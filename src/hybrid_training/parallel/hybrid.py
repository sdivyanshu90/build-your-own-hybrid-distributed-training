"""Composing data, sharding, tensor and sequence parallelism.

The composition rule
====================
The four strategies are orthogonal because each one acts on a *different* axis
of the problem:

===================  ==========================  ==========================
Dimension            Splits                      Leaves replicated
===================  ==========================  ==========================
``tensor``           weight matrices, attention  the activations entering
                     heads                       and leaving each region
``sequence``         activations along the       every parameter
                     sequence axis
``shard`` (FSDP)     parameters, gradients,      the computation itself
                     optimizer state
``data_parallel``    the global batch            every parameter
===================  ==========================  ==========================

So a model is *built* with tensor and sequence parallelism baked into its
layers, and then *wrapped* for sharding and/or replication.  The wrapper never
needs to know what the layers do, and the layers never need to know how the
model is wrapped.

Which group does what
=====================
This is the table that has to be right, because getting it wrong produces
training that converges to the wrong thing rather than an error:

==============================  ============================================
Quantity                        Reduced over
==============================  ============================================
tensor-parallel activations     ``tensor`` (all-reduce, or all-gather +
                                reduce-scatter with sequence parallelism)
FSDP parameter all-gather       ``shard``
FSDP gradient reduce-scatter    ``shard``
replica gradient all-reduce     ``data_parallel`` (after the reduce-scatter,
                                on the *sharded* gradient)
DDP gradient all-reduce         ``data_parallel``
loss / metrics                  ``dp_shard`` -- **not** the world, because
                                ranks in the same tensor/sequence group
                                processed the *same* samples and would be
                                counted twice
global gradient norm            ``world``, with per-parameter replication
                                weights (see
                                :mod:`hybrid_training.optim.sharded_optimizer`)
==============================  ============================================

Data feeding
============
Ranks that share a ``tensor_sequence`` group are collaborating on one batch, so
they **must** receive identical input.  Ranks in different ``dp_shard`` groups
must receive different input.  The sampler in
:mod:`hybrid_training.training.data` indexes by the ``dp_shard`` coordinate for
exactly this reason, and :meth:`HybridModel.validate_input_replication` checks
it at run time when asked.

Worked example: 8 ranks, ``dp=2 x shard=2 x tensor=2``
======================================================
.. code-block:: text

    rank  dp sh tp   holds                              gradient path
    ----  -- -- --   --------------------------------   -----------------------
     0     0  0  0   shard 0 of {tp-slice 0 weights}    RS over (0,2) -> AR (0,4)
     1     0  0  1   shard 0 of {tp-slice 1 weights}    RS over (1,3) -> AR (1,5)
     2     0  1  0   shard 1 of {tp-slice 0 weights}    RS over (0,2) -> AR (2,6)
     3     0  1  1   shard 1 of {tp-slice 1 weights}    RS over (1,3) -> AR (3,7)
     4     1  0  0   shard 0 of {tp-slice 0 weights}    RS over (4,6) -> AR (0,4)
     5     1  0  1   shard 0 of {tp-slice 1 weights}    RS over (5,7) -> AR (1,5)
     6     1  1  0   shard 1 of {tp-slice 0 weights}    RS over (4,6) -> AR (2,6)
     7     1  1  1   shard 1 of {tp-slice 1 weights}    RS over (5,7) -> AR (3,7)

Ranks 0 and 1 see the *same* data but different weight slices.  Ranks 0 and 2
see the same data and hold different shards of the same slice.  Ranks 0 and 4
see *different* data and hold identical shards -- which is why the replica
all-reduce runs over ``(0, 4)``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from ..config import ExperimentConfig
from ..distributed.collectives import (
    CommunicationRecorder,
    ReduceOp,
    assert_tensor_consistent,
    sum_scalar,
)
from ..distributed.context import DistributedContext
from ..distributed.groups import GroupHandle
from ..errors import ConfigurationError, ShardingError, format_error
from ..logging import get_logger
from .ddp import DistributedDataParallel
from .fsdp import FullyShardedDataParallel, PieceLayout, ShardedTensorPiece
from .tensor_parallel import all_reduce_sequence_parallel_gradients

# `hybrid_training.models` imports the parallel layers, so importing the models
# at module scope here would close an import cycle
# (models -> parallel -> hybrid -> models).  `build_model` is the only place
# that needs them, and it imports them on first call.

__all__ = [
    "HybridModel",
    "ParameterParallelInfo",
    "build_model",
    "build_parallel_model",
    "describe_parallel_plan",
]

_LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class ParameterParallelInfo:
    """How one parameter relates to the topology.

    The checkpoint layer uses this to decide what a tensor's *global identity*
    is and which ranks should write it.

    Attributes:
        name: Parameter name, wrapper-independent.
        shape: Shape as this rank holds it.  For a tensor-parallel weight this
            is the *slice* shape, and ``tensor_parallel_size > 1`` records that
            it is one of several slices.
        partition_dim: Which dimension the tensor-parallel split runs along, or
            ``None`` when the parameter is replicated across the tensor group.
        tensor_parallel_size: Width of the tensor-parallel group.
        tensor_parallel_rank: This rank's index within it.
    """

    name: str
    shape: tuple[int, ...]
    partition_dim: int | None
    tensor_parallel_size: int
    tensor_parallel_rank: int

    @property
    def storage_key(self) -> str:
        """The name under which this tensor is stored in a checkpoint.

        Tensor-parallel slices are *different tensors*: rank 0 and rank 1 of a
        tensor group hold disjoint halves of a weight matrix, and a row-parallel
        slice is not even contiguous in the full matrix's row-major layout.
        Rather than pretend a single global tensor exists, each slice is stored
        under its own key.  Resharding across *sharding* widths still works
        (the FSDP offsets are relative to the slice), while changing the
        *tensor-parallel* width is rejected explicitly by the reader.
        """
        if self.tensor_parallel_size <= 1:
            return self.name
        return f"{self.name}#tp{self.tensor_parallel_rank}of{self.tensor_parallel_size}"

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view for the checkpoint manifest."""
        return {
            "name": self.name,
            "shape": list(self.shape),
            "partition_dim": self.partition_dim,
            "tensor_parallel_size": self.tensor_parallel_size,
            "tensor_parallel_rank": self.tensor_parallel_rank,
        }


@dataclass
class ParallelismDescription:
    """Human-readable summary of a composed strategy.

    Attributes:
        strategy: Short label such as ``"fsdp+tensor"``.
        replicated_parameters: Names of the groups parameters are replicated
            over.
        sharded_parameters: Names of the groups parameters are split over.
        sharded_activations: Names of the groups activations are split over.
        gradient_reductions: Ordered list of ``(operation, group)`` pairs.
        metric_group: Group metrics must be reduced over.
        notes: Extra remarks worth logging.
    """

    strategy: str
    replicated_parameters: list[str] = field(default_factory=list)
    sharded_parameters: list[str] = field(default_factory=list)
    sharded_activations: list[str] = field(default_factory=list)
    gradient_reductions: list[tuple[str, str]] = field(default_factory=list)
    metric_group: str = "dp_shard"
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        """Multi-line description."""
        lines = [f"parallel strategy: {self.strategy}"]
        lines.append(f"  parameters replicated over : {self.replicated_parameters or ['-']}")
        lines.append(f"  parameters sharded over    : {self.sharded_parameters or ['-']}")
        lines.append(f"  activations sharded over   : {self.sharded_activations or ['-']}")
        lines.append("  gradient reduction order   :")
        for index, (operation, group) in enumerate(self.gradient_reductions, start=1):
            lines.append(f"      {index}. {operation} over {group!r}")
        lines.append(f"  metrics reduced over       : {self.metric_group!r}")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


def build_model(
    config: ExperimentConfig,
    context: DistributedContext,
    *,
    device: torch.device | None = None,
) -> nn.Module:
    """Construct the reference model with tensor/sequence parallelism baked in.

    Args:
        config: Experiment configuration.
        context: Active distributed context, used for the tensor and sequence
            groups.
        device: Construction device.  Defaults to the context's device.

    Returns:
        An unwrapped ``nn.Module``.  Data/shard parallelism is added separately
        by :func:`build_parallel_model`.

    Raises:
        ConfigurationError: If the model kind does not support the requested
            parallelism -- for example tensor parallelism on the plain MLP head.
    """
    from ..models.mlp import MLP
    from ..models.transformer import ParallelPlan, TinyTransformer

    device = device if device is not None else context.device
    tensor_group = context.group("tensor")
    sequence_group = context.group("sequence_effective")
    sequence_parallel = config.topology.sequence_parallel_enabled

    if config.model.kind == "transformer":
        plan = ParallelPlan(
            tensor_group=tensor_group,
            sequence_group=sequence_group,
            sequence_parallel=sequence_parallel,
            vocab_parallel=tensor_group.size > 1,
        )
        return TinyTransformer(config.model, plan, seed=config.training.seed, device=device)

    if sequence_parallel:
        raise ConfigurationError(
            format_error(
                "hybrid.build_model",
                "the MLP reference model has no sequence dimension, so sequence "
                "parallelism cannot be applied to it",
                rank=context.rank,
                world_size=context.world_size,
                expected="model.kind='transformer' for sequence parallelism",
                observed=config.model.kind,
                resolution="use the transformer model, or disable sequence parallelism",
            )
        )
    return MLP(
        config.model,
        seed=config.training.seed,
        tensor_parallel_group=tensor_group if tensor_group.size > 1 else None,
        device=device,
    )


def describe_parallel_plan(
    config: ExperimentConfig, context: DistributedContext
) -> ParallelismDescription:
    """Describe what the composed strategy does, without building anything.

    Args:
        config: Experiment configuration.
        context: Active distributed context.

    Returns:
        A :class:`ParallelismDescription` suitable for logging or for a test to
        assert against.
    """
    topology = context.topology
    shard_size = topology.size("shard")
    replica_size = topology.size("data_parallel")
    tensor_size = topology.size("tensor")

    labels = []
    if shard_size > 1:
        labels.append("fsdp")
    if replica_size > 1:
        labels.append("ddp" if shard_size == 1 else "replicate")
    if tensor_size > 1:
        labels.append("tensor")
    if topology.sequence_parallel_enabled:
        labels.append("sequence")
    strategy = "+".join(labels) if labels else "single-process"

    description = ParallelismDescription(strategy=strategy)
    if tensor_size > 1:
        description.sharded_parameters.append("tensor")
        description.replicated_parameters.append("data_parallel")
    if shard_size > 1:
        description.sharded_parameters.append("shard")
    if replica_size > 1 and "data_parallel" not in description.replicated_parameters:
        description.replicated_parameters.append("data_parallel")
    if topology.sequence_parallel_enabled:
        description.sharded_activations.append(topology.sequence_group_name)

    if tensor_size > 1:
        description.gradient_reductions.append(
            (
                "all-reduce of activation gradients (inside the layers, not of the weights)",
                "tensor",
            )
        )
    if shard_size > 1:
        description.gradient_reductions.append(("reduce-scatter of flat gradients", "shard"))
        if replica_size > 1:
            description.gradient_reductions.append(
                ("all-reduce of the sharded gradient", "data_parallel")
            )
    elif replica_size > 1:
        description.gradient_reductions.append(
            ("bucketed all-reduce of gradients", "data_parallel")
        )

    description.metric_group = "dp_shard"
    if tensor_size > 1:
        description.notes.append(
            "ranks in the same tensor group must receive identical input batches"
        )
    if shard_size > 1 and replica_size > 1:
        description.notes.append(
            "hybrid sharding: memory scales with 1/shard_size only; the replica "
            "dimension buys communication locality, not memory"
        )

    # The knobs that change how much is communicated belong in the plan too:
    # someone reading it should be able to predict the collective count without
    # opening the configuration file as well.
    if shard_size > 1:
        if config.fsdp.reshard_after_forward:
            description.notes.append(
                "reshard_after_forward=True: one extra all-gather per unit per step, "
                "in exchange for not holding the full parameters between forward "
                "and backward"
            )
        else:
            description.notes.append(
                "reshard_after_forward=False: the gathered parameters stay resident "
                "until backward consumes them, saving one all-gather per unit"
            )
        if config.fsdp.auto_wrap_min_num_params == 0:
            description.notes.append(
                "the whole module is one FSDP unit, so the transient all-gather "
                "buffer is the size of the entire model; set "
                "fsdp.auto_wrap_min_num_params to split it"
            )
    elif replica_size > 1:
        description.notes.append(
            f"DDP bucket cap {config.ddp.bucket_cap_mb} MiB, "
            f"async_reduction={config.ddp.async_reduction}"
        )
    if config.training.gradient_accumulation_steps > 1:
        description.notes.append(
            f"{config.training.gradient_accumulation_steps} micro-batches per "
            "optimizer step: the first N-1 run inside no_sync(), so the step "
            "costs one gradient reduction rather than N"
        )
    return description


class HybridModel(nn.Module):
    """A model wrapped for whatever combination of parallelism the topology asks for.

    The wrapper selects exactly one of three data-side strategies:

    * ``shard_parallel_size > 1``  -> FSDP over ``"shard"``, with the
      ``"data_parallel"`` group passed as the replica group when it is wider
      than one rank (hybrid sharding).
    * ``shard_parallel_size == 1 and data_parallel_size > 1`` -> the bucketed
      DDP implementation over ``"data_parallel"``.
    * neither -> no wrapper; the model is used directly.

    Tensor and sequence parallelism are already inside ``model``'s layers.

    Args:
        model: The (already tensor/sequence-parallel) module.
        config: Experiment configuration.
        context: Active distributed context.
        recorder: Optional communication instrumentation sink.

    Example:
        >>> # doctest: +SKIP
        >>> model = build_parallel_model(config, ctx)
        >>> loss = criterion(model(x), y)
        >>> loss.backward()
        >>> model.finish_backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        context: DistributedContext,
        *,
        recorder: CommunicationRecorder | None = None,
    ) -> None:
        super().__init__()
        self._config = config
        self._context = context
        self._recorder = recorder
        self._description = describe_parallel_plan(config, context)

        shard_group = context.group("shard")
        replica_group = context.group("data_parallel")
        self._sequence_parallel_group: GroupHandle | None = (
            context.group("sequence_effective")
            if config.topology.sequence_parallel_enabled
            else None
        )

        self._fsdp: FullyShardedDataParallel | None = None
        self._ddp: DistributedDataParallel | None = None

        if shard_group.size > 1:
            self._fsdp = FullyShardedDataParallel(
                model,
                shard_group,
                config.fsdp,
                replica_group=replica_group if replica_group.size > 1 else None,
                mixed_precision=config.mixed_precision,
                device=context.device,
                recorder=recorder,
            )
            self.wrapped: nn.Module = self._fsdp
        elif replica_group.size > 1:
            self._ddp = DistributedDataParallel(model, replica_group, config.ddp, recorder=recorder)
            self.wrapped = self._ddp
        else:
            self.wrapped = model

        self._inner_model = model
        _LOGGER.info("hybrid model ready\n%s", self._description.render())

    # -- nn.Module ----------------------------------------------------------
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate to the wrapped module."""
        return self.wrapped(*args, **kwargs)

    # -- training-loop integration -----------------------------------------
    def no_sync(self) -> AbstractContextManager[None]:
        """Return a context manager that suppresses gradient synchronisation.

        Both wrappers implement it; the unwrapped case returns a null context,
        because with no communication there is nothing to suppress.

        Returns:
            A context manager.
        """
        if self._ddp is not None:
            return self._ddp.no_sync()
        if self._fsdp is not None:
            return self._fsdp.no_sync()
        return nullcontext()

    def finish_backward(self) -> None:
        """Close the synchronisation boundary after ``loss.backward()``.

        Three things happen here, **in this order**:

        1. **Sequence-parallel gradient completion.**  Parameters that only ever
           saw this rank's sequence shard (LayerNorm gains, row-parallel biases)
           hold partial gradients that no collective in the autograd graph
           sums.  They are summed over the sequence-parallel group first,
           because everything downstream assumes gradients are complete.
        2. **DDP bucket completion**, if replication is in use: wait for the
           in-flight all-reduces and rebind ``param.grad`` to the reduced
           buffers.
        3. **FSDP buffer release**, if sharding is in use: the reduce-scatter
           already happened inside backward, so this only frees the transient
           full parameters and runs the optional reduction-order check.

        The ordering matters for step 1 versus step 2: reducing the
        sequence-parallel partials *after* the data-parallel all-reduce would
        average incomplete gradients and then sum them, which is not the same
        number.
        """
        if self._sequence_parallel_group is not None:
            all_reduce_sequence_parallel_gradients(
                self._inner_model, self._sequence_parallel_group, recorder=self._recorder
            )
        if self._ddp is not None:
            self._ddp.finish_gradient_synchronization()
        if self._fsdp is not None:
            self._fsdp.finish_backward()

    def optimizer_parameters(self) -> list[nn.Parameter]:
        """Return the parameters an optimizer should be built from.

        Under FSDP these are the flat shards -- which is what makes the
        optimizer state sharded without the optimizer knowing anything about
        sharding.

        Returns:
            The trainable parameters this rank owns.
        """
        return [p for p in self.wrapped.parameters() if p.requires_grad]

    # -- group accessors ----------------------------------------------------
    @property
    def metric_group(self) -> GroupHandle:
        """Group over which losses and metrics must be averaged.

        This is ``dp_shard``, **not** the world.  Ranks that share a
        tensor/sequence group processed the same samples, so averaging over the
        world would weight those samples by the tensor-parallel size.
        """
        return self._context.group("dp_shard")

    @property
    def norm_group(self) -> GroupHandle:
        """Group over which the global gradient norm is reduced.

        The world, because a hybrid job partitions parameters over several
        dimensions simultaneously and every rank holds a piece of the total.
        The per-parameter replication weighting in
        :mod:`hybrid_training.optim.sharded_optimizer` is what keeps replicated
        parameters from being counted more than once.
        """
        return self._context.group("world")

    @property
    def data_group(self) -> GroupHandle:
        """Group whose coordinate selects which slice of the batch to read."""
        return self._context.group("dp_shard")

    @property
    def description(self) -> ParallelismDescription:
        """Summary of the composed strategy."""
        return self._description

    @property
    def fsdp(self) -> FullyShardedDataParallel | None:
        """The FSDP wrapper, when sharding is active."""
        return self._fsdp

    @property
    def ddp(self) -> DistributedDataParallel | None:
        """The DDP wrapper, when pure replication is active."""
        return self._ddp

    @property
    def inner_model(self) -> nn.Module:
        """The unwrapped module, including its tensor-parallel layers."""
        return self._inner_model

    # -- metrics ------------------------------------------------------------
    def reduce_metric(self, value: float, *, average: bool = True) -> float:
        """Reduce a scalar metric over the data-processing group.

        Args:
            value: This rank's local value.
            average: Average rather than sum.

        Returns:
            The reduced value, identical on every rank.
        """
        return sum_scalar(
            value,
            self.metric_group,
            device=self._context.device,
            op=ReduceOp.AVG if average else ReduceOp.SUM,
            recorder=self._recorder,
        )

    def validate_input_replication(self, batch: torch.Tensor) -> None:
        """Assert that model-parallel peers received identical inputs.

        Ranks in the same ``tensor_sequence`` group collaborate on one batch.
        If they are fed different data, the tensor-parallel all-reduces combine
        activations from unrelated samples -- and the result still looks like a
        loss curve, just a wrong one.  This check costs one broadcast and one
        comparison, so it belongs in tests and in a debug run rather than in a
        hot loop.

        Args:
            batch: The input tensor to compare.

        Raises:
            ParameterConsistencyError: If the batches differ.
        """
        group = self._context.group("tensor_sequence")
        if group.size == 1:
            return
        assert_tensor_consistent(
            batch.detach().to(torch.float64) if batch.is_floating_point() else batch.detach(),
            group,
            name="the input batch",
            operation="hybrid.validate_input_replication",
        )

    # -- state ---------------------------------------------------------------
    def parameter_parallel_info(self) -> dict[str, ParameterParallelInfo]:
        """Describe every parameter's relationship to the tensor-parallel group.

        Returns:
            Mapping from wrapper-independent parameter name to its
            :class:`ParameterParallelInfo`.
        """
        tensor_group = self._context.group("tensor")
        if self._fsdp is not None:
            described = self._fsdp.original_named_parameters()
        else:
            described = {
                name: (tuple(param.shape), param)
                for name, param in _original_named_parameters(self._inner_model).items()
            }
        info: dict[str, ParameterParallelInfo] = {}
        for name, (shape, param) in described.items():
            replicated = bool(getattr(param, "is_tensor_parallel_replicated", True))
            partition_dim = (
                None if replicated else int(getattr(param, "tensor_parallel_partition_dim", 0))
            )
            info[name] = ParameterParallelInfo(
                name=name,
                shape=shape,
                partition_dim=partition_dim,
                tensor_parallel_size=tensor_group.size,
                tensor_parallel_rank=tensor_group.local_rank,
            )
        return info

    def optimizer_parameter_layout(self) -> list[list[PieceLayout]]:
        """Describe, for each optimizer parameter, which model tensors it covers.

        Aligned index-for-index with :meth:`optimizer_parameters`.  Under FSDP
        an optimizer parameter is a flat shard spanning several model tensors;
        otherwise it is one model tensor and the layout has a single entry.

        Returns:
            One layout list per optimizer parameter.

        Raises:
            ShardingError: If the layout cannot be aligned with the optimizer's
                parameters, which would mean the two were built from different
                module trees.
        """
        parameters = self.optimizer_parameters()
        if self._fsdp is not None:
            pairs = self._fsdp.optimizer_parameter_layout()
            if [id(p) for p, _ in pairs] != [id(p) for p in parameters]:
                raise ShardingError(
                    format_error(
                        "hybrid.optimizer_parameter_layout",
                        "the FSDP flat-parameter order does not match the optimizer's "
                        "parameter order, so optimizer state could not be attributed to "
                        "the right tensors",
                        rank=self._context.rank,
                        expected=len(parameters),
                        observed=len(pairs),
                        resolution="build the optimizer from model.optimizer_parameters()",
                    )
                )
            return [layout for _, layout in pairs]

        by_id = {
            id(param): (name, tuple(param.shape))
            for name, param in _original_named_parameters(self._inner_model).items()
        }
        layouts: list[list[PieceLayout]] = []
        for param in parameters:
            entry = by_id.get(id(param))
            if entry is None:
                raise ShardingError(
                    format_error(
                        "hybrid.optimizer_parameter_layout",
                        "an optimizer parameter does not belong to the wrapped model",
                        rank=self._context.rank,
                        expected="a parameter of the wrapped model",
                        observed=f"tensor of shape {tuple(param.shape)}",
                        resolution="build the optimizer from model.optimizer_parameters()",
                    )
                )
            name, shape = entry
            layouts.append(
                [
                    PieceLayout(
                        name=name,
                        global_shape=shape,
                        parameter_offset=0,
                        length=param.numel(),
                        local_offset=0,
                    )
                ]
            )
        return layouts

    def sharded_state_dict(self) -> dict[str, ShardedTensorPiece]:
        """Return this rank's slice of every parameter, in global coordinates.

        For an FSDP model this is genuinely partial.  For a DDP or unwrapped
        model each rank holds every parameter whole, so the pieces cover the
        full tensors -- the checkpoint writer then de-duplicates by only
        writing from ranks whose replication coordinates are zero.

        Returns:
            Mapping from parameter name to its owned piece.
        """
        if self._fsdp is not None:
            return self._fsdp.sharded_state_dict()
        pieces: dict[str, ShardedTensorPiece] = {}
        for name, param in _original_named_parameters(self._inner_model).items():
            pieces[name] = ShardedTensorPiece(
                name=name,
                global_shape=tuple(param.shape),
                offset=0,
                data=param.detach().reshape(-1).clone(),
            )
        return pieces

    def buffers_state_dict(self) -> dict[str, torch.Tensor]:
        """Return the model's buffers under wrapper-independent names."""
        if self._fsdp is not None:
            return {name: b.detach().clone() for name, b in self._fsdp.original_named_buffers()}
        return {name: b.detach().clone() for name, b in self._inner_model.named_buffers()}

    def full_state_dict(self) -> dict[str, torch.Tensor]:
        """Reconstruct the complete (per tensor-parallel rank) parameters.

        Returns:
            Mapping from parameter name to full tensor.  Under tensor
            parallelism the tensors are this rank's *slices*, because a
            tensor-parallel slice is a different tensor on every rank; use
            :meth:`parameter_parallel_info` to interpret them.
        """
        if self._fsdp is not None:
            return self._fsdp.full_state_dict()
        result = {
            name: param.detach().clone()
            for name, param in _original_named_parameters(self._inner_model).items()
        }
        result.update(self.buffers_state_dict())
        return result

    def load_full_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load a full state dict into this rank's shards.

        Args:
            state_dict: Mapping from parameter name to full tensor.
        """
        if self._fsdp is not None:
            self._fsdp.load_full_state_dict(state_dict)
            return
        with torch.no_grad():
            for name, param in _original_named_parameters(self._inner_model).items():
                if name in state_dict:
                    param.data.copy_(state_dict[name].to(param.dtype).to(param.device))
            for name, buffer in self._inner_model.named_buffers():
                if name in state_dict:
                    buffer.data.copy_(state_dict[name].to(buffer.dtype))

    @contextmanager
    def summon_full_params(self) -> Iterator[None]:
        """Materialise sharded parameters for inspection.

        A no-op when sharding is not in use, so calling code does not need to
        branch on the strategy.

        Yields:
            ``None``.
        """
        if self._fsdp is None:
            yield
            return
        with self._fsdp.summon_full_params():
            yield

    def memory_summary(self) -> dict[str, int]:
        """Per-rank byte counts, when sharding is active.

        Returns:
            The FSDP memory summary, or a parameter-byte count for the
            unsharded strategies.
        """
        if self._fsdp is not None:
            return self._fsdp.memory_summary()
        parameter_bytes = sum(p.numel() * p.element_size() for p in self._inner_model.parameters())
        gradient_bytes = sum(
            p.grad.numel() * p.grad.element_size()
            for p in self._inner_model.parameters()
            if p.grad is not None
        )
        return {
            "shard_bytes": parameter_bytes,
            "full_bytes": parameter_bytes,
            "grad_shard_bytes": gradient_bytes,
            "padding_bytes": 0,
            "units": 0,
        }

    def __repr__(self) -> str:
        return f"HybridModel({self._description.strategy}, wrapped={type(self.wrapped).__name__})"


def build_parallel_model(
    config: ExperimentConfig,
    context: DistributedContext,
    *,
    recorder: CommunicationRecorder | None = None,
    device: torch.device | None = None,
) -> HybridModel:
    """Build the model and wrap it according to the topology.

    Args:
        config: Experiment configuration.
        context: Active distributed context.
        recorder: Optional communication instrumentation sink.
        device: Construction device.

    Returns:
        A ready-to-train :class:`HybridModel`.
    """
    model = build_model(config, context, device=device)
    return HybridModel(model, config, context, recorder=recorder)


def _original_named_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    """Return ``named_parameters`` of an *unwrapped* model as a dict.

    Deduplicates tied parameters by identity while keeping the first name, so
    the mapping is a faithful description of the distinct tensors.

    Args:
        model: An unwrapped module.

    Returns:
        Ordered mapping from name to parameter.
    """
    seen: set[int] = set()
    result: dict[str, nn.Parameter] = {}
    for name, param in model.named_parameters():
        if id(param) in seen:
            continue
        seen.add(id(param))
        result[name] = param
    return result
