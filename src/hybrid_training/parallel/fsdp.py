"""FSDP-style sharding of parameters, gradients and optimizer states.

The memory problem
==================
Plain data parallelism replicates everything.  Per rank, for ``P`` parameters
trained in fp32 with Adam::

    parameters   P * 4 bytes
    gradients    P * 4 bytes
    exp_avg      P * 4 bytes
    exp_avg_sq   P * 4 bytes
    -----------------------
    total        P * 16 bytes

A 7-billion-parameter model therefore needs 112 GB per rank before a single
activation is stored.  Adding ranks does not help: every rank holds the same
112 GB.

Full sharding divides all four rows by the shard-group size ``G``::

    total        P * 16 / G bytes   +   transient all-gather buffers

With ``G = 64`` the same model needs 1.75 GB of persistent state per rank.  The
*transient* term is the cost: a unit's parameters must be whole to compute
with, so they are gathered just before use and thrown away just after.

The lifecycle
=============
Each FSDP *unit* owns one **flat parameter**: the concatenation of its
parameters' elements, padded to a multiple of ``G``, of which each rank
persistently stores exactly ``1/G``.

.. code-block:: text

    idle          rank r holds flat_param[r * S : (r+1) * S]        (S = padded/G)
      |
      | forward starts
      v
    all-gather    full = concat over ranks           -> padded_numel elements
      |           views bound to the modules' .weight/.bias attributes
      v
    forward       ordinary computation on whole tensors
      |
      | reshard_after_forward: free full's storage (views stay valid objects)
      v
    idle again    only the 1/G shard is resident
      |
      | backward reaches this unit
      v
    all-gather    refill full's storage in place, so the tensors autograd saved
      |           during forward point at correct data again
      v
    backward      gradients accumulate into one padded_numel flat gradient
      |
      v
    reduce-scatter  sum across the shard group, keep only slice r
      |
      v
    flat_param.grad = the local gradient shard  -> optimizer updates 1/G

Why a custom autograd Function rather than hooks
================================================
The all-gather is expressed as :class:`_AllGatherFlatParam`, whose adjoint is
the reduce-scatter.  Concretely, for the flat shard :math:`s_r` on rank
:math:`r` and the gathered parameter :math:`W = \\mathrm{concat}_r(s_r)`:

.. math::

    W = A(s_0, \\dots, s_{G-1}), \\qquad
    \\frac{\\partial L}{\\partial s_r}
      = \\Big[\\sum_{q} \\frac{\\partial L}{\\partial W}\\Big|_q\\Big]_{\\text{slice } r}

which is precisely ``reduce_scatter``.  Expressing it this way means autograd
does the bookkeeping: gradients for every parameter in the unit flow into the
single flat gradient buffer, the reduce-scatter happens exactly once, at
exactly the right point in the backward pass, and the result is accumulated
into ``flat_param.grad`` by the ordinary accumulation machinery.  There is no
hook counting, no "have all parameters reported yet" logic, and no possibility
of reducing twice.

Ownership summary
=================
============================  ==========================================
Object                        Who holds it
============================  ==========================================
persistent parameter shard    rank ``r`` holds elements
                              ``[r*S, (r+1)*S)`` of the padded flat buffer
full parameters               every rank, transiently, between all-gather
                              and reshard
gradient (unsharded)          every rank, transiently, during backward
gradient (sharded)            rank ``r`` holds the same slice as its
                              parameter shard -- which is exactly why the
                              local optimizer state lines up
optimizer state               rank ``r``, sized to its parameter shard
============================  ==========================================
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from ..config import FSDPConfig, MixedPrecisionConfig, resolve_dtype
from ..distributed.collectives import (
    CommunicationRecorder,
    ReduceOp,
    all_gather_object_in_group,
    all_gather_tensor,
    all_reduce,
    assert_metadata_consistent,
    broadcast,
    reduce_scatter_tensor,
)
from ..distributed.groups import GroupHandle
from ..errors import (
    ParameterConsistencyError,
    ShardingError,
    UnsupportedFeatureError,
    format_error,
)
from ..logging import get_logger
from ..utils.tensors import FlatEntry, ShardRange, build_flat_layout, intersect_ranges

__all__ = [
    "FlatParamHandle",
    "FullyShardedDataParallel",
    "PieceLayout",
    "ShardedTensorPiece",
    "reshard_state_dict_pieces",
]

_LOGGER = get_logger(__name__)

#: ``torch.autograd._unsafe_preserve_version_counter`` lets us refill a flat
#: parameter's storage during backward without tripping the "variable needed for
#: gradient computation has been modified in place" guard.  The write is safe
#: because the refilled bytes are *identical* to the ones autograd saw during
#: forward -- we are restoring the value, not changing it.  If the private API
#: ever disappears, ``reshard_after_forward`` fails loudly instead of silently
#: producing wrong gradients.
_HAS_VERSION_COUNTER_GUARD = hasattr(torch.autograd, "_unsafe_preserve_version_counter")


@dataclass(frozen=True)
class ShardedTensorPiece:
    """This rank's slice of one logical parameter, in *global* coordinates.

    The checkpoint layer speaks this language, not "rank ``r``'s flat
    parameter", so that a checkpoint written by 4 ranks can be read by 3.

    Attributes:
        name: Fully-qualified parameter name, identical on every rank.
        global_shape: Shape of the whole parameter.
        offset: Index of the first element this rank owns, in the parameter's
            own row-major flattening.
        data: 1-D tensor of length ``length`` holding the owned elements.
    """

    name: str
    global_shape: tuple[int, ...]
    offset: int
    data: torch.Tensor

    @property
    def length(self) -> int:
        """Number of elements owned."""
        return int(self.data.numel())

    @property
    def range(self) -> ShardRange:
        """The owned interval."""
        return ShardRange(start=self.offset, length=self.length)


@dataclass(frozen=True)
class PieceLayout:
    """Where one parameter's owned elements live inside this rank's flat shard.

    Attributes:
        name: Parameter name.
        global_shape: Shape of the whole parameter.
        parameter_offset: Index of the first owned element within the
            parameter's own row-major flattening.  This is the *global*
            coordinate a checkpoint records.
        length: Number of owned elements.
        local_offset: Index of the same elements within this rank's flat shard.
            This is the *local* coordinate used to read and write the data.
    """

    name: str
    global_shape: tuple[int, ...]
    parameter_offset: int
    length: int
    local_offset: int


def _free_storage(tensor: torch.Tensor) -> None:
    """Release a tensor's storage while keeping the tensor object alive.

    Views built from ``tensor`` remain valid *objects*; they simply point at a
    zero-length storage until :func:`_alloc_storage` refills it.  This is what
    lets FSDP drop the gathered parameters after forward without invalidating
    the tensors autograd saved for backward.

    Args:
        tensor: Tensor whose storage should be released.
    """
    storage = tensor.untyped_storage()
    if storage.size() > 0:
        storage.resize_(0)


def _alloc_storage(tensor: torch.Tensor, num_bytes: int) -> None:
    """Re-acquire storage previously released by :func:`_free_storage`.

    Args:
        tensor: Tensor to refill.
        num_bytes: Byte count the storage must hold.
    """
    storage = tensor.untyped_storage()
    if storage.size() != num_bytes:
        storage.resize_(num_bytes)


@contextmanager
def _preserve_version(tensor: torch.Tensor) -> Iterator[None]:
    """Restore ``tensor``'s autograd version counter after the block.

    Args:
        tensor: Tensor whose version counter should be pinned.

    Yields:
        ``None``.
    """
    if _HAS_VERSION_COUNTER_GUARD:
        with torch.autograd._unsafe_preserve_version_counter(tensor):
            yield
    else:  # pragma: no cover - depends on the torch build
        yield


class _AllGatherFlatParam(torch.autograd.Function):
    """Materialise the full flat parameter; its adjoint is the reduce-scatter.

    Forward: ``W = concat_r(shard_r)`` over the shard group.
    Backward: ``dL/dshard_r = reduce_scatter_r(dL/dW)``.

    The ``handle`` argument is a plain Python object, so autograd returns
    ``None`` as its "gradient"; only the first argument is differentiable.
    """

    @staticmethod
    def forward(ctx: Any, shard: torch.Tensor, handle: FlatParamHandle) -> torch.Tensor:
        ctx.handle = handle
        return handle.gather_full(shard)

    @staticmethod
    def backward(ctx: Any, grad_full: torch.Tensor) -> tuple[torch.Tensor | None, None]:  # type: ignore[override]
        return ctx.handle.reduce_scatter_gradient(grad_full), None


class _PreBackwardUnshard(torch.autograd.Function):
    """Identity in forward; re-materialises a unit's parameters in backward.

    Attached to a unit's *outputs*, so its backward runs before any of the
    unit's internal operator backwards -- exactly when the parameters need to
    exist again.

    It carries the **specific** gathered buffer that this forward pass bound its
    views to, rather than asking the handle for "the current one".  That matters
    whenever more than one forward happens before a backward -- a summed
    multi-task loss, or accumulation written as ``(l1 + l2 + l3).backward()``.
    Each forward allocates its own buffer and each graph saved views into *its*
    buffer, so refilling only the most recent one would leave the earlier graphs
    reading storage that was freed and never restored.
    """

    @staticmethod
    def forward(
        ctx: Any, handle: FlatParamHandle, full: torch.Tensor, *tensors: torch.Tensor
    ) -> Any:
        ctx.handle = handle
        ctx.full = full
        # Return fresh view objects so a graph node is genuinely inserted.
        outputs = tuple(t.view_as(t) for t in tensors)
        return outputs[0] if len(outputs) == 1 else outputs

    @staticmethod
    def backward(ctx: Any, *grads: torch.Tensor) -> tuple[Any, ...]:  # type: ignore[override]
        ctx.handle.refill(ctx.full)
        return (None, None, *grads)


class FlatParamHandle:
    """Owns one FSDP unit's flat parameter and drives its lifecycle.

    Args:
        named_parameters: ``(qualified_name, parameter)`` pairs the unit
            manages, in a deterministic order.
        locations: For each managed parameter (by identity), every
            ``(module, attribute_name)`` it is registered under.  A parameter
            appearing at more than one location is a *tied* weight; the same
            view is bound to all of its locations.
        shard_group: The group parameters are split across.
        config: FSDP knobs.
        mixed_precision: Precision policy.
        device: Compute device.
        replica_group: Optional outer group over which the *sharded* gradient
            is additionally all-reduced (hybrid sharding).  ``None`` for pure
            sharding.
        recorder: Optional communication instrumentation sink.

    Raises:
        UnsupportedFeatureError: If the unit mixes trainable and frozen
            parameters, which a single flat parameter cannot express.
        ShardingError: If the parameter list is empty or has duplicate names.
    """

    def __init__(
        self,
        named_parameters: Sequence[tuple[str, nn.Parameter]],
        locations: dict[int, list[tuple[nn.Module, str]]],
        shard_group: GroupHandle,
        config: FSDPConfig,
        mixed_precision: MixedPrecisionConfig,
        device: torch.device,
        *,
        replica_group: GroupHandle | None = None,
        recorder: CommunicationRecorder | None = None,
    ) -> None:
        if not named_parameters:
            raise ShardingError(
                format_error(
                    "fsdp.FlatParamHandle",
                    "an FSDP unit needs at least one parameter",
                    rank=shard_group.global_rank,
                    expected=">= 1",
                    observed=0,
                    resolution="do not create a unit for a parameter-free module",
                )
            )
        requires_grad_flags = {bool(p.requires_grad) for _, p in named_parameters}
        if len(requires_grad_flags) > 1:
            frozen = [n for n, p in named_parameters if not p.requires_grad]
            raise UnsupportedFeatureError(
                format_error(
                    "fsdp.FlatParamHandle",
                    "an FSDP unit flattens its parameters into a single tensor, so all of "
                    "them must share one requires_grad flag; a unit cannot freeze part of "
                    "its flat parameter",
                    rank=shard_group.global_rank,
                    expected="a uniform requires_grad within the unit",
                    observed=f"frozen parameters present: {frozen[:8]}",
                    resolution=(
                        "wrap the frozen submodule as its own FSDP unit (auto-wrapping "
                        "does this at module boundaries), or unfreeze it"
                    ),
                )
            )

        self._shard_group = shard_group
        self._replica_group = replica_group
        self._config = config
        self._mixed_precision = mixed_precision
        self._device = device
        self._recorder = recorder
        self._requires_grad = next(iter(requires_grad_flags))

        self._names = tuple(name for name, _ in named_parameters)
        self._params = tuple(param for _, param in named_parameters)
        self._locations = {id(param): list(locations[id(param)]) for _, param in named_parameters}

        self._master_dtype = (
            resolve_dtype(mixed_precision.master_dtype)
            if mixed_precision.enabled
            else self._params[0].dtype
        )
        self._compute_dtype = (
            resolve_dtype(mixed_precision.param_dtype)
            if mixed_precision.enabled
            else self._params[0].dtype
        )
        self._reduce_dtype = (
            resolve_dtype(mixed_precision.reduce_dtype)
            if mixed_precision.enabled
            else self._params[0].dtype
        )

        self._entries, self._total_numel = build_flat_layout(list(named_parameters))
        group_size = shard_group.size
        self._shard_numel = (self._total_numel + group_size - 1) // group_size
        self._padded_numel = self._shard_numel * group_size
        self._padding = self._padded_numel - self._total_numel

        if config.limit_all_gather_bytes:
            gathered_bytes = (
                self._padded_numel * torch.empty(0, dtype=self._compute_dtype).element_size()
            )
            if gathered_bytes > config.limit_all_gather_bytes:
                raise ShardingError(
                    format_error(
                        "fsdp.FlatParamHandle",
                        "this unit's all-gather would exceed limit_all_gather_bytes; a unit "
                        "that large defeats the purpose of sharding because its transient "
                        "full copy dominates the memory saved",
                        rank=shard_group.global_rank,
                        expected=f"<= {config.limit_all_gather_bytes} bytes",
                        observed=gathered_bytes,
                        resolution=(
                            "lower FSDPConfig.auto_wrap_min_num_params so the model is split "
                            "into more units, or raise limit_all_gather_bytes"
                        ),
                    )
                )

        self.flat_param = self._build_shard()
        self._attach_gradient_norm_scale()
        self._full: torch.Tensor | None = None
        self._full_bytes = 0
        self._is_sharded = True
        self._unsharded_grad_accumulator: torch.Tensor | None = None
        self._accumulate_without_reduction = False
        self._reduction_count = 0

        self._detach_original_parameters()

    # -- construction -------------------------------------------------------
    def _build_shard(self) -> nn.Parameter:
        """Flatten the parameters, keep this rank's slice, and free the rest.

        The full flat buffer exists only for the duration of this method.  For
        very large models the memory-efficient path is to build the model on a
        meta device and materialise shard by shard; that optimisation is called
        out in ``docs/04_fsdp_style_sharding.md`` and is deliberately not
        implemented here because it obscures the mechanism.
        """
        flat = torch.zeros(self._padded_numel, dtype=self._master_dtype, device=self._device)
        for entry, param in zip(self._entries, self._params):
            flat[entry.offset : entry.end].copy_(
                param.detach().to(device=self._device, dtype=self._master_dtype).reshape(-1)
            )
        start = self._shard_group.local_rank * self._shard_numel
        shard = flat[start : start + self._shard_numel].clone()
        if self._config.cpu_offload_params:
            shard = shard.to("cpu")
        parameter = nn.Parameter(shard, requires_grad=self._requires_grad)
        del flat
        return parameter

    def _attach_gradient_norm_scale(self) -> None:
        """Annotate the flat shard with a per-element gradient-norm weighting.

        A flat parameter can concatenate tensor-parallel weight *slices* (whose
        squared norms must be summed across the tensor group) with replicated
        tensors such as LayerNorm gains (whose squared norms must be counted
        once).  A single scalar weight for the whole flat parameter would be
        wrong for one or the other, so the weighting is stored per element.

        The value stored is the *partitioned product*: the number of ranks
        across which that element's parameter is split.  Padding elements get
        ``0`` so they contribute nothing.
        :func:`hybrid_training.optim.sharded_optimizer.build_gradient_norm_contributions`
        divides by the reduction-group size to turn it into the final weight.
        """
        full_scale = torch.zeros(self._padded_numel, dtype=torch.float32)
        for entry, param in zip(self._entries, self._params):
            partitioned = self._shard_group.size
            if not getattr(param, "is_tensor_parallel_replicated", True):
                partitioned *= int(getattr(param, "tensor_parallel_group_size", 1))
            full_scale[entry.offset : entry.end] = float(partitioned)
        local = self.local_shard_range()
        self.flat_param.gradient_norm_scale_vector = (  # type: ignore[attr-defined]
            full_scale[local.start : local.end].clone()
        )
        self.flat_param.gradient_norm_label = (  # type: ignore[attr-defined]
            f"flat_param[{self._names[0]} .. {self._names[-1]}]"
        )

    def _detach_original_parameters(self) -> None:
        """Deregister the originals so only the flat shard is a real parameter.

        After this runs, ``module.weight`` is a plain tensor attribute rather
        than an ``nn.Parameter``.  That is what makes
        ``fsdp_model.parameters()`` return *only* flat shards, so an optimizer
        constructed the obvious way updates sharded state and nothing else.
        """
        placeholder = torch.empty(0, dtype=self._compute_dtype, device=self._device)
        for param in self._params:
            for module, attribute in self._locations[id(param)]:
                if attribute in module._parameters:
                    del module._parameters[attribute]
                setattr(module, attribute, placeholder)
            # Release the original (unsharded) storage.  Without this the
            # pre-sharding model stays resident and FSDP saves no memory at
            # all -- the single most embarrassing way to get this wrong.  The
            # parameter *object* is kept because it carries the tensor-parallel
            # markers; its shape lives in `self._entries`.
            param.data = torch.empty(0, dtype=param.dtype, device=param.device)

    # -- properties ---------------------------------------------------------
    @property
    def entries(self) -> tuple[FlatEntry, ...]:
        """Layout of the parameters inside the flat buffer."""
        return self._entries

    @property
    def managed_parameters(self) -> tuple[nn.Parameter, ...]:
        """The original parameter objects this unit flattened."""
        return self._params

    @property
    def names(self) -> tuple[str, ...]:
        """Managed parameter names, in flat-buffer order."""
        return self._names

    @property
    def total_numel(self) -> int:
        """Logical element count, excluding padding."""
        return self._total_numel

    @property
    def padded_numel(self) -> int:
        """Element count including padding."""
        return self._padded_numel

    @property
    def shard_numel(self) -> int:
        """Elements stored per rank."""
        return self._shard_numel

    @property
    def padding(self) -> int:
        """Number of padding elements at the end of the flat buffer."""
        return self._padding

    @property
    def is_sharded(self) -> bool:
        """``True`` when the full parameters are not currently materialised."""
        return self._is_sharded

    @property
    def shard_group(self) -> GroupHandle:
        """The group the flat parameter is split across."""
        return self._shard_group

    @property
    def replica_group(self) -> GroupHandle | None:
        """The outer replication group, when hybrid sharding."""
        return self._replica_group

    def local_shard_range(self) -> ShardRange:
        """This rank's interval within the padded flat buffer."""
        return ShardRange(
            start=self._shard_group.local_rank * self._shard_numel, length=self._shard_numel
        )

    # -- lifecycle ----------------------------------------------------------
    def gather_full(self, shard: torch.Tensor) -> torch.Tensor:
        """All-gather ``shard`` into a fresh full flat buffer and bind views.

        Called from :class:`_AllGatherFlatParam`'s forward.

        Args:
            shard: This rank's flat shard, in master dtype.

        Returns:
            The full padded buffer, in compute dtype.
        """
        compute_shard = shard
        if compute_shard.device != self._device:
            compute_shard = compute_shard.to(self._device, non_blocking=True)
        if compute_shard.dtype != self._compute_dtype:
            compute_shard = compute_shard.to(self._compute_dtype)
        compute_shard = compute_shard.contiguous()

        full = torch.empty(self._padded_numel, dtype=self._compute_dtype, device=self._device)
        _, work = all_gather_tensor(
            compute_shard, self._shard_group, out=full, recorder=self._recorder
        )
        work.wait()

        self._full = full
        self._full_bytes = full.untyped_storage().nbytes()
        self._is_sharded = False
        return full

    def bind_views(self, full: torch.Tensor) -> None:
        """Point every managed module attribute at its slice of ``full``.

        ``torch.split`` is used rather than one ``narrow`` per parameter
        because its backward builds *one* padded-size gradient for the whole
        unit instead of one per parameter, which is the difference between an
        ``O(1)`` and an ``O(num_parameters)`` transient during backward.

        Args:
            full: The materialised flat buffer.
        """
        sizes = [entry.numel for entry in self._entries]
        if self._padding:
            sizes.append(self._padding)
        pieces = torch.split(full, sizes)
        for index, entry in enumerate(self._entries):
            view = pieces[index].view(entry.shape)
            for module, attribute in self._locations[id(self._params[index])]:
                setattr(module, attribute, view)

    def unbind_views(self) -> None:
        """Replace bound views with empty placeholders.

        Makes accidental use of a resharded unit fail with an obvious shape
        error rather than reading freed memory.
        """
        placeholder = torch.empty(0, dtype=self._compute_dtype, device=self._device)
        for param in self._params:
            for module, attribute in self._locations[id(param)]:
                setattr(module, attribute, placeholder)

    def unshard(self) -> torch.Tensor:
        """Materialise the full parameters and bind them to the modules.

        Returns:
            The full padded buffer.  It participates in autograd, so gradients
            flowing into it are automatically reduce-scattered.
        """
        full = _AllGatherFlatParam.apply(self.flat_param, self)
        assert isinstance(full, torch.Tensor)
        self.bind_views(full)
        return full

    def reshard(self) -> None:
        """Free the full buffer's storage, keeping the tensor object alive.

        The views bound to the modules -- and any tensor autograd saved during
        forward -- keep referring to this storage, which is why the object must
        survive and only its bytes are released.  :meth:`refill`
        refills exactly the same bytes.
        """
        if self._full is None or self._is_sharded:
            return
        if not _HAS_VERSION_COUNTER_GUARD:  # pragma: no cover - torch build dependent
            raise UnsupportedFeatureError(
                format_error(
                    "fsdp.reshard",
                    "this PyTorch build lacks torch.autograd._unsafe_preserve_version_counter, "
                    "which is required to refill a freed flat parameter during backward "
                    "without tripping the in-place-modification guard",
                    rank=self._shard_group.global_rank,
                    resolution="set FSDPConfig(reshard_after_forward=False)",
                )
            )
        _free_storage(self._full)
        self._is_sharded = True

    def refill(self, full: torch.Tensor) -> None:
        """Restore the bytes of a specific gathered buffer before its backward.

        The refilled values are identical to the ones produced in forward, so
        pinning the version counter across the write is sound: autograd's guard
        exists to catch *value* changes, and there is none.

        Idempotent, and keyed on the buffer's **own** storage size rather than
        on a handle-level flag, so several buffers left over from several
        forward passes can each be restored independently.  See
        :class:`_PreBackwardUnshard` for why that is necessary.

        Args:
            full: The buffer to restore -- the one whose views the graph now
                being differentiated saved during its forward pass.
        """
        required_bytes = self._padded_numel * full.element_size()
        if full.untyped_storage().nbytes() == required_bytes:
            return
        # Two guards are needed, for two different autograd checks:
        #  * `no_grad` -- the collective writes into `full` through internal
        #    chunk views; without it PyTorch rejects "a view created in no_grad
        #    mode modified in place with grad mode enabled".
        #  * `_preserve_version` -- the write bumps the version counter, which
        #    would make the tensors saved during forward look stale.  Restoring
        #    the counter is sound because the bytes written are identical to
        #    the ones forward produced.
        with torch.no_grad(), _preserve_version(full):
            _alloc_storage(full, required_bytes)
            shard = self.flat_param.detach()
            if shard.device != self._device:
                shard = shard.to(self._device, non_blocking=True)
            if shard.dtype != self._compute_dtype:
                shard = shard.to(self._compute_dtype)
            _, work = all_gather_tensor(
                shard.contiguous(),
                self._shard_group,
                out=full.detach(),
                recorder=self._recorder,
            )
            work.wait()
        if full is self._full:
            self._is_sharded = False

    def reduce_scatter_gradient(self, grad_full: torch.Tensor) -> torch.Tensor | None:
        """Reduce the unit's gradient across ranks and keep this rank's slice.

        Called from :class:`_AllGatherFlatParam`'s backward with the gradient
        of the loss with respect to the *full* flat parameter.

        Steps:

        1. Cast to the reduction dtype (fp32 by default under mixed precision,
           because summing bf16 across many ranks loses precision fast).
        2. ``reduce_scatter`` over the shard group with ``AVG``, which divides
           by ``G`` before summing.  The result on rank ``r`` is the average of
           the ranks' gradients restricted to slice ``r`` -- exactly the slice
           whose parameters rank ``r`` owns.
        3. When hybrid sharding, ``all_reduce`` the resulting shard over the
           replica group with ``AVG``, dividing by ``R``.  The two divisions
           compose to ``1/(G*R)``, the correct data-parallel mean.
        4. Zero the padding contribution.  Padding elements are not parameters;
           they receive gradient ``0`` from ``split``'s backward already, so
           this is an assertion of the invariant rather than a correction.

        Args:
            grad_full: Gradient with respect to the padded full buffer.

        Returns:
            The local gradient shard in master dtype, or ``None`` when running
            inside :meth:`FullyShardedDataParallel.no_sync`, in which case the
            unsharded gradient is accumulated for a later reduction.
        """
        grad = grad_full
        if grad.dtype != self._reduce_dtype:
            grad = grad.to(self._reduce_dtype)
        grad = grad.contiguous()

        if self._accumulate_without_reduction:
            if self._unsharded_grad_accumulator is None:
                self._unsharded_grad_accumulator = grad.clone()
            else:
                self._unsharded_grad_accumulator.add_(grad)
            # Returning None tells autograd "no gradient for this input", so
            # flat_param.grad is left untouched until the synchronised step.
            return None

        if self._unsharded_grad_accumulator is not None:
            grad = grad + self._unsharded_grad_accumulator
            self._unsharded_grad_accumulator = None

        op = ReduceOp.AVG if self._config.average_gradients else ReduceOp.SUM
        shard_grad, work = reduce_scatter_tensor(
            grad, self._shard_group, op=op, recorder=self._recorder
        )
        work.wait()

        if self._replica_group is not None and self._replica_group.size > 1:
            all_reduce(shard_grad, self._replica_group, op=op, recorder=self._recorder).wait()

        self._reduction_count += 1
        if shard_grad.dtype != self._master_dtype:
            shard_grad = shard_grad.to(self._master_dtype)
        if self._config.cpu_offload_params:
            shard_grad = shard_grad.to("cpu")
        # The tail of the last rank's shard may cover padding.  Those elements
        # are not parameters, so their gradient must be exactly zero; forcing it
        # keeps the padding out of the global gradient norm.
        self._zero_padding_region(shard_grad)
        return shard_grad

    def _zero_padding_region(self, shard_grad: torch.Tensor) -> None:
        """Zero the part of ``shard_grad`` that corresponds to padding."""
        if not self._padding:
            return
        local = self.local_shard_range()
        padding_range = ShardRange(start=self._total_numel, length=self._padding)
        overlap = intersect_ranges(local, padding_range)
        if overlap is None:
            return
        begin = overlap.start - local.start
        shard_grad[begin : begin + overlap.length].zero_()

    def free_full(self) -> None:
        """Drop the full buffer entirely, after backward has consumed it."""
        if self._full is not None:
            _free_storage(self._full)
        self._full = None
        self._is_sharded = True
        self.unbind_views()

    def set_accumulate_without_reduction(self, value: bool) -> None:
        """Enable/disable ``no_sync`` accumulation for this unit.

        Args:
            value: ``True`` to hold unsharded gradients locally.
        """
        self._accumulate_without_reduction = value

    @property
    def has_pending_accumulation(self) -> bool:
        """Whether unreduced gradients are being held from a ``no_sync`` block."""
        return self._unsharded_grad_accumulator is not None

    @property
    def reduction_count(self) -> int:
        """Number of reduce-scatters this unit has performed."""
        return self._reduction_count

    # -- state dict ---------------------------------------------------------
    def local_piece_layout(self) -> list[PieceLayout]:
        """Describe which elements of which parameters this rank owns.

        This is the bridge between the flat-parameter world (where the unit
        thinks in one contiguous buffer) and the checkpoint world (where every
        tensor has its own global coordinates).  It is pure arithmetic on
        offsets, so it holds no data and can be computed before any tensor
        exists.

        A parameter may be wholly owned, split across the shard boundary, or
        absent from this rank entirely; only the parameters this rank owns at
        least one element of appear.

        Returns:
            One :class:`PieceLayout` per (partially) owned parameter, ordered
            by position in the flat buffer.
        """
        local = self.local_shard_range()
        layout: list[PieceLayout] = []
        for entry in self._entries:
            entry_range = ShardRange(start=entry.offset, length=entry.numel)
            overlap = intersect_ranges(local, entry_range)
            if overlap is None:
                continue
            layout.append(
                PieceLayout(
                    name=entry.name,
                    global_shape=entry.shape,
                    parameter_offset=overlap.start - entry.offset,
                    length=overlap.length,
                    local_offset=overlap.start - local.start,
                )
            )
        return layout

    def local_pieces(self) -> list[ShardedTensorPiece]:
        """Return this rank's slice of each managed parameter, with data.

        Returns:
            One :class:`ShardedTensorPiece` per (partially) owned parameter.
        """
        shard = self.flat_param.detach()
        return [
            ShardedTensorPiece(
                name=item.name,
                global_shape=item.global_shape,
                offset=item.parameter_offset,
                data=shard[item.local_offset : item.local_offset + item.length].clone(),
            )
            for item in self.local_piece_layout()
        ]

    def load_local_pieces(self, pieces: dict[str, torch.Tensor]) -> None:
        """Write full parameter tensors into this rank's shard.

        Args:
            pieces: Mapping from parameter name to the *whole* parameter
                tensor.  The handle extracts the part it owns.

        Raises:
            ShardingError: If a managed parameter is missing, or a supplied
                tensor has the wrong shape.
        """
        local = self.local_shard_range()
        shard = self.flat_param.data
        for entry in self._entries:
            tensor = pieces.get(entry.name)
            if tensor is None:
                raise ShardingError(
                    format_error(
                        "fsdp.load_local_pieces",
                        "missing parameter in the state dict",
                        rank=self._shard_group.global_rank,
                        expected=entry.name,
                        observed=sorted(pieces)[:8],
                        resolution="load a state dict produced by the same model definition",
                    )
                )
            if tuple(tensor.shape) != entry.shape:
                raise ShardingError(
                    format_error(
                        "fsdp.load_local_pieces",
                        f"shape mismatch for {entry.name!r}",
                        rank=self._shard_group.global_rank,
                        expected=entry.shape,
                        observed=tuple(tensor.shape),
                        resolution="the checkpoint was written by a differently shaped model",
                    )
                )
            entry_range = ShardRange(start=entry.offset, length=entry.numel)
            overlap = intersect_ranges(local, entry_range)
            if overlap is None:
                continue
            begin = overlap.start - local.start
            source_begin = overlap.start - entry.offset
            flat_source = tensor.reshape(-1)
            shard[begin : begin + overlap.length].copy_(
                flat_source[source_begin : source_begin + overlap.length].to(
                    dtype=shard.dtype, device=shard.device
                )
            )

    def full_tensors(self) -> dict[str, torch.Tensor]:
        """All-gather and reconstruct every managed parameter.

        Returns:
            Mapping from parameter name to a full, correctly shaped tensor on
            every rank of the shard group.
        """
        shard = self.flat_param.detach().to(self._device).contiguous()
        full, work = all_gather_tensor(shard, self._shard_group, recorder=self._recorder)
        work.wait()
        return {
            entry.name: full[entry.offset : entry.end].view(entry.shape).clone()
            for entry in self._entries
        }

    def memory_summary(self) -> dict[str, int]:
        """Per-rank byte counts for this unit.

        Returns:
            Mapping with ``"shard_bytes"``, ``"full_bytes"`` (0 while
            resharded), ``"grad_shard_bytes"`` and ``"padding_bytes"``.
        """
        element = self.flat_param.element_size()
        grad = self.flat_param.grad
        return {
            "shard_bytes": self._shard_numel * element,
            "full_bytes": 0 if self._is_sharded or self._full is None else self._full_bytes,
            "grad_shard_bytes": 0 if grad is None else grad.numel() * grad.element_size(),
            "padding_bytes": self._padding * element,
        }

    def metadata(self) -> dict[str, Any]:
        """Deterministic description used by the cross-rank consistency check."""
        return {
            "names": list(self._names),
            "entries": [entry.as_dict() for entry in self._entries],
            "total_numel": self._total_numel,
            "padded_numel": self._padded_numel,
            "shard_numel": self._shard_numel,
        }

    def __repr__(self) -> str:
        return (
            f"FlatParamHandle(params={len(self._params)}, total_numel={self._total_numel}, "
            f"shard_numel={self._shard_numel}, padding={self._padding}, "
            f"sharded={self._is_sharded})"
        )


class FullyShardedDataParallel(nn.Module):
    """Shard a module's parameters, gradients and optimizer state.

    This is a from-scratch replacement for
    ``torch.distributed.fsdp.FullyShardedDataParallel``, built only from
    ``all_gather_into_tensor``, ``reduce_scatter_tensor``, ``all_reduce`` and
    ``broadcast``.

    Args:
        module: Module to shard.  Its parameters are consumed: they are
            deregistered and replaced by views onto the flat parameter.
        shard_group: Group across which parameters are split.  **Required.**
        config: FSDP knobs.
        replica_group: Optional outer data-parallel group.  When given, the
            sharded gradient is additionally averaged over it, producing
            hybrid-sharded data parallelism (shard inside, replicate outside).
        mixed_precision: Precision policy.
        device: Compute device.  Defaults to the first parameter's device.
        recorder: Optional communication instrumentation sink.
        sync_module_states: Broadcast parameters from the group source before
            sharding, so every rank shards *the same* weights.  Leave enabled
            unless the caller has already guaranteed identical initialisation.

    Raises:
        ShardingError: If the module has no parameters, or a parameter is tied
            across two different units.
        ParameterConsistencyError: If the flat-parameter layout differs across
            ranks.

    Example:
        >>> # doctest: +SKIP
        >>> model = FullyShardedDataParallel(MLP(cfg), ctx.group("shard"), FSDPConfig())
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        >>> loss = model(x).square().mean()
        >>> loss.backward()          # reduce-scatter happens inside backward
        >>> optimizer.step()         # updates only this rank's shard
    """

    def __init__(
        self,
        module: nn.Module,
        shard_group: GroupHandle,
        config: FSDPConfig | None = None,
        *,
        replica_group: GroupHandle | None = None,
        mixed_precision: MixedPrecisionConfig | None = None,
        device: torch.device | None = None,
        recorder: CommunicationRecorder | None = None,
        sync_module_states: bool = True,
    ) -> None:
        super().__init__()
        self._config = config or FSDPConfig()
        self._mixed_precision = mixed_precision or MixedPrecisionConfig()
        self._shard_group = shard_group
        self._replica_group = replica_group
        self._recorder = recorder

        parameters = list(module.parameters())
        if not parameters:
            raise ShardingError(
                format_error(
                    "fsdp.__init__",
                    "the module has no parameters to shard",
                    rank=shard_group.global_rank,
                    expected=">= 1",
                    observed=0,
                    resolution="wrap a module that has parameters",
                )
            )
        self._device = device if device is not None else parameters[0].device

        if sync_module_states:
            self._broadcast_module_states(module)

        if self._config.auto_wrap_min_num_params > 0:
            module = self._auto_wrap(module)

        named, locations = _collect_managed_parameters(module)
        self.module = module

        self._handle: FlatParamHandle | None = None
        if named:
            self._handle = FlatParamHandle(
                named,
                locations,
                shard_group,
                self._config,
                self._mixed_precision,
                self._device,
                replica_group=replica_group,
                recorder=recorder,
            )
            # Registering the shard as a real parameter is what makes
            # `fsdp_model.parameters()` yield sharded state and nothing else.
            self.register_parameter("flat_param", self._handle.flat_param)
            assert_metadata_consistent(
                self._handle.metadata(),
                shard_group,
                name="the flat parameter layout",
                operation="fsdp.verify_flat_layout",
            )

        self._check_no_cross_unit_tying()

        _LOGGER.info(
            "FSDP unit ready over group %r (size %d): %s",
            shard_group.name,
            shard_group.size,
            "no local parameters" if self._handle is None else repr(self._handle),
        )

    # -- construction helpers ----------------------------------------------
    def _broadcast_module_states(self, module: nn.Module) -> None:
        """Broadcast parameters and buffers from the shard-group source rank.

        FSDP shards *whatever it is given*.  If ranks start from different
        random weights they will shard different models and the all-gathered
        parameter will be a nonsensical mix of two initialisations -- which
        still trains, badly, and is very hard to diagnose.  Broadcasting first
        removes the failure mode.
        """
        for param in module.parameters():
            broadcast(
                param.data, self._shard_group, source_local_rank=0, recorder=self._recorder
            ).wait()
        for buffer in module.buffers():
            if buffer.numel():
                broadcast(
                    buffer.data, self._shard_group, source_local_rank=0, recorder=self._recorder
                ).wait()

    def _auto_wrap(self, module: nn.Module) -> nn.Module:
        """Wrap large submodules as nested FSDP units, bottom-up.

        Wrapping granularity is the main FSDP tuning knob.  One unit for the
        whole model gives the smallest number of collectives and the largest
        transient buffer; one unit per layer gives the opposite.  The threshold
        here selects submodules by parameter count.

        Args:
            module: Module to traverse.

        Returns:
            ``module``, with qualifying children replaced by FSDP units.
        """
        threshold = self._config.auto_wrap_min_num_params
        for name, child in list(module.named_children()):
            if isinstance(child, FullyShardedDataParallel):
                continue
            wrapped_child = self._auto_wrap(child)
            managed, _ = _collect_managed_parameters(wrapped_child)
            direct_numel = sum(param.numel() for _, param in managed)
            if direct_numel >= threshold:
                setattr(
                    module,
                    name,
                    FullyShardedDataParallel(
                        wrapped_child,
                        self._shard_group,
                        self._config_without_auto_wrap(),
                        replica_group=self._replica_group,
                        mixed_precision=self._mixed_precision,
                        device=self._device,
                        recorder=self._recorder,
                        sync_module_states=False,
                    ),
                )
            else:
                setattr(module, name, wrapped_child)
        return module

    def _config_without_auto_wrap(self) -> FSDPConfig:
        """Return a copy of the config with auto-wrapping disabled.

        Nested units must not re-run auto-wrapping, or the recursion would wrap
        each unit's children a second time.
        """
        return dataclasses.replace(self._config, auto_wrap_min_num_params=0)

    def _check_no_cross_unit_tying(self) -> None:
        """Reject a parameter shared between two different FSDP units.

        Two units mean two flat parameters, each of which would receive and
        reduce a *partial* gradient for the shared tensor, and each of which
        would be updated independently -- so the two "copies" would diverge
        immediately.  Supporting this properly requires a shared-parameter
        registry that PyTorch's FSDP also declines to provide in the general
        case.
        """
        seen: dict[int, str] = {}
        for unit in self.fsdp_units():
            if unit._handle is None:
                continue
            for name, param in zip(unit._handle.names, unit._handle.managed_parameters):
                previous = seen.get(id(param))
                if previous is not None and previous != name:
                    raise UnsupportedFeatureError(
                        format_error(
                            "fsdp.check_tied_parameters",
                            "a parameter is shared between two FSDP units; each unit would "
                            "reduce and update its own copy, so the tie would break after "
                            "the first optimizer step",
                            rank=self._shard_group.global_rank,
                            expected="tied parameters inside a single unit",
                            observed=f"{previous!r} and {name!r} are the same tensor",
                            resolution=(
                                "keep tied modules inside one FSDP unit (raise "
                                "auto_wrap_min_num_params), or untie the weights"
                            ),
                        )
                    )
                seen[id(param)] = name

    # -- transparency -------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        """Fall back to the wrapped module for unknown attributes.

        Auto-wrapping **rewrites the caller's module tree**: a nested
        ``nn.Linear`` at ``model.blocks[0].linear`` is replaced in place by a
        :class:`FullyShardedDataParallel` around it.  Without this method, every
        attribute path through a wrapped submodule breaks --
        ``blocks[0].linear.weight`` raises ``AttributeError: 'FullyShardedDataParallel'
        object has no attribute 'weight'`` -- even though nothing about the
        user's model changed.

        That is not a hypothetical: it broke `examples/train_fsdp.py`, whose
        whole point is that ``summon_full_params()`` lets *ordinary* code read
        the weights. Ordinary code does not know about wrappers, so a wrapper
        that changes attribute paths has not made the parameters inspectable;
        it has only moved them. PyTorch's own FSDP forwards for the same reason.

        Args:
            name: Attribute that ordinary lookup did not find.

        Returns:
            The attribute from the wrapped module.

        Raises:
            AttributeError: If neither this wrapper nor the wrapped module has
                it, or if lookup happens before ``self.module`` is registered.
        """
        try:
            # nn.Module's own lookup: parameters, buffers and submodules --
            # including `module` itself, which is registered as a submodule.
            return super().__getattr__(name)
        except AttributeError:
            # Guard against recursing when `module` is what could not be found,
            # which is the case during __init__ and during unpickling.
            if name == "module":
                raise
            wrapped = self.__dict__.get("_modules", {}).get("module")
            if wrapped is None:
                raise
            return getattr(wrapped, name)

    # -- forward / backward -------------------------------------------------
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """All-gather this unit's parameters, run the module, then reshard.

        Args:
            *args: Forwarded positionally.
            **kwargs: Forwarded by keyword.

        Returns:
            Whatever the wrapped module returns.  When
            ``reshard_after_forward`` is enabled and gradients are required,
            tensor outputs are routed through :class:`_PreBackwardUnshard` so
            the parameters come back before this unit's backward runs.
        """
        gathered: torch.Tensor | None = None
        if self._handle is not None:
            gathered = self._handle.unshard()

        output = self.module(*args, **kwargs)

        if self._handle is None:
            return output

        needs_backward = torch.is_grad_enabled() and self._handle.flat_param.requires_grad
        if self._config.reshard_after_forward:
            if needs_backward:
                assert gathered is not None
                output = _apply_pre_backward(self._handle, gathered, output)
            self._handle.reshard()
        elif not needs_backward:
            # Nothing will consume the gathered parameters, so release them.
            self._handle.reshard()
        return output

    @contextmanager
    def no_sync(self) -> Iterator[None]:
        """Accumulate *unsharded* gradients without reducing them.

        The trade-off differs from DDP's.  DDP's ``no_sync`` costs nothing
        extra: gradients were already unsharded.  FSDP's costs one full-size
        gradient buffer per unit, because the accumulated total must stay
        unsharded until it is finally reduce-scattered.  That is why FSDP's
        gradient accumulation saves communication but not memory.

        Yields:
            ``None``.
        """
        units = [u for u in self.fsdp_units() if u._handle is not None]
        for unit in units:
            unit._handle.set_accumulate_without_reduction(True)  # type: ignore[union-attr]
        try:
            yield
        finally:
            for unit in units:
                unit._handle.set_accumulate_without_reduction(False)  # type: ignore[union-attr]

    def finish_backward(self) -> None:
        """Release transient buffers and, optionally, verify reduction order.

        Not required for correctness -- the reduce-scatter already happened
        inside backward -- but it frees the full buffers immediately rather
        than at the next forward, and it runs the cross-rank ordering check
        when ``FSDPConfig.check_reduction_order`` is on.

        Raises:
            ParameterConsistencyError: If ranks reduced their units in
                different orders, which would eventually deadlock.
        """
        for unit in self.fsdp_units():
            if unit._handle is not None:
                unit._handle.free_full()
        if self._config.check_reduction_order:
            order = [
                unit._handle.reduction_count
                for unit in self.fsdp_units()
                if unit._handle is not None
            ]
            gathered = all_gather_object_in_group(order, self._shard_group)
            if any(entry != gathered[0] for entry in gathered):
                raise ParameterConsistencyError(
                    format_error(
                        "fsdp.finish_backward",
                        "ranks performed different numbers of reduce-scatters per unit; "
                        "the collective streams have diverged and the next step will hang",
                        rank=self._shard_group.global_rank,
                        expected=gathered[0],
                        observed=order,
                        resolution=(
                            "make the forward pass structurally identical on every rank "
                            "(no data-dependent module skipping)"
                        ),
                    )
                )

    # -- inspection ---------------------------------------------------------
    def fsdp_units(self) -> list[FullyShardedDataParallel]:
        """Return this unit and every nested unit, outermost first."""
        return [m for m in self.modules() if isinstance(m, FullyShardedDataParallel)]

    @property
    def handle(self) -> FlatParamHandle | None:
        """This unit's flat-parameter handle, or ``None`` if it manages none."""
        return self._handle

    @property
    def config(self) -> FSDPConfig:
        """The active configuration."""
        return self._config

    @property
    def shard_group(self) -> GroupHandle:
        """The group parameters are sharded across."""
        return self._shard_group

    @contextmanager
    def summon_full_params(self, *, writeback: bool = False) -> Iterator[None]:
        """Temporarily materialise every unit's parameters for inspection.

        Inside the block, ``module.weight`` and friends are whole tensors again,
        so ordinary code (printing, norm computation, a non-distributed
        evaluation) works.  Gradients are disabled inside the block because the
        gathered tensors are not part of any graph.

        Args:
            writeback: Copy modifications made inside the block back into the
                shards on exit.  Off by default because silently persisting an
                accidental in-place edit is worse than losing an intentional
                one.

        Yields:
            ``None``.

        Example:
            >>> # doctest: +SKIP
            >>> with fsdp.summon_full_params():
            ...     print(fsdp.module.blocks[0].linear.weight.shape)
        """
        units = [u for u in self.fsdp_units() if u._handle is not None]
        gathered: list[torch.Tensor] = []
        with torch.no_grad():
            for unit in units:
                handle = unit._handle
                assert handle is not None
                full = handle.gather_full(handle.flat_param.detach())
                handle.bind_views(full)
                gathered.append(full)
            try:
                yield
            finally:
                if writeback:
                    for unit, full in zip(units, gathered):
                        handle = unit._handle
                        assert handle is not None
                        local = handle.local_shard_range()
                        handle.flat_param.data.copy_(
                            full[local.start : local.end].to(handle.flat_param.dtype)
                        )
                for unit in units:
                    handle = unit._handle
                    assert handle is not None
                    handle.free_full()

    # -- state dict ---------------------------------------------------------
    def sharded_state_dict(self) -> dict[str, ShardedTensorPiece]:
        """Return this rank's slice of every parameter, in global coordinates.

        This is the checkpoint-facing view.  Keys are the *original* parameter
        names (``"module.blocks.0.linear.weight"``), so a checkpoint does not
        encode the sharding at all -- only which elements this rank happened to
        hold.

        Returns:
            Mapping from parameter name to its owned piece.  Parameters this
            rank owns no elements of are absent.
        """
        result: dict[str, ShardedTensorPiece] = {}
        for unit, prefix in self._units_with_prefix():
            if unit._handle is None:
                continue
            for piece in unit._handle.local_pieces():
                result[prefix + piece.name] = ShardedTensorPiece(
                    name=prefix + piece.name,
                    global_shape=piece.global_shape,
                    offset=piece.offset,
                    data=piece.data,
                )
        return result

    def full_state_dict(self) -> dict[str, torch.Tensor]:
        """All-gather and return the complete, unsharded parameters.

        Every rank in the shard group receives the full dictionary.  Memory
        cost is the whole model on every rank, so this is for small models,
        tests and final exports -- not for checkpointing a large job, which is
        what :meth:`sharded_state_dict` is for.

        Returns:
            Mapping from parameter name to full tensor.
        """
        result: dict[str, torch.Tensor] = {}
        for unit, prefix in self._units_with_prefix():
            if unit._handle is None:
                continue
            for name, tensor in unit._handle.full_tensors().items():
                result[prefix + name] = tensor
        for name, buffer in self.original_named_buffers():
            result[name] = buffer.detach().clone()
        return result

    def load_full_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load whole parameters, keeping only this rank's slice of each.

        Args:
            state_dict: Mapping from parameter name to full tensor, as produced
                by :meth:`full_state_dict` or by an unsharded reference model.

        Raises:
            ShardingError: If a managed parameter is missing or mis-shaped.
        """
        for unit, prefix in self._units_with_prefix():
            if unit._handle is None:
                continue
            local = {
                name: state_dict[prefix + name]
                for name in unit._handle.names
                if prefix + name in state_dict
            }
            unit._handle.load_local_pieces(local)
        for name, buffer in self.original_named_buffers():
            if name in state_dict:
                buffer.data.copy_(state_dict[name].to(buffer.dtype))

    def _units_with_prefix(self) -> list[tuple[FullyShardedDataParallel, str]]:
        """Return each unit with the *wrapper-independent* prefix of its parameters.

        A checkpoint must not encode how the model happened to be wrapped: a run
        with one FSDP unit and a run with per-layer units have to produce the
        same parameter names, or resuming across a wrapping change would fail.
        The traversal therefore skips the synthetic ``.module`` segment each
        FSDP wrapper introduces, so a parameter that the unwrapped model calls
        ``blocks.0.linear.weight`` is called that here too, regardless of how
        many wrappers sit above it.

        Returns:
            ``(unit, prefix)`` pairs, outermost first.
        """
        result: list[tuple[FullyShardedDataParallel, str]] = []

        def walk(module: nn.Module, prefix: str) -> None:
            for name, child in module.named_children():
                child_prefix = f"{prefix}{name}."
                if isinstance(child, FullyShardedDataParallel):
                    result.append((child, child_prefix))
                    walk(child.module, child_prefix)
                else:
                    walk(child, child_prefix)

        result.append((self, ""))
        walk(self.module, "")
        return result

    def optimizer_parameter_layout(self) -> list[tuple[nn.Parameter, list[PieceLayout]]]:
        """Map each flat shard to the parameter pieces it contains.

        The optimizer's state tensors have the same shape as the flat shard, so
        this layout is what lets optimizer state be checkpointed in the *same*
        global coordinates as the parameters -- which in turn is what lets
        optimizer state be resharded across different shard-group sizes.
        Without it, ``exp_avg`` would only be restorable at the world size that
        wrote it.

        The order matches ``self.parameters()``, because both walk
        ``self.modules()``.

        Returns:
            ``(flat_param, layout)`` pairs, with names carrying the
            wrapper-independent prefix.
        """
        prefixes = {id(unit): prefix for unit, prefix in self._units_with_prefix()}
        result: list[tuple[nn.Parameter, list[PieceLayout]]] = []
        for unit in self.fsdp_units():
            handle = unit._handle
            if handle is None:
                continue
            prefix = prefixes[id(unit)]
            result.append(
                (
                    handle.flat_param,
                    [
                        dataclasses.replace(item, name=prefix + item.name)
                        for item in handle.local_piece_layout()
                    ],
                )
            )
        return result

    def original_named_parameters(self) -> dict[str, tuple[tuple[int, ...], nn.Parameter]]:
        """Describe the pre-sharding parameters under wrapper-independent names.

        The returned ``nn.Parameter`` objects are the originals, kept alive for
        the metadata they carry (tensor-parallel markers, dtype).  Their
        ``.data`` has been emptied, which is why the shape is returned
        separately from the layout the handle recorded.

        Returns:
            Mapping from parameter name to ``(shape, parameter_object)``.
        """
        result: dict[str, tuple[tuple[int, ...], nn.Parameter]] = {}
        for unit, prefix in self._units_with_prefix():
            if unit._handle is None:
                continue
            for entry, param in zip(unit._handle.entries, unit._handle.managed_parameters):
                result[prefix + entry.name] = (entry.shape, param)
        return result

    def original_named_buffers(self) -> list[tuple[str, torch.Tensor]]:
        """Buffers under their wrapper-independent names.

        Returns:
            ``(name, buffer)`` pairs using the naming the *unwrapped* model
            would have produced.
        """
        found: list[tuple[str, torch.Tensor]] = []

        def walk(module: nn.Module, prefix: str) -> None:
            for name, buffer in module.named_buffers(recurse=False):
                if buffer is not None:
                    found.append((prefix + name, buffer))
            for name, child in module.named_children():
                child_prefix = f"{prefix}{name}."
                walk(
                    child.module if isinstance(child, FullyShardedDataParallel) else child,
                    child_prefix,
                )

        walk(self.module, "")
        return found

    # -- memory -------------------------------------------------------------
    def memory_summary(self) -> dict[str, int]:
        """Aggregate per-rank byte counts across every unit.

        Returns:
            Mapping with ``"shard_bytes"``, ``"full_bytes"``,
            ``"grad_shard_bytes"``, ``"padding_bytes"`` and ``"units"``.
        """
        total = {
            "shard_bytes": 0,
            "full_bytes": 0,
            "grad_shard_bytes": 0,
            "padding_bytes": 0,
            "units": 0,
        }
        for unit in self.fsdp_units():
            if unit._handle is None:
                continue
            summary = unit._handle.memory_summary()
            for key, value in summary.items():
                total[key] += value
            total["units"] += 1
        return total

    def __repr__(self) -> str:
        return (
            f"FullyShardedDataParallel(shard_group={self._shard_group.name!r}, "
            f"size={self._shard_group.size}, units={len(self.fsdp_units())}, "
            f"reshard_after_forward={self._config.reshard_after_forward})"
        )


def _collect_managed_parameters(
    root: nn.Module,
) -> tuple[list[tuple[str, nn.Parameter]], dict[int, list[tuple[nn.Module, str]]]]:
    """Find the parameters a unit owns and where each one is registered.

    Traversal stops at nested :class:`FullyShardedDataParallel` boundaries,
    because those units manage their own parameters.

    Args:
        root: The module being wrapped.

    Returns:
        ``(named_parameters, locations)`` where ``named_parameters`` is a
        deduplicated, deterministically ordered list and ``locations`` maps a
        parameter's identity to every ``(module, attribute)`` it is registered
        under -- more than one for a tied weight.
    """
    named: list[tuple[str, nn.Parameter]] = []
    locations: dict[int, list[tuple[nn.Module, str]]] = {}

    def visit(module: nn.Module, prefix: str) -> None:
        for attribute, param in list(module._parameters.items()):
            if param is None:
                continue
            key = id(param)
            if key not in locations:
                locations[key] = []
                named.append((prefix + attribute, param))
            locations[key].append((module, attribute))
        for child_name, child in module.named_children():
            if isinstance(child, FullyShardedDataParallel):
                continue
            visit(child, f"{prefix}{child_name}.")

    visit(root, "")
    return named, locations


def _apply_pre_backward(handle: FlatParamHandle, full: torch.Tensor, output: Any) -> Any:
    """Route every differentiable tensor in ``output`` through the re-gather hook.

    Args:
        handle: The unit whose parameters must be restored in backward.
        full: The gathered buffer this forward pass bound its views to.
        output: The module's output; a tensor, or an arbitrarily nested
            tuple/list/dict of tensors.

    Returns:
        The output with tensors replaced by hooked equivalents.
    """
    if isinstance(output, torch.Tensor):
        if not output.requires_grad:
            return output
        return _PreBackwardUnshard.apply(handle, full, output)

    if isinstance(output, dict):
        items: list[tuple[Any, Any]] = list(output.items())
    elif isinstance(output, tuple | list):
        items = list(enumerate(output))
    else:
        # A unit whose output holds no differentiable tensor cannot trigger a
        # backward pass through its parameters, so there is nothing to restore.
        return output

    differentiable = [
        key for key, value in items if isinstance(value, torch.Tensor) and value.requires_grad
    ]
    if not differentiable:
        return output

    hooked = _PreBackwardUnshard.apply(handle, full, *(dict(items)[k] for k in differentiable))
    if isinstance(hooked, torch.Tensor):
        hooked = (hooked,)

    rebuilt = dict(items)
    for position, key in enumerate(differentiable):
        rebuilt[key] = hooked[position]

    if isinstance(output, dict):
        return type(output)(rebuilt)
    values = [rebuilt[key] for key, _ in items]
    return tuple(values) if isinstance(output, tuple) else values


def reshard_state_dict_pieces(
    pieces: Sequence[ShardedTensorPiece], global_shape: tuple[int, ...]
) -> torch.Tensor:
    """Reassemble a whole parameter from pieces that cover it.

    Used by the checkpoint reader when the saving and loading world sizes
    differ: the pieces come from *whatever* ranks happened to hold them, and
    are placed by their global offsets.

    Args:
        pieces: Pieces of one parameter, in any order.  Together they must
            cover ``[0, prod(global_shape))`` without gaps.
        global_shape: Shape of the reassembled tensor.

    Returns:
        The reassembled tensor.

    Raises:
        ShardingError: If the pieces leave a gap or overrun the parameter.
    """
    total = 1
    for dimension in global_shape:
        total *= dimension
    if not pieces:
        raise ShardingError(
            format_error(
                "fsdp.reshard_state_dict_pieces",
                "no pieces supplied",
                expected=">= 1 piece",
                observed=0,
                resolution="check that the manifest lists shards for this tensor",
            )
        )
    flat = torch.zeros(total, dtype=pieces[0].data.dtype)
    covered = torch.zeros(total, dtype=torch.bool)
    for piece in pieces:
        end = piece.offset + piece.length
        if end > total:
            raise ShardingError(
                format_error(
                    "fsdp.reshard_state_dict_pieces",
                    f"piece for {piece.name!r} runs past the end of the tensor",
                    expected=f"offset + length <= {total}",
                    observed=end,
                    resolution="the manifest disagrees with the model definition",
                )
            )
        flat[piece.offset : end] = piece.data.reshape(-1).to(flat.dtype)
        covered[piece.offset : end] = True
    if not bool(covered.all()):
        missing = int((~covered).sum().item())
        raise ShardingError(
            format_error(
                "fsdp.reshard_state_dict_pieces",
                f"the supplied pieces leave {missing} element(s) of {pieces[0].name!r} uncovered",
                expected=f"{total} covered elements",
                observed=total - missing,
                resolution="a shard file is missing from the checkpoint",
            )
        )
    return flat.view(global_shape)
