"""Structured configuration for every subsystem.

Everything the framework does is driven by frozen dataclasses.  There are three
reasons the project does not use raw dictionaries or ``argparse`` namespaces:

1. **Cross-rank determinism.**  A configuration object is hashed into the
   checkpoint manifest and compared across ranks.  Dataclasses give a stable
   field order and a canonical ``asdict`` representation; dictionaries do not.
2. **Validation lives next to the data.**  Each dataclass validates itself in
   ``__post_init__`` so an illegal configuration fails at construction time on
   *every* rank simultaneously, rather than at the first collective where one
   rank behaves differently from the others.
3. **Documentation.**  Every field is a documented, typed, defaulted knob.

YAML files under ``configs/`` map 1:1 onto :class:`ExperimentConfig`.
Unknown keys are rejected rather than ignored -- a silently ignored typo in a
config file is one of the most expensive kinds of bug in a distributed job.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin

import torch
import yaml

from .errors import ConfigurationError, TopologyError, format_error

__all__ = [
    "CheckpointConfig",
    "DDPConfig",
    "DataConfig",
    "ExperimentConfig",
    "FSDPConfig",
    "GradScalerConfig",
    "MixedPrecisionConfig",
    "ModelConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "SequenceParallelMode",
    "TensorParallelConfig",
    "TopologyConfig",
    "TrainingConfig",
    "load_experiment_config",
    "resolve_dtype",
]

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float64": torch.float64,
    "fp64": torch.float64,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def resolve_dtype(name: str | torch.dtype) -> torch.dtype:
    """Map a dtype name onto a :class:`torch.dtype`.

    Args:
        name: Either an actual ``torch.dtype`` (returned unchanged) or one of
            ``float32``/``fp32``/``float64``/``fp64``/``float16``/``fp16``/
            ``half``/``bfloat16``/``bf16``.

    Returns:
        The resolved dtype.

    Raises:
        ConfigurationError: If the name is unknown.
    """
    if isinstance(name, torch.dtype):
        return name
    key = str(name).lower().replace("torch.", "")
    if key not in _DTYPES:
        raise ConfigurationError(
            format_error(
                "config.resolve_dtype",
                "unknown dtype name",
                expected=sorted(_DTYPES),
                observed=name,
                resolution="use one of the listed dtype names",
            )
        )
    return _DTYPES[key]


class SequenceParallelMode:
    """Enumeration of how the sequence-parallel dimension relates to others.

    This is a plain class of string constants rather than an ``enum.Enum`` so
    that YAML round-trips without custom representers and so the values appear
    verbatim in checkpoint manifests.

    Attributes:
        DISABLED: No sequence parallelism.  ``sequence_parallel_size`` must be
            ``1``.
        TENSOR_GROUP: Megatron-style fusion.  Sequence parallelism reuses the
            *tensor*-parallel process group, so activations are split along the
            sequence dimension exactly where they would otherwise be
            replicated.  ``sequence_parallel_size`` stays ``1`` in the world
            size product because the ranks are already accounted for by
            ``tensor_parallel_size``.
        INDEPENDENT: Sequence parallelism gets its own topology dimension and
            its own process group.  Useful for teaching the mechanism in
            isolation, and for splitting the sequence dimension *further* than
            the tensor-parallel width.
    """

    DISABLED = "disabled"
    TENSOR_GROUP = "tensor_group"
    INDEPENDENT = "independent"

    ALL = (DISABLED, TENSOR_GROUP, INDEPENDENT)


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TopologyConfig:
    """Sizes of the four logical parallelism dimensions.

    The world size must factor exactly::

        world_size = data_parallel_size
                   * shard_parallel_size
                   * sequence_parallel_size
                   * tensor_parallel_size

    Attributes:
        data_parallel_size: Number of *replicas*.  Parameters are replicated
            across this dimension and gradients are all-reduced over it.
        shard_parallel_size: Number of FSDP-style *shards*.  Parameters,
            gradients and optimizer states are split across this dimension.
            Combining ``data_parallel_size > 1`` with
            ``shard_parallel_size > 1`` yields hybrid-sharded data parallelism
            (shard within the inner group, replicate across the outer one).
        sequence_parallel_size: Size of the standalone sequence-parallel
            dimension.  Must be ``1`` unless
            ``sequence_parallel_mode == "independent"``.
        tensor_parallel_size: Width of the tensor-parallel groups.  Weight
            matrices are partitioned across this dimension.
        sequence_parallel_mode: See :class:`SequenceParallelMode`.

    Raises:
        TopologyError: If any size is not a positive integer, or if
            ``sequence_parallel_size`` disagrees with
            ``sequence_parallel_mode``.
    """

    data_parallel_size: int = 1
    shard_parallel_size: int = 1
    sequence_parallel_size: int = 1
    tensor_parallel_size: int = 1
    sequence_parallel_mode: str = SequenceParallelMode.DISABLED

    def __post_init__(self) -> None:
        for name in (
            "data_parallel_size",
            "shard_parallel_size",
            "sequence_parallel_size",
            "tensor_parallel_size",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise TopologyError(
                    format_error(
                        "config.TopologyConfig",
                        f"{name} must be a positive integer",
                        expected=">= 1",
                        observed=value,
                        resolution=f"set {name} to a positive integer",
                    )
                )

        if self.sequence_parallel_mode not in SequenceParallelMode.ALL:
            raise TopologyError(
                format_error(
                    "config.TopologyConfig",
                    "unknown sequence_parallel_mode",
                    expected=list(SequenceParallelMode.ALL),
                    observed=self.sequence_parallel_mode,
                    resolution="choose one of the listed modes",
                )
            )

        if (
            self.sequence_parallel_mode != SequenceParallelMode.INDEPENDENT
            and self.sequence_parallel_size != 1
        ):
            raise TopologyError(
                format_error(
                    "config.TopologyConfig",
                    "sequence_parallel_size > 1 requires mode 'independent'; in "
                    "'tensor_group' mode the sequence dimension reuses the "
                    "tensor-parallel ranks and therefore consumes no extra ranks",
                    expected=1,
                    observed=self.sequence_parallel_size,
                    resolution=(
                        "either set sequence_parallel_mode='independent', or leave "
                        "sequence_parallel_size=1 and rely on tensor_parallel_size"
                    ),
                )
            )

        if (
            self.sequence_parallel_mode == SequenceParallelMode.TENSOR_GROUP
            and self.tensor_parallel_size == 1
        ):
            raise TopologyError(
                format_error(
                    "config.TopologyConfig",
                    "sequence_parallel_mode='tensor_group' needs a tensor-parallel "
                    "group wider than one rank, because that group is what the "
                    "sequence dimension is split over",
                    expected="tensor_parallel_size > 1",
                    observed=self.tensor_parallel_size,
                    resolution="raise tensor_parallel_size or disable sequence parallelism",
                )
            )

    @property
    def world_size(self) -> int:
        """Number of ranks this topology describes."""
        return (
            self.data_parallel_size
            * self.shard_parallel_size
            * self.sequence_parallel_size
            * self.tensor_parallel_size
        )

    @property
    def effective_sequence_parallel_size(self) -> int:
        """How many ranks the sequence dimension is actually split over.

        In ``tensor_group`` mode this is the tensor-parallel size even though
        ``sequence_parallel_size`` is ``1``.
        """
        if self.sequence_parallel_mode == SequenceParallelMode.TENSOR_GROUP:
            return self.tensor_parallel_size
        if self.sequence_parallel_mode == SequenceParallelMode.INDEPENDENT:
            return self.sequence_parallel_size
        return 1

    @property
    def sequence_parallel_enabled(self) -> bool:
        """``True`` when activations are split along the sequence dimension."""
        return (
            self.sequence_parallel_mode != SequenceParallelMode.DISABLED
            and self.effective_sequence_parallel_size > 1
        )

    def validate_against_world_size(self, world_size: int) -> None:
        """Check that this topology exactly covers ``world_size`` ranks.

        Args:
            world_size: The actual number of processes in the job.

        Raises:
            TopologyError: If the product of the dimensions differs from
                ``world_size``.
        """
        if self.world_size != world_size:
            raise TopologyError(
                format_error(
                    "config.TopologyConfig.validate_against_world_size",
                    "parallel dimensions do not factor the world size "
                    f"(dp={self.data_parallel_size} x shard={self.shard_parallel_size} "
                    f"x seq={self.sequence_parallel_size} x tensor={self.tensor_parallel_size})",
                    world_size=world_size,
                    expected=world_size,
                    observed=self.world_size,
                    resolution=(
                        "adjust the topology so the four sizes multiply to the number "
                        "of launched processes, or launch a different number of processes"
                    ),
                )
            )

    @classmethod
    def for_world_size(cls, world_size: int, **overrides: Any) -> TopologyConfig:
        """Build a topology that fills ``world_size``, inferring one dimension.

        Exactly one of the four dimensions may be omitted; it is inferred as
        ``world_size / product(of the others)``.  When *all* dimensions are
        supplied the result is validated instead of inferred.  When *none* are
        supplied the whole world becomes data parallel.

        Args:
            world_size: Number of processes.
            **overrides: Any subset of the :class:`TopologyConfig` fields.

        Returns:
            A validated topology.

        Raises:
            TopologyError: If the supplied dimensions do not divide
                ``world_size``, or if more than one dimension is missing and
                the remainder is ambiguous.
        """
        dims = (
            "data_parallel_size",
            "shard_parallel_size",
            "sequence_parallel_size",
            "tensor_parallel_size",
        )
        given = {k: v for k, v in overrides.items() if k in dims}
        missing = [d for d in dims if d not in given]
        product = math.prod(given.values()) if given else 1

        if world_size % product != 0:
            raise TopologyError(
                format_error(
                    "config.TopologyConfig.for_world_size",
                    "the supplied parallel dimensions do not divide the world size",
                    world_size=world_size,
                    expected=f"world_size divisible by {product}",
                    observed=world_size,
                    resolution="pick dimension sizes whose product divides the world size",
                )
            )
        remainder = world_size // product

        resolved = dict(overrides)
        if remainder != 1:
            if not missing:
                raise TopologyError(
                    format_error(
                        "config.TopologyConfig.for_world_size",
                        "all four dimensions were given but their product is smaller "
                        "than the world size",
                        world_size=world_size,
                        expected=world_size,
                        observed=product,
                        resolution="increase one dimension or launch fewer processes",
                    )
                )
            # Fill the first missing dimension, defaulting the rest to 1.  The
            # order of `dims` makes data parallelism the default sink, which is
            # the least surprising behaviour.
            resolved[missing[0]] = remainder
        for name in missing[1:] if remainder != 1 else missing:
            resolved.setdefault(name, 1)

        config = cls(**resolved)
        config.validate_against_world_size(world_size)
        return config


# ---------------------------------------------------------------------------
# Parallel strategy knobs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DDPConfig:
    """Knobs for :class:`hybrid_training.parallel.ddp.DistributedDataParallel`.

    Attributes:
        bucket_cap_mb: Soft cap on the size of a gradient bucket, in mebibytes.
            Larger buckets mean fewer, bigger collectives (better bandwidth
            utilisation, worse overlap); smaller buckets mean the first
            all-reduce can start earlier.  ``25`` matches PyTorch's default.
        broadcast_buffers: Broadcast non-parameter buffers (e.g. BatchNorm
            running stats) from the source rank at the start of every forward.
            Disable it when buffers are intentionally rank-local.
        find_unused_parameters: Tolerate parameters that receive no gradient in
            a given iteration by contributing explicit zeros for them.  Costs
            one extra pass over the parameter list per step; leave it off when
            the graph is static.
        average_gradients: Divide the summed gradients by the group size.  When
            ``False`` the gradients are summed and the caller is responsible
            for scaling the learning rate.
        async_reduction: Launch bucket all-reduces asynchronously so that
            communication overlaps the remaining backward computation.  Set to
            ``False`` to get a strictly serial, easier-to-debug schedule.
        check_parameter_consistency: Run a collective start-up check that every
            rank has the same parameter names, shapes and dtypes in the same
            order.
        source_rank_in_group: Group-local rank whose parameters are broadcast
            to the rest of the group at construction time.
    """

    bucket_cap_mb: float = 25.0
    broadcast_buffers: bool = True
    find_unused_parameters: bool = False
    average_gradients: bool = True
    async_reduction: bool = True
    check_parameter_consistency: bool = True
    source_rank_in_group: int = 0

    def __post_init__(self) -> None:
        if self.bucket_cap_mb <= 0:
            raise ConfigurationError(
                format_error(
                    "config.DDPConfig",
                    "bucket_cap_mb must be positive",
                    expected="> 0",
                    observed=self.bucket_cap_mb,
                    resolution="use a positive bucket size, e.g. 25.0",
                )
            )
        if self.source_rank_in_group < 0:
            raise ConfigurationError(
                format_error(
                    "config.DDPConfig",
                    "source_rank_in_group must be non-negative",
                    expected=">= 0",
                    observed=self.source_rank_in_group,
                    resolution="use a group-local rank index",
                )
            )


@dataclass(frozen=True)
class FSDPConfig:
    """Knobs for :class:`hybrid_training.parallel.fsdp.FullyShardedDataParallel`.

    Attributes:
        reshard_after_forward: Free the all-gathered full parameters as soon as
            the forward pass of a unit finishes, and gather them again in
            backward.  This is the memory/communication trade-off at the heart
            of FSDP: ``True`` halves peak parameter memory at the cost of one
            extra all-gather per unit per step.
        average_gradients: Divide reduce-scattered gradients by the shard-group
            size (and by the replica-group size when hybrid sharding).
        auto_wrap_min_num_params: When greater than zero, submodules with at
            least this many parameters become their own FSDP unit.  ``0``
            wraps the whole module as a single unit.
        cpu_offload_params: Keep the persistent parameter shard in pinned CPU
            memory and copy it to the device only while it is needed.
        limit_all_gather_bytes: Guard rail that refuses to materialise a single
            flat parameter larger than this many bytes.  ``0`` disables the
            check.  Useful in teaching settings to make an accidental
            "wrap the whole 7B model as one unit" fail loudly instead of OOM.
        check_reduction_order: Debug mode that all-gathers the sequence of
            reduced units at the end of every backward pass and verifies every
            rank reduced in the same order.  Costs one tiny collective per
            step; catches the class of bug that otherwise shows up as a hang.
        use_padding: Pad each flat parameter up to a multiple of the shard
            group size.  Always ``True`` in this implementation; exposed so the
            documentation can point at a single named invariant.
    """

    reshard_after_forward: bool = True
    average_gradients: bool = True
    auto_wrap_min_num_params: int = 0
    cpu_offload_params: bool = False
    limit_all_gather_bytes: int = 0
    check_reduction_order: bool = False
    use_padding: bool = True

    def __post_init__(self) -> None:
        if self.auto_wrap_min_num_params < 0:
            raise ConfigurationError(
                format_error(
                    "config.FSDPConfig",
                    "auto_wrap_min_num_params must be non-negative",
                    expected=">= 0",
                    observed=self.auto_wrap_min_num_params,
                    resolution="use 0 to disable automatic wrapping",
                )
            )
        if self.limit_all_gather_bytes < 0:
            raise ConfigurationError(
                format_error(
                    "config.FSDPConfig",
                    "limit_all_gather_bytes must be non-negative",
                    expected=">= 0",
                    observed=self.limit_all_gather_bytes,
                    resolution="use 0 to disable the guard rail",
                )
            )
        if not self.use_padding:
            raise ConfigurationError(
                format_error(
                    "config.FSDPConfig",
                    "use_padding=False is not supported: unpadded flat parameters "
                    "make the local shard size rank-dependent, which breaks the "
                    "single-collective all-gather/reduce-scatter fast path",
                    expected=True,
                    observed=False,
                    resolution="leave use_padding at its default of True",
                )
            )


@dataclass(frozen=True)
class TensorParallelConfig:
    """Knobs for the tensor-parallel layers.

    Attributes:
        gather_output: Whether a :class:`ColumnParallelLinear` at the *end* of
            a model gathers its shards back into a full-width output.  Inside a
            transformer block the answer is ``False`` (the following
            row-parallel layer consumes the shards directly); at the model
            output it is usually ``True``.
        sequence_parallel: Enable the Megatron-style fused tensor+sequence
            parallel schedule, where the region between two tensor-parallel
            layers keeps activations split along the sequence dimension.
        async_input_gradient_allreduce: Overlap the input-gradient all-reduce
            of a column-parallel layer with the weight-gradient matmul.
        init_method_std: Standard deviation used to initialise partitioned
            weights.  Every rank initialises its own slice from a *seeded,
            rank-offset* generator so the concatenation matches what a single
            process would have produced for the same seed and shape.
    """

    gather_output: bool = False
    sequence_parallel: bool = False
    async_input_gradient_allreduce: bool = True
    init_method_std: float = 0.02

    def __post_init__(self) -> None:
        if self.init_method_std <= 0:
            raise ConfigurationError(
                format_error(
                    "config.TensorParallelConfig",
                    "init_method_std must be positive",
                    expected="> 0",
                    observed=self.init_method_std,
                    resolution="use a small positive value such as 0.02",
                )
            )


@dataclass(frozen=True)
class GradScalerConfig:
    """Loss-scaling knobs for fp16 training.

    bf16 does not need loss scaling (it has the same exponent range as fp32),
    so ``enabled`` should stay ``False`` there.

    Attributes:
        enabled: Turn dynamic loss scaling on.
        init_scale: Starting loss scale.
        growth_factor: Multiplier applied after ``growth_interval`` consecutive
            finite steps.
        backoff_factor: Multiplier applied immediately after a non-finite step.
        growth_interval: Number of consecutive finite steps before growth.
        max_scale: Upper bound on the loss scale.
    """

    enabled: bool = False
    init_scale: float = 65536.0
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    max_scale: float = 2.0**24

    def __post_init__(self) -> None:
        if self.init_scale <= 0:
            raise ConfigurationError(
                format_error(
                    "config.GradScalerConfig",
                    "init_scale must be positive",
                    expected="> 0",
                    observed=self.init_scale,
                    resolution="use a power of two such as 65536",
                )
            )
        if not self.growth_factor > 1.0:
            raise ConfigurationError(
                format_error(
                    "config.GradScalerConfig",
                    "growth_factor must exceed 1",
                    expected="> 1.0",
                    observed=self.growth_factor,
                    resolution="use 2.0",
                )
            )
        if not 0.0 < self.backoff_factor < 1.0:
            raise ConfigurationError(
                format_error(
                    "config.GradScalerConfig",
                    "backoff_factor must lie in (0, 1)",
                    expected="0 < x < 1",
                    observed=self.backoff_factor,
                    resolution="use 0.5",
                )
            )
        if self.growth_interval < 1:
            raise ConfigurationError(
                format_error(
                    "config.GradScalerConfig",
                    "growth_interval must be at least 1",
                    expected=">= 1",
                    observed=self.growth_interval,
                    resolution="use 2000",
                )
            )


@dataclass(frozen=True)
class MixedPrecisionConfig:
    """Mixed-precision policy shared by DDP, FSDP and the tensor-parallel layers.

    Attributes:
        enabled: Master switch.  When ``False`` every dtype below is ignored
            and computation runs in the model's own dtype.
        param_dtype: Dtype that all-gathered parameters are cast to for
            compute.  The *persistent* shard always stays in ``master_dtype``.
        reduce_dtype: Dtype used for gradient reductions.  Reducing in fp32
            while computing in bf16 is the standard recipe: it keeps the
            reduction numerically stable without paying fp32 compute cost.
        buffer_dtype: Dtype for non-parameter buffers.
        master_dtype: Dtype of the persistent parameter shard and optimizer
            state ("master weights").
        scaler: Loss-scaling policy.
    """

    enabled: bool = False
    param_dtype: str = "bfloat16"
    reduce_dtype: str = "float32"
    buffer_dtype: str = "float32"
    master_dtype: str = "float32"
    scaler: GradScalerConfig = field(default_factory=GradScalerConfig)

    def __post_init__(self) -> None:
        # Resolving eagerly turns a typo into a construction-time error.
        for name in ("param_dtype", "reduce_dtype", "buffer_dtype", "master_dtype"):
            resolve_dtype(getattr(self, name))
        if (
            self.enabled
            and self.scaler.enabled
            and resolve_dtype(self.param_dtype) is not torch.float16
        ):
            raise ConfigurationError(
                format_error(
                    "config.MixedPrecisionConfig",
                    "dynamic loss scaling is only meaningful for float16 compute; "
                    "bfloat16 has the same exponent range as float32 and does not "
                    "underflow in the same way",
                    expected="param_dtype=float16 when scaler.enabled",
                    observed=self.param_dtype,
                    resolution="either set param_dtype=float16 or scaler.enabled=false",
                )
            )

    @property
    def compute_dtype(self) -> torch.dtype:
        """Dtype used for forward/backward compute."""
        return resolve_dtype(self.param_dtype) if self.enabled else resolve_dtype(self.master_dtype)

    @property
    def gradient_reduce_dtype(self) -> torch.dtype:
        """Dtype used for gradient reductions."""
        return (
            resolve_dtype(self.reduce_dtype) if self.enabled else resolve_dtype(self.master_dtype)
        )


# ---------------------------------------------------------------------------
# Model / data / optimisation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    """Reference-model description.

    Attributes:
        kind: ``"mlp"`` or ``"transformer"``.
        hidden_size: Width of the MLP hidden layers / transformer ``d_model``.
        num_layers: Number of MLP blocks or transformer blocks.
        input_size: MLP input width (ignored by the transformer).
        output_size: MLP output width (ignored by the transformer).
        vocab_size: Transformer vocabulary size.
        num_heads: Transformer attention heads.  Must divide ``hidden_size``,
            and ``num_heads`` must be divisible by the tensor-parallel size.
        ffn_hidden_size: Transformer feed-forward width.  ``0`` means
            ``4 * hidden_size``.
        max_sequence_length: Longest sequence the positional table supports.
        dropout: Dropout probability.  Kept at ``0`` in tests so that runs are
            bitwise reproducible without RNG-state juggling across ranks.
        layer_norm_eps: Epsilon inside LayerNorm.
        tie_word_embeddings: Share the embedding matrix with the output
            projection.  Documented as unsupported under FSDP-style sharding
            when the two ends land in different units.
        activation: ``"gelu"`` or ``"relu"``.
    """

    kind: str = "mlp"
    hidden_size: int = 64
    num_layers: int = 2
    input_size: int = 32
    output_size: int = 8
    vocab_size: int = 256
    num_heads: int = 4
    ffn_hidden_size: int = 0
    max_sequence_length: int = 64
    dropout: float = 0.0
    layer_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False
    activation: str = "gelu"

    def __post_init__(self) -> None:
        if self.kind not in {"mlp", "transformer"}:
            raise ConfigurationError(
                format_error(
                    "config.ModelConfig",
                    "unknown model kind",
                    expected=["mlp", "transformer"],
                    observed=self.kind,
                    resolution="use 'mlp' or 'transformer'",
                )
            )
        if self.activation not in {"gelu", "relu"}:
            raise ConfigurationError(
                format_error(
                    "config.ModelConfig",
                    "unknown activation",
                    expected=["gelu", "relu"],
                    observed=self.activation,
                    resolution="use 'gelu' or 'relu'",
                )
            )
        for name in (
            "hidden_size",
            "num_layers",
            "vocab_size",
            "num_heads",
            "max_sequence_length",
            "input_size",
            "output_size",
        ):
            if getattr(self, name) < 1:
                raise ConfigurationError(
                    format_error(
                        "config.ModelConfig",
                        f"{name} must be positive",
                        expected=">= 1",
                        observed=getattr(self, name),
                        resolution=f"set {name} to a positive integer",
                    )
                )
        if self.kind == "transformer" and self.hidden_size % self.num_heads != 0:
            raise ConfigurationError(
                format_error(
                    "config.ModelConfig",
                    "hidden_size must be divisible by num_heads",
                    expected=f"hidden_size % {self.num_heads} == 0",
                    observed=self.hidden_size,
                    resolution="pick a hidden size that is a multiple of num_heads",
                )
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigurationError(
                format_error(
                    "config.ModelConfig",
                    "dropout must lie in [0, 1)",
                    expected="0 <= p < 1",
                    observed=self.dropout,
                    resolution="use 0.0 for reproducible tests",
                )
            )

    @property
    def resolved_ffn_hidden_size(self) -> int:
        """Feed-forward width, resolving the ``0`` sentinel to ``4 * hidden_size``."""
        return self.ffn_hidden_size if self.ffn_hidden_size > 0 else 4 * self.hidden_size

    @property
    def head_dim(self) -> int:
        """Per-head dimension."""
        return self.hidden_size // self.num_heads


@dataclass(frozen=True)
class DataConfig:
    """Synthetic-dataset description.

    The project ships a deterministic synthetic dataset rather than a real
    corpus so that "same seed, same data, same result" holds exactly, which is
    what the numerical-equivalence tests depend on.

    Attributes:
        micro_batch_size: Samples per *data-parallel* rank per micro-step.
        sequence_length: Tokens per sample (transformer only).
        num_train_samples: Size of the synthetic training set.
        num_eval_samples: Size of the synthetic evaluation set.
        seed: Dataset seed.  Distinct from the model-init seed so data order
            and weight init can be varied independently.
        shuffle: Shuffle sample order each epoch.
    """

    micro_batch_size: int = 4
    sequence_length: int = 16
    num_train_samples: int = 256
    num_eval_samples: int = 64
    seed: int = 1234
    shuffle: bool = True

    def __post_init__(self) -> None:
        for name in ("micro_batch_size", "sequence_length", "num_train_samples"):
            if getattr(self, name) < 1:
                raise ConfigurationError(
                    format_error(
                        "config.DataConfig",
                        f"{name} must be positive",
                        expected=">= 1",
                        observed=getattr(self, name),
                        resolution=f"set {name} to a positive integer",
                    )
                )
        if self.num_eval_samples < 0:
            raise ConfigurationError(
                format_error(
                    "config.DataConfig",
                    "num_eval_samples must be non-negative",
                    expected=">= 0",
                    observed=self.num_eval_samples,
                    resolution="use 0 to disable evaluation",
                )
            )


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer hyper-parameters.

    Attributes:
        name: ``"adamw"`` or ``"sgd"``.
        learning_rate: Peak learning rate.
        weight_decay: Decoupled weight decay for AdamW, L2 for SGD.
        betas: AdamW moment decay rates.
        eps: AdamW epsilon.
        momentum: SGD momentum.
        cpu_offload_state: Keep optimizer state (and the master shard copy) in
            CPU memory, stepping on the CPU and copying the updated shard back
            to the device.  Trades step latency for device memory.
    """

    name: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    momentum: float = 0.0
    cpu_offload_state: bool = False

    def __post_init__(self) -> None:
        if self.name not in {"adamw", "sgd"}:
            raise ConfigurationError(
                format_error(
                    "config.OptimizerConfig",
                    "unknown optimizer",
                    expected=["adamw", "sgd"],
                    observed=self.name,
                    resolution="use 'adamw' or 'sgd'",
                )
            )
        if self.learning_rate <= 0:
            raise ConfigurationError(
                format_error(
                    "config.OptimizerConfig",
                    "learning_rate must be positive",
                    expected="> 0",
                    observed=self.learning_rate,
                    resolution="use a positive learning rate",
                )
            )
        betas = tuple(self.betas)
        if len(betas) != 2 or not all(0.0 <= b < 1.0 for b in betas):
            raise ConfigurationError(
                format_error(
                    "config.OptimizerConfig",
                    "betas must be two values in [0, 1)",
                    expected="(b1, b2) with 0 <= b < 1",
                    observed=self.betas,
                    resolution="use (0.9, 0.999)",
                )
            )
        object.__setattr__(self, "betas", betas)


@dataclass(frozen=True)
class SchedulerConfig:
    """Learning-rate schedule.

    Attributes:
        name: ``"constant"``, ``"linear"`` or ``"cosine"``.
        warmup_steps: Linear warm-up length in optimizer steps.
        min_lr_ratio: Floor of the decay, as a fraction of the peak LR.
    """

    name: str = "constant"
    warmup_steps: int = 0
    min_lr_ratio: float = 0.0

    def __post_init__(self) -> None:
        if self.name not in {"constant", "linear", "cosine"}:
            raise ConfigurationError(
                format_error(
                    "config.SchedulerConfig",
                    "unknown schedule",
                    expected=["constant", "linear", "cosine"],
                    observed=self.name,
                    resolution="use one of the listed schedules",
                )
            )
        if self.warmup_steps < 0:
            raise ConfigurationError(
                format_error(
                    "config.SchedulerConfig",
                    "warmup_steps must be non-negative",
                    expected=">= 0",
                    observed=self.warmup_steps,
                    resolution="use 0 to disable warm-up",
                )
            )
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ConfigurationError(
                format_error(
                    "config.SchedulerConfig",
                    "min_lr_ratio must lie in [0, 1]",
                    expected="0 <= r <= 1",
                    observed=self.min_lr_ratio,
                    resolution="use 0.0",
                )
            )


@dataclass(frozen=True)
class CheckpointConfig:
    """Distributed-checkpoint behaviour.

    Attributes:
        directory: Root directory holding ``checkpoint-step-XXXXXX`` folders.
        save_every_steps: Checkpoint cadence.  ``0`` disables periodic saving.
        keep_last: Number of checkpoints to retain.  ``0`` keeps all of them.
        save_optimizer_state: Persist optimizer state shards.
        save_rng_state: Persist per-rank RNG state so a resume reproduces the
            same dropout masks and data order.
        verify_checksums_on_load: Recompute and compare SHA-256 digests for
            every shard file that is read.
        resume_from: Explicit checkpoint directory to resume from, or ``""`` to
            auto-detect the newest complete checkpoint under ``directory``.
    """

    directory: str = "checkpoints"
    save_every_steps: int = 0
    keep_last: int = 0
    save_optimizer_state: bool = True
    save_rng_state: bool = True
    verify_checksums_on_load: bool = True
    resume_from: str = ""

    def __post_init__(self) -> None:
        if self.save_every_steps < 0:
            raise ConfigurationError(
                format_error(
                    "config.CheckpointConfig",
                    "save_every_steps must be non-negative",
                    expected=">= 0",
                    observed=self.save_every_steps,
                    resolution="use 0 to disable periodic checkpointing",
                )
            )
        if self.keep_last < 0:
            raise ConfigurationError(
                format_error(
                    "config.CheckpointConfig",
                    "keep_last must be non-negative",
                    expected=">= 0",
                    observed=self.keep_last,
                    resolution="use 0 to keep every checkpoint",
                )
            )


@dataclass(frozen=True)
class TrainingConfig:
    """Training-loop behaviour.

    Attributes:
        max_steps: Number of optimizer steps to run.
        gradient_accumulation_steps: Micro-batches per optimizer step.  The
            first ``n-1`` micro-batches run inside ``no_sync()``.
        max_grad_norm: Global gradient-norm clip.  ``0`` disables clipping.
            The norm is computed over the *correct* union of process groups --
            see :mod:`hybrid_training.optim.sharded_optimizer`.
        seed: Master seed for model init and per-rank RNG derivation.
        log_every_steps: Logging cadence.
        eval_every_steps: Evaluation cadence.  ``0`` disables evaluation.
        deterministic: Enable deterministic algorithms and disable cuDNN
            autotuning.  Slower, but required by the equivalence tests.
        collect_metrics: Record per-step timings and communication volumes.
    """

    max_steps: int = 20
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 0.0
    seed: int = 0
    log_every_steps: int = 1
    eval_every_steps: int = 0
    deterministic: bool = True
    collect_metrics: bool = False

    def __post_init__(self) -> None:
        if self.max_steps < 0:
            raise ConfigurationError(
                format_error(
                    "config.TrainingConfig",
                    "max_steps must be non-negative",
                    expected=">= 0",
                    observed=self.max_steps,
                    resolution="use a non-negative number of steps",
                )
            )
        if self.gradient_accumulation_steps < 1:
            raise ConfigurationError(
                format_error(
                    "config.TrainingConfig",
                    "gradient_accumulation_steps must be at least 1",
                    expected=">= 1",
                    observed=self.gradient_accumulation_steps,
                    resolution="use 1 to disable accumulation",
                )
            )
        if self.max_grad_norm < 0:
            raise ConfigurationError(
                format_error(
                    "config.TrainingConfig",
                    "max_grad_norm must be non-negative",
                    expected=">= 0",
                    observed=self.max_grad_norm,
                    resolution="use 0 to disable clipping",
                )
            )


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level configuration: everything a run needs.

    Attributes:
        name: Human-readable run name, used in log lines and checkpoint paths.
        backend: ``"auto"``, ``"gloo"`` or ``"nccl"``.  ``"auto"`` picks NCCL
            when CUDA is available and Gloo otherwise.
        device: ``"auto"``, ``"cpu"`` or ``"cuda"``.
        timeout_seconds: Collective timeout handed to ``init_process_group``.
        topology: Parallel dimension sizes.
        model, data, optimizer, scheduler, training, checkpoint: Subsystem
            configs.
        ddp, fsdp, tensor_parallel, mixed_precision: Strategy knobs.
    """

    name: str = "experiment"
    backend: str = "auto"
    device: str = "auto"
    timeout_seconds: float = 300.0
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    ddp: DDPConfig = field(default_factory=DDPConfig)
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    tensor_parallel: TensorParallelConfig = field(default_factory=TensorParallelConfig)
    mixed_precision: MixedPrecisionConfig = field(default_factory=MixedPrecisionConfig)

    def __post_init__(self) -> None:
        if self.backend not in {"auto", "gloo", "nccl"}:
            raise ConfigurationError(
                format_error(
                    "config.ExperimentConfig",
                    "unknown backend",
                    expected=["auto", "gloo", "nccl"],
                    observed=self.backend,
                    resolution="use 'auto' unless you need to pin a backend",
                )
            )
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ConfigurationError(
                format_error(
                    "config.ExperimentConfig",
                    "unknown device",
                    expected=["auto", "cpu", "cuda"],
                    observed=self.device,
                    resolution="use 'auto' unless you need to pin a device type",
                )
            )
        if self.timeout_seconds <= 0:
            raise ConfigurationError(
                format_error(
                    "config.ExperimentConfig",
                    "timeout_seconds must be positive",
                    expected="> 0",
                    observed=self.timeout_seconds,
                    resolution="use a positive timeout such as 300",
                )
            )
        if self.backend == "gloo" and self.device == "cuda":
            raise ConfigurationError(
                format_error(
                    "config.ExperimentConfig",
                    "the Gloo backend on CUDA tensors is supported only for a small "
                    "subset of collectives and is never the right choice for training",
                    expected="backend='nccl' with device='cuda'",
                    observed="backend='gloo', device='cuda'",
                    resolution="use backend='nccl' for CUDA or device='cpu' for Gloo",
                )
            )
        # Cross-section consistency: sequence parallelism must be switched on in
        # *both* the topology and the tensor-parallel knobs, or in neither,
        # otherwise the layers and the process groups disagree about shapes.
        if self.tensor_parallel.sequence_parallel and not self.topology.sequence_parallel_enabled:
            raise ConfigurationError(
                format_error(
                    "config.ExperimentConfig",
                    "tensor_parallel.sequence_parallel is enabled but the topology "
                    "does not create a sequence-parallel group",
                    expected="topology.sequence_parallel_mode != 'disabled'",
                    observed=self.topology.sequence_parallel_mode,
                    resolution=(
                        "set topology.sequence_parallel_mode='tensor_group' (with "
                        "tensor_parallel_size > 1) or 'independent' with "
                        "sequence_parallel_size > 1"
                    ),
                )
            )

    # -- serialisation ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON/YAML-serialisable dictionary."""
        return dataclasses.asdict(self)

    def to_yaml(self, path: str | Path) -> Path:
        """Write this configuration to ``path`` as YAML.

        Args:
            path: Destination file.

        Returns:
            The path written.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
        return destination

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExperimentConfig:
        """Build a config from a nested dictionary, rejecting unknown keys.

        Args:
            payload: Nested mapping matching the dataclass structure.

        Returns:
            A validated :class:`ExperimentConfig`.

        Raises:
            ConfigurationError: On unknown keys or values of the wrong shape.
        """
        return _from_mapping(cls, payload, path="")


_T = TypeVar("_T")


def _from_mapping(cls: type[_T], payload: Any, *, path: str) -> _T:
    """Recursively build a (possibly nested) dataclass from a mapping.

    Args:
        cls: The dataclass type to build.
        payload: Mapping of field name to value.
        path: Dotted path used in error messages.

    Returns:
        An instance of ``cls``.

    Raises:
        ConfigurationError: If ``payload`` is not a mapping or contains keys
            that do not correspond to fields.
    """
    if not isinstance(payload, dict):
        raise ConfigurationError(
            format_error(
                "config.from_dict",
                f"expected a mapping for {path or 'the top level'}",
                expected="mapping",
                observed=type(payload).__name__,
                resolution="check the indentation of the YAML file",
            )
        )

    known = {f.name: f for f in fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(payload) - set(known))
    if unknown:
        raise ConfigurationError(
            format_error(
                "config.from_dict",
                f"unknown configuration key(s) under {path or 'the top level'!r}",
                expected=sorted(known),
                observed=unknown,
                resolution="remove the key or fix the typo; unknown keys are never ignored",
            )
        )

    kwargs: dict[str, Any] = {}
    for name, value in payload.items():
        field_type = known[name].type
        kwargs[name] = _coerce(field_type, value, path=f"{path}.{name}" if path else name)
    return cls(**kwargs)  # type: ignore[return-value]


def _coerce(field_type: Any, value: Any, *, path: str) -> Any:
    """Convert a YAML scalar/mapping into the type a dataclass field expects."""
    # Dataclass fields declared with `from __future__ import annotations` arrive
    # as strings; resolve the handful of nested config types by name.
    if isinstance(field_type, str):
        field_type = _CONFIG_TYPES.get(field_type, field_type)

    if is_dataclass(field_type) and isinstance(field_type, type):
        return _from_mapping(field_type, value, path=path)

    origin = get_origin(field_type)
    if origin is tuple and isinstance(value, list | tuple):
        return tuple(value)
    if origin is Union:
        # Only `X | None` unions appear in these configs.
        args = [a for a in get_args(field_type) if a is not type(None)]
        if len(args) == 1 and value is not None:
            return _coerce(args[0], value, path=path)
    return value


#: Name -> type map used to resolve string annotations produced by
#: ``from __future__ import annotations``.  Populated after all dataclasses
#: above have been defined.
_CONFIG_TYPES: dict[str, type] = {
    "TopologyConfig": TopologyConfig,
    "ModelConfig": ModelConfig,
    "DataConfig": DataConfig,
    "OptimizerConfig": OptimizerConfig,
    "SchedulerConfig": SchedulerConfig,
    "TrainingConfig": TrainingConfig,
    "CheckpointConfig": CheckpointConfig,
    "DDPConfig": DDPConfig,
    "FSDPConfig": FSDPConfig,
    "TensorParallelConfig": TensorParallelConfig,
    "MixedPrecisionConfig": MixedPrecisionConfig,
    "GradScalerConfig": GradScalerConfig,
}


def load_experiment_config(path: str | Path, **overrides: Any) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from a YAML file.

    Args:
        path: YAML file to read.
        **overrides: Top-level field overrides applied after parsing.  Nested
            overrides should be passed as already-constructed dataclasses.

    Returns:
        A validated configuration.

    Raises:
        ConfigurationError: If the file does not exist, is not a mapping, or
            contains unknown keys.
    """
    source = Path(path)
    if not source.is_file():
        raise ConfigurationError(
            format_error(
                "config.load_experiment_config",
                "configuration file not found",
                expected="an existing YAML file",
                observed=str(source),
                resolution="check the path; configs live under configs/",
            )
        )
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    config = ExperimentConfig.from_dict(raw)
    if overrides:
        config = dataclasses.replace(config, **overrides)
    return config
