"""Parallelism strategies: DDP, FSDP-style sharding, tensor and sequence parallelism."""

from .ddp import BucketLayout, DistributedDataParallel, GradientBucket
from .fsdp import (
    FlatParamHandle,
    FullyShardedDataParallel,
    PieceLayout,
    ShardedTensorPiece,
)
from .hybrid import (
    HybridModel,
    ParameterParallelInfo,
    build_model,
    build_parallel_model,
    describe_parallel_plan,
)
from .sequence_parallel import (
    SEQUENCE_DIM,
    SequenceParallelLayerNorm,
    SequenceShardInfo,
    gather_sequence,
    local_sequence_slice,
    pad_sequence_dimension,
    reduce_scatter_sequence,
    scatter_sequence,
    unpad_sequence_dimension,
)
from .tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    TensorParallelFeedForward,
    TensorParallelMLPBlock,
    VocabParallelEmbedding,
    all_reduce_sequence_parallel_gradients,
    init_linear_parameters,
    mark_sequence_parallel_partial,
)

__all__ = [
    "SEQUENCE_DIM",
    "BucketLayout",
    "ColumnParallelLinear",
    "DistributedDataParallel",
    "FlatParamHandle",
    "FullyShardedDataParallel",
    "GradientBucket",
    "HybridModel",
    "ParameterParallelInfo",
    "PieceLayout",
    "RowParallelLinear",
    "SequenceParallelLayerNorm",
    "SequenceShardInfo",
    "ShardedTensorPiece",
    "TensorParallelFeedForward",
    "TensorParallelMLPBlock",
    "VocabParallelEmbedding",
    "all_reduce_sequence_parallel_gradients",
    "build_model",
    "build_parallel_model",
    "describe_parallel_plan",
    "gather_sequence",
    "init_linear_parameters",
    "local_sequence_slice",
    "mark_sequence_parallel_partial",
    "pad_sequence_dimension",
    "reduce_scatter_sequence",
    "scatter_sequence",
    "unpad_sequence_dimension",
]
