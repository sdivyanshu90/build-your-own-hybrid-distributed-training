"""Shape arithmetic, memory accounting and reproducibility helpers."""

from .memory import (
    MemoryEstimate,
    MemorySnapshot,
    capture_memory,
    estimate_training_memory,
    format_bytes,
    reset_peak_memory,
)
from .reproducibility import (
    RngSnapshot,
    capture_rng_state,
    derive_seed,
    restore_rng_state,
    seed_everything,
    temporary_seed,
)
from .tensors import (
    FlatEntry,
    ShardRange,
    build_flat_layout,
    even_shard_ranges,
    flatten_dense_tensors,
    intersect_ranges,
    pad_flat_tensor,
    shard_range_for,
    split_tensor_along_dim,
    unflatten_to_views,
)

__all__ = [
    "FlatEntry",
    "MemoryEstimate",
    "MemorySnapshot",
    "RngSnapshot",
    "ShardRange",
    "build_flat_layout",
    "capture_memory",
    "capture_rng_state",
    "derive_seed",
    "estimate_training_memory",
    "even_shard_ranges",
    "flatten_dense_tensors",
    "format_bytes",
    "intersect_ranges",
    "pad_flat_tensor",
    "reset_peak_memory",
    "restore_rng_state",
    "seed_everything",
    "shard_range_for",
    "split_tensor_along_dim",
    "temporary_seed",
    "unflatten_to_views",
]
