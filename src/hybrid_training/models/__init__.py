"""Reference models: a fast MLP and a small tensor/sequence-parallel transformer."""

from .mlp import MLP, MLPBlock, build_activation
from .transformer import (
    ParallelPlan,
    TensorParallelAttention,
    TinyTransformer,
    TransformerBlock,
    build_reference_linear,
)

__all__ = [
    "MLP",
    "MLPBlock",
    "ParallelPlan",
    "TensorParallelAttention",
    "TinyTransformer",
    "TransformerBlock",
    "build_activation",
    "build_reference_linear",
]
