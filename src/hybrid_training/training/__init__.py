"""Training engine, loop state and deterministic synthetic data."""

from .data import (
    Batch,
    DistributedBatchSampler,
    SyntheticDataLoader,
    SyntheticDataset,
    SyntheticMLPDataset,
    SyntheticTokenDataset,
    build_dataset,
)
from .engine import StepMetrics, TrainingEngine, cross_entropy_loss, mse_loss
from .state import LearningRateSchedule, TrainingState

__all__ = [
    "Batch",
    "DistributedBatchSampler",
    "LearningRateSchedule",
    "StepMetrics",
    "SyntheticDataLoader",
    "SyntheticDataset",
    "SyntheticMLPDataset",
    "SyntheticTokenDataset",
    "TrainingEngine",
    "TrainingState",
    "build_dataset",
    "cross_entropy_loss",
    "mse_loss",
]
