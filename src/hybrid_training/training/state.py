"""Training-loop state and learning-rate schedules.

Both classes here are deliberately *plain*: they hold JSON-serialisable values
and no tensors, no process groups and no device state.  That is what lets the
checkpoint format keep them in ``metadata.json`` alongside the manifest, rather
than pickling them into a tensor file -- which in turn is what lets a
checkpoint be inspected (and validated) without ``torch.load`` ever running.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from ..config import SchedulerConfig
from ..errors import ConfigurationError, format_error

__all__ = ["LearningRateSchedule", "TrainingState"]


@dataclass
class TrainingState:
    """Where a run is up to.

    Attributes:
        step: Completed optimizer steps.  This is the number a checkpoint is
            named after and the number a resumed run continues from.
        epoch: Completed passes over the dataset.
        micro_step: Micro-batches consumed within the current optimizer step.
            Non-zero only if a checkpoint is taken mid-accumulation, which the
            engine avoids.
        samples_seen: Global samples consumed, summed over all data-parallel
            ranks.  Tracked so throughput can be reported in samples/second
            rather than in steps/second, which hides the batch size.
        batches_in_epoch: Index of the next batch within the epoch, so a resume
            continues the data stream where it stopped rather than restarting
            the epoch.
        last_loss: Most recent reduced training loss.
        best_eval_loss: Best evaluation loss seen, or ``inf``.
        skipped_steps: Optimizer steps skipped because of non-finite gradients.
    """

    step: int = 0
    epoch: int = 0
    micro_step: int = 0
    samples_seen: int = 0
    batches_in_epoch: int = 0
    last_loss: float = float("nan")
    best_eval_loss: float = float("inf")
    skipped_steps: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view.

        ``nan`` and ``inf`` are converted to ``None`` and a large sentinel is
        avoided, because strict JSON has no representation for either and
        round-tripping through a permissive parser is a portability trap.
        """
        payload = asdict(self)
        payload["last_loss"] = None if math.isnan(self.last_loss) else self.last_loss
        payload["best_eval_loss"] = None if math.isinf(self.best_eval_loss) else self.best_eval_loss
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainingState:
        """Rebuild from :meth:`as_dict` output.

        Args:
            payload: The mapping to restore.

        Returns:
            The reconstructed state.
        """
        data = dict(payload)
        if data.get("last_loss") is None:
            data["last_loss"] = float("nan")
        if data.get("best_eval_loss") is None:
            data["best_eval_loss"] = float("inf")
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})

    def advance_step(self, *, samples: int) -> None:
        """Record one completed optimizer step.

        Args:
            samples: Global samples consumed by the step.
        """
        self.step += 1
        self.samples_seen += samples
        self.micro_step = 0

    def describe(self) -> str:
        """One-line summary for logs."""
        loss = "nan" if math.isnan(self.last_loss) else f"{self.last_loss:.6f}"
        return (
            f"step={self.step} epoch={self.epoch} samples={self.samples_seen} "
            f"loss={loss} skipped={self.skipped_steps}"
        )


@dataclass
class LearningRateSchedule:
    """Warm-up plus optional decay, evaluated as a pure function of the step.

    Being a pure function of the step (rather than an object that mutates on
    every call) means a resumed run computes exactly the learning rate it would
    have had, from the step number alone.  No scheduler state has to be
    checkpointed beyond the configuration, and a resume cannot drift.

    The schedules, with ``w`` the warm-up length, ``T`` the total steps and
    ``m`` the floor ratio:

    .. math::

        \\text{lr}(t) = \\begin{cases}
            \\text{base} \\cdot \\frac{t+1}{w}                & t < w \\\\
            \\text{base}                                      & \\text{constant} \\\\
            \\text{base}\\,(m + (1-m)(1 - \\frac{t-w}{T-w}))    & \\text{linear} \\\\
            \\text{base}\\,(m + (1-m)\\tfrac{1}{2}(1 + \\cos(\\pi \\frac{t-w}{T-w})))
                                                              & \\text{cosine}
        \\end{cases}

    Attributes:
        config: Schedule shape.
        base_learning_rate: Peak learning rate, reached at the end of warm-up.
        total_steps: Total optimizer steps in the run; the decay horizon.
    """

    config: SchedulerConfig
    base_learning_rate: float
    total_steps: int = field(default=1)

    def __post_init__(self) -> None:
        if self.total_steps < 1:
            raise ConfigurationError(
                format_error(
                    "training.LearningRateSchedule",
                    "total_steps must be positive",
                    expected=">= 1",
                    observed=self.total_steps,
                    resolution="pass TrainingConfig.max_steps",
                )
            )
        if self.config.warmup_steps >= self.total_steps and self.config.name != "constant":
            raise ConfigurationError(
                format_error(
                    "training.LearningRateSchedule",
                    "warm-up is at least as long as the whole run, so the decay phase "
                    "would never begin",
                    expected=f"warmup_steps < {self.total_steps}",
                    observed=self.config.warmup_steps,
                    resolution="shorten the warm-up or lengthen the run",
                )
            )

    def value_at(self, step: int) -> float:
        """Return the learning rate for a given step.

        Args:
            step: Zero-based optimizer step index.

        Returns:
            The learning rate.
        """
        warmup = self.config.warmup_steps
        if warmup > 0 and step < warmup:
            return self.base_learning_rate * float(step + 1) / float(warmup)

        if self.config.name == "constant":
            return self.base_learning_rate

        span = max(self.total_steps - warmup, 1)
        progress = min(max((step - warmup) / span, 0.0), 1.0)
        floor = self.config.min_lr_ratio

        if self.config.name == "linear":
            factor = floor + (1.0 - floor) * (1.0 - progress)
        else:  # cosine
            factor = floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.base_learning_rate * factor

    def state_dict(self) -> dict[str, Any]:
        """Return the schedule's configuration for checkpointing.

        There is no mutable state: the schedule is reconstructed from these
        values and the step number.
        """
        return {
            "name": self.config.name,
            "warmup_steps": self.config.warmup_steps,
            "min_lr_ratio": self.config.min_lr_ratio,
            "base_learning_rate": self.base_learning_rate,
            "total_steps": self.total_steps,
        }

    @classmethod
    def from_state_dict(cls, payload: dict[str, Any]) -> LearningRateSchedule:
        """Rebuild a schedule from :meth:`state_dict` output.

        Args:
            payload: The mapping to restore.

        Returns:
            The reconstructed schedule.
        """
        return cls(
            config=SchedulerConfig(
                name=payload["name"],
                warmup_steps=int(payload["warmup_steps"]),
                min_lr_ratio=float(payload["min_lr_ratio"]),
            ),
            base_learning_rate=float(payload["base_learning_rate"]),
            total_steps=int(payload["total_steps"]),
        )
