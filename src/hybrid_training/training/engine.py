"""The training engine: one loop that drives every parallelism strategy.

There is deliberately **one** loop, not one per strategy.  Everything that
differs between single-process, DDP, FSDP, tensor-parallel and hybrid runs is
absorbed by :class:`~hybrid_training.parallel.hybrid.HybridModel`, so the
sequence of operations below is identical in all of them:

.. code-block:: text

    for each optimizer step:
        for each micro-batch:
            with no_sync() for all but the last micro-batch:
                loss = criterion(model(inputs), targets) / accumulation
                scaler.scale(loss).backward()
        model.finish_backward()          <- the synchronisation boundary
        scaler.unscale_(parameters)      <- before clipping, never after
        optimizer.clip_grad_norm(max)    <- one global norm, all ranks agree
        scaler.step(optimizer)           <- skipped together or taken together
        scaler.update()
        optimizer.zero_grad()

Two orderings in that list are load-bearing:

* ``finish_backward`` **before** clipping.  Clipping an unsynchronised gradient
  computes a per-rank norm, and each rank would then scale its gradients by a
  different factor -- so the averaged result is not the clipped average.
* ``unscale_`` **before** clipping.  The clip threshold is expressed in true
  gradient units; applying it to loss-scaled gradients would clip at
  ``max_norm / scale``.

Seeding
=======
Three different seeds are derived from one master seed:

============================  ==========================================
Stream                        Must be identical across
============================  ==========================================
model initialisation          every rank (the wrappers also broadcast, so
                              a mismatch is corrected rather than
                              tolerated)
data order                    every rank -- ranks then take disjoint
                              slices of the same permutation
dropout / runtime randomness  every rank in the ``tensor_sequence``
                              group, and *different* across ``dp_shard``
============================  ==========================================

The last row is the subtle one: tensor-parallel peers compute complementary
halves of one sample and must draw the *same* dropout mask, while data-parallel
ranks process different samples and should not.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from ..config import ExperimentConfig
from ..distributed.collectives import CommunicationRecorder
from ..distributed.context import DistributedContext
from ..errors import ConfigurationError, format_error
from ..logging import get_logger
from ..optim.grad_scaler import GradScaler
from ..optim.sharded_optimizer import ShardedOptimizer
from ..parallel.hybrid import HybridModel, build_parallel_model
from ..utils.memory import MemorySnapshot, capture_memory, reset_peak_memory
from ..utils.reproducibility import seed_everything
from .data import Batch, DistributedBatchSampler, SyntheticDataLoader, build_dataset
from .state import LearningRateSchedule, TrainingState

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance
    from ..checkpoint.reader import LoadedCheckpoint

# The checkpoint package imports `training.state`, so importing it at module
# scope here would close an import cycle.  The three call sites below import it
# lazily instead; the cost is one dictionary lookup per checkpoint operation,
# which happens at most once every few hundred steps.

__all__ = ["StepMetrics", "TrainingEngine", "cross_entropy_loss", "mse_loss"]

_LOGGER = get_logger(__name__)


def mse_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean squared error, averaged over every element.

    Args:
        predictions: Model output.
        targets: Ground truth of the same shape.

    Returns:
        A scalar loss.
    """
    return F.mse_loss(predictions, targets)


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Token-level cross entropy for next-token prediction.

    Args:
        logits: ``(batch, sequence, vocab)``.
        targets: ``(batch, sequence)`` integer token ids.

    Returns:
        A scalar loss averaged over all positions.
    """
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


@dataclass
class StepMetrics:
    """Per-step measurements.

    Attributes:
        step: Optimizer step index.
        loss: Loss, reduced over the data-processing group.
        learning_rate: Learning rate applied.
        grad_norm: Global gradient norm before clipping, or ``None`` when
            clipping and norm reporting are both off.
        seconds: Wall-clock duration of the step.
        forward_seconds: Time inside forward passes.
        backward_seconds: Time inside backward passes.
        samples: Global samples consumed.
        skipped: Whether the optimizer step was skipped (non-finite gradient).
    """

    step: int
    loss: float
    learning_rate: float
    grad_norm: float | None = None
    seconds: float = 0.0
    forward_seconds: float = 0.0
    backward_seconds: float = 0.0
    samples: int = 0
    skipped: bool = False

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        return {
            "step": self.step,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "grad_norm": self.grad_norm,
            "seconds": self.seconds,
            "forward_seconds": self.forward_seconds,
            "backward_seconds": self.backward_seconds,
            "samples": self.samples,
            "skipped": self.skipped,
        }


@dataclass
class TrainingEngine:
    """Builds and runs a distributed training job.

    Args:
        config: Full experiment configuration.
        context: Active distributed context.
        recorder: Optional communication instrumentation sink.  Created
            automatically when ``config.training.collect_metrics`` is set.

    Example:
        >>> # doctest: +SKIP
        >>> with distributed_context(config) as ctx:
        ...     engine = TrainingEngine(config, ctx)
        ...     engine.train()
        ...     engine.save_checkpoint()
    """

    config: ExperimentConfig
    context: DistributedContext
    recorder: CommunicationRecorder | None = None

    model: HybridModel = field(init=False)
    optimizer: ShardedOptimizer = field(init=False)
    scheduler: LearningRateSchedule = field(init=False)
    scaler: GradScaler = field(init=False)
    state: TrainingState = field(init=False)
    history: list[StepMetrics] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        config = self.config
        context = self.context
        if self.recorder is None and config.training.collect_metrics:
            self.recorder = CommunicationRecorder()

        # Model initialisation uses a rank-independent seed so every replica
        # starts from the same weights; the wrappers broadcast as well, so a
        # divergence here is corrected rather than silently tolerated.
        seed_everything(
            config.training.seed,
            stream="model-init",
            deterministic=config.training.deterministic,
        )
        self.model = build_parallel_model(config, context, recorder=self.recorder)

        # Runtime randomness (dropout) must agree inside a tensor/sequence
        # group and differ across data-parallel ranks.
        seed_everything(
            config.training.seed,
            stream="runtime",
            index=context.group("dp_shard").local_rank,
            deterministic=config.training.deterministic,
        )

        self.optimizer = ShardedOptimizer(
            self.model.optimizer_parameters(),
            config.optimizer,
            norm_group=self.model.norm_group,
            device=context.device,
            recorder=self.recorder,
        )
        self.scheduler = LearningRateSchedule(
            config.scheduler,
            base_learning_rate=config.optimizer.learning_rate,
            total_steps=max(config.training.max_steps, 1),
        )
        self.scaler = GradScaler(
            config.mixed_precision.scaler,
            context.group("world"),
            context.device,
            recorder=self.recorder,
        )
        self.state = TrainingState()

        self._criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
            cross_entropy_loss if config.model.kind == "transformer" else mse_loss
        )
        self._build_data()
        reset_peak_memory(context.device)
        _LOGGER.info(
            "engine ready: %s | %d local parameters | %d micro-batches/epoch",
            self.model.description.strategy,
            sum(p.numel() for p in self.model.optimizer_parameters()),
            len(self.train_loader),
        )

    # -- setup --------------------------------------------------------------
    def _build_data(self) -> None:
        """Construct the datasets, samplers and loaders."""
        config = self.config
        data_group = self.model.data_group
        train_dataset = build_dataset(config.model, config.data, split="train")
        self.train_sampler = DistributedBatchSampler(
            len(train_dataset),
            config.data.micro_batch_size,
            data_group,
            shuffle=config.data.shuffle,
            seed=config.data.seed,
        )
        self.train_loader = SyntheticDataLoader(
            train_dataset, self.train_sampler, self.context.device
        )

        self.eval_loader: SyntheticDataLoader | None = None
        if config.data.num_eval_samples > 0:
            eval_dataset = build_dataset(config.model, config.data, split="eval")
            eval_sampler = DistributedBatchSampler(
                len(eval_dataset),
                config.data.micro_batch_size,
                data_group,
                shuffle=False,
                seed=config.data.seed,
            )
            self.eval_loader = SyntheticDataLoader(eval_dataset, eval_sampler, self.context.device)

        needed = config.training.gradient_accumulation_steps
        if len(self.train_loader) < needed:
            raise ConfigurationError(
                format_error(
                    "engine.build_data",
                    "the training set yields fewer micro-batches per epoch than one "
                    "optimizer step consumes, so a step could never complete",
                    rank=self.context.rank,
                    world_size=self.context.world_size,
                    expected=f">= {needed} micro-batches per epoch",
                    observed=len(self.train_loader),
                    resolution=(
                        "raise data.num_train_samples, or lower "
                        "training.gradient_accumulation_steps"
                    ),
                )
            )

    # -- training -----------------------------------------------------------
    def train(self, max_steps: int | None = None) -> TrainingState:
        """Run the training loop.

        Args:
            max_steps: Stop after this many *total* optimizer steps.  Defaults
                to ``config.training.max_steps``.  Because the bound is on the
                total, calling ``train()`` again after a resume continues to
                the same finishing line.

        Returns:
            The final training state.
        """
        target = max_steps if max_steps is not None else self.config.training.max_steps
        batches = self._batch_stream()
        while self.state.step < target:
            metrics = self.train_step(batches)
            self.history.append(metrics)
            if (
                self.config.training.log_every_steps > 0
                and self.state.step % self.config.training.log_every_steps == 0
            ):
                _LOGGER.info(
                    "step %d/%d loss=%.6f lr=%.3e norm=%s %.3fs",
                    self.state.step,
                    target,
                    metrics.loss,
                    metrics.learning_rate,
                    "n/a" if metrics.grad_norm is None else f"{metrics.grad_norm:.4f}",
                    metrics.seconds,
                    extra={"primary_only": True},
                )
            if (
                self.config.training.eval_every_steps > 0
                and self.state.step % self.config.training.eval_every_steps == 0
            ):
                loss = self.evaluate()
                self.state.best_eval_loss = min(self.state.best_eval_loss, loss)
            if (
                self.config.checkpoint.save_every_steps > 0
                and self.state.step % self.config.checkpoint.save_every_steps == 0
            ):
                self.save_checkpoint()
        return self.state

    def _batch_stream(self) -> Any:
        """Yield micro-batches indefinitely, advancing the epoch counter.

        Resuming mid-epoch is supported: the stream skips
        ``state.batches_in_epoch`` batches of the current epoch before
        yielding, so a resumed run consumes the samples it had not yet seen.
        """
        while True:
            epoch = self.state.epoch
            for index, batch in enumerate(self.train_loader.iter_epoch(epoch)):
                if index < self.state.batches_in_epoch:
                    continue
                self.state.batches_in_epoch = index + 1
                yield batch
            self.state.epoch += 1
            self.state.batches_in_epoch = 0

    def train_step(self, batches: Any) -> StepMetrics:
        """Run one optimizer step, including gradient accumulation.

        Args:
            batches: Iterator produced by :meth:`_batch_stream`.

        Returns:
            The step's metrics.
        """
        started = time.perf_counter()
        accumulation = self.config.training.gradient_accumulation_steps
        learning_rate = self.scheduler.value_at(self.state.step)
        self.optimizer.set_learning_rate(learning_rate)

        forward_seconds = 0.0
        backward_seconds = 0.0
        local_loss = 0.0
        samples = 0

        for micro in range(accumulation):
            batch: Batch = next(batches)
            samples += batch.size
            is_last = micro == accumulation - 1
            # Suppressing synchronisation on all but the last micro-batch turns
            # `accumulation` gradient reductions into one.
            sync_context = self.model.no_sync() if not is_last else _null_context()
            with sync_context:
                forward_start = time.perf_counter()
                predictions = self.model(batch.inputs)
                loss = self._criterion(predictions, batch.targets) / accumulation
                forward_seconds += time.perf_counter() - forward_start

                backward_start = time.perf_counter()
                self.scaler.scale(loss).backward()
                backward_seconds += time.perf_counter() - backward_start
            local_loss += float(loss.detach().item()) * accumulation / accumulation

        self.model.finish_backward()

        parameters = self.model.optimizer_parameters()
        self.scaler.unscale_(parameters)
        grad_norm: float | None = None
        if self.config.training.max_grad_norm > 0 or self.config.training.collect_metrics:
            grad_norm = float(
                self.optimizer.clip_grad_norm(self.config.training.max_grad_norm).item()
            )
        stepped = self.scaler.step(self.optimizer, parameters, already_unscaled=True)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        global_samples = samples * self.model.data_group.size
        if stepped:
            self.state.advance_step(samples=global_samples)
        else:
            self.state.step += 1
            self.state.skipped_steps += 1
            self.state.samples_seen += global_samples

        reduced_loss = self.model.reduce_metric(local_loss)
        self.state.last_loss = reduced_loss
        return StepMetrics(
            step=self.state.step,
            loss=reduced_loss,
            learning_rate=learning_rate,
            grad_norm=grad_norm,
            seconds=time.perf_counter() - started,
            forward_seconds=forward_seconds,
            backward_seconds=backward_seconds,
            samples=global_samples,
            skipped=not stepped,
        )

    @torch.no_grad()
    def evaluate(self) -> float:
        """Run the evaluation split and return the reduced loss.

        Returns:
            The mean loss over the evaluation set, reduced over the
            data-processing group.  Returns ``nan`` when evaluation is
            disabled.
        """
        if self.eval_loader is None:
            return float("nan")
        was_training = self.model.training
        self.model.eval()
        total = 0.0
        count = 0
        for batch in self.eval_loader.iter_epoch(0):
            predictions = self.model(batch.inputs)
            total += float(self._criterion(predictions, batch.targets).item())
            count += 1
        if was_training:
            self.model.train()
        local = total / max(count, 1)
        loss = self.model.reduce_metric(local)
        _LOGGER.info(
            "eval loss %.6f at step %d", loss, self.state.step, extra={"primary_only": True}
        )
        return loss

    # -- checkpointing ------------------------------------------------------
    def save_checkpoint(self, directory: str | Path | None = None) -> Path:
        """Write a checkpoint of the current state.

        Args:
            directory: Root directory.  Defaults to
                ``config.checkpoint.directory``.

        Returns:
            The published checkpoint directory.
        """
        from ..checkpoint.writer import save_checkpoint

        root = Path(directory) if directory is not None else Path(self.config.checkpoint.directory)
        result = save_checkpoint(
            root,
            model=self.model,
            context=self.context,
            state=self.state,
            optimizer=self.optimizer,
            config=self.config,
            scheduler_state=self.scheduler.state_dict(),
            scaler_state=self.scaler.state_dict(),
            extra_metadata={
                "batches_in_epoch": self.state.batches_in_epoch,
                "dataset_size": len(self.train_loader.dataset),
                "micro_batch_size": self.config.data.micro_batch_size,
            },
            save_optimizer=self.config.checkpoint.save_optimizer_state,
            save_rng=self.config.checkpoint.save_rng_state,
            keep_last=self.config.checkpoint.keep_last,
        )
        return result.path

    def load_checkpoint(self, directory: str | Path | None = None) -> LoadedCheckpoint:
        """Restore state from a checkpoint.

        Args:
            directory: Checkpoint directory.  Defaults to
                ``config.checkpoint.resume_from``, then to the newest complete
                checkpoint under ``config.checkpoint.directory``.

        Returns:
            The load result.

        Raises:
            ConfigurationError: If no checkpoint can be found.
        """
        from ..checkpoint.reader import load_checkpoint
        from ..checkpoint.writer import find_latest_checkpoint

        target: Path | None
        if directory is not None:
            target = Path(directory)
        elif self.config.checkpoint.resume_from:
            target = Path(self.config.checkpoint.resume_from)
        else:
            target = find_latest_checkpoint(self.config.checkpoint.directory)
        if target is None:
            raise ConfigurationError(
                format_error(
                    "engine.load_checkpoint",
                    "no checkpoint found to resume from",
                    rank=self.context.rank,
                    world_size=self.context.world_size,
                    expected="a checkpoint directory",
                    observed=self.config.checkpoint.directory,
                    resolution="pass an explicit directory or set checkpoint.resume_from",
                )
            )
        result = load_checkpoint(
            target,
            model=self.model,
            context=self.context,
            optimizer=self.optimizer,
            config=self.config,
            verify_checksums=self.config.checkpoint.verify_checksums_on_load,
            load_optimizer=self.config.checkpoint.save_optimizer_state,
            load_rng=self.config.checkpoint.save_rng_state,
        )
        # Take our own copy of the restored progress.  `result` is a *record of
        # what was loaded*; the engine is about to mutate its state on every
        # step, and sharing the object would make `LoadedCheckpoint.step` report
        # the step training has reached rather than the step resumed from.
        self.state = replace(result.state)
        if result.scheduler_state:
            self.scheduler = LearningRateSchedule.from_state_dict(result.scheduler_state)
        if result.scaler_state:
            self.scaler.load_state_dict(result.scaler_state)
        extra = result.metadata.get("extra", {})
        if "batches_in_epoch" in extra:
            self.state.batches_in_epoch = int(extra["batches_in_epoch"])
        return result

    # -- diagnostics --------------------------------------------------------
    def memory_snapshot(self) -> MemorySnapshot:
        """Measure this rank's memory usage."""
        return capture_memory(self.context.device, include_gc_scan=True)

    def communication_summary(self) -> str:
        """Return the communication report, or a note when not instrumented."""
        if self.recorder is None:
            return "communication recording disabled (set training.collect_metrics=true)"
        return self.recorder.summary()

    def parameter_count(self) -> dict[str, int]:
        """Report local and global parameter counts.

        Returns:
            Mapping with ``"local"`` (this rank's optimizer parameters) and
            ``"global"`` (summed over the whole world, so tensor-parallel and
            FSDP slices add up to the true model size).
        """
        from ..distributed.collectives import sum_scalar

        local = sum(p.numel() for p in self.model.optimizer_parameters())
        total = int(
            sum_scalar(float(local), self.context.group("world"), device=self.context.device)
        )
        return {"local": local, "global": total}

    def close(self) -> None:
        """Release the DDP hooks, if any.

        The distributed context itself is owned by the caller; the engine never
        tears down a process group it did not create.
        """
        if self.model.ddp is not None:
            self.model.ddp.teardown()


class _null_context:
    """A minimal no-op context manager.

    ``contextlib.nullcontext`` would do, but this avoids an import in the hot
    loop and makes the intent obvious at the call site.
    """

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc_info: object) -> None:
        return None
