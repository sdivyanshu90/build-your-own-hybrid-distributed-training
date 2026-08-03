"""Dynamic loss scaling for fp16 training, with distributed overflow consensus.

The problem
===========
fp16 has 5 exponent bits, so the smallest normal positive value is about
``6e-5`` and gradients below roughly ``6e-8`` (the smallest subnormal) flush to
zero.  Real gradients routinely live in that range, so an unscaled fp16
backward pass silently zeroes a large fraction of the gradient.

The fix is to multiply the loss by a large constant :math:`s` before backward.
By linearity every gradient is multiplied by :math:`s` too, moving it into fp16's
representable range; the optimizer then divides by :math:`s` before stepping.
:math:`s` is chosen dynamically: raise it while everything is finite, and drop
it sharply the moment an ``inf`` or ``nan`` appears.

bf16 does **not** need this.  It has the same 8 exponent bits as fp32, so the
range problem does not arise -- only precision is reduced.  The configuration
refuses to enable scaling for bf16 rather than letting it be a silently
useless setting.

Why the overflow check must be a collective
===========================================
Rank 3 might overflow while ranks 0-2 do not.  If each rank decided
independently, rank 3 would skip its optimizer step while the others took
theirs -- and from that moment the replicas hold different weights, forever.
Every ``inf``/``nan`` decision here is therefore all-reduced with ``MAX`` over
the data-parallel group, so all ranks skip or all ranks step.

This is also why :meth:`GradScaler.step` returns a boolean: the caller (and the
learning-rate scheduler) needs to know whether the step actually happened.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from ..config import GradScalerConfig
from ..distributed.collectives import CommunicationRecorder, ReduceOp, all_reduce
from ..distributed.groups import GroupHandle
from ..logging import get_logger

__all__ = ["GradScaler", "GradScalerState"]

_LOGGER = get_logger(__name__)


@dataclass
class GradScalerState:
    """Serialisable scaler state.

    Attributes:
        scale: Current loss scale.
        growth_tracker: Consecutive finite steps since the last backoff.
        total_steps: Steps attempted.
        skipped_steps: Steps skipped because of a non-finite gradient.
    """

    scale: float
    growth_tracker: int = 0
    total_steps: int = 0
    skipped_steps: int = 0

    def as_dict(self) -> dict[str, float]:
        """JSON-serialisable view."""
        return {
            "scale": float(self.scale),
            "growth_tracker": int(self.growth_tracker),
            "total_steps": int(self.total_steps),
            "skipped_steps": int(self.skipped_steps),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GradScalerState:
        """Rebuild from :meth:`as_dict` output.

        Args:
            payload: Mapping produced by :meth:`as_dict`.

        Returns:
            The reconstructed state.
        """
        return cls(
            scale=float(payload["scale"]),
            growth_tracker=int(payload.get("growth_tracker", 0)),
            total_steps=int(payload.get("total_steps", 0)),
            skipped_steps=int(payload.get("skipped_steps", 0)),
        )


class GradScaler:
    """Dynamic loss scaler that reaches an overflow decision collectively.

    Args:
        config: Scaler policy.
        group: Group over which the overflow decision is agreed.  Must include
            every rank that participates in the same optimizer step -- normally
            the ``"world"`` group, because a hybrid job's ranks all step
            together.
        device: Device the overflow flag lives on.
        recorder: Optional instrumentation sink.

    Example:
        >>> # doctest: +SKIP
        >>> scaler = GradScaler(GradScalerConfig(enabled=True), ctx.group("world"),
        ...                     ctx.device)
        >>> scaler.scale(loss).backward()
        >>> if scaler.step(optimizer, model.parameters()):
        ...     scheduler.step()
        >>> scaler.update()
    """

    def __init__(
        self,
        config: GradScalerConfig,
        group: GroupHandle,
        device: torch.device,
        *,
        recorder: CommunicationRecorder | None = None,
    ) -> None:
        self._config = config
        self._group = group
        self._device = device
        self._recorder = recorder
        self._state = GradScalerState(scale=config.init_scale if config.enabled else 1.0)
        self._unscaled = False
        self._skipped_on_last_step = False

    # -- properties ---------------------------------------------------------
    @property
    def enabled(self) -> bool:
        """Whether scaling is active."""
        return self._config.enabled

    @property
    def scale_value(self) -> float:
        """Current loss scale (``1.0`` when disabled)."""
        return self._state.scale if self._config.enabled else 1.0

    @property
    def state(self) -> GradScalerState:
        """Current scaler state."""
        return self._state

    # -- scaling ------------------------------------------------------------
    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """Multiply the loss by the current scale.

        Args:
            loss: Scalar loss tensor.

        Returns:
            The scaled loss, or ``loss`` unchanged when disabled.
        """
        if not self._config.enabled:
            return loss
        self._unscaled = False
        return loss * self._state.scale

    def unscale_(self, parameters: Iterable[nn.Parameter]) -> bool:
        """Divide gradients by the scale and detect overflow, collectively.

        Must run before gradient clipping, because a clip threshold applies to
        the *true* gradient norm, not the scaled one.

        Args:
            parameters: Parameters whose gradients should be unscaled.

        Returns:
            ``True`` when any rank in the group produced a non-finite gradient.
            The value is identical on every rank.
        """
        params = [p for p in parameters if p.grad is not None]
        if not self._config.enabled:
            return self._detect_non_finite(params)

        inverse = 1.0 / self._state.scale
        for param in params:
            gradient = param.grad
            if gradient is not None:
                gradient.detach().mul_(inverse)
        self._unscaled = True
        return self._detect_non_finite(params)

    def _detect_non_finite(self, parameters: Sequence[nn.Parameter]) -> bool:
        """All-reduce a local finiteness flag with ``MAX`` over the group."""
        found = torch.zeros(1, dtype=torch.float32, device=self._device)
        for param in parameters:
            grad = param.grad
            if grad is None:
                continue
            if not bool(torch.isfinite(grad.detach()).all()):
                found.fill_(1.0)
                break
        all_reduce(found, self._group, op=ReduceOp.MAX, recorder=self._recorder).wait()
        return bool(found.item() > 0.0)

    def step(
        self,
        optimizer: Any,
        parameters: Iterable[nn.Parameter],
        *,
        already_unscaled: bool = False,
    ) -> bool:
        """Unscale, check for overflow, and step the optimizer if it is safe.

        Args:
            optimizer: Anything with a ``step()`` method -- a
                :class:`~hybrid_training.optim.sharded_optimizer.ShardedOptimizer`
                or a plain ``torch.optim.Optimizer``.
            parameters: The optimizer's parameters, used for the overflow scan.
            already_unscaled: Set when :meth:`unscale_` has already run this
                iteration (which it must have, if gradients were clipped).

        Returns:
            ``True`` when the optimizer stepped, ``False`` when the step was
            skipped because of a non-finite gradient.
        """
        params = list(parameters)
        found_non_finite = (
            self._detect_non_finite([p for p in params if p.grad is not None])
            if already_unscaled or self._unscaled
            else self.unscale_(params)
        )
        self._state.total_steps += 1
        self._skipped_on_last_step = found_non_finite
        if found_non_finite:
            self._state.skipped_steps += 1
            _LOGGER.warning(
                "non-finite gradient detected on at least one rank; skipping optimizer "
                "step %d and reducing the loss scale from %.1f",
                self._state.total_steps,
                self._state.scale,
            )
            return False
        optimizer.step()
        return True

    def update(self, *, found_non_finite: bool | None = None) -> float:
        """Advance the scale according to the growth/backoff policy.

        Args:
            found_non_finite: Override the outcome of the last :meth:`step`.
                Normally left at ``None``.

        Returns:
            The new scale.
        """
        if not self._config.enabled:
            return 1.0
        overflowed = (
            found_non_finite if found_non_finite is not None else self._skipped_on_last_step
        )
        if overflowed:
            self._state.scale = max(self._state.scale * self._config.backoff_factor, 1.0)
            self._state.growth_tracker = 0
        else:
            self._state.growth_tracker += 1
            if self._state.growth_tracker >= self._config.growth_interval:
                self._state.scale = min(
                    self._state.scale * self._config.growth_factor, self._config.max_scale
                )
                self._state.growth_tracker = 0
        self._unscaled = False
        return self._state.scale

    @property
    def last_step_skipped(self) -> bool:
        """Whether the most recent :meth:`step` was skipped.

        ``step`` records its own outcome so ``update`` can be called without
        the training loop having to thread the boolean through.
        """
        return self._skipped_on_last_step

    # -- serialisation ------------------------------------------------------
    def state_dict(self) -> dict[str, float]:
        """Return the scaler state for checkpointing."""
        return self._state.as_dict()

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        """Restore scaler state from a checkpoint.

        Args:
            payload: Mapping produced by :meth:`state_dict`.
        """
        self._state = GradScalerState.from_dict(payload)
        self._unscaled = False

    def __repr__(self) -> str:
        return (
            f"GradScaler(enabled={self._config.enabled}, scale={self.scale_value}, "
            f"skipped={self._state.skipped_steps}/{self._state.total_steps})"
        )
