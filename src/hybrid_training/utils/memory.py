"""Memory accounting for sharded training.

The point of FSDP is memory, so the project needs to be able to *show* the
memory it saves rather than assert it in prose.  This module provides two
things:

1. :func:`estimate_training_memory` -- a closed-form model of the per-rank
   bytes required by a given strategy, computed from parameter counts alone.
2. :func:`capture_memory` -- an actual measurement, using CUDA's allocator
   statistics when available and falling back to a tensor-inventory count on
   CPU (where PyTorch keeps no allocator counters).

The analytical model
--------------------
Let ``P`` be the total number of parameters, ``b_p`` bytes per parameter,
``b_g`` bytes per gradient element, ``k`` optimizer-state slots per parameter
(2 for Adam: exp_avg and exp_avg_sq), ``b_o`` bytes per optimizer slot, and
``G`` the size of the shard group.

**Plain data parallelism** keeps everything on every rank::

    bytes = P * (b_p + b_g + k * b_o)

For a 1-billion-parameter model in fp32 with Adam that is
``1e9 * (4 + 4 + 8) = 16 GB`` per rank *before* activations.

**Full sharding** divides all three by ``G`` and adds a transient buffer for
the unit currently being gathered::

    bytes = P * (b_p + b_g + k * b_o) / G  +  max_unit_numel * b_p * (1 + resharded)

The transient term is why FSDP wrapping granularity matters: wrapping a model
as one unit makes ``max_unit_numel == P`` and saves nothing at the peak.

**Hybrid sharding** with a replica group of size ``R`` and shard group ``G``
divides by ``G`` only -- replication across ``R`` buys communication locality,
not memory.
"""

from __future__ import annotations

import gc
from collections.abc import Iterable
from dataclasses import dataclass, field

import torch

__all__ = [
    "MemoryEstimate",
    "MemorySnapshot",
    "capture_memory",
    "estimate_training_memory",
    "format_bytes",
    "reset_peak_memory",
]

#: Optimizer state slots per parameter, by optimizer name.  SGD without
#: momentum keeps no state at all; with momentum it keeps one buffer.
OPTIMIZER_STATE_SLOTS: dict[str, int] = {
    "adamw": 2,
    "adam": 2,
    "sgd": 0,
    "sgd_momentum": 1,
}


def format_bytes(num_bytes: float) -> str:
    """Render a byte count with a binary unit suffix.

    Args:
        num_bytes: Value to format.

    Returns:
        A string such as ``"1.50 GiB"``.

    Example:
        >>> format_bytes(1536)
        '1.50 KiB'
    """
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{value:.0f} B"
        value /= 1024.0
    return f"{value:.2f} TiB"  # pragma: no cover - unreachable, loop returns


@dataclass(frozen=True)
class MemoryEstimate:
    """Analytical per-rank memory breakdown, in bytes.

    Attributes:
        persistent_parameters: Bytes held for parameters between steps.
        gradients: Bytes held for gradients at the point they all exist.
        optimizer_state: Bytes of optimizer state.
        transient_gathered_parameters: Peak extra bytes for parameters that are
            temporarily materialised in full.
        transient_gathered_gradients: Peak extra bytes for full-size gradient
            buffers before a reduce-scatter.
    """

    persistent_parameters: int
    gradients: int
    optimizer_state: int
    transient_gathered_parameters: int = 0
    transient_gathered_gradients: int = 0

    @property
    def steady_state(self) -> int:
        """Bytes held between steps."""
        return self.persistent_parameters + self.gradients + self.optimizer_state

    @property
    def peak(self) -> int:
        """Steady state plus the transient buffers."""
        return (
            self.steady_state
            + self.transient_gathered_parameters
            + self.transient_gathered_gradients
        )

    def as_dict(self) -> dict[str, int]:
        """JSON-serialisable view including the derived totals."""
        return {
            "persistent_parameters": self.persistent_parameters,
            "gradients": self.gradients,
            "optimizer_state": self.optimizer_state,
            "transient_gathered_parameters": self.transient_gathered_parameters,
            "transient_gathered_gradients": self.transient_gathered_gradients,
            "steady_state": self.steady_state,
            "peak": self.peak,
        }

    def report(self) -> str:
        """Multi-line human-readable breakdown."""
        rows = [
            ("persistent parameters", self.persistent_parameters),
            ("gradients", self.gradients),
            ("optimizer state", self.optimizer_state),
            ("transient gathered params", self.transient_gathered_parameters),
            ("transient gathered grads", self.transient_gathered_gradients),
            ("steady state", self.steady_state),
            ("peak", self.peak),
        ]
        return "\n".join(f"  {label:<28} {format_bytes(value):>12}" for label, value in rows)


def estimate_training_memory(
    total_parameters: int,
    *,
    shard_group_size: int = 1,
    largest_unit_parameters: int | None = None,
    optimizer: str = "adamw",
    parameter_bytes: int = 4,
    gradient_bytes: int = 4,
    optimizer_state_bytes: int = 4,
    reshard_after_forward: bool = True,
) -> MemoryEstimate:
    """Model per-rank memory for a given sharding configuration.

    Args:
        total_parameters: Number of parameters in the whole model.
        shard_group_size: FSDP shard-group size.  ``1`` models plain data
            parallelism.
        largest_unit_parameters: Parameter count of the biggest FSDP unit,
            which bounds the transient all-gather buffer.  Defaults to
            ``total_parameters`` (the "one unit" worst case).
        optimizer: Key into :data:`OPTIMIZER_STATE_SLOTS`.
        parameter_bytes: Bytes per parameter element.
        gradient_bytes: Bytes per gradient element.
        optimizer_state_bytes: Bytes per optimizer state element.
        reshard_after_forward: When ``True``, backward re-gathers a unit, so
            at most one unit is materialised at a time; when ``False`` the
            forward-gathered copies stay resident until backward consumes
            them, which is modelled as two units.

    Returns:
        The estimate.

    Raises:
        ValueError: If ``shard_group_size`` is not positive or the optimizer is
            unknown.

    Example:
        >>> e = estimate_training_memory(1_000_000, shard_group_size=4,
        ...                              largest_unit_parameters=100_000)
        >>> e.persistent_parameters
        1000000
        >>> e.optimizer_state
        2000000
    """
    if shard_group_size < 1:
        raise ValueError(f"shard_group_size must be positive, got {shard_group_size}")
    if optimizer not in OPTIMIZER_STATE_SLOTS:
        raise ValueError(
            f"unknown optimizer {optimizer!r}; expected one of {sorted(OPTIMIZER_STATE_SLOTS)}"
        )
    if largest_unit_parameters is None:
        largest_unit_parameters = total_parameters

    slots = OPTIMIZER_STATE_SLOTS[optimizer]
    local_parameters = (total_parameters + shard_group_size - 1) // shard_group_size

    transient_params = 0
    transient_grads = 0
    if shard_group_size > 1:
        units_resident = 1 if reshard_after_forward else 2
        transient_params = largest_unit_parameters * parameter_bytes * units_resident
        transient_grads = largest_unit_parameters * gradient_bytes

    return MemoryEstimate(
        persistent_parameters=local_parameters * parameter_bytes,
        gradients=local_parameters * gradient_bytes,
        optimizer_state=local_parameters * slots * optimizer_state_bytes,
        transient_gathered_parameters=transient_params,
        transient_gathered_gradients=transient_grads,
    )


@dataclass
class MemorySnapshot:
    """A point-in-time memory measurement for one device.

    Attributes:
        device: Device the measurement refers to.
        allocated_bytes: Currently allocated tensor bytes.
        reserved_bytes: Bytes the caching allocator holds from the driver
            (CUDA only; ``0`` on CPU).
        peak_allocated_bytes: High-water mark since the last reset (CUDA only).
        live_tensor_bytes: Sum over live tensors found by the garbage
            collector.  This is the CPU fallback and is also reported on CUDA
            as a cross-check; it *undercounts* because it misses storage held
            only by autograd internals.
        source: ``"cuda"`` or ``"gc"``, naming how the numbers were obtained.
    """

    device: torch.device
    allocated_bytes: int = 0
    reserved_bytes: int = 0
    peak_allocated_bytes: int = 0
    live_tensor_bytes: int = 0
    source: str = "gc"
    extra: dict[str, int] = field(default_factory=dict)

    def report(self) -> str:
        """Multi-line human-readable summary."""
        lines = [f"memory on {self.device} (source={self.source}):"]
        for label, value in (
            ("allocated", self.allocated_bytes),
            ("reserved", self.reserved_bytes),
            ("peak allocated", self.peak_allocated_bytes),
            ("live tensors (gc)", self.live_tensor_bytes),
        ):
            lines.append(f"  {label:<20} {format_bytes(value):>12}")
        for label, value in sorted(self.extra.items()):
            lines.append(f"  {label:<20} {format_bytes(value):>12}")
        return "\n".join(lines)


def _live_tensor_bytes(device: torch.device) -> int:
    """Sum the storage of every live tensor on ``device``.

    Walks the garbage collector's object graph.  It is O(number of Python
    objects), so it is fine for diagnostics and far too slow for a training
    loop.  Distinct tensors sharing one storage are counted once.
    """
    seen: set[int] = set()
    total = 0
    for obj in gc.get_objects():
        try:
            if not torch.is_tensor(obj):
                continue
            tensor = obj
            if tensor.device != device:
                continue
            storage = tensor.untyped_storage()
            identity = storage.data_ptr()
            if identity in seen:
                continue
            seen.add(identity)
            total += storage.nbytes()
        except (ReferenceError, RuntimeError):
            # Objects can die mid-iteration, and some tensor-like objects raise
            # when their storage is inspected (meta tensors, for instance).
            continue
    return total


def capture_memory(device: torch.device, *, include_gc_scan: bool = False) -> MemorySnapshot:
    """Measure memory usage on ``device``.

    Args:
        device: Device to inspect.
        include_gc_scan: Also run the (slow) live-tensor scan.  Always run on
            CPU, where no allocator statistics exist.

    Returns:
        The snapshot.
    """
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        snapshot = MemorySnapshot(
            device=device,
            allocated_bytes=int(torch.cuda.memory_allocated(device)),
            reserved_bytes=int(torch.cuda.memory_reserved(device)),
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
            source="cuda",
        )
        if include_gc_scan:
            snapshot.live_tensor_bytes = _live_tensor_bytes(device)
        return snapshot

    live = _live_tensor_bytes(device)
    return MemorySnapshot(
        device=device,
        allocated_bytes=live,
        live_tensor_bytes=live,
        source="gc",
    )


def reset_peak_memory(device: torch.device) -> None:
    """Reset the CUDA high-water mark so the next measurement is scoped.

    A no-op on CPU.

    Args:
        device: Device whose statistics should be reset.
    """
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def tensor_inventory(tensors: Iterable[torch.Tensor]) -> dict[str, int]:
    """Group a tensor collection by dtype and report bytes per dtype.

    Args:
        tensors: Tensors to inventory.

    Returns:
        Mapping from ``str(dtype)`` to total bytes.
    """
    inventory: dict[str, int] = {}
    for tensor in tensors:
        key = str(tensor.dtype)
        inventory[key] = inventory.get(key, 0) + tensor.numel() * tensor.element_size()
    return inventory


def module_parameter_bytes(module: torch.nn.Module) -> dict[str, int]:
    """Report parameter, gradient and buffer bytes held by ``module``.

    Only counts what is reachable through the module tree, so for an FSDP-
    wrapped module it reports the *sharded* footprint -- which is the point.

    Args:
        module: Module to measure.

    Returns:
        Mapping with ``"parameters"``, ``"gradients"`` and ``"buffers"``.
    """
    parameters = 0
    gradients = 0
    for param in module.parameters():
        parameters += param.numel() * param.element_size()
        if param.grad is not None:
            gradients += param.grad.numel() * param.grad.element_size()
    buffers = sum(b.numel() * b.element_size() for b in module.buffers())
    return {"parameters": parameters, "gradients": gradients, "buffers": buffers}
