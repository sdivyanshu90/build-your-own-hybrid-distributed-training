"""Thin, explicit, instrumented wrappers around ``torch.distributed`` collectives.

Every wrapper in this module takes a
:class:`~hybrid_training.distributed.groups.GroupHandle` as a **required**
argument.  There is no ``group=None`` default anywhere, and no wrapper falls
back to the default process group.  This is a deliberate constraint: in a
hybrid job the difference between all-reducing gradients over the data-parallel
group and over the whole world is the difference between correct training and
silently wrong training that still converges to *something*.  Making the group
mandatory turns that class of bug into a ``TypeError`` at import time instead of
a subtle numerical error at scale.

Averaging convention
--------------------
Gloo does not implement ``ReduceOp.AVG`` (it raises
``RuntimeError: Cannot use ReduceOp.AVG with Gloo``).  Every average in this
package is therefore expressed as *divide locally, then sum*::

    x_local /= group_size
    all_reduce(x_local, SUM)

rather than *sum, then divide*.  The two differ only in rounding, and dividing
first is what PyTorch's own ``Reducer`` does, which keeps the numerical
comparison against ``torch.nn.parallel.DistributedDataParallel`` tight.  It also
means an asynchronous all-reduce needs no completion callback: the result is
correct the moment the work handle is waited on.

Trivial groups
--------------
A group of size one makes every collective here the identity map.  The wrappers
short-circuit those cases.  That is not a correctness fallback -- it is the
mathematically exact result -- and it lets the *same* code path run at world
size 1, which is what makes the single-process reference in the equivalence
tests meaningful.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist

from ..errors import CollectiveError, ParameterConsistencyError, format_error
from ..logging import get_logger
from .groups import GroupHandle, validate_group_membership

__all__ = [
    "AsyncWork",
    "CommunicationRecorder",
    "CommunicationStats",
    "ReduceOp",
    "all_gather_object_in_group",
    "all_gather_tensor",
    "all_reduce",
    "all_to_all_tensor",
    "assert_metadata_consistent",
    "assert_tensor_consistent",
    "broadcast",
    "concat_shard_sizes",
    "recv_tensor",
    "reduce_scatter_tensor",
    "send_tensor",
    "sum_scalar",
    "wait_all",
]

_LOGGER = get_logger(__name__)


class ReduceOp:
    """Reduction operations supported by this package.

    ``AVG`` is implemented as "divide by the group size, then sum" rather than
    mapped onto ``dist.ReduceOp.AVG``, which Gloo does not support.
    """

    SUM = "sum"
    AVG = "avg"
    MAX = "max"
    MIN = "min"

    ALL = (SUM, AVG, MAX, MIN)


_TORCH_OPS: dict[str, Any] = {
    ReduceOp.SUM: dist.ReduceOp.SUM,
    ReduceOp.AVG: dist.ReduceOp.SUM,  # averaging is done by pre-scaling
    ReduceOp.MAX: dist.ReduceOp.MAX,
    ReduceOp.MIN: dist.ReduceOp.MIN,
}


@dataclass
class CommunicationStats:
    """Aggregate counters for one logical communication channel.

    Attributes:
        calls: Number of collectives issued.
        bytes: Total payload bytes passed to the backend.  For an all-reduce
            this counts the tensor once; the wire volume of a ring all-reduce
            is roughly ``2 * (n-1)/n`` times this, which the benchmark
            documentation spells out.
        seconds: Wall-clock time spent inside the wrapper.  For asynchronous
            collectives this measures only the *launch*; the wait time is
            attributed to :attr:`wait_seconds`.
        wait_seconds: Time spent blocked in :meth:`AsyncWork.wait`.
    """

    calls: int = 0
    bytes: int = 0
    seconds: float = 0.0
    wait_seconds: float = 0.0

    def merge(self, other: CommunicationStats) -> None:
        """Accumulate ``other`` into this object."""
        self.calls += other.calls
        self.bytes += other.bytes
        self.seconds += other.seconds
        self.wait_seconds += other.wait_seconds

    def as_dict(self) -> dict[str, float]:
        """Return a JSON-friendly view."""
        return {
            "calls": float(self.calls),
            "bytes": float(self.bytes),
            "seconds": self.seconds,
            "wait_seconds": self.wait_seconds,
        }


@dataclass
class CommunicationRecorder:
    """Per-owner collection of :class:`CommunicationStats`, keyed by operation.

    Recorders are *owned objects*, not globals: the DDP wrapper has one, each
    FSDP unit shares one, and the benchmark script creates its own.  Passing
    ``None`` wherever a recorder is accepted disables instrumentation with no
    branch in the hot path beyond a single ``is None`` test.

    Attributes:
        enabled: When ``False`` every record call is a no-op.
        by_operation: Statistics keyed by ``"<op>/<group>"``.
    """

    enabled: bool = True
    by_operation: dict[str, CommunicationStats] = field(default_factory=dict)

    def record(
        self,
        operation: str,
        group_name: str,
        *,
        num_bytes: int,
        seconds: float = 0.0,
        wait_seconds: float = 0.0,
    ) -> None:
        """Add one collective to the tally.

        Args:
            operation: Operation name such as ``"all_reduce"``.
            group_name: Group the collective ran on.
            num_bytes: Payload size in bytes.
            seconds: Launch time.
            wait_seconds: Blocking wait time.
        """
        if not self.enabled:
            return
        key = f"{operation}/{group_name}"
        stats = self.by_operation.get(key)
        if stats is None:
            stats = CommunicationStats()
            self.by_operation[key] = stats
        stats.calls += 1
        stats.bytes += num_bytes
        stats.seconds += seconds
        stats.wait_seconds += wait_seconds

    def record_wait(self, operation: str, group_name: str, *, wait_seconds: float) -> None:
        """Attribute blocking wait time to a collective that was already recorded.

        This deliberately does **not** increment ``calls``.  A launch and its
        wait are two halves of one collective; counting them separately would
        report twice as many collectives for an asynchronous operation as for
        the byte-for-byte identical synchronous one, making the headline number
        depend on *how* the call was issued rather than on what crossed the
        wire.  ``calls`` must mean "collectives", or it means nothing.

        Args:
            operation: Operation name, matching the original :meth:`record`.
            group_name: Group the collective ran on.
            wait_seconds: Time spent blocked in ``wait()``.
        """
        if not self.enabled:
            return
        key = f"{operation}/{group_name}"
        stats = self.by_operation.get(key)
        if stats is None:
            # A wait with no matching launch means the launch was not recorded;
            # keep the timing rather than dropping it, still without a call.
            stats = CommunicationStats()
            self.by_operation[key] = stats
        stats.wait_seconds += wait_seconds

    def total(self) -> CommunicationStats:
        """Sum of every channel."""
        total = CommunicationStats()
        for stats in self.by_operation.values():
            total.merge(stats)
        return total

    def reset(self) -> None:
        """Drop all counters."""
        self.by_operation.clear()

    def summary(self) -> str:
        """Multi-line human-readable report."""
        if not self.by_operation:
            return "no communication recorded"
        lines = ["communication summary:"]
        for key in sorted(self.by_operation):
            stats = self.by_operation[key]
            lines.append(
                f"  {key:<34} calls={stats.calls:<6} "
                f"MiB={stats.bytes / 1048576:9.3f} "
                f"launch_s={stats.seconds:8.4f} wait_s={stats.wait_seconds:8.4f}"
            )
        return "\n".join(lines)


class AsyncWork:
    """Handle for an in-flight collective.

    Wrapping the backend's work object lets the caller time the blocking wait
    separately from the launch, and gives a uniform interface for the
    short-circuited "trivial group" case where there is nothing to wait for.

    Args:
        work: Backend work object, or ``None`` when the collective completed
            synchronously (trivial group, or ``async_op=False``).
        recorder: Optional recorder that receives the wait time.
        operation: Operation name for the recorder.
        group_name: Group name for the recorder.
        on_complete: Optional callable invoked after the wait completes.  Used
            by wrappers whose result needs post-processing.
    """

    __slots__ = ("_done", "_group_name", "_on_complete", "_operation", "_recorder", "_work")

    def __init__(
        self,
        work: Any,
        *,
        recorder: CommunicationRecorder | None = None,
        operation: str = "",
        group_name: str = "",
        on_complete: Callable[[], None] | None = None,
    ) -> None:
        self._work = work
        self._recorder = recorder
        self._operation = operation
        self._group_name = group_name
        self._on_complete = on_complete
        self._done = work is None

    @property
    def is_completed(self) -> bool:
        """Whether the collective has already finished.

        Note that this is a *hint*: a ``False`` result may become ``True``
        without any action from the caller.  Never gate correctness on it; use
        :meth:`wait`.
        """
        if self._done:
            return True
        return bool(self._work.is_completed())

    def wait(self) -> None:
        """Block until the collective completes, then run the completion hook.

        Idempotent.
        """
        if not self._done:
            start = time.perf_counter()
            self._work.wait()
            elapsed = time.perf_counter() - start
            if self._recorder is not None:
                self._recorder.record_wait(self._operation, self._group_name, wait_seconds=elapsed)
            self._done = True
        if self._on_complete is not None:
            hook, self._on_complete = self._on_complete, None
            hook()


def _validate_op(op: str, operation: str, group: GroupHandle) -> Any:
    """Resolve a :class:`ReduceOp` constant to a backend op, validating it."""
    if op not in ReduceOp.ALL:
        raise CollectiveError(
            format_error(
                operation,
                "unknown reduction operation",
                rank=group.global_rank,
                expected=list(ReduceOp.ALL),
                observed=op,
                resolution="use one of the ReduceOp constants",
            )
        )
    return _TORCH_OPS[op]


def _check_tensor(tensor: torch.Tensor, operation: str, group: GroupHandle) -> None:
    """Reject tensors the backends cannot handle in a collective."""
    if not tensor.is_contiguous():
        raise CollectiveError(
            format_error(
                operation,
                "collectives require contiguous tensors; a non-contiguous buffer would "
                "be silently copied by some backends and rejected by others",
                rank=group.global_rank,
                expected="tensor.is_contiguous() == True",
                observed=f"shape={tuple(tensor.shape)} strides={tensor.stride()}",
                resolution="call .contiguous() before the collective",
            )
        )


def all_reduce(
    tensor: torch.Tensor,
    group: GroupHandle,
    *,
    op: str = ReduceOp.SUM,
    async_op: bool = False,
    recorder: CommunicationRecorder | None = None,
) -> AsyncWork:
    """All-reduce ``tensor`` in place over ``group``.

    Args:
        tensor: Contiguous tensor, modified in place.  Every rank in the group
            must pass the same shape and dtype.
        group: Target process group.  Required.
        op: One of the :class:`ReduceOp` constants.  ``AVG`` divides the local
            tensor by the group size *before* summing.
        async_op: Return immediately with an unwaited :class:`AsyncWork`.
        recorder: Optional instrumentation sink.

    Returns:
        An :class:`AsyncWork`.  When ``async_op`` is ``False`` it is already
        complete and calling :meth:`AsyncWork.wait` is a no-op.

    Raises:
        CollectiveError: If the rank is not in the group, the tensor is not
            contiguous, or ``op`` is unknown.
    """
    validate_group_membership(group, group.global_rank, "collectives.all_reduce")
    _check_tensor(tensor, "collectives.all_reduce", group)
    torch_op = _validate_op(op, "collectives.all_reduce", group)

    if group.is_trivial:
        # Identity.  AVG over one rank divides by one.
        return AsyncWork(None)

    start = time.perf_counter()
    if op == ReduceOp.AVG:
        tensor.div_(group.size)
    work = dist.all_reduce(tensor, op=torch_op, group=group.process_group, async_op=async_op)
    elapsed = time.perf_counter() - start
    if recorder is not None:
        recorder.record(
            "all_reduce",
            group.name,
            num_bytes=tensor.numel() * tensor.element_size(),
            seconds=elapsed,
        )
    return AsyncWork(
        work if async_op else None,
        recorder=recorder,
        operation="all_reduce",
        group_name=group.name,
    )


def broadcast(
    tensor: torch.Tensor,
    group: GroupHandle,
    *,
    source_local_rank: int = 0,
    async_op: bool = False,
    recorder: CommunicationRecorder | None = None,
) -> AsyncWork:
    """Broadcast ``tensor`` from one group member to the rest, in place.

    Args:
        tensor: Contiguous tensor.  On the source rank it is the payload; on
            every other rank it is overwritten.
        group: Target process group.
        source_local_rank: Index *within the group* of the source rank.  Using
            a group-local index (rather than a global rank) means the same call
            works in every group without the caller doing rank arithmetic.
        async_op: Return without waiting.
        recorder: Optional instrumentation sink.

    Returns:
        An :class:`AsyncWork`.

    Raises:
        CollectiveError: If ``source_local_rank`` is out of range, the rank is
            not in the group, or the tensor is not contiguous.
    """
    validate_group_membership(group, group.global_rank, "collectives.broadcast")
    _check_tensor(tensor, "collectives.broadcast", group)
    if not 0 <= source_local_rank < group.size:
        raise CollectiveError(
            format_error(
                "collectives.broadcast",
                "source_local_rank is outside the group",
                rank=group.global_rank,
                expected=f"0 <= r < {group.size}",
                observed=source_local_rank,
                resolution="pass a group-local index, not a global rank",
            )
        )
    if group.is_trivial:
        return AsyncWork(None)

    start = time.perf_counter()
    work = dist.broadcast(
        tensor,
        src=group.ranks[source_local_rank],
        group=group.process_group,
        async_op=async_op,
    )
    elapsed = time.perf_counter() - start
    if recorder is not None:
        recorder.record(
            "broadcast",
            group.name,
            num_bytes=tensor.numel() * tensor.element_size(),
            seconds=elapsed,
        )
    return AsyncWork(
        work if async_op else None,
        recorder=recorder,
        operation="broadcast",
        group_name=group.name,
    )


def all_gather_tensor(
    tensor: torch.Tensor,
    group: GroupHandle,
    *,
    out: torch.Tensor | None = None,
    recorder: CommunicationRecorder | None = None,
    async_op: bool = False,
) -> tuple[torch.Tensor, AsyncWork]:
    """Concatenate one contiguous tensor per rank along dimension 0.

    Uses ``all_gather_into_tensor``, which writes directly into a single
    pre-sized buffer instead of allocating one tensor per rank and copying.
    That matters for FSDP, where this is the hot path.

    Args:
        tensor: This rank's contribution.  Must be contiguous, and must have
            the same shape and dtype on every rank.
        group: Target process group.
        out: Optional destination of shape ``(group_size * tensor.shape[0],
            *tensor.shape[1:])``.  Allocated when omitted.
        recorder: Optional instrumentation sink.
        async_op: Return without waiting.

    Returns:
        ``(gathered, work)``.  ``gathered`` is only valid after
        ``work.wait()``.

    Raises:
        CollectiveError: On a non-contiguous input, a wrong-shaped ``out``, or
            a group the rank does not belong to.
    """
    validate_group_membership(group, group.global_rank, "collectives.all_gather_tensor")
    _check_tensor(tensor, "collectives.all_gather_tensor", group)

    expected_shape = (tensor.shape[0] * group.size, *tuple(tensor.shape[1:]))
    if out is None:
        out = torch.empty(expected_shape, dtype=tensor.dtype, device=tensor.device)
    elif tuple(out.shape) != expected_shape:
        raise CollectiveError(
            format_error(
                "collectives.all_gather_tensor",
                "destination buffer has the wrong shape",
                rank=group.global_rank,
                expected=expected_shape,
                observed=tuple(out.shape),
                resolution="size the output as (group_size * input.shape[0], *input.shape[1:])",
            )
        )
    if group.is_trivial:
        out.copy_(tensor)
        return out, AsyncWork(None)

    start = time.perf_counter()
    work = dist.all_gather_into_tensor(out, tensor, group=group.process_group, async_op=async_op)
    elapsed = time.perf_counter() - start
    if recorder is not None:
        recorder.record(
            "all_gather",
            group.name,
            num_bytes=tensor.numel() * tensor.element_size(),
            seconds=elapsed,
        )
    return out, AsyncWork(
        work if async_op else None,
        recorder=recorder,
        operation="all_gather",
        group_name=group.name,
    )


def reduce_scatter_tensor(
    tensor: torch.Tensor,
    group: GroupHandle,
    *,
    op: str = ReduceOp.SUM,
    out: torch.Tensor | None = None,
    recorder: CommunicationRecorder | None = None,
    async_op: bool = False,
) -> tuple[torch.Tensor, AsyncWork]:
    """Reduce across ranks and keep only this rank's slice.

    Given an input of ``N`` elements on every rank, the output on rank ``i`` is
    ``sum_over_ranks(input)[i * N/G : (i+1) * N/G]`` where ``G`` is the group
    size.  This is the operation at the heart of FSDP's gradient path: the sum
    is a genuine cross-rank gradient reduction, and the scatter is what leaves
    each rank holding exactly the gradient slice matching its parameter shard.

    Args:
        tensor: Contiguous input whose first dimension is divisible by the
            group size.
        group: Target process group.
        op: Reduction operation.  ``AVG`` pre-divides by the group size.
        out: Optional destination of shape ``(tensor.shape[0] // group_size,
            *tensor.shape[1:])``.
        recorder: Optional instrumentation sink.
        async_op: Return without waiting.

    Returns:
        ``(local_shard, work)``.

    Raises:
        CollectiveError: If the leading dimension is not divisible by the
            group size, the tensor is not contiguous, ``out`` is mis-sized, or
            the rank is not in the group.
    """
    validate_group_membership(group, group.global_rank, "collectives.reduce_scatter_tensor")
    _check_tensor(tensor, "collectives.reduce_scatter_tensor", group)
    torch_op = _validate_op(op, "collectives.reduce_scatter_tensor", group)

    if tensor.shape[0] % group.size != 0:
        raise CollectiveError(
            format_error(
                "collectives.reduce_scatter_tensor",
                "the leading dimension must be divisible by the group size; pad the "
                "buffer before reducing rather than letting ranks disagree about shapes",
                rank=group.global_rank,
                expected=f"shape[0] % {group.size} == 0",
                observed=tensor.shape[0],
                resolution="pad the flat buffer up to a multiple of the group size",
            )
        )

    expected_shape = (tensor.shape[0] // group.size, *tuple(tensor.shape[1:]))
    if out is None:
        out = torch.empty(expected_shape, dtype=tensor.dtype, device=tensor.device)
    elif tuple(out.shape) != expected_shape:
        raise CollectiveError(
            format_error(
                "collectives.reduce_scatter_tensor",
                "destination buffer has the wrong shape",
                rank=group.global_rank,
                expected=expected_shape,
                observed=tuple(out.shape),
                resolution="size the output as (input.shape[0] // group_size, ...)",
            )
        )

    if group.is_trivial:
        out.copy_(tensor.view(expected_shape))
        return out, AsyncWork(None)

    start = time.perf_counter()
    if op == ReduceOp.AVG:
        tensor = tensor.div(group.size)
    work = dist.reduce_scatter_tensor(
        out, tensor, op=torch_op, group=group.process_group, async_op=async_op
    )
    elapsed = time.perf_counter() - start
    if recorder is not None:
        recorder.record(
            "reduce_scatter",
            group.name,
            num_bytes=tensor.numel() * tensor.element_size(),
            seconds=elapsed,
        )
    return out, AsyncWork(
        work if async_op else None,
        recorder=recorder,
        operation="reduce_scatter",
        group_name=group.name,
    )


def all_to_all_tensor(
    tensor: torch.Tensor,
    group: GroupHandle,
    *,
    recorder: CommunicationRecorder | None = None,
) -> torch.Tensor:
    """Equal-split all-to-all along dimension 0.

    Rank ``i`` sends chunk ``j`` of its input to rank ``j`` and receives chunk
    ``i`` from every rank.  Used by the "gather queries instead of keys"
    variant of distributed attention discussed in
    ``docs/06_sequence_parallelism.md``; the baseline sequence-parallel path
    does not need it.

    Args:
        tensor: Contiguous input whose leading dimension divides evenly.
        group: Target process group.
        recorder: Optional instrumentation sink.

    Returns:
        The permuted tensor, same shape as the input.

    Raises:
        CollectiveError: On indivisible shapes or a bad group.
    """
    validate_group_membership(group, group.global_rank, "collectives.all_to_all_tensor")
    _check_tensor(tensor, "collectives.all_to_all_tensor", group)
    if tensor.shape[0] % group.size != 0:
        raise CollectiveError(
            format_error(
                "collectives.all_to_all_tensor",
                "the leading dimension must be divisible by the group size",
                rank=group.global_rank,
                expected=f"shape[0] % {group.size} == 0",
                observed=tensor.shape[0],
                resolution="pad the leading dimension before the exchange",
            )
        )
    if group.is_trivial:
        return tensor.clone()

    out = torch.empty_like(tensor)
    start = time.perf_counter()
    dist.all_to_all_single(out, tensor, group=group.process_group)
    elapsed = time.perf_counter() - start
    if recorder is not None:
        recorder.record(
            "all_to_all",
            group.name,
            num_bytes=tensor.numel() * tensor.element_size(),
            seconds=elapsed,
        )
    return out


def send_tensor(tensor: torch.Tensor, group: GroupHandle, *, destination_local_rank: int) -> None:
    """Blocking point-to-point send to a group member.

    Args:
        tensor: Contiguous payload.
        group: Group defining the local-rank numbering.
        destination_local_rank: Index within the group.

    Raises:
        CollectiveError: On a bad destination or a self-send.
    """
    validate_group_membership(group, group.global_rank, "collectives.send_tensor")
    _check_tensor(tensor, "collectives.send_tensor", group)
    if not 0 <= destination_local_rank < group.size:
        raise CollectiveError(
            format_error(
                "collectives.send_tensor",
                "destination outside the group",
                rank=group.global_rank,
                expected=f"0 <= r < {group.size}",
                observed=destination_local_rank,
                resolution="pass a group-local index",
            )
        )
    destination = group.ranks[destination_local_rank]
    if destination == group.global_rank:
        raise CollectiveError(
            format_error(
                "collectives.send_tensor",
                "a rank cannot send to itself; this deadlocks with a blocking send",
                rank=group.global_rank,
                expected="a different rank",
                observed=destination,
                resolution="skip the transfer when source and destination coincide",
            )
        )
    dist.send(tensor, dst=destination, group=group.process_group)


def recv_tensor(
    tensor: torch.Tensor, group: GroupHandle, *, source_local_rank: int
) -> torch.Tensor:
    """Blocking point-to-point receive into ``tensor``.

    Args:
        tensor: Pre-sized contiguous destination buffer.
        group: Group defining the local-rank numbering.
        source_local_rank: Index within the group.

    Returns:
        ``tensor``, filled.

    Raises:
        CollectiveError: On a bad source or a self-receive.
    """
    validate_group_membership(group, group.global_rank, "collectives.recv_tensor")
    _check_tensor(tensor, "collectives.recv_tensor", group)
    if not 0 <= source_local_rank < group.size:
        raise CollectiveError(
            format_error(
                "collectives.recv_tensor",
                "source outside the group",
                rank=group.global_rank,
                expected=f"0 <= r < {group.size}",
                observed=source_local_rank,
                resolution="pass a group-local index",
            )
        )
    source = group.ranks[source_local_rank]
    if source == group.global_rank:
        raise CollectiveError(
            format_error(
                "collectives.recv_tensor",
                "a rank cannot receive from itself; this deadlocks with a blocking recv",
                rank=group.global_rank,
                expected="a different rank",
                observed=source,
                resolution="skip the transfer when source and destination coincide",
            )
        )
    dist.recv(tensor, src=source, group=group.process_group)
    return tensor


def all_gather_object_in_group(obj: Any, group: GroupHandle) -> list[Any]:
    """Gather one picklable Python object per rank.

    Only used off the hot path -- metadata exchange, consistency checks and
    checkpoint manifest assembly -- because it pickles through CPU memory.

    Args:
        obj: This rank's object.  Must be picklable.
        group: Target process group.

    Returns:
        A list of length ``group.size`` indexed by *group-local* rank.
    """
    validate_group_membership(group, group.global_rank, "collectives.all_gather_object")
    if group.is_trivial:
        return [obj]
    gathered: list[Any] = [None] * group.size
    dist.all_gather_object(gathered, obj, group=group.process_group)
    return gathered


def assert_metadata_consistent(
    payload: Any, group: GroupHandle, *, name: str, operation: str = "collectives.consistency"
) -> None:
    """Verify that every rank in ``group`` supplied an equal ``payload``.

    This is the collective form of a precondition check.  It is used for
    parameter name/shape/dtype lists, bucket layouts and checkpoint manifests,
    all of which *must* agree for the subsequent collectives to line up.

    Args:
        payload: Comparable, picklable object.
        group: Group over which agreement is required.
        name: What is being compared, for the error message.
        operation: Dotted operation name for the error message.

    Raises:
        ParameterConsistencyError: If any rank disagrees.  The message names
            the first offending group-local rank and shows both values.
    """
    if group.is_trivial:
        return
    gathered = all_gather_object_in_group(payload, group)
    reference = gathered[0]
    for local_rank, value in enumerate(gathered):
        if value != reference:
            raise ParameterConsistencyError(
                format_error(
                    operation,
                    f"{name} differs across the {group.name!r} group; ranks in this group "
                    "must agree or the collectives that follow will mismatch",
                    rank=group.global_rank,
                    expected=f"group-local rank 0 (global {group.ranks[0]}): {reference!r}",
                    observed=(
                        f"group-local rank {local_rank} "
                        f"(global {group.ranks[local_rank]}): {value!r}"
                    ),
                    resolution=(
                        "make the model construction deterministic and identical on all "
                        "ranks of this group (same seed, same config, same module order)"
                    ),
                )
            )


def assert_tensor_consistent(
    tensor: torch.Tensor,
    group: GroupHandle,
    *,
    name: str,
    rtol: float = 0.0,
    atol: float = 0.0,
    operation: str = "collectives.tensor_consistency",
) -> None:
    """Verify that ``tensor`` holds (nearly) the same values on every rank.

    Implemented as a broadcast of the group source's copy followed by a local
    comparison.  Costs one collective plus one temporary the size of the
    tensor, so it is a debugging/validation tool, not something to call every
    step.

    Args:
        tensor: The tensor to compare.
        group: Group over which the values must agree.
        name: Description used in the error message.
        rtol: Relative tolerance.  ``0`` means bitwise-equal.
        atol: Absolute tolerance.
        operation: Dotted operation name for the error message.

    Raises:
        ParameterConsistencyError: If the local copy differs from the source's.
    """
    if group.is_trivial:
        return
    reference = tensor.detach().clone().contiguous()
    broadcast(reference, group, source_local_rank=0).wait()
    if not torch.allclose(tensor.detach(), reference, rtol=rtol, atol=atol):
        difference = (tensor.detach().float() - reference.float()).abs()
        raise ParameterConsistencyError(
            format_error(
                operation,
                f"{name} differs from the value held by group-local rank 0",
                rank=group.global_rank,
                expected=f"max |delta| <= {atol} (+ rtol {rtol})",
                observed=f"max |delta| = {difference.max().item():.3e}",
                resolution=(
                    "seed the model identically on every rank of this group, and "
                    "broadcast parameters at construction time"
                ),
            )
        )


def sum_scalar(
    value: float,
    group: GroupHandle,
    *,
    device: torch.device,
    op: str = ReduceOp.SUM,
    recorder: CommunicationRecorder | None = None,
) -> float:
    """Reduce a Python scalar over ``group`` and return the result.

    Args:
        value: Local contribution.
        group: Target process group.
        device: Device the temporary lives on.  Must match the backend
            (CUDA for NCCL, CPU for Gloo).
        op: Reduction operation.
        recorder: Optional instrumentation sink.

    Returns:
        The reduced scalar as a Python float.
    """
    if group.is_trivial:
        return float(value)
    buffer = torch.tensor([float(value)], dtype=torch.float64, device=device)
    all_reduce(buffer, group, op=op, recorder=recorder).wait()
    return float(buffer.item())


def concat_shard_sizes(numel: int, group_size: int) -> tuple[int, int]:
    """Return ``(padded_numel, shard_numel)`` for a flat buffer.

    Every flat parameter and every flat gradient in this project is padded up
    to a multiple of the group size so that ``all_gather_into_tensor`` and
    ``reduce_scatter_tensor`` -- both of which require equal-sized
    contributions -- can be used directly.  The alternative (uneven
    ``all_gather``/``reduce_scatter`` lists) costs an extra allocation per rank
    and is measurably slower for the same result.

    Args:
        numel: Logical number of elements.
        group_size: Number of shards.

    Returns:
        The padded element count and the per-rank shard size.

    Example:
        >>> concat_shard_sizes(10, 4)
        (12, 3)
        >>> concat_shard_sizes(8, 4)
        (8, 2)
    """
    if group_size < 1:
        raise CollectiveError(
            format_error(
                "collectives.concat_shard_sizes",
                "group size must be positive",
                expected=">= 1",
                observed=group_size,
                resolution="pass a real group size",
            )
        )
    shard_numel = (numel + group_size - 1) // group_size
    return shard_numel * group_size, shard_numel


def wait_all(works: Iterable[AsyncWork]) -> None:
    """Wait on a batch of asynchronous collectives, in issue order.

    Args:
        works: Handles to wait on.
    """
    for work in works:
        work.wait()


def log_group_layout(groups: Sequence[GroupHandle]) -> None:
    """Emit one debug line per group describing its membership.

    Args:
        groups: Handles to describe.
    """
    for handle in groups:
        _LOGGER.debug(
            "group %-16s size=%d local_rank=%d ranks=%s",
            handle.name,
            handle.size,
            handle.local_rank,
            handle.ranks,
        )
