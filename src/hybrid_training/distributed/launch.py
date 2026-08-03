"""In-process multi-rank launcher used by tests, examples and benchmarks.

``torchrun`` is the right tool for real jobs, and every example in this
repository runs under it.  But a *test* cannot shell out to ``torchrun`` for
each of a hundred cases: the process-startup cost dominates, the assertions
live in the parent, and a hung child needs to be killed with a diagnosis
rather than a wall-clock timeout on the whole suite.

:func:`launch_workers` fills that gap.  It spawns ``world_size`` children, gives
them a private rendezvous port, runs a plain Python function in each, collects
either the return value or the full traceback from every rank, and guarantees
that no child outlives the call.

Requirements on the entrypoint
------------------------------
The default start method is ``"spawn"`` because it is the only one that is safe
once CUDA has been initialised in the parent.  Spawn pickles the callable, so
the entrypoint **must be a module-level function** (not a lambda, not a
closure, not a local function).  Its arguments and its return value must be
picklable too.  A helpful error is raised at submit time when they are not.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import queue as queue_module
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import DistributedInitializationError, format_error
from ..logging import configure_logging, get_logger
from .context import find_free_port

__all__ = [
    "WorkerFailure",
    "WorkerResult",
    "launch_workers",
    "torchrun_environment_summary",
]

_LOGGER = get_logger(__name__)


@dataclass
class WorkerResult:
    """Outcome of one rank.

    Attributes:
        rank: Global rank of the worker.
        value: Whatever the entrypoint returned, or ``None`` when it raised.
        traceback_text: Formatted traceback when the entrypoint raised.
        exit_code: Process exit code, filled in after the join.
        duration_seconds: Wall-clock time inside the entrypoint.
    """

    rank: int
    value: Any = None
    traceback_text: str | None = None
    exit_code: int | None = None
    duration_seconds: float = 0.0

    @property
    def succeeded(self) -> bool:
        """``True`` when the entrypoint returned normally and the process exited 0."""
        return self.traceback_text is None and (self.exit_code in (0, None))


class WorkerFailure(DistributedInitializationError):
    """At least one spawned rank failed, timed out, or exited non-zero.

    The per-rank tracebacks are part of the exception *message*, not merely an
    attribute.  In a distributed failure the interesting traceback is often on
    a rank other than the one that reported first, and a message that says only
    "2 of 4 workers failed" forces whoever is debugging to go hunting for
    output that pytest may already have discarded.

    Attributes:
        results: Per-rank outcomes, including the ranks that succeeded.
        summary: The one-line headline, without the per-rank detail.
    """

    def __init__(self, message: str, results: Sequence[WorkerResult]) -> None:
        self.results: tuple[WorkerResult, ...] = tuple(results)
        self.summary = message
        super().__init__(self._render(message))

    def _render(self, headline: str) -> str:
        """Build the full multi-rank report."""
        lines = [headline]
        for result in sorted(self.results, key=lambda r: r.rank):
            status = "ok" if result.succeeded else "FAILED"
            lines.append(
                f"--- rank {result.rank}: {status} "
                f"(exit_code={result.exit_code}, {result.duration_seconds:.2f}s) ---"
            )
            if result.traceback_text:
                lines.append(result.traceback_text.rstrip())
        return "\n".join(lines)

    def detailed_report(self) -> str:
        """Return the multi-rank failure report."""
        return str(self)

    def failing_ranks(self) -> tuple[int, ...]:
        """Ranks that did not succeed."""
        return tuple(r.rank for r in self.results if not r.succeeded)


@dataclass
class _WorkerSpec:
    """Everything a child needs to bootstrap itself."""

    rank: int
    world_size: int
    master_addr: str
    master_port: int
    entrypoint: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    extra_env: dict[str, str] = field(default_factory=dict)
    log_level: str = "WARNING"


def _worker_main(spec: _WorkerSpec, result_queue: Any) -> None:
    """Child-process entrypoint: set up the environment, run, report.

    Runs in the spawned process.  Never raises: every outcome is reported
    through ``result_queue`` so the parent can produce a real diagnosis instead
    of "exit code 1".
    """
    started = time.perf_counter()
    os.environ["RANK"] = str(spec.rank)
    os.environ["WORLD_SIZE"] = str(spec.world_size)
    os.environ["LOCAL_RANK"] = str(spec.rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(spec.world_size)
    os.environ["MASTER_ADDR"] = spec.master_addr
    os.environ["MASTER_PORT"] = str(spec.master_port)
    # Multiple ranks on one host would otherwise each try to use every core and
    # thrash.  One thread per rank keeps CPU tests fast and deterministic.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.update(spec.extra_env)

    configure_logging(spec.log_level, force=True)
    exit_code = 0
    try:
        import torch

        torch.set_num_threads(1)
        value = spec.entrypoint(spec.rank, spec.world_size, *spec.args, **spec.kwargs)
        payload = WorkerResult(
            rank=spec.rank,
            value=value,
            duration_seconds=time.perf_counter() - started,
        )
    except BaseException:
        payload = WorkerResult(
            rank=spec.rank,
            value=None,
            traceback_text=traceback.format_exc(),
            duration_seconds=time.perf_counter() - started,
        )
        exit_code = 1

    # Serialise with *plain* pickle rather than letting the queue's
    # ForkingPickler do it.  torch registers reducers with ForkingPickler that
    # move tensor storage through shared-memory file descriptors, which are
    # only valid while the sending process is alive -- so a worker that returns
    # a tensor and then exits races the parent's read and intermittently fails
    # with "ConnectionResetError: [Errno 104]".  Plain pickle serialises the
    # storage bytes inline, so the payload is self-contained.
    try:
        encoded = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        encoded = pickle.dumps(
            WorkerResult(
                rank=spec.rank,
                value=None,
                traceback_text=(
                    "the worker returned successfully but its value could not be "
                    "pickled back to the parent:\n" + traceback.format_exc()
                ),
                duration_seconds=time.perf_counter() - started,
            ),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        exit_code = 1
    result_queue.put(encoded)

    # Flush before the interpreter tears down so the parent's captured output
    # contains the child's logs.
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(exit_code)


def _resource_hint() -> str:
    """Report load and free memory, so a timeout can be attributed correctly.

    A timeout caused by swapping and a timeout caused by a mismatched collective
    produce the same symptom: a rank that never reports.  Printing the machine's
    state at the moment of failure is usually enough to tell them apart without
    re-running anything.

    Returns:
        A one-line summary, or a note that the platform does not expose it.
    """
    parts: list[str] = []
    try:
        load1, load5, _ = os.getloadavg()
        parts.append(f"load {load1:.1f}/{load5:.1f} over {os.cpu_count() or 1} core(s)")
    except OSError:  # pragma: no cover - not available on every platform
        pass
    try:
        values: dict[str, int] = {}
        with Path("/proc/meminfo").open(encoding="utf-8") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                if key in {"MemAvailable", "SwapTotal", "SwapFree"}:
                    values[key] = int(rest.split()[0]) * 1024
        if "MemAvailable" in values:
            swapped = values.get("SwapTotal", 0) - values.get("SwapFree", 0)
            parts.append(
                f"{values['MemAvailable'] // (1024 * 1024)} MiB available, "
                f"{swapped // (1024 * 1024)} MiB swapped"
            )
    except OSError:  # pragma: no cover - not Linux
        pass
    if not parts:
        return "  (machine state is not available on this platform)"
    return "  Machine at failure: " + "; ".join(parts) + "."


def _check_picklable(obj: Any, description: str) -> None:
    """Fail early with a readable message when spawn would fail on pickling."""
    try:
        pickle.dumps(obj)
    except Exception as exc:
        raise DistributedInitializationError(
            format_error(
                "launch.launch_workers",
                f"{description} cannot be pickled, so it cannot cross a 'spawn' boundary",
                expected="a module-level function and picklable arguments",
                observed=repr(obj)[:200],
                resolution=(
                    "move the entrypoint to module scope (no lambdas, closures or "
                    "locally defined functions) and pass only picklable arguments"
                ),
            )
        ) from exc


def launch_workers(
    entrypoint: Callable[..., Any],
    world_size: int,
    *,
    args: Sequence[Any] = (),
    kwargs: dict[str, Any] | None = None,
    timeout_seconds: float = 180.0,
    start_method: str = "spawn",
    master_addr: str = "127.0.0.1",
    master_port: int | None = None,
    extra_env: dict[str, str] | None = None,
    log_level: str = "WARNING",
    raise_on_failure: bool = True,
) -> list[WorkerResult]:
    """Run ``entrypoint`` in ``world_size`` fresh processes and collect results.

    The children receive ``RANK``, ``WORLD_SIZE``, ``LOCAL_RANK``,
    ``LOCAL_WORLD_SIZE``, ``MASTER_ADDR`` and ``MASTER_PORT`` in their
    environment, so anything inside them can call
    :func:`~hybrid_training.distributed.context.init_distributed` with no
    arguments and come up correctly.

    Args:
        entrypoint: Module-level callable invoked as
            ``entrypoint(rank, world_size, *args, **kwargs)``.
        world_size: Number of processes to start.
        args: Extra positional arguments for the entrypoint.
        kwargs: Extra keyword arguments for the entrypoint.
        timeout_seconds: Upper bound on the whole run.  On expiry every child
            is terminated and a :class:`WorkerFailure` is raised naming the
            ranks that had not reported -- which is exactly the set of ranks
            stuck in a collective.
        start_method: ``"spawn"`` (default, CUDA-safe) or ``"fork"`` (faster,
            CPU only, unsafe once CUDA is initialised).
        master_addr: Rendezvous host.
        master_port: Rendezvous port; a free one is chosen when omitted.
        extra_env: Additional environment variables for the children.
        log_level: Logging level configured inside each child.
        raise_on_failure: When ``False``, return the results instead of raising
            so a caller can assert on an expected failure.

    Returns:
        One :class:`WorkerResult` per rank, ordered by rank.

    Raises:
        DistributedInitializationError: If ``world_size`` is not positive, or
            the entrypoint/arguments are not picklable under ``"spawn"``.
        WorkerFailure: If any rank raised, exited non-zero, or timed out (and
            ``raise_on_failure`` is ``True``).

    Example:
        >>> # doctest: +SKIP
        >>> def _worker(rank, world_size):
        ...     from hybrid_training.distributed.context import distributed_context
        ...     with distributed_context(backend="gloo") as ctx:
        ...         return ctx.rank
        >>> [r.value for r in launch_workers(_worker, 2)]
        [0, 1]
    """
    if world_size < 1:
        raise DistributedInitializationError(
            format_error(
                "launch.launch_workers",
                "world_size must be positive",
                expected=">= 1",
                observed=world_size,
                resolution="launch at least one worker",
            )
        )
    kwargs = dict(kwargs or {})
    if start_method == "spawn":
        _check_picklable(entrypoint, "the entrypoint")
        _check_picklable((tuple(args), kwargs), "the entrypoint arguments")

    port = master_port if master_port is not None else find_free_port(master_addr)
    # `BaseContext` is the declared return type; the concrete contexts all
    # provide Process/Queue, but the stub does not say so.
    context: Any = mp.get_context(start_method)
    result_queue: Any = context.Queue()
    processes: list[Any] = []

    for rank in range(world_size):
        spec = _WorkerSpec(
            rank=rank,
            world_size=world_size,
            master_addr=master_addr,
            master_port=port,
            entrypoint=entrypoint,
            args=tuple(args),
            kwargs=kwargs,
            extra_env=dict(extra_env or {}),
            log_level=log_level,
        )
        process = context.Process(target=_worker_main, args=(spec, result_queue), daemon=False)
        process.start()
        processes.append(process)

    _LOGGER.debug("launched %d workers on port %d via %s", world_size, port, start_method)

    collected: dict[int, WorkerResult] = {}
    deadline = time.monotonic() + timeout_seconds
    timed_out = False
    try:
        # Drain the queue while the children run.  Draining first (rather than
        # joining first) matters: a full pipe blocks the child in `put`, and a
        # parent that joins before reading would deadlock against it.
        while len(collected) < world_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            try:
                encoded = result_queue.get(timeout=min(remaining, 1.0))
                reported: WorkerResult = pickle.loads(encoded)
            except queue_module.Empty:
                if all(not p.is_alive() for p in processes) and result_queue.empty():
                    # Every child died without reporting -- e.g. a hard crash
                    # or an OS-level kill.  Stop waiting for messages that will
                    # never arrive.
                    break
                continue
            collected[reported.rank] = reported

        for process in processes:
            remaining = max(deadline - time.monotonic(), 0.0)
            process.join(timeout=remaining if not timed_out else 0.0)
            if process.is_alive():
                timed_out = True
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=10.0)
            if process.is_alive():  # pragma: no cover - only on a wedged kernel
                process.kill()
                process.join(timeout=10.0)
        result_queue.close()

    results: list[WorkerResult] = []
    for rank, process in enumerate(processes):
        found = collected.get(rank)
        result: WorkerResult = (
            found
            if found is not None
            else WorkerResult(
                rank=rank,
                traceback_text=(
                    "rank produced no result: it was still running when the "
                    "timeout expired, or it died before it could report.\n"
                    "Two very different causes look identical here:\n"
                    "  1. A genuine mismatch -- this rank blocked inside a "
                    "collective its peers did not issue.  Check that every rank "
                    "builds the same groups in the same order and issues the "
                    "same collectives.\n"
                    "  2. An overloaded machine -- each spawned rank costs a "
                    "full `import torch` plus its training state (~550 MB resident), so concurrent "
                    "distributed runs on a memory-tight box swap and blow the "
                    "timeout with nothing actually wrong.\n" + _resource_hint()
                ),
            )
        )
        result.exit_code = process.exitcode
        results.append(result)

    failures = [r for r in results if not r.succeeded]
    if failures and raise_on_failure:
        reason = "timed out" if timed_out else "failed"
        failure = WorkerFailure(
            format_error(
                "launch.launch_workers",
                f"{len(failures)} of {world_size} worker(s) {reason}",
                expected="every rank to return normally with exit code 0",
                observed=f"failing ranks: {[r.rank for r in failures]}",
                resolution=(
                    "read the per-rank tracebacks below; a rank with no traceback was "
                    "blocked in a collective"
                ),
            ),
            results,
        )
        raise failure
    return results


def torchrun_environment_summary(env: dict[str, str] | None = None) -> dict[str, str]:
    """Return the launcher-relevant environment variables, for diagnostics.

    Args:
        env: Mapping to inspect.  Defaults to ``os.environ``.

    Returns:
        Mapping of variable name to value, with absent variables reported as
        ``"<unset>"`` so a missing entry is visible in a log line.
    """
    source = os.environ if env is None else env
    names = (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "GROUP_RANK",
        "ROLE_RANK",
        "TORCHELASTIC_RUN_ID",
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "NCCL_DEBUG",
    )
    return {name: source.get(name, "<unset>") for name in names}
