"""Shared pytest fixtures and the distributed-test harness.

Design of the distributed tests
===============================
A distributed test needs several things a normal test does not:

* **Process isolation.**  ``torch.distributed`` keeps global state, and a
  process group cannot be re-created with a different world size in the same
  interpreter without ending up in a bad state.  Every distributed test
  therefore runs in *fresh* child processes.
* **A timeout that produces a diagnosis.**  A mismatched collective hangs
  forever.  :func:`run_distributed` bounds the run and, on expiry, reports
  which ranks never reported -- which is exactly the set stuck in a collective.
* **Per-rank tracebacks.**  The interesting failure is often not on rank 0.
  :class:`~hybrid_training.distributed.launch.WorkerFailure` carries every
  rank's traceback in its message, so pytest's output contains the whole
  picture.

Worker functions must be **module-level** (the default ``spawn`` start method
pickles them) and their return values must be picklable.  The helper
:func:`run_distributed` checks both and fails with a readable message rather
than an opaque pickling error.

Tolerances
==========
Every numerical assertion states its tolerance explicitly, and the tolerance is
justified in a comment.  The rule of thumb used throughout:

============================  =========  ==========================================
Comparison                    Tolerance  Why
============================  =========  ==========================================
same arithmetic, same order   ``0``      bitwise identical; anything else is a bug
same maths, reduction order   ``1e-6``   fp32 non-associativity over a few hundred
differs across ranks                     terms is ~1e-7; 1e-6 is ~10x headroom
after several Adam steps      ``1e-5``   the optimizer amplifies the above by the
                                         ratio lr/(sqrt(v)+eps)
============================  =========  ==========================================
"""

from __future__ import annotations

import functools
import os
import shutil
import tempfile
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, TypeVar, cast

import pytest
import torch

from hybrid_training.distributed.launch import WorkerFailure, WorkerResult, launch_workers

#: Bound to the decorated test function so ``requires_ranks`` is transparent to
#: type checkers -- it must not erase the signature of what it wraps.
F = TypeVar("F", bound=Callable[..., Any])

# Keep the CPU tests from oversubscribing: several ranks each spawning a thread
# pool on an 8-core CI box is slower *and* less deterministic than one thread
# per rank.
os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.set_num_threads(1)

#: Default upper bound on a distributed test, in seconds.  Generous enough for
#: a loaded CI machine to spawn four processes and import torch in each, tight
#: enough that a genuine hang is caught within one test rather than at the end
#: of the suite.
DEFAULT_TIMEOUT_SECONDS = 180.0

#: fp32 non-associativity: reordering a sum of ~10^3 terms of magnitude ~1
#: perturbs the result by ~1e-7.  1e-6 leaves an order of magnitude of headroom
#: while still failing on a genuinely wrong reduction (which is wrong by a
#: factor of the group size, not by 1e-6).
FLOAT32_REDUCTION_TOLERANCE = 1e-6

#: After a handful of Adam steps the reduction error above is amplified by the
#: adaptive step size.  Measured at ~3e-8 for the test models; 1e-5 is a wide
#: margin that still catches a mis-assembled shard.
OPTIMIZER_STEP_TOLERANCE = 1e-5


#: Resident memory to budget per spawned rank.
#:
#: ``import torch`` alone accounts for ~320 MB, but that is the wrong number to
#: budget with: a rank that is *training* -- holding parameters, gradients,
#: optimizer state, activations and a communication buffer -- was measured at
#: ~550 MB RSS on this project's models.  Budgeting the import figure is how
#: eight ranks got admitted onto a machine that could not hold them, and the
#: OOM killer took the whole pytest process with it.
#:
#: 600 MB is therefore the planning figure: the observed peak plus a margin, so
#: the guard errs towards skipping a test rather than towards swapping.  A guard
#: that is too generous does not degrade gracefully -- it takes the run down.
APPROXIMATE_BYTES_PER_RANK = 600 * 1024 * 1024


def available_memory_bytes() -> int:
    """Return usable physical memory, or ``0`` when it cannot be determined.

    Reads ``MemAvailable`` from ``/proc/meminfo``, which accounts for
    reclaimable cache and is a far better predictor than ``MemFree``.
    """
    try:
        with Path("/proc/meminfo").open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return 0
    return 0


def can_host_ranks(world_size: int) -> bool:
    """Whether this machine can run ``world_size`` spawned ranks comfortably.

    A distributed test that starts swapping does not fail -- it takes twenty
    minutes and then usually passes, which is the worst outcome: the suite
    looks hung and nobody learns anything.  Tests that need many ranks check
    this first and skip with a message naming the shortfall.

    Args:
        world_size: Ranks the test wants to start.

    Returns:
        ``True`` when both memory and core count look sufficient.
    """
    memory = available_memory_bytes()
    if memory and memory < world_size * APPROXIMATE_BYTES_PER_RANK:
        return False
    return os.cpu_count() is None or os.cpu_count() >= world_size  # type: ignore[operator]


def requires_ranks(world_size: int) -> Callable[[F], F]:
    """Skip a test when the machine cannot host ``world_size`` ranks.

    The check runs when the test is **called**, not when it is collected.

    That distinction is the whole point. ``pytest.mark.skipif`` evaluates its
    condition once, while pytest is importing the module -- at which moment the
    machine is usually idle and has plenty of memory, so the guard admits the
    test. The test then runs twenty minutes later, after earlier tests have
    filled memory and swap, and gets OOM-killed. That is not a hypothetical:
    it is how the eight-rank end-to-end test took down a full suite run on the
    development machine, and a collection-time guard could not have prevented
    it. The resources have to be measured at the moment the ranks are about to
    be spawned.

    Args:
        world_size: Ranks the test needs.

    Returns:
        A decorator that skips the test at call time when resources are short.
    """

    def decorator(function: F) -> F:
        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not can_host_ranks(world_size):
                memory = available_memory_bytes()
                pytest.skip(
                    f"needs {world_size} spawned ranks (~"
                    f"{world_size * APPROXIMATE_BYTES_PER_RANK // (1024 * 1024)} MiB and "
                    f"{world_size} cores); this machine reports "
                    f"{memory // (1024 * 1024)} MiB available and {os.cpu_count()} core(s)"
                )
            return function(*args, **kwargs)

        return cast("F", wrapper)

    return decorator


def pytest_configure(config: pytest.Config) -> None:
    """Register the dynamic skip reasons used by the CUDA markers."""
    config.addinivalue_line("markers", "cuda: requires at least one CUDA device")
    config.addinivalue_line("markers", "multigpu: requires at least two CUDA devices")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip GPU tests cleanly when the hardware is not present.

    Marked tests are *collected* either way, so ``pytest --collect-only``
    reports the full matrix and the skip reason names the missing hardware.
    """
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    no_cuda = pytest.mark.skip(reason=f"requires a CUDA device (found {device_count})")
    no_multigpu = pytest.mark.skip(
        reason=f"requires >= 2 CUDA devices for NCCL collectives (found {device_count})"
    )
    for item in items:
        if "multigpu" in item.keywords and device_count < 2:
            item.add_marker(no_multigpu)
        elif "cuda" in item.keywords and device_count < 1:
            item.add_marker(no_cuda)


def run_distributed(
    entrypoint: Callable[..., Any],
    world_size: int,
    *,
    kwargs: dict[str, Any] | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    log_level: str = "ERROR",
) -> list[Any]:
    """Run ``entrypoint`` on ``world_size`` ranks and return their values.

    Args:
        entrypoint: Module-level function called as
            ``entrypoint(rank, world_size, **kwargs)``.
        world_size: Number of processes.
        kwargs: Extra keyword arguments.
        timeout_seconds: Upper bound on the whole run.
        log_level: Logging level inside the children.  ``ERROR`` by default so
            a passing test is quiet; a failing one still shows the traceback.

    Returns:
        The per-rank return values, ordered by rank.

    Raises:
        pytest.Failed: If any rank fails or the run times out.  The failure
            message includes every rank's traceback.
    """
    try:
        results = launch_workers(
            entrypoint,
            world_size,
            kwargs=kwargs or {},
            timeout_seconds=timeout_seconds,
            log_level=log_level,
        )
    except WorkerFailure as failure:
        pytest.fail(failure.detailed_report(), pytrace=False)
    return [result.value for result in results]


#: Cache of completed distributed runs, keyed by ``(function, world_size,
#: kwargs)``.  See :func:`run_distributed_cached`.
_RUN_CACHE: dict[tuple[Any, ...], list[Any]] = {}


def run_distributed_cached(
    entrypoint: Callable[..., Any],
    world_size: int,
    *,
    kwargs: dict[str, Any] | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> list[Any]:
    """Like :func:`run_distributed`, but memoised for the pytest session.

    Several tests routinely assert *different properties of the same run*: the
    DDP equivalence worker returns gradients, bucket layouts, statistics and a
    loss, and there is one test for each.  Re-spawning the ranks for every
    assertion would multiply the cost of the suite by the number of properties
    checked, which is exactly backwards -- the expensive part is the processes,
    not the assertions.

    Spawning a rank costs a full ``import torch`` (hundreds of megabytes of
    resident memory and a second or more of wall clock), so on a memory-tight
    machine the difference is not marginal: it is the difference between a
    suite that finishes and one that swaps.

    The cache is safe because the workers are pure functions of
    ``(rank, world_size, **kwargs)`` and return plain data. Tests must treat
    the returned values as **read-only**; mutating them would leak into the
    next test that asks for the same run.

    Args:
        entrypoint: Module-level worker function.
        world_size: Number of processes.
        kwargs: Extra keyword arguments; must be hashable for the cache key.
        timeout_seconds: Upper bound on the run.

    Returns:
        The per-rank return values, ordered by rank.
    """
    payload = kwargs or {}
    key = (
        f"{entrypoint.__module__}.{entrypoint.__qualname__}",
        world_size,
        tuple(sorted(payload.items())),
    )
    cached = _RUN_CACHE.get(key)
    if cached is None:
        cached = run_distributed(
            entrypoint, world_size, kwargs=payload, timeout_seconds=timeout_seconds
        )
        _RUN_CACHE[key] = cached
    return cached


def expect_distributed_failure(
    entrypoint: Callable[..., Any],
    world_size: int,
    *,
    kwargs: dict[str, Any] | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> Sequence[WorkerResult]:
    """Run a worker that is *expected* to fail, and return the outcomes.

    Negative tests need to assert on the error a rank raised, which means the
    harness must not turn that error into a test failure.

    Args:
        entrypoint: Module-level worker function.
        world_size: Number of processes.
        kwargs: Extra keyword arguments.
        timeout_seconds: Upper bound on the run.

    Returns:
        The per-rank results, including tracebacks.
    """
    return launch_workers(
        entrypoint,
        world_size,
        kwargs=kwargs or {},
        timeout_seconds=timeout_seconds,
        log_level="CRITICAL",
        raise_on_failure=False,
    )


@pytest.fixture()
def temporary_directory() -> Iterator[Path]:
    """Yield a fresh temporary directory that is removed afterwards.

    Distributed tests write checkpoints, so each one needs its own directory --
    a shared one would let a leftover checkpoint from a previous test satisfy
    a "resume from the newest checkpoint" call.
    """
    path = Path(tempfile.mkdtemp(prefix="hybrid-test-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture()
def cpu_device() -> torch.device:
    """The CPU device, for tests that need an explicit one."""
    return torch.device("cpu")


@pytest.fixture(autouse=True)
def _deterministic_seed() -> Iterator[None]:
    """Seed the global RNGs before each test and restore afterwards.

    Autouse, because a test that happens to run after a test which consumed
    random numbers would otherwise see different values -- making failures
    depend on test *order*, which is the hardest kind of flake to diagnose.
    """
    state = torch.get_rng_state()
    torch.manual_seed(1234)
    try:
        yield
    finally:
        torch.set_rng_state(state)
