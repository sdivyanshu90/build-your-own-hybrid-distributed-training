"""The distributed runtime context.

Everything that needs to know "which rank am I, on what device, in what
groups?" asks a :class:`DistributedContext`.  There is exactly one active
context per process, stored in a module-level slot that only this file writes
to.  That is the "clearly defined distributed context" the design calls for:
process-group handles never leak into ad-hoc globals elsewhere in the package,
and every subsystem receives its context explicitly through its constructor.

Launch modes
------------
``torchrun``
    ``RANK``, ``LOCAL_RANK``, ``WORLD_SIZE``, ``LOCAL_WORLD_SIZE``,
    ``MASTER_ADDR`` and ``MASTER_PORT`` are read from the environment.

Explicit
    Tests and the in-process spawn helper pass ``rank``/``world_size``/
    ``master_port`` directly.  No environment variables are required.

Single process
    With no environment and no explicit arguments the context comes up as
    rank 0 of a world of 1, with a real (Gloo) process group on a private
    port.  Collectives then work as identity operations, so *the same code
    path runs* whether or not the job is distributed.  This is what makes the
    "single-process reference" in the equivalence tests meaningful: it is not
    a separate non-distributed branch of the library.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

from ..config import ExperimentConfig, TopologyConfig
from ..errors import DistributedInitializationError, format_error
from ..logging import LogContext, get_logger, set_log_context
from .groups import GroupHandle, ProcessGroupRegistry
from .topology import ParallelTopology

__all__ = [
    "DistributedContext",
    "LaunchEnvironment",
    "current_context",
    "distributed_context",
    "find_free_port",
    "init_distributed",
    "is_context_active",
]

_LOGGER = get_logger(__name__)

#: The single active context for this process.  Written only by
#: :func:`init_distributed` and :meth:`DistributedContext.shutdown`.
_ACTIVE_CONTEXT: DistributedContext | None = None

_REQUIRED_TORCHRUN_VARS = ("RANK", "WORLD_SIZE")


def find_free_port(host: str = "127.0.0.1") -> int:
    """Bind an ephemeral TCP port, release it, and return the number.

    There is an inherent race here (another process could claim the port
    between the ``close`` and the rendezvous), which is why this is used for
    tests and the single-process fallback rather than for production launches.
    ``torchrun --standalone`` does the same thing.

    Args:
        host: Interface to probe.

    Returns:
        A port number that was free a moment ago.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


@dataclass(frozen=True)
class LaunchEnvironment:
    """Rendezvous parameters, however they were obtained.

    Attributes:
        rank: Global rank.
        world_size: Total number of processes.
        local_rank: Rank within this node; selects the CUDA device.
        local_world_size: Processes on this node.
        master_addr: Rendezvous host.
        master_port: Rendezvous port.
    """

    rank: int
    world_size: int
    local_rank: int
    local_world_size: int
    master_addr: str
    master_port: int

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise DistributedInitializationError(
                format_error(
                    "context.LaunchEnvironment",
                    "world_size must be positive",
                    rank=self.rank,
                    expected=">= 1",
                    observed=self.world_size,
                    resolution="check WORLD_SIZE / the world_size argument",
                )
            )
        if not 0 <= self.rank < self.world_size:
            raise DistributedInitializationError(
                format_error(
                    "context.LaunchEnvironment",
                    "rank outside the world",
                    rank=self.rank,
                    world_size=self.world_size,
                    expected=f"0 <= rank < {self.world_size}",
                    observed=self.rank,
                    resolution="check RANK / the rank argument",
                )
            )
        if not 0 <= self.local_rank < max(self.local_world_size, 1):
            raise DistributedInitializationError(
                format_error(
                    "context.LaunchEnvironment",
                    "local_rank outside the node",
                    rank=self.rank,
                    world_size=self.world_size,
                    expected=f"0 <= local_rank < {self.local_world_size}",
                    observed=self.local_rank,
                    resolution="check LOCAL_RANK / LOCAL_WORLD_SIZE",
                )
            )
        if not 1 <= self.master_port <= 65535:
            raise DistributedInitializationError(
                format_error(
                    "context.LaunchEnvironment",
                    "master_port outside the valid range",
                    rank=self.rank,
                    world_size=self.world_size,
                    expected="1..65535",
                    observed=self.master_port,
                    resolution="check MASTER_PORT",
                )
            )

    @classmethod
    def from_environment(
        cls, env: dict[str, str] | None = None, *, allow_single_process: bool = True
    ) -> LaunchEnvironment:
        """Build a launch environment from ``torchrun``-style variables.

        Args:
            env: Environment mapping.  Defaults to ``os.environ``.
            allow_single_process: When ``True`` and no distributed variables
                are present, synthesise a rank-0-of-1 environment on a free
                local port instead of failing.

        Returns:
            The parsed environment.

        Raises:
            DistributedInitializationError: If the variables are partially
                present (which almost always means a broken launcher), if a
                value is not an integer, or if nothing is present and
                ``allow_single_process`` is ``False``.
        """
        source = dict(os.environ if env is None else env)
        present = [name for name in _REQUIRED_TORCHRUN_VARS if name in source]

        if not present:
            if not allow_single_process:
                raise DistributedInitializationError(
                    format_error(
                        "context.LaunchEnvironment.from_environment",
                        "no distributed launch variables found",
                        expected=f"{', '.join(_REQUIRED_TORCHRUN_VARS)} to be set",
                        observed="none set",
                        resolution=(
                            "launch with `torchrun --standalone --nproc-per-node=N ...`, "
                            "or pass rank/world_size explicitly to init_distributed()"
                        ),
                    )
                )
            port = find_free_port()
            _LOGGER.info(
                "no torchrun environment detected; running single-process on port %d", port
            )
            return cls(
                rank=0,
                world_size=1,
                local_rank=0,
                local_world_size=1,
                master_addr="127.0.0.1",
                master_port=port,
            )

        missing = [name for name in _REQUIRED_TORCHRUN_VARS if name not in source]
        if missing:
            raise DistributedInitializationError(
                format_error(
                    "context.LaunchEnvironment.from_environment",
                    "the distributed launch environment is only partially set, which "
                    "means the launcher failed part-way or variables were unset by hand",
                    expected=list(_REQUIRED_TORCHRUN_VARS),
                    observed=present,
                    resolution=f"set the missing variables: {', '.join(missing)}",
                )
            )

        def _int(name: str, default: int | None = None) -> int:
            raw = source.get(name)
            if raw is None:
                if default is None:  # pragma: no cover - guarded by `missing` above
                    raise DistributedInitializationError(
                        format_error(
                            "context.LaunchEnvironment.from_environment",
                            f"{name} is required but unset",
                            resolution=f"export {name}",
                        )
                    )
                return default
            try:
                return int(raw)
            except ValueError as exc:
                raise DistributedInitializationError(
                    format_error(
                        "context.LaunchEnvironment.from_environment",
                        f"{name} is not an integer",
                        expected="an integer",
                        observed=raw,
                        resolution=f"export {name} as a decimal integer",
                    )
                ) from exc

        rank = _int("RANK")
        world_size = _int("WORLD_SIZE")
        return cls(
            rank=rank,
            world_size=world_size,
            local_rank=_int("LOCAL_RANK", rank),
            local_world_size=_int("LOCAL_WORLD_SIZE", world_size),
            master_addr=source.get("MASTER_ADDR", "127.0.0.1"),
            master_port=_int("MASTER_PORT", 29500),
        )


def _resolve_backend_and_device(
    requested_backend: str,
    requested_device: str,
    env: LaunchEnvironment,
) -> tuple[str, torch.device]:
    """Choose a ``(backend, device)`` pair, failing loudly on impossible ones.

    Rules:

    * ``nccl`` requires CUDA and at least ``local_world_size`` visible devices.
      Two processes sharing one GPU is not a supported NCCL configuration; it
      deadlocks rather than erroring, so we refuse it up front.
    * ``gloo`` runs on CPU tensors.
    * ``auto`` prefers NCCL/CUDA when there are enough devices and otherwise
      falls back to Gloo/CPU with a ``WARNING`` -- a *visible* fallback, never
      a silent one.

    Args:
        requested_backend: ``"auto"``, ``"gloo"`` or ``"nccl"``.
        requested_device: ``"auto"``, ``"cpu"`` or ``"cuda"``.
        env: The launch environment (needed for ``local_rank``).

    Returns:
        The backend name and the concrete device for this rank.

    Raises:
        DistributedInitializationError: If the explicit request cannot be met.
    """
    cuda_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    enough_gpus = cuda_devices >= env.local_world_size

    if requested_backend == "nccl" or requested_device == "cuda":
        if cuda_devices == 0:
            raise DistributedInitializationError(
                format_error(
                    "context.resolve_backend",
                    "CUDA/NCCL requested but no CUDA device is visible",
                    rank=env.rank,
                    world_size=env.world_size,
                    expected=">= 1 CUDA device",
                    observed=cuda_devices,
                    resolution="use backend='gloo' and device='cpu', or fix CUDA_VISIBLE_DEVICES",
                )
            )
        if requested_backend == "nccl" and not enough_gpus:
            raise DistributedInitializationError(
                format_error(
                    "context.resolve_backend",
                    "NCCL requires one CUDA device per local process; sharing a device "
                    "between ranks hangs instead of failing",
                    rank=env.rank,
                    world_size=env.world_size,
                    expected=f">= {env.local_world_size} CUDA devices",
                    observed=cuda_devices,
                    resolution=(
                        "reduce --nproc-per-node to the number of GPUs, or use "
                        "backend='gloo' with device='cpu'"
                    ),
                )
            )
        if not dist.is_nccl_available() and requested_backend == "nccl":
            raise DistributedInitializationError(
                format_error(
                    "context.resolve_backend",
                    "this PyTorch build has no NCCL support",
                    rank=env.rank,
                    world_size=env.world_size,
                    resolution="install a CUDA build of PyTorch or use backend='gloo'",
                )
            )
        backend = "nccl" if requested_backend in {"auto", "nccl"} else requested_backend
        return backend, torch.device("cuda", env.local_rank)

    if requested_backend == "gloo" or requested_device == "cpu":
        if not dist.is_gloo_available():  # pragma: no cover - Gloo ships with every build
            raise DistributedInitializationError(
                format_error(
                    "context.resolve_backend",
                    "this PyTorch build has no Gloo support",
                    rank=env.rank,
                    world_size=env.world_size,
                    resolution="rebuild PyTorch with Gloo, or use NCCL on CUDA",
                )
            )
        return "gloo", torch.device("cpu")

    # Both "auto".
    if enough_gpus and cuda_devices > 0 and dist.is_nccl_available():
        return "nccl", torch.device("cuda", env.local_rank)
    if cuda_devices > 0:
        _LOGGER.warning(
            "backend='auto': %d CUDA device(s) visible but %d local processes requested; "
            "falling back to gloo/cpu because NCCL cannot share a device between ranks",
            cuda_devices,
            env.local_world_size,
        )
    return "gloo", torch.device("cpu")


class DistributedContext:
    """Owns the process group, topology, device and named sub-groups.

    Instances are created by :func:`init_distributed`; constructing one
    directly is possible but bypasses the double-initialisation guard.

    Args:
        env: Rendezvous parameters.
        backend: Resolved backend name.
        device: Resolved device for this rank.
        topology: Rank grid.
        groups: Registry of named sub-groups.
        owns_process_group: Whether :meth:`shutdown` should destroy the default
            process group.  ``False`` when the caller initialised
            ``torch.distributed`` themselves.
    """

    def __init__(
        self,
        env: LaunchEnvironment,
        backend: str,
        device: torch.device,
        topology: ParallelTopology,
        groups: ProcessGroupRegistry,
        *,
        owns_process_group: bool,
    ) -> None:
        self._env = env
        self._backend = backend
        self._device = device
        self._topology = topology
        self._groups = groups
        self._owns_process_group = owns_process_group
        self._active = True

    # -- identity -----------------------------------------------------------
    @property
    def rank(self) -> int:
        """Global rank."""
        return self._env.rank

    @property
    def local_rank(self) -> int:
        """Node-local rank; also the CUDA device ordinal."""
        return self._env.local_rank

    @property
    def world_size(self) -> int:
        """Total number of processes."""
        return self._env.world_size

    @property
    def local_world_size(self) -> int:
        """Number of processes on this node."""
        return self._env.local_world_size

    @property
    def backend(self) -> str:
        """Backend name, ``"gloo"`` or ``"nccl"``."""
        return self._backend

    @property
    def device(self) -> torch.device:
        """Device this rank computes on."""
        return self._device

    @property
    def env(self) -> LaunchEnvironment:
        """The rendezvous parameters used to bring the job up."""
        return self._env

    @property
    def topology(self) -> ParallelTopology:
        """The rank grid."""
        return self._topology

    @property
    def groups(self) -> ProcessGroupRegistry:
        """Named process-group registry."""
        return self._groups

    @property
    def is_primary(self) -> bool:
        """``True`` on global rank 0."""
        return self._env.rank == 0

    @property
    def is_active(self) -> bool:
        """``False`` after :meth:`shutdown`."""
        return self._active

    @property
    def coordinates(self) -> Any:
        """This rank's :class:`~hybrid_training.distributed.topology.RankCoordinates`."""
        return self._topology.coordinates_of(self._env.rank)

    def group(self, name: str) -> GroupHandle:
        """Return a named group handle.

        Args:
            name: Group name; see
                :data:`hybrid_training.distributed.groups.GROUP_CREATION_ORDER`.

        Returns:
            The handle for this rank.
        """
        return self._groups.get(name)

    def is_group_primary(self, name: str) -> bool:
        """Whether this rank is the designated source of the named group.

        Args:
            name: Group name.

        Returns:
            ``True`` when this rank is the lowest-numbered member.
        """
        handle = self._groups.get(name)
        return handle.local_rank == 0

    # -- synchronisation ----------------------------------------------------
    def barrier(self, group: str = "world", *, label: str = "") -> None:
        """Block until every rank in ``group`` reaches this call.

        Args:
            group: Named group to synchronise.  Defaults to the whole world.
            label: Optional tag written to the debug log, which makes a hang
                traceable to a specific barrier.

        Raises:
            DistributedInitializationError: If the context has been shut down.
        """
        self._require_active("context.barrier")
        handle = self._groups.get(group)
        if handle.is_trivial:
            return
        _LOGGER.debug("barrier(%s)%s", group, f" [{label}]" if label else "")
        if self._backend == "nccl":
            # NCCL barriers are implemented as an all-reduce on some device;
            # naming the device avoids PyTorch guessing (and warning) about it.
            dist.barrier(group=handle.process_group, device_ids=[self._device.index])
        else:
            dist.barrier(group=handle.process_group)

    def synchronize_device(self) -> None:
        """Wait for this rank's asynchronous device work to finish.

        A no-op on CPU.  Benchmarks and memory measurements call this; nothing
        in the training path does, because unnecessary synchronisation destroys
        the compute/communication overlap the whole design is built around.
        """
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    # -- lifecycle ----------------------------------------------------------
    def shutdown(self) -> None:
        """Destroy sub-groups and, if owned, the default process group.

        Idempotent and safe to call from an exception handler.  Ordering
        matters: sub-communicators must go before the default group, otherwise
        NCCL can abort while tearing down a communicator whose bootstrap store
        has already gone away.
        """
        global _ACTIVE_CONTEXT
        if not self._active:
            return
        self._active = False
        try:
            self._groups.destroy()
            if self._owns_process_group and dist.is_initialized():
                dist.destroy_process_group()
        finally:
            if _ACTIVE_CONTEXT is self:
                _ACTIVE_CONTEXT = None
            set_log_context(LogContext())

    def __enter__(self) -> DistributedContext:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.shutdown()

    def _require_active(self, operation: str) -> None:
        """Raise if the context has already been shut down."""
        if not self._active:
            raise DistributedInitializationError(
                format_error(
                    operation,
                    "the distributed context has been shut down",
                    rank=self._env.rank,
                    world_size=self._env.world_size,
                    resolution="create a new context with init_distributed()",
                )
            )

    def describe(self) -> str:
        """Multi-line human description of this rank's placement."""
        return (
            f"{self._topology.describe(self._env.rank)}\n"
            f"  backend={self._backend} device={self._device} "
            f"local_rank={self._env.local_rank}/{self._env.local_world_size}"
        )

    def __repr__(self) -> str:
        return (
            f"DistributedContext(rank={self.rank}/{self.world_size}, "
            f"backend={self._backend}, device={self._device}, {self._topology!r})"
        )


def init_distributed(
    topology: TopologyConfig | ExperimentConfig | None = None,
    *,
    backend: str = "auto",
    device: str = "auto",
    timeout_seconds: float = 300.0,
    rank: int | None = None,
    world_size: int | None = None,
    local_rank: int | None = None,
    master_addr: str | None = None,
    master_port: int | None = None,
    env: dict[str, str] | None = None,
    allow_single_process: bool = True,
) -> DistributedContext:
    """Bring up the distributed runtime and return the active context.

    Args:
        topology: A :class:`~hybrid_training.config.TopologyConfig`, or an
            :class:`~hybrid_training.config.ExperimentConfig` (whose topology,
            backend, device and timeout are used), or ``None`` to make the
            whole world data parallel.
        backend: ``"auto"``, ``"gloo"`` or ``"nccl"``.  Ignored when
            ``topology`` is an :class:`ExperimentConfig`.
        device: ``"auto"``, ``"cpu"`` or ``"cuda"``.  Ignored when ``topology``
            is an :class:`ExperimentConfig`.
        timeout_seconds: Collective timeout.
        rank: Explicit global rank, overriding the environment.
        world_size: Explicit world size, overriding the environment.
        local_rank: Explicit node-local rank.  Defaults to ``rank``.
        master_addr: Explicit rendezvous host.
        master_port: Explicit rendezvous port.
        env: Environment mapping to read instead of ``os.environ``.
        allow_single_process: Permit the rank-0-of-1 fallback when nothing is
            set.

    Returns:
        The freshly created, globally registered context.

    Raises:
        DistributedInitializationError: On double initialisation, an
            unsatisfiable backend/device request, or a broken launcher
            environment.
        TopologyError: If the topology does not factor the world size.

    Example:
        >>> # doctest: +SKIP
        >>> ctx = init_distributed(TopologyConfig(data_parallel_size=2), backend="gloo")
        >>> print(ctx.describe())
        >>> ctx.shutdown()
    """
    global _ACTIVE_CONTEXT
    if _ACTIVE_CONTEXT is not None and _ACTIVE_CONTEXT.is_active:
        raise DistributedInitializationError(
            format_error(
                "context.init_distributed",
                "a distributed context is already active in this process",
                rank=_ACTIVE_CONTEXT.rank,
                world_size=_ACTIVE_CONTEXT.world_size,
                resolution=(
                    "call shutdown() on the existing context first, or use the "
                    "distributed_context() context manager which does it for you"
                ),
            )
        )

    topology_config: TopologyConfig | None
    if isinstance(topology, ExperimentConfig):
        topology_config = topology.topology
        backend = topology.backend
        device = topology.device
        timeout_seconds = topology.timeout_seconds
    else:
        topology_config = topology

    # -- rendezvous parameters ---------------------------------------------
    if rank is not None or world_size is not None:
        if rank is None or world_size is None:
            raise DistributedInitializationError(
                format_error(
                    "context.init_distributed",
                    "rank and world_size must be supplied together",
                    expected="both or neither",
                    observed=f"rank={rank}, world_size={world_size}",
                    resolution="pass both explicit values, or neither",
                )
            )
        launch = LaunchEnvironment(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank if local_rank is not None else rank,
            local_world_size=world_size,
            master_addr=master_addr or "127.0.0.1",
            master_port=master_port if master_port is not None else find_free_port(),
        )
    else:
        launch = LaunchEnvironment.from_environment(env, allow_single_process=allow_single_process)
        if master_addr is not None or master_port is not None:
            launch = LaunchEnvironment(
                rank=launch.rank,
                world_size=launch.world_size,
                local_rank=launch.local_rank,
                local_world_size=launch.local_world_size,
                master_addr=master_addr or launch.master_addr,
                master_port=master_port if master_port is not None else launch.master_port,
            )

    resolved_backend, resolved_device = _resolve_backend_and_device(backend, device, launch)

    # The CUDA device must be selected *before* NCCL initialises, otherwise
    # every rank bootstraps on device 0 and the job deadlocks on the first
    # collective.
    if resolved_device.type == "cuda":
        torch.cuda.set_device(resolved_device)

    already_initialised = dist.is_initialized()
    owns_process_group = not already_initialised
    if already_initialised:
        existing_world = dist.get_world_size()
        if existing_world != launch.world_size:
            raise DistributedInitializationError(
                format_error(
                    "context.init_distributed",
                    "torch.distributed is already initialised with a different world size",
                    rank=launch.rank,
                    world_size=launch.world_size,
                    expected=launch.world_size,
                    observed=existing_world,
                    resolution="destroy the existing process group before re-initialising",
                )
            )
        _LOGGER.debug("re-using the pre-existing default process group")
    else:
        os.environ.setdefault("MASTER_ADDR", launch.master_addr)
        os.environ.setdefault("MASTER_PORT", str(launch.master_port))
        dist.init_process_group(
            backend=resolved_backend,
            init_method=f"tcp://{launch.master_addr}:{launch.master_port}",
            rank=launch.rank,
            world_size=launch.world_size,
            timeout=timedelta(seconds=timeout_seconds),
        )

    try:
        if topology_config is None:
            topology_config = TopologyConfig(data_parallel_size=launch.world_size)
        parallel_topology = ParallelTopology(topology_config, launch.world_size)
        registry = ProcessGroupRegistry(
            parallel_topology,
            launch.rank,
            timeout=timedelta(seconds=timeout_seconds),
        )
    except Exception:
        # Bringing the groups up failed on this rank.  Tear the default group
        # down so the process exits instead of leaving a half-built runtime
        # that would hang the next collective.
        if owns_process_group and dist.is_initialized():
            dist.destroy_process_group()
        raise

    context = DistributedContext(
        env=launch,
        backend=resolved_backend,
        device=resolved_device,
        topology=parallel_topology,
        groups=registry,
        owns_process_group=owns_process_group,
    )
    _ACTIVE_CONTEXT = context
    set_log_context(
        LogContext(
            rank=launch.rank,
            world_size=launch.world_size,
            local_rank=launch.local_rank,
            coordinates=parallel_topology.coordinates_of(launch.rank).label(),
        )
    )
    _LOGGER.info(
        "distributed context ready: backend=%s device=%s topology=%s",
        resolved_backend,
        resolved_device,
        parallel_topology,
    )
    return context


@contextmanager
def distributed_context(
    topology: TopologyConfig | ExperimentConfig | None = None, **kwargs: Any
) -> Iterator[DistributedContext]:
    """Context-manager wrapper around :func:`init_distributed`.

    Guarantees :meth:`DistributedContext.shutdown` runs even when the body
    raises, which is what keeps a failing rank from leaving an orphaned NCCL
    communicator behind.

    Args:
        topology: See :func:`init_distributed`.
        **kwargs: Forwarded to :func:`init_distributed`.

    Yields:
        The active context.
    """
    context = init_distributed(topology, **kwargs)
    try:
        yield context
    finally:
        context.shutdown()


def current_context() -> DistributedContext:
    """Return the active context.

    Returns:
        The context created by the most recent :func:`init_distributed`.

    Raises:
        DistributedInitializationError: If no context is active.  Library code
            should prefer taking a context as a constructor argument; this
            accessor exists for scripts and for error messages.
    """
    if _ACTIVE_CONTEXT is None or not _ACTIVE_CONTEXT.is_active:
        raise DistributedInitializationError(
            format_error(
                "context.current_context",
                "no distributed context is active in this process",
                resolution="call init_distributed() (or use distributed_context()) first",
            )
        )
    return _ACTIVE_CONTEXT


def is_context_active() -> bool:
    """Whether a context is currently active in this process."""
    return _ACTIVE_CONTEXT is not None and _ACTIVE_CONTEXT.is_active
