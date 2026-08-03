"""Deterministic process-group construction.

Why group *creation order* matters
==================================
``torch.distributed.new_group`` is a **collective** call.  Every process in the
default process group must call it, the same number of times, with the same
rank lists, in the same order.  The reason is that the backend assigns each new
group an implicit sequence number and uses it as part of the communicator
identity; NCCL additionally exchanges a unique id through the rendezvous store
under a key derived from that sequence number.

If rank 0 creates ``[0, 1]`` then ``[0, 2]`` while rank 2 creates ``[0, 2]``
then ``[0, 1]``, the two processes end up with communicators that disagree
about who is in them.  The failure is *not* an exception.  It is a hang: rank 0
posts a collective on communicator #1 while rank 2 posts on communicator #2 and
both wait forever.  This is the single most common way a hand-rolled hybrid
parallel stack deadlocks.

This module removes the possibility by construction:

* :data:`GROUP_CREATION_ORDER` fixes the order of the *named* groups.
* :meth:`ParallelTopology.all_group_rank_lists` enumerates the groups for a
  given dimension sorted by their smallest member.
* Every rank walks exactly the same nested loop and calls ``new_group`` for
  every group, keeping only the handle for the group it belongs to.  Ranks that
  are not members still have to make the call -- that is what "collective"
  means here.

Two shortcuts are taken, and both are safe for the same reason: they depend
only on the *rank list*, which every rank computes identically, so no rank can
take a different branch from its peers.

1. A group whose rank list is the whole world reuses the default process group.
2. A group with a single member gets no communicator at all.  Every collective
   over a one-member group is the identity and the wrappers short-circuit
   before touching ``process_group``, so the communicator would compute
   nothing.  This matters more than it sounds: a four-dimensional topology
   creates a *lot* of degenerate groups -- with ``dp=2, tensor=2`` the ``shard``
   and ``sequence`` dimensions contribute eight one-member groups out of
   sixteen -- and with Gloo each ``new_group`` is a real TCP-store round trip.
   Skipping them roughly halves context start-up time, which is the dominant
   cost in a test suite that builds hundreds of contexts.

Additionally, communicators are cached by rank list, so two named groups
covering the same ranks (``dp_shard`` equals ``data_parallel`` whenever the
shard dimension is 1) share one communicator.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch.distributed as dist

from ..errors import CollectiveError, DistributedInitializationError, format_error
from ..logging import get_logger
from .topology import COMPOSITE_GROUPS, DIMENSIONS, ParallelTopology

__all__ = ["GROUP_CREATION_ORDER", "GroupHandle", "ProcessGroupRegistry"]

_LOGGER = get_logger(__name__)

#: The fixed order in which named groups are created.  Do not reorder without
#: understanding the deadlock discussion in this module's docstring: any change
#: must be applied to every rank simultaneously, which in practice means every
#: process in a job must run the same version of this file.
GROUP_CREATION_ORDER: tuple[str, ...] = (
    "data_parallel",
    "shard",
    "sequence",
    "tensor",
    "dp_shard",
    "tensor_sequence",
)


@dataclass(frozen=True)
class GroupHandle:
    """A named communication group this rank belongs to.

    Attributes:
        name: Group name, e.g. ``"tensor"`` or ``"dp_shard"``.
        ranks: Ascending tuple of the *global* ranks in the group.
        local_rank: This rank's index inside :attr:`ranks`.
        global_rank: This rank's global rank.
        process_group: The backend handle to pass to collectives.  For a group
            spanning the whole world this is the default process group.
    """

    name: str
    ranks: tuple[int, ...]
    local_rank: int
    global_rank: int
    process_group: Any

    @property
    def size(self) -> int:
        """Number of ranks in the group."""
        return len(self.ranks)

    @property
    def is_trivial(self) -> bool:
        """``True`` when the group has a single member.

        Collectives over a trivial group are mathematically identity
        operations.  Call sites still issue them (so the code path is exercised
        identically at every world size), but the wrappers in
        :mod:`hybrid_training.distributed.collectives` short-circuit the ones
        that would otherwise allocate.
        """
        return len(self.ranks) == 1

    @property
    def source_rank(self) -> int:
        """Global rank of the group's designated broadcast source."""
        return self.ranks[0]

    @classmethod
    def trivial(cls, name: str = "trivial", rank: int = 0) -> GroupHandle:
        """Build a single-member group that needs no backend communicator.

        Every collective over a one-rank group is the identity, and the
        wrappers in :mod:`hybrid_training.distributed.collectives` short-circuit
        those cases before touching ``process_group``.  That makes this handle
        usable with no distributed runtime at all, which is what lets the
        tensor- and sequence-parallel layers be written *once* and run
        unmodified as the single-process reference in the equivalence tests.

        Args:
            name: Name reported in diagnostics.
            rank: The single member's rank.

        Returns:
            A one-member handle whose ``process_group`` is ``None``.
        """
        return cls(
            name=name,
            ranks=(rank,),
            local_rank=0,
            global_rank=rank,
            process_group=None,
        )

    def __repr__(self) -> str:
        return (
            f"GroupHandle(name={self.name!r}, size={self.size}, "
            f"local_rank={self.local_rank}, ranks={self.ranks})"
        )


class ProcessGroupRegistry:
    """Creates and owns every named process group for one job.

    The registry is created once by
    :class:`hybrid_training.distributed.context.DistributedContext` and is the
    only place in the package that calls ``dist.new_group``.  Nothing else may
    create groups, because doing so out of band would break the ordering
    guarantee described in the module docstring.

    Args:
        topology: The rank grid.
        rank: This process's global rank.
        timeout: Collective timeout applied to newly created groups.
        backend: Backend name for the new groups, or ``None`` to inherit the
            default group's backend.
        create: Set to ``False`` to build a registry without touching
            ``torch.distributed``.  Used by unit tests that only exercise the
            rank arithmetic.

    Raises:
        DistributedInitializationError: If ``create`` is ``True`` and the
            default process group has not been initialised.
    """

    def __init__(
        self,
        topology: ParallelTopology,
        rank: int,
        *,
        timeout: timedelta | None = None,
        backend: str | None = None,
        create: bool = True,
    ) -> None:
        if create and not dist.is_initialized():
            raise DistributedInitializationError(
                format_error(
                    "groups.ProcessGroupRegistry",
                    "the default process group must exist before sub-groups are created",
                    rank=rank,
                    world_size=topology.world_size,
                    resolution="call init_distributed() / dist.init_process_group() first",
                )
            )
        self._topology = topology
        self._rank = rank
        self._handles: dict[str, GroupHandle] = {}
        self._created_rank_lists: list[tuple[str, tuple[int, ...]]] = []
        self._destroyed = False

        world_ranks = tuple(range(topology.world_size))
        default_group = dist.group.WORLD if create else None

        # Communicators are cached by rank list.  Two named groups often cover
        # exactly the same ranks -- `dp_shard` equals `data_parallel` whenever
        # the shard dimension is 1 -- and building a second communicator for
        # the same set is pure cost.  The cache key depends only on the rank
        # list, which every rank computes identically, so reuse cannot
        # desynchronise the construction sequence.
        communicators: dict[tuple[int, ...], Any] = {}
        if create:
            communicators[world_ranks] = default_group

        for name in GROUP_CREATION_ORDER:
            my_group: Any = None
            my_ranks: tuple[int, ...] = ()
            # Iterate every group along this dimension in a deterministic order.
            for ranks in topology.all_group_rank_lists(name):
                self._created_rank_lists.append((name, ranks))
                if not create:
                    handle_group = None
                elif len(ranks) == 1:
                    # A one-member group needs no communicator: every collective
                    # over it is the identity, and the wrappers short-circuit
                    # before touching `process_group`.  Skipping it removes a
                    # rendezvous that would compute nothing -- with Gloo each
                    # `new_group` is a real TCP-store round trip, and a
                    # four-dimensional topology creates many degenerate groups.
                    # Every rank makes the same decision from the same rank
                    # list, so the remaining calls stay in lockstep.
                    handle_group = None
                elif ranks in communicators:
                    handle_group = communicators[ranks]
                else:
                    kwargs: dict[str, Any] = {}
                    if timeout is not None:
                        kwargs["timeout"] = timeout
                    if backend is not None:
                        kwargs["backend"] = backend
                    handle_group = dist.new_group(ranks=list(ranks), **kwargs)
                    communicators[ranks] = handle_group
                if rank in ranks:
                    my_group = handle_group
                    my_ranks = ranks
            if not my_ranks:  # pragma: no cover - impossible: groups partition the world
                raise DistributedInitializationError(
                    format_error(
                        "groups.ProcessGroupRegistry",
                        f"rank is not a member of any {name!r} group",
                        rank=rank,
                        world_size=topology.world_size,
                        resolution="this indicates a bug in ParallelTopology",
                    )
                )
            self._handles[name] = GroupHandle(
                name=name,
                ranks=my_ranks,
                local_rank=my_ranks.index(rank),
                global_rank=rank,
                process_group=my_group,
            )

        # "world" is registered last and always maps onto the default group.
        self._handles["world"] = GroupHandle(
            name="world",
            ranks=world_ranks,
            local_rank=rank,
            global_rank=rank,
            process_group=default_group,
        )
        _LOGGER.debug(
            "created %d process groups (%d named)",
            len(self._created_rank_lists),
            len(self._handles),
        )

    # -- lookup -------------------------------------------------------------
    def get(self, name: str) -> GroupHandle:
        """Return the handle for a named group.

        Args:
            name: One of :data:`GROUP_CREATION_ORDER`, ``"world"``, or the
                alias ``"sequence_effective"`` which resolves to whichever
                group the sequence dimension is actually split over.

        Returns:
            The group handle.

        Raises:
            CollectiveError: If the name is unknown, or the registry has been
                destroyed.
        """
        if self._destroyed:
            raise CollectiveError(
                format_error(
                    "groups.get",
                    "the process-group registry has been destroyed",
                    rank=self._rank,
                    world_size=self._topology.world_size,
                    resolution="do not use groups after shutting the context down",
                )
            )
        if name == "sequence_effective":
            name = self._topology.sequence_group_name
        handle = self._handles.get(name)
        if handle is None:
            raise CollectiveError(
                format_error(
                    "groups.get",
                    "unknown process-group name",
                    rank=self._rank,
                    world_size=self._topology.world_size,
                    expected=[*sorted(self._handles), "sequence_effective"],
                    observed=name,
                    resolution="use one of the registered group names",
                )
            )
        return handle

    def __getitem__(self, name: str) -> GroupHandle:
        """Alias for :meth:`get`."""
        return self.get(name)

    def __contains__(self, name: str) -> bool:
        """Whether ``name`` resolves to a registered group."""
        return name in self._handles or name == "sequence_effective"

    @property
    def names(self) -> tuple[str, ...]:
        """Registered group names, in creation order."""
        return (*GROUP_CREATION_ORDER, "world")

    @property
    def creation_log(self) -> tuple[tuple[str, tuple[int, ...]], ...]:
        """Every ``(name, ranks)`` pair passed to ``new_group``, in order.

        Exposed so tests can assert that all ranks agree on the construction
        sequence -- the property that prevents the deadlock described above.
        """
        return tuple(self._created_rank_lists)

    def describe(self) -> str:
        """Multi-line description of every group this rank belongs to."""
        lines = [f"process groups for rank {self._rank}:"]
        for name in self.names:
            handle = self._handles[name]
            lines.append(
                f"  {name:<16} size={handle.size:<3} local_rank={handle.local_rank:<3} "
                f"ranks={handle.ranks}"
            )
        return "\n".join(lines)

    # -- teardown -----------------------------------------------------------
    def destroy(self) -> None:
        """Destroy every non-default communicator this registry created.

        The default group is *not* destroyed here; that is the job of
        :meth:`DistributedContext.shutdown`, which must run after all
        sub-groups are gone.

        Communicators shared between named groups are destroyed once; one-member
        groups have no communicator to destroy.

        Idempotent.
        """
        if self._destroyed:
            return
        self._destroyed = True
        seen: set[int] = set()
        default_group = dist.group.WORLD if dist.is_initialized() else None
        # Destroy in reverse creation order.  NCCL tolerates any order, but
        # reverse order mirrors normal resource-stack discipline and makes the
        # sequence easy to reason about in logs.
        for name in reversed(self.names):
            handle = self._handles.get(name)
            if handle is None or handle.process_group is None:
                continue
            if handle.process_group is default_group:
                continue
            identity = id(handle.process_group)
            if identity in seen:
                continue
            seen.add(identity)
            dist.destroy_process_group(handle.process_group)
        self._handles.clear()


def validate_group_membership(handle: GroupHandle, rank: int, operation: str) -> None:
    """Assert that ``rank`` belongs to ``handle``'s group.

    Calling a collective on a group you are not a member of is undefined
    behaviour in every backend -- typically a segfault or a hang.  Every
    wrapper in :mod:`hybrid_training.distributed.collectives` runs this check
    first because the cost (a tuple membership test) is nothing next to the
    cost of debugging a hang.

    Args:
        handle: The group being used.
        rank: The caller's global rank.
        operation: Name of the operation, for the error message.

    Raises:
        CollectiveError: If ``rank`` is not in the group.
    """
    if rank not in handle.ranks:
        raise CollectiveError(
            format_error(
                operation,
                f"rank is not a member of process group {handle.name!r}",
                rank=rank,
                expected=f"rank in {handle.ranks}",
                observed=rank,
                resolution=(
                    "pass the group the current rank belongs to; group handles are "
                    "per-rank views obtained from ProcessGroupRegistry.get()"
                ),
            )
        )


def known_group_names() -> tuple[str, ...]:
    """Return every group name the registry can produce.

    Exposed for error messages and for the documentation test that checks the
    API reference lists them all.
    """
    return (*DIMENSIONS, *COMPOSITE_GROUPS, "sequence_effective")
