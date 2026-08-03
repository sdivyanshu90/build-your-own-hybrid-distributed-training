"""Distributed runtime: context, topology, process groups and collectives.

Import order inside this package is deliberately acyclic::

    topology  -> (config, errors)          pure arithmetic, no torch.distributed
    groups    -> topology                  the only caller of dist.new_group
    context   -> groups, topology          owns the process group and device
    collectives -> groups                  explicit-group collective wrappers
    launch    -> context                   multi-process launcher for tests
"""

from .collectives import (
    AsyncWork,
    CommunicationRecorder,
    CommunicationStats,
    ReduceOp,
    all_gather_object_in_group,
    all_gather_tensor,
    all_reduce,
    all_to_all_tensor,
    assert_metadata_consistent,
    assert_tensor_consistent,
    broadcast,
    reduce_scatter_tensor,
)
from .context import (
    DistributedContext,
    LaunchEnvironment,
    current_context,
    distributed_context,
    find_free_port,
    init_distributed,
    is_context_active,
)
from .groups import GROUP_CREATION_ORDER, GroupHandle, ProcessGroupRegistry
from .launch import WorkerFailure, WorkerResult, launch_workers
from .topology import COMPOSITE_GROUPS, DIMENSIONS, ParallelTopology, RankCoordinates

__all__ = [
    "COMPOSITE_GROUPS",
    "DIMENSIONS",
    "GROUP_CREATION_ORDER",
    "AsyncWork",
    "CommunicationRecorder",
    "CommunicationStats",
    "DistributedContext",
    "GroupHandle",
    "LaunchEnvironment",
    "ParallelTopology",
    "ProcessGroupRegistry",
    "RankCoordinates",
    "ReduceOp",
    "WorkerFailure",
    "WorkerResult",
    "all_gather_object_in_group",
    "all_gather_tensor",
    "all_reduce",
    "all_to_all_tensor",
    "assert_metadata_consistent",
    "assert_tensor_consistent",
    "broadcast",
    "current_context",
    "distributed_context",
    "find_free_port",
    "init_distributed",
    "is_context_active",
    "launch_workers",
    "reduce_scatter_tensor",
]
