"""Exception hierarchy for :mod:`hybrid_training`.

Design rules that the whole package follows:

* **Every** user-visible failure is a subclass of :class:`HybridTrainingError`.
  Library code never raises bare ``RuntimeError`` for a condition a user can
  actually cause, and never swallows an exception with a bare ``except``.
* Error messages are *rank aware*.  In a 32-rank job the single most useful
  piece of information is which rank produced the message, so the helper
  :func:`format_error` renders a consistent five-part message:

  ``[rank R/W] operation: <what went wrong>; expected <x>, observed <y>. Fix: <hint>``

* A distributed failure that only one rank can detect is a *hang generator*:
  the detecting rank raises and stops calling collectives while every other
  rank blocks forever.  Where a check can be made collectively we do so (see
  :func:`hybrid_training.distributed.collectives.assert_consistent_across_group`),
  and where it cannot we say so in the message so the operator knows to look
  for a partner failure on another rank.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CheckpointCorruptionError",
    "CheckpointError",
    "CheckpointTopologyError",
    "CheckpointVersionError",
    "CollectiveError",
    "ConfigurationError",
    "DistributedInitializationError",
    "HybridTrainingError",
    "IncompleteCheckpointError",
    "ParameterConsistencyError",
    "ShardingError",
    "TensorParallelError",
    "TopologyError",
    "UnsupportedFeatureError",
    "format_error",
]


def format_error(
    operation: str,
    problem: str,
    *,
    rank: int | None = None,
    world_size: int | None = None,
    expected: Any = None,
    observed: Any = None,
    resolution: str | None = None,
) -> str:
    """Render a rank-aware, actionable error message.

    Args:
        operation: The logical operation that failed, e.g. ``"fsdp.unshard"``
            or ``"checkpoint.verify"``.  Use dotted names so messages can be
            grepped.
        problem: One sentence describing what is wrong.
        rank: Global rank, when known.
        world_size: Global world size, when known.
        expected: The value the code required.  Rendered only if not ``None``.
        observed: The value the code actually saw.  Rendered only if not
            ``None``.
        resolution: A concrete next step for the operator.

    Returns:
        The formatted message.

    Example:
        >>> format_error(
        ...     "topology.validate",
        ...     "parallel sizes do not factor the world size",
        ...     rank=3, world_size=8, expected=8, observed=6,
        ...     resolution="set dp*shard*seq*tensor == world_size",
        ... )
        '[rank 3/8] topology.validate: parallel sizes do not factor the world size; expected 8, observed 6. Fix: set dp*shard*seq*tensor == world_size'
    """
    prefix = ""
    if rank is not None and world_size is not None:
        prefix = f"[rank {rank}/{world_size}] "
    elif rank is not None:
        prefix = f"[rank {rank}] "

    message = f"{prefix}{operation}: {problem}"
    if expected is not None or observed is not None:
        message += f"; expected {expected!r}, observed {observed!r}"
    if resolution:
        message += f". Fix: {resolution}"
    return message


class HybridTrainingError(Exception):
    """Base class for every error raised by this package.

    Catching this type is the supported way to distinguish "the framework told
    me I did something wrong" from "PyTorch or the OS blew up".
    """


class ConfigurationError(HybridTrainingError, ValueError):
    """A dataclass configuration is internally inconsistent or unsupported.

    Inherits :class:`ValueError` so that code validating user input with
    ``except ValueError`` keeps working.
    """


class DistributedInitializationError(HybridTrainingError):
    """The distributed runtime could not be brought up or torn down safely.

    Raised for missing ``torchrun`` environment variables, double
    initialisation, backend/device mismatches and unclean shutdown.
    """


class TopologyError(ConfigurationError):
    """The requested multi-dimensional parallel topology is invalid.

    Most commonly ``dp * shard * sequence * tensor != world_size``.
    """


class CollectiveError(HybridTrainingError):
    """A collective operation was issued incorrectly.

    This covers calling a collective with a tensor whose shape or dtype
    disagrees across the group, passing a process group the current rank does
    not belong to, and using an operation the backend does not support.
    """


class ParameterConsistencyError(HybridTrainingError):
    """Model parameters disagree across ranks that must hold identical copies.

    Raised by the DDP and FSDP wrappers during their start-up consistency
    checks, and by the tensor-parallel layers when replicated inputs differ.
    """


class ShardingError(HybridTrainingError):
    """A parameter/gradient/optimizer-state sharding invariant was violated."""


class TensorParallelError(HybridTrainingError):
    """A tensor-parallel layer was constructed or driven incorrectly.

    Typical causes: a feature dimension that is not divisible by the
    tensor-parallel size, or feeding a row-parallel layer a full-width input
    when it was configured to expect an already-sharded input.
    """


class UnsupportedFeatureError(HybridTrainingError, NotImplementedError):
    """A configuration is legal in principle but deliberately not supported.

    Used for the cases this project documents as out of scope, e.g. tied
    weights spanning two different FSDP units.  Inherits
    :class:`NotImplementedError` so it reads naturally at call sites, but it is
    still a :class:`HybridTrainingError` and is *always* raised with a message
    explaining why the limitation exists and what to do instead.
    """


class CheckpointError(HybridTrainingError):
    """Base class for checkpoint save/load failures."""


class IncompleteCheckpointError(CheckpointError):
    """A checkpoint directory is missing files the manifest requires.

    A checkpoint is only ever *published* (renamed into place) once every shard
    file exists and validates, so seeing this error normally means the
    directory was hand-edited, the filesystem lost data, or a save crashed
    while writing into the staging directory.
    """


class CheckpointCorruptionError(CheckpointError):
    """A shard file failed checksum verification or could not be deserialised."""


class CheckpointVersionError(CheckpointError):
    """The checkpoint format version is not readable by this build."""


class CheckpointTopologyError(CheckpointError):
    """The saved topology cannot be transformed into the requested topology.

    Resharding across *sharding* world sizes is supported because shards are
    described by global flat offsets.  Changing tensor-parallel width is not,
    because tensor-parallel partitioning is a property of the *model
    definition* (which slice of a weight matrix a rank computes with), not just
    of storage layout.
    """
