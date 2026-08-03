"""Checkpoint format constants, naming rules and path-safety checks.

Layout on disk
==============
.. code-block:: text

    checkpoint-step-000100/
    |-- manifest.json      # what tensors exist, who owns which bytes, checksums
    |-- metadata.json      # step, epoch, config, scheduler, scaler, RNG scalars
    |-- rank-00000.pt      # tensor payload written by rank 0
    |-- rank-00001.pt
    `-- ...

Two files, two jobs
===================
``manifest.json`` and ``metadata.json`` are **pure JSON**.  They can be read,
diffed, validated and inspected without ``torch`` being installed and without
executing anything.  Only the ``rank-*.pt`` files hold tensors, and they are
loaded with ``weights_only=True``, which restricts unpickling to tensors and a
small set of primitives.

This split is the whole security story of the format: everything that decides
*what to do* is inert JSON, and everything that is deserialised by PyTorch is
plain tensor data.  See ``docs/17_security`` notes in
``docs/08_distributed_checkpointing.md``.

Atomicity
=========
Writes go to a sibling staging directory named ``.staging-<name>-<uuid>``.  The
manifest -- the only file a reader trusts -- is written **last**, after every
shard has been written and checksummed.  The staging directory is then renamed
into place, which is atomic on POSIX filesystems within a single mount point.
A reader therefore never observes a half-written checkpoint: either the final
directory does not exist, or it exists complete.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from ..errors import CheckpointError, CheckpointVersionError, format_error

__all__ = [
    "CHECKPOINT_DIRECTORY_PATTERN",
    "CURRENT_FORMAT_VERSION",
    "MANIFEST_FILENAME",
    "METADATA_FILENAME",
    "SUPPORTED_FORMAT_VERSIONS",
    "checkpoint_directory_name",
    "file_digest",
    "resolve_inside",
    "shard_filename",
    "staging_directory_name",
    "step_from_directory_name",
    "validate_format_version",
    "validate_shard_filename",
]

#: Format version written into every manifest.  Bump the *major* component for
#: a change a previous reader could misinterpret, and the *minor* component for
#: an additive change a previous reader can ignore.
CURRENT_FORMAT_VERSION = "1.0"

#: Versions this build can read.  A reader that encounters anything else fails
#: loudly rather than guessing, because a checkpoint misread as the wrong
#: version produces plausible-looking garbage weights.
SUPPORTED_FORMAT_VERSIONS = frozenset({"1.0"})

MANIFEST_FILENAME = "manifest.json"
METADATA_FILENAME = "metadata.json"

#: ``checkpoint-step-000100``: zero-padded so lexical order matches numeric
#: order, which makes "the newest checkpoint" a simple ``max()`` over names.
CHECKPOINT_DIRECTORY_PATTERN = re.compile(r"^checkpoint-step-(\d{6,})$")

#: Shard files are ``rank-00000.pt``.  The pattern is enforced on *read* as
#: well as on write: a manifest that names any other file is rejected.
_SHARD_FILENAME_PATTERN = re.compile(r"^rank-(\d{5,})\.pt$")

_STAGING_PREFIX = ".staging-"


def checkpoint_directory_name(step: int) -> str:
    """Return the directory name for a checkpoint at ``step``.

    Args:
        step: Optimizer step.

    Returns:
        A name such as ``"checkpoint-step-000100"``.

    Raises:
        CheckpointError: If ``step`` is negative.
    """
    if step < 0:
        raise CheckpointError(
            format_error(
                "checkpoint.directory_name",
                "step must be non-negative",
                expected=">= 0",
                observed=step,
                resolution="pass the current optimizer step",
            )
        )
    return f"checkpoint-step-{step:06d}"


def step_from_directory_name(name: str) -> int | None:
    """Extract the step from a checkpoint directory name.

    Args:
        name: Directory base name.

    Returns:
        The step, or ``None`` when the name is not a checkpoint directory.
    """
    match = CHECKPOINT_DIRECTORY_PATTERN.match(name)
    return int(match.group(1)) if match else None


def staging_directory_name(final_name: str, token: str) -> str:
    """Return the staging directory name used while writing ``final_name``.

    The leading dot keeps it out of ordinary globs, and the token keeps two
    concurrent writers from colliding.

    Args:
        final_name: The directory the staging area will be renamed to.
        token: A unique suffix, normally a UUID hex fragment.

    Returns:
        The staging directory name.
    """
    return f"{_STAGING_PREFIX}{final_name}-{token}"


def shard_filename(rank: int) -> str:
    """Return the payload filename for a rank.

    Args:
        rank: Global rank that wrote the file.

    Returns:
        A name such as ``"rank-00003.pt"``.

    Raises:
        CheckpointError: If ``rank`` is negative.
    """
    if rank < 0:
        raise CheckpointError(
            format_error(
                "checkpoint.shard_filename",
                "rank must be non-negative",
                expected=">= 0",
                observed=rank,
                resolution="pass a global rank",
            )
        )
    return f"rank-{rank:05d}.pt"


def validate_shard_filename(name: str) -> str:
    """Reject any filename that is not a well-formed shard name.

    Applied to every filename read out of a manifest.  A manifest is data, and
    data from an untrusted source must never be able to steer a read towards an
    arbitrary path -- ``"../../etc/passwd"`` or ``"/etc/shadow"`` are both
    rejected here, before the path is ever joined.

    Args:
        name: Candidate filename.

    Returns:
        ``name``, unchanged, when it is valid.

    Raises:
        CheckpointError: If the name is not of the form ``rank-NNNNN.pt``.
    """
    if not _SHARD_FILENAME_PATTERN.match(name):
        raise CheckpointError(
            format_error(
                "checkpoint.validate_shard_filename",
                "the manifest names a file that does not match the shard naming rule; "
                "refusing to read it, because a manifest must never be able to point "
                "the reader at an arbitrary path",
                expected="rank-NNNNN.pt",
                observed=name,
                resolution="the checkpoint directory has been tampered with or corrupted",
            )
        )
    return name


def resolve_inside(root: Path, name: str) -> Path:
    """Join ``name`` onto ``root`` and prove the result stays inside ``root``.

    Belt and braces alongside :func:`validate_shard_filename`: the name is
    checked for shape, and the *resolved* path is checked for containment, so a
    symlink inside the checkpoint directory cannot redirect a read either.

    Args:
        root: Checkpoint directory.
        name: Relative filename.

    Returns:
        The resolved absolute path.

    Raises:
        CheckpointError: If the resolved path escapes ``root``.
    """
    root_resolved = root.resolve()
    candidate = (root_resolved / name).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise CheckpointError(
            format_error(
                "checkpoint.resolve_inside",
                "the resolved path escapes the checkpoint directory",
                expected=f"a path under {root_resolved}",
                observed=str(candidate),
                resolution="remove the symlink or path component that redirects outside",
            )
        ) from exc
    return candidate


def file_digest(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file.

    Streamed in chunks so a multi-gigabyte shard does not have to be held in
    memory to be verified.

    Args:
        path: File to hash.
        chunk_size: Read granularity in bytes.

    Returns:
        Lowercase hex digest.

    Raises:
        CheckpointError: If the file cannot be read.
    """
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise CheckpointError(
            format_error(
                "checkpoint.file_digest",
                "could not read a checkpoint file",
                expected="a readable file",
                observed=f"{path}: {exc}",
                resolution="check filesystem permissions and that the file exists",
            )
        ) from exc
    return digest.hexdigest()


def validate_format_version(version: str) -> str:
    """Check that this build can read a given checkpoint format version.

    Args:
        version: The ``format_version`` field from a manifest.

    Returns:
        ``version``, unchanged, when it is supported.

    Raises:
        CheckpointVersionError: If the version is unknown.
    """
    if version not in SUPPORTED_FORMAT_VERSIONS:
        raise CheckpointVersionError(
            format_error(
                "checkpoint.validate_format_version",
                "unsupported checkpoint format version",
                expected=sorted(SUPPORTED_FORMAT_VERSIONS),
                observed=version,
                resolution=(
                    "use the version of this package that wrote the checkpoint, or "
                    "convert the checkpoint with a migration script"
                ),
            )
        )
    return version
