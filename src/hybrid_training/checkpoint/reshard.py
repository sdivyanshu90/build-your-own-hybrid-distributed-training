"""Reading tensor ranges out of a checkpoint, across any shard layout.

Resharding is interval arithmetic
=================================
A checkpoint records, for every global tensor, a set of half-open intervals of
its row-major flattening and the file each interval lives in.  A loading rank
knows which interval *it* wants.  The bytes to read are the intersections.
That is the entire algorithm; nothing about the number of writers, the number
of readers, or the sharding strategy enters into it.

.. code-block:: text

    saved with 4 shards      |--0--|--1--|--2--|--3--|      each 3 elements
    wanted by 3 readers      |---0----|---1----|---2---|    each 4 elements

    reader 0 wants [0, 4)  -> shard 0 fully ([0,3)) + shard 1 partially ([3,4))
    reader 1 wants [4, 8)  -> shard 1 partially ([4,6)) + shard 2 fully ([6,8))
    reader 2 wants [8,12)  -> shard 2 partially ([8,9)) + shard 3 fully ([9,12))

File access
===========
A naive implementation would open a shard file once per tensor it contributes
to; a checkpoint with 200 tensors and 8 writers would then perform 1600 loads.
:class:`ShardFileCache` loads each file at most once and keeps it until the
caller drops the cache, trading memory for a linear number of reads.  Since a
loading rank typically needs a contiguous span of the global tensor space, it
usually touches only one or two files per tensor.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from ..errors import (
    CheckpointCorruptionError,
    CheckpointError,
    IncompleteCheckpointError,
    format_error,
)
from ..logging import get_logger
from ..utils.tensors import ShardRange
from .format import file_digest, resolve_inside, validate_shard_filename
from .manifest import CheckpointManifest, TensorRecord

__all__ = ["ShardFileCache", "read_tensor_range", "verify_files"]

_LOGGER = get_logger(__name__)


class ShardFileCache:
    """Lazily loads shard payload files, at most once each.

    Args:
        directory: Checkpoint directory.
        manifest: The validated manifest, used to check integrity on first
            access.
        verify_checksums: Recompute and compare each file's SHA-256 the first
            time it is opened.  Costs one extra pass over the bytes; leaving it
            on is strongly recommended, because a silently corrupted shard
            produces a model that loads cleanly and behaves wrongly.

    Example:
        >>> # doctest: +SKIP
        >>> cache = ShardFileCache(path, manifest)
        >>> payload = cache.get("rank-00000.pt")
    """

    def __init__(
        self,
        directory: Path,
        manifest: CheckpointManifest,
        *,
        verify_checksums: bool = True,
    ) -> None:
        self._directory = Path(directory)
        self._manifest = manifest
        self._verify = verify_checksums
        self._loaded: dict[str, Mapping[str, torch.Tensor]] = {}

    def get(self, filename: str) -> Mapping[str, torch.Tensor]:
        """Return the tensor mapping stored in ``filename``.

        Args:
            filename: Name from a manifest shard record.

        Returns:
            Mapping from storage key to 1-D tensor.

        Raises:
            IncompleteCheckpointError: If the file is missing.
            CheckpointCorruptionError: If the file fails verification or cannot
                be deserialised.
        """
        cached = self._loaded.get(filename)
        if cached is not None:
            return cached

        validate_shard_filename(filename)
        path = resolve_inside(self._directory, filename)
        if not path.is_file():
            raise IncompleteCheckpointError(
                format_error(
                    "reshard.ShardFileCache.get",
                    "a shard file listed in the manifest is missing",
                    expected=filename,
                    observed="not present in the checkpoint directory",
                    resolution=(
                        "the checkpoint is incomplete; restore the missing file or "
                        "resume from an earlier checkpoint"
                    ),
                )
            )

        record = self._manifest.files.get(filename)
        if record is None:
            raise CheckpointCorruptionError(
                format_error(
                    "reshard.ShardFileCache.get",
                    "the file has no integrity record in the manifest",
                    expected="an entry under manifest['files']",
                    observed=filename,
                    resolution="the manifest is truncated or hand-edited",
                )
            )
        actual_size = path.stat().st_size
        if actual_size != record.bytes:
            raise CheckpointCorruptionError(
                format_error(
                    "reshard.ShardFileCache.get",
                    f"shard file {filename!r} has the wrong size, so it was truncated "
                    "or overwritten after the checkpoint was published",
                    expected=f"{record.bytes} bytes",
                    observed=f"{actual_size} bytes",
                    resolution="restore the file from a backup or discard the checkpoint",
                )
            )
        if self._verify:
            digest = file_digest(path)
            if digest != record.sha256:
                raise CheckpointCorruptionError(
                    format_error(
                        "reshard.ShardFileCache.get",
                        f"shard file {filename!r} failed checksum verification; its "
                        "contents differ from what the writer recorded",
                        expected=record.sha256,
                        observed=digest,
                        resolution=(
                            "the file is corrupt; restore it from a backup or discard "
                            "the checkpoint. Pass verify_checksums=False only when you "
                            "have another reason to trust the bytes."
                        ),
                    )
                )

        try:
            # weights_only=True restricts unpickling to tensors and a small set
            # of primitives, so a hostile payload cannot execute code.
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise CheckpointCorruptionError(
                format_error(
                    "reshard.ShardFileCache.get",
                    f"shard file {filename!r} could not be deserialised",
                    expected="a tensor mapping saved with torch.save",
                    observed=repr(exc),
                    resolution=(
                        "the file is corrupt, or was written by a different tool; note "
                        "that this reader deliberately refuses pickled objects other "
                        "than tensors"
                    ),
                )
            ) from exc
        if not isinstance(payload, dict):
            raise CheckpointCorruptionError(
                format_error(
                    "reshard.ShardFileCache.get",
                    f"shard file {filename!r} does not contain a mapping",
                    expected="dict[str, Tensor]",
                    observed=type(payload).__name__,
                    resolution="the file was not written by this package",
                )
            )
        self._loaded[filename] = payload
        return payload

    def clear(self) -> None:
        """Drop every cached payload, releasing the memory."""
        self._loaded.clear()

    @property
    def loaded_files(self) -> tuple[str, ...]:
        """Names of the files loaded so far, for diagnostics and tests."""
        return tuple(sorted(self._loaded))


def read_tensor_range(
    record: TensorRecord,
    wanted: ShardRange,
    cache: ShardFileCache,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Assemble ``wanted`` elements of a global tensor from its shards.

    Args:
        record: The tensor's manifest record.
        wanted: The interval to materialise, in the tensor's flat coordinates.
        cache: Open shard files.
        dtype: Cast the result to this dtype.  Defaults to whatever the shards
            hold.

    Returns:
        A 1-D tensor of length ``wanted.length``.

    Raises:
        IncompleteCheckpointError: If the shards do not cover ``wanted``.
        CheckpointCorruptionError: If a shard's stored tensor has a different
            length from what the manifest claims.
    """
    if wanted.length == 0:
        return torch.empty(0, dtype=dtype or torch.float32)
    if wanted.end > record.numel:
        raise CheckpointError(
            format_error(
                "reshard.read_tensor_range",
                f"the requested range of {record.key!r} runs past the end of the tensor; "
                "the current model is larger than the one that was saved",
                expected=f"end <= {record.numel}",
                observed=wanted.end,
                resolution="resume with the model definition that produced the checkpoint",
            )
        )

    overlaps = record.shards_overlapping(wanted)
    if not overlaps:
        raise IncompleteCheckpointError(
            format_error(
                "reshard.read_tensor_range",
                f"no shard of {record.key!r} covers the requested range",
                expected=f"a shard intersecting [{wanted.start}, {wanted.end})",
                observed=[(s.offset, s.length) for s in record.shards],
                resolution="the checkpoint is incomplete",
            )
        )

    out: torch.Tensor | None = None
    filled = 0
    cursor = wanted.start
    for shard, overlap in overlaps:
        if overlap.start != cursor:
            raise IncompleteCheckpointError(
                format_error(
                    "reshard.read_tensor_range",
                    f"a gap exists in the shards of {record.key!r}",
                    expected=f"a shard starting at element {cursor}",
                    observed=f"next shard starts at {overlap.start}",
                    resolution="a shard file is missing from the checkpoint",
                )
            )
        payload = cache.get(shard.file)
        stored = payload.get(shard.key)
        if stored is None:
            raise CheckpointCorruptionError(
                format_error(
                    "reshard.read_tensor_range",
                    f"shard file {shard.file!r} does not contain key {shard.key!r}",
                    expected=shard.key,
                    observed=sorted(payload)[:8],
                    resolution="the manifest and the payload disagree; the checkpoint is corrupt",
                )
            )
        flat = stored.reshape(-1)
        if flat.numel() != shard.length:
            raise CheckpointCorruptionError(
                format_error(
                    "reshard.read_tensor_range",
                    f"shard {shard.key!r} has a different length from its manifest record",
                    expected=shard.length,
                    observed=flat.numel(),
                    resolution="the manifest and the payload disagree; the checkpoint is corrupt",
                )
            )
        if out is None:
            out = torch.empty(wanted.length, dtype=dtype or flat.dtype)
        begin = overlap.start - shard.offset
        out[filled : filled + overlap.length] = flat[begin : begin + overlap.length].to(out.dtype)
        filled += overlap.length
        cursor = overlap.end

    assert out is not None  # guaranteed: `overlaps` is non-empty
    if filled != wanted.length:
        raise IncompleteCheckpointError(
            format_error(
                "reshard.read_tensor_range",
                f"the shards of {record.key!r} cover only part of the requested range",
                expected=wanted.length,
                observed=filled,
                resolution="a shard file is missing from the checkpoint",
            )
        )
    return out


def verify_files(
    directory: Path, manifest: CheckpointManifest, *, filenames: set[str] | None = None
) -> dict[str, str]:
    """Verify shard files against the manifest's checksums.

    Args:
        directory: Checkpoint directory.
        manifest: Validated manifest.
        filenames: Subset to check.  Defaults to every referenced file.

    Returns:
        Mapping from filename to its computed digest.

    Raises:
        IncompleteCheckpointError: If a referenced file is missing.
        CheckpointCorruptionError: On a size or checksum mismatch.
    """
    targets = sorted(filenames if filenames is not None else manifest.referenced_files())
    digests: dict[str, str] = {}
    for name in targets:
        record = manifest.files.get(name)
        if record is None:
            raise CheckpointCorruptionError(
                format_error(
                    "reshard.verify_files",
                    "file has no integrity record",
                    expected="an entry under manifest['files']",
                    observed=name,
                    resolution="the manifest is truncated",
                )
            )
        path = resolve_inside(directory, validate_shard_filename(name))
        if not path.is_file():
            raise IncompleteCheckpointError(
                format_error(
                    "reshard.verify_files",
                    "a referenced shard file is missing",
                    expected=name,
                    observed="absent",
                    resolution="the checkpoint is incomplete",
                )
            )
        size = path.stat().st_size
        if size != record.bytes:
            raise CheckpointCorruptionError(
                format_error(
                    "reshard.verify_files",
                    f"{name!r} has the wrong size",
                    expected=record.bytes,
                    observed=size,
                    resolution="the file was truncated or replaced",
                )
            )
        digest = file_digest(path)
        if digest != record.sha256:
            raise CheckpointCorruptionError(
                format_error(
                    "reshard.verify_files",
                    f"{name!r} failed checksum verification",
                    expected=record.sha256,
                    observed=digest,
                    resolution="the file is corrupt",
                )
            )
        digests[name] = digest
    _LOGGER.debug("verified %d checkpoint file(s)", len(digests))
    return digests


def describe_reshard(manifest: CheckpointManifest, key: str, wanted: ShardRange) -> dict[str, Any]:
    """Describe which shards a read would touch, without reading anything.

    Used by ``scripts/inspect_checkpoint.py`` and by the resharding tests to
    assert on the *plan* rather than only on the result.

    Args:
        manifest: The manifest.
        key: Storage key of the tensor.
        wanted: The interval of interest.

    Returns:
        A mapping describing the plan.

    Raises:
        CheckpointError: If the key is unknown.
    """
    record = manifest.tensors.get(key)
    if record is None:
        raise CheckpointError(
            format_error(
                "reshard.describe_reshard",
                "unknown tensor key",
                expected=sorted(manifest.tensors)[:8],
                observed=key,
                resolution="check the key against the manifest",
            )
        )
    overlaps = record.shards_overlapping(wanted)
    return {
        "key": key,
        "global_shape": list(record.global_shape),
        "wanted": [wanted.start, wanted.length],
        "sources": [
            {
                "file": shard.file,
                "saved_rank": shard.rank,
                "shard_range": [shard.offset, shard.length],
                "overlap": [overlap.start, overlap.length],
            }
            for shard, overlap in overlaps
        ],
    }
