"""The checkpoint manifest: a rank-independent description of what was saved.

The central idea
================
A manifest describes **global tensors**, never "what rank 2 had".  Each tensor
records its global shape and a list of shards, and each shard records a
half-open interval ``[offset, offset + length)`` of that tensor's *row-major
flattening*, together with the file the bytes live in.

That is the whole reason resharding works.  To load tensor ``t`` on a rank that
wants elements ``[a, b)``, a reader intersects ``[a, b)`` with every recorded
shard interval and reads the overlaps.  It never needs to know how many ranks
wrote the checkpoint, which rank wrote which piece, or what the sharding
strategy was.  Saved rank ids appear in the manifest only as provenance.

Worked example
==============
A ``(3, 4)`` weight (12 elements) saved by 4 ranks with an FSDP shard group of
4 has shards ``[0,3) [3,6) [6,9) [9,12)``.  A later run with a shard group of 3
wants ``[0,4) [4,8) [8,12)``.  Rank 0's request ``[0,4)`` intersects saved
shards 0 (``[0,3)``, fully) and 1 (``[3,6)``, giving ``[3,4)``), so rank 0 reads
two files and concatenates 3 + 1 elements.  No arithmetic anywhere in that
process refers to "4 ranks".

What tensor-parallel slices do
==============================
A tensor-parallel slice is **not** a sub-range of the full weight -- a
row-parallel slice is a set of strided columns, which no single
``(offset, length)`` can express.  Tensor-parallel slices are therefore stored
as *distinct tensors*, keyed ``name#tpKofN``, each with its own global shape.
Resharding across FSDP widths still works within each slice; changing the
tensor-parallel width is rejected explicitly by the reader.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import (
    CheckpointCorruptionError,
    CheckpointError,
    IncompleteCheckpointError,
    format_error,
)
from ..utils.tensors import ShardRange, intersect_ranges
from .format import (
    CURRENT_FORMAT_VERSION,
    validate_format_version,
    validate_shard_filename,
)

__all__ = [
    "CheckpointManifest",
    "FileRecord",
    "ShardRecord",
    "TensorRecord",
]


@dataclass(frozen=True)
class ShardRecord:
    """One contiguous piece of a global tensor.

    Attributes:
        rank: Rank that wrote the piece.  Provenance only -- readers never key
            off it.
        file: Shard filename, validated against the naming rule.
        offset: First element index within the tensor's row-major flattening.
        length: Number of elements.
        key: Key under which the piece is stored inside ``file``.
        padding: Trailing elements of this piece that are FSDP padding and
            carry no meaning.  Recorded so a reader can assert they are never
            copied into a parameter.
    """

    rank: int
    file: str
    offset: int
    length: int
    key: str
    padding: int = 0

    def __post_init__(self) -> None:
        if self.offset < 0 or self.length < 0:
            raise CheckpointError(
                format_error(
                    "manifest.ShardRecord",
                    "offset and length must be non-negative",
                    expected="offset >= 0 and length >= 0",
                    observed=f"offset={self.offset}, length={self.length}",
                    resolution="the manifest is malformed",
                )
            )
        validate_shard_filename(self.file)

    @property
    def range(self) -> ShardRange:
        """The interval this shard covers."""
        return ShardRange(start=self.offset, length=self.length)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        return {
            "rank": self.rank,
            "file": self.file,
            "offset": self.offset,
            "length": self.length,
            "key": self.key,
            "padding": self.padding,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ShardRecord:
        """Rebuild from :meth:`as_dict` output."""
        return cls(
            rank=int(payload["rank"]),
            file=str(payload["file"]),
            offset=int(payload["offset"]),
            length=int(payload["length"]),
            key=str(payload["key"]),
            padding=int(payload.get("padding", 0)),
        )


@dataclass
class TensorRecord:
    """A global tensor and the shards that cover it.

    Attributes:
        key: Storage key; the parameter name, possibly with a ``#tpKofN``
            suffix.
        name: The parameter name without the tensor-parallel suffix.
        global_shape: Shape of the whole tensor as this tensor-parallel rank
            sees it.
        dtype: ``str(torch.dtype)`` of the saved data.
        category: ``"model"``, ``"optimizer"`` or ``"buffer"``.
        partition_dim: Tensor-parallel partition dimension, or ``None``.
        tensor_parallel_size: Width of the tensor-parallel group at save time.
        tensor_parallel_rank: Which slice this tensor is.
        state_name: For optimizer tensors, the state key (``"exp_avg"``).
        shards: Pieces covering the tensor.
    """

    key: str
    name: str
    global_shape: tuple[int, ...]
    dtype: str
    category: str = "model"
    partition_dim: int | None = None
    tensor_parallel_size: int = 1
    tensor_parallel_rank: int = 0
    state_name: str = ""
    shards: list[ShardRecord] = field(default_factory=list)

    @property
    def numel(self) -> int:
        """Total elements in the global tensor."""
        total = 1
        for dimension in self.global_shape:
            total *= dimension
        return total

    def covered_elements(self) -> int:
        """Number of distinct elements the shards cover.

        Overlapping shards are counted once, so a manifest with duplicated
        coverage does not look complete when it is not.
        """
        if not self.shards:
            return 0
        intervals = sorted((s.offset, s.offset + s.length) for s in self.shards)
        covered = 0
        current_start, current_end = intervals[0]
        for start, end in intervals[1:]:
            if start > current_end:
                covered += current_end - current_start
                current_start, current_end = start, end
            else:
                current_end = max(current_end, end)
        covered += current_end - current_start
        return covered

    def validate(self) -> None:
        """Check that the shards exactly cover the tensor.

        Raises:
            IncompleteCheckpointError: If elements are missing.
            CheckpointCorruptionError: If a shard runs past the end of the
                tensor, or if two shards report different data for the same
                elements (detected as an overlap, which this writer never
                produces).
        """
        for shard in self.shards:
            if shard.offset + shard.length > self.numel:
                raise CheckpointCorruptionError(
                    format_error(
                        "manifest.TensorRecord.validate",
                        f"a shard of {self.key!r} runs past the end of the tensor",
                        expected=f"offset + length <= {self.numel}",
                        observed=shard.offset + shard.length,
                        resolution="the manifest disagrees with the recorded global shape",
                    )
                )
        total_shard_elements = sum(s.length for s in self.shards)
        covered = self.covered_elements()
        if total_shard_elements != covered:
            raise CheckpointCorruptionError(
                format_error(
                    "manifest.TensorRecord.validate",
                    f"the shards of {self.key!r} overlap, so two files claim the same "
                    "elements and the reader cannot tell which is authoritative",
                    expected=f"{covered} elements across non-overlapping shards",
                    observed=total_shard_elements,
                    resolution="the checkpoint was written by a buggy or mixed-version writer",
                )
            )
        if covered != self.numel:
            raise IncompleteCheckpointError(
                format_error(
                    "manifest.TensorRecord.validate",
                    f"the shards of {self.key!r} do not cover the whole tensor",
                    expected=f"{self.numel} elements",
                    observed=covered,
                    resolution="a shard file is missing; the checkpoint is unusable",
                )
            )

    def shards_overlapping(self, wanted: ShardRange) -> list[tuple[ShardRecord, ShardRange]]:
        """Return the shards intersecting ``wanted``, with the overlap.

        This is the core of resharding.

        Args:
            wanted: The interval the caller needs.

        Returns:
            ``(shard, overlap)`` pairs ordered by offset.
        """
        result: list[tuple[ShardRecord, ShardRange]] = []
        for shard in sorted(self.shards, key=lambda s: s.offset):
            overlap = intersect_ranges(shard.range, wanted)
            if overlap is not None:
                result.append((shard, overlap))
        return result

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        return {
            "key": self.key,
            "name": self.name,
            "global_shape": list(self.global_shape),
            "numel": self.numel,
            "dtype": self.dtype,
            "category": self.category,
            "partition_dim": self.partition_dim,
            "tensor_parallel_size": self.tensor_parallel_size,
            "tensor_parallel_rank": self.tensor_parallel_rank,
            "state_name": self.state_name,
            "shards": [s.as_dict() for s in sorted(self.shards, key=lambda s: s.offset)],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TensorRecord:
        """Rebuild from :meth:`as_dict` output."""
        return cls(
            key=str(payload["key"]),
            name=str(payload["name"]),
            global_shape=tuple(int(d) for d in payload["global_shape"]),
            dtype=str(payload["dtype"]),
            category=str(payload.get("category", "model")),
            partition_dim=(
                None if payload.get("partition_dim") is None else int(payload["partition_dim"])
            ),
            tensor_parallel_size=int(payload.get("tensor_parallel_size", 1)),
            tensor_parallel_rank=int(payload.get("tensor_parallel_rank", 0)),
            state_name=str(payload.get("state_name", "")),
            shards=[ShardRecord.from_dict(s) for s in payload.get("shards", [])],
        )


@dataclass(frozen=True)
class FileRecord:
    """Integrity metadata for one payload file.

    Attributes:
        name: Filename.
        sha256: Hex digest of the file's bytes.
        bytes: File size, checked before hashing so a truncated file is
            reported as truncated rather than as a checksum mismatch.
    """

    name: str
    sha256: str
    bytes: int

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        return {"name": self.name, "sha256": self.sha256, "bytes": self.bytes}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FileRecord:
        """Rebuild from :meth:`as_dict` output."""
        return cls(
            name=validate_shard_filename(str(payload["name"])),
            sha256=str(payload["sha256"]),
            bytes=int(payload["bytes"]),
        )


@dataclass
class CheckpointManifest:
    """The complete, rank-independent description of a checkpoint.

    Attributes:
        format_version: Format version this manifest was written in.
        writer_world_size: Number of ranks that wrote it.  Provenance only.
        topology: Parallel dimension sizes at save time.
        step: Optimizer step.
        tensors: Tensor records keyed by storage key.
        files: Integrity records keyed by filename.
        complete: Always ``True`` in a published manifest.  The field exists so
            that a manifest recovered from a crashed staging directory is
            visibly not a valid checkpoint.
        created_by: Package name and version that wrote it.
    """

    format_version: str = CURRENT_FORMAT_VERSION
    writer_world_size: int = 1
    topology: dict[str, Any] = field(default_factory=dict)
    step: int = 0
    tensors: dict[str, TensorRecord] = field(default_factory=dict)
    files: dict[str, FileRecord] = field(default_factory=dict)
    complete: bool = False
    created_by: str = ""

    # -- construction -------------------------------------------------------
    def add_shard(self, record: TensorRecord, shard: ShardRecord) -> None:
        """Register one shard, creating the tensor record if needed.

        Args:
            record: Tensor description.  Merged with any existing record for
                the same key.
            shard: The piece to register.

        Raises:
            CheckpointCorruptionError: If two ranks describe the same tensor
                with different shapes or dtypes.
        """
        existing = self.tensors.get(record.key)
        if existing is None:
            self.tensors[record.key] = TensorRecord(
                key=record.key,
                name=record.name,
                global_shape=record.global_shape,
                dtype=record.dtype,
                category=record.category,
                partition_dim=record.partition_dim,
                tensor_parallel_size=record.tensor_parallel_size,
                tensor_parallel_rank=record.tensor_parallel_rank,
                state_name=record.state_name,
                shards=[shard],
            )
            return
        if existing.global_shape != record.global_shape or existing.dtype != record.dtype:
            raise CheckpointCorruptionError(
                format_error(
                    "manifest.add_shard",
                    f"two ranks describe tensor {record.key!r} differently, so the "
                    "manifest cannot describe a single global tensor",
                    expected=f"shape={existing.global_shape}, dtype={existing.dtype}",
                    observed=f"shape={record.global_shape}, dtype={record.dtype}",
                    resolution=(
                        "the ranks are running different model definitions; check that "
                        "every rank built the model from the same configuration"
                    ),
                )
            )
        existing.shards.append(shard)

    # -- validation ---------------------------------------------------------
    def validate(self) -> None:
        """Check the manifest is internally consistent and complete.

        Raises:
            CheckpointVersionError: On an unreadable format version.
            IncompleteCheckpointError: If a tensor is not fully covered, or a
                shard names a file with no integrity record.
            CheckpointCorruptionError: On overlapping or out-of-range shards.
        """
        validate_format_version(self.format_version)
        if not self.complete:
            raise IncompleteCheckpointError(
                format_error(
                    "manifest.validate",
                    "the manifest is not marked complete, which means the writer did not "
                    "finish; a checkpoint is only published after every shard validates",
                    expected=True,
                    observed=self.complete,
                    resolution="discard this checkpoint and resume from an earlier one",
                )
            )
        for record in self.tensors.values():
            record.validate()
            for shard in record.shards:
                if shard.file not in self.files:
                    raise IncompleteCheckpointError(
                        format_error(
                            "manifest.validate",
                            f"tensor {record.key!r} references file {shard.file!r}, which "
                            "has no integrity record",
                            expected=sorted(self.files),
                            observed=shard.file,
                            resolution="the manifest is truncated or hand-edited",
                        )
                    )

    def referenced_files(self) -> set[str]:
        """Filenames any tensor depends on."""
        return {shard.file for record in self.tensors.values() for shard in record.shards}

    def tensors_by_category(self, category: str) -> dict[str, TensorRecord]:
        """Return the tensor records of one category.

        Args:
            category: ``"model"``, ``"optimizer"`` or ``"buffer"``.

        Returns:
            Mapping from storage key to record.
        """
        return {k: v for k, v in self.tensors.items() if v.category == category}

    # -- serialisation ------------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view, with deterministic key order."""
        return {
            "format_version": self.format_version,
            "created_by": self.created_by,
            "writer_world_size": self.writer_world_size,
            "topology": self.topology,
            "step": self.step,
            "complete": self.complete,
            "tensors": {k: self.tensors[k].as_dict() for k in sorted(self.tensors)},
            "files": {k: self.files[k].as_dict() for k in sorted(self.files)},
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialise to a JSON string.

        Args:
            indent: JSON indentation; ``None`` for the compact form.

        Returns:
            The JSON text.
        """
        return json.dumps(self.as_dict(), indent=indent, sort_keys=True)

    def write(self, path: Path) -> Path:
        """Write the manifest to ``path``.

        Args:
            path: Destination file.

        Returns:
            ``path``.
        """
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CheckpointManifest:
        """Rebuild from :meth:`as_dict` output.

        Args:
            payload: The parsed JSON object.

        Returns:
            The manifest.

        Raises:
            CheckpointError: If a required field is missing or mistyped.
        """
        try:
            return cls(
                format_version=str(payload["format_version"]),
                created_by=str(payload.get("created_by", "")),
                writer_world_size=int(payload.get("writer_world_size", 1)),
                topology=dict(payload.get("topology", {})),
                step=int(payload.get("step", 0)),
                complete=bool(payload.get("complete", False)),
                tensors={
                    key: TensorRecord.from_dict(value)
                    for key, value in payload.get("tensors", {}).items()
                },
                files={
                    key: FileRecord.from_dict(value)
                    for key, value in payload.get("files", {}).items()
                },
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointError(
                format_error(
                    "manifest.from_dict",
                    "the manifest is malformed",
                    expected="a manifest written by this package",
                    observed=repr(exc),
                    resolution="the file is not a valid checkpoint manifest",
                )
            ) from exc

    @classmethod
    def read(cls, path: Path) -> CheckpointManifest:
        """Load a manifest from disk.

        Args:
            path: Manifest file.

        Returns:
            The parsed manifest, not yet validated.

        Raises:
            IncompleteCheckpointError: If the file does not exist -- the usual
                sign of a checkpoint whose write was interrupted.
            CheckpointError: If the file is not valid JSON.
        """
        if not path.is_file():
            raise IncompleteCheckpointError(
                format_error(
                    "manifest.read",
                    "no manifest found; the manifest is written last, so its absence "
                    "means the checkpoint was never completed",
                    expected=str(path),
                    observed="missing",
                    resolution="resume from an earlier checkpoint",
                )
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CheckpointError(
                format_error(
                    "manifest.read",
                    "the manifest is not valid JSON",
                    expected="a JSON object",
                    observed=str(exc),
                    resolution="the file is corrupt",
                )
            ) from exc
        if not isinstance(payload, dict):
            raise CheckpointError(
                format_error(
                    "manifest.read",
                    "the manifest is not a JSON object",
                    expected="object",
                    observed=type(payload).__name__,
                    resolution="the file is corrupt",
                )
            )
        return cls.from_dict(payload)

    def summary(self) -> str:
        """Human-readable overview, used by ``scripts/inspect_checkpoint.py``."""
        model = self.tensors_by_category("model")
        optimizer = self.tensors_by_category("optimizer")
        buffers = self.tensors_by_category("buffer")
        total_elements = sum(r.numel for r in self.tensors.values())
        lines = [
            f"checkpoint format {self.format_version} written by {self.created_by or '?'}",
            f"  step               : {self.step}",
            f"  writer world size  : {self.writer_world_size}",
            f"  topology           : {self.topology}",
            f"  complete           : {self.complete}",
            f"  model tensors      : {len(model)}",
            f"  optimizer tensors  : {len(optimizer)}",
            f"  buffer tensors     : {len(buffers)}",
            f"  total elements     : {total_elements}",
            f"  payload files      : {len(self.files)}",
        ]
        return "\n".join(lines)
