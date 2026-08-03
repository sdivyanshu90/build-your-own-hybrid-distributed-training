"""Unit tests for the checkpoint format, manifest and integrity rules.

These need no process group: a manifest is a JSON document and its validation
is pure logic.  The negative cases matter most -- a manifest is *data*, and
data that can steer a reader towards an arbitrary path or a partially written
tensor is the security and correctness surface of the whole format.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hybrid_training.checkpoint.format import (
    CURRENT_FORMAT_VERSION,
    checkpoint_directory_name,
    file_digest,
    resolve_inside,
    shard_filename,
    staging_directory_name,
    step_from_directory_name,
    validate_format_version,
    validate_shard_filename,
)
from hybrid_training.checkpoint.manifest import (
    CheckpointManifest,
    FileRecord,
    ShardRecord,
    TensorRecord,
)
from hybrid_training.errors import (
    CheckpointCorruptionError,
    CheckpointError,
    CheckpointVersionError,
    IncompleteCheckpointError,
)
from hybrid_training.utils.tensors import ShardRange


class TestNaming:
    """Directory and file naming rules."""

    def test_directory_names_sort_numerically(self) -> None:
        """Zero padding makes lexical order match numeric order."""
        names = [checkpoint_directory_name(s) for s in (5, 50, 500, 5000)]
        assert names == sorted(names)
        assert names[0] == "checkpoint-step-000005"

    def test_step_round_trip(self) -> None:
        """A directory name yields back the step it encodes."""
        assert step_from_directory_name(checkpoint_directory_name(1234)) == 1234
        assert step_from_directory_name("not-a-checkpoint") is None
        assert step_from_directory_name("checkpoint-step-12") is None  # too few digits

    def test_negative_step_rejected(self) -> None:
        """Negative steps are a programming error."""
        with pytest.raises(CheckpointError, match="step must be non-negative"):
            checkpoint_directory_name(-1)

    def test_shard_filenames(self) -> None:
        """Shard filenames are zero-padded and validated on the way back in."""
        assert shard_filename(3) == "rank-00003.pt"
        assert validate_shard_filename("rank-00003.pt") == "rank-00003.pt"
        with pytest.raises(CheckpointError, match="rank must be non-negative"):
            shard_filename(-1)

    def test_staging_directory_is_hidden_and_unique(self) -> None:
        """Staging names start with a dot and carry the uniqueness token."""
        name = staging_directory_name("checkpoint-step-000010", "abc123")
        assert name.startswith(".")
        assert "abc123" in name


class TestPathSafety:
    """A manifest must never be able to redirect a read outside the directory."""

    @pytest.mark.parametrize(
        "name",
        [
            "../../etc/passwd",
            "/etc/shadow",
            "rank-00000.pt/../../escape.pt",
            "subdir/rank-00000.pt",
            "rank-0.pt",
            "rank-00000.pth",
            "manifest.json",
            "",
        ],
    )
    def test_hostile_filenames_rejected(self, name: str) -> None:
        """Anything that is not exactly ``rank-NNNNN.pt`` is refused."""
        with pytest.raises(CheckpointError, match="does not match the shard naming rule"):
            validate_shard_filename(name)

    def test_resolve_inside_accepts_contained_paths(self, tmp_path: Path) -> None:
        """A legitimate filename resolves under the checkpoint directory."""
        resolved = resolve_inside(tmp_path, "rank-00000.pt")
        assert resolved.parent == tmp_path.resolve()

    def test_resolve_inside_rejects_traversal(self, tmp_path: Path) -> None:
        """A relative escape is caught even though the name never touches disk."""
        with pytest.raises(CheckpointError, match="escapes the checkpoint directory"):
            resolve_inside(tmp_path, "../outside.pt")

    def test_resolve_inside_rejects_symlink_escape(self, tmp_path: Path) -> None:
        """A symlink pointing outside the directory is caught by resolution."""
        outside = tmp_path.parent / "outside-target"
        outside.mkdir(exist_ok=True)
        link = tmp_path / "link"
        link.symlink_to(outside, target_is_directory=True)
        try:
            with pytest.raises(CheckpointError, match="escapes the checkpoint directory"):
                resolve_inside(tmp_path, "link/../../evil.pt")
        finally:
            link.unlink()
            outside.rmdir()


class TestVersioning:
    """Format-version gating."""

    def test_current_version_accepted(self) -> None:
        """The version this build writes is readable by this build."""
        assert validate_format_version(CURRENT_FORMAT_VERSION) == CURRENT_FORMAT_VERSION

    @pytest.mark.parametrize("version", ["0.9", "2.0", "1.1", "", "latest"])
    def test_unknown_versions_rejected(self, version: str) -> None:
        """An unreadable version fails loudly rather than being guessed at."""
        with pytest.raises(CheckpointVersionError, match="unsupported checkpoint format"):
            validate_format_version(version)


class TestDigest:
    """Checksum computation."""

    def test_digest_is_stable_and_content_sensitive(self, tmp_path: Path) -> None:
        """The digest depends on the bytes and nothing else."""
        first = tmp_path / "a.bin"
        second = tmp_path / "b.bin"
        first.write_bytes(b"hello world")
        second.write_bytes(b"hello world")
        assert file_digest(first) == file_digest(second)
        second.write_bytes(b"hello worlD")
        assert file_digest(first) != file_digest(second)

    def test_chunked_reading_matches_whole_file(self, tmp_path: Path) -> None:
        """Streaming in small chunks gives the same digest as one read."""
        path = tmp_path / "big.bin"
        path.write_bytes(bytes(range(256)) * 100)
        assert file_digest(path, chunk_size=7) == file_digest(path, chunk_size=1 << 20)

    def test_missing_file_reports_the_path(self, tmp_path: Path) -> None:
        """A missing file produces a readable error, not an OSError."""
        with pytest.raises(CheckpointError, match="could not read a checkpoint file"):
            file_digest(tmp_path / "absent.bin")


def build_manifest(shards: list[tuple[int, int, int]], numel: int = 12) -> CheckpointManifest:
    """Assemble a one-tensor manifest from ``(rank, offset, length)`` triples."""
    manifest = CheckpointManifest(complete=True, writer_world_size=len(shards))
    for rank, offset, length in shards:
        name = shard_filename(rank)
        manifest.files.setdefault(name, FileRecord(name=name, sha256="0" * 64, bytes=1))
        manifest.add_shard(
            TensorRecord(
                key="w",
                name="w",
                global_shape=(3, 4),
                dtype="torch.float32",
            ),
            ShardRecord(
                rank=rank, file=name, offset=offset, length=length, key=f"model::w@{offset}"
            ),
        )
    return manifest


class TestManifestValidation:
    """Completeness and consistency checks."""

    def test_complete_manifest_validates(self) -> None:
        """Four shards tiling a 12-element tensor is a valid checkpoint."""
        build_manifest([(0, 0, 3), (1, 3, 3), (2, 6, 3), (3, 9, 3)]).validate()

    def test_gap_is_detected(self) -> None:
        """A missing shard leaves the tensor uncovered and is rejected."""
        manifest = build_manifest([(0, 0, 3), (2, 6, 3), (3, 9, 3)])
        with pytest.raises(IncompleteCheckpointError, match="do not cover the whole tensor"):
            manifest.validate()

    def test_overlap_is_detected(self) -> None:
        """Two shards claiming the same elements is ambiguous and rejected."""
        manifest = build_manifest([(0, 0, 6), (1, 3, 3), (2, 6, 6)])
        with pytest.raises(CheckpointCorruptionError, match="overlap"):
            manifest.validate()

    def test_shard_past_the_end_is_detected(self) -> None:
        """A shard running past the global shape is corruption."""
        manifest = build_manifest([(0, 0, 12), (1, 12, 4)])
        with pytest.raises(CheckpointCorruptionError, match="runs past the end"):
            manifest.validate()

    def test_incomplete_flag_is_honoured(self) -> None:
        """A manifest not marked complete is refused even if it looks fine."""
        manifest = build_manifest([(0, 0, 12)])
        manifest.complete = False
        with pytest.raises(IncompleteCheckpointError, match="not marked complete"):
            manifest.validate()

    def test_shard_referencing_an_unlisted_file_is_rejected(self) -> None:
        """Every referenced file needs an integrity record."""
        manifest = build_manifest([(0, 0, 12)])
        manifest.files.clear()
        with pytest.raises(IncompleteCheckpointError, match="no integrity record"):
            manifest.validate()

    def test_conflicting_shapes_rejected(self) -> None:
        """Two ranks describing one tensor differently cannot be reconciled."""
        manifest = build_manifest([(0, 0, 12)])
        with pytest.raises(CheckpointCorruptionError, match="describe tensor"):
            manifest.add_shard(
                TensorRecord(key="w", name="w", global_shape=(4, 4), dtype="torch.float32"),
                ShardRecord(rank=1, file="rank-00001.pt", offset=0, length=4, key="k"),
            )

    def test_covered_elements_counts_unions_not_sums(self) -> None:
        """Overlapping shards do not make a tensor look complete."""
        record = TensorRecord(key="w", name="w", global_shape=(12,), dtype="torch.float32")
        record.shards = [
            ShardRecord(rank=0, file="rank-00000.pt", offset=0, length=8, key="a"),
            ShardRecord(rank=1, file="rank-00001.pt", offset=4, length=8, key="b"),
        ]
        assert record.covered_elements() == 12  # union
        assert sum(s.length for s in record.shards) == 16  # sum


class TestManifestQueries:
    """Shard lookup, which is the resharding primitive."""

    def test_overlapping_shards_are_ordered_and_exact(self) -> None:
        """A wanted range resolves to exactly the intersecting shards."""
        manifest = build_manifest([(0, 0, 3), (1, 3, 3), (2, 6, 3), (3, 9, 3)])
        record = manifest.tensors["w"]
        overlaps = record.shards_overlapping(ShardRange(start=0, length=4))
        assert [(s.rank, o.as_tuple()) for s, o in overlaps] == [(0, (0, 3)), (1, (3, 1))]

    def test_disjoint_request_returns_nothing(self) -> None:
        """A range beyond the shards yields no sources."""
        manifest = build_manifest([(0, 0, 3)])
        assert manifest.tensors["w"].shards_overlapping(ShardRange(start=8, length=2)) == []

    def test_category_filtering(self) -> None:
        """Model, optimizer and buffer tensors are separable."""
        manifest = build_manifest([(0, 0, 12)])
        manifest.add_shard(
            TensorRecord(
                key="w::exp_avg",
                name="w",
                global_shape=(3, 4),
                dtype="torch.float32",
                category="optimizer",
                state_name="exp_avg",
            ),
            ShardRecord(rank=0, file="rank-00000.pt", offset=0, length=12, key="opt"),
        )
        assert set(manifest.tensors_by_category("model")) == {"w"}
        assert set(manifest.tensors_by_category("optimizer")) == {"w::exp_avg"}


class TestManifestSerialisation:
    """JSON round-tripping."""

    def test_round_trip(self, tmp_path: Path) -> None:
        """A manifest survives a write/read cycle unchanged."""
        original = build_manifest([(0, 0, 6), (1, 6, 6)])
        original.step = 42
        original.topology = {"world_size": 2, "sizes": {"shard": 2}}
        path = original.write(tmp_path / "manifest.json")
        restored = CheckpointManifest.read(path)
        assert restored.as_dict() == original.as_dict()
        restored.validate()

    def test_json_is_deterministic(self) -> None:
        """Key order is stable, so manifests can be diffed."""
        manifest = build_manifest([(1, 6, 6), (0, 0, 6)])
        assert manifest.to_json() == build_manifest([(0, 0, 6), (1, 6, 6)]).to_json()

    def test_missing_manifest_reports_incompleteness(self, tmp_path: Path) -> None:
        """No manifest means the write never finished."""
        with pytest.raises(IncompleteCheckpointError, match="no manifest found"):
            CheckpointManifest.read(tmp_path / "manifest.json")

    def test_corrupt_json_rejected(self, tmp_path: Path) -> None:
        """Invalid JSON is reported as corruption, not a stack trace."""
        path = tmp_path / "manifest.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(CheckpointError, match="not valid JSON"):
            CheckpointManifest.read(path)

    def test_non_object_json_rejected(self, tmp_path: Path) -> None:
        """A JSON array is not a manifest."""
        path = tmp_path / "manifest.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(CheckpointError, match="not a JSON object"):
            CheckpointManifest.read(path)

    def test_malformed_manifest_rejected(self, tmp_path: Path) -> None:
        """A JSON object missing required fields is reported as malformed."""
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps({"tensors": {}}), encoding="utf-8")
        with pytest.raises(CheckpointError, match="manifest is malformed"):
            CheckpointManifest.read(path)

    def test_summary_mentions_the_essentials(self) -> None:
        """The human summary names the step, world size and completeness."""
        manifest = build_manifest([(0, 0, 12)])
        manifest.step = 7
        text = manifest.summary()
        assert "step               : 7" in text
        assert "complete           : True" in text
