"""Integration tests for distributed checkpointing.

Coverage:

* same-topology save/restore, bit-for-bit;
* resharding across FSDP widths and between FSDP and DDP;
* every integrity failure mode -- missing file, truncated file, corrupted
  bytes, incomplete manifest, unknown version, unsupported topology change;
* optimizer state and RNG state restoration.

Tampering tests deliberately damage a real checkpoint on disk rather than
constructing a synthetic manifest, so they exercise the same code path an
operator with a failing disk would hit.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from hybrid_training.checkpoint import (
    CheckpointManifest,
    inspect_checkpoint,
    read_metadata,
)
from hybrid_training.checkpoint.format import MANIFEST_FILENAME, shard_filename
from hybrid_training.checkpoint.reader import load_checkpoint
from hybrid_training.checkpoint.reshard import describe_reshard, verify_files
from hybrid_training.checkpoint.writer import (
    find_latest_checkpoint,
    prune_checkpoints,
    save_checkpoint,
)
from hybrid_training.config import (
    CheckpointConfig,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    TopologyConfig,
    TrainingConfig,
)
from hybrid_training.distributed.context import distributed_context
from hybrid_training.errors import (
    CheckpointCorruptionError,
    CheckpointVersionError,
    IncompleteCheckpointError,
)
from hybrid_training.optim.sharded_optimizer import ShardedOptimizer
from hybrid_training.parallel.hybrid import build_parallel_model
from hybrid_training.training.state import TrainingState
from hybrid_training.utils.tensors import ShardRange

from ..conftest import expect_distributed_failure, run_distributed

pytestmark = pytest.mark.distributed

MODEL = ModelConfig(input_size=10, hidden_size=17, num_layers=2, output_size=5)
GLOBAL_BATCH = 8


def _config(world_size: int, kind: str, directory: str, steps: int = 6) -> ExperimentConfig:
    """Build a configuration whose *global* batch is independent of world size."""
    if kind == "fsdp":
        topology = TopologyConfig(shard_parallel_size=world_size)
    elif kind == "ddp":
        topology = TopologyConfig(data_parallel_size=world_size)
    else:
        topology = TopologyConfig(data_parallel_size=world_size // 2, shard_parallel_size=2)
    return ExperimentConfig(
        backend="gloo",
        device="cpu",
        topology=topology,
        model=MODEL,
        data=DataConfig(
            micro_batch_size=GLOBAL_BATCH // world_size,
            num_train_samples=128,
            num_eval_samples=0,
            seed=7,
        ),
        optimizer=OptimizerConfig(name="adamw", learning_rate=5e-3),
        training=TrainingConfig(max_steps=steps, seed=3, max_grad_norm=1.0, log_every_steps=0),
        checkpoint=CheckpointConfig(directory=directory),
    )


def _flatten(state: dict[str, torch.Tensor]) -> torch.Tensor:
    """Concatenate a state dict into one vector for comparison."""
    return torch.cat([state[name].reshape(-1).double() for name in sorted(state)])


# --------------------------------------------------------------------------
# workers
# --------------------------------------------------------------------------
def worker_train_and_save(
    rank: int, world_size: int, kind: str = "fsdp", directory: str = "", save_at: int = 3
) -> dict:
    """Train, checkpoint mid-run, and report the state at the save point."""
    from hybrid_training.training.engine import TrainingEngine

    config = _config(world_size, kind, directory)
    with distributed_context(config) as context:
        engine = TrainingEngine(config, context)
        batches = engine._batch_stream()
        saved = ""
        state_at_save: dict[str, list[float]] = {}
        losses = []
        for _ in range(config.training.max_steps):
            losses.append(engine.train_step(batches).loss)
            if engine.state.step == save_at:
                saved = str(engine.save_checkpoint())
                state_at_save = {
                    name: value.reshape(-1).tolist()
                    for name, value in engine.model.full_state_dict().items()
                }
        final = engine.model.full_state_dict()
        engine.close()
        return {
            "checkpoint": saved,
            "losses": losses,
            "state_at_save": state_at_save,
            "final_checksum": _flatten(final).sum().item(),
            "final_norm": _flatten(final).norm().item(),
        }


def worker_load_and_finish(
    rank: int,
    world_size: int,
    kind: str = "fsdp",
    directory: str = "",
    checkpoint: str = "",
    steps: int = 6,
) -> dict:
    """Resume from a checkpoint and train to the same finishing line."""
    from hybrid_training.training.engine import TrainingEngine

    config = _config(world_size, kind, directory, steps=steps)
    with distributed_context(config) as context:
        engine = TrainingEngine(config, context)
        loaded = engine.load_checkpoint(checkpoint)
        restored_tensors = engine.model.full_state_dict()
        restored = {name: value.reshape(-1).tolist() for name, value in restored_tensors.items()}
        optimizer_state_numel = sum(
            v.numel()
            for state in engine.optimizer.inner.state.values()
            for v in state.values()
            if torch.is_tensor(v) and v.numel() > 1
        )
        # Under FSDP the tensors the optimizer steps on *are* the flat shards,
        # so their combined size is the local shard size the moments must match.
        local_shard_numel = sum(
            p.numel() for group in engine.optimizer.inner.param_groups for p in group["params"]
        )
        batches = engine._batch_stream()
        losses = []
        while engine.state.step < steps:
            losses.append(engine.train_step(batches).loss)
        final = engine.model.full_state_dict()
        engine.close()
        return {
            "resumed_at": loaded.step,
            "files_read": len(loaded.files_read),
            "file_names": sorted(loaded.files_read),
            "rng_restored": loaded.rng_restored,
            "restored_state": restored,
            "losses": losses,
            "optimizer_state_numel": optimizer_state_numel,
            # Reported rather than hard-coded so the assertions stay true if the
            # test model ever changes shape.
            "local_shard_numel": local_shard_numel,
            "global_param_numel": sum(v.numel() for v in restored_tensors.values()),
            "final_checksum": _flatten(final).sum().item(),
            "final_norm": _flatten(final).norm().item(),
        }


def worker_save_simple(rank: int, world_size: int, directory: str = "") -> dict:
    """Save a checkpoint without training, for the tampering tests."""
    config = _config(world_size, "fsdp", directory)
    with distributed_context(config) as context:
        model = build_parallel_model(config, context)
        optimizer = ShardedOptimizer(
            model.optimizer_parameters(),
            config.optimizer,
            norm_group=model.norm_group,
            device=context.device,
        )
        loss = nn.functional.mse_loss(
            model(torch.randn(2, MODEL.input_size)), torch.randn(2, MODEL.output_size)
        )
        loss.backward()
        model.finish_backward()
        optimizer.step()
        state = TrainingState(step=10)
        result = save_checkpoint(
            directory,
            model=model,
            context=context,
            state=state,
            optimizer=optimizer,
            config=config,
        )
        return {"path": str(result.path), "bytes": result.bytes_written}


def worker_load_expecting_failure(
    rank: int, world_size: int, directory: str = "", checkpoint: str = ""
) -> str:
    """Attempt a load that is expected to fail, and name the failure."""
    config = _config(world_size, "fsdp", directory)
    with distributed_context(config) as context:
        model = build_parallel_model(config, context)
        load_checkpoint(checkpoint, model=model, context=context)
        return "should not reach here"


def worker_load_with_tensor_parallel(
    rank: int, world_size: int, directory: str = "", checkpoint: str = ""
) -> str:
    """Loading an FSDP checkpoint into a tensor-parallel topology must fail."""
    config = ExperimentConfig(
        backend="gloo",
        device="cpu",
        topology=TopologyConfig(tensor_parallel_size=world_size),
        model=ModelConfig(kind="transformer", vocab_size=32, hidden_size=16, num_heads=4),
        data=DataConfig(micro_batch_size=2, num_train_samples=32, num_eval_samples=0),
        checkpoint=CheckpointConfig(directory=directory),
    )
    with distributed_context(config) as context:
        model = build_parallel_model(config, context)
        load_checkpoint(checkpoint, model=model, context=context)
        return "should not reach here"


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
class TestSameTopologyResume:
    """Save and resume at the same world size."""

    @pytest.mark.parametrize(("kind", "world_size"), [("ddp", 2), ("fsdp", 2), ("fsdp", 4)])
    def test_resume_reproduces_the_uninterrupted_run(
        self, kind: str, world_size: int, temporary_directory: Path
    ) -> None:
        """An interrupted run finishes exactly where an uninterrupted one does."""
        directory = str(temporary_directory / f"{kind}{world_size}")
        baseline = run_distributed(
            worker_train_and_save,
            world_size,
            kwargs={"kind": kind, "directory": directory},
        )
        resumed = run_distributed(
            worker_load_and_finish,
            world_size,
            kwargs={
                "kind": kind,
                "directory": directory,
                "checkpoint": baseline[0]["checkpoint"],
            },
        )
        # Identical arithmetic in identical order: bitwise equality.
        assert resumed[0]["final_checksum"] == baseline[0]["final_checksum"]
        assert resumed[0]["losses"] == baseline[0]["losses"][3:]

    @pytest.mark.parametrize("kind", ["ddp", "fsdp"])
    def test_restored_parameters_equal_the_saved_ones(
        self, kind: str, temporary_directory: Path
    ) -> None:
        """The load itself is exact, before any further training."""
        directory = str(temporary_directory / kind)
        baseline = run_distributed(
            worker_train_and_save, 2, kwargs={"kind": kind, "directory": directory}
        )
        resumed = run_distributed(
            worker_load_and_finish,
            2,
            kwargs={
                "kind": kind,
                "directory": directory,
                "checkpoint": baseline[0]["checkpoint"],
            },
        )
        saved = baseline[0]["state_at_save"]
        restored = resumed[0]["restored_state"]
        assert set(saved) == set(restored)
        for name in saved:
            assert restored[name] == saved[name], name

    def test_rng_state_is_restored(self, temporary_directory: Path) -> None:
        """Per-rank RNG state comes back, so dropout and data order continue."""
        directory = str(temporary_directory / "rng")
        baseline = run_distributed(worker_train_and_save, 2, kwargs={"directory": directory})
        resumed = run_distributed(
            worker_load_and_finish,
            2,
            kwargs={"directory": directory, "checkpoint": baseline[0]["checkpoint"]},
        )
        assert all(r["rng_restored"] for r in resumed)

    def test_optimizer_state_is_restored_sharded(self, temporary_directory: Path) -> None:
        """Adam moments come back at the *local* size, not the global one."""
        directory = str(temporary_directory / "opt")
        baseline = run_distributed(worker_train_and_save, 2, kwargs={"directory": directory})
        resumed = run_distributed(
            worker_load_and_finish,
            2,
            kwargs={"directory": directory, "checkpoint": baseline[0]["checkpoint"]},
        )
        # Adam keeps two moments (exp_avg, exp_avg_sq) per optimizer parameter,
        # and under FSDP the optimizer parameters are the *flat shards*.  So the
        # restored moment storage is twice the local shard, not twice the model.
        result = resumed[0]
        assert result["optimizer_state_numel"] == 2 * result["local_shard_numel"]
        # And the local shard really is a shard: half the model, give or take the
        # padding that rounds the flat vector up to a multiple of the shard count.
        half = result["global_param_numel"] / 2
        assert half <= result["local_shard_numel"] < half + 2
        # The point of the test: this is *not* the global figure.
        assert result["optimizer_state_numel"] < 2 * result["global_param_numel"]


class TestResharding:
    """Changing the world size between save and load."""

    @pytest.mark.parametrize(("save_ranks", "load_ranks"), [(4, 2), (2, 4)])
    def test_fsdp_width_change(
        self, save_ranks: int, load_ranks: int, temporary_directory: Path
    ) -> None:
        """A checkpoint written at one FSDP width loads at another, exactly."""
        directory = str(temporary_directory / f"reshard{save_ranks}to{load_ranks}")
        baseline = run_distributed(
            worker_train_and_save, save_ranks, kwargs={"directory": directory}
        )
        resumed = run_distributed(
            worker_load_and_finish,
            load_ranks,
            kwargs={"directory": directory, "checkpoint": baseline[0]["checkpoint"]},
        )
        saved = baseline[0]["state_at_save"]
        restored = resumed[0]["restored_state"]
        for name in saved:
            # A pure redistribution of bytes: bitwise equality is required.
            assert restored[name] == saved[name], name

    def test_reader_touches_only_the_files_it_needs(self, temporary_directory: Path) -> None:
        """Halving the rank count means each reader reads two writers' files."""
        directory = str(temporary_directory / "files")
        baseline = run_distributed(worker_train_and_save, 4, kwargs={"directory": directory})
        resumed = run_distributed(
            worker_load_and_finish,
            2,
            kwargs={"directory": directory, "checkpoint": baseline[0]["checkpoint"]},
        )
        # Each reader needs the writer files overlapping its half of the flat
        # parameter vector: at 4 -> 2 that is two files, which is the whole point
        # of indexing the manifest by interval instead of by rank.
        #
        # A reader opens one more file when its *own* rank's payload is not
        # already among them, because per-rank RNG state lives in the file that
        # rank wrote.  Reader 0's interval covers rank-00000.pt already; reader
        # 1's covers ranks 2 and 3, so it opens rank-00001.pt as well.
        for rank, result in enumerate(resumed):
            opened = set(result["file_names"])
            # Its own file, for the RNG stream...
            assert shard_filename(rank) in opened, opened
            # ...and strictly fewer than every file the four writers produced.
            assert len(opened) < 4, opened
        assert [r["files_read"] for r in resumed] == [2, 3]

    def test_sharded_to_replicated(self, temporary_directory: Path) -> None:
        """An FSDP checkpoint loads into a DDP job and vice versa."""
        directory = str(temporary_directory / "fsdp-to-ddp")
        baseline = run_distributed(
            worker_train_and_save, 2, kwargs={"kind": "fsdp", "directory": directory}
        )
        resumed = run_distributed(
            worker_load_and_finish,
            2,
            kwargs={
                "kind": "ddp",
                "directory": directory,
                "checkpoint": baseline[0]["checkpoint"],
            },
        )
        saved = baseline[0]["state_at_save"]
        for name, values in resumed[0]["restored_state"].items():
            assert values == saved[name], name

    def test_hybrid_to_pure_sharding(self, temporary_directory: Path) -> None:
        """Changing the replication degree is supported."""
        directory = str(temporary_directory / "hybrid-to-fsdp")
        baseline = run_distributed(
            worker_train_and_save, 4, kwargs={"kind": "hybrid", "directory": directory}
        )
        resumed = run_distributed(
            worker_load_and_finish,
            4,
            kwargs={
                "kind": "fsdp",
                "directory": directory,
                "checkpoint": baseline[0]["checkpoint"],
            },
        )
        saved = baseline[0]["state_at_save"]
        for name, values in resumed[0]["restored_state"].items():
            assert values == saved[name], name

    def test_tensor_parallel_change_is_rejected(self, temporary_directory: Path) -> None:
        """A tensor-parallel width change is refused with an explanation."""
        directory = str(temporary_directory / "tp-change")
        baseline = run_distributed(worker_train_and_save, 2, kwargs={"directory": directory})
        results = expect_distributed_failure(
            worker_load_with_tensor_parallel,
            2,
            kwargs={"directory": directory, "checkpoint": baseline[0]["checkpoint"]},
        )
        assert all(not r.succeeded for r in results)
        assert any("tensor-parallel width differs" in (r.traceback_text or "") for r in results)


class TestIntegrity:
    """Every way a checkpoint can be damaged."""

    @pytest.fixture()
    def saved_checkpoint(self, temporary_directory: Path) -> Path:
        """A real, valid checkpoint on disk."""
        directory = str(temporary_directory / "integrity")
        results = run_distributed(worker_save_simple, 2, kwargs={"directory": directory})
        return Path(results[0]["path"])

    def test_valid_checkpoint_passes_inspection(self, saved_checkpoint: Path) -> None:
        """The undamaged article validates and reports its contents."""
        summary = inspect_checkpoint(saved_checkpoint, verify=True)
        assert summary["complete"] is True
        assert summary["step"] == 10
        assert summary["writer_world_size"] == 2
        assert summary["tensor_counts"]["model"] > 0
        assert summary["tensor_counts"]["optimizer"] > 0
        assert summary["verification"]["files_verified"] == 2

    def test_manifest_is_written_last(self, saved_checkpoint: Path) -> None:
        """Every payload file exists alongside the manifest."""
        manifest = CheckpointManifest.read(saved_checkpoint / MANIFEST_FILENAME)
        for name in manifest.referenced_files():
            assert (saved_checkpoint / name).is_file()

    def test_no_staging_directory_survives(self, saved_checkpoint: Path) -> None:
        """The staging directory is renamed away, not left behind."""
        parent = saved_checkpoint.parent
        assert not [child for child in parent.iterdir() if child.name.startswith(".staging-")]

    def test_missing_shard_file_detected(self, saved_checkpoint: Path) -> None:
        """A deleted payload file is reported as an incomplete checkpoint."""
        (saved_checkpoint / shard_filename(1)).unlink()
        with pytest.raises(IncompleteCheckpointError, match="missing"):
            inspect_checkpoint(saved_checkpoint, verify=True)

    def test_truncated_file_detected(self, saved_checkpoint: Path) -> None:
        """A short file is reported as truncated, not as a checksum failure."""
        target = saved_checkpoint / shard_filename(0)
        data = target.read_bytes()
        target.write_bytes(data[: len(data) // 2])
        with pytest.raises(CheckpointCorruptionError, match="wrong size"):
            inspect_checkpoint(saved_checkpoint, verify=True)

    def test_corrupted_bytes_detected(self, saved_checkpoint: Path) -> None:
        """Same-length corruption is caught by the checksum."""
        target = saved_checkpoint / shard_filename(0)
        data = bytearray(target.read_bytes())
        data[len(data) // 2] ^= 0xFF
        target.write_bytes(bytes(data))
        with pytest.raises(CheckpointCorruptionError, match="failed checksum verification"):
            inspect_checkpoint(saved_checkpoint, verify=True)

    def test_missing_manifest_detected(self, saved_checkpoint: Path) -> None:
        """No manifest means the write never completed."""
        (saved_checkpoint / MANIFEST_FILENAME).unlink()
        with pytest.raises(IncompleteCheckpointError, match="no manifest found"):
            inspect_checkpoint(saved_checkpoint)

    def test_incomplete_manifest_detected(self, saved_checkpoint: Path) -> None:
        """A manifest without the complete flag is refused."""
        path = saved_checkpoint / MANIFEST_FILENAME
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["complete"] = False
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(IncompleteCheckpointError, match="not marked complete"):
            inspect_checkpoint(saved_checkpoint)

    def test_version_mismatch_detected(self, saved_checkpoint: Path) -> None:
        """An unreadable format version fails loudly."""
        path = saved_checkpoint / MANIFEST_FILENAME
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["format_version"] = "99.0"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(CheckpointVersionError, match="unsupported checkpoint format"):
            inspect_checkpoint(saved_checkpoint)

    def test_dropped_shard_record_detected(self, saved_checkpoint: Path) -> None:
        """Removing a shard from the manifest leaves a tensor uncovered."""
        path = saved_checkpoint / MANIFEST_FILENAME
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = next(k for k, v in payload["tensors"].items() if len(v["shards"]) > 1)
        payload["tensors"][key]["shards"].pop()
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(IncompleteCheckpointError, match="do not cover the whole tensor"):
            inspect_checkpoint(saved_checkpoint)

    def test_hostile_filename_in_manifest_rejected(self, saved_checkpoint: Path) -> None:
        """A manifest cannot point the reader at an arbitrary path."""
        path = saved_checkpoint / MANIFEST_FILENAME
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = next(iter(payload["tensors"]))
        payload["tensors"][key]["shards"][0]["file"] = "../../../etc/passwd"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(Exception, match="does not match the shard naming rule"):
            inspect_checkpoint(saved_checkpoint)

    def test_corruption_is_detected_during_a_real_load(
        self, saved_checkpoint: Path, temporary_directory: Path
    ) -> None:
        """The load path verifies too, not only the inspector."""
        target = saved_checkpoint / shard_filename(0)
        data = bytearray(target.read_bytes())
        data[-8] ^= 0xFF
        target.write_bytes(bytes(data))
        results = expect_distributed_failure(
            worker_load_expecting_failure,
            2,
            kwargs={
                "directory": str(temporary_directory / "integrity"),
                "checkpoint": str(saved_checkpoint),
            },
        )
        assert any("checksum" in (r.traceback_text or "") for r in results)

    def test_verify_files_returns_digests(self, saved_checkpoint: Path) -> None:
        """Explicit verification returns the digests it computed."""
        manifest = CheckpointManifest.read(saved_checkpoint / MANIFEST_FILENAME)
        digests = verify_files(saved_checkpoint, manifest)
        assert set(digests) == manifest.referenced_files()
        assert all(len(d) == 64 for d in digests.values())


class TestMetadataAndInspection:
    """The JSON side of the format."""

    @pytest.fixture()
    def saved_checkpoint(self, temporary_directory: Path) -> Path:
        """A real checkpoint produced by the training engine."""
        directory = str(temporary_directory / "metadata")
        results = run_distributed(worker_train_and_save, 2, kwargs={"directory": directory})
        return Path(results[0]["checkpoint"])

    def test_metadata_is_pure_json(self, saved_checkpoint: Path) -> None:
        """Metadata parses without ``torch`` and holds the training state."""
        metadata = read_metadata(saved_checkpoint)
        json.dumps(metadata, allow_nan=False)
        assert metadata["training_state"]["step"] == 3
        assert metadata["world_size"] == 2
        assert metadata["config"]["model"]["hidden_size"] == MODEL.hidden_size
        assert metadata["scheduler"]["name"] in {"constant", "linear", "cosine"}

    def test_manifest_describes_global_tensors(self, saved_checkpoint: Path) -> None:
        """Every model tensor's shards tile its global element count."""
        manifest = CheckpointManifest.read(saved_checkpoint / MANIFEST_FILENAME)
        for record in manifest.tensors_by_category("model").values():
            assert record.covered_elements() == record.numel
            assert sum(s.length for s in record.shards) == record.numel

    def test_reshard_plan_is_inspectable(self, saved_checkpoint: Path) -> None:
        """The read plan can be examined without reading any tensor."""
        manifest = CheckpointManifest.read(saved_checkpoint / MANIFEST_FILENAME)
        key = sorted(manifest.tensors_by_category("model"))[0]
        record = manifest.tensors[key]
        plan = describe_reshard(manifest, key, ShardRange(start=0, length=record.numel))
        assert plan["key"] == key
        assert plan["sources"]
        assert sum(s["overlap"][1] for s in plan["sources"]) == record.numel

    def test_optimizer_tensors_share_the_parameter_coordinates(
        self, saved_checkpoint: Path
    ) -> None:
        """Optimizer state is stored in the same global space as parameters."""
        manifest = CheckpointManifest.read(saved_checkpoint / MANIFEST_FILENAME)
        model = manifest.tensors_by_category("model")
        for key, record in manifest.tensors_by_category("optimizer").items():
            base = key.rsplit("::", 1)[0]
            assert base in model
            assert record.global_shape == model[base].global_shape


class TestDirectoryManagement:
    """Discovery and retention."""

    def test_find_latest_selects_the_highest_step(self, tmp_path: Path) -> None:
        """The newest *complete* checkpoint wins."""
        for step in (5, 20, 12):
            directory = tmp_path / f"checkpoint-step-{step:06d}"
            directory.mkdir()
            (directory / MANIFEST_FILENAME).write_text("{}", encoding="utf-8")
        assert find_latest_checkpoint(tmp_path).name == "checkpoint-step-000020"

    def test_find_latest_ignores_incomplete_directories(self, tmp_path: Path) -> None:
        """A directory without a manifest is never selected."""
        complete = tmp_path / "checkpoint-step-000005"
        complete.mkdir()
        (complete / MANIFEST_FILENAME).write_text("{}", encoding="utf-8")
        (tmp_path / "checkpoint-step-000099").mkdir()
        assert find_latest_checkpoint(tmp_path).name == "checkpoint-step-000005"

    def test_find_latest_on_an_empty_directory(self, tmp_path: Path) -> None:
        """No checkpoints means ``None``, not an exception."""
        assert find_latest_checkpoint(tmp_path) is None
        assert find_latest_checkpoint(tmp_path / "absent") is None

    def test_pruning_keeps_the_newest(self, tmp_path: Path) -> None:
        """Retention removes the oldest and keeps the requested count."""
        for step in (1, 2, 3, 4, 5):
            (tmp_path / f"checkpoint-step-{step:06d}").mkdir()
        removed = prune_checkpoints(tmp_path, keep_last=2)
        assert len(removed) == 3
        remaining = sorted(p.name for p in tmp_path.iterdir())
        assert remaining == ["checkpoint-step-000004", "checkpoint-step-000005"]

    def test_pruning_disabled_by_default(self, tmp_path: Path) -> None:
        """``keep_last=0`` retains everything."""
        (tmp_path / "checkpoint-step-000001").mkdir()
        assert prune_checkpoints(tmp_path, keep_last=0) == []

    def test_pruning_ignores_unrelated_directories(self, tmp_path: Path) -> None:
        """Only checkpoint-shaped names are candidates for deletion."""
        (tmp_path / "checkpoint-step-000001").mkdir()
        (tmp_path / "checkpoint-step-000002").mkdir()
        (tmp_path / "important-data").mkdir()
        prune_checkpoints(tmp_path, keep_last=1)
        assert (tmp_path / "important-data").is_dir()


def worker_duplicate_save_rejected(rank: int, world_size: int, directory: str = "") -> str:
    """Saving twice at the same step must not silently overwrite."""
    from hybrid_training.errors import CheckpointError

    config = _config(world_size, "fsdp", directory)
    with distributed_context(config) as context:
        model = build_parallel_model(config, context)
        state = TrainingState(step=1)
        save_checkpoint(directory, model=model, context=context, state=state, config=config)
        try:
            save_checkpoint(directory, model=model, context=context, state=state, config=config)
        except CheckpointError as error:
            return f"rejected: {'already exists' in str(error)}"
        return "not rejected"


def test_duplicate_save_rejected(temporary_directory: Path) -> None:
    """Overwriting an existing checkpoint requires deleting it first."""
    directory = str(temporary_directory / "duplicate")
    results = run_distributed(worker_duplicate_save_rejected, 2, kwargs={"directory": directory})
    # *Every* rank must raise, not just the one that performs the check.  Rank 0
    # decides and broadcasts the verdict; if it raised alone, the other ranks
    # would block in the next barrier until rank 0's process died, turning a
    # clean error into a hang.  Asserting on all ranks is what pins that down.
    assert results == ["rejected: True", "rejected: True"]


def test_checkpoint_survives_directory_move(temporary_directory: Path) -> None:
    """A checkpoint is self-contained: moving the directory does not break it."""
    directory = str(temporary_directory / "movable")
    results = run_distributed(worker_save_simple, 2, kwargs={"directory": directory})
    original = Path(results[0]["path"])
    moved = temporary_directory / "relocated"
    shutil.move(str(original), str(moved))
    summary = inspect_checkpoint(moved, verify=True)
    assert summary["complete"] is True
