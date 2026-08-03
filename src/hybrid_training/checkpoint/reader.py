"""Loading distributed checkpoints, including across different world sizes.

What can and cannot change between save and load
================================================
================================================  ==========================
Change                                            Supported?
================================================  ==========================
``shard_parallel_size`` (FSDP width)              **yes** -- shards are
                                                  described by global flat
                                                  offsets, so any width can
                                                  read any other
``data_parallel_size`` (replication degree)       **yes** -- replicas hold
                                                  identical data, so a new
                                                  replica simply reads the
                                                  same bytes
sharding <-> replication (FSDP <-> DDP)           **yes** -- both are
                                                  expressed as ranges of the
                                                  same global tensors
``tensor_parallel_size``                          **no** -- rejected
                                                  explicitly; see below
model architecture                                **no** -- shape mismatches
                                                  are reported per tensor
================================================  ==========================

Why tensor-parallel width cannot change
=======================================
Sharding for *memory* and partitioning for *computation* are different things.
An FSDP shard is an arbitrary contiguous range of a tensor's elements, chosen
purely to divide bytes evenly; nothing in the model cares where the cut falls.
A tensor-parallel slice is a *mathematical* decomposition: rank ``t`` owns
output features ``[tO/T, (t+1)O/T)`` of a weight matrix and computes with them.
Worse, a row-parallel slice is a set of strided columns, which no single
``(offset, length)`` interval can describe.

Converting between tensor-parallel widths therefore means re-deriving each
slice from the full matrix -- a genuine tensor transformation, not a
redistribution of bytes.  This reader refuses rather than guessing.  The
supported route is to load with the original width, write a full state dict,
and re-shard offline.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ..config import ExperimentConfig
from ..distributed.context import DistributedContext
from ..errors import (
    CheckpointError,
    CheckpointTopologyError,
    IncompleteCheckpointError,
    format_error,
)
from ..logging import get_logger
from ..optim.sharded_optimizer import ShardedOptimizer
from ..parallel.hybrid import HybridModel
from ..training.state import TrainingState
from ..utils.reproducibility import restore_rng_state, rng_state_from_serialisable
from ..utils.tensors import ShardRange
from .format import (
    MANIFEST_FILENAME,
    METADATA_FILENAME,
    resolve_inside,
    shard_filename,
    validate_format_version,
)
from .manifest import CheckpointManifest
from .reshard import ShardFileCache, read_tensor_range

__all__ = ["LoadedCheckpoint", "inspect_checkpoint", "load_checkpoint", "read_metadata"]

_LOGGER = get_logger(__name__)

_RNG_PREFIX = "rng::"


@dataclass
class LoadedCheckpoint:
    """What a resume recovered.

    Attributes:
        path: Directory that was read.
        manifest: The validated manifest.
        metadata: The parsed ``metadata.json``.
        state: Restored training progress.
        scheduler_state: Restored learning-rate schedule state.
        scaler_state: Restored gradient-scaler state.
        seconds: Wall-clock duration of the load on this rank.
        tensors_loaded: Number of model tensors restored.
        files_read: Payload files this rank opened.
        rng_restored: Whether this rank's RNG state was restored.
    """

    path: Path
    manifest: CheckpointManifest
    metadata: dict[str, Any]
    state: TrainingState
    scheduler_state: dict[str, Any] = field(default_factory=dict)
    scaler_state: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    tensors_loaded: int = 0
    files_read: tuple[str, ...] = ()
    rng_restored: bool = False

    @property
    def step(self) -> int:
        """Step the checkpoint was taken at."""
        return self.state.step

    def __repr__(self) -> str:
        return (
            f"LoadedCheckpoint(path={self.path.name!r}, step={self.step}, "
            f"tensors={self.tensors_loaded}, files={len(self.files_read)}, "
            f"{self.seconds:.3f}s)"
        )


def read_metadata(directory: str | Path) -> dict[str, Any]:
    """Read ``metadata.json`` without touching any tensor payload.

    Args:
        directory: Checkpoint directory.

    Returns:
        The parsed metadata.

    Raises:
        IncompleteCheckpointError: If the file is missing.
        CheckpointError: If it is not valid JSON.
    """
    path = Path(directory) / METADATA_FILENAME
    if not path.is_file():
        raise IncompleteCheckpointError(
            format_error(
                "checkpoint.read_metadata",
                "metadata.json is missing",
                expected=str(path),
                observed="absent",
                resolution="the checkpoint is incomplete",
            )
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CheckpointError(
            format_error(
                "checkpoint.read_metadata",
                "metadata.json is not valid JSON",
                expected="a JSON object",
                observed=str(exc),
                resolution="the file is corrupt",
            )
        ) from exc
    if not isinstance(payload, dict):
        raise CheckpointError(
            format_error(
                "checkpoint.read_metadata",
                "metadata.json is not a JSON object",
                expected="object",
                observed=type(payload).__name__,
                resolution="the file is corrupt",
            )
        )
    return payload


def _validate_topology_transition(
    manifest: CheckpointManifest, context: DistributedContext
) -> None:
    """Reject topology changes the format cannot express.

    Raises:
        CheckpointTopologyError: If the tensor-parallel width differs.
    """
    saved_sizes = dict(manifest.topology.get("sizes", {}))
    saved_tensor = int(saved_sizes.get("tensor", 1))
    current_tensor = context.topology.size("tensor")
    if saved_tensor != current_tensor:
        raise CheckpointTopologyError(
            format_error(
                "checkpoint.load",
                "the tensor-parallel width differs between the checkpoint and this run. "
                "Tensor-parallel slices are a mathematical decomposition of each weight "
                "matrix, not an arbitrary byte range, and a row-parallel slice is a set "
                "of strided columns that no single offset/length can describe",
                rank=context.rank,
                world_size=context.world_size,
                expected=f"tensor_parallel_size == {saved_tensor}",
                observed=current_tensor,
                resolution=(
                    "resume with the original tensor-parallel size; to change it, load "
                    "at the original width, export a full state dict, and rebuild"
                ),
            )
        )

    saved_sequence = int(saved_sizes.get("sequence", 1))
    current_sequence = context.topology.size("sequence")
    if saved_sequence != current_sequence:
        raise CheckpointTopologyError(
            format_error(
                "checkpoint.load",
                "the standalone sequence-parallel width differs; parameters are "
                "replicated across that dimension, but the number of ranks that must "
                "receive each parameter changes what the writer recorded",
                rank=context.rank,
                world_size=context.world_size,
                expected=f"sequence_parallel_size == {saved_sequence}",
                observed=current_sequence,
                resolution="resume with the original sequence-parallel size",
            )
        )


def load_checkpoint(
    directory: str | Path,
    *,
    model: HybridModel,
    context: DistributedContext,
    optimizer: ShardedOptimizer | None = None,
    config: ExperimentConfig | None = None,
    verify_checksums: bool = True,
    load_optimizer: bool = True,
    load_rng: bool = True,
    strict: bool = True,
) -> LoadedCheckpoint:
    """Restore model, optimizer, RNG and progress from a checkpoint.

    Args:
        directory: Checkpoint directory.
        model: The wrapped model to load into.
        context: Active distributed context.
        optimizer: Optimizer to restore state into.
        config: Current configuration, compared against the saved one for a
            warning when they differ.
        verify_checksums: Verify every file this rank reads.
        load_optimizer: Restore optimizer state.
        load_rng: Restore this rank's RNG state.
        strict: Fail when the checkpoint is missing a tensor the current model
            needs.  With ``False`` the missing tensor keeps its current value
            and a warning is logged.

    Returns:
        A :class:`LoadedCheckpoint`.

    Raises:
        IncompleteCheckpointError: If a required tensor or file is missing.
        CheckpointCorruptionError: On checksum or shape mismatches.
        CheckpointTopologyError: On an unsupported topology change.
    """
    started = time.perf_counter()
    path = Path(directory)
    manifest = CheckpointManifest.read(path / MANIFEST_FILENAME)
    validate_format_version(manifest.format_version)
    manifest.validate()
    _validate_topology_transition(manifest, context)

    metadata = read_metadata(path)
    if config is not None:
        _warn_on_config_drift(metadata.get("config"), config)

    cache = ShardFileCache(path, manifest, verify_checksums=verify_checksums)
    parallel_info = model.parameter_parallel_info()

    tensors_loaded = _load_model_tensors(
        model=model,
        manifest=manifest,
        cache=cache,
        parallel_info=parallel_info,
        context=context,
        strict=strict,
    )
    if load_optimizer and optimizer is not None:
        _load_optimizer_state(
            model=model,
            optimizer=optimizer,
            manifest=manifest,
            cache=cache,
            parallel_info=parallel_info,
            metadata=metadata,
            context=context,
            strict=strict,
        )

    rng_restored = False
    if load_rng and metadata.get("rng", {}).get("saved"):
        rng_restored = _restore_rng(path, manifest, metadata, cache, context)

    state = TrainingState.from_dict(metadata.get("training_state", {}))
    result = LoadedCheckpoint(
        path=path,
        manifest=manifest,
        metadata=metadata,
        state=state,
        scheduler_state=dict(metadata.get("scheduler", {})),
        scaler_state=dict(metadata.get("scaler", {})),
        seconds=time.perf_counter() - started,
        tensors_loaded=tensors_loaded,
        files_read=cache.loaded_files,
    )
    result.rng_restored = rng_restored
    cache.clear()
    _LOGGER.info("resumed from %s at step %d", path.name, state.step)
    return result


def _load_model_tensors(
    *,
    model: HybridModel,
    manifest: CheckpointManifest,
    cache: ShardFileCache,
    parallel_info: dict[str, Any],
    context: DistributedContext,
    strict: bool,
) -> int:
    """Load parameters and buffers into this rank's shards.

    Returns:
        The number of tensors restored.
    """
    loaded = 0
    if model.fsdp is not None:
        for flat_param, layout in model.fsdp.optimizer_parameter_layout():
            destination = flat_param.data
            for item in layout:
                info = parallel_info.get(item.name)
                key = info.storage_key if info is not None else item.name
                record = manifest.tensors.get(key)
                if record is None:
                    if strict:
                        raise IncompleteCheckpointError(
                            _missing_tensor_error(key, manifest, context)
                        )
                    _LOGGER.warning("checkpoint has no tensor %r; keeping current value", key)
                    continue
                _check_shape(record, item.global_shape, key, context)
                values = read_tensor_range(
                    record,
                    ShardRange(start=item.parameter_offset, length=item.length),
                    cache,
                    dtype=destination.dtype,
                )
                destination[item.local_offset : item.local_offset + item.length].copy_(values)
                loaded += 1
    else:
        pieces = model.sharded_state_dict()
        full: dict[str, torch.Tensor] = {}
        for name, piece in pieces.items():
            info = parallel_info.get(name)
            key = info.storage_key if info is not None else name
            record = manifest.tensors.get(key)
            if record is None:
                if strict:
                    raise IncompleteCheckpointError(_missing_tensor_error(key, manifest, context))
                _LOGGER.warning("checkpoint has no tensor %r; keeping current value", key)
                continue
            _check_shape(record, piece.global_shape, key, context)
            values = read_tensor_range(
                record,
                ShardRange(start=0, length=record.numel),
                cache,
                dtype=piece.data.dtype,
            )
            full[name] = values.view(piece.global_shape)
            loaded += 1
        model.load_full_state_dict(full)

    # Buffers are small and replicated; load them whole on every rank.
    buffers = model.buffers_state_dict()
    if buffers:
        restored: dict[str, torch.Tensor] = {}
        for name, buffer in buffers.items():
            buffer_record = manifest.tensors.get(name)
            if buffer_record is None:
                continue
            record = buffer_record
            values = read_tensor_range(
                record, ShardRange(start=0, length=record.numel), cache, dtype=buffer.dtype
            )
            restored[name] = values.view(record.global_shape)
            loaded += 1
        if restored:
            model.load_full_state_dict(restored)
    return loaded


def _load_optimizer_state(
    *,
    model: HybridModel,
    optimizer: ShardedOptimizer,
    manifest: CheckpointManifest,
    cache: ShardFileCache,
    parallel_info: dict[str, Any],
    metadata: dict[str, Any],
    context: DistributedContext,
    strict: bool,
) -> None:
    """Rebuild the optimizer's per-parameter state from the checkpoint.

    Optimizer state is stored in the *same* global coordinates as the
    parameters, so this is the same interval arithmetic as the model load and
    works across shard-group widths for free.
    """
    layouts = model.optimizer_parameter_layout()
    scalars = dict(metadata.get("optimizer_scalars", {}))

    state_names: set[str] = set()
    for record in manifest.tensors_by_category("optimizer").values():
        state_names.add(record.state_name)
    state_names.discard("")
    if not state_names and not scalars:
        _LOGGER.warning("checkpoint holds no optimizer state; the optimizer starts fresh")
        return

    rebuilt: dict[int, dict[str, Any]] = {}
    for index, param in enumerate(optimizer.parameters):
        entry: dict[str, Any] = {}
        for state_name in sorted(state_names):
            buffer = torch.zeros(param.numel(), dtype=param.dtype)
            found_any = False
            for item in layouts[index]:
                info = parallel_info.get(item.name)
                base = info.storage_key if info is not None else item.name
                state_record = manifest.tensors.get(f"{base}::{state_name}")
                if state_record is None:
                    continue
                found_any = True
                values = read_tensor_range(
                    state_record,
                    ShardRange(start=item.parameter_offset, length=item.length),
                    cache,
                    dtype=param.dtype,
                )
                buffer[item.local_offset : item.local_offset + item.length] = values
            if found_any:
                entry[state_name] = buffer.view(param.shape)
            elif strict and state_names:
                raise IncompleteCheckpointError(
                    format_error(
                        "checkpoint.load_optimizer",
                        f"optimizer state {state_name!r} is missing for parameter {index}",
                        rank=context.rank,
                        world_size=context.world_size,
                        expected=f"state entries for parameter {index}",
                        observed="none",
                        resolution="the checkpoint was written with save_optimizer=False",
                    )
                )
        for name, value in scalars.items():
            entry[name] = torch.tensor(value) if isinstance(value, int | float) else value
        if entry:
            rebuilt[index] = entry

    optimizer.load_state_dict(
        {
            "param_groups": [
                {k: v for k, v in group.items() if k != "params"}
                for group in optimizer.inner.state_dict()["param_groups"]
            ],
            "state": rebuilt,
            "num_parameters": len(optimizer.parameters),
        }
    )


def _restore_rng(
    path: Path,
    manifest: CheckpointManifest,
    metadata: dict[str, Any],
    cache: ShardFileCache,
    context: DistributedContext,
) -> bool:
    """Restore this rank's RNG state, if the checkpoint holds one for it.

    A rank that has no saved counterpart (because the world grew) keeps its
    current RNG state and a warning is logged.  Silently leaving it unchanged
    would make the run non-reproducible without saying so.

    Returns:
        ``True`` when state was restored.
    """
    saved_ranks = set(metadata.get("rng", {}).get("ranks", []))
    if context.rank not in saved_ranks:
        _LOGGER.warning(
            "the checkpoint holds RNG state for %d rank(s) but this run has %d; rank %d "
            "keeps its freshly seeded RNG, so dropout masks and data order will differ "
            "from the original run",
            len(saved_ranks),
            context.world_size,
            context.rank,
        )
        return False
    filename = shard_filename(context.rank)
    if filename not in manifest.files:
        _LOGGER.warning("no payload file for rank %d; RNG state not restored", context.rank)
        return False
    resolve_inside(path, filename)
    payload = cache.get(filename)
    tensors = {
        key[len(_RNG_PREFIX) :]: value
        for key, value in payload.items()
        if key.startswith(_RNG_PREFIX)
    }
    if not tensors:
        return False
    restore_rng_state(
        rng_state_from_serialisable(
            {"tensors": tensors, "meta": metadata.get("rng", {}).get("meta", {})}
        )
    )
    return True


def _check_shape(
    record: Any, expected_shape: tuple[int, ...], key: str, context: DistributedContext
) -> None:
    """Reject a tensor whose saved shape differs from the model's."""
    if tuple(record.global_shape) != tuple(expected_shape):
        raise CheckpointError(
            format_error(
                "checkpoint.load",
                f"tensor {key!r} has a different shape in the checkpoint",
                rank=context.rank,
                world_size=context.world_size,
                expected=tuple(expected_shape),
                observed=tuple(record.global_shape),
                resolution="resume with the model definition that produced the checkpoint",
            )
        )


def _missing_tensor_error(
    key: str, manifest: CheckpointManifest, context: DistributedContext
) -> str:
    """Build the message for a tensor the checkpoint does not contain."""
    available = sorted(manifest.tensors_by_category("model"))
    return format_error(
        "checkpoint.load",
        f"the checkpoint does not contain tensor {key!r} that the current model needs",
        rank=context.rank,
        world_size=context.world_size,
        expected=key,
        observed=available[:8] + (["..."] if len(available) > 8 else []),
        resolution=(
            "resume with the model definition that produced the checkpoint, or pass "
            "strict=False to keep the current value"
        ),
    )


def _warn_on_config_drift(saved: Any, current: ExperimentConfig) -> None:
    """Log the configuration fields that changed since the checkpoint."""
    if not isinstance(saved, dict):
        return
    differences: list[str] = []

    def normalise(value: Any) -> Any:
        """Put both sides in the shape JSON round-tripping produces.

        Without this, a ``tuple`` field such as ``optimizer.betas`` reports a
        spurious difference on every resume, because JSON has no tuple type and
        the saved copy comes back as a list.  A warning that always fires is a
        warning nobody reads.
        """
        if isinstance(value, tuple | list):
            return [normalise(v) for v in value]
        if isinstance(value, dict):
            return {k: normalise(v) for k, v in value.items()}
        return value

    def walk(a: Any, b: Any, prefix: str) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                walk(a.get(key), b.get(key), f"{prefix}.{key}" if prefix else key)
        elif a != b:
            differences.append(f"{prefix}: {a!r} -> {b!r}")

    walk(normalise(saved), normalise(current.to_dict()), "")
    if differences:
        _LOGGER.warning(
            "configuration differs from the checkpoint (%d field(s)); resuming anyway:\n  %s",
            len(differences),
            "\n  ".join(differences[:12]),
        )


def inspect_checkpoint(directory: str | Path, *, verify: bool = True) -> dict[str, Any]:
    """Summarise a checkpoint without loading it into a model.

    Reads only JSON unless ``verify`` is set, in which case the payload files
    are hashed but still never unpickled.

    Args:
        directory: Checkpoint directory.
        verify: Recompute and compare every file's checksum.

    Returns:
        A JSON-serialisable summary.

    Raises:
        CheckpointError: If the checkpoint cannot be read or fails validation.
    """
    path = Path(directory)
    manifest = CheckpointManifest.read(path / MANIFEST_FILENAME)
    validate_format_version(manifest.format_version)
    manifest.validate()
    metadata = read_metadata(path)

    verification: dict[str, Any] = {"performed": verify}
    if verify:
        from .reshard import verify_files

        verify_files(path, manifest)
        verification["files_verified"] = len(manifest.referenced_files())

    by_category: dict[str, int] = {}
    elements_by_category: dict[str, int] = {}
    for record in manifest.tensors.values():
        by_category[record.category] = by_category.get(record.category, 0) + 1
        elements_by_category[record.category] = (
            elements_by_category.get(record.category, 0) + record.numel
        )

    return {
        "path": str(path),
        "format_version": manifest.format_version,
        "created_by": manifest.created_by,
        "step": manifest.step,
        "writer_world_size": manifest.writer_world_size,
        "topology": manifest.topology,
        "complete": manifest.complete,
        "tensor_counts": by_category,
        "element_counts": elements_by_category,
        "files": len(manifest.files),
        "total_bytes": sum(f.bytes for f in manifest.files.values()),
        "training_state": metadata.get("training_state", {}),
        "verification": verification,
    }


def _console_main() -> int:  # pragma: no cover - thin console wrapper
    """Entry point for the ``hybrid-inspect-checkpoint`` console script.

    Returns:
        Process exit code.
    """
    if len(sys.argv) < 2:
        print("usage: hybrid-inspect-checkpoint <checkpoint-directory>", file=sys.stderr)
        return 2
    try:
        summary = inspect_checkpoint(sys.argv[1])
    except CheckpointError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0
