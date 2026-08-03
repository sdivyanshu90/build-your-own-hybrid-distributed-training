"""Atomic, integrity-checked distributed checkpoint writing.

The protocol
============
.. code-block:: text

    all ranks   agree on a staging directory name (rank 0 chooses, broadcast)
    all ranks   write their own payload file into the staging directory
    all ranks   compute the SHA-256 of the file they wrote
    all ranks   all-gather their tensor records + file records
       |
       | barrier: every rank has finished writing
       v
    rank 0      assemble the manifest, validate it, verify every file exists
    rank 0      write metadata.json, then manifest.json  (manifest LAST)
    rank 0      rename the staging directory into place  (atomic)
       |
       | barrier: the checkpoint is visible to everyone
       v
    all ranks   continue training

Why the manifest is written last, and why the rename
====================================================
A reader trusts exactly one thing: the manifest.  Writing it last means an
interrupted save leaves a staging directory with no manifest, which no reader
will ever look at (the name begins with a dot and the final path does not
exist).  The ``rename`` then publishes everything at once: on a POSIX
filesystem, within one mount point, a directory rename is atomic, so a
concurrent reader sees either "no such directory" or the complete checkpoint --
never a half-populated one.

The ``complete`` flag is belt and braces for the case where a staging directory
*does* get inspected: a manifest that was somehow written without the final
validation is visibly incomplete.

Who writes what
===============
Not every rank writes every tensor, and it matters which ones do:

===============================  =========================================
Dimension                        Writing policy
===============================  =========================================
``shard`` (FSDP)                 every rank writes -- they hold *different*
                                 elements
``tensor``                       every rank writes -- a tensor-parallel
                                 slice is a distinct tensor with its own
                                 storage key
``data_parallel``                only coordinate 0 writes -- the others hold
                                 byte-identical copies
``sequence`` (independent mode)  only coordinate 0 writes -- parameters are
                                 replicated across it
===============================  =========================================

RNG state is the exception: it is genuinely per-rank, so every rank writes its
own and the metadata records which ranks are present.
"""

from __future__ import annotations

import shutil
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .. import __version__
from ..config import ExperimentConfig
from ..distributed.collectives import all_gather_object_in_group
from ..distributed.context import DistributedContext
from ..errors import CheckpointError, format_error
from ..logging import get_logger
from ..optim.sharded_optimizer import ShardedOptimizer
from ..parallel.hybrid import HybridModel
from ..training.state import TrainingState
from ..utils.reproducibility import capture_rng_state, rng_state_to_serialisable
from .format import (
    CHECKPOINT_DIRECTORY_PATTERN,
    CURRENT_FORMAT_VERSION,
    MANIFEST_FILENAME,
    METADATA_FILENAME,
    checkpoint_directory_name,
    file_digest,
    shard_filename,
    staging_directory_name,
    step_from_directory_name,
)
from .manifest import CheckpointManifest, FileRecord, ShardRecord, TensorRecord

__all__ = [
    "CheckpointWriteResult",
    "find_latest_checkpoint",
    "prune_checkpoints",
    "save_checkpoint",
]

_LOGGER = get_logger(__name__)

#: Storage keys inside a payload file are ``"<category>::<key>@<offset>"`` for
#: manifest-described tensors; RNG state is per-rank and lives outside the
#: manifest's global-tensor space, so it gets its own prefix.
_RNG_PREFIX = "rng::"


class CheckpointWriteResult:
    """Outcome of a save.

    Attributes:
        path: Final checkpoint directory.
        step: Step the checkpoint was taken at.
        seconds: Wall-clock duration of the save on this rank.
        bytes_written: Payload bytes this rank wrote.
        num_tensors: Tensor records this rank contributed.
    """

    def __init__(
        self, path: Path, step: int, seconds: float, bytes_written: int, num_tensors: int
    ) -> None:
        self.path = path
        self.step = step
        self.seconds = seconds
        self.bytes_written = bytes_written
        self.num_tensors = num_tensors

    def __repr__(self) -> str:
        return (
            f"CheckpointWriteResult(path={self.path.name!r}, step={self.step}, "
            f"{self.seconds:.3f}s, {self.bytes_written} bytes, {self.num_tensors} tensors)"
        )


def _writes_replicated_tensors(context: DistributedContext) -> bool:
    """Whether this rank should write parameters replicated over dp/sequence.

    Ranks whose data-parallel and sequence coordinates are both zero are the
    designated writers of every tensor that is replicated across those
    dimensions.  Their peers hold byte-identical copies, so writing from all of
    them would multiply the checkpoint size by the replication degree and force
    the manifest to describe overlapping shards.
    """
    coordinates = context.topology.coordinates_of(context.rank)
    return coordinates.data_parallel == 0 and coordinates.sequence == 0


def save_checkpoint(
    directory: str | Path,
    *,
    model: HybridModel,
    context: DistributedContext,
    state: TrainingState,
    optimizer: ShardedOptimizer | None = None,
    config: ExperimentConfig | None = None,
    scheduler_state: dict[str, Any] | None = None,
    scaler_state: dict[str, Any] | None = None,
    extra_metadata: dict[str, Any] | None = None,
    save_optimizer: bool = True,
    save_rng: bool = True,
    keep_last: int = 0,
) -> CheckpointWriteResult:
    """Write a distributed checkpoint atomically.

    Args:
        directory: Root directory that holds ``checkpoint-step-*`` folders.
        model: The wrapped model to save.
        context: Active distributed context.
        state: Training progress.
        optimizer: Optimizer whose state should be saved.
        config: Experiment configuration, recorded for provenance and for the
            resume-time compatibility check.
        scheduler_state: Learning-rate schedule state.
        scaler_state: Gradient-scaler state.
        extra_metadata: Arbitrary JSON-serialisable extras, e.g. dataloader
            progress.
        save_optimizer: Persist optimizer state shards.
        save_rng: Persist per-rank RNG state.
        keep_last: Retain only the newest ``keep_last`` checkpoints.  ``0``
            keeps everything.

    Returns:
        A :class:`CheckpointWriteResult`.

    Raises:
        CheckpointError: If the destination already exists, or if the assembled
            manifest fails validation (in which case nothing is published).
    """
    started = time.perf_counter()
    root = Path(directory)
    final_name = checkpoint_directory_name(state.step)
    final_path = root / final_name

    world = context.group("world")
    # Rank 0 chooses the staging token *and* decides whether the destination is
    # free; every rank learns both from the same all-gather.
    #
    # The decision has to be collective.  If only rank 0 raised, every other
    # rank would walk into the barrier below and block there until rank 0's
    # process died -- converting a clean, recoverable, obviously-diagnosable
    # error into a hang, which is the single failure mode this project works
    # hardest to design out.  An error that is not raised on every rank is not
    # an error, it is a deadlock with an explanation on one node.
    decision: dict[str, Any] = (
        {"token": uuid.uuid4().hex[:12], "exists": final_path.exists()}
        if context.is_primary
        else {"token": "", "exists": False}
    )
    decision = all_gather_object_in_group(decision, world)[0]
    if decision["exists"]:
        raise CheckpointError(
            format_error(
                "checkpoint.save",
                "the destination checkpoint already exists",
                rank=context.rank,
                world_size=context.world_size,
                expected="a fresh directory",
                observed=str(final_path),
                resolution="remove the existing checkpoint or save at a different step",
            )
        )
    staging_path = root / staging_directory_name(final_name, decision["token"])

    # Only rank 0 creates the staging directory, but a failure to create it is
    # reported collectively for the same reason as above.
    mkdir_error = ""
    if context.is_primary:
        try:
            staging_path.mkdir(parents=True, exist_ok=True)
        except OSError as error:  # pragma: no cover - filesystem failure
            mkdir_error = str(error)
    mkdir_error = all_gather_object_in_group(mkdir_error, world)[0]
    if mkdir_error:
        raise CheckpointError(
            format_error(
                "checkpoint.save",
                "the staging directory could not be created",
                rank=context.rank,
                world_size=context.world_size,
                expected=str(staging_path),
                observed=mkdir_error,
                resolution="check permissions and free space on the checkpoint filesystem",
            )
        )
    context.barrier("world", label="checkpoint-staging-created")

    payload, records, optimizer_scalars = _collect_payload(
        model=model,
        optimizer=optimizer,
        context=context,
        save_optimizer=save_optimizer,
        save_rng=save_rng,
    )

    filename = shard_filename(context.rank)
    file_path = staging_path / filename
    torch.save(payload, file_path)
    size = file_path.stat().st_size
    digest = file_digest(file_path)

    contribution = {
        "file": {"name": filename, "sha256": digest, "bytes": size},
        "records": records,
        "optimizer_scalars": optimizer_scalars,
    }
    gathered = all_gather_object_in_group(contribution, world)
    context.barrier("world", label="checkpoint-payloads-written")

    # The whole publish phase happens on rank 0 alone -- assembling the
    # manifest, validating it, writing the two JSON files and renaming the
    # staging directory into place.  Any failure in it is caught and broadcast
    # so that every rank raises, for the same reason as the existence check
    # above: a raise on rank 0 alone would strand the others in the
    # "checkpoint-published" barrier.  `Exception` is caught deliberately
    # broadly -- an OSError from a full disk must not become a hang either.
    publish_error = ""
    if context.is_primary:
        try:
            _publish(
                context=context,
                state=state,
                config=config,
                scheduler_state=scheduler_state,
                scaler_state=scaler_state,
                extra_metadata=extra_metadata,
                gathered=gathered,
                save_rng=save_rng,
                staging_path=staging_path,
                final_path=final_path,
            )
        except Exception as error:
            publish_error = f"{type(error).__name__}: {error}"
    publish_error = all_gather_object_in_group(publish_error, world)[0]
    if publish_error:
        raise CheckpointError(
            format_error(
                "checkpoint.save",
                "the checkpoint could not be published; nothing was renamed into "
                "place, so the checkpoint directory is unchanged",
                rank=context.rank,
                world_size=context.world_size,
                expected="a validated manifest covering every tensor exactly once",
                observed=publish_error,
                resolution=(
                    "read the reported error from rank 0; the staging directory "
                    f"{staging_path.name!r} is left behind for inspection"
                ),
            )
        )

    context.barrier("world", label="checkpoint-published")

    if context.is_primary and keep_last > 0:
        prune_checkpoints(root, keep_last=keep_last)
    context.barrier("world", label="checkpoint-pruned")

    return CheckpointWriteResult(
        path=final_path,
        step=state.step,
        seconds=time.perf_counter() - started,
        bytes_written=size,
        num_tensors=len(records),
    )


def _publish(
    *,
    context: DistributedContext,
    state: TrainingState,
    config: ExperimentConfig | None,
    scheduler_state: dict[str, Any] | None,
    scaler_state: dict[str, Any] | None,
    extra_metadata: dict[str, Any] | None,
    gathered: list[dict[str, Any]],
    save_rng: bool,
    staging_path: Path,
    final_path: Path,
) -> None:
    """Assemble, validate and atomically publish the checkpoint on rank 0.

    Runs on the primary rank only.  Every failure path raises; the caller turns
    a raise into a collective error so no rank is left in a barrier.

    Args:
        context: Active distributed context.
        state: Training progress being recorded.
        config: Experiment configuration, stored for provenance.
        scheduler_state: Learning-rate schedule state.
        scaler_state: Gradient-scaler state.
        extra_metadata: Caller-supplied JSON-serialisable extras.
        gathered: Per-rank manifest contributions, ordered by rank.
        save_rng: Whether per-rank RNG state was collected.
        staging_path: Directory holding the payload files.
        final_path: Directory to publish to.

    Raises:
        CheckpointError: If the manifest fails validation or references a file
            that is not present, in which case nothing is published.
    """
    manifest = CheckpointManifest(
        format_version=CURRENT_FORMAT_VERSION,
        created_by=f"hybrid_training {__version__}",
        writer_world_size=context.world_size,
        topology=context.topology.summary(),
        step=state.step,
    )
    for entry in gathered:
        file_record = FileRecord.from_dict(entry["file"])
        manifest.files[file_record.name] = file_record
        for item in entry["records"]:
            manifest.add_shard(
                TensorRecord(
                    key=item["key"],
                    name=item["name"],
                    global_shape=tuple(item["global_shape"]),
                    dtype=item["dtype"],
                    category=item["category"],
                    partition_dim=item["partition_dim"],
                    tensor_parallel_size=item["tensor_parallel_size"],
                    tensor_parallel_rank=item["tensor_parallel_rank"],
                    state_name=item["state_name"],
                ),
                ShardRecord(
                    rank=item["rank"],
                    file=item["file"],
                    offset=item["offset"],
                    length=item["length"],
                    key=item["storage_key"],
                    padding=item.get("padding", 0),
                ),
            )
    manifest.complete = True
    # Validate *before* publishing: a manifest that does not describe a
    # complete, non-overlapping cover of every tensor must never be
    # renamed into place, because a reader would trust it.
    manifest.validate()
    for name in manifest.referenced_files():
        if not (staging_path / name).is_file():
            raise CheckpointError(
                format_error(
                    "checkpoint.save",
                    "a rank reported a file that is not present in the staging "
                    "directory, so the checkpoint would be incomplete",
                    rank=context.rank,
                    world_size=context.world_size,
                    expected=name,
                    observed="missing",
                    resolution="check for a filesystem error on the reporting rank",
                )
            )

    metadata = _build_metadata(
        context=context,
        state=state,
        config=config,
        scheduler_state=scheduler_state,
        scaler_state=scaler_state,
        extra_metadata=extra_metadata,
        gathered=gathered,
        save_rng=save_rng,
    )
    (staging_path / METADATA_FILENAME).write_text(_dump_json(metadata), encoding="utf-8")
    manifest.write(staging_path / MANIFEST_FILENAME)
    # Atomic publish.
    staging_path.replace(final_path)
    _LOGGER.info(
        "checkpoint written: %s (%d tensors, %d files)",
        final_path,
        len(manifest.tensors),
        len(manifest.files),
    )


def _collect_payload(
    *,
    model: HybridModel,
    optimizer: ShardedOptimizer | None,
    context: DistributedContext,
    save_optimizer: bool,
    save_rng: bool,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], dict[str, Any]]:
    """Build this rank's tensor payload and its manifest contributions.

    Returns:
        ``(payload, records, scalars)`` where ``payload`` maps storage key to a
        1-D CPU tensor, ``records`` is a list of JSON-serialisable shard
        descriptions, and ``scalars`` holds the non-tensor optimizer state
        (Adam's step counter and the like).
    """
    filename = shard_filename(context.rank)
    parallel_info = model.parameter_parallel_info()
    writes_replicated = _writes_replicated_tensors(context)

    payload: dict[str, torch.Tensor] = {}
    records: list[dict[str, Any]] = []
    scalars: dict[str, Any] = {}

    def add(
        *,
        key: str,
        name: str,
        global_shape: tuple[int, ...],
        offset: int,
        data: torch.Tensor,
        category: str,
        state_name: str = "",
    ) -> None:
        info = parallel_info.get(name)
        storage_key = f"{category}::{key}@{offset}"
        payload[storage_key] = data.detach().to("cpu").reshape(-1).clone()
        records.append(
            {
                "key": key,
                "name": name,
                "global_shape": list(global_shape),
                "dtype": str(data.dtype),
                "category": category,
                "partition_dim": None if info is None else info.partition_dim,
                "tensor_parallel_size": 1 if info is None else info.tensor_parallel_size,
                "tensor_parallel_rank": 0 if info is None else info.tensor_parallel_rank,
                "state_name": state_name,
                "rank": context.rank,
                "file": filename,
                "offset": offset,
                "length": int(data.numel()),
                "storage_key": storage_key,
            }
        )

    if writes_replicated:
        for name, piece in model.sharded_state_dict().items():
            info = parallel_info.get(name)
            key = info.storage_key if info is not None else name
            add(
                key=key,
                name=name,
                global_shape=piece.global_shape,
                offset=piece.offset,
                data=piece.data,
                category="model",
            )
        for name, buffer in model.buffers_state_dict().items():
            add(
                key=name,
                name=name,
                global_shape=tuple(buffer.shape),
                offset=0,
                data=buffer.reshape(-1),
                category="buffer",
            )

        if save_optimizer and optimizer is not None:
            layouts = model.optimizer_parameter_layout()
            inner_state = optimizer.inner.state
            for index, param in enumerate(optimizer.parameters):
                entry = inner_state.get(param)
                if not entry:
                    continue
                for state_name, value in entry.items():
                    if not torch.is_tensor(value) or value.numel() != param.numel():
                        # Non-tensor state -- Adam's step counter and the like.
                        # It has no shape to shard, so it lives in
                        # metadata.json as one value per state name.  This
                        # implementation steps every parameter together, so the
                        # values always agree; if they ever did not, silently
                        # keeping one of them would corrupt the resumed bias
                        # correction, so the disagreement is an error.
                        scalar = _as_scalar(value)
                        previous = scalars.get(state_name)
                        if previous is not None and previous != scalar:
                            raise CheckpointError(
                                format_error(
                                    "checkpoint.save",
                                    f"optimizer state {state_name!r} differs between "
                                    "parameters; this format stores one value per state "
                                    "name and cannot represent per-parameter counters",
                                    rank=context.rank,
                                    expected=previous,
                                    observed=scalar,
                                    resolution=(
                                        "step every parameter together, or extend the "
                                        "format to store per-parameter scalars"
                                    ),
                                )
                            )
                        scalars[state_name] = scalar
                        continue
                    flat = value.detach().reshape(-1)
                    for item in layouts[index]:
                        info = parallel_info.get(item.name)
                        base_key = info.storage_key if info is not None else item.name
                        add(
                            key=f"{base_key}::{state_name}",
                            name=item.name,
                            global_shape=item.global_shape,
                            offset=item.parameter_offset,
                            data=flat[item.local_offset : item.local_offset + item.length],
                            category="optimizer",
                            state_name=state_name,
                        )

    if save_rng:
        # RNG state is genuinely per-rank, so it bypasses the replication rule
        # and is stored outside the manifest's global-tensor space.
        serialised = rng_state_to_serialisable(capture_rng_state())
        for name, tensor in serialised["tensors"].items():
            payload[f"{_RNG_PREFIX}{name}"] = tensor
    return payload, records, scalars


def _as_scalar(value: Any) -> Any:
    """Convert a non-tensor optimizer state value into a JSON-safe scalar."""
    if torch.is_tensor(value):
        if value.numel() != 1:
            return [float(v) for v in value.reshape(-1).tolist()]
        return value.item()
    return value


def _build_metadata(
    *,
    context: DistributedContext,
    state: TrainingState,
    config: ExperimentConfig | None,
    scheduler_state: dict[str, Any] | None,
    scaler_state: dict[str, Any] | None,
    extra_metadata: dict[str, Any] | None,
    gathered: list[dict[str, Any]],
    save_rng: bool,
) -> dict[str, Any]:
    """Assemble the JSON metadata document."""
    optimizer_scalars: dict[str, Any] = {}
    for entry in gathered:
        optimizer_scalars.update(entry.get("optimizer_scalars", {}))
    return {
        "format_version": CURRENT_FORMAT_VERSION,
        "created_by": f"hybrid_training {__version__}",
        "training_state": state.as_dict(),
        "scheduler": scheduler_state or {},
        "scaler": scaler_state or {},
        "config": None if config is None else config.to_dict(),
        "topology": context.topology.summary(),
        "world_size": context.world_size,
        "backend": context.backend,
        "rng": {
            "saved": save_rng,
            "ranks": list(range(context.world_size)) if save_rng else [],
            "meta": (rng_state_to_serialisable(capture_rng_state())["meta"] if save_rng else {}),
        },
        "optimizer_scalars": optimizer_scalars,
        "extra": extra_metadata or {},
    }


def _dump_json(payload: dict[str, Any]) -> str:
    """Serialise metadata to JSON, rejecting values JSON cannot represent."""
    import json

    def default(value: Any) -> Any:
        if isinstance(value, set | frozenset | tuple):
            return list(value)
        if hasattr(value, "as_dict"):
            return value.as_dict()
        if hasattr(value, "__dataclass_fields__"):
            return asdict(value)
        raise TypeError(
            f"checkpoint metadata must be JSON-serialisable; {type(value).__name__} is not. "
            "Tensors belong in the payload files, not in metadata.json."
        )

    return json.dumps(payload, indent=2, sort_keys=True, default=default, allow_nan=False)


def prune_checkpoints(root: str | Path, *, keep_last: int) -> list[Path]:
    """Delete all but the newest ``keep_last`` checkpoints.

    Only directories matching the checkpoint naming pattern are considered, and
    only complete ones are counted towards the limit -- an incomplete directory
    is removed regardless, because it can never be resumed from.

    Args:
        root: Directory holding the checkpoints.
        keep_last: How many to retain.  ``0`` or negative keeps everything.

    Returns:
        The directories that were removed.
    """
    if keep_last <= 0:
        return []
    root_path = Path(root)
    if not root_path.is_dir():
        return []
    candidates: list[tuple[int, Path]] = []
    for child in root_path.iterdir():
        if not child.is_dir():
            continue
        step = step_from_directory_name(child.name)
        if step is None:
            continue
        candidates.append((step, child))
    candidates.sort(key=lambda item: item[0])
    removed: list[Path] = []
    for _, path in candidates[: max(len(candidates) - keep_last, 0)]:
        shutil.rmtree(path, ignore_errors=True)
        removed.append(path)
        _LOGGER.info("pruned old checkpoint %s", path.name)
    return removed


def find_latest_checkpoint(root: str | Path) -> Path | None:
    """Return the newest complete checkpoint directory under ``root``.

    "Complete" is decided by the presence of a manifest, which the writer
    creates last, so a crashed save is never selected.

    Args:
        root: Directory to search.

    Returns:
        The newest checkpoint path, or ``None`` if there is none.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        return None
    best: tuple[int, Path] | None = None
    for child in sorted(root_path.iterdir()):
        if not child.is_dir() or not CHECKPOINT_DIRECTORY_PATTERN.match(child.name):
            continue
        if not (child / MANIFEST_FILENAME).is_file():
            continue
        step = step_from_directory_name(child.name)
        if step is None:  # pragma: no cover - guarded by the pattern match
            continue
        if best is None or step > best[0]:
            best = (step, child)
    return None if best is None else best[1]
