#!/usr/bin/env python3
"""Inspect a distributed checkpoint without loading it into a model.

Usage::

    python scripts/inspect_checkpoint.py /path/to/checkpoint-step-000100
    python scripts/inspect_checkpoint.py CHECKPOINT --json
    python scripts/inspect_checkpoint.py CHECKPOINT --tensors --limit 20
    python scripts/inspect_checkpoint.py CHECKPOINT --plan blocks.0.linear.weight
    python scripts/inspect_checkpoint.py CHECKPOINT --no-verify

Expected output::

    checkpoint: /tmp/runs/checkpoint-step-000100
    checkpoint format 1.0 written by hybrid_training 0.1.0
      step               : 100
      writer world size  : 4
      topology           : {'world_size': 4, 'sizes': {...}}
      complete           : True
      model tensors      : 14
      optimizer tensors  : 28
      buffer tensors     : 0
      total elements     : 5571
      payload files      : 4

    training state
      step                 100
      epoch                6
      samples_seen         3200
      ...

    integrity
      files verified       4 / 4
      total payload bytes  102,400

What this tool does *not* do
===========================
It never calls ``torch.load`` on a payload file unless ``--tensors`` is given
with ``--values``, and even then it loads with ``weights_only=True``.  The
manifest and metadata are plain JSON, so a checkpoint from an untrusted source
can be examined safely before anything is deserialised.

Exit codes: ``0`` valid, ``1`` invalid or unreadable, ``2`` bad usage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from a source checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hybrid_training.checkpoint.format import MANIFEST_FILENAME
from hybrid_training.checkpoint.manifest import CheckpointManifest
from hybrid_training.checkpoint.reader import inspect_checkpoint, read_metadata
from hybrid_training.checkpoint.reshard import describe_reshard
from hybrid_training.errors import CheckpointError
from hybrid_training.utils.memory import format_bytes
from hybrid_training.utils.tensors import ShardRange


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Inspect and validate a distributed checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("checkpoint", help="checkpoint directory to inspect")
    parser.add_argument(
        "--json", action="store_true", help="emit the summary as JSON and nothing else"
    )
    parser.add_argument("--tensors", action="store_true", help="list every tensor and its shards")
    parser.add_argument(
        "--limit", type=int, default=20, help="maximum tensors to list with --tensors"
    )
    parser.add_argument(
        "--plan",
        metavar="TENSOR",
        help="show which files a full read of TENSOR would touch",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip checksum verification (faster; only for a checkpoint you trust)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    arguments = build_parser().parse_args(argv)
    path = Path(arguments.checkpoint)

    if not path.is_dir():
        print(f"error: {path} is not a directory", file=sys.stderr)
        return 2

    try:
        summary = inspect_checkpoint(path, verify=not arguments.no_verify)
        manifest = CheckpointManifest.read(path / MANIFEST_FILENAME)
        metadata = read_metadata(path)
    except CheckpointError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if arguments.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    print(f"checkpoint: {path}")
    print(manifest.summary())

    print("\ntraining state")
    for key, value in sorted(metadata.get("training_state", {}).items()):
        print(f"  {key:<20} {value}")

    scheduler = metadata.get("scheduler") or {}
    if scheduler:
        print("\nlearning-rate schedule")
        for key, value in sorted(scheduler.items()):
            print(f"  {key:<20} {value}")

    rng = metadata.get("rng") or {}
    if rng:
        print("\nRNG state")
        print(f"  {'saved':<20} {rng.get('saved')}")
        print(f"  {'ranks':<20} {rng.get('ranks')}")

    print("\nintegrity")
    verified = summary["verification"].get("files_verified", 0)
    print(f"  {'files verified':<20} {verified} / {len(manifest.files)}")
    print(f"  {'total payload bytes':<20} {format_bytes(summary['total_bytes'])}")
    if arguments.no_verify:
        print("  (checksums were NOT verified: --no-verify was given)")

    if arguments.tensors:
        print("\ntensors")
        for index, key in enumerate(sorted(manifest.tensors)):
            if index >= arguments.limit:
                remaining = len(manifest.tensors) - arguments.limit
                print(f"  ... and {remaining} more (raise --limit to see them)")
                break
            record = manifest.tensors[key]
            print(
                f"  {key:<52} {tuple(record.global_shape)!s:<16} "
                f"{record.dtype:<16} {record.category:<10} "
                f"{len(record.shards)} shard(s)"
            )
            for shard in sorted(record.shards, key=lambda s: s.offset):
                print(
                    f"      [{shard.offset:>8}, {shard.offset + shard.length:>8})  "
                    f"{shard.file}  (written by rank {shard.rank})"
                )

    if arguments.plan:
        record = manifest.tensors.get(arguments.plan)
        if record is None:
            print(f"\nerror: no tensor named {arguments.plan!r}", file=sys.stderr)
            candidates = sorted(manifest.tensors)[:10]
            print(f"       known tensors include: {candidates}", file=sys.stderr)
            return 1
        plan = describe_reshard(manifest, arguments.plan, ShardRange(start=0, length=record.numel))
        print(f"\nread plan for {arguments.plan!r} (shape {tuple(record.global_shape)})")
        for source in plan["sources"]:
            print(
                f"  {source['file']}  saved by rank {source['saved_rank']}  "
                f"shard={source['shard_range']}  overlap={source['overlap']}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
