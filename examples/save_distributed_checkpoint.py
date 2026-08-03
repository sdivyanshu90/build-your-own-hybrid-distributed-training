#!/usr/bin/env python3
"""Train briefly, write a distributed checkpoint, and inspect it.

Run it::

    torchrun --standalone --nproc-per-node=2 examples/save_distributed_checkpoint.py \
        --checkpoint-dir /tmp/hybrid-demo

Expected output (sizes and counts depend on the model)::

    ... step 10/10 loss=0.xxxxxx ...
    checkpoint written to /tmp/hybrid-demo/checkpoint-step-000010

    manifest summary
    checkpoint format 1.0 written by hybrid_training 0.1.0
      step               : 10
      writer world size  : 2
      topology           : {'world_size': 2, 'sizes': {...}, ...}
      complete           : True
      model tensors      : 14
      optimizer tensors  : 28
      buffer tensors     : 0
      total elements     : ...
      payload files      : 2

    directory contents
      manifest.json      12,345 bytes
      metadata.json       3,456 bytes
      rank-00000.pt      34,567 bytes
      rank-00001.pt      34,567 bytes

    reshard plan for 'blocks.0.linear.weight' wanting elements [0, 16)
      rank-00000.pt  saved_rank=0  shard=[0, 292]  overlap=[0, 16]

The last section shows the query the reader performs to reshard: it asks which
saved intervals overlap the interval it wants, and reads only those.  That is
why the checkpoint can be reloaded at a different world size.

Prerequisites: an editable install.  No GPU required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from _common import build_argument_parser, load_config, report_environment
from hybrid_training.checkpoint import inspect_checkpoint
from hybrid_training.checkpoint.manifest import CheckpointManifest
from hybrid_training.checkpoint.reshard import describe_reshard
from hybrid_training.distributed.context import distributed_context
from hybrid_training.training.engine import TrainingEngine
from hybrid_training.utils.tensors import ShardRange


def main() -> int:
    """Train, checkpoint and report.

    Returns:
        Process exit code.
    """
    parser = build_argument_parser("Write and inspect a distributed checkpoint.", "ddp_2gpu.yaml")
    parser.set_defaults(steps=10)
    arguments = parser.parse_args()
    config = load_config(arguments)

    with distributed_context(config) as context:
        report_environment(context, show_topology=arguments.show_topology)
        engine = TrainingEngine(config, context)
        engine.train()
        path = engine.save_checkpoint()
        engine.close()

        if not context.is_primary:
            return 0

        print(f"\ncheckpoint written to {path}")
        manifest = CheckpointManifest.read(path / "manifest.json")
        print("\nmanifest summary")
        print(manifest.summary())

        print("\ndirectory contents")
        for child in sorted(Path(path).iterdir()):
            print(f"  {child.name:<18} {child.stat().st_size:,} bytes")

        first_key = sorted(manifest.tensors_by_category("model"))[0]
        record = manifest.tensors[first_key]
        wanted = ShardRange(start=0, length=min(16, record.numel))
        plan = describe_reshard(manifest, first_key, wanted)
        print(f"\nreshard plan for {first_key!r} wanting elements [{wanted.start}, {wanted.end})")
        for source in plan["sources"]:
            print(
                f"  {source['file']}  saved_rank={source['saved_rank']}  "
                f"shard={source['shard_range']}  overlap={source['overlap']}"
            )

        print("\nfull inspection report")
        print(json.dumps(inspect_checkpoint(path), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
