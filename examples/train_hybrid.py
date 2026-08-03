#!/usr/bin/env python3
"""Compose every parallelism strategy in one job.

Run it::

    # 4 ranks: dp=2 x shard=2, the smallest interesting hybrid
    torchrun --standalone --nproc-per-node=4 examples/train_hybrid.py

    # 8 ranks: dp=2 x shard=2 x tensor=2 with sequence parallelism
    torchrun --standalone --nproc-per-node=8 examples/train_hybrid.py \
        --config configs/hybrid_8gpu.yaml

Expected output (8-rank configuration)::

    rank topology map
      rank 0/8  coordinates dp0/sh0/sq0/tp0
        data_parallel    size=2   local_rank=0   members=(0, 4)
        shard            size=2   local_rank=0   members=(0, 2)
        tensor           size=2   local_rank=0   members=(0, 1)
        dp_shard         size=4   local_rank=0   members=(0, 2, 4, 6)
        ...
    parallel strategy: fsdp+replicate+tensor+sequence
      parameters sharded over    : ['tensor', 'shard']
      parameters replicated over : ['data_parallel']
      activations sharded over   : ['tensor']
      gradient reduction order   :
          1. all-reduce of activation gradients (inside the layers, ...) over 'tensor'
          2. reduce-scatter of flat gradients over 'shard'
          3. all-reduce of the sharded gradient over 'data_parallel'
      metrics reduced over       : 'dp_shard'
    ...

The ordering in "gradient reduction order" is the specification of this run.
Every collective in the step happens on the group named there, in that order,
on every rank.

Prerequisites: an editable install.  No GPU required -- with fewer GPUs than
processes the backend falls back to Gloo on CPU and prints a warning saying so.
"""

from __future__ import annotations

import sys

from _common import build_argument_parser, load_config, report_environment, report_result
from hybrid_training.distributed.context import distributed_context
from hybrid_training.parallel.hybrid import describe_parallel_plan
from hybrid_training.training.engine import TrainingEngine


def main() -> int:
    """Run the hybrid example.

    Returns:
        Process exit code.
    """
    parser = build_argument_parser(
        "Compose data, shard, tensor and sequence parallelism.", "hybrid_4gpu.yaml"
    )
    arguments = parser.parse_args()
    config = load_config(arguments)

    with distributed_context(config) as context:
        report_environment(context, show_topology=arguments.show_topology)

        # Every rank prints its own placement.  In a real job you would log
        # this once; here it is the point of the example.
        print("rank topology map")
        print("  " + context.describe().replace("\n", "\n  "))

        plan = describe_parallel_plan(config, context)
        if context.is_primary:
            print()
            print(plan.render())

        engine = TrainingEngine(config, context)
        engine.train()
        if engine.eval_loader is not None:
            engine.evaluate()

        # Assert the two invariants that hybrid composition is most likely to
        # get wrong: model-parallel peers must share a batch, and replicas must
        # not have drifted.
        batch = next(iter(engine.train_loader.iter_epoch(0)))
        engine.model.validate_input_replication(batch.inputs)
        if context.is_primary:
            print("\ninput replication verified across the tensor/sequence group")

        report_result(context, engine)
        engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
