#!/usr/bin/env python3
"""Train with the custom distributed data parallel wrapper.

Run it::

    torchrun --standalone --nproc-per-node=2 examples/train_ddp.py

Expected output (values differ by machine, the *shape* does not)::

    launch environment:
      RANK                   0
      WORLD_SIZE             2
      ...
    ... distributed context ready: backend=gloo device=cpu topology=...
    ... hybrid model ready
    parallel strategy: ddp
      parameters replicated over : ['data_parallel']
      ...
    ... step 10/40 loss=0.xxxxxx lr=3.000e-03 norm=0.xxxx 0.0xxs
    ========================================================================
    strategy            : ddp
    world size          : 2 (gloo on cpu)
    parameters (local)  : 6,472
    parameters (global) : 12,944          <- 2 replicas of the same 6,472
    ...

Note that ``parameters (global)`` is twice ``parameters (local)``: DDP
*replicates*, so the sum over ranks double-counts.  Compare with
``train_fsdp.py``, where the two numbers are (almost) equal because the model
is *split*.

Prerequisites: an editable install (``pip install -e ".[dev]"``).  No GPU is
needed; the default ``backend: auto`` falls back to Gloo on CPU when there are
fewer GPUs than processes.
"""

from __future__ import annotations

import sys

from _common import build_argument_parser, load_config, report_environment, report_result
from hybrid_training.distributed.context import distributed_context
from hybrid_training.training.engine import TrainingEngine


def main() -> int:
    """Run the DDP example.

    Returns:
        Process exit code.
    """
    parser = build_argument_parser("Train with custom bucketed DDP.", "ddp_2gpu.yaml")
    arguments = parser.parse_args()
    config = load_config(arguments)

    with distributed_context(config) as context:
        report_environment(context, show_topology=arguments.show_topology)
        engine = TrainingEngine(config, context)
        engine.train()
        if engine.eval_loader is not None:
            engine.evaluate()
        # Prove the replicas never drifted: with correct gradient averaging
        # every rank holds bitwise-identical parameters after every step.
        if engine.model.ddp is not None:
            engine.model.ddp.verify_replica_consistency()
            if context.is_primary:
                print("replica consistency verified: all ranks hold identical parameters")
        report_result(context, engine)
        engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
