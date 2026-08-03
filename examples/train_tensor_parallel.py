#!/usr/bin/env python3
"""Train a transformer whose weight matrices are split across ranks.

Run it::

    torchrun --standalone --nproc-per-node=2 examples/train_tensor_parallel.py

Expected output (2-rank run; exact numbers depend on the config)::

    ... hybrid model ready
    parallel strategy: tensor
      parameters sharded over    : ['tensor']
      note: ranks in the same tensor group must receive identical input batches
    ...
    tensor-parallel layout on this rank:
      blocks.0.attention.query      weight (32, 64)  <- half of (64, 64)
      blocks.0.attention.output     weight (64, 32)  <- half the other way
      blocks.0.feed_forward.fc1     weight (64, 64)  <- half of (128, 64)
      blocks.0.input_norm           weight (64,)     <- replicated
    ...
    ========================================================================
    strategy            : tensor
    parameters (local)  : 47,168
    parameters (global) : 94,336        <- the sum over ranks IS the model,
                                           because the ranks hold different
                                           slices rather than copies

Two things this example demonstrates:

* Column-parallel layers keep a slice of the *output* features
  (``(32, 64)`` from ``(64, 64)``) while row-parallel layers keep a slice of
  the *input* features (``(64, 32)``).
* Both ranks are checked for identical input batches.  Tensor-parallel peers
  compute complementary parts of the *same* sample; feeding them different
  data produces a plausible-looking loss curve and a wrong model.

Prerequisites: an editable install.  No GPU required.
"""

from __future__ import annotations

import sys

from _common import build_argument_parser, load_config, report_environment, report_result
from hybrid_training.distributed.context import distributed_context
from hybrid_training.training.engine import TrainingEngine


def main() -> int:
    """Run the tensor-parallel example.

    Returns:
        Process exit code.
    """
    parser = build_argument_parser(
        "Train a transformer with tensor-parallel linear layers.",
        "tensor_parallel_2gpu.yaml",
    )
    arguments = parser.parse_args()
    config = load_config(arguments)

    with distributed_context(config) as context:
        report_environment(context, show_topology=arguments.show_topology)
        engine = TrainingEngine(config, context)

        info = engine.model.parameter_parallel_info()
        if context.is_primary:
            print("\ntensor-parallel layout on this rank:")
            for name in sorted(info)[:10]:
                entry = info[name]
                kind = (
                    "replicated"
                    if entry.partition_dim is None
                    else f"split on dim {entry.partition_dim}"
                )
                print(f"  {name:<44} {entry.shape!s:<16} {kind}")

        # Verify that model-parallel peers really do see the same data.  The
        # engine's sampler guarantees it; this asserts the guarantee holds.
        batch = next(iter(engine.train_loader.iter_epoch(0)))
        engine.model.validate_input_replication(batch.inputs)
        if context.is_primary:
            print("input replication verified across the tensor-parallel group")

        engine.train()
        if engine.eval_loader is not None:
            engine.evaluate()
        report_result(context, engine)
        engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
