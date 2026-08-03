#!/usr/bin/env python3
"""Train with tensor parallelism fused with sequence parallelism.

Run it::

    torchrun --standalone --nproc-per-node=2 examples/train_sequence_parallel.py

Expected output (2-rank run; exact numbers depend on the config)::

    ... hybrid model ready
    parallel strategy: tensor+sequence
      activations sharded over   : ['tensor']
    ...
    sequence-parallel placement:
      full sequence length     : 16
      positions held by rank 0 : [0, 8)
      LayerNorm parameters with partial gradients: 10
    ...
    ========================================================================
    strategy            : tensor+sequence

What sequence parallelism changes, and what it does not:

* The regions *between* tensor-parallel layers -- LayerNorm, dropout, the
  residual adds -- hold ``(batch, seq/2, hidden)`` instead of
  ``(batch, seq, hidden)``.  That is the memory saving, and it is free: the
  all-reduce plain tensor parallelism would perform is replaced by a
  reduce-scatter plus an all-gather, which move the same bytes.
* Attention still needs the whole sequence.  The q/k/v projections all-gather
  it, so this is *not* a communication-free way to shard the sequence through
  attention -- that is context parallelism, which this project does not
  implement (see docs/06_sequence_parallelism.md).
* LayerNorm gains only ever see this rank's positions, so their gradients are
  partial and must be summed across the sequence-parallel group after backward.
  ``HybridModel.finish_backward()`` does that; the count is printed above.

Prerequisites: an editable install.  No GPU required.
"""

from __future__ import annotations

import sys

from _common import build_argument_parser, load_config, report_environment, report_result
from hybrid_training.distributed.context import distributed_context
from hybrid_training.parallel.sequence_parallel import (
    SequenceShardInfo,
    local_sequence_slice,
)
from hybrid_training.training.engine import TrainingEngine


def main() -> int:
    """Run the sequence-parallel example.

    Returns:
        Process exit code.
    """
    parser = build_argument_parser(
        "Train a transformer with fused tensor + sequence parallelism.",
        "sequence_parallel_2gpu.yaml",
    )
    arguments = parser.parse_args()
    config = load_config(arguments)

    with distributed_context(config) as context:
        report_environment(context, show_topology=arguments.show_topology)
        engine = TrainingEngine(config, context)

        sequence_group = context.group("sequence_effective")
        length = config.data.sequence_length
        # The model pads the sequence up to a multiple of the group size before
        # scattering it, so the positions this rank owns are positions of the
        # *padded* sequence.  Ask the same helper the model uses rather than
        # recomputing the rounding here.
        shard_info = SequenceShardInfo.for_length(length, sequence_group.size)
        start, end = local_sequence_slice(shard_info.padded_length, sequence_group)
        partial = sum(
            1
            for parameter in engine.model.inner_model.parameters()
            if getattr(parameter, "sequence_parallel_partial_grad", False)
        )
        if context.is_primary:
            print("\nsequence-parallel placement:")
            print(f"  full sequence length     : {length}")
            print(
                f"  padded to                : {shard_info.padded_length}"
                f" ({shard_info.padding} position(s) of padding)"
            )
            print(f"  sequence-parallel size   : {sequence_group.size}")
            print(f"  positions held by rank 0 : [{start}, {end})")
            print(f"  parameters with partial gradients: {partial}")

        engine.train()
        if engine.eval_loader is not None:
            engine.evaluate()
        report_result(context, engine)
        engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
