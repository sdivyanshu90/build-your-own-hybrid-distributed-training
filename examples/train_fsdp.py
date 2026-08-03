#!/usr/bin/env python3
"""Train with the custom FSDP-style sharding wrapper.

Run it::

    torchrun --standalone --nproc-per-node=2 examples/train_fsdp.py
    torchrun --standalone --nproc-per-node=4 examples/train_fsdp.py \
        --config configs/fsdp_4gpu.yaml

Expected output (exact values depend on the config and the machine; the
*relationships* are what matter)::

    ... hybrid model ready
    parallel strategy: fsdp
      parameters sharded over    : ['shard']
      gradient reduction order   :
          1. reduce-scatter of flat gradients over 'shard'
    ...
    ========================================================================
    strategy            : fsdp
    parameters (local)  : P/shard_size, rounded up, plus padding
    parameters (global) : P             <- the sum over ranks IS the model
    memory (per rank)   : shard=... grad=... padding=... units=...

The two things to look at:

* ``parameters (local)`` is roughly ``parameters (global) / shard_size``.  With
  DDP those two numbers differ by a factor of the world size in the *other*
  direction.
* ``padding`` is non-zero whenever a unit's parameter count is not divisible by
  the shard-group size.  Padding never reaches the optimizer as a trainable
  value and is always zero in a gradient.

The example also demonstrates ``summon_full_params()``, which temporarily
reassembles the sharded weights so ordinary code can look at them.

Prerequisites: an editable install.  No GPU required.
"""

from __future__ import annotations

import sys

from _common import build_argument_parser, load_config, report_environment, report_result
from hybrid_training.distributed.context import distributed_context
from hybrid_training.training.engine import TrainingEngine


def main() -> int:
    """Run the FSDP example.

    Returns:
        Process exit code.
    """
    parser = build_argument_parser(
        "Train with custom FSDP-style parameter/gradient/optimizer sharding.",
        "fsdp_2gpu.yaml",
    )
    arguments = parser.parse_args()
    config = load_config(arguments)

    with distributed_context(config) as context:
        report_environment(context, show_topology=arguments.show_topology)
        engine = TrainingEngine(config, context)

        if engine.model.fsdp is not None and context.is_primary:
            handle = engine.model.fsdp.handle
            print("\nFSDP units on this rank:")
            for unit in engine.model.fsdp.fsdp_units():
                if unit.handle is not None:
                    print(f"  {unit.handle}")
            if handle is not None:
                print(f"  root unit local shard range: {handle.local_shard_range().as_tuple()}")

        engine.train()
        if engine.eval_loader is not None:
            engine.evaluate()

        # summon_full_params reassembles the sharded parameters so that
        # ordinary (non-distributed) code can inspect them.  Inside the block
        # `module.weight` is a whole tensor again; outside it, it is an empty
        # placeholder and only the flat shard is resident.
        if engine.model.fsdp is not None:
            with engine.model.summon_full_params():
                first_block = engine.model.inner_model.blocks[0]
                summoned_shape = tuple(first_block.linear.weight.shape)
            resharded_shape = tuple(engine.model.inner_model.blocks[0].linear.weight.shape)
            full = engine.model.full_state_dict()
            if context.is_primary:
                print(f"\nblocks.0.linear.weight inside summon_full_params : {summoned_shape}")
                print(f"blocks.0.linear.weight after resharding           : {resharded_shape}")
                print(f"reconstructed {len(full)} full tensors via all-gather")
                for name in list(full)[:5]:
                    print(f"  {name:<40} {tuple(full[name].shape)}")

        report_result(context, engine)
        engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
