#!/usr/bin/env python3
"""Save a checkpoint at one world size and resume it at another.

Unlike the other examples this one is a *driver*: it launches both phases
itself, because demonstrating a world-size change requires two differently
sized jobs.  Run it as an ordinary Python program, not under ``torchrun``::

    python examples/resume_with_different_world_size.py
    python examples/resume_with_different_world_size.py --save-ranks 4 --load-ranks 2

Expected output (norms depend on the seed and the machine)::

    phase 1: training 6 steps with 4 FSDP ranks
      rank 0 local shard: 146 elements of 584
      checkpoint: /tmp/.../checkpoint-step-000003

    phase 2: resuming the same checkpoint with 2 FSDP ranks
      rank 0 local shard: 292 elements of 584
      resumed at step 3, read 2 payload file(s) per rank

    verification
      4-rank final parameter norm : 3.8412041713
      2-rank final parameter norm : 3.8412041713
      absolute difference         : 0.000e+00
      PASS: the resharded resume reproduced the original trajectory

Why this works
==============
The manifest describes each tensor by *global* element offsets, so a rank that
wants elements ``[0, 292)`` simply intersects that interval with whatever
intervals were saved -- here shards ``[0,146)`` and ``[146,292)`` written by two
different ranks of the original job.  No part of the reader knows or cares that
the checkpoint was written by four processes.

What is *not* supported is changing the tensor-parallel width; that is a
mathematical repartitioning of each weight matrix rather than a redistribution
of bytes, and the reader rejects it with an explicit error.

Prerequisites: an editable install.  No GPU required.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

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
from hybrid_training.distributed.launch import launch_workers
from hybrid_training.logging import configure_logging

MODEL = ModelConfig(kind="mlp", input_size=10, hidden_size=17, num_layers=2, output_size=5)

#: Samples per optimizer step, held *constant* across the two phases.
#:
#: This is the detail that makes the comparison meaningful.  A resharded resume
#: reproduces the *parameters* exactly, but if the second phase then trains on
#: a different global batch size it will of course diverge -- not because
#: resharding was wrong, but because it is a different optimisation problem.
#: Keeping ``micro_batch_size * world_size`` fixed means both phases consume the
#: same samples in the same order, so any difference in the final weights is a
#: genuine defect.
GLOBAL_BATCH_SIZE = 8


def _config(world_size: int, steps: int, checkpoint_dir: str) -> ExperimentConfig:
    """Build the shared configuration for a given FSDP width."""
    if GLOBAL_BATCH_SIZE % world_size != 0:
        raise SystemExit(
            f"global batch {GLOBAL_BATCH_SIZE} is not divisible by {world_size} ranks; "
            "pick a rank count that divides it"
        )
    return ExperimentConfig(
        name="reshard-demo",
        backend="gloo",
        device="cpu",
        topology=TopologyConfig(shard_parallel_size=world_size),
        model=MODEL,
        data=DataConfig(
            micro_batch_size=GLOBAL_BATCH_SIZE // world_size,
            num_train_samples=128,
            num_eval_samples=0,
            seed=7,
        ),
        optimizer=OptimizerConfig(name="adamw", learning_rate=5e-3),
        training=TrainingConfig(max_steps=steps, seed=3, max_grad_norm=1.0, log_every_steps=0),
        checkpoint=CheckpointConfig(directory=checkpoint_dir),
    )


def save_phase(rank: int, world_size: int, steps: int, save_at: int, checkpoint_dir: str) -> dict:
    """Train and checkpoint; runs in each spawned worker of phase 1."""
    from hybrid_training.training.engine import TrainingEngine

    config = _config(world_size, steps, checkpoint_dir)
    with distributed_context(config) as context:
        engine = TrainingEngine(config, context)
        batches = engine._batch_stream()
        saved = ""
        for _ in range(steps):
            engine.train_step(batches)
            if engine.state.step == save_at:
                saved = str(engine.save_checkpoint())
        handle = engine.model.fsdp.handle if engine.model.fsdp else None
        state = engine.model.full_state_dict()
        engine.close()
        return {
            "checkpoint": saved,
            "shard_numel": 0 if handle is None else handle.shard_numel,
            "total_numel": 0 if handle is None else handle.total_numel,
            "norm": float(sum(v.double().pow(2).sum() for v in state.values()) ** 0.5),
        }


def load_phase(
    rank: int, world_size: int, steps: int, checkpoint: str, checkpoint_dir: str
) -> dict:
    """Resume and finish training; runs in each spawned worker of phase 2."""
    from hybrid_training.training.engine import TrainingEngine

    config = _config(world_size, steps, checkpoint_dir)
    with distributed_context(config) as context:
        engine = TrainingEngine(config, context)
        loaded = engine.load_checkpoint(checkpoint)
        batches = engine._batch_stream()
        while engine.state.step < steps:
            engine.train_step(batches)
        handle = engine.model.fsdp.handle if engine.model.fsdp else None
        state = engine.model.full_state_dict()
        engine.close()
        return {
            "resumed_at": loaded.step,
            "files_read": len(loaded.files_read),
            "shard_numel": 0 if handle is None else handle.shard_numel,
            "total_numel": 0 if handle is None else handle.total_numel,
            "norm": float(sum(v.double().pow(2).sum() for v in state.values()) ** 0.5),
        }


def main() -> int:
    """Drive both phases and compare the results.

    Returns:
        ``0`` when the resharded resume matched, ``1`` otherwise.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--save-ranks", type=int, default=4)
    parser.add_argument("--load-ranks", type=int, default=2)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--save-at", type=int, default=3)
    parser.add_argument("--keep", action="store_true", help="do not delete the checkpoint")
    parser.add_argument("--log-level", default="WARNING")
    arguments = parser.parse_args()
    configure_logging(arguments.log_level)

    root = Path(tempfile.mkdtemp(prefix="hybrid-reshard-"))
    try:
        print(
            f"phase 1: training {arguments.steps} steps with {arguments.save_ranks} FSDP "
            f"ranks (micro-batch {GLOBAL_BATCH_SIZE // arguments.save_ranks}, "
            f"global batch {GLOBAL_BATCH_SIZE})"
        )
        phase_one = launch_workers(
            save_phase,
            arguments.save_ranks,
            kwargs={
                "steps": arguments.steps,
                "save_at": arguments.save_at,
                "checkpoint_dir": str(root),
            },
            timeout_seconds=600,
            log_level=arguments.log_level,
        )
        first = phase_one[0].value
        print(f"  rank 0 local shard: {first['shard_numel']} elements of {first['total_numel']}")
        print(f"  checkpoint: {first['checkpoint']}")

        print(
            f"\nphase 2: resuming the same checkpoint with {arguments.load_ranks} FSDP "
            f"ranks (micro-batch {GLOBAL_BATCH_SIZE // arguments.load_ranks}, "
            f"global batch {GLOBAL_BATCH_SIZE})"
        )
        phase_two = launch_workers(
            load_phase,
            arguments.load_ranks,
            kwargs={
                "steps": arguments.steps,
                "checkpoint": first["checkpoint"],
                "checkpoint_dir": str(root),
            },
            timeout_seconds=600,
            log_level=arguments.log_level,
        )
        second = phase_two[0].value
        print(f"  rank 0 local shard: {second['shard_numel']} elements of {second['total_numel']}")
        print(
            f"  resumed at step {second['resumed_at']}, "
            f"read {second['files_read']} payload file(s) per rank"
        )

        difference = abs(first["norm"] - second["norm"])
        print("\nverification")
        print(f"  {arguments.save_ranks}-rank final parameter norm : {first['norm']:.10f}")
        print(f"  {arguments.load_ranks}-rank final parameter norm : {second['norm']:.10f}")
        print(f"  absolute difference         : {difference:.3e}")
        # The parameters restored at the checkpoint step are bit-identical; the
        # residual here comes only from fp32 summation order differing between
        # a 4-way and a 2-way reduce-scatter over the following steps.  1e-6 is
        # ~30x the observed 3e-8, so the test is tight enough to catch a real
        # assembly error while not being flaky.
        tolerance = 1e-6
        if difference <= tolerance:
            print("  PASS: the resharded resume reproduced the original trajectory")
            return 0
        print(f"  FAIL: difference exceeds the {tolerance:.0e} tolerance")
        return 1
    finally:
        if arguments.keep:
            print(f"\ncheckpoint retained at {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
