#!/usr/bin/env python3
"""Benchmark the parallelism strategies against each other.

Usage::

    # single process, all strategies that make sense at world size 1
    python scripts/benchmark.py

    # compare DDP and FSDP at four ranks on CPU
    python scripts/benchmark.py --world-size 4 --strategies ddp fsdp

    # under torchrun, benchmark whatever topology the launcher provides
    torchrun --standalone --nproc-per-node=4 scripts/benchmark.py --in-process

    # JSON for downstream processing
    python scripts/benchmark.py --world-size 2 --json > results.json

READ THIS BEFORE BELIEVING ANY NUMBER
=====================================
These results depend on hardware, interconnect, backend, tensor sizes, the
number of warm-up iterations and what else the machine is doing.  Gloo over
loopback on a laptop is not a proxy for NCCL over NVLink; a 4-layer MLP is not
a proxy for a 70B transformer.  The purpose of this script is to make the
*shape* of the trade-offs visible -- FSDP moves more bytes and holds less
memory, accumulation trades latency for communication -- not to produce a
number anyone should quote.

Two measurement rules the script follows
========================================
1. **Warm-up before timing.**  The first iteration pays for lazy allocator
   growth, communicator setup and kernel autotuning.  It is discarded.
2. **Synchronise only at the boundaries.**  ``torch.cuda.synchronize()`` is
   called once before the timer starts and once before it stops -- never
   inside the loop.  Synchronising every iteration would serialise the
   communication against the computation and destroy exactly the overlap the
   design exists to create, so the "measurement" would report a slower
   implementation than the one actually running.  On CPU the call is skipped
   entirely; there is nothing asynchronous to wait for.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn as nn

from hybrid_training.config import (
    DDPConfig,
    FSDPConfig,
    ModelConfig,
    OptimizerConfig,
    SequenceParallelMode,
    TopologyConfig,
)
from hybrid_training.distributed.collectives import CommunicationRecorder
from hybrid_training.distributed.context import distributed_context
from hybrid_training.distributed.launch import launch_workers
from hybrid_training.logging import configure_logging
from hybrid_training.models.mlp import MLP
from hybrid_training.models.transformer import ParallelPlan, TinyTransformer
from hybrid_training.optim.sharded_optimizer import ShardedOptimizer
from hybrid_training.parallel.ddp import DistributedDataParallel
from hybrid_training.parallel.fsdp import FullyShardedDataParallel
from hybrid_training.utils.memory import (
    capture_memory,
    format_bytes,
    reset_peak_memory,
)

STRATEGIES = ("single", "ddp", "fsdp", "tensor", "sequence", "hybrid")


@dataclass
class BenchmarkResult:
    """Timings and volumes for one strategy at one world size.

    Attributes:
        strategy: Strategy name.
        world_size: Number of ranks.
        backend: Communication backend used.
        device: Device the model ran on.
        steps: Timed iterations (warm-up excluded).
        median_step_seconds: Median wall-clock time per step.
        mean_step_seconds: Mean wall-clock time per step.
        min_step_seconds: Fastest step.
        forward_seconds: Total time inside forward.
        backward_seconds: Total time inside backward.
        communication_seconds: Total time inside collectives, launch plus wait.
        communicated_bytes: Payload bytes passed to the backend.
        collective_calls: Number of collectives issued.
        local_parameter_bytes: Persistent parameter bytes on this rank.
        optimizer_state_bytes: Optimizer state bytes on this rank.
        peak_allocated_bytes: Peak device allocation (CUDA only).
        samples_per_second: Throughput over the timed steps.
    """

    strategy: str
    world_size: int
    backend: str
    device: str
    steps: int
    median_step_seconds: float
    mean_step_seconds: float
    min_step_seconds: float
    forward_seconds: float
    backward_seconds: float
    communication_seconds: float
    communicated_bytes: int
    collective_calls: int
    local_parameter_bytes: int
    optimizer_state_bytes: int
    peak_allocated_bytes: int
    samples_per_second: float
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        return asdict(self)


def _topology(strategy: str, world_size: int) -> TopologyConfig:
    """Map a strategy name onto a topology."""
    if strategy == "single" or world_size == 1:
        return TopologyConfig()
    if strategy == "ddp":
        return TopologyConfig(data_parallel_size=world_size)
    if strategy == "fsdp":
        return TopologyConfig(shard_parallel_size=world_size)
    if strategy == "tensor":
        return TopologyConfig(tensor_parallel_size=world_size)
    if strategy == "sequence":
        return TopologyConfig(
            tensor_parallel_size=world_size,
            sequence_parallel_mode=SequenceParallelMode.TENSOR_GROUP,
        )
    if strategy == "hybrid":
        if world_size % 2 != 0:
            raise SystemExit("the hybrid strategy needs an even world size")
        return TopologyConfig(data_parallel_size=world_size // 2, shard_parallel_size=2)
    raise SystemExit(f"unknown strategy {strategy!r}")


def benchmark_worker(
    rank: int,
    world_size: int,
    strategy: str = "ddp",
    steps: int = 20,
    warmup: int = 5,
    batch_size: int = 16,
    hidden_size: int = 256,
    num_layers: int = 4,
    model_kind: str = "mlp",
    sequence_length: int = 32,
    accumulation: int = 1,
    backend: str = "auto",
) -> dict[str, Any]:
    """Run one strategy and return its measurements.

    Args:
        rank: Global rank (supplied by the launcher).
        world_size: Number of processes.
        strategy: One of :data:`STRATEGIES`.
        steps: Timed iterations.
        warmup: Iterations discarded before timing starts.
        batch_size: Samples per rank per micro-step.
        hidden_size: Model width.
        num_layers: Model depth.
        model_kind: ``"mlp"`` or ``"transformer"``.
        sequence_length: Sequence length for the transformer.
        accumulation: Micro-batches per optimizer step.
        backend: Backend override.

    Returns:
        A JSON-serialisable measurement dictionary.
    """
    topology = _topology(strategy, world_size)
    with distributed_context(topology, backend=backend) as context:
        recorder = CommunicationRecorder()
        device = context.device
        notes: list[str] = []

        if model_kind == "transformer":
            config = ModelConfig(
                kind="transformer",
                vocab_size=1024,
                hidden_size=hidden_size,
                num_heads=8,
                num_layers=num_layers,
                max_sequence_length=sequence_length,
            )
            plan = ParallelPlan(
                tensor_group=context.group("tensor"),
                sequence_group=context.group("sequence_effective"),
                sequence_parallel=topology.sequence_parallel_enabled,
                vocab_parallel=context.group("tensor").size > 1,
            )
            model: nn.Module = TinyTransformer(config, plan, seed=0, device=device)
            inputs = torch.randint(
                0, config.vocab_size, (batch_size, sequence_length), device=device
            )
            targets = torch.randint(
                0, config.vocab_size, (batch_size, sequence_length), device=device
            )

            def loss_fn(module: nn.Module) -> torch.Tensor:
                logits = module(inputs)
                return nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
                )

        else:
            config = ModelConfig(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=hidden_size,
            )
            tensor_group = context.group("tensor")
            model = MLP(
                config,
                seed=0,
                device=device,
                tensor_parallel_group=tensor_group if tensor_group.size > 1 else None,
            )
            inputs = torch.randn(batch_size, hidden_size, device=device)
            targets = torch.randn(batch_size, hidden_size, device=device)

            def loss_fn(module: nn.Module) -> torch.Tensor:
                return nn.functional.mse_loss(module(inputs), targets)

        # Wrap for the data-side strategy.
        wrapped: nn.Module = model
        ddp: DistributedDataParallel | None = None
        fsdp: FullyShardedDataParallel | None = None
        if context.group("shard").size > 1:
            fsdp = FullyShardedDataParallel(
                model,
                context.group("shard"),
                FSDPConfig(),
                replica_group=(
                    context.group("data_parallel")
                    if context.group("data_parallel").size > 1
                    else None
                ),
                device=device,
                recorder=recorder,
            )
            wrapped = fsdp
        elif context.group("data_parallel").size > 1:
            ddp = DistributedDataParallel(
                model, context.group("data_parallel"), DDPConfig(), recorder=recorder
            )
            wrapped = ddp

        optimizer = ShardedOptimizer(
            [p for p in wrapped.parameters() if p.requires_grad],
            OptimizerConfig(name="adamw", learning_rate=1e-3),
            norm_group=context.group("world"),
            device=device,
            recorder=recorder,
        )

        def one_step() -> tuple[float, float]:
            """Run one optimizer step; return (forward, backward) seconds."""
            forward_total = 0.0
            backward_total = 0.0
            optimizer.zero_grad(set_to_none=True)
            for micro in range(accumulation):
                last = micro == accumulation - 1
                sync_context = (
                    (ddp.no_sync() if ddp else fsdp.no_sync())
                    if (not last and (ddp or fsdp))
                    else _null()
                )
                with sync_context:
                    start = time.perf_counter()
                    loss = loss_fn(wrapped) / accumulation
                    forward_total += time.perf_counter() - start
                    start = time.perf_counter()
                    loss.backward()
                    backward_total += time.perf_counter() - start
            if ddp is not None:
                ddp.finish_gradient_synchronization()
            if fsdp is not None:
                fsdp.finish_backward()
            optimizer.step()
            return forward_total, backward_total

        # -- warm-up: excluded from every measurement --------------------
        for _ in range(warmup):
            one_step()
        recorder.reset()
        reset_peak_memory(device)

        # -- timed region ------------------------------------------------
        # One synchronisation before the timer and one after: never inside
        # the loop, where it would serialise communication against compute.
        context.synchronize_device()
        context.barrier("world", label="benchmark-start")

        durations: list[float] = []
        forward_seconds = 0.0
        backward_seconds = 0.0
        for _ in range(steps):
            step_start = time.perf_counter()
            forward, backward = one_step()
            durations.append(time.perf_counter() - step_start)
            forward_seconds += forward
            backward_seconds += backward

        context.synchronize_device()
        # -- end of timed region -----------------------------------------

        total = recorder.total()
        memory = capture_memory(device)
        parameter_bytes = sum(
            p.numel() * p.element_size() for p in wrapped.parameters() if p.requires_grad
        )
        if device.type != "cuda":
            notes.append(
                "peak_allocated_bytes is 0 on CPU: PyTorch keeps no allocator "
                "statistics there, so only the analytical estimate is available"
            )
        if context.backend == "gloo":
            notes.append(
                "Gloo over loopback: latency dominates and bandwidth is far below "
                "any real interconnect; treat the timings as relative only"
            )

        samples = batch_size * accumulation * steps * context.group("dp_shard").size
        result = BenchmarkResult(
            strategy=strategy,
            world_size=world_size,
            backend=context.backend,
            device=str(device),
            steps=steps,
            median_step_seconds=statistics.median(durations),
            mean_step_seconds=statistics.fmean(durations),
            min_step_seconds=min(durations),
            forward_seconds=forward_seconds,
            backward_seconds=backward_seconds,
            communication_seconds=total.seconds + total.wait_seconds,
            communicated_bytes=total.bytes,
            collective_calls=total.calls,
            local_parameter_bytes=parameter_bytes,
            optimizer_state_bytes=optimizer.state_bytes(),
            peak_allocated_bytes=memory.peak_allocated_bytes,
            samples_per_second=samples / sum(durations) if sum(durations) else 0.0,
            notes=notes,
        )
        if ddp is not None:
            ddp.teardown()
        return result.as_dict()


class _null:
    """No-op context manager used when there is nothing to suppress."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc_info: object) -> None:
        return None


def render_table(results: list[dict[str, Any]]) -> str:
    """Format results as a fixed-width table.

    Args:
        results: One dictionary per strategy, as returned by
            :func:`benchmark_worker`.

    Returns:
        The rendered table.
    """
    header = (
        f"{'strategy':<12}{'ranks':>6}{'backend':>9}{'step ms':>10}"
        f"{'fwd ms':>9}{'bwd ms':>9}{'comm ms':>9}"
        f"{'comm MiB':>10}{'calls':>7}{'params':>11}{'opt state':>11}{'samp/s':>10}"
    )
    lines = [header, "-" * len(header)]
    for entry in results:
        lines.append(
            f"{entry['strategy']:<12}"
            f"{entry['world_size']:>6}"
            f"{entry['backend']:>9}"
            f"{entry['median_step_seconds'] * 1000:>10.2f}"
            f"{entry['forward_seconds'] / entry['steps'] * 1000:>9.2f}"
            f"{entry['backward_seconds'] / entry['steps'] * 1000:>9.2f}"
            f"{entry['communication_seconds'] / entry['steps'] * 1000:>9.2f}"
            f"{entry['communicated_bytes'] / 1048576 / entry['steps']:>10.3f}"
            f"{entry['collective_calls'] // entry['steps']:>7}"
            f"{format_bytes(entry['local_parameter_bytes']):>11}"
            f"{format_bytes(entry['optimizer_state_bytes']):>11}"
            f"{entry['samples_per_second']:>10.1f}"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Benchmark the parallelism strategies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--world-size", type=int, default=1, help="ranks to spawn")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        choices=STRATEGIES,
        help="strategies to run (default: those valid at this world size)",
    )
    parser.add_argument("--steps", type=int, default=20, help="timed iterations")
    parser.add_argument("--warmup", type=int, default=5, help="discarded iterations")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--model", default="mlp", choices=["mlp", "transformer"])
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--backend", default="auto", choices=["auto", "gloo", "nccl"])
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    parser.add_argument("--log-level", default="WARNING")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark sweep.

    Returns:
        Process exit code.
    """
    arguments = build_parser().parse_args(argv)
    configure_logging(arguments.log_level)

    strategies = arguments.strategies
    if strategies is None:
        if arguments.world_size == 1:
            strategies = ["single"]
        elif arguments.world_size % 2 == 0:
            strategies = ["ddp", "fsdp", "hybrid"]
        else:
            strategies = ["ddp", "fsdp"]

    results: list[dict[str, Any]] = []
    for strategy in strategies:
        if strategy in {"tensor", "sequence"} and arguments.model != "transformer":
            print(
                f"skipping {strategy!r}: it needs --model transformer",
                file=sys.stderr,
            )
            continue
        outcomes = launch_workers(
            benchmark_worker,
            arguments.world_size,
            kwargs={
                "strategy": strategy,
                "steps": arguments.steps,
                "warmup": arguments.warmup,
                "batch_size": arguments.batch_size,
                "hidden_size": arguments.hidden_size,
                "num_layers": arguments.num_layers,
                "model_kind": arguments.model,
                "sequence_length": arguments.sequence_length,
                "accumulation": arguments.accumulation,
                "backend": arguments.backend,
            },
            timeout_seconds=900.0,
            log_level=arguments.log_level,
        )
        # Report rank 0; the others are printed only in the JSON form.
        results.append(outcomes[0].value)

    if arguments.json:
        print(json.dumps(results, indent=2, sort_keys=True))
        return 0

    print()
    print("=" * 118)
    print("BENCHMARK RESULTS -- these numbers depend on hardware, interconnect,")
    print("backend, tensor sizes and warm-up.  They show the SHAPE of the")
    print("trade-offs, not portable performance figures.  Do not quote them.")
    print("=" * 118)
    print(
        f"model={arguments.model} hidden={arguments.hidden_size} "
        f"layers={arguments.num_layers} batch={arguments.batch_size} "
        f"accumulation={arguments.accumulation} steps={arguments.steps} "
        f"warmup={arguments.warmup}"
    )
    print()
    print(render_table(results))
    print()
    for note in dict.fromkeys(n for entry in results for n in entry["notes"]):
        print(f"note: {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
