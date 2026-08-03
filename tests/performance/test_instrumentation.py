"""Performance-flavoured tests that assert on *invariants*, never on timings.

A test that asserts "the overlapped version is faster" is a flake generator:
on a shared CI machine the wall-clock ordering of two similar workloads is not
reproducible.  What *is* reproducible is the communication volume, the number
of collectives, and the memory each strategy holds -- and those are the
quantities that actually distinguish the strategies.

So these tests assert:

* how many bytes each strategy moves, and how that scales with the world size;
* how many collectives a step issues;
* that FSDP's resident memory really is a fraction of DDP's;
* that the analytical memory model agrees with the measured shard sizes.

Timings are still *recorded* (and printed on failure) so a regression in speed
is visible, but never asserted on.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from hybrid_training.config import (
    DDPConfig,
    FSDPConfig,
    ModelConfig,
    OptimizerConfig,
    TopologyConfig,
)
from hybrid_training.distributed.collectives import CommunicationRecorder
from hybrid_training.distributed.context import distributed_context
from hybrid_training.models.mlp import MLP
from hybrid_training.optim.sharded_optimizer import ShardedOptimizer
from hybrid_training.parallel.ddp import DistributedDataParallel
from hybrid_training.parallel.fsdp import FullyShardedDataParallel
from hybrid_training.utils.memory import estimate_training_memory

from ..conftest import run_distributed_cached

pytestmark = [pytest.mark.performance, pytest.mark.distributed]

MODEL = ModelConfig(input_size=32, hidden_size=64, num_layers=4, output_size=8)
MICRO_BATCH = 8
STEPS = 3


def _parameter_count() -> int:
    """Total parameters in the benchmark model."""
    return sum(p.numel() for p in MLP(MODEL, seed=0).parameters())


def worker_ddp_volume(rank: int, world_size: int, bucket_cap_mb: float = 25.0) -> dict:
    """Measure DDP's communication volume over a few steps."""
    topology = TopologyConfig(data_parallel_size=world_size)
    with distributed_context(topology, backend="gloo") as context:
        recorder = CommunicationRecorder()
        model = MLP(MODEL, seed=0)
        parameters = sum(p.numel() for p in model.parameters())
        ddp = DistributedDataParallel(
            model,
            context.group("data_parallel"),
            DDPConfig(bucket_cap_mb=bucket_cap_mb),
            recorder=recorder,
        )
        recorder.reset()  # exclude the construction-time broadcasts
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for _ in range(STEPS):
            optimizer.zero_grad(set_to_none=True)
            inputs = torch.randn(MICRO_BATCH, MODEL.input_size)
            targets = torch.randn(MICRO_BATCH, MODEL.output_size)
            nn.functional.mse_loss(ddp(inputs), targets).backward()
            ddp.finish_gradient_synchronization()
            optimizer.step()
        total = recorder.total()
        by_operation = {k: v.calls for k, v in recorder.by_operation.items()}
        parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        # Read the layout *before* teardown: teardown drops the buckets.
        num_buckets = len(ddp.bucket_layouts())
        ddp.teardown()
        return {
            "parameters": parameters,
            "bytes": total.bytes,
            "calls": total.calls,
            "by_operation": by_operation,
            "buckets": num_buckets,
            "resident_parameter_bytes": parameter_bytes,
            "seconds": total.seconds + total.wait_seconds,
        }


def worker_fsdp_volume(rank: int, world_size: int, reshard: bool = True) -> dict:
    """Measure FSDP's communication volume and resident memory."""
    topology = TopologyConfig(shard_parallel_size=world_size)
    with distributed_context(topology, backend="gloo") as context:
        recorder = CommunicationRecorder()
        model = MLP(MODEL, seed=0)
        parameters = sum(p.numel() for p in model.parameters())
        fsdp = FullyShardedDataParallel(
            model,
            context.group("shard"),
            FSDPConfig(reshard_after_forward=reshard),
            recorder=recorder,
        )
        recorder.reset()
        optimizer = ShardedOptimizer(
            fsdp.parameters(),
            OptimizerConfig(name="adamw", learning_rate=0.01),
            norm_group=context.group("world"),
            device=context.device,
        )
        for _ in range(STEPS):
            optimizer.zero_grad(set_to_none=True)
            inputs = torch.randn(MICRO_BATCH, MODEL.input_size)
            targets = torch.randn(MICRO_BATCH, MODEL.output_size)
            nn.functional.mse_loss(fsdp(inputs), targets).backward()
            fsdp.finish_backward()
            optimizer.step()
        total = recorder.total()
        memory = fsdp.memory_summary()
        return {
            "parameters": parameters,
            "bytes": total.bytes,
            "calls": total.calls,
            "by_operation": {k: v.calls for k, v in recorder.by_operation.items()},
            "resident_parameter_bytes": memory["shard_bytes"],
            "gradient_bytes": memory["grad_shard_bytes"],
            "optimizer_state_bytes": optimizer.state_bytes(),
            "full_bytes_when_idle": memory["full_bytes"],
            "seconds": total.seconds + total.wait_seconds,
        }


def worker_accumulation_volume(rank: int, world_size: int, micro_steps: int = 4) -> dict:
    """Compare communication with and without ``no_sync`` accumulation."""
    topology = TopologyConfig(data_parallel_size=world_size)
    with distributed_context(topology, backend="gloo") as context:
        results = {}
        for use_no_sync in (False, True):
            recorder = CommunicationRecorder()
            ddp = DistributedDataParallel(
                MLP(MODEL, seed=0),
                context.group("data_parallel"),
                DDPConfig(),
                recorder=recorder,
            )
            recorder.reset()
            for step in range(micro_steps):
                inputs = torch.randn(MICRO_BATCH, MODEL.input_size)
                targets = torch.randn(MICRO_BATCH, MODEL.output_size)
                loss = nn.functional.mse_loss(ddp(inputs), targets) / micro_steps
                if use_no_sync and step < micro_steps - 1:
                    with ddp.no_sync():
                        loss.backward()
                else:
                    loss.backward()
                    ddp.finish_gradient_synchronization()
                    for _, param in ddp.parameters_and_names():
                        param.grad = None if step == micro_steps - 1 else param.grad
            ddp.finish_gradient_synchronization()
            results["no_sync" if use_no_sync else "every_step"] = recorder.total().bytes
            ddp.teardown()
        return results


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
class TestCommunicationVolume:
    """Bytes on the wire, which is deterministic and hardware independent."""

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_ddp_moves_one_gradient_per_step(self, world_size: int) -> None:
        """DDP all-reduces exactly the parameter bytes, once per step."""
        for result in run_distributed_cached(worker_ddp_volume, world_size):
            expected = result["parameters"] * 4 * STEPS
            # Bucketing pads nothing, so the volume is exact.
            assert result["bytes"] == expected
            assert set(result["by_operation"]) == {"all_reduce/data_parallel"}
            assert result["by_operation"]["all_reduce/data_parallel"] == (STEPS * result["buckets"])

    def test_ddp_volume_is_independent_of_the_world_size(self) -> None:
        """Each rank sends the whole gradient regardless of how many peers."""
        two = run_distributed_cached(worker_ddp_volume, 2)[0]
        four = run_distributed_cached(worker_ddp_volume, 4)[0]
        assert two["bytes"] == four["bytes"]

    @pytest.mark.parametrize("bucket_cap_mb", [0.001, 25.0])
    def test_bucket_size_changes_call_count_not_volume(self, bucket_cap_mb: float) -> None:
        """Smaller buckets mean more, smaller collectives -- same total bytes."""
        result = run_distributed_cached(
            worker_ddp_volume, 2, kwargs={"bucket_cap_mb": bucket_cap_mb}
        )[0]
        assert result["bytes"] == result["parameters"] * 4 * STEPS
        if bucket_cap_mb < 0.01:
            assert result["buckets"] > 1

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_fsdp_moves_parameters_and_gradients(self, world_size: int) -> None:
        """FSDP all-gathers parameters and reduce-scatters gradients."""
        for result in run_distributed_cached(worker_fsdp_volume, world_size):
            operations = set(result["by_operation"])
            assert "all_gather/shard" in operations
            assert "reduce_scatter/shard" in operations
            # One all-gather in forward, one in backward (reshard is on).
            assert result["by_operation"]["all_gather/shard"] == 2 * STEPS
            assert result["by_operation"]["reduce_scatter/shard"] == STEPS

    def test_reshard_after_forward_costs_an_extra_all_gather(self) -> None:
        """The memory/communication trade-off is visible in the counters."""
        with_reshard = run_distributed_cached(worker_fsdp_volume, 2, kwargs={"reshard": True})[0]
        without = run_distributed_cached(worker_fsdp_volume, 2, kwargs={"reshard": False})[0]
        assert with_reshard["by_operation"]["all_gather/shard"] == 2 * STEPS
        assert without["by_operation"]["all_gather/shard"] == STEPS
        assert with_reshard["bytes"] > without["bytes"]

    def test_accumulation_reduces_communication(self) -> None:
        """``no_sync`` turns N reductions into one, for the same computation."""
        for result in run_distributed_cached(
            worker_accumulation_volume, 2, kwargs={"micro_steps": 4}
        ):
            assert result["no_sync"] < result["every_step"]
            assert result["every_step"] == pytest.approx(4 * result["no_sync"], rel=0.01)


class TestMemoryScaling:
    """Resident memory per rank, which is the point of sharding."""

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_fsdp_holds_a_fraction_of_the_parameters(self, world_size: int) -> None:
        """Persistent parameter bytes scale as ``1/G``."""
        ddp = run_distributed_cached(worker_ddp_volume, world_size)[0]
        fsdp = run_distributed_cached(worker_fsdp_volume, world_size)[0]
        ratio = fsdp["resident_parameter_bytes"] / ddp["resident_parameter_bytes"]
        # Padding makes the ratio slightly above 1/G; allow 5% slack.
        assert ratio <= 1.0 / world_size * 1.05

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_optimizer_state_scales_with_the_shard(self, world_size: int) -> None:
        """AdamW's two buffers are sized to the shard, not the model."""
        for result in run_distributed_cached(worker_fsdp_volume, world_size):
            assert result["optimizer_state_bytes"] == pytest.approx(
                2 * result["resident_parameter_bytes"], rel=0.01
            )

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_gradients_are_sharded_too(self, world_size: int) -> None:
        """The gradient held between steps is one shard, not the whole model."""
        for result in run_distributed_cached(worker_fsdp_volume, world_size):
            assert result["gradient_bytes"] == result["resident_parameter_bytes"]

    def test_full_parameters_are_freed_between_steps(self) -> None:
        """With reshard-after-forward nothing full is resident when idle."""
        for result in run_distributed_cached(worker_fsdp_volume, 2, kwargs={"reshard": True}):
            assert result["full_bytes_when_idle"] == 0

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_analytical_model_matches_the_measurement(self, world_size: int) -> None:
        """The closed-form estimate predicts what the code actually holds."""
        measured = run_distributed_cached(worker_fsdp_volume, world_size)[0]
        estimate = estimate_training_memory(
            measured["parameters"], shard_group_size=world_size, optimizer="adamw"
        )
        assert measured["resident_parameter_bytes"] == pytest.approx(
            estimate.persistent_parameters, rel=0.02
        )
        assert measured["optimizer_state_bytes"] == pytest.approx(
            estimate.optimizer_state, rel=0.02
        )


def test_timings_are_recorded_but_not_asserted_on() -> None:
    """Timings exist for diagnosis; the suite never depends on their values.

    Asserting on wall-clock time in a shared CI environment produces flaky
    failures that teach nothing.  This test only checks the instrumentation is
    wired up, which is what the benchmark script relies on.
    """
    result = run_distributed_cached(worker_ddp_volume, 2)[0]
    assert result["seconds"] >= 0.0
    assert result["calls"] > 0
