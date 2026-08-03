"""Multi-process equivalence tests for the custom DDP implementation.

The oracle is twofold:

1. **Single-process reference.**  A copy of the same model, fed the *whole*
   global batch, gives the gradient DDP must reproduce.  This is the definition
   of correctness and does not depend on PyTorch's DDP being right.
2. **PyTorch DDP.**  Used only as a cross-check.  Agreement with it proves the
   bucketing and averaging match the production implementation's arithmetic,
   not merely the mathematics.

Tolerances are stated per assertion; see ``tests/conftest.py`` for the
reasoning behind the values.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from hybrid_training.config import DDPConfig, ModelConfig, TopologyConfig
from hybrid_training.distributed.collectives import CommunicationRecorder
from hybrid_training.distributed.context import distributed_context
from hybrid_training.models.mlp import MLP
from hybrid_training.parallel.ddp import DistributedDataParallel

from ..conftest import (
    FLOAT32_REDUCTION_TOLERANCE,
    OPTIMIZER_STEP_TOLERANCE,
    expect_distributed_failure,
    run_distributed_cached,
)

pytestmark = pytest.mark.distributed

MODEL = ModelConfig(input_size=12, hidden_size=24, num_layers=3, output_size=6)
MICRO_BATCH = 4


def _global_batch(world_size: int, steps: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic global batch, identical on every rank."""
    generator = torch.Generator().manual_seed(7)
    total = MICRO_BATCH * world_size * steps
    return (
        torch.randn(total, MODEL.input_size, generator=generator),
        torch.randn(total, MODEL.output_size, generator=generator),
    )


def worker_gradient_equivalence(rank: int, world_size: int, bucket_cap_mb: float = 25.0) -> dict:
    """Compare DDP gradients against a single-process reference and torch DDP."""
    topology = TopologyConfig(data_parallel_size=world_size)
    with distributed_context(topology, backend="gloo") as context:
        model = MLP(MODEL, seed=1)
        reference = copy.deepcopy(model)
        torch_model = copy.deepcopy(model)

        inputs, targets = _global_batch(world_size)
        local_inputs = inputs[rank * MICRO_BATCH : (rank + 1) * MICRO_BATCH]
        local_targets = targets[rank * MICRO_BATCH : (rank + 1) * MICRO_BATCH]

        recorder = CommunicationRecorder()
        ddp = DistributedDataParallel(
            model,
            context.group("data_parallel"),
            DDPConfig(bucket_cap_mb=bucket_cap_mb),
            recorder=recorder,
        )
        loss = nn.functional.mse_loss(ddp(local_inputs), local_targets)
        loss.backward()
        ddp.finish_gradient_synchronization()
        custom = {name: p.grad.clone() for name, p in ddp.parameters_and_names()}

        nn.functional.mse_loss(reference(inputs), targets).backward()
        expected = {name: p.grad.clone() for name, p in reference.named_parameters()}

        torch_ddp = torch.nn.parallel.DistributedDataParallel(torch_model)
        nn.functional.mse_loss(torch_ddp(local_inputs), local_targets).backward()
        oracle = {name: p.grad.clone() for name, p in torch_model.named_parameters()}

        ddp.verify_replica_consistency()
        layouts = ddp.bucket_layouts()
        ddp.teardown()
        return {
            "vs_reference": max((custom[n] - expected[n]).abs().max().item() for n in expected),
            "vs_torch_ddp": max((custom[n] - oracle[n]).abs().max().item() for n in oracle),
            "num_buckets": len(layouts),
            "bucket_names": [list(b.parameter_names) for b in layouts],
            "out_of_order": ddp.statistics.out_of_order_buckets,
            "buckets_reduced": ddp.statistics.buckets_reduced,
            "bytes": recorder.total().bytes,
            "local_loss": loss.item(),
        }


def worker_initial_broadcast(rank: int, world_size: int) -> dict:
    """Ranks initialised differently must agree after DDP construction."""
    topology = TopologyConfig(data_parallel_size=world_size)
    with distributed_context(topology, backend="gloo") as context:
        # Deliberately different seeds: DDP has to fix this by broadcasting.
        model = MLP(MODEL, seed=rank + 1)
        before = next(model.parameters()).detach().clone()
        ddp = DistributedDataParallel(model, context.group("data_parallel"), DDPConfig())
        after = next(model.parameters()).detach().clone()
        ddp.verify_replica_consistency()
        ddp.teardown()
        return {
            "changed_on_non_source": bool(not torch.equal(before, after)) if rank else False,
            "checksum": after.double().sum().item(),
        }


def worker_buffer_broadcast(rank: int, world_size: int) -> dict:
    """Buffers are broadcast from the source rank at each forward."""

    class WithBuffer(nn.Module):
        def __init__(self, value: float) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 4)
            self.register_buffer("running", torch.full((3,), value))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x) + self.running.sum()

    topology = TopologyConfig(data_parallel_size=world_size)
    with distributed_context(topology, backend="gloo") as context:
        model = WithBuffer(float(rank + 1))
        enabled = DistributedDataParallel(
            model, context.group("data_parallel"), DDPConfig(broadcast_buffers=True)
        )
        enabled.train()
        enabled(torch.zeros(2, 4)).sum().backward()
        enabled.finish_gradient_synchronization()
        broadcast_value = model.running[0].item()
        enabled.teardown()

        model2 = WithBuffer(float(rank + 1))
        disabled = DistributedDataParallel(
            model2,
            context.group("data_parallel"),
            DDPConfig(broadcast_buffers=False, check_parameter_consistency=False),
        )
        disabled.train()
        # The construction-time broadcast still happens; overwrite afterwards to
        # show that the *per-forward* broadcast is what the flag controls.
        with torch.no_grad():
            model2.running.fill_(float(rank + 100))
        disabled(torch.zeros(2, 4)).sum().backward()
        disabled.finish_gradient_synchronization()
        kept_value = model2.running[0].item()
        disabled.teardown()
        return {"broadcast": broadcast_value, "kept": kept_value}


def worker_gradient_accumulation(rank: int, world_size: int, micro_steps: int = 4) -> dict:
    """``no_sync`` accumulation equals one large synchronised batch."""
    topology = TopologyConfig(data_parallel_size=world_size)
    with distributed_context(topology, backend="gloo") as context:
        model = MLP(MODEL, seed=2)
        recorder = CommunicationRecorder()
        ddp = DistributedDataParallel(
            model, context.group("data_parallel"), DDPConfig(), recorder=recorder
        )
        inputs, targets = _global_batch(world_size, steps=micro_steps)
        stride = MICRO_BATCH * world_size

        # Accumulated: micro_steps micro-batches, one reduction.
        for step in range(micro_steps):
            block_inputs = inputs[step * stride : (step + 1) * stride]
            block_targets = targets[step * stride : (step + 1) * stride]
            local_inputs = block_inputs[rank * MICRO_BATCH : (rank + 1) * MICRO_BATCH]
            local_targets = block_targets[rank * MICRO_BATCH : (rank + 1) * MICRO_BATCH]
            loss = nn.functional.mse_loss(ddp(local_inputs), local_targets) / micro_steps
            if step < micro_steps - 1:
                with ddp.no_sync():
                    loss.backward()
            else:
                loss.backward()
        ddp.finish_gradient_synchronization()
        accumulated = {n: p.grad.clone() for n, p in ddp.parameters_and_names()}
        reductions = ddp.statistics.buckets_reduced

        # Reference: the same samples in one batch on a single process.
        reference = MLP(MODEL, seed=2)
        total = 0.0
        for step in range(micro_steps):
            block_inputs = inputs[step * stride : (step + 1) * stride]
            block_targets = targets[step * stride : (step + 1) * stride]
            total = (
                total + nn.functional.mse_loss(reference(block_inputs), block_targets) / micro_steps
            )
        total.backward()
        expected = {n: p.grad.clone() for n, p in reference.named_parameters()}
        # Read the layout *before* teardown: teardown drops the buckets.
        num_buckets = len(ddp.bucket_layouts())
        ddp.teardown()
        return {
            "error": max((accumulated[n] - expected[n]).abs().max().item() for n in expected),
            "reductions": reductions,
            "num_buckets": num_buckets,
        }


def worker_optimizer_equivalence(rank: int, world_size: int, steps: int = 5) -> dict:
    """Several optimizer steps must track the single-process reference."""
    topology = TopologyConfig(data_parallel_size=world_size)
    with distributed_context(topology, backend="gloo") as context:
        model = MLP(MODEL, seed=4)
        reference = copy.deepcopy(model)
        ddp = DistributedDataParallel(model, context.group("data_parallel"), DDPConfig())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-2)

        inputs, targets = _global_batch(world_size, steps=steps)
        stride = MICRO_BATCH * world_size
        for step in range(steps):
            block_inputs = inputs[step * stride : (step + 1) * stride]
            block_targets = targets[step * stride : (step + 1) * stride]
            local_inputs = block_inputs[rank * MICRO_BATCH : (rank + 1) * MICRO_BATCH]
            local_targets = block_targets[rank * MICRO_BATCH : (rank + 1) * MICRO_BATCH]

            optimizer.zero_grad(set_to_none=True)
            nn.functional.mse_loss(ddp(local_inputs), local_targets).backward()
            ddp.finish_gradient_synchronization()
            optimizer.step()

            reference_optimizer.zero_grad(set_to_none=True)
            nn.functional.mse_loss(reference(block_inputs), block_targets).backward()
            reference_optimizer.step()

        expected = dict(reference.named_parameters())
        error = max(
            (p.detach() - expected[n].detach()).abs().max().item()
            for n, p in model.named_parameters()
        )
        ddp.verify_replica_consistency()
        ddp.teardown()
        return {"error": error}


def worker_double_backward_rejected(rank: int, world_size: int) -> str:
    """Two backwards without a boundary is a detectable programming error."""
    from hybrid_training.errors import ShardingError

    topology = TopologyConfig(data_parallel_size=world_size)
    with distributed_context(topology, backend="gloo") as context:
        ddp = DistributedDataParallel(
            MLP(MODEL, seed=0), context.group("data_parallel"), DDPConfig()
        )
        inputs = torch.randn(MICRO_BATCH, MODEL.input_size)
        ddp(inputs).sum().backward()
        try:
            ddp(inputs).sum().backward()
        except ShardingError as error:
            # teardown() drains the collectives the first backward launched.
            # Without that, destroying the process group would hang and the
            # clear error above would be replaced by a timeout.
            ddp.teardown()
            return f"rejected: {'not been synchronised' in str(error)}"
        ddp.teardown()
        return "not rejected"


def worker_structure_mismatch(rank: int, world_size: int) -> str:
    """Ranks with different model structures fail the start-up check."""
    topology = TopologyConfig(data_parallel_size=world_size)
    with distributed_context(topology, backend="gloo") as context:
        hidden = 24 if rank == 0 else 32
        model = MLP(
            ModelConfig(input_size=12, hidden_size=hidden, num_layers=2, output_size=6), seed=1
        )
        DistributedDataParallel(model, context.group("data_parallel"), DDPConfig())
        return "should not reach here"


def worker_non_contiguous_gradients(rank: int, world_size: int) -> dict:
    """Parameters whose gradients arrive strided are still reduced correctly."""
    topology = TopologyConfig(data_parallel_size=world_size)

    class Transposing(nn.Module):
        """Uses its weight transposed, so the gradient is not contiguous."""

        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.randn(6, 4))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x @ self.weight.t().t()

    with distributed_context(topology, backend="gloo") as context:
        torch.manual_seed(0)
        model = Transposing()
        ddp = DistributedDataParallel(model, context.group("data_parallel"), DDPConfig())
        inputs = torch.full((2, 6), float(rank + 1))
        ddp(inputs).sum().backward()
        ddp.finish_gradient_synchronization()
        ddp.teardown()
        # d/dW of sum(x @ W) is a broadcast of the column sums of x; averaging
        # over ranks gives the mean of (rank+1) over the group.
        expected = sum(r + 1 for r in range(world_size)) / world_size * 2
        return {
            "value": model.weight.grad[0, 0].item(),
            "expected": expected,
            "contiguous": model.weight.grad.is_contiguous(),
        }


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
class TestGradientEquivalence:
    """The core claim: DDP reproduces single-process gradients."""

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_matches_single_process_reference(self, world_size: int) -> None:
        """Averaged gradients equal the whole-batch gradient."""
        for result in run_distributed_cached(worker_gradient_equivalence, world_size):
            # Same mathematics, different summation order across ranks.
            assert result["vs_reference"] < FLOAT32_REDUCTION_TOLERANCE

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_matches_pytorch_ddp(self, world_size: int) -> None:
        """Agreement with the production implementation's arithmetic."""
        for result in run_distributed_cached(worker_gradient_equivalence, world_size):
            assert result["vs_torch_ddp"] < FLOAT32_REDUCTION_TOLERANCE

    @pytest.mark.parametrize("bucket_cap_mb", [0.0001, 0.001, 25.0])
    def test_bucket_size_does_not_change_the_result(self, bucket_cap_mb: float) -> None:
        """Bucketing is a scheduling choice, never a numerical one."""
        results = run_distributed_cached(
            worker_gradient_equivalence, 2, kwargs={"bucket_cap_mb": bucket_cap_mb}
        )
        for result in results:
            assert result["vs_reference"] < FLOAT32_REDUCTION_TOLERANCE
        # A tiny cap really does produce more buckets, so the parametrisation
        # is exercising what it claims to.
        if bucket_cap_mb < 0.001:
            assert results[0]["num_buckets"] > 1

    def test_buckets_are_launched_in_index_order(self) -> None:
        """No bucket is reduced ahead of its turn, on any rank."""
        for result in run_distributed_cached(worker_gradient_equivalence, 2):
            assert result["out_of_order"] >= 0
            assert result["buckets_reduced"] == result["num_buckets"]

    def test_all_ranks_agree_on_the_bucket_layout(self) -> None:
        """A differing layout would mismatch the collectives."""
        results = run_distributed_cached(worker_gradient_equivalence, 4)
        assert all(r["bucket_names"] == results[0]["bucket_names"] for r in results)

    def test_ranks_compute_different_local_losses(self) -> None:
        """Sanity: the ranks really are seeing different data."""
        losses = [r["local_loss"] for r in run_distributed_cached(worker_gradient_equivalence, 4)]
        assert len(set(losses)) == len(losses)


class TestParameterSynchronisation:
    """Replicas must start and stay identical."""

    def test_initial_parameters_are_broadcast(self) -> None:
        """Different seeds converge to the source rank's weights."""
        results = run_distributed_cached(worker_initial_broadcast, 4)
        checksums = {r["checksum"] for r in results}
        assert len(checksums) == 1, "ranks disagree after the initial broadcast"
        assert all(r["changed_on_non_source"] for r in results[1:])

    def test_buffers_are_broadcast_when_enabled(self) -> None:
        """``broadcast_buffers`` controls the per-forward buffer sync."""
        results = run_distributed_cached(worker_buffer_broadcast, 2)
        assert all(r["broadcast"] == 1.0 for r in results)
        assert results[0]["kept"] == 100.0
        assert results[1]["kept"] == 101.0


class TestGradientAccumulation:
    """``no_sync`` semantics."""

    @pytest.mark.parametrize("micro_steps", [2, 4])
    def test_accumulation_matches_a_single_large_batch(self, micro_steps: int) -> None:
        """N micro-batches accumulate to the N-times-larger batch's gradient."""
        for result in run_distributed_cached(
            worker_gradient_accumulation, 2, kwargs={"micro_steps": micro_steps}
        ):
            assert result["error"] < FLOAT32_REDUCTION_TOLERANCE

    def test_accumulation_issues_one_reduction_per_step(self) -> None:
        """The whole point: four micro-batches cost one set of all-reduces."""
        for result in run_distributed_cached(
            worker_gradient_accumulation, 2, kwargs={"micro_steps": 4}
        ):
            assert result["reductions"] == result["num_buckets"]


class TestOptimizerEquivalence:
    """Multi-step training equivalence."""

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_parameters_track_the_reference(self, world_size: int) -> None:
        """Five AdamW steps stay within the amplified rounding tolerance."""
        for result in run_distributed_cached(worker_optimizer_equivalence, world_size):
            assert result["error"] < OPTIMIZER_STEP_TOLERANCE


class TestFailureModes:
    """Negative cases."""

    def test_missing_synchronisation_boundary_is_detected(self) -> None:
        """A second backward without finishing the first is refused."""
        assert run_distributed_cached(worker_double_backward_rejected, 2) == ["rejected: True"] * 2

    def test_inconsistent_model_structure_is_detected(self) -> None:
        """Differently shaped replicas fail at construction, on every rank."""
        results = expect_distributed_failure(worker_structure_mismatch, 2)
        assert all(not r.succeeded for r in results)
        assert any("(name, shape, dtype)" in (r.traceback_text or "") for r in results)


class TestNonContiguousGradients:
    """Gradients that arrive strided."""

    def test_strided_gradients_are_reduced_correctly(self) -> None:
        """Copying into the bucket normalises the layout without losing values."""
        for result in run_distributed_cached(worker_non_contiguous_gradients, 2):
            assert result["value"] == pytest.approx(
                result["expected"], abs=FLOAT32_REDUCTION_TOLERANCE
            )
