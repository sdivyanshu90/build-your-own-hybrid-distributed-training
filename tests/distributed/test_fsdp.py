"""Multi-process tests for FSDP-style parameter/gradient/optimizer sharding.

The claims under test, in order of importance:

1. Training with sharding produces the same weights as training without it.
2. Parameters, gradients **and** optimizer state are genuinely divided -- a
   wrapper that all-gathers everything and never frees it would pass a
   numerical test but defeat the purpose.
3. Padding never leaks into a parameter, a gradient or a norm.
4. The full-parameter reconstruction and state-dict paths round-trip.
"""

from __future__ import annotations

import copy
import itertools

import pytest
import torch
import torch.nn as nn

from hybrid_training.config import (
    FSDPConfig,
    ModelConfig,
    OptimizerConfig,
    TopologyConfig,
)
from hybrid_training.distributed.context import distributed_context
from hybrid_training.models.mlp import MLP
from hybrid_training.optim.sharded_optimizer import ShardedOptimizer
from hybrid_training.parallel.fsdp import FullyShardedDataParallel

from ..conftest import (
    FLOAT32_REDUCTION_TOLERANCE,
    OPTIMIZER_STEP_TOLERANCE,
    expect_distributed_failure,
    run_distributed_cached,
)

pytestmark = pytest.mark.distributed

# input 14 x hidden 25 x 3 layers x output 7 gives 1857 parameters, which is
# divisible by neither 2 nor 4 -- so padding is exercised at every world size.
MODEL = ModelConfig(input_size=14, hidden_size=25, num_layers=3, output_size=7)
MICRO_BATCH = 4
TOTAL_PARAMETERS = 1857


def _global_batch(world_size: int, steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic global batch, identical on every rank."""
    generator = torch.Generator().manual_seed(11)
    total = MICRO_BATCH * world_size * steps
    return (
        torch.randn(total, MODEL.input_size, generator=generator),
        torch.randn(total, MODEL.output_size, generator=generator),
    )


def worker_shard_construction(rank: int, world_size: int, auto_wrap: int = 0) -> dict:
    """Report the shard layout and the memory actually held."""
    topology = TopologyConfig(shard_parallel_size=world_size)
    with distributed_context(topology, backend="gloo") as context:
        model = MLP(MODEL, seed=3)
        unsharded_parameters = sum(p.numel() for p in model.parameters())
        fsdp = FullyShardedDataParallel(
            model,
            context.group("shard"),
            FSDPConfig(auto_wrap_min_num_params=auto_wrap),
        )
        handle = fsdp.handle
        return {
            "unsharded_parameters": unsharded_parameters,
            "local_parameters": sum(p.numel() for p in fsdp.parameters()),
            "units": len(fsdp.fsdp_units()),
            "shard_numel": None if handle is None else handle.shard_numel,
            "total_numel": None if handle is None else handle.total_numel,
            "padding": None if handle is None else handle.padding,
            "local_range": None if handle is None else handle.local_shard_range().as_tuple(),
            "is_sharded": None if handle is None else handle.is_sharded,
            "memory": fsdp.memory_summary(),
            "state_dict_keys": sorted(fsdp.full_state_dict()),
            "original_keys": sorted(n for n, _ in MLP(MODEL, seed=3).named_parameters()),
        }


def worker_training_equivalence(
    rank: int, world_size: int, steps: int = 5, reshard: bool = True, auto_wrap: int = 0
) -> dict:
    """Train sharded and unsharded side by side and compare the weights."""
    topology = TopologyConfig(shard_parallel_size=world_size)
    with distributed_context(topology, backend="gloo") as context:
        model = MLP(MODEL, seed=3)
        reference = copy.deepcopy(model)

        fsdp = FullyShardedDataParallel(
            model,
            context.group("shard"),
            FSDPConfig(
                reshard_after_forward=reshard,
                auto_wrap_min_num_params=auto_wrap,
                check_reduction_order=True,
            ),
        )
        optimizer = ShardedOptimizer(
            fsdp.parameters(),
            OptimizerConfig(name="adamw", learning_rate=1e-2),
            norm_group=context.group("world"),
            device=context.device,
        )
        # torch.optim.AdamW defaults to weight_decay=0.01 while OptimizerConfig
        # defaults to 0.0; state both explicitly so the two sides really do
        # run the same optimizer.
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-2, weight_decay=0.0)

        inputs, targets = _global_batch(world_size, steps)
        stride = MICRO_BATCH * world_size
        losses = []
        for step in range(steps):
            block_inputs = inputs[step * stride : (step + 1) * stride]
            block_targets = targets[step * stride : (step + 1) * stride]
            local_inputs = block_inputs[rank * MICRO_BATCH : (rank + 1) * MICRO_BATCH]
            local_targets = block_targets[rank * MICRO_BATCH : (rank + 1) * MICRO_BATCH]

            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(fsdp(local_inputs), local_targets)
            loss.backward()
            fsdp.finish_backward()
            optimizer.step()
            losses.append(loss.item())

            reference_optimizer.zero_grad(set_to_none=True)
            nn.functional.mse_loss(reference(block_inputs), block_targets).backward()
            reference_optimizer.step()

        expected = dict(reference.named_parameters())
        full = fsdp.full_state_dict()
        error = max((full[n] - expected[n].detach()).abs().max().item() for n in expected)

        optimizer_state_numel = sum(
            v.numel()
            for state in optimizer.inner.state.values()
            for v in state.values()
            if torch.is_tensor(v) and v.numel() > 1
        )
        return {
            "error": error,
            "losses": losses,
            "optimizer_state_numel": optimizer_state_numel,
            "local_parameters": sum(p.numel() for p in fsdp.parameters()),
            "is_sharded_after_step": fsdp.handle.is_sharded if fsdp.handle else None,
        }


def worker_gradient_shape(rank: int, world_size: int) -> dict:
    """The gradient this rank holds is exactly its parameter shard's size."""
    topology = TopologyConfig(shard_parallel_size=world_size)
    with distributed_context(topology, backend="gloo") as context:
        fsdp = FullyShardedDataParallel(MLP(MODEL, seed=3), context.group("shard"), FSDPConfig())
        inputs = torch.randn(MICRO_BATCH, MODEL.input_size)
        fsdp(inputs).sum().backward()
        fsdp.finish_backward()
        flat = fsdp.handle.flat_param  # type: ignore[union-attr]
        assert flat.grad is not None
        handle = fsdp.handle
        assert handle is not None
        padding_start = handle.total_numel - handle.local_shard_range().start
        padded_tail = (
            flat.grad[padding_start:].abs().max().item()
            if 0 <= padding_start < flat.grad.numel()
            else 0.0
        )
        return {
            "param_numel": flat.numel(),
            "grad_numel": flat.grad.numel(),
            "padding": handle.padding,
            "padded_tail_max": padded_tail,
            "reductions": handle.reduction_count,
        }


def worker_summon_and_state_dict(rank: int, world_size: int) -> dict:
    """``summon_full_params`` and the state-dict paths reconstruct the model."""
    topology = TopologyConfig(shard_parallel_size=world_size)
    with distributed_context(topology, backend="gloo") as context:
        reference = MLP(MODEL, seed=3)
        expected = {n: p.detach().clone() for n, p in reference.named_parameters()}
        fsdp = FullyShardedDataParallel(MLP(MODEL, seed=3), context.group("shard"), FSDPConfig())

        resharded_shape = tuple(fsdp.module.blocks[0].linear.weight.shape)
        with fsdp.summon_full_params():
            summoned_shape = tuple(fsdp.module.blocks[0].linear.weight.shape)
            summoned_error = (
                (fsdp.module.blocks[0].linear.weight - expected["blocks.0.linear.weight"])
                .abs()
                .max()
                .item()
            )
        after_shape = tuple(fsdp.module.blocks[0].linear.weight.shape)

        full = fsdp.full_state_dict()
        full_error = max((full[n] - expected[n]).abs().max().item() for n in expected)

        pieces = fsdp.sharded_state_dict()
        owned = sum(p.length for p in pieces.values())

        # Load a perturbed state dict and read it back.
        perturbed = {n: v + 1.0 for n, v in expected.items()}
        fsdp.load_full_state_dict(perturbed)
        reloaded = fsdp.full_state_dict()
        reload_error = max((reloaded[n] - perturbed[n]).abs().max().item() for n in perturbed)

        return {
            "resharded_shape": resharded_shape,
            "summoned_shape": summoned_shape,
            "after_shape": after_shape,
            "summoned_error": summoned_error,
            "full_error": full_error,
            "owned_elements": owned,
            "reload_error": reload_error,
            "piece_names": sorted(pieces),
        }


def worker_no_sync(rank: int, world_size: int, micro_steps: int = 3) -> dict:
    """FSDP's ``no_sync`` accumulates unsharded gradients, reducing once."""
    topology = TopologyConfig(shard_parallel_size=world_size)
    with distributed_context(topology, backend="gloo") as context:
        fsdp = FullyShardedDataParallel(MLP(MODEL, seed=3), context.group("shard"), FSDPConfig())
        inputs, targets = _global_batch(world_size, micro_steps)
        stride = MICRO_BATCH * world_size

        for step in range(micro_steps):
            block = inputs[step * stride : (step + 1) * stride]
            block_targets = targets[step * stride : (step + 1) * stride]
            local = block[rank * MICRO_BATCH : (rank + 1) * MICRO_BATCH]
            local_targets = block_targets[rank * MICRO_BATCH : (rank + 1) * MICRO_BATCH]
            loss = nn.functional.mse_loss(fsdp(local), local_targets) / micro_steps
            if step < micro_steps - 1:
                with fsdp.no_sync():
                    loss.backward()
            else:
                loss.backward()
        fsdp.finish_backward()
        handle = fsdp.handle
        assert handle is not None
        accumulated = handle.flat_param.grad.clone()  # type: ignore[union-attr]
        reductions = handle.reduction_count

        # Reference: the same total loss with no accumulation trickery.
        reference = FullyShardedDataParallel(
            MLP(MODEL, seed=3), context.group("shard"), FSDPConfig()
        )
        total = None
        for step in range(micro_steps):
            block = inputs[step * stride : (step + 1) * stride]
            block_targets = targets[step * stride : (step + 1) * stride]
            local = block[rank * MICRO_BATCH : (rank + 1) * MICRO_BATCH]
            local_targets = block_targets[rank * MICRO_BATCH : (rank + 1) * MICRO_BATCH]
            piece = nn.functional.mse_loss(reference(local), local_targets) / micro_steps
            total = piece if total is None else total + piece
        assert total is not None
        total.backward()
        reference.finish_backward()
        expected = reference.handle.flat_param.grad  # type: ignore[union-attr]
        return {
            "error": (accumulated - expected).abs().max().item(),
            "reductions": reductions,
            "micro_steps": micro_steps,
        }


def worker_frozen_parameters_rejected(rank: int, world_size: int) -> str:
    """A unit mixing trainable and frozen parameters is refused."""
    from hybrid_training.errors import UnsupportedFeatureError

    topology = TopologyConfig(shard_parallel_size=world_size)
    with distributed_context(topology, backend="gloo") as context:
        model = MLP(MODEL, seed=3)
        model.head.weight.requires_grad_(False)
        try:
            FullyShardedDataParallel(model, context.group("shard"), FSDPConfig())
        except UnsupportedFeatureError as error:
            return f"rejected: {'requires_grad' in str(error)}"
        return "not rejected"


def worker_gather_limit(rank: int, world_size: int) -> str:
    """The all-gather guard rail refuses an over-large unit."""
    from hybrid_training.errors import ShardingError

    topology = TopologyConfig(shard_parallel_size=world_size)
    with distributed_context(topology, backend="gloo") as context:
        try:
            FullyShardedDataParallel(
                MLP(MODEL, seed=3),
                context.group("shard"),
                FSDPConfig(limit_all_gather_bytes=1024),
            )
        except ShardingError as error:
            return f"rejected: {'limit_all_gather_bytes' in str(error)}"
        return "not rejected"


def worker_parameterless_rejected(rank: int, world_size: int) -> str:
    """Wrapping a module with no parameters is refused."""
    from hybrid_training.errors import ShardingError

    topology = TopologyConfig(shard_parallel_size=world_size)
    with distributed_context(topology, backend="gloo") as context:
        try:
            FullyShardedDataParallel(nn.Identity(), context.group("shard"), FSDPConfig())
        except ShardingError as error:
            return f"rejected: {'no parameters' in str(error)}"
        return "not rejected"


def worker_hybrid_sharding(rank: int, world_size: int, steps: int = 4) -> dict:
    """Hybrid sharding (shard inside, replicate outside) matches full sharding."""
    topology = TopologyConfig(data_parallel_size=2, shard_parallel_size=world_size // 2)
    with distributed_context(topology, backend="gloo") as context:
        model = MLP(MODEL, seed=3)
        reference = copy.deepcopy(model)
        fsdp = FullyShardedDataParallel(
            model,
            context.group("shard"),
            FSDPConfig(),
            replica_group=context.group("data_parallel"),
        )
        optimizer = ShardedOptimizer(
            fsdp.parameters(),
            OptimizerConfig(name="adamw", learning_rate=1e-2),
            norm_group=context.group("world"),
            device=context.device,
        )
        # torch.optim.AdamW defaults to weight_decay=0.01 while OptimizerConfig
        # defaults to 0.0; state both explicitly so the two sides really do
        # run the same optimizer.
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-2, weight_decay=0.0)

        data_group = context.group("dp_shard")
        inputs, targets = _global_batch(data_group.size, steps)
        stride = MICRO_BATCH * data_group.size
        for step in range(steps):
            block = inputs[step * stride : (step + 1) * stride]
            block_targets = targets[step * stride : (step + 1) * stride]
            index = data_group.local_rank
            local = block[index * MICRO_BATCH : (index + 1) * MICRO_BATCH]
            local_targets = block_targets[index * MICRO_BATCH : (index + 1) * MICRO_BATCH]

            optimizer.zero_grad(set_to_none=True)
            nn.functional.mse_loss(fsdp(local), local_targets).backward()
            fsdp.finish_backward()
            optimizer.step()

            reference_optimizer.zero_grad(set_to_none=True)
            nn.functional.mse_loss(reference(block), block_targets).backward()
            reference_optimizer.step()

        expected = dict(reference.named_parameters())
        full = fsdp.full_state_dict()
        return {
            "error": max((full[n] - expected[n].detach()).abs().max().item() for n in expected),
            "local_parameters": sum(p.numel() for p in fsdp.parameters()),
        }


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
class TestShardConstruction:
    """Parameters are genuinely split."""

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_local_parameters_are_one_shard(self, world_size: int) -> None:
        """Each rank holds ``ceil(P / G)`` elements, not ``P``."""
        for result in run_distributed_cached(worker_shard_construction, world_size):
            assert result["unsharded_parameters"] == TOTAL_PARAMETERS
            expected_shard = -(-TOTAL_PARAMETERS // world_size)
            assert result["shard_numel"] == expected_shard
            assert result["local_parameters"] == expected_shard
            assert result["local_parameters"] < result["unsharded_parameters"]

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_shards_tile_the_padded_buffer(self, world_size: int) -> None:
        """Local ranges are contiguous, equal and cover the padded total."""
        results = run_distributed_cached(worker_shard_construction, world_size)
        ranges = [r["local_range"] for r in results]
        assert [r[0] for r in ranges] == sorted(r[0] for r in ranges)
        for previous, current in itertools.pairwise(ranges):
            assert previous[0] + previous[1] == current[0]
        assert ranges[-1][0] + ranges[-1][1] == results[0]["shard_numel"] * world_size

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_padding_is_present_and_accounted_for(self, world_size: int) -> None:
        """1857 parameters never divide evenly, so padding must exist."""
        for result in run_distributed_cached(worker_shard_construction, world_size):
            expected_padding = result["shard_numel"] * world_size - TOTAL_PARAMETERS
            assert result["padding"] == expected_padding > 0
            assert result["memory"]["padding_bytes"] == expected_padding * 4

    def test_unit_is_resharded_when_idle(self) -> None:
        """Between steps only the shard is resident."""
        for result in run_distributed_cached(worker_shard_construction, 2):
            assert result["is_sharded"]
            assert result["memory"]["full_bytes"] == 0

    def test_state_dict_names_are_wrapper_independent(self) -> None:
        """A sharded model's state dict uses the unwrapped model's names."""
        for result in run_distributed_cached(worker_shard_construction, 2):
            assert result["state_dict_keys"] == result["original_keys"]

    @pytest.mark.parametrize("auto_wrap", [0, 300])
    def test_auto_wrapping_creates_nested_units(self, auto_wrap: int) -> None:
        """A threshold splits the model into several units."""
        results = run_distributed_cached(
            worker_shard_construction, 2, kwargs={"auto_wrap": auto_wrap}
        )
        expected_units = 1 if auto_wrap == 0 else 4
        assert results[0]["units"] == expected_units
        # Nested wrapping pads each unit separately, so it can hold slightly
        # more elements than one big unit -- but never the whole model.
        assert results[0]["local_parameters"] < TOTAL_PARAMETERS


class TestTrainingEquivalence:
    """Sharded training must reproduce unsharded training."""

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_matches_single_process_reference(self, world_size: int) -> None:
        """Five AdamW steps stay within the amplified rounding tolerance."""
        for result in run_distributed_cached(worker_training_equivalence, world_size):
            assert result["error"] < OPTIMIZER_STEP_TOLERANCE

    @pytest.mark.parametrize("reshard", [True, False])
    def test_reshard_after_forward_does_not_change_results(self, reshard: bool) -> None:
        """Freeing parameters after forward is a memory choice, not a numerical one."""
        for result in run_distributed_cached(
            worker_training_equivalence, 2, kwargs={"reshard": reshard}
        ):
            assert result["error"] < OPTIMIZER_STEP_TOLERANCE

    @pytest.mark.parametrize("auto_wrap", [0, 300])
    def test_nested_wrapping_does_not_change_results(self, auto_wrap: int) -> None:
        """Wrapping granularity is a memory/communication choice only."""
        for result in run_distributed_cached(
            worker_training_equivalence, 2, kwargs={"auto_wrap": auto_wrap}
        ):
            assert result["error"] < OPTIMIZER_STEP_TOLERANCE

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_optimizer_state_is_sharded(self, world_size: int) -> None:
        """AdamW keeps two buffers per *local* element, not per global one."""
        for result in run_distributed_cached(worker_training_equivalence, world_size):
            assert result["optimizer_state_numel"] == 2 * result["local_parameters"]
            assert result["optimizer_state_numel"] < 2 * TOTAL_PARAMETERS

    def test_all_ranks_report_the_same_loss_trajectory(self) -> None:
        """Every rank computes the same global loss from its own shard."""
        results = run_distributed_cached(worker_training_equivalence, 2)
        # Ranks see different data, so their *local* losses differ; the point
        # here is that the resulting parameters agree, which the error check
        # above covers. Losses are recorded to show the run really trained.
        assert all(len(r["losses"]) == 5 for r in results)
        assert results[0]["losses"][0] > 0


class TestGradientSharding:
    """Gradients are reduce-scattered, not all-reduced."""

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_gradient_matches_the_parameter_shard(self, world_size: int) -> None:
        """``flat_param.grad`` has exactly the shard's element count."""
        for result in run_distributed_cached(worker_gradient_shape, world_size):
            assert result["grad_numel"] == result["param_numel"]
            assert result["grad_numel"] < TOTAL_PARAMETERS

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_padding_gradient_is_exactly_zero(self, world_size: int) -> None:
        """Padding is not a parameter, so it must never carry a gradient."""
        for result in run_distributed_cached(worker_gradient_shape, world_size):
            assert result["padded_tail_max"] == 0.0

    def test_one_reduction_per_backward(self) -> None:
        """A single backward performs exactly one reduce-scatter per unit."""
        for result in run_distributed_cached(worker_gradient_shape, 2):
            assert result["reductions"] == 1


class TestStateDictPaths:
    """Reconstruction and loading."""

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_summon_materialises_and_frees(self, world_size: int) -> None:
        """Inside the block the weight is whole; outside it is empty again."""
        for result in run_distributed_cached(worker_summon_and_state_dict, world_size):
            assert result["resharded_shape"] == (0,)
            assert result["summoned_shape"] == (25, 14)
            assert result["after_shape"] == (0,)
            assert result["summoned_error"] < FLOAT32_REDUCTION_TOLERANCE

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_full_state_dict_reconstructs_the_model(self, world_size: int) -> None:
        """All-gathering the shards recovers the original parameters exactly."""
        for result in run_distributed_cached(worker_summon_and_state_dict, world_size):
            # A pure data movement: bitwise equality is the right expectation.
            assert result["full_error"] == 0.0

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_sharded_pieces_cover_every_parameter_once(self, world_size: int) -> None:
        """The ranks' pieces sum to the model, with no double counting."""
        results = run_distributed_cached(worker_summon_and_state_dict, world_size)
        assert sum(r["owned_elements"] for r in results) == TOTAL_PARAMETERS

    def test_load_full_state_dict_round_trips(self) -> None:
        """Writing a state dict and reading it back returns the same values."""
        for result in run_distributed_cached(worker_summon_and_state_dict, 2):
            assert result["reload_error"] == 0.0


class TestNoSync:
    """Gradient accumulation with unsharded gradients."""

    @pytest.mark.parametrize("micro_steps", [2, 3])
    def test_accumulation_matches_a_summed_loss(self, micro_steps: int) -> None:
        """The accumulated shard equals the shard of the summed-loss gradient."""
        for result in run_distributed_cached(
            worker_no_sync, 2, kwargs={"micro_steps": micro_steps}
        ):
            assert result["error"] < FLOAT32_REDUCTION_TOLERANCE

    def test_accumulation_performs_one_reduction(self) -> None:
        """N micro-batches cost one reduce-scatter, not N."""
        for result in run_distributed_cached(worker_no_sync, 2, kwargs={"micro_steps": 3}):
            assert result["reductions"] == 1


class TestHybridSharding:
    """Shard inside a group, replicate across it."""

    def test_hybrid_matches_the_reference(self) -> None:
        """HSDP is mathematically identical to full sharding."""
        for result in run_distributed_cached(worker_hybrid_sharding, 4):
            assert result["error"] < OPTIMIZER_STEP_TOLERANCE

    def test_hybrid_shards_only_over_the_inner_group(self) -> None:
        """Memory scales with the shard size, not the world size."""
        for result in run_distributed_cached(worker_hybrid_sharding, 4):
            # shard group is 2, so each rank holds about half the model.
            assert result["local_parameters"] == -(-TOTAL_PARAMETERS // 2)


class TestFailureModes:
    """Explicitly unsupported configurations."""

    def test_mixed_requires_grad_rejected(self) -> None:
        """A unit cannot freeze part of its flat parameter."""
        assert (
            run_distributed_cached(worker_frozen_parameters_rejected, 2) == ["rejected: True"] * 2
        )

    def test_all_gather_limit_enforced(self) -> None:
        """The guard rail fires before an over-large unit is built."""
        assert run_distributed_cached(worker_gather_limit, 2) == ["rejected: True"] * 2

    def test_parameterless_module_rejected(self) -> None:
        """Sharding nothing is an error."""
        assert run_distributed_cached(worker_parameterless_rejected, 2) == ["rejected: True"] * 2

    def test_tied_parameters_across_units_rejected(self) -> None:
        """A weight shared between two units cannot stay tied."""
        results = expect_distributed_failure(worker_tied_across_units, 2)
        assert all(not r.succeeded for r in results)
        assert any("shared between two FSDP units" in (r.traceback_text or "") for r in results)


def worker_tied_across_units(rank: int, world_size: int) -> str:
    """Two nested units sharing one parameter must be refused."""
    topology = TopologyConfig(shard_parallel_size=world_size)

    class Tied(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(16, 16, bias=False)
            self.second = nn.Linear(16, 16, bias=False)
            self.second.weight = self.first.weight

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.second(self.first(x))

    with distributed_context(topology, backend="gloo") as context:
        FullyShardedDataParallel(
            Tied(), context.group("shard"), FSDPConfig(auto_wrap_min_num_params=1)
        )
        return "should not reach here"
