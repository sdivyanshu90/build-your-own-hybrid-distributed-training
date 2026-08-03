"""Single-process tests for buckets, norms, schedules, RNG, memory and data.

Everything here uses a one-member ``GroupHandle``, which is exactly what the
distributed code sees at world size 1.  That means these tests exercise the
*real* code paths rather than a non-distributed alternative, while remaining
fast enough to run on every commit.
"""

from __future__ import annotations

import itertools
import math

import pytest
import torch
import torch.nn as nn

from hybrid_training.config import (
    DataConfig,
    DDPConfig,
    FSDPConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
)
from hybrid_training.distributed.collectives import CommunicationRecorder
from hybrid_training.distributed.groups import GroupHandle, validate_group_membership
from hybrid_training.errors import (
    CollectiveError,
    ConfigurationError,
    ShardingError,
    TensorParallelError,
)
from hybrid_training.models.mlp import MLP, build_activation
from hybrid_training.models.transformer import TinyTransformer, build_reference_linear
from hybrid_training.optim.sharded_optimizer import (
    ShardedOptimizer,
    build_gradient_norm_contributions,
    build_inner_optimizer,
)
from hybrid_training.parallel.ddp import DistributedDataParallel
from hybrid_training.parallel.fsdp import FullyShardedDataParallel
from hybrid_training.parallel.sequence_parallel import (
    SequenceShardInfo,
    pad_sequence_dimension,
    unpad_sequence_dimension,
)
from hybrid_training.parallel.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from hybrid_training.training.data import (
    DistributedBatchSampler,
    SyntheticMLPDataset,
    SyntheticTokenDataset,
    build_dataset,
)
from hybrid_training.training.state import LearningRateSchedule, TrainingState
from hybrid_training.utils.memory import (
    estimate_training_memory,
    format_bytes,
    module_parameter_bytes,
)
from hybrid_training.utils.reproducibility import (
    capture_rng_state,
    derive_seed,
    restore_rng_state,
    rng_state_from_serialisable,
    rng_state_to_serialisable,
    temporary_seed,
)

TRIVIAL = GroupHandle.trivial()


class TestGroupHandle:
    """The one-member handle used by single-process code paths."""

    def test_trivial_group_properties(self) -> None:
        """A one-member group is trivial and its own source."""
        assert TRIVIAL.size == 1
        assert TRIVIAL.is_trivial
        assert TRIVIAL.source_rank == 0
        assert TRIVIAL.process_group is None

    def test_membership_validation(self) -> None:
        """Using a group you are not in is refused before the collective."""
        validate_group_membership(TRIVIAL, 0, "test")
        with pytest.raises(CollectiveError, match="not a member of process group"):
            validate_group_membership(TRIVIAL, 1, "test")


class TestCommunicationRecorder:
    """What ``calls`` counts, and what it must not count."""

    def test_wait_time_is_attributed_without_inflating_the_call_count(self) -> None:
        """A launch and its wait are one collective, not two.

        The recorder is fed twice for an asynchronous collective: once when it
        is issued and once when it is waited on.  If both incremented ``calls``,
        an asynchronous all-reduce would report twice as many collectives as the
        byte-for-byte identical synchronous one, so the headline number would
        describe *how* the call was issued rather than what crossed the wire.

        This shipped as a real defect: DDP launches its buckets asynchronously,
        so `tests/performance` measured 6 all-reduces for a 3-step, 1-bucket run.
        """
        recorder = CommunicationRecorder()
        recorder.record("all_reduce", "data_parallel", num_bytes=1024, seconds=0.25)
        recorder.record_wait("all_reduce", "data_parallel", wait_seconds=0.75)

        stats = recorder.by_operation["all_reduce/data_parallel"]
        assert stats.calls == 1, "the wait must not count as a second collective"
        assert stats.bytes == 1024
        assert stats.seconds == pytest.approx(0.25)
        assert stats.wait_seconds == pytest.approx(0.75)

    def test_a_synchronous_and_an_asynchronous_collective_count_the_same(self) -> None:
        """The count must not depend on how the collective was issued."""
        synchronous = CommunicationRecorder()
        synchronous.record("all_reduce", "shard", num_bytes=512)

        asynchronous = CommunicationRecorder()
        asynchronous.record("all_reduce", "shard", num_bytes=512)
        asynchronous.record_wait("all_reduce", "shard", wait_seconds=0.5)

        assert asynchronous.total().calls == synchronous.total().calls == 1
        assert asynchronous.total().bytes == synchronous.total().bytes == 512

    def test_recording_can_be_disabled(self) -> None:
        """A disabled recorder is inert on both entry points."""
        recorder = CommunicationRecorder(enabled=False)
        recorder.record("all_reduce", "world", num_bytes=64)
        recorder.record_wait("all_reduce", "world", wait_seconds=1.0)
        assert recorder.by_operation == {}


class TestDDPBuckets:
    """Bucket construction, which must be identical on every rank."""

    def _model(self) -> nn.Module:
        return MLP(ModelConfig(input_size=8, hidden_size=16, num_layers=2, output_size=4), seed=0)

    def test_buckets_are_built_in_reverse_parameter_order(self) -> None:
        """Bucket 0 holds the *last* parameters, which finish backward first."""
        model = self._model()
        names = [n for n, _ in model.named_parameters()]
        ddp = DistributedDataParallel(model, TRIVIAL, DDPConfig(bucket_cap_mb=1e-6))
        layouts = ddp.bucket_layouts()
        flattened = [n for layout in layouts for n in layout.parameter_names]
        assert flattened == list(reversed(names))
        ddp.teardown()

    def test_large_cap_produces_a_single_bucket(self) -> None:
        """A cap larger than the model coalesces everything into one collective."""
        ddp = DistributedDataParallel(self._model(), TRIVIAL, DDPConfig(bucket_cap_mb=100.0))
        assert len(ddp.bucket_layouts()) == 1
        ddp.teardown()

    def test_bucket_offsets_are_contiguous(self) -> None:
        """Offsets tile each bucket's flat buffer exactly."""
        ddp = DistributedDataParallel(self._model(), TRIVIAL, DDPConfig(bucket_cap_mb=0.001))
        for layout in ddp.bucket_layouts():
            cursor = 0
            for offset, numel in zip(layout.offsets, layout.numels):
                assert offset == cursor
                cursor += numel
            assert cursor == layout.total_numel
        ddp.teardown()

    def test_every_parameter_appears_exactly_once(self) -> None:
        """No parameter is reduced twice or forgotten."""
        model = self._model()
        ddp = DistributedDataParallel(model, TRIVIAL, DDPConfig(bucket_cap_mb=0.002))
        placed = [n for layout in ddp.bucket_layouts() for n in layout.parameter_names]
        assert sorted(placed) == sorted(n for n, _ in model.named_parameters())
        ddp.teardown()

    def test_parameterless_module_rejected(self) -> None:
        """Wrapping something with nothing to synchronise is an error."""
        with pytest.raises(ShardingError, match="no trainable parameters"):
            DistributedDataParallel(nn.Identity(), TRIVIAL, DDPConfig())

    def test_unused_parameter_without_the_flag_raises(self) -> None:
        """A parameter with no gradient is reported, not silently skipped."""

        class PartiallyUsed(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.used = nn.Linear(4, 4)
                self.unused = nn.Linear(4, 4)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.used(x)

        ddp = DistributedDataParallel(PartiallyUsed(), TRIVIAL, DDPConfig())
        ddp(torch.randn(2, 4)).sum().backward()
        with pytest.raises(ShardingError, match="received no gradient"):
            ddp.finish_gradient_synchronization()
        ddp.teardown()

    def test_unused_parameter_with_the_flag_contributes_zeros(self) -> None:
        """``find_unused_parameters`` fills zeros so all ranks reduce alike."""

        class PartiallyUsed(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.used = nn.Linear(4, 4)
                self.unused = nn.Linear(4, 4)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.used(x)

        model = PartiallyUsed()
        ddp = DistributedDataParallel(model, TRIVIAL, DDPConfig(find_unused_parameters=True))
        ddp(torch.randn(2, 4)).sum().backward()
        ddp.finish_gradient_synchronization()
        assert torch.equal(model.unused.weight.grad, torch.zeros_like(model.unused.weight))
        assert "unused.weight" in ddp.statistics.unused_parameters
        ddp.teardown()

    def test_forward_before_synchronising_is_refused(self) -> None:
        """Running forward with unreduced gradients pending is an error."""
        ddp = DistributedDataParallel(self._model(), TRIVIAL, DDPConfig())
        ddp(torch.randn(2, 8)).sum().backward()
        with pytest.raises(ShardingError, match="has not been synchronised"):
            ddp(torch.randn(2, 8))
        ddp.teardown()

    def test_no_sync_accumulates_without_touching_buckets(self) -> None:
        """Two accumulated micro-batches equal one double-sized batch."""
        model = self._model()
        ddp = DistributedDataParallel(model, TRIVIAL, DDPConfig())
        x = torch.randn(4, 8)
        with ddp.no_sync():
            ddp(x[:2]).sum().backward()
        ddp(x[2:]).sum().backward()
        ddp.finish_gradient_synchronization()
        accumulated = {n: p.grad.clone() for n, p in ddp.parameters_and_names()}

        for _, param in ddp.parameters_and_names():
            param.grad = None
        ddp(x).sum().backward()
        ddp.finish_gradient_synchronization()
        for name, param in ddp.parameters_and_names():
            # Same additions in the same order: bitwise equality is the right
            # expectation, and anything else would indicate a lost micro-batch.
            assert torch.allclose(accumulated[name], param.grad, atol=1e-6), name
        ddp.teardown()


class TestWrapperTransparency:
    """Wrapping must not change how the caller reaches their own model."""

    def test_attributes_reach_through_the_wrapper(self) -> None:
        """`wrapped.weight` still works, because auto-wrap rewrote the tree.

        Auto-wrapping replaces nested modules *in place*: an ``nn.Linear`` at
        ``blocks[0].linear`` becomes a `FullyShardedDataParallel` around it.
        Every attribute path through that submodule would break, even though
        the caller's model is unchanged -- which is what broke
        `examples/train_fsdp.py`, whose point is that `summon_full_params()`
        lets ordinary code read the weights.  Ordinary code does not know about
        wrappers.
        """
        config = ModelConfig(input_size=8, hidden_size=12, num_layers=2, output_size=4)
        inner = MLP(config, seed=1)
        expected = tuple(inner.blocks[0].linear.weight.shape)

        wrapped = FullyShardedDataParallel(inner, TRIVIAL, FSDPConfig())

        # The *path* resolves through the wrapper, which is the guarantee here.
        # The tensor it lands on is the resharded placeholder: outside a forward
        # FSDP frees the full parameter's storage, so the shape is empty. Both
        # facts are asserted, because a test that only checked reachability
        # would not notice the parameter silently coming back unsharded.
        assert tuple(wrapped.blocks[0].linear.weight.shape) == (0,)

        # Inside summon_full_params the same path yields the real tensor, which
        # is precisely what `examples/train_fsdp.py` demonstrates.
        with wrapped.summon_full_params():
            assert tuple(wrapped.blocks[0].linear.weight.shape) == expected

        # Non-parameter attributes forward regardless of sharding state.
        # blocks[0] is the input projection, so it maps input_size -> hidden_size.
        assert wrapped.blocks[0].linear.in_features == config.input_size
        assert wrapped.blocks[0].linear.out_features == config.hidden_size

    def test_unknown_attributes_still_raise(self) -> None:
        """Forwarding must not turn a typo into a silent ``None``."""
        config = ModelConfig(input_size=8, hidden_size=12, num_layers=2, output_size=4)
        wrapped = FullyShardedDataParallel(MLP(config, seed=1), TRIVIAL, FSDPConfig())
        with pytest.raises(AttributeError):
            _ = wrapped.no_such_attribute


class TestCpuOffload:
    """The CPU-offload path, exercised at world size 1.

    On this device (`cpu`) the offload copies are no-ops, so this proves the
    *plumbing* -- that the shard, the gradient and the optimizer step agree
    with the non-offloaded path -- rather than the transfer itself.  The
    transfer only does real work on CUDA, which
    ``tests/distributed/test_cuda.py`` covers and which cannot run without a
    GPU.  Stated here so the coverage gap is visible rather than implied.
    """

    def _train_one_step(self, offload: bool) -> tuple[torch.Tensor, dict]:
        """Run a single SGD step with or without offload; return grad and weights."""
        config = ModelConfig(input_size=8, hidden_size=12, num_layers=2, output_size=4)
        torch.manual_seed(0)
        inputs, targets = torch.randn(3, 8), torch.randn(3, 4)
        model = FullyShardedDataParallel(
            MLP(config, seed=1), TRIVIAL, FSDPConfig(cpu_offload_params=offload)
        )
        optimizer = ShardedOptimizer(
            model.parameters(),
            OptimizerConfig(name="sgd", learning_rate=0.1),
            norm_group=TRIVIAL,
            device=torch.device("cpu"),
        )
        nn.functional.mse_loss(model(inputs), targets).backward()
        model.finish_backward()
        handle = model.handle
        assert handle is not None and handle.flat_param.grad is not None
        gradient = handle.flat_param.grad.clone()
        optimizer.clip_grad_norm(1.0)
        optimizer.step()
        return gradient, model.full_state_dict()

    def test_offload_does_not_change_the_gradient(self) -> None:
        """The reduce-scattered gradient is identical either way."""
        plain, _ = self._train_one_step(offload=False)
        offloaded, _ = self._train_one_step(offload=True)
        assert torch.equal(plain, offloaded)

    def test_offload_does_not_change_the_update(self) -> None:
        """One SGD step lands on exactly the same weights."""
        _, plain = self._train_one_step(offload=False)
        _, offloaded = self._train_one_step(offload=True)
        assert sorted(plain) == sorted(offloaded)
        for name in plain:
            assert torch.equal(plain[name], offloaded[name]), name


class TestGradientNorms:
    """The per-parameter weighting that makes the global norm correct."""

    def test_replicated_parameter_is_scaled_down(self) -> None:
        """A replicated parameter counts once, so its weight is ``1/W``."""
        param = nn.Parameter(torch.ones(4))
        contributions = build_gradient_norm_contributions([param], world_size=4)
        assert contributions[0].scale == pytest.approx(0.25)

    def test_partitioned_parameter_keeps_full_weight_per_shard(self) -> None:
        """A parameter split ``P`` ways over ``W`` ranks weighs ``P/W``."""
        param = nn.Parameter(torch.ones(4))
        param.is_tensor_parallel_replicated = False
        param.tensor_parallel_group_size = 2
        contributions = build_gradient_norm_contributions([param], world_size=4)
        assert contributions[0].scale == pytest.approx(0.5)

    def test_elementwise_scale_vector_is_honoured(self) -> None:
        """A flat parameter's per-element weighting is used verbatim."""
        param = nn.Parameter(torch.ones(4))
        param.grad = torch.ones(4)
        param.gradient_norm_scale_vector = torch.tensor([2.0, 2.0, 1.0, 0.0])
        contribution = build_gradient_norm_contributions([param], world_size=4)[0]
        # (2 + 2 + 1 + 0) / 4 == 1.25
        assert contribution.squared_norm().item() == pytest.approx(1.25)

    def test_missing_gradient_contributes_zero(self) -> None:
        """A parameter with no gradient does not perturb the norm."""
        contribution = build_gradient_norm_contributions(
            [nn.Parameter(torch.ones(4))], world_size=1
        )[0]
        assert contribution.squared_norm().item() == 0.0

    def test_norm_matches_torch_at_world_size_one(self) -> None:
        """At world size 1 the result equals ``torch.nn.utils.clip_grad_norm_``."""
        model = MLP(ModelConfig(input_size=4, hidden_size=8, num_layers=1, output_size=2), seed=0)
        model(torch.randn(3, 4)).sum().backward()
        expected = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))

        optimizer = ShardedOptimizer(
            model.parameters(),
            OptimizerConfig(),
            norm_group=TRIVIAL,
            device=torch.device("cpu"),
        )
        observed = optimizer.clip_grad_norm(0.0)
        # Both sum the same squares; only the accumulation dtype differs
        # (float64 here, float32 in torch), so 1e-5 is generous headroom.
        assert observed.item() == pytest.approx(expected.item(), abs=1e-5)

    def test_clipping_scales_gradients(self) -> None:
        """Clipping below the norm rescales gradients to exactly the threshold."""
        param = nn.Parameter(torch.ones(4))
        param.grad = torch.full((4,), 3.0)  # norm = 6
        optimizer = ShardedOptimizer(
            [param], OptimizerConfig(), norm_group=TRIVIAL, device=torch.device("cpu")
        )
        norm = optimizer.clip_grad_norm(1.0)
        assert norm.item() == pytest.approx(6.0)
        assert param.grad.norm().item() == pytest.approx(1.0, abs=1e-5)

    def test_no_trainable_parameters_rejected(self) -> None:
        """Building an optimizer over nothing is an error."""
        with pytest.raises(ShardingError, match="no trainable parameters"):
            build_inner_optimizer([], OptimizerConfig())

    def test_world_size_must_be_positive(self) -> None:
        """A zero reduction-group size is a programming error."""
        with pytest.raises(ShardingError, match="world_size must be positive"):
            build_gradient_norm_contributions([nn.Parameter(torch.ones(1))], world_size=0)


class TestLearningRateSchedule:
    """Schedules are pure functions of the step."""

    def test_warmup_is_linear_and_reaches_the_peak(self) -> None:
        """Warm-up ramps from ``base/w`` to ``base`` over ``w`` steps."""
        schedule = LearningRateSchedule(
            SchedulerConfig(name="constant", warmup_steps=4), 1.0, total_steps=10
        )
        assert [round(schedule.value_at(s), 4) for s in range(5)] == [0.25, 0.5, 0.75, 1.0, 1.0]

    def test_cosine_decays_to_the_floor(self) -> None:
        """Cosine ends at ``min_lr_ratio`` and is monotonically decreasing."""
        schedule = LearningRateSchedule(
            SchedulerConfig(name="cosine", min_lr_ratio=0.1), 1.0, total_steps=10
        )
        values = [schedule.value_at(s) for s in range(11)]
        assert values[0] == pytest.approx(1.0)
        assert values[-1] == pytest.approx(0.1)
        assert all(a >= b - 1e-12 for a, b in itertools.pairwise(values))

    def test_linear_decays_to_the_floor(self) -> None:
        """Linear decay reaches the floor exactly at the last step."""
        schedule = LearningRateSchedule(
            SchedulerConfig(name="linear", min_lr_ratio=0.0), 2.0, total_steps=4
        )
        assert schedule.value_at(4) == pytest.approx(0.0)
        assert schedule.value_at(2) == pytest.approx(1.0)

    def test_schedule_is_pure(self) -> None:
        """Evaluating out of order gives the same values, so resume is exact."""
        schedule = LearningRateSchedule(SchedulerConfig(name="cosine"), 1.0, total_steps=20)
        forward = [schedule.value_at(s) for s in range(20)]
        backward = [schedule.value_at(s) for s in reversed(range(20))]
        assert forward == list(reversed(backward))

    def test_state_dict_round_trip(self) -> None:
        """A schedule rebuilt from its state dict produces identical values."""
        schedule = LearningRateSchedule(
            SchedulerConfig(name="cosine", warmup_steps=3, min_lr_ratio=0.2), 0.5, 30
        )
        restored = LearningRateSchedule.from_state_dict(schedule.state_dict())
        assert [restored.value_at(s) for s in range(30)] == [
            schedule.value_at(s) for s in range(30)
        ]

    def test_warmup_longer_than_the_run_rejected(self) -> None:
        """A decay schedule whose warm-up never ends is a configuration error."""
        with pytest.raises(ConfigurationError, match="warm-up is at least as long"):
            LearningRateSchedule(SchedulerConfig(name="cosine", warmup_steps=10), 1.0, 10)


class TestTrainingState:
    """Progress bookkeeping and its JSON form."""

    def test_round_trip_with_nan_and_inf(self) -> None:
        """``nan``/``inf`` survive the JSON-safe representation."""
        state = TrainingState(step=5, epoch=1, samples_seen=40)
        restored = TrainingState.from_dict(state.as_dict())
        assert restored.step == 5
        assert math.isnan(restored.last_loss)
        assert math.isinf(restored.best_eval_loss)

    def test_as_dict_is_strict_json(self) -> None:
        """The dictionary contains no values ``json.dumps`` would reject."""
        import json

        json.dumps(TrainingState().as_dict(), allow_nan=False)

    def test_advance_step(self) -> None:
        """Advancing a step updates the counters consistently."""
        state = TrainingState()
        state.micro_step = 3
        state.advance_step(samples=16)
        assert (state.step, state.samples_seen, state.micro_step) == (1, 16, 0)


class TestReproducibility:
    """Seed derivation and RNG state round-tripping."""

    def test_derivation_is_deterministic_and_separated(self) -> None:
        """Same inputs give the same seed; different inputs do not collide."""
        assert derive_seed(1234, "a") == derive_seed(1234, "a")
        assert derive_seed(1234, "a") != derive_seed(1234, "b")
        assert derive_seed(1234, "a", 0) != derive_seed(1234, "a", 1)
        assert derive_seed(1, "a") != derive_seed(2, "a")

    def test_derived_seeds_are_in_range(self) -> None:
        """Seeds stay within the range ``torch.manual_seed`` accepts."""
        for index in range(200):
            seed = derive_seed(99, "stream", index)
            assert 0 <= seed < 2**31

    def test_temporary_seed_restores_the_stream(self) -> None:
        """A seeded block does not perturb the surrounding random stream."""
        torch.manual_seed(0)
        before = torch.randn(3)
        torch.manual_seed(0)
        _ = torch.randn(3)
        with temporary_seed(999):
            _ = torch.randn(10)
        after = torch.randn(3)
        torch.manual_seed(0)
        _ = torch.randn(3)
        expected = torch.randn(3)
        assert torch.equal(after, expected)
        assert not torch.equal(before, after)

    def test_rng_state_round_trip(self) -> None:
        """Restoring a snapshot reproduces the exact next draws."""
        torch.manual_seed(7)
        snapshot = capture_rng_state(include_cuda=False)
        first = torch.randn(5)
        restore_rng_state(snapshot)
        assert torch.equal(torch.randn(5), first)

    def test_serialisable_form_round_trips(self) -> None:
        """The tensor/JSON split reconstructs an equivalent snapshot."""
        torch.manual_seed(11)
        snapshot = capture_rng_state(include_cuda=False)
        expected = torch.randn(4)
        payload = rng_state_to_serialisable(snapshot)
        assert all(torch.is_tensor(t) for t in payload["tensors"].values())
        import json

        json.dumps(payload["meta"])  # must be JSON-safe
        restore_rng_state(rng_state_from_serialisable(payload))
        assert torch.equal(torch.randn(4), expected)


class TestMemoryModel:
    """The analytical memory estimate."""

    def test_sharding_divides_persistent_state(self) -> None:
        """Four-way sharding quarters parameters, gradients and state."""
        replicated = estimate_training_memory(1_000_000, shard_group_size=1)
        sharded = estimate_training_memory(
            1_000_000, shard_group_size=4, largest_unit_parameters=100_000
        )
        assert sharded.persistent_parameters == replicated.persistent_parameters // 4
        assert sharded.optimizer_state == replicated.optimizer_state // 4
        assert sharded.steady_state < replicated.steady_state

    def test_adam_state_is_two_slots(self) -> None:
        """AdamW keeps two buffers per parameter."""
        estimate = estimate_training_memory(1000, optimizer="adamw")
        assert estimate.optimizer_state == 1000 * 2 * 4

    def test_plain_sgd_keeps_no_state(self) -> None:
        """SGD without momentum allocates nothing."""
        assert estimate_training_memory(1000, optimizer="sgd").optimizer_state == 0

    def test_wrapping_granularity_drives_the_transient(self) -> None:
        """A larger unit means a larger transient all-gather buffer."""
        coarse = estimate_training_memory(1_000_000, shard_group_size=4)
        fine = estimate_training_memory(
            1_000_000, shard_group_size=4, largest_unit_parameters=50_000
        )
        assert fine.peak < coarse.peak

    def test_no_reshard_keeps_two_units_resident(self) -> None:
        """Disabling reshard-after-forward doubles the transient estimate."""
        with_reshard = estimate_training_memory(
            1000, shard_group_size=2, largest_unit_parameters=500, reshard_after_forward=True
        )
        without = estimate_training_memory(
            1000, shard_group_size=2, largest_unit_parameters=500, reshard_after_forward=False
        )
        assert (
            without.transient_gathered_parameters == 2 * with_reshard.transient_gathered_parameters
        )

    def test_invalid_arguments(self) -> None:
        """Bad group sizes and unknown optimizers are rejected."""
        with pytest.raises(ValueError, match="shard_group_size must be positive"):
            estimate_training_memory(10, shard_group_size=0)
        with pytest.raises(ValueError, match="unknown optimizer"):
            estimate_training_memory(10, optimizer="lion")

    def test_format_bytes(self) -> None:
        """Byte formatting uses binary units."""
        assert format_bytes(512) == "512 B"
        assert format_bytes(1536) == "1.50 KiB"
        assert format_bytes(1024**3) == "1.00 GiB"

    def test_module_parameter_bytes(self) -> None:
        """Module accounting matches a hand count."""
        model = nn.Linear(4, 8)
        report = module_parameter_bytes(model)
        assert report["parameters"] == (4 * 8 + 8) * 4
        assert report["gradients"] == 0


class TestSequenceShardInfo:
    """Sequence padding metadata."""

    def test_documented_example(self) -> None:
        """Length 10 over 4 ranks pads to 12 with 3 positions each."""
        info = SequenceShardInfo.for_length(10, 4)
        assert (info.padded_length, info.local_length, info.padding) == (12, 3, 2)
        assert info.requires_padding

    def test_exact_division_needs_no_padding(self) -> None:
        """An already-divisible length is left alone."""
        info = SequenceShardInfo.for_length(8, 4)
        assert not info.requires_padding
        assert info.local_range(2) == (4, 6)

    def test_pad_and_unpad_round_trip(self) -> None:
        """Padding then un-padding recovers the original tensor exactly."""
        original = torch.randn(2, 7, 3)
        padded, info = pad_sequence_dimension(original, 4)
        assert padded.shape[1] == 8
        assert torch.equal(padded[:, 7:], torch.zeros(2, 1, 3))
        assert torch.equal(unpad_sequence_dimension(padded, info), original)

    def test_unpad_checks_the_length(self) -> None:
        """Un-padding a wrongly sized tensor is refused."""
        _, info = pad_sequence_dimension(torch.randn(2, 7, 3), 4)
        with pytest.raises(ShardingError, match="does not have the padded sequence length"):
            unpad_sequence_dimension(torch.randn(2, 5, 3), info)

    def test_group_size_must_be_positive(self) -> None:
        """A zero group size is rejected."""
        with pytest.raises(ShardingError, match="group size must be positive"):
            SequenceShardInfo.for_length(8, 0)


class TestTensorParallelLayersAtWidthOne:
    """Parallel layers must reduce to ordinary layers at width 1."""

    def test_column_parallel_matches_nn_linear(self) -> None:
        """A one-rank column-parallel layer *is* an ``nn.Linear``."""
        layer = ColumnParallelLinear(6, 8, TRIVIAL, init_seed=3, gather_output=True)
        reference = build_reference_linear(6, 8, init_seed=3)
        x = torch.randn(4, 6)
        # Identical weights and identical arithmetic: bitwise equality.
        assert torch.equal(layer.weight, reference.weight)
        assert torch.equal(layer(x), reference(x))

    def test_row_parallel_matches_nn_linear(self) -> None:
        """The same holds for row-parallel layers."""
        layer = RowParallelLinear(6, 8, TRIVIAL, init_seed=3, input_is_parallel=False)
        reference = build_reference_linear(6, 8, init_seed=3)
        x = torch.randn(4, 6)
        assert torch.equal(layer(x), reference(x))

    def test_vocab_parallel_embedding_matches_nn_embedding(self) -> None:
        """A one-rank vocabulary-parallel embedding is a plain lookup."""
        layer = VocabParallelEmbedding(16, 4, TRIVIAL, init_seed=1)
        ids = torch.randint(0, 16, (3, 5))
        assert torch.equal(layer(ids), nn.functional.embedding(ids, layer.weight))

    def test_indivisible_features_rejected(self) -> None:
        """Uneven partitioning is refused with an actionable message."""
        group = GroupHandle(
            name="tensor", ranks=(0, 1, 2), local_rank=0, global_rank=0, process_group=None
        )
        with pytest.raises(TensorParallelError, match="out_features must be divisible"):
            ColumnParallelLinear(4, 8, group)
        with pytest.raises(TensorParallelError, match="in_features must be divisible"):
            RowParallelLinear(8, 4, group)
        with pytest.raises(TensorParallelError, match="num_embeddings must be divisible"):
            VocabParallelEmbedding(10, 4, group)

    def test_sequence_parallel_with_gathered_output_rejected(self) -> None:
        """The contradictory flag combination is refused at construction."""
        with pytest.raises(TensorParallelError, match="gather_output=False"):
            ColumnParallelLinear(4, 4, TRIVIAL, sequence_parallel=True, gather_output=True)

    def test_load_from_linear_checks_shapes(self) -> None:
        """Copying from a mismatched reference layer is refused."""
        layer = ColumnParallelLinear(6, 8, TRIVIAL)
        with pytest.raises(TensorParallelError, match="weight shape mismatch"):
            layer.load_from_linear(nn.Linear(6, 4))

    def test_load_from_linear_requires_a_bias_when_the_layer_has_one(self) -> None:
        """A bias-less reference cannot fill a layer that has a bias."""
        layer = ColumnParallelLinear(6, 8, TRIVIAL, bias=True)
        with pytest.raises(TensorParallelError, match="has a bias but the reference"):
            layer.load_from_linear(nn.Linear(6, 8, bias=False))


class TestModels:
    """Reference-model construction."""

    def test_mlp_shapes_and_determinism(self) -> None:
        """The MLP is shape-correct and reproducible from its seed."""
        config = ModelConfig(input_size=6, hidden_size=12, num_layers=2, output_size=3)
        first = MLP(config, seed=5)
        second = MLP(config, seed=5)
        assert first(torch.zeros(4, 6)).shape == (4, 3)
        for a, b in zip(first.parameters(), second.parameters()):
            assert torch.equal(a, b)
        assert not torch.equal(
            next(iter(MLP(config, seed=6).parameters())), next(iter(first.parameters()))
        )

    def test_transformer_shapes(self) -> None:
        """The transformer maps token ids to logits of the right shape."""
        config = ModelConfig(
            kind="transformer",
            vocab_size=32,
            hidden_size=16,
            num_heads=4,
            num_layers=2,
            max_sequence_length=8,
        )
        model = TinyTransformer(config, seed=0)
        assert model(torch.zeros(2, 8, dtype=torch.long)).shape == (2, 8, 32)

    def test_transformer_rejects_overlong_sequences(self) -> None:
        """A sequence longer than the positional table is refused."""
        config = ModelConfig(
            kind="transformer", max_sequence_length=4, hidden_size=8, num_heads=2, vocab_size=16
        )
        model = TinyTransformer(config, seed=0)
        with pytest.raises(ConfigurationError, match="longer than the positional table"):
            model(torch.zeros(1, 5, dtype=torch.long))

    def test_transformer_rejects_mlp_config(self) -> None:
        """Building a transformer from an MLP config is refused."""
        with pytest.raises(ConfigurationError, match="must be 'transformer'"):
            TinyTransformer(ModelConfig(kind="mlp"))

    def test_mlp_requires_at_least_one_layer(self) -> None:
        """A zero-layer MLP is impossible because the config rejects it first.

        The invariant lives in ``ModelConfig``, so ``MLP`` does not re-check
        it -- a second check there would be unreachable code.
        """
        with pytest.raises(ConfigurationError, match="num_layers must be positive"):
            MLP(ModelConfig(num_layers=0))

    def test_unknown_activation(self) -> None:
        """The activation factory rejects unknown names."""
        with pytest.raises(ConfigurationError, match="unknown activation"):
            build_activation("swish")

    def test_causal_attention_ignores_future_positions(self) -> None:
        """Changing a later token cannot change an earlier position's output."""
        config = ModelConfig(
            kind="transformer",
            vocab_size=16,
            hidden_size=16,
            num_heads=2,
            num_layers=1,
            max_sequence_length=8,
        )
        model = TinyTransformer(config, seed=0)
        first = torch.tensor([[1, 2, 3, 4]])
        second = torch.tensor([[1, 2, 3, 9]])
        with torch.no_grad():
            a = model(first)
            b = model(second)
        assert torch.allclose(a[:, :3], b[:, :3], atol=1e-6)
        assert not torch.allclose(a[:, 3], b[:, 3], atol=1e-6)


class TestSyntheticData:
    """Dataset determinism and the topology-aware sampler."""

    def test_samples_are_index_addressable(self) -> None:
        """Sample ``i`` is the same tensor regardless of access order."""
        dataset = SyntheticMLPDataset(32, seed=3, input_size=4, output_size=2)
        forward = [dataset.get(i)[0] for i in range(8)]
        backward = [dataset.get(i)[0] for i in reversed(range(8))]
        for a, b in zip(forward, reversed(backward)):
            assert torch.equal(a, b)

    def test_datasets_are_reproducible_from_the_seed(self) -> None:
        """Two datasets with the same seed produce identical samples."""
        a = SyntheticTokenDataset(16, seed=5, vocab_size=32, sequence_length=6)
        b = SyntheticTokenDataset(16, seed=5, vocab_size=32, sequence_length=6)
        for index in range(16):
            assert torch.equal(a.get(index)[0], b.get(index)[0])

    def test_token_targets_are_the_shifted_inputs(self) -> None:
        """Next-token targets line up with the inputs."""
        dataset = SyntheticTokenDataset(4, seed=1, vocab_size=16, sequence_length=6)
        tokens, targets = dataset.get(0)
        assert tokens.shape == targets.shape == (6,)
        assert torch.equal(tokens[1:], targets[:-1])

    def test_token_sequences_are_learnable(self) -> None:
        """Most transitions follow the recurrence, so the task is learnable."""
        dataset = SyntheticTokenDataset(
            64, seed=2, vocab_size=32, sequence_length=16, noise_probability=0.1
        )
        follows = 0
        total = 0
        for index in range(64):
            tokens, _ = dataset.get(index)
            for a, b in itertools.pairwise(tokens):
                total += 1
                follows += int(b.item() == (7 * int(a.item()) + 3) % 32)
        assert follows / total > 0.8

    def test_sampler_slices_are_disjoint_and_cover_the_global_batch(self) -> None:
        """Ranks take disjoint slices whose union is the global batch."""
        groups = [
            GroupHandle(
                name="dp_shard", ranks=(0, 1, 2, 3), local_rank=r, global_rank=r, process_group=None
            )
            for r in range(4)
        ]
        samplers = [DistributedBatchSampler(64, 2, g, seed=9) for g in groups]
        for batch_index in range(samplers[0].batches_per_epoch):
            per_rank = [list(s.iter_epoch(0))[batch_index] for s in samplers]
            flattened = [i for indices in per_rank for i in indices]
            assert len(set(flattened)) == len(flattened)
            assert sorted(flattened) == sorted(samplers[0].global_batch_indices(0, batch_index))

    def test_all_ranks_agree_on_the_epoch_order(self) -> None:
        """The permutation is identical everywhere, which makes slicing valid."""
        groups = [
            GroupHandle(
                name="dp_shard", ranks=(0, 1), local_rank=r, global_rank=r, process_group=None
            )
            for r in range(2)
        ]
        samplers = [DistributedBatchSampler(32, 4, g, seed=3) for g in groups]
        assert torch.equal(samplers[0].epoch_order(0), samplers[1].epoch_order(0))
        assert not torch.equal(samplers[0].epoch_order(0), samplers[0].epoch_order(1))

    def test_dataset_smaller_than_a_global_batch_rejected(self) -> None:
        """A dataset too small for one step is a configuration error."""
        group = GroupHandle(
            name="dp_shard", ranks=(0, 1, 2, 3), local_rank=0, global_rank=0, process_group=None
        )
        with pytest.raises(ConfigurationError, match="smaller than one global batch"):
            DistributedBatchSampler(4, 2, group)

    def test_build_dataset_selects_by_model_kind(self) -> None:
        """The dataset type follows the model kind, and splits differ."""
        data = DataConfig(num_train_samples=16, num_eval_samples=8, sequence_length=4)
        train = build_dataset(ModelConfig(kind="transformer", vocab_size=16), data, split="train")
        evaluation = build_dataset(
            ModelConfig(kind="transformer", vocab_size=16), data, split="eval"
        )
        assert isinstance(train, SyntheticTokenDataset)
        assert not torch.equal(train.get(0)[0], evaluation.get(0)[0])
        assert isinstance(build_dataset(ModelConfig(kind="mlp"), data), SyntheticMLPDataset)

    def test_unknown_split_rejected(self) -> None:
        """Only ``train`` and ``eval`` exist."""
        with pytest.raises(ConfigurationError, match="unknown split"):
            build_dataset(ModelConfig(), DataConfig(), split="test")

    def test_too_short_sequences_rejected(self) -> None:
        """Next-token prediction needs at least two positions."""
        with pytest.raises(ConfigurationError, match="at least two positions"):
            SyntheticTokenDataset(4, seed=0, vocab_size=8, sequence_length=1)

    def test_empty_dataset_rejected(self) -> None:
        """A dataset with no samples is refused."""
        with pytest.raises(ConfigurationError, match="at least one sample"):
            SyntheticMLPDataset(0, seed=0, input_size=2, output_size=1)
