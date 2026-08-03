"""Multi-process tests for tensor-parallel layers and sequence parallelism.

The reference is an *unsharded* layer built from the same seed, so the
comparisons are structural rather than approximate: a column-parallel layer's
gathered weight must be bit-for-bit the reference's weight, and its gathered
output must be bit-for-bit the reference's output.  Only the gradients that
pass through a cross-rank sum carry floating-point reordering error.
"""

from __future__ import annotations

import itertools

import pytest
import torch
import torch.nn as nn

from hybrid_training.config import ModelConfig, SequenceParallelMode, TopologyConfig
from hybrid_training.distributed.context import distributed_context
from hybrid_training.models.transformer import (
    ParallelPlan,
    TinyTransformer,
    build_reference_linear,
)
from hybrid_training.parallel.sequence_parallel import (
    gather_sequence,
    local_sequence_slice,
    reduce_scatter_sequence,
    scatter_sequence,
)
from hybrid_training.parallel.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    all_reduce_sequence_parallel_gradients,
)

from ..conftest import (
    FLOAT32_REDUCTION_TOLERANCE,
    OPTIMIZER_STEP_TOLERANCE,
    expect_distributed_failure,
    run_distributed_cached,
)

pytestmark = pytest.mark.distributed

IN_FEATURES, OUT_FEATURES, BATCH = 12, 16, 5
INIT_SEED = 5

TRANSFORMER = ModelConfig(
    kind="transformer",
    vocab_size=32,
    hidden_size=16,
    num_heads=4,
    num_layers=2,
    ffn_hidden_size=32,
    max_sequence_length=16,
    dropout=0.0,
)


def _reference_pass(
    layer: nn.Linear, inputs: torch.Tensor, grad_output: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run an unsharded layer and return outputs plus all gradients."""
    inputs = inputs.clone().requires_grad_(True)
    outputs = layer(inputs)
    outputs.backward(grad_output)
    assert inputs.grad is not None and layer.weight.grad is not None
    return (
        outputs.detach(),
        inputs.grad.detach(),
        layer.weight.grad.detach(),
        None if layer.bias is None else layer.bias.grad,
    )


def worker_column_parallel(rank: int, world_size: int, bias: bool = True) -> dict:
    """Compare a column-parallel layer against an unsharded ``nn.Linear``."""
    topology = TopologyConfig(tensor_parallel_size=world_size)
    with distributed_context(topology, backend="gloo") as context:
        group = context.group("tensor")
        generator = torch.Generator().manual_seed(99)
        inputs = torch.randn(BATCH, IN_FEATURES, generator=generator)
        grad_output = torch.randn(BATCH, OUT_FEATURES, generator=generator)

        reference = build_reference_linear(
            IN_FEATURES, OUT_FEATURES, bias=bias, init_seed=INIT_SEED
        )
        expected_output, expected_input_grad, expected_weight_grad, expected_bias_grad = (
            _reference_pass(reference, inputs, grad_output)
        )

        layer = ColumnParallelLinear(
            IN_FEATURES,
            OUT_FEATURES,
            group,
            bias=bias,
            gather_output=True,
            init_seed=INIT_SEED,
        )
        parallel_inputs = inputs.clone().requires_grad_(True)
        outputs = layer(parallel_inputs)
        outputs.backward(grad_output)
        assert parallel_inputs.grad is not None and layer.weight.grad is not None

        weight_slice = expected_weight_grad.chunk(world_size, dim=0)[rank]
        result = {
            "weight_shape": tuple(layer.weight.shape),
            "output_error": (outputs.detach() - expected_output).abs().max().item(),
            "input_grad_error": (parallel_inputs.grad - expected_input_grad).abs().max().item(),
            "weight_grad_error": (layer.weight.grad - weight_slice).abs().max().item(),
            "full_weight_error": (layer.full_weight() - reference.weight.detach())
            .abs()
            .max()
            .item(),
        }
        if bias:
            assert layer.bias is not None and expected_bias_grad is not None
            bias_slice = expected_bias_grad.chunk(world_size, dim=0)[rank]
            result["bias_grad_error"] = (layer.bias.grad - bias_slice).abs().max().item()
            result["full_bias_error"] = (
                (
                    layer.full_bias() - reference.bias.detach()  # type: ignore[union-attr]
                )
                .abs()
                .max()
                .item()
            )

        # Sharded-output mode must give exactly this rank's feature slice.
        local_layer = ColumnParallelLinear(
            IN_FEATURES,
            OUT_FEATURES,
            group,
            bias=bias,
            gather_output=False,
            init_seed=INIT_SEED,
        )
        local_output = local_layer(inputs.clone())
        result["local_output_shape"] = tuple(local_output.shape)
        result["local_output_error"] = (
            (local_output.detach() - expected_output.chunk(world_size, dim=-1)[rank])
            .abs()
            .max()
            .item()
        )

        # One optimizer step on both sides must keep them equal.
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        optimizer.step()
        reference_optimizer.step()
        result["after_step_error"] = (
            (layer.full_weight() - reference.weight.detach()).abs().max().item()
        )
        return result


def worker_row_parallel(
    rank: int, world_size: int, bias: bool = True, input_is_parallel: bool = False
) -> dict:
    """Compare a row-parallel layer against an unsharded ``nn.Linear``."""
    topology = TopologyConfig(tensor_parallel_size=world_size)
    with distributed_context(topology, backend="gloo") as context:
        group = context.group("tensor")
        generator = torch.Generator().manual_seed(21)
        inputs = torch.randn(BATCH, OUT_FEATURES, generator=generator)
        grad_output = torch.randn(BATCH, IN_FEATURES, generator=generator)

        reference = build_reference_linear(
            OUT_FEATURES, IN_FEATURES, bias=bias, init_seed=INIT_SEED
        )
        expected_output, expected_input_grad, expected_weight_grad, expected_bias_grad = (
            _reference_pass(reference, inputs, grad_output)
        )

        layer = RowParallelLinear(
            OUT_FEATURES,
            IN_FEATURES,
            group,
            bias=bias,
            input_is_parallel=input_is_parallel,
            init_seed=INIT_SEED,
        )
        if input_is_parallel:
            local_inputs = (
                inputs.chunk(world_size, dim=-1)[rank].clone().contiguous().requires_grad_(True)
            )
        else:
            local_inputs = inputs.clone().requires_grad_(True)
        outputs = layer(local_inputs)
        outputs.backward(grad_output)
        assert local_inputs.grad is not None and layer.weight.grad is not None

        expected_grad = (
            expected_input_grad.chunk(world_size, dim=-1)[rank]
            if input_is_parallel
            else expected_input_grad
        )
        result = {
            "weight_shape": tuple(layer.weight.shape),
            "output_error": (outputs.detach() - expected_output).abs().max().item(),
            "input_grad_error": (local_inputs.grad - expected_grad).abs().max().item(),
            "weight_grad_error": (
                layer.weight.grad - expected_weight_grad.chunk(world_size, dim=1)[rank]
            )
            .abs()
            .max()
            .item(),
            "full_weight_error": (layer.full_weight() - reference.weight.detach())
            .abs()
            .max()
            .item(),
        }
        if bias:
            assert layer.bias is not None and expected_bias_grad is not None
            # The bias is replicated, so its gradient is the *whole* gradient.
            result["bias_grad_error"] = (layer.bias.grad - expected_bias_grad).abs().max().item()
            result["bias_shape"] = tuple(layer.bias.shape)
        return result


def worker_vocab_parallel_embedding(rank: int, world_size: int) -> dict:
    """A partitioned embedding table reproduces the unsharded lookup."""
    topology = TopologyConfig(tensor_parallel_size=world_size)
    with distributed_context(topology, backend="gloo") as context:
        group = context.group("tensor")
        layer = VocabParallelEmbedding(32, 8, group, init_seed=3)
        reference_weight = layer.full_weight()
        ids = torch.randint(0, 32, (4, 6), generator=torch.Generator().manual_seed(1))
        output = layer(ids)
        expected = nn.functional.embedding(ids, reference_weight)
        return {
            "shard_shape": tuple(layer.weight.shape),
            "vocab_range": (layer.vocab_start, layer.vocab_end),
            "error": (output - expected).abs().max().item(),
        }


def worker_indivisible_features(rank: int, world_size: int) -> str:
    """A feature dimension that does not divide evenly is refused."""
    from hybrid_training.errors import TensorParallelError

    topology = TopologyConfig(tensor_parallel_size=world_size)
    with distributed_context(topology, backend="gloo") as context:
        try:
            ColumnParallelLinear(8, world_size + 1, context.group("tensor"))
        except TensorParallelError as error:
            return f"rejected: {'divisible' in str(error)}"
        return "not rejected"


def worker_sequence_round_trip(rank: int, world_size: int) -> dict:
    """Scatter/gather round-trips, and their adjoints are correct."""
    topology = TopologyConfig(
        tensor_parallel_size=world_size,
        sequence_parallel_mode=SequenceParallelMode.TENSOR_GROUP,
    )
    with distributed_context(topology, backend="gloo") as context:
        group = context.group("sequence_effective")
        generator = torch.Generator().manual_seed(13)
        full = torch.randn(2, 4 * world_size, 3, generator=generator, requires_grad=True)

        shard = scatter_sequence(full, group)
        recovered = gather_sequence(shard, group)
        round_trip_error = (recovered - full).abs().max().item()

        # Backward through scatter->gather must be the identity.
        grad_output = torch.randn(recovered.shape, generator=generator)
        recovered.backward(grad_output)
        assert full.grad is not None
        adjoint_error = (full.grad - grad_output).abs().max().item()

        # reduce_scatter_sequence sums across ranks, then keeps this slice.
        partial = torch.full((2, 4 * world_size, 3), float(rank + 1), requires_grad=True)
        reduced = reduce_scatter_sequence(partial, group)
        expected_value = sum(r + 1 for r in range(world_size))
        start, end = local_sequence_slice(4 * world_size, group)

        return {
            "shard_shape": tuple(shard.shape),
            "round_trip_error": round_trip_error,
            "adjoint_error": adjoint_error,
            "reduced_value": reduced[0, 0, 0].item(),
            "expected_reduced": float(expected_value),
            "reduced_shape": tuple(reduced.shape),
            "positions": (start, end),
        }


def worker_odd_sequence_length(rank: int, world_size: int) -> str:
    """An indivisible sequence length is refused with padding guidance."""
    from hybrid_training.errors import ShardingError

    topology = TopologyConfig(
        tensor_parallel_size=world_size,
        sequence_parallel_mode=SequenceParallelMode.TENSOR_GROUP,
    )
    with distributed_context(topology, backend="gloo") as context:
        group = context.group("sequence_effective")
        try:
            scatter_sequence(torch.randn(2, 4 * world_size + 1, 3), group)
        except ShardingError as error:
            return f"rejected: {'pad_sequence_dimension' in str(error)}"
        return "not rejected"


def worker_transformer(
    rank: int, world_size: int, sequence_parallel: bool = False, sequence_length: int = 8
) -> dict:
    """Compare a tensor/sequence-parallel transformer against a single-process one."""
    mode = SequenceParallelMode.TENSOR_GROUP if sequence_parallel else SequenceParallelMode.DISABLED
    topology = TopologyConfig(tensor_parallel_size=world_size, sequence_parallel_mode=mode)
    with distributed_context(topology, backend="gloo") as context:
        plan = ParallelPlan(
            tensor_group=context.group("tensor"),
            sequence_group=context.group("sequence_effective"),
            sequence_parallel=sequence_parallel,
            vocab_parallel=True,
        )
        model = TinyTransformer(TRANSFORMER, plan, seed=INIT_SEED)
        reference = TinyTransformer(TRANSFORMER, ParallelPlan.single_process(), seed=INIT_SEED)

        generator = torch.Generator().manual_seed(4)
        ids = torch.randint(0, TRANSFORMER.vocab_size, (3, sequence_length), generator=generator)
        targets = torch.randint(
            0, TRANSFORMER.vocab_size, (3, sequence_length), generator=generator
        )

        logits = model(ids)
        reference_logits = reference(ids)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, TRANSFORMER.vocab_size), targets.reshape(-1)
        )
        reference_loss = nn.functional.cross_entropy(
            reference_logits.reshape(-1, TRANSFORMER.vocab_size), targets.reshape(-1)
        )
        loss.backward()
        reference_loss.backward()

        reduced = 0
        if sequence_parallel:
            reduced = all_reduce_sequence_parallel_gradients(
                model, context.group("sequence_effective")
            )

        layer_norm = model.blocks[0].input_norm.weight.grad
        reference_layer_norm = reference.blocks[0].input_norm.weight.grad
        query = model.blocks[0].attention.query.weight.grad
        reference_query = reference.blocks[0].attention.query.weight.grad.chunk(world_size, dim=0)[
            rank
        ]
        row_bias = model.blocks[0].attention.output.bias.grad
        reference_row_bias = reference.blocks[0].attention.output.bias.grad

        return {
            "logits_shape": tuple(logits.shape),
            "logit_error": (logits.detach() - reference_logits.detach()).abs().max().item(),
            "loss_error": abs(loss.item() - reference_loss.item()),
            "layer_norm_grad_error": (layer_norm - reference_layer_norm).abs().max().item(),
            "query_grad_error": (query - reference_query).abs().max().item(),
            "row_bias_grad_error": (row_bias - reference_row_bias).abs().max().item(),
            "local_parameters": model.num_parameters(),
            "reference_parameters": reference.num_parameters(),
            "partial_gradients_reduced": reduced,
        }


def worker_mixed_dtype(rank: int, world_size: int) -> dict:
    """Column/row parallel layers work in float64 as well as float32."""
    topology = TopologyConfig(tensor_parallel_size=world_size)
    with distributed_context(topology, backend="gloo") as context:
        group = context.group("tensor")
        errors = {}
        for dtype, tolerance_name in ((torch.float32, "fp32"), (torch.float64, "fp64")):
            layer = ColumnParallelLinear(
                IN_FEATURES,
                OUT_FEATURES,
                group,
                gather_output=True,
                init_seed=INIT_SEED,
                dtype=dtype,
            )
            reference = build_reference_linear(
                IN_FEATURES, OUT_FEATURES, init_seed=INIT_SEED, dtype=dtype
            )
            inputs = torch.randn(
                BATCH, IN_FEATURES, generator=torch.Generator().manual_seed(2), dtype=dtype
            )
            errors[tolerance_name] = (
                (layer(inputs).detach() - reference(inputs).detach()).abs().max().item()
            )
            errors[f"{tolerance_name}_dtype"] = str(layer.weight.dtype)
        return errors


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
class TestColumnParallel:
    """Column partitioning: split the output features."""

    @pytest.mark.parametrize("world_size", [2, 4])
    @pytest.mark.parametrize("bias", [True, False])
    def test_forward_is_exact(self, world_size: int, bias: bool) -> None:
        """Gathering the shards reproduces the unsharded output bit-for-bit."""
        for result in run_distributed_cached(
            worker_column_parallel, world_size, kwargs={"bias": bias}
        ):
            # Each output feature is computed by exactly one rank with exactly
            # the same arithmetic, so there is no reordering error at all.
            assert result["output_error"] == 0.0
            assert result["weight_shape"] == (OUT_FEATURES // world_size, IN_FEATURES)

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_weights_are_the_reference_sliced(self, world_size: int) -> None:
        """The concatenated shards *are* the reference weight."""
        for result in run_distributed_cached(worker_column_parallel, world_size):
            assert result["full_weight_error"] == 0.0
            assert result["full_bias_error"] == 0.0

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_weight_gradients_are_exact(self, world_size: int) -> None:
        """A weight-shard gradient involves no cross-rank sum, so it is exact."""
        for result in run_distributed_cached(worker_column_parallel, world_size):
            assert result["weight_grad_error"] == 0.0

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_input_gradients_are_all_reduced(self, world_size: int) -> None:
        """The input gradient is a cross-rank sum, so only rounding differs."""
        for result in run_distributed_cached(worker_column_parallel, world_size):
            assert 0.0 <= result["input_grad_error"] < FLOAT32_REDUCTION_TOLERANCE

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_bias_is_partitioned(self, world_size: int) -> None:
        """The bias follows the output features it belongs to."""
        for result in run_distributed_cached(worker_column_parallel, world_size):
            assert result["bias_grad_error"] < FLOAT32_REDUCTION_TOLERANCE

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_sharded_output_mode(self, world_size: int) -> None:
        """Without gathering, the output is exactly this rank's feature slice."""
        for result in run_distributed_cached(worker_column_parallel, world_size):
            assert result["local_output_shape"] == (BATCH, OUT_FEATURES // world_size)
            assert result["local_output_error"] == 0.0

    def test_optimizer_step_keeps_the_layers_equal(self) -> None:
        """An SGD step applied to shards equals the same step unsharded."""
        for result in run_distributed_cached(worker_column_parallel, 2):
            assert result["after_step_error"] < OPTIMIZER_STEP_TOLERANCE


class TestRowParallel:
    """Row partitioning: split the input features."""

    @pytest.mark.parametrize("world_size", [2, 4])
    @pytest.mark.parametrize("bias", [True, False])
    def test_forward_matches_reference(self, world_size: int, bias: bool) -> None:
        """The all-reduced partial sums equal the unsharded product."""
        for result in run_distributed_cached(
            worker_row_parallel, world_size, kwargs={"bias": bias}
        ):
            # Forward *is* a cross-rank sum here, so rounding applies.
            assert result["output_error"] < FLOAT32_REDUCTION_TOLERANCE
            assert result["weight_shape"] == (IN_FEATURES, OUT_FEATURES // world_size)

    @pytest.mark.parametrize("input_is_parallel", [True, False])
    def test_both_input_modes(self, input_is_parallel: bool) -> None:
        """Auto-splitting and pre-split inputs both work."""
        for result in run_distributed_cached(
            worker_row_parallel, 2, kwargs={"input_is_parallel": input_is_parallel}
        ):
            assert result["output_error"] < FLOAT32_REDUCTION_TOLERANCE
            assert result["input_grad_error"] < FLOAT32_REDUCTION_TOLERANCE

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_weight_gradients_are_exact(self, world_size: int) -> None:
        """Weight-shard gradients are local, so exact."""
        for result in run_distributed_cached(worker_row_parallel, world_size):
            assert result["weight_grad_error"] < FLOAT32_REDUCTION_TOLERANCE

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_bias_is_replicated_and_added_after_the_reduction(self, world_size: int) -> None:
        """A replicated bias keeps the full width and the full gradient."""
        for result in run_distributed_cached(worker_row_parallel, world_size):
            assert result["bias_shape"] == (IN_FEATURES,)
            assert result["bias_grad_error"] < FLOAT32_REDUCTION_TOLERANCE


class TestVocabParallelEmbedding:
    """Vocabulary partitioning."""

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_masked_lookup_matches_the_full_table(self, world_size: int) -> None:
        """Exactly one rank contributes a non-zero row per token."""
        results = run_distributed_cached(worker_vocab_parallel_embedding, world_size)
        for result in results:
            assert result["error"] < FLOAT32_REDUCTION_TOLERANCE
            assert result["shard_shape"] == (32 // world_size, 8)
        ranges = [r["vocab_range"] for r in results]
        assert ranges[0][0] == 0 and ranges[-1][1] == 32
        for previous, current in itertools.pairwise(ranges):
            assert previous[1] == current[0]


class TestDivisibility:
    """Uneven partitioning is refused."""

    def test_indivisible_features_rejected(self) -> None:
        """An out-features value that does not divide is an error."""
        assert run_distributed_cached(worker_indivisible_features, 2) == ["rejected: True"] * 2


class TestSequenceParallelOperations:
    """Scatter, gather and reduce-scatter along the sequence."""

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_scatter_gather_round_trip(self, world_size: int) -> None:
        """Gathering the scattered shards recovers the input exactly."""
        for result in run_distributed_cached(worker_sequence_round_trip, world_size):
            assert result["round_trip_error"] == 0.0
            assert result["shard_shape"] == (2, 4, 3)

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_adjoint_of_the_round_trip_is_the_identity(self, world_size: int) -> None:
        """Backward through scatter-then-gather returns the incoming gradient."""
        for result in run_distributed_cached(worker_sequence_round_trip, world_size):
            assert result["adjoint_error"] == 0.0

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_reduce_scatter_sums_then_splits(self, world_size: int) -> None:
        """The value is the cross-rank sum; the shape is ``1/G`` of the input."""
        for result in run_distributed_cached(worker_sequence_round_trip, world_size):
            assert result["reduced_value"] == pytest.approx(result["expected_reduced"])
            assert result["reduced_shape"] == (2, 4, 3)

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_positions_tile_the_sequence(self, world_size: int) -> None:
        """The ranks' position ranges partition the sequence."""
        results = run_distributed_cached(worker_sequence_round_trip, world_size)
        positions = [r["positions"] for r in results]
        assert positions[0][0] == 0
        assert positions[-1][1] == 4 * world_size
        for previous, current in itertools.pairwise(positions):
            assert previous[1] == current[0]

    def test_odd_sequence_length_rejected(self) -> None:
        """An indivisible sequence is refused, pointing at the padding helper."""
        assert run_distributed_cached(worker_odd_sequence_length, 2) == ["rejected: True"] * 2


class TestTransformerEquivalence:
    """The whole model, with and without sequence parallelism."""

    @pytest.mark.parametrize("world_size", [2, 4])
    @pytest.mark.parametrize("sequence_parallel", [False, True])
    def test_logits_match_the_single_process_model(
        self, world_size: int, sequence_parallel: bool
    ) -> None:
        """Forward is equivalent regardless of how the model is partitioned."""
        for result in run_distributed_cached(
            worker_transformer,
            world_size,
            kwargs={"sequence_parallel": sequence_parallel},
        ):
            assert result["logit_error"] < FLOAT32_REDUCTION_TOLERANCE
            assert result["loss_error"] < FLOAT32_REDUCTION_TOLERANCE

    @pytest.mark.parametrize("sequence_parallel", [False, True])
    def test_partitioned_weight_gradients_match(self, sequence_parallel: bool) -> None:
        """The query weight's gradient equals the reference's matching slice."""
        for result in run_distributed_cached(
            worker_transformer, 2, kwargs={"sequence_parallel": sequence_parallel}
        ):
            assert result["query_grad_error"] < FLOAT32_REDUCTION_TOLERANCE

    @pytest.mark.parametrize("sequence_parallel", [False, True])
    def test_replicated_parameter_gradients_match(self, sequence_parallel: bool) -> None:
        """LayerNorm and row-parallel-bias gradients are complete.

        Under sequence parallelism these are *partial* until
        ``all_reduce_sequence_parallel_gradients`` sums them; this is the test
        that would fail if that step were missing.
        """
        for result in run_distributed_cached(
            worker_transformer, 2, kwargs={"sequence_parallel": sequence_parallel}
        ):
            assert result["layer_norm_grad_error"] < FLOAT32_REDUCTION_TOLERANCE
            assert result["row_bias_grad_error"] < FLOAT32_REDUCTION_TOLERANCE

    def test_sequence_parallelism_marks_partial_gradients(self) -> None:
        """The LayerNorms and row-parallel biases are the marked parameters."""
        without = run_distributed_cached(worker_transformer, 2, kwargs={"sequence_parallel": False})
        with_sp = run_distributed_cached(worker_transformer, 2, kwargs={"sequence_parallel": True})
        assert without[0]["partial_gradients_reduced"] == 0
        # 2 blocks x 2 LayerNorms x (weight + bias) + final norm x 2
        # + 2 row-parallel biases x 2 blocks = 14
        assert with_sp[0]["partial_gradients_reduced"] == 14

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_parameters_are_actually_partitioned(self, world_size: int) -> None:
        """Each rank holds strictly fewer parameters than the whole model."""
        for result in run_distributed_cached(worker_transformer, world_size):
            assert result["local_parameters"] < result["reference_parameters"]

    def test_shorter_sequences_still_match(self) -> None:
        """A sequence shorter than the maximum works in both modes."""
        for result in run_distributed_cached(
            worker_transformer, 2, kwargs={"sequence_length": 4, "sequence_parallel": True}
        ):
            assert result["logit_error"] < FLOAT32_REDUCTION_TOLERANCE
            assert result["logits_shape"] == (3, 4, TRANSFORMER.vocab_size)


class TestDtypes:
    """Multiple floating-point widths."""

    def test_float32_and_float64(self) -> None:
        """Both dtypes reproduce the reference within their own precision."""
        for result in run_distributed_cached(worker_mixed_dtype, 2):
            assert result["fp32"] == 0.0
            assert result["fp64"] == 0.0
            assert result["fp32_dtype"] == "torch.float32"
            assert result["fp64_dtype"] == "torch.float64"


def worker_plan_rejects_trivial_sequence_group(rank: int, world_size: int) -> str:
    """Requesting sequence parallelism over a one-rank group is refused."""
    from hybrid_training.distributed.groups import GroupHandle
    from hybrid_training.errors import ConfigurationError

    with distributed_context(TopologyConfig(data_parallel_size=world_size), backend="gloo"):
        try:
            ParallelPlan(
                tensor_group=GroupHandle.trivial(),
                sequence_group=GroupHandle.trivial(),
                sequence_parallel=True,
            )
        except ConfigurationError as error:
            return f"rejected: {'nothing to split' in str(error)}"
        return "not rejected"


def test_plan_validation() -> None:
    """A contradictory parallel plan is refused at construction."""
    assert (
        run_distributed_cached(worker_plan_rejects_trivial_sequence_group, 2)
        == ["rejected: True"] * 2
    )


def test_tied_embeddings_with_vocab_parallel_rejected() -> None:
    """Tied embeddings plus a partitioned vocabulary is explicitly unsupported."""
    results = expect_distributed_failure(worker_tied_embeddings, 2)
    assert all(not r.succeeded for r in results)
    assert any("tied word embeddings" in (r.traceback_text or "") for r in results)


def worker_tied_embeddings(rank: int, world_size: int) -> str:
    """Build a transformer with an unsupported tie."""
    topology = TopologyConfig(tensor_parallel_size=world_size)
    with distributed_context(topology, backend="gloo") as context:
        config = ModelConfig(
            kind="transformer",
            vocab_size=32,
            hidden_size=16,
            num_heads=4,
            num_layers=1,
            tie_word_embeddings=True,
        )
        plan = ParallelPlan(
            tensor_group=context.group("tensor"),
            sequence_group=context.group("sequence_effective"),
            vocab_parallel=True,
        )
        TinyTransformer(config, plan, seed=0)
        return "should not reach here"
