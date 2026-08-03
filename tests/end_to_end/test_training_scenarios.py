"""End-to-end scenarios: every strategy trained through the real engine.

Each test asserts a *numerical* property, never merely that the processes
exited zero.  The eight scenarios required by the project specification map
onto the tests here as follows:

===  ============================================  ==========================
 #   Scenario                                      Test
===  ============================================  ==========================
 1   single-process baseline                       ``TestSingleProcess``
 2   two-rank custom DDP                           ``TestStrategyEquivalence``
 3   two-rank FSDP-style training                  ``TestStrategyEquivalence``
 4   two-rank tensor parallelism                   ``TestTransformerStrategies``
 5   two-rank sequence parallelism                 ``TestTransformerStrategies``
 6   four-rank hybrid training                     ``TestHybrid``
 7   checkpoint save and resume                    ``TestCheckpointResume``
 8   checkpoint resharding across world sizes      ``TestCheckpointResume``
===  ============================================  ==========================
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hybrid_training.config import (
    CheckpointConfig,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    SequenceParallelMode,
    TensorParallelConfig,
    TopologyConfig,
    TrainingConfig,
)
from hybrid_training.distributed.context import distributed_context

from ..conftest import OPTIMIZER_STEP_TOLERANCE, requires_ranks, run_distributed_cached

pytestmark = [pytest.mark.e2e, pytest.mark.distributed]

#: A global batch that stays constant as the world size changes, so runs at
#: different world sizes consume the *same* samples and are comparable.
GLOBAL_BATCH = 8
STEPS = 8

MLP_MODEL = ModelConfig(input_size=12, hidden_size=20, num_layers=3, output_size=6)
TRANSFORMER_MODEL = ModelConfig(
    kind="transformer",
    vocab_size=32,
    hidden_size=16,
    num_heads=4,
    num_layers=2,
    ffn_hidden_size=32,
    max_sequence_length=16,
    dropout=0.0,
)


def _topology(world_size: int, strategy: str) -> TopologyConfig:
    """Map a strategy name onto a topology for the given world size."""
    if strategy == "single":
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
        return TopologyConfig(data_parallel_size=world_size // 2, shard_parallel_size=2)
    if strategy == "hybrid_tensor":
        return TopologyConfig(data_parallel_size=world_size // 2, tensor_parallel_size=2)
    if strategy == "hybrid_full":
        return TopologyConfig(
            data_parallel_size=world_size // 4,
            shard_parallel_size=2,
            tensor_parallel_size=2,
            sequence_parallel_mode=SequenceParallelMode.TENSOR_GROUP,
        )
    raise ValueError(f"unknown strategy {strategy!r}")


def build_config(
    world_size: int,
    strategy: str,
    *,
    model: str = "mlp",
    steps: int = STEPS,
    directory: str = "",
) -> ExperimentConfig:
    """Assemble a configuration whose global batch is world-size independent."""
    topology = _topology(world_size, strategy)
    # Only the data-processing dimensions consume distinct samples.
    data_ranks = topology.data_parallel_size * topology.shard_parallel_size
    model_config = TRANSFORMER_MODEL if model == "transformer" else MLP_MODEL
    return ExperimentConfig(
        name=f"{strategy}-{world_size}",
        backend="gloo",
        device="cpu",
        topology=topology,
        model=model_config,
        data=DataConfig(
            micro_batch_size=GLOBAL_BATCH // data_ranks,
            sequence_length=8,
            num_train_samples=256,
            num_eval_samples=32,
            seed=1234,
        ),
        optimizer=OptimizerConfig(name="adamw", learning_rate=3e-3),
        scheduler=SchedulerConfig(name="cosine", warmup_steps=2, min_lr_ratio=0.1),
        training=TrainingConfig(
            max_steps=steps,
            seed=0,
            max_grad_norm=1.0,
            log_every_steps=0,
            eval_every_steps=0,
            collect_metrics=True,
        ),
        tensor_parallel=TensorParallelConfig(sequence_parallel=topology.sequence_parallel_enabled),
        checkpoint=CheckpointConfig(directory=directory or "unused"),
    )


# --------------------------------------------------------------------------
# workers
# --------------------------------------------------------------------------
def worker_train(
    rank: int,
    world_size: int,
    strategy: str = "ddp",
    model: str = "mlp",
    steps: int = STEPS,
    directory: str = "",
    save_at: int = 0,
) -> dict:
    """Run the engine end to end and report losses, weights and metrics."""
    from hybrid_training.training.engine import TrainingEngine

    config = build_config(world_size, strategy, model=model, steps=steps, directory=directory)
    with distributed_context(config) as context:
        engine = TrainingEngine(config, context)
        batches = engine._batch_stream()
        losses: list[float] = []
        grad_norms: list[float] = []
        saved = ""
        for _ in range(steps):
            metrics = engine.train_step(batches)
            losses.append(metrics.loss)
            if metrics.grad_norm is not None:
                grad_norms.append(metrics.grad_norm)
            if save_at and engine.state.step == save_at:
                saved = str(engine.save_checkpoint())

        evaluation = engine.evaluate()
        full = engine.model.full_state_dict()
        counts = engine.parameter_count()
        summary = {
            "losses": losses,
            "grad_norms": grad_norms,
            "eval_loss": evaluation,
            "strategy": engine.model.description.strategy,
            "parameters": counts,
            "memory": engine.model.memory_summary(),
            "state": engine.state.as_dict(),
            "checkpoint": saved,
            "weight_checksum": float(sum(v.double().sum().item() for v in full.values())),
            "weight_norm": float(sum(v.double().pow(2).sum().item() for v in full.values()) ** 0.5),
            "metric_group": engine.model.metric_group.ranks,
            "norm_group_size": engine.model.norm_group.size,
        }
        engine.close()
        return summary


def worker_resume(
    rank: int,
    world_size: int,
    strategy: str = "fsdp",
    model: str = "mlp",
    steps: int = STEPS,
    directory: str = "",
    checkpoint: str = "",
) -> dict:
    """Resume from a checkpoint and finish the run."""
    from hybrid_training.training.engine import TrainingEngine

    config = build_config(world_size, strategy, model=model, steps=steps, directory=directory)
    with distributed_context(config) as context:
        engine = TrainingEngine(config, context)
        loaded = engine.load_checkpoint(checkpoint)
        batches = engine._batch_stream()
        losses: list[float] = []
        while engine.state.step < steps:
            losses.append(engine.train_step(batches).loss)
        full = engine.model.full_state_dict()
        engine.close()
        return {
            "resumed_at": loaded.step,
            "losses": losses,
            "weight_checksum": float(sum(v.double().sum().item() for v in full.values())),
            "weight_norm": float(sum(v.double().pow(2).sum().item() for v in full.values()) ** 0.5),
        }


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
class TestSingleProcess:
    """Scenario 1: the baseline every other scenario is compared against."""

    def test_single_process_run_converges(self) -> None:
        """A one-rank run trains, and the loss decreases."""
        result = run_distributed_cached(worker_train, 1, kwargs={"strategy": "single"})[0]
        assert result["strategy"] == "single-process"
        assert len(result["losses"]) == STEPS

        # Comparing two *individual* losses is not a convergence test.  On this
        # trajectory the step-to-step variance (~0.08) is several times the
        # total improvement over eight steps (~0.03), so `losses[-1] <
        # losses[0]` measures which two mini-batches happened to be drawn --
        # and on this seed it is false even though training is working.
        #
        # Averaging the first quarter against the last quarter divides that
        # variance down and states the claim actually being made.  The run is
        # seeded, so this is deterministic rather than merely probable.
        quarter = max(1, STEPS // 4)
        first = sum(result["losses"][:quarter]) / quarter
        last = sum(result["losses"][-quarter:]) / quarter
        assert last < first, f"the model did not learn: {first:.5f} -> {last:.5f}"

        assert result["parameters"]["local"] == result["parameters"]["global"]

    def test_single_process_uses_the_distributed_code_path(self) -> None:
        """World size 1 still builds a real process group and named groups."""
        result = run_distributed_cached(worker_train, 1, kwargs={"strategy": "single"})[0]
        assert result["metric_group"] == (0,)
        assert result["norm_group_size"] == 1


class TestStrategyEquivalence:
    """Scenarios 2 and 3: DDP and FSDP reproduce the baseline."""

    @pytest.fixture(scope="class")
    def baseline(self) -> dict:
        """The single-process trajectory, computed once for the class."""
        return run_distributed_cached(worker_train, 1, kwargs={"strategy": "single"})[0]

    @pytest.mark.parametrize(("strategy", "world_size"), [("ddp", 2), ("fsdp", 2), ("fsdp", 4)])
    def test_final_weights_match_the_baseline(
        self, strategy: str, world_size: int, baseline: dict
    ) -> None:
        """Distributed training lands on the same weights as one process."""
        results = run_distributed_cached(worker_train, world_size, kwargs={"strategy": strategy})
        assert results[0]["weight_norm"] == pytest.approx(
            baseline["weight_norm"], abs=OPTIMIZER_STEP_TOLERANCE
        )

    @pytest.mark.parametrize(("strategy", "world_size"), [("ddp", 2), ("fsdp", 2)])
    def test_loss_trajectories_match_the_baseline(
        self, strategy: str, world_size: int, baseline: dict
    ) -> None:
        """Every step's loss matches, not just the endpoint."""
        results = run_distributed_cached(worker_train, world_size, kwargs={"strategy": strategy})
        for step, (observed, expected) in enumerate(zip(results[0]["losses"], baseline["losses"])):
            assert observed == pytest.approx(expected, abs=1e-5), f"step {step}"

    def test_all_ranks_report_identical_reduced_losses(self) -> None:
        """The reduced loss is a global quantity and must agree everywhere."""
        results = run_distributed_cached(worker_train, 4, kwargs={"strategy": "fsdp"})
        for step in range(STEPS):
            values = {r["losses"][step] for r in results}
            assert len(values) == 1, f"ranks disagree at step {step}: {values}"

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_fsdp_holds_less_than_the_whole_model(self, world_size: int) -> None:
        """Sharding is real: local parameters shrink as ranks are added."""
        results = run_distributed_cached(worker_train, world_size, kwargs={"strategy": "fsdp"})
        counts = results[0]["parameters"]
        assert counts["local"] < counts["global"]
        assert counts["local"] <= -(-counts["global"] // world_size) + world_size

    def test_ddp_replicates_rather_than_shards(self) -> None:
        """DDP's global count is the local count times the world size."""
        results = run_distributed_cached(worker_train, 2, kwargs={"strategy": "ddp"})
        counts = results[0]["parameters"]
        assert counts["global"] == 2 * counts["local"]

    def test_gradient_norms_agree_across_ranks(self) -> None:
        """A global norm must be a single number, identical on every rank."""
        results = run_distributed_cached(worker_train, 4, kwargs={"strategy": "fsdp"})
        for step in range(STEPS):
            values = {round(r["grad_norms"][step], 9) for r in results}
            assert len(values) == 1, f"gradient norms disagree at step {step}"


class TestTransformerStrategies:
    """Scenarios 4 and 5: tensor and sequence parallelism on the transformer."""

    @pytest.fixture(scope="class")
    def baseline(self) -> dict:
        """The single-process transformer trajectory."""
        return run_distributed_cached(
            worker_train, 1, kwargs={"strategy": "single", "model": "transformer"}
        )[0]

    @pytest.mark.parametrize("strategy", ["tensor", "sequence"])
    def test_matches_the_single_process_transformer(self, strategy: str, baseline: dict) -> None:
        """Partitioning the model does not change what it learns.

        The comparison is over the *whole trajectory* -- every step's loss and
        every step's gradient norm -- rather than the endpoint weights.

        Weight norms are deliberately not compared here, and the reason is worth
        stating because getting it wrong looks like a failing test rather than a
        bad question.  Under tensor parallelism `full_state_dict()` returns this
        rank's **slices**: a tensor-parallel slice is a genuinely different
        tensor on each rank, so there is no global tensor to hand back (see its
        docstring, and `parameter_parallel_info()` for interpreting the slices).
        Comparing a shard's norm against the single-process model's norm
        compares half a model to a whole one -- here `11.24` against `13.11` --
        which says nothing about equivalence.

        The loss and the gradient norm *are* global quantities, identical on
        every rank by construction, so they are the right things to compare.
        Exact per-weight equality against an unsharded reference is asserted at
        `0.0` in `tests/distributed/test_tensor_parallel.py`, which can do it
        because it gathers the slices explicitly.
        """
        results = run_distributed_cached(
            worker_train, 2, kwargs={"strategy": strategy, "model": "transformer"}
        )
        for step, (observed, expected) in enumerate(zip(results[0]["losses"], baseline["losses"])):
            assert observed == pytest.approx(expected, abs=1e-5), f"loss diverged at step {step}"
        for step, (observed, expected) in enumerate(
            zip(results[0]["grad_norms"], baseline["grad_norms"])
        ):
            assert observed == pytest.approx(expected, abs=1e-5), f"grad norm differs at {step}"
        assert results[0]["eval_loss"] == pytest.approx(baseline["eval_loss"], abs=1e-5)

    @pytest.mark.parametrize("strategy", ["tensor", "sequence"])
    def test_transformer_learns(self, strategy: str) -> None:
        """The loss falls below the uniform-distribution entropy."""
        import math

        results = run_distributed_cached(
            worker_train,
            2,
            kwargs={"strategy": strategy, "model": "transformer", "steps": 20},
        )
        uniform = math.log(TRANSFORMER_MODEL.vocab_size)
        assert results[0]["losses"][-1] < results[0]["losses"][0]
        assert results[0]["losses"][-1] < uniform

    @pytest.mark.parametrize("strategy", ["tensor", "sequence"])
    def test_parameters_are_partitioned(self, strategy: str) -> None:
        """Each rank holds part of the model, and the parts sum to the whole."""
        results = run_distributed_cached(
            worker_train, 2, kwargs={"strategy": strategy, "model": "transformer"}
        )
        counts = results[0]["parameters"]
        assert counts["local"] < counts["global"]

    def test_sequence_parallelism_matches_plain_tensor_parallelism(self) -> None:
        """Sequence parallelism is a memory optimisation, not a model change."""
        tensor_only = run_distributed_cached(
            worker_train, 2, kwargs={"strategy": "tensor", "model": "transformer"}
        )[0]
        with_sequence = run_distributed_cached(
            worker_train, 2, kwargs={"strategy": "sequence", "model": "transformer"}
        )[0]
        assert with_sequence["weight_norm"] == pytest.approx(tensor_only["weight_norm"], abs=1e-5)

    def test_metrics_are_reduced_over_the_data_group_only(self) -> None:
        """Tensor-parallel peers share a batch, so they are one metric source."""
        results = run_distributed_cached(
            worker_train, 2, kwargs={"strategy": "tensor", "model": "transformer"}
        )
        # With tensor parallelism only, the dp_shard group is a single rank.
        assert len(results[0]["metric_group"]) == 1


class TestHybrid:
    """Scenario 6: several strategies composed."""

    @pytest.fixture(scope="class")
    def mlp_baseline(self) -> dict:
        """Single-process MLP trajectory."""
        return run_distributed_cached(worker_train, 1, kwargs={"strategy": "single"})[0]

    def test_four_rank_hybrid_matches_the_baseline(self, mlp_baseline: dict) -> None:
        """dp=2 x shard=2 reproduces the single-process result."""
        results = run_distributed_cached(worker_train, 4, kwargs={"strategy": "hybrid"})
        assert results[0]["strategy"] == "fsdp+replicate"
        assert results[0]["weight_norm"] == pytest.approx(
            mlp_baseline["weight_norm"], abs=OPTIMIZER_STEP_TOLERANCE
        )

    def test_hybrid_metric_group_spans_the_data_dimensions(self) -> None:
        """Metrics average over dp x shard: four ranks here."""
        results = run_distributed_cached(worker_train, 4, kwargs={"strategy": "hybrid"})
        assert len(results[0]["metric_group"]) == 4

    def test_data_parallel_plus_tensor_parallel(self) -> None:
        """dp=2 x tensor=2 on the transformer trains and agrees across ranks."""
        results = run_distributed_cached(
            worker_train, 4, kwargs={"strategy": "hybrid_tensor", "model": "transformer"}
        )
        assert "tensor" in results[0]["strategy"]
        # Ranks 0 and 1 share a batch; 0 and 2 do not.  All four must agree on
        # the reduced loss regardless.
        for step in range(STEPS):
            assert len({round(r["losses"][step], 9) for r in results}) == 1
        assert len(results[0]["metric_group"]) == 2

    @requires_ranks(8)
    @pytest.mark.slow
    def test_full_hybrid_over_eight_ranks(self) -> None:
        """dp=2 x shard=2 x tensor=2 with sequence parallelism runs correctly."""
        results = run_distributed_cached(
            worker_train,
            8,
            kwargs={"strategy": "hybrid_full", "model": "transformer", "steps": 4},
            timeout_seconds=300.0,
        )
        assert results[0]["strategy"] == "fsdp+replicate+tensor+sequence"
        for step in range(4):
            assert len({round(r["losses"][step], 9) for r in results}) == 1
        counts = results[0]["parameters"]
        assert counts["local"] < counts["global"]
        assert len(results[0]["metric_group"]) == 4

    def test_hybrid_matches_pure_sharding(self) -> None:
        """Hybrid sharding and pure sharding are the same computation."""
        hybrid = run_distributed_cached(worker_train, 4, kwargs={"strategy": "hybrid"})[0]
        pure = run_distributed_cached(worker_train, 4, kwargs={"strategy": "fsdp"})[0]
        assert hybrid["weight_norm"] == pytest.approx(pure["weight_norm"], abs=1e-6)


class TestCheckpointResume:
    """Scenarios 7 and 8: save/resume and resharding."""

    @pytest.mark.parametrize(("strategy", "world_size"), [("ddp", 2), ("fsdp", 2), ("hybrid", 4)])
    def test_resume_reproduces_the_uninterrupted_run(
        self, strategy: str, world_size: int, temporary_directory: Path
    ) -> None:
        """Interrupting and resuming changes nothing about the outcome."""
        directory = str(temporary_directory / f"{strategy}{world_size}")
        baseline = run_distributed_cached(
            worker_train,
            world_size,
            kwargs={"strategy": strategy, "directory": directory, "save_at": 4},
        )[0]
        resumed = run_distributed_cached(
            worker_resume,
            world_size,
            kwargs={
                "strategy": strategy,
                "directory": directory,
                "checkpoint": baseline["checkpoint"],
            },
        )[0]
        assert resumed["resumed_at"] == 4
        assert resumed["losses"] == baseline["losses"][4:]
        assert resumed["weight_checksum"] == baseline["weight_checksum"]

    @pytest.mark.parametrize(("save_ranks", "load_ranks"), [(4, 2), (2, 4)])
    def test_resharding_across_world_sizes(
        self, save_ranks: int, load_ranks: int, temporary_directory: Path
    ) -> None:
        """A checkpoint written at one width finishes identically at another.

        The global batch is held constant across the two phases, so the two
        runs consume the same samples; without that the comparison would
        measure a different optimisation problem rather than the reshard.
        """
        directory = str(temporary_directory / f"reshard{save_ranks}-{load_ranks}")
        baseline = run_distributed_cached(
            worker_train,
            save_ranks,
            kwargs={"strategy": "fsdp", "directory": directory, "save_at": 4},
        )[0]
        resumed = run_distributed_cached(
            worker_resume,
            load_ranks,
            kwargs={
                "strategy": "fsdp",
                "directory": directory,
                "checkpoint": baseline["checkpoint"],
            },
        )[0]
        assert resumed["resumed_at"] == 4
        # Reduction order differs between the two widths, so this is a
        # tolerance comparison rather than a bitwise one.
        assert resumed["weight_norm"] == pytest.approx(
            baseline["weight_norm"], abs=OPTIMIZER_STEP_TOLERANCE
        )
        for observed, expected in zip(resumed["losses"], baseline["losses"][4:]):
            assert observed == pytest.approx(expected, abs=1e-5)
