"""Unit tests for configuration validation and serialisation.

A configuration error that is *not* caught here becomes a distributed hang or a
silently wrong reduction later, so the tests focus on the rejections rather
than the happy path.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch
import yaml

from hybrid_training.config import (
    CheckpointConfig,
    DataConfig,
    DDPConfig,
    ExperimentConfig,
    FSDPConfig,
    GradScalerConfig,
    MixedPrecisionConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    SequenceParallelMode,
    TensorParallelConfig,
    TopologyConfig,
    TrainingConfig,
    load_experiment_config,
    resolve_dtype,
)
from hybrid_training.errors import ConfigurationError

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIRECTORY = REPOSITORY_ROOT / "configs"


class TestDtypeResolution:
    """dtype name handling."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("float32", torch.float32),
            ("fp32", torch.float32),
            ("bfloat16", torch.bfloat16),
            ("bf16", torch.bfloat16),
            ("float16", torch.float16),
            ("half", torch.float16),
            ("torch.float64", torch.float64),
        ],
    )
    def test_known_names(self, name: str, expected: torch.dtype) -> None:
        """Every documented alias resolves."""
        assert resolve_dtype(name) is expected

    def test_dtype_passes_through(self) -> None:
        """An actual dtype is returned unchanged."""
        assert resolve_dtype(torch.float32) is torch.float32

    def test_unknown_name_rejected(self) -> None:
        """A typo raises with the list of valid names."""
        with pytest.raises(ConfigurationError, match="unknown dtype name"):
            resolve_dtype("float8")


class TestSubsystemValidation:
    """Per-dataclass rejections."""

    def test_ddp_bucket_size_positive(self) -> None:
        """A zero bucket cap would make every parameter its own bucket."""
        with pytest.raises(ConfigurationError, match="bucket_cap_mb must be positive"):
            DDPConfig(bucket_cap_mb=0)

    def test_ddp_source_rank_non_negative(self) -> None:
        """The broadcast source is a group-local index, never negative."""
        with pytest.raises(ConfigurationError, match="source_rank_in_group"):
            DDPConfig(source_rank_in_group=-1)

    def test_fsdp_unpadded_rejected(self) -> None:
        """Disabling padding is explicitly unsupported, with a reason."""
        with pytest.raises(ConfigurationError, match="use_padding=False is not supported"):
            FSDPConfig(use_padding=False)

    def test_fsdp_negative_thresholds_rejected(self) -> None:
        """Auto-wrap and guard-rail thresholds must be non-negative."""
        with pytest.raises(ConfigurationError, match="auto_wrap_min_num_params"):
            FSDPConfig(auto_wrap_min_num_params=-1)
        with pytest.raises(ConfigurationError, match="limit_all_gather_bytes"):
            FSDPConfig(limit_all_gather_bytes=-1)

    def test_model_head_divisibility(self) -> None:
        """A transformer's hidden size must divide by its head count."""
        with pytest.raises(ConfigurationError, match="divisible by num_heads"):
            ModelConfig(kind="transformer", hidden_size=10, num_heads=4)

    def test_model_kind_and_activation(self) -> None:
        """Unknown model kinds and activations are rejected."""
        with pytest.raises(ConfigurationError, match="unknown model kind"):
            ModelConfig(kind="rnn")
        with pytest.raises(ConfigurationError, match="unknown activation"):
            ModelConfig(activation="swish")

    def test_dropout_range(self) -> None:
        """Dropout must be a probability below 1."""
        with pytest.raises(ConfigurationError, match=r"dropout must lie in \[0, 1\)"):
            ModelConfig(dropout=1.0)

    def test_optimizer_validation(self) -> None:
        """Unknown optimizers, bad learning rates and bad betas are rejected."""
        with pytest.raises(ConfigurationError, match="unknown optimizer"):
            OptimizerConfig(name="lamb")
        with pytest.raises(ConfigurationError, match="learning_rate must be positive"):
            OptimizerConfig(learning_rate=0.0)
        with pytest.raises(ConfigurationError, match="betas must be two values"):
            OptimizerConfig(betas=(0.9, 1.5))

    def test_optimizer_betas_normalised_to_tuple(self) -> None:
        """A list from YAML becomes a tuple, so configs compare equal."""
        assert OptimizerConfig(betas=[0.9, 0.95]).betas == (0.9, 0.95)  # type: ignore[arg-type]

    def test_scheduler_validation(self) -> None:
        """Unknown schedules and out-of-range ratios are rejected."""
        with pytest.raises(ConfigurationError, match="unknown schedule"):
            SchedulerConfig(name="exponential")
        with pytest.raises(ConfigurationError, match=r"min_lr_ratio must lie in \[0, 1\]"):
            SchedulerConfig(min_lr_ratio=1.5)

    def test_training_validation(self) -> None:
        """Accumulation and clipping thresholds are validated."""
        with pytest.raises(ConfigurationError, match="gradient_accumulation_steps"):
            TrainingConfig(gradient_accumulation_steps=0)
        with pytest.raises(ConfigurationError, match="max_grad_norm must be non-negative"):
            TrainingConfig(max_grad_norm=-1.0)

    def test_data_validation(self) -> None:
        """Batch and dataset sizes are validated."""
        with pytest.raises(ConfigurationError, match="micro_batch_size must be positive"):
            DataConfig(micro_batch_size=0)
        with pytest.raises(ConfigurationError, match="num_eval_samples must be non-negative"):
            DataConfig(num_eval_samples=-1)

    def test_checkpoint_validation(self) -> None:
        """Cadence and retention counts are validated."""
        with pytest.raises(ConfigurationError, match="save_every_steps"):
            CheckpointConfig(save_every_steps=-1)
        with pytest.raises(ConfigurationError, match="keep_last"):
            CheckpointConfig(keep_last=-1)

    def test_grad_scaler_validation(self) -> None:
        """Scaler growth/backoff parameters are validated."""
        with pytest.raises(ConfigurationError, match="init_scale must be positive"):
            GradScalerConfig(init_scale=0)
        with pytest.raises(ConfigurationError, match="growth_factor must exceed 1"):
            GradScalerConfig(growth_factor=1.0)
        with pytest.raises(ConfigurationError, match=r"backoff_factor must lie in \(0, 1\)"):
            GradScalerConfig(backoff_factor=1.5)

    def test_tensor_parallel_init_std(self) -> None:
        """Initialisation spread must be positive."""
        with pytest.raises(ConfigurationError, match="init_method_std must be positive"):
            TensorParallelConfig(init_method_std=0.0)


class TestMixedPrecision:
    """Precision-policy validation and derived dtypes."""

    def test_loss_scaling_requires_fp16(self) -> None:
        """bf16 does not underflow the way fp16 does, so scaling is refused."""
        with pytest.raises(ConfigurationError, match="only meaningful for float16"):
            MixedPrecisionConfig(
                enabled=True,
                param_dtype="bfloat16",
                scaler=GradScalerConfig(enabled=True),
            )

    def test_fp16_with_scaling_is_accepted(self) -> None:
        """The supported fp16 combination constructs cleanly."""
        config = MixedPrecisionConfig(
            enabled=True, param_dtype="float16", scaler=GradScalerConfig(enabled=True)
        )
        assert config.compute_dtype is torch.float16
        assert config.gradient_reduce_dtype is torch.float32

    def test_disabled_policy_uses_master_dtype(self) -> None:
        """With mixed precision off, compute happens in the master dtype."""
        config = MixedPrecisionConfig(enabled=False, param_dtype="bfloat16")
        assert config.compute_dtype is torch.float32

    def test_bad_dtype_name_caught_at_construction(self) -> None:
        """A dtype typo fails when the config is built, not at the first cast."""
        with pytest.raises(ConfigurationError, match="unknown dtype name"):
            MixedPrecisionConfig(reduce_dtype="float8")


class TestExperimentConfig:
    """Top-level cross-section validation."""

    def test_gloo_on_cuda_rejected(self) -> None:
        """Gloo on CUDA tensors is never the right choice and is refused."""
        with pytest.raises(ConfigurationError, match="Gloo backend on CUDA"):
            ExperimentConfig(backend="gloo", device="cuda")

    def test_unknown_backend_and_device(self) -> None:
        """Backend and device names are validated."""
        with pytest.raises(ConfigurationError, match="unknown backend"):
            ExperimentConfig(backend="mpi")
        with pytest.raises(ConfigurationError, match="unknown device"):
            ExperimentConfig(device="tpu")

    def test_sequence_parallel_flags_must_agree(self) -> None:
        """Enabling SP in the layers but not the topology is rejected."""
        with pytest.raises(ConfigurationError, match="does not create a sequence-parallel"):
            ExperimentConfig(
                topology=TopologyConfig(tensor_parallel_size=2),
                tensor_parallel=TensorParallelConfig(sequence_parallel=True),
            )

    def test_consistent_sequence_parallel_config_is_accepted(self) -> None:
        """The matching combination constructs cleanly."""
        config = ExperimentConfig(
            topology=TopologyConfig(
                tensor_parallel_size=2,
                sequence_parallel_mode=SequenceParallelMode.TENSOR_GROUP,
            ),
            tensor_parallel=TensorParallelConfig(sequence_parallel=True),
        )
        assert config.topology.sequence_parallel_enabled

    def test_timeout_must_be_positive(self) -> None:
        """A non-positive collective timeout is rejected."""
        with pytest.raises(ConfigurationError, match="timeout_seconds must be positive"):
            ExperimentConfig(timeout_seconds=0)


class TestSerialisation:
    """Dictionary and YAML round-tripping."""

    def test_round_trip_through_dict(self) -> None:
        """``from_dict(to_dict(c)) == c`` for a non-trivial configuration."""
        original = ExperimentConfig(
            name="round-trip",
            topology=TopologyConfig(data_parallel_size=2, tensor_parallel_size=2),
            model=ModelConfig(kind="transformer", hidden_size=32, num_heads=4),
            training=TrainingConfig(max_steps=7, max_grad_norm=0.5),
        )
        assert ExperimentConfig.from_dict(original.to_dict()) == original

    def test_round_trip_through_yaml(self, tmp_path: Path) -> None:
        """A config survives a YAML write/read cycle unchanged."""
        original = ExperimentConfig(name="yaml", training=TrainingConfig(max_steps=3))
        path = original.to_yaml(tmp_path / "config.yaml")
        assert load_experiment_config(path) == original

    def test_unknown_top_level_key_rejected(self) -> None:
        """An unknown key is an error, never silently ignored."""
        with pytest.raises(ConfigurationError, match="unknown configuration key"):
            ExperimentConfig.from_dict({"nmae": "typo"})

    def test_unknown_nested_key_rejected_with_path(self) -> None:
        """The error names the section the bad key was in."""
        with pytest.raises(ConfigurationError, match="topology"):
            ExperimentConfig.from_dict({"topology": {"data_paralel_size": 2}})

    def test_non_mapping_section_rejected(self) -> None:
        """A scalar where a section is expected reports an indentation problem."""
        with pytest.raises(ConfigurationError, match="expected a mapping"):
            ExperimentConfig.from_dict({"topology": 4})

    def test_missing_file_rejected(self) -> None:
        """A missing config file names the path it looked for."""
        with pytest.raises(ConfigurationError, match="configuration file not found"):
            load_experiment_config("/nonexistent/config.yaml")

    def test_overrides_applied_after_parsing(self, tmp_path: Path) -> None:
        """Keyword overrides replace top-level fields."""
        path = ExperimentConfig(name="base").to_yaml(tmp_path / "c.yaml")
        assert load_experiment_config(path, name="overridden").name == "overridden"


class TestShippedConfigs:
    """Every YAML under ``configs/`` must load and be internally consistent."""

    @pytest.mark.parametrize(
        "filename",
        sorted(p.name for p in CONFIG_DIRECTORY.glob("*.yaml")),
    )
    def test_config_loads_and_validates(self, filename: str) -> None:
        """A shipped config parses, and its topology factors its world size."""
        config = load_experiment_config(CONFIG_DIRECTORY / filename)
        config.topology.validate_against_world_size(config.topology.world_size)
        assert config.topology.world_size >= 1

    @pytest.mark.parametrize(
        ("filename", "expected_world_size"),
        [
            ("single_process.yaml", 1),
            ("ddp_2gpu.yaml", 2),
            ("fsdp_4gpu.yaml", 4),
            ("tensor_parallel_2gpu.yaml", 2),
            ("sequence_parallel_2gpu.yaml", 2),
            ("hybrid_8gpu.yaml", 8),
        ],
    )
    def test_world_sizes_match_their_names(self, filename: str, expected_world_size: int) -> None:
        """A config named ``*_4gpu`` really does describe four ranks."""
        config = load_experiment_config(CONFIG_DIRECTORY / filename)
        assert config.topology.world_size == expected_world_size

    def test_no_config_contains_unknown_keys(self) -> None:
        """Strict parsing means this is implied, but assert it per file."""
        for path in sorted(CONFIG_DIRECTORY.glob("*.yaml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            ExperimentConfig.from_dict(raw)

    def test_tensor_parallel_configs_have_divisible_models(self) -> None:
        """Head and feed-forward widths divide by the tensor-parallel size."""
        for path in sorted(CONFIG_DIRECTORY.glob("*.yaml")):
            config = load_experiment_config(path)
            width = config.topology.tensor_parallel_size
            if width == 1 or config.model.kind != "transformer":
                continue
            assert config.model.num_heads % width == 0, path.name
            assert config.model.resolved_ffn_hidden_size % width == 0, path.name
            assert config.model.vocab_size % width == 0, path.name


def test_dataclasses_replace_preserves_validation() -> None:
    """``dataclasses.replace`` re-runs ``__post_init__``, so it cannot bypass checks."""
    config = ExperimentConfig()
    with pytest.raises(ConfigurationError, match="unknown backend"):
        dataclasses.replace(config, backend="mpi")


def test_resolved_ffn_and_head_dim() -> None:
    """Derived model properties compute as documented."""
    model = ModelConfig(kind="transformer", hidden_size=64, num_heads=8, ffn_hidden_size=0)
    assert model.resolved_ffn_hidden_size == 256
    assert model.head_dim == 8
