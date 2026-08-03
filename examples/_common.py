"""Shared argument parsing and reporting for the examples.

The examples are meant to be *read*, so anything that is the same in all of
them lives here instead of being copy-pasted five times.  What stays in each
example is only the part that differs: the topology it builds and the thing it
demonstrates.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Any

from hybrid_training.config import ExperimentConfig, load_experiment_config
from hybrid_training.distributed.context import DistributedContext
from hybrid_training.distributed.launch import torchrun_environment_summary
from hybrid_training.logging import configure_logging

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIRECTORY = REPOSITORY_ROOT / "configs"


def build_argument_parser(description: str, default_config: str) -> argparse.ArgumentParser:
    """Return the argument parser shared by every example.

    Args:
        description: Text shown in ``--help``.
        default_config: Config filename under ``configs/``.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        default=str(CONFIG_DIRECTORY / default_config),
        help="YAML experiment configuration to load",
    )
    parser.add_argument("--steps", type=int, default=None, help="override training.max_steps")
    parser.add_argument("--seed", type=int, default=None, help="override training.seed")
    parser.add_argument(
        "--backend",
        default=None,
        choices=["auto", "gloo", "nccl"],
        help="override the communication backend",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["auto", "cpu", "cuda"],
        help="override the compute device",
    )
    parser.add_argument("--checkpoint-dir", default=None, help="override checkpoint.directory")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="logging verbosity",
    )
    parser.add_argument(
        "--show-topology",
        action="store_true",
        help="print this rank's coordinates and group membership, then continue",
    )
    return parser


def load_config(arguments: argparse.Namespace) -> ExperimentConfig:
    """Load the configuration and apply the command-line overrides.

    Args:
        arguments: Parsed arguments from :func:`build_argument_parser`.

    Returns:
        The effective configuration.
    """
    configure_logging(arguments.log_level)
    config = load_experiment_config(arguments.config)

    if arguments.backend is not None:
        config = dataclasses.replace(config, backend=arguments.backend)
    if arguments.device is not None:
        config = dataclasses.replace(config, device=arguments.device)
    if arguments.steps is not None:
        config = dataclasses.replace(
            config, training=dataclasses.replace(config.training, max_steps=arguments.steps)
        )
    if arguments.seed is not None:
        config = dataclasses.replace(
            config, training=dataclasses.replace(config.training, seed=arguments.seed)
        )
    if arguments.checkpoint_dir is not None:
        config = dataclasses.replace(
            config,
            checkpoint=dataclasses.replace(config.checkpoint, directory=arguments.checkpoint_dir),
        )
    return config


def report_environment(context: DistributedContext, *, show_topology: bool) -> None:
    """Print the launch environment and, optionally, this rank's placement.

    Args:
        context: Active distributed context.
        show_topology: Print the full per-rank group membership.
    """
    if context.is_primary:
        summary = torchrun_environment_summary()
        print("launch environment:")
        for name, value in summary.items():
            print(f"  {name:<22} {value}")
    if show_topology:
        print(context.describe())


def report_result(context: DistributedContext, engine: Any) -> None:
    """Print a short training summary from the primary rank.

    Args:
        context: Active distributed context.
        engine: The :class:`~hybrid_training.training.engine.TrainingEngine`.
    """
    # `parameter_count()` all-reduces the local count over the world group, so
    # EVERY rank must call it.  Putting it after the `is_primary` guard meant
    # rank 0 issued an all-reduce that its peers -- already returned and on
    # their way out of the process -- never joined, and the example died with
    # `Connection closed by peer` *after* printing a successful training run.
    #
    # The rule is the same one the checkpoint writer follows: compute
    # collectively, print conditionally.  Anything above this guard must be
    # safe to run on every rank; anything below it must be purely local.
    counts = engine.parameter_count()

    if not context.is_primary:
        return
    print()
    print("=" * 72)
    print(f"strategy            : {engine.model.description.strategy}")
    print(f"world size          : {context.world_size} ({context.backend} on {context.device})")
    print(f"parameters (local)  : {counts['local']:,}")
    print(f"parameters (global) : {counts['global']:,}")
    print(f"final state         : {engine.state.describe()}")
    if engine.history:
        first = engine.history[0].loss
        last = engine.history[-1].loss
        print(f"loss                : {first:.6f} -> {last:.6f}")
    memory = engine.model.memory_summary()
    print(
        "memory (per rank)   : "
        f"shard={memory['shard_bytes']:,}B grad={memory['grad_shard_bytes']:,}B "
        f"padding={memory['padding_bytes']:,}B units={memory['units']}"
    )
    print(engine.communication_summary())
    print("=" * 72)
