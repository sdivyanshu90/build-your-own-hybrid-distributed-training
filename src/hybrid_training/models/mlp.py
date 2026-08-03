"""A small MLP used as the fast-path reference model.

Why an MLP is the right model for correctness tests
---------------------------------------------------
Every numerical-equivalence assertion in this project has the form "the
distributed result equals the single-process result to within tolerance ``t``".
The tighter ``t`` can be, the more bugs the test can catch.  Tolerance is set by
floating-point non-associativity, which grows with depth and with the number of
terms in each reduction.  A 2-to-4 layer MLP over 32-to-64 features accumulates
so little error that ``atol=1e-6`` in fp32 is comfortable, which is tight
enough to catch a wrong reduction group, a missing division, or an off-by-one
in a shard offset.

The MLP is also cheap enough to run four ranks of it on a laptop CPU under
Gloo, which is what makes the distributed test suite runnable in CI without
GPUs.

Structure::

    x -> [Linear(in, hidden) -> act] -> [Linear(hidden, hidden) -> act] * (L-1)
      -> Linear(hidden, out)

The optional ``tensor_parallel_group`` argument replaces the hidden-layer pairs
with a column-parallel/row-parallel pair, which is the same partitioning a
transformer's feed-forward block uses (see
:mod:`hybrid_training.parallel.tensor_parallel`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from ..config import ModelConfig
from ..distributed.groups import GroupHandle
from ..errors import ConfigurationError, format_error
from ..utils.reproducibility import derive_seed, temporary_seed

__all__ = ["MLP", "MLPBlock", "build_activation"]


def build_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the activation function named by ``name``.

    Args:
        name: ``"gelu"`` or ``"relu"``.

    Returns:
        The functional activation.

    Raises:
        ConfigurationError: If the name is unknown.
    """
    if name == "gelu":
        # 'tanh' approximation is deterministic across CPU/CUDA builds, which
        # matters for the cross-device equivalence assertions.
        return lambda x: torch.nn.functional.gelu(x, approximate="tanh")
    if name == "relu":
        return torch.nn.functional.relu
    raise ConfigurationError(
        format_error(
            "models.build_activation",
            "unknown activation",
            expected=["gelu", "relu"],
            observed=name,
            resolution="use 'gelu' or 'relu'",
        )
    )


class MLPBlock(nn.Module):
    """One hidden block: ``Linear -> activation``.

    Kept as a separate module so that FSDP auto-wrapping and nested wrapping
    have a natural boundary to wrap at, and so the tensor-parallel variant can
    swap the linear layer out without touching the rest of the model.

    Args:
        in_features: Input width.
        out_features: Output width.
        activation: Activation name.
        bias: Whether the linear layer has a bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        activation: str = "gelu",
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.activation_name = activation
        self._activation = build_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear layer followed by the activation.

        Args:
            x: Input of shape ``(..., in_features)``.

        Returns:
            Output of shape ``(..., out_features)``.
        """
        return self._activation(self.linear(x))


class MLP(nn.Module):
    """Configurable multi-layer perceptron.

    Args:
        config: Model configuration.  ``input_size``, ``hidden_size``,
            ``num_layers``, ``output_size`` and ``activation`` are used.
        seed: Seed for weight initialisation.  Every rank that holds a
            *replica* of this model must pass the same seed; the DDP and FSDP
            wrappers additionally broadcast parameters at construction time, so
            a mismatched seed is corrected rather than silently tolerated.
        bias: Whether the linear layers have biases.
        tensor_parallel_group: When given, the hidden blocks are built from
            column- and row-parallel linear layers over this group.
        device: Device to construct on.
        dtype: Parameter dtype.

    Example:
        >>> from hybrid_training.config import ModelConfig
        >>> model = MLP(ModelConfig(input_size=8, hidden_size=16, num_layers=2,
        ...                         output_size=4), seed=0)
        >>> model(torch.zeros(3, 8)).shape
        torch.Size([3, 4])
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        seed: int = 0,
        bias: bool = True,
        tensor_parallel_group: GroupHandle | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        # `num_layers >= 1` is guaranteed by ModelConfig.__post_init__; a
        # re-check here would be unreachable code, so the invariant is stated
        # rather than re-tested.  See tests/unit/test_single_process_components.py
        # ::TestModels::test_mlp_requires_at_least_one_layer.
        self.config = config
        self.tensor_parallel_group = tensor_parallel_group

        factory: dict[str, Any] = {}
        if device is not None:
            factory["device"] = device
        if dtype is not None:
            factory["dtype"] = dtype

        # Initialising inside `temporary_seed` makes construction independent of
        # whatever random draws happened earlier in the process, so two models
        # built at different points in a script are still identical.
        with temporary_seed(derive_seed(seed, "model-init")):
            blocks: list[nn.Module] = []
            widths = [config.input_size] + [config.hidden_size] * config.num_layers
            if tensor_parallel_group is None or tensor_parallel_group.size == 1:
                for index in range(config.num_layers):
                    blocks.append(
                        MLPBlock(
                            widths[index],
                            widths[index + 1],
                            activation=config.activation,
                            bias=bias,
                        )
                    )
            else:
                from ..parallel.tensor_parallel import TensorParallelMLPBlock

                for index in range(config.num_layers):
                    blocks.append(
                        TensorParallelMLPBlock(
                            widths[index],
                            widths[index + 1],
                            group=tensor_parallel_group,
                            activation=config.activation,
                            bias=bias,
                            **factory,
                        )
                    )
            self.blocks = nn.ModuleList(blocks)
            self.head = nn.Linear(config.hidden_size, config.output_size, bias=bias, **factory)

        # Move/cast the plain layers after construction so the RNG draw is
        # identical regardless of the target device.  The parallel layers were
        # already built on the requested device, so this applies only to the
        # unpartitioned path.
        if (tensor_parallel_group is None or tensor_parallel_group.size == 1) and (
            device is not None or dtype is not None
        ):
            self.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP.

        Args:
            x: Input of shape ``(batch, input_size)`` (or any leading dims).

        Returns:
            Output of shape ``(..., output_size)``.
        """
        for block in self.blocks:
            x = block(x)
        return self.head(x)

    def num_parameters(self) -> int:
        """Total number of parameter elements held by this rank."""
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        tp = self.tensor_parallel_group.size if self.tensor_parallel_group is not None else 1
        return (
            f"MLP(in={self.config.input_size}, hidden={self.config.hidden_size}, "
            f"layers={self.config.num_layers}, out={self.config.output_size}, "
            f"tensor_parallel={tp})"
        )
