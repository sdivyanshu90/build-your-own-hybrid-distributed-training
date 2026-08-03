"""Sharded optimizer and distributed gradient-norm clipping.

Optimizer sharding is (almost) free
===================================
Once :class:`~hybrid_training.parallel.fsdp.FullyShardedDataParallel` has
replaced a module's parameters with a single flat *shard*, the optimizer needs
no special support at all: ``torch.optim.AdamW(fsdp_model.parameters())`` sees
one parameter of ``P/G`` elements per unit, allocates ``2 * P/G`` elements of
state, and updates ``P/G`` values.  The sharding of optimizer state falls out
of the sharding of parameters.

What this module adds on top is the part that is *not* free:

* a **correct distributed gradient norm**, which is the subtlest piece of
  arithmetic in the whole system;
* optional **CPU offload** of optimizer state;
* a **sharded state dict** that names its tensors in a way the checkpoint layer
  can reshard.

The distributed gradient norm
=============================
Gradient clipping needs :math:`\\|g\\|_2` over the *whole* model, but no rank
holds the whole gradient.  The naive fix -- sum every rank's local
:math:`\\|g_{\\text{local}}\\|^2` and take the square root -- is wrong whenever
any parameter is replicated, because a replicated parameter's contribution gets
counted once per rank that holds a copy.

The correct statement is: reduce over the whole world, weighting each
parameter's contribution by the reciprocal of how many ranks hold a copy of it.

.. math::

    \\|g\\|_2^2 = \\sum_{r=0}^{W-1} \\sum_{p \\in P_r}
        \\frac{\\|g_p^{(r)}\\|^2}{\\rho_p}, \\qquad
    \\rho_p = \\frac{W}{\\prod_{d \\in \\text{split}(p)} |d|}

where :math:`\\text{split}(p)` is the set of topology dimensions the parameter
is *partitioned* over.  Worked examples, all with ``W = 4``:

=========================================  ==============  =========  =========
Parameter                                  split over      rho        scale
=========================================  ==============  =========  =========
LayerNorm weight, dp=4                     nothing         4          1/4
LayerNorm weight, dp=2 x tp=2              nothing         4          1/4
column-parallel weight, dp=2 x tp=2        tensor (2)      2          1/2
FSDP flat parameter, shard=4               shard (4)       1          1
FSDP flat parameter, shard=2 x tp=2        shard, tensor   1          1
=========================================  ==============  =========  =========

An FSDP flat parameter is the interesting case: it concatenates
tensor-parallel weight slices (partitioned over ``tensor``) with LayerNorm
weights (replicated over ``tensor``), so a *single* scale for the whole flat
parameter would be wrong.  The scale is therefore computed **per element**,
using the layout the flat parameter already records.  Padding elements get
scale ``0``, which is how the padding stays out of the norm.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from ..config import OptimizerConfig
from ..distributed.collectives import CommunicationRecorder, ReduceOp, all_reduce
from ..distributed.groups import GroupHandle
from ..errors import ShardingError, format_error
from ..logging import get_logger

__all__ = [
    "NormContribution",
    "ShardedOptimizer",
    "build_gradient_norm_contributions",
    "build_inner_optimizer",
    "distributed_gradient_norm",
]

_LOGGER = get_logger(__name__)


def build_inner_optimizer(
    parameters: Iterable[nn.Parameter], config: OptimizerConfig
) -> torch.optim.Optimizer:
    """Construct the underlying PyTorch optimizer.

    Args:
        parameters: Parameters to optimise.  Under FSDP these are the flat
            shards, so the resulting optimizer state is sharded.
        config: Optimizer hyper-parameters.

    Returns:
        A configured ``torch.optim.Optimizer``.

    Raises:
        ShardingError: If there are no parameters to optimise.
    """
    params = [p for p in parameters if p.requires_grad]
    if not params:
        raise ShardingError(
            format_error(
                "optim.build_inner_optimizer",
                "no trainable parameters were supplied",
                expected=">= 1 parameter with requires_grad=True",
                observed=0,
                resolution="check that the model was wrapped before the optimizer was built",
            )
        )
    if config.name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            # `OptimizerConfig.__post_init__` guarantees exactly two values;
            # the cast tells mypy what the validation already enforces.
            betas=(float(config.betas[0]), float(config.betas[1])),
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
    return torch.optim.SGD(
        params,
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )


@dataclass
class NormContribution:
    """One parameter's contribution to the global gradient norm.

    Attributes:
        parameter: The parameter whose gradient is measured.
        scale: Either a scalar weight applied to :math:`\\|g\\|^2`, or a
            per-element weight vector of the same length as the (flattened)
            gradient.  See the module docstring for how it is derived.
        label: Human-readable name, for diagnostics.
    """

    parameter: nn.Parameter
    scale: float | torch.Tensor
    label: str

    def squared_norm(self) -> torch.Tensor:
        """Return this parameter's weighted squared gradient norm.

        Returns:
            A scalar tensor.  Zero when the parameter has no gradient.
        """
        grad = self.parameter.grad
        if grad is None:
            return torch.zeros((), dtype=torch.float64)
        squared = grad.detach().to(torch.float64).pow(2)
        if isinstance(self.scale, torch.Tensor):
            weights = self.scale.to(device=squared.device, dtype=torch.float64)
            return (squared.reshape(-1) * weights).sum()
        return squared.sum() * float(self.scale)


def build_gradient_norm_contributions(
    parameters: Sequence[nn.Parameter], *, world_size: int
) -> list[NormContribution]:
    """Compute the norm weighting for each parameter.

    Args:
        parameters: The optimizer's parameters.
        world_size: Number of ranks the norm will be reduced over.  This must
            be the size of the group passed to
            :func:`distributed_gradient_norm`.

    Returns:
        One :class:`NormContribution` per parameter.

    Raises:
        ShardingError: If ``world_size`` is not positive.
    """
    if world_size < 1:
        raise ShardingError(
            format_error(
                "optim.build_gradient_norm_contributions",
                "world_size must be positive",
                expected=">= 1",
                observed=world_size,
                resolution="pass the size of the reduction group",
            )
        )
    contributions: list[NormContribution] = []
    for index, param in enumerate(parameters):
        scale_vector = getattr(param, "gradient_norm_scale_vector", None)
        if scale_vector is not None:
            # An FSDP flat parameter: mixed partitioned/replicated content, so
            # the weighting varies element by element.
            contributions.append(
                NormContribution(
                    parameter=param,
                    scale=scale_vector.to(torch.float64) / world_size,
                    label=getattr(param, "gradient_norm_label", f"flat_param#{index}"),
                )
            )
            continue
        partitioned = 1
        if not getattr(param, "is_tensor_parallel_replicated", True):
            partitioned = int(getattr(param, "tensor_parallel_group_size", 1))
        contributions.append(
            NormContribution(
                parameter=param,
                scale=partitioned / world_size,
                label=f"param#{index}",
            )
        )
    return contributions


def distributed_gradient_norm(
    contributions: Sequence[NormContribution],
    group: GroupHandle,
    *,
    device: torch.device,
    recorder: CommunicationRecorder | None = None,
) -> torch.Tensor:
    """Compute the global L2 gradient norm across ``group``.

    Args:
        contributions: Per-parameter weightings from
            :func:`build_gradient_norm_contributions`.
        group: Group to reduce over.  Must be the group whose size was used to
            build the contributions -- normally ``"world"``, because a hybrid
            job partitions parameters over several dimensions at once.
        device: Device the reduction temporary lives on.
        recorder: Optional instrumentation sink.

    Returns:
        A scalar ``float64`` tensor holding :math:`\\|g\\|_2`.
    """
    total = torch.zeros((), dtype=torch.float64, device=device)
    for contribution in contributions:
        total = total + contribution.squared_norm().to(device)
    buffer = total.reshape(1).contiguous()
    all_reduce(buffer, group, op=ReduceOp.SUM, recorder=recorder).wait()
    return buffer[0].clamp_min(0).sqrt()


class ShardedOptimizer:
    """Optimizer wrapper adding distributed clipping and optional CPU offload.

    Args:
        parameters: Parameters to optimise.  Under FSDP these are flat shards.
        config: Optimizer hyper-parameters.
        norm_group: Group the gradient norm is reduced over.  **Required** --
            the correct group in a hybrid job is the *world*, and defaulting to
            anything else silently produces a wrong norm.
        device: Compute device.
        recorder: Optional instrumentation sink.

    Raises:
        ShardingError: If no trainable parameters are supplied.

    Example:
        >>> # doctest: +SKIP
        >>> optimizer = ShardedOptimizer(model.parameters(), OptimizerConfig(),
        ...                              norm_group=ctx.group("world"),
        ...                              device=ctx.device)
        >>> loss.backward()
        >>> total_norm = optimizer.clip_grad_norm(1.0)
        >>> optimizer.step()
        >>> optimizer.zero_grad()
    """

    def __init__(
        self,
        parameters: Iterable[nn.Parameter],
        config: OptimizerConfig,
        *,
        norm_group: GroupHandle,
        device: torch.device,
        recorder: CommunicationRecorder | None = None,
    ) -> None:
        self._parameters = [p for p in parameters if p.requires_grad]
        self._config = config
        self._norm_group = norm_group
        self._device = device
        self._recorder = recorder
        self._offload = config.cpu_offload_state and device.type != "cpu"

        if self._offload:
            # Master copies live on the CPU; the optimizer only ever touches
            # them, so no optimizer state is allocated on the device at all.
            self._cpu_parameters = [
                nn.Parameter(p.detach().to("cpu").clone(), requires_grad=True)
                for p in self._parameters
            ]
            self._optimizer = build_inner_optimizer(self._cpu_parameters, config)
        else:
            self._cpu_parameters = []
            self._optimizer = build_inner_optimizer(self._parameters, config)

        self._contributions = build_gradient_norm_contributions(
            self._parameters, world_size=norm_group.size
        )
        _LOGGER.debug(
            "sharded optimizer over %d parameters (%d elements), norm group %r, offload=%s",
            len(self._parameters),
            sum(p.numel() for p in self._parameters),
            norm_group.name,
            self._offload,
        )

    # -- optimisation -------------------------------------------------------
    def clip_grad_norm(self, max_norm: float) -> torch.Tensor:
        """Clip gradients to a global L2 norm and return the pre-clip norm.

        Every rank computes the *same* total norm (it is all-reduced), so every
        rank applies the *same* scaling factor.  A per-rank norm would scale
        the shards of one parameter differently and silently change the update
        direction.

        Args:
            max_norm: Clipping threshold.  ``0`` or negative disables clipping,
                and the norm is still computed and returned.

        Returns:
            A scalar tensor with the global gradient norm before clipping.
        """
        total_norm = distributed_gradient_norm(
            self._contributions,
            self._norm_group,
            device=self._device,
            recorder=self._recorder,
        )
        if max_norm <= 0:
            return total_norm
        clip_coefficient = float(max_norm) / (float(total_norm.item()) + 1e-6)
        if clip_coefficient < 1.0:
            for param in self._parameters:
                if param.grad is not None:
                    param.grad.detach().mul_(clip_coefficient)
        return total_norm

    def step(self) -> None:
        """Apply one optimizer step to this rank's shards."""
        if self._offload:
            for device_param, cpu_param in zip(self._parameters, self._cpu_parameters):
                if device_param.grad is None:
                    cpu_param.grad = None
                else:
                    if cpu_param.grad is None:
                        cpu_param.grad = torch.empty_like(cpu_param)
                    cpu_param.grad.copy_(device_param.grad.detach().to("cpu"))
            self._optimizer.step()
            with torch.no_grad():
                for device_param, cpu_param in zip(self._parameters, self._cpu_parameters):
                    device_param.data.copy_(cpu_param.data.to(device_param.device))
        else:
            self._optimizer.step()

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        """Clear gradients.

        Args:
            set_to_none: Release the gradient tensors instead of zeroing them.
                Releasing is faster and saves memory; zeroing keeps the buffers
                so the next accumulation reuses them.
        """
        for param in self._parameters:
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                param.grad.detach().zero_()
        if self._offload:
            for param in self._cpu_parameters:
                param.grad = None

    # -- introspection ------------------------------------------------------
    @property
    def parameters(self) -> tuple[nn.Parameter, ...]:
        """The parameters this optimizer updates."""
        return tuple(self._parameters)

    @property
    def inner(self) -> torch.optim.Optimizer:
        """The underlying PyTorch optimizer."""
        return self._optimizer

    @property
    def norm_group(self) -> GroupHandle:
        """The group the gradient norm is reduced over."""
        return self._norm_group

    @property
    def learning_rate(self) -> float:
        """Current learning rate of the first parameter group."""
        return float(self._optimizer.param_groups[0]["lr"])

    def set_learning_rate(self, value: float) -> None:
        """Set the learning rate on every parameter group.

        Args:
            value: New learning rate.
        """
        for group in self._optimizer.param_groups:
            group["lr"] = value

    def state_bytes(self) -> int:
        """Total bytes of optimizer state held by this rank.

        Returns:
            Sum over every state tensor.  Under FSDP this is ``1/G`` of what
            a replicated optimizer would hold, which is the number the memory
            tests assert on.
        """
        total = 0
        for state in self._optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    total += value.numel() * value.element_size()
        return total

    # -- serialisation ------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """Return this rank's optimizer state, keyed by parameter index.

        Keys are positional indices rather than the opaque integers PyTorch
        uses, so the checkpoint layer can pair a state tensor with the
        parameter shard it belongs to and reshard both together.

        Returns:
            ``{"param_groups": [...], "state": {index: {name: tensor|scalar}}}``.
        """
        inner = self._optimizer.state_dict()
        state: dict[int, dict[str, Any]] = {}
        for index, entry in inner["state"].items():
            state[int(index)] = {
                key: (value.detach().cpu().clone() if torch.is_tensor(value) else value)
                for key, value in entry.items()
            }
        return {
            "param_groups": [
                {k: v for k, v in group.items() if k != "params"} for group in inner["param_groups"]
            ],
            "state": state,
            "num_parameters": len(self._parameters),
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        """Restore optimizer state produced by :meth:`state_dict`.

        Args:
            payload: The mapping to restore.

        Raises:
            ShardingError: If the payload describes a different number of
                parameters than this optimizer manages.
        """
        expected = len(self._parameters)
        observed = int(payload.get("num_parameters", -1))
        if observed != expected:
            raise ShardingError(
                format_error(
                    "optim.load_state_dict",
                    "the checkpoint holds optimizer state for a different number of "
                    "parameters, so the state cannot be matched to the current model",
                    rank=self._norm_group.global_rank,
                    expected=expected,
                    observed=observed,
                    resolution=(
                        "resume with the same model definition and the same wrapping "
                        "granularity that produced the checkpoint"
                    ),
                )
            )
        target = self._cpu_parameters if self._offload else self._parameters
        inner_groups = self._optimizer.state_dict()["param_groups"]
        rebuilt_groups: list[dict[str, Any]] = []
        for saved, existing in zip(payload["param_groups"], inner_groups):
            merged = dict(existing)
            merged.update(saved)
            merged["params"] = existing["params"]
            rebuilt_groups.append(merged)

        state: dict[int, dict[str, Any]] = {}
        for index, entry in payload["state"].items():
            index = int(index)
            reference = target[index]
            state[index] = {
                key: (
                    value.to(device=reference.device, dtype=reference.dtype)
                    if torch.is_tensor(value) and value.dtype.is_floating_point
                    else value
                )
                for key, value in entry.items()
            }
        self._optimizer.load_state_dict({"state": state, "param_groups": rebuilt_groups})

    def __repr__(self) -> str:
        return (
            f"ShardedOptimizer({self._config.name}, params={len(self._parameters)}, "
            f"norm_group={self._norm_group.name!r}, offload={self._offload})"
        )
