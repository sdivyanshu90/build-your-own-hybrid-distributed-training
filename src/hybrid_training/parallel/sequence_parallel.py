"""Sequence parallelism: splitting activations along the sequence dimension.

What it is, and what it is not
==============================
Activations in a transformer are shaped ``(batch, sequence, hidden)``.  The
five ways to cut that tensor across ranks are routinely confused, so:

======================  ==============================  =========================
Strategy                What is split                   What is replicated
======================  ==============================  =========================
data parallel           ``batch``                       all parameters,
                                                        all activations
                                                        within a sample
**sequence parallel**   ``sequence``, in the regions    parameters
                        *between* tensor-parallel
                        layers
tensor parallel         ``hidden`` (and the weight      the activations
                        matrices)                       entering/leaving a
                                                        parallel region
context parallel        ``sequence``, **including       parameters
                        inside attention**, using a
                        ring/all-to-all exchange of
                        keys and values
pipeline parallel       ``layers``                      nothing within a stage
======================  ==============================  =========================

The distinction between *sequence* and *context* parallelism is the one that
matters most here.  Sequence parallelism splits the sequence only where the
computation is **pointwise along the sequence** -- LayerNorm, dropout, residual
adds, elementwise activations.  Attention needs every position to see every
other position, so a sequence-parallel implementation **gathers the sequence
back** before attention.  Context parallelism is the harder technique that
keeps the sequence split *through* attention by exchanging keys and values
between ranks; it is out of scope here and is discussed as a future extension
in ``docs/06_sequence_parallelism.md``.

This module therefore makes no claim that attention is communication-free under
sequence sharding.  It is not.  The baseline strategy implemented here gathers
the sequence at the entrance to each tensor-parallel region -- which, in the
fused Megatron schedule, is *free*, because the column-parallel layer at that
entrance had to communicate anyway.

Why sequence parallelism is worth it
====================================
Tensor parallelism leaves the LayerNorm/dropout/residual regions replicated:
every one of the ``T`` ranks stores the same ``(B, S, H)`` activation.  Sequence
parallelism stores ``(B, S/T, H)`` on each rank instead, cutting that class of
activation memory by ``T``.  The communication cost is *zero* when fused with
tensor parallelism: an all-reduce of ``(B, S, H)`` is replaced by a
reduce-scatter of ``(B, S, H)`` plus an all-gather of ``(B, S, H)``, and

.. math::

    \\text{all-reduce} \\;\\equiv\\; \\text{reduce-scatter} + \\text{all-gather}

moves the same number of bytes.  Sequence parallelism is, in that sense, free
activation memory.

Adjoints
========
Every operation here is a differentiable collective whose adjoint is stated
exactly:

========================  ======================  ======================
Operation                 Forward                 Backward
========================  ======================  ======================
``scatter_sequence``      keep local slice        all-gather
``gather_sequence``       all-gather              keep local slice
``reduce_scatter_sequence``  sum + keep slice     all-gather
========================  ======================  ======================
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..autograd.collectives import (
    gather_from_group,
    reduce_scatter_to_group,
    scatter_to_group,
)
from ..distributed.groups import GroupHandle
from ..errors import ShardingError, format_error

__all__ = [
    "SEQUENCE_DIM",
    "SequenceParallelLayerNorm",
    "SequenceShardInfo",
    "gather_sequence",
    "local_sequence_slice",
    "pad_sequence_dimension",
    "reduce_scatter_sequence",
    "scatter_sequence",
    "unpad_sequence_dimension",
]

#: Activations are ``(batch, sequence, hidden)`` throughout this project, so the
#: sequence axis is always 1.  Naming it means no call site contains a bare
#: ``dim=1`` whose meaning has to be guessed.
SEQUENCE_DIM = 1


@dataclass(frozen=True)
class SequenceShardInfo:
    """Bookkeeping for a sequence split that required padding.

    A sequence length that is not divisible by the group size cannot be split
    evenly, and the equal-size collectives this project uses require even
    splits.  The sequence is therefore right-padded with zeros to the next
    multiple, and this record carries what is needed to undo that.

    Attributes:
        original_length: Sequence length before padding.
        padded_length: Length after padding; a multiple of ``group_size``.
        group_size: Number of ranks the sequence is split across.
        local_length: Elements per rank, ``padded_length // group_size``.

    Example:
        >>> info = SequenceShardInfo.for_length(10, 4)
        >>> (info.padded_length, info.local_length, info.padding)
        (12, 3, 2)
    """

    original_length: int
    padded_length: int
    group_size: int
    local_length: int

    @classmethod
    def for_length(cls, length: int, group_size: int) -> SequenceShardInfo:
        """Compute the padding needed to split ``length`` across ``group_size``.

        Args:
            length: Sequence length.
            group_size: Number of shards.

        Returns:
            The shard bookkeeping.

        Raises:
            ShardingError: If ``group_size`` is not positive.
        """
        if group_size < 1:
            raise ShardingError(
                format_error(
                    "sequence_parallel.SequenceShardInfo",
                    "group size must be positive",
                    expected=">= 1",
                    observed=group_size,
                    resolution="pass the sequence-parallel group size",
                )
            )
        padded = ((length + group_size - 1) // group_size) * group_size
        return cls(
            original_length=length,
            padded_length=padded,
            group_size=group_size,
            local_length=padded // group_size,
        )

    @property
    def padding(self) -> int:
        """Number of padding positions appended to the sequence."""
        return self.padded_length - self.original_length

    @property
    def requires_padding(self) -> bool:
        """Whether any padding was needed."""
        return self.padding > 0

    def local_range(self, local_rank: int) -> tuple[int, int]:
        """Return the ``[start, end)`` positions owned by ``local_rank``.

        Args:
            local_rank: Index within the sequence-parallel group.

        Returns:
            Half-open interval in *padded* coordinates.
        """
        start = local_rank * self.local_length
        return start, start + self.local_length


def pad_sequence_dimension(
    x: torch.Tensor, group_size: int, *, dim: int = SEQUENCE_DIM
) -> tuple[torch.Tensor, SequenceShardInfo]:
    """Right-pad ``x`` along ``dim`` so it splits evenly across ``group_size``.

    Zero padding is used, and it is the caller's responsibility to make sure
    the padded positions are masked out of any attention computation -- which
    :class:`~hybrid_training.models.transformer.TransformerBlock` does via its
    attention mask.

    Args:
        x: Activation tensor.
        group_size: Number of shards.
        dim: Sequence dimension.

    Returns:
        ``(padded_tensor, info)``.  When no padding is needed the original
        tensor is returned unchanged.
    """
    info = SequenceShardInfo.for_length(x.shape[dim], group_size)
    if not info.requires_padding:
        return x, info
    pad_shape = list(x.shape)
    pad_shape[dim] = info.padding
    padding = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, padding], dim=dim), info


def unpad_sequence_dimension(
    x: torch.Tensor, info: SequenceShardInfo, *, dim: int = SEQUENCE_DIM
) -> torch.Tensor:
    """Remove the padding added by :func:`pad_sequence_dimension`.

    Args:
        x: Tensor whose ``dim`` has length ``info.padded_length``.
        info: The bookkeeping returned when padding.
        dim: Sequence dimension.

    Returns:
        A tensor whose ``dim`` has length ``info.original_length``.

    Raises:
        ShardingError: If ``x`` does not have the padded length along ``dim``.
    """
    if x.shape[dim] != info.padded_length:
        raise ShardingError(
            format_error(
                "sequence_parallel.unpad_sequence_dimension",
                "tensor does not have the padded sequence length",
                expected=info.padded_length,
                observed=x.shape[dim],
                resolution="pass the same SequenceShardInfo that produced the padding",
            )
        )
    if not info.requires_padding:
        return x
    return x.narrow(dim, 0, info.original_length)


def scatter_sequence(
    x: torch.Tensor, group: GroupHandle, *, dim: int = SEQUENCE_DIM
) -> torch.Tensor:
    """Keep only this rank's slice of the sequence.

    Forward is local (the tensor is already replicated); backward all-gathers
    the gradient, because the upstream replicated tensor needs every rank's
    contribution.

    Args:
        x: Replicated activation ``(batch, sequence, hidden)``.  ``sequence``
            must be divisible by the group size -- pad with
            :func:`pad_sequence_dimension` first if it is not.
        group: Sequence-parallel group.
        dim: Sequence dimension.

    Returns:
        ``(batch, sequence / G, hidden)``.

    Raises:
        ShardingError: If the sequence is not divisible by the group size.
    """
    _require_divisible(x, group, dim, "sequence_parallel.scatter_sequence")
    return scatter_to_group(x, group, dim=dim)


def gather_sequence(
    x: torch.Tensor, group: GroupHandle, *, dim: int = SEQUENCE_DIM
) -> torch.Tensor:
    """Reassemble the full sequence from the group's shards.

    Forward all-gathers; backward keeps this rank's slice of the gradient.

    Args:
        x: This rank's shard, ``(batch, sequence / G, hidden)``.
        group: Sequence-parallel group.
        dim: Sequence dimension.

    Returns:
        ``(batch, sequence, hidden)``, identical on every rank.
    """
    return gather_from_group(x, group, dim=dim)


def reduce_scatter_sequence(
    x: torch.Tensor, group: GroupHandle, *, dim: int = SEQUENCE_DIM
) -> torch.Tensor:
    """Sum partial activations across the group and keep this rank's slice.

    This is the operation that replaces a tensor-parallel all-reduce when
    sequence parallelism is enabled: the sum that the all-reduce would have
    performed still happens, but only ``1/G`` of the result is materialised per
    rank.

    Forward reduce-scatters; backward all-gathers.

    Args:
        x: This rank's partial sum, full length along ``dim``.
        group: Sequence-parallel group.
        dim: Sequence dimension.

    Returns:
        ``(batch, sequence / G, hidden)``.

    Raises:
        ShardingError: If the sequence is not divisible by the group size.
    """
    _require_divisible(x, group, dim, "sequence_parallel.reduce_scatter_sequence")
    return reduce_scatter_to_group(x, group, dim=dim)


def local_sequence_slice(total_length: int, group: GroupHandle) -> tuple[int, int]:
    """Return the ``[start, end)`` sequence positions this rank owns.

    Needed by anything that indexes the sequence by absolute position -- a
    learned positional embedding, a causal mask, a relative-position bias.

    Args:
        total_length: Full (already padded) sequence length.
        group: Sequence-parallel group.

    Returns:
        Half-open interval of positions.

    Raises:
        ShardingError: If ``total_length`` is not divisible by the group size.
    """
    if total_length % group.size != 0:
        raise ShardingError(
            format_error(
                "sequence_parallel.local_sequence_slice",
                "sequence length is not divisible by the sequence-parallel size",
                rank=group.global_rank,
                expected=f"{total_length} % {group.size} == 0",
                observed=total_length % group.size,
                resolution="pad the sequence with pad_sequence_dimension() first",
            )
        )
    local = total_length // group.size
    start = group.local_rank * local
    return start, start + local


def _require_divisible(x: torch.Tensor, group: GroupHandle, dim: int, operation: str) -> None:
    """Raise unless ``x.shape[dim]`` splits evenly across ``group``."""
    if x.shape[dim] % group.size != 0:
        raise ShardingError(
            format_error(
                operation,
                "the sequence dimension must be divisible by the sequence-parallel size; "
                "an uneven split would give ranks different shapes and the equal-size "
                "collectives would mismatch",
                rank=group.global_rank,
                expected=f"{x.shape[dim]} % {group.size} == 0",
                observed=x.shape[dim] % group.size,
                resolution=(
                    "call pad_sequence_dimension() before scattering and "
                    "unpad_sequence_dimension() after gathering"
                ),
            )
        )


class SequenceParallelLayerNorm(nn.Module):
    """LayerNorm that operates directly on a sequence shard.

    **No communication is required, and that is the whole point.** LayerNorm
    normalises over the *hidden* dimension:

    .. math::

        y_{b,s,:} = \\gamma \\odot
            \\frac{x_{b,s,:} - \\mu_{b,s}}{\\sqrt{\\sigma^2_{b,s} + \\epsilon}}
            + \\beta

    The statistics :math:`\\mu_{b,s}` and :math:`\\sigma^2_{b,s}` are computed
    per ``(batch, position)`` pair over the hidden axis only, so a rank holding
    positions ``[s0, s1)`` has everything it needs for those positions.  Had the
    normalisation been over the sequence -- as in some other architectures --
    sequence sharding would have required an all-reduce of the statistics.

    The parameters :math:`\\gamma` and :math:`\\beta` are **replicated** across
    the sequence-parallel group, so their gradients must be all-reduced over it.
    That reduction is handled by the surrounding data-parallel/tensor-parallel
    machinery via the replication marker set here.

    Args:
        hidden_size: Width of the normalised dimension.
        eps: Numerical stabiliser.
        group: The sequence-parallel group, recorded so gradient-norm
            computation knows the parameters are replicated over it.
        sequence_parallel: Whether this layer will see sequence *shards*.  When
            ``True`` the parameters are marked as having partial gradients, so
            that :func:`~hybrid_training.parallel.tensor_parallel.all_reduce_sequence_parallel_gradients`
            sums them across the group after backward.  Getting this flag wrong
            in either direction is a silent correctness bug: ``False`` when it
            should be ``True`` drops every other rank's positions from the
            gradient, and ``True`` when it should be ``False`` multiplies the
            gradient by the group size.
        device: Construction device.
        dtype: Parameter dtype.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        eps: float = 1e-5,
        group: GroupHandle | None = None,
        sequence_parallel: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.sequence_parallel = sequence_parallel
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(hidden_size, device=device, dtype=dtype))
        if group is not None:
            from .tensor_parallel import _mark_replicated, mark_sequence_parallel_partial

            _mark_replicated(self.weight, group)
            _mark_replicated(self.bias, group)
            if sequence_parallel:
                mark_sequence_parallel_partial(self.weight)
                mark_sequence_parallel_partial(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise over the last dimension.

        Args:
            x: ``(..., hidden_size)``.  The sequence dimension may be sharded;
                this function neither knows nor cares.

        Returns:
            The normalised tensor, same shape.
        """
        return torch.nn.functional.layer_norm(
            x, (self.hidden_size,), self.weight, self.bias, self.eps
        )

    def extra_repr(self) -> str:
        """Describe the layer in ``print(model)`` output."""
        return (
            f"hidden_size={self.hidden_size}, eps={self.eps}, "
            f"sequence_parallel={self.sequence_parallel}"
        )
