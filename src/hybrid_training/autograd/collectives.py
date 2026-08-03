"""Differentiable collectives: the four primitives every model-parallel layer needs.

A collective inside a model is a *function*, and autograd needs its adjoint.
This module defines the four that matter, together with proofs sketched in the
docstrings.  Throughout, ``G`` is the process group, ``r`` indexes ranks, and
:math:`\\bar{g} = \\partial L / \\partial Y` is the incoming gradient.

======================  ==========================  ==========================
Name                    Forward                     Backward (adjoint)
======================  ==========================  ==========================
``copy_to_group``       :math:`Y_r = X`             :math:`\\bar{X} = \\sum_r \\bar{g}_r`
(``f`` in Megatron)     identity                    all-reduce
``reduce_from_group``   :math:`Y = \\sum_r X_r`      :math:`\\bar{X}_r = \\bar{g}`
(``g`` in Megatron)     all-reduce                  identity
``gather_from_group``   :math:`Y = [X_0 \\ldots X_{G-1}]`  :math:`\\bar{X}_r = \\bar{g}[r]`
                        all-gather along a dim      split along that dim
``scatter_to_group``    :math:`Y_r = X[r]`          :math:`\\bar{X} = [\\bar{g}_0 \\ldots]`
                        split along a dim           all-gather along that dim
``reduce_scatter_to_group``  :math:`Y_r = (\\sum_q X_q)[r]`  :math:`\\bar{X}_q = [\\bar{g}_0 \\ldots]`
                        sum then split              all-gather along that dim
======================  ==========================  ==========================

Why ``copy_to_group``'s adjoint is an all-reduce
------------------------------------------------
Suppose a replicated activation :math:`X` (identical on every rank) feeds a
column-parallel weight slice on each rank.  Formally there are ``G`` copies
:math:`Y_r = X`, and the loss depends on all of them:

.. math::

    \\frac{\\partial L}{\\partial X}
      = \\sum_{r=0}^{G-1} \\frac{\\partial L}{\\partial Y_r}
        \\frac{\\partial Y_r}{\\partial X}
      = \\sum_{r=0}^{G-1} \\bar{g}_r

Each rank computes only its own :math:`\\bar{g}_r`, so producing the true
:math:`\\partial L/\\partial X` requires summing across the group -- an
all-reduce.  Forgetting it is the classic tensor-parallel bug: the model still
trains, but every rank's input gradient is a factor of ``G`` too small *and*
missing the other ranks' contributions, so layers below the split learn the
wrong thing.

Why ``reduce_from_group``'s adjoint is the identity
---------------------------------------------------
Here :math:`Y = \\sum_r X_r` with :math:`Y` replicated.  Then
:math:`\\partial Y/\\partial X_r = I` for every ``r``, so
:math:`\\bar{X}_r = \\bar{g}`.  Since the all-reduce already made
:math:`\\bar{g}` identical on every rank, nothing needs to be communicated.

Gathering along a dimension other than 0
----------------------------------------
``all_gather_into_tensor`` and ``reduce_scatter_tensor`` concatenate along
dimension 0 only.  To operate on dimension ``d`` the tensor is transposed so
``d`` becomes 0, made contiguous, communicated, and transposed back.  The
transposes cost a copy; the alternative (a list-based ``all_gather`` plus
``torch.cat``) costs a copy *and* an extra allocation per rank, so this is the
cheaper of two imperfect options.  ``docs/06_sequence_parallelism.md`` shows
the shapes at each step.
"""

from __future__ import annotations

from typing import Any

import torch

from ..distributed.collectives import (
    CommunicationRecorder,
    ReduceOp,
    all_gather_tensor,
    all_reduce,
    reduce_scatter_tensor,
)
from ..distributed.groups import GroupHandle
from ..errors import CollectiveError, format_error
from ..utils.tensors import split_tensor_along_dim

__all__ = [
    "all_gather_along_dim",
    "copy_to_group",
    "gather_from_group",
    "gather_from_sequence_parallel_region",
    "reduce_from_group",
    "reduce_scatter_along_dim",
    "reduce_scatter_to_group",
    "scatter_to_group",
]


def all_gather_along_dim(
    tensor: torch.Tensor,
    group: GroupHandle,
    dim: int,
    *,
    recorder: CommunicationRecorder | None = None,
) -> torch.Tensor:
    """Concatenate one tensor per rank along ``dim`` (non-differentiable).

    Args:
        tensor: This rank's contribution.  Shapes must match across the group.
        group: Communication group.
        dim: Dimension to concatenate along.
        recorder: Optional instrumentation sink.

    Returns:
        A tensor whose ``dim`` is ``group.size`` times longer.

    Example:
        With ``group.size == 2`` and inputs of shape ``(4, 3, 8)`` gathered
        along ``dim=1``, the result has shape ``(4, 6, 8)``, where rows
        ``[0:3]`` come from rank 0 and ``[3:6]`` from rank 1.
    """
    if group.is_trivial:
        return tensor
    dim = dim % tensor.dim()
    moved = tensor.movedim(dim, 0).contiguous()
    gathered, work = all_gather_tensor(moved, group, recorder=recorder)
    work.wait()
    return gathered.movedim(0, dim).contiguous()


def reduce_scatter_along_dim(
    tensor: torch.Tensor,
    group: GroupHandle,
    dim: int,
    *,
    op: str = ReduceOp.SUM,
    recorder: CommunicationRecorder | None = None,
) -> torch.Tensor:
    """Sum across ranks and keep this rank's slice of ``dim`` (non-differentiable).

    Args:
        tensor: Input, identical in shape on every rank, with ``dim``
            divisible by the group size.
        group: Communication group.
        dim: Dimension to scatter along.
        op: Reduction operation.
        recorder: Optional instrumentation sink.

    Returns:
        A tensor whose ``dim`` is ``group.size`` times shorter.

    Raises:
        CollectiveError: If ``dim`` is not divisible by the group size.
    """
    if group.is_trivial:
        return tensor
    dim = dim % tensor.dim()
    if tensor.shape[dim] % group.size != 0:
        raise CollectiveError(
            format_error(
                "autograd.reduce_scatter_along_dim",
                f"dimension {dim} must be divisible by the group size",
                rank=group.global_rank,
                expected=f"{tensor.shape[dim]} % {group.size} == 0",
                observed=tensor.shape[dim] % group.size,
                resolution="pad the dimension or choose a parallel size that divides it",
            )
        )
    moved = tensor.movedim(dim, 0).contiguous()
    scattered, work = reduce_scatter_tensor(moved, group, op=op, recorder=recorder)
    work.wait()
    return scattered.movedim(0, dim).contiguous()


class _CopyToGroup(torch.autograd.Function):
    """Identity forward; all-reduce backward (Megatron's ``f``)."""

    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor, group: GroupHandle) -> torch.Tensor:
        ctx.group = group
        return tensor

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore[override]
        group: GroupHandle = ctx.group
        if group.is_trivial:
            return grad, None
        grad = grad.contiguous()
        all_reduce(grad, group, op=ReduceOp.SUM).wait()
        return grad, None


class _ReduceFromGroup(torch.autograd.Function):
    """All-reduce forward; identity backward (Megatron's ``g``)."""

    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor, group: GroupHandle) -> torch.Tensor:
        if group.is_trivial:
            return tensor
        out = tensor.contiguous().clone()
        all_reduce(out, group, op=ReduceOp.SUM).wait()
        return out

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore[override]
        return grad, None


class _GatherFromGroup(torch.autograd.Function):
    """All-gather forward along ``dim``; split backward."""

    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor, group: GroupHandle, dim: int) -> torch.Tensor:
        ctx.group = group
        ctx.dim = dim % tensor.dim()
        return all_gather_along_dim(tensor, group, ctx.dim)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None, None]:  # type: ignore[override]
        group: GroupHandle = ctx.group
        if group.is_trivial:
            return grad, None, None
        parts = split_tensor_along_dim(grad, ctx.dim, group.size)
        return parts[group.local_rank], None, None


class _ScatterToGroup(torch.autograd.Function):
    """Split forward along ``dim``; all-gather backward.

    Forward is purely local -- every rank already holds the whole tensor and
    simply keeps its own slice -- but the backward pass genuinely communicates,
    because the upstream tensor is replicated and needs every rank's gradient.
    """

    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor, group: GroupHandle, dim: int) -> torch.Tensor:
        ctx.group = group
        ctx.dim = dim % tensor.dim()
        if group.is_trivial:
            return tensor
        parts = split_tensor_along_dim(tensor, ctx.dim, group.size)
        return parts[group.local_rank]

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None, None]:  # type: ignore[override]
        group: GroupHandle = ctx.group
        if group.is_trivial:
            return grad, None, None
        return all_gather_along_dim(grad.contiguous(), group, ctx.dim), None, None


class _ReduceScatterToGroup(torch.autograd.Function):
    """Reduce-scatter forward along ``dim``; all-gather backward.

    This is the pair that makes sequence parallelism cheap.  In the fused
    tensor+sequence schedule, a row-parallel layer's output is *partial* on
    every rank (each holds a partial sum) and the next region wants it split
    along the sequence.  A single reduce-scatter does both jobs, moving
    ``1/G`` of the data an all-reduce would have moved.
    """

    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor, group: GroupHandle, dim: int) -> torch.Tensor:
        ctx.group = group
        ctx.dim = dim % tensor.dim()
        return reduce_scatter_along_dim(tensor, group, ctx.dim, op=ReduceOp.SUM)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None, None]:  # type: ignore[override]
        group: GroupHandle = ctx.group
        if group.is_trivial:
            return grad, None, None
        return all_gather_along_dim(grad.contiguous(), group, ctx.dim), None, None


class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    """All-gather forward along ``dim``; **reduce-scatter** backward.

    This is *not* the same as :class:`_GatherFromGroup`, and the difference is
    the single subtlest point in the whole sequence-parallel implementation.

    Both gather shards into a full tensor :math:`Y`.  What differs is how
    :math:`Y` is consumed, and therefore what :math:`\\partial L/\\partial Y`
    means:

    *Feature gather* (:class:`_GatherFromGroup`, used at the end of a
    tensor-parallel region).  :math:`Y` is a **replicated activation**: every
    rank goes on to perform the *same* computation with it.  Each rank
    redundantly computes the same :math:`\\partial L/\\partial Y`, so that value
    is already the true total, and the adjoint is simply "take my slice".

    *Sequence gather* (this class, used at the **entrance** to a
    tensor-parallel region).  :math:`Y` is consumed **differently** on each
    rank -- rank ``t`` multiplies it by *its* slice of the weight matrix, i.e.
    computes with different attention heads.  Each rank therefore holds only a
    *partial* :math:`\\partial L/\\partial Y`, and the true total is the sum
    over ranks:

    .. math::

        \\frac{\\partial L}{\\partial X_r}
          = \\Big[\\sum_{q=0}^{G-1}
            \\frac{\\partial L}{\\partial Y}\\Big|_q\\Big]_{\\text{slice } r}

    Summing and then slicing is exactly ``reduce_scatter``.  Using the split
    adjoint here instead produces a forward pass that is numerically perfect
    and gradients that are missing every other rank's contribution -- a bug
    that shows up as slightly-wrong training rather than as an error.
    """

    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor, group: GroupHandle, dim: int) -> torch.Tensor:
        ctx.group = group
        ctx.dim = dim % tensor.dim()
        return all_gather_along_dim(tensor, group, ctx.dim)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None, None]:  # type: ignore[override]
        group: GroupHandle = ctx.group
        if group.is_trivial:
            return grad, None, None
        return (
            reduce_scatter_along_dim(grad.contiguous(), group, ctx.dim, op=ReduceOp.SUM),
            None,
            None,
        )


def gather_from_sequence_parallel_region(
    tensor: torch.Tensor, group: GroupHandle, dim: int = 1
) -> torch.Tensor:
    """Gather a sequence shard at the entrance to a tensor-parallel region.

    Forward all-gathers; backward reduce-scatters.  See
    :class:`_GatherFromSequenceParallelRegion` for why the adjoint differs from
    :func:`gather_from_group`.

    Args:
        tensor: This rank's sequence shard.
        group: Sequence-parallel group.
        dim: Sequence dimension.

    Returns:
        The full-sequence tensor.
    """
    return _GatherFromSequenceParallelRegion.apply(tensor, group, dim)


def copy_to_group(tensor: torch.Tensor, group: GroupHandle) -> torch.Tensor:
    """Mark ``tensor`` as replicated over ``group``.

    Forward is the identity; backward all-reduces the gradient.  Apply this to
    an activation *before* it enters a column-parallel layer.

    Args:
        tensor: Replicated activation.
        group: Tensor-parallel group.

    Returns:
        ``tensor``, with the all-reduce adjoint attached.
    """
    return _CopyToGroup.apply(tensor, group)


def reduce_from_group(tensor: torch.Tensor, group: GroupHandle) -> torch.Tensor:
    """Sum partial results across ``group`` and replicate the total.

    Forward all-reduces; backward is the identity.  Apply this to the output of
    a row-parallel layer.

    Args:
        tensor: This rank's partial sum.
        group: Tensor-parallel group.

    Returns:
        The full sum, identical on every rank.
    """
    return _ReduceFromGroup.apply(tensor, group)


def gather_from_group(tensor: torch.Tensor, group: GroupHandle, dim: int = -1) -> torch.Tensor:
    """Concatenate the group's shards along ``dim``.

    Forward all-gathers; backward keeps this rank's slice of the gradient.

    Args:
        tensor: This rank's shard.
        group: Communication group.
        dim: Dimension to concatenate along.  ``-1`` (the feature dimension)
            for a column-parallel output; the sequence dimension for sequence
            parallelism.

    Returns:
        The concatenated tensor.
    """
    return _GatherFromGroup.apply(tensor, group, dim)


def scatter_to_group(tensor: torch.Tensor, group: GroupHandle, dim: int = -1) -> torch.Tensor:
    """Keep only this rank's slice of ``dim``.

    Forward splits (locally); backward all-gathers the gradient.

    Args:
        tensor: A replicated tensor.
        group: Communication group.
        dim: Dimension to split.

    Returns:
        This rank's slice.
    """
    return _ScatterToGroup.apply(tensor, group, dim)


def reduce_scatter_to_group(tensor: torch.Tensor, group: GroupHandle, dim: int = 1) -> torch.Tensor:
    """Sum partial results across the group and keep this rank's slice of ``dim``.

    Forward reduce-scatters; backward all-gathers.

    Args:
        tensor: This rank's partial sum, full length along ``dim``.
        group: Communication group.
        dim: Dimension to scatter along.  Defaults to ``1``, the sequence
            dimension of a ``(batch, sequence, hidden)`` activation.

    Returns:
        The reduced tensor, ``group.size`` times shorter along ``dim``.
    """
    return _ReduceScatterToGroup.apply(tensor, group, dim)
