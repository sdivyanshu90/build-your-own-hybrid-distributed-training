"""Tensor-parallel linear layers.

The idea in one matrix equation
===============================
A linear layer computes :math:`Y = XW^\\top + b` with
:math:`W \\in \\mathbb{R}^{O \\times I}`.  There are exactly two ways to cut
:math:`W` across ``T`` ranks, and they differ in *which* dimension of the
product each rank ends up owning.

**Column parallel** splits the output features::

    W = [W_0 ; W_1 ; ... ; W_{T-1}]   stacked along dim 0 (out_features)
    W_t in R^{(O/T) x I}

    Y_t = X W_t^T + b_t     ->  Y = [Y_0 | Y_1 | ... ]  concatenated on features

Every rank needs the *whole* :math:`X` and produces a *slice* of :math:`Y`.
Forward needs no communication at all (assuming :math:`X` is already
replicated); backward must all-reduce :math:`\\partial L/\\partial X`, because
each rank computed only its own contribution
:math:`\\bar{Y}_t W_t`.

**Row parallel** splits the input features::

    W = [W_0 | W_1 | ... | W_{T-1}]   split along dim 1 (in_features)
    W_t in R^{O x (I/T)}
    X   = [X_0 | X_1 | ... ]          split the same way

    Y = sum_t X_t W_t^T + b

Every rank needs a *slice* of :math:`X` and produces a *partial sum* of the
whole :math:`Y`.  Forward must all-reduce; backward needs no communication for
:math:`\\partial L/\\partial X_t`, because the all-reduced
:math:`\\bar{Y}` is already identical everywhere.

The pairing that makes transformers cheap
=========================================
Because a column-parallel layer *produces* feature shards and a row-parallel
layer *consumes* them, chaining them costs **one** collective for the pair
instead of one per layer::

    x  --f-->  ColumnParallel  -->  act  -->  RowParallel  --g-->  y
       (identity fwd,                                     (all-reduce fwd,
        all-reduce bwd)                                    identity bwd)

The activation in the middle is elementwise, so it operates happily on shards.
This is exactly :class:`TensorParallelFeedForward`, and the same structure with
attention heads in place of the hidden dimension is
:class:`~hybrid_training.models.transformer.TensorParallelAttention`.

Initialisation
==============
Each rank draws the **whole** weight matrix from an identical seed and keeps
only its slice.  The concatenation of the slices is then bit-for-bit what a
single process would have produced, which is what lets the tests assert exact
structural equality against an unsharded ``nn.Linear``.  The alternative --
each rank drawing only its own slice -- is cheaper but produces a different
(equally valid) initialisation, and makes exact equivalence untestable.  See
:mod:`hybrid_training.utils.reproducibility` for the full discussion.

Divisibility
============
``out_features`` (column parallel) and ``in_features`` (row parallel) must be
divisible by the tensor-parallel size.  Uneven splits are rejected rather than
padded, because a padded feature dimension changes the *mathematics* -- a bias
or a LayerNorm applied after an all-gather would see the padding as real
features.  ``docs/05_tensor_parallelism.md`` shows what padding would require.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..autograd.collectives import (
    copy_to_group,
    gather_from_group,
    gather_from_sequence_parallel_region,
    reduce_from_group,
    reduce_scatter_to_group,
    scatter_to_group,
)
from ..distributed.groups import GroupHandle
from ..errors import TensorParallelError, format_error
from ..logging import get_logger
from ..utils.reproducibility import derive_seed, temporary_seed
from ..utils.tensors import split_tensor_along_dim

__all__ = [
    "ColumnParallelLinear",
    "RowParallelLinear",
    "TensorParallelFeedForward",
    "TensorParallelMLPBlock",
    "VocabParallelEmbedding",
    "all_reduce_sequence_parallel_gradients",
    "init_linear_parameters",
    "mark_sequence_parallel_partial",
]

_LOGGER = get_logger(__name__)

#: Sequence-parallel layers keep activations as ``(batch, sequence, hidden)``
#: and split dimension 1.  Naming the constant means the layers never contain a
#: bare ``dim=1`` whose meaning has to be inferred.
SEQUENCE_DIM = 1


def init_linear_parameters(
    weight: torch.Tensor, bias: torch.Tensor | None, *, in_features: int
) -> None:
    """Initialise a linear layer exactly as ``nn.Linear.reset_parameters`` does.

    Reproducing PyTorch's own initialisation -- rather than inventing a scheme
    -- means a tensor-parallel layer built with seed ``s`` and an ``nn.Linear``
    built with seed ``s`` hold the same numbers, which is the basis of the
    equivalence tests.

    Args:
        weight: Weight tensor of shape ``(out_features, in_features)``.  For a
            partitioned layer this is the **full** matrix; the caller slices it
            afterwards.
        bias: Optional bias of shape ``(out_features,)``.
        in_features: Fan-in used for the bias bound.  Passed explicitly because
            a row-parallel shard's ``weight.shape[1]`` is not the true fan-in.
    """
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    if bias is not None:
        bound = 1 / math.sqrt(in_features) if in_features > 0 else 0
        nn.init.uniform_(bias, -bound, bound)


def _validate_divisible(value: int, parts: int, *, what: str, group: GroupHandle) -> int:
    """Check divisibility and return the per-rank size."""
    if value % parts != 0:
        raise TensorParallelError(
            format_error(
                "tensor_parallel.validate_divisible",
                f"{what} must be divisible by the tensor-parallel size; an uneven split "
                "would give ranks different shapes, which the single-buffer collectives "
                "cannot express and which changes what a following bias or normalisation "
                "layer computes",
                rank=group.global_rank,
                expected=f"{what} % {parts} == 0",
                observed=f"{what}={value}, remainder={value % parts}",
                resolution=(
                    f"choose {what} as a multiple of {parts}, or reduce the tensor-parallel size"
                ),
            )
        )
    return value // parts


class ColumnParallelLinear(nn.Module):
    """Linear layer with ``out_features`` partitioned across a group.

    Args:
        in_features: Full input width.  Replicated: every rank sees all of it.
        out_features: Full output width.  Must be divisible by ``group.size``.
        group: Tensor-parallel group.  **Required.**
        bias: Whether to include a bias.  The bias is partitioned exactly like
            the output, so rank ``t`` holds ``b[t*O/T : (t+1)*O/T]``.
        gather_output: All-gather the output shards into the full width.
            ``False`` (the default) is right when the next layer is
            row-parallel; ``True`` is right at the end of a parallel region.
        sequence_parallel: Treat the input as split along the sequence
            dimension and all-gather it first.  Only meaningful together with
            ``gather_output=False``; see the class notes.
        init_seed: Seed for the full-matrix draw.  Every rank must pass the
            same value.
        device: Construction device.
        dtype: Parameter dtype.

    Raises:
        TensorParallelError: If ``out_features`` is not divisible by the group
            size, or if an inconsistent combination of flags is requested.

    Example:
        >>> # doctest: +SKIP
        >>> layer = ColumnParallelLinear(16, 32, group, gather_output=True)
        >>> layer(torch.randn(4, 16)).shape
        torch.Size([4, 32])
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group: GroupHandle,
        *,
        bias: bool = True,
        gather_output: bool = False,
        sequence_parallel: bool = False,
        init_seed: int = 0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if sequence_parallel and gather_output:
            raise TensorParallelError(
                format_error(
                    "tensor_parallel.ColumnParallelLinear",
                    "sequence_parallel implies the output stays sharded on features so a "
                    "row-parallel layer can consume it; gathering the output as well would "
                    "undo both savings",
                    rank=group.global_rank,
                    expected="gather_output=False when sequence_parallel=True",
                    observed="both True",
                    resolution="set gather_output=False",
                )
            )
        self.in_features = in_features
        self.out_features = out_features
        self.group = group
        self.gather_output = gather_output
        self.sequence_parallel = sequence_parallel
        self.output_features_per_partition = _validate_divisible(
            out_features, group.size, what="out_features", group=group
        )

        factory: dict[str, Any] = {}
        if device is not None:
            factory["device"] = device
        if dtype is not None:
            factory["dtype"] = dtype

        # Draw the whole matrix identically on every rank, then keep one slice.
        with temporary_seed(derive_seed(init_seed, "tensor-parallel-init")):
            full_weight = torch.empty(out_features, in_features, **factory)
            full_bias = torch.empty(out_features, **factory) if bias else None
            init_linear_parameters(full_weight, full_bias, in_features=in_features)

        weight_shard = split_tensor_along_dim(full_weight, 0, group.size)[group.local_rank]
        self.weight = nn.Parameter(weight_shard.clone())
        if bias:
            assert full_bias is not None
            bias_shard = split_tensor_along_dim(full_bias, 0, group.size)[group.local_rank]
            self.bias: nn.Parameter | None = nn.Parameter(bias_shard.clone())
        else:
            self.register_parameter("bias", None)

        # Recorded so the distributed gradient-norm computation knows this
        # parameter is *partitioned* (its shards must be summed) rather than
        # *replicated* (which must be counted once).
        _mark_partitioned(self.weight, group, partition_dim=0)
        if self.bias is not None:
            _mark_partitioned(self.bias, group, partition_dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the partitioned linear layer.

        Args:
            x: Input of shape ``(..., in_features)``.  With
                ``sequence_parallel=True`` the sequence dimension is expected
                to be already split, i.e. ``(batch, seq/T, in_features)``.

        Returns:
            ``(..., out_features)`` when ``gather_output`` is set, otherwise
            ``(..., out_features / T)``.
        """
        if self.sequence_parallel:
            # The parallel region needs the whole sequence, so gather it.  The
            # adjoint is a *reduce-scatter*, not a split: the gathered
            # activation is consumed by a different weight slice on every rank,
            # so each rank holds only a partial gradient with respect to it.
            x = gather_from_sequence_parallel_region(x, self.group, dim=SEQUENCE_DIM)
        else:
            # X is replicated; mark it so backward sums the input gradients.
            x = copy_to_group(x, self.group)

        output = F.linear(x, self.weight, self.bias)
        if self.gather_output:
            output = gather_from_group(output, self.group, dim=-1)
        return output

    def full_weight(self) -> torch.Tensor:
        """All-gather the weight shards into the full matrix.

        For tests and inspection only: it materialises the whole layer on every
        rank.

        Returns:
            The full ``(out_features, in_features)`` weight.
        """
        from ..autograd.collectives import all_gather_along_dim

        return all_gather_along_dim(self.weight.detach(), self.group, 0)

    def full_bias(self) -> torch.Tensor | None:
        """All-gather the bias shards, or ``None`` when the layer has no bias."""
        if self.bias is None:
            return None
        from ..autograd.collectives import all_gather_along_dim

        return all_gather_along_dim(self.bias.detach(), self.group, 0)

    def load_from_linear(self, linear: nn.Linear) -> None:
        """Copy the matching slice out of an unsharded ``nn.Linear``.

        Args:
            linear: Reference layer with the full weight.

        Raises:
            TensorParallelError: If the shapes do not match this layer.
        """
        if linear.weight.shape != (self.out_features, self.in_features):
            raise TensorParallelError(
                format_error(
                    "tensor_parallel.load_from_linear",
                    "weight shape mismatch",
                    rank=self.group.global_rank,
                    expected=(self.out_features, self.in_features),
                    observed=tuple(linear.weight.shape),
                    resolution="the reference layer has different features",
                )
            )
        with torch.no_grad():
            shard = split_tensor_along_dim(linear.weight.detach(), 0, self.group.size)
            self.weight.copy_(shard[self.group.local_rank])
            if self.bias is not None:
                if linear.bias is None:
                    raise TensorParallelError(
                        format_error(
                            "tensor_parallel.load_from_linear",
                            "this layer has a bias but the reference layer does not",
                            rank=self.group.global_rank,
                            expected="a bias tensor",
                            observed=None,
                            resolution="construct the parallel layer with bias=False",
                        )
                    )
                bias_shard = split_tensor_along_dim(linear.bias.detach(), 0, self.group.size)
                self.bias.copy_(bias_shard[self.group.local_rank])

    def extra_repr(self) -> str:
        """Describe the partitioning in ``print(model)`` output."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"per_partition={self.output_features_per_partition}, "
            f"tp_size={self.group.size}, bias={self.bias is not None}, "
            f"gather_output={self.gather_output}, sequence_parallel={self.sequence_parallel}"
        )


class RowParallelLinear(nn.Module):
    """Linear layer with ``in_features`` partitioned across a group.

    Args:
        in_features: Full input width.  Must be divisible by ``group.size``.
        out_features: Full output width.  Replicated.
        group: Tensor-parallel group.  **Required.**
        bias: Whether to include a bias.  The bias is **replicated**, and it is
            added *after* the reduction -- adding it before would count it
            ``T`` times.
        input_is_parallel: The input is already split along its feature
            dimension (the normal case, when the previous layer was
            column-parallel).  When ``False``, the layer splits a replicated
            input itself, which costs an extra all-gather in backward.
        sequence_parallel: Emit the output split along the sequence dimension,
            using a reduce-scatter instead of an all-reduce.  This is the
            Megatron sequence-parallel schedule: the same collective performs
            the cross-rank sum *and* the sequence split, moving ``1/T`` of the
            bytes an all-reduce would.
        init_seed: Seed for the full-matrix draw.
        device: Construction device.
        dtype: Parameter dtype.

    Raises:
        TensorParallelError: If ``in_features`` is not divisible by the group
            size.

    Example:
        >>> # doctest: +SKIP
        >>> layer = RowParallelLinear(32, 16, group, input_is_parallel=False)
        >>> layer(torch.randn(4, 32)).shape
        torch.Size([4, 16])
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group: GroupHandle,
        *,
        bias: bool = True,
        input_is_parallel: bool = True,
        sequence_parallel: bool = False,
        init_seed: int = 0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group = group
        self.input_is_parallel = input_is_parallel
        self.sequence_parallel = sequence_parallel
        self.input_features_per_partition = _validate_divisible(
            in_features, group.size, what="in_features", group=group
        )

        factory: dict[str, Any] = {}
        if device is not None:
            factory["device"] = device
        if dtype is not None:
            factory["dtype"] = dtype

        with temporary_seed(derive_seed(init_seed, "tensor-parallel-init")):
            full_weight = torch.empty(out_features, in_features, **factory)
            full_bias = torch.empty(out_features, **factory) if bias else None
            init_linear_parameters(full_weight, full_bias, in_features=in_features)

        weight_shard = split_tensor_along_dim(full_weight, 1, group.size)[group.local_rank]
        self.weight = nn.Parameter(weight_shard.clone())
        if bias:
            assert full_bias is not None
            # Replicated, not partitioned: every rank holds the whole bias.
            self.bias: nn.Parameter | None = nn.Parameter(full_bias.clone())
        else:
            self.register_parameter("bias", None)

        _mark_partitioned(self.weight, group, partition_dim=1)
        if self.bias is not None:
            _mark_replicated(self.bias, group)
            if sequence_parallel:
                # The bias is added *after* the reduce-scatter, so it only ever
                # sees this rank's slice of the sequence.  Its gradient is
                # therefore partial and must be summed across the group before
                # the optimizer step -- see mark_sequence_parallel_partial.
                mark_sequence_parallel_partial(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the partitioned linear layer.

        Args:
            x: ``(..., in_features / T)`` when ``input_is_parallel``, otherwise
                ``(..., in_features)``.

        Returns:
            ``(..., out_features)``.  With ``sequence_parallel=True`` the
            sequence dimension is additionally divided by ``T``.
        """
        if not self.input_is_parallel:
            x = scatter_to_group(x, self.group, dim=-1)

        # Bias is deliberately excluded here: each rank computes a partial sum,
        # and adding a replicated bias to every partial would multiply it by T.
        partial = F.linear(x, self.weight, None)

        if self.sequence_parallel:
            output = reduce_scatter_to_group(partial, self.group, dim=SEQUENCE_DIM)
        else:
            output = reduce_from_group(partial, self.group)

        if self.bias is not None:
            output = output + self.bias
        return output

    def full_weight(self) -> torch.Tensor:
        """All-gather the weight shards into the full matrix."""
        from ..autograd.collectives import all_gather_along_dim

        return all_gather_along_dim(self.weight.detach(), self.group, 1)

    def load_from_linear(self, linear: nn.Linear) -> None:
        """Copy the matching slice out of an unsharded ``nn.Linear``.

        Args:
            linear: Reference layer with the full weight.

        Raises:
            TensorParallelError: If shapes do not match.
        """
        if linear.weight.shape != (self.out_features, self.in_features):
            raise TensorParallelError(
                format_error(
                    "tensor_parallel.load_from_linear",
                    "weight shape mismatch",
                    rank=self.group.global_rank,
                    expected=(self.out_features, self.in_features),
                    observed=tuple(linear.weight.shape),
                    resolution="the reference layer has different features",
                )
            )
        with torch.no_grad():
            shard = split_tensor_along_dim(linear.weight.detach(), 1, self.group.size)
            self.weight.copy_(shard[self.group.local_rank])
            if self.bias is not None:
                if linear.bias is None:
                    raise TensorParallelError(
                        format_error(
                            "tensor_parallel.load_from_linear",
                            "this layer has a bias but the reference layer does not",
                            rank=self.group.global_rank,
                            expected="a bias tensor",
                            observed=None,
                            resolution="construct the parallel layer with bias=False",
                        )
                    )
                self.bias.copy_(linear.bias.detach())

    def extra_repr(self) -> str:
        """Describe the partitioning in ``print(model)`` output."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"per_partition={self.input_features_per_partition}, "
            f"tp_size={self.group.size}, bias={self.bias is not None}, "
            f"input_is_parallel={self.input_is_parallel}, "
            f"sequence_parallel={self.sequence_parallel}"
        )


class VocabParallelEmbedding(nn.Module):
    """Embedding whose vocabulary is partitioned across a group.

    Rank ``t`` stores rows ``[t*V/T, (t+1)*V/T)``.  A lookup produces zeros for
    tokens outside the local range, and a single all-reduce sums the per-rank
    contributions -- exactly one rank contributes a non-zero row per token, so
    the sum is the correct embedding.

    This is how the embedding table of a large-vocabulary model is kept off any
    single device.

    Args:
        num_embeddings: Vocabulary size.  Must be divisible by ``group.size``.
        embedding_dim: Embedding width, replicated.
        group: Tensor-parallel group.
        init_seed: Seed for the full-table draw.
        init_std: Standard deviation of the normal initialisation.
        device: Construction device.
        dtype: Parameter dtype.

    Raises:
        TensorParallelError: If the vocabulary is not divisible by the group
            size.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        group: GroupHandle,
        *,
        init_seed: int = 0,
        init_std: float = 0.02,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.group = group
        self.embeddings_per_partition = _validate_divisible(
            num_embeddings, group.size, what="num_embeddings", group=group
        )
        self.vocab_start = group.local_rank * self.embeddings_per_partition
        self.vocab_end = self.vocab_start + self.embeddings_per_partition

        factory: dict[str, Any] = {}
        if device is not None:
            factory["device"] = device
        if dtype is not None:
            factory["dtype"] = dtype

        with temporary_seed(derive_seed(init_seed, "tensor-parallel-init")):
            full = torch.empty(num_embeddings, embedding_dim, **factory)
            nn.init.normal_(full, mean=0.0, std=init_std)
        shard = split_tensor_along_dim(full, 0, group.size)[group.local_rank]
        self.weight = nn.Parameter(shard.clone())
        _mark_partitioned(self.weight, group, partition_dim=0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up embeddings for ``token_ids``.

        Args:
            token_ids: Integer tensor of any shape.

        Returns:
            ``(*token_ids.shape, embedding_dim)``.
        """
        if self.group.size == 1:
            return F.embedding(token_ids, self.weight)
        mask = (token_ids < self.vocab_start) | (token_ids >= self.vocab_end)
        local_ids = (token_ids - self.vocab_start).clamp_(0, self.embeddings_per_partition - 1)
        embedded = F.embedding(local_ids, self.weight)
        embedded = embedded.masked_fill(mask.unsqueeze(-1), 0.0)
        return reduce_from_group(embedded, self.group)

    def full_weight(self) -> torch.Tensor:
        """All-gather the partitioned table into the full one."""
        from ..autograd.collectives import all_gather_along_dim

        return all_gather_along_dim(self.weight.detach(), self.group, 0)

    def extra_repr(self) -> str:
        """Describe the partitioning in ``print(model)`` output."""
        return (
            f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, "
            f"per_partition={self.embeddings_per_partition}, tp_size={self.group.size}"
        )


class TensorParallelFeedForward(nn.Module):
    """The canonical column-then-row transformer feed-forward block.

    ``ColumnParallelLinear(hidden -> ffn)`` feeds ``activation`` feeds
    ``RowParallelLinear(ffn -> hidden)``.  No collective happens between the two
    projections: the column layer's sharded output is exactly the sharded input
    the row layer wants.  The whole block costs one all-reduce forward and one
    all-reduce backward (or, with ``sequence_parallel``, one all-gather and one
    reduce-scatter each way, which move the same total bytes but leave the
    activations sharded).

    Args:
        hidden_size: Model width.
        ffn_hidden_size: Inner width.  Must be divisible by ``group.size``.
        group: Tensor-parallel group.
        activation: ``"gelu"`` or ``"relu"``.
        bias: Whether the projections have biases.
        sequence_parallel: Use the fused tensor+sequence schedule.
        init_seed: Base seed; the two projections derive distinct sub-seeds.
        device: Construction device.
        dtype: Parameter dtype.
    """

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        group: GroupHandle,
        *,
        activation: str = "gelu",
        bias: bool = True,
        sequence_parallel: bool = False,
        init_seed: int = 0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        from ..models.mlp import build_activation

        self.fc1 = ColumnParallelLinear(
            hidden_size,
            ffn_hidden_size,
            group,
            bias=bias,
            gather_output=False,
            sequence_parallel=sequence_parallel,
            init_seed=derive_seed(init_seed, "ffn-fc1"),
            device=device,
            dtype=dtype,
        )
        self.fc2 = RowParallelLinear(
            ffn_hidden_size,
            hidden_size,
            group,
            bias=bias,
            input_is_parallel=True,
            sequence_parallel=sequence_parallel,
            init_seed=derive_seed(init_seed, "ffn-fc2"),
            device=device,
            dtype=dtype,
        )
        self.activation_name = activation
        self._activation = build_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the feed-forward block.

        Args:
            x: ``(batch, sequence, hidden)``, sequence-sharded when
                ``sequence_parallel`` is enabled.

        Returns:
            Same shape as the input.
        """
        return self.fc2(self._activation(self.fc1(x)))


class TensorParallelMLPBlock(nn.Module):
    """Drop-in tensor-parallel replacement for :class:`~hybrid_training.models.mlp.MLPBlock`.

    A single ``Linear -> activation`` whose weight is column-partitioned and
    whose output is gathered back to full width, so the block is
    *mathematically identical* to the unsharded version and can be swapped in
    without changing the model definition.

    This is deliberately **not** the communication-optimal pattern -- gathering
    the output costs one all-gather per layer, where a column/row pair costs one
    all-reduce per *pair*.  It is used for the MLP reference model because
    equivalence with the unsharded model is what that model is for;
    :class:`TensorParallelFeedForward` shows the optimal pattern.

    Args:
        in_features: Input width.
        out_features: Output width.
        group: Tensor-parallel group.
        activation: Activation name.
        bias: Whether the layer has a bias.
        init_seed: Seed for the full-matrix draw.
        device: Construction device.
        dtype: Parameter dtype.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group: GroupHandle,
        *,
        activation: str = "gelu",
        bias: bool = True,
        init_seed: int = 0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        from ..models.mlp import build_activation

        self.linear = ColumnParallelLinear(
            in_features,
            out_features,
            group,
            bias=bias,
            gather_output=True,
            init_seed=init_seed,
            device=device,
            dtype=dtype,
        )
        self.activation_name = activation
        self._activation = build_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the partitioned linear layer followed by the activation.

        Args:
            x: ``(..., in_features)``, replicated.

        Returns:
            ``(..., out_features)``, replicated.
        """
        return self._activation(self.linear(x))


def _mark_partitioned(param: nn.Parameter, group: GroupHandle, *, partition_dim: int) -> None:
    """Record that ``param`` holds a distinct slice on each rank of ``group``.

    The global gradient norm must **sum** the squared norms of partitioned
    parameters across the group (each rank holds different numbers) but must
    **not** sum those of replicated ones (each rank holds the same numbers, so
    summing would multiply the contribution by the group size).  Marking the
    parameters is how :mod:`hybrid_training.optim.sharded_optimizer` tells the
    two apart.

    Args:
        param: The parameter to annotate.
        group: The group it is partitioned over.
        partition_dim: Which dimension is split.
    """
    param.tensor_parallel_partition_dim = partition_dim  # type: ignore[attr-defined]
    param.tensor_parallel_group_name = group.name  # type: ignore[attr-defined]
    param.tensor_parallel_group_size = group.size  # type: ignore[attr-defined]
    param.is_tensor_parallel_replicated = False  # type: ignore[attr-defined]


def _mark_replicated(param: nn.Parameter, group: GroupHandle) -> None:
    """Record that ``param`` holds identical values on every rank of ``group``.

    Args:
        param: The parameter to annotate.
        group: The group it is replicated over.
    """
    param.tensor_parallel_partition_dim = None  # type: ignore[attr-defined]
    param.tensor_parallel_group_name = group.name  # type: ignore[attr-defined]
    param.tensor_parallel_group_size = group.size  # type: ignore[attr-defined]
    param.is_tensor_parallel_replicated = True  # type: ignore[attr-defined]


def mark_sequence_parallel_partial(param: nn.Parameter) -> None:
    """Record that ``param``'s gradient covers only this rank's sequence shard.

    Autograd cannot discover this on its own.  A parameter such as a LayerNorm
    gain is *replicated* across the sequence-parallel group, but under sequence
    parallelism it is only ever applied to the positions this rank holds, so
    the gradient each rank computes is a partial sum over positions:

    .. math::

        \\frac{\\partial L}{\\partial \\gamma}
          = \\sum_{s=0}^{S-1} g_s
          = \\sum_{r=0}^{G-1} \\underbrace{\\sum_{s \\in P_r} g_s}_{
                \\text{what rank } r \\text{ computes}}

    Nothing in the autograd graph performs that outer sum: the parameter is a
    leaf, and no collective sits between it and the loss.  It therefore has to
    be done explicitly after backward, by
    :func:`all_reduce_sequence_parallel_gradients`.

    Without sequence parallelism the same parameter is applied to the *whole*
    sequence redundantly on every rank, so its gradient is already complete and
    summing would multiply it by the group size.  That is why this marker
    exists rather than a blanket rule.

    Args:
        param: The parameter whose gradient is partial.
    """
    param.sequence_parallel_partial_grad = True  # type: ignore[attr-defined]


def all_reduce_sequence_parallel_gradients(
    module: nn.Module,
    group: GroupHandle,
    *,
    recorder: Any = None,
) -> int:
    """Sum the partial gradients of sequence-parallel replicated parameters.

    Must be called after ``backward()`` and before the optimizer step, whenever
    sequence parallelism is active.  Megatron-LM performs the equivalent step
    in ``finalize_model_grads`` (``_allreduce_layernorm_grads``); the operation
    is unavoidable in any sequence-parallel implementation.

    Args:
        module: Root module to scan for marked parameters.
        group: Sequence-parallel group to sum over.
        recorder: Optional communication instrumentation sink.

    Returns:
        The number of parameters whose gradients were reduced.
    """
    if group.size == 1:
        return 0
    from ..distributed.collectives import ReduceOp, all_reduce

    reduced = 0
    for param in module.parameters():
        if not getattr(param, "sequence_parallel_partial_grad", False):
            continue
        if param.grad is None:
            continue
        grad = param.grad.detach()
        if grad.is_contiguous():
            all_reduce(grad, group, op=ReduceOp.SUM, recorder=recorder).wait()
        else:
            # `.contiguous()` on a strided gradient returns a *copy*, so the
            # reduced values have to be written back explicitly.
            buffer = grad.contiguous()
            all_reduce(buffer, group, op=ReduceOp.SUM, recorder=recorder).wait()
            grad.copy_(buffer)
        reduced += 1
    return reduced
