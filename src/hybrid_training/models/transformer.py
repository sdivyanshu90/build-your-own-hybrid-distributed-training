"""A small transformer with tensor and sequence parallelism built in.

Design decision: one code path, not two
=======================================
The model is written *only* in terms of the parallel layers.  When tensor
parallelism is disabled the caller passes a one-member
:meth:`~hybrid_training.distributed.groups.GroupHandle.trivial` group, every
collective short-circuits to the identity, and the result is bit-for-bit what a
plain ``nn.Linear`` model would have produced.

The alternative -- an ``if tensor_parallel: ... else: ...`` in every layer --
would mean the single-process reference used by the equivalence tests exercises
*different code* from the distributed model, so the tests would prove much
less.

Shapes
======
Activations are ``(batch, sequence, hidden)`` everywhere.  With a
tensor-parallel size ``T`` and a sequence-parallel size ``G``:

=====================================  ================================
Location                               Shape on each rank
=====================================  ================================
input token ids                        ``(B, S)``
after embedding                        ``(B, S, H)``
after sequence scatter                 ``(B, S/G, H)``
inside a block, after LayerNorm        ``(B, S/G, H)``
after q/k/v column-parallel            ``(B, S, H/T)``    (sequence gathered)
attention scores                       ``(B, n_head/T, S, S)``
after attention output row-parallel    ``(B, S/G, H)``    (reduce-scattered)
after the feed-forward's fc1           ``(B, S, F/T)``
after the feed-forward's fc2           ``(B, S/G, H)``
logits                                 ``(B, S, V)``
=====================================  ================================

Attention is *not* communication-free
=====================================
The q/k/v projections all-gather the sequence, so every rank sees every
position for the heads it owns.  Attention itself then needs no communication,
because a head's computation is self-contained.  But the gather is real: it
moves ``(B, S, H)`` bytes per parallel region.  In the fused Megatron schedule
that gather replaces half of an all-reduce that would have happened anyway, so
it is free relative to plain tensor parallelism -- but it is not free relative
to *no* parallelism, and no implementation that keeps ordinary attention can
make it so.  Keeping the sequence split *through* attention requires context
parallelism (ring attention); see ``docs/06_sequence_parallelism.md``.

Padding and causal masking
==========================
When the sequence length is not divisible by ``G`` it is right-padded with
zeros.  Under **causal** attention this is harmless for the real positions:
position ``i`` attends only to ``j <= i``, and every padded position has
``j > i`` for all real ``i``.  The padded outputs are garbage and are discarded
when the sequence is un-padded.  Under non-causal attention a key mask would be
required, which is why :class:`TinyTransformer` defaults to causal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from ..distributed.groups import GroupHandle
from ..errors import ConfigurationError, UnsupportedFeatureError, format_error
from ..parallel.sequence_parallel import (
    SEQUENCE_DIM,
    SequenceParallelLayerNorm,
    SequenceShardInfo,
    gather_sequence,
    local_sequence_slice,
    pad_sequence_dimension,
    scatter_sequence,
    unpad_sequence_dimension,
)
from ..parallel.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    TensorParallelFeedForward,
    VocabParallelEmbedding,
    init_linear_parameters,
)
from ..utils.reproducibility import derive_seed, temporary_seed

__all__ = [
    "ParallelPlan",
    "TensorParallelAttention",
    "TinyTransformer",
    "TransformerBlock",
]


@dataclass(frozen=True)
class ParallelPlan:
    """Which groups the model's layers should use.

    Attributes:
        tensor_group: Group over which weight matrices and attention heads are
            partitioned.  Pass a trivial group to disable tensor parallelism.
        sequence_group: Group over which the sequence dimension is split.  In
            the fused Megatron schedule this is the *same handle* as
            ``tensor_group``.
        sequence_parallel: Whether to split the sequence in the regions between
            tensor-parallel layers.
        vocab_parallel: Whether the embedding table is partitioned over
            ``tensor_group``.
    """

    tensor_group: GroupHandle
    sequence_group: GroupHandle
    sequence_parallel: bool = False
    vocab_parallel: bool = True

    @classmethod
    def single_process(cls) -> ParallelPlan:
        """A plan with no parallelism at all, for the reference model."""
        trivial = GroupHandle.trivial("tensor")
        return cls(tensor_group=trivial, sequence_group=trivial, sequence_parallel=False)

    def __post_init__(self) -> None:
        if self.sequence_parallel and self.sequence_group.size == 1:
            raise ConfigurationError(
                format_error(
                    "models.ParallelPlan",
                    "sequence_parallel was requested but the sequence group has one member, "
                    "so there is nothing to split the sequence across",
                    expected="sequence_group.size > 1",
                    observed=self.sequence_group.size,
                    resolution="pass a real sequence-parallel group or disable the flag",
                )
            )


class TensorParallelAttention(nn.Module):
    """Multi-head self-attention with heads partitioned across a group.

    Each rank owns ``num_heads / T`` complete heads.  Because a head's
    computation never crosses head boundaries, partitioning by head is exact:
    the concatenation of the ranks' outputs *is* the unsharded output, and the
    output projection's row-parallel reduction reassembles it.

    Separate q/k/v projections are used rather than one fused ``3H`` projection.
    A fused projection is one matmul instead of three, but a column-parallel
    split of ``[Q; K; V]`` gives rank ``t`` a contiguous band of the stacked
    matrix, which is *not* ``(Q_t, K_t, V_t)`` unless the full matrix is stored
    in an interleaved-by-partition layout.  Megatron does exactly that
    interleaving; here the three-projection form is used because it keeps the
    weights directly comparable with an unsharded reference, which the tests
    depend on.  The cost is documented in ``docs/05_tensor_parallelism.md``.

    Args:
        hidden_size: Model width.
        num_heads: Total heads.  Must be divisible by the tensor-parallel size.
        plan: Which groups to use.
        dropout: Attention dropout probability.
        bias: Whether the projections have biases.
        causal: Apply a causal mask.
        init_seed: Base seed for weight initialisation.
        device: Construction device.
        dtype: Parameter dtype.

    Raises:
        ConfigurationError: If the heads do not divide across the group.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        plan: ParallelPlan,
        *,
        dropout: float = 0.0,
        bias: bool = True,
        causal: bool = True,
        init_seed: int = 0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        tensor_size = plan.tensor_group.size
        if num_heads % tensor_size != 0:
            raise ConfigurationError(
                format_error(
                    "models.TensorParallelAttention",
                    "attention heads must divide evenly across the tensor-parallel group; "
                    "splitting a head would break the softmax, which is computed over a "
                    "whole head's scores",
                    rank=plan.tensor_group.global_rank,
                    expected=f"num_heads % {tensor_size} == 0",
                    observed=num_heads,
                    resolution="choose num_heads as a multiple of the tensor-parallel size",
                )
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.heads_per_partition = num_heads // tensor_size
        self.causal = causal
        self.dropout = dropout
        self.plan = plan

        sequence_parallel = plan.sequence_parallel
        common: dict[str, Any] = {
            "bias": bias,
            "gather_output": False,
            "sequence_parallel": sequence_parallel,
            "device": device,
            "dtype": dtype,
        }
        self.query = ColumnParallelLinear(
            hidden_size,
            hidden_size,
            plan.tensor_group,
            init_seed=derive_seed(init_seed, "attn-q"),
            **common,
        )
        self.key = ColumnParallelLinear(
            hidden_size,
            hidden_size,
            plan.tensor_group,
            init_seed=derive_seed(init_seed, "attn-k"),
            **common,
        )
        self.value = ColumnParallelLinear(
            hidden_size,
            hidden_size,
            plan.tensor_group,
            init_seed=derive_seed(init_seed, "attn-v"),
            **common,
        )
        self.output = RowParallelLinear(
            hidden_size,
            hidden_size,
            plan.tensor_group,
            bias=bias,
            input_is_parallel=True,
            sequence_parallel=sequence_parallel,
            init_seed=derive_seed(init_seed, "attn-o"),
            device=device,
            dtype=dtype,
        )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape ``(B, S, H/T)`` into ``(B, heads/T, S, head_dim)``."""
        batch, sequence, _ = x.shape
        return x.view(batch, sequence, self.heads_per_partition, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Run partitioned self-attention.

        Args:
            x: ``(batch, sequence, hidden)``; sequence-sharded when the plan
                enables sequence parallelism.
            attention_mask: Optional additive mask broadcastable to
                ``(batch, 1, sequence, sequence)``.  Added to the scores before
                the softmax, so masked entries should be large negatives.

        Returns:
            Same shape as ``x``.
        """
        q = self._split_heads(self.query(x))
        k = self._split_heads(self.key(x))
        v = self._split_heads(self.value(x))

        # Attention is computed explicitly rather than through
        # scaled_dot_product_attention so that the arithmetic is visible and
        # bitwise reproducible across CPU and CUDA backends.
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.causal:
            sequence = scores.shape[-1]
            causal_mask = torch.triu(
                torch.ones(sequence, sequence, dtype=torch.bool, device=scores.device),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))
        if attention_mask is not None:
            scores = scores + attention_mask

        weights = torch.softmax(scores, dim=-1)
        if self.dropout > 0.0 and self.training:
            weights = F.dropout(weights, p=self.dropout)

        context = torch.matmul(weights, v)
        batch, _, sequence, _ = context.shape
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch, sequence, self.heads_per_partition * self.head_dim)
        )
        return self.output(context)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: ``x + attn(ln(x))`` then ``x + ffn(ln(x))``.

    Pre-norm (rather than post-norm) is used because it trains without a
    warm-up schedule, which keeps the test configurations small and stable.

    Under sequence parallelism the residual stream stays sequence-sharded from
    end to end: the LayerNorms operate on shards (no communication), the
    attention and feed-forward blocks gather the sequence internally and
    reduce-scatter it back, and both residual adds therefore see matching
    shapes.

    Args:
        config: Model configuration.
        plan: Parallel groups.
        layer_index: Position in the stack; mixed into the initialisation seed
            so layers are not identical.
        init_seed: Base seed.
        device: Construction device.
        dtype: Parameter dtype.
    """

    def __init__(
        self,
        config: ModelConfig,
        plan: ParallelPlan,
        *,
        layer_index: int = 0,
        init_seed: int = 0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        seed = derive_seed(init_seed, "block", layer_index)
        self.input_norm = SequenceParallelLayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            group=plan.tensor_group,
            sequence_parallel=plan.sequence_parallel,
            device=device,
            dtype=dtype,
        )
        self.attention = TensorParallelAttention(
            config.hidden_size,
            config.num_heads,
            plan,
            dropout=config.dropout,
            causal=True,
            init_seed=derive_seed(seed, "attn"),
            device=device,
            dtype=dtype,
        )
        self.post_attention_norm = SequenceParallelLayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            group=plan.tensor_group,
            sequence_parallel=plan.sequence_parallel,
            device=device,
            dtype=dtype,
        )
        self.feed_forward = TensorParallelFeedForward(
            config.hidden_size,
            config.resolved_ffn_hidden_size,
            plan.tensor_group,
            activation=config.activation,
            sequence_parallel=plan.sequence_parallel,
            init_seed=derive_seed(seed, "ffn"),
            device=device,
            dtype=dtype,
        )
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Run the block.

        Args:
            x: ``(batch, sequence, hidden)``, possibly sequence-sharded.
            attention_mask: Optional additive attention mask.

        Returns:
            Same shape as ``x``.
        """
        hidden = self.attention(self.input_norm(x), attention_mask)
        if self.dropout > 0.0 and self.training:
            hidden = F.dropout(hidden, p=self.dropout)
        x = x + hidden

        hidden = self.feed_forward(self.post_attention_norm(x))
        if self.dropout > 0.0 and self.training:
            hidden = F.dropout(hidden, p=self.dropout)
        return x + hidden


class TinyTransformer(nn.Module):
    """A small decoder-only transformer with optional tensor/sequence parallelism.

    Args:
        config: Model configuration; ``kind`` must be ``"transformer"``.
        plan: Which groups to parallelise over.  Defaults to no parallelism.
        seed: Master seed for weight initialisation.
        device: Construction device.
        dtype: Parameter dtype.

    Raises:
        ConfigurationError: If the configuration is not a transformer
            configuration.
        UnsupportedFeatureError: If tied embeddings are requested together with
            a partitioned vocabulary, which would require the output projection
            and the embedding to be sharded along incompatible dimensions.

    Example:
        >>> from hybrid_training.config import ModelConfig
        >>> cfg = ModelConfig(kind="transformer", vocab_size=32, hidden_size=16,
        ...                   num_heads=4, num_layers=2, max_sequence_length=8)
        >>> model = TinyTransformer(cfg, seed=0)
        >>> model(torch.zeros(2, 8, dtype=torch.long)).shape
        torch.Size([2, 8, 32])
    """

    def __init__(
        self,
        config: ModelConfig,
        plan: ParallelPlan | None = None,
        *,
        seed: int = 0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if config.kind != "transformer":
            raise ConfigurationError(
                format_error(
                    "models.TinyTransformer",
                    "model kind must be 'transformer'",
                    expected="transformer",
                    observed=config.kind,
                    resolution="set ModelConfig.kind='transformer'",
                )
            )
        self.config = config
        self.plan = plan if plan is not None else ParallelPlan.single_process()
        if (
            config.tie_word_embeddings
            and self.plan.vocab_parallel
            and self.plan.tensor_group.size > 1
        ):
            raise UnsupportedFeatureError(
                format_error(
                    "models.TinyTransformer",
                    "tied word embeddings with a vocabulary-parallel embedding are not "
                    "supported: the embedding is sharded along the vocabulary while the "
                    "output projection needs the same matrix sharded along its output "
                    "features, so one tensor cannot serve both without a transpose-aware "
                    "all-to-all",
                    expected="tie_word_embeddings=False, or vocab_parallel=False",
                    observed="both enabled",
                    resolution="untie the weights, or disable vocabulary partitioning",
                )
            )

        vocab_group = (
            self.plan.tensor_group if self.plan.vocab_parallel else GroupHandle.trivial("tensor")
        )
        self.token_embedding = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            vocab_group,
            init_seed=derive_seed(seed, "token-embedding"),
            device=device,
            dtype=dtype,
        )

        with temporary_seed(derive_seed(seed, "position-embedding")):
            position_weight = torch.empty(
                config.max_sequence_length, config.hidden_size, device=device, dtype=dtype
            )
            nn.init.normal_(position_weight, mean=0.0, std=0.02)
        self.position_embedding = nn.Embedding(
            config.max_sequence_length, config.hidden_size, device=device, dtype=dtype
        )
        with torch.no_grad():
            self.position_embedding.weight.copy_(position_weight)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config,
                    self.plan,
                    layer_index=index,
                    init_seed=derive_seed(seed, "blocks"),
                    device=device,
                    dtype=dtype,
                )
                for index in range(config.num_layers)
            ]
        )
        self.final_norm = SequenceParallelLayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            group=self.plan.tensor_group,
            sequence_parallel=self.plan.sequence_parallel,
            device=device,
            dtype=dtype,
        )
        self.output_projection = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            self.plan.tensor_group,
            bias=False,
            gather_output=True,
            sequence_parallel=False,
            init_seed=derive_seed(seed, "output-projection"),
            device=device,
            dtype=dtype,
        )
        if config.tie_word_embeddings:
            # Only reachable when the vocabulary is not partitioned, so both
            # tensors are the full (vocab, hidden) matrix.
            self.output_projection.weight = self.token_embedding.weight

        self.dropout = config.dropout

    def forward(
        self, token_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Map token ids to logits.

        Args:
            token_ids: ``(batch, sequence)`` integer tensor.
            attention_mask: Optional additive mask broadcastable to
                ``(batch, 1, sequence, sequence)``.

        Returns:
            ``(batch, sequence, vocab_size)`` logits, gathered on every rank.

        Raises:
            ConfigurationError: If the sequence exceeds
                ``max_sequence_length``.
        """
        batch, sequence = token_ids.shape
        if sequence > self.config.max_sequence_length:
            raise ConfigurationError(
                format_error(
                    "models.TinyTransformer.forward",
                    "sequence longer than the positional table",
                    expected=f"<= {self.config.max_sequence_length}",
                    observed=sequence,
                    resolution="raise ModelConfig.max_sequence_length or shorten the input",
                )
            )

        hidden = self.token_embedding(token_ids)
        positions = torch.arange(sequence, device=token_ids.device)
        hidden = hidden + self.position_embedding(positions).unsqueeze(0)
        if self.dropout > 0.0 and self.training:
            hidden = F.dropout(hidden, p=self.dropout)

        shard_info: SequenceShardInfo | None = None
        if self.plan.sequence_parallel:
            hidden, shard_info = pad_sequence_dimension(
                hidden, self.plan.sequence_group.size, dim=SEQUENCE_DIM
            )
            if attention_mask is not None and shard_info.requires_padding:
                raise UnsupportedFeatureError(
                    format_error(
                        "models.TinyTransformer.forward",
                        "an explicit attention mask combined with sequence padding is not "
                        "supported: the mask would have to be extended to cover the padded "
                        "positions, and its correct extension depends on what the mask means",
                        expected="a sequence length divisible by the sequence-parallel size",
                        observed=f"length {sequence}, group {self.plan.sequence_group.size}",
                        resolution=(
                            "pad the batch to a multiple of the sequence-parallel size "
                            "before calling forward, or drop the explicit mask and rely on "
                            "causal masking"
                        ),
                    )
                )
            hidden = scatter_sequence(hidden, self.plan.sequence_group, dim=SEQUENCE_DIM)

        for block in self.blocks:
            hidden = block(hidden, attention_mask)

        hidden = self.final_norm(hidden)

        if self.plan.sequence_parallel:
            hidden = gather_sequence(hidden, self.plan.sequence_group, dim=SEQUENCE_DIM)
            assert shard_info is not None
            hidden = unpad_sequence_dimension(hidden, shard_info, dim=SEQUENCE_DIM)

        return self.output_projection(hidden)

    def local_positions(self, sequence_length: int) -> tuple[int, int]:
        """Return the sequence positions this rank owns after scattering.

        Exposed for diagnostics and for tests that assert the split is what the
        documentation claims.

        Args:
            sequence_length: Padded sequence length.

        Returns:
            ``(start, end)`` positions.
        """
        if not self.plan.sequence_parallel:
            return 0, sequence_length
        return local_sequence_slice(sequence_length, self.plan.sequence_group)

    def num_parameters(self) -> int:
        """Number of parameter elements held by this rank.

        Under tensor parallelism this is smaller than the model's total
        parameter count, because each rank owns only its slices.
        """
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (
            f"TinyTransformer(layers={self.config.num_layers}, "
            f"hidden={self.config.hidden_size}, heads={self.config.num_heads}, "
            f"vocab={self.config.vocab_size}, tp={self.plan.tensor_group.size}, "
            f"sp={self.plan.sequence_group.size if self.plan.sequence_parallel else 1})"
        )


def build_reference_linear(
    in_features: int,
    out_features: int,
    *,
    bias: bool = True,
    init_seed: int = 0,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> nn.Linear:
    """Build an ``nn.Linear`` whose weights match a parallel layer's full matrix.

    Uses the same seed derivation and the same initialisation calls as
    :class:`~hybrid_training.parallel.tensor_parallel.ColumnParallelLinear`, so
    a test can compare the two directly without copying weights across.

    Args:
        in_features: Input width.
        out_features: Output width.
        bias: Whether to include a bias.
        init_seed: The value passed as the parallel layer's ``init_seed``.
        device: Construction device.
        dtype: Parameter dtype.

    Returns:
        The reference layer.
    """
    linear = nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
    with temporary_seed(derive_seed(init_seed, "tensor-parallel-init")), torch.no_grad():
        weight = torch.empty(out_features, in_features, device=device, dtype=dtype)
        bias_tensor = torch.empty(out_features, device=device, dtype=dtype) if bias else None
        init_linear_parameters(weight, bias_tensor, in_features=in_features)
        linear.weight.copy_(weight)
        if bias_tensor is not None:
            assert linear.bias is not None
            linear.bias.copy_(bias_tensor)
    return linear
