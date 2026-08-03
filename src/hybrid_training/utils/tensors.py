"""Flattening, padding and partitioning utilities.

These are the primitives every sharding path in the project is built from, and
they are all pure functions of shapes and sizes.  Keeping them here -- rather
than inlining the arithmetic into the FSDP and checkpoint code -- means the
off-by-one errors that would otherwise manifest as a hang or as silently
corrupted weights can be caught by single-process unit tests.

The flat-parameter model
------------------------
An FSDP unit owns an ordered list of parameters.  Concatenating their
``reshape(-1)`` views produces one *flat parameter*::

    p0: shape (2, 3), numel 6      offsets [0, 6)
    p1: shape (4,),   numel 4      offsets [6, 10)
    p2: shape (3, 2), numel 6      offsets [10, 16)
                                   total numel 16

With a shard group of 3 ranks, 16 is padded to 18 and each rank keeps 6
elements::

    rank 0: flat[0:6]    -> all of p0
    rank 1: flat[6:12]   -> all of p1, plus the first 2 elements of p2
    rank 2: flat[12:18]  -> the last 4 elements of p2, plus 2 padding slots

Two facts fall out of this picture and drive most of the FSDP implementation:

* A parameter can straddle a shard boundary, so "which rank owns parameter *p*"
  is not a well-posed question.  Ownership is per *element*.
* Padding exists only in the flat buffer.  It is never exposed to the
  optimizer as a trainable value, it is always zeroed before a reduction, and
  it is dropped when a full state dict is reconstructed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch

from ..errors import ShardingError, format_error

__all__ = [
    "FlatEntry",
    "ShardRange",
    "build_flat_layout",
    "even_shard_ranges",
    "flatten_dense_tensors",
    "intersect_ranges",
    "pad_flat_tensor",
    "shard_range_for",
    "split_tensor_along_dim",
    "unflatten_to_views",
]


@dataclass(frozen=True)
class FlatEntry:
    """Where one tensor lives inside a flat buffer.

    Attributes:
        name: Fully-qualified parameter name, e.g. ``"blocks.0.mlp.fc1.weight"``.
        shape: Original tensor shape.
        offset: Index of the first element inside the flat buffer.
        numel: Number of elements.
        dtype_str: ``str(dtype)`` of the original tensor, recorded so that
            metadata can round-trip through JSON without importing ``torch``.
        requires_grad: Whether the original parameter required gradients.
    """

    name: str
    shape: tuple[int, ...]
    offset: int
    numel: int
    dtype_str: str
    requires_grad: bool = True

    @property
    def end(self) -> int:
        """One past the last element index."""
        return self.offset + self.numel

    def as_dict(self) -> dict[str, object]:
        """JSON-serialisable view, used by the checkpoint manifest."""
        return {
            "name": self.name,
            "shape": list(self.shape),
            "offset": self.offset,
            "numel": self.numel,
            "dtype": self.dtype_str,
            "requires_grad": self.requires_grad,
        }


@dataclass(frozen=True)
class ShardRange:
    """A half-open interval ``[start, start + length)`` of a flat buffer.

    Attributes:
        start: First element index.
        length: Number of elements.  May be zero for a rank that owns nothing,
            which happens when a flat parameter is smaller than the shard
            group.
    """

    start: int
    length: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.length < 0:
            raise ShardingError(
                format_error(
                    "tensors.ShardRange",
                    "shard ranges must be non-negative",
                    expected="start >= 0 and length >= 0",
                    observed=f"start={self.start}, length={self.length}",
                    resolution="check the offset arithmetic that produced this range",
                )
            )

    @property
    def end(self) -> int:
        """One past the last element index."""
        return self.start + self.length

    @property
    def is_empty(self) -> bool:
        """Whether the range covers no elements."""
        return self.length == 0

    def as_tuple(self) -> tuple[int, int]:
        """``(start, length)``."""
        return (self.start, self.length)


def build_flat_layout(
    named_tensors: Sequence[tuple[str, torch.Tensor]],
) -> tuple[tuple[FlatEntry, ...], int]:
    """Compute the layout of a flat buffer holding ``named_tensors`` in order.

    Args:
        named_tensors: ``(name, tensor)`` pairs, in the order they will be
            concatenated.  Order is part of the contract: every rank must
            produce the same order or their flat buffers will not correspond.

    Returns:
        ``(entries, total_numel)``.

    Raises:
        ShardingError: If ``named_tensors`` is empty or contains duplicate
            names.  Duplicates almost always mean a tied/shared parameter was
            handed to the same unit twice, which would make unflattening
            ambiguous.
    """
    if not named_tensors:
        raise ShardingError(
            format_error(
                "tensors.build_flat_layout",
                "cannot flatten an empty tensor list",
                expected=">= 1 tensor",
                observed=0,
                resolution="do not create a flat parameter for a module with no parameters",
            )
        )
    seen: set[str] = set()
    entries: list[FlatEntry] = []
    offset = 0
    for name, tensor in named_tensors:
        if name in seen:
            raise ShardingError(
                format_error(
                    "tensors.build_flat_layout",
                    "duplicate tensor name in a flat parameter; this indicates a shared "
                    "(tied) tensor being registered twice in the same unit",
                    expected="unique names",
                    observed=name,
                    resolution=(
                        "de-duplicate tied parameters before flattening, or place the "
                        "tied tensors in the same unit exactly once"
                    ),
                )
            )
        seen.add(name)
        entries.append(
            FlatEntry(
                name=name,
                shape=tuple(tensor.shape),
                offset=offset,
                numel=tensor.numel(),
                dtype_str=str(tensor.dtype),
                requires_grad=bool(tensor.requires_grad),
            )
        )
        offset += tensor.numel()
    return tuple(entries), offset


def flatten_dense_tensors(
    tensors: Iterable[torch.Tensor], *, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Concatenate tensors into one contiguous 1-D buffer.

    Args:
        tensors: Tensors to concatenate.  All must share a dtype and device.
        out: Optional pre-allocated destination of the right size.

    Returns:
        The flat buffer.  When ``out`` is given it is returned; otherwise a new
        tensor is allocated.

    Raises:
        ShardingError: On dtype/device mismatch or a wrongly sized ``out``.
    """
    tensor_list = list(tensors)
    if not tensor_list:
        raise ShardingError(
            format_error(
                "tensors.flatten_dense_tensors",
                "nothing to flatten",
                expected=">= 1 tensor",
                observed=0,
                resolution="pass at least one tensor",
            )
        )
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device
    total = 0
    for tensor in tensor_list:
        if tensor.dtype != dtype or tensor.device != device:
            raise ShardingError(
                format_error(
                    "tensors.flatten_dense_tensors",
                    "all tensors in a flat buffer must share dtype and device",
                    expected=f"{dtype} on {device}",
                    observed=f"{tensor.dtype} on {tensor.device}",
                    resolution="group parameters by dtype/device before flattening",
                )
            )
        total += tensor.numel()

    if out is None:
        out = torch.empty(total, dtype=dtype, device=device)
    elif out.numel() != total:
        raise ShardingError(
            format_error(
                "tensors.flatten_dense_tensors",
                "destination buffer is the wrong size",
                expected=total,
                observed=out.numel(),
                resolution="allocate a buffer with exactly the total element count",
            )
        )

    offset = 0
    for tensor in tensor_list:
        numel = tensor.numel()
        out[offset : offset + numel].copy_(tensor.detach().reshape(-1))
        offset += numel
    return out


def unflatten_to_views(flat: torch.Tensor, entries: Sequence[FlatEntry]) -> list[torch.Tensor]:
    """Return one *view* per entry into ``flat``.

    Views, not copies: writing through a returned tensor writes into ``flat``.
    This is what lets an FSDP unit rebind a module's parameters to slices of an
    all-gathered buffer without any data movement.

    Args:
        flat: 1-D buffer at least as long as the entries require.
        entries: Layout produced by :func:`build_flat_layout`.

    Returns:
        Views shaped like the original tensors, in entry order.

    Raises:
        ShardingError: If ``flat`` is not 1-D or is too short.
    """
    if flat.dim() != 1:
        raise ShardingError(
            format_error(
                "tensors.unflatten_to_views",
                "the flat buffer must be one-dimensional",
                expected="dim() == 1",
                observed=flat.dim(),
                resolution="reshape the buffer to (-1,) before unflattening",
            )
        )
    required = entries[-1].end if entries else 0
    if flat.numel() < required:
        raise ShardingError(
            format_error(
                "tensors.unflatten_to_views",
                "flat buffer is too short for the layout",
                expected=f">= {required} elements",
                observed=flat.numel(),
                resolution="allocate the padded size, not the logical size",
            )
        )
    return [flat[entry.offset : entry.end].view(entry.shape) for entry in entries]


def pad_flat_tensor(flat: torch.Tensor, multiple: int, *, value: float = 0.0) -> torch.Tensor:
    """Right-pad a 1-D tensor so its length is a multiple of ``multiple``.

    Args:
        flat: 1-D tensor.
        multiple: Desired divisor, normally the shard-group size.
        value: Padding value.  Always ``0`` in this project so that padding
            contributes nothing to a sum reduction or a norm.

    Returns:
        ``flat`` itself when it is already aligned, otherwise a new padded
        tensor.

    Raises:
        ShardingError: If ``flat`` is not 1-D or ``multiple`` is not positive.
    """
    if flat.dim() != 1:
        raise ShardingError(
            format_error(
                "tensors.pad_flat_tensor",
                "expected a one-dimensional tensor",
                expected="dim() == 1",
                observed=flat.dim(),
                resolution="flatten before padding",
            )
        )
    if multiple < 1:
        raise ShardingError(
            format_error(
                "tensors.pad_flat_tensor",
                "the alignment multiple must be positive",
                expected=">= 1",
                observed=multiple,
                resolution="pass the shard-group size",
            )
        )
    remainder = flat.numel() % multiple
    if remainder == 0:
        return flat
    padding = multiple - remainder
    return torch.cat([flat, torch.full((padding,), value, dtype=flat.dtype, device=flat.device)])


def even_shard_ranges(total_numel: int, num_shards: int) -> tuple[ShardRange, ...]:
    """Split ``[0, total_numel)`` into ``num_shards`` equal padded ranges.

    Every range has the *same* length ``ceil(total_numel / num_shards)``.  The
    last range(s) may extend past ``total_numel`` -- those elements are the
    padding.  Equal lengths are what make ``all_gather_into_tensor`` and
    ``reduce_scatter_tensor`` usable.

    Args:
        total_numel: Logical element count before padding.
        num_shards: Number of shards.

    Returns:
        One :class:`ShardRange` per shard, in rank order.

    Raises:
        ShardingError: If ``num_shards`` is not positive.

    Example:
        >>> [r.as_tuple() for r in even_shard_ranges(10, 4)]
        [(0, 3), (3, 3), (6, 3), (9, 3)]
    """
    if num_shards < 1:
        raise ShardingError(
            format_error(
                "tensors.even_shard_ranges",
                "num_shards must be positive",
                expected=">= 1",
                observed=num_shards,
                resolution="pass the shard-group size",
            )
        )
    shard_numel = (total_numel + num_shards - 1) // num_shards
    return tuple(ShardRange(start=i * shard_numel, length=shard_numel) for i in range(num_shards))


def shard_range_for(total_numel: int, num_shards: int, shard_index: int) -> ShardRange:
    """Return one rank's padded range within a flat buffer.

    Args:
        total_numel: Logical element count.
        num_shards: Number of shards.
        shard_index: Which shard.

    Returns:
        The range for ``shard_index``.

    Raises:
        ShardingError: If ``shard_index`` is out of range.
    """
    if not 0 <= shard_index < num_shards:
        raise ShardingError(
            format_error(
                "tensors.shard_range_for",
                "shard index outside the group",
                expected=f"0 <= i < {num_shards}",
                observed=shard_index,
                resolution="pass the rank's index within the shard group",
            )
        )
    return even_shard_ranges(total_numel, num_shards)[shard_index]


def intersect_ranges(a: ShardRange, b: ShardRange) -> ShardRange | None:
    """Intersect two flat-buffer ranges.

    This is the whole of checkpoint resharding: a saved shard and a wanted
    shard are both intervals in the *same global flat coordinate system*, so
    the bytes to copy are exactly their intersection.  No knowledge of the
    saving world size is needed beyond the offsets recorded in the manifest.

    Args:
        a: First range.
        b: Second range.

    Returns:
        The overlapping range, or ``None`` when they are disjoint.

    Example:
        >>> intersect_ranges(ShardRange(0, 6), ShardRange(4, 6)).as_tuple()
        (4, 2)
        >>> intersect_ranges(ShardRange(0, 4), ShardRange(4, 4)) is None
        True
    """
    start = max(a.start, b.start)
    end = min(a.end, b.end)
    if end <= start:
        return None
    return ShardRange(start=start, length=end - start)


def split_tensor_along_dim(
    tensor: torch.Tensor, dim: int, num_parts: int, *, contiguous: bool = True
) -> tuple[torch.Tensor, ...]:
    """Split a tensor into ``num_parts`` equal slices along ``dim``.

    Used by the tensor-parallel layers (splitting weight matrices and
    activations) and by the sequence-parallel scatter.  Divisibility is
    *required*: an uneven split would make the per-rank shapes differ, which
    breaks the single-buffer collectives and silently changes the mathematics
    of an all-gather.

    Args:
        tensor: Tensor to split.
        dim: Dimension to split along.  Negative values are normalised.
        num_parts: Number of equal parts.
        contiguous: Call ``.contiguous()`` on each slice.  Required before
            passing a slice to a collective.

    Returns:
        ``num_parts`` slices in order.

    Raises:
        ShardingError: If the dimension is not divisible by ``num_parts`` or
            ``dim`` is out of range.

    Example:
        >>> import torch
        >>> parts = split_tensor_along_dim(torch.arange(8).view(2, 4), dim=1, num_parts=2)
        >>> parts[0].tolist()
        [[0, 1], [4, 5]]
    """
    if num_parts < 1:
        raise ShardingError(
            format_error(
                "tensors.split_tensor_along_dim",
                "num_parts must be positive",
                expected=">= 1",
                observed=num_parts,
                resolution="pass the group size",
            )
        )
    if not -tensor.dim() <= dim < tensor.dim():
        raise ShardingError(
            format_error(
                "tensors.split_tensor_along_dim",
                "dimension index out of range",
                expected=f"-{tensor.dim()} <= dim < {tensor.dim()}",
                observed=dim,
                resolution="check the tensor rank",
            )
        )
    dim = dim % tensor.dim()
    size = tensor.shape[dim]
    if size % num_parts != 0:
        raise ShardingError(
            format_error(
                "tensors.split_tensor_along_dim",
                f"dimension {dim} of size {size} is not divisible by {num_parts}",
                expected=f"{size} % {num_parts} == 0",
                observed=size % num_parts,
                resolution=(
                    "choose a parallel size that divides the dimension, or pad the "
                    "tensor before splitting (see docs/05_tensor_parallelism.md)"
                ),
            )
        )
    chunk = size // num_parts
    parts = tuple(tensor.narrow(dim, i * chunk, chunk) for i in range(num_parts))
    if contiguous:
        parts = tuple(p.contiguous() for p in parts)
    return parts


def numel_of(named_tensors: Iterable[tuple[str, torch.Tensor]]) -> int:
    """Total element count of a named tensor collection.

    Args:
        named_tensors: ``(name, tensor)`` pairs.

    Returns:
        Sum of ``tensor.numel()``.
    """
    return sum(tensor.numel() for _, tensor in named_tensors)


def bytes_of(tensors: Iterable[torch.Tensor]) -> int:
    """Total storage size in bytes of the given tensors.

    Args:
        tensors: Tensors to measure.

    Returns:
        Sum of ``numel * element_size``.
    """
    return sum(t.numel() * t.element_size() for t in tensors)
