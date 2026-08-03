"""Unit tests for flattening, padding, sharding and partition arithmetic.

Everything here is the offset arithmetic that FSDP and the checkpoint format
are built on.  An error in it produces silently wrong weights rather than an
exception, so the tests cover the boundary cases explicitly: empty shards,
parameters that straddle a shard boundary, and padding.
"""

from __future__ import annotations

import itertools

import pytest
import torch

from hybrid_training.errors import ShardingError
from hybrid_training.utils.tensors import (
    ShardRange,
    build_flat_layout,
    bytes_of,
    even_shard_ranges,
    flatten_dense_tensors,
    intersect_ranges,
    numel_of,
    pad_flat_tensor,
    shard_range_for,
    split_tensor_along_dim,
    unflatten_to_views,
)


class TestFlatLayout:
    """Flattening a parameter list into one buffer."""

    def test_layout_matches_the_documented_example(self) -> None:
        """The worked example in the module docstring is reproduced exactly."""
        tensors = [
            ("p0", torch.zeros(2, 3)),
            ("p1", torch.zeros(4)),
            ("p2", torch.zeros(3, 2)),
        ]
        entries, total = build_flat_layout(tensors)
        assert total == 16
        assert [(e.name, e.offset, e.numel) for e in entries] == [
            ("p0", 0, 6),
            ("p1", 6, 4),
            ("p2", 10, 6),
        ]
        assert entries[2].end == 16

    def test_flatten_unflatten_is_lossless(self) -> None:
        """Round-tripping through the flat buffer preserves every value."""
        originals = [torch.randn(2, 3), torch.randn(4), torch.randn(3, 2)]
        named = [(f"p{i}", t) for i, t in enumerate(originals)]
        entries, total = build_flat_layout(named)
        flat = flatten_dense_tensors(originals)
        assert flat.numel() == total
        views = unflatten_to_views(flat, entries)
        for original, view in zip(originals, views):
            # Bitwise: this is a copy, not an arithmetic operation.
            assert torch.equal(original, view)

    def test_views_alias_the_flat_buffer(self) -> None:
        """Unflattened tensors are views, so writes go through to the buffer."""
        entries, _ = build_flat_layout([("a", torch.zeros(2, 2)), ("b", torch.zeros(3))])
        flat = torch.zeros(7)
        views = unflatten_to_views(flat, entries)
        views[0][0, 0] = 5.0
        views[1][2] = 9.0
        assert flat[0].item() == 5.0
        assert flat[6].item() == 9.0

    def test_duplicate_names_rejected(self) -> None:
        """A repeated name means a tied tensor was registered twice."""
        with pytest.raises(ShardingError, match="duplicate tensor name"):
            build_flat_layout([("w", torch.zeros(2)), ("w", torch.zeros(2))])

    def test_empty_list_rejected(self) -> None:
        """A unit with no parameters cannot have a flat parameter."""
        with pytest.raises(ShardingError, match="cannot flatten an empty tensor list"):
            build_flat_layout([])

    def test_dtype_mismatch_rejected(self) -> None:
        """One flat buffer cannot hold two dtypes."""
        with pytest.raises(ShardingError, match="share dtype and device"):
            flatten_dense_tensors([torch.zeros(2), torch.zeros(2, dtype=torch.float64)])

    def test_wrong_sized_output_rejected(self) -> None:
        """A pre-allocated destination of the wrong size is an error."""
        with pytest.raises(ShardingError, match="wrong size"):
            flatten_dense_tensors([torch.zeros(3)], out=torch.zeros(4))

    def test_unflatten_requires_one_dimensional_buffer(self) -> None:
        """Unflattening a 2-D "flat" buffer is rejected."""
        entries, _ = build_flat_layout([("a", torch.zeros(4))])
        with pytest.raises(ShardingError, match="one-dimensional"):
            unflatten_to_views(torch.zeros(2, 2), entries)

    def test_unflatten_requires_a_long_enough_buffer(self) -> None:
        """A short buffer is rejected rather than silently truncating."""
        entries, _ = build_flat_layout([("a", torch.zeros(8))])
        with pytest.raises(ShardingError, match="too short"):
            unflatten_to_views(torch.zeros(4), entries)


class TestPadding:
    """Alignment of flat buffers to the shard-group size."""

    @pytest.mark.parametrize(
        ("numel", "multiple", "expected"),
        [(10, 4, 12), (8, 4, 8), (1, 3, 3), (7, 1, 7), (0, 4, 0)],
    )
    def test_pad_to_multiple(self, numel: int, multiple: int, expected: int) -> None:
        """Padding rounds up to the next multiple and no further."""
        padded = pad_flat_tensor(torch.ones(numel), multiple)
        assert padded.numel() == expected

    def test_padding_is_zero(self) -> None:
        """Padding is zero so it contributes nothing to a sum or a norm."""
        padded = pad_flat_tensor(torch.ones(5), 4)
        assert padded.numel() == 8
        assert torch.equal(padded[5:], torch.zeros(3))

    def test_aligned_input_is_returned_unchanged(self) -> None:
        """An already-aligned buffer is not copied."""
        original = torch.ones(8)
        assert pad_flat_tensor(original, 4) is original

    def test_invalid_arguments(self) -> None:
        """Non-1-D input and non-positive multiples are rejected."""
        with pytest.raises(ShardingError, match="one-dimensional"):
            pad_flat_tensor(torch.ones(2, 2), 4)
        with pytest.raises(ShardingError, match="alignment multiple must be positive"):
            pad_flat_tensor(torch.ones(4), 0)


class TestShardRanges:
    """Splitting a padded buffer into equal ranges."""

    def test_documented_example(self) -> None:
        """``even_shard_ranges(10, 4)`` produces four 3-element ranges."""
        ranges = even_shard_ranges(10, 4)
        assert [r.as_tuple() for r in ranges] == [(0, 3), (3, 3), (6, 3), (9, 3)]
        # The last range extends past 10: those two elements are the padding.
        assert ranges[-1].end == 12

    def test_all_ranges_have_equal_length(self) -> None:
        """Equal lengths are what makes the single-buffer collectives usable."""
        for numel in range(1, 40):
            for shards in range(1, 7):
                lengths = {r.length for r in even_shard_ranges(numel, shards)}
                assert len(lengths) == 1, (numel, shards)

    def test_ranges_are_contiguous_and_cover_the_padded_buffer(self) -> None:
        """Ranges tile ``[0, padded_numel)`` with no gaps or overlaps."""
        ranges = even_shard_ranges(17, 4)
        for previous, current in itertools.pairwise(ranges):
            assert previous.end == current.start
        assert ranges[0].start == 0
        assert ranges[-1].end == 4 * ranges[0].length

    def test_shard_smaller_than_group_gives_every_rank_something(self) -> None:
        """A 2-element tensor across 4 shards still yields equal ranges."""
        ranges = even_shard_ranges(2, 4)
        assert all(r.length == 1 for r in ranges)
        # Ranks 2 and 3 own only padding.
        assert ranges[2].start >= 2

    def test_shard_range_for_index(self) -> None:
        """The per-rank helper agrees with the full enumeration."""
        assert shard_range_for(10, 4, 2).as_tuple() == (6, 3)
        with pytest.raises(ShardingError, match="shard index outside the group"):
            shard_range_for(10, 4, 4)

    def test_negative_range_rejected(self) -> None:
        """A negative offset or length is a programming error."""
        with pytest.raises(ShardingError, match="non-negative"):
            ShardRange(start=-1, length=4)

    def test_num_shards_must_be_positive(self) -> None:
        """Zero shards is rejected."""
        with pytest.raises(ShardingError, match="num_shards must be positive"):
            even_shard_ranges(8, 0)


class TestIntersection:
    """The interval arithmetic that implements resharding."""

    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            ((0, 6), (4, 6), (4, 2)),
            ((0, 4), (4, 4), None),
            ((0, 10), (2, 3), (2, 3)),
            ((5, 5), (0, 3), None),
            ((0, 3), (0, 3), (0, 3)),
            ((3, 0), (0, 10), None),
        ],
    )
    def test_intersections(
        self, a: tuple[int, int], b: tuple[int, int], expected: tuple[int, int] | None
    ) -> None:
        """Overlaps and disjoint pairs both behave."""
        result = intersect_ranges(ShardRange(*a), ShardRange(*b))
        assert (None if result is None else result.as_tuple()) == expected

    def test_intersection_is_symmetric(self) -> None:
        """Intersecting in either order gives the same interval."""
        for start_a in range(0, 8):
            for start_b in range(0, 8):
                first = ShardRange(start_a, 4)
                second = ShardRange(start_b, 3)
                forward = intersect_ranges(first, second)
                backward = intersect_ranges(second, first)
                assert (forward is None) == (backward is None)
                if forward is not None and backward is not None:
                    assert forward.as_tuple() == backward.as_tuple()

    def test_reshard_scenario_from_the_documentation(self) -> None:
        """The 12-element 4-writer / 3-reader example resolves as documented."""
        saved = even_shard_ranges(12, 4)  # [0,3) [3,6) [6,9) [9,12)
        wanted = even_shard_ranges(12, 3)  # [0,4) [4,8) [8,12)
        overlaps = [
            [intersect_ranges(s, w) for s in saved if intersect_ranges(s, w)] for w in wanted
        ]
        assert [[o.as_tuple() for o in row if o] for row in overlaps] == [
            [(0, 3), (3, 1)],
            [(4, 2), (6, 2)],
            [(8, 1), (9, 3)],
        ]


class TestSplitAlongDim:
    """Splitting tensors for tensor and sequence parallelism."""

    def test_split_values(self) -> None:
        """Splitting reproduces the documented example."""
        parts = split_tensor_along_dim(torch.arange(8).view(2, 4), dim=1, num_parts=2)
        assert parts[0].tolist() == [[0, 1], [4, 5]]
        assert parts[1].tolist() == [[2, 3], [6, 7]]

    def test_concatenation_recovers_the_original(self) -> None:
        """The split is a partition, so concatenation is the inverse."""
        original = torch.randn(4, 6, 8)
        for dim in (0, 1, 2, -1):
            parts = split_tensor_along_dim(original, dim=dim, num_parts=2)
            assert torch.equal(torch.cat(parts, dim=dim), original)

    def test_indivisible_dimension_is_rejected(self) -> None:
        """An uneven split raises with an actionable message."""
        with pytest.raises(ShardingError, match="not divisible by 3"):
            split_tensor_along_dim(torch.zeros(4, 5), dim=1, num_parts=3)

    def test_out_of_range_dimension_is_rejected(self) -> None:
        """A dimension index outside the tensor's rank is an error."""
        with pytest.raises(ShardingError, match="dimension index out of range"):
            split_tensor_along_dim(torch.zeros(4, 4), dim=5, num_parts=2)

    def test_parts_are_contiguous_by_default(self) -> None:
        """Slices are made contiguous so they can be passed to a collective."""
        parts = split_tensor_along_dim(torch.randn(4, 6), dim=1, num_parts=2)
        assert all(part.is_contiguous() for part in parts)

    def test_non_positive_parts_rejected(self) -> None:
        """Splitting into zero parts is rejected."""
        with pytest.raises(ShardingError, match="num_parts must be positive"):
            split_tensor_along_dim(torch.zeros(4), dim=0, num_parts=0)


def test_measurement_helpers() -> None:
    """Element and byte counting helpers agree with PyTorch."""
    named = [("a", torch.zeros(2, 3)), ("b", torch.zeros(4, dtype=torch.float64))]
    assert numel_of(named) == 10
    assert bytes_of([t for _, t in named]) == 6 * 4 + 4 * 8
