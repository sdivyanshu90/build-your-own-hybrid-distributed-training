"""Unit tests for the rank topology.

These run in a single process with no process group at all: the topology is
pure arithmetic, and an off-by-one in it manifests as a *hang* in a real job,
so it is exactly the kind of logic that deserves cheap, exhaustive tests.
"""

from __future__ import annotations

import itertools

import pytest

from hybrid_training.config import SequenceParallelMode, TopologyConfig
from hybrid_training.distributed.topology import (
    COMPOSITE_GROUPS,
    DIMENSIONS,
    ParallelTopology,
    RankCoordinates,
)
from hybrid_training.errors import TopologyError


def make(
    dp: int = 1, shard: int = 1, sequence: int = 1, tensor: int = 1, mode: str | None = None
) -> ParallelTopology:
    """Build a topology whose world size is the product of the dimensions.

    ``mode`` defaults to ``independent`` whenever ``sequence > 1``, because a
    standalone sequence dimension is only legal in that mode -- the helper
    should not force every caller to repeat the rule.
    """
    if mode is None:
        mode = SequenceParallelMode.INDEPENDENT if sequence > 1 else SequenceParallelMode.DISABLED
    config = TopologyConfig(
        data_parallel_size=dp,
        shard_parallel_size=shard,
        sequence_parallel_size=sequence,
        tensor_parallel_size=tensor,
        sequence_parallel_mode=mode,
    )
    return ParallelTopology(config, dp * shard * sequence * tensor)


class TestCoordinates:
    """Rank <-> coordinate round-tripping."""

    @pytest.mark.parametrize(
        ("dp", "shard", "sequence", "tensor"),
        [(1, 1, 1, 1), (2, 1, 1, 1), (1, 4, 1, 1), (2, 2, 1, 2), (2, 1, 2, 2), (3, 2, 1, 2)],
    )
    def test_round_trip_is_a_bijection(
        self, dp: int, shard: int, sequence: int, tensor: int
    ) -> None:
        """Every rank maps to unique coordinates and back again."""
        topology = make(dp, shard, sequence, tensor)
        seen = set()
        for rank in range(topology.world_size):
            coordinates = topology.coordinates_of(rank)
            assert topology.rank_of(coordinates) == rank
            assert coordinates.as_tuple() not in seen
            seen.add(coordinates.as_tuple())
        assert len(seen) == topology.world_size

    def test_documented_eight_rank_example(self) -> None:
        """The table in the topology module docstring is exactly reproduced.

        Documentation that drifts from the code is worse than no documentation,
        so the worked example is asserted rather than merely written down.
        """
        topology = make(dp=2, shard=2, tensor=2)
        expected = {
            0: ("dp0/sh0/sq0/tp0", (0, 4), (0, 2), (0, 1)),
            1: ("dp0/sh0/sq0/tp1", (1, 5), (1, 3), (0, 1)),
            2: ("dp0/sh1/sq0/tp0", (2, 6), (0, 2), (2, 3)),
            3: ("dp0/sh1/sq0/tp1", (3, 7), (1, 3), (2, 3)),
            4: ("dp1/sh0/sq0/tp0", (0, 4), (4, 6), (4, 5)),
            5: ("dp1/sh0/sq0/tp1", (1, 5), (5, 7), (4, 5)),
            6: ("dp1/sh1/sq0/tp0", (2, 6), (4, 6), (6, 7)),
            7: ("dp1/sh1/sq0/tp1", (3, 7), (5, 7), (6, 7)),
        }
        for rank, (label, dp_group, shard_group, tensor_group) in expected.items():
            assert topology.coordinates_of(rank).label() == label
            assert topology.group_ranks("data_parallel", rank) == dp_group
            assert topology.group_ranks("shard", rank) == shard_group
            assert topology.group_ranks("tensor", rank) == tensor_group

    def test_tensor_dimension_varies_fastest(self) -> None:
        """Adjacent ranks share a tensor group, which is the placement goal."""
        topology = make(dp=2, shard=2, tensor=2)
        assert topology.group_ranks("tensor", 0) == (0, 1)
        assert topology.group_ranks("tensor", 2) == (2, 3)
        # ...while data-parallel peers are as far apart as possible.
        assert topology.group_ranks("data_parallel", 0) == (0, 4)

    def test_coordinate_lookup_by_name(self) -> None:
        """Coordinates can be indexed by dimension name."""
        coordinates = RankCoordinates(data_parallel=1, shard=2, sequence=0, tensor=3)
        assert coordinates["data_parallel"] == 1
        assert coordinates["tensor"] == 3
        with pytest.raises(KeyError, match="unknown dimension"):
            _ = coordinates["pipeline"]

    def test_rank_out_of_range_is_rejected(self) -> None:
        """A rank outside the grid raises rather than wrapping silently."""
        topology = make(dp=2)
        with pytest.raises(TopologyError, match="rank outside the topology"):
            topology.coordinates_of(2)
        with pytest.raises(TopologyError, match="rank outside the topology"):
            topology.coordinates_of(-1)

    def test_coordinate_out_of_range_is_rejected(self) -> None:
        """An impossible coordinate raises rather than aliasing another rank."""
        topology = make(dp=2, tensor=2)
        with pytest.raises(TopologyError, match="out of range"):
            topology.rank_of(RankCoordinates(data_parallel=5, shard=0, sequence=0, tensor=0))
        with pytest.raises(TopologyError, match="wrong number of coordinates"):
            topology.rank_of((0, 0))


class TestGroups:
    """Group membership properties."""

    @pytest.mark.parametrize(
        ("dp", "shard", "sequence", "tensor"),
        [(2, 2, 1, 2), (4, 2, 1, 1), (1, 1, 2, 2), (2, 3, 1, 2)],
    )
    def test_groups_partition_the_world(
        self, dp: int, shard: int, sequence: int, tensor: int
    ) -> None:
        """Every dimension's groups tile the ranks exactly once."""
        topology = make(dp, shard, sequence, tensor)
        for name in (*DIMENSIONS, *COMPOSITE_GROUPS):
            groups = topology.all_group_rank_lists(name)
            flattened = [rank for group in groups for rank in group]
            assert sorted(flattened) == list(range(topology.world_size)), name
            assert len(set(flattened)) == len(flattened), f"{name} groups overlap"

    def test_group_enumeration_is_deterministic(self) -> None:
        """Groups come out in a stable order, sorted by their first member.

        This is the property that makes ``dist.new_group`` safe to call in a
        loop on every rank: all ranks walk the same sequence.
        """
        topology = make(dp=2, shard=2, tensor=2)
        for name in DIMENSIONS:
            groups = topology.all_group_rank_lists(name)
            assert [g[0] for g in groups] == sorted(g[0] for g in groups)
            assert groups == topology.all_group_rank_lists(name)

    def test_group_ranks_are_ascending_and_contain_the_rank(self) -> None:
        """Group rank lists are sorted and always include the querying rank."""
        topology = make(dp=2, shard=2, sequence=1, tensor=2)
        for rank in range(topology.world_size):
            for name in (*DIMENSIONS, *COMPOSITE_GROUPS):
                members = topology.group_ranks(name, rank)
                assert list(members) == sorted(members)
                assert rank in members

    def test_composite_group_sizes(self) -> None:
        """Composite groups have the product of their member dimensions' sizes."""
        topology = make(dp=2, shard=2, tensor=2)
        assert topology.size("dp_shard") == 4
        assert topology.size("tensor_sequence") == 2
        assert topology.size("world") == 8
        assert topology.group_ranks("dp_shard", 0) == (0, 2, 4, 6)

    def test_local_rank_and_source(self) -> None:
        """Local rank is the index in the group; the source is its first member."""
        topology = make(dp=2, shard=2, tensor=2)
        assert topology.local_rank_in_group("data_parallel", 4) == 1
        assert topology.group_source_rank("data_parallel", 4) == 0
        assert topology.local_rank_in_group("tensor", 3) == 1
        assert topology.group_source_rank("tensor", 3) == 2

    def test_neighbour_ranks_form_a_ring(self) -> None:
        """Ring neighbours wrap around and degenerate correctly at size 1."""
        topology = make(dp=1, shard=1, tensor=4)
        assert topology.neighbour_ranks("tensor", 0) == (3, 1)
        assert topology.neighbour_ranks("tensor", 3) == (2, 0)
        trivial = make(dp=2)
        assert trivial.neighbour_ranks("tensor", 1) == (1, 1)

    def test_unknown_dimension_is_rejected(self) -> None:
        """An unknown group name raises instead of returning an empty group."""
        topology = make(dp=2)
        with pytest.raises(TopologyError, match="unknown dimension or group name"):
            topology.group_ranks("pipeline", 0)


class TestSequenceParallelModes:
    """How the sequence dimension resolves under each mode."""

    def test_tensor_group_mode_reuses_the_tensor_group(self) -> None:
        """In fused mode the sequence group *is* the tensor group."""
        topology = make(tensor=4, mode=SequenceParallelMode.TENSOR_GROUP)
        assert topology.sequence_group_name == "tensor"
        assert topology.sequence_parallel_size == 4
        assert topology.sequence_parallel_enabled
        assert topology.world_size == 4  # the sequence consumes no extra ranks

    def test_independent_mode_uses_its_own_dimension(self) -> None:
        """In independent mode the sequence has its own group and ranks."""
        topology = make(sequence=2, tensor=2, mode=SequenceParallelMode.INDEPENDENT)
        assert topology.sequence_group_name == "sequence"
        assert topology.sequence_parallel_size == 2
        assert topology.world_size == 4

    def test_disabled_mode_is_a_trivial_group(self) -> None:
        """With sequence parallelism off the group has one member."""
        topology = make(tensor=2)
        assert topology.sequence_parallel_size == 1
        assert not topology.sequence_parallel_enabled


class TestConfigValidation:
    """Configuration-level topology validation."""

    def test_product_must_equal_world_size(self) -> None:
        """A topology that does not factor the world size is rejected."""
        config = TopologyConfig(data_parallel_size=2, tensor_parallel_size=2)
        config.validate_against_world_size(4)
        with pytest.raises(TopologyError, match="do not factor the world size"):
            config.validate_against_world_size(6)

    def test_for_world_size_infers_the_missing_dimension(self) -> None:
        """One omitted dimension is inferred from the world size."""
        config = TopologyConfig.for_world_size(8, tensor_parallel_size=2)
        assert config.data_parallel_size == 8 // 2
        assert config.world_size == 8

    def test_for_world_size_defaults_everything_to_data_parallel(self) -> None:
        """With no hints the whole world becomes data parallel."""
        assert TopologyConfig.for_world_size(4).data_parallel_size == 4

    def test_for_world_size_rejects_indivisible_dimensions(self) -> None:
        """Dimensions that do not divide the world size are rejected."""
        with pytest.raises(TopologyError, match="do not divide the world size"):
            TopologyConfig.for_world_size(6, tensor_parallel_size=4)

    def test_negative_and_zero_sizes_rejected(self) -> None:
        """Dimension sizes must be positive integers."""
        with pytest.raises(TopologyError, match="must be a positive integer"):
            TopologyConfig(data_parallel_size=0)
        with pytest.raises(TopologyError, match="must be a positive integer"):
            TopologyConfig(tensor_parallel_size=-1)

    def test_booleans_are_not_accepted_as_sizes(self) -> None:
        """``True`` is an ``int`` in Python; it is still not a valid size."""
        with pytest.raises(TopologyError, match="must be a positive integer"):
            TopologyConfig(data_parallel_size=True)  # type: ignore[arg-type]

    def test_sequence_size_requires_independent_mode(self) -> None:
        """``sequence_parallel_size > 1`` is only legal in independent mode."""
        with pytest.raises(TopologyError, match="requires mode 'independent'"):
            TopologyConfig(sequence_parallel_size=2)

    def test_tensor_group_mode_requires_a_real_tensor_group(self) -> None:
        """Fused mode with a one-rank tensor group is a configuration error."""
        with pytest.raises(TopologyError, match="needs a tensor-parallel group"):
            TopologyConfig(
                tensor_parallel_size=1,
                sequence_parallel_mode=SequenceParallelMode.TENSOR_GROUP,
            )

    def test_unknown_mode_rejected(self) -> None:
        """An unknown sequence-parallel mode is rejected."""
        with pytest.raises(TopologyError, match="unknown sequence_parallel_mode"):
            TopologyConfig(sequence_parallel_mode="ring")


def test_describe_lists_every_group() -> None:
    """The human description covers every named group, for debuggability."""
    topology = make(dp=2, shard=2, tensor=2)
    text = topology.describe(3)
    for name in (*DIMENSIONS, *COMPOSITE_GROUPS):
        assert name in text
    assert "dp0/sh1/sq0/tp1" in text


def test_summary_is_json_serialisable() -> None:
    """The checkpoint metadata form contains the sizes and dimension order."""
    import json

    topology = make(dp=2, shard=2, tensor=2)
    payload = json.loads(json.dumps(topology.summary()))
    assert payload["world_size"] == 8
    assert payload["sizes"]["tensor"] == 2
    assert payload["dimension_order"] == list(DIMENSIONS)


def test_all_dimension_combinations_up_to_eight_ranks() -> None:
    """Exhaustive sweep: every legal factorisation of 1..8 ranks behaves."""
    for world_size in range(1, 9):
        for dp, shard, tensor in itertools.product(range(1, 9), repeat=3):
            if dp * shard * tensor != world_size:
                continue
            topology = make(dp, shard, 1, tensor)
            assert topology.world_size == world_size
            for rank in range(world_size):
                assert topology.rank_of(topology.coordinates_of(rank)) == rank
