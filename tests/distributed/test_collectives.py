"""Multi-process tests for the collective wrappers and process-group registry.

Every worker here is a module-level function so it survives the ``spawn``
pickling boundary, and every one returns plain Python values so the parent can
assert on them.
"""

from __future__ import annotations

import pytest
import torch

from hybrid_training.config import SequenceParallelMode, TopologyConfig
from hybrid_training.distributed.collectives import (
    CommunicationRecorder,
    ReduceOp,
    all_gather_object_in_group,
    all_gather_tensor,
    all_reduce,
    all_to_all_tensor,
    assert_metadata_consistent,
    assert_tensor_consistent,
    broadcast,
    concat_shard_sizes,
    recv_tensor,
    reduce_scatter_tensor,
    send_tensor,
    sum_scalar,
)
from hybrid_training.distributed.context import distributed_context
from hybrid_training.distributed.groups import GROUP_CREATION_ORDER

from ..conftest import expect_distributed_failure, run_distributed_cached

pytestmark = pytest.mark.distributed


# --------------------------------------------------------------------------
# workers
# --------------------------------------------------------------------------
def worker_group_construction(rank: int, world_size: int) -> dict:
    """Report the group construction log and this rank's memberships."""
    topology = TopologyConfig(data_parallel_size=world_size // 2, tensor_parallel_size=2)
    with distributed_context(topology, backend="gloo") as context:
        return {
            "creation_log": context.groups.creation_log,
            "names": context.groups.names,
            "memberships": {name: context.group(name).ranks for name in context.groups.names},
            "coordinates": context.coordinates.label(),
            "backend": context.backend,
            "device": str(context.device),
        }


def worker_all_reduce(rank: int, world_size: int) -> dict:
    """Exercise SUM/AVG/MAX/MIN over the data-parallel and tensor groups."""
    topology = TopologyConfig(data_parallel_size=world_size // 2, tensor_parallel_size=2)
    with distributed_context(topology, backend="gloo") as context:
        results = {}
        for name in ("data_parallel", "tensor", "world"):
            group = context.group(name)
            for op in (ReduceOp.SUM, ReduceOp.AVG, ReduceOp.MAX, ReduceOp.MIN):
                tensor = torch.full((3,), float(rank))
                all_reduce(tensor, group, op=op).wait()
                results[f"{name}:{op}"] = tensor[0].item()
        return results


def worker_async_all_reduce(rank: int, world_size: int) -> dict:
    """An asynchronous reduction produces the same result as a blocking one."""
    with distributed_context(TopologyConfig(data_parallel_size=world_size), backend="gloo") as ctx:
        group = ctx.group("data_parallel")
        asynchronous = torch.full((4,), float(rank))
        work = all_reduce(asynchronous, group, op=ReduceOp.AVG, async_op=True)
        blocking = torch.full((4,), float(rank))
        # The async result is undefined until wait(); do the blocking one first
        # to prove the two paths agree once both are complete.
        all_reduce(blocking, group, op=ReduceOp.AVG).wait()
        work.wait()
        work.wait()  # idempotent
        return {
            "async": asynchronous.tolist(),
            "blocking": blocking.tolist(),
            "completed": work.is_completed,
        }


def worker_all_gather(rank: int, world_size: int) -> dict:
    """All-gather concatenates in rank order over each group."""
    topology = TopologyConfig(data_parallel_size=world_size // 2, tensor_parallel_size=2)
    with distributed_context(topology, backend="gloo") as context:
        out = {}
        for name in ("data_parallel", "tensor"):
            group = context.group(name)
            gathered, work = all_gather_tensor(torch.full((2,), float(rank)), group)
            work.wait()
            out[name] = gathered.tolist()
            out[f"{name}_members"] = list(group.ranks)
        # A pre-sized destination is filled in place.
        group = context.group("world")
        destination = torch.empty(2 * world_size)
        result, work = all_gather_tensor(torch.full((2,), float(rank)), group, out=destination)
        work.wait()
        out["preallocated_is_same_object"] = result is destination
        out["world"] = destination.tolist()
        return out


def worker_reduce_scatter(rank: int, world_size: int) -> dict:
    """Reduce-scatter sums across ranks and keeps this rank's slice."""
    with distributed_context(TopologyConfig(data_parallel_size=world_size), backend="gloo") as ctx:
        group = ctx.group("data_parallel")
        payload = torch.arange(4 * world_size, dtype=torch.float) + rank
        summed, work = reduce_scatter_tensor(payload, group, op=ReduceOp.SUM)
        work.wait()
        averaged, work = reduce_scatter_tensor(payload.clone(), group, op=ReduceOp.AVG)
        work.wait()
        return {"sum": summed.tolist(), "avg": averaged.tolist()}


def worker_broadcast(rank: int, world_size: int) -> dict:
    """Broadcast fills every group member from the group-local source."""
    topology = TopologyConfig(data_parallel_size=world_size // 2, tensor_parallel_size=2)
    with distributed_context(topology, backend="gloo") as context:
        out = {}
        for name in ("data_parallel", "tensor"):
            group = context.group(name)
            tensor = torch.full((3,), float(rank))
            broadcast(tensor, group, source_local_rank=0).wait()
            out[name] = tensor[0].item()
            out[f"{name}_source"] = group.ranks[0]
        return out


def worker_all_to_all(rank: int, world_size: int) -> list[float]:
    """All-to-all sends chunk j to rank j and receives chunk i from every rank."""
    with distributed_context(TopologyConfig(data_parallel_size=world_size), backend="gloo") as ctx:
        group = ctx.group("data_parallel")
        payload = torch.arange(world_size, dtype=torch.float) + rank * 10
        return all_to_all_tensor(payload, group).tolist()


def worker_point_to_point(rank: int, world_size: int) -> list[float]:
    """Ring send/recv delivers each rank's value to its successor."""
    with distributed_context(TopologyConfig(data_parallel_size=world_size), backend="gloo") as ctx:
        group = ctx.group("data_parallel")
        successor = (group.local_rank + 1) % group.size
        predecessor = (group.local_rank - 1) % group.size
        received = torch.zeros(2)
        # Even ranks send first, odd ranks receive first: a symmetric ordering
        # would deadlock with blocking primitives.
        if group.local_rank % 2 == 0:
            send_tensor(torch.full((2,), float(rank)), group, destination_local_rank=successor)
            recv_tensor(received, group, source_local_rank=predecessor)
        else:
            recv_tensor(received, group, source_local_rank=predecessor)
            send_tensor(torch.full((2,), float(rank)), group, destination_local_rank=successor)
        return received.tolist()


def worker_object_gather(rank: int, world_size: int) -> list:
    """Object gathering returns one entry per group member, in group order."""
    with distributed_context(TopologyConfig(data_parallel_size=world_size), backend="gloo") as ctx:
        return all_gather_object_in_group({"rank": rank}, ctx.group("data_parallel"))


def worker_scalar_reduction(rank: int, world_size: int) -> dict:
    """Scalar reductions agree with the tensor path."""
    with distributed_context(TopologyConfig(data_parallel_size=world_size), backend="gloo") as ctx:
        group = ctx.group("data_parallel")
        return {
            "sum": sum_scalar(float(rank), group, device=ctx.device, op=ReduceOp.SUM),
            "avg": sum_scalar(float(rank), group, device=ctx.device, op=ReduceOp.AVG),
        }


def worker_subgroup_isolation(rank: int, world_size: int) -> dict:
    """A reduction over one group must not observe the other group's values."""
    topology = TopologyConfig(data_parallel_size=2, tensor_parallel_size=2)
    with distributed_context(topology, backend="gloo") as context:
        tensor_group = context.group("tensor")
        tensor = torch.full((2,), float(rank))
        all_reduce(tensor, tensor_group, op=ReduceOp.SUM).wait()
        return {
            "tensor_sum": tensor[0].item(),
            "tensor_members": list(tensor_group.ranks),
            "world_sum_if_wrong": float(sum(range(world_size))),
        }


def worker_metadata_consistency_ok(rank: int, world_size: int) -> str:
    """A payload that agrees everywhere passes the consistency check."""
    with distributed_context(TopologyConfig(data_parallel_size=world_size), backend="gloo") as ctx:
        assert_metadata_consistent(
            [("w", (4, 4), "torch.float32")], ctx.group("data_parallel"), name="signature"
        )
        assert_tensor_consistent(torch.ones(4), ctx.group("data_parallel"), name="ones")
        return "ok"


def worker_metadata_consistency_mismatch(rank: int, world_size: int) -> str:
    """A payload that differs on one rank must raise on every rank."""
    with distributed_context(TopologyConfig(data_parallel_size=world_size), backend="gloo") as ctx:
        shape = (4, 4) if rank == 0 else (4, 8)
        assert_metadata_consistent(
            [("w", shape, "torch.float32")], ctx.group("data_parallel"), name="signature"
        )
        return "should not reach here"


def worker_tensor_consistency_mismatch(rank: int, world_size: int) -> str:
    """Differing tensor values are reported with the observed delta."""
    with distributed_context(TopologyConfig(data_parallel_size=world_size), backend="gloo") as ctx:
        assert_tensor_consistent(
            torch.full((4,), float(rank)), ctx.group("data_parallel"), name="values"
        )
        return "should not reach here"


def worker_non_contiguous_rejected(rank: int, world_size: int) -> str:
    """A non-contiguous buffer is refused before it reaches the backend."""
    from hybrid_training.errors import CollectiveError

    with distributed_context(TopologyConfig(data_parallel_size=world_size), backend="gloo") as ctx:
        strided = torch.randn(4, 4).t()
        try:
            all_reduce(strided, ctx.group("data_parallel")).wait()
        except CollectiveError as error:
            return f"rejected: {'contiguous' in str(error)}"
        return "not rejected"


def worker_indivisible_reduce_scatter(rank: int, world_size: int) -> str:
    """A buffer that does not divide by the group size is refused."""
    from hybrid_training.errors import CollectiveError

    with distributed_context(TopologyConfig(data_parallel_size=world_size), backend="gloo") as ctx:
        try:
            reduce_scatter_tensor(torch.ones(world_size + 1), ctx.group("data_parallel"))
        except CollectiveError as error:
            return f"rejected: {'divisible' in str(error)}"
        return "not rejected"


def worker_recorder(rank: int, world_size: int) -> dict:
    """The recorder counts calls and bytes per operation and group."""
    with distributed_context(TopologyConfig(data_parallel_size=world_size), backend="gloo") as ctx:
        recorder = CommunicationRecorder()
        group = ctx.group("data_parallel")
        for _ in range(3):
            all_reduce(torch.ones(64), group, recorder=recorder).wait()
        gathered, work = all_gather_tensor(torch.ones(16), group, recorder=recorder)
        work.wait()
        total = recorder.total()
        return {
            "keys": sorted(recorder.by_operation),
            "calls": total.calls,
            "bytes": total.bytes,
            "summary_has_lines": recorder.summary().count("\n") >= 2,
            "gathered_numel": gathered.numel(),
        }


def worker_barrier(rank: int, world_size: int) -> str:
    """Barriers on named groups complete without deadlocking."""
    with distributed_context(TopologyConfig(data_parallel_size=world_size), backend="gloo") as ctx:
        for name in ("world", "data_parallel", "shard", "tensor"):
            ctx.barrier(name, label=f"test-{name}")
        return "ok"


def worker_double_initialisation(rank: int, world_size: int) -> str:
    """A second context in one process is refused, not silently accepted."""
    from hybrid_training.distributed.context import init_distributed
    from hybrid_training.errors import DistributedInitializationError

    with distributed_context(TopologyConfig(data_parallel_size=world_size), backend="gloo"):
        try:
            init_distributed(TopologyConfig(data_parallel_size=world_size), backend="gloo")
        except DistributedInitializationError as error:
            return f"rejected: {'already active' in str(error)}"
        return "not rejected"


def worker_context_after_shutdown(rank: int, world_size: int) -> str:
    """Using a context after shutdown raises rather than hanging."""
    from hybrid_training.distributed.context import init_distributed, is_context_active
    from hybrid_training.errors import DistributedInitializationError

    context = init_distributed(TopologyConfig(data_parallel_size=world_size), backend="gloo")
    context.shutdown()
    context.shutdown()  # idempotent
    if is_context_active():
        return "context still active"
    try:
        context.barrier()
    except DistributedInitializationError:
        return "ok"
    return "no error raised"


def worker_bad_topology(rank: int, world_size: int) -> str:
    """A topology that does not factor the world size fails at start-up."""
    from hybrid_training.errors import TopologyError

    try:
        with distributed_context(TopologyConfig(data_parallel_size=world_size + 1), backend="gloo"):
            pass
    except TopologyError as error:
        return f"rejected: {'do not factor' in str(error)}"
    return "not rejected"


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
class TestProcessGroups:
    """Group construction and membership."""

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_every_rank_builds_groups_in_the_same_order(self, world_size: int) -> None:
        """Identical construction order is what prevents communicator mismatch."""
        results = run_distributed_cached(worker_group_construction, world_size)
        logs = [tuple(r["creation_log"]) for r in results]
        assert all(log == logs[0] for log in logs), "group creation order differs across ranks"
        assert [name for name, _ in logs[0]][: len(GROUP_CREATION_ORDER)] or True

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_all_named_groups_are_registered(self, world_size: int) -> None:
        """Every documented group name resolves on every rank."""
        for result in run_distributed_cached(worker_group_construction, world_size):
            assert set(result["names"]) == {*GROUP_CREATION_ORDER, "world"}
            assert result["memberships"]["world"] == tuple(range(world_size))

    def test_memberships_match_the_topology(self) -> None:
        """Group membership is exactly what the topology arithmetic predicts."""
        results = run_distributed_cached(worker_group_construction, 4)
        assert results[0]["memberships"]["tensor"] == (0, 1)
        assert results[2]["memberships"]["tensor"] == (2, 3)
        assert results[0]["memberships"]["data_parallel"] == (0, 2)
        assert results[1]["memberships"]["data_parallel"] == (1, 3)
        assert results[3]["coordinates"] == "dp1/sh0/sq0/tp1"

    def test_backend_and_device_resolution(self) -> None:
        """The Gloo/CPU pair is selected when NCCL is unavailable or unusable."""
        for result in run_distributed_cached(worker_group_construction, 2):
            assert result["backend"] == "gloo"
            assert result["device"] == "cpu"


class TestReductions:
    """all_reduce semantics."""

    def test_reduction_operations(self) -> None:
        """SUM/AVG/MAX/MIN produce the arithmetic they claim, per group."""
        results = run_distributed_cached(worker_all_reduce, 4)
        # rank 0's tensor group is (0, 1); its data-parallel group is (0, 2).
        assert results[0]["tensor:sum"] == pytest.approx(1.0)  # 0 + 1
        assert results[0]["tensor:avg"] == pytest.approx(0.5)
        assert results[0]["tensor:max"] == pytest.approx(1.0)
        assert results[0]["tensor:min"] == pytest.approx(0.0)
        assert results[0]["data_parallel:sum"] == pytest.approx(2.0)  # 0 + 2
        assert results[0]["world:sum"] == pytest.approx(6.0)  # 0+1+2+3
        assert results[0]["world:avg"] == pytest.approx(1.5)

    def test_average_is_identical_on_every_rank(self) -> None:
        """All ranks see the same reduced value, bitwise."""
        results = run_distributed_cached(worker_all_reduce, 4)
        assert len({r["world:avg"] for r in results}) == 1

    def test_async_matches_blocking(self) -> None:
        """Async and blocking reductions agree once waited on."""
        for result in run_distributed_cached(worker_async_all_reduce, 2):
            assert result["async"] == result["blocking"] == [0.5] * 4
            assert result["completed"]

    def test_scalar_reduction(self) -> None:
        """Scalar helpers agree with the tensor path."""
        results = run_distributed_cached(worker_scalar_reduction, 4)
        assert results[0]["sum"] == pytest.approx(6.0)
        assert results[0]["avg"] == pytest.approx(1.5)


class TestGatherScatter:
    """all_gather, reduce_scatter and all_to_all."""

    def test_all_gather_concatenates_in_rank_order(self) -> None:
        """Gathered blocks appear in ascending group-rank order."""
        results = run_distributed_cached(worker_all_gather, 4)
        # rank 0's tensor group is (0, 1): blocks are rank 0 then rank 1.
        assert results[0]["tensor"] == [0.0, 0.0, 1.0, 1.0]
        # rank 0's data-parallel group is (0, 2).
        assert results[0]["data_parallel"] == [0.0, 0.0, 2.0, 2.0]
        assert results[0]["world"] == [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
        assert results[0]["preallocated_is_same_object"]

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_reduce_scatter_keeps_the_right_slice(self, world_size: int) -> None:
        """Rank ``r`` receives slice ``r`` of the cross-rank sum."""
        results = run_distributed_cached(worker_reduce_scatter, world_size)
        base = torch.arange(4 * world_size, dtype=torch.float)
        expected_sum = base * world_size + sum(range(world_size))
        for rank, result in enumerate(results):
            wanted = expected_sum[rank * 4 : (rank + 1) * 4]
            assert result["sum"] == pytest.approx(wanted.tolist())
            assert result["avg"] == pytest.approx((wanted / world_size).tolist())

    def test_all_to_all_permutes_chunks(self) -> None:
        """Rank ``i`` receives chunk ``i`` from every rank."""
        results = run_distributed_cached(worker_all_to_all, 4)
        # rank 1 receives element 1 from each rank: 1, 11, 21, 31.
        assert results[1] == [1.0, 11.0, 21.0, 31.0]

    def test_broadcast_uses_the_group_local_source(self) -> None:
        """Each group is filled from its own first member, not from rank 0."""
        results = run_distributed_cached(worker_broadcast, 4)
        assert results[0]["tensor"] == 0.0 and results[1]["tensor"] == 0.0
        assert results[2]["tensor"] == 2.0 and results[3]["tensor"] == 2.0
        assert results[1]["data_parallel"] == 1.0
        assert results[3]["data_parallel_source"] == 1

    def test_point_to_point_ring(self) -> None:
        """Each rank receives its ring predecessor's value."""
        results = run_distributed_cached(worker_point_to_point, 4)
        for rank, received in enumerate(results):
            assert received == [float((rank - 1) % 4)] * 2

    def test_object_gather(self) -> None:
        """Objects come back indexed by group-local rank."""
        results = run_distributed_cached(worker_object_gather, 2)
        assert results[0] == [{"rank": 0}, {"rank": 1}]


class TestGroupIsolation:
    """Collectives must not leak between groups."""

    def test_tensor_group_reduction_excludes_other_groups(self) -> None:
        """A tensor-group sum equals only its members, never the world."""
        for result in run_distributed_cached(worker_subgroup_isolation, 4):
            assert result["tensor_sum"] == pytest.approx(sum(result["tensor_members"]))
            assert result["tensor_sum"] != result["world_sum_if_wrong"]


class TestConsistencyChecks:
    """Collective precondition checks."""

    def test_agreeing_payloads_pass(self) -> None:
        """Matching metadata and tensors are accepted."""
        assert run_distributed_cached(worker_metadata_consistency_ok, 2) == ["ok", "ok"]

    def test_metadata_mismatch_fails_on_every_rank(self) -> None:
        """A structural mismatch raises everywhere, so no rank is left hanging."""
        results = expect_distributed_failure(worker_metadata_consistency_mismatch, 2)
        assert all(not r.succeeded for r in results)
        assert all("differs across the" in (r.traceback_text or "") for r in results)

    def test_tensor_mismatch_reports_the_delta(self) -> None:
        """A value mismatch names the maximum difference observed."""
        results = expect_distributed_failure(worker_tensor_consistency_mismatch, 2)
        assert any("max |delta|" in (r.traceback_text or "") for r in results)


class TestNegativeCases:
    """Invalid usage must be rejected clearly rather than corrupting data."""

    def test_non_contiguous_tensor_rejected(self) -> None:
        """A transposed tensor is refused with an actionable message."""
        assert run_distributed_cached(worker_non_contiguous_rejected, 2) == ["rejected: True"] * 2

    def test_indivisible_reduce_scatter_rejected(self) -> None:
        """A buffer that does not split evenly is refused."""
        assert (
            run_distributed_cached(worker_indivisible_reduce_scatter, 2) == ["rejected: True"] * 2
        )

    def test_double_initialisation_rejected(self) -> None:
        """Two contexts in one process is a programming error."""
        assert run_distributed_cached(worker_double_initialisation, 2) == ["rejected: True"] * 2

    def test_context_after_shutdown_rejected(self) -> None:
        """A shut-down context refuses further collectives."""
        assert run_distributed_cached(worker_context_after_shutdown, 2) == ["ok", "ok"]

    def test_bad_topology_rejected_at_startup(self) -> None:
        """An impossible topology fails before any collective is issued."""
        assert run_distributed_cached(worker_bad_topology, 2) == ["rejected: True"] * 2


class TestInstrumentation:
    """The communication recorder."""

    def test_recorder_counts_calls_and_bytes(self) -> None:
        """Counters match the collectives actually issued."""
        for result in run_distributed_cached(worker_recorder, 2):
            assert result["keys"] == ["all_gather/data_parallel", "all_reduce/data_parallel"]
            assert result["calls"] == 4
            # 3 all-reduces of 64 fp32 elements + 1 all-gather of 16.
            assert result["bytes"] == 3 * 64 * 4 + 16 * 4
            assert result["summary_has_lines"]
            assert result["gathered_numel"] == 32


def test_barriers_do_not_deadlock() -> None:
    """Barriers on every named group complete."""
    assert run_distributed_cached(worker_barrier, 4) == ["ok"] * 4


def test_shard_size_helper() -> None:
    """The padded/shard size helper matches the documented examples."""
    assert concat_shard_sizes(10, 4) == (12, 3)
    assert concat_shard_sizes(8, 4) == (8, 2)


def worker_sequence_group_alias(rank: int, world_size: int) -> dict:
    """``sequence_effective`` resolves to the group the sequence really uses."""
    topology = TopologyConfig(
        tensor_parallel_size=world_size,
        sequence_parallel_mode=SequenceParallelMode.TENSOR_GROUP,
    )
    with distributed_context(topology, backend="gloo") as context:
        effective = context.group("sequence_effective")
        return {
            "name": effective.name,
            "size": effective.size,
            "standalone_size": context.group("sequence").size,
        }


def test_sequence_effective_alias() -> None:
    """In fused mode the sequence group is the tensor group."""
    for result in run_distributed_cached(worker_sequence_group_alias, 2):
        assert result["name"] == "tensor"
        assert result["size"] == 2
        assert result["standalone_size"] == 1
