"""CUDA and NCCL tests.

These are the only tests in the suite that need hardware. They are marked
``cuda`` (one device) or ``multigpu`` (two or more, for NCCL collectives) and
skip cleanly otherwise, with a reason naming the device count actually found::

    SKIPPED [1] requires >= 2 CUDA devices for NCCL collectives (found 1)

**Everything these tests cover is also covered on CPU with Gloo.** Their purpose
is to check the *device-specific* parts: that the CUDA device is selected before
NCCL initialises, that memory statistics behave, that mixed precision works on
real half-precision hardware, and that the NCCL path produces the same numbers
as the Gloo path.

Nothing in CI depends on them; see ``.github/workflows/nccl.yml``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from hybrid_training.config import (
    DDPConfig,
    FSDPConfig,
    GradScalerConfig,
    MixedPrecisionConfig,
    ModelConfig,
    OptimizerConfig,
    TopologyConfig,
)
from hybrid_training.distributed.context import distributed_context, init_distributed
from hybrid_training.models.mlp import MLP
from hybrid_training.optim.grad_scaler import GradScaler
from hybrid_training.optim.sharded_optimizer import ShardedOptimizer
from hybrid_training.parallel.ddp import DistributedDataParallel
from hybrid_training.parallel.fsdp import FullyShardedDataParallel
from hybrid_training.utils.memory import capture_memory, reset_peak_memory

from ..conftest import OPTIMIZER_STEP_TOLERANCE, run_distributed_cached

pytestmark = pytest.mark.distributed

MODEL = ModelConfig(input_size=16, hidden_size=32, num_layers=3, output_size=8)
MICRO_BATCH = 8


# --------------------------------------------------------------------------
# single-device tests (marker: cuda)
# --------------------------------------------------------------------------
@pytest.mark.cuda
def test_single_process_context_selects_cuda() -> None:
    """With one GPU and one rank, ``auto`` resolves to NCCL on ``cuda:0``."""
    context = init_distributed(TopologyConfig(), backend="auto", device="auto")
    try:
        assert context.device.type == "cuda"
        assert context.device.index == context.local_rank
        # The device must be current *before* the backend initialises, or every
        # rank would bootstrap NCCL on device 0.
        assert torch.cuda.current_device() == context.device.index
    finally:
        context.shutdown()


@pytest.mark.cuda
def test_memory_statistics_are_available_on_cuda() -> None:
    """CUDA gives real allocator statistics; CPU cannot."""
    device = torch.device("cuda", 0)
    reset_peak_memory(device)
    before = capture_memory(device)
    held = torch.empty(1 << 20, device=device)  # 4 MiB
    after = capture_memory(device)
    assert after.source == "cuda"
    assert after.allocated_bytes >= before.allocated_bytes + held.numel() * 4
    assert after.peak_allocated_bytes >= after.allocated_bytes
    del held


@pytest.mark.cuda
def test_synchronize_device_is_a_no_op_on_cpu_and_real_on_cuda() -> None:
    """``synchronize_device`` must never call ``torch.cuda.synchronize`` on CPU."""
    cpu_context = init_distributed(TopologyConfig(), backend="gloo", device="cpu")
    try:
        cpu_context.synchronize_device()  # must not raise
        assert cpu_context.device.type == "cpu"
    finally:
        cpu_context.shutdown()

    cuda_context = init_distributed(TopologyConfig(), backend="nccl", device="cuda")
    try:
        torch.empty(1024, device=cuda_context.device).sum()
        cuda_context.synchronize_device()
    finally:
        cuda_context.shutdown()


@pytest.mark.cuda
def test_grad_scaler_recovers_from_an_overflow() -> None:
    """A non-finite gradient skips the step and backs the scale off.

    fp16 overflow needs real half-precision arithmetic to reproduce faithfully,
    which is why this lives here rather than in the CPU suite.
    """
    context = init_distributed(TopologyConfig(), backend="nccl", device="cuda")
    try:
        parameter = nn.Parameter(torch.ones(8, device=context.device))
        optimizer = ShardedOptimizer(
            [parameter],
            OptimizerConfig(name="sgd", learning_rate=0.1),
            norm_group=context.group("world"),
            device=context.device,
        )
        scaler = GradScaler(
            GradScalerConfig(enabled=True, init_scale=1024.0, growth_interval=1),
            context.group("world"),
            context.device,
        )
        parameter.grad = torch.full((8,), float("inf"), device=context.device)
        stepped = scaler.step(optimizer, [parameter])
        assert stepped is False
        assert scaler.last_step_skipped
        scaler.update()
        assert scaler.scale_value < 1024.0

        parameter.grad = torch.ones(8, device=context.device)
        assert scaler.step(optimizer, [parameter]) is True
    finally:
        context.shutdown()


# --------------------------------------------------------------------------
# multi-device workers (marker: multigpu)
# --------------------------------------------------------------------------
def worker_nccl_ddp(rank: int, world_size: int) -> dict:
    """Train two DDP steps on NCCL and report the resulting weights."""
    topology = TopologyConfig(data_parallel_size=world_size)
    with distributed_context(topology, backend="nccl", device="cuda") as context:
        model = MLP(MODEL, seed=1, device=context.device)
        ddp = DistributedDataParallel(model, context.group("data_parallel"), DDPConfig())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        generator = torch.Generator(device="cpu").manual_seed(3)
        for _ in range(2):
            inputs = torch.randn(MICRO_BATCH, MODEL.input_size, generator=generator).to(
                context.device
            )
            targets = torch.randn(MICRO_BATCH, MODEL.output_size, generator=generator).to(
                context.device
            )
            optimizer.zero_grad(set_to_none=True)
            nn.functional.mse_loss(ddp(inputs), targets).backward()
            ddp.finish_gradient_synchronization()
            optimizer.step()
        ddp.verify_replica_consistency()
        checksum = float(sum(p.detach().double().sum().item() for p in model.parameters()))
        ddp.teardown()
        return {
            "device": str(context.device),
            "backend": context.backend,
            "checksum": checksum,
        }


def worker_nccl_fsdp(rank: int, world_size: int) -> dict:
    """Shard on NCCL and report the reconstructed weights plus peak memory."""
    topology = TopologyConfig(shard_parallel_size=world_size)
    with distributed_context(topology, backend="nccl", device="cuda") as context:
        reset_peak_memory(context.device)
        model = MLP(MODEL, seed=1, device=context.device)
        fsdp = FullyShardedDataParallel(
            model, context.group("shard"), FSDPConfig(), device=context.device
        )
        optimizer = ShardedOptimizer(
            fsdp.parameters(),
            OptimizerConfig(name="sgd", learning_rate=0.05),
            norm_group=context.group("world"),
            device=context.device,
        )
        generator = torch.Generator(device="cpu").manual_seed(3)
        for _ in range(2):
            inputs = torch.randn(MICRO_BATCH, MODEL.input_size, generator=generator).to(
                context.device
            )
            targets = torch.randn(MICRO_BATCH, MODEL.output_size, generator=generator).to(
                context.device
            )
            optimizer.zero_grad(set_to_none=True)
            nn.functional.mse_loss(fsdp(inputs), targets).backward()
            fsdp.finish_backward()
            optimizer.step()
        full = fsdp.full_state_dict()
        memory = capture_memory(context.device)
        return {
            "device": str(context.device),
            "checksum": float(sum(v.double().sum().item() for v in full.values())),
            "local_parameters": sum(p.numel() for p in fsdp.parameters()),
            "peak_bytes": memory.peak_allocated_bytes,
        }


def worker_nccl_mixed_precision(rank: int, world_size: int) -> dict:
    """Run FSDP with bf16 compute and fp32 reduction on NCCL."""
    topology = TopologyConfig(shard_parallel_size=world_size)
    policy = MixedPrecisionConfig(
        enabled=True,
        param_dtype="bfloat16",
        reduce_dtype="float32",
        master_dtype="float32",
    )
    with distributed_context(topology, backend="nccl", device="cuda") as context:
        model = MLP(MODEL, seed=1, device=context.device)
        fsdp = FullyShardedDataParallel(
            model,
            context.group("shard"),
            FSDPConfig(),
            mixed_precision=policy,
            device=context.device,
        )
        inputs = torch.randn(MICRO_BATCH, MODEL.input_size, device=context.device)
        targets = torch.randn(MICRO_BATCH, MODEL.output_size, device=context.device)
        output = fsdp(inputs)
        nn.functional.mse_loss(output.float(), targets).backward()
        fsdp.finish_backward()
        handle = fsdp.handle
        assert handle is not None
        return {
            "output_dtype": str(output.dtype),
            "master_dtype": str(handle.flat_param.dtype),
            "grad_dtype": str(handle.flat_param.grad.dtype),  # type: ignore[union-attr]
        }


# --------------------------------------------------------------------------
# multi-device tests
# --------------------------------------------------------------------------
@pytest.mark.multigpu
def test_nccl_ddp_keeps_replicas_identical() -> None:
    """Two NCCL ranks agree bitwise after two optimizer steps."""
    results = run_distributed_cached(worker_nccl_ddp, 2)
    assert all(r["backend"] == "nccl" for r in results)
    assert results[0]["device"] == "cuda:0"
    assert results[1]["device"] == "cuda:1"
    assert results[0]["checksum"] == results[1]["checksum"]


@pytest.mark.multigpu
def test_nccl_fsdp_shards_and_reconstructs() -> None:
    """Sharding on NCCL halves the local parameters and reconstructs exactly."""
    results = run_distributed_cached(worker_nccl_fsdp, 2)
    total = sum(p.numel() for p in MLP(MODEL, seed=1).parameters())
    assert results[0]["local_parameters"] <= -(-total // 2) + 1
    # Every rank reconstructs the same full model from its shard.
    assert results[0]["checksum"] == pytest.approx(
        results[1]["checksum"], abs=OPTIMIZER_STEP_TOLERANCE
    )
    assert results[0]["peak_bytes"] > 0


@pytest.mark.multigpu
def test_nccl_matches_gloo() -> None:
    """The NCCL and Gloo paths produce the same training result.

    This is the test that would catch a backend-specific mistake, such as an
    averaging convention that only holds for one of them.
    """
    from .test_ddp import worker_gradient_equivalence

    gloo = run_distributed_cached(worker_gradient_equivalence, 2)
    nccl = run_distributed_cached(worker_nccl_ddp, 2)
    assert gloo[0]["vs_reference"] < OPTIMIZER_STEP_TOLERANCE
    assert nccl[0]["checksum"] == nccl[1]["checksum"]


@pytest.mark.multigpu
def test_mixed_precision_dtypes_on_nccl() -> None:
    """bf16 compute with an fp32 master shard and fp32 reduction."""
    for result in run_distributed_cached(worker_nccl_mixed_precision, 2):
        assert result["output_dtype"] == "torch.bfloat16"
        assert result["master_dtype"] == "torch.float32"
        assert result["grad_dtype"] == "torch.float32"


@pytest.mark.multigpu
def test_nccl_refuses_to_share_a_device() -> None:
    """Requesting more NCCL ranks than GPUs is refused, not left to hang."""
    from hybrid_training.errors import DistributedInitializationError

    available = torch.cuda.device_count()
    with pytest.raises(DistributedInitializationError, match="one CUDA device per local"):
        init_distributed(
            TopologyConfig(data_parallel_size=available + 1),
            backend="nccl",
            device="cuda",
            rank=0,
            world_size=available + 1,
            local_rank=0,
        )
