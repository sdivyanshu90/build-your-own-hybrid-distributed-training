# build-your-own-hybrid-distributed-training

A from-scratch, correctness-first implementation of the parallelism strategies
used to train large models — **distributed data parallelism, FSDP-style
sharding, tensor parallelism, sequence parallelism, their hybrid composition,
and a resharding distributed checkpoint format** — built on nothing but
`torch.distributed`'s raw collectives.

Nothing here wraps `torch.nn.parallel.DistributedDataParallel`,
`torch.distributed.fsdp`, DTensor, `torch.distributed.tensor.parallel` or
`torch.distributed.checkpoint`. PyTorch's DDP appears exactly once, in
`tests/distributed/test_ddp.py`, as a **test oracle** — and the custom
implementation matches it to `0.0` at world size 2.

Everything runs on CPU with Gloo. **No GPU is required** for the full
correctness suite.

---

## Install

```bash
python -m pip install -e ".[dev]"
```

## Run something

```bash
# two-rank data parallelism
torchrun --standalone --nproc-per-node=2 examples/train_ddp.py

# four-rank parameter/gradient/optimizer sharding
torchrun --standalone --nproc-per-node=4 examples/train_fsdp.py \
    --config configs/fsdp_4gpu.yaml

# tensor parallelism, then tensor + sequence parallelism
torchrun --standalone --nproc-per-node=2 examples/train_tensor_parallel.py
torchrun --standalone --nproc-per-node=2 examples/train_sequence_parallel.py

# everything at once
torchrun --standalone --nproc-per-node=4 examples/train_hybrid.py

# save a checkpoint at one world size, resume it at another
python examples/resume_with_different_world_size.py

# inspect a checkpoint without loading any tensor
python scripts/inspect_checkpoint.py runs/exp/checkpoint-step-000100
```

Each example writes checkpoints under `runs/<name>/`, and saving refuses to
overwrite an existing checkpoint — silently replacing one is how you lose the
run you actually wanted. So running the same example twice stops with

```
CheckpointError: the destination checkpoint already exists ...
Fix: remove the existing checkpoint or save at a different step
```

which is deliberate. Delete `runs/<name>/` or pass `--checkpoint-dir` to point
somewhere fresh. The error is raised on *every* rank, not just rank 0 — see
`docs/08_distributed_checkpointing.md` §6 for why that distinction matters.

## Test it

```bash
pytest -q tests/unit                        # ~25 s, no processes spawned
pytest -q -m "not cuda and not multigpu"    # what CI runs
./scripts/run_tests_distributed.sh          # multi-process suites, one file at a time
```

---

## What is implemented

| Strategy | Module | Mechanism |
|---|---|---|
| **DDP** | `parallel/ddp.py` | gradient bucketing, post-accumulate hooks, asynchronous all-reduce overlapped with backward, index-ordered launches, `no_sync()` accumulation, unused-parameter handling |
| **FSDP** | `parallel/fsdp.py` | flat parameters with padding, all-gather/reshard lifecycle via storage `resize_`, reduce-scatter expressed as an autograd adjoint, nested wrapping, `summon_full_params()`, hybrid sharding |
| **Tensor parallel** | `parallel/tensor_parallel.py` | column/row-partitioned linear layers, vocabulary-parallel embedding, the `f`/`g` autograd pair, head-partitioned attention |
| **Sequence parallel** | `parallel/sequence_parallel.py` | sequence scatter/gather/reduce-scatter with correct adjoints, padding metadata, explicit completion of partial parameter gradients |
| **Hybrid** | `parallel/hybrid.py` | four-dimensional rank grid, explicit group assignment for every collective, rendered and asserted parallel plan |
| **Checkpointing** | `checkpoint/` | JSON manifest over global flat intervals, SHA-256 integrity, atomic publish by rename, resharding across world sizes |

---

## Results

Measured on this repository with Gloo on CPU. Every number is asserted by the
test named beside it.

### DDP vs references

| Comparison | Max gradient difference | Test |
|---|---|---|
| custom DDP vs single-process reference (`W=2`) | `1.5e-08` | `test_matches_single_process_reference` |
| custom DDP vs **PyTorch DDP** (`W=2`) | `0.0` | `test_matches_pytorch_ddp` |
| custom DDP vs single-process reference (`W=4`) | `7.5e-09` | same |
| `no_sync` accumulation vs one large batch | `1.5e-08` | `test_accumulation_matches_a_single_large_batch` |

### FSDP sharding is real

1857-parameter MLP, deliberately divisible by neither 2 nor 4:

| World size | shard numel | padding | optimizer state | weight error after 5 AdamW steps |
|---|---|---|---|---|
| 2 | 929 | 1 | 1858 | `3.0e-08` |
| 4 | 465 | 3 | 930 | `4.5e-08` |

`full_bytes` between steps is `0`; `full_state_dict()` reconstructs the original
parameters **bitwise**.

### Tensor parallelism is exact

| Quantity | `T=2` | `T=4` |
|---|---|---|
| forward output (gathered) vs `nn.Linear` | `0.0` | `0.0` |
| gathered weight vs reference weight | `0.0` | `0.0` |
| weight-shard gradient vs reference slice | `0.0` | `0.0` |
| input gradient (passes through an all-reduce) | `1.2e-07` | `2.4e-07` |

### Checkpoint resharding

| Scenario | Result |
|---|---|
| save at 4 FSDP ranks, restore at 2 | restored parameters **bitwise identical**; 2 of 4 files read per rank |
| resume vs uninterrupted, same world size | final weights **bitwise identical**; losses identical |
| resume across world sizes, global batch held constant | `1.4e-09` after continuing to the same step |

---

## Repository layout

```text
configs/     eight YAML experiment configurations, single-process to 8-rank hybrid
docs/        fourteen documents; every mechanism derived, every number sourced
examples/    seven runnable programs, each with its expected output documented
scripts/     test driver, benchmark sweep, checkpoint inspector
src/hybrid_training/
  config.py logging.py errors.py
  distributed/  topology, groups, context, collectives, launcher
  autograd/     differentiable collectives and their adjoints
  parallel/     ddp, fsdp, tensor_parallel, sequence_parallel, hybrid
  optim/        sharded optimizer, distributed norm, fp16 loss scaling
  checkpoint/   format, manifest, writer, reader, reshard
  models/       MLP and a tensor/sequence-parallel transformer
  training/     engine, state, deterministic synthetic data
  utils/        shape arithmetic, memory accounting, reproducibility
tests/       unit, distributed, integration, end_to_end, performance
```

---

## Documentation

Start with [`docs/00_overview.md`](docs/00_overview.md).

| | |
|---|---|
| [01 Foundations](docs/01_distributed_systems_foundations.md) | ranks, coordinates, groups, why creation order causes hangs |
| [02 Collectives](docs/02_collective_communication.md) | every operation, its cost, its adjoint |
| [03 DDP](docs/03_distributed_data_parallel.md) | bucketing, overlap, the synchronisation boundary |
| [04 FSDP](docs/04_fsdp_style_sharding.md) | flat parameters, padding, the storage-resize trick |
| [05 Tensor parallel](docs/05_tensor_parallelism.md) | column vs row, and why they pair |
| [06 Sequence parallel](docs/06_sequence_parallelism.md) | what it is, and what it is *not* |
| [07 Hybrid](docs/07_hybrid_parallelism.md) | which group carries which traffic; worked 8-rank example |
| [08 Checkpointing](docs/08_distributed_checkpointing.md) | manifest format, atomicity, resharding, security |
| [09 Debugging](docs/09_failure_modes_and_debugging.md) | every error, and the silent-failure checklist |
| [10 Performance](docs/10_performance_engineering.md) | volume, overlap, and the known limitations |
| [11 Testing](docs/11_testing_strategy.md) | tolerances, the harness, what is *not* tested |
| [12 API reference](docs/12_api_reference.md) | every public symbol |
| [13 Deliverables](docs/13_deliverables.md) | feature matrix, test matrix, measured results, acceptance criteria |

---

## Design principles

**Explicit groups, always.** No collective wrapper accepts `group=None`. In a
hybrid job the difference between reducing over the data-parallel group and over
the world is the difference between correct training and training that
converges to the wrong thing without ever raising.

**One code path, not two.** At world size 1 the framework still creates a real
process group and still calls every collective — they are identity operations
over a one-member group. The single-process reference the tests compare against
therefore exercises the *same code* as the distributed run.

**Fail loudly and collectively.** Preconditions that can be checked with a
collective are, so a mismatch raises on every rank instead of hanging the ones
that did not notice.

**Determinism before speed.** Collectives are issued in a fixed order that does
not depend on which gradient happened to finish first.

**A forward-pass comparison is not a correctness test.** Three real bugs found
while building this produced a *perfect* forward pass and gradients wrong by six
orders of magnitude. Every equivalence test compares gradients and
post-optimizer-step weights, not only outputs.

---

## Known limitations

Stated plainly rather than implied by absence. Details in
[`docs/10_performance_engineering.md`](docs/10_performance_engineering.md) §6.

- **Python floor is 3.10, not 3.11.** The reference environment ships 3.10.12,
  and running the suite for real was judged more valuable than an unrunnable
  floor. No 3.11-only syntax is used; raising it is a one-line change.
- **No FSDP prefetching.** The all-gather latency is exposed rather than hidden
  behind the previous unit's compute.
- **No fused kernels, no FlashAttention.** Attention materialises the
  `(B, heads/T, S, S)` score matrix, because the explicit form is bitwise
  reproducible and the tests depend on that.
- **No pipeline or context parallelism.** Both are genuinely different in
  character; §8 of the sequence-parallelism document describes what context
  parallelism would need.
- **Tensor-parallel width cannot be changed by resharding.** Rejected explicitly
  with an explanation, not silently mishandled.
- **Single node only.** The rendezvous code is the same, but no test crosses a
  network.
- **Coverage under-reports.** Distributed code runs in child processes and child
  coverage is not merged.

---

## Requirements

- Python ≥ 3.10 (3.11+ works; see above)
- PyTorch ≥ 2.1 — developed against 2.3
- NumPy, PyYAML

## Licence

MIT. See [LICENSE](LICENSE).
