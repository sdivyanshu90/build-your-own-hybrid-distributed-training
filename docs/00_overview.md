# 00 — Overview

This repository implements, from the collectives up, the parallelism
strategies used to train large models:

- **Distributed data parallelism (DDP)** — replicate the model, split the batch,
  average the gradients.
- **FSDP-style sharding** — split parameters, gradients *and* optimizer state
  across ranks, materialising each unit only while it is needed.
- **Tensor parallelism** — split individual weight matrices and attention heads.
- **Sequence parallelism** — split activations along the sequence axis in the
  regions where that is mathematically free.
- **Hybrid composition** — all of the above at once, over a four-dimensional
  rank grid.
- **Distributed checkpointing** — a manifest-based format that can be resharded
  across world sizes.

Nothing here wraps `torch.nn.parallel.DistributedDataParallel`,
`torch.distributed.fsdp`, DTensor, `torch.distributed.tensor.parallel` or
`torch.distributed.checkpoint`. The only PyTorch distributed APIs used are the
raw collectives: `all_reduce`, `all_gather_into_tensor`,
`reduce_scatter_tensor`, `all_to_all_single`, `broadcast`, `barrier`, `send`
and `recv`. PyTorch's DDP appears exactly once, in
`tests/distributed/test_ddp.py`, as a *test oracle*.

---

## Reading order

| Document | What it covers |
|---|---|
| [01 — Distributed systems foundations](01_distributed_systems_foundations.md) | Ranks, groups, the runtime context, why group creation order matters |
| [02 — Collective communication](02_collective_communication.md) | Every collective, its cost model, and its adjoint |
| [03 — Distributed data parallel](03_distributed_data_parallel.md) | Bucketing, overlap, `no_sync`, the synchronisation boundary |
| [04 — FSDP-style sharding](04_fsdp_style_sharding.md) | Flat parameters, padding, the unshard/reshard lifecycle |
| [05 — Tensor parallelism](05_tensor_parallelism.md) | Column/row partitioning and the `f`/`g` autograd pair |
| [06 — Sequence parallelism](06_sequence_parallelism.md) | Activation sharding, and what it is *not* |
| [07 — Hybrid parallelism](07_hybrid_parallelism.md) | Composing all four dimensions, with a worked 8-rank example |
| [08 — Distributed checkpointing](08_distributed_checkpointing.md) | The manifest format, atomicity, resharding |
| [09 — Failure modes and debugging](09_failure_modes_and_debugging.md) | How each failure presents, and how to diagnose it |
| [10 — Performance engineering](10_performance_engineering.md) | Communication volume, overlap, why not to over-synchronise |
| [11 — Testing strategy](11_testing_strategy.md) | What is tested, how, and with what tolerances |
| [12 — API reference](12_api_reference.md) | Every public class and function |
| [13 — Deliverables](13_deliverables.md) | Feature matrix, test matrix, results, acceptance-criteria mapping |

---

## The one-paragraph version of each strategy

**DDP.** Every rank holds the whole model and a slice of the batch. The
gradient of the global mean loss is the mean of the per-rank gradients, so the
only required operation is one all-reduce per step. Everything else — bucketing
gradients so the first collective can start before backward finishes — is
performance engineering around that fact.

**FSDP.** Every rank holds `1/G` of the parameters. Before a unit computes, its
parameters are all-gathered; afterwards they are thrown away. Gradients flow
into one flat buffer and are reduce-scattered, so each rank ends up with the
gradient slice matching its parameter slice — and therefore its optimizer state
slice. Persistent memory drops by a factor of `G`; communication rises.

**Tensor parallelism.** A weight matrix is cut either by output features
(column parallel: no forward communication, one backward all-reduce) or by
input features (row parallel: one forward all-reduce, no backward
communication). Chaining a column-parallel layer into a row-parallel one costs
*one* collective for the pair, which is why transformer blocks are built that
way.

**Sequence parallelism.** Between tensor-parallel layers the activations are
replicated and idle. Splitting them along the sequence costs nothing extra,
because an all-reduce and a (reduce-scatter + all-gather) move the same bytes.
Attention still needs the whole sequence, so it is gathered — this is *not*
context parallelism.

**Hybrid.** The four strategies act on four different axes, so they compose. The
only hard part is making sure every collective runs on the right group, which
is why no collective wrapper in this repository has a default group.

---

## Repository layout

```text
src/hybrid_training/
├── config.py              frozen dataclasses; every knob, validated
├── errors.py              exception hierarchy; rank-aware messages
├── logging.py             rank-aware structured logging
├── distributed/
│   ├── topology.py        pure rank arithmetic (no torch.distributed)
│   ├── groups.py          the only caller of dist.new_group
│   ├── context.py         owns the process group, device and groups
│   ├── collectives.py     explicit-group collective wrappers
│   └── launch.py          in-process multi-rank launcher for tests
├── autograd/
│   └── collectives.py     differentiable collectives and their adjoints
├── parallel/
│   ├── ddp.py             bucketed data parallelism
│   ├── fsdp.py            flat parameters, unshard/reshard, reduce-scatter
│   ├── tensor_parallel.py column/row/vocab parallel layers
│   ├── sequence_parallel.py  sequence scatter/gather/reduce-scatter
│   └── hybrid.py          composition, and the group-assignment table
├── optim/
│   ├── sharded_optimizer.py  distributed norm, CPU offload
│   └── grad_scaler.py     fp16 loss scaling with collective overflow consensus
├── checkpoint/
│   ├── format.py          naming, path safety, versioning
│   ├── manifest.py        the global-tensor description
│   ├── writer.py          atomic, integrity-checked writes
│   ├── reader.py          loading and resharding
│   └── reshard.py         interval arithmetic and file caching
├── models/
│   ├── mlp.py             the fast correctness-test model
│   └── transformer.py     tensor/sequence-parallel transformer
├── training/
│   ├── engine.py          one training loop for every strategy
│   ├── state.py           progress and learning-rate schedules
│   └── data.py            deterministic synthetic data, topology-aware sampler
└── utils/
    ├── tensors.py         flatten/pad/shard/intersect arithmetic
    ├── memory.py          analytical model and measurement
    └── reproducibility.py seeding and RNG state
```

---

## Installation

```bash
python -m pip install -e ".[dev]"
```

Requirements: PyTorch ≥ 2.1 (2.3 is what the project is developed against),
NumPy and PyYAML. No GPU is required — the whole correctness suite runs on CPU
with the Gloo backend.

### Python version

The project specification asks for Python ≥ 3.11; `pyproject.toml` declares
`requires-python = ">=3.10"`. The reference development environment for this
repository ships Python 3.10.12, and running the test suite *for real* was
judged more valuable than enforcing a floor that would have made it unrunnable.

The source avoids every 3.11-only construct — no PEP 695 generics, no
`typing.Self`, no `except*`, no `enum.StrEnum` — so raising the floor is a
one-line change in `pyproject.toml` plus the `target-version` settings for Ruff
and mypy. Nothing else needs to move.

---

## Commands

```bash
# tests
pytest -q                                   # everything
pytest -q -m "not cuda"                     # skip GPU-only tests
pytest -q -m "not cuda and not multigpu"    # what CI runs
./scripts/run_tests_distributed.sh          # one file at a time (see below)

# training examples (no GPU required)
torchrun --standalone --nproc-per-node=2 examples/train_ddp.py
torchrun --standalone --nproc-per-node=2 examples/train_fsdp.py
torchrun --standalone --nproc-per-node=2 examples/train_tensor_parallel.py
torchrun --standalone --nproc-per-node=2 examples/train_sequence_parallel.py
torchrun --standalone --nproc-per-node=4 examples/train_hybrid.py

# checkpointing
torchrun --standalone --nproc-per-node=2 examples/save_distributed_checkpoint.py
python examples/resume_with_different_world_size.py
python scripts/inspect_checkpoint.py /path/to/checkpoint-step-000100

# benchmarks
python scripts/benchmark.py --world-size 2 --strategies ddp fsdp
```

`scripts/run_tests_distributed.sh` exists because the distributed tests spawn
child processes: running several test *files* concurrently multiplies the
process count and, on a machine with fewer cores than ranks, turns a two-minute
suite into a twenty-minute one that looks like a hang. The script runs files
sequentially and ranks in parallel, which is the right split.

---

## Design principles

1. **Explicit groups, always.** No collective wrapper accepts `group=None`. In a
   hybrid job the difference between reducing over the data-parallel group and
   over the world is the difference between correct training and training that
   converges to the wrong thing without ever raising.

2. **One code path, not two.** At world size 1 the framework still creates a
   real process group and still calls every collective; they are identity
   operations over a one-member group. The "single-process reference" the tests
   compare against therefore exercises the *same code* as the distributed run.

3. **Fail loudly and collectively.** Preconditions that can be checked with a
   collective are checked with one, so a mismatch raises on every rank instead
   of hanging the ones that did not notice.

4. **Determinism before speed.** Collectives are issued in a fixed order that
   does not depend on which gradient happened to finish first.

5. **Every number in the documentation is asserted somewhere.** The 8-rank
   topology table in `07_hybrid_parallelism.md` is checked by
   `tests/unit/test_topology.py::TestCoordinates::test_documented_eight_rank_example`.

---

## What this project deliberately does not do

- Pipeline parallelism, expert parallelism, context (ring) attention.
- Custom CUDA kernels — everything is PyTorch operations, and the places where
  a production system would need a fused kernel are called out in
  `10_performance_engineering.md`.
- Elastic membership changes during a step, automatic restart, parameter
  servers, or scheduler integration.
- Any claim about trillion-parameter scalability. The mechanisms here are the
  real ones; the constants are not tuned for scale.

These are discussed as future extensions in the relevant documents rather than
being silently absent.
