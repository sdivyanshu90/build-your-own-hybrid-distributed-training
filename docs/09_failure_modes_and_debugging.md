# 09 — Failure modes and debugging

> Every error quoted here is produced by the code and, unless marked otherwise,
> asserted by a test.

---

## 1. The two shapes of distributed failure

**Loud failures** raise an exception naming the rank, the operation and the
values involved. They are the easy case.

**Quiet failures** are the ones this project is designed around:

| Symptom | Usual cause |
|---|---|
| the job hangs | ranks issued different collectives, or on different groups, or in different orders |
| the loss decreases but converges to the wrong thing | reduction over the wrong group; a missing adjoint |
| the loss is fine but replicas differ | a missing synchronisation boundary; different initial weights |
| memory does not drop when sharding is enabled | the unsharded model is still resident |
| results change between runs | a seed stream that should have been identical was not |

A hang gives you no information at all. Most of the design decisions in this
codebase — mandatory groups, fixed collective ordering, collective precondition
checks — exist to turn hangs into exceptions.

---

## 2. Diagnosing a hang

### Step 0: rule out an overloaded machine first

A rank that never reports has two very different causes that look identical:

| Cause | Signature |
|---|---|
| a genuine collective mismatch | reproduces in isolation, on an idle machine |
| an overloaded or swapping box | disappears when nothing else is running |

`launch_workers` prints the machine's state alongside every timeout for exactly
this reason:

```
--- rank 3: FAILED (exit_code=None, 0.00s) ---
rank produced no result: it was still running when the timeout expired, or it
died before it could report.
Two very different causes look identical here:
  1. A genuine mismatch -- this rank blocked inside a collective its peers did
     not issue. ...
  2. An overloaded machine -- each spawned rank costs a full `import torch`
     (~320 MB resident), so concurrent distributed runs on a memory-tight box
     swap and blow the timeout with nothing actually wrong.
  Machine at failure: load 24.7/21.3 over 8 core(s); 1240 MiB available,
  1795 MiB swapped.
```

Read that last line before anything else. A load average three times the core
count, or swap in active use, means **re-run the test alone on an idle machine
before investigating further**. This happened during development: a
tensor-parallel test failed once in a full-suite run under load average 24, and
passed on every isolated re-run.

### Step 1: find out which ranks are stuck

Under the test harness this is automatic. Ranks that never report are named:

```
WorkerFailure: launch.launch_workers: 2 of 4 worker(s) timed out;
observed 'failing ranks: [1, 3]'.
Fix: read the per-rank tracebacks below; a rank with no traceback was blocked
in a collective
--- rank 0: ok (exit_code=0, 1.83s) ---
--- rank 1: FAILED (exit_code=None, 0.00s) ---
rank produced no result: it was still running when the timeout expired, ...
  Machine at failure: load 1.2/1.1 over 8 core(s); 4100 MiB available, 0 MiB swapped.
```

Here the machine was idle, so the timeout is a real mismatch and worth
investigating.

Under `torchrun`, use `py-spy dump --pid <pid>` on each rank. The rank whose
stack is *not* in a collective is the one that went somewhere else.

### Step 2: check the group construction log

```python
print(context.groups.describe())
print(context.groups.creation_log)
```

Every rank must produce an identical `creation_log`. If they do not, the
communicators disagree about membership and every subsequent collective is a
coin flip.

```
process groups for rank 3:
  data_parallel    size=2   local_rank=1   ranks=(1, 3)
  shard            size=1   local_rank=0   ranks=(3,)
  sequence         size=1   local_rank=0   ranks=(3,)
  tensor           size=2   local_rank=1   ranks=(2, 3)
  dp_shard         size=2   local_rank=1   ranks=(1, 3)
  tensor_sequence  size=2   local_rank=1   ranks=(2, 3)
  world            size=4   local_rank=3   ranks=(0, 1, 2, 3)
```

### Step 3: check the collective ordering

Enable the FSDP order check:

```yaml
fsdp:
  check_reduction_order: true
```

It all-gathers the per-unit reduction counts at the end of every backward and
raises if they differ — one small collective per step, and it catches the class
of bug that otherwise manifests as a hang on step 200.

### Step 4: raise the log level

```bash
HYBRID_LOG_FORMAT=json torchrun --standalone --nproc-per-node=4 \
    examples/train_hybrid.py --log-level DEBUG 2> ranks.jsonl
jq -r 'select(.level=="DEBUG") | "\(.rank) \(.message)"' ranks.jsonl | sort -s -k1,1n
```

Every barrier logs at DEBUG with its label, so the last barrier each rank
reached tells you where they diverged.

### Step 5: if exactly one rank is missing, look for a one-sided `raise`

There is a distinctive signature worth learning:

```
--- rank 0: ok (exit_code=0) ---
--- rank 1: FAILED ---
RuntimeError: [.../gloo/transport/tcp/pair.cc:534] Connection closed by peer
```

Rank 1 is blamed, but rank 1 is the *victim*. `Connection closed by peer` means
the other end's process went away while rank 1 was inside a collective. So the
question is never "what is wrong with rank 1" — it is "why did rank 0 leave".

Almost always the answer is a guarded raise:

```python
if context.is_primary:
    if something_is_wrong:
        raise SomeError(...)      # rank 0 unwinds...
context.barrier("world")          # ...and everyone else blocks here forever
```

The error is real and the message is good; it just never reaches the other
ranks, and they convert it into a hang. This exact bug shipped in
`save_checkpoint` and is dissected in `08_distributed_checkpointing.md` §6.

The rule that prevents it: **a decision that gates a collective must itself be
collective.** Compute the verdict wherever it is cheapest, then all-gather it
and have every rank raise the same error. Grep for `is_primary` followed by
`raise` — that pairing is the smell.

---

## 3. Failure catalogue

### Configuration

| Condition | Error | Where |
|---|---|---|
| topology does not factor the world size | `TopologyError: parallel dimensions do not factor the world size (dp=2 x shard=2 x seq=1 x tensor=2); expected 4, observed 8` | `TopologyConfig.validate_against_world_size` |
| `sequence_parallel_size > 1` without independent mode | `TopologyError: sequence_parallel_size > 1 requires mode 'independent'` | `TopologyConfig.__post_init__` |
| `tensor_group` mode with `tensor_parallel_size == 1` | `TopologyError: needs a tensor-parallel group wider than one rank` | same |
| unknown config key | `ConfigurationError: unknown configuration key(s) under 'topology'` | `ExperimentConfig.from_dict` |
| Gloo requested on CUDA | `ConfigurationError: the Gloo backend on CUDA tensors is supported only for a small subset of collectives` | `ExperimentConfig.__post_init__` |
| SP enabled in layers but not topology | `ConfigurationError: tensor_parallel.sequence_parallel is enabled but the topology does not create a sequence-parallel group` | same |
| loss scaling with bf16 | `ConfigurationError: dynamic loss scaling is only meaningful for float16 compute` | `MixedPrecisionConfig.__post_init__` |

### Runtime bring-up

| Condition | Error |
|---|---|
| partially set launch environment | `DistributedInitializationError: the distributed launch environment is only partially set, which means the launcher failed part-way` |
| NCCL with fewer GPUs than ranks | `DistributedInitializationError: NCCL requires one CUDA device per local process; sharing a device between ranks hangs instead of failing` |
| CUDA requested with no device | `DistributedInitializationError: CUDA/NCCL requested but no CUDA device is visible` |
| second context in one process | `DistributedInitializationError: a distributed context is already active in this process` |
| using a context after shutdown | `DistributedInitializationError: the distributed context has been shut down` |

### Collectives

| Condition | Error |
|---|---|
| non-contiguous tensor | `CollectiveError: collectives require contiguous tensors; a non-contiguous buffer would be silently copied by some backends and rejected by others` |
| indivisible reduce-scatter | `CollectiveError: the leading dimension must be divisible by the group size` |
| group the rank is not in | `CollectiveError: rank is not a member of process group 'tensor'` |
| self send/recv | `CollectiveError: a rank cannot send to itself; this deadlocks with a blocking send` |

### DDP

| Condition | Error |
|---|---|
| model structure differs across ranks | `ParameterConsistencyError: the (name, shape, dtype) list of trainable parameters differs across the 'data_parallel' group` |
| bucket layout differs | `ParameterConsistencyError: the gradient bucket layout differs across the 'data_parallel' group` |
| unused parameter, flag off | `ShardingError: some parameters received no gradient, so their buckets can never become ready and the all-reduce would never be issued (every other rank would block waiting for it)` |
| forward before the boundary | `ShardingError: a previous backward pass has not been synchronised; stepping now would use gradients that were never averaged across ranks` |
| two backwards, no boundary | `ShardingError: a gradient arrived for a bucket that has already been reduced this iteration` |

### FSDP

| Condition | Error |
|---|---|
| mixed `requires_grad` in one unit | `UnsupportedFeatureError: an FSDP unit flattens its parameters into a single tensor, so all of them must share one requires_grad flag` |
| parameter tied across units | `UnsupportedFeatureError: a parameter is shared between two FSDP units; each unit would reduce and update its own copy, so the tie would break after the first optimizer step` |
| unit too large | `ShardingError: this unit's all-gather would exceed limit_all_gather_bytes` |
| flat layout differs across ranks | `ParameterConsistencyError: the flat parameter layout differs across the 'shard' group` |
| reduction counts differ | `ParameterConsistencyError: ranks performed different numbers of reduce-scatters per unit; the collective streams have diverged and the next step will hang` |

### Tensor and sequence parallelism

| Condition | Error |
|---|---|
| indivisible features | `TensorParallelError: out_features must be divisible by the tensor-parallel size; an uneven split would give ranks different shapes` |
| indivisible heads | `ConfigurationError: attention heads must divide evenly across the tensor-parallel group; splitting a head would break the softmax` |
| indivisible sequence | `ShardingError: the sequence dimension must be divisible by the sequence-parallel size` |
| tied embeddings + vocab parallel | `UnsupportedFeatureError: the embedding is sharded along the vocabulary while the output projection needs the same matrix sharded along its output features` |
| SP with a one-rank group | `ConfigurationError: sequence_parallel was requested but the sequence group has one member` |
| model-parallel peers fed different data | `ParameterConsistencyError: the input batch differs from the value held by group-local rank 0` |

### Checkpointing

See `08_distributed_checkpointing.md` §10 for the full table.

---

## 4. Silent-failure checklist

When training runs but the numbers are wrong, work through these in order.

**1. Are the replicas identical?**

```python
model.ddp.verify_replica_consistency()   # atol=0
```

If they differ, a synchronisation boundary is missing or the initial broadcast
did not happen.

**2. Are model-parallel peers seeing the same batch?**

```python
model.validate_input_replication(batch.inputs)
```

If they differ, the sampler is indexing by the wrong group. This produces a
plausible loss curve and a wrong model.

**3. Is the loss reduced over the right group?**

```python
assert model.metric_group.name == "dp_shard"
print(model.metric_group.ranks)
```

Reducing over the world weights each sample by the tensor-parallel size.

**4. Do all ranks agree on the gradient norm?**

```python
print(optimizer.clip_grad_norm(0.0).item())   # must be identical everywhere
```

If not, the per-parameter replication weighting is wrong — usually a parameter
that should be marked partitioned but is not, or vice versa.

**5. Under sequence parallelism, were the partial gradients completed?**

```python
n = sum(1 for p in model.inner_model.parameters()
        if getattr(p, "sequence_parallel_partial_grad", False))
print("parameters needing an explicit reduction:", n)
```

Zero when sequence parallelism is on means the markers are missing, and the
LayerNorm gradients are each missing `(G−1)/G` of their value.

**6. Did sharding actually save memory?**

```python
print(model.memory_summary())
# {'shard_bytes': …, 'full_bytes': 0, 'grad_shard_bytes': …, 'padding_bytes': …}
```

`shard_bytes` should be roughly `total/G`. If it is not, the original
parameters were never released — the single most embarrassing FSDP bug, and the
reason `_detach_original_parameters` sets `param.data = torch.empty(0)`.

---

## 5. Why the "silent" bugs are silent

Three real bugs found while building this project, all of which produced a
*perfect* forward pass:

**Sequence-gather adjoint.** Using the split adjoint instead of reduce-scatter
gave logits matching the single-process reference to `4.6e-07` and LayerNorm
gradients wrong by `3.3e-02`. Only a test that compares *gradients* against an
unsharded reference catches it.

**Partial LayerNorm gradients.** Even with the right adjoint, the replicated
parameters' gradients were each missing the other rank's positions. Same
signature: forward perfect, gradients wrong.

**Tensor returns from spawned workers.** `mp.Queue` moves torch tensors through
shared-memory file descriptors that die with the sender, so a worker returning
a tensor and exiting raced the parent's read. Intermittent
`ConnectionResetError`, ~1 run in 5. The fix was to serialise with plain pickle.

The lesson encoded in `11_testing_strategy.md`: **a forward-pass comparison is
not a correctness test.** Every equivalence test in this repository compares
gradients and post-optimizer-step weights, not only outputs.

---

## 6. Reproducibility checklist

When two runs of the same configuration disagree:

| Check | Command |
|---|---|
| deterministic algorithms on | `training.deterministic: true` in the config |
| same seed reaching the model | `derive_seed(seed, "model-init")` is rank-independent |
| dropout disabled for comparisons | `model.dropout: 0.0` |
| data order identical | `sampler.epoch_order(0)` must match across ranks |
| RNG restored on resume | `LoadedCheckpoint.rng_restored` is `True` |
| same world size | reduction order differs between world sizes; expect `~1e-8`, not `0` |

Note the last row: a *bitwise* comparison is only valid between runs at the same
world size. Across world sizes, floating-point summation order changes and the
right expectation is `1e-6`-ish, which is what the reshard tests assert.

---

## 7. Environment variables

| Variable | Effect |
|---|---|
| `HYBRID_LOG_FORMAT=json` | one JSON object per log line, for multi-rank analysis |
| `OMP_NUM_THREADS=1` | one thread per rank; without it four ranks each start an eight-thread pool |
| `NCCL_DEBUG=INFO` | NCCL's own topology and ring construction output |
| `TORCH_DISTRIBUTED_DEBUG=DETAIL` | PyTorch's collective mismatch detection |
| `CUDA_LAUNCH_BLOCKING=1` | makes CUDA errors point at the right line, at a large speed cost |

`torchrun_environment_summary()` prints the launcher-relevant variables with
`<unset>` for absent ones, so a missing variable is visible rather than implied.

---

## 8. Common mistakes, ranked by how long they take to find

1. **Reducing over the wrong group.** No error, wrong model. Prevented
   structurally: no collective wrapper has a default group.
2. **A missing adjoint.** No error, wrong gradients, perfect forward pass.
   Caught only by gradient-level equivalence tests.
3. **Inconsistent group creation order.** Hang, with no useful message from the
   backend. Prevented by the single ordered registry.
4. **Feeding model-parallel peers different data.** No error, plausible loss.
   Caught by `validate_input_replication`.
5. **Forgetting the synchronisation boundary.** Replicas silently diverge.
   Caught by the guard in `DDP.forward`.
6. **Not releasing the unsharded model.** No error, no memory saving.
   Caught by `memory_summary()` and by the memory-scaling tests.
7. **Comparing across world sizes without holding the global batch constant.**
   Looks like a resharding bug, is not. Caught by reading the test carefully —
   this one bit the first version of the reshard example.
