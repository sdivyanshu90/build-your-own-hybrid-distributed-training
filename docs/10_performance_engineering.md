# 10 — Performance engineering

> Instrumentation in `distributed/collectives.py` (`CommunicationRecorder`),
> `utils/memory.py`. Benchmark driver: `scripts/benchmark.py`.
> Invariant tests: `tests/performance/test_instrumentation.py`.

**Read this first.** This project optimises for correctness and clarity. Where a
production system would need a fused CUDA kernel or an overlapping stream
schedule, this one uses plain PyTorch operations and says so. The numbers below
describe the *shape* of the trade-offs; they are not portable performance
figures and should not be quoted as such.

---

## 1. What actually costs time

| Cost | Scales with | Reduced by |
|---|---|---|
| gradient all-reduce (DDP) | parameters | accumulation, bucketing (overlap) |
| parameter all-gather (FSDP) | parameters × units | larger units, prefetching |
| gradient reduce-scatter (FSDP) | parameters | — (irreducible) |
| activation all-reduce (TP) | batch × sequence × hidden × layers | keeping TP inside a node |
| activation memory | batch × sequence × hidden × layers | sequence parallelism, checkpointing |
| optimizer step | local parameters | sharding |

The single most useful rule: **tensor parallelism communicates activations,
FSDP communicates parameters.** Activation traffic scales with batch and
sequence length and is issued *per layer*; parameter traffic is independent of
batch size and is issued *per unit*. That is why tensor parallelism belongs
inside a node (NVLink) and FSDP can span nodes.

---

## 2. Communication volume, measured

From `tests/performance/test_instrumentation.py`, MLP with 32→64×4→8
(≈ 17k parameters), 3 steps, Gloo:

| Strategy | bytes moved per step | collectives per step |
|---|---|---|
| DDP | `4P` (the full gradient) | 1 per bucket |
| DDP, `no_sync` over 4 micro-batches | `4P` per *optimizer* step | 1 per bucket per step |
| DDP, 4 synchronised micro-batches | `16P` per optimizer step | 4× |
| FSDP, `reshard_after_forward=True` | `4P` (gather) × 2 + `4P` (scatter) | 2 all-gather + 1 reduce-scatter per unit |
| FSDP, `reshard_after_forward=False` | `4P` (gather) + `4P` (scatter) | 1 all-gather + 1 reduce-scatter per unit |

Two properties the tests assert rather than assume:

- **DDP's volume is independent of the world size.** Each rank sends the whole
  gradient regardless of how many peers it has. (The *wire* volume of a ring
  all-reduce is `2N(G−1)/G`, but the payload each rank hands the backend is
  `N`.)
- **Bucket size changes the call count, not the volume.** Caps of 0.001 MiB and
  25 MiB move identical totals.

---

## 3. Overlap, and why not to over-synchronise

DDP launches each bucket's all-reduce asynchronously as soon as the bucket
fills, so the collective proceeds while backward continues on earlier layers:

```
compute:  [=== backward ===]
network:      [=b3=][=b2=][=b1=][=b0=]
```

The recorder separates **launch time** from **wait time** for exactly this
reason. A wait time near zero means the collective finished before anyone
needed it — overlap is working. A wait time close to the launch-to-wait
interval means it is not.

**`calls` counts collectives, not recorder events.** Splitting launch from wait
means an asynchronous collective touches the recorder twice, and the first
version of this code incremented `calls` on both. The result was that a 3-step,
1-bucket DDP run reported **6** all-reduces — with the byte total still exactly
right, because the wait was recorded with `num_bytes=0`. The number was wrong in
the one way that is hardest to notice: plausible, self-consistent, and off by
exactly the factor that makes bucketing look twice as chatty as it is.

Worse, it made the metric depend on *how* a collective was issued rather than on
what crossed the wire: switching `async_reduction` off would have halved the
reported call count without changing a single byte of traffic. A performance
number that moves when you change nothing observable is not a measurement.

`AsyncWork.wait()` therefore calls `record_wait()`, which accumulates
`wait_seconds` and deliberately does not touch `calls`. If you add a collective
wrapper, record the launch once and route the wait through `record_wait` —
`tests/unit/test_single_process_components.py::TestCommunicationRecorder` pins
the invariant that a synchronous and an asynchronous collective count the same.

The report's shape (values vary by machine):

```
communication summary:
  all_reduce/data_parallel   calls=8   MiB=0.066  launch_s=…  wait_s=…
                                                              ^^^^^^^^
                                          near zero means overlap is working
```

### The synchronisation anti-pattern

```python
# WRONG: destroys the overlap you are trying to measure
for step in range(steps):
    torch.cuda.synchronize()      # <-- inside the loop
    t0 = time.perf_counter()
    train_step()
    torch.cuda.synchronize()      # <-- inside the loop
    durations.append(time.perf_counter() - t0)
```

Synchronising every iteration forces every asynchronous collective to complete
before the next iteration's compute begins. The "measurement" then reports a
*serialised* implementation, which is slower than the one actually running —
and worse, it reports the same number regardless of whether overlap works, so a
regression in overlap is invisible.

```python
# RIGHT: synchronise once at each boundary of the timed region
for _ in range(warmup):
    one_step()
context.synchronize_device()          # before the timer
t0 = time.perf_counter()
for _ in range(steps):
    one_step()
context.synchronize_device()          # after the loop
total = time.perf_counter() - t0
```

`scripts/benchmark.py` follows this. On CPU, `synchronize_device()` is a no-op —
`torch.cuda.synchronize()` is never called on a CPU path, because there is
nothing asynchronous to wait for and the call would raise on a CUDA-less build.

### Warm-up

The first iteration pays for lazy allocator growth, communicator setup and
kernel autotuning. It is discarded (`--warmup`, default 5). Reporting it would
make short runs look several times slower than they are.

---

## 4. Memory, analytically and measured

The closed-form model (`utils/memory.py`) for `P` parameters, fp32, Adam:

```
plain data parallel:   16P
full sharding (G):     16P/G  +  4·max_unit_numel·(1 if reshard else 2)
hybrid (shard G):      16P/G  +  same transient
```

`TestMemoryScaling::test_analytical_model_matches_the_measurement` asserts the
formula agrees with the measured shard sizes to within 2 %, so the model is not
merely decorative.

Measured for the 17k-parameter MLP:

| Configuration | resident parameter bytes | optimizer state bytes | full bytes when idle |
|---|---|---|---|
| DDP, `W=2` | `4P` | `8P` | `4P` |
| FSDP, `W=2` | `≈ 4P/2` | `≈ 8P/2` | `0` |
| FSDP, `W=4` | `≈ 4P/4` | `≈ 8P/4` | `0` |

The `0` in the last column is the point of `reshard_after_forward`: nothing
full is resident between steps.

### Why wrapping granularity matters

The transient term is `max_unit_numel`, not `P/G`. Wrapping the whole model as
one unit makes `max_unit_numel == P` and saves nothing at the peak, even though
the *steady-state* memory is correctly `16P/G`. That is the most common way to
configure FSDP and get no benefit.

`limit_all_gather_bytes` exists to make that mistake fail loudly.

---

## 5. Sequence parallelism's memory saving

Tensor parallelism leaves the LayerNorm/dropout/residual regions replicated:
`T` copies of `(B, S, H)`. Sequence parallelism stores `(B, S/T, H)` per rank.

For a 2-layer block, `B=8`, `S=2048`, `H=4096`, fp32, `T=8`:

| | bytes per rank in the replicated regions |
|---|---|
| tensor parallel only | `2 × 8 × 2048 × 4096 × 4 = 512 MiB` |
| tensor + sequence parallel | `64 MiB` |

At **no extra communication**, because `all-reduce ≡ reduce-scatter +
all-gather` moves the same bytes. This is why sequence parallelism is close to
free in practice.

---

## 6. Known performance limitations

These are deliberate. Each says what a production implementation does instead.

### No FSDP prefetching

Production FSDP issues the all-gather for unit `i+1` while unit `i` is still
computing, hiding the latency entirely. This implementation gathers unit `i`
synchronously at the start of its forward.

*Effect:* the all-gather latency is exposed. On a fast interconnect with small
units this can be a large fraction of the step.
*What it would take:* a forward-order queue and a second CUDA stream, plus
careful event synchronisation to avoid using a buffer before its gather lands.

### No fused kernels

Megatron fuses bias+GeLU+dropout, softmax+mask+scale, and LayerNorm. Each
fusion removes a full-activation round trip to HBM.

*Effect:* the transformer block does several more HBM passes than it needs to.
*What it would take:* custom CUDA, explicitly out of scope.

### Attention computed explicitly

`TensorParallelAttention` writes out `q kᵀ / √d`, masks, softmaxes and
multiplies by `v`, materialising the `(B, heads/T, S, S)` score matrix.
`scaled_dot_product_attention` (and FlashAttention behind it) avoids that
materialisation entirely.

*Effect:* attention memory is `O(S²)` rather than `O(S)`, which bounds the
sequence length reachable.
*Why:* the explicit form is bitwise reproducible across CPU and CUDA backends,
which the equivalence tests depend on. A production model should use
`scaled_dot_product_attention`.

### Three q/k/v projections instead of one

Three matmul launches per attention block instead of one. See
`05_tensor_parallelism.md` §4 for why.

### `movedim` copies in sequence-parallel collectives

Gathering along the sequence dimension transposes to put it first, communicates,
and transposes back. A production implementation keeps the sequence dimension
leading throughout (`(S, B, H)` layout) and avoids both copies. Megatron does
exactly this.

### Python-level bucket bookkeeping

DDP's hook, copy and counter logic runs in Python once per parameter per step.
PyTorch's `Reducer` is C++. For a model with thousands of parameters this is
measurable; for the models here it is not.

### No CPU/GPU overlap for offload

`ShardedOptimizer`'s CPU offload copies gradients to the host, steps, and copies
back — all synchronously. Production implementations overlap the transfer with
compute using pinned memory and a copy stream.

### No `torch.compile` or CUDA graphs

Both would help the small-tensor regime substantially, and both make the
executed code harder to relate to the source.

---

## 7. Using the benchmark script

```bash
# single process
python scripts/benchmark.py

# compare strategies at four ranks
python scripts/benchmark.py --world-size 4 --strategies ddp fsdp hybrid

# transformer with tensor and sequence parallelism
python scripts/benchmark.py --world-size 2 --model transformer \
    --strategies tensor sequence

# machine-readable
python scripts/benchmark.py --world-size 2 --json > results.json
```

The output is a fixed-width table with one row per strategy, preceded by a
banner repeating the caveat above. Its columns are: median step time, forward
time, backward time, communication time, communicated MiB per step, collectives
per step, resident parameter bytes, optimizer-state bytes, samples/second — plus
`note:` lines naming anything that makes the run unrepresentative (Gloo on
loopback, missing CUDA allocator statistics on CPU).

What to read from it, rather than the absolute numbers:

- **FSDP moves more and holds less.** It issues roughly three collectives per
  unit per step (two all-gathers with `reshard_after_forward`, one
  reduce-scatter) against DDP's one per bucket, and holds `1/G` of the
  parameters and `1/G` of the optimizer state. Whether that is a good trade
  depends entirely on whether you were running out of memory.
- **`comm ms` against `step ms`** says whether communication is on the critical
  path. Under Gloo on loopback it usually is, which is a property of the
  transport rather than of the strategy.
- **`params` against `opt state`** should be roughly 1:2 for AdamW. If it is
  not, something that should be sharded is not.

A sample table is deliberately **not** reproduced here. Any figures printed in
this document would be from one machine on one day, and quoting them is exactly
the mistake the banner warns against. Run the script on the hardware you care
about; the *invariants* — byte counts, collective counts, memory ratios — are
the parts asserted by `tests/performance/`, and those are reproducible.

---

## 8. Why the performance tests assert on volume, not time

A test that asserts "the overlapped version is faster" is a flake generator: on
a shared CI machine the wall-clock ordering of two similar workloads is not
reproducible, and the failure teaches nothing.

What *is* reproducible is:

- bytes moved,
- number of collectives,
- resident bytes per strategy,
- agreement between the analytical model and the measurement.

So `tests/performance/` asserts on those and merely *records* timings.
`test_timings_are_recorded_but_not_asserted_on` documents the choice in the
suite itself.

---

## 9. Tuning guide

| Symptom | Likely cause | Try |
|---|---|---|
| DDP: high wait time on the last bucket | it cannot overlap with anything | smaller `bucket_cap_mb`, or accept it |
| DDP: many `out_of_order_buckets` | bucket order does not match backward order | this is what PyTorch's rebuild-after-first-iteration fixes; not implemented |
| FSDP: memory did not drop | one unit, so `max_unit_numel == P` | lower `auto_wrap_min_num_params` |
| FSDP: too many small collectives | units too fine | raise `auto_wrap_min_num_params` |
| FSDP: high all-gather wait | no prefetching | fewer, larger units; or accept it |
| TP: communication dominates | tensor parallelism across nodes | keep TP inside a node; use FSDP across nodes |
| activation memory dominates | replicated norm/residual regions | enable sequence parallelism |
| step time dominated by the optimizer | replicated optimizer state | enable sharding |
