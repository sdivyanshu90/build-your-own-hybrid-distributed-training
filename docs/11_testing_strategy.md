# 11 — Testing strategy

> `tests/conftest.py` holds the harness; the suites live under
> `tests/{unit,distributed,integration,end_to_end,performance}`.

---

## 1. The principle

**A forward-pass comparison is not a correctness test.**

Three real bugs found while building this project all produced a *perfect*
forward pass and wrong gradients:

| Bug | forward error | gradient error |
|---|---|---|
| sequence-gather adjoint was `split` instead of `reduce_scatter` | `4.6e-07` ✅ | `1.1e-02` ❌ |
| partial LayerNorm gradients never summed | `4.6e-07` ✅ | `3.3e-02` ❌ |
| tensor returns racing the child's exit | — | intermittent crash, ~1 run in 5 |

So every equivalence test here compares **gradients and post-optimizer-step
weights**, not only outputs. Where a property is structural (a weight shard *is*
a slice of the reference weight) the assertion is for exact equality, because
anything else would indicate a bug rather than rounding.

---

## 2. The layers

| Suite | World size | What it proves | Runtime |
|---|---|---|---|
| `tests/unit` | 1 (no process group) | shape/offset arithmetic, config validation, manifest logic, schedules, RNG, CPU-offload transparency | ~25 s |
| `tests/distributed` | 2 and 4 | collectives, DDP, FSDP, TP, SP against single-process references | minutes |
| `tests/integration` | 2 and 4 | checkpoint write/read/reshard and every integrity failure | minutes |
| `tests/end_to_end` | 1, 2, 4, (8) | the eight required scenarios through the real engine | minutes |
| `tests/performance` | 2 and 4 | communication volume and memory scaling — never timings | minutes |

The unit layer is where the *hang-generating* logic lives. An off-by-one in a
shard offset manifests in a distributed run as a hang or as silently corrupted
weights; in a unit test it is a one-line assertion. That is why
`tests/unit/test_topology.py` sweeps every legal factorisation of 1–8 ranks
exhaustively.

**One suite reads the source instead of running it.**
`tests/unit/test_source_invariants.py` parses every module and asserts two
structural properties:

| Invariant | Why a runtime test cannot replace it |
|---|---|
| no `raise` inside `if is_primary:` | the bug only fires on the ranks you are *not* looking at, and only when that specific error condition occurs; a structural check covers every error path at once, including ones nobody has written a scenario for |
| no `dist.<collective>(...)` without `group=` | omitting the group does not raise or hang — it trains a subtly wrong model, so there is no failing behaviour for a runtime test to detect |

Both were added after the corresponding bug shipped. The first is dissected in
`08_distributed_checkpointing.md` §6: a duplicate-save check that raised on rank
0 alone left every other rank blocked in the following barrier until rank 0's
process died, reporting `Connection closed by peer` — an error naming a TCP
socket rather than the duplicate checkpoint. The tests are written to fail on
planted violations, not merely to pass on clean code.

---

## 3. Tolerances, and why each is what it is

| Comparison | Tolerance | Reason |
|---|---|---|
| same arithmetic, same order | `0` (bitwise) | anything else is a bug, not rounding |
| same maths, reduction order differs across ranks | `1e-6` | fp32 non-associativity over ~10³ terms is ~`1e-7`; 10× headroom |
| after several Adam steps | `1e-5` | the adaptive step size amplifies the above by `lr/(√v + ε)` |

These are named constants in `conftest.py` (`FLOAT32_REDUCTION_TOLERANCE`,
`OPTIMIZER_STEP_TOLERANCE`), so a test that needs a looser bound has to say so
explicitly rather than quietly widening it.

The bitwise cases are worth listing, because they are the strongest assertions
in the suite:

- a column-parallel layer's gathered output vs an unsharded `nn.Linear`: `0.0`
- a weight-shard gradient vs the reference's matching slice: `0.0`
- custom DDP vs PyTorch DDP at world size 2: `0.0`
- FSDP's `full_state_dict` vs the original parameters: `0.0`
- a resharded checkpoint's restored parameters vs the saved ones: `0.0`
- resumed vs uninterrupted training at the same world size: `0.0`

A tolerance is only used where a *cross-rank sum* genuinely reorders the
arithmetic.

---

## 4. The distributed harness

A distributed test needs three things a normal test does not.

**Process isolation.** `torch.distributed` keeps global state, and a process
group cannot be re-created with a different world size in the same interpreter.
Every distributed test therefore runs in fresh child processes.

**A timeout that produces a diagnosis.** A mismatched collective hangs forever.
`run_distributed` bounds the run and, on expiry, names the ranks that never
reported — precisely the set stuck in a collective.

**Per-rank tracebacks.** The interesting failure is often not on rank 0.
`WorkerFailure` carries every rank's traceback *in its message*, so pytest's
output contains the whole picture rather than "2 of 4 workers failed".

Workers must be **module-level** functions (`spawn` pickles them) returning
picklable values. Both are checked at submit time with a readable error rather
than an opaque pickling failure.

### Memoising expensive runs

Several tests assert *different properties of the same run*. The DDP
equivalence worker returns gradients, bucket layouts, statistics and a loss, and
there is one test for each. Re-spawning the ranks per assertion would multiply
the suite cost by the number of properties checked — backwards, since the
expensive part is the processes.

`run_distributed_cached` memoises by `(function, world_size, kwargs)`. It is
safe because the workers are pure functions returning plain data; tests must
treat the results as read-only.

The saving is not marginal. Spawning a rank costs a full `import torch`
(hundreds of megabytes resident, a second or more of wall clock), so on a
memory-tight machine the cache is the difference between a suite that finishes
and one that swaps.

### Resource guards

`requires_ranks(n)` skips a test when the machine cannot host `n` spawned
ranks, reading `MemAvailable` from `/proc/meminfo` and comparing against
`n × 600 MB`. The skip reason names the shortfall:

```
SKIPPED [1] needs 8 spawned ranks (~4800 MiB and 8 cores); this machine
reports 3381 MiB available and 8 core(s)
```

A test that starts swapping does not fail — it takes twenty minutes and then
usually passes, which is the worst outcome: the suite looks hung and nobody
learns anything.

**The guard runs at call time, not at collection time.** This looks like a
detail and is not. The obvious implementation is `pytest.mark.skipif(...)`, but
a marker's condition is evaluated once, while pytest imports the module. At
that moment the machine is idle and has plenty of memory, so the guard admits
the eight-rank test. Twenty minutes later, after earlier tests have filled
memory and swap, that test finally runs — and gets OOM-killed, taking the whole
pytest process with it (`exit=137`) and leaving orphaned rank processes holding
several gigabytes.

That is not hypothetical; it is exactly how a full end-to-end run died on the
development machine, and no collection-time guard could have prevented it,
because at collection time the resources genuinely were there. `requires_ranks`
is therefore a decorator that calls `pytest.skip()` from inside the wrapper,
measuring the machine at the moment the ranks are about to be spawned.

The general lesson is worth stating: **a resource check is only meaningful at
the instant the resource is claimed.** Anything earlier is a prediction.

**Budget the peak, not the import.** The first version of the guard used 400 MB
per rank, reasoning from `import torch` (~320 MB resident). That is the wrong
quantity. A rank that is *training* also holds parameters, gradients, optimizer
state, activations and a communication buffer, and was measured at **~550 MB**.
Budgeting the import figure is precisely how eight ranks were admitted onto a
machine that could not hold them.

The asymmetry matters: a guard that is too strict skips a test and says so,
which is a visible, recoverable outcome. A guard that is too generous does not
degrade — the OOM killer takes the pytest process down mid-run and leaves
orphaned ranks holding gigabytes. The figure is therefore 600 MB: the observed
peak plus margin, deliberately erring towards the recoverable failure.

### Running the suite

pytest already runs tests serially inside one process, so the rule is not about
ordering — it is that **two pytest processes must not run distributed tests at
the same time**. Each spawned rank costs a full `import torch` plus its
training state (~550 MB resident), so concurrent runs on a memory-constrained machine swap, and a
five-minute suite takes twenty and looks like a hang. `pytest-xdist` is wrong
here for the same reason.

`scripts/run_tests_distributed.sh` runs one file per pytest invocation. That is
for *isolation and reporting* — a suite that leaves a wedged process group
behind cannot affect the next one, and each file gets its own pass/fail line —
and it never runs two invocations concurrently.

---

## 5. Negative tests

Roughly a third of the suite asserts on *failures*. Every documented error path
has a test:

| Category | Examples |
|---|---|
| configuration | unknown keys, impossible topologies, Gloo-on-CUDA, bf16 loss scaling |
| bring-up | double initialisation, use after shutdown, partial launch environment |
| collectives | non-contiguous tensors, indivisible reduce-scatter, wrong group |
| DDP | structure mismatch across ranks, unused parameters, missing boundary, double backward |
| FSDP | mixed `requires_grad`, cross-unit tied weights, oversized units |
| TP/SP | indivisible features/heads/sequences, tied embeddings with vocab parallel |
| checkpoint | missing file, truncation, bit flip, incomplete manifest, version mismatch, hostile filename, TP-width change |

`expect_distributed_failure` runs a worker that is *expected* to fail and
returns the outcomes, so the test can assert on the message rather than on the
harness turning it into a test failure.

The checkpoint integrity tests deliberately damage a **real** checkpoint on
disk rather than constructing a synthetic manifest, so they exercise the same
code path an operator with a failing disk would hit.

---

## 6. The eight required end-to-end scenarios

| # | Scenario | Test | Asserts |
|---|---|---|---|
| 1 | single-process baseline | `TestSingleProcess` | loss decreases; distributed code path used at `W=1` |
| 2 | two-rank custom DDP | `TestStrategyEquivalence` | final weights and every step's loss match the baseline |
| 3 | two-rank FSDP | `TestStrategyEquivalence` | same, plus local parameters < global |
| 4 | two-rank tensor parallel | `TestTransformerStrategies` | logits and weights match; parameters partitioned |
| 5 | two-rank sequence parallel | `TestTransformerStrategies` | matches plain TP to `1e-5` |
| 6 | four-rank hybrid | `TestHybrid` | matches the baseline; `metric_group` spans dp×shard |
| 7 | checkpoint save/resume | `TestCheckpointResume` | resumed losses **identical** to the uninterrupted tail |
| 8 | checkpoint resharding | `TestCheckpointResume` | restored parameters bitwise identical; trajectory matches |

Every one asserts a numerical property. None passes merely because the
processes exited zero.

### The subtlety in scenario 8

Comparing post-resume training across world sizes is only meaningful if the
**global batch is held constant**. Changing the rank count while keeping
`micro_batch_size` fixed changes the global batch (16 vs 8), so the two runs
solve different optimisation problems and diverge — not because resharding was
wrong.

The first version of `examples/resume_with_different_world_size.py` made exactly
this mistake and reported a `1.2e-02` difference. The test and the example both
scale `micro_batch_size` inversely with the rank count now, and both say why.

---

## 7. Determinism

Tests would otherwise be order-dependent, which is the hardest kind of flake to
diagnose. Three measures:

- an autouse fixture seeds and restores the global RNGs around every test;
- `training.deterministic: true` enables `torch.use_deterministic_algorithms`;
- dropout is `0` in every test model, so no RNG stream has to be matched across
  ranks for the comparisons to hold.

Dimensions are fixed, not randomised. Randomised shapes produce failures that
cannot be reproduced from the test name alone. Where variety matters — padding,
uneven shards — the values are chosen *deliberately*: the FSDP test model has
1857 parameters precisely because that is divisible by neither 2 nor 4, so
padding is exercised at every world size.

---

## 8. GPU tests

Marked `cuda` (needs one device) or `multigpu` (needs two for NCCL collectives),
and skipped with a reason naming the device count found:

```
SKIPPED [1] requires >= 2 CUDA devices for NCCL collectives (found 1)
```

CPU/Gloo coverage is complete: every correctness claim in this project is
verified without a GPU. CI does not require any GPU, and the multi-GPU workflow
is separate and manual.

---

## 9. Coverage

`make coverage` reports line and branch coverage for the non-GPU suite.

**The reported number under-counts.** Distributed code runs in *child*
processes, and this project does not merge child coverage data, so the paths
exercised only inside a spawned worker — most of `collectives.py`, `ddp.py`,
`fsdp.py` — appear less covered than they are. That is a known limitation
stated here rather than papered over with a misleading badge.

Exclusions are narrow and justified:

- `distributed/launch.py` — a process/argv shim whose failure modes are the
  timeouts the suite already exercises;
- `if TYPE_CHECKING:` blocks;
- `pragma: no cover` on genuinely unreachable defensive branches, each with a
  comment saying why it is unreachable.

Coverage is not chased by testing trivial getters. A property that returns a
stored value is covered incidentally or not at all.

---

## 10. Running

```bash
pytest -q                                  # everything
pytest -q -m "not cuda and not multigpu"   # what CI runs
pytest -q tests/unit                       # fast, ~10 s
./scripts/run_tests_distributed.sh         # one file at a time
./scripts/run_tests_distributed.sh unit
TIMEOUT=600 ./scripts/run_tests_distributed.sh distributed
```

Timeouts come from two places: `run_distributed`'s own bound (per test) and
`pytest-timeout` (per test, belt and braces). Both exist so that a hang fails
one test with a diagnosis rather than wedging the run.

---

## 11. What is not tested

Stated plainly rather than implied by absence:

- **Multi-node.** Everything is single-node, multi-process. The rendezvous code
  is the same, but no test crosses a network.
- **NCCL correctness.** The NCCL path is exercised only by the CUDA-marked
  tests, which need two GPUs to run at all.
- **Failure injection mid-collective.** No test kills a rank halfway through an
  all-reduce; the recovery story for that is "the timeout fires".
- **Very large models.** The models are small by design, so nothing here
  validates behaviour at a scale where the constants change.
- **Long-run convergence.** Tests run tens of steps. That they train correctly
  for tens of thousands is not established.
