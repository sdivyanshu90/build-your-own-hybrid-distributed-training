# 02 — Collective communication

> Implemented by `distributed/collectives.py` and `autograd/collectives.py`.
> Tested by `tests/distributed/test_collectives.py`.

---

## 1. The operations

With `G` ranks in a group and a payload of `N` elements per rank:

| Operation | Result on rank `r` | Bytes each rank sends (ring) |
|---|---|---|
| `broadcast` | `x` from the source | `≈ N` |
| `all_reduce` | `Σ_q x_q` (all ranks) | `2N(G−1)/G` |
| `reduce_scatter` | slice `r` of `Σ_q x_q` | `N(G−1)/G` |
| `all_gather` | `concat_q(x_q)` | `N(G−1)/G` |
| `all_to_all` | `concat_q(x_q[r])` | `N(G−1)/G` |

The identity that drives the whole design of sequence parallelism:

```
all_reduce  ≡  reduce_scatter  +  all_gather
```

An all-reduce moves exactly twice what either half moves. So replacing an
all-reduce by a reduce-scatter *and* an all-gather costs nothing — and if the
region between them can work on the scattered form, it saves a factor of `G` in
activation memory for free. That is sequence parallelism in one line.

---

## 2. Averaging: divide first, then sum

Gloo does not implement `ReduceOp.AVG`:

```
RuntimeError: Cannot use ReduceOp.AVG with Gloo
```

Every average in this project is therefore expressed as **divide locally, then
sum**:

```python
x_local /= group_size
all_reduce(x_local, SUM)
```

rather than sum-then-divide. Three reasons:

1. It is backend-agnostic.
2. It is what PyTorch's own `Reducer` does, which keeps the numerical
   comparison against `torch.nn.parallel.DistributedDataParallel` tight — the
   equivalence test measures `0.0` at world size 2.
3. An **asynchronous** all-reduce needs no completion callback: the result is
   correct the moment the handle is waited on. Sum-then-divide would require
   scheduling a division after every wait.

`ReduceOp.AVG` in this codebase is a *convention*, not a backend op — it
pre-scales and then issues `SUM`.

---

## 3. Explicit groups, no exceptions

Every wrapper takes a `GroupHandle` as a **required** argument. There is no
`group=None` default anywhere:

```python
def all_reduce(tensor, group: GroupHandle, *, op=ReduceOp.SUM, ...) -> AsyncWork
```

In a hybrid job, reducing gradients over the world instead of the data-parallel
group does not raise — it produces training that converges to the wrong thing.
Making the group mandatory converts that class of bug into a `TypeError` at
import time.

Every wrapper also calls `validate_group_membership` first. Issuing a
collective on a group you are not in is undefined behaviour in every backend —
typically a segfault or a hang — and the check costs a tuple membership test.

---

## 4. Trivial groups

A group of size one makes every collective the identity map, and the wrappers
short-circuit those cases:

```python
if group.is_trivial:
    return AsyncWork(None)          # all_reduce
    out.copy_(tensor); ...          # all_gather / reduce_scatter
```

This is not a correctness fallback. It is the mathematically exact result, and
it is what lets the *same* code run at world size 1 — which is what makes the
single-process reference in the equivalence tests worth anything.

`GroupHandle.trivial()` builds a one-member handle with `process_group=None`,
usable with no distributed runtime at all. The tensor-parallel layers are
written once against it and run unmodified as both the reference and the
distributed implementation.

---

## 5. Buffer-shaped collectives

Two of the wrappers use the "into tensor" variants rather than the list-based
ones:

- `all_gather_into_tensor(out, x)` writes directly into one pre-sized buffer.
- `reduce_scatter_tensor(out, x)` reads directly from one contiguous buffer.

The list-based `all_gather(list_of_tensors, x)` allocates one tensor per rank
and then requires a `torch.cat`. For FSDP, where this is the hot path, that is
an extra allocation and an extra copy per unit per step.

The price is that every rank's contribution must be **the same size**, which is
why every flat buffer in this project is padded to a multiple of the group
size. See `04_fsdp_style_sharding.md` §3.

Both wrappers validate:

```python
>>> reduce_scatter_tensor(torch.ones(5), group_of_2)
CollectiveError: [rank 0] collectives.reduce_scatter_tensor: the leading dimension
must be divisible by the group size; pad the buffer before reducing rather than
letting ranks disagree about shapes; expected 'shape[0] % 2 == 0', observed 1.
Fix: pad the flat buffer up to a multiple of the group size
```

Non-contiguous tensors are also refused: some backends silently copy them and
others reject them, so the wrapper makes the requirement explicit rather than
backend-dependent.

---

## 6. Gathering along a dimension other than 0

`all_gather_into_tensor` and `reduce_scatter_tensor` concatenate along
dimension 0 only. To operate on dimension `d`:

```python
moved = tensor.movedim(d, 0).contiguous()   # (B, S, H) -> (S, B, H) for d=1
gathered = all_gather(moved)                # (G·S, B, H)
result = gathered.movedim(0, d).contiguous()  # (B, G·S, H)
```

The block written by rank `r` lands at positions `[r·S, (r+1)·S)` along `d`,
which is exactly concatenation in rank order.

The transposes cost a copy. The alternative — a list-based gather plus
`torch.cat` — costs a copy *and* an allocation per rank, so this is the cheaper
of two imperfect options. A production implementation would keep the sequence
dimension leading throughout the model to avoid both.

---

## 7. Differentiable collectives and their adjoints

A collective inside a model is a function, and autograd needs its adjoint. The
five that matter:

| Name | Forward | Backward |
|---|---|---|
| `copy_to_group` (Megatron's `f`) | `Y_r = X` (identity) | `X̄ = Σ_r ḡ_r` (all-reduce) |
| `reduce_from_group` (Megatron's `g`) | `Y = Σ_r X_r` (all-reduce) | `X̄_r = ḡ` (identity) |
| `gather_from_group` | `Y = [X_0 … X_{G−1}]` | `X̄_r = ḡ[r]` (split) |
| `scatter_to_group` | `Y_r = X[r]` (local split) | `X̄ = [ḡ_0 … ḡ_{G−1}]` (all-gather) |
| `reduce_scatter_to_group` | `Y_r = (Σ_q X_q)[r]` | `X̄_q = [ḡ_0 … ḡ_{G−1}]` (all-gather) |

### Why `copy_to_group`'s adjoint is an all-reduce

A replicated activation `X` feeds a *different* weight slice on each rank.
Formally there are `G` copies `Y_r = X`, and the loss depends on all of them:

```
∂L/∂X = Σ_r (∂L/∂Y_r)(∂Y_r/∂X) = Σ_r ḡ_r
```

Each rank computes only its own `ḡ_r`, so producing the true `∂L/∂X` requires
summing across the group. Omitting it is the classic tensor-parallel bug: the
model still trains, but every layer below the split learns from a gradient that
is missing the other ranks' contributions entirely.

### Why `reduce_from_group`'s adjoint is the identity

Here `Y = Σ_r X_r` with `Y` replicated, so `∂Y/∂X_r = I` and `X̄_r = ḡ`. The
forward all-reduce already made `ḡ` identical on every rank, so nothing needs
to move.

### The sixth one: `gather_from_sequence_parallel_region`

This is the subtlest point in the whole implementation, and getting it wrong
produces a forward pass that is numerically perfect and gradients that are
quietly incomplete.

Two operations both gather shards into a full tensor `Y`. They differ in how
`Y` is *consumed*, and therefore in what `∂L/∂Y` means:

**Feature gather** — `gather_from_group`, at the *end* of a tensor-parallel
region. `Y` is a replicated activation: every rank performs the *same*
computation with it, redundantly computing the same `∂L/∂Y`. That value is
already the true total, so the adjoint is "take my slice".

**Sequence gather** — `gather_from_sequence_parallel_region`, at the
*entrance* to a tensor-parallel region. `Y` is consumed **differently** on each
rank: rank `t` multiplies it by *its* weight slice, i.e. computes with
different attention heads. Each rank therefore holds only a *partial*
`∂L/∂Y`, and the true total is the sum over ranks:

```
∂L/∂X_r = [ Σ_q (∂L/∂Y)|_q ]_slice r
```

Summing and then slicing is exactly `reduce_scatter`.

This project originally used the split adjoint in both places. The
forward pass matched the single-process reference to `4.6e-07`; the LayerNorm
gradients were wrong by `3.3e-02`. The test that catches it is
`tests/distributed/test_tensor_parallel.py::TestTransformerEquivalence::test_replicated_parameter_gradients_match`.

---

## 8. What autograd cannot do for you

Some gradients are partial and **no** collective in the graph fixes them.

Under sequence parallelism a LayerNorm gain is *replicated* across the group but
is only ever applied to the positions this rank holds. Its gradient is a
partial sum over positions:

```
∂L/∂γ = Σ_{s=0}^{S−1} g_s = Σ_{r=0}^{G−1} [ Σ_{s ∈ P_r} g_s ]
                                            └── what rank r computes ──┘
```

Nothing performs the outer sum: the parameter is a leaf, and no collective sits
between it and the loss. It has to be done explicitly after backward, by
`all_reduce_sequence_parallel_gradients`. Megatron-LM does the same thing in
`finalize_model_grads`; it is unavoidable in any sequence-parallel
implementation.

*Without* sequence parallelism the same parameter is applied to the whole
sequence redundantly on every rank, so its gradient is already complete and
summing would multiply it by `G`. That is why the affected parameters carry an
explicit marker (`mark_sequence_parallel_partial`) rather than there being a
blanket rule.

---

## 9. Instrumentation

`CommunicationRecorder` tallies calls, bytes, launch time and wait time, keyed
by `operation/group`. Recorders are **owned objects**, not globals: DDP has
one, each FSDP unit shares one, the benchmark creates its own. Passing `None`
disables instrumentation with a single `is None` test in the hot path.

Launch time and wait time are recorded separately, because for an asynchronous
collective the launch is nearly free and the wait is where the cost shows up —
and a wait time near zero is the signal that overlap is working.

```
communication summary:
  all_gather/shard          calls=24     MiB=    0.244  launch_s=  0.0021 wait_s=  0.0009
  reduce_scatter/shard      calls=12     MiB=    0.122  launch_s=  0.0014 wait_s=  0.0006
```

---

## 10. Collective consistency checks

Two helpers turn "a precondition that can only be checked collectively" into an
error that fires on **every** rank:

- `assert_metadata_consistent(payload, group, name=…)` — gathers a picklable
  object per rank and compares. Used for parameter name/shape/dtype lists,
  bucket layouts and flat-parameter layouts.
- `assert_tensor_consistent(tensor, group, name=…)` — broadcasts the source's
  copy and compares locally, reporting the maximum observed difference.

The point is coordination. A check that only one rank can perform is a hang
generator: the detecting rank raises and stops calling collectives while its
peers block forever. These fire everywhere, so the job dies with a diagnosis.

```
ParameterConsistencyError: [rank 1] ddp.verify_parameter_structure: the
(name, shape, dtype) list of trainable parameters differs across the
'data_parallel' group; ranks in this group must agree or the collectives that
follow will mismatch; expected "group-local rank 0 (global 0): [('blocks.0...',
(24, 12), 'torch.float32'), ...]", observed "group-local rank 1 (global 1):
[('blocks.0...', (32, 12), 'torch.float32'), ...]". Fix: make the model
construction deterministic and identical on all ranks of this group
```

---

## 11. Gloo capability notes

Measured on PyTorch 2.3 (`tests/distributed/test_collectives.py` covers all of
these):

| Operation | Gloo | Note |
|---|---|---|
| `all_reduce` | ✅ | |
| `broadcast` | ✅ | |
| `all_gather_into_tensor` | ✅ | |
| `reduce_scatter_tensor` | ✅ | supported in 2.3, unlike older documentation suggests |
| `all_to_all_single` | ✅ | |
| `send`/`recv` | ✅ | blocking; needs an asymmetric order to avoid deadlock |
| `ReduceOp.AVG` | ❌ | hence divide-then-sum |

The point-to-point test uses an even/odd send-first ordering. A symmetric
"everyone sends then everyone receives" deadlocks with blocking primitives
once the payload exceeds the socket buffer.

---

## 12. Comparison with PyTorch

| Concern | PyTorch | This project |
|---|---|---|
| Group argument | optional, defaults to world | mandatory |
| Averaging | `ReduceOp.AVG` on NCCL, manual elsewhere | always divide-then-sum |
| Membership check | none (UB if wrong) | validated on every call |
| Instrumentation | external profilers | built-in per-operation recorder |
| Differentiable collectives | in `torch.distributed.nn` / DTensor | explicit `Function`s with documented adjoints |
