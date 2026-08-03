# 13 — Deliverables, matrices and acceptance criteria

This document is the audit trail: what was built, what was tested, what was
actually executed, and where each requirement is satisfied.

---

## 1. Architecture summary

Five layers, each depending only on the ones above it.

```
config / errors / logging          frozen dataclasses, exception hierarchy,
                                   rank-aware structured logging
        │
distributed/                       topology (pure arithmetic) → groups (the only
                                   caller of new_group) → context (owns the PG,
                                   device, registry) → collectives (explicit-group
                                   wrappers) → launch (multi-rank test launcher)
        │
autograd/                          differentiable collectives; each Function
                                   documents its adjoint
        │
parallel/ + optim/                 ddp, fsdp, tensor_parallel, sequence_parallel,
                                   hybrid; sharded optimizer, distributed norm,
                                   loss scaling
        │
models/ + training/ + checkpoint/  reference models, one training engine for every
                                   strategy, manifest-based checkpoints
```

**The four-dimensional rank grid.** A rank is a mixed-radix number over
`(data_parallel, shard, sequence, tensor)`, most-significant first, so the
highest-traffic dimension (`tensor`) occupies contiguous ranks and the
lowest-traffic one (`data_parallel`) is outermost. Everything else — group
membership, who writes a checkpoint, which group a metric is reduced over —
falls out of that arithmetic.

**Three structural decisions carry most of the correctness weight.**

1. *No collective wrapper accepts a default group.* Reducing over the wrong
   group produces training that converges to the wrong thing without raising;
   making the group mandatory turns that into a `TypeError` at import time.
2. *World size 1 runs the distributed code path.* A real one-member process
   group is created and every collective executes as an identity, so the
   single-process reference used by the equivalence tests exercises the same
   code as the distributed run.
3. *Collectives are issued in a fixed order.* Group construction walks one
   ordered registry; DDP launches buckets by index, not by readiness. Both
   remove hang-by-divergent-ordering by construction.

---

## 2. Feature-completeness matrix

| Requirement | Status | Where | Verified by |
|---|---|---|---|
| **Distributed runtime** | | | |
| init/destroy process groups safely | ✅ | `distributed/context.py` | `test_collectives.py::TestProcessGroups` |
| read `torchrun` env vars | ✅ | `LaunchEnvironment.from_environment` | `TestNegativeCases` |
| explicit init params for tests | ✅ | `init_distributed(rank=, world_size=)` | used by every distributed test |
| expose rank/local rank/world/backend/device/groups | ✅ | `DistributedContext` properties | `test_backend_and_device_resolution` |
| set CUDA device per local rank | ✅ | before `init_process_group` | `test_cuda.py::test_single_process_context_selects_cuda` |
| detect invalid launch configurations | ✅ | `_resolve_backend_and_device` | `test_cuda.py::test_nccl_refuses_to_share_a_device` |
| avoid double initialisation | ✅ | `_ACTIVE_CONTEXT` guard | `test_double_initialisation_rejected` |
| context-manager cleanup | ✅ | `distributed_context()` | `test_context_after_shutdown_rejected` |
| rank-aware barriers | ✅ | `ctx.barrier(name, label=)` | `test_barriers_do_not_deadlock` |
| timeout configuration | ✅ | `timeout_seconds` → `init_process_group` | — |
| deterministic group creation | ✅ | `GROUP_CREATION_ORDER` + sorted lists | `test_every_rank_builds_groups_in_the_same_order` |
| **Topology** | | | |
| 4 dimensions, validated product | ✅ | `TopologyConfig`, `ParallelTopology` | `test_topology.py` (36 tests) |
| coordinates, group ranks, local rank, neighbours | ✅ | `ParallelTopology` | same |
| TP/SP shared-group mode | ✅ | `sequence_parallel_mode="tensor_group"` | `test_sequence_effective_alias` |
| **DDP** | | | |
| replicate; broadcast params and buffers | ✅ | `_broadcast_initial_state` | `TestParameterSynchronisation` |
| gradient hooks | ✅ | `register_post_accumulate_grad_hook` | `TestGradientEquivalence` |
| average or sum consistently | ✅ | `average_gradients` | `test_matches_single_process_reference` |
| sync before optimizer step | ✅ | `finish_gradient_synchronization` | `test_missing_synchronisation_boundary_is_detected` |
| parameters without gradients | ✅ | `find_unused_parameters` | `test_unused_parameter_*` |
| `no_sync()` accumulation | ✅ | `DDP.no_sync` | `TestGradientAccumulation` |
| configurable bucket sizes | ✅ | `bucket_cap_mb` | `test_bucket_size_does_not_change_the_result` |
| flatten gradients into buckets | ✅ | `GradientBucket` | `TestDDPBuckets` |
| async all-reduce when ready | ✅ | `_launch_ready_buckets` | `test_buckets_are_launched_in_index_order` |
| wait before optimizer updates | ✅ | boundary + `forward` guard | as above |
| reconstruct gradient views | ✅ | `param.grad = bucket.view_for(i)` | `TestGradientEquivalence` |
| preserve dtypes | ✅ | buckets keyed by `(dtype, device)` | `TestDDPBuckets` |
| validate parameter ordering across ranks | ✅ | `_verify_parameter_structure` | `test_inconsistent_model_structure_is_detected` |
| detect shape/dtype mismatch early | ✅ | same | same |
| buffer broadcast behaviour | ✅ | `broadcast_buffers` | `test_buffers_are_broadcast_when_enabled` |
| communication statistics | ✅ | `DDPStatistics`, `CommunicationRecorder` | `TestInstrumentation` |
| numerically matches single process | ✅ | `1.5e-08` | `test_matches_single_process_reference` |
| numerically matches PyTorch DDP | ✅ | `0.0` at `W=2` | `test_matches_pytorch_ddp` |
| **FSDP** | | | |
| shard parameters / gradients / optimizer state | ✅ | `FlatParamHandle` | `TestShardConstruction`, `test_optimizer_state_is_sharded` |
| store only the local shard when idle | ✅ | storage `resize_(0)` | `test_unit_is_resharded_when_idle` |
| all-gather before forward, bind views | ✅ | `unshard`, `bind_views` | `TestTrainingEquivalence` |
| reshard per policy | ✅ | `reshard_after_forward` | `test_reshard_after_forward_does_not_change_results` |
| reconstruct params during backward | ✅ | `_PreBackwardUnshard` + `refill` | same |
| flatten, reduce-scatter, keep local shard | ✅ | `_AllGatherFlatParam.backward` | `TestGradientSharding` |
| update local shard with local state | ✅ | flat shard *is* the optimizer param | `test_optimizer_state_is_sharded` |
| parameter flattening | ✅ | `build_flat_layout` | `test_tensor_utils.py::TestFlatLayout` |
| alignment / padding | ✅ | `even_shard_ranges` | `test_padding_is_present_and_accounted_for` |
| reconstruction metadata | ✅ | `FlatEntry`, `PieceLayout` | `TestStateDictPaths` |
| shard ownership | ✅ | `local_shard_range`, `local_pieces` | `test_sharded_pieces_cover_every_parameter_once` |
| full materialisation | ✅ | `full_tensors`, `full_state_dict` | `test_full_state_dict_reconstructs_the_model`. Undoes *FSDP* sharding; under tensor parallelism it returns this rank's slices by design (a TP slice is a different tensor per rank) — see §11 |
| safe view replacement/restoration | ✅ | `bind_views` / `unbind_views` | `test_summon_materialises_and_frees` |
| optional CPU offload | ✅ | `cpu_offload_params`, `cpu_offload_state` | `TestCpuOffload` (plumbing, world size 1); the transfer itself is CUDA-only, see §5 |
| mixed-precision configuration | ✅ | `MixedPrecisionConfig` | `test_cuda.py::test_mixed_precision_dtypes_on_nccl` |
| distributed global-norm clipping | ✅ | `ShardedOptimizer.clip_grad_norm` | `test_gradient_norms_agree_across_ranks` |
| `summon_full_params()` | ✅ | `FSDP.summon_full_params` | `test_summon_materialises_and_frees` |
| wrapping is attribute-transparent | ✅ | `FSDP.__getattr__` forwards to the wrapped module | `TestWrapperTransparency` — auto-wrap rewrites the caller's module tree, so `blocks[0].linear.weight` must keep resolving |
| full and sharded state dicts | ✅ | both | `TestStateDictPaths` |
| nested wrapping | ✅ | `auto_wrap_min_num_params` | `test_auto_wrapping_creates_nested_units` |
| tied-weight limitations | ✅ | rejected across units, supported within | `test_tied_parameters_across_units_rejected` |
| empty/uneven shard edge cases | ✅ | equal padded ranges | `test_shard_smaller_than_group_gives_every_rank_something` |
| memory accounting | ✅ | `memory_summary`, `estimate_training_memory` | `TestMemoryScaling` |
| **Tensor parallel** | | | |
| column-parallel linear | ✅ | `ColumnParallelLinear` | `TestColumnParallel` (7 tests) |
| local and gathered output modes | ✅ | `gather_output` | `test_sharded_output_mode` |
| sharded bias | ✅ | partitioned with the output | `test_bias_is_partitioned` |
| autograd-correct input gradients | ✅ | `copy_to_group` | `test_input_gradients_are_all_reduced` |
| row-parallel linear | ✅ | `RowParallelLinear` | `TestRowParallel` |
| sharded and auto-splitting input | ✅ | `input_is_parallel` | `test_both_input_modes` |
| output all-reduce, replicated bias | ✅ | bias added after reduction | `test_bias_is_replicated_and_added_after_the_reduction` |
| transformer MLP composition | ✅ | `TensorParallelFeedForward` | `TestTransformerEquivalence` |
| divisibility validation | ✅ | `_validate_divisible` | `test_indivisible_features_rejected` |
| equivalence vs `nn.Linear` | ✅ | `0.0` forward and weight grads | `TestColumnParallel`, `TestRowParallel` |
| multiple dtypes | ✅ | fp32 and fp64 | `TestDtypes` |
| with and without bias | ✅ | parametrised | both classes |
| **Sequence parallel** | | | |
| scatter / gather / reduce-scatter sequence | ✅ | `sequence_parallel.py` | `TestSequenceParallelOperations` |
| autograd-compatible all-gather | ✅ | `gather_from_sequence_parallel_region` | `test_replicated_parameter_gradients_match` |
| padding metadata for uneven lengths | ✅ | `SequenceShardInfo` | `TestSequenceShardInfo` |
| transformer-block demonstration | ✅ | `TransformerBlock` | `TestTransformerEquivalence` |
| elementwise ops local on shards | ✅ | LayerNorm, activations, residuals | same |
| LayerNorm path correct | ✅ | `SequenceParallelLayerNorm` + partial-grad reduction | `test_replicated_parameter_gradients_match` |
| gather only when required | ✅ | at the TP-region entrance | documented in `06_*` |
| sharded/gathered output modes | ✅ | model gathers before the head | `test_shorter_sequences_still_match` |
| distinguish SP/DP/CP/TP/PP | ✅ | table in `06_*` §1 | — |
| one correct attention strategy | ✅ | gather-then-attend, cost documented | `TestTransformerEquivalence` |
| **Hybrid** | | | |
| DDP + TP | ✅ | `hybrid_tensor` | `test_data_parallel_plus_tensor_parallel` |
| FSDP + TP | ✅ | `hybrid_full` | `test_full_hybrid_over_eight_ranks` |
| TP + SP | ✅ | fused schedule | `TestTransformerStrategies` |
| DP + shard + TP | ✅ | `hybrid_full` | `test_full_hybrid_over_eight_ranks` |
| full hybrid demonstration | ✅ | `examples/train_hybrid.py`, `configs/hybrid_4gpu.yaml` (dp×shard) and `configs/hybrid_8gpu.yaml` (dp×shard×tensor×sequence) | same |
| document ownership and ordering | ✅ | `describe_parallel_plan`, `07_*` | `test_hybrid_metric_group_spans_the_data_dimensions` |
| prevent wrong-group reduction | ✅ | mandatory group arguments | structural |
| 8-rank worked example | ✅ | `07_*` §4 | `test_documented_eight_rank_example` |
| **Checkpointing** | | | |
| model / optimizer / scaler / scheduler / step / epoch / RNG / topology / metadata / config / version | ✅ | `writer.py`, `manifest.py` | `TestMetadataAndInspection` |
| manifest-based directory format | ✅ | `manifest.json` + `rank-*.pt` | `test_manifest_describes_global_tensors` |
| offsets, lengths, padding, owners, checksums, completeness | ✅ | `ShardRecord`, `FileRecord` | `test_checkpoint_format.py` |
| temporary dir, per-rank writes, sync, verify, atomic rename | ✅ | `save_checkpoint` | `TestIntegrity` |
| incomplete / corrupt detection | ✅ | 10 distinct failure modes | `TestIntegrity` (11 tests) |
| load same topology | ✅ | `load_checkpoint` | `TestSameTopologyResume` |
| full state dict reconstruction | ✅ | `full_state_dict` | `test_full_state_dict_reconstructs_the_model` |
| reshard across sharding world size | ✅ | interval intersection | `test_fsdp_width_change` |
| change replication degree | ✅ | supported | `test_hybrid_to_pure_sharding` |
| load TP shards into matching TP topology | ✅ | `#tpKofN` keys | writer/reader |
| reject unsupported transformations | ✅ | `CheckpointTopologyError` | `test_tensor_parallel_change_is_rejected` |
| reshard by global metadata, not rank ids | ✅ | `read_tensor_range` | `test_reader_touches_only_the_files_it_needs` |
| resume equals uninterrupted | ✅ | bitwise | `test_resume_reproduces_the_uninterrupted_run` |
| save errors raise on **every** rank, never on one | ✅ | collective verdict + `_publish` broadcast | `test_duplicate_save_rejected` (asserts all ranks), `unit/test_source_invariants.py` |
| **Engine and models** | | | |
| model construction, wrapping, optimizer, scheduler | ✅ | `TrainingEngine` | `tests/end_to_end` |
| gradient accumulation, mixed precision, clipping | ✅ | same | same |
| periodic evaluation, checkpoint save/resume | ✅ | same | `TestCheckpointResume` |
| deterministic synthetic data | ✅ | `training/data.py` | `TestSyntheticData` |
| rank-aware logging | ✅ | `logging.py` | — |
| metrics reduced over the DP group | ✅ | `metric_group` = `dp_shard` | `test_metrics_are_reduced_over_the_data_group_only` |
| configurable seeds | ✅ | `derive_seed` streams | `TestReproducibility` |
| graceful cleanup | ✅ | `distributed_context`, `engine.close()` | — |
| MLP reference model | ✅ | `models/mlp.py` | `TestModels` |
| transformer with embedding/positional/LayerNorm/attention/FFN/residual/output | ✅ | `models/transformer.py` | same |
| TP and SP integrated into the transformer | ✅ | `ParallelPlan` | `TestTransformerEquivalence` |
| **Not implemented (documented non-goals)** | | | |
| pipeline parallelism | ❌ | — | `07_*` §12 |
| context (ring) attention | ❌ | — | `06_*` §8 |
| expert parallelism, optimizer quantisation | ❌ | — | `00_*` |
| custom CUDA kernels | ❌ | — | `10_*` §6 |
| elastic membership, multi-node fault tolerance | ❌ | — | `00_*` |

---

## 3. Test matrix

| Test group | World size | Backend | Hardware | Behaviour validated |
|---|---|---|---|---|
| `unit/test_topology.py` | 1 | none | any | rank↔coordinate bijection, group partitioning, deterministic enumeration, all validations |
| `unit/test_tensor_utils.py` | 1 | none | any | flatten/unflatten losslessness, padding, equal shard ranges, interval intersection, dim splitting |
| `unit/test_config.py` | 1 | none | any | every config rejection, YAML round-trip, shipped configs load and are self-consistent |
| `unit/test_checkpoint_format.py` | 1 | none | any | naming, path-traversal rejection, versioning, digests, manifest completeness/overlap |
| `unit/test_single_process_components.py` | 1 | none (trivial group) | any | DDP buckets, gradient-norm weighting, schedules, RNG, memory model, TP layers at width 1, data sampler |
| `unit/test_source_invariants.py` | 1 (parses source, runs none) | none | any | over `src/`, `examples/` and `scripts/`: no `raise` guarded by a rank check; no rank-guarded `return` above a collective; no `dist.*` collective without an explicit `group=` |
| `distributed/test_collectives.py` | 2, 4 | Gloo | CPU | group construction order, all-reduce/gather/reduce-scatter/all-to-all/p2p, subgroup isolation, consistency checks, negative cases, recorder |
| `distributed/test_ddp.py` | 2, 4 | Gloo | CPU | gradient equivalence vs single process **and** PyTorch DDP, bucket layout, initial broadcast, buffers, accumulation, optimizer equivalence, failure modes, strided gradients |
| `distributed/test_fsdp.py` | 2, 4 | Gloo | CPU | shard construction and padding, training equivalence, gradient sharding, state dicts, `summon_full_params`, `no_sync`, hybrid sharding, unsupported configurations |
| `distributed/test_tensor_parallel.py` | 2, 4 | Gloo | CPU | column/row/vocab-parallel exactness, sequence scatter/gather/reduce-scatter adjoints, transformer equivalence with and without SP, dtypes, divisibility |
| `distributed/test_cuda.py` | 1, 2 | NCCL | **1–2 GPUs** | device selection before NCCL init, memory statistics, loss-scaler overflow recovery, NCCL≡Gloo, mixed-precision dtypes, device-sharing refusal |
| `integration/test_checkpoint.py` | 2, 4 | Gloo | CPU | same-topology resume, resharding 4↔2, FSDP↔DDP, hybrid→pure, every integrity failure, metadata, retention |
| `end_to_end/test_training_scenarios.py` | 1, 2, 4, 8 | Gloo | CPU | the eight required scenarios, each with a numerical assertion |
| `performance/test_instrumentation.py` | 2, 4 | Gloo | CPU | communication volume per strategy, collective counts, memory scaling, analytical model agreement |

CUDA tests are marked `cuda` / `multigpu` and skip with a reason naming the
device count found. The 8-rank end-to-end test is marked `slow` and skips when
the machine cannot host 8 spawned ranks.

---

## 4. Commands

```bash
# install
python -m pip install -e ".[dev]"

# test
pytest -q                                   # everything
pytest -q -m "not cuda"                     # no GPU needed
pytest -q -m "not cuda and not multigpu"    # what CI runs
pytest -q tests/unit                        # ~10 s
./scripts/run_tests_distributed.sh          # multi-process, one file at a time

# quality
ruff check src tests examples scripts
ruff format --check src tests examples scripts
mypy
make check                                  # all of the above plus the CPU suite

# train
torchrun --standalone --nproc-per-node=2 examples/train_ddp.py
torchrun --standalone --nproc-per-node=2 examples/train_fsdp.py
torchrun --standalone --nproc-per-node=2 examples/train_tensor_parallel.py
torchrun --standalone --nproc-per-node=2 examples/train_sequence_parallel.py
torchrun --standalone --nproc-per-node=4 examples/train_hybrid.py
torchrun --standalone --nproc-per-node=8 examples/train_hybrid.py \
    --config configs/hybrid_8gpu.yaml

# checkpoint
torchrun --standalone --nproc-per-node=2 examples/save_distributed_checkpoint.py
python examples/resume_with_different_world_size.py
python scripts/inspect_checkpoint.py /path/to/checkpoint-step-000100
python scripts/inspect_checkpoint.py CHECKPOINT --tensors --plan blocks.0.linear.weight

# benchmark
python scripts/benchmark.py --world-size 2 --strategies ddp fsdp
python scripts/benchmark.py --world-size 2 --model transformer --strategies tensor sequence
```

---

## 5. Known limitations

| Limitation | Consequence | Where discussed |
|---|---|---|
| Python floor is 3.10, not 3.11 | none functionally; a one-line change to raise | `00_*`, `pyproject.toml` |
| No pipeline parallelism | cannot partition by layer | `07_*` §12 |
| No context/ring attention | sequence cannot stay split through attention | `06_*` §8 |
| TP width cannot be resharded | must resume at the original width | `08_*` §5 |
| Single node only | no test crosses a network | `11_*` §11 |
| No meta-device init | the unsharded model is built before sharding | `04_*` §12 |
| Frozen parameters need their own unit | a flat parameter has one `requires_grad` | `04_*` §10 |
| Tied weights cannot span units | rejected explicitly | `04_*` §10 |
| Coverage under-reports | child-process coverage is not merged | `11_*` §9 |
| CPU offload's *transfer* is CUDA-only | on CPU the copies are no-ops, so only the plumbing is verified here; `TestCpuOffload` asserts it is numerically transparent, and the CUDA path is written but unrun | `04_*` §12 |

---

## 6. Performance limitations

All deliberate; each states what a production system does instead
(`10_*` §6).

| Limitation | Effect |
|---|---|
| no FSDP prefetching | all-gather latency exposed rather than hidden behind the previous unit |
| no fused kernels | several extra HBM round trips per transformer block |
| explicit attention (no FlashAttention) | `O(S²)` attention memory; bounds reachable sequence length |
| three q/k/v projections | three matmul launches instead of one |
| `movedim` copies in sequence collectives | two extra copies per sequence gather |
| Python bucket bookkeeping | per-parameter Python work each step |
| synchronous CPU offload | no transfer/compute overlap |
| no `torch.compile` / CUDA graphs | small-tensor regime unoptimised |

---

## 7. Security considerations

| Risk | Mitigation | Test |
|---|---|---|
| pickle code execution from a checkpoint | `torch.load(weights_only=True)` for all tensor payloads | `TestIntegrity` |
| manifest steering a read to an arbitrary path | filename validated against `^rank-\d{5,}\.pt$` | `test_hostile_filenames_rejected` |
| symlink escape from inside the directory | `resolve_inside()` resolves and asserts containment | `test_resolve_inside_rejects_symlink_escape` |
| silent corruption | SHA-256 per file plus a size check | `test_corrupted_bytes_detected` |
| truncation | size checked before hashing, reported as truncation | `test_truncated_file_detected` |
| type confusion from metadata | no type name from a checkpoint is imported or constructed | by construction |
| writes outside the target | the writer only creates files under the staging directory it made | by construction |
| shell injection in scripts | no script builds a shell command from input | by construction |

**Trust boundary.** `manifest.json` and `metadata.json` are inert JSON and safe
to parse from anywhere; `scripts/inspect_checkpoint.py` reads only those and
never calls `torch.load`. The `rank-*.pt` payloads must only be loaded from
sources you trust: `weights_only=True` is a real reduction in attack surface,
not a guarantee — a hostile payload can still force a large allocation or a
shape confusion.

---

## 8. Suggested future improvements

Ordered by value per unit of work.

1. **FSDP prefetching.** Issue unit `i+1`'s all-gather during unit `i`'s
   compute. The largest single performance win available.
2. **Meta-device initialisation.** Build the model on `meta` and materialise
   shard by shard, removing the unsharded peak at construction.
3. **Asynchronous checkpoint save.** Overlap the write with subsequent steps.
4. **Pipeline parallelism.** The one missing dimension; needs micro-batch
   scheduling (1F1B/interleaved) and bubble accounting.
5. **Context parallelism.** Ring or all-to-all attention; the primitives
   (`neighbour_ranks`, `all_to_all_tensor`, `send/recv`) already exist.
6. **Bucket rebuild after the first iteration**, as PyTorch's `Reducer` does,
   using observed autograd order.
7. **Fused q/k/v projection** with an interleaved-by-partition layout.
8. **Communication hooks** for gradient compression.
9. **Pluggable checkpoint storage** (object stores).
10. **Merged child-process coverage**, so the reported number reflects reality.

---

## 9. Acceptance-criteria mapping

| # | Criterion | Satisfied by | Evidence |
|---|---|---|---|
| 1 | every required source file contains working code | all 30 modules under `src/hybrid_training/` | the suite imports and exercises every one |
| 2 | no required feature is pseudocode | — | no `pass`, `TODO` or `NotImplementedError` in required paths; `grep` clean |
| 3 | custom DDP matches a reference | `parallel/ddp.py` | `1.5e-08` vs single process, `0.0` vs PyTorch DDP |
| 4 | FSDP genuinely shards params, grads, optimizer state | `parallel/fsdp.py`, `optim/sharded_optimizer.py` | `TestShardConstruction`, `TestGradientSharding`, `test_optimizer_state_is_sharded`, `TestMemoryScaling` |
| 5 | tensor-parallel layers correct forward and backward | `parallel/tensor_parallel.py` | `TestColumnParallel`, `TestRowParallel` — forward and weight grads exactly `0.0` |
| 6 | sequence-parallel operations autograd-correct | `parallel/sequence_parallel.py`, `autograd/collectives.py` | `TestSequenceParallelOperations`, `test_replicated_parameter_gradients_match` |
| 7 | hybrid groups explicitly constructed and used | `distributed/groups.py`, `parallel/hybrid.py` | `TestHybrid`, `test_documented_eight_rank_example` |
| 8 | checkpoint integrity + a resharding path | `checkpoint/` | `TestIntegrity` (11 modes), `TestResharding` (4↔2, FSDP↔DDP) |
| 9 | CPU distributed tests run with Gloo | all of `tests/distributed`, `integration`, `end_to_end`, `performance` | executed; see §10 |
| 10 | CUDA tests available and skipped without GPUs | `tests/distributed/test_cuda.py` | 9 tests, skip reason names the device count |
| 11 | end-to-end training and resume verify equivalence | `tests/end_to_end` | resumed losses **bitwise identical** to the uninterrupted tail |
| 12 | documentation explains algorithms and implementation | `docs/00`–`docs/13` | ~10 000 lines; every mechanism derived |
| 13 | documented commands correspond to real files/APIs | `README.md`, `docs/12_api_reference.md` | **11 of 12 runnable commands executed, all exit 0** (§12); the 12th (8-rank hybrid) exceeds this machine's memory and is recorded as not-run. Every API symbol exists. Running them found three defects the test suite could not reach — §11 items 7-9 |
| 14 | static analysis and type checking pass | `ruff`, `mypy` | `ruff check`, `ruff format --check` and `mypy` all clean over 38 source files. Three narrowly scoped exceptions, each documented where it lives: `TID252` (relative imports are deliberate inside the package) and `ARG001/ARG004` (launcher and `autograd.Function` signatures are fixed by their callers) in `pyproject.toml`; `# type: ignore[override]` on each `Function.backward`, because PyTorch's stub declares `backward(ctx, *grad_outputs) -> Any` and a *more precise* signature reads to mypy as narrowing |
| 15 | no placeholder implementations | — | criterion 2 |
| 16 | negative and failure-path tests | ~60 negative tests | `TestNegativeCases`, `TestFailureModes`, `TestIntegrity`, all config rejections |
| 17 | multi-process tests have timeouts and useful logs | `tests/conftest.py`, `distributed/launch.py` | `WorkerFailure` carries every rank's traceback in its message |
| 18 | educational code distinguished from industrial optimisations | every `docs/*` closes with a "Comparison with PyTorch/Megatron" section | `03_*`–`08_*` §12 |

---

## 10. Tests actually executed

Executed on the development machine: 8 cores, 5.9 GB RAM, PyTorch 2.3.0,
Python 3.10.12, Gloo on CPU. The single GPU present (2 GB) is not enough for
the multi-GPU tests, which skip.

| Suite | Result | Duration |
|---|---|---|
| `tests/unit` (6 files) | **377 passed, 1 skipped** | 14 s |
| `tests/distributed/test_collectives.py` | **30 passed** | 13 m 56 s |
| `tests/distributed/test_ddp.py` | **20 passed** | 5 m 43 s |
| `tests/distributed/test_fsdp.py` | **40 passed** | 6 m 06 s |
| `tests/distributed/test_tensor_parallel.py` | **52 passed** | 6 m 57 s |
| `tests/integration` | **38 passed** | 8 m 08 s |
| `tests/end_to_end` | **29 passed, 1 skipped** (8-rank scenario, resource guard) | 7 m 45 s |
| `tests/performance` | **19 passed** | 3 m 43 s |
| `tests/distributed/test_cuda.py` | **9 collected, all skipped** | — |
| `ruff check` | clean | 1 s |
| `ruff format --check` | clean (69 files) | 1 s |
| `mypy` | clean (38 source files) | 40 s |

**Every suite above was run to completion after the last fix**, not carried over
from an earlier round. The totals reconcile against collection exactly:

```
377 + 30 + 20 + 40 + 52 + 38 + 29 + 19  = 605 passed
  1 (source-invariant: context.py, by design)
+ 1 (8-rank end-to-end, resource guard)
+ 9 (CUDA / multi-GPU, no hardware)      =  11 skipped
                                          ---
                                           616 collected
```

616 is up from 485 before final validation. The additions are the source-invariant
checks (three invariants over every module in `src/`, `examples/` and `scripts/`),
the communication-recorder tests, the wrapper-transparency tests, and the
parametrisations added while fixing the items in §11.

**Honest notes on what that means.**

- The CPU/Gloo results are real: every correctness claim above was verified by
  a process that actually ran.
- The `multigpu` tests have **not** been executed — this machine has one small
  GPU. They are written, they collect, and they skip with a reason. Their
  assertions are statically reviewed, not runtime-verified.
- The 8-rank end-to-end test skips on this machine (it needs ~4.8 GB of
  spawned-rank memory against 3.4 GB available) and has been verified at 4
  ranks instead. The skip is decided when the test is *called*, not when it is
  collected — see §11 defect 6 for why that distinction cost a whole suite run.
- Distributed suites are slow here (minutes per file) because each spawned rank
  costs a full `import torch` on a memory-constrained box. This is an
  environment property, not a property of the code.
- Two tests have failed **once each** to a launcher timeout under swap, and
  both passed on isolated re-run. They are recorded rather than quietly
  re-run-until-green, because the distinction between "flaky environment" and
  "intermittent bug" is exactly what an audit trail is for.

  `test_ddp.py::TestOptimizerEquivalence::test_parameters_track_the_reference[4]`
  is the clearer of the two, and its shape is diagnostic: ranks 0, 1 and 2 all
  exited **0 after 27.8 s** while rank 3 was SIGTERM'd. A rank stuck in a
  collective cannot produce that pattern — its peers would be stuck with it.
  The machine reported 1983 MiB swapped at the moment of failure. Re-run alone
  with 4.8 GB free: 2 passed in 73 s.

- The other (`test_logits_match_the_single_process_model[True-4]`)
  failed during a full-file run taken while the machine was at load
  average 24 with 1.8 GB swapped. It passed on every isolated re-run and in the
  clean full-file run. The cause was the launcher timeout expiring under swap,
  not a numerical failure. `launch_workers` now prints the machine's load and
  free memory alongside every timeout so this is attributable at a glance
  instead of being investigated as a correctness bug; `docs/09` §2 step 0
  describes the procedure.

---

## 11. Defects found during final validation

Running the suites end to end — rather than assuming they would pass — found
nine real defects and four bad assertions. They are recorded here because
*which* bugs a test run catches is the most useful evidence about what the
tests are worth.

| # | Defect | Where | Class |
|---|---|---|---|
| 1 | duplicate-save check raised on rank 0 only | `checkpoint/writer.py` | **hang**: other ranks blocked in the next barrier until rank 0's process died, surfacing as `Connection closed by peer` |
| 2 | staging `mkdir` and the whole publish phase had the same shape | `checkpoint/writer.py` | **hang**, same mechanism |
| 3 | `LoadedCheckpoint.state` aliased the engine's live state | `training/engine.py` | **wrong value**: `loaded.step` reported the step training had reached, not the step resumed from |
| 4 | `calls` counted a collective's launch *and* its wait | `distributed/collectives.py` | **wrong metric**: asynchronous collectives reported 2x; DDP showed 6 all-reduces for a 3-step, 1-bucket run |
| 5 | `(x.items() if c else {}).items()` | `tests/end_to_end` | **crash**: took out the whole suite (2 failures + 5 fixture errors) |
| 6 | `requires_ranks` was a collection-time `skipif` | `tests/conftest.py` | **OOM**: the 8-rank test was admitted while the machine was idle and killed 20 minutes later, taking the pytest process with it |
| 7 | `if not is_primary: return` sat above a collective | `examples/_common.py` | **hang/crash**: every `torchrun` example in the README printed a complete, successful training run and then died on the way out with `Connection closed by peer` |
| 8 | auto-wrap rewrote the caller's module tree without forwarding attributes | `parallel/fsdp.py` | **broken API**: `blocks[0].linear.weight` raised `AttributeError` once FSDP replaced the submodule, defeating the point of `summon_full_params()` |
| 9 | `python` assumed to exist | `scripts/run_hybrid_example.sh`, `Makefile` | **exit 127**: a stock Debian/Ubuntu has only `python3`, so the documented script died with `python: command not found` |

Four further items were wrong *expectations* rather than wrong code, which is
worth distinguishing — the implementation was right and the test was lying:

- `2 x 929` optimizer elements: a stale figure from a different model. The
  checkpoint suite's MLP has **583** parameters, not 1857 (that is the *FSDP*
  suite's model, where the number is correct). The assertion now derives the
  expectation from the model at runtime so it cannot go stale again. The same
  stale number had propagated into `08_*` §11.
- `files_read == 2` for every reader at 4 -> 2: correct for reader 0, wrong for
  reader 1, which legitimately opens a third file for its own RNG stream
  (`08_*` §7).
- `losses[-1] < losses[0]` as a convergence test: step-to-step variance (~0.08)
  is several times the total improvement over eight steps (~0.03), so this
  measured which two mini-batches were drawn. It is false on this seed even
  though training works. Now a first-quarter/last-quarter mean comparison.
- Comparing a tensor-parallel run's `weight_norm` against the single-process
  model's: `11.24` vs `13.11`, which looks alarming and means nothing. Under
  tensor parallelism `full_state_dict()` returns this rank's **slices** — a
  tensor-parallel slice is a genuinely different tensor on each rank, and the
  method's docstring says so. The test was comparing half a model to a whole
  one. Every step's loss and gradient norm agreed to `1e-7` throughout, which
  is the actual equivalence claim, so the test now asserts the whole trajectory
  instead of the endpoint weights. Exact per-weight equality against an
  unsharded reference is asserted at `0.0` in
  `tests/distributed/test_tensor_parallel.py`, which gathers the slices first.

  This one is worth dwelling on: the *code* was right, the *docstring* was
  right, and the test still failed — because the question it asked was not
  well-formed. A distributed assertion has to name the frame it is asserting
  in (this rank's shard, or the global tensor); "the weights match" does not.

**Defects 7, 8 and 9 were found only by running the documented commands.** None
of them is reachable from the test suite: the tests build models directly rather
than through `examples/_common.py`, they navigate modules by `named_parameters()`
rather than by attribute path, and they never invoke the shell scripts. A green
suite said nothing about whether the README worked. It did not.

**What the pattern says.** Six of the nine defects (1, 2, 3, 6, 7, 8) are
invisible on one rank and in a short run: they need either a second rank or a
long enough suite to exhaust memory. Defect 4 is invisible *always* — it
produced a plausible, self-consistent, exactly-2x-wrong number that no assertion
on bytes would ever catch. That is the argument for the structural tests in
`tests/unit/test_source_invariants.py`: the failure modes that matter most in a
distributed system are the ones with no failing behaviour to observe.

Defects 1, 2 and 7 are now prevented by construction. Three invariants are
asserted against the parsed source of every module in `src/`, `examples/` and
`scripts/`:

| Invariant | Catches |
|---|---|
| no `raise` under a rank guard | defects 1, 2 — the raiser leaves, everyone else blocks in the next barrier |
| no rank-guarded `return` above a collective | defect 7 — the primary issues a collective alone |
| no `dist.<collective>()` without `group=` | silently reducing over the world instead of the intended group |

Note that defect 7 lived in `examples/`, which the first version of the
structural test did not scan. An example launched under `torchrun` deadlocks a
user's job exactly as thoroughly as the library does, so the scan now covers
everything that can run on more than one rank. Each invariant is verified to
fail on a planted violation, not merely to pass on clean code.

---

## 12. Documented commands actually executed

Every runnable command in `README.md`, in this document and in the example
docstrings was executed on the development machine. This section exists because
running them found three defects (§11, items 7–9) that the entire test suite
could not reach.

| Command | Result |
|---|---|
| `torchrun --nproc-per-node=2 examples/train_ddp.py` | ✅ |
| `torchrun --nproc-per-node=2 examples/train_fsdp.py` | ✅ |
| `torchrun --nproc-per-node=4 examples/train_fsdp.py --config configs/fsdp_4gpu.yaml` | ✅ |
| `torchrun --nproc-per-node=2 examples/train_tensor_parallel.py` | ✅ |
| `torchrun --nproc-per-node=2 examples/train_sequence_parallel.py` | ✅ |
| `torchrun --nproc-per-node=4 examples/train_hybrid.py` | ✅ |
| `python examples/resume_with_different_world_size.py --save-ranks 4 --load-ranks 2` | ✅ |
| `torchrun --nproc-per-node=2 examples/save_distributed_checkpoint.py --checkpoint-dir …` | ✅ |
| `python scripts/inspect_checkpoint.py <ckpt> --tensors --plan blocks.0.linear.weight` | ✅ |
| `bash scripts/run_hybrid_example.sh 4` | ✅ |
| `python scripts/benchmark.py --world-size 2 --strategies ddp fsdp` | ✅ |
| `torchrun --nproc-per-node=8 examples/train_hybrid.py --config configs/hybrid_8gpu.yaml` | **not run** — needs ~4.8 GB of spawned ranks against ~3.4 GB available |
| `pytest`, `ruff check`, `ruff format --check`, `mypy`, `make check` targets | ✅ (§10) |

The 8-rank hybrid is the one documented command that has **not** been executed.
It is recorded as not-run rather than assumed to work, for the same reason the
8-rank end-to-end test skips rather than being quietly dropped: this machine
cannot host it, and a command nobody has run is not a verified command.

**Two configurations were added because the documented commands demanded them.**
`examples/train_fsdp.py` documented a 2-rank invocation while defaulting to a
4-way shard config, and `examples/train_hybrid.py` documented a 4-rank one while
defaulting to an 8-rank config. Both would have failed validation immediately.
Rather than weaken the documentation to match the code, `configs/fsdp_2gpu.yaml`
and `configs/hybrid_4gpu.yaml` were added so the commands work as written; the
derived process groups of each were checked against the rank tables in their
headers.

**One ergonomic consequence is documented rather than fixed.** Examples write to
`runs/<name>/`, and saving refuses to overwrite an existing checkpoint, so
running the same example twice stops with a `CheckpointError` naming the
directory. That is the intended safety property — silently replacing a
checkpoint is how a run is lost — so the README says so and points at
`--checkpoint-dir`.
