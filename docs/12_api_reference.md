# 12 — API reference

Every symbol listed here exists in the code; every code sample matches the real
signatures. Where a sample is not runnable standalone (because it needs a live
process group) it is marked.

---

## `hybrid_training.config`

Frozen dataclasses. Each validates itself in `__post_init__`, so an illegal
configuration fails at construction on *every* rank simultaneously.

| Class | Key fields |
|---|---|
| `TopologyConfig` | `data_parallel_size`, `shard_parallel_size`, `sequence_parallel_size`, `tensor_parallel_size`, `sequence_parallel_mode` |
| `DDPConfig` | `bucket_cap_mb`, `broadcast_buffers`, `find_unused_parameters`, `average_gradients`, `async_reduction`, `check_parameter_consistency`, `source_rank_in_group` |
| `FSDPConfig` | `reshard_after_forward`, `average_gradients`, `auto_wrap_min_num_params`, `cpu_offload_params`, `limit_all_gather_bytes`, `check_reduction_order`, `use_padding` |
| `TensorParallelConfig` | `gather_output`, `sequence_parallel`, `async_input_gradient_allreduce`, `init_method_std` |
| `MixedPrecisionConfig` | `enabled`, `param_dtype`, `reduce_dtype`, `buffer_dtype`, `master_dtype`, `scaler` |
| `GradScalerConfig` | `enabled`, `init_scale`, `growth_factor`, `backoff_factor`, `growth_interval`, `max_scale` |
| `ModelConfig` | `kind`, `hidden_size`, `num_layers`, `input_size`, `output_size`, `vocab_size`, `num_heads`, `ffn_hidden_size`, `max_sequence_length`, `dropout`, `layer_norm_eps`, `tie_word_embeddings`, `activation` |
| `DataConfig` | `micro_batch_size`, `sequence_length`, `num_train_samples`, `num_eval_samples`, `seed`, `shuffle` |
| `OptimizerConfig` | `name`, `learning_rate`, `weight_decay`, `betas`, `eps`, `momentum`, `cpu_offload_state` |
| `SchedulerConfig` | `name`, `warmup_steps`, `min_lr_ratio` |
| `TrainingConfig` | `max_steps`, `gradient_accumulation_steps`, `max_grad_norm`, `seed`, `log_every_steps`, `eval_every_steps`, `deterministic`, `collect_metrics` |
| `CheckpointConfig` | `directory`, `save_every_steps`, `keep_last`, `save_optimizer_state`, `save_rng_state`, `verify_checksums_on_load`, `resume_from` |
| `ExperimentConfig` | `name`, `backend`, `device`, `timeout_seconds`, plus one field per config above |

```python
from hybrid_training.config import (
    ExperimentConfig, TopologyConfig, load_experiment_config, resolve_dtype,
)

config = load_experiment_config("configs/hybrid_8gpu.yaml")
config.topology.world_size            # 8
config.topology.sequence_parallel_enabled   # True
config.model.resolved_ffn_hidden_size # 128

topology = TopologyConfig.for_world_size(8, tensor_parallel_size=2)
topology.data_parallel_size           # 4  (inferred)

resolve_dtype("bf16")                 # torch.bfloat16
config.to_dict()                      # nested, JSON-safe
config.to_yaml("out.yaml")
ExperimentConfig.from_dict(payload)   # rejects unknown keys
```

`SequenceParallelMode` holds the constants `DISABLED`, `TENSOR_GROUP`,
`INDEPENDENT`.

---

## `hybrid_training.errors`

```
HybridTrainingError
├── ConfigurationError (also ValueError)
│   └── TopologyError
├── DistributedInitializationError
│   └── WorkerFailure                (in distributed.launch)
├── CollectiveError
├── ParameterConsistencyError
├── ShardingError
├── TensorParallelError
├── UnsupportedFeatureError (also NotImplementedError)
└── CheckpointError
    ├── IncompleteCheckpointError
    ├── CheckpointCorruptionError
    ├── CheckpointVersionError
    └── CheckpointTopologyError
```

```python
from hybrid_training.errors import format_error

format_error("topology.validate", "sizes do not factor the world size",
             rank=3, world_size=8, expected=8, observed=6,
             resolution="set dp*shard*seq*tensor == world_size")
# '[rank 3/8] topology.validate: sizes do not factor the world size;
#  expected 8, observed 6. Fix: set dp*shard*seq*tensor == world_size'
```

---

## `hybrid_training.distributed`

### `ParallelTopology`

Pure arithmetic; no `torch.distributed` involvement.

```python
from hybrid_training.config import TopologyConfig
from hybrid_training.distributed.topology import ParallelTopology

topo = ParallelTopology(
    TopologyConfig(data_parallel_size=2, shard_parallel_size=2,
                   tensor_parallel_size=2), 8)

topo.coordinates_of(3).label()          # 'dp0/sh1/sq0/tp1'
topo.rank_of(topo.coordinates_of(3))    # 3
topo.group_ranks("tensor", 3)           # (2, 3)
topo.group_ranks("dp_shard", 3)         # (1, 3, 5, 7)
topo.all_group_rank_lists("shard")      # ((0, 2), (1, 3), (4, 6), (5, 7))
topo.local_rank_in_group("shard", 3)    # 1
topo.group_source_rank("shard", 3)      # 1
topo.neighbour_ranks("tensor", 3)       # (2, 2)
topo.size("dp_shard")                   # 4
topo.sequence_group_name                # 'sequence'
topo.sequence_parallel_size             # 1
print(topo.describe(3))
topo.summary()                          # JSON-safe, for checkpoint metadata
```

Constants: `DIMENSIONS`, `COMPOSITE_GROUPS`. Dataclass: `RankCoordinates`.

### `DistributedContext`

```python
from hybrid_training.distributed.context import (
    distributed_context, init_distributed, current_context,
    is_context_active, find_free_port,
)

with distributed_context(TopologyConfig(data_parallel_size=2),
                         backend="gloo") as ctx:
    ctx.rank, ctx.local_rank, ctx.world_size, ctx.local_world_size
    ctx.backend, ctx.device, ctx.is_primary, ctx.is_active
    ctx.topology, ctx.groups, ctx.coordinates, ctx.env
    ctx.group("data_parallel")          # GroupHandle
    ctx.group("sequence_effective")     # resolves per the SP mode
    ctx.is_group_primary("shard")
    ctx.barrier("world", label="after-setup")
    ctx.synchronize_device()            # no-op on CPU
    print(ctx.describe())
```

`init_distributed(topology=None, *, backend="auto", device="auto",
timeout_seconds=300.0, rank=None, world_size=None, local_rank=None,
master_addr=None, master_port=None, env=None, allow_single_process=True)`.
Accepts a `TopologyConfig` or a whole `ExperimentConfig`.

### `GroupHandle` and `ProcessGroupRegistry`

```python
handle = ctx.group("tensor")
handle.name, handle.ranks, handle.size, handle.local_rank
handle.global_rank, handle.is_trivial, handle.source_rank
handle.process_group                    # backend handle

from hybrid_training.distributed.groups import GroupHandle
GroupHandle.trivial()                   # one member, no communicator

ctx.groups.names                        # creation order + 'world'
ctx.groups.creation_log                 # every (name, ranks) passed to new_group
print(ctx.groups.describe())
```

Constant: `GROUP_CREATION_ORDER`.

### Collectives

Every one takes `group: GroupHandle` as a **required** argument.

```python
from hybrid_training.distributed.collectives import (
    ReduceOp, CommunicationRecorder, AsyncWork,
    all_reduce, broadcast, all_gather_tensor, reduce_scatter_tensor,
    all_to_all_tensor, send_tensor, recv_tensor,
    all_gather_object_in_group, assert_metadata_consistent,
    assert_tensor_consistent, sum_scalar, concat_shard_sizes, wait_all,
)

recorder = CommunicationRecorder()

work = all_reduce(t, group, op=ReduceOp.AVG, async_op=True, recorder=recorder)
work.wait()

broadcast(t, group, source_local_rank=0).wait()

gathered, work = all_gather_tensor(shard, group, out=None, recorder=recorder)
work.wait()

local, work = reduce_scatter_tensor(flat, group, op=ReduceOp.AVG)
work.wait()

permuted = all_to_all_tensor(t, group)
send_tensor(t, group, destination_local_rank=1)
recv_tensor(buffer, group, source_local_rank=0)

objects = all_gather_object_in_group({"rank": ctx.rank}, group)
assert_metadata_consistent(signature, group, name="parameter list")
assert_tensor_consistent(t, group, name="weights", atol=0.0)
total = sum_scalar(1.0, group, device=ctx.device, op=ReduceOp.SUM)
concat_shard_sizes(10, 4)               # (12, 3)

print(recorder.summary())
recorder.total().as_dict()
recorder.reset()
```

`ReduceOp`: `SUM`, `AVG`, `MAX`, `MIN`.

### Launcher

```python
from hybrid_training.distributed.launch import (
    launch_workers, WorkerResult, WorkerFailure, torchrun_environment_summary,
)

def worker(rank, world_size, scale=1.0):   # MUST be module-level
    ...

results = launch_workers(worker, 4, kwargs={"scale": 2.0},
                         timeout_seconds=180, start_method="spawn",
                         log_level="WARNING", raise_on_failure=True)
[r.value for r in results]
results[0].rank, results[0].succeeded, results[0].duration_seconds
```

---

## `hybrid_training.autograd.collectives`

```python
from hybrid_training.autograd.collectives import (
    copy_to_group, reduce_from_group, gather_from_group, scatter_to_group,
    reduce_scatter_to_group, gather_from_sequence_parallel_region,
    all_gather_along_dim, reduce_scatter_along_dim,
)
```

| Function | Forward | Backward |
|---|---|---|
| `copy_to_group(t, group)` | identity | all-reduce |
| `reduce_from_group(t, group)` | all-reduce | identity |
| `gather_from_group(t, group, dim=-1)` | all-gather | split |
| `scatter_to_group(t, group, dim=-1)` | split | all-gather |
| `reduce_scatter_to_group(t, group, dim=1)` | reduce-scatter | all-gather |
| `gather_from_sequence_parallel_region(t, group, dim=1)` | all-gather | **reduce-scatter** |

`all_gather_along_dim` and `reduce_scatter_along_dim` are the
non-differentiable primitives underneath.

---

## `hybrid_training.parallel`

### `DistributedDataParallel`

```python
from hybrid_training.parallel.ddp import DistributedDataParallel

ddp = DistributedDataParallel(module, group, DDPConfig(), recorder=None)

loss = criterion(ddp(x), y)
loss.backward()
ddp.finish_gradient_synchronization()   # the synchronisation boundary
optimizer.step()

with ddp.no_sync():
    (criterion(ddp(x), y) / n).backward()

ddp.group, ddp.config, ddp.statistics
ddp.bucket_layouts()                    # tuple[BucketLayout, ...]
ddp.parameters_and_names()
ddp.communication_summary()
ddp.verify_replica_consistency(atol=0.0)
ddp.teardown()                          # remove hooks
```

`DDPStatistics`: `steps`, `buckets_reduced`, `bytes_reduced`,
`unused_parameters`, `out_of_order_buckets`.

### `FullyShardedDataParallel`

```python
from hybrid_training.parallel.fsdp import FullyShardedDataParallel

fsdp = FullyShardedDataParallel(
    module, shard_group, FSDPConfig(),
    replica_group=None, mixed_precision=None, device=None,
    recorder=None, sync_module_states=True,
)

optimizer = torch.optim.AdamW(fsdp.parameters(), lr=1e-3)  # sharded state
loss.backward()                         # reduce-scatter happens inside
fsdp.finish_backward()
optimizer.step()

with fsdp.no_sync():
    loss.backward()

with fsdp.summon_full_params(writeback=False):
    print(fsdp.module.blocks[0].linear.weight.shape)   # whole tensor

fsdp.handle                             # FlatParamHandle | None
fsdp.fsdp_units()                       # this unit and every nested one
fsdp.shard_group, fsdp.config
fsdp.sharded_state_dict()               # {name: ShardedTensorPiece}
fsdp.full_state_dict()                  # {name: full tensor}
fsdp.load_full_state_dict(state)
fsdp.original_named_parameters()        # {name: (shape, parameter)}
fsdp.original_named_buffers()
fsdp.optimizer_parameter_layout()       # [(flat_param, [PieceLayout, ...])]
fsdp.memory_summary()
```

`FlatParamHandle`: `entries`, `names`, `managed_parameters`, `total_numel`,
`padded_numel`, `shard_numel`, `padding`, `is_sharded`, `shard_group`,
`replica_group`, `reduction_count`, `local_shard_range()`,
`local_piece_layout()`, `local_pieces()`, `full_tensors()`,
`load_local_pieces()`, `memory_summary()`, `metadata()`.

Dataclasses: `ShardedTensorPiece(name, global_shape, offset, data)`,
`PieceLayout(name, global_shape, parameter_offset, length, local_offset)`.

### Tensor-parallel layers

```python
from hybrid_training.parallel.tensor_parallel import (
    ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding,
    TensorParallelFeedForward, TensorParallelMLPBlock,
    init_linear_parameters, mark_sequence_parallel_partial,
    all_reduce_sequence_parallel_gradients,
)

col = ColumnParallelLinear(in_features, out_features, group,
                           bias=True, gather_output=False,
                           sequence_parallel=False, init_seed=0,
                           device=None, dtype=None)
col.full_weight(); col.full_bias(); col.load_from_linear(reference)
col.output_features_per_partition

row = RowParallelLinear(in_features, out_features, group,
                        bias=True, input_is_parallel=True,
                        sequence_parallel=False, init_seed=0)
row.full_weight(); row.load_from_linear(reference)
row.input_features_per_partition

emb = VocabParallelEmbedding(num_embeddings, embedding_dim, group,
                             init_seed=0, init_std=0.02)
emb.full_weight(); emb.vocab_start; emb.vocab_end

ffn = TensorParallelFeedForward(hidden, ffn_hidden, group,
                                activation="gelu", bias=True,
                                sequence_parallel=False, init_seed=0)

n = all_reduce_sequence_parallel_gradients(model, sequence_group)
```

### Sequence-parallel operations

```python
from hybrid_training.parallel.sequence_parallel import (
    SEQUENCE_DIM, SequenceShardInfo, SequenceParallelLayerNorm,
    scatter_sequence, gather_sequence, reduce_scatter_sequence,
    pad_sequence_dimension, unpad_sequence_dimension, local_sequence_slice,
)

shard = scatter_sequence(x, group)                 # (B, S/G, H)
full = gather_sequence(shard, group)               # (B, S, H)
reduced = reduce_scatter_sequence(partial, group)  # (B, S/G, H)

padded, info = pad_sequence_dimension(x, group.size)
info.original_length, info.padded_length, info.local_length, info.padding
x = unpad_sequence_dimension(padded, info)

start, end = local_sequence_slice(padded_length, group)

norm = SequenceParallelLayerNorm(hidden, eps=1e-5, group=group,
                                 sequence_parallel=True)
```

### Hybrid composition

```python
from hybrid_training.parallel.hybrid import (
    HybridModel, ParameterParallelInfo, build_model, build_parallel_model,
    describe_parallel_plan,
)

model = build_parallel_model(config, context, recorder=None, device=None)

model(x)
with model.no_sync():
    loss.backward()
model.finish_backward()

model.optimizer_parameters()
model.metric_group      # dp_shard
model.norm_group        # world
model.data_group        # dp_shard
model.description       # ParallelismDescription
model.fsdp, model.ddp, model.inner_model
model.reduce_metric(value, average=True)
model.validate_input_replication(batch)
model.parameter_parallel_info()         # {name: ParameterParallelInfo}
model.optimizer_parameter_layout()
model.sharded_state_dict(); model.full_state_dict(); model.buffers_state_dict()
model.load_full_state_dict(state)
with model.summon_full_params(): ...
model.memory_summary()

print(describe_parallel_plan(config, context).render())
```

`ParameterParallelInfo`: `name`, `shape`, `partition_dim`,
`tensor_parallel_size`, `tensor_parallel_rank`, `storage_key`.

---

## `hybrid_training.optim`

```python
from hybrid_training.optim.sharded_optimizer import (
    ShardedOptimizer, NormContribution,
    build_inner_optimizer, build_gradient_norm_contributions,
    distributed_gradient_norm,
)

optimizer = ShardedOptimizer(parameters, OptimizerConfig(),
                             norm_group=ctx.group("world"),
                             device=ctx.device, recorder=None)
norm = optimizer.clip_grad_norm(max_norm=1.0)   # returns the pre-clip norm
optimizer.step(); optimizer.zero_grad(set_to_none=True)
optimizer.parameters, optimizer.inner, optimizer.norm_group
optimizer.learning_rate; optimizer.set_learning_rate(3e-4)
optimizer.state_bytes()
optimizer.state_dict(); optimizer.load_state_dict(payload)
```

```python
from hybrid_training.optim.grad_scaler import GradScaler, GradScalerState

scaler = GradScaler(GradScalerConfig(enabled=True), ctx.group("world"),
                    ctx.device)
scaler.scale(loss).backward()
found = scaler.unscale_(parameters)     # collective overflow consensus
stepped = scaler.step(optimizer, parameters, already_unscaled=True)
scaler.update()
scaler.enabled, scaler.scale_value, scaler.state, scaler.last_step_skipped
scaler.state_dict(); scaler.load_state_dict(payload)
```

---

## `hybrid_training.checkpoint`

```python
from hybrid_training.checkpoint import (
    save_checkpoint, load_checkpoint, inspect_checkpoint, read_metadata,
    find_latest_checkpoint, prune_checkpoints,
    CheckpointManifest, TensorRecord, ShardRecord, FileRecord,
    ShardFileCache, read_tensor_range, verify_files, describe_reshard,
    checkpoint_directory_name, shard_filename,
    validate_format_version, validate_shard_filename, file_digest,
    CURRENT_FORMAT_VERSION, SUPPORTED_FORMAT_VERSIONS,
    MANIFEST_FILENAME, METADATA_FILENAME,
)

result = save_checkpoint(
    "runs/exp", model=model, context=ctx, state=state,
    optimizer=optimizer, config=config,
    scheduler_state=scheduler.state_dict(), scaler_state=scaler.state_dict(),
    extra_metadata={"note": "…"},
    save_optimizer=True, save_rng=True, keep_last=3,
)
result.path, result.step, result.seconds, result.bytes_written

loaded = load_checkpoint(
    result.path, model=model, context=ctx, optimizer=optimizer, config=config,
    verify_checksums=True, load_optimizer=True, load_rng=True, strict=True,
)
loaded.step, loaded.state, loaded.scheduler_state, loaded.scaler_state
loaded.manifest, loaded.metadata, loaded.files_read, loaded.rng_restored

summary = inspect_checkpoint(path, verify=True)     # never calls torch.load
manifest = CheckpointManifest.read(path / MANIFEST_FILENAME)
manifest.validate(); print(manifest.summary())
manifest.tensors_by_category("model")
manifest.tensors["w"].shards_overlapping(ShardRange(0, 16))
describe_reshard(manifest, "w", ShardRange(0, 16))
```

---

## `hybrid_training.models`

```python
from hybrid_training.models.mlp import MLP, MLPBlock, build_activation
from hybrid_training.models.transformer import (
    TinyTransformer, TransformerBlock, TensorParallelAttention,
    ParallelPlan, build_reference_linear,
)

model = MLP(ModelConfig(...), seed=0, bias=True,
            tensor_parallel_group=None, device=None, dtype=None)
model.num_parameters()

plan = ParallelPlan(tensor_group=ctx.group("tensor"),
                    sequence_group=ctx.group("sequence_effective"),
                    sequence_parallel=True, vocab_parallel=True)
plan = ParallelPlan.single_process()    # no parallelism, for references

transformer = TinyTransformer(ModelConfig(kind="transformer", ...), plan,
                              seed=0, device=None, dtype=None)
transformer(token_ids, attention_mask=None)   # (B, S, vocab)
transformer.local_positions(sequence_length)
```

---

## `hybrid_training.training`

```python
from hybrid_training.training.engine import (
    TrainingEngine, StepMetrics, mse_loss, cross_entropy_loss,
)

engine = TrainingEngine(config, context, recorder=None)
engine.train(max_steps=None)            # -> TrainingState
engine.evaluate()                       # -> float
engine.save_checkpoint(directory=None)  # -> Path
engine.load_checkpoint(directory=None)  # -> LoadedCheckpoint
engine.parameter_count()                # {'local': …, 'global': …}
engine.memory_snapshot(); engine.communication_summary()
engine.model, engine.optimizer, engine.scheduler, engine.scaler
engine.state, engine.history            # list[StepMetrics]
engine.train_loader, engine.eval_loader, engine.train_sampler
engine.close()
```

```python
from hybrid_training.training.state import TrainingState, LearningRateSchedule

state = TrainingState(step=0, epoch=0)
state.advance_step(samples=64); state.describe(); state.as_dict()

schedule = LearningRateSchedule(SchedulerConfig(name="cosine", warmup_steps=100),
                                base_learning_rate=3e-4, total_steps=10_000)
schedule.value_at(step)                 # pure function of the step
```

```python
from hybrid_training.training.data import (
    Batch, SyntheticDataset, SyntheticMLPDataset, SyntheticTokenDataset,
    DistributedBatchSampler, SyntheticDataLoader, build_dataset,
)

dataset = build_dataset(config.model, config.data, split="train")
sampler = DistributedBatchSampler(len(dataset), micro_batch_size,
                                  ctx.group("dp_shard"), shuffle=True, seed=0)
loader = SyntheticDataLoader(dataset, sampler, ctx.device)
for batch in loader.iter_epoch(0):
    batch.inputs, batch.targets, batch.indices, batch.size
loader.global_batch(0, 0)               # the union, for reference comparisons
```

---

## `hybrid_training.utils`

```python
from hybrid_training.utils.tensors import (
    FlatEntry, ShardRange, build_flat_layout, flatten_dense_tensors,
    unflatten_to_views, pad_flat_tensor, even_shard_ranges, shard_range_for,
    intersect_ranges, split_tensor_along_dim,
)

entries, total = build_flat_layout([("w", w), ("b", b)])
flat = flatten_dense_tensors([w, b])
views = unflatten_to_views(flat, entries)          # views, not copies
padded = pad_flat_tensor(flat, multiple=4)
even_shard_ranges(10, 4)                           # equal, padded ranges
intersect_ranges(ShardRange(0, 6), ShardRange(4, 6))   # ShardRange(4, 2)
split_tensor_along_dim(t, dim=1, num_parts=2)
```

```python
from hybrid_training.utils.memory import (
    MemoryEstimate, MemorySnapshot, estimate_training_memory,
    capture_memory, reset_peak_memory, format_bytes, module_parameter_bytes,
)

estimate = estimate_training_memory(1_000_000, shard_group_size=4,
                                    largest_unit_parameters=100_000,
                                    optimizer="adamw",
                                    reshard_after_forward=True)
estimate.steady_state, estimate.peak; print(estimate.report())
capture_memory(device, include_gc_scan=False)
```

```python
from hybrid_training.utils.reproducibility import (
    seed_everything, derive_seed, temporary_seed,
    capture_rng_state, restore_rng_state, RngSnapshot,
    rng_state_to_serialisable, rng_state_from_serialisable,
)

seed_everything(1234, stream="model-init", index=0, deterministic=True)
derive_seed(1234, "dropout", rank)
with temporary_seed(999):
    ...                                  # restores the outer stream on exit
```

---

## `hybrid_training.logging`

```python
from hybrid_training.logging import (
    configure_logging, get_logger, LogContext, set_log_context, log_context,
)

configure_logging("INFO", fmt="json")    # or HYBRID_LOG_FORMAT=json
logger = get_logger(__name__)
logger.info("step %d", step, extra={"primary_only": True})
```
