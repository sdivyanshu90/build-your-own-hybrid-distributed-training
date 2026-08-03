# Diagrams

The diagrams for this project are written as [Mermaid](https://mermaid.js.org/)
blocks **inside** the documents that explain them, rather than as separate image
files. Three reasons:

1. **They cannot drift.** A diagram that lives next to the paragraph describing
   it is edited in the same commit as that paragraph. A separate `.png` is not.
2. **They are diffable.** A reviewer can see that an arrow changed direction;
   nobody can review a re-rendered image.
3. **They need no toolchain.** GitHub, GitLab and most Markdown viewers render
   Mermaid natively, so there is no build step and no committed binary.

This file is the index. Each entry links to the document and section that
contains the diagram.

| Diagram | Document | Section |
|---|---|---|
| Process-group topology (8 ranks, `dp=2 × shard=2 × tensor=2`) | [01 — Distributed systems foundations](../01_distributed_systems_foundations.md) | §2 |
| DDP gradient flow (hooks → buckets → network) | [03 — Distributed data parallel](../03_distributed_data_parallel.md) | §3 |
| DDP bucket lifecycle (state machine) | [03 — Distributed data parallel](../03_distributed_data_parallel.md) | §3 |
| FSDP parameter lifecycle (shard → gather → compute → reshard → reduce-scatter) | [04 — FSDP-style sharding](../04_fsdp_style_sharding.md) | §3 |
| Tensor-parallel MLP flow (`f` → column → activation → row → `g`) | [05 — Tensor parallelism](../05_tensor_parallelism.md) | §2 |
| Sequence-parallel activation flow through a transformer block | [06 — Sequence parallelism](../06_sequence_parallelism.md) | §4 |
| Hybrid parallelism topology (which group carries which traffic) | [07 — Hybrid parallelism](../07_hybrid_parallelism.md) | §4 |
| Hybrid collective ordering within one step | [07 — Hybrid parallelism](../07_hybrid_parallelism.md) | §3 |
| Distributed-checkpoint save protocol | [08 — Distributed checkpointing](../08_distributed_checkpointing.md) | §6 |
| Checkpoint resharding (interval intersection) | [08 — Distributed checkpointing](../08_distributed_checkpointing.md) | §4 |

## Rendering them elsewhere

If you need standalone images — for slides, say — the Mermaid CLI will produce
them without any change to the sources:

```bash
npm install -g @mermaid-js/mermaid-cli

# extract and render every fenced mermaid block in a document
python - <<'PY'
import pathlib, re, subprocess
for doc in sorted(pathlib.Path("docs").glob("*.md")):
    blocks = re.findall(r"```mermaid\n(.*?)```", doc.read_text(), re.S)
    for index, block in enumerate(blocks):
        source = pathlib.Path(f"docs/diagrams/{doc.stem}-{index}.mmd")
        source.write_text(block)
        subprocess.run(["mmdc", "-i", str(source),
                        "-o", str(source.with_suffix(".svg"))], check=True)
PY
```

Generated `.mmd` and `.svg` files are deliberately **not** committed: they would
be the second copy that goes stale.
