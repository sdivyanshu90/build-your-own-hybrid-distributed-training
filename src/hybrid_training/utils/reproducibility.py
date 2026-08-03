"""Seeding and RNG-state management for partitioned computation.

Random numbers are subtle in a hybrid-parallel job because different ranks play
different roles, and the *same* physical model can require both identical and
independent random streams at the same time:

============================  ========================  =========================
Random draw                   Ranks that must agree     Ranks that must differ
============================  ========================  =========================
Weight initialisation of a    all data-parallel and     none (see below)
replicated parameter          shard ranks
Weight initialisation of a    none                      tensor-parallel ranks
tensor-parallel slice         (each holds a different    hold different slices
                              slice)
Dropout on a *replicated*     all ranks in the tensor/  none
activation                    sequence group
Dropout on a *sharded*        none                      every rank in the
activation                                              sequence group
Data sampling                 tensor/sequence group     data-parallel and shard
                              (they share a batch)      ranks
============================  ========================  =========================

Weight initialisation for partitioned parameters
------------------------------------------------
Two strategies exist:

*Slice-a-full-draw* (what this project does).  Every tensor-parallel rank draws
the *entire* weight matrix from an identical seed and then keeps only its
slice.  The concatenation of the slices is then bit-for-bit what a single
process would have produced, which is what makes
``tests/distributed/test_tensor_parallel.py`` able to assert exact structural
equivalence against an unsharded ``nn.Linear``.  The cost is a transient
full-size allocation during construction and some wasted RNG work.

*Draw-only-your-slice*.  Each rank offsets its seed and draws only the elements
it keeps.  This costs nothing but produces a different (equally valid)
initialisation from the single-process reference, which makes exact
equivalence tests impossible.  Megatron-LM does this and accepts the
consequence.

The trade-off is documented in ``docs/05_tensor_parallelism.md``.
"""

from __future__ import annotations

import os
import random
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..logging import get_logger

__all__ = [
    "RngSnapshot",
    "capture_rng_state",
    "derive_seed",
    "restore_rng_state",
    "seed_everything",
    "temporary_seed",
]

_LOGGER = get_logger(__name__)

#: Large odd multipliers used to decorrelate derived seeds.  They are arbitrary
#: but fixed, because a *reproducible* derivation is the entire point; any two
#: distinct (base, stream, index) triples must map to distinct seeds with
#: overwhelming probability.
_STREAM_MULTIPLIER = 0x9E3779B1  # 2**32 / golden ratio, the usual choice
_INDEX_MULTIPLIER = 0x85EBCA6B
_SEED_MASK = (1 << 31) - 1


def derive_seed(base_seed: int, stream: str, index: int = 0) -> int:
    """Derive a reproducible sub-seed from a base seed, a name and an index.

    Args:
        base_seed: The experiment's master seed.
        stream: A name identifying the purpose, e.g. ``"model-init"`` or
            ``"dropout"``.  Hashing the name means adding a new random stream
            never perturbs the existing ones.
        index: A per-rank or per-group index; use ``0`` for streams that must
            be identical everywhere.

    Returns:
        A non-negative seed below ``2**31``.

    Example:
        >>> derive_seed(1234, "model-init") == derive_seed(1234, "model-init")
        True
        >>> derive_seed(1234, "dropout", 0) != derive_seed(1234, "dropout", 1)
        True
    """
    # A stable string hash: Python's built-in hash() is salted per process.
    stream_hash = 0
    for char in stream.encode("utf-8"):
        stream_hash = (stream_hash * 131 + char) & 0xFFFFFFFF
    mixed = (
        (base_seed & 0xFFFFFFFF)
        ^ ((stream_hash * _STREAM_MULTIPLIER) & 0xFFFFFFFF)
        ^ ((index * _INDEX_MULTIPLIER) & 0xFFFFFFFF)
    )
    # One xorshift round so nearby indices do not produce nearby seeds.
    mixed ^= (mixed >> 16) & 0xFFFFFFFF
    mixed = (mixed * 0x2545F491) & 0xFFFFFFFF
    mixed ^= (mixed >> 13) & 0xFFFFFFFF
    return int(mixed & _SEED_MASK)


@dataclass
class RngSnapshot:
    """Captured state of every RNG this project can perturb.

    Attributes:
        python_state: ``random.getstate()`` payload.
        numpy_state: ``numpy.random.get_state()`` payload.
        torch_cpu_state: CPU generator state as a ``ByteTensor``.
        torch_cuda_states: Per-device CUDA generator states, or ``None`` when
            CUDA is unavailable.
    """

    python_state: Any
    numpy_state: Any
    torch_cpu_state: torch.Tensor
    torch_cuda_states: list[torch.Tensor] | None = None


def capture_rng_state(*, include_cuda: bool = True) -> RngSnapshot:
    """Snapshot the Python, NumPy and PyTorch RNG states.

    Saved into every checkpoint so that a resumed run draws exactly the
    dropout masks and data order it would have drawn had it never stopped.

    Args:
        include_cuda: Also capture CUDA generator states when CUDA is present.

    Returns:
        The snapshot.
    """
    cuda_states: list[torch.Tensor] | None = None
    if include_cuda and torch.cuda.is_available():
        cuda_states = [state.clone() for state in torch.cuda.get_rng_state_all()]
    return RngSnapshot(
        python_state=random.getstate(),
        numpy_state=np.random.get_state(),
        torch_cpu_state=torch.get_rng_state().clone(),
        torch_cuda_states=cuda_states,
    )


def restore_rng_state(snapshot: RngSnapshot) -> None:
    """Restore RNG state captured by :func:`capture_rng_state`.

    CUDA states are restored only when the current process has at least as many
    devices as the snapshot recorded; otherwise the CUDA part is skipped with a
    warning, because forcing it would raise on a legitimate CPU-only resume.

    Args:
        snapshot: State to restore.
    """
    random.setstate(snapshot.python_state)
    np.random.set_state(snapshot.numpy_state)
    torch.set_rng_state(snapshot.torch_cpu_state.to(torch.uint8).cpu())
    if snapshot.torch_cuda_states is not None and torch.cuda.is_available():
        if torch.cuda.device_count() >= len(snapshot.torch_cuda_states):
            torch.cuda.set_rng_state_all(
                [state.to(torch.uint8).cpu() for state in snapshot.torch_cuda_states]
            )
        else:
            _LOGGER.warning(
                "checkpoint holds %d CUDA RNG states but only %d device(s) are visible; "
                "CUDA RNG state was not restored",
                len(snapshot.torch_cuda_states),
                torch.cuda.device_count(),
            )


def seed_everything(
    seed: int,
    *,
    stream: str = "global",
    index: int = 0,
    deterministic: bool = True,
    warn_only: bool = True,
) -> int:
    """Seed Python, NumPy and PyTorch, and optionally force deterministic kernels.

    Args:
        seed: Master seed.
        stream: Named sub-stream; see :func:`derive_seed`.
        index: Per-rank index for streams that must differ across ranks.
        deterministic: Enable ``torch.use_deterministic_algorithms`` and
            disable cuDNN benchmarking.  Slower, but required for the
            equivalence tests.
        warn_only: Pass ``warn_only=True`` to
            ``torch.use_deterministic_algorithms`` so that an operator without
            a deterministic kernel warns rather than raising.  Set ``False``
            to make such an operator a hard error.

    Returns:
        The derived seed that was actually applied.
    """
    derived = derive_seed(seed, stream, index)
    random.seed(derived)
    np.random.seed(derived % (2**32))
    torch.manual_seed(derived)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(derived)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        # Required by cuBLAS for deterministic GEMMs; harmless on CPU.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return derived


@contextmanager
def temporary_seed(seed: int, *, devices: list[torch.device] | None = None) -> Iterator[None]:
    """Run a block under a fixed seed, restoring the previous state afterwards.

    Used when initialising tensor-parallel weight slices: every rank must draw
    the *same* full matrix, but doing so must not disturb the surrounding
    random stream (which the rest of model construction depends on).

    Args:
        seed: Seed applied inside the block.
        devices: CUDA devices whose generators should also be forked.  Defaults
            to the current device when CUDA is available.

    Yields:
        ``None``.
    """
    snapshot = capture_rng_state(include_cuda=torch.cuda.is_available())
    try:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            if devices is None:
                torch.cuda.manual_seed_all(seed)
            else:
                for device in devices:
                    with torch.cuda.device(device):
                        torch.cuda.manual_seed(seed)
        yield
    finally:
        restore_rng_state(snapshot)


def rng_state_to_serialisable(snapshot: RngSnapshot) -> dict[str, Any]:
    """Convert a snapshot into tensors and JSON-safe primitives.

    The checkpoint format keeps tensor payloads in ``.pt`` files loaded with
    ``weights_only=True``, which refuses arbitrary pickled objects.  Python's
    and NumPy's RNG states are tuples containing an array, so they are split
    into a tensor part and a small JSON part here.

    Args:
        snapshot: State to convert.

    Returns:
        A mapping with two keys: ``"tensors"`` (name -> ``ByteTensor`` /
        ``IntTensor``) and ``"meta"`` (JSON-safe scalars).
    """
    python_version, python_keys, python_gauss = snapshot.python_state
    numpy_name, numpy_keys, numpy_pos, numpy_has_gauss, numpy_gauss = snapshot.numpy_state

    tensors: dict[str, torch.Tensor] = {
        "torch_cpu": snapshot.torch_cpu_state.to(torch.uint8).cpu(),
        # Python's Mersenne-Twister key is 625 unsigned 32-bit words; int64
        # holds them without sign trouble.
        "python_keys": torch.tensor(list(python_keys), dtype=torch.int64),
        "numpy_keys": torch.tensor(np.asarray(numpy_keys, dtype=np.int64), dtype=torch.int64),
    }
    if snapshot.torch_cuda_states is not None:
        for index, state in enumerate(snapshot.torch_cuda_states):
            tensors[f"torch_cuda_{index}"] = state.to(torch.uint8).cpu()

    meta: dict[str, Any] = {
        "python_version": int(python_version),
        "python_gauss": python_gauss,
        "numpy_name": str(numpy_name),
        "numpy_pos": int(numpy_pos),
        "numpy_has_gauss": int(numpy_has_gauss),
        "numpy_gauss": float(numpy_gauss),
        "num_cuda_states": 0
        if snapshot.torch_cuda_states is None
        else len(snapshot.torch_cuda_states),
    }
    return {"tensors": tensors, "meta": meta}


def rng_state_from_serialisable(payload: dict[str, Any]) -> RngSnapshot:
    """Inverse of :func:`rng_state_to_serialisable`.

    Args:
        payload: Mapping with ``"tensors"`` and ``"meta"`` keys.

    Returns:
        The reconstructed snapshot.
    """
    tensors = payload["tensors"]
    meta = payload["meta"]

    python_state = (
        meta["python_version"],
        tuple(int(v) for v in tensors["python_keys"].tolist()),
        meta["python_gauss"],
    )
    numpy_state = (
        meta["numpy_name"],
        np.asarray(tensors["numpy_keys"].tolist(), dtype=np.uint32),
        meta["numpy_pos"],
        meta["numpy_has_gauss"],
        meta["numpy_gauss"],
    )
    cuda_states: list[torch.Tensor] | None = None
    count = int(meta.get("num_cuda_states", 0))
    if count:
        cuda_states = [tensors[f"torch_cuda_{i}"] for i in range(count)]
    return RngSnapshot(
        python_state=python_state,
        numpy_state=numpy_state,
        torch_cpu_state=tensors["torch_cpu"],
        torch_cuda_states=cuda_states,
    )
