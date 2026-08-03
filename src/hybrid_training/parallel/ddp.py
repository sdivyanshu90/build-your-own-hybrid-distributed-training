"""Distributed Data Parallel, implemented from the collectives up.

What DDP actually is
====================
Every rank holds a *complete* copy of the model and processes a different slice
of the global batch.  If rank *r* computes the loss ``L_r`` as the mean over its
own ``B`` samples, then the gradient of the *global* mean loss over ``R * B``
samples is

.. math::

    \\nabla L = \\frac{1}{R}\\sum_{r=0}^{R-1} \\nabla L_r

so the only thing DDP has to do is **average the gradients across ranks before
the optimizer steps**.  Everything else -- bucketing, hooks, asynchronous
collectives -- is performance engineering around that one equation.

Two consequences that trip people up:

* If you *sum* instead of averaging, every gradient is ``R`` times too large,
  which is exactly equivalent to multiplying the learning rate by ``R``.  That
  is a legitimate choice (set ``average_gradients=False``), but it must be a
  choice, not an accident.
* If the per-rank batch sizes differ, the average of the per-rank means is not
  the mean over the global batch.  Getting that right requires weighting by
  sample count; this implementation requires equal per-rank batches and the
  data loader enforces it.

Overlapping communication with computation
==========================================
Backward runs layer by layer from the output towards the input.  The gradient
of the *last* layer is ready long before the gradient of the *first*.  A naive
implementation waits for the whole backward pass and then issues one giant
all-reduce, leaving the network idle during backward and the GPU idle during
the reduction::

    compute:  [====== backward ======]
    network:                          [==== all-reduce ====]

Bucketing fixes this.  Parameters are grouped into fixed buckets in
*approximately reverse* order of the forward pass, which is approximately the
order gradients become ready.  As soon as every parameter in a bucket has a
gradient, that bucket's all-reduce is launched asynchronously::

    compute:  [=== backward ===]
    network:      [=b3=][=b2=][=b1=][=b0=]

The last bucket still cannot overlap with anything, which is why the first
bucket in reverse order (`bucket 0` here, holding the *last* layers) is
deliberately allowed to be smaller in production implementations.

Bucket ordering is a correctness property, not just a performance one
=====================================================================
Collectives issued on the same process group must be issued in the same order
on every rank.  Gradient *readiness* order can differ between ranks -- a rank
whose activation happened to be a leaf, an autograd hook that fired slightly
differently, a conditional branch -- so "reduce whichever bucket happens to
fill up first" is not safe.  This implementation therefore keeps a launch
pointer: bucket ``k`` is only launched once buckets ``0..k-1`` have been
launched.  Buckets that fill out of order are marked ready and wait their turn.
That costs a little overlap in exchange for making the collective order
identical on every rank by construction.

Synchronisation boundary
========================
The optimizer must not step until every bucket's all-reduce has completed.  In
this implementation the boundary is an explicit call::

    loss.backward()
    ddp.finish_gradient_synchronization()   # <-- the boundary
    optimizer.step()

PyTorch's DDP hides this behind an autograd engine callback.  Making it
explicit is a deliberate teaching decision: the boundary is the single most
important line in the training loop, and a missing one is caught here by a
guard in :meth:`DistributedDataParallel.forward` rather than by silently
training on unreduced gradients.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from ..config import DDPConfig
from ..distributed.collectives import (
    AsyncWork,
    CommunicationRecorder,
    ReduceOp,
    all_reduce,
    assert_metadata_consistent,
    assert_tensor_consistent,
    broadcast,
)
from ..distributed.groups import GroupHandle
from ..errors import ShardingError, format_error
from ..logging import get_logger

__all__ = ["BucketLayout", "DistributedDataParallel", "GradientBucket"]

_LOGGER = get_logger(__name__)

_BYTES_PER_MIB = 1024 * 1024


@dataclass(frozen=True)
class BucketLayout:
    """Static description of one bucket, identical on every rank.

    Attributes:
        index: Position in the launch order.  Bucket ``0`` is reduced first.
        parameter_names: Names of the parameters in the bucket, in buffer
            order.
        offsets: Element offset of each parameter inside the flat buffer.
        numels: Element count of each parameter.
        total_numel: Size of the flat buffer.
        dtype_str: Common dtype of the bucket.
        device_str: Common device of the bucket.
    """

    index: int
    parameter_names: tuple[str, ...]
    offsets: tuple[int, ...]
    numels: tuple[int, ...]
    total_numel: int
    dtype_str: str
    device_str: str

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view used by the cross-rank consistency check."""
        return {
            "index": self.index,
            "parameter_names": list(self.parameter_names),
            "offsets": list(self.offsets),
            "numels": list(self.numels),
            "total_numel": self.total_numel,
            "dtype": self.dtype_str,
        }


class GradientBucket:
    """A contiguous buffer holding the gradients of several parameters.

    The buffer is allocated lazily on first use so that a model that never runs
    backward (inference) pays nothing.

    Args:
        layout: The static layout.
        parameters: The parameters, in buffer order.
        dtype: Buffer dtype.
        device: Buffer device.
    """

    __slots__ = (
        "_buffer",
        "_device",
        "_dtype",
        "launched",
        "layout",
        "parameters",
        "pending",
        "ready",
        "work",
    )

    def __init__(
        self,
        layout: BucketLayout,
        parameters: Sequence[nn.Parameter],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.layout = layout
        self.parameters = tuple(parameters)
        self._dtype = dtype
        self._device = device
        self._buffer: torch.Tensor | None = None
        self.pending: int = len(parameters)
        self.ready: bool = False
        self.launched: bool = False
        self.work: AsyncWork | None = None

    @property
    def buffer(self) -> torch.Tensor:
        """The flat gradient buffer, allocated on first access."""
        if self._buffer is None:
            self._buffer = torch.zeros(
                self.layout.total_numel, dtype=self._dtype, device=self._device
            )
        return self._buffer

    @property
    def is_allocated(self) -> bool:
        """Whether the flat buffer exists yet."""
        return self._buffer is not None

    def view_for(self, position: int) -> torch.Tensor:
        """Return the buffer slice for the ``position``-th parameter, reshaped.

        Args:
            position: Index into :attr:`parameters`.

        Returns:
            A view of the flat buffer with the parameter's shape.
        """
        offset = self.layout.offsets[position]
        numel = self.layout.numels[position]
        return self.buffer[offset : offset + numel].view(self.parameters[position].shape)

    def reset(self) -> None:
        """Prepare the bucket for another backward pass."""
        self.pending = len(self.parameters)
        self.ready = False
        self.launched = False
        self.work = None

    def num_bytes(self) -> int:
        """Size of the flat buffer in bytes."""
        return self.layout.total_numel * torch.empty(0, dtype=self._dtype).element_size()

    def __repr__(self) -> str:
        return (
            f"GradientBucket(index={self.layout.index}, params={len(self.parameters)}, "
            f"numel={self.layout.total_numel}, pending={self.pending})"
        )


@dataclass
class DDPStatistics:
    """Counters exposed for debugging and for the benchmark script.

    Attributes:
        steps: Completed synchronisation boundaries.
        buckets_reduced: Total bucket all-reduces issued.
        bytes_reduced: Total bytes passed to all-reduce.
        unused_parameters: Parameters that received no gradient in the most
            recent backward pass.
        out_of_order_buckets: Buckets that became ready before their turn and
            had to wait for the launch pointer.  A large number means the
            bucket order is a poor match for the backward order, which costs
            overlap (but never correctness).
    """

    steps: int = 0
    buckets_reduced: int = 0
    bytes_reduced: int = 0
    unused_parameters: tuple[str, ...] = ()
    out_of_order_buckets: int = 0

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        return {
            "steps": self.steps,
            "buckets_reduced": self.buckets_reduced,
            "bytes_reduced": self.bytes_reduced,
            "unused_parameters": list(self.unused_parameters),
            "out_of_order_buckets": self.out_of_order_buckets,
        }


class DistributedDataParallel(nn.Module):
    """Replicate a module across a process group and average its gradients.

    This is a from-scratch replacement for
    ``torch.nn.parallel.DistributedDataParallel``.  It uses only
    ``broadcast`` and ``all_reduce``.

    Args:
        module: The module to replicate.  It must already be on the correct
            device.
        group: The data-parallel process group.  **Required** -- there is no
            default -- because in a hybrid job the wrong group here is a silent
            correctness bug.
        config: Bucketing and synchronisation knobs.
        recorder: Optional communication instrumentation sink.

    Raises:
        ParameterConsistencyError: If the parameter structure differs across
            ranks in ``group`` and ``config.check_parameter_consistency`` is
            enabled.
        ShardingError: If the module has no trainable parameters.

    Example:
        >>> # doctest: +SKIP
        >>> ddp = DistributedDataParallel(model, ctx.group("data_parallel"), DDPConfig())
        >>> loss = ddp(x).square().mean()
        >>> loss.backward()
        >>> ddp.finish_gradient_synchronization()
        >>> optimizer.step()
    """

    def __init__(
        self,
        module: nn.Module,
        group: GroupHandle,
        config: DDPConfig | None = None,
        *,
        recorder: CommunicationRecorder | None = None,
    ) -> None:
        super().__init__()
        self.module = module
        self._group = group
        self._config = config or DDPConfig()
        self._recorder = recorder

        # Deduplicate by identity: tied weights appear once in the bucket list,
        # so their (already summed) gradient is reduced exactly once.
        self._named_parameters: list[tuple[str, nn.Parameter]] = []
        seen: set[int] = set()
        for name, param in module.named_parameters():
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            self._named_parameters.append((name, param))

        if not self._named_parameters:
            raise ShardingError(
                format_error(
                    "ddp.__init__",
                    "the module has no trainable parameters, so there is nothing to synchronise",
                    rank=group.global_rank,
                    expected=">= 1 parameter with requires_grad=True",
                    observed=0,
                    resolution="wrap a module that has trainable parameters",
                )
            )

        self._name_of: dict[int, str] = {id(p): n for n, p in self._named_parameters}
        self._buckets: list[GradientBucket] = []
        self._bucket_of: dict[int, tuple[GradientBucket, int]] = {}
        self._handles: list[Any] = []
        self._require_backward_grad_sync = True
        self._backward_in_progress = False
        self._next_bucket_to_launch = 0
        self._statistics = DDPStatistics()

        if self._config.check_parameter_consistency:
            self._verify_parameter_structure()

        self._broadcast_initial_state()
        self._build_buckets()
        self._register_hooks()

        _LOGGER.info(
            "DDP ready over group %r (size %d): %d parameters in %d buckets",
            group.name,
            group.size,
            len(self._named_parameters),
            len(self._buckets),
        )

    # -- construction helpers ----------------------------------------------
    def _verify_parameter_structure(self) -> None:
        """Check that every rank in the group has the same parameters, in order.

        Collective, so it either passes everywhere or raises everywhere.  The
        alternative -- discovering the mismatch when an all-reduce receives
        differently shaped buffers -- produces a backend-level crash whose
        message names neither the parameter nor the rank.
        """
        signature = [
            (name, tuple(param.shape), str(param.dtype)) for name, param in self._named_parameters
        ]
        assert_metadata_consistent(
            signature,
            self._group,
            name="the (name, shape, dtype) list of trainable parameters",
            operation="ddp.verify_parameter_structure",
        )

    def _broadcast_initial_state(self) -> None:
        """Make every rank start from the source rank's parameters and buffers.

        Without this step each rank would start from its own random
        initialisation.  Averaging gradients would then keep the *replicas*
        different forever, because averaged gradients applied to different
        starting points give different results.  Model replicas must be
        identical at step 0 and stay identical by induction.
        """
        source = self._config.source_rank_in_group
        for _, param in self._named_parameters:
            broadcast(
                param.data, self._group, source_local_rank=source, recorder=self._recorder
            ).wait()
        for _, buffer in self.module.named_buffers():
            if buffer is None or buffer.numel() == 0:
                continue
            broadcast(
                buffer.data, self._group, source_local_rank=source, recorder=self._recorder
            ).wait()

    def _build_buckets(self) -> None:
        """Group parameters into flat buckets in approximate backward order.

        Parameters are visited in **reverse** ``named_parameters()`` order,
        which for a sequentially defined model approximates the order in which
        backward produces gradients.  A bucket is closed when adding the next
        parameter would exceed the byte cap, or when the dtype/device changes
        (a single collective cannot span dtypes).
        """
        cap_bytes = int(self._config.bucket_cap_mb * _BYTES_PER_MIB)
        current: list[tuple[str, nn.Parameter]] = []
        current_bytes = 0
        current_key: tuple[torch.dtype, torch.device] | None = None

        def close() -> None:
            nonlocal current, current_bytes, current_key
            if not current:
                return
            index = len(self._buckets)
            offsets: list[int] = []
            numels: list[int] = []
            offset = 0
            for _, param in current:
                offsets.append(offset)
                numels.append(param.numel())
                offset += param.numel()
            layout = BucketLayout(
                index=index,
                parameter_names=tuple(name for name, _ in current),
                offsets=tuple(offsets),
                numels=tuple(numels),
                total_numel=offset,
                dtype_str=str(current[0][1].dtype),
                device_str=str(current[0][1].device),
            )
            bucket = GradientBucket(
                layout=layout,
                parameters=[param for _, param in current],
                dtype=current[0][1].dtype,
                device=current[0][1].device,
            )
            self._buckets.append(bucket)
            for position, (_, param) in enumerate(current):
                self._bucket_of[id(param)] = (bucket, position)
            current = []
            current_bytes = 0
            current_key = None

        for name, param in reversed(self._named_parameters):
            key = (param.dtype, param.device)
            param_bytes = param.numel() * param.element_size()
            if current_key is not None and (
                key != current_key or current_bytes + param_bytes > cap_bytes
            ):
                close()
            current_key = key
            current.append((name, param))
            current_bytes += param_bytes
        close()

        if self._config.check_parameter_consistency:
            assert_metadata_consistent(
                [b.layout.as_dict() for b in self._buckets],
                self._group,
                name="the gradient bucket layout",
                operation="ddp.verify_bucket_layout",
            )

    def _register_hooks(self) -> None:
        """Attach a post-accumulate-gradient hook to every trainable parameter.

        ``register_post_accumulate_grad_hook`` fires *after* autograd has
        written ``param.grad``, which is exactly when the value is safe to copy
        into a bucket.  A plain ``register_hook`` on the tensor fires when the
        gradient is *computed*, before accumulation, and would therefore miss
        contributions from a parameter used more than once in the graph.
        """
        for _, param in self._named_parameters:
            handle = param.register_post_accumulate_grad_hook(self._on_grad_ready)
            self._handles.append(handle)

    # -- backward machinery -------------------------------------------------
    def _on_grad_ready(self, param: torch.Tensor) -> None:
        """Copy a finished gradient into its bucket and maybe launch reductions.

        Args:
            param: The parameter whose ``.grad`` has just been accumulated.
        """
        if not self._require_backward_grad_sync:
            # Inside no_sync(): let gradients pile up in `param.grad` and do
            # not touch the buckets.  The next synchronised backward will pick
            # up the accumulated total.
            return
        entry = self._bucket_of.get(id(param))
        if entry is None:  # pragma: no cover - only reachable if hooks outlive the wrapper
            return
        bucket, position = entry
        if bucket.pending == 0:
            # Already reduced this iteration.  Reaching here means backward was
            # run twice without an intervening finish_gradient_synchronization().
            raise ShardingError(
                format_error(
                    "ddp.on_grad_ready",
                    "a gradient arrived for a bucket that has already been reduced this "
                    "iteration; backward() ran twice without a synchronisation boundary",
                    rank=self._group.global_rank,
                    expected="one backward() per finish_gradient_synchronization()",
                    observed=f"bucket {bucket.layout.index}",
                    resolution=(
                        "call finish_gradient_synchronization() after each backward(), or "
                        "wrap accumulation micro-steps in no_sync()"
                    ),
                )
            )

        assert param.grad is not None  # guaranteed by the hook contract
        bucket.view_for(position).copy_(param.grad)
        bucket.pending -= 1
        self._backward_in_progress = True
        if bucket.pending == 0:
            bucket.ready = True
            if bucket.layout.index != self._next_bucket_to_launch:
                self._statistics.out_of_order_buckets += 1
            if self._config.async_reduction:
                self._launch_ready_buckets()

    def _launch_ready_buckets(self) -> None:
        """Launch every ready bucket that is next in line, in index order.

        This is the mechanism that makes the collective order identical on all
        ranks; see the module docstring.
        """
        while (
            self._next_bucket_to_launch < len(self._buckets)
            and self._buckets[self._next_bucket_to_launch].ready
        ):
            bucket = self._buckets[self._next_bucket_to_launch]
            self._launch(bucket, async_op=self._config.async_reduction)
            self._next_bucket_to_launch += 1

    def _launch(self, bucket: GradientBucket, *, async_op: bool) -> None:
        """Issue one bucket's all-reduce."""
        op = ReduceOp.AVG if self._config.average_gradients else ReduceOp.SUM
        bucket.work = all_reduce(
            bucket.buffer,
            self._group,
            op=op,
            async_op=async_op,
            recorder=self._recorder,
        )
        bucket.launched = True
        self._statistics.buckets_reduced += 1
        self._statistics.bytes_reduced += bucket.num_bytes()

    def finish_gradient_synchronization(self) -> None:
        """Complete every outstanding reduction and rebind gradients.

        This is the synchronisation boundary.  It must be called after
        ``backward()`` and before ``optimizer.step()``.  It:

        1. fills in zero gradients for unused parameters (when
           ``find_unused_parameters`` is on) so all ranks reduce the same
           buckets;
        2. launches any bucket that was not launched during backward;
        3. waits for every in-flight all-reduce;
        4. re-points each ``param.grad`` at its slice of the reduced bucket, so
           the optimizer reads averaged gradients with no extra copy.

        Calling it when no backward has happened since the last call is a
        no-op, which makes it safe to put unconditionally in a training loop.

        Raises:
            ShardingError: If a parameter has no gradient and
                ``find_unused_parameters`` is disabled.
        """
        if not self._backward_in_progress:
            return
        if not self._require_backward_grad_sync:
            # Backward ran inside no_sync(); gradients stay local this round.
            self._backward_in_progress = False
            return

        missing: list[str] = []
        for bucket in self._buckets:
            if bucket.pending == 0:
                continue
            for position, param in enumerate(bucket.parameters):
                if param.grad is None:
                    name = self._name_of.get(id(param), f"<param#{position}>")
                    missing.append(name)
                    if self._config.find_unused_parameters:
                        # Contribute explicit zeros so every rank reduces the
                        # same buffer.  The parameter keeps grad=None locally
                        # until the rebind below hands it the averaged value,
                        # which may be non-zero if another rank used it.
                        bucket.view_for(position).zero_()
                        bucket.pending -= 1
            if bucket.pending == 0:
                bucket.ready = True

        if missing and not self._config.find_unused_parameters:
            raise ShardingError(
                format_error(
                    "ddp.finish_gradient_synchronization",
                    "some parameters received no gradient, so their buckets can never "
                    "become ready and the all-reduce would never be issued (every other "
                    "rank would block waiting for it)",
                    rank=self._group.global_rank,
                    world_size=self._group.size,
                    expected="every trainable parameter to receive a gradient",
                    observed=f"{len(missing)} without gradients: {missing[:8]}",
                    resolution=(
                        "set DDPConfig(find_unused_parameters=True), or stop marking "
                        "unused parameters as requires_grad=True"
                    ),
                )
            )
        self._statistics.unused_parameters = tuple(missing)

        # Launch anything still pending, in index order.
        self._launch_ready_buckets()
        for bucket in self._buckets:
            if not bucket.launched:  # pragma: no cover - all buckets are ready by here
                raise ShardingError(
                    format_error(
                        "ddp.finish_gradient_synchronization",
                        f"bucket {bucket.layout.index} never became ready",
                        rank=self._group.global_rank,
                        expected="all buckets ready",
                        observed=f"{bucket.pending} parameters still pending",
                        resolution="this indicates a bug in the bucket bookkeeping",
                    )
                )

        for bucket in self._buckets:
            if bucket.work is not None:
                bucket.work.wait()

        # Point each gradient at the reduced buffer.  This is PyTorch's
        # `gradient_as_bucket_view=True` behaviour: the optimizer then reads
        # directly out of the communication buffer with no copy back.
        for bucket in self._buckets:
            for position, param in enumerate(bucket.parameters):
                param.grad = bucket.view_for(position)
            bucket.reset()

        self._next_bucket_to_launch = 0
        self._backward_in_progress = False
        self._statistics.steps += 1

    # -- nn.Module interface ------------------------------------------------
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the wrapped module's forward pass.

        Before delegating, this checks that the previous iteration's gradients
        were synchronised.  Detecting the mistake here -- rather than letting
        the optimizer step on unreduced gradients -- turns a silent
        divergence-between-replicas bug into an immediate, explicit failure.

        Args:
            *args: Forwarded positionally.
            **kwargs: Forwarded by keyword.

        Returns:
            Whatever the wrapped module returns.

        Raises:
            ShardingError: If a backward pass is still unsynchronised.
        """
        if self._backward_in_progress and self._require_backward_grad_sync:
            raise ShardingError(
                format_error(
                    "ddp.forward",
                    "a previous backward pass has not been synchronised; stepping now "
                    "would use gradients that were never averaged across ranks",
                    rank=self._group.global_rank,
                    expected="finish_gradient_synchronization() after every backward()",
                    observed="pending bucket reductions",
                    resolution=(
                        "call ddp.finish_gradient_synchronization() before the next "
                        "forward, or use the TrainingEngine which does it for you"
                    ),
                )
            )
        if self._config.broadcast_buffers and self.training:
            self._sync_buffers()
        return self.module(*args, **kwargs)

    def _sync_buffers(self) -> None:
        """Broadcast buffers from the source rank.

        Buffers (BatchNorm running statistics, for instance) are updated by the
        forward pass rather than by the optimizer, so they drift apart across
        replicas unless they are explicitly re-synchronised.  Broadcasting at
        the *start* of forward means the statistics used for the step are the
        source rank's, which matches PyTorch DDP.
        """
        source = self._config.source_rank_in_group
        for _, buffer in self.module.named_buffers():
            if buffer.numel() == 0:
                continue
            broadcast(
                buffer.data, self._group, source_local_rank=source, recorder=self._recorder
            ).wait()

    @contextmanager
    def no_sync(self) -> Iterator[None]:
        """Accumulate gradients locally without communicating.

        Used for gradient accumulation: run ``N-1`` micro-batches inside
        ``no_sync()`` and the last one outside it.  Gradients accumulate into
        ``param.grad`` throughout, and the single all-reduce on the final
        micro-batch averages the *sum* over all micro-batches, which is exactly
        the gradient of the larger effective batch.

        The saving is real: ``N`` micro-batches cost one all-reduce instead of
        ``N``.

        Yields:
            ``None``.

        Example:
            >>> # doctest: +SKIP
            >>> for i, batch in enumerate(micro_batches):
            ...     ctx = ddp.no_sync() if i < len(micro_batches) - 1 else nullcontext()
            ...     with ctx:
            ...         (ddp(batch).mean() / len(micro_batches)).backward()
            >>> ddp.finish_gradient_synchronization()
        """
        previous = self._require_backward_grad_sync
        self._require_backward_grad_sync = False
        try:
            yield
        finally:
            self._require_backward_grad_sync = previous

    # -- introspection ------------------------------------------------------
    @property
    def group(self) -> GroupHandle:
        """The data-parallel process group used for gradient reduction."""
        return self._group

    @property
    def config(self) -> DDPConfig:
        """The active configuration."""
        return self._config

    @property
    def statistics(self) -> DDPStatistics:
        """Cumulative bucket/communication counters."""
        return self._statistics

    def bucket_layouts(self) -> tuple[BucketLayout, ...]:
        """Return the static bucket layout, for tests and diagnostics."""
        return tuple(bucket.layout for bucket in self._buckets)

    def communication_summary(self) -> str:
        """Human-readable communication report, or a note if not instrumented."""
        if self._recorder is None:
            return "communication recording is disabled (pass recorder= to enable it)"
        return self._recorder.summary()

    def parameters_and_names(self) -> tuple[tuple[str, nn.Parameter], ...]:
        """The deduplicated trainable ``(name, parameter)`` pairs, in bucket-source order."""
        return tuple(self._named_parameters)

    def verify_replica_consistency(self, *, atol: float = 0.0) -> None:
        """Assert that every replica currently holds the same parameters.

        A debugging aid: replicas that have silently diverged (a forgotten
        synchronisation boundary, a rank-dependent code path) are otherwise
        very hard to spot, because training still "works".

        Args:
            atol: Absolute tolerance.  ``0`` demands bitwise equality, which is
                the right expectation on a single machine with a deterministic
                reduction order.

        Raises:
            ParameterConsistencyError: On the first parameter that differs.
        """
        for name, param in self._named_parameters:
            assert_tensor_consistent(
                param.data,
                self._group,
                name=f"parameter {name!r}",
                atol=atol,
                operation="ddp.verify_replica_consistency",
            )

    def drain_pending_reductions(self) -> int:
        """Wait for any launched-but-unwaited bucket all-reduce.

        Normally :meth:`finish_gradient_synchronization` waits for everything.
        This exists for the path where it never runs: a backward pass launched
        some buckets asynchronously and then something raised -- a shape error,
        a caught ``ShardingError``, a user-level exception.

        Leaving collectives in flight is not merely untidy. Destroying a
        process group with outstanding work **hangs**, so a job that failed
        with a clear error message would go on to hang during teardown and
        report a timeout instead.  Draining here converts that into a normal
        exit carrying the original diagnosis.

        Returns:
            The number of collectives that were still outstanding.
        """
        drained = 0
        for bucket in self._buckets:
            if bucket.work is not None:
                bucket.work.wait()
                bucket.work = None
                drained += 1
            bucket.reset()
        self._next_bucket_to_launch = 0
        self._backward_in_progress = False
        if drained:
            _LOGGER.debug("drained %d outstanding bucket reduction(s)", drained)
        return drained

    def teardown(self) -> None:
        """Drain outstanding collectives and remove the autograd hooks.

        Necessary when the same module is re-wrapped (a test that builds DDP
        twice around one model), because stale hooks would write into buckets
        belonging to a dead wrapper -- and necessary after an exception,
        because an in-flight collective would otherwise hang the process-group
        shutdown.
        """
        self.drain_pending_reductions()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._bucket_of.clear()
        self._buckets.clear()

    def __repr__(self) -> str:
        return (
            f"DistributedDataParallel(group={self._group.name!r}, size={self._group.size}, "
            f"buckets={len(self._buckets)}, params={len(self._named_parameters)})"
        )
