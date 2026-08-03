"""Rank-aware structured logging.

Why a bespoke module instead of ``logging.basicConfig``?

1. **Every line must carry the rank.** When 8 processes interleave on one
   terminal, a log line without a rank is nearly useless.
2. **Most lines should only come from one rank.** Printing the loss from all 8
   ranks is noise; printing it from none is worse.  The helpers here make
   "primary rank only" the *default* for progress output and require an
   explicit opt-in for all-rank output.
3. **Machine-readable mode.** ``HYBRID_LOG_FORMAT=json`` emits one JSON object
   per line so a multi-rank run can be sorted/filtered offline.  This is how
   the distributed test harness captures child logs.

The module holds a single piece of mutable state -- the rank metadata used by
the filter.  It is set by :func:`configure_logging` and by
:class:`hybrid_training.distributed.context.DistributedContext` when it comes
up, and it is *only* metadata: no process group handles live here.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

__all__ = [
    "LogContext",
    "RankFilter",
    "configure_logging",
    "get_logger",
    "log_context",
    "set_log_context",
]

_DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s [%(rank_tag)s] %(name)s: %(message)s"
_ROOT_LOGGER_NAME = "hybrid_training"


@dataclass(frozen=True)
class LogContext:
    """Rank metadata attached to every log record.

    Attributes:
        rank: Global rank, or ``-1`` before the distributed runtime is up.
        world_size: Global world size, or ``1`` when not distributed.
        local_rank: Node-local rank, used to identify the device.
        coordinates: Human-readable topology coordinates such as
            ``"dp0/sh1/sp0/tp1"``.  Empty until a topology exists.
    """

    rank: int = -1
    world_size: int = 1
    local_rank: int = 0
    coordinates: str = ""

    @property
    def tag(self) -> str:
        """Short human tag used by the text formatter."""
        base = "rank ?" if self.rank < 0 else f"rank {self.rank}/{self.world_size}"
        return f"{base} {self.coordinates}" if self.coordinates else base

    @property
    def is_primary(self) -> bool:
        """``True`` on rank 0 and in the not-yet-initialised single-process case."""
        return self.rank <= 0


_LOG_CONTEXT = LogContext()


def set_log_context(context: LogContext) -> LogContext:
    """Install rank metadata used by subsequent log records.

    Args:
        context: The new metadata.

    Returns:
        The metadata that was previously installed, so callers can restore it.
    """
    global _LOG_CONTEXT
    previous = _LOG_CONTEXT
    _LOG_CONTEXT = context
    return previous


def get_log_context() -> LogContext:
    """Return the currently installed rank metadata."""
    return _LOG_CONTEXT


@contextmanager
def log_context(context: LogContext) -> Iterator[LogContext]:
    """Temporarily install rank metadata.

    Args:
        context: Metadata to install for the duration of the block.

    Yields:
        The installed context.
    """
    previous = set_log_context(context)
    try:
        yield context
    finally:
        set_log_context(previous)


class RankFilter(logging.Filter):
    """Injects rank metadata into records and implements primary-only filtering.

    A record is dropped when it carries ``record.primary_only = True`` and the
    current rank is not the primary rank.  ``configure_logging`` attaches one
    instance of this filter to the package handler.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        ctx = _LOG_CONTEXT
        record.rank = ctx.rank
        record.world_size = ctx.world_size
        record.local_rank = ctx.local_rank
        record.coordinates = ctx.coordinates
        record.rank_tag = ctx.tag
        return not (getattr(record, "primary_only", False) and not ctx.is_primary)


class JsonFormatter(logging.Formatter):
    """One JSON object per line, suitable for post-hoc multi-rank analysis."""

    #: Attributes present on every ``LogRecord`` that we do not want to copy
    #: into the ``extra`` payload.
    _STANDARD_ATTRS = frozenset(
        {
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "message",
            "module",
            "msecs",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "taskName",
            "thread",
            "threadName",
            "rank",
            "world_size",
            "local_rank",
            "coordinates",
            "rank_tag",
            "primary_only",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "rank": getattr(record, "rank", -1),
            "world_size": getattr(record, "world_size", 1),
            "local_rank": getattr(record, "local_rank", 0),
            "coordinates": getattr(record, "coordinates", ""),
            "message": record.getMessage(),
        }
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in self._STANDARD_ATTRS and not key.startswith("_")
        }
        if extras:
            payload["extra"] = {k: _jsonable(v) for k, v in extras.items()}
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True)


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of arbitrary log payloads to JSON-safe values."""
    if isinstance(value, str | int | float | bool | type(None)):
        return value
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return repr(value)


def configure_logging(
    level: int | str = logging.INFO,
    *,
    fmt: str | None = None,
    stream: Any = None,
    force: bool = False,
) -> logging.Logger:
    """Configure the ``hybrid_training`` logger tree.

    Idempotent: calling it twice does not duplicate handlers unless ``force``
    is set.  This matters because examples, the training engine and tests all
    call it.

    Args:
        level: Logging level, as an ``int`` or a level name.
        fmt: ``"text"`` or ``"json"``.  Defaults to the ``HYBRID_LOG_FORMAT``
            environment variable, then ``"text"``.
        stream: Output stream.  Defaults to ``sys.stderr`` so that ``stdout``
            remains clean for program output.
        force: Replace existing handlers instead of leaving them alone.

    Returns:
        The configured ``hybrid_training`` logger.

    Raises:
        ValueError: If ``fmt`` is not ``"text"`` or ``"json"``.
    """
    resolved_fmt = (fmt or os.environ.get("HYBRID_LOG_FORMAT") or "text").lower()
    if resolved_fmt not in {"text", "json"}:
        raise ValueError(f"fmt must be 'text' or 'json', got {resolved_fmt!r}")

    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    if logger.handlers and not force:
        logger.setLevel(level)
        return logger

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(
        JsonFormatter() if resolved_fmt == "json" else logging.Formatter(_DEFAULT_FORMAT)
    )
    handler.addFilter(RankFilter())
    logger.addHandler(handler)
    logger.setLevel(level)
    # Package logs are self-contained; propagating to the root logger would
    # duplicate lines in applications that also call basicConfig().
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``hybrid_training`` namespace.

    Args:
        name: Usually ``__name__``.  A leading ``hybrid_training.`` is
            tolerated and not duplicated.

    Returns:
        The child logger.  Handlers are installed lazily by
        :func:`configure_logging`; if the caller never configures logging the
        records are simply dropped by the standard library's
        "no handler" behaviour, which keeps library imports side-effect free.
    """
    if name == _ROOT_LOGGER_NAME or name.startswith(_ROOT_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")


def log_primary(logger: logging.Logger, level: int, message: str, **extra: Any) -> None:
    """Log ``message`` only from the primary rank.

    Args:
        logger: Target logger.
        level: Logging level.
        message: Message, already formatted.
        **extra: Structured fields attached to the record.
    """
    logger.log(level, message, extra={"primary_only": True, **extra})
