"""Architectural invariants checked against the source itself.

Most tests in this suite run code. These read it.

That is the right tool for a specific class of bug: one where the *shape* of
the code is wrong in a way that only shows up on a rank you are not looking at,
under a condition you did not think to trigger. A runtime test can only catch
such a bug if it happens to exercise that exact path on that exact rank; a
structural test catches every instance, including the ones nobody has written a
scenario for yet.

Both invariants below were added after the corresponding bug shipped.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src" / "hybrid_training"

#: Methods and functions that issue a collective, so every rank must reach them.
COLLECTIVE_CALLS = frozenset(
    {
        "all_gather_tensor",
        "all_reduce",
        "all_to_all_tensor",
        "barrier",
        "broadcast",
        "parameter_count",
        "reduce_scatter_tensor",
        "sum_scalar",
    }
)


def source_files() -> list[Path]:
    """Every Python module in the package, sorted for stable test ids."""
    return sorted(SOURCE_ROOT.rglob("*.py"))


def rank_aware_files() -> list[Path]:
    """Every module that can run on more than one rank.

    Deliberately wider than :func:`source_files`.  The examples and scripts are
    launched under ``torchrun`` exactly like the library, so a rank-asymmetry
    bug in them deadlocks a user's job just as thoroughly -- and one did.
    """
    return sorted(
        [
            *SOURCE_ROOT.rglob("*.py"),
            *(REPOSITORY_ROOT / "examples").rglob("*.py"),
            *(REPOSITORY_ROOT / "scripts").rglob("*.py"),
        ]
    )


def _describes_primary_rank(test: ast.expr) -> bool:
    """Whether an ``if`` test selects a single rank.

    Args:
        test: The condition expression of an ``if`` statement.

    Returns:
        ``True`` when the condition is rank-dependent in the usual ways.
    """
    rendered = ast.unparse(test)
    return "is_primary" in rendered or "rank == 0" in rendered or "rank != 0" in rendered


class TestCollectiveErrorReporting:
    """A raise that only one rank reaches is a deadlock, not an error."""

    @pytest.mark.parametrize("path", source_files(), ids=lambda p: p.name)
    def test_no_raise_is_guarded_by_a_rank_check(self, path: Path) -> None:
        """``if is_primary: ... raise ...`` strands every other rank.

        The other ranks do not see the exception. They continue into whatever
        collective comes next and block there until the raising rank's *process*
        exits, at which point they report a transport error naming a TCP socket
        rather than the actual problem.

        `save_checkpoint` shipped with exactly this bug; see
        `docs/08_distributed_checkpointing.md` section 6. The fix is always the
        same: compute the verdict on one rank, broadcast it, raise everywhere.

        A raise that the same block also catches is fine -- it never escapes.
        """
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        offenders: list[str] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.If) or not _describes_primary_rank(node.test):
                continue
            for statement in node.body:
                for inner in ast.walk(statement):
                    # A raise inside a `try` in this block is caught locally and
                    # cannot strand anyone, so it is not an offender.
                    if isinstance(inner, ast.Try):
                        continue
                    if isinstance(inner, ast.Raise):
                        offenders.append(
                            f"{path.name}:{inner.lineno} raises inside "
                            f"`if {ast.unparse(node.test)}:` (opened at line {node.lineno})"
                        )

        assert not offenders, (
            "a rank-guarded raise leaves every other rank in the next collective:\n  "
            + "\n  ".join(offenders)
        )


class TestExplicitGroups:
    """No collective may fall back to the default process group."""

    @pytest.mark.parametrize("path", source_files(), ids=lambda p: p.name)
    def test_no_collective_is_called_without_a_group(self, path: Path) -> None:
        """Reducing over the wrong group converges to the wrong answer silently.

        `torch.distributed` defaults to the world group when ``group=`` is
        omitted, which is the one failure this project cannot tolerate: it does
        not raise, it does not hang, it just trains a subtly wrong model. Every
        call site must name its group.

        Only `context.py` may talk to `torch.distributed` without one, because
        it is what creates the groups in the first place.
        """
        if path.name == "context.py":
            pytest.skip("context.py owns process-group creation and teardown")

        collectives = {
            "all_reduce",
            "all_gather",
            "all_gather_into_tensor",
            "all_to_all_single",
            "broadcast",
            "reduce_scatter",
            "reduce_scatter_tensor",
        }
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        offenders: list[str] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in collectives:
                continue
            # Only calls made through the `dist` module are raw collectives; the
            # project's own wrappers take a required `GroupHandle` positionally.
            target = node.func.value
            if not (isinstance(target, ast.Name) and target.id == "dist"):
                continue
            if not any(keyword.arg == "group" for keyword in node.keywords):
                offenders.append(f"{path.name}:{node.lineno} dist.{node.func.attr}(...)")

        assert not offenders, (
            "these collectives would silently use the default (world) group:\n  "
            + "\n  ".join(offenders)
        )


class TestCollectiveParticipation:
    """Every rank must reach every collective, or the job deadlocks."""

    @pytest.mark.parametrize("path", rank_aware_files(), ids=lambda p: f"{p.parent.name}/{p.name}")
    def test_no_rank_guarded_return_skips_a_collective(self, path: Path) -> None:
        """``if not is_primary: return`` above a collective strands rank 0.

        This is the mirror image of the guarded-``raise`` bug, and it is easier
        to write by accident because the guard looks like a pure output
        optimisation -- "only rank 0 prints the summary". But if anything below
        the guard issues a collective, rank 0 issues it alone and the peers have
        already left, so rank 0 fails with a transport error naming a socket.

        `examples/_common.py::report_result` shipped with exactly this shape:
        the guard sat above `engine.parameter_count()`, which all-reduces the
        local count over the world group. Every `torchrun` example in the README
        printed a complete, successful training run and then died on the way
        out. Fixed by hoisting the collective above the guard -- compute
        collectively, print conditionally.
        """
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        offenders: list[str] = []

        functions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        ]
        for function in functions:
            for position, statement in enumerate(function.body):
                is_guarded_return = (
                    isinstance(statement, ast.If)
                    and _describes_primary_rank(statement.test)
                    and any(isinstance(inner, ast.Return) for inner in statement.body)
                )
                if not is_guarded_return:
                    continue
                for later in function.body[position + 1 :]:
                    for node in ast.walk(later):
                        if (
                            isinstance(node, ast.Call)
                            and isinstance(node.func, ast.Attribute)
                            and node.func.attr in COLLECTIVE_CALLS
                        ):
                            offenders.append(
                                f"{path.name}:{node.lineno} calls .{node.func.attr}() after "
                                f"`if {ast.unparse(statement.test)}: return` "
                                f"in {function.name}() (line {statement.lineno})"
                            )

        assert not offenders, (
            "these collectives are unreachable on non-primary ranks, so the "
            "primary would issue them alone:\n  " + "\n  ".join(offenders)
        )
