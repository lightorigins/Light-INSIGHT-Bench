"""The CLI's exit status survives an Omniverse close that hard-exits with 0.

The bug this gate exists for: ``IsaacAppLifecycle.start`` used to register an
``atexit`` hook that closed the Kit application, and on a real Isaac build
``SimulationApp.close()`` does not close and return -- it exits the process with
status 0. Interpreter-exit hooks run on the way out of a *failed* run too, so
that hook silently rewrote every failure into a success. The standard library
alone shows the mechanism::

    python -c "import atexit,os,sys; atexit.register(lambda: os._exit(0)); sys.exit(2)"
    # exits 0

So ``insight-bench run`` reporting a non-completed ``RunResult`` (exit 1) and
``insight-bench`` dying on an exception (exit 2) would both have reached the shell
as exit 0: no traceback, no failure, a green CI job over a broken run.

These are subprocesses because an exit status is only observable from outside the
process. Each case arms the process-global lifecycle with an application whose
``close()`` behaves the way the operator measured -- write a marker, then
``os._exit(0)`` -- and then runs the real :func:`insight_bench.cli.run_cli`. The
marker file is the whole story. ``close`` means Kit was released; ``after-close``
is written by an interpreter-exit hook the driver owns, so it appears only when
the process came down normally. A failing case must read exactly
``after-close``: no release, and a shutdown that ran to completion.

Nothing here needs a GPU, Isaac, or the ``vln`` extra.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from support import write_run_result

# The fake application, the arming, and the entry point: run in a fresh
# interpreter per case, with the marker path as argv[1].
DRIVER = '''
import atexit
import os
import sys
from pathlib import Path

marker = Path(sys.argv[1])


def mark(word):
    # Opened and closed per call: os._exit() flushes nothing.
    with marker.open("a", encoding="utf-8") as handle:
        handle.write(word + "\\n")


class HardExitingApp:
    """Omniverse's measured behaviour: closing Kit exits the process with 0."""

    def close(self):
        mark("close")
        os._exit(0)


# Proof the fake is a real hard exit and not a mock that returns: if `close`
# ran, nothing queued after it did, so this line never appends.
atexit.register(mark, "after-close")

from insight_bench.simulator.isaac import app as app_module

app_module._LIFECYCLE = app_module.IsaacAppLifecycle(launch=lambda args: HardExitingApp())
assert app_module.isaac_app().start() is True, "the fake Kit application did not come up"

ENTRY
'''

_RUN_CLI_ENTRY = """
from insight_bench.cli import run_cli

raise SystemExit(run_cli(sys.argv[2:]))
"""


def _run_driver(
    entry: str,
    argv: Sequence[str],
    *,
    mkdir: Sequence[str] = (),
    run_results: Sequence[str] = (),
) -> tuple[int, str, str]:
    """Run the hard-exiting-Kit driver with *entry* as its last statements.

    ``{root}`` in an argument is replaced with the temporary directory; every
    name in *mkdir* is created inside it first, and every name in *run_results*
    is written there as a real run result. Returns the exit status, stdout, and
    the marker text ("" when nothing wrote one).
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in mkdir:
            (root / name).mkdir(parents=True, exist_ok=True)
        for name in run_results:
            write_run_result(root / name)
        marker = root / "marker.txt"
        resolved = [arg.format(root=root.as_posix()) for arg in argv]
        completed = subprocess.run(
            [sys.executable, "-c", DRIVER.replace("ENTRY", entry), str(marker), *resolved],
            cwd=tmp,
            capture_output=True,
            text=True,
        )
        return (
            completed.returncode,
            completed.stdout,
            marker.read_text(encoding="utf-8") if marker.exists() else "",
        )


def _run_cli(argv: Sequence[str], **setup: Sequence[str]) -> tuple[int, str, str]:
    return _run_driver(_RUN_CLI_ENTRY, argv, **setup)


def test_a_failing_run_still_leaves_the_shell_a_failing_status() -> None:
    """Exit 1 stays exit 1.

    ``check-data`` answering "not ready" takes the same ``return 1`` in
    :func:`insight_bench.cli.main` that a non-completed ``RunResult`` takes; the
    release guard is command-agnostic and only ever sees that integer. Driven
    through the cheap command because a real ``run`` needs a simulator and this
    gate is about the integer, not about what produced it.
    """
    rc, _stdout, marker = _run_cli(
        ["check-data", "--episodes", "{root}/absent.jsonl", "--scene-root", "{root}/scenes"],
        mkdir=["scenes"],
    )
    assert rc == 1, f"a failing run exited {rc}; the Isaac close swallowed the status"
    assert marker.splitlines() == ["after-close"], f"Kit was released at exit 1: {marker!r}"


def test_an_exception_still_leaves_the_shell_a_failing_status() -> None:
    """Exit 2 stays exit 2: ``main`` caught the error and returned 2.

    Packing a run result that is not there raises inside ``main``, which is the
    ``except Exception`` path -- distinct from the argparse exit below, which
    never reaches that handler at all.
    """
    rc, _stdout, marker = _run_cli(["pack", "{root}/absent.json", "{root}/evidence"])
    assert rc == 2, f"an erroring run exited {rc}; the Isaac close swallowed the status"
    assert marker.splitlines() == ["after-close"], f"Kit was released at exit 2: {marker!r}"


def test_an_error_that_raises_past_main_still_fails() -> None:
    """``main`` can raise, not only return: argparse exits 2 from inside it.

    The release must therefore not sit in a ``finally``. A guard that ran while
    unwinding a raising ``main`` would hard-exit 0 and lose the usage error too.
    """
    rc, _stdout, marker = _run_cli([])
    assert rc == 2, f"a usage error exited {rc}"
    assert marker.splitlines() == ["after-close"], (
        f"a release path ran while unwinding an exception: {marker!r}"
    )


def test_a_successful_run_releases_kit_and_still_exits_zero() -> None:
    """The other half: on a clean 0 the release does happen, and output survives.

    A close that hard-exits skips the interpreter's own flush, so the guard
    flushes first; without that, this JSON would die with the process. The
    missing ``after-close`` line is what proves the close really did hard-exit,
    so the failing cases above are not passing on a mock that merely returns.
    """
    rc, stdout, marker = _run_cli(
        ["pack", "{root}/run-result.json", "{root}/evidence"],
        run_results=["run-result.json"],
    )
    assert rc == 0
    assert marker.splitlines() == ["close"], f"expected one close and nothing after it: {marker!r}"
    assert '"artifacts"' in stdout, "the result JSON was lost to the hard exit"


@pytest.mark.parametrize("code", [1, 2, 130])
def test_the_release_guard_never_closes_at_a_non_zero_code(code: int) -> None:
    """The guard called directly, with a hard-exiting application armed.

    Had it closed, this subprocess would exit 0 instead of *code*.
    """
    entry = (
        "from insight_bench.simulator.isaac.app import release_isaac_app_on_success\n\n"
        f"raise SystemExit(release_isaac_app_on_success({code}))\n"
    )
    rc, _stdout, marker = _run_driver(entry, [])
    assert rc == code
    assert marker.splitlines() == ["after-close"], (
        f"the guard closed Kit at exit {code}: {marker!r}"
    )
