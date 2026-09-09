"""``python -m insight_bench`` -- the console script's module form.

Isaac Lab is driven through a wrapper (``./isaaclab.sh -p``) that runs its own
interpreter, and a console script installed into that interpreter's ``bin`` is
not on ``PATH``. Module invocation is the form that works there, so it is the
one the README documents for a simulator run.
"""

from __future__ import annotations

from insight_bench.cli import run_cli

# `raise SystemExit(...)`, not a bare call: the console script's generated
# wrapper passes the return value to `sys.exit`, but module execution discards
# it. Without this the two invocations disagree -- `insight-bench run` failing
# exits 2, `python -m insight_bench run` failing exits 0 -- and the module form
# is the one the README recommends, so a failing run in CI looked like a pass.
raise SystemExit(run_cli())
