"""``vis`` must work on a bare interpreter, and must not smuggle a dependency in.

The read side is the one part of the visualisation that a reader with no
simulator and no GPU runs: they were handed a run directory and want to look at
it. So every third-party module the run side reaches for -- the distribution's
own numpy, Pillow and requests, and the web stack that is not a dependency of
anything here and must stay that way -- is blocked at import time, and the
command has to work anyway. It is the read-side twin of the import-purity gate
in ``test_independence.py``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

BLOCKED = ("numpy", "requests", "PIL", "yaml", "fastapi", "uvicorn")

SCRIPT = f"""
import builtins, json, sys
BLOCKED = {BLOCKED!r}
_real_import = builtins.__import__

def guarded(name, *args, **kwargs):
    root = name.split(".")[0]
    if root in BLOCKED:
        raise ModuleNotFoundError(f"No module named {{root!r}}")
    return _real_import(name, *args, **kwargs)

builtins.__import__ = guarded
from insight_bench.cli import main
assert main(["vis", sys.argv[1]]) == 0
leaked = [m for m in sys.modules if m.split(".")[0] in BLOCKED]
assert not leaked, leaked
print("vis base-install ok")
"""


def _run_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "ep-a").mkdir()
    (root / "ep-a" / "trace.json").write_text(
        json.dumps(
            {
                "episode_id": "ep-a",
                "instruction": "go",
                "steps": [
                    {
                        "step": index,
                        "time_s": float(index),
                        "position": [float(index), 0.0, 0.0],
                        "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
                        "measures": {"distance_to_goal": 3.0 - index},
                        "observation": {"frame_path": ""},
                        "metadata": {},
                        "action": {},
                    }
                    for index in range(3)
                ],
                "final_measures": {},
                "termination_reason": "stop",
                "stop_step": 2,
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )
    (root / "run-result.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "run_id": "run-0123456789abcdef",
                "run_fingerprint": "b" * 64,
                "benchmark_id": "insight-bench-v1",
                "benchmark_version": "2.0.0",
                "adapter": {"adapter_id": "http-policy", "model_id": "demo/model"},
                "seed": 0,
                "status": "completed",
                "episodes": [
                    {
                        "episode_id": "ep-a",
                        "task_id": "objnav",
                        "status": "passed",
                        "metrics": {"spl": 0.5, "distance_to_goal": 0.4, "stopped": 1.0},
                    }
                ],
                "metrics": {"success-rate": 1.0, "spl": 0.5},
            }
        ),
        encoding="utf-8",
    )
    return root


def test_vis_renders_without_the_run_side_dependencies(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path / "run")
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(run_dir)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "vis base-install ok" in result.stdout
    assert (run_dir / "vis.html").is_file()


def test_the_page_module_is_a_template_and_nothing_else() -> None:
    """``vis_page`` holds one string. Anything else there would be page logic
    living outside the module that is type-checked and tested for it."""
    from insight_bench import vis_page

    exported = [name for name in vars(vis_page) if not name.startswith("_")]
    assert exported == ["annotations", "Final", "PAGE_TEMPLATE"]
