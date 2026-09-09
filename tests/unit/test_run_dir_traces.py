"""The per-episode trace is written where the run directory says it is.

The path is ``<run_dir>/<episode_id>/trace.json``. That is the convention
``insight-bench vis`` reads back, the one an evidence reader is pointed at, and
the one :func:`insight_bench.runners._shared.run_scored_episode` writes through
``safe_join`` after every rollout -- driven end to end in
``tests/gate/test_insight_bench_runner.py``. These two tests pin the writer side
of it, which is where the original bug was: ``write_rollout_trace`` existed with
no callers at all, so the evidence a verifier reads was built in memory, handed
to scoring and dropped.

A third test here used to check that ``run_benchmark`` accepted a ``RunContext``
and that the text runner left nothing in it. Both the text runner and the
in-process adapter it took are gone, and with a single simulator runner left
there is no runner for which "ignores the artifact directory" is the correct
behaviour, so that case was deleted rather than rewritten.
"""

from __future__ import annotations

import json
from pathlib import Path

from insight_bench.vln_runtime.traces.schema import RolloutTrace
from insight_bench.vln_runtime.traces.writer import write_rollout_trace


def _trace(episode_id: str) -> RolloutTrace:
    return RolloutTrace.from_camera_walk_records(
        episode_id=episode_id,
        instruction="go",
        records=[
            {
                "step": 0,
                "pose_before": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
                "pose_after": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
                "measures": {},
                "observation": {"rgb": {"dtype": "uint8", "shape": [4, 6, 3]}},
                "metadata": {"terrain_query_available": False, "terrain_raycast_hit": False},
            }
        ],
    )


def test_the_documented_path_is_the_one_written(tmp_path: Path) -> None:
    episode_id = "objnav_00013-sfbj7jspYWj_4"
    path = write_rollout_trace(tmp_path / episode_id / "trace.json", _trace(episode_id))
    assert path == tmp_path / episode_id / "trace.json"
    assert path.is_file()


def test_a_stand_still_episode_is_a_valid_trace_not_an_empty_one(tmp_path: Path) -> None:
    """An identical pose on every step is the correct stand-still outcome, not a missing run.

    This is the case the first version of the acceptance got wrong by requiring distinct poses, so
    it is worth pinning: the trace still has steps, positions and orientations.
    """
    trace = _trace("ep-still")
    path = write_rollout_trace(tmp_path / "ep-still" / "trace.json", trace)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["steps"], "a stand-still episode still records steps"
    positions = [tuple(step["position"]) for step in payload["steps"]]
    assert len(set(positions)) == 1, "stand-still means one position, and that is allowed"
