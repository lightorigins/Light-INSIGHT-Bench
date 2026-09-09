"""The loop closes: what ``run --vis`` writes is what ``insight-bench vis`` reads.

Both halves are asserted separately elsewhere -- the write side in
``test_vis_write_side.py``, the read side in ``tests/unit/test_vis.py`` -- with
fixtures each half wrote for itself. That leaves the one thing neither can catch:
a disagreement about the contract between them. The file names the sink chooses,
the step a frame path lands on, the keys in ``episode.json`` and the stride in
``vis-manifest.json`` are all crossing a seam here, so this drives the real
producers and then renders the real page.

The simulator is not involved: the rollout's trace builder is, which is the
component that decides which step owns a frame.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from insight_bench.runners._shared import VisSession
from insight_bench.vis import render_run_page
from insight_bench.vln_runtime.episodes import EpisodeSpec
from insight_bench.vln_runtime.episodes.scenes import bind_scene_asset
from insight_bench.vln_runtime.traces.frames import JpegFrameSink
from insight_bench.vln_runtime.traces.schema import RolloutTrace
from insight_bench.vln_runtime.traces.writer import write_rollout_trace

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "objnav_published_episodes.jsonl"
DECISIONS = 3


def _episode_spec(scene_root: Path | None = None) -> EpisodeSpec:
    """The first published-fixture episode, with its scene asset bound.

    Binding matters here: it is what puts an absolute host path on the spec, so
    without it the "no host path on the page" assertion below would pass for the
    wrong reason.
    """
    record = json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[0])
    episode = EpisodeSpec.from_dict(record)
    if scene_root is None:
        return episode
    return bind_scene_asset(episode, scene_root=scene_root)


def _pose(index: int) -> dict[str, float]:
    return {"x": -8.0 + index * 0.4, "y": 0.3 + index * 0.2, "z": -0.16, "yaw": 0.2 * index}


def _round_trip(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    root.mkdir(parents=True, exist_ok=True)
    scene_root = root.parent / "scenes"
    (scene_root / "hm3d" / "00013-sfbj7jspYWj").mkdir(parents=True, exist_ok=True)
    (scene_root / "hm3d" / "00013-sfbj7jspYWj" / "sfbj7jspYWj.usd").write_bytes(b"usd")
    episode = _episode_spec(scene_root)
    assert episode.scene.asset_path and episode.scene.asset_path.startswith("/")
    sink = JpegFrameSink(root=root, stride=1)
    session = VisSession(artifact_dir=root, sink=sink, episode_count=1, max_frames_per_episode=100)
    session.write_episode(episode)

    frame = np.zeros((270, 480, 3), dtype=np.uint8)
    frame[:, :, 1] = 128
    records = []
    for decision in range(DECISIONS):
        records.append(
            {
                "step": decision,
                "pose_before": _pose(decision),
                "pose_after": _pose(decision + 1),
                "measures": {"distance_to_goal": 4.0 - decision},
                "frame_path": sink(episode.episode_id, decision, frame),
                "raw_output": '<point x="812" y="344">table</point>',
                "apos_id": 1273,
                "apos_kind": 0,
                "selected_waypoint": {"forward_m": 0.4, "lateral_m": 0.1, "yaw_rad": 0.05},
            }
        )
    trace = RolloutTrace.from_camera_walk_records(
        episode_id=episode.episode_id,
        instruction=episode.instruction,
        records=records,
        termination_reason="stop",
        stop_step=DECISIONS - 1,
    )
    write_rollout_trace(root / episode.episode_id / "trace.json", trace)
    session.write_manifest()

    (root / "run-result.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "run_id": "run-0123456789abcdef",
                "run_fingerprint": "d" * 64,
                "benchmark_id": "insight-bench-v1",
                "benchmark_version": "2.0.0",
                "adapter": {"adapter_id": "http-policy", "model_id": "demo/model"},
                "seed": 0,
                "status": "completed",
                "episodes": [
                    {
                        "episode_id": episode.episode_id,
                        "task_id": "objnav",
                        "status": "passed",
                        "metrics": {"success": 1.0, "distance_to_goal": 1.4, "stopped": 1.0},
                        "response": {
                            "schema_version": "1.0.0",
                            "request_id": "req-0123456789abcdef",
                            "adapter": {"adapter_id": "http-policy", "model_id": "demo/model"},
                            "status": "success",
                            "output": "stop",
                            "metadata": {"score_labels": {"capability": "LR_towards"}},
                        },
                    }
                ],
                "metrics": {"success-rate": 1.0, "distance_to_goal": 1.4},
            }
        ),
        encoding="utf-8",
    )
    summary = render_run_page(root)
    html = (root / "vis.html").read_text(encoding="utf-8")
    marker = '<script type="application/json" id="insight-bench-data">'
    start = html.index(marker) + len(marker)
    return summary, json.loads(html[start : html.index("</script>", start)])


def test_the_page_finds_every_frame_the_sink_wrote(tmp_path):
    run_dir = tmp_path / "run"
    summary, data = _round_trip(run_dir)
    episode_id = _episode_spec().episode_id
    assert summary["frames"] == DECISIONS
    assert summary["episodes_with_frames"] == 1
    assert summary["episodes_with_replay"] == 1
    assert summary["warnings"] == [], summary["warnings"]

    entry = data["replay"][episode_id]
    # The numbering contract, end to end: decision k is trace step k + 1, and
    # step_0000.jpg does not exist.
    assert entry["step"] == [1, 2, 3]
    assert entry["frame"] == [f"{episode_id}/frames/step_{index:04d}.jpg" for index in (1, 2, 3)]
    for reference in entry["frame"]:
        assert (run_dir / reference).is_file(), reference
    assert not (run_dir / episode_id / "frames" / "step_0000.jpg").exists()


def test_the_page_reads_the_stride_and_the_categories_the_run_recorded(tmp_path):
    summary, data = _round_trip(tmp_path / "run")
    assert summary["frame_stride"] == 1
    assert summary["episodes_with_categories"] == 1

    breakdown = data["breakdown"]
    assert breakdown is not None
    # The fixture's first episode is instr_type=Direction, scene_class=House.
    assert [axis["label"] for axis in breakdown["instr_type"]] == ["Direction"]
    assert [axis["label"] for axis in breakdown["scene_class"]] == ["House"]
    assert breakdown["matrix"]["grand_total"]["episodes"] == 1


def test_the_page_never_shows_the_host_path_of_the_scene_asset(tmp_path):
    _, data = _round_trip(tmp_path / "run")
    blob = json.dumps(data, ensure_ascii=False)
    assert "asset_path" not in blob
    assert ".usd" not in blob
    assert str(tmp_path) not in blob
    entry = data["replay"][_episode_spec().episode_id]
    # The geometry the top-down panel needs did come through.
    assert entry["goal"] and entry["reference_path"]


def test_the_pointing_tokens_do_not_reach_the_page(tmp_path):
    """A real policy emits them; the page carries none of it.

    They were printed over the frame beside a marker decoded from them, and the
    decode was never verified against a pointing policy. Marker, caveat and
    values are all gone, so a whole trace's worth of them is dead weight in the
    payload. The trace on disk beside the page still holds every field.
    """
    run_dir = tmp_path / "run"
    _, data = _round_trip(run_dir)
    entry = data["replay"][_episode_spec().episode_id]

    for absent in ("apos_id", "apos_xy", "opos_id", "opos_xy", "raw"):
        assert absent not in entry, absent

    # ... and the policy really did emit them, so this is the page dropping
    # them rather than the fixture never producing any.
    trace = json.loads(
        (run_dir / _episode_spec().episode_id / "trace.json").read_text(encoding="utf-8")
    )
    emitted = [step.get("metadata", {}).get("apos_id") for step in trace["steps"]]
    assert 1273 in emitted
