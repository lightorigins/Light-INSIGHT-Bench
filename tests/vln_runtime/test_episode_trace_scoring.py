from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from insight_bench.vln_runtime.episodes import EpisodeSpec, Pose3D, SceneSpec
from insight_bench.vln_runtime.managers import MeasureTerm
from insight_bench.vln_runtime.measures.manager import MeasureManagerCfg, build_measure_manager
from insight_bench.vln_runtime.scoring import navigation_terms
from insight_bench.vln_runtime.scoring.manager import ScoreManager
from insight_bench.vln_runtime.scoring.metrics import evaluate_navigation
from insight_bench.vln_runtime.scoring.term_cfg import ScoreTerm
from insight_bench.vln_runtime.traces import RolloutTrace, StepTrace


def _episode() -> EpisodeSpec:
    return EpisodeSpec(
        episode_id="ep-1",
        instruction="go to the goal",
        scene=SceneSpec(scene_id="scene-1", asset_path="/scene/scene.usdz", scene_type="hm3d_usd"),
        start_pose=Pose3D.from_xy_yaw(0.0, 0.0, 0.5, 0.0),
        goal_position=(2.0, 0.0, 0.5),
        goal_radius_m=0.25,
    )


def test_pose3d_from_xy_yaw_uses_wxyz_quaternion() -> None:
    pose = Pose3D.from_xy_yaw(1.0, 2.0, 0.5, math.pi)

    assert pose.position == pytest.approx((1.0, 2.0, 0.5))
    assert pose.orientation_wxyz[0] == pytest.approx(0.0, abs=1e-6)
    assert pose.orientation_wxyz[3] == pytest.approx(1.0)


def test_navigation_scoring_success_with_stop() -> None:
    trace = RolloutTrace(
        episode_id="ep-1",
        steps=[
            StepTrace(step=0, time_s=0.0, position=(0.0, 0.0, 0.5)),
            StepTrace(step=1, time_s=1.0, position=(1.0, 0.0, 0.5)),
            StepTrace(step=2, time_s=2.0, position=(2.0, 0.0, 0.5)),
        ],
        stop_step=2,
    )

    score = evaluate_navigation(_episode(), trace)

    assert score.sr is True
    assert score.osr == pytest.approx(1.0)
    assert score.ne == pytest.approx(0.0)
    assert score.path_length == pytest.approx(2.0)
    assert score.spl == pytest.approx(1.0)
    assert score.failure_reason is None


def test_navigation_scoring_requires_stop_for_sr_but_keeps_osr() -> None:
    trace = RolloutTrace(
        episode_id="ep-1",
        steps=[
            StepTrace(step=0, time_s=0.0, position=(0.0, 0.0, 0.5)),
            StepTrace(step=1, time_s=1.0, position=(2.0, 0.0, 0.5)),
        ],
        stop_step=-1,
    )

    score = evaluate_navigation(_episode(), trace, require_stop=True)
    relaxed_score = evaluate_navigation(_episode(), trace, require_stop=False)

    assert score.sr is False
    assert score.osr == pytest.approx(1.0)
    assert score.failure_reason == "stop_not_requested"
    assert relaxed_score.sr is True


def test_camera_walk_records_include_initial_pose_for_path_length() -> None:
    records = [
        {
            "step": 0,
            "pose_before": {"x": 0.0, "y": 0.0, "z": 0.5, "yaw": 0.0},
            "pose_after": {"x": 1.0, "y": 0.0, "z": 0.5, "yaw": 0.0},
            "selected_waypoint": {"forward_m": 1.0, "lateral_m": 0.0, "yaw_rad": 0.0},
            "frame_path": "/tmp/frame_0000.png",
        },
        {
            "step": 1,
            "pose_before": {"x": 1.0, "y": 0.0, "z": 0.5, "yaw": 0.0},
            "pose_after": {"x": 2.0, "y": 0.0, "z": 0.5, "yaw": 0.0},
            "selected_waypoint": {"forward_m": 1.0, "lateral_m": 0.0, "yaw_rad": 0.0},
            "frame_path": "/tmp/frame_0001.png",
            "measures": {"goal_distance_xy": 0.0},
        },
    ]

    trace = RolloutTrace.from_camera_walk_records(
        episode_id="ep-1",
        instruction="go forward",
        records=records,
    )

    assert [step.step for step in trace.steps] == [0, 1, 2]
    assert trace.path_length_xy() == pytest.approx(2.0)
    assert trace.steps[0].action == {}
    assert trace.steps[-1].observation["frame_path"] == "/tmp/frame_0001.png"
    assert trace.steps[-1].measures["goal_distance_xy"] == pytest.approx(0.0)


def test_score_manager_exports_configured_metrics() -> None:
    trace = RolloutTrace(
        episode_id="ep-1",
        steps=[
            StepTrace(step=0, time_s=0.0, position=(0.0, 0.0, 0.5)),
            StepTrace(step=1, time_s=1.0, position=(2.0, 0.0, 0.5)),
        ],
        stop_step=1,
    )
    manager = ScoreManager(
        {
            "success_rate": ScoreTerm(func=navigation_terms.sr, params={"require_stop": True}),
            "nav_error": ScoreTerm(func=navigation_terms.ne),
        }
    )

    score = manager.evaluate_episode(_episode(), trace)

    assert score.metrics["success_rate"] is True
    assert score.metrics["nav_error"] == pytest.approx(0.0)


def test_measure_manager_collects_online_evidence_from_trace_context() -> None:
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        episode=_episode(),
        position=(2.0, 0.0, 0.5),
        positions=[(0.0, 0.0, 0.5), (1.0, 0.0, 0.5), (2.0, 0.0, 0.5)],
        episode_length_buf=[2],
    )
    manager = build_measure_manager(MeasureManagerCfg.default_navigation(), env)

    measures = manager.compute()

    assert measures["goal_distance_xy"] == pytest.approx(0.0)
    assert measures["path_length_xy"] == pytest.approx(2.0)
    assert measures["oracle_goal_distance_xy"] == pytest.approx(0.0)
    assert measures["step_count"] == 2


def test_measure_manager_uses_cfg_field_name_as_term_name() -> None:
    env = SimpleNamespace(num_envs=1, device="cpu")
    manager = build_measure_manager({"custom_measure": MeasureTerm(func=lambda env: 42)}, env)

    assert manager.active_terms == ["custom_measure"]
    assert manager.compute() == {"custom_measure": 42}


def test_measure_manager_validates_env_argument_and_required_params() -> None:
    env = SimpleNamespace(num_envs=1, device="cpu")

    with pytest.raises(ValueError, match="environment as its first argument"):
        build_measure_manager({"bad": MeasureTerm(func=lambda: 1)}, env)

    def needs_scale(env, scale: float) -> float:
        del env
        return float(scale)

    with pytest.raises(ValueError, match="missing mandatory params"):
        build_measure_manager({"needs_scale": MeasureTerm(func=needs_scale)}, env)
