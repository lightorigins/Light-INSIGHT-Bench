"""The simulator-neutral episode loop, exercised without a simulator.

The rollout takes a backend and a policy as protocols precisely so this can
run in CI: a fake free-camera world plus a scripted policy reproduce every
branch the Isaac backend will take.

The task fixture is the published objnav suite's own config -- 480x270 at HFOV
120, a 300-step budget, a 0.1 s control period and the suite's three
terminations -- because that is the only rig this SDK runs. The episode is a
verbatim record from the published fixture file, parsed but not scene-bound: no
scene is opened here, and binding is tested where it lives, in
``test_objnav_scene_binding.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from insight_bench.vln_runtime.episodes import EpisodeSpec
from insight_bench.vln_runtime.episodes.loader import load_episode_records
from insight_bench.vln_runtime.motion.pose import CameraPose
from insight_bench.vln_runtime.policy.client import PolicyResponse
from insight_bench.vln_runtime.rollout import (
    POSE_SOURCE_COMMANDED_FALLBACK,
    POSE_SOURCE_READBACK,
    STRUCTURELESS_FRAME_EDGE,
    STRUCTURELESS_FRAME_STD,
    EpisodeRollout,
    PoseReadbackError,
    RolloutOptions,
    describe_blank_scene_frame,
    frame_structure_energy,
    run_camera_walk_episode,
)
from insight_bench.vln_runtime.scoring import ScoreManager
from insight_bench.vln_runtime.suite import get_task_config

PUBLISHED_EPISODES = (
    Path(__file__).resolve().parents[1] / "fixtures" / "objnav_published_episodes.jsonl"
)

# The one fixture episode whose object sits straight ahead of the start heading
# (4.8 m away, bearing 0.1 deg). "Drive forward" is therefore real progress
# toward the goal, which is what lets the scoring tests below reach the success
# radius without the fake world ever having to steer.
FORWARD_EPISODE = "objnav_v2_human_manual_8194nk5LbLH_49"


class FakeCameraWalkBackend:
    """A flat, obstacle-free world that renders a frame encoding its pose."""

    def __init__(self, *, floor_z: float | None = None, blank: bool = False):
        self._floor_z = floor_z
        self._blank = blank
        self.pose = CameraPose(0.0, 0.0, 0.0, 0.0)
        self.steps = 0
        self.reset_calls: list[dict[str, Any]] = []

    def resolve_start_pose(self, pose: CameraPose) -> CameraPose:
        if self._floor_z is None:
            return pose
        return CameraPose(pose.x, pose.y, self._floor_z, pose.yaw)

    def resolve_motion(self, prev: CameraPose, candidate: CameraPose) -> CameraPose:
        if self._floor_z is None:
            return candidate
        return CameraPose(candidate.x, candidate.y, self._floor_z, candidate.yaw)

    def reset(self, pose, *, warmup_steps, converge_max_steps, converge_tol, log):
        self.reset_calls.append(
            {
                "warmup_steps": warmup_steps,
                "converge_max_steps": converge_max_steps,
                "converge_tol": converge_tol,
            }
        )
        self.pose = pose
        return self._frame()

    def set_pose(self, pose: CameraPose) -> None:
        self.pose = pose

    def step_and_capture(self, *, settle_steps: int) -> np.ndarray:
        self.steps += settle_steps
        return self._frame()

    def _frame(self) -> np.ndarray:
        if self._blank:
            return np.zeros((4, 4, 3), dtype=np.uint8)
        rng = np.random.default_rng(abs(int(self.pose.x * 1000)) % 2**32)
        return rng.integers(0, 256, size=(4, 4, 3), dtype=np.uint8)


class ScriptedPolicy:
    """Emits forward waypoints, then whatever the caller scripted as the tail."""

    def __init__(self, *, forward_m: float = 0.25, stop_after: int, tail: str = "zero"):
        self.forward_m = forward_m
        self.stop_after = stop_after
        self.tail = tail
        self.reset_calls: list[str] = []
        self.finish_calls: list[str] = []
        self.frames_seen = 0

    def reset(self, instruction, *, episode_id, first_frame_rgb):
        self.reset_calls.append(instruction)
        # The rollout deliberately sends no frame here: the first `act` delivers
        # it, and sending it twice made a stateful adapter count step 0 twice.
        assert first_frame_rgb is None
        return {"success": True}

    def act(self, rgb_frame, *, episode_id, frame_id):
        self.frames_seen += 1
        if frame_id < self.stop_after:
            return _waypoint_response(frame_id, forward_m=self.forward_m)
        if self.tail == "stop_token":
            return _waypoint_response(frame_id, forward_m=0.0, raw_output="stop")
        return _waypoint_response(frame_id, forward_m=0.0)

    def finish(self, *, episode_id):
        self.finish_calls.append(episode_id)
        return {"success": True}


def _waypoint_response(frame_id: int, *, forward_m: float, raw_output: str = "") -> PolicyResponse:
    return PolicyResponse(
        success=True,
        action_name="WAYPOINT",
        raw_output=raw_output,
        waypoint={
            "cluster_id": 0,
            "waypoints": [
                {"forward_m": forward_m, "lateral_m": 0.0, "yaw_rad": 0.0} for _ in range(4)
            ],
        },
        waypoint_cluster_id=0,
        waypoint_horizon=4,
        frame_id=frame_id,
    )


def _episode(episode_id: str = FORWARD_EPISODE):
    task_config = get_task_config("insight_bench")
    records = {record["episode_id"]: record for record in load_episode_records(PUBLISHED_EPISODES)}
    return task_config, EpisodeSpec.from_dict(records[episode_id])


def _run(policy: ScriptedPolicy, **backend_kwargs: Any) -> tuple[EpisodeRollout, Any, Any]:
    task_config, episode = _episode()
    env = FakeCameraWalkBackend(**backend_kwargs)
    rollout = run_camera_walk_episode(
        env=env,
        policy=policy,
        episode=episode,
        task_config=task_config,
        options=RolloutOptions.from_task_config(task_config),
    )
    return rollout, env, episode


def test_a_stop_token_terminates_the_episode_immediately() -> None:
    policy = ScriptedPolicy(stop_after=5, tail="stop_token")
    rollout, _env, episode = _run(policy)

    assert rollout.termination_reason == "policy_stop"
    assert rollout.stop_step == 5
    assert rollout.steps_taken == 6
    assert policy.reset_calls == [episode.instruction]
    # finish() always runs, even though the loop exited on a termination.
    assert policy.finish_calls == [episode.episode_id]
    # The stop step must not move the camera.
    assert rollout.records[-1]["pose_before"] == rollout.records[-1]["pose_after"]
    assert rollout.records[-1]["waypoint_execution"] == {"mode": "policy_stop"}


def test_forward_waypoints_integrate_into_real_motion() -> None:
    policy = ScriptedPolicy(forward_m=0.25, stop_after=4, tail="stop_token")
    rollout, _env, episode = _run(policy)

    # The start yaw comes out of the record's quaternion, so "forward" is not an
    # axis and the displacement is what can be measured. 0.25 m of waypoint over
    # a 0.1 s control period clamps to 2.5 m/s, i.e. exactly 0.25 m per step.
    start_x, start_y, _start_z = episode.start_pose.position
    after = rollout.records[3]["pose_after"]
    moved = math.hypot(after["x"] - start_x, after["y"] - start_y)
    assert moved == pytest.approx(4 * 0.25, abs=1e-6)
    assert rollout.records[0]["velocity_command"]["linear_velocity_mps"] == pytest.approx(2.5)


def test_a_zero_velocity_command_stops_the_episode_at_once() -> None:
    # The objnav suite declares zero_velocity_stop as the single-step term: a
    # waypoint policy signals stop by emitting its zero-motion cluster, so the
    # first near-zero command ends the episode rather than opening a countdown.
    policy = ScriptedPolicy(stop_after=2, tail="zero")
    rollout, _env, _episode = _run(policy)

    assert rollout.termination_reason == "zero_velocity_stop"
    assert rollout.stop_step == 2
    assert rollout.steps_taken == 3


def test_a_duration_configured_zero_velocity_stop_waits_out_its_duration() -> None:
    # The other half of the same seam: the termination manager is built once and
    # driven across steps, so a term that accumulates state accumulates it. A
    # rollout that rebuilt the manager per step would stop at the first zero
    # command here and look identical to the test above.
    from dataclasses import replace

    from insight_bench.vln_runtime.managers import DoneTerm
    from insight_bench.vln_runtime.terminations import terms as termination_terms

    task_config, episode = _episode()
    task_config = replace(
        task_config,
        terminations={
            "zero_velocity_stop": DoneTerm(
                func=termination_terms.VelocityControlBelowThresholdForDuration,
                params={"duration_sec": 4.0},
            )
        },
    )
    rollout = run_camera_walk_episode(
        env=FakeCameraWalkBackend(),
        policy=ScriptedPolicy(stop_after=2, tail="zero"),
        episode=episode,
        task_config=task_config,
        options=RolloutOptions.from_task_config(task_config),
    )

    # 4.0 s at a 0.1 s control period: 40 consecutive near-zero commands, not one.
    assert rollout.termination_reason == "zero_velocity_stop"
    assert rollout.stop_step == 2 + 40 - 1
    assert rollout.steps_taken == rollout.stop_step + 1


def test_the_trace_carries_the_initial_pose_and_every_step() -> None:
    policy = ScriptedPolicy(stop_after=3, tail="stop_token")
    rollout, _env, episode = _run(policy)

    trace = rollout.trace
    assert trace.episode_id == episode.episode_id
    assert trace.instruction == episode.instruction
    # One extra step for the pre-action starting pose.
    assert len(trace.steps) == rollout.steps_taken + 1
    assert trace.steps[0].position == pytest.approx(episode.start_pose.position)
    assert trace.termination_reason == "policy_stop"
    assert trace.stop_step == rollout.stop_step
    assert trace.metadata["task"] == "insight_bench"


def test_the_rollout_scores_through_the_task_score_manager() -> None:
    task_config, episode = _episode()
    # 14 steps of forward motion is 3.5 m, which enters the episode's 2.0 m
    # success radius with the object still inside the camera's FOV.
    policy = ScriptedPolicy(stop_after=14, tail="stop_token")
    env = FakeCameraWalkBackend()
    rollout = run_camera_walk_episode(
        env=env,
        policy=policy,
        episode=episode,
        task_config=task_config,
        options=RolloutOptions.from_task_config(task_config),
    )
    score = ScoreManager(task_config.scoring).evaluate_episode(
        episode, rollout.trace, final_measures=rollout.final_measures, task_config=task_config
    )
    assert set(score.metrics) == {
        "success",
        "spl",
        "distance_to_goal",
        "path_length",
        "stop_step",
        "capability",
        "stopped",
    }
    # The geometry above is the assertion, not a comment: a rollout that reached
    # the radius and stopped there is an LR-towards success.
    assert score.metrics["success"] is True
    assert score.metrics["capability"] == "LR_towards"
    assert score.metrics["path_length"] > 0.0
    assert math.isfinite(float(score.metrics["distance_to_goal"]))


def test_the_step_budget_is_reported_as_the_configured_time_out() -> None:
    # objnav declares a time_out term, so exhausting num_steps is a real
    # termination with a name, not the anonymous fallback.
    task_config, episode = _episode()
    policy = ScriptedPolicy(stop_after=10_000, tail="forward")
    env = FakeCameraWalkBackend()
    options = RolloutOptions.from_task_config(task_config)
    rollout = run_camera_walk_episode(
        env=env, policy=policy, episode=episode, task_config=task_config, options=options
    )
    assert rollout.steps_taken == options.num_steps
    assert rollout.termination_reason == "time_out"
    assert rollout.termination["time_out"] is True
    assert rollout.stop_step == -1


def test_the_fallback_reason_covers_a_task_with_no_terminating_term() -> None:
    from dataclasses import dataclass, field, replace

    from insight_bench.vln_runtime.managers import DoneTerm, TerminationTermCfg
    from insight_bench.vln_runtime.terminations import terms as termination_terms

    @dataclass
    class OnlyPolicyStopCfg:
        policy_stop: TerminationTermCfg = field(
            default_factory=lambda: DoneTerm(func=termination_terms.policy_stop_requested)
        )

    task_config, episode = _episode()
    task_config = replace(task_config, terminations=OnlyPolicyStopCfg())
    options = RolloutOptions.from_task_config(task_config)
    rollout = run_camera_walk_episode(
        env=FakeCameraWalkBackend(),
        policy=ScriptedPolicy(stop_after=10_000, tail="forward"),
        episode=episode,
        task_config=task_config,
        options=options,
    )
    assert rollout.steps_taken == options.num_steps
    assert rollout.termination_reason == "num_steps_exhausted"
    assert rollout.termination["done"] is False


def test_the_start_pose_is_snapped_onto_the_backend_floor() -> None:
    policy = ScriptedPolicy(stop_after=1, tail="stop_token")
    rollout, _env, _episode = _run(policy, floor_z=1.25)
    assert rollout.records[0]["pose_before"]["z"] == pytest.approx(1.25)


def test_reset_forwards_the_render_convergence_budget() -> None:
    task_config, episode = _episode()
    env = FakeCameraWalkBackend()
    options = RolloutOptions.from_task_config(task_config)
    run_camera_walk_episode(
        env=env,
        policy=ScriptedPolicy(stop_after=0, tail="stop_token"),
        episode=episode,
        task_config=task_config,
        options=options,
    )
    assert env.reset_calls == [
        {
            "warmup_steps": options.warmup_steps,
            "converge_max_steps": options.render_converge_max_steps,
            "converge_tol": options.render_converge_tol,
        }
    ]


def test_a_blank_frame_is_described_because_the_suite_requires_real_assets() -> None:
    from dataclasses import replace

    task_config, _episode_spec = _episode()
    blank = np.zeros((8, 8, 3), dtype=np.uint8)

    # The objnav scenes are user-provided USD stages, so the suite declares
    # require_scene_assets: a stage that failed to load renders flat, and a
    # reader has to be able to tell that from a model that simply did badly.
    note = describe_blank_scene_frame(blank, task_config=task_config, episode_id="e0")
    assert note is not None and "appears blank" in note

    # A task that declares no asset requirement may legitimately render a
    # near-uniform frame, and nothing there is verifiable against.
    procedural = replace(task_config, assets={**task_config.assets, "require_scene_assets": False})
    assert describe_blank_scene_frame(blank, task_config=procedural, episode_id="e0") is None


def test_a_blank_first_frame_is_recorded_and_the_episode_still_runs() -> None:
    """One bad start pose must not cost the whole suite its submittability.

    A blank first frame used to raise, which the runner turned into an errored
    episode and a `partial` run. One real episode of the published suite starts
    half a metre from a bare wall, so that single pose was enough to make every
    full run unsubmittable. It is scored like any other episode now -- a policy
    shown nothing fails it unaided -- and the note survives in the trace, which
    is what still distinguishes this from a scene that never loaded.
    """
    task_config, episode = _episode()
    policy = ScriptedPolicy(stop_after=1, tail="stop_token")

    rollout = run_camera_walk_episode(
        env=FakeCameraWalkBackend(blank=True),
        policy=policy,
        episode=episode,
        task_config=task_config,
        options=RolloutOptions.from_task_config(task_config),
    )

    assert policy.reset_calls == [episode.instruction]
    warning = rollout.trace.metadata["first_frame_warning"]
    assert "appears blank" in warning


def test_a_first_frame_with_content_leaves_no_warning_behind() -> None:
    task_config, episode = _episode()

    rollout = run_camera_walk_episode(
        env=FakeCameraWalkBackend(),
        policy=ScriptedPolicy(stop_after=1, tail="stop_token"),
        episode=episode,
        task_config=task_config,
        options=RolloutOptions.from_task_config(task_config),
    )

    assert "first_frame_warning" not in rollout.trace.metadata


def test_a_policy_failure_still_finishes_the_episode_on_the_policy_side() -> None:
    task_config, episode = _episode()

    class ExplodingPolicy(ScriptedPolicy):
        def act(self, rgb_frame, *, episode_id, frame_id):
            raise RuntimeError("model crashed")

    policy = ExplodingPolicy(stop_after=0)
    with pytest.raises(RuntimeError, match="model crashed"):
        run_camera_walk_episode(
            env=FakeCameraWalkBackend(),
            policy=policy,
            episode=episode,
            task_config=task_config,
            options=RolloutOptions.from_task_config(task_config),
        )
    assert policy.finish_calls == [episode.episode_id]


# --- the pose a step reports -----------------------------------------------
#
# The bug this section pins: every step wrote `next_pose` -- the *resolved command* --
# into positions, state, measures, scoring, records and the trace, while the only call
# that read the simulator back lived in the observation summary and had its exceptions
# swallowed. On a real L20 job `read_camera_pose` raised on every step; the trace filled
# up with commands, scored perfectly against itself, and the run read as green.


class ReadbackBackend(FakeCameraWalkBackend):
    """A world whose camera lands somewhere other than where it was sent.

    The offset is fixed and deliberately large. A readback that merely *equals* the
    command cannot tell the two apart, so it would pass the old code and the new code
    alike; only a backend that disagrees with its command can prove which one the
    rollout wrote down.
    """

    OFFSET_X = 0.5
    OFFSET_YAW = 0.25

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.captures = 0

    def _actual(self) -> CameraPose:
        return CameraPose(
            self.pose.x + self.OFFSET_X,
            self.pose.y,
            self.pose.z,
            self.pose.yaw + self.OFFSET_YAW,
        )

    def capture_observation(self):
        from insight_bench.simulator.observation import make_observation, quat_wxyz_from_yaw

        self.captures += 1
        actual = self._actual()
        return make_observation(
            self._frame(),
            pose=actual,
            orientation_wxyz=quat_wxyz_from_yaw(actual.yaw),
            depth=None,
            intrinsics=None,
            frame_id=self.captures,
        )


def _run_backend(env, policy: ScriptedPolicy) -> EpisodeRollout:
    task_config, episode = _episode()
    return run_camera_walk_episode(
        env=env,
        policy=policy,
        episode=episode,
        task_config=task_config,
        options=RolloutOptions.from_task_config(task_config),
    )


def test_the_record_reports_the_readback_pose_not_the_command() -> None:
    env = ReadbackBackend()
    rollout = _run_backend(env, ScriptedPolicy(stop_after=3, tail="stop_token"))

    record = rollout.records[0]
    commanded = record["metadata"]["commanded_pose"]
    assert record["pose_after"]["x"] == pytest.approx(commanded["x"] + env.OFFSET_X)
    assert record["pose_after"]["yaw"] != pytest.approx(commanded["yaw"])
    assert record["metadata"]["pose_source"] == POSE_SOURCE_READBACK


def test_the_trace_and_positions_follow_the_readback() -> None:
    """The trace is what an acceptance run reads, so it is what has to be actual."""
    env = ReadbackBackend()
    rollout = _run_backend(env, ScriptedPolicy(stop_after=3, tail="stop_token"))

    moving = 0
    for step, record in zip(rollout.trace.steps[1:], rollout.records, strict=True):
        assert step.position[0] == pytest.approx(record["pose_after"]["x"])
        assert step.metadata["pose_source"] == POSE_SOURCE_READBACK
        if record["waypoint_execution"]["mode"] == "policy_stop":
            # A stop step re-commands the pose it was already at, so command and
            # readback coincide there by arithmetic. Only a commanded move can show
            # which of the two the trace wrote down.
            continue
        moving += 1
        assert step.position[0] != pytest.approx(record["metadata"]["commanded_pose"]["x"])
    assert moving > 0, "the episode must contain a commanded move for this to prove anything"


def test_scoring_consumes_the_readback_trajectory() -> None:
    """A path length computed from commands is a claim about the policy, not the sim."""
    task_config, episode = _episode()
    policy = ScriptedPolicy(stop_after=40, tail="stop_token")
    commanded_only = run_camera_walk_episode(
        env=FakeCameraWalkBackend(),
        policy=ScriptedPolicy(stop_after=40, tail="stop_token"),
        episode=episode,
        task_config=task_config,
        options=RolloutOptions.from_task_config(task_config),
    )
    readback = run_camera_walk_episode(
        env=ReadbackBackend(),
        policy=policy,
        episode=episode,
        task_config=task_config,
        options=RolloutOptions.from_task_config(task_config),
    )

    def _score(rollout):
        return ScoreManager(task_config.scoring).evaluate_episode(
            episode, rollout.trace, final_measures=rollout.final_measures, task_config=task_config
        )

    # The camera that kept landing off-command must not score the same as the one that
    # landed on it. If it does, scoring never saw the simulator.
    assert _score(readback).metrics["distance_to_goal"] != pytest.approx(
        _score(commanded_only).metrics["distance_to_goal"]
    )


def test_the_backend_is_captured_exactly_once_per_step() -> None:
    # Once for the initial pose, then once per step. A second capture would be a second
    # reading of a moving camera, and the record's pose would not match its own summary.
    env = ReadbackBackend()
    rollout = _run_backend(env, ScriptedPolicy(stop_after=3, tail="stop_token"))

    assert env.captures == rollout.steps_taken + 1


def test_the_initial_pose_is_a_readback_and_says_so() -> None:
    # Step 0 of the trace is the origin every distance measure is relative to; leaving it
    # a command while every later step is a readback would put an invisible seam there.
    env = ReadbackBackend()
    rollout = _run_backend(env, ScriptedPolicy(stop_after=1, tail="stop_token"))

    assert rollout.trace.metadata["initial_pose_source"] == POSE_SOURCE_READBACK
    assert rollout.trace.steps[0].position[0] == pytest.approx(
        _episode()[1].start_pose.position[0] + env.OFFSET_X
    )


def test_the_step_metadata_carries_the_canonical_orientation() -> None:
    """The step trace rebuilds orientation from yaw alone, so the measured one is kept.

    A simulator reporting pitch or roll would otherwise have it silently discarded on the
    way into the trace, and nothing downstream could tell.
    """
    env = ReadbackBackend()
    rollout = _run_backend(env, ScriptedPolicy(stop_after=2, tail="stop_token"))

    from insight_bench.simulator.observation import quat_wxyz_from_yaw

    metadata = rollout.trace.steps[1].metadata
    expected = quat_wxyz_from_yaw(rollout.records[0]["pose_after"]["yaw"])
    assert metadata["orientation_wxyz"] == pytest.approx(list(expected))


def test_a_failing_capture_ends_the_episode_instead_of_scoring_the_command() -> None:
    """The L20 shape exactly: the capability is there and it raises every time.

    Before, the exception was recorded on the step summary and the rollout carried on
    with commanded poses -- a full trace, a clean score, and a broken run.
    """

    class _BrokenReadback(FakeCameraWalkBackend):
        def capture_observation(self):
            raise RuntimeError("'Camera' object has no attribute 'get_world_poses'")

    policy = ScriptedPolicy(stop_after=3, tail="stop_token")
    with pytest.raises(PoseReadbackError, match="get_world_poses"):
        _run_backend(_BrokenReadback(), policy)


class _FailsAfter(FakeCameraWalkBackend):
    """Reads back cleanly *n* times, then breaks -- the mid-episode failure shape."""

    def __init__(self, *, after: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self._after = after
        self.captures = 0

    def capture_observation(self):
        from insight_bench.simulator.observation import make_observation, quat_wxyz_from_yaw

        self.captures += 1
        if self.captures > self._after:
            raise RuntimeError("camera exploded")
        return make_observation(
            self._frame(),
            pose=self.pose,
            orientation_wxyz=quat_wxyz_from_yaw(self.pose.yaw),
            frame_id=self.captures,
        )


def test_a_capture_failing_mid_episode_still_finishes_it_on_the_policy_side() -> None:
    # Failing loudly must not leak the policy-side episode: finish() runs from the
    # rollout's finally block whatever ended the loop.
    policy = ScriptedPolicy(stop_after=3, tail="stop_token")
    task_config, episode = _episode()
    with pytest.raises(PoseReadbackError):
        run_camera_walk_episode(
            env=_FailsAfter(after=2),
            policy=policy,
            episode=episode,
            task_config=task_config,
            options=RolloutOptions.from_task_config(task_config),
        )
    assert policy.finish_calls == [episode.episode_id]


def test_a_capture_failing_at_reset_refuses_before_the_policy_is_engaged() -> None:
    """The initial readback is taken before ``policy.reset``, so nothing to finish.

    Ordering, not an accident: a run that cannot read its own camera should not open an
    episode on the policy side at all, and an unmatched ``finish`` would be a lie about
    an episode that never started.
    """
    policy = ScriptedPolicy(stop_after=3, tail="stop_token")
    task_config, episode = _episode()
    with pytest.raises(PoseReadbackError):
        run_camera_walk_episode(
            env=_FailsAfter(after=0),
            policy=policy,
            episode=episode,
            task_config=task_config,
            options=RolloutOptions.from_task_config(task_config),
        )
    assert policy.reset_calls == []
    assert policy.finish_calls == []


def test_a_backend_with_no_capture_surface_is_marked_a_fallback() -> None:
    """A frame-only backend still runs, and its trace says the pose is an echo.

    This is the honest version of what the old code did for every backend: the pose is
    the command, so comparing it to the command proves nothing. The marker is what lets
    an acceptance run refuse to read it as a readback.
    """
    env = FakeCameraWalkBackend()
    assert not hasattr(env, "capture_observation"), "the fake must reproduce the legacy shape"
    rollout = _run_backend(env, ScriptedPolicy(stop_after=2, tail="stop_token"))

    assert rollout.trace.metadata["initial_pose_source"] == POSE_SOURCE_COMMANDED_FALLBACK
    for record in rollout.records:
        assert record["metadata"]["pose_source"] == POSE_SOURCE_COMMANDED_FALLBACK
        # And the tautology is visible rather than hidden: pose equals command.
        assert record["pose_after"] == record["metadata"]["commanded_pose"]


def test_the_two_sources_are_distinguishable_in_the_persisted_trace(tmp_path: Path) -> None:
    # An acceptance run reads JSON, not objects.
    from insight_bench.vln_runtime.traces.writer import write_rollout_trace

    readback = _run_backend(ReadbackBackend(), ScriptedPolicy(stop_after=1, tail="stop_token"))
    fallback = _run_backend(
        FakeCameraWalkBackend(), ScriptedPolicy(stop_after=1, tail="stop_token")
    )

    sources = []
    for name, rollout in (("readback", readback), ("fallback", fallback)):
        path = write_rollout_trace(tmp_path / name / "trace.json", rollout.trace)
        payload = json.loads(path.read_text(encoding="utf-8"))
        sources.append(payload["steps"][-1]["metadata"]["pose_source"])
    assert sources == [POSE_SOURCE_READBACK, POSE_SOURCE_COMMANDED_FALLBACK]


def test_the_terrain_record_survives_beside_the_pose_provenance() -> None:
    """G2 reads both out of the same metadata dict; adding one must not shadow the other.

    An objnav scene is a USD mesh, so the query runs and the ray finds a floor --
    two flags that have to arrive alongside the pose provenance rather than in
    place of it.
    """

    class _MeshReadback(ReadbackBackend):
        def last_terrain_query(self):
            return {"terrain_query_available": True, "terrain_raycast_hit": True}

    rollout = _run_backend(_MeshReadback(), ScriptedPolicy(stop_after=1, tail="stop_token"))

    metadata = rollout.trace.steps[-1].metadata
    assert metadata["terrain_query_available"] is True
    assert metadata["terrain_raycast_hit"] is True
    assert metadata["pose_source"] == POSE_SOURCE_READBACK


def test_trace_step_zero_carries_the_same_provenance_as_every_other_step() -> None:
    """F-1: step 0 was the one step in every trace with empty metadata.

    It is the pre-action pose and has no step record of its own, so it used to
    arrive with nothing attached -- which meant "does every step say where its
    pose came from" had to special-case it, or report it as missing. It now
    carries the same three fields the rest do.
    """
    env = ReadbackBackend()
    rollout = _run_backend(env, ScriptedPolicy(stop_after=2, tail="stop_token"))

    step0 = rollout.trace.steps[0].metadata
    assert step0["pose_source"] == POSE_SOURCE_READBACK
    assert isinstance(step0["commanded_pose"], dict)
    assert len(step0["orientation_wxyz"]) == 4

    # "The same as every other step" is the actual requirement, so compare shapes
    # rather than asserting three names and hoping they stay in step.
    later = rollout.trace.steps[1].metadata
    assert set(step0) == {"pose_source", "commanded_pose", "orientation_wxyz"}
    assert set(step0) <= set(later)


def test_trace_step_zero_reports_the_readback_measured_against_the_start_command() -> None:
    # The numbers, not just the keys: step 0's pose is the readback and its
    # commanded_pose is the resolved start pose, so the two can be subtracted.
    env = ReadbackBackend()
    rollout = _run_backend(env, ScriptedPolicy(stop_after=1, tail="stop_token"))

    step0 = rollout.trace.steps[0]
    commanded = step0.metadata["commanded_pose"]
    assert step0.position[0] == pytest.approx(commanded["x"] + env.OFFSET_X)
    assert step0.position[0] == pytest.approx(_episode()[1].start_pose.position[0] + env.OFFSET_X)


def test_a_fallback_backend_labels_step_zero_a_fallback_too() -> None:
    # The label has to be honest on the degraded path as well, or step 0 becomes a
    # place where a commanded pose is silently presented as a readback.
    rollout = _run_backend(FakeCameraWalkBackend(), ScriptedPolicy(stop_after=1, tail="stop_token"))

    step0 = rollout.trace.steps[0].metadata
    assert step0["pose_source"] == POSE_SOURCE_COMMANDED_FALLBACK
    assert step0["commanded_pose"] == {
        "x": rollout.trace.steps[0].position[0],
        "y": rollout.trace.steps[0].position[1],
        "z": rollout.trace.steps[0].position[2],
        "yaw": pytest.approx(_yaw_of(rollout.trace.steps[0])),
    }


def _yaw_of(step) -> float:
    from insight_bench.simulator.observation import yaw_from_quat_wxyz

    return yaw_from_quat_wxyz(step.orientation_wxyz)


@pytest.mark.parametrize(
    "make_backend, expected_source",
    [
        (ReadbackBackend, POSE_SOURCE_READBACK),
        (FakeCameraWalkBackend, POSE_SOURCE_COMMANDED_FALLBACK),
    ],
    ids=["readback", "commanded-fallback"],
)
def test_no_step_in_a_trace_is_exempt_from_declaring_its_provenance(
    make_backend, expected_source, tmp_path: Path
) -> None:
    """The step-0-class invariant, stated as a rule rather than as cases.

    The individual tests above pin the steps that exist today, which is how step 0
    went unnoticed: it was not a step anyone had written a test for. This pins the
    *rule* instead -- every element that reaches `trace.steps` carries
    `pose_source`, `commanded_pose` and a canonical `orientation_wxyz`, with no
    exemption -- so a future element added without provenance fails here instead
    of quietly becoming the next step 0.

    Run on both paths. A readback trace and a fallback trace must each be
    uniformly labelled: a trace that is honest about most of its steps and silent
    about one is exactly the shape that let a commanded pose pass as a readback.
    """
    rollout = _run_backend(make_backend(), ScriptedPolicy(stop_after=3, tail="stop_token"))

    steps = rollout.trace.steps
    # Not vacuous: the initial pose plus one element per recorded step, so a
    # truncated or empty trace cannot satisfy the loop below by having nothing in it.
    assert len(steps) == rollout.steps_taken + 1 >= 2

    for step in steps:
        metadata = step.metadata
        assert metadata.get("pose_source") == expected_source, f"step {step.step}"
        assert isinstance(metadata.get("commanded_pose"), dict), f"step {step.step}"
        assert len(metadata.get("orientation_wxyz") or []) == 4, f"step {step.step}"

    # And it has to survive serialisation, because that is the form an acceptance
    # run reads -- an invariant that holds only in memory protects nobody.
    from insight_bench.vln_runtime.traces.writer import write_rollout_trace

    path = write_rollout_trace(tmp_path / "ep" / "trace.json", rollout.trace)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["steps"]) == len(steps)
    for step in payload["steps"]:
        metadata = step["metadata"]
        assert metadata["pose_source"] == expected_source, step["step"]
        assert isinstance(metadata["commanded_pose"], dict), step["step"]
        assert len(metadata["orientation_wxyz"]) == 4, step["step"]


# -- the structureless-frame guard -------------------------------------------------
#
# Numbers measured on real renders, not chosen: 604 first frames from a full
# 1,097-episode run across all 210 published scenes, against one scene composed
# in the wrong frame. See camera_walk.STRUCTURELESS_FRAME_* for the distribution.


def _frame(mean: float, *, structure: float, height: int = 270, width: int = 480) -> np.ndarray:
    """A frame with a chosen mean and a chosen amount of local contrast."""
    grid = np.full((height, width), float(mean), dtype=np.float32)
    grid[:, ::2] += structure
    grid[:, 1::2] -= structure
    return np.clip(grid, 0.0, 255.0).astype(np.uint8)


def test_a_uniform_frame_is_still_reported() -> None:
    task_config = get_task_config("insight_bench")
    note = describe_blank_scene_frame(
        _frame(236.0, structure=0.0), task_config=task_config, episode_id="e"
    )
    assert note is not None and "appears blank" in note


def test_a_frame_with_spread_but_no_structure_is_reported() -> None:
    """The wrong-frame failure: bright fog, unremarkable spread, no local contrast.

    This is the case the spread check alone let through, and every other check
    in a run passes on it, so nothing else in a run would say the scene was
    invisible.
    """
    fog = np.asarray(_frame(236.0, structure=0.0), dtype=np.float32)
    fog[:, :24] = 210.0  # one broad band: spread ~8.5, structure ~0.2
    note = describe_blank_scene_frame(
        fog.astype(np.uint8), task_config=get_task_config("insight_bench"), episode_id="e"
    )
    assert note is not None and "no structure" in note


def test_the_dimmest_real_frames_measured_still_pass() -> None:
    """Breaching one floor is not enough to report a frame, and that is deliberate.

    Of the 604 real first frames, the two closest to refusal each breached one
    floor and cleared the other: structure 0.350 at spread 16.74, and spread
    10.54 at structure 0.811. If either floor refused on its own, both of those
    real episodes would die. So the cases are built to sit in exactly those two
    regions -- asserted against the floors themselves, not against numbers that
    could drift away from them -- and both must pass.
    """
    task_config = get_task_config("insight_bench")

    # Structure below its floor, spread above its: a broad ramp with almost no
    # neighbour-to-neighbour change. This is the "0.350 at 16.74" corner.
    ramp = np.tile(np.linspace(80.0, 138.0, 270, dtype=np.float32)[:, None], (1, 480))
    low_structure = np.clip(ramp, 0.0, 255.0).astype(np.uint8)
    assert frame_structure_energy(low_structure) < STRUCTURELESS_FRAME_EDGE
    assert float(low_structure.std()) >= STRUCTURELESS_FRAME_STD
    assert (
        describe_blank_scene_frame(low_structure, task_config=task_config, episode_id="ls") is None
    )

    # Spread below its floor, structure above its: a two-tone frame with a fine
    # texture on it. This is the "10.54 at 0.811" corner.
    two_tone = np.full((270, 480), 236.0, dtype=np.float32)
    two_tone[:, :60] = 212.0
    two_tone[::2, :] += 2.0
    low_spread = np.clip(two_tone, 0.0, 255.0).astype(np.uint8)
    assert float(low_spread.std()) < STRUCTURELESS_FRAME_STD
    assert frame_structure_energy(low_spread) >= STRUCTURELESS_FRAME_EDGE
    assert describe_blank_scene_frame(low_spread, task_config=task_config, episode_id="lsp") is None
