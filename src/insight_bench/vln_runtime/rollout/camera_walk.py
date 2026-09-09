"""Single-episode camera-walk rollout: the simulator-neutral execution loop.

This is the closure that turns "a backend that can pose a camera and hand back
frames" plus "a policy that answers with waypoints" into episode evidence
(records, a :class:`RolloutTrace`, termination facts). It never imports Isaac,
torch or warp: the backend and the policy are protocols, so the whole loop is
exercised in CI against fakes and the Isaac backend is the only thing that has
to change when a simulator line changes.

Deliberately *not* migrated from the operator harness: video/top-down
rendering, frame and depth persistence, the async artifact pipeline, the
profiler, and multi-env batching. Those are operator conveniences, and every
one of them widens the trust and dependency surface of a public SDK.

One contract is load-bearing enough to state here. **A step reports where the
camera actually is, not where it was told to go.** When the backend can capture
an observation, that observation's pose is what reaches positions, state,
measures, scoring, records and the trace; the resolved command is kept beside it
as evidence, never in place of it. See :func:`_pose_reading` and
:data:`POSE_SOURCE_READBACK`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from insight_bench.vln_runtime.episodes import EpisodeSpec
from insight_bench.vln_runtime.managers import TerminationManager
from insight_bench.vln_runtime.measures import build_measure_manager
from insight_bench.vln_runtime.motion.pose import CameraPose, normalize_yaw
from insight_bench.vln_runtime.motion.waypoint_execution import (
    VelocityControlCommand,
    WaypointDelta,
    integrate_velocity_control,
    velocity_control_from_waypoint,
    waypoint_from_policy_response,
)
from insight_bench.vln_runtime.policy.client import PolicyResponse
from insight_bench.vln_runtime.rollout.options import RolloutOptions
from insight_bench.vln_runtime.scoring.event_terms import navigation_step_events
from insight_bench.vln_runtime.scoring.progress_terms import navigation_progress_status
from insight_bench.vln_runtime.suite.config import BenchmarkTaskConfig
from insight_bench.vln_runtime.terminations import is_policy_stop_response
from insight_bench.vln_runtime.traces import RolloutTrace
from insight_bench.vln_runtime.traces.frames import FrameSink

if TYPE_CHECKING:  # the runtime import stays inside a function; see _canonical_yaw_quat
    from insight_bench.simulator.observation import Observation

XYZ = tuple[float, float, float]
QUAT_WXYZ = tuple[float, float, float, float]
LogCallback = Any

POSE_SOURCE_READBACK = "simulator-readback"
"""``pose_source`` for a step whose pose was read back out of the simulator.

This is the only value an acceptance run may treat as a readback. It means the
pose in the record is what the backend's ``capture_observation`` reported, so
comparing it against ``metadata["commanded_pose"]`` is a real measurement of
whether the simulator moved -- which is exactly what the G2 yaw check asks.
"""

POSE_SOURCE_COMMANDED_FALLBACK = "commanded-fallback"
"""``pose_source`` for a step whose backend cannot capture an observation.

The pose is then the resolved command echoed back, and a comparison against the
command is a tautology. Legacy fakes and frame-only backends land here; the
marker exists so acceptance can refuse to count such a run as readback-verified
instead of reading a plausible number and believing it.
"""


class CameraWalkBackend(Protocol):
    """What the rollout needs from a simulator backend. Isaac-free by design."""

    def resolve_start_pose(self, pose: CameraPose) -> CameraPose:
        """Snap a start pose onto the floor, or return it unchanged."""
        ...

    def resolve_motion(self, prev: CameraPose, candidate: CameraPose) -> CameraPose:
        """Resolve an integrated pose against scene geometry (terrain + walls)."""
        ...

    def reset(
        self,
        pose: CameraPose,
        *,
        warmup_steps: int,
        converge_max_steps: int,
        converge_tol: float,
        log: Any | None,
    ) -> np.ndarray:
        """Pose the camera and return the first settled RGB frame."""
        ...

    def set_pose(self, pose: CameraPose) -> None:
        """Command the camera to a pose."""
        ...

    def step_and_capture(self, *, settle_steps: int) -> np.ndarray:
        """Advance the simulator and return the current RGB frame."""
        ...


class PolicyDriver(Protocol):
    """The four-call policy surface. ``LocalVlnPolicyClient`` satisfies it."""

    def reset(
        self, instruction: str, *, episode_id: str, first_frame_rgb: Any | None
    ) -> dict[str, Any]: ...

    def act(self, rgb_frame: Any, *, episode_id: str, frame_id: int) -> PolicyResponse: ...

    def finish(self, *, episode_id: str) -> dict[str, Any]: ...


@dataclass
class CameraWalkEpisodeState:
    """Environment-shaped online state consumed by measures and terminations."""

    episode: EpisodeSpec
    task_config: BenchmarkTaskConfig
    max_episode_length: int
    control_dt_sec: float
    pose: CameraPose
    position: XYZ
    positions: list[XYZ]
    num_envs: int = 1
    device: str = "cpu"
    episode_length_buf: list[int] = field(default_factory=lambda: [0])
    last_policy_response: Any | None = None
    last_selected_waypoint: Any | None = None
    last_velocity_command: VelocityControlCommand | None = None
    last_executed_delta: WaypointDelta | None = None
    stop_step: int = -1
    termination_terms: tuple[str, ...] = field(default_factory=tuple)
    measure_values: dict[str, Any] = field(default_factory=dict)

    def update_after_action(
        self,
        *,
        response: Any,
        selected_waypoint: Any | None,
        velocity_command: VelocityControlCommand,
        executed_delta: WaypointDelta,
        pose: CameraPose,
        position: XYZ,
        positions: list[XYZ],
    ) -> None:
        self.last_policy_response = response
        self.last_selected_waypoint = selected_waypoint
        self.last_velocity_command = velocity_command
        self.last_executed_delta = executed_delta
        self.pose = pose
        self.position = position
        self.positions = positions
        self.episode_length_buf[0] += 1

    def update_measures(self, measures: dict[str, Any]) -> None:
        self.measure_values = dict(measures)

    def update_termination_terms(self, terms: tuple[str, ...]) -> None:
        self.termination_terms = tuple(terms)


@dataclass(frozen=True)
class EpisodeRollout:
    """Everything one finished episode produced, ready for offline scoring."""

    episode_id: str
    records: list[dict[str, Any]]
    trace: RolloutTrace
    termination: dict[str, Any]
    termination_reason: str
    stop_step: int
    final_measures: dict[str, Any]
    steps_taken: int
    wall_time_sec: float


class PoseReadbackError(RuntimeError):
    """Raised when a backend that can read its camera pose back failed to.

    Deliberately fatal to the episode. The alternative -- recording the failure
    and carrying on with the commanded pose -- is what made a real L20 job look
    green: ``read_camera_pose`` raised on every step, the summary swallowed it,
    and the trace was filled with commands that scored perfectly against
    themselves. A backend that offers the capability and then cannot deliver it
    has not run a measurable episode, and saying so is the only honest outcome.

    A backend with no capture surface at all is a different fact and is not an
    error: see :data:`POSE_SOURCE_COMMANDED_FALLBACK`.
    """


@dataclass(frozen=True)
class _PoseReading:
    """Where the camera is for one step, and how that was determined."""

    pose: CameraPose
    orientation_wxyz: QUAT_WXYZ
    source: str

    @property
    def is_readback(self) -> bool:
        return self.source == POSE_SOURCE_READBACK


def run_camera_walk_episode(
    *,
    env: CameraWalkBackend,
    policy: PolicyDriver,
    episode: EpisodeSpec,
    task_config: BenchmarkTaskConfig,
    options: RolloutOptions,
    log: LogCallback | None = None,
    frame_sink: FrameSink | None = None,
) -> EpisodeRollout:
    """Run one episode end to end and return its evidence.

    The loop is the operator harness's, minus the artifact machinery: reset,
    then for each step ask the policy, decode a waypoint into a velocity
    command, integrate it, let the backend resolve it against scene geometry,
    re-render, and evaluate measures and terminations.

    *frame_sink*, when given, is handed the frame the policy is about to be
    shown -- before it acts, so the persisted picture is the one the decision
    was made on rather than the one after it. Its return value is recorded as
    that step's ``frame_path``. Without a sink no ``frame_path`` key is
    produced at all, and the trace is byte-for-byte the trace this rollout
    produced before frames could be persisted. See
    :mod:`insight_bench.vln_runtime.traces.frames` for the numbering.
    """
    start_x, start_y, start_z = episode.start_pose.position
    commanded_start = env.resolve_start_pose(
        CameraPose(start_x, start_y, start_z, normalize_yaw(_episode_start_yaw(episode)))
    )

    # Reset before the episode state is built, so the state can start from where the
    # camera actually landed rather than from where it was sent. The initial pose is
    # step 0 of the trace and the origin every distance measure is relative to; leaving
    # it as a command while every later step is a readback would put a seam at step 0
    # that nothing downstream could see.
    frame = env.reset(
        commanded_start,
        warmup_steps=options.warmup_steps,
        converge_max_steps=options.render_converge_max_steps,
        converge_tol=options.render_converge_tol,
        log=log,
    )
    blank_frame_note = describe_blank_scene_frame(
        frame, task_config=task_config, episode_id=episode.episode_id
    )
    if blank_frame_note is not None and log is not None:
        log(f"[{episode.episode_id}] {blank_frame_note}")
    initial_reading = _pose_reading(capture_step_observation(env), commanded=commanded_start)
    pose = initial_reading.pose

    positions: list[XYZ] = [_position_from_pose(pose)]
    state = CameraWalkEpisodeState(
        episode=episode,
        task_config=task_config,
        max_episode_length=options.num_steps,
        control_dt_sec=options.control_dt_sec,
        pose=pose,
        position=positions[-1],
        positions=positions,
    )
    measure_manager = build_measure_manager(task_config.measures, state)
    measure_manager.reset()
    termination_manager = TerminationManager(task_config.terminations, state)
    termination_manager.reset()
    stop_terms = _stop_termination_terms(task_config)

    # No first frame here. The loop below opens with `act(frame, frame_id=0)` on
    # this very frame, before the simulator has stepped, so sending it with the
    # reset delivered the same observation twice -- and any adapter that keeps a
    # history counted step 0 as two distinct moments unless it happened to guess
    # that it should drop one. The frame is not lost: it arrives immediately, as
    # the first thing the policy is asked to act on.
    policy.reset(episode.instruction, episode_id=episode.episode_id, first_frame_rgb=None)

    records: list[dict[str, Any]] = []
    measures: dict[str, Any] = {}
    started_at = time.monotonic()
    final_termination = _termination_payload(termination_manager)

    try:
        for step in range(options.num_steps):
            frame_path = None if frame_sink is None else frame_sink(episode.episode_id, step, frame)
            act_started = time.monotonic()
            response = policy.act(frame, episode_id=episode.episode_id, frame_id=step)
            client_step_ms = (time.monotonic() - act_started) * 1000.0

            if is_policy_stop_response(response):
                waypoint_index: int | None = None
                selected_waypoint: WaypointDelta | None = None
                velocity_command = _zero_velocity_command()
                executed_delta = _zero_waypoint_delta()
                next_pose = pose
                next_frame = frame
                waypoint_execution: dict[str, Any] = {"mode": "policy_stop"}
            else:
                waypoint_index, selected_waypoint = waypoint_from_policy_response(
                    response,
                    options.waypoint_index,
                    waypoint_atol=options.waypoint_atol,
                    yaw_lookahead_steps=options.yaw_lookahead_steps,
                    yaw_lookahead_hybrid=options.yaw_lookahead_hybrid,
                    yaw_hybrid_thresh_rad=options.yaw_hybrid_thresh_rad,
                )
                velocity_command = velocity_control_from_waypoint(
                    selected_waypoint,
                    dt_sec=options.control_dt_sec,
                    lin_vel_range_mps=options.lin_vel_range_mps,
                    ang_vel_range_dps=options.ang_vel_range_dps,
                    drop_lateral=options.drop_lateral,
                )
                next_pose, executed_delta = integrate_velocity_control(
                    pose,
                    velocity_command,
                    dt_sec=options.control_dt_sec,
                    motion_scale=options.motion_scale,
                    yaw_scale=options.yaw_scale,
                )
                # Snap z to the floor (stairs/ramps) and clamp at walls before the
                # pose reaches rendering, positions, measures and scoring.
                next_pose = env.resolve_motion(pose, next_pose)
                waypoint_execution = {
                    "mode": "waypoint_to_velocity_control",
                    "dt_sec": options.control_dt_sec,
                    "lin_vel_range_mps": list(options.lin_vel_range_mps),
                    "ang_vel_range_dps": list(options.ang_vel_range_dps),
                    "waypoint_atol": options.waypoint_atol,
                    "yaw_hybrid_thresh_deg": options.yaw_hybrid_thresh_deg,
                    "lateral_m_dropped": options.drop_lateral,
                    "yaw_lookahead_steps": options.yaw_lookahead_steps,
                    "yaw_lookahead_hybrid": options.yaw_lookahead_hybrid,
                }
                env.set_pose(next_pose)
                next_frame = env.step_and_capture(settle_steps=options.render_settle_steps)

            # Exactly one capture per step, on both branches, and before anything reads
            # a pose. `next_pose` is what the camera was asked for; `reading.pose` is
            # what it reports. Everything from here down -- positions, state, measures,
            # terminations, scoring, the record and the trace -- uses the reading. A
            # policy-stop step is captured too rather than assumed unchanged: "the
            # camera did not move because we did not move it" is the assumption this
            # whole path exists to stop making.
            observation = capture_step_observation(env)
            reading = _pose_reading(observation, commanded=next_pose)

            positions.append(_position_from_pose(reading.pose))
            state.update_after_action(
                response=response,
                selected_waypoint=selected_waypoint,
                velocity_command=velocity_command,
                executed_delta=executed_delta,
                pose=reading.pose,
                position=_position_from_pose(reading.pose),
                positions=positions,
            )
            measures = measure_manager.compute()
            state.update_measures(measures)
            termination_manager.compute()
            triggered = termination_manager.triggered_terms(env_idx=0)
            state.update_termination_terms(triggered)
            if state.stop_step < 0 and any(term in stop_terms for term in triggered):
                state.stop_step = step
            termination = _termination_payload(termination_manager)
            final_termination = termination
            subtask_status = navigation_progress_status(
                episode, positions, stop_step=state.stop_step
            )
            events = navigation_step_events(
                episode,
                positions,
                stop_requested=state.stop_step == step,
                termination_reason=termination.get("reason"),
                progress_status=subtask_status,
            )

            records.append(
                {
                    "step": step,
                    "episode_id": episode.episode_id,
                    "pose_before": pose.to_dict(),
                    "pose_after": reading.pose.to_dict(),
                    "raw_output": response.raw_output,
                    "waypoint": response.waypoint,
                    "waypoint_cluster_id": response.waypoint_cluster_id,
                    "apos_id": response.apos_id,
                    "opos_id": response.opos_id,
                    "apos_xy": response.apos_xy,
                    "opos_xy": response.opos_xy,
                    "apos_kind": response.apos_kind,
                    "opos_kind": response.opos_kind,
                    "waypoint_index": waypoint_index,
                    "waypoint_index_override": options.waypoint_index,
                    "selected_waypoint": (
                        None if selected_waypoint is None else selected_waypoint.to_dict()
                    ),
                    "velocity_command": velocity_command.to_dict(),
                    "executed_delta": executed_delta.to_dict(),
                    "waypoint_execution": waypoint_execution,
                    "measures": measures,
                    "policy_inference_time_ms": response.inference_time_ms,
                    "client_step_time_ms": client_step_ms,
                    "timings_ms": response.timings_ms,
                    "termination": termination,
                    "subtask_status": subtask_status,
                    "events": events,
                    "observation": _observation_summary(observation, next_frame),
                    "metadata": {
                        **_terrain_metadata(env),
                        **_pose_metadata(reading, commanded=next_pose),
                    },
                    # Absent, not empty, when no sink is attached: the trace
                    # builder defaults the field to "", so a run without
                    # `--vis` keeps producing exactly the bytes it did before.
                    **({} if frame_path is None else {"frame_path": frame_path}),
                }
            )

            # The next step integrates from where the camera is, not from where the last
            # command aimed. Integrating from the command would let the two drift apart
            # with nothing recording that they had.
            pose = reading.pose
            frame = next_frame
            if termination["done"]:
                break
    finally:
        _finish_policy_episode(policy, episode.episode_id, log=log)

    if final_termination["done"]:
        termination_reason = str(final_termination["reason"] or "terminated")
        termination_payload = final_termination
    else:
        termination_reason = "num_steps_exhausted"
        termination_payload = {**final_termination, "reason": termination_reason}

    final_measures = dict(measures)
    trace = RolloutTrace.from_camera_walk_records(
        episode_id=episode.episode_id,
        instruction=episode.instruction,
        records=records,
        termination_reason=termination_reason,
        stop_step=state.stop_step,
        final_measures=final_measures,
        metadata={
            "task": task_config.name,
            "backend": task_config.backend,
            # Kept alongside the per-step field below: this one is episode-level and
            # survives even a trace whose steps were filtered or truncated.
            "initial_pose_source": initial_reading.source,
            # Present only when the first frame had nothing in it, so a reader
            # can tell one bad start pose from a scene that never loaded.
            **({"first_frame_warning": blank_frame_note} if blank_frame_note else {}),
        },
        # Step 0 is the pre-action pose and has no step record to carry provenance
        # in, so it is passed separately. Without it, step 0 was the single step in
        # every trace that could not say where its pose came from.
        initial_metadata=_pose_metadata(initial_reading, commanded=commanded_start),
    )
    return EpisodeRollout(
        episode_id=episode.episode_id,
        records=records,
        trace=trace,
        termination=termination_payload,
        termination_reason=termination_reason,
        stop_step=state.stop_step,
        final_measures=final_measures,
        steps_taken=len(records),
        wall_time_sec=time.monotonic() - started_at,
    )


#: A frame this uniform never came from a loaded scene.
BLANK_FRAME_STD = 5.0

#: A frame can be far from uniform and still show the agent nothing. A scene
#: loaded in the wrong frame renders as bright fog with a streak or two: its
#: spread is unremarkable but it carries almost no local structure. Both of
#: these have to be low at once to call a frame structureless, because either
#: alone has real frames on the wrong side of it.
#:
#: Measured, not chosen. Over 604 first frames from a full 1,097-episode run
#: across all 210 published scenes, the lowest structure was 0.350 and the
#: lowest spread 10.54; the median frame sat at 3.649 and 36.76. A scene
#: loaded in a wrong frame measured 0.215 and 8.53 -- under both floors, while
#: every one of the 604 real frames clears at least one of them.
STRUCTURELESS_FRAME_EDGE = 0.30
STRUCTURELESS_FRAME_STD = 10.0


def frame_structure_energy(array: Any) -> float:
    """Mean absolute difference between neighbouring pixels, in grey levels.

    Local contrast, which is what "can the agent see anything" comes down to.
    Global spread cannot answer it: a uniform grey wall and a room full of
    furniture can share a standard deviation.
    """
    grey = np.asarray(array, dtype=np.float32)
    if grey.ndim == 3:
        grey = grey.mean(axis=2)
    if grey.ndim != 2 or grey.shape[0] < 2 or grey.shape[1] < 2:
        return 0.0
    vertical = float(np.abs(np.diff(grey, axis=0)).mean())
    horizontal = float(np.abs(np.diff(grey, axis=1)).mean())
    return (vertical + horizontal) / 2.0


def describe_blank_scene_frame(
    frame: Any,
    *,
    task_config: BenchmarkTaskConfig,
    episode_id: str,
) -> str | None:
    """Describe a first frame with nothing in it, or ``None`` when it is fine.

    Only applies to tasks that declare ``assets["require_scene_assets"]``: a
    procedural scene may legitimately render a near-uniform frame, but a
    Matterport/HM3D scene that comes back flat means the USD never loaded, and
    scoring that rollout would manufacture a number out of an empty room.

    Two failures, not one. A scene that did not load at all comes back uniform,
    which spread alone catches. A scene that loaded in the *wrong frame* does
    not: it renders bright fog, spread 8.5 where a real room is 37, and every
    other check in the run passes -- the start pose is read from the episode
    file and written into the stage, so a wrong frame moves the scene out from
    under an agent standing exactly where it was told to. Poses, metrics,
    evidence and `verify` all come out identical to a good run. The pixels are
    the only place it shows.

    This reports rather than raises. Erroring the episode was too blunt: one
    start pose that legitimately faces a blank wall was enough to turn a
    complete suite into a `partial` run that could not be submitted, and the
    published measurement of this benchmark scored that episode like any other.
    The caller records what this returns, so a whole scene that failed to load
    is still visible instead of quietly depressing every episode in it.
    """
    if not bool(task_config.assets.get("require_scene_assets")):
        return None
    array = np.asarray(frame)
    std = float(array.std()) if array.size else 0.0
    mean = float(array.mean()) if array.size else 0.0
    edge = frame_structure_energy(array) if array.size else 0.0
    if std < BLANK_FRAME_STD:
        return (
            "rendered scene frame appears blank "
            f"(mean={mean:.2f}, std={std:.2f}); "
            "check the simulator environment and USD scene asset loading"
        )
    if edge < STRUCTURELESS_FRAME_EDGE and std < STRUCTURELESS_FRAME_STD:
        return (
            "rendered scene frame carries no structure "
            f"(mean={mean:.2f}, std={std:.2f}, edge={edge:.3f}); "
            "the scene asset loaded but nothing in it is visible from the start pose -- "
            "usually a scene composed in the wrong frame, which every other check passes"
        )
    return None


def _episode_start_yaw(episode: EpisodeSpec) -> float:
    """Yaw of the episode's start pose, via the canonical WXYZ extraction.

    Calls ``yaw_from_quat_wxyz`` rather than repeating a yaw-only special case of it. The special
    case -- ``atan2(2wz, 1 - 2z^2)`` -- is exactly right while x and y are zero and silently wrong
    the day a start pose carries pitch or roll, and it would then disagree with every other yaw in
    the system while looking correct in isolation.
    """
    from insight_bench.simulator.observation import yaw_from_quat_wxyz

    quaternion = episode.start_pose.orientation_wxyz
    if quaternion is None:
        return 0.0
    return float(yaw_from_quat_wxyz(quaternion))


def _error_detail(error: BaseException) -> str:
    """Render an exception for an artifact: type and message, nothing else.

    Deliberately not a traceback. A trace is written to a shared artifact directory and may be
    handed to whoever is reading a result; a frame dump carries absolute paths and sometimes the
    contents of locals. The type and message are what make a failure diagnosable.
    """
    message = str(error).strip()
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


def _canonical_yaw_quat(yaw: float) -> QUAT_WXYZ:
    """Canonical WXYZ for a pure-yaw pose, via the one shared implementation.

    Imported inside the function for the same reason :func:`_episode_start_yaw`
    does it: this module is the simulator-neutral loop and stays importable
    without reaching into the simulator package at import time.
    """
    from insight_bench.simulator.observation import quat_wxyz_from_yaw

    return quat_wxyz_from_yaw(yaw)


def capture_step_observation(env: CameraWalkBackend) -> Observation | None:
    """Capture this step's observation, exactly once, or explain why there is none.

    Three outcomes, and the whole point of this function is that they are three
    and not two:

    * the backend has no ``capture_observation`` -> ``None``. A frame-only
      backend is a legitimate, if less measurable, thing to run against; the
      caller marks the step :data:`POSE_SOURCE_COMMANDED_FALLBACK`.
    * it has one and it answered -> the :class:`Observation`.
    * it has one and it raised, or returned ``None`` in violation of its own
      protocol -> :class:`PoseReadbackError`, ending the episode.

    The third case used to be folded into an ``{"error": ...}`` field on the
    step summary while the rollout carried on with the commanded pose. That is
    how an episode whose every pose readback crashed still produced a full trace
    and a passing gate. A capability that exists and fails is a broken run, not
    a degraded one.
    """
    capture = getattr(env, "capture_observation", None)
    if not callable(capture):
        return None
    try:
        observation = capture()
    except Exception as error:
        # Type and message, never a traceback: this reaches a shared artifact
        # directory, and a frame dump carries absolute paths and locals.
        raise PoseReadbackError(
            "the backend offers capture_observation but it failed, so this step has "
            f"no measured pose: {_error_detail(error)}"
        ) from error
    if observation is None:
        raise PoseReadbackError(
            "the backend offers capture_observation but it returned None; a pose "
            "readback is not something to substitute a command for"
        )
    return observation


def _pose_reading(observation: Observation | None, *, commanded: CameraPose) -> _PoseReading:
    """Decide the step's pose: the readback when there is one, else the command.

    The readback wins whenever it exists, because the question a step answers is
    where the camera *is*. Falling back to *commanded* is not a quieter version
    of the same answer -- it cannot disagree with the command, so every check
    that compares the two passes by construction. The source travels with the
    pose so nothing downstream has to guess which one it got.
    """
    if observation is not None:
        return _PoseReading(
            pose=observation.pose,
            orientation_wxyz=tuple(float(part) for part in observation.orientation_wxyz),  # type: ignore[arg-type]
            source=POSE_SOURCE_READBACK,
        )
    return _PoseReading(
        pose=commanded,
        orientation_wxyz=_canonical_yaw_quat(commanded.yaw),
        source=POSE_SOURCE_COMMANDED_FALLBACK,
    )


def _pose_metadata(reading: _PoseReading, *, commanded: CameraPose) -> dict[str, Any]:
    """The provenance an acceptance run needs to believe -- or reject -- a step.

    Three fields. ``pose_source`` says which surface produced the pose.
    ``orientation_wxyz`` is the canonical quaternion as measured, kept because
    the step trace reconstructs its own orientation from yaw alone and would
    silently drop any pitch or roll the simulator reported. ``commanded_pose``
    is the resolved command the step asked for: without it the recorded pose is
    unfalsifiable, since the G2 check is precisely |commanded - read back|.
    """
    return {
        "pose_source": reading.source,
        "orientation_wxyz": list(reading.orientation_wxyz),
        "commanded_pose": commanded.to_dict(),
    }


def _observation_summary(observation: Observation | None, frame: Any) -> dict[str, Any]:
    """Describe the step's observation in the shape the trace contract asks for.

    Shapes and dtypes rather than pixels: a trace is evidence that the run rendered what it says
    it rendered, and embedding frames would make it unreadable and enormous. The values are read
    off the real arrays -- never asserted -- so a backend that quietly returned float rgb or a
    depth map of the wrong size shows up here instead of being discovered downstream.

    Takes the observation the step already captured rather than capturing its own. Two calls per
    step would be two different readings of a moving camera, and the pose the record reports would
    not be the pose the summary describes. Falls back to the raw frame when the step had no
    observation to summarise; depth and intrinsics are then genuinely unavailable and are reported
    absent rather than invented.
    """
    if observation is not None:
        summary: dict[str, Any] = {
            "rgb": {"dtype": str(observation.rgb.dtype), "shape": list(observation.rgb.shape)}
        }
        if observation.depth is not None:
            summary["depth"] = {
                "dtype": str(observation.depth.dtype),
                "shape": list(observation.depth.shape),
            }
        if observation.intrinsics is not None:
            summary["intrinsics"] = [[float(v) for v in row] for row in observation.intrinsics]
        return summary
    if frame is None:
        return {}
    dtype = getattr(frame, "dtype", None)
    shape = getattr(frame, "shape", None)
    if dtype is None or shape is None:
        return {}
    return {"rgb": {"dtype": str(dtype), "shape": list(shape)}}


def _terrain_metadata(env: CameraWalkBackend) -> dict[str, Any]:
    """Report what the last motion resolution did to the scene geometry.

    Two flags. ``terrain_query_available`` is false for a plane terrain, which has no mesh and
    therefore never casts a ray; ``terrain_raycast_hit`` is whether a ray that *was* cast found a
    floor. Collapsing them into one would make "nothing to hit" indistinguishable from "terrain
    following failed", and those need different reactions.
    """
    report = getattr(env, "last_terrain_query", None)
    if callable(report):
        try:
            return dict(report())
        except Exception as error:
            # Same reasoning as the observation summary: an empty dict would read as "this scene
            # reported nothing", which is what a plane terrain legitimately does. A raising query
            # is a different fact and has to look different.
            return {"error": _error_detail(error)}
    return {}


def _position_from_pose(pose: CameraPose) -> XYZ:
    return pose.x, pose.y, pose.z


def _zero_velocity_command() -> VelocityControlCommand:
    return VelocityControlCommand(
        linear_velocity_mps=0.0,
        lateral_velocity_mps=0.0,
        angular_velocity_dps=0.0,
    )


def _zero_waypoint_delta() -> WaypointDelta:
    return WaypointDelta(forward_m=0.0, lateral_m=0.0, yaw_rad=0.0)


def _termination_payload(manager: TerminationManager) -> dict[str, Any]:
    triggered = manager.triggered_terms(env_idx=0)
    return {
        "done": manager.dones[0],
        "terminated": manager.terminated[0],
        "time_out": manager.time_outs[0],
        "triggered_terms": list(triggered),
        "reason": triggered[0] if triggered else None,
    }


def _stop_termination_terms(task_config: BenchmarkTaskConfig) -> set[str]:
    return set(task_config.metadata.get("stop_termination_terms") or ())


def _finish_policy_episode(policy: PolicyDriver, episode_id: str, *, log: LogCallback) -> None:
    finish = getattr(policy, "finish", None)
    if not callable(finish):
        return
    try:
        finish(episode_id=episode_id)
    except Exception as exc:  # policies are an untrusted extension boundary
        if log is not None:
            log(f"policy finish skipped: episode={episode_id} {type(exc).__name__}: {exc}")
