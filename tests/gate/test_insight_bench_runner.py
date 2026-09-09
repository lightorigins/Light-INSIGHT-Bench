"""The objnav closed loop, end to end, on a machine with no GPU.

Neither CI nor a laptop can run Isaac Sim, so two things are substituted and
nothing else is: the simulator backend (at the one seam the runner uses to
obtain one) and the *model* behind the policy server. The policy server itself
is real -- a stdlib HTTP server on a loopback port, spoken to by the shipped
:class:`~insight_bench.vln_runtime.policy.client.LocalVlnPolicyClient` over the
real four-endpoint protocol, JPEG frames and all. Everything between those two
substitutions is the production path: the published episode records, scene
resolution against a scene root, the success-radius policy, scene grouping, the
camera-walk rollout, the NavNuances scoring protocol, the run result, and the
runtime attestation.

The substituted world moves the camera toward the goal whenever the policy asks
for forward motion and halts it at a fixed distance. That distance is the whole
trick: parked at 2.5 m from the object, an indoor episode (2.0 m radius) fails
and an outdoor one (3.0 m radius) passes, so the mixed pass/fail result is
decided by exactly the pinned radius policy this suite is defined by.
"""

from __future__ import annotations

import base64
import json
import math
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from insight_bench._json import sha256_file, write_stable_json
from insight_bench.contracts import (
    BenchmarkManifest,
    RegistryContract,
    RegistryEntry,
    RuntimeAttestation,
)
from insight_bench.registry import Registry
from insight_bench.runner import RunContext, RunnerError, run_benchmark
from insight_bench.runners import insight_bench as runner_module
from insight_bench.runners.insight_bench import (
    DEFAULT_POLICY_URL,
    RUNNER_ID,
    insight_bench_runner,
)
from insight_bench.simulator.base import SimulatorNotAvailableError
from insight_bench.vis import SCENE_CLASS_ORDER
from insight_bench.vln_runtime.motion.pose import CameraPose
from insight_bench.vln_runtime.rollout import RolloutOptions, run_camera_walk_episode

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "objnav_published_episodes.jsonl"
COORDINATE = "insight-bench-v1@1.0.0"
MODEL_ID = "test-lab/objnav-walker-v1"

SCENE_ASSETS = (
    "habitat_gs/scene56/scene56.usd",
    "habitat_gs/scene58/scene58_zup.usda",
    "habitat_gs/scene63/scene63_zup.usda",
    "hm3d/00013-sfbj7jspYWj/sfbj7jspYWj.usd",
    "interiorgs/interior_0405_840145/interior_0405_840145_zup.usda",
    "mp3d/8194nk5LbLH/8194nk5LbLH.usd",
)

# Fixture episodes, by the radius their scene class selects.
OUTDOOR_EPISODES = {
    "objnav_v2_outdoor_batch_scene56_0",
    "objnav_v2_outdoor_batch_scene58_24",
}
INDOOR_EPISODES = {
    "objnav_00013-sfbj7jspYWj_4",
    "objnav_00013-sfbj7jspYWj_5",
    "objnav_scene63_103",
    "objnav_v2_human_manual_8194nk5LbLH_49",
    "objnav_v2_human_manual_interior_0405_840145_73",
}


# --- the real policy server, with a scripted model behind it ----------------


class _PolicyHandler(BaseHTTPRequestHandler):
    """The four endpoints, answered exactly as the wire protocol specifies."""

    server_version = "ScriptedPolicy/1.0"
    stop_after_frames = 12
    model_id: str | None = MODEL_ID
    # A model can go away while its server keeps answering. `unhealthy_after`
    # is how many /health reads stay true before the rest report the model gone.
    health_calls = 0
    unhealthy_after: int | None = None

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # do_GET / do_POST are BaseHTTPRequestHandler's spelling, not this repo's.
    def do_GET(self) -> None:
        if self.path != "/health":
            self._json({"error": "not found"}, status=404)
            return
        cls = type(self)
        cls.health_calls += 1
        health: dict[str, object] = {
            "healthy": cls.unhealthy_after is None or cls.health_calls <= cls.unhealthy_after,
            "action_mode": "waypoint",
            "model_version": "0.0.1",
            "num_active_sessions": 0,
            "active_episode_ids": [],
        }
        if self.model_id is not None:
            health["model_id"] = self.model_id
        self._json(health)

    def do_POST(self) -> None:
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"] or 0)) or b"{}")
        if self.path == "/reset":
            # The first frame is a real base64 JPEG produced by the client.
            # No frame rides with the reset: the first `/act` carries it, and
            # sending it in both made a stateful policy see step 0 twice.
            assert "first_frame_jpeg_b64" not in payload
            self.server.seen_instructions[payload["episode_id"]] = payload["instruction"]  # type: ignore[attr-defined]
            self._json({"success": True})
            return
        if self.path == "/act":
            assert base64.b64decode(payload["image_jpeg_b64"])[:2] == b"\xff\xd8"
            frame_id = int(payload["frame_id"])
            self.server.act_calls.append((payload["episode_id"], frame_id))  # type: ignore[attr-defined]
            if frame_id >= self.stop_after_frames:
                self._json(
                    {
                        "success": True,
                        "action_id": 0,
                        "action_name": "STOP",
                        "raw_output": "stop",
                        "frame_id": frame_id,
                    }
                )
                return
            self._json(
                {
                    "success": True,
                    "action_id": 1,
                    "action_name": "WAYPOINT",
                    "raw_output": "",
                    "waypoint": {
                        "cluster_id": 3,
                        "waypoints": [
                            {"forward_m": 0.5, "lateral_m": 0.0, "yaw_rad": 0.0} for _ in range(4)
                        ],
                    },
                    "waypoint_cluster_id": 3,
                    "waypoint_horizon": 4,
                    "frame_id": frame_id,
                    "inference_time_ms": 1.0,
                }
            )
            return
        if self.path == "/finish":
            self.server.finished.append(payload["episode_id"])  # type: ignore[attr-defined]
            self._json({"success": True})
            return
        self._json({"error": "not found"}, status=404)


@pytest.fixture
def policy_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _PolicyHandler)
    server.seen_instructions = {}  # type: ignore[attr-defined]
    server.act_calls = []  # type: ignore[attr-defined]
    server.finished = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    server.url = f"http://127.0.0.1:{server.server_address[1]}"  # type: ignore[attr-defined]
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# --- the substituted simulator ---------------------------------------------


def _frame() -> np.ndarray:
    """A textured 16x16 frame: the blank-scene guard is on for this suite."""
    grid = np.arange(16 * 16 * 3, dtype=np.uint16).reshape(16, 16, 3)
    return ((grid * 37) % 251).astype(np.uint8)


class _ApproachWorld:
    """Moves the camera toward the goal, halting at ``stop_distance_m`` from it."""

    def __init__(self, episode, *, stop_distance_m: float):
        self.episode = episode
        self.stop_distance_m = float(stop_distance_m)
        self.goal = episode.goal_position or episode.start_pose.position
        self.pose = CameraPose(*episode.start_pose.position, 0.0)

    # -- CameraWalkBackend surface
    def resolve_start_pose(self, pose: CameraPose) -> CameraPose:
        return self._facing(pose)

    def resolve_motion(self, prev: CameraPose, candidate: CameraPose) -> CameraPose:
        moved = math.hypot(candidate.x - prev.x, candidate.y - prev.y) > 1e-9
        if not moved:
            return self._facing(prev)
        remaining = math.hypot(self.goal[0] - prev.x, self.goal[1] - prev.y)
        target = max(self.stop_distance_m, remaining * 0.5)
        if remaining <= 1e-9:
            return self._facing(prev)
        ratio = (remaining - target) / remaining
        return self._facing(
            CameraPose(
                prev.x + (self.goal[0] - prev.x) * ratio,
                prev.y + (self.goal[1] - prev.y) * ratio,
                prev.z,
                prev.yaw,
            )
        )

    def reset(self, pose, *, warmup_steps, converge_max_steps, converge_tol, log):
        self.pose = pose
        self.reset_kwargs = {
            "warmup_steps": warmup_steps,
            "converge_max_steps": converge_max_steps,
            "converge_tol": converge_tol,
        }
        return _frame()

    def set_pose(self, pose: CameraPose) -> None:
        self.pose = pose

    def step_and_capture(self, *, settle_steps: int):
        return _frame()

    def capture_observation(self):
        from insight_bench.simulator.observation import make_observation, quat_wxyz_from_yaw

        return make_observation(
            _frame(),
            pose=self.pose,
            orientation_wxyz=quat_wxyz_from_yaw(self.pose.yaw),
        )

    def _facing(self, pose: CameraPose) -> CameraPose:
        return CameraPose(
            pose.x, pose.y, pose.z, math.atan2(self.goal[1] - pose.y, self.goal[0] - pose.x)
        )


class StandInBackend:
    """The surface `insight_bench_runner` requires of a loaded backend."""

    def __init__(self, *, stop_distance_m: float = 2.5):
        self.stop_distance_m = stop_distance_m
        self.opened: list[tuple[str, str | None]] = []
        self.closed = 0
        self.reset_kwargs: list[dict] = []
        self._task_config = None

    def runtime_attestation(self) -> RuntimeAttestation:
        return RuntimeAttestation(
            backend_id="isaac-5.1",
            backend_status="official",
            launcher_id="linux-native",
            runtime_kind="native",
            os="linux",
        )

    def open_scene(self, task_config, *, scene_extras=None, log=None) -> None:
        self.opened.append((task_config.name, task_config.terrain.usd_path))
        self._task_config = task_config

    def close(self, *, log=None) -> None:
        self.closed += 1

    def run_episode_with_policy(self, episode, policy, *, options=None, log=None):
        assert self._task_config is not None, "run before open_scene"
        assert self._task_config.terrain.usd_path == episode.scene.asset_path
        world = _ApproachWorld(episode, stop_distance_m=self.stop_distance_m)
        rollout = run_camera_walk_episode(
            env=world,
            policy=policy,
            episode=episode,
            task_config=self._task_config,
            options=options or RolloutOptions.from_task_config(self._task_config),
        )
        self.reset_kwargs.append(world.reset_kwargs)
        return rollout


# --- fixtures ---------------------------------------------------------------


@pytest.fixture
def manifest() -> BenchmarkManifest:
    """The shipped v2 manifest, re-pinned to the seven-episode fixture."""
    published = Registry.load().get(COORDINATE)
    return published.model_copy(
        update={
            "dataset": published.dataset.model_copy(update={"sha256": sha256_file(FIXTURE)}),
            "metadata": {**published.metadata, "episode_count": 7},
        }
    )


@pytest.fixture
def scene_root(tmp_path: Path) -> Path:
    root = tmp_path / "scenes"
    for asset in SCENE_ASSETS:
        path = root / asset
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#usda 1.0\n", encoding="utf-8")
    return root


@pytest.fixture
def stand_in(monkeypatch) -> StandInBackend:
    backend = StandInBackend()
    monkeypatch.setattr(runner_module, "load_backend", lambda backend_id: backend)
    return backend


@pytest.fixture
def context(scene_root, policy_server) -> RunContext:
    return RunContext(scene_root=scene_root, policy_url=policy_server.url)


def _run(
    manifest,
    context,
    *,
    seed: int = 0,
    max_episodes: int | None = None,
    artifact_dir: Path | None = None,
    resume: bool = False,
) -> object:
    if max_episodes is not None:
        context = replace(context, max_episodes=max_episodes)
    if artifact_dir is not None:
        context = replace(context, artifact_dir=artifact_dir)
    if resume:
        context = replace(context, resume=True)
    return insight_bench_runner(manifest, FIXTURE, seed, context=context)


# --- the executing path -----------------------------------------------------


def test_a_full_run_produces_a_submittable_result(manifest, context, stand_in) -> None:
    result = _run(manifest, context)

    assert result.benchmark_id == "insight-bench-v1"
    assert result.benchmark_version == "1.0.0"
    # Three-part semver: the submission service rejects "2.0".
    assert result.benchmark_version.count(".") == 2
    assert result.status == "completed"
    assert len(result.episodes) == 7
    assert [episode.episode_id for episode in result.episodes] == sorted(
        INDOOR_EPISODES | OUTDOOR_EPISODES
    )
    # The headline metric key is `success-rate`, not a display label.
    assert "success-rate" in result.metrics
    assert "Success Rate" not in result.metrics
    for episode in result.episodes:
        assert episode.task_id == "objnav"
        assert episode.response is not None
        assert set(episode.metrics) >= {"success", "spl", "distance_to_goal", "path_length"}


def test_the_success_radius_policy_decides_the_mixed_result(manifest, context, stand_in) -> None:
    # Every episode ends 2.5 m from its object, facing it. Outdoor episodes are
    # scored at 3.0 m and pass; indoor ones at 2.0 m and fail. Same behaviour,
    # opposite verdicts, decided only by metadata.scene_class.
    result = _run(manifest, context)
    by_status = {episode.episode_id: episode.status for episode in result.episodes}
    assert {name for name, status in by_status.items() if status == "passed"} == OUTDOOR_EPISODES
    assert {name for name, status in by_status.items() if status == "failed"} == INDOOR_EPISODES
    assert result.metrics["success-rate"] == pytest.approx(2 / 7)
    # Failing on the merits is not an execution error, and a run of them is
    # still a completed run.
    assert all(episode.error is None for episode in result.episodes)
    assert result.status == "completed"


def test_reaching_the_object_passes_every_episode(manifest, context, monkeypatch) -> None:
    monkeypatch.setattr(
        runner_module, "load_backend", lambda _id: StandInBackend(stop_distance_m=1.0)
    )
    result = _run(manifest, context)
    assert result.metrics["success-rate"] == pytest.approx(1.0)
    assert all(episode.status == "passed" for episode in result.episodes)


def test_each_scene_asset_is_opened_exactly_once(manifest, context, stand_in) -> None:
    # Loading a scene is the expensive part of an objnav run, and two fixture
    # episodes share one HM3D scene: six scene assets, seven episodes. The key
    # is the asset, not the scene id -- 8 of the suite's 210 scenes are reached
    # through two entry points, and two entry points are two stages to compose.
    _run(manifest, context)
    opened = [usd for _name, usd in stand_in.opened]
    assert sorted(opened) == sorted(str(context.scene_root / asset) for asset in SCENE_ASSETS)
    assert len(opened) == len(set(opened)) == len(SCENE_ASSETS)
    assert {name for name, _usd in stand_in.opened} == {"insight_bench"}
    assert stand_in.closed == len(SCENE_ASSETS)


def test_mesh_scene_families_are_rendered_before_the_gaussian_splat_ones(
    manifest, context, stand_in
) -> None:
    # Not cosmetic. Loading a Gaussian-splat stage leaves renderer state that
    # persists for the life of the process and shifts how mesh scenes render
    # afterwards; this process cannot restart itself, so the ordering is the
    # remedy. Every mesh asset must be opened before the first splat one.
    _run(manifest, context)
    families = [
        Path(usd).relative_to(context.scene_root).parts[0] for _name, usd in stand_in.opened
    ]
    mesh = {"hm3d", "mp3d"}
    neural = {"habitat_gs", "interiorgs"}
    assert set(families) == mesh | neural
    first_neural = min(index for index, family in enumerate(families) if family in neural)
    assert all(family in mesh for family in families[:first_neural])
    assert all(family in neural for family in families[first_neural:])
    # Families stay contiguous, so each is entered exactly once: dropping
    # repeats leaves the four families in the order they were rendered.
    entered = [
        family
        for index, family in enumerate(families)
        if index == 0 or family != families[index - 1]
    ]
    assert entered == ["hm3d", "mp3d", "habitat_gs", "interiorgs"]


def test_the_first_frame_convergence_gate_reaches_the_backend(manifest, context, stand_in) -> None:
    # The gate is what makes the first frame reproducible on this scene mix.
    # Its budget comes from the suite's rollout block, and zero -- the dataclass
    # default -- disables it, so assert the calibrated values arrive at reset.
    _run(manifest, context)
    assert stand_in.reset_kwargs
    for kwargs in stand_in.reset_kwargs:
        assert kwargs["converge_max_steps"] == 60
        assert kwargs["converge_tol"] == pytest.approx(0.5)
        assert kwargs["warmup_steps"] == 3


def test_the_run_talks_the_real_policy_protocol(manifest, context, stand_in, policy_server) -> None:
    result = _run(manifest, context)
    episode_ids = {episode.episode_id for episode in result.episodes}
    # reset once per episode, with that episode's instruction ...
    assert set(policy_server.seen_instructions) == episode_ids
    assert all(text for text in policy_server.seen_instructions.values())
    # ... act with a monotonic frame_id per episode ...
    per_episode: dict[str, list[int]] = {}
    for episode_id, frame_id in policy_server.act_calls:
        per_episode.setdefault(episode_id, []).append(frame_id)
    assert set(per_episode) == episode_ids
    for frames in per_episode.values():
        assert frames == list(range(len(frames)))
        assert frames[-1] == _PolicyHandler.stop_after_frames
    # ... and finish once per episode.
    assert sorted(policy_server.finished) == sorted(episode_ids)


def test_the_model_identity_comes_from_the_policy_server(manifest, context, stand_in) -> None:
    # The model is whatever answered the URL, so that is what the result says
    # produced its numbers -- nothing in this process names the model.
    result = _run(manifest, context)
    assert result.adapter.model_id == MODEL_ID
    assert result.adapter.adapter_id == "vln-http-policy"
    assert result.adapter.metadata["policy_protocol"] == "vln-http/1"
    assert result.adapter.metadata["model_version"] == "0.0.1"
    assert result.adapter.deterministic is False
    # The endpoint itself is deployment detail and stays out of the evidence.
    assert not any("127.0.0.1" in str(value) for value in result.adapter.metadata.values())


def test_the_result_carries_the_backends_attestation_unaltered(manifest, context, stand_in) -> None:
    result = _run(manifest, context)
    attestation = result.runtime_attestation
    assert attestation is not None
    assert attestation.backend_id == "isaac-5.1"
    assert attestation.backend_status == "official"
    assert attestation.os == "linux"
    # The stand-in recorded no driver, so the run is not publishable and the
    # runner neither grants nor withholds that -- it passes the record through.
    assert attestation.driver_version == ""
    assert attestation.publishable is False


def test_the_run_is_reproducibly_identified(manifest, context, stand_in) -> None:
    first = _run(manifest, context)
    second = _run(manifest, context)
    other_seed = _run(manifest, context, seed=7)
    assert (first.run_id, first.run_fingerprint) == (second.run_id, second.run_fingerprint)
    assert other_seed.run_id != first.run_id


def test_per_episode_traces_are_written_as_evidence(manifest, context, stand_in, tmp_path) -> None:
    artifact_dir = tmp_path / "artifacts"
    result = _run(
        manifest,
        RunContext(
            scene_root=context.scene_root,
            policy_url=context.policy_url,
            artifact_dir=artifact_dir,
        ),
    )
    for episode in result.episodes:
        trace = json.loads((artifact_dir / episode.episode_id / "trace.json").read_text())
        assert trace["episode_id"] == episode.episode_id
        assert trace["steps"], "a trace with no steps is not evidence"
        assert trace["metadata"]["task"] == "insight_bench"
        # Poses must be measurements, not the commands echoed back.
        assert trace["metadata"]["initial_pose_source"] == "simulator-readback"


# --- the smoke-sized subset (`--max-episodes`) ------------------------------


def test_max_episodes_runs_exactly_that_many_episodes(manifest, context, stand_in) -> None:
    """A truncated run is a short run, not a different kind of run.

    The point of the flag is proving a setup works without paying for the whole
    suite, so what comes back has to be the same object a full run produces --
    scored episodes, over the scenes those episodes actually need. It differs in
    exactly one way: it reports `partial`, because 2 of 7 episodes is not a
    measurement of a 7-episode benchmark.
    """
    result = _run(manifest, context, max_episodes=2)

    assert result.status == "partial"
    assert len(result.episodes) == 2
    # The first two records of the file, which share one HM3D scene -- so one
    # scene load, not seven.
    assert [episode.episode_id for episode in result.episodes] == [
        "objnav_00013-sfbj7jspYWj_4",
        "objnav_00013-sfbj7jspYWj_5",
    ]
    assert len(stand_in.opened) == 1
    for episode in result.episodes:
        assert episode.task_id == "objnav"
        assert episode.error is None
        assert set(episode.metrics) >= {"success", "spl", "distance_to_goal", "path_length"}


def test_max_episodes_lifts_the_episode_count_check_it_would_otherwise_trip(
    manifest, context, stand_in, tmp_path
) -> None:
    # This exact file is refused without the flag (see the truncated-file test
    # below): the manifest is defined over 7 episodes and it holds 3. Asking for
    # a subset is the one way a run over fewer than 7 is allowed to proceed.
    short = tmp_path / "short.jsonl"
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()[:3]
    short.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = insight_bench_runner(manifest, short, 0, context=replace(context, max_episodes=2))
    assert len(result.episodes) == 2
    # ... and the run says it is not the whole benchmark. The flag lifts the
    # count check so the run can proceed; it does not make 2 of 7 a measurement.
    assert result.status == "partial"


def test_a_truncated_run_says_so_in_the_identity_it_publishes(manifest, context, stand_in) -> None:
    # A partial measurement that does not admit it is one is the failure mode.
    # The note rides on the run's own model identity, so it survives into the
    # run result, the evidence pack and anything reading either.
    subset = _run(manifest, context, max_episodes=3)
    note = subset.adapter.metadata["episode_subset"]
    assert "3 of 7" in note
    assert "not a full measurement" in note
    # ... and a full run carries no such note, so the label means something.
    assert "episode_subset" not in _run(manifest, context).adapter.metadata


# --- the refusing paths -----------------------------------------------------


def test_a_missing_scene_root_is_refused_by_name(manifest, policy_server) -> None:
    with pytest.raises(RunnerError, match="needs --scene-root"):
        _run(manifest, RunContext(policy_url=policy_server.url))


def test_missing_scene_assets_are_all_reported_before_anything_starts(
    manifest, policy_server, tmp_path, stand_in
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RunnerError) as caught:
        _run(manifest, RunContext(scene_root=empty, policy_url=policy_server.url))
    message = str(caught.value)
    assert "7 of 7 episodes have no scene asset" in message
    # Reported once per missing asset, not once per episode: the user needs the
    # list of files to go and place -- six of them here, over seven episodes.
    assert "across 6 missing scene assets" in message
    assert "hm3d/00013-sfbj7jspYWj/sfbj7jspYWj.usd" in message
    # Nothing was opened: the refusal happens before the simulator is used.
    assert stand_in.opened == []


def test_a_truncated_episode_file_cannot_be_run_as_the_coordinate(
    manifest, context, stand_in, tmp_path
) -> None:
    short = tmp_path / "short.jsonl"
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()[:3]
    short.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(RunnerError, match="defined over 7 episodes but the episode file holds 3"):
        insight_bench_runner(manifest, short, 0, context=context)


def test_an_empty_episode_file_is_refused(manifest, context, stand_in, tmp_path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(RunnerError, match="episode file is empty"):
        insight_bench_runner(manifest, empty, 0, context=context)


def test_no_policy_server_is_refused_with_the_url_it_tried(manifest, scene_root, stand_in) -> None:
    with pytest.raises(RunnerError) as caught:
        _run(manifest, RunContext(scene_root=scene_root, policy_url="http://127.0.0.1:1/"))
    message = str(caught.value)
    assert "/health" in message
    assert "--policy-url" in message
    # A refused run still tears the scene down and never fabricates episodes.
    assert stand_in.opened == []


def test_a_policy_server_with_no_model_id_is_refused(
    manifest, context, stand_in, monkeypatch
) -> None:
    monkeypatch.setattr(_PolicyHandler, "model_id", None)
    with pytest.raises(RunnerError, match="reported no model_id"):
        _run(manifest, context)


def test_a_task_family_this_runner_cannot_drive_is_refused(manifest, context, stand_in) -> None:
    stripped = manifest.model_copy(
        update={"metadata": {**manifest.metadata, "task_family": "vlnce_r2r"}}
    )
    with pytest.raises(RunnerError, match="is not a family the"):
        _run(stripped, context)


def test_an_undeclared_scene_dataset_is_refused(manifest, context, stand_in) -> None:
    # The licence block is the published record of what a reader may reproduce,
    # so it has to cover what the run actually loaded.
    stripped = manifest.model_copy(
        update={
            "asset_licenses": tuple(
                item for item in manifest.asset_licenses if item.family != "hm3d"
            )
        }
    )
    with pytest.raises(RunnerError, match="declares no licence for: hm3d"):
        _run(stripped, context)


def test_an_episode_that_crashes_is_recorded_as_an_error_not_a_pass(
    manifest, context, stand_in
) -> None:
    def explode(episode, policy, *, options=None, log=None):
        raise RuntimeError("renderer died")

    stand_in.run_episode_with_policy = explode
    result = _run(manifest, context)
    assert result.status == "failed"
    assert result.metrics["success-rate"] == 0.0
    assert all(episode.status == "error" for episode in result.episodes)
    assert all("renderer died" in (episode.error or "") for episode in result.episodes)
    assert stand_in.closed == len(SCENE_ASSETS)


def test_a_blank_scene_frame_is_recorded_and_the_run_still_completes(
    manifest, context, stand_in, monkeypatch
) -> None:
    """A flat first frame is a note on the episode, not the end of the run.

    A user-provided USD that did not load renders flat, and a reader has to be
    able to tell that from a model that merely did badly. It used to error the
    episode, which cost the whole suite its `completed` status over a single
    start pose -- and one episode of the published suite legitimately starts
    facing a bare wall, so every full run came out unsubmittable.
    """
    monkeypatch.setattr(
        "insight_bench.vln_runtime.rollout.camera_walk.np.asarray",
        lambda *args, **kwargs: np.zeros((4, 4, 3), dtype=np.uint8),
    )
    result = _run(manifest, context)
    assert result.status == "completed"
    assert all(episode.error is None for episode in result.episodes)


def test_without_a_runtime_the_run_refuses_with_the_environmental_reason(
    manifest, context, monkeypatch
) -> None:
    def refuse(backend_id: str):
        raise SimulatorNotAvailableError(
            f"{backend_id} cannot execute here: missing Isaac import surface: isaacsim"
        )

    monkeypatch.setattr(runner_module, "load_backend", refuse)
    with pytest.raises(RunnerError) as caught:
        _run(manifest, context)
    message = str(caught.value)
    assert "7 episodes validated and ready" in message
    assert "isaac-5.1 cannot execute here" in message


# --- the CLI and registry plumbing -----------------------------------------


@pytest.fixture
def cli_registry(manifest, tmp_path: Path) -> Path:
    """A registry root holding the re-pinned v2 manifest, for the CLI path."""
    root = tmp_path / "registry"
    relative = f"{manifest.benchmark_id}/{manifest.version}/manifest.json"
    (root / relative).parent.mkdir(parents=True, exist_ok=True)
    write_stable_json(root / relative, manifest)
    write_stable_json(
        root / "index.json",
        RegistryContract(
            registry_version="1.0.0",
            entries=(
                RegistryEntry(
                    benchmark_id=manifest.benchmark_id,
                    version=manifest.version,
                    manifest_path=relative,
                    sha256=sha256_file(root / relative),
                ),
            ),
        ),
    )
    return root


def test_the_cli_runs_the_suite_against_a_policy_url(
    cli_registry, scene_root, policy_server, stand_in, tmp_path, capsys
) -> None:
    from insight_bench.cli import main

    output = tmp_path / "run-result.json"
    exit_code = main(
        [
            "run",
            COORDINATE,
            "--registry",
            str(cli_registry),
            "--episodes",
            str(FIXTURE),
            "--scene-root",
            str(scene_root),
            "--policy-url",
            policy_server.url,
            "--output",
            str(output),
        ]
    )
    capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["benchmark_id"] == "insight-bench-v1"
    assert payload["benchmark_version"] == "1.0.0"
    assert payload["adapter"]["model_id"] == MODEL_ID
    assert payload["metrics"]["success-rate"] == pytest.approx(2 / 7)
    assert payload["runtime_attestation"]["backend_id"] == "isaac-5.1"
    assert len(payload["episodes"]) == 7


def test_the_cli_defaults_the_policy_url_to_the_pinned_loopback_endpoint(scene_root) -> None:
    from insight_bench.cli import build_parser

    args = build_parser().parse_args(
        ["run", "--episodes", str(FIXTURE), "--scene-root", str(scene_root)]
    )
    assert args.policy_url == DEFAULT_POLICY_URL == "http://127.0.0.1:18081"
    # A bare `run` evaluates the published suite this file drives, whole.
    assert args.benchmark == COORDINATE
    assert args.max_episodes is None


def test_the_cli_will_not_guess_the_episode_file_or_the_scene_root() -> None:
    # Both are user-provided by licence, so there is no location to fall back
    # to; the parser refuses rather than running against something else.
    from insight_bench.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", COORDINATE])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", COORDINATE, "--episodes", str(FIXTURE)])


def test_running_without_the_episode_file_is_refused_by_name(cli_registry) -> None:
    registry = Registry.load(cli_registry)
    with pytest.raises(RunnerError, match="--episodes"):
        run_benchmark(COORDINATE, registry=registry)


def test_an_episode_file_that_is_not_the_published_one_is_refused(
    cli_registry, scene_root, policy_server, stand_in, tmp_path
) -> None:
    tampered = tmp_path / "tampered.jsonl"
    records = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]
    records[0]["goal_position"] = [0.0, 0.0, 0.0]
    tampered.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records)
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RunnerError, match="does not match the published dataset digest"):
        run_benchmark(
            COORDINATE,
            registry=Registry.load(cli_registry),
            context=RunContext(
                episode_dataset=tampered,
                scene_root=scene_root,
                policy_url=policy_server.url,
            ),
        )


def test_the_published_file_passes_the_digest_gate(
    cli_registry, scene_root, policy_server, stand_in
) -> None:
    result = run_benchmark(
        COORDINATE,
        registry=Registry.load(cli_registry),
        context=RunContext(
            episode_dataset=FIXTURE,
            scene_root=scene_root,
            policy_url=policy_server.url,
        ),
    )
    assert result.status == "completed"
    assert result.benchmark_id == "insight-bench-v1"


def test_the_runner_stays_registered_under_its_stable_id() -> None:
    from insight_bench.runner import available_runners

    assert RUNNER_ID in available_runners()
    assert Registry.load().get(COORDINATE).runtime.runner == RUNNER_ID


def test_a_failed_episode_is_explained_by_the_protocol_that_scored_it(
    manifest, context, stand_in
) -> None:
    """The reason and the metrics must come from one success criterion.

    These records carry a ``metadata["ovon"]`` block as well as their
    NavNuances one, and the shared scorer matched the ObjectNav branch first --
    so a run's ``failure_reason`` was an ObjectNav verdict (at ObjectNav's own
    2.5 m distance) sitting beside NavNuances metrics.
    """
    result = _run(manifest, context)
    reasons = {
        episode.episode_id: (
            episode.response.metadata["failure_reason"] if episode.response else None
        )
        for episode in result.episodes
    }
    assert {reasons[name] for name in INDOOR_EPISODES} == {"criterion_not_met"}
    assert {reasons[name] for name in OUTDOOR_EPISODES} == {None}
    # And the capability label says which protocol branch scored it.
    labels = {
        episode.response.metadata["score_labels"]["capability"]
        for episode in result.episodes
        if episode.response
    }
    assert labels == {"LR_towards"}


def test_the_manifest_declares_every_metric_the_result_carries(manifest, context, stand_in) -> None:
    # A metric the result carries but the manifest omits is a metric no
    # consumer knows to read, and one the manifest claims but the result never
    # produces is a promise nothing keeps.
    result = _run(manifest, context)
    assert set(result.metrics) == set(manifest.metrics)


# --- replay material (`run --vis`) ------------------------------------------


class _FrameSinkBackend(StandInBackend):
    """A backend that accepts the frames-era keyword.

    :class:`StandInBackend` deliberately does not, and stays that way: a
    backend written before ``--vis`` must keep working, which is what the
    conditional keyword in ``run_scored_episode`` guarantees and what every
    other test in this file exercises.
    """

    def run_episode_with_policy(self, episode, policy, *, options=None, log=None, frame_sink=None):
        assert self._task_config is not None, "run before open_scene"
        world = _ApproachWorld(episode, stop_distance_m=self.stop_distance_m)
        rollout = run_camera_walk_episode(
            env=world,
            policy=policy,
            episode=episode,
            task_config=self._task_config,
            options=options or RolloutOptions.from_task_config(self._task_config),
            frame_sink=frame_sink,
        )
        self.reset_kwargs.append(world.reset_kwargs)
        return rollout


def test_vis_leaves_replay_material_beside_every_trace(manifest, context, monkeypatch, tmp_path):
    """The published coordinate is what `--vis` exists for, so wire it here too.

    Four frames per episode over this suite's 300 steps means a stride of 75:
    sampled across the whole episode rather than truncated to its opening, so
    every episode of a long run stays inspectable.
    """
    from insight_bench.runner import VisOptions

    monkeypatch.setattr(runner_module, "load_backend", lambda _id: _FrameSinkBackend())
    artifact_dir = tmp_path / "artifacts"
    result = _run(
        manifest,
        RunContext(
            scene_root=context.scene_root,
            policy_url=context.policy_url,
            artifact_dir=artifact_dir,
            vis=VisOptions(max_frames_per_episode=4),
        ),
    )

    vis_manifest = json.loads((artifact_dir / "vis-manifest.json").read_text())
    assert vis_manifest["frame_stride"] == 75
    assert vis_manifest["episode_count"] == len(result.episodes)
    assert vis_manifest["frames_failed"] == 0
    assert vis_manifest["frames_written"] > 0

    written = 0
    for episode in result.episodes:
        episode_dir = artifact_dir / episode.episode_id
        sidecar = json.loads((episode_dir / "episode.json").read_text())
        # The whitelist, on the path where scene.asset_path is always populated.
        assert sidecar["episode_id"] == episode.episode_id
        assert sidecar["scene_id"]
        # Carried verbatim, whatever the dataset says: the page shows the values
        # it is given rather than the ones it recognises. The published vocabulary
        # is English, and SCENE_CLASS_ORDER is what keys the scene-class radar on
        # it -- so a value outside it would render off the designed axis order.
        assert sidecar["scene_class"] in set(SCENE_CLASS_ORDER)
        assert sidecar["instr_type"]
        assert sidecar["suite"]
        assert "asset_path" not in (episode_dir / "episode.json").read_text()
        assert str(context.scene_root) not in (episode_dir / "episode.json").read_text()

        steps = json.loads((episode_dir / "trace.json").read_text())["steps"]
        paths = [str(step["observation"].get("frame_path", "")) for step in steps]
        assert paths[0] == "", "trace step 0 made no decision and was shown no frame"
        indexes = [index for index, path in enumerate(paths) if path]
        assert indexes and all((index - 1) % 75 == 0 for index in indexes)
        for index in indexes:
            frame = episode_dir / f"frames/step_{index:04d}.jpg"
            assert paths[index] == f"frames/step_{index:04d}.jpg"
            assert frame.read_bytes()[:3] == b"\xff\xd8\xff"
        written += len(indexes)
    assert vis_manifest["frames_written"] == written


def test_a_full_run_over_every_episode_is_completed(manifest, context, stand_in) -> None:
    """The counterpart, so `partial` above means something.

    Truncation is what downgrades a truncated run, not the flag being present or
    the run being small: this one runs the whole 7-episode benchmark and comes
    back `completed`.
    """
    result = _run(manifest, context)

    assert result.status == "completed"
    assert len(result.episodes) == 7
    assert "episode_subset" not in result.adapter.metadata


# --- resuming an interrupted run --------------------------------------------


def test_every_finished_episode_leaves_its_score_on_disk(manifest, context, stand_in, tmp_path):
    """The file that makes an interrupted run worth something.

    A trace records what the rollout did; nothing on disk recorded what it
    scored, so a run that stopped before writing its result had produced
    evidence and no measurement.
    """
    run_dir = tmp_path / "run"
    result = _run(manifest, context, artifact_dir=run_dir)

    for episode in result.episodes:
        stored = json.loads((run_dir / episode.episode_id / "result.json").read_text())
        assert stored["episode_id"] == episode.episode_id
        assert stored["status"] == episode.status
        assert stored["metrics"] == pytest.approx(episode.metrics)
    assert (run_dir / "resume-key.json").is_file()


def test_resume_reuses_finished_episodes_and_runs_only_the_rest(
    manifest, context, stand_in, tmp_path
):
    run_dir = tmp_path / "run"
    _run(manifest, context, max_episodes=3, artifact_dir=run_dir)
    scenes_opened_first = len(stand_in.opened)

    stand_in.opened.clear()
    resumed = _run(manifest, context, artifact_dir=run_dir, resume=True)

    assert resumed.status == "completed"
    assert len(resumed.episodes) == 7
    # The three already scored were not run again: only the scenes the remaining
    # four need were opened, which is the cost the flag exists to avoid.
    assert scenes_opened_first > 0
    assert len(stand_in.opened) < len(SCENE_ASSETS)


def test_resume_refuses_a_directory_written_by_a_different_model(
    manifest, context, stand_in, tmp_path
):
    """The failure this guard exists for is silent, not loud.

    Adopting another model's episodes would average two measurements into one
    success rate, and the number would look entirely ordinary.
    """
    run_dir = tmp_path / "run"
    _run(manifest, context, max_episodes=3, artifact_dir=run_dir)

    _PolicyHandler.model_id = "other-lab/imposter-v1"
    try:
        with pytest.raises(RunnerError) as caught:
            _run(manifest, context, artifact_dir=run_dir, resume=True)
    finally:
        _PolicyHandler.model_id = MODEL_ID
    message = str(caught.value)
    assert "refusing to resume" in message
    assert "model_id" in message


def test_resume_without_a_previous_run_says_so(manifest, context, stand_in, tmp_path):
    with pytest.raises(RunnerError, match="needs a previous run"):
        _run(manifest, context, artifact_dir=tmp_path / "empty", resume=True)


def test_a_corrupt_stored_result_is_rerun_rather_than_trusted(
    manifest, context, stand_in, tmp_path
):
    run_dir = tmp_path / "run"
    first = _run(manifest, context, artifact_dir=run_dir)
    victim = first.episodes[0].episode_id
    (run_dir / victim / "result.json").write_text("{ not json", encoding="utf-8")

    resumed = _run(manifest, context, artifact_dir=run_dir, resume=True)

    assert resumed.status == "completed"
    assert len(resumed.episodes) == 7
    assert any(episode.episode_id == victim for episode in resumed.episodes)


def test_the_templates_placeholder_model_id_is_refused(manifest, context, stand_in) -> None:
    """Non-empty is not the same as filled in.

    `policies/template` ships `model_id = "TODO/your-model"`, which passed the
    emptiness check and would have become the identity of a whole run in the
    evidence pack -- for anyone who copied the template and forgot one line.
    """
    _PolicyHandler.model_id = "TODO/your-model"
    try:
        with pytest.raises(RunnerError) as caught:
            _run(manifest, context)
    finally:
        _PolicyHandler.model_id = MODEL_ID

    message = str(caught.value)
    assert "placeholder" in message
    assert "TODO/your-model" in message


def test_a_real_model_id_is_accepted_unchanged(manifest, context, stand_in) -> None:
    result = _run(manifest, context)

    assert result.adapter.model_id == MODEL_ID


def test_a_policy_whose_model_dies_mid_run_stops_the_run(
    manifest, context, stand_in, policy_server, monkeypatch
) -> None:
    """One refusal with a reason, instead of every episode failing separately.

    /health used to be read once, before the first episode. A policy whose
    model died after that kept answering, so ten episodes were each driven and
    each failed on connection refused, one at a time, with the card held for
    all of them -- and the evidence recorded the startup reading, attesting
    `healthy: true` for a run whose model was gone throughout.
    """
    monkeypatch.setattr(_PolicyHandler, "health_calls", 0)
    monkeypatch.setattr(_PolicyHandler, "unhealthy_after", 1)  # the pre-run read, then gone

    with pytest.raises(RunnerError) as raised:
        _run(manifest, context)

    message = str(raised.value)
    assert "healthy=false" in message
    assert "re-run to resume" in message


def test_a_liveness_reading_is_not_published_as_the_run_identity(
    manifest, context, stand_in
) -> None:
    """`healthy` describes an instant, and the descriptor describes a run."""
    result = _run(manifest, context)
    assert "healthy" not in result.adapter.metadata
    assert "policy_child_alive" not in result.adapter.metadata
    # What the server says about its configuration still travels.
    assert result.adapter.metadata["action_mode"] == "waypoint"
