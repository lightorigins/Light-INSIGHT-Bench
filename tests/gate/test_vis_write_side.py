"""`run --vis` writes the replay material, and a run without it writes today's bytes.

CI has no Isaac and no model, so two things are substituted and nothing else is:
the simulator backend, at the one seam the runner uses to obtain one, and the
policy client, at the one seam the runner uses to obtain that. The backend
stand-in accepts ``frame_sink`` through ``**kwargs`` precisely so this file can
assert that a run *without* ``--vis`` never receives the keyword at all.

The policy is substituted in process rather than served over HTTP. The wire
protocol has its own gate (``tests/gate/test_insight_bench_runner.py`` drives a
real stdlib server through the shipped client); what this file is about is what
a run *writes*, and keeping the model in process leaves the JPEG encoder used by
exactly one thing -- the frame sink -- which is what makes the "visualisation
failed" case below observable without also breaking the transport.

Everything between those two substitutions is the production path: the published
episode records, scene resolution against a scene root, the rollout loop, the
trace builder that decides which step a frame belongs to, the frame sink, the
whitelist that keeps the user's own filesystem out of ``episode.json``, and the
run manifest.

The default matters as much as the feature. ``--vis`` is opt-in, and a run
without it must produce exactly what it produced before ``--vis`` existed: no
``frame_path`` key in the records, therefore the same trace bytes, the same run
directory, and the same run identity.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from insight_bench.contracts import RuntimeAttestation
from insight_bench.registry import Registry
from insight_bench.runner import RunContext, RunnerError, VisOptions
from insight_bench.runners import _shared
from insight_bench.runners import insight_bench as runner_module
from insight_bench.runners._shared import VisSession, open_vis_session
from insight_bench.vln_runtime.episodes import EpisodeSpec
from insight_bench.vln_runtime.policy import client as policy_client
from insight_bench.vln_runtime.policy.client import PolicyResponse
from insight_bench.vln_runtime.rollout import RolloutOptions, run_camera_walk_episode
from insight_bench.vln_runtime.traces.frames import JpegFrameSink

COORDINATE = "insight-bench-v1@1.0.0"
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "objnav_published_episodes.jsonl"
JPEG_MAGIC = b"\xff\xd8\xff"

# The scene assets the seven fixture episodes name, relative to a scene root.
SCENE_ASSETS = (
    "habitat_gs/scene56/scene56.usd",
    "habitat_gs/scene58/scene58_zup.usda",
    "habitat_gs/scene63/scene63_zup.usda",
    "hm3d/00013-sfbj7jspYWj/sfbj7jspYWj.usd",
    "interiorgs/interior_0405_840145/interior_0405_840145_zup.usda",
    "mp3d/8194nk5LbLH/8194nk5LbLH.usd",
)


class WalkingPolicy:
    """The model, in process: walks forward for the whole episode.

    Never stops, so every episode runs the full step budget and there are long
    traces to sample frames out of. Its ``health`` is what the runner records as
    the model identity of the run.
    """

    def __init__(self, base_url: str = "", **_: object) -> None:
        self.base_url = base_url
        self.closed = 0

    def health(self) -> dict[str, object]:
        return {"healthy": True, "model_id": "test/vis-walker", "action_mode": "waypoint"}

    def reset(self, instruction: str, *, episode_id: str = "", first_frame_rgb=None):
        assert first_frame_rgb is None, "the rollout sends no frame with the reset"
        return {"success": True}

    def act(self, rgb_frame, *, episode_id: str = "", frame_id: int = 0) -> PolicyResponse:
        return PolicyResponse(
            success=True,
            action_name="WAYPOINT",
            waypoint={
                "waypoints": [
                    {"forward_m": 0.05, "lateral_m": 0.0, "yaw_rad": 0.0} for _ in range(4)
                ]
            },
            waypoint_horizon=4,
            frame_id=frame_id,
        )

    def finish(self, *, episode_id: str = "") -> dict[str, object]:
        return {"success": True}

    def close(self) -> None:
        self.closed += 1


class _FlatWorld:
    """A minimal camera-walk backend: no geometry, deterministic frames."""

    def resolve_start_pose(self, pose):
        return pose

    def resolve_motion(self, prev, candidate):
        return candidate

    def reset(self, pose, *, warmup_steps, converge_max_steps, converge_tol, log):
        return self._frame()

    def set_pose(self, pose):
        return None

    def step_and_capture(self, *, settle_steps):
        return self._frame()

    def _frame(self):
        # Textured, not flat: this suite requires real scene assets, and the
        # rollout refuses a frame that looks like an empty room.
        grid = np.arange(8 * 8 * 3, dtype=np.uint16)
        return ((grid * 37) % 251).astype(np.uint8).reshape(8, 8, 3)


class StandInBackend:
    """The backend surface the runner needs, with the frames-era keyword optional.

    ``**kwargs`` rather than a declared ``frame_sink``: a backend written
    before ``--vis`` does not accept the keyword, and what this file has to be
    able to prove is that a run without ``--vis`` does not pass it.
    """

    def __init__(self) -> None:
        self.opened: list[str] = []
        self.closed = 0
        self.calls: list[dict[str, object]] = []
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
        self.opened.append(task_config.name)
        self._task_config = task_config

    def close(self, *, log=None) -> None:
        self.closed += 1

    def run_episode_with_policy(self, episode, policy, *, options=None, log=None, **kwargs):
        assert self._task_config is not None, "run before open_scene"
        self.calls.append(dict(kwargs))
        return run_camera_walk_episode(
            env=_FlatWorld(),
            policy=policy,
            episode=episode,
            task_config=self._task_config,
            options=options or RolloutOptions.from_task_config(self._task_config),
            frame_sink=kwargs.get("frame_sink"),
        )


@pytest.fixture
def manifest():
    """The shipped v2 manifest, re-pinned to the seven-episode fixture.

    Only the episode count moves: the runner refuses a file shorter than the
    coordinate is defined over, which is exactly the guarantee that makes a
    seven-episode fixture unusable as the published 1097.
    """
    published = Registry.load().get(COORDINATE)
    return published.model_copy(update={"metadata": {**published.metadata, "episode_count": 7}})


@pytest.fixture
def scene_root(tmp_path: Path) -> Path:
    root = tmp_path / "scenes"
    for asset in SCENE_ASSETS:
        path = root / asset
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#usda 1.0\n", encoding="utf-8")
    return root


@pytest.fixture
def context(scene_root: Path) -> RunContext:
    return RunContext(scene_root=scene_root)


@pytest.fixture
def stand_in(monkeypatch):
    backend = StandInBackend()
    monkeypatch.setattr(runner_module, "load_backend", lambda backend_id: backend)
    monkeypatch.setattr(policy_client, "LocalVlnPolicyClient", WalkingPolicy)
    return backend


def _run(manifest, context, **context_kwargs):
    return runner_module.insight_bench_runner(
        manifest, FIXTURE, 0, context=replace(context, **context_kwargs)
    )


def _episode_dirs(run_dir: Path) -> list[Path]:
    return sorted(path for path in run_dir.iterdir() if path.is_dir())


def _steps(episode_dir: Path) -> list[dict]:
    payload = json.loads((episode_dir / "trace.json").read_text(encoding="utf-8"))
    return list(payload["steps"])


def _frame_paths(episode_dir: Path) -> list[str]:
    """Every step's frame path, absent counting as none.

    Step 0's observation is ``{}`` -- it is built from a pose alone, so the key
    is not merely empty there but missing, and the read side has to use a
    default rather than an index.
    """
    return [str(step["observation"].get("frame_path", "")) for step in _steps(episode_dir)]


def _num_steps() -> int:
    """The suite's step budget: what the frame stride is computed against."""
    from insight_bench.vln_runtime.suite import get_task_config

    return RolloutOptions.from_task_config(get_task_config("insight_bench")).num_steps


# --- the default is off, and off costs nothing -------------------------------


def test_a_run_without_the_flag_writes_no_replay_material(manifest, context, stand_in, tmp_path):
    run_dir = tmp_path / "plain"
    result = _run(manifest, context, artifact_dir=run_dir)

    # The keyword is not passed, so a backend that predates --vis still runs.
    assert stand_in.calls
    assert all(call == {} for call in stand_in.calls)
    # No manifest and no frames. Each episode holds its trace and its scored
    # result -- the result is what --resume reads, and it is written whether or
    # not replay material was asked for.
    assert not (run_dir / "vis-manifest.json").exists()
    for episode_dir in _episode_dirs(run_dir):
        assert sorted(path.name for path in episode_dir.iterdir()) == [
            "result.json",
            "trace.json",
        ]
        # The trace field keeps the value it has had since it was reserved.
        assert set(_frame_paths(episode_dir)) == {""}
    assert result.status == "completed"


def test_no_frame_path_key_reaches_the_trace_builder_without_the_flag(
    manifest, context, stand_in, tmp_path, monkeypatch
):
    """The executable form of "byte-identical": the key is absent, not empty.

    ``RolloutTrace.from_camera_walk_records`` defaults ``frame_path`` to ``""``
    when a record does not carry it, so a record with no such key produces the
    trace bytes it produced before frames could be persisted.
    """
    from insight_bench.vln_runtime.rollout import camera_walk as rollout_module

    seen: list[list[dict]] = []
    original = rollout_module.RolloutTrace.from_camera_walk_records

    def spy(**kwargs):
        seen.append(list(kwargs["records"]))
        return original(**kwargs)

    monkeypatch.setattr(rollout_module.RolloutTrace, "from_camera_walk_records", spy)
    _run(manifest, context, artifact_dir=tmp_path / "plain")

    assert seen and any(records for records in seen)
    assert not [record for records in seen for record in records if "frame_path" in record]


def test_the_flag_does_not_change_the_identity_of_the_run(manifest, context, stand_in, tmp_path):
    # run_id and run_fingerprint derive from the execution key and the results,
    # and a visualisation flag is neither. A --vis run must be submittable as
    # the same run as the one without it.
    plain = _run(manifest, context, artifact_dir=tmp_path / "plain")
    visualised = _run(manifest, context, artifact_dir=tmp_path / "vis", vis=VisOptions())
    assert plain.run_id == visualised.run_id
    assert plain.run_fingerprint == visualised.run_fingerprint


# --- what --vis writes -------------------------------------------------------


def test_each_frame_is_named_for_the_trace_step_that_records_its_decision(
    manifest, context, stand_in, tmp_path
):
    run_dir = tmp_path / "vis"
    _run(manifest, context, artifact_dir=run_dir, vis=VisOptions())

    assert all(call["frame_sink"] is not None for call in stand_in.calls)
    for episode_dir in _episode_dirs(run_dir):
        steps = _steps(episode_dir)
        assert [step["step"] for step in steps] == list(range(len(steps)))
        paths = _frame_paths(episode_dir)
        # Step 0 is the pre-action pose: no decision, no frame, no broken path.
        assert paths[0] == ""
        assert not (episode_dir / "frames" / "step_0000.jpg").exists()
        referenced = []
        for index, path in enumerate(paths):
            if not path:
                continue
            assert path == f"frames/step_{index:04d}.jpg"
            assert (episode_dir / path).read_bytes()[:3] == JPEG_MAGIC
            referenced.append(path)
        assert referenced, "every episode of this run was shown frames"
        # No orphans: what is on disk is exactly what the trace points at.
        on_disk = sorted(f"frames/{path.name}" for path in (episode_dir / "frames").iterdir())
        assert on_disk == sorted(referenced)


def test_the_stride_samples_the_episode_instead_of_truncating_it(
    manifest, context, stand_in, tmp_path
):
    # num_steps is 300 for this suite, so a cap of 5 frames means every 60th
    # decision -- sampled across the whole episode, not the first five steps.
    stride = _num_steps() // 5
    run_dir = tmp_path / "vis"
    _run(
        manifest,
        context,
        artifact_dir=run_dir,
        vis=VisOptions(max_frames_per_episode=5),
    )
    manifest_payload = json.loads((run_dir / "vis-manifest.json").read_text(encoding="utf-8"))
    assert manifest_payload["frame_stride"] == stride
    assert manifest_payload["max_frames_per_episode"] == 5

    for episode_dir in _episode_dirs(run_dir):
        indexes = [index for index, path in enumerate(_frame_paths(episode_dir)) if path]
        assert indexes, "the first decision of every episode is always sampled"
        assert indexes[0] == 1
        assert all((index - 1) % stride == 0 for index in indexes)
        assert len(indexes) <= 5


def test_the_run_manifest_says_what_the_frames_beside_it_mean(
    manifest, context, stand_in, tmp_path
):
    run_dir = tmp_path / "vis"
    _run(manifest, context, artifact_dir=run_dir, vis=VisOptions())
    payload = json.loads((run_dir / "vis-manifest.json").read_text(encoding="utf-8"))

    assert payload["schema"] == "insight-bench/vis-manifest/1"
    assert payload["episode_count"] == len(_episode_dirs(run_dir))
    assert payload["max_frames_per_episode"] == VisOptions().max_frames_per_episode
    assert payload["frames_failed"] == 0
    assert payload["frame_failure_reason"] == ""
    on_disk = sum(len(list((path / "frames").iterdir())) for path in _episode_dirs(run_dir))
    assert payload["frames_written"] == on_disk
    # A reader must not have to guess which pose took the picture.
    assert "PREVIOUS step's position" in payload["frame_path"]


def test_visualisation_failing_never_fails_the_episode(
    manifest, context, stand_in, tmp_path, monkeypatch
):
    plain = _run(manifest, context, artifact_dir=tmp_path / "plain")

    def refuse(frame, **kwargs):
        raise ValueError("expected RGB frame with shape (H, W, >=3), got (8, 8, 2)")

    monkeypatch.setattr(policy_client, "encode_rgb_jpeg", refuse)
    run_dir = tmp_path / "vis"
    visualised = _run(manifest, context, artifact_dir=run_dir, vis=VisOptions())

    # The measurement is untouched: same statuses, same metrics, same identity.
    assert visualised.status == plain.status
    assert [episode.status for episode in visualised.episodes] == [
        episode.status for episode in plain.episodes
    ]
    assert visualised.run_fingerprint == plain.run_fingerprint

    payload = json.loads((run_dir / "vis-manifest.json").read_text(encoding="utf-8"))
    assert payload["frames_written"] == 0
    assert payload["frames_failed"] > 0
    assert "expected RGB frame" in payload["frame_failure_reason"]
    for episode_dir in _episode_dirs(run_dir):
        assert not (episode_dir / "frames").exists()
        assert set(_frame_paths(episode_dir)) == {""}


# --- episode.json ------------------------------------------------------------


def test_the_episode_sidecar_carries_the_page_fields_and_not_the_host(tmp_path):
    # A run directory is meant to be tarred and handed to someone else, and
    # scene.asset_path is a path into the machine that produced it.
    episode = EpisodeSpec.from_dict(
        {
            "episode_id": "ep-1",
            "instruction": "find the chair",
            "scene": {
                "scene_id": "scene56",
                "asset_path": "/Users/nobody/scenes/habitat_gs/scene56/scene56.usd",
                "dataset": "habitat_gs",
                "metadata": {"internal_note": "/Users/nobody/notes.txt"},
            },
            "start_pose": {"position": [1.0, 2.0, 0.0]},
            "goal_position": [4.0, 5.0, 0.0],
            "goal_radius_m": 3.0,
            "reference_path": [[1.0, 2.0, 0.0], [4.0, 5.0, 0.0]],
            "metadata": {
                "suite": "insight-bench-v1",
                "difficulty_label": "hard",
                "instr_type": "attribute",
                "scene_class": "House",
            },
        }
    )
    session = VisSession(
        artifact_dir=tmp_path,
        sink=JpegFrameSink(root=tmp_path),
        episode_count=1,
        max_frames_per_episode=100,
    )
    session.write_episode(episode)

    text = (tmp_path / "ep-1" / "episode.json").read_text(encoding="utf-8")
    assert "/Users/nobody" not in text
    assert "asset_path" not in text
    assert "internal_note" not in text
    payload = json.loads(text)
    assert payload == {
        "episode_id": "ep-1",
        "instruction": "find the chair",
        "scene_id": "scene56",
        "goal_position": [4.0, 5.0, 0.0],
        "goal_radius_m": 3.0,
        "reference_path": [[1.0, 2.0, 0.0], [4.0, 5.0, 0.0]],
        "suite": "insight-bench-v1",
        "difficulty_label": "hard",
        "instr_type": "attribute",
        "scene_class": "House",
    }


def test_metadata_the_dataset_does_not_carry_is_omitted_not_nulled(tmp_path):
    # Missing data is not an empty value: the reader must be able to tell
    # "this dataset has no instruction types" from "this one is blank".
    episode = EpisodeSpec.from_dict(
        {
            "episode_id": "ep-2",
            "instruction": "go",
            "scene": {"scene_id": "s"},
            "start_pose": {"position": [0.0, 0.0, 0.0]},
        }
    )
    session = VisSession(
        artifact_dir=tmp_path,
        sink=JpegFrameSink(root=tmp_path),
        episode_count=1,
        max_frames_per_episode=100,
    )
    session.write_episode(episode)
    payload = json.loads((tmp_path / "ep-2" / "episode.json").read_text(encoding="utf-8"))
    assert set(payload) == {
        "episode_id",
        "instruction",
        "scene_id",
        "goal_position",
        "goal_radius_m",
        "reference_path",
    }
    assert payload["goal_position"] is None


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("goal_radius_m", float("inf")),
        ("goal_position", [float("inf"), 0.0, 0.0]),
        ("reference_path", [[0.0, 0.0, 0.0], [float("nan"), 1.0, 0.0]]),
    ],
)
def test_a_sidecar_the_dataset_makes_unwritable_is_counted_not_raised(tmp_path, field_name, value):
    """`EpisodeSpec` accepts non-finite floats; stable JSON refuses them.

    Both are correct: the dataset parser is not a JSON writer, and
    `stable_json_bytes` uses `allow_nan=False` deliberately. What must not
    happen is the third thing -- one such episode in an untrusted dataset
    aborting the whole run with a raw ValueError before anything is measured.
    """
    payload = {
        "episode_id": "ep-nonfinite",
        "instruction": "go",
        "scene": {"scene_id": "s"},
        "start_pose": {"position": [0.0, 0.0, 0.0]},
        field_name: value,
    }
    episode = EpisodeSpec.from_dict(payload)
    session = VisSession(
        artifact_dir=tmp_path,
        sink=JpegFrameSink(root=tmp_path),
        episode_count=1,
        max_frames_per_episode=100,
    )
    session.write_episode(episode)

    assert session.sidecars_failed == 1
    assert "ValueError" in session.first_sidecar_failure
    assert not (tmp_path / "ep-nonfinite" / "episode.json").exists()
    # And it is reported, once, where the reader of the run will see it.
    session.write_manifest()
    manifest_payload = json.loads((tmp_path / "vis-manifest.json").read_text(encoding="utf-8"))
    assert manifest_payload["sidecars_failed"] == 1
    assert "ValueError" in manifest_payload["sidecar_failure_reason"]


def test_a_sidecar_write_that_cannot_succeed_never_fails_the_run(
    manifest, context, stand_in, tmp_path, monkeypatch
):
    """The end-to-end half of the same guarantee, for the sidecar and manifest.

    ``write_manifest`` runs in the runner's ``finally``, so an exception there
    would replace a completed run's result with a stack trace about a
    description of some pictures.
    """
    plain = _run(manifest, context, artifact_dir=tmp_path / "plain")

    def refuse(path, payload):
        raise OSError("Read-only file system")

    monkeypatch.setattr(_shared, "write_stable_json", refuse)
    run_dir = tmp_path / "vis"
    visualised = _run(manifest, context, artifact_dir=run_dir, vis=VisOptions())

    assert visualised.status == plain.status
    assert visualised.run_fingerprint == plain.run_fingerprint
    assert not (run_dir / "vis-manifest.json").exists()
    # The traces -- the evidence -- are untouched by any of it.
    for episode_dir in _episode_dirs(run_dir):
        assert (episode_dir / "trace.json").is_file()
        assert not (episode_dir / "episode.json").exists()


def test_an_episode_that_errors_is_still_described(
    manifest, context, stand_in, tmp_path, monkeypatch
):
    # episode.json is written before the rollout, not beside the trace after
    # it: an errored episode never reaches the trace write, and it is exactly
    # the one a reader wants named rather than filed under "unknown".
    def explode(episode, policy, *, options=None, log=None, **kwargs):
        raise RuntimeError("renderer died")

    monkeypatch.setattr(stand_in, "run_episode_with_policy", explode)
    run_dir = tmp_path / "vis"
    result = _run(manifest, context, artifact_dir=run_dir, vis=VisOptions())

    assert result.status == "failed"
    assert all(episode.status == "error" for episode in result.episodes)
    for episode in result.episodes:
        sidecar = run_dir / episode.episode_id / "episode.json"
        assert sidecar.is_file()
        assert not (run_dir / episode.episode_id / "trace.json").exists()


# --- refusing before anything runs -------------------------------------------


def test_vis_without_an_artifact_dir_is_refused(manifest, context, stand_in):
    with pytest.raises(RunnerError, match="--vis needs --artifact-dir"):
        _run(manifest, context, vis=VisOptions())
    assert stand_in.opened == []


def test_a_disk_too_small_is_refused_before_the_first_episode(
    manifest, context, stand_in, tmp_path, monkeypatch
):
    # The footprint is exactly computable before the run: episode count times
    # the per-episode frame cap. So --vis refuses up front rather than filling
    # a disk halfway through and leaving the tail of the run unusable.
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=4096))
    run_dir = tmp_path / "vis"
    with pytest.raises(RunnerError, match="MiB free"):
        _run(manifest, context, artifact_dir=run_dir, vis=VisOptions())
    assert stand_in.opened == []
    assert list(run_dir.iterdir()) == []


def test_the_precheck_counts_the_frames_it_will_actually_write(tmp_path, monkeypatch):
    # num_steps 300 with a cap of 100 costs 100 frames per episode, not 300:
    # the stride is what makes the footprint independent of episode length.
    from insight_bench.vln_runtime.suite import get_task_config

    seen: list[int] = []

    def tiny(path):
        seen.append(0)
        return SimpleNamespace(free=1)

    monkeypatch.setattr(shutil, "disk_usage", tiny)
    with pytest.raises(RunnerError) as excinfo:
        open_vis_session(
            RunContext(artifact_dir=tmp_path, vis=VisOptions()),
            task_config=get_task_config("insight_bench"),
            episode_ids=[f"ep-{index}" for index in range(10)],
        )
    assert "1000 frames and their traces across 10 episodes" in str(excinfo.value)
    assert seen, "the check reads the real free space, not a guess"


def test_the_reserve_is_a_real_upper_bound_on_a_real_frame():
    """Measured, not assumed: the constant must exceed what the encoder produces.

    `VIS_FRAME_BYTES` is the whole substance of the disk refusal, and it was
    first written from an estimate that a real 480x270 render exceeds by up to
    2x. This encodes the two facts that keep the promise honest: the sensor
    size the reserve is quoted for, and a frame at that size with content
    coarser than any indoor render, encoded by the code's own encoder.
    """
    from insight_bench.vln_runtime.policy.client import encode_rgb_jpeg
    from insight_bench.vln_runtime.suite import get_task_config

    sensor = get_task_config("insight_bench").sensors[0]
    assert (sensor.width, sensor.height) == (480, 270)

    rows, columns = np.mgrid[0 : sensor.height, 0 : sensor.width]
    gradient = np.stack(
        [columns * 255 // sensor.width, rows * 255 // sensor.height, (columns + rows) % 256],
        axis=-1,
    ).astype(np.int64)
    noise = np.random.default_rng(7).integers(-25, 26, gradient.shape)
    textured = np.clip(gradient + noise, 0, 255).astype(np.uint8)

    assert len(encode_rgb_jpeg(textured)) < _shared.VIS_FRAME_BYTES


def test_the_reserve_pays_for_the_traces_beside_the_frames():
    # The headroom used to be justified by "the traces sit beside them" while
    # covering a fraction of them: 1097 objnav episodes of trace come to
    # ~820 MB against a ~450 MB margin. They are reserved explicitly now.
    frames_only = 10 * 100 * _shared.VIS_FRAME_BYTES * _shared.VIS_DISK_HEADROOM
    required = _shared.vis_required_bytes(
        episode_count=10, num_steps=300, max_frames_per_episode=100
    )
    assert required > frames_only
    assert required - frames_only == pytest.approx(
        10 * 300 * _shared.VIS_TRACE_BYTES_PER_STEP * _shared.VIS_DISK_HEADROOM, rel=1e-6
    )


def test_an_episode_named_after_a_run_level_file_is_refused_before_episode_1(tmp_path):
    """A legal episode id can name the manifest. That collision is not survivable.

    `validate_episode_id` accepts "vis-manifest.json", and episode ids name
    directories at the same root the manifest is a file in. Refused here, at
    zero cost, rather than discovered when the episode's *trace* fails to
    write -- a trace is evidence, and guarding that write would hide a real
    failure to protect a picture.
    """
    from insight_bench.vln_runtime.episodes import EpisodeSpec
    from insight_bench.vln_runtime.suite import get_task_config

    for name in _shared.VIS_RUN_LEVEL_FILENAMES:
        assert (
            EpisodeSpec.from_dict(
                {
                    "episode_id": name,
                    "instruction": "go",
                    "scene": {"scene_id": "s"},
                    "start_pose": {"position": [0.0, 0.0, 0.0]},
                }
            ).episode_id
            == name
        ), "the dataset parser accepts this id, so --vis has to be the one that refuses"
        with pytest.raises(RunnerError, match="run root"):
            open_vis_session(
                RunContext(artifact_dir=tmp_path, vis=VisOptions()),
                task_config=get_task_config("insight_bench"),
                episode_ids=["ep-1", name],
            )
    assert list(tmp_path.iterdir()) == []


def test_the_manifest_records_the_camera_the_frames_came_from(
    manifest, context, stand_in, tmp_path
):
    """The page needs one field of view for the run; the task config has it.

    Read from the sensor the frames come out of rather than from the trace: a
    backend fills `observation.intrinsics` only when it has them, and the
    top-down view otherwise degrades to a heading arrow that says nothing about
    what the frame beside it could see.
    """
    from insight_bench.vln_runtime.suite import get_task_config

    run_dir = tmp_path / "vis"
    _run(manifest, context, artifact_dir=run_dir, vis=VisOptions())
    payload = json.loads((run_dir / "vis-manifest.json").read_text(encoding="utf-8"))

    declared = get_task_config("insight_bench").sensors[0].params["hfov_deg"]
    assert payload["camera"] == {"camera_hfov_deg": declared}


def test_a_task_with_no_camera_records_an_empty_block_not_a_missing_key(tmp_path):
    """The manifest shape must not depend on which suite ran.

    An absent key and an empty block read the same to the page, but only one of
    them lets a reader tell "this suite declares no camera" apart from "this
    version of the manifest predates the field".
    """
    import dataclasses

    from insight_bench.vln_runtime.suite import get_task_config

    task_config = get_task_config("insight_bench")
    assert _shared.vis_camera_hfov_deg(task_config) is not None
    without = dataclasses.replace(task_config, sensors=())
    assert _shared.vis_camera_hfov_deg(without) is None

    session = open_vis_session(
        RunContext(artifact_dir=tmp_path, vis=VisOptions()),
        task_config=without,
        episode_ids=["ep-1"],
    )
    assert session is not None
    payload = json.loads((tmp_path / "vis-manifest.json").read_text(encoding="utf-8"))
    assert payload["camera"] == {}
