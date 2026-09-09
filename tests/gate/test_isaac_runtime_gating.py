"""Honesty gates for the Isaac execution backend.

Everything here holds on a machine with no simulator, which is the machine CI
runs on. The point is that "no Isaac" must produce a precise refusal and never
a fabricated result, and that merely importing the SDK must not reach for
Isaac at all.
"""

from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from insight_bench.simulator.attestation import evaluate_official_profile
from insight_bench.simulator.base import SimulatorNotAvailableError, get_backend_descriptor
from insight_bench.simulator.isaac import (
    IMPLEMENTED_BACKENDS,
    RUNTIME_DEPENDENCIES,
    isaac_backend_probe,
    load_camera_walk_backend,
    load_isaac_backend,
    missing_runtime_dependencies,
)
from insight_bench.simulator.isaac.backend import (
    ISAAC_IMPORT_SURFACE,
    IsaacCameraWalkBackend,
    _require_scene_assets,
    isaac_import_probe,
)
from insight_bench.simulator.isaac.camera_walk import (
    CAMERA_POSE_READ_SURFACES,
    CameraWalkEnv,
    CameraWalkSettings,
    _capture_depth,
    _capture_rgb,
    _create_camera,
    frame_delta,
    native_orientation,
    read_world_pose_rows,
    render_until_converged,
)
from insight_bench.simulator.isaac.terrain_query import _triangulate
from insight_bench.simulator.observation import (
    G2_YAW_TOLERANCE_RAD,
    quat_wxyz_from_yaw,
    yaw_from_quat_wxyz,
)
from insight_bench.vln_runtime.suite import get_task_config

ISAAC_AVAILABLE, MISSING_ISAAC_MODULES = isaac_import_probe()
needs_no_isaac = pytest.mark.skipif(
    ISAAC_AVAILABLE, reason="asserts the refusal path; this machine has Isaac installed"
)


# --- lazy import discipline -------------------------------------------------


def test_importing_the_sdk_pulls_no_simulator_modules() -> None:
    # Assembled at runtime so this file stays clean for the token scanner.
    forbidden = ("isaac" + "sim", "isaac" + "lab", "warp", "omni", "carb", "pxr", "torch")
    script = (
        "import sys;"
        "import insight_bench, insight_bench.runner, insight_bench.simulator.isaac;"
        f"names={forbidden!r};"
        "leaked=[m for m in sys.modules "
        "if any(m == n or m.startswith(n + '.') for n in names)];"
        "print(leaked)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]", result.stdout


def test_the_import_surface_is_a_fixed_package_owned_list() -> None:
    # Not manifest data and not a caller-supplied module path: the Phase 1
    # trust boundary allows importing names this distribution itself fixes.
    assert ISAAC_IMPORT_SURFACE == ("isaacsim", "isaaclab", "warp")


# --- loader honesty ---------------------------------------------------------


def test_only_the_official_51_line_has_an_implementation() -> None:
    assert IMPLEMENTED_BACKENDS == ("isaac-5.1",)


def test_the_experimental_60_line_refuses_instead_of_running_51_code() -> None:
    with pytest.raises(SimulatorNotAvailableError) as excinfo:
        load_isaac_backend("isaac-6.0")
    message = str(excinfo.value)
    assert "isaac-6.0" in message
    assert "experimental" in message
    assert "nothing here has been verified against it" in message


def test_an_unknown_backend_id_refuses_with_the_known_set() -> None:
    with pytest.raises(SimulatorNotAvailableError, match="unknown simulator backend"):
        load_isaac_backend("isaac-9.9")
    probe = isaac_backend_probe("isaac-9.9")
    assert probe["available"] is False


@needs_no_isaac
def test_loading_without_isaac_names_the_missing_modules() -> None:
    with pytest.raises(SimulatorNotAvailableError) as excinfo:
        load_camera_walk_backend("isaac-5.1")
    message = str(excinfo.value)
    assert "isaac-5.1 cannot execute here" in message
    for name in MISSING_ISAAC_MODULES:
        assert name in message


def test_the_probe_reports_facts_rather_than_raising() -> None:
    probe = isaac_backend_probe("isaac-5.1")
    assert probe["backend_id"] == "isaac-5.1"
    assert probe["available"] is (not probe["reasons"])
    # No runtime image digest is published for any backend in this release.
    assert probe["image_digest"] is None
    assert probe["scene_open"] is False


def test_the_60_probe_explains_itself_without_touching_the_backend() -> None:
    probe = isaac_backend_probe("isaac-6.0")
    assert probe["available"] is False
    assert any("experimental" in reason for reason in probe["reasons"])


# --- fail-closed on assets and on scene state -------------------------------


def test_a_usd_task_with_no_configured_scene_path_fails_closed() -> None:
    # The shipped suite is exactly this case: a USD terrain whose usd_path is
    # filled in per episode from the user-provided scene root, so the bare
    # config carries none and a run that never resolved one must refuse.
    task = get_task_config("insight_bench")
    assert task.terrain.kind == "usd" and task.terrain.usd_path is None
    with pytest.raises(SimulatorNotAvailableError, match="no usd_path is configured"):
        _require_scene_assets(task)


def test_a_usd_task_pointing_at_a_missing_file_fails_closed(tmp_path) -> None:
    task = get_task_config("insight_bench")
    missing = replace(task.terrain, usd_path=str(tmp_path / "absent.usd"))
    with pytest.raises(SimulatorNotAvailableError, match="scene asset does not exist"):
        _require_scene_assets(replace(task, terrain=missing))


def test_a_present_scene_file_passes_the_asset_gate(tmp_path) -> None:
    scene = tmp_path / "scene.usd"
    scene.write_text("", encoding="utf-8")
    task = get_task_config("insight_bench")
    _require_scene_assets(replace(task, terrain=replace(task.terrain, usd_path=str(scene))))


def test_a_non_usd_terrain_needs_no_scene_assets() -> None:
    """The gate's early return, which is what keeps it from over-refusing.

    Only a USD terrain names a file on disk. A procedurally generated one --
    the plane a `TerrainConfig` describes by default -- has nothing to check,
    and a gate that refused it would refuse a task it has no business refusing.
    No shipped suite is procedural any more, so the task is built here.
    """
    task = get_task_config("insight_bench")
    plane = replace(task.terrain, kind="plane", usd_path=None)
    _require_scene_assets(replace(task, terrain=plane))


def test_every_scene_call_refuses_before_a_scene_is_open() -> None:
    backend = IsaacCameraWalkBackend()
    from insight_bench.vln_runtime.motion.pose import CameraPose

    pose = CameraPose(0.0, 0.0, 0.0, 0.0)
    for call in (
        lambda: backend.capture_observation(),
        # The exact call an external probe script reached for in a fresh process
        # after only loading the backend: it must refuse, not read stale state.
        lambda: backend.read_camera_pose(),
        lambda: backend.resolve_start_pose(pose),
        lambda: backend.resolve_motion(pose, pose),
        lambda: backend.set_pose(pose),
        lambda: backend.step_and_capture(),
        lambda: backend.step_motion(pose, pose),
        lambda: backend.last_terrain_query(),
    ):
        with pytest.raises(SimulatorNotAvailableError, match="no scene is open"):
            call()


# --- runtime attestation ----------------------------------------------------


def test_a_docker_run_publishes_without_an_image_digest(monkeypatch) -> None:
    backend = IsaacCameraWalkBackend()
    monkeypatch.setattr("insight_bench.simulator.isaac.backend.sys.platform", "linux")
    monkeypatch.setattr("insight_bench.simulator.isaac.backend._in_docker", lambda: True)
    monkeypatch.setattr(
        "insight_bench.simulator.isaac.backend.observed_driver_version", lambda: "580.65.06"
    )
    attestation = backend.runtime_attestation()
    assert attestation.backend_id == "isaac-5.1"
    assert attestation.backend_status == "official"
    assert attestation.runtime_kind == "docker"
    assert attestation.os == "linux"
    # No digest exists to lock to (the image embeds Isaac Sim): recorded
    # honestly, and no longer what decides publication (ADR 0008).
    assert attestation.image_digest is None
    assert attestation.digest_locked is False
    assert attestation.driver_version == "580.65.06"
    assert attestation.publishable is True


def test_a_native_linux_run_is_attested_as_native_and_publishes(monkeypatch) -> None:
    backend = IsaacCameraWalkBackend()
    monkeypatch.setattr("insight_bench.simulator.isaac.backend.sys.platform", "linux")
    monkeypatch.setattr("insight_bench.simulator.isaac.backend._in_docker", lambda: False)
    monkeypatch.setattr(
        "insight_bench.simulator.isaac.backend.observed_driver_version", lambda: "535.216.03"
    )
    attestation = backend.runtime_attestation()
    assert (attestation.runtime_kind, attestation.launcher_id) == ("native", "linux-native")
    assert attestation.driver_version == "535.216.03"
    assert attestation.publishable is True


def test_a_run_whose_driver_could_not_be_read_does_not_publish(monkeypatch) -> None:
    backend = IsaacCameraWalkBackend()
    monkeypatch.setattr("insight_bench.simulator.isaac.backend.sys.platform", "linux")
    monkeypatch.setattr("insight_bench.simulator.isaac.backend._in_docker", lambda: False)
    monkeypatch.setattr("insight_bench.simulator.isaac.backend.observed_driver_version", lambda: "")
    attestation = backend.runtime_attestation()
    assert attestation.driver_version == ""
    assert attestation.publishable is False


def test_a_docker_run_with_no_readable_driver_does_not_publish_either(monkeypatch) -> None:
    # The container is not what decides it (ADR 0008): Docker with an unread
    # driver is withheld for exactly the reason a native run would be.
    backend = IsaacCameraWalkBackend()
    monkeypatch.setattr("insight_bench.simulator.isaac.backend.sys.platform", "linux")
    monkeypatch.setattr("insight_bench.simulator.isaac.backend._in_docker", lambda: True)
    monkeypatch.setattr("insight_bench.simulator.isaac.backend.observed_driver_version", lambda: "")
    attestation = backend.runtime_attestation()
    assert (attestation.runtime_kind, attestation.launcher_id) == ("docker", "docker")
    assert attestation.driver_version == ""
    assert attestation.publishable is False
    ok, reasons = evaluate_official_profile(attestation, get_backend_descriptor("isaac-5.1"))
    assert not ok and reasons == ("driver version was not recorded",)


def test_an_unsupported_platform_cannot_produce_an_attestation(monkeypatch) -> None:
    backend = IsaacCameraWalkBackend()
    monkeypatch.setattr("insight_bench.simulator.isaac.backend.sys.platform", "darwin")
    with pytest.raises(SimulatorNotAvailableError, match="not a supported Isaac Sim platform"):
        backend.runtime_attestation()


# --- the quaternion boundary ------------------------------------------------


def test_the_pose_command_converts_canonical_wxyz_into_the_native_order() -> None:
    yaw = 0.7
    wxyz = native_orientation(yaw, "wxyz")
    xyzw = native_orientation(yaw, "xyzw")
    assert wxyz == (xyzw[3], xyzw[0], xyzw[1], xyzw[2])
    # Round-tripping the canonical order recovers the yaw far inside the G2 gate.
    recovered = yaw_from_quat_wxyz(wxyz)
    assert abs(recovered - yaw) < 1e-12 < G2_YAW_TOLERANCE_RAD
    # Reading the Lab 3.x order as if it were WXYZ is the failure G2 catches.
    assert abs(yaw_from_quat_wxyz(xyzw) - yaw) > G2_YAW_TOLERANCE_RAD


def test_the_51_line_defaults_to_its_native_wxyz_order() -> None:
    settings = CameraWalkSettings.from_task_config(get_task_config("insight_bench"))
    assert settings.native_quat_order == "wxyz"


# --- capture never aliases a reused render buffer ---------------------------


class _FakeTensor:
    """Mimics the torch tensor Isaac hands back: .detach().cpu().numpy() is a view."""

    def __init__(self, array: np.ndarray):
        self._array = array

    def __getitem__(self, index: int) -> _FakeTensor:
        return _FakeTensor(self._array[index])

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._array


class _FakeCameraData:
    def __init__(self, output: dict[str, object]):
        self.output = output
        self.intrinsic_matrices = _FakeTensor(np.eye(3, dtype=np.float32)[None, ...])


class _FakeCamera:
    def __init__(self, buffer: np.ndarray, depth: np.ndarray | None = None):
        payload: dict[str, object] = {"rgb": _FakeTensor(buffer)}
        if depth is not None:
            payload["distance_to_image_plane"] = _FakeTensor(depth)
        self.data = _FakeCameraData(payload)


def test_a_captured_rgb_frame_does_not_alias_the_render_buffer() -> None:
    # This is the concrete bug the copy prevents: Isaac reuses the buffer on the
    # next step, and a CPU-device camera hands back a view onto it.
    buffer = np.zeros((1, 4, 6, 3), dtype=np.uint8)
    frame = _capture_rgb(_FakeCamera(buffer))
    buffer[:] = 255
    assert not frame.any(), "the captured frame followed a later write to the render buffer"
    assert frame.dtype == np.uint8 and frame.shape == (4, 6, 3)


def test_an_rgba_capture_is_trimmed_to_three_contiguous_channels() -> None:
    buffer = np.zeros((1, 4, 6, 4), dtype=np.uint8)
    buffer[..., 3] = 255
    frame = _capture_rgb(_FakeCamera(buffer))
    assert frame.shape == (4, 6, 3)
    assert frame.flags["C_CONTIGUOUS"]
    buffer[:] = 7
    assert not frame.any()


def test_a_float_capture_without_a_declared_range_fails_loudly() -> None:
    # Isaac Lab emits uint8 by default. A float frame means the camera was
    # configured differently, and only the configuration knows in which range.
    buffer = np.ones((1, 2, 2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="no rgb_float_range is configured"):
        _capture_rgb(_FakeCamera(buffer))


def test_a_unit_ranged_float_capture_is_scaled_to_bytes_and_copied() -> None:
    buffer = np.ones((1, 2, 2, 3), dtype=np.float32)
    frame = _capture_rgb(_FakeCamera(buffer), float_range="unit")
    assert frame.dtype == np.uint8
    assert frame.max() == 255
    buffer[:] = 0.0
    assert frame.max() == 255  # still a private copy


def test_a_dark_byte_ranged_float_frame_is_not_brightened() -> None:
    # The regression that range-guessing caused: every pixel of this frame is
    # below 1.0, but it is already in 0-255 units. Inferring "max <= 1.0, so
    # scale by 255" turned a nearly black frame into a mid-grey one, and
    # nothing downstream could tell the difference.
    dark = np.full((1, 2, 2, 3), 0.8, dtype=np.float32)
    frame = _capture_rgb(_FakeCamera(dark), float_range="byte")
    assert frame.dtype == np.uint8
    assert frame.max() == 0  # 0.8 truncates to 0 in byte units, as it should
    assert _capture_rgb(_FakeCamera(dark), float_range="unit").max() == 204


def test_float_captures_are_clipped_into_the_byte_range() -> None:
    hot = np.full((1, 2, 2, 3), 300.0, dtype=np.float32)
    assert _capture_rgb(_FakeCamera(hot), float_range="byte").max() == 255
    negative = np.full((1, 2, 2, 3), -5.0, dtype=np.float32)
    assert _capture_rgb(_FakeCamera(negative), float_range="byte").min() == 0


def test_the_float_range_comes_from_the_rgb_sensor_config() -> None:
    from dataclasses import replace

    from insight_bench.vln_runtime.suite.config import SensorConfig

    task = get_task_config("insight_bench")
    # Default: uint8 expected, nothing declared.
    assert CameraWalkSettings.from_task_config(task).rgb_float_range is None

    rgb = next(sensor for sensor in task.sensors if sensor.name == "rgb")
    declared = replace(rgb, params={**rgb.params, "rgb_float_range": "byte"})
    assert (
        CameraWalkSettings.from_task_config(replace(task, sensors=(declared,))).rgb_float_range
        == "byte"
    )

    bogus = SensorConfig(name="rgb", kind="pinhole_camera", params={"rgb_float_range": "0-1"})
    with pytest.raises(ValueError, match="must be 'unit' or 'byte'"):
        CameraWalkSettings.from_task_config(replace(task, sensors=(bogus,)))


def test_a_captured_depth_frame_does_not_alias_either() -> None:
    rgb = np.zeros((1, 2, 2, 3), dtype=np.uint8)
    depth = np.ones((1, 2, 2), dtype=np.float32)
    captured = _capture_depth(_FakeCamera(rgb, depth))
    assert captured is not None
    depth[:] = 9.0
    assert float(captured.max()) == 1.0


def test_depth_is_absent_when_the_camera_did_not_record_it() -> None:
    assert _capture_depth(_FakeCamera(np.zeros((1, 2, 2, 3), dtype=np.uint8))) is None


# --- render convergence and mesh helpers (pure, GPU-free) -------------------


def test_convergence_waits_for_streaming_then_for_a_stable_image() -> None:
    frames = [np.full((2, 2, 3), value, dtype=np.uint8) for value in (10, 40, 60, 61, 61, 61, 61)]
    calls = {"n": 0}
    busy_for = 2

    def capture() -> np.ndarray:
        frame = frames[min(calls["n"], len(frames) - 1)]
        calls["n"] += 1
        return frame

    def busy() -> bool:
        return calls["n"] < busy_for

    logs: list[str] = []
    final = render_until_converged(
        capture,
        np.zeros((2, 2, 3), dtype=np.uint8),
        max_steps=10,
        tol=0.5,
        busy=busy,
        log=logs.append,
    )
    assert int(final[0, 0, 0]) == 61
    summary = next(line for line in logs if line.startswith("first-frame converge"))
    assert f"streaming_steps={busy_for}" in summary
    assert "converged=True" in summary


def test_convergence_is_skipped_when_no_budget_is_given() -> None:
    previous = np.zeros((2, 2, 3), dtype=np.uint8)

    def capture() -> np.ndarray:  # pragma: no cover - must never be called
        raise AssertionError("capture called with a zero budget")

    assert render_until_converged(capture, previous, max_steps=0, tol=0.0) is previous


def test_a_scene_that_never_settles_warns_and_returns_the_newest_frame() -> None:
    counter = {"n": 0}

    def capture() -> np.ndarray:
        counter["n"] += 1
        return np.full((2, 2, 3), counter["n"] * 10 % 250, dtype=np.uint8)

    logs: list[str] = []
    render_until_converged(
        capture, np.zeros((2, 2, 3), dtype=np.uint8), max_steps=4, tol=0.1, log=logs.append
    )
    assert any("did not converge" in line for line in logs)
    assert counter["n"] == 4


def test_frame_delta_handles_shape_changes_and_empty_frames() -> None:
    assert frame_delta(np.zeros((2, 2)), np.zeros((3, 3))) == math.inf
    assert frame_delta(np.zeros((0,)), np.zeros((0,))) == 0.0
    assert frame_delta(np.zeros((2, 2)), np.full((2, 2), 4.0)) == pytest.approx(4.0)


def test_polygon_meshes_are_fan_triangulated_and_triangles_pass_through() -> None:
    already = _triangulate(np.arange(6, dtype=np.int64), np.array([3, 3], dtype=np.int64))
    assert already.tolist() == [[0, 1, 2], [3, 4, 5]]

    quad = _triangulate(np.arange(4, dtype=np.int64), np.array([4], dtype=np.int64))
    assert quad.tolist() == [[0, 1, 2], [0, 2, 3]]

    degenerate = _triangulate(np.arange(2, dtype=np.int64), np.array([2], dtype=np.int64))
    assert degenerate.shape == (0, 3)


def test_a_non_mapping_camera_output_raises_instead_of_reporting_no_depth() -> None:
    class _OpaqueData:
        output = object()
        intrinsic_matrices = None

    class _OpaqueCamera:
        data = _OpaqueData()

    with pytest.raises(TypeError, match="not mapping-shaped"):
        _capture_depth(_OpaqueCamera())


def test_the_backend_offers_the_whole_rollout_backend_surface() -> None:
    # The rollout drives the backend, not the raw env, so the delegating
    # surface has to be complete or an episode fails at the first call.
    required = {
        "resolve_start_pose",
        "resolve_motion",
        "reset",
        "set_pose",
        "step_and_capture",
    }
    assert required <= {name for name in dir(IsaacCameraWalkBackend) if not name.startswith("_")}


def test_the_frame_counter_follows_renders_and_resets_per_episode() -> None:
    from insight_bench.vln_runtime.motion.pose import CameraPose

    class _CountingEnv:
        def reset(self, pose, *, warmup_steps, converge_max_steps, converge_tol, log):
            return np.zeros((2, 2, 3), dtype=np.uint8)

        def step_and_capture(self, *, settle_steps):
            return np.zeros((2, 2, 3), dtype=np.uint8)

    backend = IsaacCameraWalkBackend()
    # Hand-installing the scene is the only way to exercise the counter without
    # a simulator; open_scene() itself is gated on a real Isaac runtime.
    backend._env = _CountingEnv()
    backend._task_config = get_task_config("insight_bench")

    backend.step_and_capture()
    backend.step_and_capture()
    assert backend._frame_id == 2
    backend.reset(CameraPose(0.0, 0.0, 0.0, 0.0), warmup_steps=1)
    assert backend._frame_id == 0


# --- a base install must refuse, not explode --------------------------------
#
# `pip install insight_bench` brings pydantic and nothing else. The backend
# module imports numpy (and, through the policy client, requests and Pillow) at
# its own module scope, so the loader has to answer before importing it or the
# caller gets a bare ModuleNotFoundError instead of "this backend cannot
# execute here".


def test_the_runtime_dependency_list_is_the_vln_extra() -> None:
    assert {distribution for _module, distribution in RUNTIME_DEPENDENCIES} == {
        "numpy",
        "pillow",
        "requests",
    }


def test_a_full_install_reports_no_missing_runtime_dependencies() -> None:
    # The dev environment has the vln extra, so this is the "nothing missing"
    # branch; the isolated tests below cover the other one.
    assert missing_runtime_dependencies() == ()


def test_a_base_install_refuses_by_name_instead_of_raising_importerror(monkeypatch) -> None:
    """Primary guard: find_spec reports the dependencies absent."""
    import importlib.util as importlib_util

    absent = {module for module, _distribution in RUNTIME_DEPENDENCIES}
    real_find_spec = importlib_util.find_spec

    def fake_find_spec(name, package=None):
        if name.split(".")[0] in absent:
            return None
        return real_find_spec(name, package)

    monkeypatch.setattr(importlib_util, "find_spec", fake_find_spec)

    assert missing_runtime_dependencies() == ("numpy", "pillow", "requests")

    for loader in (load_camera_walk_backend, load_isaac_backend):
        with pytest.raises(SimulatorNotAvailableError) as excinfo:
            loader("isaac-5.1")
        message = str(excinfo.value)
        assert "isaac-5.1 cannot execute here" in message
        for distribution in ("numpy", "pillow", "requests"):
            assert distribution in message
        assert "insight_bench[isaac]" in message

    probe = isaac_backend_probe("isaac-5.1")
    assert probe["available"] is False
    assert probe["missing_dependencies"] == ["numpy", "pillow", "requests"]
    assert "insight_bench[isaac]" in probe["reasons"][0]


def test_no_bare_importerror_escapes_when_the_dependencies_cannot_import() -> None:
    """Fallback guard: the packages exist on disk but importing them fails.

    find_spec and a real import can disagree (a broken install, or an
    environment that blocks the import itself), so the loader must convert that
    into the same honest refusal. Run in a subprocess because it rewrites
    __import__.
    """
    script = """
import builtins
BLOCKED = ("numpy", "requests", "PIL")
_real = builtins.__import__


def guarded(name, *args, **kwargs):
    root = name.split(".")[0]
    if root in BLOCKED:
        raise ModuleNotFoundError(f"No module named {root!r}", name=root)
    return _real(name, *args, **kwargs)


builtins.__import__ = guarded

from insight_bench.simulator.base import SimulatorNotAvailableError
from insight_bench.simulator.isaac import isaac_backend_probe, load_camera_walk_backend

try:
    load_camera_walk_backend("isaac-5.1")
    raise AssertionError("loader returned a backend without its dependencies")
except SimulatorNotAvailableError as exc:
    assert "cannot execute here" in str(exc), exc
    assert "insight_bench[isaac]" in str(exc), exc

probe = isaac_backend_probe("isaac-5.1")
assert probe["available"] is False, probe
assert "insight_bench[isaac]" in probe["reasons"][0], probe
print("refused cleanly")
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "refused cleanly" in result.stdout


# --- reading a camera pose back, across Isaac Lab lines ---------------------
#
# The bug this section pins: `read_camera_pose` called `camera.get_world_poses()`,
# which `isaaclab.sensors.Camera` does not have. On a real L20 job that surfaced
# as an AttributeError mid-run and took G1 and G2 to 0. The method belongs to the
# `isaacsim.core.prims` XForm views, not to a Lab sensor, so this was never a
# deprecation -- the call was aimed at the wrong object from the start.


class _PoseData:
    """The `camera.data` of an Isaac Lab sensor, pose fields only."""

    def __init__(self, pos_w, quat_w_world, *, quat_w_ros=None) -> None:
        self.pos_w = _FakeTensor(np.asarray(pos_w, dtype=np.float32))
        self.quat_w_world = _FakeTensor(np.asarray(quat_w_world, dtype=np.float32))
        # Present on a real sensor, and the wrong frame to read a world pose
        # from. Carried here so a test can prove it is not what gets used.
        if quat_w_ros is not None:
            self.quat_w_ros = _FakeTensor(np.asarray(quat_w_ros, dtype=np.float32))


class _Lab23Camera:
    """Isaac Lab 2.3's Camera: `data.pos_w` / `data.quat_w_world`, no legacy method.

    Deliberately does NOT define `get_world_poses`, so the old code path raises
    the exact AttributeError the L20 run hit.
    """

    def __init__(self, pos_w, quat_w_world, *, quat_w_ros=None) -> None:
        self.data = _PoseData(pos_w, quat_w_world, quat_w_ros=quat_w_ros)


class _LegacyPrimViewCamera:
    """An `isaacsim.core.prims` XForm view: `get_world_poses()`, no pose in `data`."""

    def __init__(self, positions, orientations) -> None:
        self.data = _FakeCameraData({})
        self._positions = _FakeTensor(np.asarray(positions, dtype=np.float32))
        self._orientations = _FakeTensor(np.asarray(orientations, dtype=np.float32))
        self.calls = 0

    def get_world_poses(self):
        self.calls += 1
        return self._positions, self._orientations


class _BothSurfacesCamera(_Lab23Camera):
    """Offers both surfaces, with different answers, so precedence is observable."""

    def __init__(self, pos_w, quat_w_world, legacy_positions, legacy_orientations) -> None:
        super().__init__(pos_w, quat_w_world)
        self._legacy = (
            _FakeTensor(np.asarray(legacy_positions, dtype=np.float32)),
            _FakeTensor(np.asarray(legacy_orientations, dtype=np.float32)),
        )
        self.legacy_calls = 0

    def get_world_poses(self):
        self.legacy_calls += 1
        return self._legacy


class _NoPoseSurfaceCamera:
    """A camera object this SDK does not know how to read a pose from."""

    def __init__(self) -> None:
        self.data = _FakeCameraData({})


def test_the_lab_23_camera_is_read_through_data_not_get_world_poses() -> None:
    """REGRESSION (the L20 failure): Lab 2.3's Camera has no `get_world_poses`.

    The old implementation called it unconditionally, so this camera -- which is
    shaped like the real one -- raised AttributeError and the run recorded
    G1=0/G2=0. Reading `data.pos_w`/`data.quat_w_world` is the documented
    surface and is what makes the readback work at all.
    """
    camera = _Lab23Camera([[1.5, -2.0, 1.25]], [[0.0, 0.0, 0.0, 1.0]])
    assert not hasattr(camera, "get_world_poses"), "the fake must reproduce the real shape"

    position, orientation = read_world_pose_rows(camera)

    assert position == pytest.approx([1.5, -2.0, 1.25])
    assert orientation == pytest.approx([0.0, 0.0, 0.0, 1.0])


def test_the_world_convention_quaternion_is_the_one_read() -> None:
    """`set_pose` writes with convention="world", so the readback reads `quat_w_world`.

    `quat_w_ros` is a real orientation in a different frame. Reading it would
    return a plausible quaternion carrying the wrong yaw, which is a worse
    failure than the crash: nothing downstream could tell.
    """
    camera = _Lab23Camera(
        [[0.0, 0.0, 1.0]],
        [[1.0, 0.0, 0.0, 0.0]],
        quat_w_ros=[[0.5, -0.5, 0.5, -0.5]],
    )

    _position, orientation = read_world_pose_rows(camera)

    assert orientation == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_a_camera_that_only_has_the_legacy_method_is_still_readable() -> None:
    # The XForm-view surface the original call was written against. Kept working
    # so a camera-like object that really does provide it is not broken by the fix.
    camera = _LegacyPrimViewCamera([[3.0, 4.0, 5.0]], [[1.0, 0.0, 0.0, 0.0]])

    position, orientation = read_world_pose_rows(camera)

    assert camera.calls == 1
    assert position == pytest.approx([3.0, 4.0, 5.0])
    assert orientation == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_the_buffered_sensor_pose_wins_when_both_surfaces_exist() -> None:
    # `camera.data` is the pose the renderer used for the frame just captured,
    # so it is the one a readback should report.
    camera = _BothSurfacesCamera(
        [[1.0, 1.0, 1.0]],
        [[1.0, 0.0, 0.0, 0.0]],
        [[9.0, 9.0, 9.0]],
        [[0.0, 0.0, 0.0, 1.0]],
    )

    position, _orientation = read_world_pose_rows(camera)

    assert position == pytest.approx([1.0, 1.0, 1.0])
    assert camera.legacy_calls == 0


def test_a_camera_with_no_pose_surface_refuses_and_names_what_it_tried() -> None:
    with pytest.raises(SimulatorNotAvailableError) as excinfo:
        read_world_pose_rows(_NoPoseSurfaceCamera())
    message = str(excinfo.value)
    for surface in CAMERA_POSE_READ_SURFACES:
        assert surface in message, message
    # The version is in the refusal because that is the first thing anyone
    # debugging this on a GPU host will want.
    assert "isaaclab" in message


@pytest.mark.parametrize("field", ["position", "orientation"])
def test_a_non_finite_readback_fails_loudly_instead_of_being_reported(field) -> None:
    """NaN is a documented Isaac failure mode, not a pose.

    It appears when a camera is moved by a parent transform the sensor pose does
    not follow. Passing it on would surface much later as a frozen or nonsense
    trajectory, with nothing left to point at the cause.
    """
    position = [[float("nan"), 0.0, 1.0]] if field == "position" else [[0.0, 0.0, 1.0]]
    orientation = (
        [[float("nan"), 0.0, 0.0, 0.0]] if field == "orientation" else [[1.0, 0.0, 0.0, 0.0]]
    )
    camera = _Lab23Camera(position, orientation)

    with pytest.raises(RuntimeError, match=f"non-finite {field}"):
        read_world_pose_rows(camera)


def test_a_readback_of_the_wrong_shape_fails_loudly() -> None:
    # A three-component "quaternion" is not a quaternion; believing it would put
    # a wrong yaw into the trace.
    camera = _Lab23Camera([[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
    with pytest.raises(RuntimeError, match=r"orientation of shape \(1, 3\)"):
        read_world_pose_rows(camera)


def test_an_empty_batch_is_refused_rather_than_indexed() -> None:
    camera = _Lab23Camera(np.zeros((0, 3)), np.zeros((0, 4)))
    with pytest.raises(RuntimeError, match="at least one row"):
        read_world_pose_rows(camera)


def _env_with_camera(camera, **settings_kwargs) -> CameraWalkEnv:
    """A CameraWalkEnv wired to *camera*, with no Isaac underneath."""
    env = CameraWalkEnv(CameraWalkSettings(**settings_kwargs))
    env.camera = camera
    env._torch = object()  # only _require_camera looks at it
    return env


def test_read_camera_pose_recovers_the_floor_pose_on_the_wxyz_line() -> None:
    # The 2.3 line: native WXYZ, so the canonical order is what the camera gave.
    # The rendered pose sits at floor + camera_height_m; the logical pose does not.
    yaw = math.pi / 3
    env = _env_with_camera(
        _Lab23Camera([[2.0, -1.0, 1.6]], [native_orientation(yaw, "wxyz")]),
        camera_height_m=1.6,
        native_quat_order="wxyz",
    )

    pose, quat = env.read_camera_pose()

    assert (pose.x, pose.y) == pytest.approx((2.0, -1.0))
    # float32 in, float32 out: the camera height cancels to a float32 zero.
    assert pose.z == pytest.approx(0.0, abs=1e-6)
    assert pose.yaw == pytest.approx(yaw, abs=1e-6)
    assert quat == pytest.approx(quat_wxyz_from_yaw(yaw), abs=1e-6)


def test_read_camera_pose_converts_the_xyzw_line_back_to_canonical_wxyz() -> None:
    """The 6.0 conversion, still explicit and still exercised through the new surface.

    The camera reports in the line-native order; canonical WXYZ is what leaves
    this module, on either line. Dropping the conversion when the read path
    moved would have silently reordered every quaternion on the 3.x line.
    """
    yaw = -math.pi / 4
    env = _env_with_camera(
        _Lab23Camera([[0.0, 0.0, 1.0]], [native_orientation(yaw, "xyzw")]),
        camera_height_m=1.0,
        native_quat_order="xyzw",
    )

    pose, quat = env.read_camera_pose()

    assert pose.z == pytest.approx(0.0, abs=1e-6)
    assert pose.yaw == pytest.approx(yaw, abs=1e-6)
    assert quat == pytest.approx(quat_wxyz_from_yaw(yaw), abs=1e-6)
    # Canonical, not the native order it arrived in.
    assert quat[0] == pytest.approx(math.cos(yaw / 2), abs=1e-6)


def test_set_pose_then_read_camera_pose_round_trips_on_both_lines() -> None:
    """Write and read are mirrors: whatever `set_pose` sends, the readback returns.

    This is the invariant the G2 yaw check depends on, and the one that makes the
    native-order setting a single decision rather than two that can disagree.
    """
    for order in ("wxyz", "xyzw"):
        for yaw in (0.0, 0.05, math.pi / 2, -2.5):
            written = native_orientation(yaw, order)
            env = _env_with_camera(
                _Lab23Camera([[1.0, 2.0, 1.0 + 0.75]], [written]),
                camera_height_m=0.75,
                native_quat_order=order,
            )
            pose, quat = env.read_camera_pose()
            assert pose.yaw == pytest.approx(yaw, abs=1e-6), (order, yaw)
            assert pose.z == pytest.approx(1.0, abs=1e-6), (order, yaw)
            assert quat == pytest.approx(quat_wxyz_from_yaw(yaw), abs=1e-6), (order, yaw)


def test_no_module_in_the_package_calls_get_world_poses_unguarded() -> None:
    """The call that broke L20 must not come back as a bare attribute access.

    `read_world_pose_rows` reaches it only behind a `callable(...)` check. Any
    other call site would be the same bug again, on whichever line drops the
    method next.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "src" / "insight_bench"
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get_world_poses"
            ):
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert offenders == [], (
        "camera.get_world_poses() is called as a direct attribute here, which is "
        f"the AttributeError that took G1 and G2 to 0 on Isaac Lab 2.3: {offenders}"
    )


# --- pose buffer preconditions ---------------------------------------------
#
# Naming the right surface is only half of a pose readback. `camera.data`'s pose
# fields exist unconditionally but are *filled* only when
# `CameraCfg.update_latest_camera_pose` is on -- it defaults to False -- and are
# *current* only after an `update`, because `SensorBase.data` is lazy. Miss
# either and the readback still returns finite floats: zeroes, or the pose from
# before the last `set_pose`. That is a quieter failure than the AttributeError
# it replaced, because every downstream check passes on a plausible number.


class _RecordingOffsetCfg:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _RecordingCameraCfg:
    """Stands in for `isaaclab.sensors.CameraCfg`, recording how it was built."""

    OffsetCfg = _RecordingOffsetCfg

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _RecordingCamera:
    def __init__(self, cfg) -> None:
        self.cfg = cfg


class _StubSimUtils:
    @staticmethod
    # Capitalised to mirror the Isaac Lab class name `_create_camera` looks up.
    def PinholeCameraCfg(**kwargs):
        return kwargs


def test_the_camera_config_turns_the_pose_buffers_on() -> None:
    """TRAP: `update_latest_camera_pose` defaults to False on 2.3 and 3.x alike.

    With it False a plain `Camera` only zeroes `data.pos_w`/`data.quat_w_world`
    in `_create_buffers`, and `_update_buffers_impl` skips the pose refresh
    entirely -- so every readback is (0, 0, 0) with a zero quaternion. Nothing
    raises; the G2 yaw gate simply compares against a fabricated pose. Pure
    config, so this holds on a machine with no Isaac.
    """
    camera = _create_camera(
        CameraWalkSettings(),
        _StubSimUtils,
        _RecordingCamera,
        _RecordingCameraCfg,
        prim_path="/World/Camera",
    )

    assert camera.cfg.kwargs["update_latest_camera_pose"] is True


def test_the_pose_buffer_flag_is_set_on_the_config_not_after_construction() -> None:
    # `CameraCfg` is a configclass the sensor reads at __init__ time; a flag
    # assigned afterwards would arrive too late to change what gets buffered.
    camera = _create_camera(
        CameraWalkSettings(),
        _StubSimUtils,
        _RecordingCamera,
        _RecordingCameraCfg,
        prim_path="/World/Camera",
    )

    assert isinstance(camera.cfg, _RecordingCameraCfg)
    assert "update_latest_camera_pose" in camera.cfg.kwargs


class _LazyLab23Camera:
    """Lab 2.3's Camera *including* the laziness that makes the refresh matter.

    `SensorBase.data` recomputes only while `_is_outdated` is set, and only
    `update(dt)` sets it. This fake reproduces exactly that: the pose it
    publishes is whatever the most recent `update` promoted, so reading without
    updating returns the previous contents -- which, before the first update,
    are the zeroes `_create_buffers` left there.
    """

    def __init__(self, pos_w, quat_w_world) -> None:
        self._staged = _PoseData(pos_w, quat_w_world)
        self._published = _PoseData([[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0]])
        self.events: list[str] = []
        self.update_calls: list[tuple[float, bool]] = []

    def update(self, dt: float, force_recompute: bool = False) -> None:
        self.events.append("update")
        self.update_calls.append((dt, force_recompute))
        self._published = self._staged

    @property
    def data(self) -> _PoseData:
        self.events.append("data")
        return self._published


def test_the_pose_buffers_are_refreshed_before_data_is_read() -> None:
    """TRAP: a `set_pose` with no intervening step reads back the *previous* pose.

    `capture_observation` can be called outside `step_motion`, and there is no
    `sim.step()` on that path to mark the sensor outdated. An off-by-one-step
    readback is the worst shape this bug takes -- finite, plausible, and wrong.
    """
    camera = _LazyLab23Camera([[1.0, 2.0, 3.0]], [[1.0, 0.0, 0.0, 0.0]])

    position, orientation = read_world_pose_rows(camera)

    assert camera.events[0] == "update", camera.events
    assert "data" in camera.events, camera.events
    assert camera.update_calls == [(0.0, True)]
    assert position == pytest.approx([1.0, 2.0, 3.0])
    assert orientation == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_the_unrefreshed_camera_really_would_have_read_zeroes() -> None:
    # Proves the ordering test above is not vacuous: this is what `read_camera_pose`
    # returns when the refresh is dropped -- a full pose made entirely of zeroes.
    camera = _LazyLab23Camera([[1.0, 2.0, 3.0]], [[1.0, 0.0, 0.0, 0.0]])

    assert np.array_equal(camera.data.pos_w.numpy(), np.zeros((1, 3), dtype=np.float32))
    assert np.array_equal(camera.data.quat_w_world.numpy(), np.zeros((1, 4), dtype=np.float32))
    assert "update" not in camera.events


class _DtOnlyUpdateCamera(_Lab23Camera):
    """A sensor whose `update` predates `force_recompute`."""

    def __init__(self, pos_w, quat_w_world) -> None:
        super().__init__(pos_w, quat_w_world)
        self.update_calls: list[tuple[float, ...]] = []

    def update(self, dt: float) -> None:
        self.update_calls.append((dt,))


class _RefusingUpdateCamera(_Lab23Camera):
    """A sensor that accepts the keyword and then fails inside the refresh."""

    def update(self, dt: float, force_recompute: bool = False) -> None:
        raise TypeError("the renderer refused the refresh")


def test_an_update_without_force_recompute_is_still_called() -> None:
    # Compatibility is decided from the signature, so an older `update` gets the
    # dt-only call rather than a TypeError. The refresh still happens: `dt=0.0`
    # marks the sensor outdated and the `.data` access recomputes.
    camera = _DtOnlyUpdateCamera([[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0]])

    position, _orientation = read_world_pose_rows(camera)

    assert camera.update_calls == [(0.0,)]
    assert position == pytest.approx([0.0, 0.0, 1.0])


def test_a_typeerror_from_inside_the_refresh_is_not_mistaken_for_an_old_signature() -> None:
    """The reason compatibility is asked of the signature and not of a `try`.

    A `try: update(force_recompute=True) except TypeError: update(dt)` cannot
    tell "no such keyword" from "the refresh ran and raised" -- it would swallow
    the second, silently restore the stale-read bug, and delete the evidence.
    """
    with pytest.raises(TypeError, match="the renderer refused the refresh"):
        read_world_pose_rows(_RefusingUpdateCamera([[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0]]))


def test_a_camera_with_no_update_at_all_is_left_alone() -> None:
    # An `isaacsim.core.prims` XForm view reads USD directly and has no `update`.
    # No buffers to refresh is not an error.
    camera = _LegacyPrimViewCamera([[3.0, 4.0, 5.0]], [[1.0, 0.0, 0.0, 0.0]])
    assert not hasattr(camera, "update"), "the fake must reproduce the real shape"

    position, _orientation = read_world_pose_rows(camera)

    assert position == pytest.approx([3.0, 4.0, 5.0])


class _ProxyArray:
    """Isaac Lab 3.x's `isaaclab.utils.warp.ProxyArray`: views, and no `.numpy()`."""

    def __init__(self, array) -> None:
        self.torch = _FakeTensor(np.asarray(array, dtype=np.float32))


class _Lab3PoseData:
    def __init__(self, pos_w, quat_w_world) -> None:
        self.pos_w = _ProxyArray(pos_w)
        self.quat_w_world = _ProxyArray(quat_w_world)


class _Lab3Camera:
    """Lab 3.x's Camera: same field names, ProxyArray values, XYZW-native order."""

    def __init__(self, pos_w, quat_w_world) -> None:
        self.data = _Lab3PoseData(pos_w, quat_w_world)
        self.update_calls: list[tuple[float, bool]] = []

    def update(self, dt: float, force_recompute: bool = False) -> None:
        self.update_calls.append((dt, force_recompute))


def test_a_lab_3_proxy_array_pose_is_read_through_the_same_surface() -> None:
    """TRAP: on Lab 3.x the field names are unchanged but the values are not tensors.

    `pos_w` and `quat_w_world` come back as ProxyArray, which has neither
    `.detach()` nor `.numpy()`. The conversion unwraps `.torch` -- the view Lab
    instruments with its WXYZ->XYZW migration warning -- and the surface, the
    field names and the refusals are all unchanged.
    """
    camera = _Lab3Camera([[1.5, -2.0, 1.25]], [[0.0, 0.0, 0.0, 1.0]])
    assert not hasattr(camera.data.pos_w, "numpy"), "the fake must reproduce the real shape"

    position, orientation = read_world_pose_rows(camera)

    assert position == pytest.approx([1.5, -2.0, 1.25])
    assert orientation == pytest.approx([0.0, 0.0, 0.0, 1.0])
    assert camera.update_calls == [(0.0, True)]


def test_a_lab_3_proxy_array_readback_still_fails_loudly_on_a_bad_value() -> None:
    # The checks are on the converted values, so the new source does not get a
    # quieter contract than the old one.
    with pytest.raises(RuntimeError, match="non-finite position"):
        read_world_pose_rows(_Lab3Camera([[float("nan"), 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0]]))
    with pytest.raises(RuntimeError, match=r"orientation of shape \(1, 3\)"):
        read_world_pose_rows(_Lab3Camera([[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]]))


def test_a_lab_3_camera_round_trips_yaw_on_its_native_xyzw_order() -> None:
    # The whole chain on the 3.x line: ProxyArray -> numpy -> XYZW-to-canonical.
    yaw = 1.1
    env = _env_with_camera(
        _Lab3Camera([[4.0, 0.5, 1.7]], [native_orientation(yaw, "xyzw")]),
        camera_height_m=1.7,
        native_quat_order="xyzw",
    )

    pose, quat = env.read_camera_pose()

    assert (pose.x, pose.y) == pytest.approx((4.0, 0.5))
    assert pose.z == pytest.approx(0.0, abs=1e-6)
    assert pose.yaw == pytest.approx(yaw, abs=1e-6)
    assert quat == pytest.approx(quat_wxyz_from_yaw(yaw), abs=1e-6)
