"""Isaac Lab free-camera backend for camera-walk benchmark tasks.

A single free camera is teleported to kinematically integrated poses and
rendered. Geometry awareness (terrain following and wall collision) comes from
:mod:`insight_bench.simulator.isaac.terrain_query`.

Every Isaac / torch / Omniverse import lives inside a function body, so this
module imports cleanly on a machine with no simulator and the published wheel
stays Isaac-free. Those imports also only *work* once the Omniverse Kit
application is up, so :meth:`CameraWalkEnv.initialize` requires a running
application before it reaches for one -- ordering enforced, not documented (see
:mod:`insight_bench.simulator.isaac.app`).

Four contract points from the Isaac 6.0.1 compatibility audit are enforced here
rather than left implicit:

1. **Quaternion order.** Poses are built as canonical WXYZ
   (:func:`~insight_bench.simulator.observation.quat_wxyz_from_yaw`) and
   converted to the line-native order exactly once, at
   :meth:`CameraWalkEnv.set_pose`. Isaac Lab 2.3 is WXYZ-native; Lab 3.x is
   XYZW-native, so the conversion site is a setting, not a rewrite.
2. **Buffer reuse.** Camera outputs are views onto a render buffer Isaac
   overwrites on the next step. Every capture therefore returns a private copy
   (see :func:`_capture_rgb`); returning the view would let a stored frame
   silently change value after the fact.
3. **Pose readback surface.** A pose is written with
   :meth:`CameraWalkEnv.set_pose` and read back through
   :func:`read_world_pose_rows`, which reads ``camera.data`` -- the sensor pose
   the renderer actually used -- and refuses rather than guessing. Isaac Lab's
   ``Camera`` has no ``get_world_poses()``; reaching for one is an
   ``AttributeError`` mid-run. See :data:`CAMERA_POSE_READ_SURFACES`.
4. **Pose buffer preconditions.** Naming the right surface is not enough to get
   a number out of it. ``camera.data``'s pose fields are only filled when
   ``CameraCfg.update_latest_camera_pose`` is set (it defaults to False, and a
   plain ``Camera`` zeroes those buffers and never revisits them), and they are
   only *current* after an ``update`` (:func:`_refresh_camera_buffers`). Miss
   either and the readback still returns floats -- zeroes, or last step's pose
   -- which is a quieter failure than the crash it replaced. Both are enforced
   in code: :func:`_create_camera` sets the flag, :func:`read_world_pose_rows`
   refreshes before it reads.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from insight_bench.simulator.base import SimulatorNotAvailableError
from insight_bench.simulator.isaac.app import IsaacAppLifecycle, require_isaac_app
from insight_bench.simulator.observation import (
    Quat,
    quat_wxyz_from_yaw,
    quat_wxyz_to_xyzw,
    to_numpy,
)
from insight_bench.vln_runtime.motion.pose import CameraPose
from insight_bench.vln_runtime.suite.config import BenchmarkTaskConfig

LogCallback = Callable[[str], None]
SceneExtrasCallback = Callable[[BenchmarkTaskConfig | None, Any], None]
NativeQuatOrder = Literal["wxyz", "xyzw"]
RgbFloatRange = Literal["unit", "byte"]

# A fixed render warmup is a race: Omniverse resolves materials and streams textures
# asynchronously, so a handful of renders after a stage load or a camera teleport
# capture whatever happened to have arrived. After the fixed warmup we therefore keep
# rendering the *unchanged* pose until consecutive captures stop moving, which is what
# actually makes the first frame reproducible.
_CONVERGE_STABLE_COMPARISONS = 2

# NVIDIA's own capture paths disable async material/texture loading instead of waiting
# for it, and Isaac Sim's app .kit files never set these keys. Waiting alone is not
# enough: under shared-storage load there is a multi-render stall BEFORE streaming
# starts, during which the streaming-status flag reads idle and the flat
# default-material frame is pixel-stable, so both wait gates pass and a half-loaded
# frame gets captured. Synchronous loads close that hole by construction: a render does
# not complete until the material compiles and the texture mips it requested are
# resident. Only these three keys measurably helped; streaming stays ON and is merely
# made synchronous (disabling it outright moves textures to an async non-streamed
# loader that no sync key covers and that never registers with the status manager).
_SYNC_LOAD_SETTINGS: tuple[tuple[str, Any], ...] = (
    ("/rtx/materialDb/syncLoads", True),
    ("/rtx/hydra/materialSyncLoads", True),
    ("/rtx-transient/resourcemanager/texturestreaming/async", False),
)

_DOME_LIGHT_INTENSITY = 1500.0
_DOME_LIGHT_COLOR = (0.9, 0.92, 1.0)


@dataclass(frozen=True)
class CameraWalkSettings:
    """Sensor, lighting and geometry tunables for one camera-walk scene."""

    camera_width: int = 480
    camera_height_px: int = 270
    # The camera renders at ``floor_z + camera_height_m``; the logical pose the
    # rollout tracks stays at floor level for scoring and waypoint integration.
    camera_height_m: float = 1.0
    camera_hfov_deg: float = 90.0
    sim_dt_sec: float = 1.0 / 30.0
    focal_length_cm: float = 24.0
    focus_distance_cm: float = 400.0
    clipping_range_m: tuple[float, float] = (0.05, 100.0)
    record_depth: bool = False
    dome_light_intensity: float = _DOME_LIGHT_INTENSITY
    dome_light_color: tuple[float, float, float] = _DOME_LIGHT_COLOR
    render_sync_loads: bool = True
    terrain_follow: bool = True
    wall_collision: bool = True
    # Step-up stays just above a stair riser (~0.18 m): larger values let terrain
    # following climb onto furniture or wall ledges, floating the camera off the
    # navmesh. Step-down stays permissive.
    max_step_up_m: float = 0.25
    max_step_down_m: float = 0.45
    collision_radius_m: float = 0.25
    collision_probe_height_m: float = 0.5
    ground_ray_margin_m: float = 0.1
    # Isaac Lab 2.3 takes WXYZ; Lab 3.x takes XYZW. Canonical stays WXYZ either way.
    native_quat_order: NativeQuatOrder = "wxyz"
    # Declared value range of a floating-point rgb frame. None means the camera
    # is expected to emit uint8, which is what Isaac Lab does by default; a
    # float frame then fails loudly rather than being guessed at. See
    # _capture_rgb for why guessing is not an option.
    rgb_float_range: RgbFloatRange | None = None

    @classmethod
    def from_task_config(
        cls,
        task_config: BenchmarkTaskConfig,
        *,
        native_quat_order: NativeQuatOrder = "wxyz",
    ) -> CameraWalkSettings:
        """Read the ``rgb`` sensor entry of *task_config*; defaults fill the rest."""
        params: dict[str, Any] = {}
        width: int | None = None
        height: int | None = None
        for sensor in task_config.sensors:
            if sensor.name == "rgb":
                params = dict(sensor.params)
                width = sensor.width
                height = sensor.height
                break
        clipping = params.get("clipping_range_m", (0.05, 100.0))
        rgb_float_range = params.get("rgb_float_range")
        if rgb_float_range is not None and rgb_float_range not in ("unit", "byte"):
            raise ValueError(
                f"rgb sensor param rgb_float_range must be 'unit' or 'byte', "
                f"got {rgb_float_range!r}"
            )
        return cls(
            camera_width=int(width if width is not None else 480),
            camera_height_px=int(height if height is not None else 270),
            camera_height_m=float(params.get("camera_height_m", 1.0)),
            camera_hfov_deg=float(params.get("hfov_deg", 90.0)),
            sim_dt_sec=float(params.get("sim_dt_sec", 1.0 / 30.0)),
            focal_length_cm=float(params.get("focal_length_cm", 24.0)),
            focus_distance_cm=float(params.get("focus_distance_cm", 400.0)),
            clipping_range_m=(float(clipping[0]), float(clipping[1])),
            record_depth=bool(params.get("record_depth", False)),
            native_quat_order=native_quat_order,
            rgb_float_range=rgb_float_range,
        )


def apply_sync_load_settings(log: LogCallback | None = None) -> bool:
    """Make material compilation and texture streaming synchronous with rendering.

    Returns ``True`` when the carb settings were applied and ``False`` when carb
    is unavailable (non-Omniverse environments). Idempotent.
    """
    try:
        import carb.settings

        settings = carb.settings.get_settings()
    except Exception as exc:
        if log is not None:
            log(f"sync-load settings skipped (carb unavailable: {type(exc).__name__})")
        return False
    for key, value in _SYNC_LOAD_SETTINGS:
        settings.set(key, value)
    if log is not None:
        log("sync-load settings applied: " + ", ".join(f"{k}={v}" for k, v in _SYNC_LOAD_SETTINGS))
    return True


def stage_streaming_busy() -> bool:
    """True while Omniverse is still resolving materials or streaming textures.

    ``omni.usd``'s streaming-status manager aggregates the texture-streaming,
    MDL material-db and hydra geometry/material clients, so unlike a pixel
    heuristic this is the renderer's own answer to "are the assets for this view
    resident yet". It is sampler-feedback driven, so poll it *after* at least one
    render of the posed camera, never before. Returns ``False`` when the API is
    unavailable so non-Omniverse builds keep working.
    """
    try:
        import omni.usd

        return bool(omni.usd.get_context().get_stage_streaming_status())
    except Exception:
        return False


def frame_delta(previous: np.ndarray, current: np.ndarray) -> float:
    """Mean absolute per-channel difference between two captures, in 0-255 units."""
    a = np.asarray(previous, dtype=np.float32)
    b = np.asarray(current, dtype=np.float32)
    if a.shape != b.shape:
        return float("inf")
    if a.size == 0:
        return 0.0
    return float(np.abs(a - b).mean())


def render_until_converged(
    capture: Callable[[], np.ndarray],
    previous: np.ndarray,
    *,
    max_steps: int,
    tol: float,
    busy: Callable[[], bool] | None = None,
    log: LogCallback | None = None,
) -> np.ndarray:
    """Re-render a held pose until the assets are resident and the image stops changing.

    Two gates run in order, sharing one *max_steps* budget:

    1. asset streaming -- while ``busy()`` reports Omniverse is still resolving
       materials or streaming textures, keep rendering. This is the renderer's
       own readiness signal.
    2. temporal settling -- then wait for two consecutive captures whose
       :func:`frame_delta` is ``<= tol``. Streaming can finish while the image
       is still moving, so the pixel gate is still needed after gate 1.

    Always returns the newest frame, so a scene that never settles within
    *max_steps* still yields a better-settled frame than a fixed warmup did
    (plus a warning).
    """
    if max_steps <= 0:
        return previous
    frame = previous
    steps = 0
    streaming_steps = 0
    if busy is not None:
        while steps < max_steps and busy():
            frame = capture()
            steps += 1
        streaming_steps = steps
    stable = 0
    delta = 0.0
    converged = False
    while steps < max_steps:
        current = capture()
        steps += 1
        delta = frame_delta(frame, current)
        frame = current
        if delta <= tol:
            stable += 1
            if stable >= _CONVERGE_STABLE_COMPARISONS:
                converged = True
                break
        else:
            stable = 0
    if log is not None:
        if not converged:
            log(
                f"render did not converge within {max_steps} extra steps "
                f"(last frame delta {delta:.3f} > tol {tol:.3f}); "
                "the first frame may not be reproducible"
            )
        log(
            f"first-frame converge: streaming_steps={streaming_steps} "
            f"settle_steps={steps - streaming_steps} last_delta={delta:.3f} "
            f"converged={converged} budget={max_steps}"
        )
    return frame


def native_orientation(yaw: float, order: NativeQuatOrder) -> tuple[float, ...]:
    """Canonical WXYZ for *yaw*, reordered into the line-native quaternion order.

    The canonical order never changes; only this one call site knows what the
    installed Isaac Lab expects.
    """
    canonical: Quat = quat_wxyz_from_yaw(yaw)
    if order == "wxyz":
        return canonical
    return quat_wxyz_to_xyzw(canonical)


CAMERA_POSE_READ_SURFACES: tuple[str, ...] = (
    "camera.data.pos_w + camera.data.quat_w_world",
    "camera.get_world_poses()",
)
"""The public camera pose surfaces this SDK reads, in the order it tries them.

Both are public API; there is no private fallback, and the reason is recorded
because the obvious candidates do not work. ``isaaclab.sensors.Camera`` has no
``_view`` worth reaching for, and ``camera._sensor_prims`` is a list of
``UsdGeom.Camera`` prims -- USD schema objects with no ``get_world_poses`` on
them at all -- so a private tier would not be a fallback, it would be a second
way to raise ``AttributeError``. If a future line breaks both public surfaces,
add the private tier *then*, guarded on the ``isaaclab`` version that needs it,
rather than shipping an unexercised guess now.

Order matters. ``camera.data`` is the buffered sensor pose -- the one the
renderer actually used for the frame just captured -- so it is what a readback
should report. ``get_world_poses()`` is second because that is what this module
called before, and it is kept so a camera-like object that still provides it
(an ``isaacsim.core.prims`` XForm view, which is where the call came from
originally) keeps working. It is not a deprecation shim for Isaac Lab: Lab's
``Camera`` never had the method, which is the bug this list exists to fix.
"""


def _isaaclab_version() -> str:
    """``isaaclab.__version__`` if it can be read, else ``"unknown"``.

    Diagnostics only, and deliberately total: this runs while building a message
    about a runtime that has already surprised us, so it must not add a second
    exception on top of the first.
    """
    try:
        import isaaclab

        return str(getattr(isaaclab, "__version__", "unknown"))
    except Exception:
        return "unknown"


def _update_accepts_force_recompute(update: Any) -> bool:
    """Whether *update* declares a ``force_recompute`` parameter.

    Asked of the signature rather than by calling and catching ``TypeError``.
    That catch cannot tell "this build's ``update`` has no such keyword" from
    "the refresh ran and raised ``TypeError`` inside the renderer", and
    swallowing the second would put the stale-read bug straight back with the
    evidence deleted.
    """
    try:
        parameters = inspect.signature(update).parameters
    except (TypeError, ValueError):
        # Only reachable for a callable whose signature cannot be introspected
        # -- a C function, some mocks. Not an error, and not a case to refuse
        # on: the dt-only call is a valid `SensorBase.update` on every Isaac Lab
        # line, it just leaves the refresh to the `.data` access instead of
        # forcing it here.
        return False
    return "force_recompute" in parameters


def _refresh_camera_buffers(camera: Any) -> None:
    """Bring the sensor buffers up to date, before anything reads them.

    ``SensorBase.data`` is lazy: it recomputes only while ``_is_outdated`` is
    set, and only ``update(dt)`` sets it. So a ``set_pose`` followed by a read
    with no intervening ``sim.step()`` -- which is exactly what
    ``capture_observation`` does when it is called outside ``step_motion`` --
    would report the *previous* pose. An off-by-one-step readback is the worst
    shape this bug can take: every value is finite and plausible, so no check
    downstream fires, and the trajectory is simply wrong by one step.

    ``dt=0.0`` is enough to set the flag (the sensor's ``update_period`` is 0.0
    and the comparison carries a ``1e-6`` epsilon), and ``force_recompute``
    makes the refresh happen here rather than at the first attribute touch, so
    the ordering is in the call sequence instead of in an invariant someone has
    to remember. A camera-like object with no ``update`` -- an
    ``isaacsim.core.prims`` XForm view reads USD directly and has none -- has no
    buffers to refresh and is left alone.
    """
    update = getattr(camera, "update", None)
    if not callable(update):
        return
    if _update_accepts_force_recompute(update):
        update(dt=0.0, force_recompute=True)
    else:
        update(dt=0.0)


def read_world_pose_rows(camera: Any) -> tuple[list[float], list[float]]:
    """Return ``(position_xyz, native_quaternion)`` for the camera's world pose.

    The quaternion comes back in the *line-native* order, exactly as
    :func:`native_orientation` writes it, so the caller applies one conversion
    and the read stays the mirror of the write. Position is world-frame metres.

    Isaac Lab 2.3's ``Camera`` has no ``get_world_poses()``; calling it is an
    ``AttributeError`` mid-run, which is how a real L20 job lost G1 and G2. The
    documented surface is ``camera.data``: ``pos_w`` and ``quat_w_world``, the
    latter named for the same ``convention="world"`` that
    :meth:`CameraWalkEnv.set_pose` writes with. Reading ``quat_w_ros`` or
    ``quat_w_opengl`` instead would return a real orientation in the wrong
    frame -- a plausible number with the wrong yaw in it, which is worse than
    the crash.

    Fails loud on both halves of the problem, because a pose readback that
    guesses is indistinguishable from a simulator that is not moving:

    * no recognised surface -> :class:`SimulatorNotAvailableError` naming every
      surface tried and the Isaac Lab version that offered none of them;
    * a surface that answered with the wrong shape or a non-finite number ->
      :class:`RuntimeError` naming the surface and the value. ``NaN`` here is a
      documented failure mode when a camera is moved by a parent transform that
      the sensor pose does not follow, and silently returning it would show up
      as a frozen or nonsense trajectory much further downstream.

    The buffers are refreshed first -- see :func:`_refresh_camera_buffers` --
    because ``camera.data`` is lazy and would otherwise be free to answer with
    the pose from before the last ``set_pose``.
    """
    _refresh_camera_buffers(camera)
    data = getattr(camera, "data", None)
    positions = getattr(data, "pos_w", None)
    orientations = getattr(data, "quat_w_world", None)
    if positions is not None and orientations is not None:
        return _pose_rows(positions, orientations, source=CAMERA_POSE_READ_SURFACES[0])

    legacy = getattr(camera, "get_world_poses", None)
    if callable(legacy):
        positions, orientations = legacy()
        return _pose_rows(positions, orientations, source=CAMERA_POSE_READ_SURFACES[1])

    raise SimulatorNotAvailableError(
        "this camera exposes no pose surface this SDK can read: tried "
        + ", ".join(CAMERA_POSE_READ_SURFACES)
        + f" on {type(camera).__name__} (isaaclab {_isaaclab_version()}). "
        "Isaac Lab's Camera reports its pose through camera.data.pos_w and "
        "camera.data.quat_w_world; a runtime providing neither cannot be read "
        "back, and a pose readback is not something to approximate"
    )


def _pose_rows(
    positions: Any, orientations: Any, *, source: str
) -> tuple[list[float], list[float]]:
    """Row 0 of a batched pose readback, checked before it is believed."""
    position = _finite_row(positions, width=3, source=source, field="position")
    orientation = _finite_row(orientations, width=4, source=source, field="orientation")
    return position, orientation


def _finite_row(value: Any, *, width: int, source: str, field: str) -> list[float]:
    array = np.asarray(to_numpy(value), dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != width:
        raise RuntimeError(
            f"{source} returned a {field} of shape {array.shape}; a batched "
            f"(N, {width}) readback with at least one row is required"
        )
    row = array[0]
    if not np.all(np.isfinite(row)):
        raise RuntimeError(
            f"{source} returned a non-finite {field} {row.tolist()}. The camera "
            "pose is unreadable, which a readback must say rather than pass on "
            "as a measurement"
        )
    return [float(component) for component in row]


def _capture_rgb(camera: Any, *, float_range: RgbFloatRange | None = None) -> np.ndarray:
    """Return a private uint8 ``(H, W, 3)`` copy of the camera's RGB output.

    Two contracts, both deliberate.

    **The copy is mandatory**, not defensive style: ``camera.data.output``
    hands back a view onto a render buffer that the next ``sim.step()``
    overwrites, so a stored frame would silently change value. ``.cpu()``
    happens to copy for CUDA tensors, but a CPU-device camera aliases.

    **The value range is declared, never inferred.** Isaac Lab emits uint8 by
    default, so a float frame means the camera was configured differently and
    the caller has to say in which range. Inferring it from ``frame.max()``
    -- "max <= 1.0, so this must be 0.0-1.0" -- is wrong for exactly the frames
    that matter: a legitimately dark 0.0-255.0 frame whose brightest pixel is
    0.8 would be multiplied by 255 and come out ~200x too bright, and nothing
    downstream could tell. Set the rgb sensor param ``rgb_float_range`` to
    ``"unit"`` or ``"byte"`` instead.
    """
    rgb = camera.data.output["rgb"]
    if rgb is None:
        raise RuntimeError("camera.data.output['rgb'] is None")

    frame = np.asarray(rgb[0].detach().cpu().numpy())
    if frame.ndim == 3 and frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.dtype != np.uint8:
        frame = _rgb_from_float(frame, float_range)
    return np.array(frame[:, :, :3], dtype=np.uint8, copy=True, order="C")


def _rgb_from_float(frame: np.ndarray, float_range: RgbFloatRange | None) -> np.ndarray:
    """Convert a non-uint8 rgb frame using the *declared* range."""
    if float_range is None:
        raise ValueError(
            f"camera returned {frame.dtype} rgb, but no rgb_float_range is configured. "
            "Set the rgb sensor param rgb_float_range to 'unit' (0.0-1.0) or 'byte' "
            "(0.0-255.0). This is not inferred: a dark byte-ranged frame and a bright "
            "unit-ranged one are indistinguishable from their values alone, and "
            "guessing silently rescales real pixels."
        )
    scaled = frame * 255.0 if float_range == "unit" else frame
    converted: np.ndarray = np.clip(scaled, 0.0, 255.0).astype(np.uint8)
    return converted


def _capture_depth(camera: Any) -> np.ndarray | None:
    """Return a private float32 distance-to-image-plane frame, or ``None``.

    ``None`` means the camera was not configured to record depth. An output
    object that is not mapping-shaped is a different situation -- the Isaac
    API changed underneath us -- and raises rather than silently reporting
    "no depth", which would let the G2 observation check pass on a gap.
    """
    out = camera.data.output
    getter = getattr(out, "get", None)
    if not callable(getter):
        raise TypeError(f"camera.data.output is not mapping-shaped, got {type(out).__name__}")
    depth = getter("distance_to_image_plane")
    if depth is None:
        return None
    return np.array(depth[0].detach().cpu().numpy(), dtype=np.float32, copy=True)


def _capture_intrinsics(camera: Any) -> np.ndarray | None:
    """Return a private float ``(3, 3)`` intrinsics matrix, or ``None``."""
    matrices = camera.data.intrinsic_matrices
    if matrices is None:
        return None
    return np.array(matrices[0].detach().cpu().numpy(), dtype=np.float64, copy=True)


def _rgb_camera_prim_path(task_config: BenchmarkTaskConfig | None) -> str:
    if task_config is None:
        return "/World/PolicyLoopCam"
    rgb_sensor = next(
        (sensor for sensor in task_config.sensors if sensor.name == "rgb"),
        None,
    )
    if rgb_sensor is None or not rgb_sensor.prim_path:
        return "/World/PolicyLoopCam"
    return rgb_sensor.prim_path


#: Where the scene's only light lives. Fixed, so it has to be cleared between scenes.
DOME_LIGHT_PRIM_PATH = "/World/DomeLight"


def _remove_existing_prim(prim_path: str) -> None:
    try:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
    except Exception:
        return
    if stage is None:
        return
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        stage.RemovePrim(prim_path)


def _no_interactive_hold(self: object, *args: object, **kwargs: object) -> None:
    """Return, where Isaac Lab would render until somebody plays the timeline."""
    return None


def _disable_interactive_stop_handler() -> None:
    """Stop Isaac Lab's timeline-stop handler hanging a headless run.

    ``SimulationContext.__init__`` subscribes to the timeline's STOP event, and
    the handler it installs is::

        if not self._disable_app_control_on_stop_handle:
            while not omni.timeline.get_timeline_interface().is_playing():
                self.render()

    It exists so a person driving Isaac Sim from a terminal keeps a live
    viewport after pressing stop. In a benchmark run nothing will ever play the
    timeline again, so it renders forever -- and it is subscribed and armed
    entirely inside the constructor, so there is no moment afterwards at which a
    caller could set ``_disable_app_control_on_stop_handle`` in time.

    The first scene of a run survives, because stopping an already-stopped
    timeline emits no STOP event. The second does not: opening a scene tears the
    previous context down, the new constructor stops the timeline for real, its
    own freshly-subscribed handler fires, and the process spins at 200% CPU
    producing nothing -- no output, no log line, no cache growth. Measured at 40
    minutes on both Isaac Lab 2.2.1 and 2.3.2. Every run over more than one
    scene ended there, which is every real run.

    Replacing the method is deliberate, and narrower than it looks. Isaac Lab's
    own switch for this, ``builtins.ISAAC_LAUNCHED_FROM_TERMINAL``, also gates an
    ``app.update()`` that a freshly created stage needs; setting it traded the
    hang for a physics context that was never built. This overrides one method,
    whose entire body is the interactive hold, and leaves everything else alone.
    """
    from isaaclab.sim import SimulationContext

    if getattr(SimulationContext, "_insight_stop_hold_disabled", False):
        return
    SimulationContext._app_control_on_stop_handle_fn = _no_interactive_hold
    SimulationContext._insight_stop_hold_disabled = True


def _spawn_terrain(task_config: BenchmarkTaskConfig | None, sim_utils: Any) -> None:
    terrain = task_config.terrain if task_config is not None else None
    if terrain is None or terrain.kind == "plane":
        prim_path = terrain.prim_path if terrain is not None else "/World/GroundPlane"
        _remove_existing_prim(prim_path)
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func(prim_path, ground_cfg)
        return
    if terrain.kind == "usd":
        from isaaclab.terrains import TerrainImporter, TerrainImporterCfg

        _remove_existing_prim(terrain.prim_path)
        TerrainImporter(TerrainImporterCfg(**terrain.to_isaaclab_terrain_kwargs()))
        return
    raise ValueError(f"unsupported terrain kind for the camera-walk backend: {terrain.kind}")


def _build_dome_light(
    sim_utils: Any, *, intensity: float, color: tuple[float, float, float]
) -> Any:
    """Build the scene's ambient DomeLight cfg (extracted so it is unit-testable)."""
    return sim_utils.DomeLightCfg(intensity=float(intensity), color=tuple(color))


def _create_camera(
    settings: CameraWalkSettings,
    sim_utils: Any,
    camera_cls: Any,
    camera_cfg_cls: Any,
    *,
    prim_path: str,
) -> Any:
    _remove_existing_prim(prim_path)
    horizontal_aperture_cm = (
        2.0 * settings.focal_length_cm * math.tan(math.radians(settings.camera_hfov_deg) / 2.0)
    )
    data_types = ["rgb", "distance_to_image_plane"] if settings.record_depth else ["rgb"]
    camera_cfg = camera_cfg_cls(
        prim_path=prim_path,
        update_period=0.0,
        width=settings.camera_width,
        height=settings.camera_height_px,
        data_types=data_types,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=settings.focal_length_cm,
            focus_distance=settings.focus_distance_cm,
            horizontal_aperture=horizontal_aperture_cm,
            clipping_range=settings.clipping_range_m,
        ),
        offset=camera_cfg_cls.OffsetCfg(
            pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"
        ),
        # The precondition for reading a pose back at all. `CameraCfg` defaults
        # this to False on both the 2.3 and the 3.x line, and with it False a
        # plain `Camera` never writes `data.pos_w` / `data.quat_w_world`: its
        # `_create_buffers` only zeroes them and `_update_buffers_impl` skips
        # the pose refresh entirely, so a readback returns (0, 0, 0) and a zero
        # quaternion. That is the failure this flag exists to prevent, and it is
        # a worse one than the AttributeError it replaced -- it compiles, it
        # runs, and it hands the G2 yaw gate a wrong number with nothing to
        # point at. Set unconditionally rather than probed: a line old enough
        # not to have the field (pre-2.1) should refuse loudly here, at camera
        # construction, instead of running a whole job on zeroes. Costs an
        # XformPrim/FrameView pose read per update, which is noise for one
        # teleporting camera.
        update_latest_camera_pose=True,
    )
    return camera_cls(cfg=camera_cfg)


class CameraWalkEnv:
    """A single Isaac Lab free camera that executes poses and returns RGB frames."""

    def __init__(
        self,
        settings: CameraWalkSettings,
        *,
        task_config: BenchmarkTaskConfig | None = None,
        scene_extras: SceneExtrasCallback | None = None,
        app: IsaacAppLifecycle | None = None,
    ) -> None:
        self.settings = settings
        self.task_config = task_config
        self.scene_extras = scene_extras
        # Which application must be up before this env may touch Isaac. Defaults
        # to the process-global one; the backend passes its own so the two can
        # never disagree about which Kit is running.
        self._app = app
        self._terrain_query: Any | None = None
        # Outcome of the most recent resolve_motion, read by the rollout for the step trace.
        # Two flags rather than one: "no mesh to query" and "queried and missed" are different
        # facts about a scene and collapsing them loses the one a reader needs.
        self._last_terrain_query_available: bool = False
        self._last_terrain_raycast_hit: bool = False
        self._torch: Any | None = None
        self.sim: Any | None = None
        self.camera: Any | None = None
        self.sync_loads_applied: bool = False

    # -- lifecycle ---------------------------------------------------------

    def initialize(self, log: LogCallback | None = None) -> None:
        """Build the simulation context, scene and camera. Requires a running app.

        The guard is the first statement on purpose: every import below it exists
        only because Omniverse Kit is up, and without the guard a caller that
        forgot to start it gets ``No module named 'pxr'`` from somewhere in the
        stage loader instead of being told what it actually did wrong. This never
        starts the application itself -- see
        :func:`~insight_bench.simulator.isaac.app.require_isaac_app`.
        """
        require_isaac_app(self._app)

        import isaaclab.sim as sim_utils
        import torch
        from isaaclab.sensors.camera import Camera, CameraCfg
        from isaaclab.sim import SimulationCfg, SimulationContext

        # Before anything renders: make asset loads synchronous with rendering.
        self.sync_loads_applied = (
            apply_sync_load_settings(log) if self.settings.render_sync_loads else False
        )

        self._torch = torch
        _disable_interactive_stop_handler()
        self.sim = SimulationContext(SimulationCfg(dt=self.settings.sim_dt_sec))
        _spawn_terrain(self.task_config, sim_utils)
        dome_cfg = _build_dome_light(
            sim_utils,
            intensity=self.settings.dome_light_intensity,
            color=self.settings.dome_light_color,
        )
        # The terrain and the camera already clear their own paths; the light did
        # not, and it is the first fixed path a second scene collides with --
        # "a prim already exists at path: '/World/DomeLight'". Isaac's own
        # SimulationContext.clear() is documented to leave /World standing and
        # did not remove it here, so the scene clears the paths it owns itself
        # rather than depending on what a stage clear happens to sweep up.
        _remove_existing_prim(DOME_LIGHT_PRIM_PATH)
        dome_cfg.func(DOME_LIGHT_PRIM_PATH, dome_cfg)
        if self.scene_extras is not None:
            self.scene_extras(self.task_config, sim_utils)
        self.camera = _create_camera(
            self.settings,
            sim_utils,
            Camera,
            CameraCfg,
            prim_path=_rgb_camera_prim_path(self.task_config),
        )
        self.sim.reset()
        self._build_terrain_query()

    def close(self, *, clear_stage: bool = False, log: LogCallback | None = None) -> None:
        if self.sim is None:
            return
        sim = self.sim

        def _log(message: str) -> None:
            if log is not None:
                log(message)

        try:
            self.camera = None
            _log("camera backend close: disabling the Isaac Lab stop callback")
            if hasattr(sim, "_disable_app_control_on_stop_handle"):
                sim._disable_app_control_on_stop_handle = True
            stop_handle = getattr(sim, "_app_control_on_stop_handle", None)
            if stop_handle is not None:
                try:
                    stop_handle.unsubscribe()
                finally:
                    sim._app_control_on_stop_handle = None

            _log("camera backend close: destroying Replicator hydra textures")
            try:
                import omni.replicator.core as rep

                rep.vp_manager.destroy_hydra_textures("Replicator")
            except Exception as exc:
                _log(
                    "camera backend close: skipped Replicator texture cleanup "
                    f"({type(exc).__name__}: {exc})"
                )

            _log("camera backend close: stopping the timeline")
            if hasattr(sim, "_timeline"):
                sim._timeline.stop()
            else:
                sim.stop()

            if clear_stage:
                _log("camera backend close: clearing the stage")
                sim.clear()

            if hasattr(sim, "clear_all_callbacks"):
                sim.clear_all_callbacks()
            sim.clear_instance()
            _log("camera backend close: complete")
        finally:
            self.sim = None
            self.camera = None
            self._terrain_query = None
            self._torch = None

    # -- geometry ----------------------------------------------------------

    def _build_terrain_query(self) -> None:
        """Build the Warp geometry query from the scene's USD terrain meshes."""
        self._terrain_query = None
        if not (self.settings.terrain_follow or self.settings.wall_collision):
            return
        terrain = self.task_config.terrain if self.task_config is not None else None
        if terrain is None or terrain.kind != "usd":
            return
        from insight_bench.simulator.isaac.terrain_query import TerrainQuery

        device = str(self.camera.device) if self.camera is not None else "cuda:0"
        self._terrain_query = TerrainQuery.build_from_stage(terrain.prim_path, device)

    def resolve_start_pose(self, pose: CameraPose) -> CameraPose:
        """Snap a start pose's ``z`` onto the floor surface (terrain following only)."""
        query = self._terrain_query
        if query is None or not self.settings.terrain_follow:
            return pose
        ground_z = query.ground_height_at(
            pose.x,
            pose.y,
            z_ref=pose.z,
            max_step_up_m=self.settings.max_step_up_m,
            max_step_down_m=self.settings.max_step_down_m,
            margin_m=self.settings.ground_ray_margin_m,
        )
        if ground_z is None:
            return pose
        return CameraPose(pose.x, pose.y, ground_z, pose.yaw)

    def resolve_motion(self, prev: CameraPose, candidate: CameraPose) -> CameraPose:
        """Resolve an integrated pose against scene geometry.

        Clamps the horizontal move at walls and snaps ``z`` to the floor under
        the resolved ``(x, y)``. If no floor is found within the step window at
        the target -- a ledge, a void, or a wall leading to an unrelated floor --
        the translation is rejected and only the yaw is applied. Returns
        *candidate* unchanged when no geometry query is available (plane
        terrains have no mesh).
        """
        query = self._terrain_query
        if query is None:
            # Plane terrains have no mesh, so no ray is cast at all. Recorded as a distinct state
            # from "cast and missed": a verifier that cannot tell them apart would read a scene
            # with nothing to hit as a scene where terrain following silently failed.
            self._last_terrain_query_available = False
            self._last_terrain_raycast_hit = False
            return candidate
        self._last_terrain_query_available = True

        x, y = candidate.x, candidate.y
        if self.settings.wall_collision:
            frac = query.path_blocked(
                prev.x,
                prev.y,
                candidate.x,
                candidate.y,
                probe_z=prev.z + self.settings.collision_probe_height_m,
                radius_m=self.settings.collision_radius_m,
            )
            if frac < 1.0:
                x = prev.x + frac * (candidate.x - prev.x)
                y = prev.y + frac * (candidate.y - prev.y)

        z = candidate.z
        if self.settings.terrain_follow:
            ground_z = query.ground_height_at(
                x,
                y,
                z_ref=prev.z,
                max_step_up_m=self.settings.max_step_up_m,
                max_step_down_m=self.settings.max_step_down_m,
                margin_m=self.settings.ground_ray_margin_m,
            )
            self._last_terrain_raycast_hit = ground_z is not None
            if ground_z is None:
                return CameraPose(prev.x, prev.y, prev.z, candidate.yaw)
            z = ground_z
        else:
            # Terrain following disabled: the wall query ran, the ground ray did not.
            self._last_terrain_raycast_hit = False

        return CameraPose(x, y, z, candidate.yaw)

    def last_terrain_query(self) -> dict[str, bool]:
        """Report what the most recent ``resolve_motion`` actually did to the geometry."""
        return {
            "terrain_query_available": self._last_terrain_query_available,
            "terrain_raycast_hit": self._last_terrain_raycast_hit,
        }

    # -- stepping and capture ---------------------------------------------

    def reset(
        self,
        pose: CameraPose,
        *,
        warmup_steps: int,
        converge_max_steps: int = 0,
        converge_tol: float = 0.0,
        log: LogCallback | None = None,
    ) -> np.ndarray:
        self.set_pose(pose)
        frame = self.step_and_capture(settle_steps=max(warmup_steps, 1))
        return render_until_converged(
            lambda: self.step_and_capture(settle_steps=1),
            frame,
            max_steps=converge_max_steps,
            tol=converge_tol,
            busy=stage_streaming_busy,
            log=log,
        )

    def set_pose(self, pose: CameraPose) -> None:
        camera, torch = self._require_camera()
        render_pose = CameraPose(pose.x, pose.y, pose.z + self.settings.camera_height_m, pose.yaw)
        position = torch.tensor(
            [[render_pose.x, render_pose.y, render_pose.z]],
            dtype=torch.float32,
            device=camera.device,
        )
        orientation = torch.tensor(
            [list(native_orientation(render_pose.yaw, self.settings.native_quat_order))],
            dtype=torch.float32,
            device=camera.device,
        )
        camera.set_world_poses(positions=position, orientations=orientation, convention="world")

    def step_and_capture(self, *, settle_steps: int = 1) -> np.ndarray:
        camera, _ = self._require_camera()
        if self.sim is None:
            raise RuntimeError("CameraWalkEnv.initialize() must be called before stepping")
        frame: np.ndarray | None = None
        for _ in range(max(1, int(settle_steps))):
            self.sim.step()
            camera.update(dt=self.sim.get_physics_dt())
            frame = _capture_rgb(camera, float_range=self.settings.rgb_float_range)
        assert frame is not None
        return frame

    def capture_rgb(self) -> np.ndarray:
        camera, _ = self._require_camera()
        return _capture_rgb(camera, float_range=self.settings.rgb_float_range)

    def capture_depth(self) -> np.ndarray | None:
        if self.camera is None:
            return None
        return _capture_depth(self.camera)

    def capture_intrinsics(self) -> np.ndarray | None:
        if self.camera is None:
            return None
        return _capture_intrinsics(self.camera)

    def read_camera_pose(self) -> tuple[CameraPose, Quat]:
        """Read the commanded camera pose back out of the simulator.

        Returned in canonical WXYZ regardless of the line-native order; the
        logical (floor-level) pose is recovered by subtracting the eye height.

        The world pose comes from :func:`read_world_pose_rows`, which reads the
        camera's own documented surface and refuses rather than guessing; see
        :data:`CAMERA_POSE_READ_SURFACES`.
        """
        camera, _ = self._require_camera()
        position, native = read_world_pose_rows(camera)
        if self.settings.native_quat_order == "wxyz":
            quat: Quat = (native[0], native[1], native[2], native[3])
        else:
            from insight_bench.simulator.observation import quat_xyzw_to_wxyz

            quat = quat_xyzw_to_wxyz(native)
        from insight_bench.simulator.observation import yaw_from_quat_wxyz

        pose = CameraPose(
            position[0],
            position[1],
            position[2] - self.settings.camera_height_m,
            yaw_from_quat_wxyz(quat),
        )
        return pose, quat

    def _require_camera(self) -> tuple[Any, Any]:
        if self.camera is None or self._torch is None:
            raise RuntimeError("CameraWalkEnv.initialize() must be called first")
        return self.camera, self._torch
