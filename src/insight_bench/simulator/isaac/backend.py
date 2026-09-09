"""The Isaac Sim 5.1 execution backend (Isaac Lab 2.3).

This is the only line with a real implementation in this release. It is the
official, default backend (``isaac-5.1``); ``isaac-6.0`` stays experimental and
unimplemented, because nothing has been verified against Isaac Sim 6.0.1 here.

Nothing in this module runs -- or claims it could -- outside a machine that
actually has Isaac installed. :func:`isaac_import_probe` reports the truth, and
every entry point raises :class:`SimulatorNotAvailableError` with the specific
missing piece rather than degrading into a fake success.

:meth:`IsaacCameraWalkBackend.open_scene` is also where the Omniverse Kit
application is started, before anything that needs ``pxr``/``omni``/
``isaaclab.sim`` exists -- see :mod:`insight_bench.simulator.isaac.app` for why
that ordering is enforced here rather than documented.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any

from insight_bench.contracts import RuntimeAttestation
from insight_bench.simulator.attestation import evaluate_official_profile
from insight_bench.simulator.base import (
    SimBackendDescriptor,
    SimulatorBackend,
    SimulatorNotAvailableError,
    get_backend_descriptor,
)
from insight_bench.simulator.observation import Observation, Quat, make_observation
from insight_bench.vln_runtime.episodes import EpisodeSpec
from insight_bench.vln_runtime.motion.pose import CameraPose
from insight_bench.vln_runtime.rollout import (
    EpisodeRollout,
    RolloutOptions,
    run_camera_walk_episode,
)
from insight_bench.vln_runtime.suite.config import BenchmarkTaskConfig
from insight_bench.vln_runtime.traces.frames import FrameSink

from .app import (
    ISAAC_RUNTIME_IMPORT_SURFACE,
    IsaacAppLifecycle,
    isaac_app,
    require_runtime_symbols,
    runtime_symbol_probe,
)
from .camera_walk import CameraWalkEnv, CameraWalkSettings, LogCallback, SceneExtrasCallback

BACKEND_ID = "isaac-5.1"

ISAAC_IMPORT_SURFACE: tuple[str, ...] = ("isaacsim", "isaaclab", "warp")
"""The G1 *static* import surface. A fixed, package-owned list of module names.

These are constants shipped by this distribution, never registry or manifest
data, so resolving them is not the plugin/module-path loading the Phase 1 trust
boundary forbids.

Static means exactly that: whether the interpreter can *see* these packages,
which is all that can be asked before Omniverse Kit is up. The symbols that only
exist after ``AppLauncher`` runs are the second G1 tier,
:data:`~insight_bench.simulator.isaac.app.ISAAC_RUNTIME_IMPORT_SURFACE`; passing
this tier alone is what let a runtime with no importable ``pxr`` look healthy.
"""


def isaac_import_probe() -> tuple[bool, tuple[str, ...]]:
    """Return ``(available, missing_modules)`` for the static Isaac import surface.

    Uses ``find_spec`` rather than importing: a probe must not start Omniverse
    as a side effect of asking whether Omniverse exists. The post-launch tier is
    :func:`~insight_bench.simulator.isaac.app.runtime_symbol_probe`.
    """
    missing: list[str] = []
    for name in ISAAC_IMPORT_SURFACE:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(name)
    return (not missing, tuple(missing))


def _observed_os() -> str | None:
    """Map the running platform onto the attestation's OS vocabulary."""
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform.startswith("win"):
        return "windows"
    return None


NVML_LIBRARY_NAMES: tuple[str, ...] = ("libnvidia-ml.so.1", "libnvidia-ml.so")
"""NVML shared objects, in the order they are tried. Constants, never data."""

NVML_SUCCESS = 0
"""``NVML_SUCCESS``. Every other return code means the call did not answer."""

NVML_DRIVER_VERSION_BUFFER_SIZE = 80
"""``NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE`` from ``nvml.h``."""

NVIDIA_DRIVER_VERSION_FILES: tuple[str, ...] = (
    "/sys/module/nvidia/version",
    "/proc/driver/nvidia/version",
)
"""Kernel-module driver version files, read when NVML cannot be loaded at all."""

_DRIVER_VERSION_RE = re.compile(r"\b(\d+(?:\.\d+)+)\b")
"""A dotted numeric version. Anything else is not recorded as a driver version."""


def _first_driver_version(text: str) -> str:
    """First dotted-numeric token in *text*, or ``""``.

    Shape-checked rather than trusted. ``/proc/driver/nvidia/version`` wraps the
    number in prose, and a string that is not a version must not be recorded as
    one: the official-profile gate would then report it as *unparseable* -- a
    claim that something was read -- instead of *unreported*, which is what
    actually happened.
    """
    for line in text.splitlines():
        match = _DRIVER_VERSION_RE.search(line)
        if match:
            return match.group(1)
    return ""


def _driver_version_from_nvml() -> str:
    """Ask NVML in-process through ``ctypes``, or return ``""``.

    No subprocess. ``ctypes`` and ``libnvidia-ml.so.1`` are what ``nvidia-smi``
    and ``pynvml`` are both thin wrappers over, so this is the same source
    without spawning anything and without adding a dependency: ``ctypes`` is the
    standard library, and the library is already present wherever the driver is.

    Loaded inside the function on purpose. ``import insight_bench`` must not
    reach for anything simulator- or driver-shaped, and a module-level
    ``CDLL`` would run at import time on every machine, including the ones with
    no NVIDIA driver at all.

    Every failure -- library absent, symbol missing, init refused, query
    refused -- returns ``""``. ``nvmlShutdown`` runs in a ``finally`` so a
    failed query cannot leave NVML initialised inside a long-lived Isaac
    process.
    """
    import ctypes

    library = None
    for name in NVML_LIBRARY_NAMES:
        try:
            library = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if library is None:
        return ""

    try:
        # _v2 is the modern initialiser; the unversioned symbol is the fallback
        # for an older library. AttributeError here means this .so does not
        # export it, which is a "cannot answer", not a crash.
        init = getattr(library, "nvmlInit_v2", None) or getattr(library, "nvmlInit", None)
        query = getattr(library, "nvmlSystemGetDriverVersion", None)
        if init is None or query is None:
            return ""
        if init() != NVML_SUCCESS:
            return ""
        try:
            buffer = ctypes.create_string_buffer(NVML_DRIVER_VERSION_BUFFER_SIZE)
            if query(buffer, ctypes.c_uint(NVML_DRIVER_VERSION_BUFFER_SIZE)) != NVML_SUCCESS:
                return ""
            return _first_driver_version(buffer.value.decode("utf-8", errors="replace"))
        finally:
            shutdown = getattr(library, "nvmlShutdown", None)
            if shutdown is not None:
                shutdown()
    except (OSError, AttributeError, ValueError, UnicodeError):
        return ""


def _driver_version_from_kernel_module() -> str:
    """Read the driver version the kernel module reports, or return ``""``."""
    for path in NVIDIA_DRIVER_VERSION_FILES:
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        version = _first_driver_version(text)
        if version:
            return version
    return ""


def observed_driver_version() -> str:
    """The installed NVIDIA driver version, or ``""`` when it cannot be read.

    NVML first, because it is the authoritative source and is in-process; the
    kernel-module files are a fallback for a container that has the driver
    mounted but not the NVML library.

    **An unreadable driver is reported as unread, never as a value.** There is
    no default, no ``"unknown"``, no zero: those are strings that would flow
    into :func:`~insight_bench.simulator.attestation.evaluate_official_profile`
    and read as a recorded driver. ``""`` is the vocabulary the contract
    already has for "not recorded", and the gate treats it as an unmet
    requirement rather than a skipped check -- the honest outcome when the
    question could not be answered.

    This is the whole point of collecting it: a publishable run must say which
    driver its numbers came from (ADR 0008). The value is recorded, never
    compared against a floor; recording ``""`` on a machine that has a driver
    made every such run indistinguishable from one that had none.
    """
    for source in (_driver_version_from_nvml, _driver_version_from_kernel_module):
        version = source()
        if version:
            return version
    return ""


def observed_isaac_versions() -> dict[str, str]:
    """What the Isaac stack on this machine reports about itself.

    Read on the same terms as :func:`observed_driver_version`: each value is
    evidence, nothing compares it against the backend descriptor's declared
    version, and a value that cannot be read is ``""`` rather than a guess.

    ``isaaclab.__version__`` is the package version, which is not the version
    number an Isaac Lab repository advertises for the framework -- an install
    reporting ``0.46.3`` can sit in a tree whose ``VERSION`` says ``2.2.1``.
    Both are real; this records the one the running code actually reports,
    because that is the one whose API the run met.

    Isaac Sim is harder: ``isaacsim.__version__`` is absent on a source build,
    so the ``VERSION`` file beside the install is the fallback. Neither is
    tried in a way that can raise into a run.
    """
    versions = {"isaac_lab_version": "", "isaac_sim_version": "", "python_version": ""}
    versions["python_version"] = platform.python_version()

    try:
        import isaaclab

        versions["isaac_lab_version"] = str(getattr(isaaclab, "__version__", "") or "")
    except Exception:
        pass

    try:
        import isaacsim

        declared = str(getattr(isaacsim, "__version__", "") or "")
        if declared:
            versions["isaac_sim_version"] = declared
        else:
            root = os.environ.get("ISAAC_PATH") or str(
                Path(getattr(isaacsim, "__file__", "") or ".").resolve().parent
            )
            version_file = Path(root) / "VERSION"
            if version_file.is_file():
                versions["isaac_sim_version"] = version_file.read_text(
                    encoding="utf-8", errors="replace"
                ).strip()[:64]
    except Exception:
        pass

    return versions


def _in_docker() -> bool:
    """Best-effort container detection for the runtime attestation.

    Only ever used to record ``runtime_kind`` honestly. It never grants
    anything: publication does not depend on the container (ADR 0008); the kind
    is recorded so a reviewer knows how the run was brought up.
    """
    if Path("/.dockerenv").exists():
        return True
    cgroup = Path("/proc/self/cgroup")
    try:
        return "docker" in cgroup.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


class IsaacCameraWalkBackend(SimulatorBackend):
    """Free-camera episode execution on the Isaac Sim 5.1 line."""

    def __init__(
        self,
        descriptor: SimBackendDescriptor | None = None,
        *,
        app: IsaacAppLifecycle | None = None,
    ) -> None:
        self._descriptor = descriptor or get_backend_descriptor(BACKEND_ID)
        # Kit is process-global, so the process-global lifecycle is the default;
        # the parameter exists so the ordering can be tested without a GPU.
        self._app = isaac_app() if app is None else app
        self._env: CameraWalkEnv | None = None
        self._task_config: BenchmarkTaskConfig | None = None
        self._frame_id = 0

    # -- declarative facts -------------------------------------------------

    @property
    def task_config(self) -> BenchmarkTaskConfig | None:
        """The task config of the open scene, or ``None`` when none is open.

        What was actually opened, so a caller reporting on the scene cannot
        disagree with the scene: resolving the task a second time from its name
        would be a second answer to a question already settled.
        """
        return self._task_config

    @property
    def descriptor(self) -> SimBackendDescriptor:
        return self._descriptor

    def probe(self) -> dict[str, Any]:
        """Report truthfully whether this backend can execute here, and why not."""
        _available, missing = isaac_import_probe()
        reasons: list[str] = []
        if missing:
            reasons.append(f"missing Isaac import surface: {', '.join(missing)}")
        observed_os = _observed_os()
        if observed_os is None:
            reasons.append(
                f"{sys.platform} is not a supported Isaac Sim platform (linux or windows)"
            )
        return {
            "backend_id": self._descriptor.backend_id,
            "available": not reasons,
            "reasons": reasons,
            "import_surface": list(ISAAC_IMPORT_SURFACE),
            "missing_modules": list(missing),
            "os": observed_os or platform.system().lower(),
            "scene_open": self._env is not None,
            # The two G1 tiers, reported separately. `available` stays keyed on
            # the static tier and the platform, because that is what decides
            # whether open_scene() may be attempted at all; the runtime tier
            # cannot be known before open_scene() starts the application, and
            # reporting an unknown as a failure would make a probe on a healthy
            # machine read as broken.
            "runtime_import_surface": list(ISAAC_RUNTIME_IMPORT_SURFACE),
            "app": self._app.to_dict(),
            "runtime_symbols": runtime_symbol_probe(self._app).to_dict(),
            # No runtime image digest is published for any backend (the image
            # embeds Isaac Sim and is not redistributable); informational only.
            "image_digest": self._descriptor.image_digest,
        }

    def require_available(self) -> None:
        """Raise unless this backend can actually execute on this machine."""
        report = self.probe()
        if not report["available"]:
            raise SimulatorNotAvailableError(
                f"{self._descriptor.backend_id} cannot execute here: "
                + "; ".join(report["reasons"])
            )

    def runtime_attestation(self, *, launcher_id: str | None = None) -> RuntimeAttestation:
        """Record where this run actually executed. Never optimistic.

        ``image_digest`` is ``None`` and ``digest_locked`` is ``False``: the
        runtime image embeds NVIDIA Isaac Sim and is not redistributable, so
        there is no published digest to lock to. Neither field gates
        publication (ADR 0008).

        ``driver_version`` is read off the machine (see
        :func:`observed_driver_version`) and left empty when it cannot be read.
        A recorded driver is evidence, not a floor; an unrecorded one is the
        one driver outcome the official-profile gate rejects.

        ``publishable`` is judged, not declared: the record is built with
        ``publishable=False``, put through
        :func:`~insight_bench.simulator.attestation.evaluate_official_profile`
        against this backend's own descriptor, and re-validated with the
        verdict, so the flag can never say more than the facts beside it.
        """
        observed_os = _observed_os()
        if observed_os is None:
            raise SimulatorNotAvailableError(
                f"{sys.platform} is not a supported Isaac Sim platform (linux or windows)"
            )
        docker = observed_os == "linux" and _in_docker()
        observed = observed_isaac_versions()
        resolved_launcher = launcher_id or ("docker" if docker else f"{observed_os}-native")
        candidate = RuntimeAttestation(
            backend_id=self._descriptor.backend_id,
            backend_status=self._descriptor.status,
            launcher_id=resolved_launcher,
            runtime_kind="docker" if docker else "native",
            os=observed_os,  # type: ignore[arg-type]
            image_digest=None,
            digest_locked=False,
            # Collected, not declared. It was hard-coded to "" here, so every run
            # looked exactly like a run on a machine with no driver at all and
            # the record could never say which driver its numbers came from.
            driver_version=observed_driver_version(),
            # Collected for the same reason as the driver: without it a run on
            # Isaac Lab 0.46.3 and one on the declared 2.3.0 leave identical
            # records, and only one of them is what the reader thinks it is.
            isaac_lab_version=observed["isaac_lab_version"],
            isaac_sim_version=observed["isaac_sim_version"],
            python_version=observed["python_version"],
            publishable=False,
        )
        publishable, _reasons = evaluate_official_profile(candidate, self._descriptor)
        # Re-validated rather than patched: the contract validator has the last
        # word on whether `publishable` is representable beside these facts.
        return RuntimeAttestation.model_validate(
            {**candidate.model_dump(), "publishable": publishable}
        )

    # -- scene lifecycle ---------------------------------------------------

    def open_scene(
        self,
        task_config: BenchmarkTaskConfig,
        *,
        scene_extras: SceneExtrasCallback | None = None,
        log: LogCallback | None = None,
    ) -> None:
        """Bring up the simulator for *task_config*, or refuse with the reason.

        The order below is the point of this method, not an implementation
        detail. The Kit application starts first, because it is what makes
        ``pxr``/``omni``/``isaaclab.sim`` importable at all; the runtime symbols
        are then proven present, so a runtime missing one says so here instead of
        as a bare ``ModuleNotFoundError`` from whichever scene-loading line got
        there first; only then is the scene built.

        A second call reuses the running application (Kit starts once per
        process) and replaces only the scene.
        """
        self.require_available()
        _require_scene_assets(task_config)
        if self._env is not None:
            # Clear the stage, not just the scene. The prims of the scene being
            # replaced otherwise survive into the next one, and the first thing
            # that collides -- the dome light, at a fixed path -- fails the run
            # with "a prim already exists at path", two scenes in.
            self.close(clear_stage=True, log=log)
        self._app.start(log=log)
        require_runtime_symbols(self._app, log=log)
        settings = CameraWalkSettings.from_task_config(task_config)
        env = CameraWalkEnv(
            settings, task_config=task_config, scene_extras=scene_extras, app=self._app
        )
        try:
            env.initialize(log=log)
        except BaseException:
            # Release whatever initialize managed to build (a SimulationContext,
            # a stage) before the failure propagates. The application itself is
            # left alone: closing Kit here would hard-exit this process with
            # status 0 and turn this failure into a silent success. Process exit
            # reclaims it (see insight_bench.simulator.isaac.app).
            try:
                env.close(log=log)
            except Exception as cleanup_error:
                if log is not None:
                    log(
                        "camera backend open_scene: cleanup after a failed initialize also "
                        f"failed ({type(cleanup_error).__name__}: {cleanup_error})"
                    )
            raise
        self._env = env
        self._task_config = task_config
        self._frame_id = 0

    def close(self, *, clear_stage: bool = False, log: LogCallback | None = None) -> None:
        """Tear the scene down. Idempotent; the application deliberately stays up.

        ``clear_stage`` empties the USD stage as well, which is required when
        another scene is about to be opened in this process and pointless at the
        end of a run.

        Scene teardown is what a run needs between scenes and at the end of one.
        The Kit application is *not* closed here: on a real Isaac build closing
        it hard-exits the process with status 0, and this method is called from a
        ``finally`` with a run result still to build. Releasing it is a top-level
        owner's decision, taken only on a successful exit
        (:func:`~insight_bench.simulator.isaac.app.release_isaac_app_on_success`);
        otherwise process exit reclaims it.
        """
        if self._env is not None:
            self._env.close(clear_stage=clear_stage, log=log)
        self._env = None
        self._task_config = None

    def shutdown(self, *, log: LogCallback | None = None) -> None:
        """Close the scene, then release the Kit application. Ordered, idempotent.

        The explicit full teardown, for a caller that owns the process and is
        finished with it: the scene first (the stage and camera it built), the
        application second. Safe to call twice, and safe with no scene open.

        **Expect this to terminate the process.** On a real Isaac build
        ``SimulationApp.close()`` hard-exits with status 0 during Omniverse
        shutdown, so nothing after this call is guaranteed to run -- write out
        whatever you need first. That is why it is a separate method:
        :meth:`close` is the one to call from a ``finally`` with a run result
        still to build, and the default is to release nothing at all and let
        process exit reclaim Kit. A caller that is about to exit and knows the
        status it is exiting with wants
        :func:`~insight_bench.simulator.isaac.app.release_isaac_app_on_success`
        instead: this method closes unconditionally, so on a failing run it
        would publish the failure as a status-0 success (see
        :mod:`insight_bench.simulator.isaac.app`).
        """
        self.close(log=log)
        self._app.close(log=log)

    # -- execution ---------------------------------------------------------

    def run_episode(
        self, episode: Any, policy_url: str, *, frame_sink: FrameSink | None = None
    ) -> dict[str, Any]:
        """Execute one episode against an HTTP policy server and return its outputs.

        The in-process path (an adapter the caller already holds) is driven by
        :meth:`run_episode_with_policy` instead; this signature exists for the
        hosted case, where the policy lives in another process.
        """
        from insight_bench.vln_runtime.policy.client import LocalVlnPolicyClient

        client = LocalVlnPolicyClient(policy_url)
        try:
            rollout = self.run_episode_with_policy(episode, client, frame_sink=frame_sink)
        finally:
            client.close()
        return {
            "episode_id": rollout.episode_id,
            "termination": rollout.termination,
            "termination_reason": rollout.termination_reason,
            "stop_step": rollout.stop_step,
            "steps": rollout.steps_taken,
            "final_measures": rollout.final_measures,
            "trace": rollout.trace.to_dict(),
        }

    def run_episode_with_policy(
        self,
        episode: EpisodeSpec,
        policy: Any,
        *,
        options: RolloutOptions | None = None,
        log: LogCallback | None = None,
        frame_sink: FrameSink | None = None,
    ) -> EpisodeRollout:
        """Run one episode against an already-constructed policy driver.

        *frame_sink* is passed straight through to the rollout, which owns what
        a frame is worth persisting and when; this backend only renders.
        """
        _env, task_config = self._require_scene()
        resolved = options or RolloutOptions.from_task_config(task_config)
        # Pass ``self``, not the raw env: the delegating surface below is what
        # keeps the backend's frame counter in step with what was rendered, so
        # a capture_observation() taken mid-episode is labelled correctly.
        return run_camera_walk_episode(
            env=self,
            policy=policy,
            episode=episode,
            task_config=task_config,
            options=resolved,
            log=log,
            frame_sink=frame_sink,
        )

    # -- CameraWalkBackend surface (delegated, so the backend is usable directly)

    def reset(
        self,
        pose: CameraPose,
        *,
        warmup_steps: int,
        converge_max_steps: int = 0,
        converge_tol: float = 0.0,
        log: LogCallback | None = None,
    ) -> Any:
        env, _ = self._require_scene()
        self._frame_id = 0
        return env.reset(
            pose,
            warmup_steps=warmup_steps,
            converge_max_steps=converge_max_steps,
            converge_tol=converge_tol,
            log=log,
        )

    def resolve_start_pose(self, pose: CameraPose) -> CameraPose:
        env, _ = self._require_scene()
        return env.resolve_start_pose(pose)

    def resolve_motion(self, prev: CameraPose, candidate: CameraPose) -> CameraPose:
        env, _ = self._require_scene()
        return env.resolve_motion(prev, candidate)

    def set_pose(self, pose: CameraPose) -> None:
        env, _ = self._require_scene()
        env.set_pose(pose)

    def step_and_capture(self, *, settle_steps: int = 1) -> Any:
        env, _ = self._require_scene()
        self._frame_id += 1
        return env.step_and_capture(settle_steps=settle_steps)

    def last_terrain_query(self) -> dict[str, bool]:
        """Forward the env's report of what the last ``resolve_motion`` did to the geometry."""
        env, _ = self._require_scene()
        return env.last_terrain_query()

    def step_motion(
        self,
        prev: CameraPose,
        candidate: CameraPose,
        *,
        settle_steps: int = 1,
    ) -> dict[str, Any]:
        """Resolve, command and settle one motion, then read the pose back. One call.

        The three-call sequence (resolve_motion -> set_pose -> step_and_capture) is what the
        rollout does per step, and a caller that only wants to prove the camera moves should not
        have to reproduce it -- a probe that stitches those calls itself is testing its own
        stitching as much as the backend. This returns the readback pose rather than the commanded
        one, because the question a motion probe asks is whether the simulator moved, not whether
        we asked it to.
        """
        resolved = self.resolve_motion(prev, candidate)
        terrain = self.last_terrain_query()
        self.set_pose(resolved)
        self.step_and_capture(settle_steps=settle_steps)
        readback_pose, readback_orientation = self.read_camera_pose()
        return {
            "commanded": candidate,
            "resolved": resolved,
            "readback": readback_pose,
            "orientation_wxyz": readback_orientation,
            **terrain,
        }

    # -- canonical observation --------------------------------------------

    def capture_observation(self) -> Observation:
        """Return the current camera observation in the canonical numpy contract."""
        env, _ = self._require_scene()
        pose, orientation = env.read_camera_pose()
        return make_observation(
            env.capture_rgb(),
            pose=pose,
            orientation_wxyz=orientation,
            depth=env.capture_depth(),
            intrinsics=env.capture_intrinsics(),
            frame_id=self._frame_id,
        )

    def read_camera_pose(self) -> tuple[CameraPose, Quat]:
        """Read back the current pose and canonical WXYZ orientation."""
        env, _ = self._require_scene()
        return env.read_camera_pose()

    # -- internals ---------------------------------------------------------

    def _require_scene(self) -> tuple[CameraWalkEnv, BenchmarkTaskConfig]:
        if self._env is None or self._task_config is None:
            raise SimulatorNotAvailableError(
                f"{self._descriptor.backend_id}: no scene is open; call open_scene() first"
            )
        return self._env, self._task_config


def _require_scene_assets(task_config: BenchmarkTaskConfig) -> None:
    """Fail closed when a USD scene the task needs is not present on disk."""
    terrain = task_config.terrain
    if terrain.kind != "usd":
        return
    if not terrain.usd_path:
        raise SimulatorNotAvailableError(
            f"task {task_config.name!r} declares a USD terrain but no usd_path is "
            "configured; scene assets are user-provided and EULA-gated, and this SDK "
            "never downloads them"
        )
    if not Path(terrain.usd_path).exists():
        raise SimulatorNotAvailableError(
            f"task {task_config.name!r}: scene asset does not exist at the configured path"
        )
