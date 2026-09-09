"""The Omniverse Kit application lifecycle, proven without a GPU.

A GPU run of this backend failed with ``No module named 'pxr'`` on a machine
with a working Isaac install: nothing here started the Kit application, and
``pxr``/``omni``/``isaaclab.sim`` exist only after ``isaaclab.app.AppLauncher``
has. The static G1 probe passed anyway, because it only asks whether the
``isaacsim``/``isaaclab``/``warp`` *packages* are visible.

These tests pin the fix at the level CI can actually check: launch ordering,
one launch per process, no restart, no retry after a failure, cleanup when the
scene fails to build, and a close that can be called twice. The single seam
substituted is the callable that would construct ``AppLauncher``; everything
above it is the production path.

They also pin the *second* bug found here: the release used to be an ``atexit``
hook, and interpreter-exit hooks run on the way out of a failed run too, so the
hook turned every non-zero exit into exit 0. Nothing may arm an automatic close
now, and the only sanctioned release refuses at a non-zero status. The end-to-end
proof -- a simulated hard-exiting ``close`` under the real CLI -- lives in
``tests/gate/test_cli_exit_status.py``; pinned here are the lifecycle-level
facts: no exit hook is registered, no module in the package registers one, and
the release guard holds its nose at any non-zero code.

Nothing here claims G1 or G2 passes -- that needs a GPU. It claims the ordering
these gates depend on is now a property of the code.
"""

from __future__ import annotations

import ast
import atexit
import importlib.util
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import pytest

from insight_bench.simulator.base import SimulatorNotAvailableError
from insight_bench.simulator.isaac import app as app_module
from insight_bench.simulator.isaac import backend as backend_module
from insight_bench.simulator.isaac.app import (
    ISAAC_RUNTIME_IMPORT_SURFACE,
    IsaacAppLifecycle,
    require_isaac_app,
    require_runtime_symbols,
    runtime_import_smoke,
    runtime_symbol_probe,
)
from insight_bench.simulator.isaac.backend import ISAAC_IMPORT_SURFACE, IsaacCameraWalkBackend
from insight_bench.simulator.isaac.camera_walk import CameraWalkEnv, CameraWalkSettings
from insight_bench.vln_runtime.suite import get_task_config

ROOT = Path(__file__).resolve().parents[2]


# --- stand-ins --------------------------------------------------------------


class FakeApp:
    """Stands in for the Omniverse application ``AppLauncher`` hands back."""

    def __init__(self, *, events: list[str] | None = None, close_error: Exception | None = None):
        self.closes = 0
        self._events = events
        self._close_error = close_error

    def close(self) -> None:
        self.closes += 1
        if self._events is not None:
            self._events.append("app.close")
        if self._close_error is not None:
            raise self._close_error


class FakeLauncher:
    """The one seam a GPU-free lifecycle test needs: what ``AppLauncher`` would do."""

    def __init__(
        self,
        *,
        events: list[str] | None = None,
        error: Exception | None = None,
        returns_none: bool = False,
        close_error: Exception | None = None,
    ):
        self.calls: list[dict[str, object]] = []
        self.apps: list[FakeApp] = []
        self._events = events
        self._error = error
        self._returns_none = returns_none
        self._close_error = close_error

    def __call__(self, args: Mapping[str, Any]) -> FakeApp | None:
        self.calls.append(dict(args))
        if self._events is not None:
            self._events.append("app.start")
        if self._error is not None:
            raise self._error
        if self._returns_none:
            return None
        app = FakeApp(events=self._events, close_error=self._close_error)
        self.apps.append(app)
        return app


def _lifecycle(**kwargs) -> tuple[IsaacAppLifecycle, FakeLauncher]:
    launcher = FakeLauncher(**kwargs)
    return IsaacAppLifecycle(launch=launcher), launcher


# --- starting ---------------------------------------------------------------


def test_a_fresh_lifecycle_has_started_nothing() -> None:
    lifecycle, launcher = _lifecycle()
    assert (lifecycle.state, lifecycle.running) == ("idle", False)
    assert launcher.calls == []
    with pytest.raises(SimulatorNotAvailableError, match="is not running"):
        _ = lifecycle.app


def test_the_headless_camera_enabled_args_are_what_kit_is_started_with() -> None:
    lifecycle, launcher = _lifecycle()
    lifecycle.start()
    # headless because an evaluation runtime has no display; enable_cameras
    # because the Camera sensor's render pipeline is otherwise off, and every
    # observation this backend produces comes from that camera.
    assert launcher.calls == [{"headless": True, "enable_cameras": True}]


def test_kit_is_launched_once_and_the_second_caller_is_told_it_did_not_start_it() -> None:
    lifecycle, launcher = _lifecycle()
    logs: list[str] = []
    assert lifecycle.start(log=logs.append) is True
    assert lifecycle.start(log=logs.append) is False
    assert len(launcher.calls) == 1
    assert lifecycle.running is True
    assert lifecycle.app is launcher.apps[0]
    assert any("already running" in line for line in logs)


def test_starting_arms_no_automatic_release() -> None:
    """The regression: ``start`` used to register an ``atexit`` close.

    That hook fired on the way out of *failed* runs too, and closing Kit
    hard-exits with status 0, so a run that returned 1 or died on a traceback
    exited 0. Starting must now arm nothing at all.
    """
    before = list(getattr(atexit, "_exithandlers", []))
    lifecycle, _launcher = _lifecycle()
    lifecycle.start()
    lifecycle.start()
    assert lifecycle.running is True
    assert list(getattr(atexit, "_exithandlers", [])) == before
    assert not hasattr(app_module, "atexit"), (
        "app.py imports atexit again; the only safe release is a top-level, "
        "exit-code-checked one (release_isaac_app_on_success)"
    )


# --- releasing --------------------------------------------------------------


def test_close_is_idempotent_and_only_the_first_call_closes_kit() -> None:
    lifecycle, launcher = _lifecycle()
    lifecycle.start()
    assert lifecycle.close() is True
    assert lifecycle.close() is False
    assert lifecycle.close() is False
    assert launcher.apps[0].closes == 1
    assert lifecycle.state == "closed"


def test_closing_a_lifecycle_that_never_started_is_a_no_op() -> None:
    lifecycle, launcher = _lifecycle()
    assert lifecycle.close() is False
    assert lifecycle.state == "idle"
    assert launcher.apps == []


def test_no_automatic_release_path_survives_on_the_lifecycle() -> None:
    """No attribute anyone can hand to ``atexit.register`` is left behind."""
    lifecycle, _launcher = _lifecycle()
    assert not hasattr(lifecycle, "_release_at_exit")
    assert not hasattr(lifecycle, "_register_release_hook")


def test_a_close_that_raises_is_reported_and_does_not_propagate() -> None:
    lifecycle, _launcher = _lifecycle(close_error=RuntimeError("kit shutdown blew up"))
    lifecycle.start()
    logs: list[str] = []
    assert lifecycle.close(log=logs.append) is True
    assert lifecycle.state == "closed"
    assert any("close failed" in line and "kit shutdown blew up" in line for line in logs)


def test_the_hard_exit_is_documented_on_every_call_that_can_trigger_it() -> None:
    """The one thing that must not be hidden.

    Five public surfaces can close Kit, and each has to say both halves of what
    that does: the process ends, and it ends with status 0. Either half alone
    leaves a caller able to reach for one of these from a failure path and
    publish the failure as a success.
    """
    from insight_bench.simulator.isaac.app import close_isaac_app

    ends_the_process = (
        "hard-exit",
        "exits the process",
        "terminates the process",
        "terminate the process",
        "does not return",
        "will not return",
    )
    for surface in (
        runtime_import_smoke,
        close_isaac_app,
        app_module.release_isaac_app_on_success,
        IsaacAppLifecycle.close,
        IsaacCameraWalkBackend.shutdown,
    ):
        doc = surface.__doc__ or ""
        assert any(phrase in doc for phrase in ends_the_process), (
            f"{surface.__qualname__} can close Kit without saying the process ends"
        )
        assert "status 0" in doc, (
            f"{surface.__qualname__} says the process ends but not that it ends at status 0"
        )


def test_the_release_guard_refuses_every_non_zero_exit_status(monkeypatch) -> None:
    """``release_isaac_app_on_success`` is the only sanctioned early release.

    It exists because ``SimulationApp.close()`` hard-exits with status 0: closing
    at exit code 1 or 2 would publish a failed run as a success. So it closes at
    0 and nowhere else, and always returns the code it was given.
    """
    lifecycle, launcher = _lifecycle()
    monkeypatch.setattr(app_module, "_LIFECYCLE", lifecycle)
    lifecycle.start()

    for code in (1, 2, -1, 130):
        assert app_module.release_isaac_app_on_success(code) == code
    assert launcher.apps[0].closes == 0, "Kit was closed at a failing exit status"
    assert lifecycle.running is True

    logs: list[str] = []
    assert app_module.release_isaac_app_on_success(0, log=logs.append) == 0
    assert launcher.apps[0].closes == 1
    assert lifecycle.state == "closed"
    # Idempotent: a second finalize on a spent lifecycle closes nothing again.
    assert app_module.release_isaac_app_on_success(0) == 0
    assert launcher.apps[0].closes == 1


def test_the_release_guard_is_a_no_op_when_kit_never_started(monkeypatch) -> None:
    lifecycle, launcher = _lifecycle()
    monkeypatch.setattr(app_module, "_LIFECYCLE", lifecycle)
    assert app_module.release_isaac_app_on_success(0) == 0
    assert launcher.calls == [] and lifecycle.state == "idle"


def test_no_module_in_the_package_registers_an_interpreter_exit_hook() -> None:
    """Package-wide gate: nothing may schedule work for interpreter shutdown.

    ``atexit`` handlers run on the way out of a *failed* run, and the only thing
    this package would ever want to do there -- close Kit -- hard-exits with
    status 0. Verified with the standard library alone::

        python -c "import atexit,os,sys; atexit.register(lambda: os._exit(0)); sys.exit(2)"
        # exits 0

    A future hook that does not hard-exit would still be a hook this gate wants
    to see reviewed, which is why the rule is "none", not "none that exit".
    """
    offenders: list[str] = []
    for path in sorted((ROOT / "src" / "insight_bench").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            target = node.func
            name = (
                f"{target.value.id}.{target.attr}"
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                else target.id
                if isinstance(target, ast.Name)
                else ""
            )
            if name in {"atexit.register", "register_atexit"}:
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}: {name}")
    assert not offenders, f"interpreter-exit hooks registered in the package: {offenders}"


def test_kit_is_never_restarted_in_a_process_that_already_shut_it_down() -> None:
    lifecycle, launcher = _lifecycle()
    lifecycle.start()
    lifecycle.close()
    with pytest.raises(SimulatorNotAvailableError, match="cannot be restarted in-process"):
        lifecycle.start()
    assert len(launcher.calls) == 1


# --- refusing ---------------------------------------------------------------


def test_a_failed_launch_is_remembered_rather_than_retried() -> None:
    lifecycle, launcher = _lifecycle(error=RuntimeError("no display and no GPU"))
    with pytest.raises(SimulatorNotAvailableError, match="failed to start"):
        lifecycle.start()
    assert lifecycle.state == "failed"
    with pytest.raises(SimulatorNotAvailableError, match="is not retried"):
        lifecycle.start()
    # A half-started Kit is not a starting point: one attempt, not two.
    assert len(launcher.calls) == 1


def test_a_refusal_from_the_launcher_keeps_its_own_message() -> None:
    lifecycle, _launcher = _lifecycle(
        error=SimulatorNotAvailableError("isaaclab.app.AppLauncher is not importable")
    )
    with pytest.raises(SimulatorNotAvailableError, match="AppLauncher is not importable"):
        lifecycle.start()


def test_a_launcher_that_returns_no_application_is_a_failure_not_a_running_app() -> None:
    lifecycle, _launcher = _lifecycle(returns_none=True)
    with pytest.raises(SimulatorNotAvailableError, match="returned no application object"):
        lifecycle.start()
    assert lifecycle.running is False


def test_kit_is_not_launched_twice_even_from_a_second_lifecycle(monkeypatch) -> None:
    # The interlock the fake launcher cannot exercise: two IsaacAppLifecycle
    # objects (a caller's own alongside the process-global one) must not both
    # bring Kit up. An attempt counts, not just a success -- a launch that raised
    # may still have left Kit half-up.
    monkeypatch.setattr(app_module, "_KIT_LAUNCH_ATTEMPTED", True)
    with pytest.raises(SimulatorNotAvailableError, match="already been launched in this process"):
        IsaacAppLifecycle().start()


def test_the_real_launch_site_is_the_default_and_refuses_without_isaac() -> None:
    # No launcher injected: the production path, which on this machine gets as
    # far as discovering there is no Isaac to launch.
    if importlib.util.find_spec("isaaclab") is not None:
        # Never call the real launcher on a machine that would honour it.
        pytest.skip("this machine has Isaac Lab installed")
    with pytest.raises(SimulatorNotAvailableError, match="AppLauncher is not importable"):
        app_module._launch_via_app_launcher(app_module.APP_LAUNCH_ARGS)


def test_require_isaac_app_refuses_without_starting_anything() -> None:
    lifecycle, launcher = _lifecycle()
    with pytest.raises(SimulatorNotAvailableError) as excinfo:
        require_isaac_app(lifecycle)
    message = str(excinfo.value)
    assert "AppLauncher" in message
    for name in ISAAC_RUNTIME_IMPORT_SURFACE:
        assert name in message
    assert launcher.calls == [], "the guard started Kit implicitly"

    lifecycle.start()
    assert require_isaac_app(lifecycle) is lifecycle


# --- the two G1 tiers -------------------------------------------------------


def test_the_two_g1_tiers_are_distinct_surfaces() -> None:
    assert ISAAC_IMPORT_SURFACE == ("isaacsim", "isaaclab", "warp")
    assert ISAAC_RUNTIME_IMPORT_SURFACE == ("pxr", "omni.usd", "carb.settings", "isaaclab.sim")
    assert not set(ISAAC_IMPORT_SURFACE) & set(ISAAC_RUNTIME_IMPORT_SURFACE)


def test_every_runtime_tier_name_is_one_this_backend_actually_imports() -> None:
    """The surface is grounded in the code, not in a plausible-looking list.

    A probe that checks a module nothing imports would fail a runtime this
    backend could have used; one that misses a module the backend does import
    passes a runtime the backend cannot use as intended. `carb.settings` was in
    the second category until it was added here -- its absence does not crash
    `apply_sync_load_settings`, it silently costs the sync-load determinism the
    captured frames depend on, which is exactly the kind of pass a gate is
    supposed to prevent.
    """
    sources = "".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "src" / "insight_bench" / "simulator" / "isaac").rglob("*.py"))
    )
    for name in ISAAC_RUNTIME_IMPORT_SURFACE:
        root = name.split(".")[0]
        assert f"import {name}" in sources or f"from {root} import" in sources, (
            f"{name} is in the G1 runtime surface but nothing in the backend imports it"
        )


def test_the_runtime_tier_is_unknown_until_the_app_starts() -> None:
    lifecycle, _launcher = _lifecycle()
    probe = runtime_symbol_probe(lifecycle)
    # Unknown, and reported as such: an empty `missing` on an unchecked tier is
    # the difference between "not started yet" and "installed wrong".
    assert (probe.checked, probe.available, probe.missing) == (False, False, ())
    assert probe.app_state == "idle"
    assert "AppLauncher" in probe.reason
    # And asking must not have imported the thing it is asking about.
    assert "pxr" not in sys.modules


def test_the_runtime_tier_reports_missing_symbols_once_the_app_is_running() -> None:
    lifecycle, _launcher = _lifecycle()
    lifecycle.start()
    probe = runtime_symbol_probe(lifecycle, names=("json", "insight_bench.contracts"))
    assert (probe.checked, probe.available, probe.missing) == (True, True, ())

    absent = ("json", "a_module_no_isaac_runtime_ships")
    broken = runtime_symbol_probe(lifecycle, names=absent)
    assert (broken.checked, broken.available) == (True, False)
    assert broken.missing == ("a_module_no_isaac_runtime_ships",)
    assert "could not be imported" in broken.reason


def test_requiring_the_runtime_tier_names_the_missing_modules() -> None:
    lifecycle, _launcher = _lifecycle()
    lifecycle.start()

    # On a laptop the real surface is absent, which is the failure this replaces:
    # one precise refusal instead of a bare ModuleNotFoundError from the stage
    # loader.
    if any(name in sys.modules for name in ISAAC_RUNTIME_IMPORT_SURFACE):
        pytest.skip("this machine has an Omniverse runtime loaded")
    with pytest.raises(SimulatorNotAvailableError) as excinfo:
        require_runtime_symbols(lifecycle)
    assert "could not be imported" in str(excinfo.value)


def test_the_g1_smoke_starts_the_application_then_imports() -> None:
    # The gate entry point: one call, start included, because that start is what
    # makes these modules exist at all.
    events: list[str] = []
    lifecycle, _launcher = _lifecycle(events=events)
    probe = runtime_import_smoke(lifecycle, names=("json", "insight_bench.contracts"))
    assert events == ["app.start"]
    assert (probe.checked, probe.available, probe.missing) == (True, True, ())
    assert lifecycle.running is True


def test_the_g1_smoke_reuses_an_application_that_is_already_up() -> None:
    events: list[str] = []
    lifecycle, launcher = _lifecycle(events=events)
    lifecycle.start()
    runtime_import_smoke(lifecycle, names=("json",))
    assert len(launcher.calls) == 1


def test_the_g1_smoke_returns_a_verdict_instead_of_raising_on_a_missing_module() -> None:
    lifecycle, _launcher = _lifecycle()
    probe = runtime_import_smoke(lifecycle, names=("json", "a_module_no_runtime_ships"))
    assert (probe.checked, probe.available) == (True, False)
    assert probe.missing == ("a_module_no_runtime_ships",)


def test_the_g1_smoke_reports_an_application_that_could_not_start() -> None:
    # A gate wants something it can record, not a traceback. `checked` is what
    # separates "never came up" from "came up and a module was missing".
    lifecycle, _launcher = _lifecycle(error=RuntimeError("no display and no GPU"))
    probe = runtime_import_smoke(lifecycle, names=("json",))
    assert (probe.checked, probe.available) == (False, False)
    assert probe.app_state == "failed"
    assert "could not be checked" in probe.reason
    assert "no display and no GPU" in probe.reason


def test_the_g1_smoke_releases_nothing_by_default() -> None:
    # A gate that never returns cannot record its verdict, and there is no exit
    # hook to release Kit either: process exit reclaims it.
    events: list[str] = []
    lifecycle, _launcher = _lifecycle(events=events)
    runtime_import_smoke(lifecycle, names=("json",))
    assert "app.close" not in events
    assert lifecycle.running is True


def test_the_g1_smoke_can_be_asked_to_release_and_then_says_so() -> None:
    # release=True closes in a finally. On a real build that hard-exits the
    # process with status 0, which is why it is opt-in and documented rather
    # than the default -- a gate that never returns cannot record its verdict,
    # and a hard exit to 0 from a failing gate would publish it as a pass.
    events: list[str] = []
    lifecycle, _launcher = _lifecycle(events=events)
    probe = runtime_import_smoke(lifecycle, names=("json",), release=True)
    assert events == ["app.start", "app.close"]
    assert probe.available is True
    assert lifecycle.state == "closed"


def test_the_backend_probe_separates_the_static_tier_from_the_runtime_tier() -> None:
    lifecycle, _launcher = _lifecycle()
    probe = IsaacCameraWalkBackend(app=lifecycle).probe()
    assert probe["import_surface"] == list(ISAAC_IMPORT_SURFACE)
    assert probe["runtime_import_surface"] == list(ISAAC_RUNTIME_IMPORT_SURFACE)
    assert probe["app"]["state"] == "idle"
    assert probe["runtime_symbols"]["checked"] is False
    # `available` stays keyed on the static tier and the platform: the runtime
    # tier cannot be known before open_scene() starts the application.
    assert probe["available"] is (not probe["reasons"])


# --- the backend's ordering -------------------------------------------------


def _fake_env_class(
    events: list[str],
    *,
    initialize_error: Exception | None = None,
    close_error: Exception | None = None,
) -> type:
    class _FakeEnv:
        instances: ClassVar[list[_FakeEnv]] = []

        def __init__(
            self,
            settings: CameraWalkSettings,
            *,
            task_config: object = None,
            scene_extras: object = None,
            app: IsaacAppLifecycle | None = None,
        ) -> None:
            self.settings = settings
            self.task_config = task_config
            self.app = app
            self.closes = 0
            _FakeEnv.instances.append(self)

        def initialize(self, log: object = None) -> None:
            events.append("env.initialize")
            # Asserted from the inside: whatever the caller believes about
            # ordering, the env only ever runs with a live application.
            if self.app is None or not self.app.running:
                raise AssertionError("env.initialize ran before the application started")
            if initialize_error is not None:
                raise initialize_error

        def close(self, *, clear_stage: bool = False, log: object = None) -> None:
            self.closes += 1
            events.append("env.close")
            if close_error is not None:
                raise close_error

    return _FakeEnv


@pytest.fixture
def isaac_looks_installed(monkeypatch):
    """Satisfy the static tier and the platform check on a laptop."""
    monkeypatch.setattr(backend_module, "isaac_import_probe", lambda: (True, ()))
    monkeypatch.setattr(backend_module, "_observed_os", lambda: "linux")


@pytest.fixture
def scene_task():
    # Plane terrain, so open_scene() reaches the lifecycle rather than stopping
    # at the asset gate. The shipped suite is USD-backed and its scenes are
    # user-provided, so the terrain is swapped for the one kind that names no
    # file on disk; nothing below this line looks at the terrain again.
    task = get_task_config("insight_bench")
    return replace(task, terrain=replace(task.terrain, kind="plane", usd_path=None))


def _wire_backend(
    monkeypatch, events: list[str], **env_kwargs
) -> tuple[IsaacCameraWalkBackend, FakeLauncher, type]:
    lifecycle, launcher = _lifecycle(events=events)
    env_cls = _fake_env_class(events, **env_kwargs)
    monkeypatch.setattr(backend_module, "CameraWalkEnv", env_cls)
    monkeypatch.setattr(
        backend_module,
        "require_runtime_symbols",
        lambda app, log=None: events.append("runtime.check"),
    )
    return IsaacCameraWalkBackend(app=lifecycle), launcher, env_cls


def test_open_scene_starts_kit_then_proves_the_symbols_then_builds_the_scene(
    monkeypatch, isaac_looks_installed, scene_task
) -> None:
    events: list[str] = []
    backend, launcher, _env_cls = _wire_backend(monkeypatch, events)
    backend.open_scene(scene_task)
    # The whole point of the fix, in one assertion.
    assert events == ["app.start", "runtime.check", "env.initialize"]
    assert len(launcher.calls) == 1


def test_two_scenes_in_one_process_share_the_one_application(
    monkeypatch, isaac_looks_installed, scene_task
) -> None:
    events: list[str] = []
    backend, launcher, env_cls = _wire_backend(monkeypatch, events)
    backend.open_scene(scene_task)
    backend.open_scene(scene_task)
    assert len(launcher.calls) == 1, "Kit was launched twice in one process"
    assert events == [
        "app.start",
        "runtime.check",
        "env.initialize",
        # The second open replaces the scene and reuses the application.
        "env.close",
        "runtime.check",
        "env.initialize",
    ]
    assert env_cls.instances[0].closes == 1


def test_a_scene_that_fails_to_build_is_torn_down_and_leaves_no_scene_behind(
    monkeypatch, isaac_looks_installed, scene_task
) -> None:
    events: list[str] = []
    backend, launcher, env_cls = _wire_backend(
        monkeypatch, events, initialize_error=RuntimeError("stage load failed")
    )
    with pytest.raises(RuntimeError, match="stage load failed"):
        backend.open_scene(scene_task)
    assert events == ["app.start", "runtime.check", "env.initialize", "env.close"]
    assert env_cls.instances[0].closes == 1
    # No half-open scene is left claimable.
    with pytest.raises(SimulatorNotAvailableError, match="no scene is open"):
        backend.read_camera_pose()
    # The application stays up on purpose: closing Kit here would hard-exit this
    # process with status 0 and turn this failure into a silent success. Nothing
    # releases it automatically; process exit reclaims it.
    assert backend._app.running is True
    assert len(launcher.calls) == 1


def test_a_cleanup_failure_does_not_replace_the_error_that_caused_it(
    monkeypatch, isaac_looks_installed, scene_task
) -> None:
    events: list[str] = []
    backend, _launcher, _env_cls = _wire_backend(
        monkeypatch,
        events,
        initialize_error=RuntimeError("stage load failed"),
        close_error=RuntimeError("teardown also failed"),
    )
    logs: list[str] = []
    with pytest.raises(RuntimeError, match="stage load failed"):
        backend.open_scene(scene_task, log=logs.append)
    assert any("cleanup after a failed initialize" in line for line in logs)


def test_a_retry_after_a_failed_scene_reuses_the_running_application(
    monkeypatch, isaac_looks_installed, scene_task
) -> None:
    events: list[str] = []
    lifecycle, launcher = _lifecycle(events=events)
    monkeypatch.setattr(
        backend_module,
        "require_runtime_symbols",
        lambda app, log=None: events.append("runtime.check"),
    )
    backend = IsaacCameraWalkBackend(app=lifecycle)

    monkeypatch.setattr(
        backend_module,
        "CameraWalkEnv",
        _fake_env_class(events, initialize_error=RuntimeError("stage load failed")),
    )
    with pytest.raises(RuntimeError):
        backend.open_scene(scene_task)

    monkeypatch.setattr(backend_module, "CameraWalkEnv", _fake_env_class(events))
    backend.open_scene(scene_task)
    assert len(launcher.calls) == 1


def test_closing_the_backend_releases_the_scene_and_keeps_the_application(
    monkeypatch, isaac_looks_installed, scene_task
) -> None:
    events: list[str] = []
    backend, launcher, env_cls = _wire_backend(monkeypatch, events)
    backend.open_scene(scene_task)
    backend.close()
    backend.close()
    backend.close()
    # Torn down once however often close() is called -- it is called from a
    # finally, and a second teardown of a released stage is not a no-op in Kit.
    assert env_cls.instances[0].closes == 1
    assert events.count("env.close") == 1
    # Still up: the run result is built after close() returns, and closing Kit
    # would hard-exit this process first.
    assert backend._app.running is True
    assert launcher.apps[0].closes == 0


def test_a_backend_with_no_scene_can_be_closed_without_touching_the_application(
    monkeypatch, isaac_looks_installed
) -> None:
    events: list[str] = []
    backend, launcher, _env_cls = _wire_backend(monkeypatch, events)
    backend.close()
    assert events == []
    assert launcher.calls == []


def test_shutdown_closes_the_scene_before_the_application_and_repeats_safely(
    monkeypatch, isaac_looks_installed, scene_task
) -> None:
    events: list[str] = []
    backend, launcher, env_cls = _wire_backend(monkeypatch, events)
    backend.open_scene(scene_task)
    backend.shutdown()
    backend.shutdown()
    assert events == [
        "app.start",
        "runtime.check",
        "env.initialize",
        # Scene first, application second, each exactly once.
        "env.close",
        "app.close",
    ]
    assert env_cls.instances[0].closes == 1
    assert launcher.apps[0].closes == 1
    assert backend._app.state == "closed"


def test_shutdown_with_no_scene_open_still_releases_nothing_it_did_not_start(
    monkeypatch, isaac_looks_installed
) -> None:
    events: list[str] = []
    backend, launcher, _env_cls = _wire_backend(monkeypatch, events)
    backend.shutdown()
    assert events == []
    assert launcher.calls == []


def test_the_default_backend_uses_the_process_global_application() -> None:
    from insight_bench.simulator.isaac.app import isaac_app

    assert IsaacCameraWalkBackend()._app is isaac_app()


# --- the env's own guard ----------------------------------------------------


def test_the_env_refuses_to_initialize_before_the_application_starts(scene_task) -> None:
    lifecycle, launcher = _lifecycle()
    env = CameraWalkEnv(
        CameraWalkSettings.from_task_config(scene_task),
        task_config=scene_task,
        app=lifecycle,
    )
    # The regression this guards: without it, the first line of initialize()
    # imports isaaclab.sim and the caller gets ModuleNotFoundError from a
    # machine whose Isaac install is fine.
    with pytest.raises(SimulatorNotAvailableError) as excinfo:
        env.initialize()
    assert "AppLauncher" in str(excinfo.value)
    assert launcher.calls == [], "the env started Kit behind the caller's back"


# --- import structure (no GPU, no Isaac) ------------------------------------


def test_no_sdk_module_imports_the_isaac_runtime_at_module_scope() -> None:
    """Static twin of the subprocess sys.modules gate.

    That gate proves the modules it imports stay Isaac-free; this one proves it
    for every module in the package, including ones no test imports yet. Both
    are needed: a lazy import that drifts to module scope is exactly how the
    published wheel would stop installing on a laptop.
    """
    # Assembled at runtime so this file stays clean for the token scanner.
    forbidden = {"isaac" + "sim", "isaac" + "lab", "warp", "omni", "carb", "pxr", "torch"}
    offenders: list[str] = []
    for path in sorted((ROOT / "src" / "insight_bench").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:  # module scope only: what runs on `import`
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                roots = [node.module.split(".")[0]]
            else:
                continue
            for root in roots:
                if root in forbidden:
                    offender = path.relative_to(ROOT)
                    offenders.append(f"{offender}:{node.lineno}: {root}")
    assert not offenders, f"simulator imports at module scope: {offenders}"
