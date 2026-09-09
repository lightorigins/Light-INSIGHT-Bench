"""The Isaac (Omniverse Kit) application lifecycle.

Isaac Sim is not a library that can be imported on demand. ``isaaclab.app.AppLauncher``
starts an Omniverse Kit application, and *that start* is what puts ``pxr``,
``omni.*``, ``carb`` and ``isaaclab.sim`` on the interpreter's import path. Code
that reaches for them earlier fails with ``No module named 'pxr'`` on a machine
with a perfectly good Isaac install -- which is what a GPU run of this backend
produced, because nothing here launched the application at all.

Four rules, enforced here rather than left to callers:

**Started once, explicitly.** Kit may be launched at most once per process, so
:meth:`IsaacAppLifecycle.start` is idempotent: the second caller gets the
running application and is told so, never a second Kit. A process-level
interlock at the real launch site refuses a second launch even when it comes
from a different lifecycle object. Nothing deeper in the stack starts one
implicitly either; :func:`require_isaac_app` refuses instead, because an
implicit launch from inside a scene load is how a process ends up with a Kit
nobody owns.

**A failed launch is not retried.** A half-started Kit is not a starting point,
so a failure is remembered and every later ``start`` refuses with the original
reason.

**Never restarted.** Kit cannot be brought back up in a process that already
shut it down. ``start`` after ``close`` therefore refuses and says to use a new
process rather than launching something that misbehaves in ways no test catches.

**Never released automatically, because the release decides the exit status.**
Omniverse's ``SimulationApp.close()`` hard-exits the process with status 0
during shutdown (observed on the 5.1 line). It therefore does not merely close
Kit: it *chooses the process's exit code*, and the code it chooses is success.
Calling it from an ``except``/``finally`` would replace a failure with a silent
success -- no traceback, no result, exit 0.

An ``atexit`` hook is not a safe home for it either, which is what this module
got wrong before. The interpreter runs its exit hooks on the way out of a
*failed* run too, so a hook that closes Kit rewrites exit 1, exit 2 and an
unhandled traceback alike into exit 0::

    python -c "import atexit,os,sys; atexit.register(lambda: os._exit(0)); sys.exit(2)"
    # exits 0

Nothing here registers such a hook. Kit is reclaimed by process exit like any
other resource the OS owns, and the only sanctioned early release is
:func:`release_isaac_app_on_success`: a top-level owner that has already
written its results calls it with the exit code it is about to exit with, and
the close happens only if that code is ``0``, where a hard exit to 0 is what
was going to happen anyway. Failure paths tear the *scene* down (see
``IsaacCameraWalkBackend.close``) and never touch the application.

:meth:`IsaacAppLifecycle.close` and :func:`close_isaac_app` stay available and
idempotent for a caller that owns the process and has decided for itself that
closing is safe. On a real Isaac build they do not return.

Every Isaac import in this module lives inside a function body, so importing it
on a machine with no simulator stays free.
"""

from __future__ import annotations

import contextlib
import importlib
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from insight_bench.simulator.base import SimulatorNotAvailableError

LogCallback = Callable[[str], None]
AppLaunch = Callable[[Mapping[str, Any]], Any]

AppState = Literal["idle", "running", "closed", "failed"]
"""``idle`` never started, ``running`` usable, ``closed`` spent, ``failed`` refused."""

ISAAC_RUNTIME_IMPORT_SURFACE: tuple[str, ...] = (
    "pxr",
    "omni.usd",
    "carb.settings",
    "isaaclab.sim",
)
"""The G1 *runtime* tier: modules that exist only once Kit is up.

A fixed, package-owned list, like
:data:`~insight_bench.simulator.isaac.backend.ISAAC_IMPORT_SURFACE` -- never
registry or manifest data. Each name is one this backend genuinely imports
after the launch, not a plausible-looking guess:

- ``pxr`` -- USD, reached through the terrain mesh collector, and the name the
  L20 run actually failed on;
- ``omni.usd`` -- the stage handle and the streaming-status manager;
- ``carb.settings`` -- the synchronous material/texture load switches applied
  before the first render. ``apply_sync_load_settings`` degrades rather than
  raising when carb is absent, so this one is a *deliberate* tightening: a
  runtime missing it would still run, silently, without the sync-load
  determinism the captured frames depend on;
- ``isaaclab.sim`` -- the simulation context and spawners.

Asking about them before the application starts would report a healthy install
as broken, so :func:`runtime_symbol_probe` answers "not checked" instead of
guessing, and nothing ever imports them ahead of ``AppLauncher``.
"""

APP_LAUNCH_ARGS: Mapping[str, Any] = MappingProxyType(
    {
        "headless": True,
        "enable_cameras": True,
    }
)
"""Kit launch arguments for the camera-walk backend.

``headless``: an evaluation runtime has no display, and the official runtime is
a container. ``enable_cameras``: the Camera sensor's render pipeline stays off
in headless Kit unless it is asked for, and every observation this backend
produces comes from that camera.
"""


def _log(log: LogCallback | None, message: str) -> None:
    if log is not None:
        log(message)


def _warn(log: LogCallback | None, message: str) -> None:
    """Report a failure that must not be swallowed silently, log callback or not."""
    if log is not None:
        log(message)
    else:
        print(message, file=sys.stderr)


_KIT_LAUNCH_ATTEMPTED = False
"""Whether this *process* has already tried to launch Kit, whichever object did.

The interlock lives at the real launch site rather than on the lifecycle, so a
caller holding its own :class:`IsaacAppLifecycle` cannot bring up a second Kit
alongside the process-global one. An attempt counts, not just a success: a
launch that raised may still have left Kit half-up, and that is not something to
launch into.
"""


def _launch_via_app_launcher(args: Mapping[str, Any]) -> Any:
    """Start Kit through ``isaaclab.app.AppLauncher`` and return the application.

    The single seam tests substitute: everything above it is lifecycle logic
    that must be provable without a GPU.
    """
    global _KIT_LAUNCH_ATTEMPTED
    if _KIT_LAUNCH_ATTEMPTED:
        raise SimulatorNotAvailableError(
            "Omniverse Kit has already been launched in this process and cannot be "
            "launched twice; use insight_bench.simulator.isaac.app.isaac_app() rather "
            "than a second IsaacAppLifecycle, or run this scene in a new process"
        )
    try:
        from isaaclab.app import AppLauncher
    except ImportError as exc:
        raise SimulatorNotAvailableError(
            "the Isaac app cannot start: isaaclab.app.AppLauncher is not importable "
            f"({exc}); this backend runs only inside an Isaac Sim runtime"
        ) from exc
    _KIT_LAUNCH_ATTEMPTED = True
    # AppLauncher takes a mapping (or an argparse namespace) of launcher args and
    # fills its own defaults for everything not named here.
    return AppLauncher(dict(args)).app


class IsaacAppLifecycle:
    """One Omniverse Kit application: started at most once, released exactly once."""

    def __init__(
        self,
        *,
        launch: AppLaunch | None = None,
        args: Mapping[str, Any] | None = None,
    ) -> None:
        self._launch: AppLaunch = _launch_via_app_launcher if launch is None else launch
        self._args: dict[str, Any] = dict(APP_LAUNCH_ARGS if args is None else args)
        self._app: Any | None = None
        self._state: AppState = "idle"
        self._failure: str = ""
        # Kit is process-global; two threads racing to start it would launch twice.
        self._lock = threading.RLock()

    # -- facts -------------------------------------------------------------

    @property
    def state(self) -> AppState:
        return self._state

    @property
    def running(self) -> bool:
        return self._state == "running"

    @property
    def args(self) -> Mapping[str, Any]:
        return MappingProxyType(self._args)

    @property
    def app(self) -> Any:
        """The Kit application, or a refusal naming the reason it is not there."""
        with self._lock:
            if self._app is None:
                raise SimulatorNotAvailableError(
                    f"the Isaac app is not running (state: {self._state}); "
                    "start it before using anything that needs Omniverse"
                )
            return self._app

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self._state,
            "running": self.running,
            "args": dict(self._args),
            "failure": self._failure,
        }

    # -- lifecycle ---------------------------------------------------------

    def start(self, *, log: LogCallback | None = None) -> bool:
        """Start Kit if it is not up. Returns whether *this* call started it.

        Idempotent, and deliberately not forgiving: a spent or failed lifecycle
        refuses rather than launching a second Kit into the same process.
        """
        with self._lock:
            if self._state == "running":
                _log(log, "isaac app: already running; reused (Kit starts once per process)")
                return False
            if self._state == "closed":
                raise SimulatorNotAvailableError(
                    "the Isaac app was already closed in this process, and Omniverse Kit "
                    "cannot be restarted in-process; run the next scene in a new process"
                )
            if self._state == "failed":
                raise SimulatorNotAvailableError(
                    "the Isaac app failed to start earlier in this process and is not "
                    f"retried: {self._failure}"
                )
            _log(log, f"isaac app: starting Kit with {self._args}")
            try:
                app = self._launch(MappingProxyType(dict(self._args)))
            except Exception as exc:
                self._fail(f"{type(exc).__name__}: {exc}", log=log)
                if isinstance(exc, SimulatorNotAvailableError):
                    raise
                raise SimulatorNotAvailableError(
                    f"the Isaac app failed to start: {type(exc).__name__}: {exc}"
                ) from exc
            if app is None:
                # Fail closed: a launcher that produced no application has not
                # started Kit, whatever it returned.
                self._fail("the launcher returned no application object", log=log)
                raise SimulatorNotAvailableError(
                    "the Isaac app failed to start: the launcher returned no application object"
                )
            self._app = app
            self._state = "running"
            _log(log, "isaac app: started")
            return True

    def close(self, *, log: LogCallback | None = None) -> bool:
        """Release Kit if it is up. Returns whether *this* call closed it.

        Idempotent. **On a real Isaac build this does not return**: it exits the
        process with status 0 (see the module docstring). Only call it with
        results already written and a successful outcome already decided --
        :func:`release_isaac_app_on_success` is that check, written down. Callers
        that still have work to do, or that got here from a failure, must not
        reach for it; leaving Kit to process exit costs nothing.
        """
        with self._lock:
            if self._state != "running":
                return False
            app: Any = self._app
            # Marked spent *before* the close call: Kit's shutdown can terminate
            # the process, and a re-entrant close (anything reached during that
            # shutdown) must not close the same application twice.
            self._app = None
            self._state = "closed"
            _log(log, "isaac app: closing Kit")
            try:
                app.close()
            except Exception as exc:
                _warn(
                    log,
                    "isaac app: close failed and was reported rather than raised "
                    f"({type(exc).__name__}: {exc}); a close in a finally must not "
                    "mask the failure that led there",
                )
                return True
            _log(log, "isaac app: closed")
            return True

    # -- internals ---------------------------------------------------------

    def _fail(self, reason: str, *, log: LogCallback | None) -> None:
        self._state = "failed"
        self._failure = reason
        self._app = None
        _log(log, f"isaac app: start failed ({reason})")


@dataclass(frozen=True)
class RuntimeSymbolProbe:
    """The G1 runtime tier's answer. ``checked`` separates unknown from missing."""

    app_state: AppState
    checked: bool
    available: bool
    missing: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "app_state": self.app_state,
            "checked": self.checked,
            "available": self.available,
            "missing": list(self.missing),
            "reason": self.reason,
        }


def _importable(name: str) -> bool:
    try:
        importlib.import_module(name)
    except Exception:
        return False
    return True


def runtime_symbol_probe(
    lifecycle: IsaacAppLifecycle | None = None,
    *,
    names: tuple[str, ...] = ISAAC_RUNTIME_IMPORT_SURFACE,
) -> RuntimeSymbolProbe:
    """Report whether the post-launch Isaac symbols are importable.

    Before the application starts these modules are absent *by construction*, so
    this reports ``checked=False`` with an empty ``missing`` rather than naming
    them. "Not started yet" and "installed wrong" are different facts, and a
    probe that collapses them either fails a healthy machine or -- what actually
    happened on the L20 run -- lets the static tier pass while this tier was
    never asked at all.

    Once the application is up the modules are imported for real: G1 is an
    import smoke test, and ``find_spec("omni.usd")`` would have to import
    ``omni`` anyway, so there is nothing to gain by pretending otherwise.
    """
    app = isaac_app() if lifecycle is None else lifecycle
    if not app.running:
        return RuntimeSymbolProbe(
            app_state=app.state,
            checked=False,
            available=False,
            missing=(),
            reason=(
                f"the Isaac app is not running (state: {app.state}), so {', '.join(names)} "
                "are unavailable by design until isaaclab.app.AppLauncher starts it; "
                "this tier is unknown, not failed"
            ),
        )
    missing = tuple(name for name in names if not _importable(name))
    if missing:
        reason = (
            f"the Isaac app is running but {', '.join(missing)} could not be imported; "
            "the Omniverse runtime is incomplete for this backend"
        )
    else:
        reason = f"the Isaac app is running and {', '.join(names)} import"
    return RuntimeSymbolProbe(
        app_state=app.state,
        checked=True,
        available=not missing,
        missing=missing,
        reason=reason,
    )


_LIFECYCLE = IsaacAppLifecycle()
"""The process-global lifecycle. Kit is process-global, so this is too."""


def isaac_app() -> IsaacAppLifecycle:
    """Return the process-global Isaac application lifecycle."""
    return _LIFECYCLE


def start_isaac_app(*, log: LogCallback | None = None) -> bool:
    """Start the process-global Kit application; True when this call started it."""
    return _LIFECYCLE.start(log=log)


def close_isaac_app(*, log: LogCallback | None = None) -> bool:
    """Release the process-global Kit application; True when this call closed it.

    On a real Isaac build this terminates the process with status 0 and does not
    return. See the module docstring, and prefer
    :func:`release_isaac_app_on_success` unless you have already established that
    a status-0 exit is the correct outcome.
    """
    return _LIFECYCLE.close(log=log)


def _flush_streams() -> None:
    """Flush stdout/stderr before a close that may never return.

    Kit's shutdown exits the process outright, which skips the flush the
    interpreter would otherwise do on the way out -- and the CLI's result JSON
    is sitting in that buffer. Streams may already be closed here, hence the
    suppression.
    """
    with contextlib.suppress(Exception):
        sys.stdout.flush()
    with contextlib.suppress(Exception):
        sys.stderr.flush()


def release_isaac_app_on_success(exit_code: int, *, log: LogCallback | None = None) -> int:
    """Release Kit for a top-level owner about to exit with *exit_code*.

    Returns *exit_code* unchanged, so the whole finalize reads
    ``raise SystemExit(release_isaac_app_on_success(main(argv)))``.

    The release happens only for a clean ``0``. ``SimulationApp.close()``
    hard-exits with status 0, so calling it at any other code -- 1, 2, or a code
    this function was never told about -- would publish a failing run as a
    success. At a non-zero code Kit is left to process exit, which reclaims it
    just as thoroughly and cannot rewrite the status.

    Only ever called with output already written: a hard exit skips the
    interpreter's own flush, so this flushes first.
    """
    if exit_code != 0:
        _log(log, f"isaac app: not released; exit status is {exit_code}, not success")
        return exit_code
    if not _LIFECYCLE.running:
        return exit_code
    _flush_streams()
    _LIFECYCLE.close(log=log)
    return exit_code


def require_isaac_app(lifecycle: IsaacAppLifecycle | None = None) -> IsaacAppLifecycle:
    """Return the running application lifecycle, or refuse naming what to do.

    The guard that keeps ordering a property of the code rather than of the
    caller's memory. It never starts anything: an implicit launch from inside a
    scene load is how a process ends up with a Kit nobody owns and nobody closes.
    """
    app = isaac_app() if lifecycle is None else lifecycle
    if not app.running:
        raise SimulatorNotAvailableError(
            f"the Isaac app is not running (state: {app.state}); isaaclab.app.AppLauncher "
            "must start the headless Kit application before anything imports "
            f"{', '.join(ISAAC_RUNTIME_IMPORT_SURFACE)}. Use "
            "IsaacCameraWalkBackend.open_scene(), which starts it, or call "
            "insight_bench.simulator.isaac.app.start_isaac_app() first"
        )
    return app


def require_runtime_symbols(
    lifecycle: IsaacAppLifecycle | None = None,
    *,
    log: LogCallback | None = None,
) -> RuntimeSymbolProbe:
    """Prove the post-launch Isaac symbols import, or refuse naming the missing ones.

    Called right after the application starts, so a runtime that cannot supply
    ``pxr`` says so once, in one place, instead of surfacing as a bare
    ``ModuleNotFoundError`` from whichever scene-loading line happened to get
    there first.
    """
    app = require_isaac_app(lifecycle)
    probe = runtime_symbol_probe(app)
    if not probe.available:
        raise SimulatorNotAvailableError(probe.reason)
    _log(log, f"isaac app: runtime symbols available ({', '.join(ISAAC_RUNTIME_IMPORT_SURFACE)})")
    return probe


def runtime_import_smoke(
    lifecycle: IsaacAppLifecycle | None = None,
    *,
    names: tuple[str, ...] = ISAAC_RUNTIME_IMPORT_SURFACE,
    release: bool = False,
    log: LogCallback | None = None,
) -> RuntimeSymbolProbe:
    """Run the G1 runtime tier end to end: start the application, then import.

    The gate entry point. It starts Kit if it is not already up -- that start is
    what makes these modules exist -- imports every name in *names*, and returns
    what it found. It does not raise for a missing piece, including an
    application that could not start at all: a gate wants a verdict it can
    record, and ``available=False`` with a reason is that verdict. Read
    ``checked`` to tell "the application never came up" from "it came up and a
    module was missing".

    *release* decides the teardown, and the choice is not cosmetic:

    - ``False`` (default) leaves Kit to process exit, and this function returns
      normally -- which is the only way a gate gets to record what it found.
      Nothing releases the application automatically; there is no exit hook.
    - ``True`` closes it in a ``finally`` -- and on a real Isaac build
      ``SimulationApp.close()`` hard-exits the process with status 0, so **this
      call will not return and nothing after it will run**. Persist the verdict
      before asking for that.

    A caller that has already written its verdict and is about to exit should
    prefer :func:`release_isaac_app_on_success`, which closes only at a clean
    exit status and so cannot rewrite a failure as a success.
    """
    app = isaac_app() if lifecycle is None else lifecycle
    try:
        app.start(log=log)
    except SimulatorNotAvailableError as exc:
        # Not a crash, a G1 result: the tier is unavailable and this is why.
        return RuntimeSymbolProbe(
            app_state=app.state,
            checked=False,
            available=False,
            missing=(),
            reason=f"the G1 runtime tier could not be checked: {exc}",
        )
    try:
        probe = runtime_symbol_probe(app, names=names)
        _log(log, f"isaac app: G1 runtime import smoke -> {probe.reason}")
        return probe
    finally:
        if release:
            # Documented above: on a real build this does not come back.
            app.close(log=log)
