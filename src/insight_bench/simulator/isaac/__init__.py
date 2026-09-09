"""Isaac Sim execution code (lazy; runtime-image only).

This package holds the Isaac-specific execution modules: scene loading, camera
stepping, terrain raycasts, and the canonical-observation boundary. Isaac Sim
itself is never pip-installable -- the ``isaac`` extra only pulls the
pure-Python dependencies the runtime image needs on top of its own Isaac
install -- so nothing here may be imported eagerly.

**Import discipline.** This module imports only from the SDK. The backend
implementation (and, through it, torch / isaaclab / omni / carb / pxr / warp)
is imported inside :func:`load_isaac_backend`, so ``import insight_bench`` on a
laptop with no simulator stays Isaac-free. The gate test
``tests/gate/test_independence.py`` pins that.

**Application lifecycle.** Those imports also only resolve once Omniverse Kit
is running, which is :mod:`insight_bench.simulator.isaac.app`'s job:
``open_scene`` starts the headless application before the scene is built.
Nothing releases the application automatically -- closing Kit hard-exits the
process with status 0, so only a top-level owner that has written its results
and succeeded may release it (``release_isaac_app_on_success``); otherwise
process exit reclaims it. Loading a backend never starts anything.

**Driving a live camera.** A backend obtained from the loader alone has no
scene: call ``open_scene`` first, and every camera call on a scene-less backend
refuses by name.

**Honesty.** ``isaac-5.1`` is the official, default line and is the only one
implemented here. ``isaac-6.0`` is experimental: nothing in this repository has
been verified against Isaac Sim 6.0.1, so asking for it refuses rather than
silently running 5.1 code against a 6.0 runtime. Where Isaac is absent, or the
platform cannot host it, the loader raises
:class:`~insight_bench.simulator.base.SimulatorNotAvailableError` naming the
missing piece; it never returns a stub that pretends to simulate.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

from insight_bench.simulator.base import (
    SimulatorBackend,
    SimulatorNotAvailableError,
    get_backend_descriptor,
)
from insight_bench.simulator.isaac.app import (
    ISAAC_RUNTIME_IMPORT_SURFACE,
    IsaacAppLifecycle,
    close_isaac_app,
    isaac_app,
    release_isaac_app_on_success,
    runtime_import_smoke,
    runtime_symbol_probe,
    start_isaac_app,
)

if TYPE_CHECKING:
    from insight_bench.simulator.isaac.backend import IsaacCameraWalkBackend

IMPLEMENTED_BACKENDS: tuple[str, ...] = ("isaac-5.1",)
"""Backend ids with a real execution implementation in this release."""

RUNTIME_DEPENDENCIES: tuple[tuple[str, str], ...] = (
    ("numpy", "numpy"),
    ("PIL", "pillow"),
    ("requests", "requests"),
)
"""(import name, distribution name) of the pure-Python deps the backend needs.

These come from the ``vln`` extra, which the ``isaac`` extra pulls in. A base
install (``pip install insight_bench``, pydantic only) has none of them, and the
backend module imports them at its own module scope -- so the loader has to
check before importing it, or the caller gets a bare ModuleNotFoundError
instead of the honest "this backend cannot execute here" answer.
"""


def missing_runtime_dependencies() -> tuple[str, ...]:
    """Distribution names of the backend's missing pure-Python dependencies.

    Uses ``find_spec``, so asking the question never imports anything.
    """
    missing: list[str] = []
    for module_name, distribution in RUNTIME_DEPENDENCIES:
        try:
            found = importlib.util.find_spec(module_name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(distribution)
    return tuple(missing)


_INSTALL_HINT = "install insight_bench[isaac] (or [vln]) inside the runtime image"


def _dependency_reason(missing: tuple[str, ...]) -> str:
    return (
        f"the backend needs {', '.join(missing)}, which this install does not have; {_INSTALL_HINT}"
    )


def _import_failure_reason(exc: ImportError) -> str:
    """Explain an unexpected backend-import failure in the same shape.

    ``find_spec`` can disagree with a real import (a package present on disk
    but broken, or an environment that blocks the import itself), so this path
    is reachable even when the dependency check passed. Name the module and
    give the same remedy rather than leaking a bare ImportError.
    """
    module = getattr(exc, "name", None)
    subject = f"{module!r} could not be imported" if module else f"an import failed ({exc})"
    return (
        "the backend module needs dependencies this install does not have: "
        f"{subject}; {_INSTALL_HINT}"
    )


_UNIMPLEMENTED_MESSAGE = (
    "no execution backend is implemented for this line in this release; nothing "
    "here has been verified against it, and running the 5.1 implementation "
    "against a different runtime would produce numbers nobody can trust"
)


def load_camera_walk_backend(backend_id: str) -> IsaacCameraWalkBackend:
    """Return the concrete camera-walk backend for *backend_id*.

    Raises :class:`SimulatorNotAvailableError` when the backend has no
    implementation, or when this machine cannot actually run it (no Isaac
    import surface, unsupported platform). First-party runners use this
    typed entry point; :func:`load_isaac_backend` is the generic one.
    """
    try:
        descriptor = get_backend_descriptor(backend_id)
    except KeyError as exc:
        raise SimulatorNotAvailableError(str(exc)) from exc

    if descriptor.backend_id not in IMPLEMENTED_BACKENDS:
        implemented = ", ".join(IMPLEMENTED_BACKENDS)
        raise SimulatorNotAvailableError(
            f"cannot load {backend_id!r} ({descriptor.status}): {_UNIMPLEMENTED_MESSAGE}; "
            f"implemented backends: {implemented}"
        )

    missing = missing_runtime_dependencies()
    if missing:
        raise SimulatorNotAvailableError(
            f"{descriptor.backend_id} cannot execute here: {_dependency_reason(missing)}"
        )

    # Imported here, never at module scope: this is the line that must not run
    # when the SDK is merely installed. The guard above should have caught a
    # thin install already; this catches anything else the backend module needs
    # and turns it into the same honest refusal rather than a raw ImportError.
    try:
        from insight_bench.simulator.isaac.backend import IsaacCameraWalkBackend
    except ImportError as exc:
        raise SimulatorNotAvailableError(
            f"{descriptor.backend_id} cannot execute here: {_import_failure_reason(exc)}"
        ) from exc

    backend = IsaacCameraWalkBackend(descriptor)
    backend.require_available()
    return backend


def load_isaac_backend(backend_id: str) -> SimulatorBackend:
    """Return the Isaac implementation for *backend_id* as the abstract interface."""
    return load_camera_walk_backend(backend_id)


def isaac_backend_probe(backend_id: str) -> dict[str, object]:
    """Diagnose *backend_id* without constructing a usable backend.

    Safe to call anywhere: it reports why a backend is unavailable instead of
    raising, which is what a doctor/diagnostics surface needs.
    """
    try:
        descriptor = get_backend_descriptor(backend_id)
    except KeyError as exc:
        return {"backend_id": backend_id, "available": False, "reasons": [str(exc)]}
    if descriptor.backend_id not in IMPLEMENTED_BACKENDS:
        return {
            "backend_id": descriptor.backend_id,
            "available": False,
            "reasons": [f"{descriptor.backend_id} ({descriptor.status}): {_UNIMPLEMENTED_MESSAGE}"],
        }

    missing = missing_runtime_dependencies()
    if missing:
        return {
            "backend_id": descriptor.backend_id,
            "available": False,
            "reasons": [_dependency_reason(missing)],
            "missing_dependencies": list(missing),
        }

    try:
        from insight_bench.simulator.isaac.backend import IsaacCameraWalkBackend
    except ImportError as exc:
        return {
            "backend_id": descriptor.backend_id,
            "available": False,
            "reasons": [_import_failure_reason(exc)],
            "missing_dependencies": [exc.name] if exc.name else [],
        }

    return IsaacCameraWalkBackend(descriptor).probe()


__all__ = [
    "IMPLEMENTED_BACKENDS",
    "ISAAC_RUNTIME_IMPORT_SURFACE",
    "RUNTIME_DEPENDENCIES",
    "IsaacAppLifecycle",
    "close_isaac_app",
    "isaac_app",
    "isaac_backend_probe",
    "load_camera_walk_backend",
    "load_isaac_backend",
    "missing_runtime_dependencies",
    "release_isaac_app_on_success",
    "runtime_import_smoke",
    "runtime_symbol_probe",
    "start_isaac_app",
]
