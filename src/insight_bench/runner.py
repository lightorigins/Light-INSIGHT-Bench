"""Generic runner dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from insight_bench._json import sha256_file, write_stable_json
from insight_bench._paths import UnsafePathError, require_regular_file
from insight_bench.contracts import BenchmarkManifest, RunResult
from insight_bench.registry import Registry, RegistryError


@dataclass(frozen=True)
class VisOptions:
    """What ``run --vis`` persists per episode, and how much of it.

    One number, because one is all the mechanism needs: the rollout writes
    every ``ceil(num_steps / max_frames_per_episode)``-th decision frame, so an
    episode of 100 steps and an episode of 300 steps both cost at most this
    many frames (~2 MB at 480x270 JPEG q85). That makes the footprint of a run
    exactly computable before it starts -- which is what lets ``--vis`` refuse
    up front instead of filling a disk halfway through.
    """

    max_frames_per_episode: int = 100


@dataclass(frozen=True)
class RunContext:
    """Everything a runner needs that is not the benchmark, the adapter or the seed.

    A typed object rather than a widening parameter list. Two runners already disagreed about
    whether they accept ``run_dir``, and a ``Callable[..., RunResult]`` alias hid that from mypy
    until the only thing standing between it and a user was an unrelated fail-closed gate.

    ``artifact_dir`` is where per-episode artifacts go -- ``<artifact_dir>/<episode_id>/trace.json``
    for a simulator runner. ``None`` means the caller wants none, which is right for a text runner
    and for anyone who only reads the RunResult.

    ``sim_backend`` names the simulator line to drive. It is carried here rather than inferred so
    that "which line produced this number" is an input to the run, not a guess about the host.

    The last three are what a licence-gated suite cannot ship and the user must therefore point
    at. ``episode_dataset`` is the published episode file they downloaded, ``scene_root`` the
    directory their converted scene assets live under, and ``policy_url`` the HTTP policy server
    they are running. All three are *data* -- a filesystem path and a URL -- never a module path
    or an object to import, so they stay inside the execution boundary in CLAUDE.md (ADR 0009).

    ``max_episodes`` truncates the run for smoke-testing a setup. It makes the result cover
    part of the suite, which is why the runner labels it in the run's own adapter metadata
    rather than letting a partial measurement pass for a full one.

    ``vis`` asks the runner to persist replay material next to the traces, for
    ``insight-bench vis``. ``None`` -- the default -- does not mean "visualise
    with defaults": it means the run writes exactly what it wrote before
    ``--vis`` existed, byte for byte, and pays nothing for the option existing.

    ``resume`` reuses the per-episode results already in ``artifact_dir`` instead
    of running those episodes again. A full suite is hours to a day depending on
    the model, and a run that stopped at episode 900 used to be worth nothing:
    the traces were all on disk and no result could be built from them. It is
    refused unless the run is configured identically to the one that wrote them,
    because resuming across a different model or dataset would blend two
    measurements into one number.
    """

    artifact_dir: Path | None = None
    sim_backend: str | None = None
    episode_dataset: Path | None = None
    scene_root: Path | None = None
    policy_url: str | None = None
    max_episodes: int | None = None
    vis: VisOptions | None = None
    resume: bool = False


class Runner(Protocol):
    """The runner contract, checkable.

    Written as a Protocol with a named, keyword-only ``context`` so mypy rejects a runner that
    forgot it. That is not hypothetical: it is exactly the mistake this replaced an ellipsis alias
    to catch.
    """

    def __call__(
        self,
        manifest: BenchmarkManifest,
        dataset_path: Path,
        seed: int,
        *,
        context: RunContext,
    ) -> RunResult: ...


_RUNNERS: dict[str, Runner] = {}


class RunnerError(RuntimeError):
    """Raised for unavailable or invalid benchmark execution."""


def register_runner(name: str, runner: Runner) -> None:
    if name in _RUNNERS:
        raise ValueError(f"runner {name!r} is already registered")
    _RUNNERS[name] = runner


def available_runners() -> tuple[str, ...]:
    return tuple(sorted(_RUNNERS))


def run_benchmark(
    reference: str,
    *,
    registry: Registry | None = None,
    seed: int = 0,
    output: Path | None = None,
    context: RunContext | None = None,
) -> RunResult:
    """Resolve a manifest and run it locally through its declared runner plugin."""
    selected_registry = registry or Registry.load()
    manifest = selected_registry.get(reference)
    if not manifest.runnable:
        missing = ", ".join(manifest.requires_configuration) or "benchmark assets"
        raise RunnerError(f"{manifest.coordinate} is not runnable; configure: {missing}")
    try:
        runner = _RUNNERS[manifest.runtime.runner]
    except KeyError as exc:
        raise RunnerError(
            f"runner {manifest.runtime.runner!r} is not installed; "
            f"available: {', '.join(available_runners())}"
        ) from exc
    resolved_context = context or RunContext()
    try:
        result = runner(
            manifest,
            _resolve_dataset_path(manifest, selected_registry, resolved_context),
            seed,
            context=resolved_context,
        )
    except RegistryError as exc:
        raise RunnerError(str(exc)) from exc
    if output is not None:
        write_stable_json(output, result)
    return result


def _resolve_dataset_path(
    manifest: BenchmarkManifest, registry: Registry, context: RunContext
) -> Path:
    """Locate the episode file for *manifest*, bundled or user-supplied.

    A bundled dataset comes from the registry, which already verified its digest
    at load time. A user-supplied one cannot: the licence-gated suite ships no
    episodes, so the caller passes the published file they downloaded and this
    checks it against the digest the manifest pins. A run against a file that is
    not the published one is not a run of that benchmark, and it would otherwise
    reach a leaderboard under that benchmark's coordinate.
    """
    if manifest.dataset.availability == "bundled":
        return registry.dataset_path(manifest)
    if context.episode_dataset is None:
        raise RunnerError(
            f"{manifest.coordinate} needs its episode file, passed with --episodes "
            f"(RunContext.episode_dataset). {manifest.dataset.notice}"
        )
    path = Path(context.episode_dataset)
    try:
        require_regular_file(path)
    except UnsafePathError as exc:
        raise RunnerError(
            f"{manifest.coordinate}: --episodes {path} is not readable: {exc}"
        ) from exc
    expected = manifest.dataset.sha256
    actual = sha256_file(path)
    if expected is not None and actual != expected:
        raise RunnerError(
            f"{manifest.coordinate}: the episode file does not match the published dataset "
            f"digest this coordinate pins (expected {expected}, got {actual}); a run on a "
            "different file is not a run of this benchmark"
        )
    return path


# Registered last: this runner module needs register_runner above.
from insight_bench.runners import insight_bench as _insight_bench  # noqa: E402,F401
