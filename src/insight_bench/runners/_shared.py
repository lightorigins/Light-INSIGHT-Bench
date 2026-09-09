"""What the simulator runners share: one scored episode, and one run result.

Two runners now drive the camera-walk rollout and turn its output into a
:class:`RunResult` -- the colored-blocks smoke runner and the insight-bench
runner. The rules they share are the ones a reader of a leaderboard row
depends on: which simulator line a ``--sim-backend`` name resolves to, what
evidence an episode leaves behind, what makes an episode ``error`` rather than
``failed``, what ``success-rate`` is averaged over, what makes a run ``partial``
rather than ``completed``, and how a run is identified.
They live here in one copy, because two copies would agree only until one of
them was edited.

Nothing here knows which benchmark asked; the runner passes its own id in.
"""

from __future__ import annotations

import math
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

from pydantic import ValidationError

from insight_bench._json import load_json, sha256_bytes, stable_json_bytes, write_stable_json
from insight_bench._paths import UnsafePathError, safe_join
from insight_bench._redaction import redact_data, redact_text
from insight_bench.contracts import (
    AdapterAction,
    AdapterDescriptor,
    AdapterResponse,
    BenchmarkManifest,
    EpisodeRunResult,
    RunResult,
    RuntimeAttestation,
)
from insight_bench.simulator.base import default_backend, get_backend_descriptor

if TYPE_CHECKING:
    from insight_bench.runner import RunContext
    from insight_bench.simulator.base import SimBackendDescriptor
    from insight_bench.vln_runtime.episodes import EpisodeSpec
    from insight_bench.vln_runtime.scoring import EpisodeScore, ScoreManager
    from insight_bench.vln_runtime.suite.config import BenchmarkTaskConfig
    from insight_bench.vln_runtime.traces.frames import JpegFrameSink

# Score-term names that decide pass/fail, in preference order. Task families
# name their success term either `sr` (navigation) or `success` (the rest).
SUCCESS_TERMS = ("sr", "success")


def resolve_backend_descriptor(context: RunContext) -> SimBackendDescriptor:
    """The simulator line to load: the one the caller named, or the official default.

    Named-and-ignored is the one outcome a runner must not produce. Reading
    ``--sim-backend`` and then loading the default anyway means a caller who
    asked for ``isaac-6.0`` gets 5.1 code against a 6.0 expectation and no
    refusal, which is exactly the by-name check
    :func:`~insight_bench.simulator.isaac.load_camera_walk_backend` exists to
    perform. An undeclared name fails here, before a simulator is reached.
    """
    if context.sim_backend is None:
        return default_backend()
    # Imported in the body: insight_bench.runner imports the runner modules that
    # import this one, so a module-scope import of it would be a cycle.
    from insight_bench.runner import RunnerError

    try:
        return get_backend_descriptor(context.sim_backend)
    except KeyError as exc:
        raise RunnerError(str(exc)) from exc


def request_id(manifest: BenchmarkManifest, episode_id: str, *, runner_id: str, seed: int) -> str:
    digest = sha256_bytes(
        stable_json_bytes(
            {
                "benchmark": manifest.coordinate,
                "episode_id": episode_id,
                "runner": runner_id,
                "seed": seed,
            }
        )
    )
    return f"req-{digest[:16]}"


def split_metrics(score: EpisodeScore) -> tuple[dict[str, float], dict[str, str]]:
    """Split score terms into the numeric contract and everything else.

    ``EpisodeRunResult.metrics`` is ``dict[str, float]``; categorical terms
    (a NavNuances capability, a first-failed subtask) are still evidence, so
    they are preserved as labels on the response instead of being dropped.
    """
    numeric: dict[str, float] = {}
    labels: dict[str, str] = {}
    for name, value in score.metrics.items():
        # bool is a subclass of int, so this covers success flags too.
        if isinstance(value, int | float):
            numeric[name] = float(value)
        else:
            labels[name] = str(value)
    return numeric, labels


def episode_passed(score: EpisodeScore) -> bool:
    for term in SUCCESS_TERMS:
        if term in score.metrics:
            return bool(score.metrics[term])
    return False


# Reserved per persisted frame. Measured, not assumed: `encode_rgb_jpeg` at
# q85 on the insight-bench sensor size (480x270, `suite/insight_bench/config.py`)
# comes to 6.8 KB for a smooth gradient, 22.5 KB for flat rooms with fine
# noise, 25.4 KB for downscaled photographs and 40.0 KB for a gradient under
# +-25 noise -- the last being the closest stand-in available here for a
# texture-heavy render. 48 KiB is above all of them; pure uniform noise (99 KB)
# is not a render and is not reserved for. No real Isaac render has been
# measured in this environment, so this is the one number to raise if one ever
# exceeds it -- and it must only ever be raised, because it is the bound the
# pre-run refusal is made of.
VIS_FRAME_BYTES = 49_152

# Reserved per rollout step for the trace written beside the frames. Measured
# on four real smoke-run traces at 2,427-2,534 bytes/step. Reserved per *step*
# rather than per frame because the trace records every step whatever the frame
# stride is. A policy server's verbatim `raw_output` rides in these records and
# is not bounded by anything here; the headroom below absorbs a modest excess,
# a garrulous server is not something a pre-run check can size.
VIS_TRACE_BYTES_PER_STEP = 2_560

# Refuse unless the free space exceeds that requirement by this much. What it
# funds: filesystem block granularity across ~110,000 small files, the
# per-episode sidecars, the run manifest, and a margin on the two figures
# above. Deliberately not the traces: they were once left to this multiplier,
# and at ~820 MB over a 1097-episode suite they are far too big to be somebody
# else's rounding error. They have their own line above.
VIS_DISK_HEADROOM = 1.2

# Files `--vis` and `insight-bench vis` write at the run root. Episode ids name
# directories at that same root, and `validate_episode_id` accepts every one of
# these as an id, so a dataset could name an episode `vis-manifest.json` and
# collide a directory with the manifest file. Refused at session open rather
# than survived per-episode: the collision would take the episode's *trace*
# down too, and a trace is evidence, not a picture.
VIS_RUN_LEVEL_FILENAMES = ("vis-manifest.json", "vis.html")

# Episode fields a visualisation may see. Explicitly not EpisodeSpec.to_dict():
# that carries scene.asset_path, which is the absolute path of the user's own
# scene asset on this machine, plus a free-form scene.metadata -- and a run
# directory is meant to be tarred and handed to someone else. A whitelist also
# fails safe when a future EpisodeSpec field carries a path, where redacting a
# full dump only fails safe for the patterns the redactor already knows about.
VIS_EPISODE_METADATA_KEYS = ("suite", "difficulty_label", "instr_type", "scene_class")


@dataclass
class VisSession:
    """Run-level ``--vis`` state: one frame sink, and the manifest describing it.

    Opened once, before the first episode, because that is the only moment the
    footprint can still be refused: the episode count and ``num_steps`` are
    both known by then, and every episode writes at most
    ``max_frames_per_episode`` frames whatever its length, so the *count* of
    what will be written is exact and only the per-item sizes are measured
    figures (:func:`vis_required_bytes`). There is deliberately no run-time
    byte budget -- see
    :func:`~insight_bench.vln_runtime.traces.frames.frame_stride_for`.
    """

    artifact_dir: Path
    sink: JpegFrameSink
    episode_count: int
    max_frames_per_episode: int
    camera_hfov_deg: float | None = None
    sidecars_failed: int = 0
    first_sidecar_failure: str = ""

    def write_episode(self, episode: EpisodeSpec) -> None:
        """Write ``<episode_id>/episode.json``: what the page needs, and nothing else.

        Written before the rollout rather than beside the trace afterwards. An
        episode that errors out never reaches the trace write, and an errored
        episode is precisely the one a reader wants named and categorised
        rather than filed under "unknown".

        Never raises. The episode spec is untrusted dataset content, and it can
        be perfectly valid to :class:`EpisodeSpec` and still unserialisable
        here -- ``goal_radius_m: Infinity`` parses, and ``stable_json_bytes``
        refuses non-finite floats by design. A picture the page cannot draw
        must not be able to abort a run before a single episode is measured, so
        this counts the failure the way the frame sink counts an unencodable
        frame, and the manifest reports it.
        """
        try:
            self._write_episode(episode)
        except Exception as exc:  # untrusted spec, and a filesystem underneath
            self.sidecars_failed += 1
            if not self.first_sidecar_failure:
                detail = redact_text(f"{type(exc).__name__}: {exc}")
                self.first_sidecar_failure = detail or type(exc).__name__

    def _write_episode(self, episode: EpisodeSpec) -> None:
        payload: dict[str, Any] = {
            "episode_id": episode.episode_id,
            "instruction": episode.instruction,
            # The scene id, not the scene block: the block holds the resolved
            # asset path.
            "scene_id": episode.scene.scene_id,
            "goal_position": None if episode.goal_position is None else list(episode.goal_position),
            "goal_radius_m": episode.goal_radius_m,
            "reference_path": [list(point) for point in episode.reference_path],
        }
        for key in VIS_EPISODE_METADATA_KEYS:
            value = episode.metadata.get(key)
            if value is not None:
                payload[key] = str(value)
        # The second lock, after the whitelist. redact_data leaves
        # reference_path alone -- its value is a list of numbers, and the
        # *_path key rule only rewrites strings.
        redacted: Any = redact_data(payload)
        write_stable_json(
            safe_join(self.artifact_dir, f"{episode.episode_id}/episode.json"), redacted
        )

    def write_manifest(self) -> None:
        """Describe the material on disk, at the run root.

        Written when the session opens as well as when it closes, so a run that
        dies halfway still says what the frames beside it mean.

        Never raises either, and for a sharper reason than
        :meth:`write_episode`: this one runs in the runner's ``finally``, so an
        exception here would replace a completed run's result -- every episode
        executed, scored and fingerprinted -- with a stack trace about a
        description of some pictures.
        """
        try:
            self._write_manifest()
        except Exception:  # nothing left to record it in; the run is what matters
            return

    def _write_manifest(self) -> None:
        write_stable_json(
            self.artifact_dir / "vis-manifest.json",
            {
                "schema": "insight-bench/vis-manifest/1",
                "episode_count": self.episode_count,
                "frame_stride": self.sink.stride,
                "max_frames_per_episode": self.max_frames_per_episode,
                "frames_written": self.sink.written,
                "frames_failed": self.sink.failed,
                "frame_failure_reason": self.sink.first_failure,
                "sidecars_failed": self.sidecars_failed,
                "sidecar_failure_reason": self.first_sidecar_failure,
                # The resolved camera geometry, so the top-down view can draw
                # the field of view the frames were actually taken with instead
                # of falling back to a bare heading arrow. An empty block, not
                # an absent key, when the task declares no camera: the shape of
                # the manifest then does not depend on the suite.
                "camera": (
                    {}
                    if self.camera_hfov_deg is None
                    else {"camera_hfov_deg": self.camera_hfov_deg}
                ),
                "frame_path": (
                    "<episode_id>/frames/step_NNNN.jpg, where NNNN is the trace step whose "
                    "observation.frame_path names it: the frame the policy was shown for the "
                    "decision that step records. The camera stood at the PREVIOUS step's "
                    "position when it took that frame. step_0000.jpg never exists -- trace "
                    "step 0 is the pre-action pose and made no decision."
                ),
            },
        )


def vis_camera_hfov_deg(task_config: BenchmarkTaskConfig) -> float | None:
    """Horizontal field of view of the task's first camera sensor, if it has one.

    Read from the sensor the frames come out of, not from the trace: the trace
    records per-step ``observation.intrinsics`` only when the backend fills
    them, and the page needs one number for the whole run.
    """
    for sensor in task_config.sensors:
        value = sensor.params.get("hfov_deg")
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        if math.isfinite(value) and value > 0:
            return float(value)
    return None


def vis_required_bytes(*, episode_count: int, num_steps: int, max_frames_per_episode: int) -> int:
    """Free space ``--vis`` insists on before it starts: frames, traces, headroom.

    Counts are exact -- the stride caps every episode at
    ``max_frames_per_episode`` frames whatever its length, and the episode
    count and ``num_steps`` are both known before episode 1. The two per-item
    sizes are measured upper bounds, which is the whole substance of the
    refusal: quote them too low and ``--vis`` accepts a disk it then fills.
    Public so the footprint quoted to the user is derived from the same
    constants the refusal uses, instead of being a number in a help string
    that drifts away from them.
    """
    frames = max(0, episode_count) * min(max(0, num_steps), max(0, max_frames_per_episode))
    steps = max(0, episode_count) * max(0, num_steps)
    return int((frames * VIS_FRAME_BYTES + steps * VIS_TRACE_BYTES_PER_STEP) * VIS_DISK_HEADROOM)


def open_vis_session(
    context: RunContext, *, task_config: BenchmarkTaskConfig, episode_ids: Sequence[str]
) -> VisSession | None:
    """Start persisting visualisation material, or refuse before episode 1.

    ``None`` in, ``None`` out: without ``--vis`` no session exists, no sink is
    constructed, and nothing downstream changes.

    Takes the ids, not just a count, because one of the two things it can
    refuse is an id that collides with a run-level file -- see
    :data:`VIS_RUN_LEVEL_FILENAMES`.
    """
    if context.vis is None:
        return None
    # Imported in the body: insight_bench.runner imports the runner modules that
    # import this one, so a module-scope import of it would be a cycle.
    from insight_bench.runner import RunnerError

    if context.artifact_dir is None:
        raise RunnerError("--vis needs --artifact-dir: there is nowhere to persist frames")

    colliding = sorted(set(episode_ids) & set(VIS_RUN_LEVEL_FILENAMES))
    if colliding:
        raise RunnerError(
            f"--vis cannot write into this run directory: episode {colliding[0]!r} would need a "
            f"directory named after a file `vis` writes at the run root "
            f"({', '.join(VIS_RUN_LEVEL_FILENAMES)}). Rename the episode in the dataset, or drop "
            "--vis (the run itself does not need it)."
        )

    from insight_bench.vln_runtime.rollout.options import RolloutOptions
    from insight_bench.vln_runtime.traces.frames import JpegFrameSink, frame_stride_for

    episode_count = len(episode_ids)
    max_frames = context.vis.max_frames_per_episode
    num_steps = RolloutOptions.from_task_config(task_config).num_steps
    frames = episode_count * min(num_steps, max_frames)
    required = vis_required_bytes(
        episode_count=episode_count, num_steps=num_steps, max_frames_per_episode=max_frames
    )
    artifact_dir = Path(context.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(artifact_dir).free
    if free < required:
        raise RunnerError(
            f"--vis needs about {required / 2**20:.0f} MiB for {frames} frames and their traces "
            f"across {episode_count} episodes, and {artifact_dir} has "
            f"{free / 2**20:.0f} MiB free. Free space, point --artifact-dir at a larger disk, "
            "or drop --vis (the run itself does not need it)."
        )
    session = VisSession(
        artifact_dir=artifact_dir,
        sink=JpegFrameSink(
            root=artifact_dir, stride=frame_stride_for(num_steps, max_frames=max_frames)
        ),
        episode_count=episode_count,
        max_frames_per_episode=max_frames,
        camera_hfov_deg=vis_camera_hfov_deg(task_config),
    )
    session.write_manifest()
    return session


def run_scored_episode(
    *,
    simulator: Any,
    policy: Any,
    episode: EpisodeSpec,
    task_config: BenchmarkTaskConfig,
    score_manager: ScoreManager,
    adapter_descriptor: AdapterDescriptor,
    manifest: BenchmarkManifest,
    runner_id: str,
    seed: int,
    artifact_dir: Path | None = None,
    vis: VisSession | None = None,
) -> EpisodeRunResult:
    """Run one episode on an open scene, persist its trace, and score it.

    ``status`` is the measurement, not the plumbing: ``passed``/``failed`` for
    reached/not-reached, and ``error`` only when the episode could not execute.
    A model that navigated somewhere wrong produced a valid measurement.
    """
    task_id = manifest.tasks[0].task_id
    response_request_id = request_id(manifest, episode.episode_id, runner_id=runner_id, seed=seed)
    if vis is not None:
        vis.write_episode(episode)
    try:
        # The frame sink is passed only when there is one. Not fastidiousness:
        # a backend implementation predating `--vis` does not accept the
        # keyword, and a run without `--vis` must reach exactly the call it
        # reached before.
        extra = {"frame_sink": vis.sink} if vis is not None else {}
        rollout = simulator.run_episode_with_policy(episode, policy, **extra)
    except Exception as exc:  # simulator and policy are both failure sources
        return EpisodeRunResult(
            episode_id=episode.episode_id,
            task_id=task_id,
            status="error",
            metrics={},
            error=redact_text(str(exc)) or type(exc).__name__,
        )
    # Persist the rollout before scoring it. The trace is the per-episode evidence -- poses,
    # canonical orientations, the observation contract, what the terrain query did -- and none of
    # that survives into RunResult, which carries metrics and a summary by design (ADR 0005: no
    # opaque metadata blobs). Written first so a scoring failure still leaves the evidence behind,
    # which is exactly the run you most want to look at.
    if artifact_dir is not None:
        # safe_join, not a bare `/`: episode ids come from a user-supplied dataset. An id of
        # "../../x" or "/etc/cron.d/x" would otherwise have write_json mkdir -p and write outside
        # the artifact directory entirely. EpisodeSpec rejects such ids at parse time too; this is
        # the second lock, on the side that actually touches the filesystem.
        from insight_bench.vln_runtime.traces.writer import write_rollout_trace

        write_rollout_trace(
            safe_join(Path(artifact_dir), f"{episode.episode_id}/trace.json"), rollout.trace
        )
    score = score_manager.evaluate_episode(
        episode, rollout.trace, final_measures=rollout.final_measures, task_config=task_config
    )
    numeric, non_numeric = split_metrics(score)
    result = EpisodeRunResult(
        episode_id=episode.episode_id,
        task_id=task_id,
        status="passed" if episode_passed(score) else "failed",
        response=AdapterResponse(
            request_id=response_request_id,
            adapter=adapter_descriptor,
            status="success",
            output=rollout.termination_reason,
            actions=(AdapterAction(name="stop"),) if rollout.stop_step >= 0 else (),
            metadata={
                "steps": rollout.steps_taken,
                "stop_step": rollout.stop_step,
                "termination": rollout.termination,
                "final_measures": rollout.final_measures,
                "score_labels": non_numeric,
                "failure_reason": score.failure_reason,
            },
        ),
        metrics=numeric,
    )
    if artifact_dir is not None:
        save_episode_result(Path(artifact_dir), result)
    return result


#: Per-episode result file, written as each episode finishes so an interrupted
#: run is worth something. The trace beside it is evidence about the rollout;
#: this is the scored outcome, which nothing else on disk carried.
EPISODE_RESULT_NAME: Final = "result.json"


def save_episode_result(artifact_dir: Path, result: EpisodeRunResult) -> None:
    """Persist one scored episode. Best effort: a run must not die over this."""
    try:
        write_stable_json(
            safe_join(artifact_dir, f"{result.episode_id}/{EPISODE_RESULT_NAME}"),
            result.model_dump(mode="json"),
        )
    except (OSError, UnsafePathError):
        # Losing a resume point costs a re-run of one episode. Losing the run
        # because a disk filled while writing 2 KB of JSON costs the whole thing.
        return


def load_episode_results(
    artifact_dir: Path, episode_ids: Iterable[str]
) -> dict[str, EpisodeRunResult]:
    """Scored episodes already on disk, keyed by id.

    Anything unreadable or no longer matching the contract is simply absent, so
    the episode runs again. A resume must never turn a corrupt file into a
    measurement.
    """
    found: dict[str, EpisodeRunResult] = {}
    for episode_id in episode_ids:
        try:
            path = safe_join(artifact_dir, f"{episode_id}/{EPISODE_RESULT_NAME}")
        except UnsafePathError:
            continue
        if not path.is_file():
            continue
        try:
            found[episode_id] = EpisodeRunResult.model_validate(load_json(path))
        except (OSError, ValueError, ValidationError):
            continue
    return found


def build_run_result(
    *,
    manifest: BenchmarkManifest,
    adapter_descriptor: AdapterDescriptor,
    runner_id: str,
    seed: int,
    episode_results: list[EpisodeRunResult],
    attestation: RuntimeAttestation,
    expected_episodes: int | None = None,
) -> RunResult:
    execution_key = {
        "adapter": adapter_descriptor,
        "backend": attestation.backend_id,
        "benchmark": manifest.coordinate,
        "dataset_sha256": manifest.dataset.sha256,
        "runner": runner_id,
        "seed": seed,
    }
    run_id = f"run-{sha256_bytes(stable_json_bytes(execution_key))[:16]}"
    scored = [result for result in episode_results if result.status != "error"]
    metrics = aggregate_metrics(scored, total=len(episode_results))
    errors = len(episode_results) - len(scored)
    # Run status reports whether the RUN executed the benchmark, not how well the
    # model did. An episode that finished and did not reach its goal is a
    # legitimate measurement -- a 0% success rate is a completed run with a low
    # score, not a partial one.
    #
    # Two things downgrade it. Episodes that failed to execute, and a run that
    # covered fewer episodes than the benchmark defines: a smoke run over the
    # first few episodes is not a measurement of this benchmark, and it used to
    # come out `completed` while the README promised otherwise. The episode file
    # is ordered by source, so a prefix is not even a fair sample of one.
    truncated = expected_episodes is not None and len(episode_results) < expected_episodes
    status: Literal["completed", "partial", "failed"] = (
        "failed"
        if errors and errors == len(episode_results)
        else "partial"
        if errors or truncated
        else "completed"
    )
    fingerprint = sha256_bytes(
        stable_json_bytes(
            {
                **execution_key,
                "attestation": attestation,
                "episodes": episode_results,
                "metrics": metrics,
                "run_id": run_id,
                "status": status,
            }
        )
    )
    return RunResult(
        run_id=run_id,
        run_fingerprint=fingerprint,
        benchmark_id=manifest.benchmark_id,
        benchmark_version=manifest.version,
        adapter=adapter_descriptor,
        seed=seed,
        status=status,
        episodes=tuple(episode_results),
        metrics=metrics,
        runtime_attestation=attestation,
    )


def aggregate_metrics(scored: list[EpisodeRunResult], *, total: int) -> dict[str, float]:
    """Mean each numeric term over the episodes that produced it.

    ``success-rate`` is always reported over *all* episodes, errored ones
    included: an episode that crashed did not succeed, and averaging it away
    would inflate the headline number.
    """
    if total == 0:
        return {"success-rate": 0.0}
    metrics: dict[str, float] = {
        "success-rate": sum(result.status == "passed" for result in scored) / total
    }
    names = sorted({name for result in scored for name in result.metrics})
    for name in names:
        values = [result.metrics[name] for result in scored if name in result.metrics]
        if values:
            metrics[name] = sum(values) / len(values)
    return metrics
