"""The runner behind the published objnav suite.

What it does, in the order it does it, because the order is the point:

1. resolves the task family the manifest declares and the scene root the
   caller supplied, refusing by name if either is missing;
2. binds every published episode record to a scene asset that actually exists
   under that root, and to the success radius the suite scores it at -- and
   reports *every* missing scene asset at once, before a simulator is started,
   since the alternative is discovering the twelfth missing one an hour into a
   run;
3. checks the episode count against the manifest, so a truncated file cannot be
   submitted as a full run of the coordinate;
4. loads the simulator backend and its runtime attestation;
5. asks the user's policy server who it is (``GET /health``) and records that as
   the model identity of the run;
6. groups the episodes by scene asset, opens each asset once, orders the mesh
   scene families ahead of the Gaussian-splat ones (see :func:`_group_by_scene_asset`
   -- the order changes the numbers), and drives every episode of an asset
   through the camera-walk rollout against the policy;
7. sorts the results by episode id and assembles a :class:`RunResult` whose
   headline metric is ``success-rate``.

The policy is reached over HTTP, through
:class:`~insight_bench.vln_runtime.policy.client.LocalVlnPolicyClient`, at a URL
the caller supplies. **The model is that server**, so the run's
:class:`~insight_bench.contracts.AdapterDescriptor` is built from what the server
reports about itself; the in-process ``adapter`` argument of the runner protocol
has no role on this path and is not consulted. A URL is data, not a module to
import, so this stays inside the execution boundary -- see ADR 0009 for the
reasoning and for what trusting a local network endpoint implies.

Nothing here fabricates a run. No simulator, no scene, no policy server, a
short episode file, or a task family this runner cannot drive all raise
:class:`~insight_bench.runner.RunnerError` naming the missing piece.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from insight_bench._json import load_json, write_stable_json
from insight_bench.contracts import (
    AdapterDescriptor,
    BenchmarkManifest,
    EpisodeRunResult,
    RunResult,
    RuntimeAttestation,
)
from insight_bench.runner import RunContext, RunnerError, register_runner
from insight_bench.runners._shared import (
    build_run_result,
    load_episode_results,
    open_vis_session,
    resolve_backend_descriptor,
    run_scored_episode,
)
from insight_bench.simulator.base import SimulatorNotAvailableError
from insight_bench.vln_runtime.episodes.loader import load_episode_records

if TYPE_CHECKING:
    from insight_bench.simulator.isaac.backend import IsaacCameraWalkBackend
    from insight_bench.vln_runtime.episodes import EpisodeSpec
    from insight_bench.vln_runtime.policy.client import LocalVlnPolicyClient
    from insight_bench.vln_runtime.suite.config import BenchmarkTaskConfig

RUNNER_ID = "objnav-http-policy-v1"

DEFAULT_POLICY_URL = "http://127.0.0.1:18081"
"""Where the reference policy server binds. Loopback, and meant to stay there."""

# Task families this runner can drive. A manifest names one in
# `metadata["task_family"]`; the name is data resolved through the static
# suite table, never an import path.
_EXECUTABLE_FAMILIES: tuple[str, ...] = ("insight_bench",)

# How many distinct missing scene assets to name before summarising the rest.
_MAX_REPORTED_SCENE_FAILURES = 8


def load_backend(backend_id: str) -> IsaacCameraWalkBackend:
    """Obtain the camera-walk backend. The single seam tests substitute.

    Imported here rather than at module scope: insight_bench.runner imports this
    module eagerly to register the runner, and the backend module pulls the
    `vln` extra's dependencies.
    """
    from insight_bench.simulator.isaac import load_camera_walk_backend

    return load_camera_walk_backend(backend_id)


def insight_bench_runner(
    manifest: BenchmarkManifest,
    dataset_path: Path,
    seed: int,
    *,
    context: RunContext | None = None,
) -> RunResult:
    """Run a published objnav suite against the caller's HTTP policy server."""
    resolved_context = context or RunContext()
    task_config = _task_config_for(manifest)
    scene_root = _scene_root(manifest, resolved_context)
    episodes = _bind_episodes(
        manifest, dataset_path, scene_root, max_episodes=resolved_context.max_episodes
    )
    # Before the simulator is loaded, so `--vis` refuses on a disk too small
    # for its frames while the run has still cost the user nothing.
    vis = open_vis_session(
        resolved_context,
        task_config=task_config,
        episode_ids=[episode.episode_id for episode in episodes],
    )

    backend_descriptor = resolve_backend_descriptor(resolved_context)
    try:
        simulator = load_backend(backend_descriptor.backend_id)
    except SimulatorNotAvailableError as exc:
        raise RunnerError(
            f"{manifest.coordinate}: {len(episodes)} episodes validated and ready, but {exc}"
        ) from exc
    attestation = _attestation(manifest, simulator)

    from insight_bench.vln_runtime.policy.client import LocalVlnPolicyClient
    from insight_bench.vln_runtime.scoring import ScoreManager

    policy_url = resolved_context.policy_url or DEFAULT_POLICY_URL
    policy = LocalVlnPolicyClient(policy_url)
    score_manager = ScoreManager(task_config.scoring)
    episode_results: list[EpisodeRunResult] = []
    try:
        adapter_descriptor = _policy_descriptor(manifest, policy, policy_url)
        if resolved_context.max_episodes is not None:
            adapter_descriptor = _label_episode_subset(adapter_descriptor, manifest, len(episodes))
        # Episodes already scored in this directory by an identically configured
        # run. Read after the policy descriptor exists, because that descriptor
        # is most of what "identically configured" means.
        done = _resumable_results(
            manifest,
            resolved_context,
            adapter_descriptor=adapter_descriptor,
            seed=seed,
            episodes=episodes,
        )
        if done:
            episode_results.extend(done.values())
            episodes = [episode for episode in episodes if episode.episode_id not in done]
        for group in _group_by_scene_asset(task_config, episodes):
            scene_task = _bind_task_config_to_episode(task_config, group[0])
            try:
                simulator.open_scene(scene_task)
            except SimulatorNotAvailableError as exc:
                raise RunnerError(
                    f"{manifest.coordinate}: cannot open the scene for "
                    f"{group[0].scene.scene_id!r}: {exc}"
                ) from exc
            try:
                for episode in group:
                    _require_live_policy(manifest, policy, policy_url)
                    episode_results.append(
                        run_scored_episode(
                            simulator=simulator,
                            policy=policy,
                            episode=episode,
                            task_config=scene_task,
                            score_manager=score_manager,
                            adapter_descriptor=adapter_descriptor,
                            manifest=manifest,
                            runner_id=RUNNER_ID,
                            seed=seed,
                            artifact_dir=resolved_context.artifact_dir,
                            vis=vis,
                        )
                    )
            finally:
                simulator.close()
    finally:
        policy.close()
        if vis is not None:
            vis.write_manifest()

    # Executed grouped by scene asset, in scene-family order, to amortise scene
    # loading and keep the renderer state consistent; reported by episode id, so
    # the result -- and its fingerprint -- does not depend on the order the
    # scenes were visited in.
    episode_results.sort(key=lambda result: result.episode_id)
    return build_run_result(
        manifest=manifest,
        adapter_descriptor=adapter_descriptor,
        runner_id=RUNNER_ID,
        seed=seed,
        episode_results=episode_results,
        attestation=attestation,
        # What the benchmark defines, so a run over fewer episodes than that
        # reports `partial` instead of looking like a full measurement.
        expected_episodes=_declared_episode_count(manifest),
    )


#: Written at the run root the first time a run writes results there, and
#: checked before any of them are reused.
RESUME_KEY_NAME: Final = "resume-key.json"


def _resume_key(
    manifest: BenchmarkManifest,
    context: RunContext,
    *,
    adapter_descriptor: AdapterDescriptor,
    seed: int,
) -> dict[str, Any]:
    """What has to match for episodes from an earlier run to be reusable.

    The model, the benchmark, the exact episode bytes, the scene root, the seed
    and the simulator line. Resuming across a change in any of them would build
    one success rate out of two different measurements, which is worse than
    losing the run: the number would look ordinary.
    """
    return {
        "backend": context.sim_backend or "",
        "benchmark": manifest.coordinate,
        "dataset_sha256": manifest.dataset.sha256 or "",
        "model_id": adapter_descriptor.model_id,
        "runner": RUNNER_ID,
        "scene_root": str(context.scene_root or ""),
        "seed": seed,
    }


def _resumable_results(
    manifest: BenchmarkManifest,
    context: RunContext,
    *,
    adapter_descriptor: AdapterDescriptor,
    seed: int,
    episodes: list[EpisodeSpec],
) -> dict[str, EpisodeRunResult]:
    """Scored episodes this run may adopt, and the key that licenses adopting them."""
    artifact_dir = context.artifact_dir
    if artifact_dir is None:
        return {}
    artifact_dir = Path(artifact_dir)
    key = _resume_key(manifest, context, adapter_descriptor=adapter_descriptor, seed=seed)
    key_path = artifact_dir / RESUME_KEY_NAME
    stored: dict[str, Any] | None = None
    if key_path.is_file():
        try:
            loaded = load_json(key_path)
            stored = loaded if isinstance(loaded, dict) else None
        except (OSError, ValueError):
            stored = None
    if not context.resume:
        # Not resuming: still record the key, so a later --resume against this
        # directory has something to check itself against.
        if stored != key:
            write_stable_json(key_path, key)
        return {}
    if stored is None:
        raise RunnerError(
            f"{manifest.coordinate}: --resume needs a previous run's {RESUME_KEY_NAME} in "
            f"{artifact_dir}, and there is none. Drop --resume to start this run."
        )
    if stored != key:
        differing = sorted(name for name in key if stored.get(name) != key[name])
        raise RunnerError(
            f"{manifest.coordinate}: refusing to resume a run configured differently -- "
            f"{', '.join(differing)} changed. Resuming would average two measurements "
            f"into one number; use a fresh --run-dir instead."
        )
    return load_episode_results(artifact_dir, (episode.episode_id for episode in episodes))


def _declared_episode_count(manifest: BenchmarkManifest) -> int | None:
    """How many episodes the coordinate says it has, when it says.

    Read rather than assumed: a manifest that omits the count leaves the run
    status decided by execution errors alone, which is what it was before.
    """
    declared = manifest.metadata.get("episode_count")
    return declared if isinstance(declared, int) and declared > 0 else None


def _task_config_for(manifest: BenchmarkManifest) -> BenchmarkTaskConfig:
    family = manifest.metadata.get("task_family")
    if family not in _EXECUTABLE_FAMILIES:
        known = ", ".join(_EXECUTABLE_FAMILIES)
        raise RunnerError(
            f"{manifest.coordinate}: metadata.task_family={family!r} is not a family the "
            f"{RUNNER_ID} runner can drive; it drives: {known}"
        )
    from insight_bench.vln_runtime.suite import get_task_config

    return get_task_config(str(family))


def _scene_root(manifest: BenchmarkManifest, context: RunContext) -> Path:
    """The directory the user's converted scene assets live under, or a refusal."""
    if context.scene_root is None:
        families = ", ".join(item.family for item in manifest.asset_licenses)
        raise RunnerError(
            f"{manifest.coordinate} needs --scene-root: its scenes are user-provided and "
            f"this SDK never downloads them. Convert the {families} scenes into one "
            "directory and pass it (see the manifest's asset_licenses for each source "
            "and its licence terms)."
        )
    scene_root = Path(context.scene_root)
    if not scene_root.is_dir():
        raise RunnerError(f"{manifest.coordinate}: --scene-root {scene_root} is not a directory")
    return scene_root


def _bind_episodes(
    manifest: BenchmarkManifest,
    dataset_path: Path,
    scene_root: Path,
    *,
    max_episodes: int | None = None,
) -> list[EpisodeSpec]:
    """Parse every record, bind it to a real scene, and check the episode count.

    Scene failures are collected rather than raised one at a time, and reported
    once per missing *asset* rather than once per episode: a suite of 1097
    episodes over 218 scene assets with an empty scene root would otherwise
    print the same handful of messages a thousand times. What a user needs is
    the list of files to go and put there -- and it is a list of assets, since
    8 of insight-bench-v1's 210 scenes are referenced through two entry points.
    """
    from insight_bench.vln_runtime.episodes.adapters.objnav import make_objnav_episode_from_record
    from insight_bench.vln_runtime.episodes.scenes import SceneAssetError

    records = load_episode_records(dataset_path)
    if not records:
        raise RunnerError(f"{manifest.coordinate}: the episode file is empty")

    episodes: list[EpisodeSpec] = []
    missing_assets: dict[str, str] = {}
    affected = 0
    for record in records:
        try:
            episodes.append(make_objnav_episode_from_record(record, scene_root=scene_root))
        except SceneAssetError as exc:
            scene = record.get("scene")
            key = str(scene.get("asset") if isinstance(scene, dict) else record.get("episode_id"))
            missing_assets.setdefault(key, str(exc))
            affected += 1
        except (KeyError, TypeError, ValueError) as exc:
            raise RunnerError(
                f"{manifest.coordinate}: unusable episode record "
                f"{record.get('episode_id')!r}: {exc}"
            ) from exc
    if missing_assets:
        shown = list(missing_assets.values())[:_MAX_REPORTED_SCENE_FAILURES]
        remaining = len(missing_assets) - len(shown)
        suffix = f" ... and {remaining} more missing scene assets." if remaining else ""
        raise RunnerError(
            f"{manifest.coordinate}: {affected} of {len(records)} episodes have no scene "
            f"asset under the scene root, across {len(missing_assets)} missing scene assets. "
            + " ".join(shown)
            + suffix
        )

    expected = manifest.metadata.get("episode_count")
    if max_episodes is not None:
        # A deliberate subset, so the count check below would refuse the very
        # thing the flag exists to do. The result says so in its own metadata.
        if max_episodes < 1:
            raise RunnerError("--max-episodes must be at least 1")
        _require_declared_asset_licenses(manifest, episodes)
        return episodes[:max_episodes]
    if isinstance(expected, int) and len(episodes) != expected:
        raise RunnerError(
            f"{manifest.coordinate} is defined over {expected} episodes but the episode "
            f"file holds {len(episodes)}; a partial file is not a run of this coordinate"
        )
    _require_declared_asset_licenses(manifest, episodes)
    return episodes


def _require_declared_asset_licenses(
    manifest: BenchmarkManifest, episodes: list[EpisodeSpec]
) -> None:
    """Refuse a run over a scene family the manifest does not license.

    The licence block is the published record of what an outside reader may
    legally reproduce, so it has to cover what the run actually loaded. A
    dataset appearing in the episodes and not in the manifest means one of the
    two is wrong, and neither is safe to guess about.
    """
    declared = {item.family for item in manifest.asset_licenses}
    used = {episode.scene.dataset for episode in episodes if episode.scene.dataset}
    undeclared = sorted(used - declared)
    if undeclared:
        raise RunnerError(
            f"{manifest.coordinate}: the episodes reference scene datasets the manifest "
            f"declares no licence for: {', '.join(undeclared)}"
        )


def _attestation(
    manifest: BenchmarkManifest, simulator: IsaacCameraWalkBackend
) -> RuntimeAttestation:
    try:
        return simulator.runtime_attestation()
    except SimulatorNotAvailableError as exc:
        raise RunnerError(f"{manifest.coordinate}: {exc}") from exc


def _require_live_policy(manifest: Any, policy: Any, policy_url: str) -> None:
    """Stop before an episode the policy can no longer answer.

    ``/health`` is read once before the first episode, which is enough to
    attribute a run and not enough to protect one. A policy whose model died
    after that kept answering: ten episodes each failed on connection refused,
    one at a time, with the card held for all of them, and the run reported
    itself as a set of failures rather than as a dead policy. Asking again
    between episodes turns that into a single refusal that says why.

    Only an explicit ``false`` stops a run. A server that does not report the
    field at all -- anything predating the liveness contract -- is left alone
    rather than refused for its silence.
    """
    try:
        health = policy.health()
    except Exception as exc:  # a user-run server is an untrusted boundary
        raise RunnerError(
            f"{manifest.coordinate}: the policy server at {policy_url} stopped answering "
            f"GET /health partway through the run ({type(exc).__name__}: {exc}). The "
            "episodes already scored are on disk; fix the server and re-run to resume."
        ) from exc
    if health.get("healthy") is False:
        raise RunnerError(
            f"{manifest.coordinate}: the policy server at {policy_url} reports "
            "healthy=false, so the model behind it is gone even though the server still "
            "answers. Stopping here rather than scoring episodes it cannot drive. The "
            "episodes already scored are on disk; fix the server and re-run to resume."
        )


def _policy_descriptor(
    manifest: BenchmarkManifest, policy: LocalVlnPolicyClient, policy_url: str
) -> AdapterDescriptor:
    """Ask the policy server who it is, and record that as the run's model.

    Required, not optional. The numbers a submitted run carries were produced by
    whatever answered this URL, and the only thing that knows what that was is
    the server itself -- so a server that will not say is refused rather than
    labelled with a placeholder or with a text adapter's identity.
    """
    try:
        health = policy.health()
    except Exception as exc:  # a user-run server is an untrusted boundary
        raise RunnerError(
            f"{manifest.coordinate}: no policy server answered GET {policy_url}/health "
            f"({type(exc).__name__}: {exc}). Start your model's policy server "
            "(see policies/ in this repository) and pass its URL with --policy-url."
        ) from exc
    model_id = str(health.get("model_id") or "").strip()
    # The shipped template's placeholder is non-empty, so the emptiness check
    # below let it through: a reader who copied the template and forgot this one
    # line would have stamped `TODO/your-model` into an evidence pack as the
    # identity of the whole run, and nothing would have said so.
    if model_id.lower().startswith(("todo/", "todo-", "your-", "changeme")):
        raise RunnerError(
            f"{manifest.coordinate}: the policy server reports model_id={model_id!r}, which is the "
            "template's placeholder. Set model_id on your policy class to the name this run should "
            "be attributed to -- it is the identity of every number the run produces."
        )
    if not model_id:
        raise RunnerError(
            f"{manifest.coordinate}: the policy server reported no model_id on /health, so "
            "this run has nothing to attribute its numbers to; set model_id on your "
            "policy class"
        )
    # Everything else the server said about itself, as flat strings. The URL is
    # deliberately absent: it is deployment detail, and evidence packs are shared.
    # Scalars keep their type; anything structured is rendered rather than
    # dropped. Stringifying everything turned `healthy: true` into the string
    # "True" in every descriptor, which reads as a server that answered with a
    # word rather than a boolean.
    # `/health` is read once, here, before the first episode. So only what a
    # server says about its identity and configuration means anything; anything
    # live is captured at zero and reads as data. `active_episode_id` in the
    # singular is what the shipped server sends, and it used to slip through the
    # plural spelling below and publish an empty string in every run.
    reported = {
        key: value if isinstance(value, (bool, int, float, str)) else json.dumps(value, indent=None)
        for key, value in sorted(health.items())
        if key
        not in {
            "model_id",
            "active_episode_id",
            "active_episode_ids",
            "num_active_sessions",
            # Liveness, not configuration, and publishing this one was worse
            # than publishing it at zero: a run whose model was dead from the
            # first episode carried `healthy: true` in its evidence, because
            # that is what the server said in the second before it died. The
            # run is now stopped when this goes false, below, rather than
            # attested with a reading that has expired.
            "healthy",
            "policy_child_alive",
        }
    }
    return AdapterDescriptor(
        adapter_id="vln-http-policy",
        model_id=model_id,
        deterministic=False,
        metadata={"policy_protocol": "vln-http/1", **reported},
    )


def _label_episode_subset(
    descriptor: AdapterDescriptor, manifest: BenchmarkManifest, ran: int
) -> AdapterDescriptor:
    """Record, in the run's own identity, that it covered part of the suite.

    ``--max-episodes`` exists so a user can prove a setup works without paying
    for a full suite, and the number it produces is real -- for those episodes.
    Carrying the fact in the descriptor means a result cannot be mistaken for a
    full measurement later, by a reader or by a leaderboard.
    """
    total = manifest.metadata.get("episode_count")
    of_total = f" of {total}" if isinstance(total, int) else ""
    return descriptor.model_copy(
        update={
            "metadata": {
                **descriptor.metadata,
                "episode_subset": f"{ran}{of_total} episodes; smoke run, not a full measurement",
            }
        }
    )


def _bind_task_config_to_episode(
    task_config: BenchmarkTaskConfig, episode: EpisodeSpec
) -> BenchmarkTaskConfig:
    """Point the task's USD terrain at this episode's resolved scene asset."""
    from dataclasses import replace

    from insight_bench.vln_runtime.suite.config import TerrainConfig

    asset_path = episode.scene.asset_path
    if task_config.terrain.kind != "usd" or not asset_path:
        return task_config
    if task_config.terrain.usd_path == asset_path:
        return task_config
    terrain = TerrainConfig(
        kind="usd",
        prim_path=task_config.terrain.prim_path,
        usd_path=asset_path,
        env_spacing_m=task_config.terrain.env_spacing_m,
        metadata={
            **task_config.terrain.metadata,
            "episode_scene_id": episode.scene.scene_id,
            "episode_scene_dataset": episode.scene.dataset,
        },
    )
    return replace(task_config, terrain=terrain)


def _scene_key(task_config: BenchmarkTaskConfig, episode: EpisodeSpec) -> tuple[Any, ...]:
    terrain = _bind_task_config_to_episode(task_config, episode).terrain
    return (terrain.kind, terrain.prim_path, terrain.usd_path or "", terrain.env_spacing_m)


# Which renderer path a scene family goes down decides where its episodes run in
# the sequence: everything mesh first, everything neural after. Ordering, not
# naming -- the families themselves are declared in the scene-layout table.
_RENDERER_ORDER = {"mesh": 0, "neural": 1}


def _group_by_scene_asset(
    task_config: BenchmarkTaskConfig, episodes: list[EpisodeSpec]
) -> list[list[EpisodeSpec]]:
    """Order episodes so every scene asset is loaded once, mesh families first.

    Two rules, and only the first is about speed.

    **One load per asset.** Loading a scene is the expensive part of an objnav
    run -- minutes of USD composition against seconds of rollout -- so episodes
    are sorted by the stage they open and consecutive episodes sharing one are
    handed back as a group. Sorting (rather than grouping whatever order the
    file happened to be in) is what makes "loaded once" a guarantee instead of a
    property of the dataset. The key is the *asset*, not the scene id: a scene
    published with two equivalent entry points -- the Habitat-GS stubs, 8 of the
    210 scenes of insight-bench-v1 -- is two stages to compose, not one.

    **Mesh families before neural ones.** This one changes the numbers. Loading
    a Gaussian-splat stage leaves the renderer in a different state that
    persists for the life of the process: measured on the reference
    implementation, reloading one mesh scene repeatedly moves its first frame by
    at most 1.13/255, inserting another mesh scene by 1.18, and inserting a
    splat scene by 8.78 -- after which it stays in the new state. Clearing the
    stage does not undo it and no renderer setting differs across the boundary;
    only a fresh process does, and this one cannot restart itself (Omniverse Kit
    cannot be brought back up in a process that has already started it). What is
    left is the order, and it is enough for the direction that was measured to
    matter: with every mesh family ahead of every neural one, no mesh episode is
    ever rendered after a splat load. The reverse -- splat scenes in a process
    that has loaded mesh scenes -- is not measured, and is the residual
    difference from restarting at each family boundary; both manifests'
    comparability notes say so, and ADR 0009 records the decision.
    """
    from insight_bench.vln_runtime.episodes.scenes import scene_asset_tree, scene_renderer

    def order(episode: EpisodeSpec) -> tuple[Any, ...]:
        return (
            _RENDERER_ORDER[scene_renderer(episode.scene)],
            scene_asset_tree(episode.scene),
            str(_scene_key(task_config, episode)),
            episode.episode_id,
        )

    groups: list[list[EpisodeSpec]] = []
    current_key: tuple[Any, ...] | None = None
    for episode in sorted(episodes, key=order):
        key = _scene_key(task_config, episode)
        if not groups or key != current_key:
            groups.append([])
            current_key = key
        groups[-1].append(episode)
    return groups


register_runner(RUNNER_ID, insight_bench_runner)
