"""``insight-bench check-data``: is this machine ready to run a suite?

Answering that costs a few seconds and no GPU, whereas finding out from a run
costs an Isaac Sim startup and, historically, a wall of the same message
repeated once per episode. This reads the episode file the caller downloaded,
checks it against the digest the coordinate pins, resolves every episode's scene
against the scene root, and reports what is missing -- once per missing *asset*,
not once per episode that wanted it.

It never downloads anything and never starts a simulator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from insight_bench._json import sha256_file
from insight_bench._paths import UnsafePathError, require_regular_file
from insight_bench.contracts import BenchmarkManifest
from insight_bench.registry import Registry

#: How many missing assets to name before summarising the rest.
MAX_REPORTED_MISSING = 20


def check_data(
    reference: str,
    *,
    episodes: Path,
    scene_root: Path,
    registry: Registry | None = None,
) -> dict[str, Any]:
    """Report whether *episodes* and *scene_root* can run *reference*."""
    selected = registry or Registry.load()
    manifest = selected.get(reference)
    report: dict[str, Any] = {
        "benchmark": manifest.coordinate,
        "ready": False,
        "problems": [],
    }
    problems: list[str] = report["problems"]

    episode_report = _check_episode_file(manifest, Path(episodes))
    report["episodes"] = episode_report
    if not episode_report["readable"]:
        problems.append(f"episode file is not readable: {episode_report['error']}")
        return report
    if episode_report["digest_matches"] is False:
        problems.append(
            "the episode file is not the published one this coordinate pins; "
            "a run on a different file is not a run of this benchmark"
        )

    scene_report = _check_scenes(Path(episodes), Path(scene_root))
    report["scenes"] = scene_report
    if scene_report["error"]:
        problems.append(scene_report["error"])
        return report
    if scene_report["missing_assets"]:
        # The count is the real total, not the length of the list beside it:
        # that list is capped, and reporting its length said "20 missing
        # assets" where 200 were missing.
        problems.append(
            f"{scene_report['episodes_blocked']} of {scene_report['episodes_total']} episodes "
            f"have no scene under the scene root, across "
            f"{scene_report['missing_assets_total']} missing assets "
            f"({len(scene_report['missing_assets'])} named below)"
        )

    expected = manifest.metadata.get("episode_count")
    if isinstance(expected, int):
        report["episodes"]["expected_records"] = expected
        if episode_report["records"] != expected:
            problems.append(
                f"the coordinate is defined over {expected} episodes but the file holds "
                f"{episode_report['records']}"
            )

    report["ready"] = not problems
    return report


def _check_episode_file(manifest: BenchmarkManifest, path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "readable": False,
        "error": "",
        "records": 0,
        "sha256": "",
        "expected_sha256": manifest.dataset.sha256 or "",
        "digest_matches": None,
    }
    try:
        require_regular_file(path)
    except UnsafePathError as exc:
        result["error"] = str(exc)
        return result
    result["readable"] = True
    result["sha256"] = sha256_file(path)
    if manifest.dataset.sha256:
        result["digest_matches"] = result["sha256"] == manifest.dataset.sha256

    from insight_bench.vln_runtime.episodes.loader import load_episode_records

    try:
        result["records"] = len(load_episode_records(path))
    except Exception as exc:  # a downloaded file is an untrusted boundary
        result["readable"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _check_scenes(episodes: Path, scene_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "scene_root": str(scene_root),
        "error": "",
        "episodes_total": 0,
        "episodes_blocked": 0,
        "assets_required": 0,
        "assets_found": 0,
        "missing_assets": [],
        "missing_assets_total": 0,
        "missing_assets_omitted": 0,
    }
    if not scene_root.is_dir():
        result["error"] = f"--scene-root {scene_root} is not a directory"
        return result

    from insight_bench.vln_runtime.episodes.adapters.objnav import make_objnav_episode_from_record
    from insight_bench.vln_runtime.episodes.loader import load_episode_records
    from insight_bench.vln_runtime.episodes.scenes import SceneAssetError

    records = load_episode_records(episodes)
    result["episodes_total"] = len(records)
    seen: set[str] = set()
    missing: set[str] = set()
    for record in records:
        scene = record.get("scene")
        asset = str(scene.get("asset", "")) if isinstance(scene, dict) else ""
        seen.add(asset)
        try:
            make_objnav_episode_from_record(record, scene_root=scene_root)
        except SceneAssetError:
            missing.add(asset)
            result["episodes_blocked"] += 1
        except (KeyError, TypeError, ValueError) as exc:
            result["error"] = (
                f"unusable episode record {record.get('episode_id')!r}: {type(exc).__name__}: {exc}"
            )
            return result
    result["assets_required"] = len(seen)
    result["assets_found"] = len(seen) - len(missing)
    ordered = sorted(missing)
    result["missing_assets_total"] = len(ordered)
    result["missing_assets"] = ordered[:MAX_REPORTED_MISSING]
    result["missing_assets_omitted"] = max(0, len(ordered) - MAX_REPORTED_MISSING)
    return result
