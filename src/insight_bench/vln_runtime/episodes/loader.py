"""Small JSON/JSONL loaders for benchmark episode files."""

from __future__ import annotations

import gzip
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from .schema import EpisodeSpec

if TYPE_CHECKING:  # a runtime import would cycle: suite configs import this package's schema
    from ..suite.config import BenchmarkTaskConfig


def load_episode_file(path: str | Path) -> list[EpisodeSpec]:
    """Load one JSON file containing one episode or an episode list.

    Accepted shapes:
    - ``{"episodes": [...]}``
    - ``[...]``
    - ``{episode object}``
    """

    file_path = Path(path)
    if file_path.suffix == ".gz":
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("episodes"), list):
        raw_items = payload["episodes"]
    elif isinstance(payload, list):
        raw_items = payload
    elif isinstance(payload, dict):
        raw_items = [payload]
    else:
        raise ValueError(f"Unsupported episode JSON payload in {file_path}")
    return [EpisodeSpec.from_dict(item) for item in raw_items if isinstance(item, dict)]


def load_episode_records(path: str | Path) -> list[dict[str, object]]:
    """Load public episode records from JSON, JSONL, or a directory.

    These records may be dataset-friendly inputs that still need a task-specific
    binder before becoming final ``EpisodeSpec`` objects.
    """

    records: list[dict[str, object]] = []
    for file_path in iter_episode_record_files(path):
        records.extend(load_episode_record_file(file_path))
    return records


def load_episode_record_file(path: str | Path) -> list[dict[str, object]]:
    file_path = Path(path)
    if file_path.suffix == ".jsonl":
        records: list[dict[str, object]] = []
        with file_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if not isinstance(payload, dict):
                    raise ValueError(f"Episode JSONL line must be an object: {file_path}:{line_no}")
                records.append(dict(payload))
        return records

    if file_path.suffix == ".gz":
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("episodes"), list):
        raw_items = payload["episodes"]
    elif isinstance(payload, list):
        raw_items = payload
    elif isinstance(payload, dict):
        raw_items = [payload]
    else:
        raise ValueError(f"Unsupported episode record payload in {file_path}")
    return [dict(item) for item in raw_items if isinstance(item, dict)]


def iter_episode_files(path: str | Path) -> Iterable[Path]:
    root = Path(path)
    if root.is_file():
        yield root
        return
    if not root.is_dir():
        raise FileNotFoundError(f"Episode path not found: {root}")
    patterns = ("*.json", "*.json.gz")
    for pattern in patterns:
        yield from sorted(item for item in root.rglob(pattern) if item.is_file())


def iter_episode_record_files(path: str | Path) -> Iterable[Path]:
    root = Path(path)
    if root.is_file():
        yield root
        return
    if not root.is_dir():
        raise FileNotFoundError(f"Episode path not found: {root}")
    patterns = ("*.jsonl", "*.json", "*.json.gz")
    for pattern in patterns:
        yield from sorted(
            item
            for item in root.rglob(pattern)
            if item.is_file() and not item.name.endswith(".meta.json")
        )


def load_episodes(path: str | Path) -> list[EpisodeSpec]:
    episodes: list[EpisodeSpec] = []
    for file_path in iter_episode_files(path):
        episodes.extend(load_episode_file(file_path))
    return episodes


def load_episodes_for_task(
    task_config: BenchmarkTaskConfig,
    episode_dataset_path: str | Path,
) -> list[EpisodeSpec]:
    """Load a task's episode dataset, dispatching on the task's own name.

    A suite's episode file is not self-describing: the published objnav records are
    EpisodeSpec-shaped but carry a *relative* ``scene.asset``, so reading them needs the task's
    scene root. Reading a file therefore needs to know which task asked, and this is the one
    place that mapping lives.

    Raises ``ValueError`` naming the task and the registered alternatives when the task has no
    loader, rather than falling back to a generic reader. A dataset silently parsed by the wrong
    loader produces episodes that look fine and score against the wrong thing.
    """
    loader = _TASK_EPISODE_LOADERS.get(task_config.name)
    if loader is None:
        available = ", ".join(sorted(_TASK_EPISODE_LOADERS))
        raise ValueError(
            f"no episode dataset loader registered for task {task_config.name!r}; available: {available}"
        )
    return loader(task_config, Path(episode_dataset_path))


def _load_insight_bench_episodes(task_config: BenchmarkTaskConfig, path: Path) -> list[EpisodeSpec]:
    """Published objnav records, bound to the user's scene root.

    The records are EpisodeSpec-shaped but carry a *relative* ``scene.asset``,
    so reading them as pre-built specs would leave every episode with no
    resolvable scene. The scene root is a task asset, like the VLN-CE one.
    """
    from ..episodes.adapters.objnav import make_objnav_episode_from_record

    scene_root = str(task_config.assets.get("scene_root") or "")
    if not scene_root:
        raise ValueError(
            f"task {task_config.name!r} needs assets['scene_root']: the published objnav "
            "episodes carry relative scene assets and the scenes are user-provided"
        )
    return [
        make_objnav_episode_from_record(record, scene_root=scene_root)
        for record in load_episode_records(path)
    ]


# Task name -> loader. Keys match `insight_bench.vln_runtime.suite.registry.available_task_names()`;
# a test asserts that, so a suite added to the registry without a loader fails there rather than at
# the first attempt to run it.
_TASK_EPISODE_LOADERS: dict[str, Callable[[BenchmarkTaskConfig, Path], list[EpisodeSpec]]] = {
    "insight_bench": _load_insight_bench_episodes,
}
