"""Published insight-bench episode adapter: scene binding and the success radius.

The published objnav records are already ``EpisodeSpec``-shaped, so this
adapter does exactly two things the raw record cannot do for itself.

**Scene binding.** ``scene.asset`` is relative to a scene root the user
supplies; :mod:`insight_bench.vln_runtime.episodes.scenes` resolves it and
fails closed when the scene is absent.

**The success radius.** The suite scores at 2.0 m indoors and 3.0 m outdoors,
and the discriminator is ``metadata.scene_class`` -- not ``scene.dataset``.
That distinction is the whole reason the rule is applied here rather than
trusted from the file: the Habitat-GS family is mostly outdoor captures but
contains indoor ones, and reading the radius off the dataset mis-scored 51
episodes of one release. Records that carry a ``scene_class`` therefore have
their radius *derived* from it and written into all three places the runtime
reads a radius (``goal_radius_m``, each subtask's ``radius_m``, and
``metadata.navnuances.success_dist_m``, which is the one the NavNuances
LR-towards protocol actually consults), so the three cannot disagree.

Records with no ``scene_class`` keep the radius they declare: the first suite
release predates the indoor/outdoor policy and scored per scene family, and
re-deriving its radii would silently restate its published numbers.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from insight_bench.vln_runtime.episodes.scenes import bind_scene_asset
from insight_bench.vln_runtime.episodes.schema import EpisodeSpec

__all__ = [
    "INDOOR_SUCCESS_RADIUS_M",
    "OUTDOOR_SCENE_CLASS",
    "OUTDOOR_SUCCESS_RADIUS_M",
    "load_objnav_episodes",
    "make_objnav_episode_from_record",
    "success_radius_m",
]

OUTDOOR_SCENE_CLASS = "Outdoor"
"""The one ``metadata.scene_class`` value that selects the outdoor radius.

Compared case-insensitively. The published suite spells it ``Outdoor``; the
retired ``1.0.0``/``2.0.0`` files spelled it ``outdoor``, and a record carrying
a scene class never takes the "keep the declared radius" path, so an unmatched
spelling would not refuse -- it would quietly score an outdoor episode at the
indoor 2.0 m radius.
"""

INDOOR_SUCCESS_RADIUS_M = 2.0
OUTDOOR_SUCCESS_RADIUS_M = 3.0


def success_radius_m(episode: EpisodeSpec) -> float:
    """The radius this episode is scored at, from its scene class when it has one."""
    scene_class = episode.metadata.get("scene_class")
    if scene_class is None:
        return float(episode.goal_radius_m)
    if str(scene_class).casefold() == OUTDOOR_SCENE_CLASS.casefold():
        return OUTDOOR_SUCCESS_RADIUS_M
    return INDOOR_SUCCESS_RADIUS_M


def make_objnav_episode_from_record(
    record: dict[str, Any], *, scene_root: str | Path
) -> EpisodeSpec:
    """Bind one published objnav record to a local scene and its success radius."""
    episode = bind_scene_asset(EpisodeSpec.from_dict(record), scene_root=Path(scene_root))
    return _with_success_radius(episode, success_radius_m(episode))


def load_objnav_episodes(path: str | Path, *, scene_root: str | Path) -> list[EpisodeSpec]:
    """Load a published objnav episode file (JSON or JSONL) bound to *scene_root*."""
    from insight_bench.vln_runtime.episodes.loader import load_episode_records

    return [
        make_objnav_episode_from_record(record, scene_root=scene_root)
        for record in load_episode_records(path)
    ]


def _with_success_radius(episode: EpisodeSpec, radius_m: float) -> EpisodeSpec:
    navnuances = {**(episode.metadata.get("navnuances") or {}), "success_dist_m": radius_m}
    return replace(
        episode,
        goal_radius_m=radius_m,
        subtasks=[replace(subtask, radius_m=radius_m) for subtask in episode.subtasks],
        metadata={**episode.metadata, "navnuances": navnuances},
    )
