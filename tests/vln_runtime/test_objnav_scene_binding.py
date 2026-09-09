"""Binding published objnav episode records to real scenes on a real machine.

Every record here is a verbatim line from the published episode file, one per
(scene family, asset suffix) combination the four datasets actually use, so
these tests fail if the published shape and the parser ever disagree again.
The regression they exist for: ``SceneSpec.from_dict`` read only
``asset_path``, the published records carry ``asset``, and all 1097 episodes
parsed cleanly with no scene at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from insight_bench.vln_runtime.episodes import SceneSpec
from insight_bench.vln_runtime.episodes.adapters.objnav import (
    INDOOR_SUCCESS_RADIUS_M,
    OUTDOOR_SUCCESS_RADIUS_M,
    load_objnav_episodes,
    make_objnav_episode_from_record,
)
from insight_bench.vln_runtime.episodes.loader import load_episode_records, load_episodes_for_task
from insight_bench.vln_runtime.episodes.scenes import (
    SceneAssetError,
    expected_scene_assets,
    resolve_scene_asset,
    scene_asset_tree,
    scene_renderer,
)
from insight_bench.vln_runtime.suite import get_task_config

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "objnav_published_episodes.jsonl"

# The four on-disk layouts, as they appear in the published records.
PUBLISHED_ASSETS = (
    "habitat_gs/scene56/scene56.usd",
    "habitat_gs/scene58/scene58_zup.usda",
    "habitat_gs/scene63/scene63_zup.usda",
    "hm3d/00013-sfbj7jspYWj/sfbj7jspYWj.usd",
    "interiorgs/interior_0405_840145/interior_0405_840145_zup.usda",
    "mp3d/8194nk5LbLH/8194nk5LbLH.usd",
)


@pytest.fixture
def records() -> list[dict]:
    return load_episode_records(FIXTURE)


@pytest.fixture
def scene_root(tmp_path: Path) -> Path:
    root = tmp_path / "scenes"
    for asset in PUBLISHED_ASSETS:
        path = root / asset
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#usda 1.0\n", encoding="utf-8")
    return root


# --- the published record shape --------------------------------------------


def test_the_published_scene_asset_survives_parsing(records) -> None:
    scenes = [SceneSpec.from_dict(record["scene"]) for record in records]
    assert {scene.asset for scene in scenes} == set(PUBLISHED_ASSETS)
    # `asset_path` is the resolved half and is genuinely absent until binding.
    assert all(scene.asset_path is None for scene in scenes)
    assert {scene.dataset for scene in scenes} == {"habitat_gs", "hm3d", "interiorgs", "mp3d"}


def test_asset_path_records_still_parse_and_round_trip() -> None:
    """The converter-produced shape keeps working; the new field is additive."""
    scene = SceneSpec.from_dict({"scene_id": "s", "asset_path": "/scenes/s/s.usd"})
    assert scene.asset_path == "/scenes/s/s.usd"
    assert scene.asset is None
    assert SceneSpec.from_dict(scene.to_dict()) == scene


# --- the four layouts -------------------------------------------------------


@pytest.mark.parametrize(
    ("dataset", "scene_id", "expected"),
    [
        ("hm3d", "00013-sfbj7jspYWj", "hm3d/00013-sfbj7jspYWj/sfbj7jspYWj.usd"),
        ("mp3d", "8194nk5LbLH", "mp3d/8194nk5LbLH/8194nk5LbLH.usd"),
        (
            "interiorgs",
            "interior_0405_840145",
            "interiorgs/interior_0405_840145/interior_0405_840145_zup.usda",
        ),
        ("habitat_gs", "scene56", "habitat_gs/scene56/scene56.usd"),
    ],
)
def test_each_dataset_layout_is_derivable_from_the_scene_id(dataset, scene_id, expected) -> None:
    # HM3D is the one that is not just the scene id: the folder is
    # <index>-<hash> and the converted stage is named after the hash alone.
    assert expected_scene_assets(dataset, scene_id)[0] == expected


def test_every_published_asset_resolves_under_a_scene_root(records, scene_root) -> None:
    resolved = {
        str(resolve_scene_asset(SceneSpec.from_dict(record["scene"]), scene_root=scene_root))
        for record in records
    }
    assert resolved == {str(scene_root / asset) for asset in PUBLISHED_ASSETS}


def test_a_scene_with_no_declared_asset_falls_back_to_its_layout(scene_root) -> None:
    scene = SceneSpec(scene_id="8194nk5LbLH", dataset="mp3d")
    assert resolve_scene_asset(scene, scene_root=scene_root) == (
        scene_root / "mp3d/8194nk5LbLH/8194nk5LbLH.usd"
    )


# --- failing closed ---------------------------------------------------------


def test_a_missing_scene_names_the_file_and_its_dataset(records, tmp_path) -> None:
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    record = next(record for record in records if record["scene"]["dataset"] == "hm3d")
    with pytest.raises(SceneAssetError) as caught:
        make_objnav_episode_from_record(record, scene_root=empty_root)
    message = str(caught.value)
    assert "00013-sfbj7jspYWj" in message
    assert "hm3d" in message
    assert "hm3d/00013-sfbj7jspYWj/sfbj7jspYWj.usd" in message
    # And it must say the SDK will not fetch it, or the reader waits for a download.
    assert "never downloads" in message


def test_an_unknown_scene_dataset_is_refused_not_guessed() -> None:
    with pytest.raises(SceneAssetError, match="no known asset layout"):
        resolve_scene_asset(SceneSpec(scene_id="x", dataset="martian_gs"), scene_root=Path("/"))


def test_a_scene_with_no_dataset_is_refused() -> None:
    with pytest.raises(SceneAssetError, match="declares no dataset"):
        resolve_scene_asset(SceneSpec(scene_id="x"), scene_root=Path("/"))


def test_an_escaping_asset_path_is_rejected(scene_root) -> None:
    scene = SceneSpec(scene_id="s", dataset="mp3d", asset="../../../etc/passwd")
    with pytest.raises(SceneAssetError, match="unusable asset path"):
        resolve_scene_asset(scene, scene_root=scene_root)


def test_a_symlinked_scene_is_rejected(scene_root, tmp_path) -> None:
    target = tmp_path / "elsewhere"
    target.mkdir()
    (target / "s.usd").write_text("#usda 1.0\n", encoding="utf-8")
    link = scene_root / "mp3d" / "linked"
    link.symlink_to(target)
    scene = SceneSpec(scene_id="linked", dataset="mp3d", asset="mp3d/linked/s.usd")
    with pytest.raises(SceneAssetError, match="unusable asset path"):
        resolve_scene_asset(scene, scene_root=scene_root)


# --- which renderer path a family goes down ---------------------------------


def test_every_published_family_is_classified_by_renderer(records) -> None:
    # The objnav runner orders mesh families ahead of neural ones because a
    # Gaussian-splat load leaves state that changes how mesh scenes render for
    # the rest of the process. That ordering is only as good as this table.
    classified = {
        scene_asset_tree(SceneSpec.from_dict(record["scene"])): scene_renderer(
            SceneSpec.from_dict(record["scene"])
        )
        for record in records
    }
    assert classified == {
        "habitat_gs": "neural",
        "hm3d": "mesh",
        "interiorgs": "neural",
        "mp3d": "mesh",
    }


def test_the_family_is_the_asset_tree_not_the_dataset_label(scene_root) -> None:
    # A dataset label is an annotation a build can get wrong, and one earlier
    # release tagged episodes pointing at the same USD with two names. The tree
    # the renderer actually loaded is the one that decides the family.
    mislabelled = SceneSpec(
        scene_id="scene56", dataset="hm3d", asset="habitat_gs/scene56/scene56.usd"
    )
    assert scene_asset_tree(mislabelled) == "habitat_gs"
    assert scene_renderer(mislabelled) == "neural"
    # With no published asset the label is all there is, and it is used.
    assert scene_asset_tree(SceneSpec(scene_id="x", dataset="mp3d")) == "mp3d"
    assert scene_renderer(SceneSpec(scene_id="x", dataset="mp3d")) == "mesh"


def test_an_unclassifiable_scene_sorts_with_the_neural_families() -> None:
    # It cannot contaminate anything ordered after it, which is the safe side
    # to guess on. Resolution refuses such a scene long before this matters.
    assert scene_renderer(SceneSpec(scene_id="x", dataset="martian_gs")) == "neural"


# --- the success radius -----------------------------------------------------


def test_the_success_radius_comes_from_the_scene_class(records, scene_root) -> None:
    episodes = {
        episode.episode_id: episode
        for episode in (
            make_objnav_episode_from_record(record, scene_root=scene_root) for record in records
        )
    }
    outdoor = [
        episode for episode in episodes.values() if episode.metadata.get("scene_class") == "Outdoor"
    ]
    indoor = [
        episode
        for episode in episodes.values()
        if episode.metadata.get("scene_class") not in (None, "Outdoor")
    ]
    assert outdoor and indoor
    for episode in outdoor:
        assert episode.goal_radius_m == OUTDOOR_SUCCESS_RADIUS_M
    for episode in indoor:
        assert episode.goal_radius_m == INDOOR_SUCCESS_RADIUS_M
    # The Habitat-GS family is not the discriminator: scene63 is an indoor
    # capture in an otherwise outdoor family, and reading the radius off
    # scene.dataset mis-scored exactly this case in an earlier release.
    indoor_gs = episodes["objnav_scene63_103"]
    assert indoor_gs.scene.dataset == "habitat_gs"
    assert indoor_gs.goal_radius_m == INDOOR_SUCCESS_RADIUS_M


def test_the_radius_is_written_everywhere_the_runtime_reads_one(records, scene_root) -> None:
    # Three places carry a radius and the NavNuances protocol reads the third.
    # Leaving them free to disagree is how a suite scores at a radius nobody
    # declared.
    for record in records:
        episode = make_objnav_episode_from_record(record, scene_root=scene_root)
        radius = episode.goal_radius_m
        assert episode.metadata["navnuances"]["success_dist_m"] == radius
        assert [subtask.radius_m for subtask in episode.subtasks] == [radius] * len(
            episode.subtasks
        )


def test_a_tampered_radius_is_overridden_by_the_scene_class(records, scene_root) -> None:
    record = json.loads(
        json.dumps(next(r for r in records if r["metadata"]["scene_class"] == "Outdoor"))
    )
    record["goal_radius_m"] = 2.0
    record["metadata"]["navnuances"]["success_dist_m"] = 2.0
    episode = make_objnav_episode_from_record(record, scene_root=scene_root)
    assert episode.goal_radius_m == OUTDOOR_SUCCESS_RADIUS_M
    assert episode.metadata["navnuances"]["success_dist_m"] == OUTDOOR_SUCCESS_RADIUS_M


def test_records_with_no_scene_class_keep_their_declared_radius(records, scene_root) -> None:
    # The retired v1 suite predated the indoor/outdoor policy and scored per
    # scene family (2.0 / 2.5 / 3.0 m). Its records are still on disk, and
    # re-deriving those radii would silently restate its published numbers.
    record = json.loads(json.dumps(records[0]))
    del record["metadata"]["scene_class"]
    record["goal_radius_m"] = 2.5
    record["metadata"]["navnuances"]["success_dist_m"] = 2.5
    episode = make_objnav_episode_from_record(record, scene_root=scene_root)
    assert episode.goal_radius_m == 2.5


# --- the loader seam --------------------------------------------------------


def test_the_task_loader_binds_the_scene_root_it_is_configured_with(scene_root) -> None:
    from dataclasses import replace

    task_config = get_task_config("insight_bench")
    configured = replace(task_config, assets={**task_config.assets, "scene_root": str(scene_root)})
    episodes = load_episodes_for_task(configured, FIXTURE)
    assert len(episodes) == 7
    assert all(episode.scene.asset_path is not None for episode in episodes)
    assert episodes == load_objnav_episodes(FIXTURE, scene_root=scene_root)


def test_the_task_loader_refuses_an_unconfigured_scene_root() -> None:
    with pytest.raises(ValueError, match="assets\\['scene_root'\\]"):
        load_episodes_for_task(get_task_config("insight_bench"), FIXTURE)


# --- the published scene-class vocabulary -----------------------------------


def test_the_published_outdoor_scene_class_selects_the_outdoor_radius(records, scene_root) -> None:
    """The vocabulary the published files actually carry, spelled as they spell it.

    ``insight-bench-v1@1.0.0``, the one published suite, labels scene classes
    in English -- ``Outdoor``, not ``outdoor``. Matching the old spelling here
    fails *open* in the worst possible way: an outdoor episode keeps a
    ``scene_class``, so it never takes the "no class, keep the declared radius"
    path, and instead falls through to the indoor 2.0 m branch. Every outdoor
    episode of the suite would then be scored at a radius nobody declared, with
    nothing in the output saying so.
    """
    record = json.loads(
        json.dumps(next(r for r in records if r["metadata"]["scene_class"] == "Outdoor"))
    )
    episode = make_objnav_episode_from_record(record, scene_root=scene_root)
    assert episode.goal_radius_m == OUTDOOR_SUCCESS_RADIUS_M
    assert episode.metadata["navnuances"]["success_dist_m"] == OUTDOOR_SUCCESS_RADIUS_M


def test_the_retired_lowercase_spelling_is_still_read_as_outdoor(records, scene_root) -> None:
    """A retired file's ``outdoor`` must not become an indoor score.

    The retired ``1.0.0``/``2.0.0`` files spelled this value lowercase. Nothing
    was ever published against them, but a local run directory or a stored
    record can still hold that spelling, and turning it into a 2.0 m score
    silently is worse than refusing it.
    """
    record = json.loads(
        json.dumps(next(r for r in records if r["metadata"]["scene_class"] == "Outdoor"))
    )
    record["metadata"]["scene_class"] = "outdoor"
    episode = make_objnav_episode_from_record(record, scene_root=scene_root)
    assert episode.goal_radius_m == OUTDOOR_SUCCESS_RADIUS_M
