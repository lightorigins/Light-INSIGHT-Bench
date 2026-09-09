"""Resolve a published episode's relative scene asset against a user scene root.

A published episode carries ``scene.asset`` -- ``<dataset>/<scene_id>/<basename>``
-- and nothing else. The USD it names is not in the dataset download for three
of the four scene families: HM3D, MP3D and InteriorGS are licence-gated, and
this SDK never fetches them. So the episode file alone cannot say where a scene
is; only the user can, by pointing at the root they converted their scenes into.

This module is the whole of that step. It knows the on-disk layout of the four
shipped scene families (:data:`SCENE_ASSET_LAYOUTS`), joins a relative asset
under a root without following symlinks or escaping it, and **fails closed**:
a scene that is not there raises :class:`SceneAssetError` naming the scene, its
dataset and every path that was tried. It never returns a path that does not
exist, and it never leaves the asset unresolved -- the failure mode this
replaced was an episode parsing cleanly with no scene at all and scoring
against an empty room.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Literal

from insight_bench._paths import UnsafePathError, safe_join
from insight_bench.vln_runtime.episodes.schema import EpisodeSpec, SceneSpec

__all__ = [
    "SCENE_ASSET_LAYOUTS",
    "SceneAssetError",
    "SceneAssetLayout",
    "bind_scene_asset",
    "expected_scene_assets",
    "resolve_scene_asset",
    "scene_asset_tree",
    "scene_renderer",
]


class SceneAssetError(RuntimeError):
    """Raised when an episode's scene asset cannot be resolved under a scene root."""


StemRule = Literal["scene_id", "hash_suffix"]

Renderer = Literal["mesh", "neural"]
"""Which rendering path a scene family goes down.

``mesh`` families are ordinary USD geometry with baked materials and streamed
textures. ``neural`` families are Gaussian-splat captures converted to USD, and
the renderer resolves them through its neural-volume path instead. The
distinction is not cosmetic: loading a neural scene leaves persistent state
behind that changes how *mesh* scenes render for the rest of the process, which
is why the objnav runner orders scene families by this field.
"""


@dataclass(frozen=True)
class SceneAssetLayout:
    """How one scene family lays a single scene out below a scene root.

    ``basenames`` are the asset filenames that are valid entry points for a
    scene, most preferred first, as templates over the asset ``{stem}``. Several
    families publish more than one: the Habitat-GS conversions ship a ``.usdz``
    stage plus small ``.usd`` / ``_zup.usda`` stubs that reference it, and an
    episode may name any of them.

    ``stem_rule`` says how the basename stem comes from the scene id. HM3D scene
    ids are ``<5-digit index>-<scene hash>`` while the converted asset is named
    after the hash alone; everywhere else the stem is the whole scene id.

    ``renderer`` records which rendering path the family goes down (see
    :data:`Renderer`); it is a property of how the assets were produced, not of
    any one benchmark, which is why it is declared here beside the layout.
    """

    dataset: str
    scene_type: str
    basenames: tuple[str, ...]
    stem_rule: StemRule = "scene_id"
    renderer: Renderer = "mesh"

    def stem(self, scene_id: str) -> str:
        if self.stem_rule == "hash_suffix" and "-" in scene_id:
            return scene_id.split("-", 1)[1]
        return scene_id

    def assets(self, scene_id: str) -> tuple[str, ...]:
        stem = self.stem(scene_id)
        return tuple(
            f"{self.dataset}/{scene_id}/{name.format(stem=stem)}" for name in self.basenames
        )


SCENE_ASSET_LAYOUTS: dict[str, SceneAssetLayout] = {
    "habitat_gs": SceneAssetLayout(
        dataset="habitat_gs",
        scene_type="habitat_gs_usd",
        basenames=("{stem}.usd", "{stem}_zup.usda", "{stem}_final.usdz"),
        renderer="neural",
    ),
    "hm3d": SceneAssetLayout(
        dataset="hm3d",
        scene_type="hm3d_usd",
        basenames=("{stem}.usd",),
        stem_rule="hash_suffix",
    ),
    "interiorgs": SceneAssetLayout(
        dataset="interiorgs",
        scene_type="interiorgs_usd",
        basenames=("{stem}_zup.usda",),
        renderer="neural",
    ),
    "mp3d": SceneAssetLayout(
        dataset="mp3d",
        scene_type="matterport_usd",
        basenames=("{stem}.usd",),
    ),
}
"""The four scene families the published objnav suite references."""


def scene_asset_tree(scene: SceneSpec) -> str:
    """The asset tree a scene lives in: the leading segment of its published asset.

    Keyed on the asset rather than on ``scene.dataset`` on purpose. The tree is
    what the renderer actually loaded, while the dataset label is an annotation
    that a dataset build can get wrong -- and has: an earlier release tagged
    episodes pointing at the *same* USD with two different dataset names, which
    split one scene across two families. Keying on the asset guarantees
    same-asset-same-family. The label is still the fallback for a record with no
    published asset, and for one whose asset names a tree with no known layout.
    """
    if scene.asset:
        parts = PurePosixPath(scene.asset).parts
        if parts and parts[0] in SCENE_ASSET_LAYOUTS:
            return parts[0]
    return scene.dataset or "unknown"


def scene_renderer(scene: SceneSpec) -> Renderer:
    """Which rendering path *scene* goes down (see :data:`Renderer`).

    An unrecognised tree is reported as ``neural``, which is the side that
    cannot contaminate anything ordered after it.
    """
    layout = SCENE_ASSET_LAYOUTS.get(scene_asset_tree(scene))
    return layout.renderer if layout is not None else "neural"


def expected_scene_assets(dataset: str, scene_id: str) -> tuple[str, ...]:
    """Relative asset paths a scene of *dataset* may occupy under a scene root."""
    layout = SCENE_ASSET_LAYOUTS.get(dataset)
    if layout is None:
        known = ", ".join(sorted(SCENE_ASSET_LAYOUTS))
        raise SceneAssetError(
            f"scene {scene_id!r} declares dataset {dataset!r}, which has no known asset "
            f"layout; known scene datasets: {known}"
        )
    return layout.assets(scene_id)


def resolve_scene_asset(scene: SceneSpec, *, scene_root: Path) -> Path:
    """Return the existing USD for *scene* below *scene_root*, or refuse by name.

    The scene's own ``asset`` is tried first -- it is what the dataset published
    and the only value that distinguishes two equivalent entry points for the
    same stage -- then the layout's expected names. A relative asset from a
    user-supplied dataset is untrusted input, so it goes through
    :func:`~insight_bench._paths.safe_join`: an ``asset`` of ``../../etc/x`` or a
    symlinked scene directory is rejected rather than opened.
    """
    if scene.dataset is None:
        raise SceneAssetError(
            f"scene {scene.scene_id!r} declares no dataset, so its asset layout is unknown; "
            "published episode records carry scene.dataset"
        )
    candidates: list[str] = []
    if scene.asset is not None:
        candidates.append(scene.asset)
    candidates.extend(
        asset
        for asset in expected_scene_assets(scene.dataset, scene.scene_id)
        if asset not in candidates
    )

    tried: list[str] = []
    for candidate in candidates:
        try:
            path = safe_join(scene_root, candidate)
        except UnsafePathError as exc:
            raise SceneAssetError(
                f"scene {scene.scene_id!r} ({scene.dataset}) declares an unusable asset "
                f"path {candidate!r}: {exc}"
            ) from exc
        if path.is_file():
            return path
        tried.append(candidate)
    raise SceneAssetError(
        f"scene asset for {scene.scene_id!r} ({scene.dataset}) is missing under the "
        f"scene root {scene_root}: none of {', '.join(tried)} exists. Scene assets are "
        f"user-provided -- this SDK never downloads them -- so obtain the {scene.dataset} "
        "scenes, convert them, and place them at one of those relative paths (see the "
        "benchmark manifest's asset_licenses for the source and its licence terms)."
    )


def bind_scene_asset(episode: EpisodeSpec, *, scene_root: Path) -> EpisodeSpec:
    """Return *episode* with its scene's ``asset_path`` resolved under *scene_root*.

    ``asset`` is left as published, so the record stays traceable to the dataset
    it came from; only the machine-specific half is filled in.
    """
    resolved = resolve_scene_asset(episode.scene, scene_root=scene_root)
    return replace(episode, scene=replace(episode.scene, asset_path=str(resolved)))
