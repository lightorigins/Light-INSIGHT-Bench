"""The published objnav coordinate, pinned to the values other repositories use.

These are cross-repository constants. The submission service validates a
result against the benchmark row it names, so a disagreement here is not a
cosmetic difference: it is every submission for that suite rejected. The audit
that produced this file found exactly that -- the service pinned
``benchmark_version`` "2.0" against a contract that requires three-part semver,
and ``primary_metric`` "Success Rate" against a runner that emits
``success-rate``. Each assertion below is one of those pairs, stated on the SDK
side.
"""

from __future__ import annotations

from insight_bench.registry import Registry, is_internal_fixture
from insight_bench.simulator.base import SIM_BACKENDS
from insight_bench.vis import INSTR_TYPE_ORDER, SCENE_CLASS_ORDER

V2 = "insight-bench-v1@1.0.0"

# The values the submission service and the leaderboard must agree with.
PINNED = {
    "benchmark_id": "insight-bench-v1",
    "version": "1.0.0",
    "episode_count": 1097,
    "dataset_sha256": "a1828911ce642521d4bdd38b6c23a4a7a9f2623cbced8688f88a578a6a3c55af",
}

PRIMARY_METRIC = "success-rate"
SCENE_FAMILIES = ("habitat_gs", "hm3d", "interiorgs", "mp3d")


def test_the_suite_is_published_at_the_pinned_coordinate() -> None:
    registry = Registry.load()
    manifest = registry.get(V2)
    assert manifest.benchmark_id == PINNED["benchmark_id"]
    assert manifest.version == PINNED["version"]
    # Three-part semver, because the contract requires it and the service's
    # row said "1.0"/"2.0".
    assert manifest.version.count(".") == 2
    assert not is_internal_fixture(manifest), "a published suite is not a fixture"
    assert manifest.coordinate in {m.coordinate for m in registry.public_catalog()}


def test_the_primary_metric_is_the_key_the_runner_emits() -> None:
    manifest = Registry.load().get(V2)
    assert manifest.metadata["primary_metric"] == PRIMARY_METRIC
    # Declared first, and declared at all: a metric the result carries but
    # the manifest omits is a metric no consumer knows to read.
    assert manifest.metrics[0] == PRIMARY_METRIC


def test_the_pinned_episode_count_and_published_digest_are_declared() -> None:
    manifest = Registry.load().get(V2)
    assert manifest.metadata["episode_count"] == PINNED["episode_count"]
    # The PUBLISHED file's digest -- the one a user can compute over their
    # own download -- never the internal scored file's.
    assert manifest.dataset.sha256 == PINNED["dataset_sha256"]


def test_the_dataset_is_declared_user_provided_and_still_runnable() -> None:
    manifest = Registry.load().get(V2)
    assert manifest.runnable is True
    assert manifest.dataset.availability == "user-provided"
    assert manifest.dataset.path is None, "nothing is bundled for this suite"
    assert manifest.license.status == "declared"
    assert list(manifest.requires_configuration) == [
        "episode-dataset",
        "scene-root",
        "policy-server",
    ]


def test_every_scene_family_carries_its_licence_and_a_way_to_obtain_it() -> None:
    manifest = Registry.load().get(V2)
    licenses = {item.family: item for item in manifest.asset_licenses}
    assert tuple(sorted(licenses)) == SCENE_FAMILIES
    for family, declaration in licenses.items():
        assert declaration.license.strip(), family
        assert declaration.obtain_url.startswith("https://"), family
    # Three of the four are licence-gated; saying so is what tells a reader
    # whether they can reproduce the number at all.
    assert licenses["hm3d"].tier == "eula_gated"
    assert licenses["mp3d"].tier == "eula_gated"
    assert licenses["interiorgs"].tier == "eula_gated"
    assert licenses["habitat_gs"].tier == "public"


def test_the_official_backend_line_is_isaac_51() -> None:
    descriptors = {item.backend_id: item for item in SIM_BACKENDS}
    manifest = Registry.load().get(V2)
    variants = {variant.backend_id: variant for variant in manifest.runtime_variants}
    assert set(variants) == set(descriptors)
    official = variants["isaac-5.1"]
    assert official.status == "official"
    assert official.default is True
    assert (official.isaac_sim, official.isaac_lab) == ("5.1.0", "2.3.0")
    assert variants["isaac-6.0"].status == "experimental"


def test_the_declared_task_family_is_one_the_runner_can_drive() -> None:
    from insight_bench.runners.insight_bench import _EXECUTABLE_FAMILIES, RUNNER_ID

    manifest = Registry.load().get(V2)
    assert manifest.runtime.runner == RUNNER_ID
    assert manifest.metadata["task_family"] in _EXECUTABLE_FAMILIES


def test_comparability_is_scoped_to_the_official_backend_at_the_declared_rig() -> None:
    # A success rate is only a number about *something*, and what it is about
    # is one backend line at one camera rig. Runs from a different backend or a
    # different rig are a different measurement, and the note has to say so
    # rather than leaving a leaderboard to assume the rows are interchangeable.
    notes = " ".join(Registry.load().get(V2).comparability_notes)
    assert "Comparable only across runs on the isaac-5.1 official backend" in notes
    assert "480x270 RGB, 1.0 m eye height, HFOV 120 deg, 0.05-200 m clip range" in notes


def test_the_v2_radius_policy_is_written_down_with_its_discriminator() -> None:
    notes = " ".join(Registry.load().get(V2).comparability_notes)
    assert "2.0 m indoors and 3.0 m outdoors" in notes
    # Naming the discriminator matters: reading the radius off scene.dataset is
    # the mistake this policy exists to prevent.
    assert "metadata.scene_class" in notes


def test_the_declared_rig_does_not_claim_to_replay_the_historical_ones() -> None:
    # The constituent episode sets were scored at two different rigs (300 steps
    # / 200 m far plane outdoors, 200 / 100 m indoors). This coordinate declares
    # one uniform rig, which makes it a new measurement -- and a leaderboard row
    # that implied otherwise would be inviting a comparison that does not hold.
    notes = " ".join(Registry.load().get(V2).comparability_notes)
    assert "300-step" in notes
    assert "going forward" in notes
    assert "200 steps with a 100 m" in notes
    assert "not expected to reproduce" in notes


def test_the_single_process_scene_family_order_is_disclosed() -> None:
    # Loading a Gaussian-splat stage leaves renderer state that persists for the
    # life of the process, so the order families render in is part of what the
    # number means. A reader of the row has to be able to find that out.
    notes = " ".join(Registry.load().get(V2).comparability_notes)
    assert "before" in notes and "Gaussian-splat" in notes
    # Including the half that is *not* fixed by ordering.
    assert "not measured" in notes


def test_the_label_vocabulary_the_manifest_declares_is_the_one_vis_charts() -> None:
    """One vocabulary, pinned across the two surfaces that each hard-code it.

    ``dataset.notice`` enumerates the ``scene_class`` and ``instr_type`` values a
    reader will find in the published file, and ``vis`` keys its two radar
    charts on the same values via ``SCENE_CLASS_ORDER``/``INSTR_TYPE_ORDER``.
    Nothing derives one from the other, and drift between them is silent: an
    unlisted value still renders, just sorted in after the listed ones, so the
    charts lose the designed order without failing. That is exactly what
    happened when the labels were translated -- both tuples still held the
    Chinese and lowercase vocabulary and matched nothing.
    """
    notice = Registry.load().get(V2).dataset.notice
    for value in SCENE_CLASS_ORDER:
        assert value in notice, f"{V2}: scene_class {value} is not declared"
    for value in INSTR_TYPE_ORDER:
        assert value in notice, f"{V2}: instr_type {value} is not declared"


def test_the_outdoor_scene_class_the_scorer_matches_is_a_declared_value() -> None:
    """The scorer's one magic string has to be a value the file actually carries.

    ``success_radius_m`` compares ``metadata.scene_class`` against
    ``OUTDOOR_SCENE_CLASS`` to pick 3.0 m over 2.0 m. A record carrying any
    scene class never takes the "keep the declared radius" path, so a spelling
    that matches nothing does not refuse -- it scores every outdoor episode at
    the indoor radius.
    """
    from insight_bench.vln_runtime.episodes.adapters.objnav import OUTDOOR_SCENE_CLASS

    assert OUTDOOR_SCENE_CLASS in SCENE_CLASS_ORDER
    assert OUTDOOR_SCENE_CLASS in Registry.load().get(V2).dataset.notice
