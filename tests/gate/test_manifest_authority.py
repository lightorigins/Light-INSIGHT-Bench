"""The typed registry manifest is the single authoritative rig record.

Suite facts (asset licences, runtime variants and digests, comparability
rules) live as typed manifest fields — not as opaque YAML copies in
``metadata`` — and the in-code backend descriptors must agree with the
manifest, so promotion stays a registry data change.
"""

from __future__ import annotations

from insight_bench.registry import Registry
from insight_bench.simulator.base import SIM_BACKENDS

OBJNAV_V2 = "insight-bench-v1@1.0.0"

# Every coordinate that declares a simulator rig. Listing them is the point:
# a new suite that forgets its runtime_variants fails here rather than at the
# first attempt to promote a backend.
SIMULATOR_COORDINATES = (OBJNAV_V2,)


def test_manifest_runtime_variants_agree_with_the_code_side_descriptors() -> None:
    registry = Registry.load()
    for coordinate in SIMULATOR_COORDINATES:
        manifest = registry.get(coordinate)
        variants = {variant.backend_id: variant for variant in manifest.runtime_variants}
        descriptors = {descriptor.backend_id: descriptor for descriptor in SIM_BACKENDS}
        assert set(variants) == set(descriptors), coordinate
        for backend_id, variant in variants.items():
            descriptor = descriptors[backend_id]
            assert variant.status == descriptor.status, (coordinate, backend_id)
            assert variant.isaac_sim == descriptor.isaac_sim, (coordinate, backend_id)
            assert variant.isaac_lab == descriptor.isaac_lab, (coordinate, backend_id)
            assert variant.image_digest == descriptor.image_digest, (coordinate, backend_id)
            assert variant.default == descriptor.default, (coordinate, backend_id)


def test_manifests_carry_no_opaque_suite_copies() -> None:
    # The metadata block stays a small set of scalar facts; structured suite
    # data must use the typed fields, never a YAML/JSON blob copy.
    registry = Registry.load()
    allowed = {
        OBJNAV_V2: {
            "episode_count",
            "primary_metric",
            "simulation_assets_required",
            "task_family",
        },
    }
    for coordinate, keys in allowed.items():
        manifest = registry.get(coordinate)
        assert set(manifest.metadata) == keys, coordinate
        for value in manifest.metadata.values():
            assert not isinstance(value, str) or len(value) < 200, coordinate


def test_structured_suite_facts_use_the_typed_fields() -> None:
    """The objnav suite's licences, rigs and口径 are typed, not metadata prose.

    ``metadata`` above is four scalars. Everything structured a reader needs --
    which scene families the suite loads and under what terms, which simulator
    lines exist, and what the numbers may be compared with -- lives in
    ``asset_licenses`` / ``runtime_variants`` / ``comparability_notes``, where
    it is validated and machine-readable.
    """
    registry = Registry.load()
    for coordinate in (OBJNAV_V2,):
        manifest = registry.get(coordinate)
        assert len(manifest.asset_licenses) == 4, coordinate
        assert len(manifest.runtime_variants) == len(SIM_BACKENDS), coordinate
        assert manifest.comparability_notes, coordinate


def test_the_manifest_schema_is_published_for_downstream_consumers() -> None:
    from insight_bench.contracts import PUBLIC_SCHEMAS, BenchmarkManifest

    assert PUBLIC_SCHEMAS["benchmark-manifest"] is BenchmarkManifest
