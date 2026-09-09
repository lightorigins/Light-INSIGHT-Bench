"""A retired coordinate must fail closed, on every surface that can reach one.

`vln-suite-v5`, `vln-suite-v5-smoke`, `internal-smoke`, `bench-tiny` and
`env-check-sim` were removed from this distribution. The risk after a removal is
not that they still work -- that is obvious and loud -- but that they fail
*open*: resolve to something else, fall back to a default, or be silently
accepted and produce a run under a different benchmark. Each check below names
the surface it covers, because "the coordinate is gone" is only true if every
entry point agrees.

`internal-smoke@1.0.0` was retired by the rename to `env-check-sim@1.0.0`, and
`env-check-sim@1.0.0` in turn by the removal of the bundled fixtures. Both are
listed here for the same reason the others are: a renamed coordinate that still
resolves is worse than one that is gone, because the run it produces is filed
under a name nobody can look up.

`bench-tiny@1.0.0` and `env-check-sim@1.0.0` were the two internal fixtures. They
are gone with their manifests, their episode files and their runners, so this
distribution now ships exactly the benchmarks it publishes and nothing else.

`objnav-suite-v1` and `objnav-suite-v2` were retired whole by the rename to
`insight-bench-v1` -- this distribution publishes one suite, not two -- so for
each of them `1.0.0`, `1.1.0`/`2.1.0` and the bare id are all coordinates that
must now be refused. `insight-bench-v1@1.0.0` is the live one and is asserted
below to be the entire published catalogue. The bare retired id is the dangerous
one: `Registry.get` resolves a bare id to the highest version *of that
benchmark*, so the failure to watch for is a stale `objnav-suite-v2` mention
quietly resolving to something and filing a run against a suite the submitter
never named.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from insight_bench.cli import main
from insight_bench.registry import Registry, is_internal_fixture
from insight_bench.runner import available_runners, run_benchmark

#: The episode file a `run` needs before it can even be parsed. Which file is
#: irrelevant here -- every case below is refused at coordinate resolution,
#: before anything reads it -- but `run` will not guess one, so it is passed.
EPISODES = Path(__file__).resolve().parents[1] / "fixtures" / "objnav_published_episodes.jsonl"

RETIRED = (
    "vln-suite-v5@1.1.0",
    "vln-suite-v5",
    "vln-suite-v5-smoke@1.0.0",
    "vln-suite-v5-smoke",
    "internal-smoke@1.0.0",
    "internal-smoke",
    "bench-tiny@1.0.0",
    "bench-tiny",
    "env-check-sim@1.0.0",
    "env-check-sim",
    # The ObjectNav-Suite naming, retired whole when the suite became
    # insight-bench-v1. These are the strings a stale reference actually says,
    # so these are the strings that have to be refused -- writing the new name
    # here would assert that the live benchmark is gone.
    "objnav-suite-v1@1.0.0",
    "objnav-suite-v1@1.1.0",
    "objnav-suite-v1",
    "objnav-suite-v2@2.0.0",
    "objnav-suite-v2@2.1.0",
    "objnav-suite-v2",
)

#: The benchmark ids above, without versions -- what a stale mention looks like in
#: data or in a manifest.
RETIRED_IDS = (
    "vln-suite-v5",
    "vln-suite-v5-smoke",
    "internal-smoke",
    "bench-tiny",
    "env-check-sim",
    "objnav-suite-v1",
    "objnav-suite-v2",
)

#: Runners that existed only to drive a retired coordinate. A registered runner
#: with no coordinate is how one comes back.
RETIRED_RUNNERS = ("deterministic-json-v1", "isaac-camera-walk-v1")

#: The one coordinate this distribution publishes.
PUBLISHED = "insight-bench-v1@1.0.0"


@pytest.mark.parametrize("coordinate", RETIRED)
def test_the_registry_refuses_rather_than_resolving_to_something_else(coordinate):
    registry = Registry.load()
    with pytest.raises((KeyError, ValueError)) as caught:
        registry.get(coordinate)
    # The refusal must name what was asked for; a bare KeyError sends the reader
    # looking for a bug in their own code.
    message = str(caught.value)
    assert any(retired_id in message for retired_id in RETIRED_IDS) or coordinate in message


@pytest.mark.parametrize("coordinate", RETIRED)
def test_the_cli_run_exits_non_zero(coordinate, tmp_path, capsys):
    exit_code = main(
        ["run", coordinate, "--episodes", str(EPISODES), "--scene-root", str(tmp_path)]
    )
    assert exit_code != 0
    # And it exits *because the coordinate is gone*. Asserting only the status
    # would pass just as well on a run that got further and failed for some
    # unrelated reason, which is exactly what a fail-open regression looks like.
    assert "unknown benchmark" in capsys.readouterr().err


@pytest.mark.parametrize("coordinate", RETIRED)
def test_run_benchmark_raises_rather_than_falling_back_to_a_default(coordinate):
    with pytest.raises(Exception) as caught:
        run_benchmark(coordinate)
    assert not isinstance(caught.value, AssertionError)


def test_no_retired_coordinate_is_in_the_catalog_or_the_published_catalog():
    registry = Registry.load()
    registered = {m.coordinate for m in registry.catalog()}
    published = {m.coordinate for m in registry.public_catalog()}
    for coordinate in RETIRED:
        assert coordinate not in registered
        assert coordinate not in published
    # Not merely absent from the coordinates: absent from the manifests entirely,
    # so a retired id cannot survive as the benchmark_id of a renamed entry.
    assert not {m.benchmark_id for m in registry.catalog()} & set(RETIRED_IDS)


def test_no_retired_runner_is_registered():
    registered = available_runners()
    assert not any("vln-suite" in name or "vln_suite" in name for name in registered)
    for runner_id in RETIRED_RUNNERS:
        assert runner_id not in registered


def test_the_registry_index_does_not_mention_a_retired_coordinate():
    """The index is data; a stale entry there would resolve before any code ran."""
    from insight_bench.registry import builtin_registry_root

    index = (builtin_registry_root() / "index.json").read_text(encoding="utf-8")
    for retired_id in RETIRED_IDS:
        assert retired_id not in index


@pytest.mark.parametrize("retired_id", RETIRED_IDS)
def test_a_retired_benchmark_is_gone_from_the_shipped_registry_tree(retired_id):
    """Removal from the bundled registry is *how* a coordinate is retired here.

    There is no ``retired`` field on a manifest. Leaving the directory in place
    while dropping its index entry would ship bytes nothing pins -- manifests
    and, for the fixtures, episode files -- which is how the next person re-adds
    one by accident.
    """
    from insight_bench.registry import builtin_registry_root

    assert not (builtin_registry_root() / retired_id).exists()


def test_the_published_catalog_is_exactly_the_insight_bench():
    """What this distribution offers, stated as a list on purpose.

    The published catalog is a product claim: every coordinate in it is a
    benchmark someone may submit a result against. ``insight-bench-v1@1.0.0`` is
    that claim and the whole of it; anything else appearing here -- or it
    vanishing -- is a change somebody has to make deliberately.
    """
    registry = Registry.load()
    assert [manifest.coordinate for manifest in registry.public_catalog()] == [
        "insight-bench-v1@1.0.0",
    ]
    for manifest in registry.public_catalog():
        assert not is_internal_fixture(manifest), manifest.coordinate


def test_nothing_is_registered_but_unpublished():
    """The fixtures are gone, so the two catalogs are now the same list.

    :func:`is_internal_fixture` and the ``include_fixtures`` split stay -- they
    are how a fixture would be kept out of the product claim if one were added
    back -- but nothing in this distribution uses them any more. A coordinate
    that resolves and runs while being absent from the published catalog is
    exactly what that machinery hides, so assert there is none.
    """
    registry = Registry.load()
    assert registry.catalog() == registry.public_catalog()
    assert [m.coordinate for m in registry.catalog() if is_internal_fixture(m)] == []


# --- retired versions of benchmarks that still exist ------------------------


def test_the_bare_id_resolves_to_the_published_version():
    """A bare id must land on what is published, and on nothing else.

    ``Registry.get`` resolves an id with no version to the highest version of
    that benchmark. That is the convenience a stale reference rides in on, so
    the property is asserted directly rather than left implied: the bare id
    resolves, it resolves to the coordinate in the published catalogue, and the
    tree holds no other version of it for a future bare lookup to drift onto.
    """
    registry = Registry.load()
    published = [manifest.coordinate for manifest in registry.public_catalog()]
    assert published == [PUBLISHED], published

    benchmark_id, version = PUBLISHED.split("@")
    assert registry.get(benchmark_id).coordinate == PUBLISHED

    from insight_bench.registry import builtin_registry_root

    versions = sorted(
        path.name for path in (builtin_registry_root() / benchmark_id).iterdir() if path.is_dir()
    )
    assert versions == [version], versions
