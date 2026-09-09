"""Generic runner dispatch: what is registered, and what it refuses.

The three tests this file used to hold drove ``bench-tiny`` through
``EchoAdapter``: a deterministic text fixture, a reference adapter over the same
manifest, and an unknown-coordinate refusal. The fixture benchmark, both
adapters and the ``deterministic-json-v1`` runner behind them no longer exist,
so the first two are gone with the code they exercised. What survives is what
:mod:`insight_bench.runner` still does for every benchmark: resolve a coordinate,
look up its declared runner, and locate its episode file -- refusing by name at
each step rather than falling back to something that would produce a run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from insight_bench.registry import RegistryError
from insight_bench.runner import RunContext, RunnerError, available_runners, run_benchmark


def test_the_objnav_http_runner_is_the_only_one_registered() -> None:
    """One product, one runner. Asserted as the exact tuple on purpose.

    ``deterministic-json-v1`` and ``isaac-camera-walk-v1`` were retired with the
    benchmarks that declared them. A runner reappearing here is a manifest
    becoming executable again, which is a decision somebody has to make
    deliberately rather than discover.
    """
    assert available_runners() == ("objnav-http-policy-v1",)


@pytest.mark.parametrize("coordinate", ["bench-tiny@1.0.0", "bench-tiny", "vln-suite-v5@1.1.0"])
def test_an_unknown_coordinate_is_refused_rather_than_guessed(coordinate: str) -> None:
    """A coordinate the registry does not hold must refuse, not resolve to a neighbour."""
    with pytest.raises(RegistryError) as caught:
        run_benchmark(coordinate)
    assert coordinate.split("@")[0] in str(caught.value)


def test_a_user_provided_episode_file_is_required_before_anything_starts() -> None:
    """The published suite ships no episodes, so a run without one refuses by name.

    This is the step between "the coordinate resolved" and "the runner ran": no
    simulator, no policy server and no scene root are reached, and the message
    has to name the flag, because the file is something only the user can supply.
    """
    with pytest.raises(RunnerError, match="--episodes"):
        run_benchmark("insight-bench-v1@1.0.0")


def test_an_episode_file_that_is_not_a_readable_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(RunnerError, match="is not readable"):
        run_benchmark(
            "insight-bench-v1@1.0.0",
            context=RunContext(episode_dataset=tmp_path / "absent.jsonl"),
        )
