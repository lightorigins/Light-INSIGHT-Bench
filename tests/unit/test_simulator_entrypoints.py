"""A simulator line can be named through the CLI, and the name is what runs.

This file used to pin two things. The first was the pair of shipped in-process
baselines -- a stand-still policy and a constant-forward one -- and their
selection through ``--adapter``. Those policies, the adapters that wrapped them
and the flag itself are gone: the model of a run is now an HTTP policy server
the user starts, identified by its own ``GET /health``, so there is no
in-process baseline left to select and no text adapter left to refuse. Those
tests went with the code.

The second is still true and still needs pinning: a simulator path no published
entry point reaches is not a path. So what remains here is the naming half --
which lines ``run`` offers, which field ``--run-dir`` lands in, and that a named
line is the line the runner resolves rather than one it reads and ignores. How a
named line then loads (or honestly refuses to) on a machine without Isaac is
``tests/gate/test_isaac_runtime_gating.py``'s subject, not this file's.
"""

from __future__ import annotations

import re

import pytest

from insight_bench.cli import DEFAULT_BENCHMARK, build_parser
from insight_bench.registry import Registry
from insight_bench.runner import RunContext, RunnerError, available_runners
from insight_bench.runners._shared import resolve_backend_descriptor
from insight_bench.simulator.base import SIM_BACKENDS, default_backend

# The two paths every `run` needs; irrelevant to what is asserted below, but
# argparse requires them before it will parse anything else.
DATA_ARGS = ["--episodes", "/tmp/episodes.jsonl", "--scene-root", "/tmp/scenes"]


def _run_args(*extra: str):
    return build_parser().parse_args(["run", *DATA_ARGS, *extra])


class TestTheCliNamesASimulatorLine:
    def test_every_declared_line_is_selectable(self) -> None:
        # Named as they are everywhere else: the flag value is the backend id,
        # not a friendly alias that something later has to translate.
        for descriptor in SIM_BACKENDS:
            assert _run_args("--sim-backend", descriptor.backend_id).sim_backend == (
                descriptor.backend_id
            )

    def test_an_undeclared_line_is_refused_by_the_parser(self, capsys) -> None:
        with pytest.raises(SystemExit):
            _run_args("--sim-backend", "isaac-7.0")
        message = capsys.readouterr().err
        for descriptor in SIM_BACKENDS:
            assert descriptor.backend_id in message, "the refusal names what is on offer"

    def test_naming_no_line_leaves_the_choice_to_the_runner(self) -> None:
        assert _run_args().sim_backend is None

    def test_run_dir_stays_accepted_as_an_alias_on_the_same_field(self) -> None:
        # Renaming a published flag without an alias breaks whatever already
        # calls it, and two fields for one directory would let them disagree.
        assert str(_run_args("--run-dir", "/tmp/y").artifact_dir) == "/tmp/y"
        assert str(_run_args("--artifact-dir", "/tmp/y").artifact_dir) == "/tmp/y"
        assert not hasattr(_run_args("--run-dir", "/tmp/y"), "run_dir")


class TestTheNamedLineIsTheLineThatRuns:
    """Named-and-ignored is the one outcome that must not be possible.

    Reading ``--sim-backend`` and loading the default anyway means a caller who
    asked for the experimental line gets official-line code with no refusal, and
    a number nobody can attribute to a runtime.
    """

    def test_a_named_line_is_the_descriptor_the_run_resolves(self) -> None:
        for descriptor in SIM_BACKENDS:
            resolved = resolve_backend_descriptor(RunContext(sim_backend=descriptor.backend_id))
            assert resolved is descriptor

    def test_naming_nothing_resolves_to_the_official_default(self) -> None:
        resolved = resolve_backend_descriptor(RunContext())
        assert resolved is default_backend()
        assert (resolved.backend_id, resolved.status) == ("isaac-5.1", "official")

    def test_an_undeclared_name_refuses_before_a_simulator_is_reached(self) -> None:
        with pytest.raises(RunnerError, match=re.escape("isaac-9.9")):
            resolve_backend_descriptor(RunContext(sim_backend="isaac-9.9"))


class TestThePathTheDefaultRunTakes:
    def test_the_coordinate_a_bare_run_targets_is_registered_and_runnable(self) -> None:
        manifest = Registry.load().get(DEFAULT_BENCHMARK)
        assert manifest.coordinate == DEFAULT_BENCHMARK
        assert manifest.runnable
        # The declared runner is data; this is the check that the data names
        # something that exists, which is what makes the path reachable at all.
        assert manifest.runtime.runner in available_runners()

    def test_that_coordinate_is_a_simulator_suite(self) -> None:
        manifest = Registry.load().get(DEFAULT_BENCHMARK)
        assert manifest.metadata["simulation_assets_required"] is True
        assert {variant.backend_id for variant in manifest.runtime_variants} == {
            descriptor.backend_id for descriptor in SIM_BACKENDS
        }
