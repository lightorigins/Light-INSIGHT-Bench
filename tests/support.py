"""A run result for the tests that need one but need no simulator.

Evidence packing, tamper detection, CLI plumbing and the v1 wire-compatibility
gate all need a :class:`~insight_bench.contracts.RunResult` and none of them need
a scene, a GPU or a policy server. They used to get one by running the
``bench-tiny`` text fixture through ``EchoAdapter``; that benchmark, that
adapter and the runner behind them are gone with the rest of the non-objnav
product, so the run result is built here instead.

Built, not mocked. :func:`insight_bench.runners._shared.build_run_result` is the
function the objnav runner itself calls to turn scored episodes into a run
result, and it is fed here with a manifest read from the shipped registry and a
request id from the same helper the runner uses. So the run id, the fingerprint,
the aggregated metrics and the run status are all computed by production code;
only the per-episode outcomes are written by hand.

Import it as ``from support import ...`` after putting ``tests/`` on the path --
see the two-line preamble in each consumer. There is no ``conftest.py`` in this
tree and this module deliberately does not introduce one: it defines plain
functions, not fixtures, so a reader can see where the values come from.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any, Literal

from insight_bench._json import write_stable_json
from insight_bench.contracts import (
    AdapterAction,
    AdapterDescriptor,
    AdapterResponse,
    BenchmarkManifest,
    EpisodeRunResult,
    RunResult,
    RuntimeAttestation,
)
from insight_bench.registry import Registry
from insight_bench.runners._shared import build_run_result, request_id
from insight_bench.runners.insight_bench import RUNNER_ID

#: The coordinate these results are filed under: the suite a bare ``run`` runs.
BENCHMARK = "insight-bench-v1@1.0.0"

#: Two ids taken verbatim from the published episode records in
#: ``tests/fixtures/objnav_published_episodes.jsonl``.
EPISODE_IDS = ("objnav_00013-sfbj7jspYWj_4", "objnav_v2_human_manual_8194nk5LbLH_49")

#: What a policy server reported about itself on ``GET /health``, in the shape
#: :func:`insight_bench.runners.insight_bench._policy_descriptor` records it.
POLICY = AdapterDescriptor(
    adapter_id="vln-http-policy",
    model_id="insight-test-policy-0.1",
    deterministic=False,
    metadata={"policy_protocol": "vln-http/1", "healthy": True},
)


@cache
def manifest() -> BenchmarkManifest:
    """The real shipped manifest for :data:`BENCHMARK`."""
    return Registry.load().get(BENCHMARK)


def attestation(**overrides: Any) -> RuntimeAttestation:
    """A publishable Linux/Docker attestation on the official backend."""
    values: dict[str, Any] = {
        "backend_id": "isaac-5.1",
        "backend_status": "official",
        "launcher_id": "docker",
        "runtime_kind": "docker",
        "os": "linux",
        "driver_version": "580.65.06",
        "isaac_sim_version": "5.1.0",
        "isaac_lab_version": "2.3.0",
        "python_version": "3.11.15",
        "publishable": True,
    }
    values.update(overrides)
    return RuntimeAttestation(**values)


def episode_result(
    episode_id: str,
    *,
    status: Literal["passed", "failed"] = "passed",
    seed: int = 0,
) -> EpisodeRunResult:
    """One scored episode, shaped the way ``run_scored_episode`` shapes one."""
    benchmark = manifest()
    reached = status == "passed"
    return EpisodeRunResult(
        episode_id=episode_id,
        task_id=benchmark.tasks[0].task_id,
        status=status,
        response=AdapterResponse(
            request_id=request_id(benchmark, episode_id, runner_id=RUNNER_ID, seed=seed),
            adapter=POLICY,
            status="success",
            output="policy_stop_requested" if reached else "time_out",
            actions=(AdapterAction(name="stop"),) if reached else (),
            metadata={
                "steps": 61 if reached else 300,
                "stop_step": 61 if reached else -1,
                "score_labels": {"navnuances_category": "LR"},
                "failure_reason": "" if reached else "did not stop within the step budget",
            },
        ),
        metrics={
            "success": 1.0 if reached else 0.0,
            "spl": 0.82 if reached else 0.0,
            "distance_to_goal": 1.24 if reached else 7.5,
            "path_length": 9.4 if reached else 24.1,
            "stopped": 1.0 if reached else 0.0,
        },
    )


def run_result(
    *,
    seed: int = 0,
    runtime_attestation: RuntimeAttestation | None = None,
) -> RunResult:
    """A two-episode objnav run -- one reached, one did not -- as the runner builds it."""
    return build_run_result(
        manifest=manifest(),
        adapter_descriptor=POLICY,
        runner_id=RUNNER_ID,
        seed=seed,
        episode_results=[
            episode_result(EPISODE_IDS[0], status="passed", seed=seed),
            episode_result(EPISODE_IDS[1], status="failed", seed=seed),
        ],
        attestation=runtime_attestation or attestation(),
    )


def run_result_without_attestation(**kwargs: Any) -> RunResult:
    """The same run result with the optional attestation absent.

    Every shipped runner drives a simulator and therefore always attaches one,
    so ``build_run_result`` requires it. The contract does not: the field is
    ``RuntimeAttestation | None`` and its absence has a serialization rule of its
    own, which is what the compatibility gate is about. Copied from a built
    result rather than hand-assembled so everything else -- run id, fingerprint,
    metrics, status -- is still what production code computed.
    """
    return run_result(**kwargs).model_copy(update={"runtime_attestation": None})


def write_run_result(path: Path, **kwargs: Any) -> Path:
    """Write :func:`run_result` to *path* exactly as ``run --output`` writes one."""
    write_stable_json(path, run_result(**kwargs))
    return path
