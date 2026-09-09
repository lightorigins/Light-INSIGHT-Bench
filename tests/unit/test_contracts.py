from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from insight_bench._json import load_json, stable_json_bytes
from insight_bench.contracts import PUBLIC_SCHEMAS, AdapterRequest, BenchmarkManifest


def test_checked_in_json_schemas_match_pydantic_contracts() -> None:
    root = Path(__file__).parents[2] / "src" / "insight_bench" / "schemas" / "v1"
    for name, model in PUBLIC_SCHEMAS.items():
        expected = model.model_json_schema(mode="validation", ref_template="#/$defs/{model}")
        expected["$id"] = (
            f"https://www.lightorigins.com/schemas/insight-bench/v1/{name}.schema.json"
        )
        expected["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        assert load_json(root / f"{name}.schema.json") == expected


def test_contracts_forbid_unknown_fields_and_invalid_versions() -> None:
    payload = {
        "request_id": "req-0123456789abcdef",
        "benchmark_id": "bench-tiny",
        "benchmark_version": "1.0",
        "task_id": "echo-command",
        "episode_id": "episode-1",
        "instruction": "stop",
        "unexpected": True,
    }
    with pytest.raises(ValidationError):
        AdapterRequest.model_validate(payload)


def test_stable_json_is_sorted_utf8_and_newline_terminated() -> None:
    assert stable_json_bytes({"z": 1, "a": "路径"}) == ('{\n  "a": "路径",\n  "z": 1\n}\n'.encode())


def _runnable_manifest(**overrides: object) -> dict:
    """A minimal runnable manifest whose dataset the user supplies."""
    payload: dict = {
        "benchmark_id": "gated-suite",
        "version": "1.0.0",
        "title": "Gated Suite",
        "description": "A runnable suite whose episode file the user downloads.",
        "license": {"status": "declared", "spdx_id": "CC-BY-4.0", "notice": "CC BY 4.0."},
        "runtime": {"runner": "objnav-http-policy-v1"},
        "dataset": {
            "availability": "user-provided",
            "sha256": "a" * 64,
            "notice": "Download the published episode file.",
        },
        "tasks": [{"task_id": "objnav", "description": "Navigate to the object."}],
        "metrics": ["success-rate"],
        "runnable": True,
        "requires_configuration": ["episode-dataset"],
    }
    payload.update(overrides)
    return payload


def test_a_runnable_manifest_may_have_a_user_provided_dataset() -> None:
    """`runnable` means nothing is undeclared, not that everything is bundled.

    The original rule refused this outright, which made a published suite whose
    dataset is licence-gated impossible to describe honestly.
    """
    manifest = BenchmarkManifest.model_validate(_runnable_manifest())
    assert manifest.runnable is True
    assert manifest.dataset.availability == "user-provided"


def test_a_runnable_user_provided_dataset_must_pin_its_digest() -> None:
    # Without the digest there is nothing to check a downloaded file against,
    # and a run over some other file would carry this coordinate.
    payload = _runnable_manifest()
    payload["dataset"] = {**payload["dataset"], "sha256": None}  # type: ignore[index]
    with pytest.raises(ValidationError, match="requires the sha256 of the published file"):
        BenchmarkManifest.model_validate(payload)


def test_a_runnable_user_provided_dataset_must_say_what_the_user_supplies() -> None:
    with pytest.raises(ValidationError, match="requires_configuration"):
        BenchmarkManifest.model_validate(_runnable_manifest(requires_configuration=[]))


def test_a_runnable_bundled_dataset_still_may_not_require_configuration() -> None:
    # The original rule for the bundled case is unchanged: a benchmark that
    # ships its data has nothing left for the user to configure.
    payload = _runnable_manifest(requires_configuration=["scene-root"])
    payload["dataset"] = {
        "availability": "bundled",
        "path": "episodes.json",
        "sha256": "b" * 64,
        "notice": "Bundled.",
    }
    with pytest.raises(ValidationError, match="cannot require configuration"):
        BenchmarkManifest.model_validate(payload)


def test_a_runnable_manifest_with_no_dataset_at_all_is_refused() -> None:
    payload = _runnable_manifest()
    payload["dataset"] = {"availability": "unconfigured", "notice": "Nothing configured."}
    with pytest.raises(ValidationError, match="bundled or user-provided dataset"):
        BenchmarkManifest.model_validate(payload)


def test_a_runnable_manifest_still_requires_a_declared_license() -> None:
    payload = _runnable_manifest()
    payload["license"] = {"status": "placeholder", "notice": "To be decided."}
    with pytest.raises(ValidationError, match="requires a declared license"):
        BenchmarkManifest.model_validate(payload)
