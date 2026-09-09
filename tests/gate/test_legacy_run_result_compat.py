"""A run result emitted today must stay readable by a pre-attestation consumer.

``runtime_attestation`` was added to ``RunResult`` inside schema version
1.0.0, which is an additive change only if the payload stays acceptable to
consumers still validating against the older copy of the schema. That older
copy is ``additionalProperties: false`` and does not declare the field, so
serializing an explicit ``"runtime_attestation": null`` — which is what a
result with no runtime to attest to used to produce — was rejected outright
by those consumers. The fix was a serializer that omits the key when there is
nothing to say, and that serializer is what this file pins.

The result with no attestation is now built directly from
:mod:`insight_bench.contracts` (via ``tests/support.py``) rather than run out of
a local fixture benchmark: the fixture benchmarks and their runner were
removed, so every *shipped* runner drives a simulator and always attaches an
attestation. The contract still declares the field optional, so the payload
shape it produces is still a shape a consumer can be handed, and its rule is
still the one that decides this compatibility question.

``legacy_v1_run_result.schema.json`` next to this file is the checked-in
run-result schema exactly as published before the attestation field existed
(``git show 7a22e80:src/insight_bench/schemas/v1/run-result.schema.json``,
the commit preceding 913dbbf). It is a frozen fixture: never regenerate it
from the current contracts, or the compatibility claim becomes circular.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

from insight_bench._json import stable_json_text
from insight_bench.contracts import RunResult
from support import run_result, run_result_without_attestation

_LEGACY_SCHEMA = json.loads(
    Path(__file__).with_name("legacy_v1_run_result.schema.json").read_text()
)


def _legacy_rejections(payload: dict[str, Any]) -> list[str]:
    """Apply the two top-level legacy rules a newer payload can violate.

    This is not a JSON Schema engine; it is the pair of constraints that
    decides this specific compatibility question — an undeclared key under
    ``additionalProperties: false``, and a missing required key.
    """
    declared = set(_LEGACY_SCHEMA["properties"])
    required = set(_LEGACY_SCHEMA.get("required", ()))
    assert _LEGACY_SCHEMA["additionalProperties"] is False
    return [
        *(f"undeclared property {key!r}" for key in sorted(set(payload) - declared)),
        *(f"missing required property {key!r}" for key in sorted(required - set(payload))),
    ]


def _emitted(result: RunResult) -> dict[str, Any]:
    """The payload as the SDK actually writes it (CLI, evidence packs, golden)."""
    return json.loads(stable_json_text(result))


def test_a_run_without_an_attestation_omits_the_field_entirely() -> None:
    payload = _emitted(run_result_without_attestation())
    assert "runtime_attestation" not in payload


def test_a_legacy_consumer_accepts_a_newly_emitted_result() -> None:
    payload = _emitted(run_result_without_attestation())
    assert _legacy_rejections(payload) == []
    assert payload["schema_version"] == "1.0.0"


def test_the_previous_serialization_would_have_been_rejected() -> None:
    # Guards against a vacuous test: the null the SDK used to emit really is
    # what the legacy consumer refused.
    payload = _emitted(run_result_without_attestation())
    payload["runtime_attestation"] = None
    assert _legacy_rejections(payload) == ["undeclared property 'runtime_attestation'"]


def test_a_real_attestation_is_still_serialized() -> None:
    # Omission expresses absence only. A simulator run -- which is every run
    # this SDK can now produce -- must keep publishing the record its numbers
    # are judged on.
    payload = _emitted(run_result())
    assert payload["runtime_attestation"]["backend_id"] == "isaac-5.1"
    assert payload["runtime_attestation"]["publishable"] is True
    # Such a payload is legitimately outside the legacy schema; the field is
    # only omittable when there is nothing to say.
    assert _legacy_rejections(payload) == ["undeclared property 'runtime_attestation'"]


def test_omission_loses_nothing_on_round_trip() -> None:
    for result in (run_result_without_attestation(), run_result()):
        assert RunResult.model_validate(_emitted(result)) == result
