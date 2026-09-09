"""Termination config helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from insight_bench.vln_runtime.managers import DoneTerm, TerminationTermCfg

from . import terms


@dataclass
class NavigationTerminationsCfg:
    """Default camera-walk navigation termination terms."""

    policy_stop: TerminationTermCfg = field(
        default_factory=lambda: DoneTerm(func=terms.policy_stop_requested)
    )
    time_out: TerminationTermCfg = field(
        default_factory=lambda: DoneTerm(func=terms.time_out, time_out=True)
    )


def terminations_from_dict(payload: dict[str, Any] | None) -> object:
    """Parse YAML/dict termination config into a manager-consumable config."""

    if not payload:
        return NavigationTerminationsCfg()

    raw_terms = payload.get("terms", payload)
    if isinstance(raw_terms, list):
        terms_by_name = {}
        for item in raw_terms:
            name = str(item["name"])
            term_payload = {key: value for key, value in item.items() if key != "name"}
            terms_by_name[name] = TerminationTermCfg.from_dict(term_payload)
        return terms_by_name
    if isinstance(raw_terms, dict):
        return {
            str(name): TerminationTermCfg.from_dict(dict(term_payload))
            for name, term_payload in raw_terms.items()
            if term_payload is not None
        }
    raise TypeError(f"terminations must be a mapping or list, got {type(raw_terms)!r}")
