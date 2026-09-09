"""Default online measure configs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from insight_bench.vln_runtime.managers import MeasureTerm

from . import terms as nav_terms


@dataclass
class NavigationMeasuresCfg:
    """Online navigation diagnostics shared by camera-walk tasks."""

    goal_distance_xy: MeasureTerm = field(
        default_factory=lambda: MeasureTerm(func=nav_terms.goal_distance_xy)
    )
    path_length_xy: MeasureTerm = field(
        default_factory=lambda: MeasureTerm(func=nav_terms.path_length_xy)
    )
    oracle_goal_distance_xy: MeasureTerm = field(
        default_factory=lambda: MeasureTerm(func=nav_terms.oracle_goal_distance_xy)
    )
    step_count: MeasureTerm = field(default_factory=lambda: MeasureTerm(func=nav_terms.step_count))


class MeasureManagerCfg:
    """Factory namespace for measure manager configs."""

    @classmethod
    def default_navigation(cls) -> NavigationMeasuresCfg:
        return NavigationMeasuresCfg()

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> object:
        if not payload:
            return cls.default_navigation()
        return {
            str(name): MeasureTerm.from_dict(dict(term_payload))
            for name, term_payload in payload.items()
            if term_payload is not None
        }
