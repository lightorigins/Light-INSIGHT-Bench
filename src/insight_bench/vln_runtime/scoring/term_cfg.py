"""Offline score term configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from insight_bench.vln_runtime.managers import ManagerTermBaseCfg


@dataclass
class ScoreTermCfg(ManagerTermBaseCfg):
    """Configuration for one offline scalar score term."""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ScoreTermCfg:
        return cls(
            func=str(payload["func"]),
            params=dict(payload.get("params") or {}),
        )


ScoreTerm = ScoreTermCfg
