"""Configuration terms for lightweight manager-based benchmark environments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ManagerTermBaseCfg:
    """Configuration for one manager term.

    This mirrors IsaacLab's shape: ``func`` is called with the environment as
    the first argument, and ``params`` are passed as keyword arguments.
    """

    func: Callable[..., Any] | str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class MeasureTermCfg(ManagerTermBaseCfg):
    """Configuration for one online measure term."""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MeasureTermCfg:
        return cls(
            func=str(payload["func"]),
            params=dict(payload.get("params") or {}),
        )


@dataclass
class TerminationTermCfg(ManagerTermBaseCfg):
    """Configuration for one termination term."""

    time_out: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TerminationTermCfg:
        return cls(
            func=str(payload["func"]),
            params=dict(payload.get("params") or {}),
            time_out=bool(payload.get("time_out", False)),
        )
