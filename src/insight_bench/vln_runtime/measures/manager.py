"""Public online measure manager helpers."""

from __future__ import annotations

from typing import Any

from insight_bench.vln_runtime.managers import MeasureManager

from .config import MeasureManagerCfg, NavigationMeasuresCfg


def build_measure_manager(cfg: object | None = None, env: Any | None = None) -> MeasureManager:
    if env is None:
        raise TypeError("build_measure_manager now requires an env/state object")
    return MeasureManager(cfg or MeasureManagerCfg.default_navigation(), env)


__all__ = ["MeasureManager", "MeasureManagerCfg", "NavigationMeasuresCfg", "build_measure_manager"]
