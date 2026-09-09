"""Rollout tunables for a camera-walk episode.

These describe *how a task is executed* (control period, velocity clamps,
waypoint selection, render settling), not which benchmark asked for it, so they
live on the task config (``BenchmarkTaskConfig.metadata["rollout"]``) rather
than in a benchmark-ID branch or in a frozen registry manifest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from insight_bench.vln_runtime.suite.config import BenchmarkTaskConfig


@dataclass(frozen=True)
class RolloutOptions:
    """One episode's execution parameters. All defaults are explicit."""

    num_steps: int = 100
    warmup_steps: int = 3
    render_settle_steps: int = 1
    # 0 disables the first-frame convergence gate; the Isaac backend sets a real
    # budget when it can actually re-render a held pose.
    render_converge_max_steps: int = 0
    render_converge_tol: float = 0.0
    control_dt_sec: float = 0.1
    lin_vel_range_mps: tuple[float, float] = (0.0, 2.5)
    ang_vel_range_dps: tuple[float, float] = (-300.0, 300.0)
    waypoint_atol: float = 1e-6
    yaw_hybrid_thresh_deg: float = 1.0
    yaw_lookahead_steps: int = 1
    yaw_lookahead_hybrid: bool = True
    # None means "auto-select the first non-zero forward/yaw waypoint" (VLN-CE).
    waypoint_index: int | None = None
    drop_lateral: bool = True
    motion_scale: float = 1.0
    yaw_scale: float = 1.0

    @property
    def yaw_hybrid_thresh_rad(self) -> float:
        return math.radians(self.yaw_hybrid_thresh_deg)

    def __post_init__(self) -> None:
        if self.num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {self.num_steps}")
        if not math.isfinite(self.control_dt_sec) or self.control_dt_sec <= 0.0:
            raise ValueError(
                f"control_dt_sec must be positive and finite, got {self.control_dt_sec}"
            )
        if self.yaw_lookahead_steps < 1:
            raise ValueError(f"yaw_lookahead_steps must be >= 1, got {self.yaw_lookahead_steps}")

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> RolloutOptions:
        if not payload:
            return cls()
        known = {field_name for field_name in cls.__dataclass_fields__}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"unknown rollout options: {unknown}")
        values = dict(payload)
        for pair_key in ("lin_vel_range_mps", "ang_vel_range_dps"):
            if pair_key in values:
                low, high = values[pair_key]
                values[pair_key] = (float(low), float(high))
        return cls(**values)

    @classmethod
    def from_task_config(cls, task_config: BenchmarkTaskConfig) -> RolloutOptions:
        payload = task_config.metadata.get("rollout")
        if payload is not None and not isinstance(payload, dict):
            raise ValueError(
                f"task {task_config.name!r} metadata['rollout'] must be a mapping, got {payload!r}"
            )
        return cls.from_dict(payload)
