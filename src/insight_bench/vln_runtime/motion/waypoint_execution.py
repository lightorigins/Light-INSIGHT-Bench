"""Waypoint response selection and camera-walk execution helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from insight_bench.vln_runtime.motion.pose import CameraPose, normalize_yaw
from insight_bench.vln_runtime.policy.client import PolicyResponse


@dataclass(frozen=True)
class WaypointDelta:
    forward_m: float
    lateral_m: float
    yaw_rad: float

    def to_dict(self) -> dict[str, float]:
        return {
            "forward_m": self.forward_m,
            "lateral_m": self.lateral_m,
            "yaw_rad": self.yaw_rad,
        }


@dataclass(frozen=True)
class VelocityControlCommand:
    linear_velocity_mps: float
    lateral_velocity_mps: float
    angular_velocity_dps: float

    def to_dict(self) -> dict[str, float]:
        return {
            "linear_velocity_mps": self.linear_velocity_mps,
            "lateral_velocity_mps": self.lateral_velocity_mps,
            "angular_velocity_dps": self.angular_velocity_dps,
        }

    def to_delta(self, dt_sec: float) -> WaypointDelta:
        if not math.isfinite(dt_sec) or dt_sec <= 0.0:
            raise ValueError(f"dt_sec must be a positive finite float, got {dt_sec}")
        return WaypointDelta(
            forward_m=self.linear_velocity_mps * dt_sec,
            lateral_m=self.lateral_velocity_mps * dt_sec,
            yaw_rad=math.radians(self.angular_velocity_dps * dt_sec),
        )


def apply_waypoint_delta(
    pose: CameraPose,
    waypoint: WaypointDelta,
    *,
    motion_scale: float = 1.0,
    yaw_scale: float = 1.0,
) -> CameraPose:
    forward = waypoint.forward_m * motion_scale
    lateral = waypoint.lateral_m * motion_scale
    cos_yaw = math.cos(pose.yaw)
    sin_yaw = math.sin(pose.yaw)
    return CameraPose(
        x=pose.x + forward * cos_yaw - lateral * sin_yaw,
        y=pose.y + forward * sin_yaw + lateral * cos_yaw,
        z=pose.z,
        yaw=normalize_yaw(pose.yaw + waypoint.yaw_rad * yaw_scale),
    )


def integrate_velocity_control(
    pose: CameraPose,
    command: VelocityControlCommand,
    *,
    dt_sec: float,
    motion_scale: float = 1.0,
    yaw_scale: float = 1.0,
) -> tuple[CameraPose, WaypointDelta]:
    delta = command.to_delta(dt_sec)
    return apply_waypoint_delta(pose, delta, motion_scale=motion_scale, yaw_scale=yaw_scale), delta


def _waypoint_payloads_from_policy_response(response: PolicyResponse) -> list[WaypointDelta]:
    if response.waypoint is None:
        raise RuntimeError(f"policy response has no waypoint field: {response}")
    waypoints = response.waypoint["waypoints"]
    if not isinstance(waypoints, list) or not waypoints:
        raise RuntimeError(f"policy response waypoint list is empty: {response.waypoint}")
    return [
        WaypointDelta(
            forward_m=float(waypoint["forward_m"]),
            lateral_m=float(waypoint["lateral_m"]),
            yaw_rad=float(waypoint["yaw_rad"]),
        )
        for waypoint in waypoints
    ]


def select_vlnce_action_waypoint_index(
    waypoints: list[WaypointDelta],
    *,
    atol: float,
) -> int:
    for index, waypoint in enumerate(waypoints):
        if abs(waypoint.forward_m) > atol or abs(waypoint.yaw_rad) > atol:
            return index
    return 0


def _apply_yaw_lookahead(
    waypoints: list[WaypointDelta],
    waypoint_index: int,
    waypoint: WaypointDelta,
    *,
    yaw_lookahead_steps: int,
    yaw_lookahead_hybrid: bool,
    yaw_hybrid_thresh_rad: float,
) -> WaypointDelta:
    if yaw_lookahead_steps < 1:
        raise ValueError(f"yaw_lookahead_steps must be >= 1, got {yaw_lookahead_steps}")
    if yaw_lookahead_steps == 1:
        return waypoint

    end_index = min(waypoint_index + yaw_lookahead_steps - 1, len(waypoints) - 1)
    avg_yaw_rate_rad = waypoints[end_index].yaw_rad / float(end_index + 1)
    should_apply = (not yaw_lookahead_hybrid) or abs(waypoint.yaw_rad) < yaw_hybrid_thresh_rad
    if should_apply and abs(avg_yaw_rate_rad) > 0.0:
        return WaypointDelta(
            forward_m=waypoint.forward_m,
            lateral_m=waypoint.lateral_m,
            yaw_rad=avg_yaw_rate_rad,
        )
    return waypoint


def waypoint_from_policy_response(
    response: PolicyResponse,
    waypoint_index: int | None = None,
    *,
    waypoint_atol: float,
    yaw_lookahead_steps: int,
    yaw_lookahead_hybrid: bool,
    yaw_hybrid_thresh_rad: float,
) -> tuple[int, WaypointDelta]:
    waypoints = _waypoint_payloads_from_policy_response(response)
    selected_index = (
        select_vlnce_action_waypoint_index(waypoints, atol=waypoint_atol)
        if waypoint_index is None
        else int(waypoint_index)
    )
    if selected_index < 0 or selected_index >= len(waypoints):
        raise ValueError(
            f"waypoint_index={selected_index} out of range for horizon={len(waypoints)}"
        )

    waypoint = _apply_yaw_lookahead(
        waypoints,
        selected_index,
        waypoints[selected_index],
        yaw_lookahead_steps=yaw_lookahead_steps,
        yaw_lookahead_hybrid=yaw_lookahead_hybrid,
        yaw_hybrid_thresh_rad=yaw_hybrid_thresh_rad,
    )
    return selected_index, waypoint


def velocity_control_from_waypoint(
    waypoint: WaypointDelta,
    *,
    dt_sec: float,
    lin_vel_range_mps: tuple[float, float],
    ang_vel_range_dps: tuple[float, float],
    drop_lateral: bool,
) -> VelocityControlCommand:
    if not math.isfinite(dt_sec) or dt_sec <= 0.0:
        raise ValueError(f"dt_sec must be a positive finite float, got {dt_sec}")

    values = np.asarray(
        [waypoint.forward_m, waypoint.lateral_m, waypoint.yaw_rad], dtype=np.float32
    )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"waypoint must be finite, got {values}")

    lin_mps = float(
        np.clip(waypoint.forward_m / dt_sec, lin_vel_range_mps[0], lin_vel_range_mps[1])
    )
    lateral_mps = 0.0 if drop_lateral else float(waypoint.lateral_m / dt_sec)
    ang_dps = float(
        np.clip(math.degrees(waypoint.yaw_rad) / dt_sec, ang_vel_range_dps[0], ang_vel_range_dps[1])
    )
    return VelocityControlCommand(
        linear_velocity_mps=lin_mps,
        lateral_velocity_mps=lateral_mps,
        angular_velocity_dps=ang_dps,
    )
