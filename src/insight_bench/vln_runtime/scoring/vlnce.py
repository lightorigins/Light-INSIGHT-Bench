"""VLN-CE / NaVILA-style scoring.

NaVILA computes distance-to-goal by snapping the agent to the nearest dense
``gt_locations`` waypoint and adding the remaining dense path length. This is
different from the simple XY goal-distance scorer in ``metrics.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from insight_bench.vln_runtime.episodes import EpisodeSpec
from insight_bench.vln_runtime.traces import RolloutTrace

XYZ = tuple[float, float, float]


@dataclass(frozen=True)
class VlnceEpisodeScore:
    """NaVILA/VLN-CE scalar metrics for one rollout."""

    episode_id: str
    success: bool
    spl: float
    distance_to_goal: float
    path_length: float
    oracle_navigation_error: float
    oracle_success: float
    stop_step: int
    initial_distance: float
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "success": bool(self.success),
            "spl": float(self.spl),
            "distance_to_goal": float(self.distance_to_goal),
            "path_length": float(self.path_length),
            "oracle_navigation_error": float(self.oracle_navigation_error),
            "oracle_success": float(self.oracle_success),
            "stop_step": int(self.stop_step),
            "initial_distance": float(self.initial_distance),
            "failure_reason": self.failure_reason,
        }


def evaluate_vlnce_navigation(
    episode: EpisodeSpec,
    trace: RolloutTrace,
    *,
    require_stop: bool = True,
) -> VlnceEpisodeScore:
    """Evaluate a finished rollout with NaVILA's dense-path metrics."""

    gt_path = dense_gt_path_from_episode(episode)
    initial_distance = distance_to_goal_along_dense_path(episode.start_pose.position, gt_path)

    if not trace.steps:
        return VlnceEpisodeScore(
            episode_id=episode.episode_id,
            success=False,
            spl=0.0,
            distance_to_goal=float("inf"),
            path_length=0.0,
            oracle_navigation_error=float("inf"),
            oracle_success=0.0,
            stop_step=trace.stop_step,
            initial_distance=initial_distance,
            failure_reason="empty_trace",
        )

    final_position = trace.steps[-1].position
    distance_to_goal = distance_to_goal_along_dense_path(final_position, gt_path)
    oracle_navigation_error = min(
        distance_to_goal_along_dense_path(step.position, gt_path) for step in trace.steps
    )
    path_length = path_length_3d([step.position for step in trace.steps])
    stopped = trace.stop_step >= 0
    success = distance_to_goal < episode.goal_radius_m and (stopped or not require_stop)
    oracle_success = 1.0 if oracle_navigation_error < episode.goal_radius_m else 0.0
    spl = _spl(success=success, initial_distance=initial_distance, path_length=path_length)

    failure_reason = None
    if not success:
        if require_stop and not stopped:
            failure_reason = "stop_not_requested"
        else:
            failure_reason = "goal_not_reached"

    return VlnceEpisodeScore(
        episode_id=episode.episode_id,
        success=success,
        spl=spl,
        distance_to_goal=distance_to_goal,
        path_length=path_length,
        oracle_navigation_error=oracle_navigation_error,
        oracle_success=oracle_success,
        stop_step=trace.stop_step,
        initial_distance=initial_distance,
        failure_reason=failure_reason,
    )


def dense_gt_path_from_episode(episode: EpisodeSpec) -> list[XYZ]:
    """Return dense GT path, falling back to reference path and goal."""

    gt_locations = episode.metadata.get("gt_locations")
    if isinstance(gt_locations, list) and gt_locations:
        return [_float3(point, name="metadata.gt_locations[]") for point in gt_locations]
    if episode.reference_path:
        return list(episode.reference_path)
    if episode.goal_position is not None:
        return [episode.start_pose.position, episode.goal_position]
    return [episode.start_pose.position]


def distance_to_goal_along_dense_path(position: XYZ, dense_path: list[XYZ]) -> float:
    """Nearest dense waypoint distance plus remaining dense path length."""

    if not dense_path:
        return float("inf")
    closest_idx = min(
        range(len(dense_path)), key=lambda idx: euclidean_distance(position, dense_path[idx])
    )
    return euclidean_distance(position, dense_path[closest_idx]) + path_length_3d(
        dense_path[closest_idx:]
    )


def path_length_3d(points: list[XYZ]) -> float:
    if len(points) < 2:
        return 0.0
    return sum(euclidean_distance(prev, cur) for prev, cur in zip(points, points[1:]))


def euclidean_distance(a: XYZ, b: XYZ) -> float:
    return math.sqrt(
        (float(a[0]) - float(b[0])) ** 2
        + (float(a[1]) - float(b[1])) ** 2
        + (float(a[2]) - float(b[2])) ** 2
    )


def _spl(*, success: bool, initial_distance: float, path_length: float) -> float:
    if not success:
        return 0.0
    if initial_distance <= 0.0:
        return 1.0 if path_length <= 0.0 else 0.0
    return float(initial_distance / max(initial_distance, path_length))


def _float3(value: Any, *, name: str) -> XYZ:
    if not isinstance(value, list | tuple) or len(value) < 3:
        raise ValueError(f"{name} must be a 3-element sequence, got {value!r}")
    return float(value[0]), float(value[1]), float(value[2])
