"""Online measure terms for camera-walk style navigation environments."""

from __future__ import annotations

from typing import Any

from insight_bench.vln_runtime.traces.schema import distance_xy

XYZ = tuple[float, float, float]


def goal_distance_xy(env: Any) -> float:
    """XY distance from the current agent position to the episode goal."""

    goal = _goal_position(env)
    position = _position(env)
    if goal is None or position is None:
        return float("inf")
    return distance_xy(position, goal)


def path_length_xy(env: Any) -> float:
    """Executed XY path length from the online position history."""

    positions = _positions(env)
    if len(positions) < 2:
        return 0.0
    return sum(distance_xy(prev, cur) for prev, cur in zip(positions, positions[1:]))


def oracle_goal_distance_xy(env: Any) -> float:
    """Best XY distance to goal reached anywhere in the online trajectory."""

    goal = _goal_position(env)
    positions = _positions(env)
    if goal is None or not positions:
        return float("inf")
    return min(distance_xy(position, goal) for position in positions)


def step_count(env: Any) -> int:
    """Number of policy-control steps already applied in this episode."""

    lengths = getattr(env, "episode_length_buf", None)
    if lengths:
        return int(lengths[0])
    positions = _positions(env)
    return max(len(positions) - 1, 0)


def vlnce_path_length(env: Any) -> float:
    """Executed 3D path length from online positions."""

    from insight_bench.vln_runtime.scoring.vlnce import path_length_3d

    return path_length_3d(_positions(env))


def route_distance_remaining(env: Any) -> float:
    """How much of the reference path is left: NaVILA's dense-path distance.

    Distance to the nearest point on the episode's reference path, plus the
    length of that path beyond it. This is a route-following quantity and is
    not the navigation error the run result reports -- that one is the straight
    line to the target object, computed in scoring/navnuances.py. Do not read
    this to compute NE.
    """

    episode = getattr(env, "episode", None)
    position = _position(env)
    if episode is None or position is None:
        return float("inf")

    from insight_bench.vln_runtime.scoring.vlnce import (
        dense_gt_path_from_episode,
        distance_to_goal_along_dense_path,
    )

    return distance_to_goal_along_dense_path(position, dense_gt_path_from_episode(episode))


def vlnce_oracle_navigation_error(env: Any) -> float:
    """Best dense-path distance reached so far."""

    episode = getattr(env, "episode", None)
    positions = _positions(env)
    if episode is None or not positions:
        return float("inf")

    from insight_bench.vln_runtime.scoring.vlnce import (
        dense_gt_path_from_episode,
        distance_to_goal_along_dense_path,
    )

    dense_path = dense_gt_path_from_episode(episode)
    return min(distance_to_goal_along_dense_path(position, dense_path) for position in positions)


def vlnce_oracle_success(env: Any) -> float:
    """Whether the online trajectory has ever reached the VLN-CE goal radius."""

    episode = getattr(env, "episode", None)
    if episode is None:
        return 0.0
    return 1.0 if vlnce_oracle_navigation_error(env) < float(episode.goal_radius_m) else 0.0


def _position(env: Any) -> XYZ | None:
    position = getattr(env, "position", None)
    if position is None:
        return None
    return _float3(position)


def _positions(env: Any) -> list[XYZ]:
    return [_float3(position) for position in getattr(env, "positions", [])]


def _goal_position(env: Any) -> XYZ | None:
    episode = getattr(env, "episode", None)
    goal = None if episode is None else getattr(episode, "goal_position", None)
    if goal is None:
        return None
    return _float3(goal)


def _float3(value: Any) -> XYZ:
    return float(value[0]), float(value[1]), float(value[2])
