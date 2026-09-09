"""Failure-event evidence terms for VLN navigation rollouts."""

from __future__ import annotations

from collections import Counter
from typing import Any

from insight_bench.vln_runtime.episodes import EpisodeSpec
from insight_bench.vln_runtime.scoring.vlnce import (
    dense_gt_path_from_episode,
    distance_to_goal_along_dense_path,
    path_length_3d,
)

from .state import EpisodeEvalState

XYZ = tuple[float, float, float]


def event_count(state: EpisodeEvalState) -> int:
    return sum(_event_counts(state).values())


def primary_event(state: EpisodeEvalState) -> str:
    counts = _event_counts(state)
    if not counts:
        return ""
    return counts.most_common(1)[0][0]


def wrong_stop_count(state: EpisodeEvalState) -> int:
    return int(_event_counts(state).get("wrong_stop", 0))


def navigation_step_events(
    episode: EpisodeSpec,
    positions: list[XYZ],
    *,
    stop_requested: bool = False,
    termination_reason: str | None = None,
    progress_status: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return event evidence for the latest trajectory step."""

    if not positions:
        return []
    current = positions[-1]
    dense_path = dense_gt_path_from_episode(episode)
    distance_to_goal = distance_to_goal_along_dense_path(current, dense_path)
    oracle_distance = min(
        (distance_to_goal_along_dense_path(position, dense_path) for position in positions),
        default=float("inf"),
    )
    events: list[dict[str, Any]] = []

    if stop_requested and distance_to_goal > float(episode.goal_radius_m):
        events.append({"type": "wrong_stop", "distance_to_goal": distance_to_goal})
    if oracle_distance <= float(episode.goal_radius_m) and distance_to_goal > float(
        episode.goal_radius_m
    ):
        events.append(
            {
                "type": "goal_overshoot",
                "distance_to_goal": distance_to_goal,
                "oracle_distance": oracle_distance,
            }
        )
    if progress_status and progress_status.get("regressed"):
        events.append(
            {
                "type": "path_regression",
                "best_path_progress": progress_status.get("best_path_progress"),
            }
        )
    if _is_stuck(positions):
        events.append({"type": "stuck"})
    if _loop_detected(positions):
        events.append({"type": "loop_detected"})
    if termination_reason in {"time_out", "max_steps", "num_steps_exhausted"}:
        events.append({"type": "timeout"})
    return events


def _event_counts(state: EpisodeEvalState) -> Counter[str]:
    key = ("events", ())
    if key in state.cache:
        return state.cache[key]
    counts: Counter[str] = Counter()
    for step in state.trace.steps:
        events = step.metadata.get("events")
        if not isinstance(events, list):
            continue
        for event in events:
            if isinstance(event, dict) and event.get("type"):
                counts[str(event["type"])] += 1
            elif isinstance(event, str):
                counts[event] += 1
    if state.trace.termination_reason == "num_steps_exhausted":
        counts["timeout"] += 1
    state.cache[key] = counts
    return counts


def _is_stuck(positions: list[XYZ], *, window: int = 20, min_displacement_m: float = 0.20) -> bool:
    if len(positions) < window:
        return False
    recent = positions[-window:]
    return path_length_3d(recent) < min_displacement_m


def _loop_detected(
    positions: list[XYZ], *, window: int = 24, close_radius_m: float = 0.50, min_path_m: float = 2.0
) -> bool:
    if len(positions) < window:
        return False
    recent = positions[-window:]
    dx = float(recent[-1][0]) - float(recent[0][0])
    dy = float(recent[-1][1]) - float(recent[0][1])
    if (dx * dx + dy * dy) ** 0.5 > close_radius_m:
        return False
    return path_length_3d(recent) >= min_path_m
