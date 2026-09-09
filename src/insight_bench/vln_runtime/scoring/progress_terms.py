"""Subtask progress terms for hard VLN navigation episodes."""

from __future__ import annotations

from typing import Any

from insight_bench.vln_runtime.episodes import EpisodeSpec
from insight_bench.vln_runtime.scoring.vlnce import (
    dense_gt_path_from_episode,
    distance_to_goal_along_dense_path,
    euclidean_distance,
)
from insight_bench.vln_runtime.traces import RolloutTrace

from .state import EpisodeEvalState

XYZ = tuple[float, float, float]


def progress_score(state: EpisodeEvalState) -> float:
    return float(_final_status(state).get("progress_score", 0.0))


def subtask_completion_rate(state: EpisodeEvalState) -> float:
    return float(_final_status(state).get("subtask_completion_rate", 0.0))


def first_failed_subtask(state: EpisodeEvalState) -> str:
    value = _final_status(state).get("first_failed_subtask")
    return "" if value is None else str(value)


def completed_subtask_count(state: EpisodeEvalState) -> int:
    return len(_final_status(state).get("completed", []) or [])


def navigation_progress_status(
    episode: EpisodeSpec,
    positions: list[XYZ],
    *,
    stop_step: int = -1,
) -> dict[str, Any]:
    """Compute ordered navigation-subtask status from the current trajectory."""

    current = positions[-1] if positions else episode.start_pose.position
    distance_from_start = euclidean_distance(episode.start_pose.position, current)
    stopped = stop_step >= 0

    viewpoints = _ovon_viewpoints(episode)
    if viewpoints:
        # ObjectNav (OVON) has no reference path; the goal is "reach any viewpoint of the
        # goal object". Track the minimum distance to any viewpoint and turn the distance
        # reduction into a progress ratio so the online monitor reflects real goal proximity
        # instead of the degenerate single-point [start] path (distance-from-spawn, progress=1).
        initial = _min_viewpoint_distance(episode.start_pose.position, viewpoints)
        distance_to_goal = _min_viewpoint_distance(current, viewpoints)
        best_distance = min(
            (_min_viewpoint_distance(position, viewpoints) for position in positions),
            default=distance_to_goal,
        )
        current_progress = _distance_progress(distance_to_goal, initial)
        best_progress = _distance_progress(best_distance, initial)
    else:
        dense_path = dense_gt_path_from_episode(episode)
        best_progress = max(
            (_reference_progress_ratio(position, dense_path) for position in positions), default=0.0
        )
        current_progress = _reference_progress_ratio(current, dense_path)
        distance_to_goal = distance_to_goal_along_dense_path(current, dense_path)

    subtasks = _subtask_defs(episode)
    completed: list[str] = []
    for subtask in subtasks:
        name = subtask["name"]
        kind = subtask["type"]
        radius = float(subtask.get("radius_m", episode.goal_radius_m) or episode.goal_radius_m)
        if kind == "distance_from_start":
            done = distance_from_start >= radius
        elif kind == "path_progress":
            done = best_progress >= float(subtask.get("progress_threshold", 0.0))
        elif kind in ("goal_distance", "objectnav_goal"):
            done = distance_to_goal <= radius
        elif kind == "stop_success":
            done = distance_to_goal <= radius and stopped
        else:
            done = False
        if done:
            completed.append(name)

    first_failed = next(
        (subtask["name"] for subtask in subtasks if subtask["name"] not in completed), None
    )
    active = first_failed or (subtasks[-1]["name"] if subtasks else "")
    score = sum(
        float(subtask.get("score", 0.0)) for subtask in subtasks if subtask["name"] in completed
    )
    total_score = sum(float(subtask.get("score", 0.0)) for subtask in subtasks) or 1.0
    return {
        "active": active,
        "completed": completed,
        "progress_score": min(score / total_score, 1.0),
        "subtask_completion_rate": len(completed) / max(len(subtasks), 1),
        "first_failed_subtask": first_failed,
        "current_path_progress": current_progress,
        "best_path_progress": best_progress,
        "distance_to_goal": distance_to_goal,
        "regressed": current_progress + 0.10 < best_progress,
    }


def _final_status(state: EpisodeEvalState) -> dict[str, Any]:
    key = ("progress", ())
    if key in state.cache:
        return state.cache[key]
    status = _last_trace_status(state.trace)
    if status is None:
        status = navigation_progress_status(
            state.episode,
            [step.position for step in state.trace.steps],
            stop_step=state.trace.stop_step,
        )
    state.cache[key] = status
    return status


def _last_trace_status(trace: RolloutTrace) -> dict[str, Any] | None:
    for step in reversed(trace.steps):
        payload = step.metadata.get("subtask_status")
        if isinstance(payload, dict):
            return dict(payload)
    return None


def _subtask_defs(episode: EpisodeSpec) -> list[dict[str, Any]]:
    if episode.subtasks:
        out: list[dict[str, Any]] = []
        for subtask in episode.subtasks:
            metadata = dict(subtask.metadata or {})
            name = str(metadata.get("name") or subtask.type)
            out.append(
                {
                    "name": name,
                    "type": subtask.type,
                    "score": float(metadata.get("score", 1.0)),
                    "progress_threshold": metadata.get("progress_threshold"),
                    "radius_m": subtask.radius_m,
                }
            )
        return out
    return [
        {"name": "leave_start_zone", "type": "distance_from_start", "score": 0.10, "radius_m": 1.0},
        {
            "name": "follow_reference_prefix_25",
            "type": "path_progress",
            "score": 0.20,
            "progress_threshold": 0.25,
        },
        {
            "name": "follow_reference_prefix_50",
            "type": "path_progress",
            "score": 0.20,
            "progress_threshold": 0.50,
        },
        {
            "name": "follow_reference_prefix_75",
            "type": "path_progress",
            "score": 0.20,
            "progress_threshold": 0.75,
        },
        {"name": "enter_goal_zone", "type": "goal_distance", "score": 0.20},
        {"name": "stop_in_goal_zone", "type": "stop_success", "score": 0.10},
    ]


def _reference_progress_ratio(position: XYZ, dense_path: list[XYZ]) -> float:
    if len(dense_path) <= 1:
        return 1.0
    closest_idx = min(
        range(len(dense_path)), key=lambda idx: euclidean_distance(position, dense_path[idx])
    )
    return closest_idx / max(len(dense_path) - 1, 1)


def _ovon_viewpoints(episode: EpisodeSpec) -> list[XYZ]:
    """Goal-object viewpoints (Isaac frame) for an OVON episode, else an empty list."""
    ovon = episode.metadata.get("ovon")
    if not isinstance(ovon, dict):
        return []
    viewpoints: list[XYZ] = []
    for goal in ovon.get("goals", []) or []:
        for vp in goal.get("viewpoints", []) or []:
            if isinstance(vp, list | tuple) and len(vp) >= 3:
                viewpoints.append((float(vp[0]), float(vp[1]), float(vp[2])))
    return viewpoints


def _min_viewpoint_distance(position: XYZ, viewpoints: list[XYZ]) -> float:
    return min(euclidean_distance(position, vp) for vp in viewpoints)


def _distance_progress(distance: float, initial: float) -> float:
    """Fraction of the initial start->goal distance covered, clamped to ``[0, 1]``."""
    if initial <= 0.0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - distance / initial))
