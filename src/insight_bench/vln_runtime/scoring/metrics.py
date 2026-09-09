"""Basic IsaacLab-free navigation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from insight_bench.vln_runtime.episodes import EpisodeSpec
from insight_bench.vln_runtime.traces.schema import RolloutTrace, distance_xy


@dataclass(frozen=True)
class NavigationEpisodeScore:
    """Official-style scalar navigation metrics for one episode."""

    episode_id: str
    sr: bool
    spl: float
    ne: float
    osr: float
    path_length: float
    oracle_ne: float
    stop_step: int
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "sr": bool(self.sr),
            "spl": float(self.spl),
            "ne": float(self.ne),
            "osr": float(self.osr),
            "path_length": float(self.path_length),
            "oracle_ne": float(self.oracle_ne),
            "stop_step": int(self.stop_step),
            "failure_reason": self.failure_reason,
        }


def compute_spl(*, success: bool, shortest_path: float, path_length: float) -> float:
    if not success:
        return 0.0
    if shortest_path <= 0.0:
        return 1.0 if path_length <= 0.0 else 0.0
    if path_length <= 0.0:
        return 1.0
    return float(shortest_path / max(path_length, shortest_path))


def shortest_path_from_episode(episode: EpisodeSpec) -> float:
    if len(episode.reference_path) >= 2:
        return sum(
            distance_xy(prev, cur)
            for prev, cur in zip(episode.reference_path, episode.reference_path[1:])
        )
    if episode.goal_position is None:
        return 0.0
    return distance_xy(episode.start_pose.position, episode.goal_position)


def evaluate_navigation(
    episode: EpisodeSpec,
    trace: RolloutTrace,
    *,
    require_stop: bool = True,
) -> NavigationEpisodeScore:
    """Evaluate basic navigation metrics from an episode spec and rollout trace."""

    if episode.goal_position is None:
        return NavigationEpisodeScore(
            episode_id=episode.episode_id,
            sr=False,
            spl=0.0,
            ne=0.0,
            osr=0.0,
            path_length=trace.path_length_xy(),
            oracle_ne=0.0,
            stop_step=trace.stop_step,
            failure_reason="missing_goal",
        )

    final_position = trace.final_position()
    if final_position is None:
        return NavigationEpisodeScore(
            episode_id=episode.episode_id,
            sr=False,
            spl=0.0,
            ne=float("inf"),
            osr=0.0,
            path_length=0.0,
            oracle_ne=float("inf"),
            stop_step=trace.stop_step,
            failure_reason="empty_trace",
        )

    ne = distance_xy(final_position, episode.goal_position)
    oracle_ne = trace.oracle_distance_xy(episode.goal_position)
    stopped = trace.stop_step >= 0
    success = ne <= episode.goal_radius_m and (stopped or not require_stop)
    osr = 1.0 if oracle_ne <= episode.goal_radius_m else 0.0
    path_length = trace.path_length_xy()
    shortest_path = shortest_path_from_episode(episode)
    spl = compute_spl(success=success, shortest_path=shortest_path, path_length=path_length)

    failure_reason = None
    if not success:
        if require_stop and not stopped:
            failure_reason = "stop_not_requested"
        else:
            failure_reason = "goal_not_reached"

    return NavigationEpisodeScore(
        episode_id=episode.episode_id,
        sr=success,
        spl=spl,
        ne=ne,
        osr=osr,
        path_length=path_length,
        oracle_ne=oracle_ne,
        stop_step=trace.stop_step,
        failure_reason=failure_reason,
    )
