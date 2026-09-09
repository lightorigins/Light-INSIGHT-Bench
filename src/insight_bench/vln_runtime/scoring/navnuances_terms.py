"""Offline NavNuances score terms (per-capability success protocols)."""

from __future__ import annotations

from insight_bench.vln_runtime.scoring.navnuances import NavNuancesScore, evaluate_navnuances

from .state import EpisodeEvalState


def _score(state: EpisodeEvalState) -> NavNuancesScore:
    key = ("navnuances", ())
    if key not in state.cache:
        state.cache[key] = evaluate_navnuances(state.episode, state.trace)
    return state.cache[key]


def success(state: EpisodeEvalState) -> bool:
    return _score(state).success


def spl(state: EpisodeEvalState) -> float:
    return _score(state).spl


def distance_to_goal(state: EpisodeEvalState) -> float:
    return _score(state).distance_to_goal


def path_length(state: EpisodeEvalState) -> float:
    return _score(state).path_length


def stop_step(state: EpisodeEvalState) -> int:
    return _score(state).stop_step


def capability(state: EpisodeEvalState) -> str:
    score = _score(state)
    return f"{score.capability}_{score.subtype}"


def stopped(state: EpisodeEvalState) -> bool:
    return _score(state).stopped
