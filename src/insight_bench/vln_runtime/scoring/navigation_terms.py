"""Offline navigation score terms."""

from __future__ import annotations

from insight_bench.vln_runtime.scoring.metrics import NavigationEpisodeScore, evaluate_navigation

from .state import EpisodeEvalState


def sr(state: EpisodeEvalState, require_stop: bool = True) -> bool:
    return _navigation_score(state, require_stop=require_stop).sr


def spl(state: EpisodeEvalState, require_stop: bool = True) -> float:
    return _navigation_score(state, require_stop=require_stop).spl


def ne(state: EpisodeEvalState, require_stop: bool = True) -> float:
    del require_stop
    return _navigation_score(state, require_stop=False).ne


def osr(state: EpisodeEvalState, require_stop: bool = True) -> float:
    del require_stop
    return _navigation_score(state, require_stop=False).osr


def path_length(state: EpisodeEvalState, require_stop: bool = True) -> float:
    del require_stop
    return _navigation_score(state, require_stop=False).path_length


def oracle_ne(state: EpisodeEvalState, require_stop: bool = True) -> float:
    del require_stop
    return _navigation_score(state, require_stop=False).oracle_ne


def stop_step(state: EpisodeEvalState) -> int:
    return int(state.trace.stop_step)


def _navigation_score(state: EpisodeEvalState, *, require_stop: bool) -> NavigationEpisodeScore:
    key = ("navigation", (("require_stop", require_stop),))
    if key not in state.cache:
        state.cache[key] = evaluate_navigation(
            state.episode, state.trace, require_stop=require_stop
        )
    return state.cache[key]
