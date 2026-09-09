"""Field-configured offline scoring manager."""

from __future__ import annotations

import copy
import importlib
import inspect
from dataclasses import dataclass, field
from typing import Any

from insight_bench.vln_runtime.episodes import EpisodeSpec
from insight_bench.vln_runtime.traces import RolloutTrace

from . import navigation_terms
from .metrics import NavigationEpisodeScore, evaluate_navigation
from .navnuances import NavNuancesScore
from .state import EpisodeEvalState
from .term_cfg import ScoreTerm, ScoreTermCfg
from .vlnce import VlnceEpisodeScore, evaluate_vlnce_navigation

RawScore = NavigationEpisodeScore | VlnceEpisodeScore | NavNuancesScore


@dataclass
class NavigationScoresCfg:
    """Default offline scores for generic navigation."""

    sr: ScoreTermCfg = field(default_factory=lambda: ScoreTerm(func=navigation_terms.sr))
    spl: ScoreTermCfg = field(default_factory=lambda: ScoreTerm(func=navigation_terms.spl))
    ne: ScoreTermCfg = field(default_factory=lambda: ScoreTerm(func=navigation_terms.ne))
    osr: ScoreTermCfg = field(default_factory=lambda: ScoreTerm(func=navigation_terms.osr))
    path_length: ScoreTermCfg = field(
        default_factory=lambda: ScoreTerm(func=navigation_terms.path_length)
    )
    oracle_ne: ScoreTermCfg = field(
        default_factory=lambda: ScoreTerm(func=navigation_terms.oracle_ne)
    )
    stop_step: ScoreTermCfg = field(
        default_factory=lambda: ScoreTerm(func=navigation_terms.stop_step)
    )


class ScoreManagerCfg:
    """Factory namespace for offline scoring configs."""

    @classmethod
    def default_navigation(cls, *, require_stop: bool = True) -> NavigationScoresCfg:
        return NavigationScoresCfg(
            sr=ScoreTerm(func=navigation_terms.sr, params={"require_stop": require_stop}),
            spl=ScoreTerm(func=navigation_terms.spl, params={"require_stop": require_stop}),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> object:
        if not payload:
            return cls.default_navigation()
        return {
            str(name): ScoreTerm.from_dict(dict(term_payload))
            for name, term_payload in payload.items()
            if term_payload is not None
        }


@dataclass(frozen=True)
class EpisodeScore:
    """A config-shaped score record for one episode."""

    episode_id: str
    metrics: dict[str, float | int | bool | str]
    failure_reason: str | None = None
    raw_navigation: RawScore | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "metrics": self.metrics,
            "failure_reason": self.failure_reason,
            "raw_navigation": None
            if self.raw_navigation is None
            else self.raw_navigation.to_dict(),
        }


class ScoreManager:
    """Runs configured score terms over a finished episode artifact."""

    def __init__(self, cfg: object | None = None):
        self.cfg = copy.deepcopy(cfg or ScoreManagerCfg.default_navigation())
        self._term_names: list[str] = []
        self._term_cfgs: list[ScoreTermCfg] = []
        self._prepare_terms()

    @property
    def active_terms(self) -> list[str]:
        return list(self._term_names)

    def evaluate(
        self,
        state: EpisodeEvalState,
    ) -> EpisodeScore:
        metrics: dict[str, float | int | bool] = {}
        for name, term_cfg in zip(self._term_names, self._term_cfgs):
            metrics[name] = term_cfg.func(state, **term_cfg.params)
        raw_score = _raw_score_from_cache(state)
        return EpisodeScore(
            episode_id=state.episode.episode_id,
            metrics=metrics,
            failure_reason=None if raw_score is None else raw_score.failure_reason,
            raw_navigation=raw_score,
        )

    def evaluate_episode(
        self,
        episode: EpisodeSpec,
        trace: RolloutTrace,
        *,
        final_measures: dict[str, Any] | None = None,
        task_config: Any | None = None,
    ) -> EpisodeScore:
        return self.evaluate(
            EpisodeEvalState(
                episode=episode,
                trace=trace,
                final_measures=dict(final_measures or trace.final_measures),
                task_config=task_config,
            )
        )

    def _prepare_terms(self) -> None:
        for term_name, term_cfg in _cfg_items(self.cfg):
            if term_cfg is None:
                continue
            if not isinstance(term_cfg, ScoreTermCfg):
                raise TypeError(
                    f"Configuration for score {term_name!r} must be ScoreTermCfg, got {type(term_cfg)!r}."
                )
            _resolve_score_term_cfg(term_name, term_cfg)
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)


def _cfg_items(cfg: object) -> list[tuple[str, Any]]:
    if isinstance(cfg, dict):
        return list(cfg.items())
    return [(key, value) for key, value in vars(cfg).items() if not key.startswith("_")]


def _resolve_score_term_cfg(term_name: str, term_cfg: ScoreTermCfg) -> None:
    if isinstance(term_cfg.func, str):
        term_cfg.func = _resolve_func_string(term_cfg.func)
    if not callable(term_cfg.func):
        raise AttributeError(f"The score term {term_name!r} is not callable: {term_cfg.func!r}")
    _validate_score_term_signature(term_name, term_cfg)


def _resolve_func_string(func_name: str) -> Any:
    if ":" in func_name:
        module_name, attr_name = func_name.split(":", 1)
        return getattr(importlib.import_module(module_name), attr_name)
    if "." in func_name:
        module_name, attr_name = func_name.rsplit(".", 1)
        return getattr(importlib.import_module(module_name), attr_name)
    raise ValueError(f"Cannot resolve unqualified score term function {func_name!r}.")


def _validate_score_term_signature(term_name: str, term_cfg: ScoreTermCfg) -> None:
    signature = inspect.signature(term_cfg.func)
    positional = []
    has_var_positional = False
    has_var_keyword = False
    for parameter in signature.parameters.values():
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional.append(parameter)
        elif parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            has_var_positional = True
        elif parameter.kind == inspect.Parameter.VAR_KEYWORD:
            has_var_keyword = True

    if not has_var_positional and len(positional) < 1:
        raise ValueError(
            f"Score term {term_name!r} must accept EpisodeEvalState as its first argument."
        )

    required_after_state = [
        parameter.name
        for parameter in positional[1:]
        if parameter.default is inspect.Parameter.empty
    ]
    missing = [name for name in required_after_state if name not in term_cfg.params]
    if missing:
        raise ValueError(f"Score term {term_name!r} is missing mandatory params: {missing}")

    if has_var_keyword:
        return
    valid_names = {parameter.name for parameter in positional[1:]}
    extra = [name for name in term_cfg.params if name not in valid_names]
    if extra:
        raise ValueError(f"Score term {term_name!r} received unexpected params: {extra}")


def _raw_score_from_cache(
    state: EpisodeEvalState,
) -> RawScore | None:
    """Return the score the metrics terms already computed, not a fresh one.

    The raw block and ``failure_reason`` beside a run's metrics have to come
    from the protocol that produced those metrics. Recomputing here with a
    different success criterion is how a run ends up reporting a reason that
    contradicts its own numbers.
    """
    scored = [
        value
        for (kind, _params), value in state.cache.items()
        if kind in {"navigation", "navnuances", "vlnce"}
    ]
    if scored:
        return scored[0]
    if _looks_like_vlnce(state):
        return evaluate_vlnce_navigation(state.episode, state.trace, require_stop=True)
    return evaluate_navigation(state.episode, state.trace, require_stop=True)


def _looks_like_vlnce(state: EpisodeEvalState) -> bool:
    task_name = getattr(state.task_config, "name", None)
    return task_name == "vlnce_r2r" or "gt_locations" in state.episode.metadata
