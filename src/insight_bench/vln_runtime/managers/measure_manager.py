"""Online measure manager for lightweight benchmark environments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import MeasureTermCfg


class MeasureManager(ManagerBase):
    """Compute online diagnostic measures from an environment-shaped state."""

    def __init__(self, cfg: object | None, env: Any):
        self._term_names: list[str] = []
        self._term_cfgs: list[MeasureTermCfg] = []
        self._class_term_cfgs: list[MeasureTermCfg] = []
        self._values: dict[str, Any] = {}
        super().__init__(cfg, env)

    @property
    def active_terms(self) -> list[str]:
        return self._term_names

    @property
    def values(self) -> dict[str, Any]:
        return dict(self._values)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        self._values.clear()
        for term_cfg in self._class_term_cfgs:
            term_cfg.func.reset(env_ids=env_ids)
        return {}

    def compute(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for name, term_cfg in zip(self._term_names, self._term_cfgs):
            values[name] = term_cfg.func(self._env, **term_cfg.params)
        self._values = values
        return dict(values)

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        del env_idx
        return [
            (name, [float(value)]) for name, value in self._values.items() if _is_numeric(value)
        ]

    def set_term_cfg(self, term_name: str, cfg: MeasureTermCfg) -> None:
        self._term_cfgs[self._term_names.index(term_name)] = cfg

    def get_term_cfg(self, term_name: str) -> MeasureTermCfg:
        return self._term_cfgs[self._term_names.index(term_name)]

    def _prepare_terms(self) -> None:
        for term_name, term_cfg in self._cfg_items():
            if term_cfg is None:
                continue
            if not isinstance(term_cfg, MeasureTermCfg):
                raise TypeError(
                    f"Configuration for measure {term_name!r} must be MeasureTermCfg, got {type(term_cfg)!r}."
                )
            self._resolve_common_term_cfg(term_name, term_cfg, min_argc=1)
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)
            if isinstance(term_cfg.func, ManagerTermBase):
                self._class_term_cfgs.append(term_cfg)


def _is_numeric(value: Any) -> bool:
    return isinstance(value, int | float | bool)
