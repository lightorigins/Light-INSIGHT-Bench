"""Termination manager for lightweight benchmark environments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import TerminationTermCfg


class TerminationManager(ManagerBase):
    """Compute terminated and timeout signals from configured terms."""

    def __init__(self, cfg: object | None, env: Any):
        self._term_names: list[str] = []
        self._term_cfgs: list[TerminationTermCfg] = []
        self._class_term_cfgs: list[TerminationTermCfg] = []
        super().__init__(cfg, env)
        self._term_name_to_term_idx = {name: index for index, name in enumerate(self._term_names)}
        self._term_values = [[False for _ in self._term_names] for _ in range(self.num_envs)]
        self._terminated_buf = [False for _ in range(self.num_envs)]
        self._time_out_buf = [False for _ in range(self.num_envs)]

    @property
    def active_terms(self) -> list[str]:
        return self._term_names

    @property
    def dones(self) -> tuple[bool, ...]:
        return tuple(
            terminated or time_out
            for terminated, time_out in zip(self._terminated_buf, self._time_out_buf)
        )

    @property
    def terminated(self) -> tuple[bool, ...]:
        return tuple(self._terminated_buf)

    @property
    def time_outs(self) -> tuple[bool, ...]:
        return tuple(self._time_out_buf)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        indices = self._resolve_env_ids(env_ids)
        extras: dict[str, float] = {}
        denominator = max(len(indices), 1)
        for term_index, term_name in enumerate(self._term_names):
            extras[f"Episode_Termination/{term_name}"] = (
                sum(float(self._term_values[env_id][term_index]) for env_id in indices)
                / denominator
            )
        for env_id in indices:
            self._terminated_buf[env_id] = False
            self._time_out_buf[env_id] = False
            for term_index in range(len(self._term_names)):
                self._term_values[env_id][term_index] = False
        for term_cfg in self._class_term_cfgs:
            term_cfg.func.reset(env_ids=env_ids)
        return extras

    def compute(self) -> tuple[bool, ...]:
        self._terminated_buf = [False for _ in range(self.num_envs)]
        self._time_out_buf = [False for _ in range(self.num_envs)]
        self._term_values = [[False for _ in self._term_names] for _ in range(self.num_envs)]

        for term_index, term_cfg in enumerate(self._term_cfgs):
            values = self._as_bool_list(term_cfg.func(self._env, **term_cfg.params))
            for env_id, value in enumerate(values):
                if not value:
                    continue
                self._term_values[env_id][term_index] = True
                if term_cfg.time_out:
                    self._time_out_buf[env_id] = True
                else:
                    self._terminated_buf[env_id] = True
        return self.dones

    def get_term(self, name: str) -> tuple[bool, ...]:
        term_index = self._term_name_to_term_idx[name]
        return tuple(row[term_index] for row in self._term_values)

    def triggered_terms(self, env_idx: int = 0) -> tuple[str, ...]:
        return tuple(
            name for name, value in zip(self._term_names, self._term_values[env_idx]) if value
        )

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        return [
            (name, [float(self._term_values[env_idx][term_index])])
            for term_index, name in enumerate(self._term_names)
        ]

    def set_term_cfg(self, term_name: str, cfg: TerminationTermCfg) -> None:
        self._term_cfgs[self._term_name_to_term_idx[term_name]] = cfg

    def get_term_cfg(self, term_name: str) -> TerminationTermCfg:
        return self._term_cfgs[self._term_name_to_term_idx[term_name]]

    def _prepare_terms(self) -> None:
        for term_name, term_cfg in self._cfg_items():
            if term_cfg is None:
                continue
            if not isinstance(term_cfg, TerminationTermCfg):
                raise TypeError(
                    f"Configuration for term {term_name!r} must be TerminationTermCfg, got {type(term_cfg)!r}."
                )
            self._resolve_common_term_cfg(term_name, term_cfg, min_argc=1)
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)
            if isinstance(term_cfg.func, ManagerTermBase):
                self._class_term_cfgs.append(term_cfg)

    def _resolve_func_string(self, func_name: str) -> Any:
        if func_name in TERMINATION_FUNC_REGISTRY:
            return TERMINATION_FUNC_REGISTRY[func_name]
        return super()._resolve_func_string(func_name)

    def _as_bool_list(self, value: Any) -> list[bool]:
        if isinstance(value, bool):
            return [value for _ in range(self.num_envs)]
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, tuple):
            value = list(value)
        if not isinstance(value, list):
            raise TypeError(
                f"Termination term must return bool or sequence of bools, got {type(value)!r}"
            )
        if len(value) != self.num_envs:
            raise ValueError(
                f"Termination term returned {len(value)} values for num_envs={self.num_envs}"
            )
        return [bool(item) for item in value]

    def _resolve_env_ids(self, env_ids: Sequence[int] | None) -> list[int]:
        if env_ids is None:
            return list(range(self.num_envs))
        return [int(env_id) for env_id in env_ids]


TERMINATION_FUNC_REGISTRY: dict[str, Any] = {}
