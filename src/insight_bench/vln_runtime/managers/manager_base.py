"""Small manager base modeled after IsaacLab's manager workflow."""

from __future__ import annotations

import copy
import importlib
import inspect
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from .manager_term_cfg import ManagerTermBaseCfg


class ManagerTermBase(ABC):
    """Base class for stateful callable manager terms."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: Any):
        self.cfg = cfg
        self._env = env

    @property
    def num_envs(self) -> int:
        return int(self._env.num_envs)

    @property
    def device(self) -> str:
        return str(getattr(self._env, "device", "cpu"))

    @property
    def __name__(self) -> str:
        return self.__class__.__name__

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        return None

    @abstractmethod
    def __call__(self, *args: Any) -> Any:
        raise NotImplementedError


class ManagerBase(ABC):
    """Base class for parsing config-shaped manager terms."""

    def __init__(self, cfg: object | None, env: Any):
        self.cfg = copy.deepcopy(cfg)
        self._env = env
        if self.cfg:
            self._prepare_terms()

    @property
    def num_envs(self) -> int:
        return int(self._env.num_envs)

    @property
    def device(self) -> str:
        return str(getattr(self._env, "device", "cpu"))

    @property
    @abstractmethod
    def active_terms(self) -> list[str] | dict[str, list[str]]:
        raise NotImplementedError

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        return {}

    def find_terms(self, name_keys: str | Sequence[str]) -> list[str]:
        keys = [name_keys] if isinstance(name_keys, str) else list(name_keys)
        active = self.active_terms
        names = (
            [term for terms in active.values() for term in terms]
            if isinstance(active, dict)
            else active
        )
        return [name for name in names if any(key == name or key in name for key in keys)]

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        raise NotImplementedError

    @abstractmethod
    def _prepare_terms(self) -> None:
        raise NotImplementedError

    def _cfg_items(self) -> list[tuple[str, Any]]:
        if isinstance(self.cfg, dict):
            return list(self.cfg.items())
        return [(key, value) for key, value in vars(self.cfg).items() if not key.startswith("_")]

    def _resolve_common_term_cfg(
        self, term_name: str, term_cfg: ManagerTermBaseCfg, *, min_argc: int = 1
    ) -> None:
        if not isinstance(term_cfg, ManagerTermBaseCfg):
            raise TypeError(
                f"Configuration for term {term_name!r} must be ManagerTermBaseCfg, got {type(term_cfg)!r}."
            )

        if isinstance(term_cfg.func, str):
            term_cfg.func = self._resolve_func_string(term_cfg.func)
        if not callable(term_cfg.func):
            raise AttributeError(f"The term {term_name!r} is not callable: {term_cfg.func!r}")

        if inspect.isclass(term_cfg.func):
            if not issubclass(term_cfg.func, ManagerTermBase):
                raise TypeError(f"Class term {term_name!r} must inherit ManagerTermBase.")
            func_static = term_cfg.func.__call__
            min_argc += 1
        else:
            func_static = term_cfg.func

        self._validate_term_signature(term_name, func_static, term_cfg, min_argc=min_argc)

        if inspect.isclass(term_cfg.func):
            term_cfg.func = term_cfg.func(cfg=term_cfg, env=self._env)

    def _resolve_func_string(self, func_name: str) -> Any:
        if ":" in func_name:
            module_name, attr_name = func_name.split(":", 1)
            return getattr(importlib.import_module(module_name), attr_name)
        if "." in func_name:
            module_name, attr_name = func_name.rsplit(".", 1)
            return getattr(importlib.import_module(module_name), attr_name)
        raise ValueError(f"Cannot resolve unqualified manager term function {func_name!r}.")

    @staticmethod
    def _validate_term_signature(
        term_name: str,
        func: Any,
        term_cfg: ManagerTermBaseCfg,
        *,
        min_argc: int,
    ) -> None:
        signature = inspect.signature(func)
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

        if not has_var_positional and len(positional) < min_argc:
            raise ValueError(
                f"Term {term_name!r} must accept the environment as its first argument."
            )

        required_after_env = [
            parameter.name
            for parameter in positional[min_argc:]
            if parameter.default is inspect.Parameter.empty
        ]
        missing = [name for name in required_after_env if name not in term_cfg.params]
        if missing:
            raise ValueError(f"Term {term_name!r} is missing mandatory params: {missing}")

        if has_var_keyword:
            return
        valid_names = {parameter.name for parameter in positional[min_argc:]}
        extra = [name for name in term_cfg.params if name not in valid_names]
        if extra:
            raise ValueError(f"Term {term_name!r} received unexpected params: {extra}")
