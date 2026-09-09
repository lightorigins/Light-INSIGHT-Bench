"""Termination term functions for camera-walk benchmark environments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from insight_bench.vln_runtime.managers import ManagerTermBase, TerminationTermCfg
from insight_bench.vln_runtime.managers.termination_manager import TERMINATION_FUNC_REGISTRY


def policy_stop_requested(env: Any) -> list[bool]:
    """Terminate when the policy explicitly emits a stop action/token."""

    return [_policy_response_is_stop(getattr(env, "last_policy_response", None))]


def velocity_control_below_threshold(
    env: Any,
    min_abs_lin_speed_mps: float = 0.01,
    min_abs_ang_speed_dps: float = 0.5,
    consider_lateral_speed: bool = False,
) -> list[bool]:
    """Terminate when the decoded velocity command is effectively zero."""

    command = getattr(env, "last_velocity_command", None)
    if command is None:
        return [False]

    linear = abs(_command_float(command, "linear_velocity_mps", "linear_velocity", default=0.0))
    angular = abs(_command_float(command, "angular_velocity_dps", "angular_velocity", default=0.0))
    if linear > float(min_abs_lin_speed_mps) or angular > float(min_abs_ang_speed_dps):
        return [False]
    if consider_lateral_speed:
        lateral = abs(
            _command_float(command, "lateral_velocity_mps", "lateral_velocity", default=0.0)
        )
        return [lateral <= float(min_abs_lin_speed_mps)]
    return [True]


class VelocityControlBelowThresholdForDuration(ManagerTermBase):
    """Terminate after velocity control stays effectively zero for a duration."""

    def __init__(self, cfg: TerminationTermCfg, env: Any):
        super().__init__(cfg, env)
        self._stationary_duration_s = [0.0 for _ in range(self.num_envs)]

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        indices = range(self.num_envs) if env_ids is None else env_ids
        for env_id in indices:
            self._stationary_duration_s[int(env_id)] = 0.0

    def __call__(
        self,
        env: Any,
        min_abs_lin_speed_mps: float = 0.01,
        min_abs_ang_speed_dps: float = 0.5,
        duration_sec: float = 4.0,
        consider_lateral_speed: bool = False,
    ) -> list[bool]:
        command = getattr(env, "last_velocity_command", None)
        if command is None:
            self.reset()
            return [False for _ in range(self.num_envs)]

        is_stationary = _velocity_command_below_threshold(
            command,
            min_abs_lin_speed_mps=min_abs_lin_speed_mps,
            min_abs_ang_speed_dps=min_abs_ang_speed_dps,
            consider_lateral_speed=consider_lateral_speed,
        )
        dt_sec = _control_dt_sec(env)
        results: list[bool] = []
        for env_id in range(self.num_envs):
            if is_stationary:
                self._stationary_duration_s[env_id] += dt_sec
            else:
                self._stationary_duration_s[env_id] = 0.0
            results.append(self._stationary_duration_s[env_id] >= float(duration_sec))
        return results


def time_out(env: Any) -> list[bool]:
    """Terminate when episode length reaches the configured max length."""

    max_episode_length = int(env.max_episode_length)
    lengths = env.episode_length_buf
    return [int(length) >= max_episode_length for length in lengths]


def is_policy_stop_response(response: Any) -> bool:
    """Return whether a policy response should be treated as a STOP action."""

    return _policy_response_is_stop(response)


def _policy_response_is_stop(response: Any) -> bool:
    action_name = str(getattr(response, "action_name", "")).strip().lower()
    raw_output = str(getattr(response, "raw_output", "")).strip().lower()
    raw_prefix = raw_output.split("<|im_end|>", 1)[0].strip()
    return (
        action_name == "stop"
        or raw_prefix == "stop"
        or raw_prefix.startswith("stop ")
        or raw_prefix == "<stop>"
        or raw_prefix.startswith("<stop>")
    )


def _velocity_command_below_threshold(
    command: Any,
    *,
    min_abs_lin_speed_mps: float,
    min_abs_ang_speed_dps: float,
    consider_lateral_speed: bool,
) -> bool:
    linear = abs(_command_float(command, "linear_velocity_mps", "linear_velocity", default=0.0))
    angular = abs(_command_float(command, "angular_velocity_dps", "angular_velocity", default=0.0))
    if linear > float(min_abs_lin_speed_mps) or angular > float(min_abs_ang_speed_dps):
        return False
    if consider_lateral_speed:
        lateral = abs(
            _command_float(command, "lateral_velocity_mps", "lateral_velocity", default=0.0)
        )
        return lateral <= float(min_abs_lin_speed_mps)
    return True


def _command_float(command: Any, *names: str, default: float) -> float:
    for name in names:
        if isinstance(command, dict) and name in command:
            return float(command[name])
        if hasattr(command, name):
            return float(getattr(command, name))
    return default


def _control_dt_sec(env: Any) -> float:
    return float(getattr(env, "control_dt_sec", getattr(env, "dt_sec", 1.0)))


TERMINATION_FUNC_REGISTRY.update(
    {
        "policy_stop": policy_stop_requested,
        "policy_stop_requested": policy_stop_requested,
        "zero_velocity_stop": VelocityControlBelowThresholdForDuration,
        "velocity_control_below_threshold_for_duration": VelocityControlBelowThresholdForDuration,
        "velocity_control_below_threshold": velocity_control_below_threshold,
        "time_out": time_out,
        "max_steps": time_out,
        "max_steps_reached": time_out,
    }
)
