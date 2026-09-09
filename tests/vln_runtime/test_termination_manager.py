from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from insight_bench.vln_runtime.managers import DoneTerm, TerminationManager, TerminationTermCfg
from insight_bench.vln_runtime.motion.waypoint_execution import VelocityControlCommand
from insight_bench.vln_runtime.policy.client import PolicyResponse
from insight_bench.vln_runtime.suite.insight_bench.config import default_task_config
from insight_bench.vln_runtime.terminations import terms as termination_terms


class DummyEnv:
    num_envs = 1
    device = "cpu"

    def __init__(self) -> None:
        self.episode_length_buf = [0]
        self.max_episode_length = 50
        self.control_dt_sec = 0.1
        self.last_policy_response: Any | None = None
        self.last_velocity_command: VelocityControlCommand | None = None


@dataclass
class PolicyStopCfg:
    policy_stop: TerminationTermCfg = field(
        default_factory=lambda: DoneTerm(func=termination_terms.policy_stop_requested)
    )


@dataclass
class ZeroVelocityCfg:
    zero_velocity_stop: TerminationTermCfg = field(
        default_factory=lambda: DoneTerm(
            func=termination_terms.VelocityControlBelowThresholdForDuration,
            params={
                "min_abs_lin_speed_mps": 0.01,
                "min_abs_ang_speed_dps": 0.5,
                "duration_sec": 0.15,
            },
        )
    )


@dataclass
class TimeOutCfg:
    time_out: TerminationTermCfg = field(
        default_factory=lambda: DoneTerm(func=termination_terms.time_out, time_out=True)
    )


def test_policy_stop_sets_terminated_buffer() -> None:
    env = DummyEnv()
    env.last_policy_response = PolicyResponse(success=True, action_name="STOP", raw_output="<stop>")
    manager = TerminationManager(PolicyStopCfg(), env)

    result = manager.compute()

    assert result == (True,)
    assert manager.terminated == (True,)
    assert manager.time_outs == (False,)
    assert manager.triggered_terms() == ("policy_stop",)


def test_zero_velocity_stop_requires_continuous_duration() -> None:
    env = DummyEnv()
    env.last_velocity_command = VelocityControlCommand(
        linear_velocity_mps=0.0,
        lateral_velocity_mps=0.0,
        angular_velocity_dps=0.0,
    )
    manager = TerminationManager(ZeroVelocityCfg(), env)

    first_result = manager.compute()
    second_result = manager.compute()

    assert first_result == (False,)
    assert second_result == (True,)
    assert manager.terminated == (True,)
    assert manager.triggered_terms() == ("zero_velocity_stop",)


def test_zero_velocity_stop_string_alias_uses_duration_term() -> None:
    env = DummyEnv()
    env.last_velocity_command = VelocityControlCommand(
        linear_velocity_mps=0.0,
        lateral_velocity_mps=0.0,
        angular_velocity_dps=0.0,
    )
    manager = TerminationManager(
        {"zero_velocity_stop": DoneTerm(func="zero_velocity_stop", params={"duration_sec": 0.15})},
        env,
    )

    assert manager.compute() == (False,)
    assert manager.compute() == (True,)


def test_nonzero_velocity_does_not_stop() -> None:
    env = DummyEnv()
    env.last_velocity_command = VelocityControlCommand(
        linear_velocity_mps=0.2,
        lateral_velocity_mps=0.0,
        angular_velocity_dps=0.0,
    )
    manager = TerminationManager(ZeroVelocityCfg(), env)

    manager.compute()
    env.last_velocity_command = VelocityControlCommand(
        linear_velocity_mps=0.2,
        lateral_velocity_mps=0.0,
        angular_velocity_dps=0.0,
    )
    result = manager.compute()

    assert result == (False,)
    assert manager.triggered_terms() == ()


def test_time_out_sets_timeout_not_terminated() -> None:
    env = DummyEnv()
    env.episode_length_buf = [50]
    manager = TerminationManager(TimeOutCfg(), env)

    result = manager.compute()

    assert result == (True,)
    assert manager.terminated == (False,)
    assert manager.time_outs == (True,)
    assert manager.triggered_terms() == ("time_out",)


def test_objnav_default_config_enables_zero_velocity_stop() -> None:
    # The published suite stops on the FIRST near-zero velocity command, which is
    # why the term carries speed thresholds and no `duration_sec`: a waypoint
    # policy signals stop by emitting its zero-motion cluster, and waiting out a
    # duration walked the agent past the goal before it halted.
    task = default_task_config()
    env = DummyEnv()
    manager = TerminationManager(task.terminations, env)

    assert manager.active_terms == ["policy_stop", "zero_velocity_stop", "time_out"]
    zero_velocity_cfg = manager.get_term_cfg("zero_velocity_stop")
    assert zero_velocity_cfg.params["min_abs_lin_speed_mps"] == 0.01
    assert "duration_sec" not in zero_velocity_cfg.params
    assert zero_velocity_cfg.time_out is False
    env.last_velocity_command = VelocityControlCommand(
        linear_velocity_mps=0.0,
        lateral_velocity_mps=0.0,
        angular_velocity_dps=0.0,
    )
    assert manager.compute() == (True,)
