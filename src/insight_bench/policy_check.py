"""``insight-bench check-policy`` -- drive a policy server the way a run will.

The contract document describes what the runner does with a reply; this makes it
observable. Without it, the only way to learn that a plan decodes to a stop, or
that a forward delta is clamped to a fifth of what the model asked for, is to run
the whole suite and read the trace afterwards. A handful of steps against a
synthetic frame answers the same question in a minute, before any data or scene
conversion is in place -- and it answers the one question that otherwise fails a
run at its most expensive moment: whether ``/health`` names a model at all.
"""

from __future__ import annotations

from typing import Any

#: Enough of a frame for the wire. What is being checked is the reply.
_FRAME_SHAPE = (270, 480, 3)

#: The suite whose motion parameters a checked reply is decoded against.
_TASK = "insight_bench"


class _Env:
    """The one attribute ``velocity_control_below_threshold`` reads."""

    def __init__(self, command: Any) -> None:
        self.last_velocity_command = command


def check_policy(
    policy_url: str, *, steps: int, instruction: str, timeout_sec: float
) -> dict[str, Any]:
    """Run ``/health -> /reset -> /act * steps -> /finish`` and report each step.

    Every step reports what the runner would execute, not what the server
    replied: which waypoint of the plan is selected, the velocity command that
    decodes to after clamping, and whether that reads as an arrival.
    """
    import numpy as np

    from insight_bench.vln_runtime.motion.waypoint_execution import (
        velocity_control_from_waypoint,
        waypoint_from_policy_response,
    )
    from insight_bench.vln_runtime.policy.client import LocalVlnPolicyClient
    from insight_bench.vln_runtime.suite.registry import get_task_config
    from insight_bench.vln_runtime.terminations.terms import (
        is_policy_stop_response,
        velocity_control_below_threshold,
    )

    task = get_task_config(_TASK)
    motion = dict(task.metadata.get("rollout", {}))
    zero_stop = task.terminations.zero_velocity_stop  # type: ignore[attr-defined]
    thresholds = dict(zero_stop.params)

    client = LocalVlnPolicyClient(base_url=policy_url, timeout_sec=timeout_sec)
    health = client.health()

    episode_id = "check-policy"
    frame = np.zeros(_FRAME_SHAPE, dtype=np.uint8)
    client.reset(instruction, episode_id=episode_id, first_frame_rgb=None)

    records: list[dict[str, Any]] = []
    stopped_at: int | None = None
    try:
        for step in range(steps):
            response = client.act(frame, episode_id=episode_id, frame_id=step)
            explicit_stop = is_policy_stop_response(response)
            if explicit_stop:
                # The rollout checks this first and never decodes the plan of a
                # reply that says it arrived -- an explicit stop may carry no
                # waypoints at all. Mirror that order, or this would raise on
                # exactly the replies it exists to confirm.
                records.append(
                    {
                        "step": step,
                        "action_name": response.action_name,
                        "waypoints_returned": response.waypoint_horizon,
                        "waypoint_index_executed": None,
                        "cluster_id": response.waypoint_cluster_id,
                        "requested": None,
                        "executed_velocity": None,
                        "stop_kind": "policy_stop",
                    }
                )
                stopped_at = step
                break
            index, waypoint = waypoint_from_policy_response(
                response,
                None,
                waypoint_atol=float(motion["waypoint_atol"]),
                yaw_lookahead_steps=int(motion["yaw_lookahead_steps"]),
                yaw_lookahead_hybrid=bool(motion["yaw_lookahead_hybrid"]),
                yaw_hybrid_thresh_rad=float(motion.get("yaw_hybrid_thresh_rad", 0.0)),
            )
            command = velocity_control_from_waypoint(
                waypoint,
                dt_sec=float(motion["control_dt_sec"]),
                lin_vel_range_mps=tuple(motion["lin_vel_range_mps"]),
                ang_vel_range_dps=tuple(motion["ang_vel_range_dps"]),
                drop_lateral=bool(motion["drop_lateral"]),
            )
            zero = bool(velocity_control_below_threshold(_Env(command), **thresholds)[0])
            records.append(
                {
                    "step": step,
                    "action_name": response.action_name,
                    "waypoints_returned": response.waypoint_horizon,
                    "waypoint_index_executed": index,
                    "cluster_id": response.waypoint_cluster_id,
                    "requested": waypoint.to_dict(),
                    "executed_velocity": command.to_dict(),
                    "stop_kind": "zero_velocity_stop" if zero else None,
                }
            )
            if zero:
                stopped_at = step
                break
    finally:
        client.finish(episode_id=episode_id)

    return {
        "policy_url": policy_url,
        # Said out loud in the report: these steps prove the wire, not the model.
        # The frame is a synthetic black image, so "it kept walking" says nothing
        # about whether it would stop on a real observation.
        "frame": "synthetic black 480x270; this checks the wire, not the policy",
        "model_id": str(health.get("model_id") or "").strip(),
        "health": dict(health),
        "steps_taken": len(records),
        "steps": records,
        "stopped_at_step": stopped_at,
        # Reported, not judged. A model that never stops on a real observation is
        # scored wherever the budget leaves it, which is expensive -- but that
        # cannot be concluded from a synthetic frame and a handful of steps.
        "never_stopped": stopped_at is None,
        "step_budget": int(motion["num_steps"]),
    }
