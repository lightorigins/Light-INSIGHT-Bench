"""Rollout trace schema used by online measures and offline scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

XYZ = tuple[float, float, float]
QUAT_WXYZ = tuple[float, float, float, float]


def _float3(value: Any, *, name: str) -> XYZ:
    if not isinstance(value, list | tuple) or len(value) < 3:
        raise ValueError(f"{name} must be a 3-element sequence, got {value!r}")
    return float(value[0]), float(value[1]), float(value[2])


def distance_xy(a: XYZ, b: XYZ) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return math.hypot(dx, dy)


def distance_xyz(a: XYZ, b: XYZ) -> float:
    """Full 3D euclidean distance. Unlike :func:`distance_xy` the height (z) axis is
    included, so points on different floors are correctly far apart."""
    return math.dist(
        (float(a[0]), float(a[1]), float(a[2])), (float(b[0]), float(b[1]), float(b[2]))
    )


@dataclass(frozen=True)
class MeasureTrace:
    """Per-step or final measurement payload."""

    name: str
    value: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MeasureTrace:
        return cls(
            name=str(payload["name"]),
            value=payload.get("value"),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class StepTrace:
    """One recorded simulator step."""

    step: int
    time_s: float
    position: XYZ
    orientation_wxyz: QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)
    action: dict[str, Any] = field(default_factory=dict)
    measures: dict[str, Any] = field(default_factory=dict)
    observation: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": int(self.step),
            "time_s": float(self.time_s),
            "position": list(self.position),
            "orientation_wxyz": list(self.orientation_wxyz),
            "action": self.action,
            "measures": self.measures,
            "observation": self.observation,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StepTrace:
        orientation = payload.get(
            "orientation_wxyz", payload.get("orientation", (1.0, 0.0, 0.0, 0.0))
        )
        if not isinstance(orientation, list | tuple) or len(orientation) < 4:
            raise ValueError(
                f"step.orientation_wxyz must be a 4-element sequence, got {orientation!r}"
            )
        return cls(
            step=int(payload["step"]),
            time_s=float(payload.get("time_s", 0.0)),
            position=_float3(payload["position"], name="step.position"),
            orientation_wxyz=(
                float(orientation[0]),
                float(orientation[1]),
                float(orientation[2]),
                float(orientation[3]),
            ),
            action=dict(payload.get("action") or {}),
            measures=dict(payload.get("measures") or {}),
            observation=dict(payload.get("observation") or {}),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class RolloutTrace:
    """Episode rollout evidence consumed by offline scoring."""

    episode_id: str
    instruction: str = ""
    steps: list[StepTrace] = field(default_factory=list)
    final_measures: dict[str, Any] = field(default_factory=dict)
    termination_reason: str | None = None
    stop_step: int = -1
    metadata: dict[str, Any] = field(default_factory=dict)

    def path_length_xy(self) -> float:
        if len(self.steps) < 2:
            return 0.0
        return sum(
            distance_xy(prev.position, cur.position)
            for prev, cur in zip(self.steps, self.steps[1:])
        )

    def final_position(self) -> XYZ | None:
        return self.steps[-1].position if self.steps else None

    def oracle_distance_xy(self, goal_position: XYZ) -> float:
        if not self.steps:
            return float("inf")
        return min(distance_xy(step.position, goal_position) for step in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "instruction": self.instruction,
            "steps": [step.to_dict() for step in self.steps],
            "final_measures": self.final_measures,
            "termination_reason": self.termination_reason,
            "stop_step": int(self.stop_step),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RolloutTrace:
        return cls(
            episode_id=str(payload["episode_id"]),
            instruction=str(payload.get("instruction", "")),
            steps=[StepTrace.from_dict(item) for item in payload.get("steps", [])],
            final_measures=dict(payload.get("final_measures") or {}),
            termination_reason=(
                None
                if payload.get("termination_reason") is None
                else str(payload.get("termination_reason"))
            ),
            stop_step=int(payload.get("stop_step", -1)),
            metadata=dict(payload.get("metadata") or {}),
        )

    @classmethod
    def from_camera_walk_records(
        cls,
        *,
        episode_id: str,
        instruction: str,
        records: list[dict[str, Any]],
        termination_reason: str | None = None,
        stop_step: int = -1,
        final_measures: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        include_initial_pose: bool = True,
        initial_metadata: dict[str, Any] | None = None,
    ) -> RolloutTrace:
        """Convert the existing camera-walk JSONL records into a RolloutTrace.

        *initial_metadata* is the provenance of step 0. Step 0 is the pre-action
        pose and has no step record of its own, so without this it was the one
        step in every trace whose metadata was empty -- and a reader checking
        "does every step say where its pose came from" had to special-case it or
        report it as missing. It carries the same ``pose_source`` /
        ``commanded_pose`` / ``orientation_wxyz`` fields every other step does.
        """

        steps: list[StepTrace] = []
        if include_initial_pose and records:
            pose_before = records[0].get("pose_before")
            if isinstance(pose_before, dict):
                steps.append(
                    _camera_walk_pose_to_step_trace(
                        step=0,
                        time_s=0.0,
                        pose=pose_before,
                        metadata=dict(initial_metadata or {}),
                    )
                )

        for record in records:
            pose = record.get("pose_after") or {}
            if not isinstance(pose, dict):
                continue
            record_step = int(record.get("step", len(steps)))
            steps.append(
                _camera_walk_pose_to_step_trace(
                    step=record_step + 1 if include_initial_pose else record_step,
                    time_s=float(
                        record.get(
                            "time_s", record_step + 1 if include_initial_pose else record_step
                        )
                    ),
                    pose=pose,
                    action={
                        "selected_waypoint": record.get("selected_waypoint"),
                        "velocity_command": record.get("velocity_command"),
                        "executed_delta": record.get("executed_delta"),
                    },
                    measures=dict(record.get("measures") or {}),
                    # The rendered-observation contract (rgb/depth dtypes and shapes, K) is
                    # carried through verbatim when the backend reported one. It is the evidence
                    # that a step actually rendered, so it must survive into the persisted trace
                    # rather than being summarised away.
                    observation={
                        "frame_path": record.get("frame_path", ""),
                        **(record.get("observation") or {}),
                    },
                    metadata={
                        **(record.get("metadata") or {}),
                        "raw_output": record.get("raw_output"),
                        "waypoint_cluster_id": record.get("waypoint_cluster_id"),
                        "apos_id": record.get("apos_id"),
                        "opos_id": record.get("opos_id"),
                        "apos_xy": record.get("apos_xy"),
                        "opos_xy": record.get("opos_xy"),
                        "apos_kind": record.get("apos_kind"),
                        "opos_kind": record.get("opos_kind"),
                        "policy_inference_time_ms": record.get("policy_inference_time_ms"),
                        "termination": record.get("termination"),
                        "subtask_status": record.get("subtask_status"),
                        "events": record.get("events", []),
                    },
                )
            )
        return cls(
            episode_id=episode_id,
            instruction=instruction,
            steps=steps,
            final_measures=dict(final_measures or {}),
            termination_reason=termination_reason,
            stop_step=int(stop_step),
            metadata=dict(metadata or {}),
        )


def _camera_walk_pose_to_step_trace(
    *,
    step: int,
    time_s: float,
    pose: dict[str, Any],
    action: dict[str, Any] | None = None,
    measures: dict[str, Any] | None = None,
    observation: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> StepTrace:
    position = (
        float(pose.get("x", 0.0)),
        float(pose.get("y", 0.0)),
        float(pose.get("z", 0.0)),
    )
    yaw = float(pose.get("yaw", 0.0))
    half = yaw * 0.5
    return StepTrace(
        step=step,
        time_s=time_s,
        position=position,
        orientation_wxyz=(math.cos(half), 0.0, 0.0, math.sin(half)),
        action=action or {},
        measures=measures or {},
        observation=observation or {},
        metadata=metadata or {},
    )
