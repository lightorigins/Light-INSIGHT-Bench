"""InternVLA-N1 (InternNav DualVLN) -- port of ``internvla_n1_agent_realworld``.

An asynchronous dual system: System-2 is the VLM that replans, System-1 is a
trajectory head that refines between replans. Upstream calls ``agent.step``
EVERY frame and lets the agent decide internally which system answers, so this
policy never builds an action queue -- every frame runs inference, and
``queue_feed_frames`` is therefore False (there are no queued frames to feed).

Semantics preserved:

* S2's discrete output is 0 STOP / 1 forward / 2 left / 3 right / 5 look-down;
  a 5 means "re-step this SAME frame with look_down=True", which upstream's own
  HTTP server does, bounded here to two hops so a look-down loop cannot hang
  the episode;
* S1 answers with a trajectory that it replans every frame, so only its FIRST
  discrete primitive is executed -- queueing the rest would execute a plan the
  model has already superseded.

RGB-only operation follows the authors' inference-only notebook: depth is a
constant plane, pose is identity, and intrinsics are computed from OUR camera
(after any crop) rather than borrowed from their rig. The DepthAnything tower
inside the checkpoint is S1's RGB encoder and loads from ``depth_model_path``.

Validated against upstream InternNav @ 7a5c624. Needs the repo on PYTHONPATH
and ``--opt depth_model_path=`` (the DepthAnything v2 metric checkpoint).
"""

from __future__ import annotations

from typing import Any, ClassVar

from insight_policy import Policy, Step

ACTION_TO_PRIMITIVE = {1: "forward", 2: "left", 3: "right"}
LOOK_DOWN_ACTION = 5
MAX_LOOK_DOWN_HOPS = 2


def trajectory_to_discrete_actions(
    trajectory: Any,
    *,
    step_size: float = 0.25,
    turn_angle_deg: float = 15.0,
    lookahead: int = 4,
) -> list[int]:
    """Upstream's pure-pursuit trajectory follower.

    Transcribed from the nested helper inside ``traj_to_actions`` in
    ``internnav/model/utils/vln_utils.py``; it is defined inside that function
    upstream and so cannot be imported.
    """
    import numpy as np

    actions: list[int] = []
    yaw = 0.0
    pos = trajectory[0]
    turn_angle_rad = np.deg2rad(turn_angle_deg)
    traj = trajectory
    goal = trajectory[-1]

    def normalize_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    while np.linalg.norm(pos - goal) > 0.2:
        dists = np.linalg.norm(traj - pos, axis=1)
        nearest_idx = np.argmin(dists)
        target_idx = min(nearest_idx + lookahead, len(traj) - 1)
        target = traj[target_idx]
        target_dir = target - pos
        if np.linalg.norm(target_dir) < 1e-6:
            break
        target_yaw = np.arctan2(target_dir[1], target_dir[0])
        delta_yaw = normalize_angle(target_yaw - yaw)
        n_turns = int(round(delta_yaw / turn_angle_rad))
        if n_turns > 0:
            actions += [2] * n_turns
        elif n_turns < 0:
            actions += [3] * (-n_turns)
        yaw = normalize_angle(yaw + n_turns * turn_angle_rad)
        next_pos = pos + step_size * np.array([np.cos(yaw), np.sin(yaw)])
        if np.linalg.norm(next_pos - goal) > np.linalg.norm(pos - goal):
            break
        actions.append(1)
        pos = next_pos
    return actions


class InternVlaN1Policy(Policy):
    model_id: ClassVar[str] = "InternRobotics/InternVLA-N1-DualVLN"  # ModelScope

    defaults: ClassVar[dict[str, Any]] = {
        "forward_m": 0.25,
        "turn_deg": 15.0,
        # Every frame runs inference; nothing ever sits in the queue.
        "queue_feed_frames": False,
        "on_empty": "forward",
        # Set center_crop_hfov_deg=79.0 crop_aspect=1.3333333 for the
        # matched-FOV variant; the intrinsics below follow the crop.
        "source_hfov_deg": 120.0,
        "center_crop_hfov_deg": None,
        "crop_aspect": None,
        "resize": None,
        "num_history": 8,
        "plan_step_gap": 4,
        "depth_model_path": "",
        "resize_w": 384,
        "resize_h": 384,
        # Constant depth plane, in metres. RGB-only operation: the value only
        # has to be far enough not to read as an obstacle.
        "depth_plane_m": 10.0,
    }

    def __init__(self, model_path: str, **options: Any):
        super().__init__(model_path, **options)
        self.plan_step_gap = self.opt_int("plan_step_gap")
        self.num_history = self.opt_int("num_history")
        self.depth_plane_m = self.opt_float("depth_plane_m")
        self._instruction = ""
        self._inferences = 0
        self._agent: Any = None
        self._load(self.opt_path("depth_model_path"))

    def _load(self, depth_model_path: str) -> None:
        import argparse

        # The depth-encoder tower path is a module constant upstream, relative
        # to the working directory; it has to be patched before the model loads.
        import internnav.model.basemodel.internvla_n1.internvla_n1_arch as arch

        arch.MODEL_PATH_TO = depth_model_path

        from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent

        args = argparse.Namespace(
            device="cuda:0",
            model_path=self.model_path,
            resize_w=self.opt_int("resize_w"),
            resize_h=self.opt_int("resize_h"),
            num_history=self.num_history,
            plan_step_gap=self.plan_step_gap,
        )
        self._agent = InternVLAN1AsyncAgent(args)

    def reset(self, instruction: str) -> None:
        self._instruction = instruction
        self._agent.reset()

    def act(self, rgb: Any) -> Step:
        import numpy as np
        import torch

        agent = self._agent
        height, width = rgb.shape[:2]
        depth = np.full((height, width), self.depth_plane_m, dtype=np.float32)
        pose = np.eye(4)
        intrinsic = self._intrinsic(width, height)

        # The deploy loop runs gradient-free; without no_grad the S1 trajectory
        # head returns grad-requiring tensors that the follower cannot .numpy().
        with torch.no_grad():
            out = agent.step(rgb, depth, pose, self._instruction, intrinsic=intrinsic)
            self._inferences += 1
            hops = 0
            while (
                out.output_action is not None
                and LOOK_DOWN_ACTION in list(out.output_action)
                and hops < MAX_LOOK_DOWN_HOPS
            ):
                out = agent.step(
                    rgb, depth, pose, self._instruction, intrinsic=intrinsic, look_down=True
                )
                hops += 1

        if out.output_action is not None:
            raw = f"action={out.output_action}"
            names: list[str] = []
            stop = False
            for action in list(out.output_action):
                if action == 0:
                    stop = True
                    break
                if action in ACTION_TO_PRIMITIVE:
                    names.append(ACTION_TO_PRIMITIVE[action])
            return Step.primitives(names, then_stop=stop, raw_output=raw)

        if out.output_trajectory is not None:
            actions = trajectory_to_discrete_actions(
                out.output_trajectory,
                step_size=self.opt_float("forward_m"),
                turn_angle_deg=self.opt_float("turn_deg"),
            )
            raw = f"trajectory->{actions[:8]}"
            # S1 replans every frame: execute only the first primitive.
            head = [ACTION_TO_PRIMITIVE[a] for a in actions[:1] if a in ACTION_TO_PRIMITIVE]
            return Step.primitives(head, raw_output=raw)

        # Neither system answered: on_empty decides whether that is recoverable.
        return Step.primitives([], raw_output="no output")

    def finish(self) -> None:
        self._instruction = ""

    def describe(self) -> dict[str, Any]:
        # No counters: /health is read before the run and would report zero.
        return {"family": "internvla-n1", "plan_step_gap": self.plan_step_gap}

    def _intrinsic(self, width: int, height: int) -> Any:
        """Pinhole intrinsics of OUR frames: the crop HFOV if cropping, else the rig's."""
        import math

        import numpy as np

        hfov_deg = self.options.get("center_crop_hfov_deg") or self.options["source_hfov_deg"]
        fx = width / 2.0 / math.tan(math.radians(float(hfov_deg)) / 2.0)
        intrinsic = np.eye(4)
        intrinsic[0, 0] = fx
        intrinsic[1, 1] = fx
        intrinsic[0, 2] = width / 2.0
        intrinsic[1, 2] = height / 2.0
        return intrinsic
