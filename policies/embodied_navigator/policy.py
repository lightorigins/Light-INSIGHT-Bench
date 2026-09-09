"""TAMP-Nav / Embodied-Navigator (ZJU-OmniAI, internally DThinkVLN).

Both names belong to one model, so both are written down here: the paper is
"Embodied-Navigator: Point, Think, Memorize, and Align for Efficient Navigation"
(arXiv 2608.17512), while the LightNav-0 paper and the INSIGHT-Bench leaderboard
row call it TAMP-Nav.

This model does not emit actions. It emits a VIEW and a PIXEL -- "walk to this
point in this image" -- and upstream's own FastAPI service owns the prompt
assembly, the centre-crop-to-280 preprocessing and the mapping of the predicted
pixel back to original-image coordinates. Reimplementing that would be guessing
at a tested code path, so this policy runs that service as a child process on
localhost and keeps it verbatim. What this file owns is the conversion between
their contract and ours:

* our single forward view is sent as ``rgb_front`` only. Their code tolerates
  any subset of the four cameras; sending black or duplicated side views would
  feed the model imagery that does not exist;
* their service requires a per-step front pose and hard-fails without one, so
  the pose is dead-reckoned by integrating the motions we emit, from an episode
  origin of (0, 0, 0);
* pixel -> waypoint by intersecting the ray with the ground plane at our own
  intrinsics (camera height 1.0 m), clamped to [0.3, 3.0] m; a pixel at or above
  the horizon is a vertical target (a door, a piece of furniture) and gets the
  horizon fallback distance instead of an infinite one;
* a hallucinated ``choice`` of a camera we never sent is read as "the target is
  in that direction": turn 90/180 deg and re-query on the next frame;
* an upstream parse failure arrives as a CMD action and means STOP.

One model call produces a whole waypoint, which the server then drains as
several capped motions (~9 calls per episode, matching upstream's cadence), so
``queue_feed_frames`` is False: the model is not meant to see the frames in
between.

Validated against upstream Embodied-Navigator @ 7adaed7. Needs ``--opt
upstream_repo=`` (the checkout whose ``src.server.pixel_agent_api`` is served).
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, ClassVar

from insight_policy import Policy, Step, Waypoint

# Fixed on purpose: upstream's session dict never garbage-collects, so reusing
# one id replaces the state instead of growing a new entry per episode.
SESSION_ID = "insight-bench-episode"

SIDE_VIEW_TURN_DEG = {"left": 90.0, "right": -90.0, "back": 180.0}


def _post_json(url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    """POST JSON with the standard library, so this policy adds no dependency."""
    data = json.dumps(payload).encode("utf-8")
    # Fixed localhost URL built from --opt child_port; never user-supplied.
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class EmbodiedNavigatorPolicy(Policy):
    model_id: ClassVar[str] = "UnderTides/Embodied-Navigator-7B-GRPO"  # Hugging Face

    defaults: ClassVar[dict[str, Any]] = {
        "forward_m": 0.25,
        "turn_deg": 15.0,
        "queue_feed_frames": False,
        "on_empty": "forward",
        # Their service centre-crops to about 88.5 deg itself, so cropping again
        # on this side would compound two crops. Left off deliberately.
        "source_hfov_deg": 120.0,
        "center_crop_hfov_deg": None,
        "crop_aspect": None,
        "resize": None,
        "upstream_repo": "",
        # Ground-plane geometry.
        "camera_height_m": 1.0,
        "min_waypoint_m": 0.3,
        "max_waypoint_m": 3.0,
        "horizon_fallback_m": 2.0,
        # Child service.
        "child_port": 18191,
        # How long the child gets to exit on its own before it is killed.
        "child_stop_timeout_s": 10.0,
        "child_log": "",
        "child_startup_timeout_s": 1500.0,
        "step_timeout_s": 280.0,
        # Official eval decoding values.
        "temperature": 0.4,
        "top_p": 0.6,
        "max_new_tokens": 512,
    }

    def __init__(self, model_path: str, **options: Any):
        super().__init__(model_path, **options)
        self.camera_height_m = self.opt_float("camera_height_m")
        self.hfov_deg = self.opt_float("source_hfov_deg")
        self.min_wp_m = self.opt_float("min_waypoint_m")
        self.max_wp_m = self.opt_float("max_waypoint_m")
        self.horizon_fallback_m = self.opt_float("horizon_fallback_m")
        self.child_port = self.opt_int("child_port")
        self.child_stop_timeout_s = self.opt_float("child_stop_timeout_s")
        self.step_timeout_s = self.opt_float("step_timeout_s")
        # Dead-reckoned front pose: x, y, theta (radians, positive left).
        self._pose = [0.0, 0.0, 0.0]
        self._inferences = 0
        self._proc: subprocess.Popen | None = None
        self._log: Any = None
        self._base = f"http://127.0.0.1:{self.child_port}"
        self._start_child(self.opt_path("upstream_repo"))

    def _start_child(self, upstream_repo: str) -> None:
        env = dict(os.environ)
        env.update(
            # Default 1 loads a second huge depth model at startup that this
            # RGB-only path never uses.
            DTHINK_PRELOAD_DA3="0",
            DTHINK_MODEL_PATH=self.model_path,
            DTHINK_MAX_NEW_TOKENS=str(self.opt_int("max_new_tokens")),
            DTHINK_TEMPERATURE=str(self.opt_float("temperature")),
            DTHINK_TOP_P=str(self.opt_float("top_p")),
        )
        child_log = self.opt_str("child_log")
        if child_log:
            self._log = Path(child_log).expanduser().open("a")
            stdout: Any = self._log
        else:
            stdout = sys.stderr
        # Fixed argv, no shell: upstream's own service, in this same venv.
        self._proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "src.server.pixel_agent_api:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.child_port),
            ],
            cwd=upstream_repo,
            env=env,
            stdout=stdout,
            stderr=subprocess.STDOUT,
        )
        self._await_child()

    def _await_child(self) -> None:
        deadline = time.monotonic() + self.opt_float("child_startup_timeout_s")
        while True:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"embodied-navigator child service exited (rc={self._proc.returncode}); "
                    "check --opt child_log"
                )
            try:
                with urllib.request.urlopen(f"{self._base}/docs", timeout=3):
                    return
            except (urllib.error.URLError, OSError):
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        "embodied-navigator child service did not come up in "
                        f"{self.opt_float('child_startup_timeout_s')}s"
                    ) from None
                time.sleep(5)

    def reset(self, instruction: str) -> None:
        self._pose = [0.0, 0.0, 0.0]
        _post_json(
            f"{self._base}/reset",
            {"session_id": SESSION_ID, "instruction": instruction, "episode_id": SESSION_ID},
            timeout=60.0,
        )

    def act(self, rgb: Any) -> Step:
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(rgb).convert("RGB").save(buf, "JPEG", quality=90)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")

        response = _post_json(
            f"{self._base}/step",
            {
                "session_id": SESSION_ID,
                "obs": {"rgb_front": encoded},
                "sensor_poses": {
                    "rgb_front": {
                        "x": self._pose[0],
                        "y": self._pose[1],
                        "theta": self._pose[2],
                    }
                },
                "return_model_text": True,
            },
            timeout=self.step_timeout_s,
        )
        self._inferences += 1
        env_action = response.get("env_action") or {}
        raw = str(response.get("model_text") or env_action)[-400:]

        if isinstance(env_action, dict) and env_action.get("type") == "CMD":
            # Covers upstream's own parse failures, which it reports as STOP.
            return Step.stop(raw_output=raw)

        action = env_action.get("action") if isinstance(env_action, dict) else None
        sensor = str((env_action or {}).get("sensor", "rgb_front"))
        motions: list[Waypoint] = []
        if "front" not in sensor:
            for key, degrees in SIDE_VIEW_TURN_DEG.items():
                if key in sensor:
                    motions = [Waypoint(yaw_rad=math.radians(degrees))]
                    break
        # (list, tuple) rather than list | tuple: this file parses under py3.8.
        elif isinstance(action, (list, tuple)) and len(action) == 2:  # noqa: UP038
            height, width = rgb.shape[:2]
            azimuth, distance = self._pixel_to_polar(
                float(action[0]), float(action[1]), width, height
            )
            # Turning and advancing are ONE waypoint, not two in a row.
            # A waypoint list is a plan, and the runner executes only its first
            # entry before asking for a new one, so a turn queued ahead of the
            # advance is an advance that never happens: measured over 609
            # consecutive inferences the agent changed heading every step and
            # moved a path length of exactly 0.000 m. The runner caps a step to
            # 0.25 m and 30 degrees, which is what turns this into a walk.
            # Azimuth is right-positive in image coordinates; our yaw is +left.
            motions.append(Waypoint(yaw_rad=-azimuth, forward_m=distance))

        self._advance_pose(motions)
        return Step.waypoints(
            motions,
            raw_output=raw,
            extras={"env_action": env_action, "pose": [round(v, 3) for v in self._pose]},
        )

    def finish(self) -> None:
        self._pose = [0.0, 0.0, 0.0]

    def close(self) -> None:
        """Stop the child, so the next run's child can have the port back.

        Without this the child outlived the server and was re-parented to
        init, still holding its port. The run that leaked it looked completely
        healthy -- it was the *next* run whose child died at startup on
        "address already in use", having loaded four checkpoint shards first.
        """
        proc = self._proc
        self._proc = None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=self.child_stop_timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=self.child_stop_timeout_s)

    def healthy(self) -> bool:
        """False once the child stops answering.

        A live pid is not enough. The failure this exists to catch had the
        launcher process still running while the service behind it was gone,
        so ``poll()`` said healthy and every episode then recorded zero motion
        without an error. Ask the child the same question startup asks.
        """
        if self._proc is None or self._proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"{self._base}/docs", timeout=2):
                return True
        except (urllib.error.URLError, OSError):
            return False

    def describe(self) -> dict[str, Any]:
        alive = self._proc is not None and self._proc.poll() is None
        # Configuration and liveness only: the runner reads /health before the
        # run, so a counter here would always be published as zero.
        return {"family": "embodied-navigator", "child_alive": alive}

    # ---- geometry --------------------------------------------------------------------

    def _pixel_to_polar(self, u: float, v: float, width: int, height: int) -> tuple[float, float]:
        """(u, v) in the original frame -> (azimuth rad, right-positive; ground distance m)."""
        fx = width / 2.0 / math.tan(math.radians(self.hfov_deg) / 2.0)
        cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
        azimuth = math.atan((u - cx) / fx)
        depression = math.atan((v - cy) / fx)  # square pixels, so fy == fx
        if depression > 1e-3:
            distance = self.camera_height_m / math.tan(depression)
        else:
            distance = self.horizon_fallback_m
        return azimuth, max(self.min_wp_m, min(self.max_wp_m, distance))

    def _advance_pose(self, motions: list[Waypoint]) -> None:
        """Dead-reckon the pose this adapter reports, the way the runner moves.

        The runner translates along the heading it already had and rotates
        afterwards. Rotating first and stepping in the new heading, which this
        did, drifts from the pose actually simulated -- and this value is only
        worth carrying in a trace if it agrees with the run.
        """
        for motion in motions:
            if motion.forward_m:
                self._pose[0] += motion.forward_m * math.cos(self._pose[2])
                self._pose[1] += motion.forward_m * math.sin(self._pose[2])
            self._pose[2] += motion.yaw_rad
