"""Copy this directory, rename it, fill in the three methods.

    cp -r policies/template policies/my_model
    $EDITOR policies/my_model/policy.py
    python -m insight_policy --policy policies/my_model --model-path /weights/my-ckpt

Everything below is a working policy already: it walks forward forever. Run it
first, confirm the benchmark talks to it, then replace the marked bodies with
your model. That order turns "does my model work" into one question at a time.

Rules this file has to keep:

* import nothing but the standard library, numpy, pillow and YOUR model. Never
  import ``insight_bench`` -- the evaluation SDK lives in a different process,
  in a different environment, and speaks to you only over HTTP.
* do the heavy loading in ``__init__``. The server binds its socket only after
  your constructor returns, so a healthy port means a warm model.
* return a :class:`Step` from ``act``. Nothing else is a valid answer.
"""

from __future__ import annotations

from typing import Any, ClassVar

from insight_policy import Policy, Step


class TemplatePolicy(Policy):
    """Rename me. One Policy subclass per policy.py -- the loader expects one."""

    # What a leaderboard entry is attributed to. Use the published identifier of
    # the weights you are serving (a Hugging Face or ModelScope repo id), not a
    # nickname: this string travels into the run's evidence.
    model_id: ClassVar[str] = "TODO/your-model"

    # Every option this policy accepts, with its default. `--opt key=value`
    # can override any of these and nothing else. Inherited keys (forward_m,
    # turn_deg, queue_feed_frames, on_empty, source_hfov_deg,
    # center_crop_hfov_deg, crop_aspect, resize) stay available -- list here
    # only what differs from insight_policy.Policy plus your own knobs.
    defaults: ClassVar[dict[str, Any]] = {
        # How far one "forward" goes and how far one "left"/"right" turns. Match
        # your model's training convention: emitting a 0.25 m step when the model
        # was trained on 0.75 m ones is the single most common porting bug.
        "forward_m": 0.25,
        "turn_deg": 30.0,
        # Set True if your model's history should include the frames rendered
        # while a queued motion drains (see observe() below).
        "queue_feed_frames": False,
        # "forward" (keep going, recoverable) or "stop" (end the episode) when
        # act() returns a Step with nothing in it.
        "on_empty": "forward",
        # The rig is 480x270 at 120 deg horizontal FOV. If your model was trained
        # narrower, ask for a centre crop to its training FOV and the server will
        # do the geometry for you, e.g.
        #   "center_crop_hfov_deg": 79.0, "crop_aspect": 1.3333333
        # and optionally "resize": (height, width).
        "center_crop_hfov_deg": None,
        "crop_aspect": None,
        "resize": None,
    }

    def __init__(self, model_path: str, **options: Any):
        super().__init__(model_path, **options)
        # ---- REPLACE: load your weights here. -------------------------------
        # import torch
        # self.model = MyModel.from_pretrained(model_path).eval().cuda()
        # ---------------------------------------------------------------------
        self._instruction = ""
        self._frames = 0

    def reset(self, instruction: str) -> None:
        """A new episode starts. Drop per-episode state; keep the weights.

        ``instruction`` is the natural-language goal, e.g. "go to the second
        chair on your left". Called once before the first act().
        """
        self._instruction = instruction
        self._frames = 0
        # ---- REPLACE: clear your KV cache / frame history / planner. ---------

    def act(self, rgb: Any) -> Step:
        """One frame in, one Step out.

        ``rgb`` is an HWC uint8 numpy array in RGB order: 480x270 as captured,
        or whatever your crop/resize options asked for. Return exactly one of:

            Step.waypoints([(forward_m, lateral_m, yaw_rad), ...])
                Explicit displacements in the camera's body frame. Positive
                forward walks along the optical axis, positive lateral goes
                left, positive yaw turns left (counter-clockwise).

            Step.primitives(["forward", "left", "right"])
                Named moves, expanded at this policy's forward_m / turn_deg.

            Step.stop()
                "I have arrived." The benchmark scores the pose of this frame,
                and the episode ends. Only success counts, so stopping early
                costs the episode -- but so does never stopping.

        Several motions at once are fine, and the two kinds behave differently
        on purpose. A waypoint chunk is a PLAN: every row goes to the runner in
        one reply, the runner executes the first row that moves, and you are
        asked again on the very next frame -- which is what a trajectory model's
        own evaluation loop does. Named primitives are a QUEUE: the server
        releases one per act(), so a frame is rendered for each, which is what a
        discrete-action model's evaluation does.

        Attach the model's own text with ``raw_output=`` -- it lands in the trace
        and is what you will read when an episode goes wrong.
        """
        self._frames += 1
        # ---- REPLACE: run your model. ---------------------------------------
        # text = self.model.generate(rgb, self._instruction)
        # if "stop" in text:
        #     return Step.stop(raw_output=text)
        # return Step.primitives(["forward"], raw_output=text)
        # ---------------------------------------------------------------------
        return Step.primitives(["forward"], raw_output="template: always forward")

    def finish(self) -> None:
        """The episode ended (stopped, arrived, or ran out of steps)."""
        self._instruction = ""
        # ---- REPLACE: free per-episode buffers. ------------------------------

    # ---- optional --------------------------------------------------------------------

    def observe(self, rgb: Any) -> None:
        """A frame rendered while a queued motion was executing.

        Only called when ``queue_feed_frames`` is True. Append it to your history
        and return -- this is not the place to run inference. Leave this alone if
        your model only sees the frames it acts on.
        """

    def describe(self) -> dict[str, Any]:
        """Extra fields for /health. Useful for spotting a stuck run early."""
        return {"frames_this_episode": self._frames}
