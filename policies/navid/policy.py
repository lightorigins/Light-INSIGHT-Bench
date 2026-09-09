"""NaVid (RSS'24) -- port of NaVid-VLN-CE ``agent_navid.py`` NaVid_Agent.

Vicuna-7B / LLaMA-VID video VLM. It answers in text with a magnitude ("move
forward 75 cm", "turn left 60 degrees"), which is parsed into repeats of a
25 cm / 30 deg primitive, at most three per call -- the upstream convention.

Upstream eval semantics preserved:

* every frame counts, including the ones rendered while queued actions drain,
  so this policy sets ``queue_feed_frames: True`` and appends them in observe();
* the FULL history video tensor is fed to every inference, preprocessed
  incrementally so only new frames pay the cost;
* an unparsable answer, or one whose magnitude rounds to zero primitives,
  becomes ONE random primitive -- upstream's own recovery, kept because
  dropping it would change the score, not just the plumbing;
* the frame is centre-cropped to the 90 deg training FOV, which is the setting
  the published INSIGHT-Bench row was produced with.

Validated against upstream jzhzhang/NaVid-VLN-CE @ ce2f804 in a Python 3.8 venv.
Needs ``--opt upstream_repo=`` (the checkout, whose ./model_zoo and
./navid/processor the checkpoint config resolves against) and ``--opt
vit_path=`` (eva_vit_g.pth).
"""

from __future__ import annotations

import os
import random
import re
from typing import Any, ClassVar

from _llamavid_common import LlamaVidBundle, ensure_model_zoo_link, predict_inference
from insight_policy import Policy, Step

PROMPT_TEMPLATE = (
    "Imagine you are a robot programmed for navigation tasks. You have been given a video "
    "of historical observations and an image of the current observation <image>. Your "
    "assigned task is: '{}'. Analyze this series of images to decide your next move, which "
    "could involve turning left or right by a specific degree or moving forward a certain "
    "distance."
)


def extract_result(output: str) -> tuple[int | None, float | None]:
    """Upstream ``agent_navid.py`` parser: 0 stop / 1 forward / 2 left / 3 right."""
    if "stop" in output:
        return 0, None
    for action_id, keyword in ((1, "forward"), (2, "left"), (3, "right")):
        if keyword in output:
            match = re.search(r"-?\d+", output)
            if match is None:
                return None, None
            return action_id, float(match.group())
    return None, None


class NavidPolicy(Policy):
    model_id: ClassVar[str] = "Jzzhang/NaVid"  # Hugging Face

    defaults: ClassVar[dict[str, Any]] = {
        "forward_m": 0.25,
        "turn_deg": 30.0,
        "queue_feed_frames": True,
        "on_empty": "forward",
        # Training HFOV is 90 deg at 4:3 against the rig's 120, and the published
        # INSIGHT-Bench NaVid row is the CROPPED variant: an A/B on the eval split
        # left SR level and moved oracle success from 62% to 72%. Pass
        # --opt center_crop_hfov_deg=none to score the model on the raw frame.
        "source_hfov_deg": 120.0,
        "center_crop_hfov_deg": 90.0,
        "crop_aspect": 1.3333333,
        "resize": None,
        # Upstream repo checkout and the ViT the checkpoint config points at.
        "upstream_repo": "",
        "vit_path": "",
        # At most this many primitives per answer, upstream's cap.
        "max_prims_per_call": 3,
        "temperature": 0.2,
    }

    def __init__(self, model_path: str, **options: Any):
        super().__init__(model_path, **options)
        self.forward_m = self.opt_float("forward_m")
        self.turn_deg = self.opt_float("turn_deg")
        self.max_prims = self.opt_int("max_prims_per_call")
        self.temperature = self.opt_float("temperature")
        self._instruction = ""
        self._rgb_list: list[Any] = []
        self._history_rgb_tensor = None
        self._inferences = 0
        self._bundle = self._load(self.opt_path("upstream_repo"), self.opt_path("vit_path"))

    def _load(self, upstream_repo: str, vit_path: str) -> LlamaVidBundle:
        import numpy as np
        import torch

        ensure_model_zoo_link(upstream_repo, vit_path)
        # The checkpoint's config.json uses CWD-relative ./model_zoo and
        # ./navid/processor paths, so the process has to sit in the checkout.
        os.chdir(upstream_repo)
        torch.manual_seed(30)
        np.random.seed(30)

        from navid.constants import (
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IMAGE_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from navid.conversation import SeparatorStyle, conv_templates
        from navid.mm_utils import (
            KeywordsStoppingCriteria,
            get_model_name_from_path,
            tokenizer_image_token,
        )
        from navid.model.builder import load_pretrained_model

        tokenizer, model, image_processor, _ = load_pretrained_model(
            self.model_path, None, get_model_name_from_path(self.model_path)
        )
        return LlamaVidBundle(
            tokenizer=tokenizer,
            model=model,
            image_processor=image_processor,
            conv_templates=conv_templates,
            SeparatorStyle=SeparatorStyle,
            tokenizer_image_token=tokenizer_image_token,
            KeywordsStoppingCriteria=KeywordsStoppingCriteria,
            IMAGE_TOKEN_INDEX=IMAGE_TOKEN_INDEX,
            DEFAULT_IMAGE_TOKEN=DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN=DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN=DEFAULT_IM_END_TOKEN,
        )

    def reset(self, instruction: str) -> None:
        self._instruction = instruction
        self._rgb_list = []
        self._history_rgb_tensor = None

    def observe(self, rgb: Any) -> None:
        self._rgb_list.append(rgb)

    def act(self, rgb: Any) -> Step:
        import numpy as np
        import torch

        bundle = self._bundle
        self._rgb_list.append(rgb)

        # Incremental preprocess; the full history tensor is fed every call.
        start = 0 if self._history_rgb_tensor is None else self._history_rgb_tensor.shape[0]
        batch = np.asarray(self._rgb_list[start:])
        video = (
            bundle.image_processor.preprocess(batch, return_tensors="pt")["pixel_values"]
            .half()
            .cuda()
        )
        if self._history_rgb_tensor is None:
            self._history_rgb_tensor = video
        else:
            self._history_rgb_tensor = torch.cat((self._history_rgb_tensor, video), dim=0)

        output = predict_inference(
            bundle,
            PROMPT_TEMPLATE.format(self._instruction),
            [self._history_rgb_tensor],
            temperature=self.temperature,
        )
        self._inferences += 1

        action_id, magnitude = extract_result(output)
        if action_id == 0:
            return Step.stop(raw_output=output)

        names: list[str] = []
        if magnitude is not None:
            if action_id == 1:
                names = ["forward"] * min(self.max_prims, int(magnitude / 25))
            elif action_id == 2:
                names = ["left"] * min(self.max_prims, int(magnitude / 30))
            elif action_id == 3:
                names = ["right"] * min(self.max_prims, int(magnitude / 30))
        if not names:
            # Upstream recovery for an unparsable answer: one random move.
            names = [random.choice(("forward", "left", "right"))]
            return Step.primitives(names, raw_output=output, extras={"random_fallback": True})
        return Step.primitives(names, raw_output=output)

    def finish(self) -> None:
        self._rgb_list = []
        self._history_rgb_tensor = None

    def describe(self) -> dict[str, Any]:
        # No counters: /health is read before the run and would report zero.
        return {"family": "llama-vid"}
