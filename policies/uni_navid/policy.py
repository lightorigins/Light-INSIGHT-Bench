"""Uni-NaVid (RSS'25) -- port of NaVid-VLN-CE ``agent_uninavid.py``.

Vicuna-7B video VLA on the unified VLN + ObjectNav + EQA + following checkpoint.
It answers with a space-separated action chunk ("forward forward left stop").

Upstream eval semantics preserved:

* every frame counts, queued steps included (``queue_feed_frames: True``);
* each inference feeds only the frames accumulated since the previous one --
  the model's own online feature cache carries the rest -- and then clears the
  pending list;
* the prompt asks for four actions but upstream executes at most two per
  inference "to accelerate", so the chunk is capped at two slots including stop;
* per-episode reset must set ``run_type="eval"`` and reinitialise the online
  inference cache, otherwise the second episode reads the first one's features.

Training HFOV is 120 deg, identical to the benchmark rig, so no crop variant
exists for this model.

Validated against upstream jzhzhang/Uni-NaVid @ 79ef5ea. Needs ``--opt
upstream_repo=`` and ``--opt vit_path=`` (eva_vit_g.pth, shared with NaVid).
"""

from __future__ import annotations

import os
from typing import Any, ClassVar

from _llamavid_common import LlamaVidBundle, ensure_model_zoo_link, predict_inference
from insight_policy import Policy, Step

PROMPT_TEMPLATE = (
    "Imagine you are a robot programmed for navigation tasks. You have been given a video "
    "of historical observations and an image of the current observation <image>. Your "
    "assigned task is: '{}'. Analyze this series of images to determine your next four "
    "actions. The predicted action should be one of the following: forward, left, right, or "
    "stop."
)

TOKEN_TO_PRIMITIVE = {"forward": "forward", "left": "left", "right": "right"}


class UniNavidPolicy(Policy):
    model_id: ClassVar[str] = "Jzzhang/Uni-NaVid"  # Hugging Face

    defaults: ClassVar[dict[str, Any]] = {
        "forward_m": 0.25,
        "turn_deg": 30.0,
        "queue_feed_frames": True,
        "on_empty": "forward",
        # Trained at 120 deg, the same as the rig: nothing to crop.
        "source_hfov_deg": 120.0,
        "center_crop_hfov_deg": None,
        "crop_aspect": None,
        "resize": None,
        "upstream_repo": "",
        "vit_path": "",
        # Upstream executes at most 2 of the 4 predicted slots per inference.
        "max_slots_per_call": 2,
        "temperature": 0.2,
    }

    def __init__(self, model_path: str, **options: Any):
        super().__init__(model_path, **options)
        self.max_slots = self.opt_int("max_slots_per_call")
        self.temperature = self.opt_float("temperature")
        self._instruction = ""
        self._rgb_list: list[Any] = []
        self._inferences = 0
        self._bundle = self._load(self.opt_path("upstream_repo"), self.opt_path("vit_path"))

    def _load(self, upstream_repo: str, vit_path: str) -> LlamaVidBundle:
        import numpy as np
        import torch

        ensure_model_zoo_link(upstream_repo, vit_path)
        # Checkpoint config.json resolves ./model_zoo and ./uninavid/processor
        # relative to the working directory.
        os.chdir(upstream_repo)
        torch.manual_seed(30)
        np.random.seed(30)

        from uninavid.constants import (
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IMAGE_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from uninavid.conversation import SeparatorStyle, conv_templates
        from uninavid.mm_utils import (
            KeywordsStoppingCriteria,
            get_model_name_from_path,
            tokenizer_image_token,
        )
        from uninavid.model.builder import load_pretrained_model

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
        # Per-episode online-inference cache init. Skipping this leaks the
        # previous episode's features into this one.
        self._bundle.model.config.run_type = "eval"
        self._bundle.model.get_model().initialize_online_inference_nav_feat_cache()
        self._bundle.model.get_model().new_frames = 0

    def observe(self, rgb: Any) -> None:
        self._rgb_list.append(rgb)

    def act(self, rgb: Any) -> Step:
        import numpy as np

        bundle = self._bundle
        self._rgb_list.append(rgb)

        # Only the frames since the last inference are fed; the model's online
        # feature cache carries the rest.
        batch = np.asarray(self._rgb_list)
        bundle.model.get_model().new_frames = len(self._rgb_list)
        video = (
            bundle.image_processor.preprocess(batch, return_tensors="pt")["pixel_values"]
            .half()
            .cuda()
        )
        self._rgb_list = []

        output = predict_inference(
            bundle,
            PROMPT_TEMPLATE.format(self._instruction),
            [video],
            temperature=self.temperature,
        )
        self._inferences += 1

        names: list[str] = []
        stop = False
        unknown: list[str] = []
        for token in output.split(" "):
            if token == "stop":
                stop = True
            elif token in TOKEN_TO_PRIMITIVE:
                names.append(TOKEN_TO_PRIMITIVE[token])
            else:
                # Upstream raises "wrong actions!" here; keeping the parsed
                # prefix turns a garbled tail into a shorter chunk rather than
                # a dead episode.
                unknown.append(token)
                break
            if stop or len(names) >= self.max_slots:
                break

        extras = {"unknown_tokens": unknown} if unknown else None
        # A chunk may move first and stop after: the moves still execute, one
        # per /act, and the stop is the reply that follows them.
        return Step.primitives(names, then_stop=stop, raw_output=output, extras=extras)

    def finish(self) -> None:
        self._rgb_list = []

    def describe(self) -> dict[str, Any]:
        # No counters: /health is read before the run and would report zero.
        return {"family": "llama-vid"}
