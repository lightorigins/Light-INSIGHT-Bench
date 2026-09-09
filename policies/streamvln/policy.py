"""StreamVLN (ICRA'26, InternRobotics) -- port of ``streamvln/streamvln_eval.py``.

LLaVA-Video Qwen2-7B that emits a chunk of four actions per inference from
{forward 25 cm, left 15 deg, right 15 deg, STOP}, truncated at the first STOP.

Loop structure transcribed from the habitat benchmark eval:

* every step appends the REAL current frame, so ``queue_feed_frames: True``;
* inference runs only when the parsed chunk is exhausted;
* the KV cache and conversation window reset every ``num_frames`` (32) steps,
  and the next inference re-injects a strided history through the <memory>
  token -- that boundary is what keeps a 300-step episode inside the context;
* an empty parse means STOP, upstream's own reading;
* the frame is centre-cropped to the 79 deg training FOV, which is the setting
  the published INSIGHT-Bench row was produced with.

Inputs follow the authors' RGB-only realworld server: depth zeroed, pose
identity, fixed intrinsics. The model ignores them -- it does no voxel pooling
-- and feeding a fabricated depth map would be worse than feeding none.

Validated against upstream InternRobotics/StreamVLN @ e48f6ff. Needs the
upstream repo on PYTHONPATH (its ``model``, ``streamvln`` and ``utils``
packages are imported directly).
"""

from __future__ import annotations

import argparse
import copy
from typing import Any, ClassVar

from insight_policy import Policy, Step

# 0 STOP / 1 forward / 2 left / 3 right, upstream's action ids.
ACTION_TO_PRIMITIVE = {1: "forward", 2: "left", 3: "right"}

# Intrinsics from the authors' RGB-only realworld server. Carried verbatim so
# the tensor shapes match what the checkpoint was served with; unused by the
# model itself.
REALWORLD_RGB_HEIGHT = 1.25
REALWORLD_INTRINSIC = (
    (192.0, 0.0, 191.42857143, 0.0),
    (0.0, 192.0, 191.42857143, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


class StreamVlnPolicy(Policy):
    # Hugging Face; R2R val-unseen SR 56.4 as published.
    model_id: ClassVar[str] = "mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3"

    defaults: ClassVar[dict[str, Any]] = {
        "forward_m": 0.25,
        "turn_deg": 15.0,
        "queue_feed_frames": True,
        "on_empty": "forward",
        # Training HFOV is 79 deg at 4:3 against the rig's 120, and the published
        # INSIGHT-Bench StreamVLN row is the CROPPED variant: an A/B on the eval
        # split put it at SR 10% against 8% raw. Pass
        # --opt center_crop_hfov_deg=none to score the model on the raw frame.
        "source_hfov_deg": 120.0,
        "center_crop_hfov_deg": 79.0,
        "crop_aspect": 1.3333333,
        "resize": None,
        # Upstream defaults: window length, strided history depth, chunk size.
        "num_frames": 32,
        "num_history": 8,
        "num_future_steps": 4,
        "attn_implementation": "flash_attention_2",
    }

    def __init__(self, model_path: str, **options: Any):
        super().__init__(model_path, **options)
        self.num_frames = self.opt_int("num_frames")
        self.num_history = self.opt_int("num_history")
        self.num_future_steps = self.opt_int("num_future_steps")
        self._instruction = ""
        self._inferences = 0
        self._clear_episode()
        self._evaluator: Any = None
        self._model: Any = None
        self._load()

    def _clear_episode(self) -> None:
        self._rgb_list: list[Any] = []
        self._depth_list: list[Any] = []
        self._pose_list: list[Any] = []
        self._intrinsic_list: list[Any] = []
        self._time_ids: list[int] = []
        self._step_id = 0
        # Step id of the frame just appended (upstream's step_id semantics).
        self._cur_step = 0
        self._output_ids = None
        self._past_key_values = None

    def _load(self) -> None:
        import numpy as np
        import torch
        import transformers
        from model.stream_video_vln import StreamVLNForCausalLM
        from streamvln.streamvln_agent import VLNEvaluator

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_path, model_max_length=4096, padding_side="right"
        )
        config = transformers.AutoConfig.from_pretrained(self.model_path)
        model = StreamVLNForCausalLM.from_pretrained(
            self.model_path,
            attn_implementation=self.opt_str("attn_implementation"),
            torch_dtype=torch.bfloat16,
            config=config,
            low_cpu_mem_usage=False,
        )
        model.model.num_history = self.num_history
        model.reset(1)
        model.requires_grad_(False)
        model.to("cuda:0")
        model.eval()

        args = argparse.Namespace(
            num_future_steps=self.num_future_steps,
            num_frames=self.num_frames,
            num_history=self.num_history,
            model_max_length=4096,
            device="cuda:0",
        )
        sensor_config = {
            "rgb_height": REALWORLD_RGB_HEIGHT,
            "camera_intrinsic": np.array(REALWORLD_INTRINSIC),
        }
        self._evaluator = VLNEvaluator(sensor_config, model=model, tokenizer=tokenizer, args=args)
        self._model = model

    def reset(self, instruction: str) -> None:
        self._instruction = instruction
        self._clear_episode()
        self._model.reset_for_env(0)

    def observe(self, rgb: Any) -> None:
        self._append_frame(rgb)

    def act(self, rgb: Any) -> Step:
        import torch
        from utils.utils import DEFAULT_MEMORY_TOKEN, DEFAULT_VIDEO_TOKEN, dict_to_cuda

        evaluator = self._evaluator
        self._append_frame(rgb)

        # The first inference of a window carries the full prompt (plus the
        # <memory> token once past step 0); continuations send an empty human
        # turn and reuse the KV cache.
        if self._output_ids is None:
            sources = copy.deepcopy(evaluator.conversation)
            sources[0]["value"] = sources[0]["value"].replace(
                " Where should you go next to stay on track?",
                " Please devise an action sequence to follow the instruction which may include "
                "turning left or right by a certain degree, moving forward by a certain distance "
                "or stopping once the task is complete.",
            )
            if self._cur_step != 0:
                sources[0]["value"] += (
                    f" These are your historical observations {DEFAULT_MEMORY_TOKEN}."
                )
            sources[0]["value"] = sources[0]["value"].replace(DEFAULT_VIDEO_TOKEN + "\n", "")
            sources[0]["value"] = sources[0]["value"].replace("<instruction>.", self._instruction)
            add_system = True
        else:
            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
            add_system = False

        input_ids, _ = evaluator.preprocess_qwen(
            [sources], evaluator.tokenizer, True, add_system=add_system
        )
        if self._output_ids is not None:
            input_ids = torch.cat([self._output_ids, input_ids.to(self._output_ids.device)], dim=1)

        images = self._rgb_list[-1:]
        depths = self._depth_list[-1:]
        poses = self._pose_list[-1:]
        intrinsics = self._intrinsic_list[-1:]
        if self._cur_step != 0 and self._cur_step % self.num_frames == 0:
            history_ids = slice(0, self._time_ids[0], (self._time_ids[0] // self.num_history))
            images = self._rgb_list[history_ids] + images
            depths = self._depth_list[history_ids] + depths
            poses = self._pose_list[history_ids] + poses
            intrinsics = self._intrinsic_list[history_ids] + intrinsics

        input_dict = {
            "images": torch.stack(images).unsqueeze(0),
            "depths": torch.stack(depths).unsqueeze(0),
            "poses": torch.stack(poses).unsqueeze(0),
            "intrinsics": torch.stack(intrinsics).unsqueeze(0),
            "inputs": input_ids,
            "env_id": 0,
            "time_ids": [self._time_ids],
            "task_type": [0],
        }
        input_dict = dict_to_cuda(input_dict, evaluator.device)
        for key in ("images", "depths", "poses", "intrinsics"):
            input_dict[key] = input_dict[key].to(torch.bfloat16)

        outputs = self._model.generate(
            **input_dict,
            do_sample=False,
            num_beams=1,
            max_new_tokens=10000,
            use_cache=True,
            return_dict_in_generate=True,
            past_key_values=self._past_key_values,
        )
        self._output_ids = outputs.sequences
        self._past_key_values = outputs.past_key_values
        llm_outputs = evaluator.tokenizer.batch_decode(self._output_ids, skip_special_tokens=False)[
            0
        ].strip()
        action_seq = evaluator.parse_actions(llm_outputs)
        self._inferences += 1

        # An empty parse is upstream's STOP.
        stop = len(action_seq) == 0
        names: list[str] = []
        for action in action_seq:
            if action == 0:
                stop = True
                break
            if action in ACTION_TO_PRIMITIVE:
                names.append(ACTION_TO_PRIMITIVE[action])
        return Step.primitives(names, then_stop=stop, raw_output=llm_outputs)

    def finish(self) -> None:
        self._clear_episode()

    def describe(self) -> dict[str, Any]:
        # No counters: /health is read before the run and would report zero.
        return {"family": "streamvln"}

    def _append_frame(self, rgb: Any) -> None:
        """The real frame is appended every step; the window resets every 32."""
        import numpy as np
        import torch
        from PIL import Image

        evaluator = self._evaluator
        # The boundary reset lands after step num_frames-1 has been executed,
        # i.e. before this append rather than after it.
        if self._step_id != 0 and self._step_id % self.num_frames == 0 and self._time_ids:
            self._model.reset_for_env(0)
            self._output_ids = None
            self._past_key_values = None
            self._time_ids = []
        self._cur_step = self._step_id
        self._time_ids.append(self._step_id)

        image = Image.fromarray(rgb).convert("RGB")
        image = evaluator.image_processor.preprocess(images=image, return_tensors="pt")[
            "pixel_values"
        ][0]
        depth = np.zeros((rgb.shape[0], rgb.shape[1], 1))
        self._rgb_list.append(image)
        self._depth_list.append(torch.from_numpy(depth).float())
        self._pose_list.append(torch.from_numpy(np.eye(4)))
        self._intrinsic_list.append(torch.from_numpy(evaluator.intrinsic_matrix).float())
        self._step_id += 1
