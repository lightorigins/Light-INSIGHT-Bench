"""JanusVLN -- port of upstream ``src/evaluation.py`` JanusVLN_Inference.

A Qwen2.5-VL variant with two implicit memories, both of which have to survive
across steps or the model degrades to single-frame reasoning:

* spatial: the VGGT KV cache, accumulated across the episode and cleared only on
  reset. Only the CURRENT frame enters VGGT each step;
* semantic: the full-episode RGB history, from which each step takes a uniform
  ``np.linspace`` sample of ``num_history`` (8) frames plus the current one.

One discrete action per inference (MOVE_FORWARD 25 cm / TURN_LEFT|TURN_RIGHT
15 deg / STOP), greedy decoding, 24 new tokens. Anything the exact-match table
does not recognise is STOP, upstream's own reading -- so a garbled answer ends
the episode rather than wandering. Because one inference yields one action,
nothing ever queues and ``queue_feed_frames`` is False.

Validated against upstream JanusVLN @ 8cfe312. Needs the upstream repo's ``src``
on PYTHONPATH for the ``qwen_vl`` package. Upstream's ``evaluation.py`` is never
imported: it pulls habitat in at module scope.
"""

from __future__ import annotations

from typing import Any, ClassVar

from insight_policy import Policy, Step

SYSTEM_PROMPT = (
    "You are a visual language navigation model, and your should go to the locations to "
    "complete the given task. Compare the observation and instruction to infer your current "
    "progress, and then select the correct direction from the candidates to go to the target "
    "location and finish the task."
)
CONTEXT_TEMPLATE = (
    "These images are your historical observations and your current observation.\n"
    " Your task is to {task} \n"
    " You should take one of the following actions:\n"
    " MOVE_FORWARD\n TURN_LEFT\n TURN_RIGHT\n STOP."
)

ANSWER_TO_PRIMITIVE = {
    "MOVE_FORWARD": "forward",
    "TURN_LEFT": "left",
    "TURN_RIGHT": "right",
}


class JanusVlnPolicy(Policy):
    # ModelScope: the authors' corrected re-upload of the weights, and the
    # checkpoint the published INSIGHT-Bench row was produced with. The upstream
    # repository declares no licence -- check your own terms before publishing a
    # result under this policy.
    model_id: ClassVar[str] = "misstl/JanusVLN_Extra"

    defaults: ClassVar[dict[str, Any]] = {
        "forward_m": 0.25,
        "turn_deg": 15.0,
        "queue_feed_frames": False,
        "on_empty": "forward",
        # Set center_crop_hfov_deg=79.0 crop_aspect=1.3333333 for the
        # matched-FOV variant.
        "source_hfov_deg": 120.0,
        "center_crop_hfov_deg": None,
        "crop_aspect": None,
        "resize": None,
        "num_history": 8,
        "attn_implementation": "flash_attention_2",
        # Upstream overrides the checkpoint's own preprocessor pixel budget.
        "max_pixels": 1605632,
        "min_pixels": 784,
        "max_new_tokens": 24,
    }

    def __init__(self, model_path: str, **options: Any):
        super().__init__(model_path, **options)
        self.num_history = self.opt_int("num_history")
        self.max_new_tokens = self.opt_int("max_new_tokens")
        self._instruction = ""
        self._rgb_list: list[Any] = []
        self._inferences = 0
        self._model: Any = None
        self._tokenizer: Any = None
        self._processor: Any = None
        self._load()

    def _load(self) -> None:
        import torch
        from qwen_vl.model.modeling_qwen2_5_vl import (
            Qwen2_5_VLForConditionalGenerationForJanusVLN,
        )
        from transformers import AutoConfig, AutoProcessor, AutoTokenizer

        config = AutoConfig.from_pretrained(self.model_path)
        self._model = Qwen2_5_VLForConditionalGenerationForJanusVLN.from_pretrained(
            self.model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            attn_implementation=self.opt_str("attn_implementation"),
            mode="evaluation",  # keeps the VGGT KV cache alive across steps
        ).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, padding_side="left")
        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            max_pixels=self.opt_int("max_pixels"),
            min_pixels=self.opt_int("min_pixels"),
            padding_side="left",
        )

    def reset(self, instruction: str) -> None:
        self._instruction = instruction
        self._rgb_list = []
        self._model.model.past_key_values_vggt = None

    def act(self, rgb: Any) -> Step:
        import numpy as np
        import torch
        from PIL import Image
        from qwen_vl.model.vggt.utils.load_fn import load_and_preprocess_images

        model, tokenizer, processor = self._model, self._tokenizer, self._processor
        self._rgb_list.append(Image.fromarray(rgb).convert("RGB"))

        # History sample plus the current frame, at most num_history + 1 images.
        history_len = len(self._rgb_list) - 1
        if history_len <= self.num_history:
            images = self._rgb_list[:history_len] + [self._rgb_list[-1]]
        else:
            indices = np.linspace(0, history_len, self.num_history + 1, dtype=int)
            images = [self._rgb_list[i] for i in indices]

        message = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [{"type": "image", "image": img} for img in images]
                + [{"type": "text", "text": CONTEXT_TEMPLATE.format(task=self._instruction)}],
            },
        ]
        text = processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True)

        patch_size = processor.image_processor.patch_size
        merge_size = processor.image_processor.merge_size
        image_inputs = []
        cur_images_vggt = []
        for i, img in enumerate(images):
            tensor = load_and_preprocess_images([img])[0]
            if i == len(images) - 1:
                cur_images_vggt.append(tensor)
            _, height, width = tensor.shape
            if (width // patch_size) % merge_size > 0:
                width = width - (width // patch_size) % merge_size * patch_size
            if (height // patch_size) % merge_size > 0:
                height = height - (height // patch_size) % merge_size * patch_size
            image_inputs.append(tensor[:, :height, :width])

        inputs = processor(
            text=text,
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
            do_rescale=False,  # load_and_preprocess_images already rescaled
        )
        device = model.device
        inputs["images_vggt"] = [torch.stack(cur_images_vggt).to(device)]
        inputs = inputs.to(device)

        with torch.inference_mode():
            cont = model.generate(
                **inputs,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
                temperature=0,
                top_p=None,
                num_beams=1,
                max_new_tokens=self.max_new_tokens,
            )
        # zip without strict=: this file parses under py3.8, where it does not exist.
        trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, cont)]  # noqa: B905
        answer = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        self._inferences += 1

        primitive = ANSWER_TO_PRIMITIVE.get(answer.strip())
        if primitive is None:
            return Step.stop(raw_output=answer)
        return Step.primitives([primitive], raw_output=answer)

    def finish(self) -> None:
        self._rgb_list = []
        if self._model is not None:
            self._model.model.past_key_values_vggt = None

    def describe(self) -> dict[str, Any]:
        # Configuration only. The runner reads /health once, before the run, so
        # a counter here would be published as zero however many frames follow.
        # Per-step inference evidence lives in the trace instead.
        return {"family": "janusvln"}
