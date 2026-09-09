"""Shared inference plumbing for the LLaMA-VID family (NaVid and Uni-NaVid).

Both upstream repos ship the same module layout under different package names
(``navid`` / ``uninavid``) and assemble tokens identically, so the two policies
pass their imported namespace in through :class:`LlamaVidBundle` and share one
copy of the generate call below.

``predict_inference`` transcribes upstream ``predict_inference`` from
NaVid-VLN-CE ``agent_navid.py`` (and the identical ``agent_uninavid.py``): a
vicuna_v1 conversation whose ``<image>`` slot is expanded into the special-token
sandwich ``<video_special><image_sep>[img]</video_special>``
``<image_special></image_special>[Navigation]``, ``model.update_prompt`` before
generate, sampled decoding.

Imports torch lazily so this module -- and every policy that imports it -- can
be read, linted and unit-tested without a GPU environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LlamaVidBundle:
    """One loaded LLaMA-VID model plus the upstream helpers that go with it."""

    tokenizer: Any
    model: Any
    image_processor: Any
    conv_templates: Any
    SeparatorStyle: Any  # upstream attribute name
    tokenizer_image_token: Any
    KeywordsStoppingCriteria: Any  # upstream attribute name
    IMAGE_TOKEN_INDEX: int  # upstream attribute name
    DEFAULT_IMAGE_TOKEN: str  # upstream attribute name
    DEFAULT_IM_START_TOKEN: str  # upstream attribute name
    DEFAULT_IM_END_TOKEN: str  # upstream attribute name


def ensure_model_zoo_link(upstream_repo: str, vit_path: str) -> None:
    """Link the ViT where the checkpoint expects it.

    The published config.json references a CWD-relative ``./model_zoo/eva_vit_g.pth``,
    so the repo checkout needs that name pointing at wherever the weights live.
    """
    model_zoo = Path(upstream_repo) / "model_zoo"
    model_zoo.mkdir(parents=True, exist_ok=True)
    link = model_zoo / "eva_vit_g.pth"
    if not link.exists():
        link.symlink_to(Path(vit_path))


def predict_inference(
    bundle: LlamaVidBundle,
    prompt: str,
    images: list,
    *,
    temperature: float = 0.2,
) -> str:
    """One LLM call. ``images`` is the tensor list upstream passes to generate."""
    import torch

    tokenizer = bundle.tokenizer
    question = prompt.replace(bundle.DEFAULT_IMAGE_TOKEN, "").replace("\n", "")
    qs = prompt

    def _tok(text: str):
        return tokenizer(text, return_tensors="pt").input_ids[0][1:].cuda()

    image_start_special_token = _tok("<image_special>")
    image_end_special_token = _tok("</image_special>")
    video_start_special_token = _tok("<video_special>")
    video_end_special_token = _tok("</video_special>")
    navigation_special_token = _tok("[Navigation]")
    image_seperator = _tok("<image_sep>")

    if bundle.model.config.mm_use_im_start_end:
        qs = (
            bundle.DEFAULT_IM_START_TOKEN
            + bundle.DEFAULT_IMAGE_TOKEN
            + bundle.DEFAULT_IM_END_TOKEN
            + "\n"
            + qs.replace("<image>", "")
        )
    else:
        qs = bundle.DEFAULT_IMAGE_TOKEN + "\n" + qs.replace("<image>", "")

    conv = bundle.conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    token_prompt = bundle.tokenizer_image_token(
        full_prompt, tokenizer, bundle.IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).cuda()
    indices_to_replace = torch.where(token_prompt == -200)[0]
    new_list = []
    while indices_to_replace.numel() > 0:
        idx = indices_to_replace[0]
        new_list.append(token_prompt[:idx])
        new_list.append(video_start_special_token)
        new_list.append(image_seperator)
        new_list.append(token_prompt[idx : idx + 1])
        new_list.append(video_end_special_token)
        new_list.append(image_start_special_token)
        new_list.append(image_end_special_token)
        new_list.append(navigation_special_token)
        token_prompt = token_prompt[idx + 1 :]
        indices_to_replace = torch.where(token_prompt == -200)[0]
    if token_prompt.numel() > 0:
        new_list.append(token_prompt)
    input_ids = torch.cat(new_list, dim=0).unsqueeze(0)

    stop_str = conv.sep if conv.sep_style != bundle.SeparatorStyle.TWO else conv.sep2
    stopping_criteria = bundle.KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    with torch.inference_mode():
        bundle.model.update_prompt([[question]])
        output_ids = bundle.model.generate(
            input_ids,
            images=images,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    input_token_len = input_ids.shape[1]
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    return outputs.strip()
