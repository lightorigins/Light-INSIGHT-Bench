"""LightNav-0 on INSIGHT-Bench.

Runs inside LightNav-0's own virtualenv (Python 3.11, vLLM), which is where
``lightnav`` is importable:

    git clone https://github.com/lightorigins/LightNav-0 && cd LightNav-0
    python3.11 -m venv .venv && . .venv/bin/activate
    pip install -e ".[vllm]"
    hf download LightOriginsHQ/LightNav-0 --local-dir checkpoints/LightNav-0

Then, from this repository:

    MODEL_PATH=/path/to/checkpoints/LightNav-0 \
    MODEL_PYTHON=/path/to/LightNav-0/.venv/bin/python \
    bash scripts/eval_lightnav0.sh

This wrapper reproduces the loop LightNav-0's own Habitat evaluation runs, so
the numbers mean the same thing:

* one ``observe`` then one inference per step -- the frame is appended to the
  history *before* the inference that consumes it;
* the navigation prompt (``task_type="vlnce_traj"``), with the instruction
  passed through verbatim;
* three RVQ action tokens decoded to a 10-step SE(2) waypoint chunk, handed over
  WHOLE as one plan. The benchmark executes the first row with a forward or yaw
  component and then asks again on the next frame, which is the closed-loop cadence
  the model was evaluated at; the rows are cumulative poses, so replaying them one
  per step would apply i steps of motion as a single step;
* the model's own stop convention: the RVQ stop codeword, or a chunk whose whole
  decode sits inside :data:`RVQ_STOP_ATOL`;
* an unparseable output ends the episode;
* an explicit stop as the last action of the step budget.

The checkpoint carries its own decoder in ``eval_config.json``, so nothing here
needs a vocabulary or bundle path.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from insight_policy import Policy, Step

#: What the benchmark rig gives every model: 300 actions per episode.
DEFAULT_STEP_BUDGET = 300

#: A decoded chunk whose every component sits within this is the RVQ decoder's
#: SECOND stop condition: not the explicit stop codeword, but a residual codeword
#: sum that rounds to standing still. Without it the benchmark's row selection
#: chases a sub-millimetre micro-step instead of stopping, and the episode walks
#: on past the goal. This is the checkpoint's own tolerance, not a benchmark
#: constant -- it must stay in step with the LightNav-0 action tokenizer's value.
RVQ_STOP_ATOL = 5e-3


class LightNav0Policy(Policy):
    model_id = "LightOriginsHQ/LightNav-0"

    defaults = {
        # This is an admission check, not a budget. LightNav-0 pins its KV cache
        # at 2 GiB explicitly, so vLLM logs "does not respect the
        # gpu_memory_utilization config" and takes about 11 GB whatever this
        # says -- 9.1 GB of bf16 weights plus that cache. What the number does
        # do is refuse to start unless that fraction of the card is free, and
        # Isaac Sim wants another ~13 GB beside it. 0.30 of a 46 GB card is
        # ~14 GB: comfortably above what the model uses, low enough to start on
        # a card somebody else is already on.
        "gpu_memory_utilization": 0.30,
        "max_new_tokens": 64,
        "max_num_seqs": 1,
        # "stretch" is what the checkpoint was trained with: every frame is
        # resized to the model's 256x448 regardless of the source aspect.
        "aspect_mode": "stretch",
        # None keeps the checkpoint's own window; SlowFast keeps the whole
        # episode anyway, so this rarely matters.
        "num_history_frames": None,
        # Where the RVQ action tokenizer lives. None resolves it the way the
        # checkpoint says to, which is right for a released checkpoint: it ships
        # the bundle beside the weights and eval_config.json points at it
        # relatively. A checkpoint straight off a training run points at an
        # absolute path on the machine that trained it, and that path is not
        # here -- so name the bundle directory instead.
        "action_tokenizer_bundle": None,
        # Emit a stop as action number `step_budget` without running inference,
        # exactly as LightNav-0's Habitat client does at --max_steps.
        "step_budget": DEFAULT_STEP_BUDGET,
        # The benchmark rig already matches the training field of view, so no
        # crop: leaving these at the base defaults documents that on /health.
    }

    def __init__(self, model_path: str, **options: Any) -> None:
        super().__init__(model_path, **options)
        from lightnav.inference import InferenceConfig, build_engine
        from lightnav.tracking import TrackingAgent, resolve_action_decoder_from_config
        from lightnav.traj_vocab import load_rvq_bundle

        config = InferenceConfig(
            model_path=self.model_path,
            backend="vllm_local",
            max_new_tokens=self.opt_int("max_new_tokens"),
            gpu_memory_utilization=self.opt_float("gpu_memory_utilization"),
            max_num_seqs=self.opt_int("max_num_seqs"),
            aspect_mode=self.opt_str("aspect_mode"),
            num_history_frames=self.options["num_history_frames"],
        )
        # "vlnce" is the navigation task entry in the checkpoint's eval_config;
        # "trackvla" would select the tracking prompt and its own decoder.
        engine, bundle = build_engine(config, task_type="vlnce")
        bundle_dir, horizon = self._resolve_action_tokenizer(resolve_action_decoder_from_config)
        rvq = load_rvq_bundle(bundle_dir, horizon)
        self.agent = TrackingAgent(
            engine,
            num_history_frames=int(bundle.num_history_frames),
            rvq_bundle=rvq,
        )
        self.horizon = int(self.agent.H)
        self.instruction = ""
        self.step = 0

    def _resolve_action_tokenizer(self, resolve_from_config: Any) -> tuple[str, int]:
        """Find the RVQ bundle, and how many waypoints it decodes to.

        ``--opt action_tokenizer_bundle=<dir>`` wins when it is given, because
        the reason to give it is that the checkpoint's own answer is wrong for
        this machine. Otherwise the checkpoint decides, which is what a released
        checkpoint is set up for.
        """
        override = self.options["action_tokenizer_bundle"]
        if override:
            bundle_dir = self.opt_path("action_tokenizer_bundle")
            return bundle_dir, _bundle_horizon(bundle_dir)

        decoder = resolve_from_config(self.model_path, "vlnce")
        if decoder is None or decoder.get("method") != "rvq":
            raise ValueError(
                f"no RVQ action tokenizer for the vlnce task of {self.model_path}. A released "
                "checkpoint ships one in action_tokenizer/; a checkpoint from a training run "
                "names an absolute path on the machine that trained it, and you have to point "
                "at the bundle yourself with --opt action_tokenizer_bundle=<dir>"
            )
        return str(decoder["bundle_path"]), int(decoder["horizon"])

    def reset(self, instruction: str) -> None:
        self.instruction = instruction
        self.step = 0
        # Clears the frame history, the frame ids, the SlowFast buffer, the
        # session ViT cache and the engine's episode state.
        self.agent.reset(instruction)

    def act(self, rgb: Any) -> Step:
        self.step += 1
        budget = self.opt_int("step_budget")
        if budget and self.step >= budget:
            return Step.stop(raw_output="step budget reached")

        self.agent.observe(rgb)
        try:
            waypoints, raw, _latency_ms = self.agent.predict_waypoints(
                self.instruction, task_type="vlnce_traj"
            )
        except ValueError as exc:
            # A chunk that did not decode is not a trajectory, and there is no
            # honest motion to invent from it. End the episode and say why.
            return Step.stop(raw_output=f"decode failed: {exc}")

        array = np.asarray(waypoints, dtype=np.float32)
        if not np.any(np.abs(array) > RVQ_STOP_ATOL):
            return Step.stop(raw_output=raw)
        return Step.waypoints(array, raw_output=raw, extras={"cluster_id": _cluster_id(raw)})

    def finish(self) -> None:
        # Nothing survives reset(), and the weights stay loaded for the next
        # episode. Releasing them here would reload the model 1,097 times.
        pass

    def describe(self) -> dict[str, Any]:
        return {
            "backend": "vllm_local",
            "decoder": "rvq",
            "horizon": self.horizon,
            "step": self.step,
        }


def _bundle_horizon(bundle_dir: str) -> int:
    """How many waypoints the bundle decodes to, from the bundle's own manifest."""
    import json
    from pathlib import Path

    manifest = Path(bundle_dir) / "manifest.json"
    if not manifest.is_file():
        raise ValueError(f"{bundle_dir} is not an RVQ bundle: no manifest.json in it")
    horizon = json.loads(manifest.read_text()).get("horizon")
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError(f"{manifest} declares no usable horizon (got {horizon!r})")
    return horizon


def _cluster_id(raw: str) -> int:
    """The coarse RVQ codeword, recorded per step so a run can be inspected.

    Never fatal: this is telemetry on the trace, and a policy that answered with
    a usable trajectory must not fail because its text was shaped unusually.
    """
    try:
        from lightnav.vln_utils import parse_rvq_action_tokens

        codes = parse_rvq_action_tokens(raw)
        return int(codes[0]) if codes else -1
    except Exception:
        return -1
