#!/usr/bin/env bash
# StreamVLN (InternRobotics) -- streaming VLN with SlowFast context modelling,
# a LLaVA-Video Qwen2-7B answering in 4-action chunks.
# Paper: https://arxiv.org/abs/2507.05240
#
# This environment pins Python 3.9 and torch 2.1.2 (the PyPI wheel is the cu121
# build, which is what upstream pins), transformers 4.45.1, flash-attn 2.5.8.
# Upstream: https://github.com/InternRobotics/StreamVLN @ e48f6ff7e9201d93aae003e8f64b04c00cec13bc
#
# Pitfalls a first run would otherwise hit:
#   * PYTHONPATH needs BOTH the checkout and its inner streamvln/ directory --
#     the code uses "from streamvln.streamvln_agent ..." and bare "from model...
#     / from utils... / from llava..." in the same process. eval_streamvln.sh
#     already sets both; this script clones so that layout comes out right.
#   * The vision tower downloads AT LOAD TIME. The checkpoint's mm_tunable_parts
#     contains mm_vision_tower, so SigLipVisionTower.__init__ calls
#     from_pretrained("google/siglip-so400m-patch14-384") even though delay_load
#     is true. Prefetch it (the command is printed at the end) or the first run
#     stalls on a 3.5GB download -- and set HF_HUB_OFFLINE=1, as eval_streamvln.sh
#     does, so from_pretrained never even sends the HEAD request.
#   * The v1_3 checkpoint has historically shipped WITHOUT tokenizer files (just
#     config plus the safetensors shards). If AutoTokenizer fails, pull them from
#     the real_world sibling repo -- that command is printed too.
#   * There is no preprocessor_config.json and you do not need one: the image
#     processor is a SigLipImageProcessor constructed in the vendored
#     llava/model/multimodal_encoder/siglip_encoder.py.
#   * av is pinned to 13.1.0, NOT upstream's 14.4.0, which has no cp39 wheel
#     (it needs py>=3.10) and fails to build from source. av is only reached
#     through a try/except in llava/utils.py, and 13.1.0 is the last cp39 wheel.
#   * numpy-quaternion is missing from upstream's requirements.txt but
#     streamvln_agent.py imports quaternion at module level. Added here.
#   * flash-attn is installed from a PREBUILT wheel only -- never compile it from
#     source. If the download fails the env still runs; pass
#     POLICY_OPTS="--opt attn_implementation=sdpa".
#
# Idempotent: with the venv and the clone already in place this does nothing.
set -euo pipefail

EVAL_DIR="${EVAL_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Caches default under $EVAL_DIR so a machine with a small root disk works; any
# of these already set in your environment is used as it stands.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$EVAL_DIR/.cache/uv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$EVAL_DIR/.cache/pip}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$EVAL_DIR/.cache/uv-python}"
export HF_HOME="${HF_HOME:-$EVAL_DIR/.cache/hf}"
export TMPDIR="${TMPDIR:-$EVAL_DIR/.cache/tmp}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$HF_HOME" "$TMPDIR"

ENV_DIR="$EVAL_DIR/envs/streamvln"
REPO_DIR="$EVAL_DIR/third_party/streamvln"
CKPT_DIR="$EVAL_DIR/checkpoints/StreamVLN"
UPSTREAM_URL="https://github.com/InternRobotics/StreamVLN.git"
UPSTREAM_COMMIT="e48f6ff7e9201d93aae003e8f64b04c00cec13bc"
READY="$ENV_DIR/.insight_env_ready"

UV="${UV:-$(command -v uv || true)}"
if [ -z "$UV" ]; then
    echo "uv not found. Install it (https://docs.astral.sh/uv/) or set UV=/path/to/uv." >&2
    exit 1
fi

# --- 1. upstream code, at the pinned commit ---------------------------------
# eval_streamvln.sh puts $REPO_DIR AND $REPO_DIR/streamvln on PYTHONPATH, so the
# clone has to land exactly here.
if [ "$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || true)" = "$UPSTREAM_COMMIT" ]; then
    echo "[streamvln] upstream already at $UPSTREAM_COMMIT in $REPO_DIR -- nothing to do"
else
    echo "[streamvln] cloning $UPSTREAM_URL @ $UPSTREAM_COMMIT"
    rm -rf "$REPO_DIR"
    mkdir -p "$(dirname "$REPO_DIR")"
    git init -q "$REPO_DIR"
    git -C "$REPO_DIR" remote add origin "$UPSTREAM_URL"
    # Shallow fetch of just the pinned object; full history only if the server
    # refuses fetch-by-sha.
    git -C "$REPO_DIR" fetch -q --depth 1 origin "$UPSTREAM_COMMIT" \
        || git -C "$REPO_DIR" fetch -q origin
    git -C "$REPO_DIR" checkout -q "$UPSTREAM_COMMIT"
fi

# --- 2. the virtualenv ------------------------------------------------------
if [ -f "$READY" ]; then
    echo "[streamvln] venv already built at $ENV_DIR -- nothing to do"
else
    echo "[streamvln] building the venv at $ENV_DIR (Python 3.9, upstream's target)"
    [ -x "$ENV_DIR/bin/python" ] || "$UV" venv --python 3.9 "$ENV_DIR"
    PY="$ENV_DIR/bin/python"

    # torch stack. The PyPI torch 2.1.2 wheel IS the cu121 build, which matches
    # upstream's pin -- no extra index needed.
    "$UV" pip install --python "$PY" torch==2.1.2 torchvision==0.16.2

    # Inference deps, upstream requirements.txt pins except where noted above.
    # Skipped training-only deps: deepspeed, peft, bitsandbytes, wandb, timm,
    # open-clip-torch, clip(git), datasets, torch-scatter, the habitat stack,
    # trl -- all either unused on the inference import path or guarded by
    # try/except in the vendored llava code. (torch_scatter is referenced
    # nowhere in the repo at all; it is a requirements leftover.)
    # flask is here because upstream's own http_realworld_server.py uses it.
    "$UV" pip install --python "$PY" \
        numpy==1.26.1 \
        transformers==4.45.1 \
        tokenizers==0.20.3 \
        accelerate==0.28.0 \
        decord==0.6.0 \
        einops==0.6.1 \
        einops-exts==0.0.4 \
        pillow==11.2.1 \
        omegaconf==2.3.0 \
        av==13.1.0 \
        safetensors==0.5.3 \
        huggingface-hub==0.33.0 \
        sentencepiece==0.1.99 \
        numpy-quaternion \
        opencv-python==4.11.0.86 \
        fastapi uvicorn requests pyyaml flask

    # depth_camera_filtering: the git pin from upstream's requirements.txt.
    "$UV" pip install --python "$PY" \
        "depth-camera-filtering @ git+https://github.com/naokiyokoyama/depth_camera_filtering@39d6e2f391c8b2198a67ad96f94bf6da0acd48a0"

    # flash-attn: PREBUILT wheel only, never a source build. The cp39 /
    # torch2.1 / cu122-tagged wheel works against torch 2.1.2+cu121 because both
    # are pre-cxx11-ABI (torch.compiled_with_cxx11_abi() is False).
    FA_WHL="flash_attn-2.5.8+cu122torch2.1cxx11abiFALSE-cp39-cp39-linux_x86_64.whl"
    FA_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/$FA_WHL"
    if curl -fsSL --retry 5 --connect-timeout 15 -o "$TMPDIR/$FA_WHL" "$FA_URL"; then
        "$UV" pip install --python "$PY" "$TMPDIR/$FA_WHL"
        rm -f "$TMPDIR/$FA_WHL"
    else
        echo "[streamvln] WARNING: could not fetch $FA_WHL" >&2
        echo "[streamvln]          run the eval with POLICY_OPTS=\"--opt attn_implementation=sdpa\"" >&2
    fi

    touch "$READY"
fi

echo "[streamvln] env $ENV_DIR"
echo "[streamvln] code $REPO_DIR"

cat <<EOF

Weights are yours to fetch -- this script never downloads them, because several
baselines are gated and you should accept their licence yourself:

  hf download mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3 \\
    --local-dir "$CKPT_DIR"

The vision tower is a SEPARATE, REQUIRED download -- the checkpoint pulls it from
the hub at load time, so fetch it once into the cache this script just set up:

  HF_HOME="$HF_HOME" hf download google/siglip-so400m-patch14-384

If AutoTokenizer then complains that the checkpoint has no tokenizer, take those
files from the real_world sibling of the same weights:

  hf download mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_real_world \\
    --include "tokenizer*" "*token*" "vocab*" "merges*" --local-dir "$CKPT_DIR"

Then run:

  MODEL_PATH=$CKPT_DIR bash scripts/eval_streamvln.sh
EOF
