#!/usr/bin/env bash
# NaVid (RSS'24) -- a video VLM that plans the next VLN step from RGB alone.
# Paper: https://arxiv.org/abs/2402.15852
#
# This environment pins Python 3.8 and torch 2.0.1+cu118 -- the LLaMA-VID era the
# checkpoint was trained in. transformers is held at 4.31.0 because the released
# model code is written against it.
# Upstream: https://github.com/jzhzhang/NaVid-VLN-CE @ ce2f804a43fb23f1d52431ec24be1acf74c3f4f4
#
# Pitfalls a first run would otherwise hit:
#   * The checkpoint's config.json carries CWD-RELATIVE paths, LLaMA-VID style:
#     "mm_vision_tower": "./model_zoo/eva_vit_g.pth" -- build_vision_tower()
#     os.path.exists() it and raises ValueError when it is missing -- and
#     "image_processor": "./navid/processor/clip-patch14-224", a directory that
#     lives inside the checkout. That is why the policy chdirs into the clone
#     and needs --opt upstream_repo= and --opt vit_path=; it creates the
#     model_zoo/eva_vit_g.pth symlink for you.
#   * eva_vit_g.pth is NOT part of the NaVid weight repo. It is the LLaMA-VID
#     vision encoder; the download block printed at the end says where it is.
#   * The weights directory name must contain "navid" in lower case --
#     get_model_name_from_path() switches on that substring. Do not rename it.
#   * huggingface_hub MUST stay below 0.20 or transformers 4.31 fails to import.
#   * flash-attn is not needed. It is imported only by navid/train/*
#     (a training monkey-patch); inference runs stock eager LLaMA attention.
#   * The PyTorch CDN stalls on some networks. UV_HTTP_TIMEOUT is raised to 600s
#     and re-running is safe: uv resumes from its cache one wheel at a time.
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

ENV_DIR="$EVAL_DIR/envs/navid"
REPO_DIR="$EVAL_DIR/third_party/navid"
CKPT_DIR="$EVAL_DIR/checkpoints/NaVid"
UPSTREAM_URL="https://github.com/jzhzhang/NaVid-VLN-CE.git"
UPSTREAM_COMMIT="ce2f804a43fb23f1d52431ec24be1acf74c3f4f4"
READY="$ENV_DIR/.insight_env_ready"

UV="${UV:-$(command -v uv || true)}"
if [ -z "$UV" ]; then
    echo "uv not found. Install it (https://docs.astral.sh/uv/) or set UV=/path/to/uv." >&2
    exit 1
fi

# --- 1. upstream code, at the pinned commit ---------------------------------
# eval_navid.sh puts $REPO_DIR on PYTHONPATH, so the clone has to sit exactly here.
if [ "$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || true)" = "$UPSTREAM_COMMIT" ]; then
    echo "[navid] upstream already at $UPSTREAM_COMMIT in $REPO_DIR -- nothing to do"
else
    echo "[navid] cloning $UPSTREAM_URL @ $UPSTREAM_COMMIT"
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
    echo "[navid] venv already built at $ENV_DIR -- nothing to do"
else
    echo "[navid] building the venv at $ENV_DIR (Python 3.8)"
    [ -x "$ENV_DIR/bin/python" ] || "$UV" venv --python 3.8 "$ENV_DIR"
    # Pins follow NaVid-VLN-CE requirements.txt (the model-side subset; habitat/
    # gym/gradio/bitsandbytes/dtw/scikit-learn/decord/fairscale/deepspeed are
    # deliberately skipped -- none of them are imported by the navid package).
    # huggingface_hub<0.20 is required for transformers 4.31 imports to work.
    "$UV" pip install --python "$ENV_DIR/bin/python" \
        --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cu118 \
        torch==2.0.1+cu118 \
        torchvision==0.15.2+cu118 \
        transformers==4.31.0 \
        tokenizers==0.13.3 \
        accelerate==0.21.0 \
        peft==0.4.0 \
        sentencepiece==0.1.99 \
        huggingface_hub==0.16.4 \
        safetensors \
        timm==0.6.13 \
        numpy==1.23.5 \
        pillow \
        opencv-python \
        imageio \
        tqdm \
        fastapi eval_type_backport uvicorn requests pyyaml
    touch "$READY"
fi

echo "[navid] env $ENV_DIR"
echo "[navid] code $REPO_DIR"

cat <<EOF

Weights are yours to fetch -- this script never downloads them, because several
baselines are gated and you should accept their licence yourself:

  hf download Jzzhang/NaVid --local-dir "$CKPT_DIR"

eva_vit_g.pth is a separate file and is not in that repo. NaVid's README points
at the LLaMA-VID vision encoder for it:

  https://github.com/dvlab-research/LLaMA-VID  (model_zoo/LAVIS/eva_vit_g.pth)

Put it at $CKPT_DIR/eva_vit_g.pth, then run:

  MODEL_PATH=$CKPT_DIR/navid-7b-full-224-video-fps-1-grid-2-r2r-rxr-training-split \\
  POLICY_OPTS="--opt upstream_repo=$REPO_DIR --opt vit_path=$CKPT_DIR/eva_vit_g.pth" \\
  bash scripts/eval_navid.sh
EOF
