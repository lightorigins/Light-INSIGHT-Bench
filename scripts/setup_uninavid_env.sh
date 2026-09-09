#!/usr/bin/env bash
# Uni-NaVid (RSS'25) -- one video VLA unifying VLN, ObjectNav, EQA and following.
# Paper: https://arxiv.org/abs/2412.06224
#
# This environment pins Python 3.10 and torch 2.0.1+cu118, with transformers held
# at 4.31.0: the released model code is written against it.
# Upstream: https://github.com/jzhzhang/Uni-NaVid @ 79ef5ea3fea14c205342d1ab070563d84c7a966a
#
# Pitfalls a first run would otherwise hit:
#   * Same LLaMA-VID trap as NaVid: the checkpoint's config.json has CWD-RELATIVE
#     paths -- "mm_vision_tower": "./model_zoo/eva_vit_g.pth" (existence-checked,
#     ValueError when missing) and "image_processor": "./uninavid/processor/
#     clip-patch14-224", inside the checkout. The policy chdirs into the clone and
#     symlinks model_zoo/eva_vit_g.pth, which is why it needs --opt upstream_repo=
#     and --opt vit_path=.
#   * eva_vit_g.pth is the same LLaMA-VID vision encoder NaVid uses; if you have
#     already fetched it for NaVid, point vit_path at that copy.
#   * The weights directory name must contain "vid" -- get_model_name_from_path()
#     switches on it to pick the LLaVA-LLaMA-VID class. Do not rename the folder.
#   * huggingface_hub MUST stay below 0.20 or transformers 4.31 fails to import.
#   * flash-attn is not needed. The README's flash-attn line is for finetuning;
#     grep only finds it under uninavid/train/*. Inference runs eager attention.
#     (If you ever did need it: 2.5.9.post1 has no torch-2.0 wheel, so the
#     matching prebuilt would be a 2.3.x cu118/torch2.0/cp310 one. Never build
#     flash-attn from source here.)
#   * Importing the uninavid package prints "Setting WANDB_MODE to offline".
#     Harmless -- wandb is not installed.
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

# eval_uninavid.sh looks for envs/uni_navid and third_party/uni_navid.
ENV_DIR="$EVAL_DIR/envs/uni_navid"
REPO_DIR="$EVAL_DIR/third_party/uni_navid"
CKPT_DIR="$EVAL_DIR/checkpoints/Uni-NaVid"
UPSTREAM_URL="https://github.com/jzhzhang/Uni-NaVid.git"
UPSTREAM_COMMIT="79ef5ea3fea14c205342d1ab070563d84c7a966a"
READY="$ENV_DIR/.insight_env_ready"

UV="${UV:-$(command -v uv || true)}"
if [ -z "$UV" ]; then
    echo "uv not found. Install it (https://docs.astral.sh/uv/) or set UV=/path/to/uv." >&2
    exit 1
fi

# --- 1. upstream code, at the pinned commit ---------------------------------
# eval_uninavid.sh puts $REPO_DIR on PYTHONPATH, so the clone has to sit here.
if [ "$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || true)" = "$UPSTREAM_COMMIT" ]; then
    echo "[uni_navid] upstream already at $UPSTREAM_COMMIT in $REPO_DIR -- nothing to do"
else
    echo "[uni_navid] cloning $UPSTREAM_URL @ $UPSTREAM_COMMIT"
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
    echo "[uni_navid] venv already built at $ENV_DIR -- nothing to do"
else
    echo "[uni_navid] building the venv at $ENV_DIR (Python 3.10)"
    [ -x "$ENV_DIR/bin/python" ] || "$UV" venv --python 3.10 "$ENV_DIR"
    # Inference subset of the upstream pyproject deps. Skipped on purpose:
    # deepspeed/wandb (train), gradio/gradio_client/markdown2/httpx (serve),
    # bitsandbytes (only for load_8bit/4bit), scikit-learn, fairscale, decord,
    # shortuuid -- none are imported by the inference chain.
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
        einops==0.6.1 \
        tqdm \
        fastapi uvicorn requests pyyaml
    touch "$READY"
fi

echo "[uni_navid] env $ENV_DIR"
echo "[uni_navid] code $REPO_DIR"

cat <<EOF

Weights are yours to fetch -- this script never downloads them, because several
baselines are gated and you should accept their licence yourself:

  hf download Jzzhang/Uni-NaVid --local-dir "$CKPT_DIR"

eva_vit_g.pth is a separate file and is not in that repo. It is the LLaMA-VID
vision encoder, shared with NaVid:

  https://github.com/dvlab-research/LLaMA-VID  (model_zoo/LAVIS/eva_vit_g.pth)

Put it at $CKPT_DIR/eva_vit_g.pth, then run:

  MODEL_PATH=$CKPT_DIR/uninavid-7b-full-224-video-fps-1-grid-2 \\
  POLICY_OPTS="--opt upstream_repo=$REPO_DIR --opt vit_path=$CKPT_DIR/eva_vit_g.pth" \\
  bash scripts/eval_uninavid.sh
EOF
