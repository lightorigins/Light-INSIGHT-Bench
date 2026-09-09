#!/usr/bin/env bash
# JanusVLN -- Qwen2.5-VL-7B with a VGGT spatial tower, holding two memories:
# semantic (RGB history, re-encoded each step) and spatial (a VGGT KV cache that
# persists across the episode).
# Paper: https://arxiv.org/abs/2509.22548
#
# This environment pins Python 3.9 and torch 2.5.0+cu124 (torchvision
# 0.20.0+cu124), transformers 4.50.0, flash-attn 2.7.1.post4.
# Upstream: https://github.com/MIV-XJTU/JanusVLN @ 8cfe3129f7af4ca276bd5e38f1fae7e5367a09ec
#
# The authors distribute this environment as a prebuilt conda-pack tarball that
# is really StreamVLN's environment (JanusVLN reuses it, and their tutorial says
# so). A conda-pack tarball cannot be relocated after unpacking and is awkward to
# audit, so this script rebuilds the same versions with uv, from that tarball's
# own package manifest, minus habitat-lab/sim/baselines -- nothing on the
# inference path imports habitat.
#
# Pitfalls a first run would otherwise hit:
#   * PYTHONPATH needs BOTH the checkout and its src/ directory: the code does
#     "from qwen_vl..." and bare "import utils / import habitat_extensions" in
#     the same process. eval_janusvln.sh sets both; this script clones so that
#     layout comes out right.
#   * NEVER import src/evaluation.py. Its first two dozen lines import habitat
#     and habitat_baselines. The parts worth having -- the model class, the VGGT
#     loader, the prompt -- are transcribed into the policy instead.
#     habitat_extensions.measures and .maps are habitat-entangled the same way;
#     the bare "import habitat_extensions" is fine.
#   * The checkpoint's preprocessor_config.json says max_pixels=451584, but
#     upstream overrides it at construction with max_pixels=1605632,
#     min_pixels=784. The policy carries those as defaults; do not "fix" them.
#   * mode='evaluation' is load-bearing at from_pretrained(): it keeps the VGGT
#     KV cache across steps. In training mode the cache is cleared every call.
#   * flash-attn is installed from a PREBUILT wheel only -- never compile it from
#     source. If the download fails the env still runs; pass
#     POLICY_OPTS="--opt attn_implementation=sdpa".
#   * The upstream repository declares no licence. Check before you rely on it.
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

ENV_DIR="$EVAL_DIR/envs/janusvln"
REPO_DIR="$EVAL_DIR/third_party/janusvln"
CKPT_DIR="$EVAL_DIR/checkpoints/JanusVLN"
UPSTREAM_URL="https://github.com/MIV-XJTU/JanusVLN.git"
UPSTREAM_COMMIT="8cfe3129f7af4ca276bd5e38f1fae7e5367a09ec"
READY="$ENV_DIR/.insight_env_ready"

UV="${UV:-$(command -v uv || true)}"
if [ -z "$UV" ]; then
    echo "uv not found. Install it (https://docs.astral.sh/uv/) or set UV=/path/to/uv." >&2
    exit 1
fi

# --- 1. upstream code, at the pinned commit ---------------------------------
# eval_janusvln.sh puts $REPO_DIR AND $REPO_DIR/src on PYTHONPATH, so the clone
# has to land exactly here.
if [ "$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || true)" = "$UPSTREAM_COMMIT" ]; then
    echo "[janusvln] upstream already at $UPSTREAM_COMMIT in $REPO_DIR -- nothing to do"
else
    echo "[janusvln] cloning $UPSTREAM_URL @ $UPSTREAM_COMMIT"
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
    echo "[janusvln] venv already built at $ENV_DIR -- nothing to do"
else
    echo "[janusvln] building the venv at $ENV_DIR (Python 3.9)"
    [ -x "$ENV_DIR/bin/python" ] || "$UV" venv --python 3.9 "$ENV_DIR"
    PY="$ENV_DIR/bin/python"

    "$UV" pip install --python "$PY" \
        --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.5.0+cu124 torchvision==0.20.0+cu124

    # Versions taken from the official environment's own package manifest. The
    # inference subset only: skipped are the training and benchmarking entries of
    # upstream's requirements.txt (deepspeed, peft, wandb, datasets, evaluate,
    # decord, torchcodec, timm, nltk, sacrebleu, matplotlib and the rest), which
    # are reached only from qwen_vl/train/*, qwen_vl/data/* and the VGGT
    # visualisation helper -- never from the model, processor or VGGT loader.
    "$UV" pip install --python "$PY" \
        transformers==4.50.0 \
        tokenizers==0.21.4 \
        accelerate==0.28.0 \
        huggingface-hub==0.33.4 \
        safetensors==0.5.3 \
        qwen-vl-utils==0.0.11 \
        einops==0.8.1 \
        numpy==1.26.4 \
        pillow==11.3.0 \
        requests pyyaml

    # flash-attn: PREBUILT wheel only, never a source build. cp39 / torch2.5 /
    # cu12, cxx11abiFALSE -- the Dao-AILab release naming for this combination.
    # A miss here is not fatal: sdpa loads the same checkpoint.
    FA_WHL="flash_attn-2.7.1.post4+cu12torch2.5cxx11abiFALSE-cp39-cp39-linux_x86_64.whl"
    FA_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.1.post4/$FA_WHL"
    if curl -fsSL --retry 5 --connect-timeout 15 -o "$TMPDIR/$FA_WHL" "$FA_URL"; then
        "$UV" pip install --python "$PY" "$TMPDIR/$FA_WHL"
        rm -f "$TMPDIR/$FA_WHL"
    else
        echo "[janusvln] WARNING: could not fetch $FA_WHL" >&2
        echo "[janusvln]          run the eval with POLICY_OPTS=\"--opt attn_implementation=sdpa\"" >&2
    fi

    touch "$READY"
fi

echo "[janusvln] env $ENV_DIR"
echo "[janusvln] code $REPO_DIR"

cat <<EOF

Weights are yours to fetch -- this script never downloads them, because several
baselines are gated and you should accept their licence yourself.

JanusVLN's weights are published on ModelScope, not on Hugging Face, so there is
no "hf download" line for them. The build notes name the corrected re-upload:

  https://modelscope.cn/models/misstl/JanusVLN_Extra

  modelscope download --model misstl/JanusVLN_Extra --local_dir "$CKPT_DIR"

(Upstream's README also lists https://modelscope.cn/models/misstl/JanusVLN_Base.
The published numbers here were produced with the Extra weights.)

Then run:

  MODEL_PATH=$CKPT_DIR bash scripts/eval_janusvln.sh
EOF
