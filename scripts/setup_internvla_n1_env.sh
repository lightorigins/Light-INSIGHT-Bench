#!/usr/bin/env bash
# InternVLA-N1 (InternNav DualVLN) -- a dual-system navigation foundation model:
# a Qwen2.5-VL-7B System-2 that replans and a diffusion System-1 that refines
# between replans.
# Project page: https://internrobotics.github.io/internvla-n1.github.io/
#
# This environment pins Python 3.9 and torch 2.6.0+cu124 (torchvision 0.21.0+cu124),
# transformers 4.51.0, diffusers 0.31.0, accelerate 1.4.0, flash-attn 2.7.4.post1.
# Upstream: https://github.com/InternRobotics/InternNav @ 7a5c62400ac45b313d9b709c740b64191556a242
#
# Pitfalls a first run would otherwise hit:
#   * NOTHING is installed from upstream's setup.py. Its install_requires pulls
#     requirements/core_requirements.txt (gym + gymnasium + ray + gunicorn +
#     pytest + sentry, and pillow<10 on py3.9) pinned for their eval stack. The
#     model is used through PYTHONPATH instead, and eval_internvla_n1.sh sets it.
#   * diffusers is held at 0.31.0. 0.33's new enable_gradient_checkpointing
#     signature is incompatible with upstream's NextDiT.
#   * setuptools must stay below 81: LongCLIP does "from pkg_resources import
#     packaging", and pkg_resources was removed in setuptools 81.
#   * numpy is installed BEFORE torch and pinned to 1.26.4. Install it after, and
#     resolving against the PyTorch index drags in numpy 2.x.
#   * LongCLIP is an import-time requirement, not an optional extra: the encoder
#     package __init__ imports it as soon as the depth encoder is built. It is a
#     git submodule, initialised below.
#   * DepthAnything-V2-Metric-Hypersim-Small is REQUIRED even for RGB-only
#     inference. This checkpoint's config says "system1": "nextdit_async", so
#     InternVLAN1MetaModel.__init__ calls build_depthanythingv2() at
#     from_pretrained() time and torch.loads the .pth. Its DINOv2-S tower IS the
#     System-1 RGB encoder: the depth IMAGES can be a constant plane, the depth
#     ENCODER WEIGHTS cannot be skipped. The download line is printed at the end.
#   * That load path uses a relative module-level constant, MODEL_PATH_TO, in
#     internnav/model/basemodel/internvla_n1/internvla_n1_arch.py. The policy
#     rewrites it, which is why it needs --opt depth_model_path=.
#   * Upstream's own scripts/realworld/http_internvla_server.py forgets
#     --plan_step_gap and would crash; the released notebook uses 4, and that is
#     this policy's default.
#   * flash-attn is installed from a PREBUILT wheel only -- never compile it from
#     source. The agent asks for flash_attention_2; if the wheel is unavailable,
#     sdpa is the documented fallback.
#   * The PyTorch CDN stalls on some networks; re-running is safe, uv resumes
#     from its cache one wheel at a time.
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

# eval_internvla_n1.sh puts third_party/internnav on PYTHONPATH -- the checkout
# keeps upstream's own package name.
ENV_DIR="$EVAL_DIR/envs/internvla_n1"
REPO_DIR="$EVAL_DIR/third_party/internnav"
CKPT_DIR="$EVAL_DIR/checkpoints/InternVLA-N1"
UPSTREAM_URL="https://github.com/InternRobotics/InternNav.git"
UPSTREAM_COMMIT="7a5c62400ac45b313d9b709c740b64191556a242"
READY="$ENV_DIR/.insight_env_ready"

UV="${UV:-$(command -v uv || true)}"
if [ -z "$UV" ]; then
    echo "uv not found. Install it (https://docs.astral.sh/uv/) or set UV=/path/to/uv." >&2
    exit 1
fi

# --- 1. upstream code, at the pinned commit ---------------------------------
if [ "$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || true)" = "$UPSTREAM_COMMIT" ]; then
    echo "[internvla_n1] upstream already at $UPSTREAM_COMMIT in $REPO_DIR -- nothing to do"
else
    echo "[internvla_n1] cloning $UPSTREAM_URL @ $UPSTREAM_COMMIT"
    rm -rf "$REPO_DIR"
    mkdir -p "$(dirname "$REPO_DIR")"
    git init -q "$REPO_DIR"
    git -C "$REPO_DIR" remote add origin "$UPSTREAM_URL"
    # Shallow fetch of just the pinned object; full history only if the server
    # refuses fetch-by-sha.
    git -C "$REPO_DIR" fetch -q --depth 1 origin "$UPSTREAM_COMMIT" \
        || git -C "$REPO_DIR" fetch -q origin
    git -C "$REPO_DIR" checkout -q "$UPSTREAM_COMMIT"

    # Submodules, at the shas .gitmodules records for this commit:
    #   internnav/model/basemodel/LongCLIP <- beichenzbc/Long-CLIP
    #                                         @ 3966af9ae9331666309a22128468b734db4672a7
    #   third_party/diffusion-policy       <- real-stanford/diffusion_policy
    #                                         @ 5ba07ac6661db573af695b419a7947ecb704690f
    # LongCLIP is import-time required. diffusion-policy is only reached by the
    # navdp System-1 path, which this checkpoint's config never builds -- taken
    # for completeness so the checkout matches upstream.
    echo "[internvla_n1] fetching submodules"
    git -C "$REPO_DIR" submodule update --init \
        internnav/model/basemodel/LongCLIP third_party/diffusion-policy
fi

# --- 2. the virtualenv ------------------------------------------------------
if [ -f "$READY" ]; then
    echo "[internvla_n1] venv already built at $ENV_DIR -- nothing to do"
else
    echo "[internvla_n1] building the venv at $ENV_DIR (Python 3.9)"
    [ -x "$ENV_DIR/bin/python" ] || "$UV" venv --python 3.9 "$ENV_DIR"
    PY="$ENV_DIR/bin/python"

    # numpy FIRST and pinned, so the torch resolve keeps 1.x.
    "$UV" pip install --python "$PY" numpy==1.26.4

    "$UV" pip install --python "$PY" \
        --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.6.0+cu124 torchvision==0.21.0+cu124

    # Model and serving deps. "setuptools<81": LongCLIP needs pkg_resources.
    "$UV" pip install --python "$PY" \
        numpy==1.26.4 \
        transformers==4.51.0 \
        accelerate==1.4.0 \
        diffusers==0.31.0 \
        ftfy==6.3.1 \
        opencv-python pillow imageio \
        gym==0.26.2 \
        numpy-quaternion \
        pydantic \
        fastapi uvicorn requests pyyaml \
        "setuptools<81" \
        regex tqdm einops

    # flash-attn: PREBUILT wheel only, never a source build. cp39 / torch2.6 /
    # cu12, cxx11abiFALSE -- matching torch.compiled_with_cxx11_abi() == False.
    FA_WHL="flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp39-cp39-linux_x86_64.whl"
    FA_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/$FA_WHL"
    if curl -fsSL --retry 5 --connect-timeout 15 -o "$TMPDIR/$FA_WHL" "$FA_URL"; then
        "$UV" pip install --python "$PY" "$TMPDIR/$FA_WHL"
        rm -f "$TMPDIR/$FA_WHL"
    else
        echo "[internvla_n1] WARNING: could not fetch $FA_WHL" >&2
        echo "[internvla_n1]          the model asks for flash_attention_2; sdpa is the" >&2
        echo "[internvla_n1]          fallback the authors document." >&2
    fi

    touch "$READY"
fi

echo "[internvla_n1] env $ENV_DIR"
echo "[internvla_n1] code $REPO_DIR"

cat <<EOF

Weights are yours to fetch -- this script never downloads them, because several
baselines are gated and you should accept their licence yourself:

  hf download InternRobotics/InternVLA-N1-DualVLN --local-dir "$CKPT_DIR/dualvln"

The depth encoder is a SEPARATE, REQUIRED download: this checkpoint builds
DepthAnything-V2 at from_pretrained() time, and its DINOv2-S tower is System-1's
RGB encoder.

  hf download depth-anything/Depth-Anything-V2-Metric-Hypersim-Small \\
    --local-dir "$CKPT_DIR/depth_anything_v2_metric_hypersim_small"

Then run:

  MODEL_PATH=$CKPT_DIR/dualvln \\
  POLICY_OPTS="--opt depth_model_path=$CKPT_DIR/depth_anything_v2_metric_hypersim_small" \\
  bash scripts/eval_internvla_n1.sh
EOF
