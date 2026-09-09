#!/usr/bin/env bash
# Embodied-Navigator (TAMP-Nav) -- a Qwen2.5-VL-7B that looks at a 360-degree
# camera ring and answers with a pixel to walk to: {"choice": "<rgb_...>",
# "pixel": [u, v]}, optionally after a <|think_start|>...<|think_end|> chain.
# Paper: https://arxiv.org/abs/2608.17512
#
# This environment pins Python 3.10 and torch 2.6.0 (the PyPI wheel is the cu124
# build) with torchvision 0.21.0, transformers 4.57.3.
# Upstream: https://github.com/ZJU-OmniAI/Embodied-Navigator @ 7adaed7c8e1845c253fd99dbc1e8dec6c0b241c5
#
# Pitfalls a first run would otherwise hit:
#   * DTHINK_PRELOAD_DA3 defaults to "1", and the FastAPI app then loads a large
#     DepthAnything3 model at startup and refuses to start if that fails. This
#     RGB-only path never uses it; the policy sets DTHINK_PRELOAD_DA3=0 when it
#     spawns the service, and you must do the same if you start it by hand.
#   * The service must run with the checkout as its working directory: src is a
#     namespace package with no __init__.py. The policy passes cwd for you, which
#     is why it needs --opt upstream_repo=.
#   * transformers is pinned to 4.57.3 because that is the checkpoint's own
#     transformers_version -- the repo's vendored modeling code is a fork of it
#     and uses 4.57 internals (masking_utils, modeling_layers, TransformersKwargs).
#   * liger-kernel is a hard dependency for a silly reason: modeling_qwen2_5_vl.py
#     has an unguarded module-level import of it, even though the patch function
#     it provides is never called on the inference path. It drags in triton and
#     needs setuptools.
#   * The HF datasets library is likewise pulled in by import, not by use:
#     src/agent/* imports src.dataset.collate_fn, whose package __init__ reaches
#     load_dataset.py.
#   * flash-attn is NOT needed and NOT installed: the loader never passes
#     attn_implementation, so the model loads with sdpa.
#   * The upstream repo has no requirements.txt, despite what its README says.
#     This list came from walking the actual import chain.
#   * Every /step must carry a sensor pose whose name contains "front" -- the
#     agent indexes [0] into the filtered list and raises IndexError otherwise.
#   * The service's session dict is never garbage-collected and each entry holds
#     PIL images, so reuse ONE session id per episode rather than a new uuid.
#   * A model answer the parser cannot read is returned as STOP, which ends the
#     episode. A run that stops after two steps is usually a formatting failure,
#     not a navigation decision.
#   * Neither the repository nor the weights declare a licence. Check before you
#     rely on them.
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

ENV_DIR="$EVAL_DIR/envs/embodied_navigator"
REPO_DIR="$EVAL_DIR/third_party/embodied_navigator"
CKPT_DIR="$EVAL_DIR/checkpoints/Embodied-Navigator"
UPSTREAM_URL="https://github.com/ZJU-OmniAI/Embodied-Navigator.git"
UPSTREAM_COMMIT="7adaed7c8e1845c253fd99dbc1e8dec6c0b241c5"
READY="$ENV_DIR/.insight_env_ready"

UV="${UV:-$(command -v uv || true)}"
if [ -z "$UV" ]; then
    echo "uv not found. Install it (https://docs.astral.sh/uv/) or set UV=/path/to/uv." >&2
    exit 1
fi

# --- 1. upstream code, at the pinned commit ---------------------------------
# eval_embodied_navigator.sh puts $REPO_DIR on PYTHONPATH, and the policy runs
# the service with $REPO_DIR as its cwd, so the clone has to sit exactly here.
if [ "$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || true)" = "$UPSTREAM_COMMIT" ]; then
    echo "[embodied_navigator] upstream already at $UPSTREAM_COMMIT in $REPO_DIR -- nothing to do"
else
    echo "[embodied_navigator] cloning $UPSTREAM_URL @ $UPSTREAM_COMMIT"
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
    echo "[embodied_navigator] venv already built at $ENV_DIR -- nothing to do"
else
    echo "[embodied_navigator] building the venv at $ENV_DIR (Python 3.10)"
    [ -x "$ENV_DIR/bin/python" ] || "$UV" venv --python 3.10 "$ENV_DIR"
    # The inference and serving path only. No habitat (only src/env and src/eval
    # want it), no cv2, no quaternion, no flash-attn, no depth_anything_3 --
    # that one is switched off with DTHINK_PRELOAD_DA3=0.
    "$UV" pip install --python "$ENV_DIR/bin/python" \
        "torch==2.6.0" "torchvision==0.21.0" "transformers==4.57.3" \
        "accelerate==1.14.0" "datasets==5.0.0" "liger-kernel==0.7.0" "setuptools" \
        "fastapi>=0.110" "uvicorn[standard]" "pillow" "requests" "pyyaml" \
        "numpy<2.2" "packaging"
    touch "$READY"
fi

echo "[embodied_navigator] env $ENV_DIR"
echo "[embodied_navigator] code $REPO_DIR"

cat <<EOF

Weights are yours to fetch -- this script never downloads them, because several
baselines are gated and you should accept their licence yourself:

  hf download UnderTides/Embodied-Navigator-7B-GRPO --local-dir "$CKPT_DIR"

Then run:

  MODEL_PATH=$CKPT_DIR \\
  POLICY_OPTS="--opt upstream_repo=$REPO_DIR" \\
  bash scripts/eval_embodied_navigator.sh
EOF
