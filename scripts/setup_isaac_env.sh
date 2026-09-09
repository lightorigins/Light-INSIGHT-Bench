#!/usr/bin/env bash
# Install Isaac Sim 5.1.0 and Isaac Lab 2.3.2 into a self-contained directory.
#
#   ISAAC_DIR=/where/there/is/room bash scripts/setup_isaac_env.sh
#
# This is the Installation step 1 of the README, as a script. It needs an NVIDIA
# GPU, about 110 GB of disk and roughly an hour of downloading. If you already
# have Isaac Lab 2.3.x on Isaac Sim 5.1.0, skip this and point ISAACLAB_DIR at it.
#
#   ISAAC_DIR      where to install (default: $EVAL_DIR/isaac)
#   ISAAC_PYTHON   python3.11 to build the virtualenv from (default: python3.11)
#
# Caches default to $ISAAC_DIR/.cache so a small root filesystem is not filled;
# override UV_CACHE_DIR / PIP_CACHE_DIR / TMPDIR if you want them elsewhere.
set -euo pipefail

EVAL_DIR="${EVAL_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ISAAC_DIR="${ISAAC_DIR:-$EVAL_DIR/isaac}"
ISAAC_PYTHON="${ISAAC_PYTHON:-python3.11}"
ISAACLAB_TAG="v2.3.2"
ISAACSIM_VERSION="5.1.0"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$ISAAC_DIR/.cache/uv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ISAAC_DIR/.cache/pip}"
export TMPDIR="${TMPDIR:-$ISAAC_DIR/.cache/tmp}"
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR" "$ISAAC_DIR"

VENV="$ISAAC_DIR/venv"
LAB="$ISAAC_DIR/IsaacLab"

echo "[isaac] installing into $ISAAC_DIR (about 110 GB)"

if [ ! -x "$VENV/bin/python" ]; then
    if command -v uv >/dev/null 2>&1; then
        uv venv --python 3.11 --seed "$VENV"
    else
        "$ISAAC_PYTHON" -m venv "$VENV"
    fi
fi
PY="$VENV/bin/python"
"$PY" -m pip install --upgrade pip

# Isaac Sim 5.1.0 is built against this exact torch line.
"$PY" -m pip install -U torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install "isaacsim[all,extscache]==${ISAACSIM_VERSION}" \
    --extra-index-url https://pypi.nvidia.com

if [ ! -d "$LAB/.git" ]; then
    git clone https://github.com/isaac-sim/IsaacLab.git --branch "$ISAACLAB_TAG" "$LAB"
fi
# `none` installs Isaac Lab without any reinforcement-learning framework:
# evaluation drives the simulator directly and needs none of them.
(cd "$LAB" && ./isaaclab.sh --install none)

# isaaclab.sh looks for an activated virtualenv or this shim. Writing the shim
# means every later command works from any shell, with nothing to activate.
mkdir -p "$LAB/_isaac_sim"
cat > "$LAB/_isaac_sim/python.sh" <<SHIM
#!/bin/bash
# Points isaaclab.sh -p at the virtualenv Isaac Sim was pip-installed into.
exec "$VENV/bin/python" "\$@"
SHIM
chmod +x "$LAB/_isaac_sim/python.sh"

echo
echo "[isaac] verifying"
# Both are needed for an unattended import: without the EULA variable the first
# `import isaacsim` waits on an interactive licence prompt and never returns.
OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}" OMNI_KIT_ALLOW_ROOT=1 \
    "$LAB/isaaclab.sh" -p -c "import isaacsim, isaaclab; print('isaac sim + isaac lab import ok')"

cat <<EOF

Done. Add these to your shell, then continue with Installation step 2:

    export ISAACLAB_DIR=$LAB
    export OMNI_KIT_ACCEPT_EULA=YES     # you are accepting NVIDIA's Omniverse licence
    export OMNI_KIT_ALLOW_ROOT=1        # only if you run as root

EOF
