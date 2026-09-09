#!/usr/bin/env bash
# Evaluate TAMP-Nav (Embodied-Navigator) on INSIGHT-Bench.
#
#   bash scripts/setup_embodied_navigator_env.sh          # once: virtualenv + upstream code
#   MODEL_PATH=/path/to/weights bash scripts/eval_embodied_navigator.sh
#
# Upstream: https://arxiv.org/abs/2608.17512
# Add MAX_EPISODES=5 for a smoke run. See scripts/eval_common.sh for every knob.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/eval_common.sh"

POLICY=embodied_navigator
MODEL_PYTHON_DEFAULT="$EVAL_DIR/envs/embodied_navigator/bin/python"
# Where the model's own code sits. Point UPSTREAM_REPO at a clone you
# already have; the default is what setup_embodied_navigator_env.sh writes.
UPSTREAM_REPO="${UPSTREAM_REPO:-$EVAL_DIR/third_party/embodied_navigator}"
UPSTREAM_PYTHONPATH="$UPSTREAM_REPO"
MODEL_ENV="HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
insight_bench_main
