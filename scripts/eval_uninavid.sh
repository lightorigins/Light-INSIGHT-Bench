#!/usr/bin/env bash
# Evaluate Uni-NaVid on INSIGHT-Bench.
#
#   bash scripts/setup_uni_navid_env.sh          # once: virtualenv + upstream code
#   MODEL_PATH=/path/to/weights bash scripts/eval_uni_navid.sh
#
# Upstream: https://github.com/jzhzhang/Uni-NaVid
# Add MAX_EPISODES=5 for a smoke run. See scripts/eval_common.sh for every knob.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/eval_common.sh"

POLICY=uni_navid
MODEL_PYTHON_DEFAULT="$EVAL_DIR/envs/uni_navid/bin/python"
# Where the model's own code sits. Point UPSTREAM_REPO at a clone you
# already have; the default is what setup_uninavid_env.sh writes.
UPSTREAM_REPO="${UPSTREAM_REPO:-$EVAL_DIR/third_party/uni_navid}"
UPSTREAM_PYTHONPATH="$UPSTREAM_REPO"
MODEL_ENV="HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
insight_bench_main
