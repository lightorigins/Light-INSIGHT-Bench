#!/usr/bin/env bash
# Evaluate NaVid on INSIGHT-Bench.
#
#   bash scripts/setup_navid_env.sh          # once: virtualenv + upstream code
#   MODEL_PATH=/path/to/weights bash scripts/eval_navid.sh
#
# Upstream: https://github.com/jzhzhang/NaVid-VLN-CE
# Add MAX_EPISODES=5 for a smoke run. See scripts/eval_common.sh for every knob.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/eval_common.sh"

POLICY=navid
MODEL_PYTHON_DEFAULT="$EVAL_DIR/envs/navid/bin/python"
# Where the model's own code sits. Point UPSTREAM_REPO at a clone you
# already have; the default is what setup_navid_env.sh writes.
UPSTREAM_REPO="${UPSTREAM_REPO:-$EVAL_DIR/third_party/navid}"
UPSTREAM_PYTHONPATH="$UPSTREAM_REPO"
MODEL_ENV="HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
insight_bench_main
