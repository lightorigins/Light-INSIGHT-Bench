#!/usr/bin/env bash
# Evaluate StreamVLN on INSIGHT-Bench.
#
#   bash scripts/setup_streamvln_env.sh          # once: virtualenv + upstream code
#   MODEL_PATH=/path/to/weights bash scripts/eval_streamvln.sh
#
# Upstream: https://github.com/InternRobotics/StreamVLN
# Add MAX_EPISODES=5 for a smoke run. See scripts/eval_common.sh for every knob.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/eval_common.sh"

POLICY=streamvln
MODEL_PYTHON_DEFAULT="$EVAL_DIR/envs/streamvln/bin/python"
# Where the model's own code sits. Point UPSTREAM_REPO at a clone you
# already have; the default is what setup_streamvln_env.sh writes.
UPSTREAM_REPO="${UPSTREAM_REPO:-$EVAL_DIR/third_party/streamvln}"
UPSTREAM_PYTHONPATH="$UPSTREAM_REPO:$UPSTREAM_REPO/streamvln"
MODEL_ENV="HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
insight_bench_main
