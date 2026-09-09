#!/usr/bin/env bash
# Evaluate InternVLA-N1 on INSIGHT-Bench.
#
#   bash scripts/setup_internvla_n1_env.sh          # once: virtualenv + upstream code
#   MODEL_PATH=/path/to/weights bash scripts/eval_internvla_n1.sh
#
# Upstream: https://internrobotics.github.io/internvla-n1.github.io/
# Add MAX_EPISODES=5 for a smoke run. See scripts/eval_common.sh for every knob.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/eval_common.sh"

POLICY=internvla_n1
MODEL_PYTHON_DEFAULT="$EVAL_DIR/envs/internvla_n1/bin/python"
# Where the model's own code sits. Point UPSTREAM_REPO at a clone you
# already have; the default is what setup_internvla_n1_env.sh writes.
UPSTREAM_REPO="${UPSTREAM_REPO:-$EVAL_DIR/third_party/internnav}"
UPSTREAM_PYTHONPATH="$UPSTREAM_REPO"
MODEL_ENV="HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
insight_bench_main
