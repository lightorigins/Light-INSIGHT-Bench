#!/usr/bin/env bash
# Evaluate LightNav-0 on INSIGHT-Bench.
#
#   MODEL_PATH=/path/to/checkpoints/LightNav-0 \
#   MODEL_PYTHON=/path/to/LightNav-0/.venv/bin/python \
#   bash scripts/eval_lightnav0.sh
#
# The model environment is LightNav-0's own virtualenv, built by its repository:
#   git clone https://github.com/lightorigins/LightNav-0 && cd LightNav-0
#   python3.11 -m venv .venv && . .venv/bin/activate && pip install -e ".[vllm]"
#   hf download LightOriginsHQ/LightNav-0 --local-dir checkpoints/LightNav-0
#
# Add MAX_EPISODES=5 for a smoke run. See scripts/eval_common.sh for every knob.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/eval_common.sh"

POLICY=lightnav0
# The GPU share lives in the policy's own defaults, so there is one place to
# read it and one place to change it. Override it here or on the command line
# with POLICY_OPTS="--opt gpu_memory_utilization=..." if your card differs.
insight_bench_main
