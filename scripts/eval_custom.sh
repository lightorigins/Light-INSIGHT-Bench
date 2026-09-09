#!/usr/bin/env bash
# Evaluate your own model on INSIGHT-Bench.
#
#   cp -r policies/template policies/my_model      # then implement three methods
#   POLICY=policies/my_model \
#   MODEL_PYTHON=/path/to/your/env/bin/python \
#   MODEL_PATH=/path/to/your/checkpoint \
#   bash scripts/eval_custom.sh
#
# Your environment needs Python 3.8 or newer with numpy and pillow. It does not
# need this repository installed: the policy server is plain files under
# policies/, put on PYTHONPATH for you.
#
# Debug the wire first, without the simulator:
#   PYTHONPATH=policies python -m insight_policy --policy policies/my_model --model-path ...
#   $ISAACLAB_DIR/isaaclab.sh -p -m insight_bench check-policy
#
# Add MAX_EPISODES=5 for a smoke run. See scripts/eval_common.sh for every knob.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/eval_common.sh"

if [ -z "${POLICY:-}" ]; then
    echo "set POLICY to your policy directory, e.g. POLICY=policies/my_model" >&2
    exit 2
fi
insight_bench_main
