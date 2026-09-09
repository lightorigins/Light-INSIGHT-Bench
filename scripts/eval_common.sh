#!/usr/bin/env bash
# Shared body of every scripts/eval_*.sh. Sourced, never run directly.
#
# One evaluation is two processes:
#
#   your model's Python              this repository, in Isaac Lab's Python
#   python -m insight_policy   <--->  isaaclab.sh -p -m insight_bench run
#          (127.0.0.1:$PORT)          /health /reset /act /finish
#
# The model process is started here, waited for, and killed on the way out, so
# the caller only ever runs one command. Nothing is installed, downloaded or
# written outside $RUN_DIR.
#
# Inputs (environment variables):
#   MODEL_PATH        required  the weights directory
#   MODEL_PYTHON      required  an interpreter that can run this model
#   ISAACLAB_DIR      required  your Isaac Lab checkout
#   EVAL_DIR          this repository (default: the directory holding scripts/)
#   DATA_DIR          benchmark data root (default: $EVAL_DIR/data), holding
#                     insight_bench/<version>/episodes.jsonl and scenes/
#   SUITE             benchmark coordinate (default: insight-bench-v1@1.0.0)
#   EPISODES          episode file (default: derived from DATA_DIR and SUITE)
#   SCENE_ROOT        scene assets root (default: $DATA_DIR/scenes)
#   RUN_DIR           output directory (default: $EVAL_DIR/runs/<model>_<timestamp>)
#   MAX_EPISODES      stop after N episodes; a smoke run, not a submittable result
#   RESUME            1 continues an interrupted run in RUN_DIR instead of
#                     re-running the episodes it already scored. Needs the same
#                     RUN_DIR, and the same model, data and seed
#   VIS               1 (default) keeps camera frames for the replay page; 0 does not
#   PORT              loopback port for the model process (default 18081)
#   POLICY_OPTS       extra "--opt key=value" pairs passed to the policy
#   HEALTH_TIMEOUT    seconds to wait for the model to load (default 1800)
#   DRY_RUN           1 prints the commands, checks the port and the upstream
#                     checkout, and exits without touching the GPU
#   UPSTREAM_REPO     a clone of a baseline's own code you already have; the
#                     eval script warns if it is not at the pinned commit
#
# Each eval script sets POLICY (a directory under policies/) and may set
# MODEL_PYTHON_DEFAULT, UPSTREAM_PYTHONPATH, MODEL_ENV and POLICY_OPTS_DEFAULT
# before calling insight_bench_main.

set -euo pipefail

# Resolved at source time, so an eval script can build paths from it before
# calling insight_bench_main.
EVAL_DIR="${EVAL_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

insight_bench_main() {
    local tag="[${POLICY##*/}]"

    MODEL_PYTHON="${MODEL_PYTHON:-${MODEL_PYTHON_DEFAULT:-}}"
    DATA_DIR="${DATA_DIR:-$EVAL_DIR/data}"
    SUITE="${SUITE:-insight-bench-v1@1.0.0}"
    PORT="${PORT:-18081}"
    VIS="${VIS:-1}"
    HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"
    # Appended, not substituted. Each eval script carries the options its model
    # needs; a caller setting POLICY_OPTS wants to add one, and replacing the
    # list instead silently dropped the per-model defaults -- so asking for an
    # action tokenizer path also threw away the GPU-memory setting. Later
    # --opt wins on a repeated key, so the caller's still overrides.
    POLICY_OPTS="${POLICY_OPTS_DEFAULT:-} ${POLICY_OPTS:-}"
    # Three baselines load their model through their own upstream package and
    # need to be told where it is. The script already knows -- it puts the same
    # path on PYTHONPATH -- so passing it saves a reader from a required option
    # that no recipe mentioned and that failed seconds into a run, with the GPU
    # already held. A caller's own --opt still wins, since later wins.
    case "$POLICY" in
        navid | uni_navid | embodied_navigator)
            POLICY_OPTS="--opt upstream_repo=${UPSTREAM_REPO:-} $POLICY_OPTS"
            ;;
    esac

    # Embodied-Navigator runs its model in a child service on a second port, so
    # this baseline occupies two. That port used to be a constant, which meant
    # PORT could not separate two runs of this model and the port check could
    # not see the one that actually collided: a leaked child kept the constant,
    # and the next run's child died on it after loading four checkpoint shards.
    # Deriving it from PORT makes PORT do what the docs promise for this model
    # too, and EXTRA_PORTS puts it under the same pre-flight check.
    case "$POLICY" in
        embodied_navigator)
            CHILD_PORT="${CHILD_PORT:-$((PORT + 3))}"
            EXTRA_PORTS="${EXTRA_PORTS:-} $CHILD_PORT"
            POLICY_OPTS="--opt child_port=$CHILD_PORT $POLICY_OPTS"
            ;;
    esac

    # insight-bench-v1@1.0.0 -> insight_bench/v1/episodes.jsonl
    local suite_id="${SUITE%@*}"
    local suite_version="${suite_id##*-}"
    local suite_name="${suite_id%-*}"
    EPISODES="${EPISODES:-$DATA_DIR/${suite_name//-/_}/$suite_version/episodes.jsonl}"
    SCENE_ROOT="${SCENE_ROOT:-$DATA_DIR/scenes}"
    RUN_DIR="${RUN_DIR:-$EVAL_DIR/runs/${POLICY##*/}_$(date +%Y%m%d_%H%M%S)}"

    _insight_require MODEL_PATH "$MODEL_PATH" "the weights directory"
    _insight_require MODEL_PYTHON "$MODEL_PYTHON" "an interpreter that can run this model"
    _insight_require ISAACLAB_DIR "${ISAACLAB_DIR:-}" "your Isaac Lab checkout"
    _insight_require_path "$MODEL_PATH" "MODEL_PATH"
    _insight_require_path "$EPISODES" "EPISODES"
    _insight_require_path "$SCENE_ROOT" "SCENE_ROOT"
    _insight_require_path "$MODEL_PYTHON" "MODEL_PYTHON"
    _insight_require_path "$ISAACLAB_DIR/isaaclab.sh" "ISAACLAB_DIR/isaaclab.sh"
    _insight_require_path "$EVAL_DIR/policies/${POLICY##*/}" "policies/${POLICY##*/}"

    local isaac_py=("$ISAACLAB_DIR/isaaclab.sh" -p -m insight_bench)
    local model_cmd=(
        "$MODEL_PYTHON" -m insight_policy
        --policy "$EVAL_DIR/policies/${POLICY##*/}"
        --model-path "$MODEL_PATH"
        --host 127.0.0.1 --port "$PORT"
    )
    # shellcheck disable=SC2206  # POLICY_OPTS is a deliberate word-split list
    [ -n "$POLICY_OPTS" ] && model_cmd+=($POLICY_OPTS)

    local run_cmd=(
        "${isaac_py[@]}" run "$SUITE"
        --episodes "$EPISODES"
        --scene-root "$SCENE_ROOT"
        --policy-url "http://127.0.0.1:$PORT"
        --run-dir "$RUN_DIR"
        --output "$RUN_DIR/run-result.json"
    )
    [ -n "${MAX_EPISODES:-}" ] && run_cmd+=(--max-episodes "$MAX_EPISODES")
    [ "${RESUME:-}" = "1" ] && run_cmd+=(--resume)
    [ "$VIS" = "1" ] && run_cmd+=(--vis)

    echo "$tag suite=$SUITE episodes=$EPISODES"
    echo "$tag model=$MODEL_PATH python=$MODEL_PYTHON port=$PORT"
    # A baseline whose policy runs its model in a child service occupies a
    # second port. Saying so is the difference between a reader who knows what
    # to look for when a run leaves something behind and one who finds a port
    # held by a process they never started.
    [ -n "${CHILD_PORT:-}" ] && echo "$tag this model also runs a child service on port $CHILD_PORT"
    echo "$tag run_dir=$RUN_DIR"
    if [ "${DRY_RUN:-}" = "1" ]; then
        echo "$tag would start: ${model_cmd[*]}"
        echo "$tag would run:   ${run_cmd[*]}"
        # A dry run that only prints is worth little: the point of it is to
        # find a mistake before you hold a GPU for twenty minutes. So it makes
        # the same checks the real run makes at the same point -- an occupied
        # port and a missing upstream checkout both used to pass here and fail
        # an hour later, after the weights had loaded.
        local dry_status=0
        if [ -n "${UPSTREAM_REPO:-}" ] && [ ! -d "$UPSTREAM_REPO" ]; then
            echo "$tag UPSTREAM_REPO does not exist: $UPSTREAM_REPO" >&2
            dry_status=1
        fi
        _insight_check_upstream_commit
        _insight_require_free_port || dry_status=1
        [ "$dry_status" = "0" ] && echo "$tag dry run: the preconditions this can check all pass"
        return "$dry_status"
    fi

    _insight_check_upstream_commit

    mkdir -p "$RUN_DIR"

    # --- 1. the model process, in the model's own environment ---------------
    # Refuse a port somebody else already holds. Without this, a leftover server
    # from an earlier run answers the health probe below, our own process dies
    # on bind, and the whole evaluation silently measures the WRONG model. That
    # is not a hypothetical: it happened, and the run looked perfectly normal.
    _insight_require_free_port
    echo "$tag starting the model on 127.0.0.1:$PORT (log: $RUN_DIR/policy.log)"
    (
        export PYTHONPATH="$EVAL_DIR/policies${UPSTREAM_PYTHONPATH:+:$UPSTREAM_PYTHONPATH}${PYTHONPATH:+:$PYTHONPATH}"
        export PYTHONUNBUFFERED=1
        # shellcheck disable=SC2086  # MODEL_ENV is a deliberate KEY=VALUE list
        exec env ${MODEL_ENV:-} "${model_cmd[@]}"
    ) >"$RUN_DIR/policy.log" 2>&1 &
    INSIGHT_POLICY_PID=$!
    trap _insight_cleanup EXIT INT TERM

    _insight_wait_healthy

    # --- 2. the evaluation, in Isaac Lab's Python ---------------------------
    echo "$tag evaluating"
    "${run_cmd[@]}" >"$RUN_DIR/run.log" 2>&1 || {
        echo "$tag the run failed; last 40 lines of $RUN_DIR/run.log:" >&2
        _insight_tail_log "$RUN_DIR/run.log" >&2
        return 1
    }

    # --- 3. report, page, evidence -----------------------------------------
    "$MODEL_PYTHON" - "$RUN_DIR/run-result.json" <<'PY'
import json, sys
result = json.load(open(sys.argv[1]))
metrics = result.get("metrics", {})
print("  model      ", result.get("adapter", {}).get("model_id", "?"))
print("  status     ", result.get("status", "?"))
print("  episodes   ", len(result.get("episodes", [])))
for key in sorted(metrics):
    print(f"  {key:<11}", f"{metrics[key]:.4f}")
subset = result.get("adapter", {}).get("metadata", {}).get("episode_subset")
if subset:
    print("  NOTE       ", subset)
# A full suite that comes back "partial" is otherwise unexplained: say how many
# episodes failed and name the first few, because that is what has to be fixed
# before the result is submittable.
failed = [e for e in result.get("episodes", []) if e.get("error")]
if failed and not subset:
    print(f"  NOTE        {len(failed)} of {len(result.get('episodes', []))} episodes failed;"
          " the result is not submittable until they run")
    for episode in failed[:5]:
        print(f"     {episode.get('episode_id')}: {str(episode.get('error'))[:110]}")
    if len(failed) > 5:
        print(f"     ... and {len(failed) - 5} more, see run-result.json")
PY

    if [ "$VIS" = "1" ]; then
        "${isaac_py[@]}" vis "$RUN_DIR" --no-open >"$RUN_DIR/vis.log" 2>&1 \
            && echo "$tag page   $RUN_DIR/vis.html" \
            || echo "$tag WARNING: vis failed, see $RUN_DIR/vis.log" >&2
    fi

    local traces=()
    while IFS= read -r trace; do traces+=(--include "$trace"); done \
        < <(find "$RUN_DIR" -name trace.json -maxdepth 2 | sort)
    if "${isaac_py[@]}" pack "$RUN_DIR/run-result.json" "$RUN_DIR/evidence-pack" \
            "${traces[@]}" >"$RUN_DIR/pack.log" 2>&1 \
        && "${isaac_py[@]}" verify "$RUN_DIR/evidence-pack" >>"$RUN_DIR/pack.log" 2>&1; then
        # Zipped from INSIDE the pack, so evidence-manifest.json sits at the
        # archive root -- a submission is rejected when it sits one level down.
        # Python rather than the zip binary, which many minimal images lack.
        "$MODEL_PYTHON" - "$RUN_DIR/evidence-pack" "$RUN_DIR/evidence-pack.zip" <<'ZIP'
import pathlib, sys, zipfile
pack, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(pack.rglob("*")):
        if path.is_file():
            archive.write(path, path.relative_to(pack).as_posix())
ZIP
        echo "$tag upload $RUN_DIR/evidence-pack.zip"
    else
        echo "$tag WARNING: pack/verify failed, see $RUN_DIR/pack.log" >&2
    fi
    if [ -n "${MAX_EPISODES:-}" ]; then
        echo "$tag this was a smoke run over $MAX_EPISODES episodes and is not submittable"
    fi
}

_insight_require() {
    if [ -z "$2" ]; then
        echo "set $1: $3" >&2
        exit 2
    fi
}

_insight_require_path() {
    if [ ! -e "$1" ]; then
        echo "$2 does not exist: $1" >&2
        exit 2
    fi
}

# Isaac Lab's launcher writes terminal escapes into the log, so a plain tail of
# the last 40 lines buries the error that matters under tab stops.
_insight_tail_log() {
    # Strip the escape sequences first: delete ESC up front and the sed below
    # starts eating real text, turning "[INFO] Using" into "NFO] Usin".
    tail -40 "$1" \
        | sed $'s/\033\\[[0-9;?]*[ -\/]*[@-~]//g; s/\033[@-Z\\-_]//g' \
        | tr -d $'\000-\010\013-\037'
}

# A baseline's number is only that baseline's if the code behind it is the code
# the result was measured against. UPSTREAM_REPO lets a reader point at a clone
# they already have, and nothing checked it: a clone a few commits ahead of the
# pin evaluates a quietly different policy and reports it under the model's name.
# Warns rather than refuses -- the pin is the setup script's, and a reader may
# have a good reason to run something else, as long as they know they are.
_insight_check_upstream_commit() {
    local setup="$EVAL_DIR/scripts/setup_${POLICY}_env.sh"
    [ "$POLICY" = "uni_navid" ] && setup="$EVAL_DIR/scripts/setup_uninavid_env.sh"
    [ -n "${UPSTREAM_REPO:-}" ] || return 0
    [ -f "$setup" ] || return 0
    local pinned
    pinned=$(sed -n 's/^UPSTREAM_COMMIT="\([0-9a-f]*\)".*/\1/p' "$setup" | head -1)
    [ -n "$pinned" ] || return 0
    local actual
    actual=$(git -C "$UPSTREAM_REPO" rev-parse HEAD 2>/dev/null || true)
    if [ -z "$actual" ]; then
        echo "$tag WARNING: $UPSTREAM_REPO is not a git checkout, so the upstream" >&2
        echo "$tag   commit cannot be verified. The measured one is $pinned." >&2
    elif [ "$actual" != "$pinned" ]; then
        echo "$tag WARNING: upstream is at $actual" >&2
        echo "$tag   but this baseline was measured at $pinned." >&2
        echo "$tag   A different commit is a different policy; the number will not be" >&2
        echo "$tag   comparable. Run setup_$(basename "$setup" .sh | sed 's/^setup_//') to pin it." >&2
    fi
}

_insight_require_free_port() {
    local port
    for port in $PORT ${EXTRA_PORTS:-}; do
        if "$MODEL_PYTHON" - "$port" <<'PORTCHECK'
import socket, sys
probe = socket.socket()
probe.settimeout(2)
sys.exit(0 if probe.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PORTCHECK
        then
            echo "port $port is already in use; something else would answer for your model." >&2
            if [ "$port" != "$PORT" ]; then
                echo "That is this model's child service, not the policy server. A previous run" >&2
                echo "may have left one behind: find it with 'ss -lntp | grep $port' and stop it" >&2
                echo "by pid, or set PORT to move both." >&2
            else
                echo "Stop it, or set PORT to a free one." >&2
            fi
            exit 2
        fi
    done
}

_insight_wait_healthy() {
    local deadline=$((SECONDS + HEALTH_TIMEOUT))
    while :; do
        # Liveness BEFORE health. The other order means a stranger on the port
        # can answer for a model process that already died.
        if ! kill -0 "$INSIGHT_POLICY_PID" 2>/dev/null; then
            echo "[eval] the model process exited before it was ready; last 40 lines:" >&2
            _insight_tail_log "$RUN_DIR/policy.log" >&2
            exit 1
        fi
        if "$MODEL_PYTHON" - "http://127.0.0.1:$PORT/health" <<'PY' 2>/dev/null
import json, sys, urllib.request
with urllib.request.urlopen(sys.argv[1], timeout=5) as response:
    sys.exit(0 if json.load(response).get("model_id") else 1)
PY
        then
            echo "[eval] the model is loaded and answering"
            return 0
        fi
        if [ "$SECONDS" -ge "$deadline" ]; then
            echo "[eval] the model did not answer within ${HEALTH_TIMEOUT}s; last 40 lines:" >&2
            _insight_tail_log "$RUN_DIR/policy.log" >&2
            exit 1
        fi
        sleep 5
    done
}

_insight_cleanup() {
    if [ -n "${INSIGHT_POLICY_PID:-}" ] && kill -0 "$INSIGHT_POLICY_PID" 2>/dev/null; then
        kill "$INSIGHT_POLICY_PID" 2>/dev/null || true
        sleep 2
        kill -9 "$INSIGHT_POLICY_PID" 2>/dev/null || true
    fi
}
