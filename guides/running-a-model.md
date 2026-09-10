# Running a model, and trusting the result

[Back to Evaluation](../README.md#how-it-works)

## Options every `eval_*.sh` takes

Three are required — `ISAACLAB_DIR`, `MODEL_PATH`, `MODEL_PYTHON` — and the rest you reach for when
you hit the situation they describe.

| | |
| :--- | :--- |
| `ISAACLAB_DIR` | the Isaac Lab checkout from Installation step 1. Required by every script |
| `MODEL_PATH` | the weights directory — the one that holds `config.json`. Required |
| `MODEL_PYTHON` | an interpreter that can run this model. Each baseline script defaults to the environment its own `setup_*_env.sh` builds, so set this whenever your environment is somewhere else — including for a baseline |
| `MAX_EPISODES` | stop after this many episodes. The run reports `partial` and is not submittable. It takes a **prefix, not a sample** — the first 60 episodes are all Base and Direction, and the first Outdoor scene is episode 372 — so use it to prove a setup works, never to estimate a score |
| `PORT` | the loopback port the model listens on, `18081` by default. Change it to evaluate two models at once, or when a previous run left a server behind — the script refuses to start on a port somebody else holds rather than evaluate whatever answers it. A baseline whose policy runs a child service derives that second port from this one, and the same check covers it, so moving `PORT` moves everything the run listens on |
| `VIS` | `0` skips the camera frames the replay page shows |
| `RUN_DIR` | where the run writes. Defaults to `runs/<model>_<timestamp>` inside the checkout; point it elsewhere to keep results out of your working tree |
| `POLICY_OPTS` | extra `--opt key=value` for the policy; repeat the flag for several. It is **added to** the options the script already sets, and a repeated key wins |
| `DATA_DIR` | the benchmark data root, if it is not `$EVAL_DIR/data` |
| `EPISODES` | the episode file, if your copy is not laid out as the package is. Derived from `DATA_DIR` otherwise |
| `SCENE_ROOT` | the directory the four scene families sit in. Derived from `DATA_DIR` otherwise |
| `UPSTREAM_REPO` | a clone of the model's own code you already have. Only the six baseline scripts read it; each defaults to what its `setup_*_env.sh` writes. That script pins an exact commit and the eval script warns if your clone is elsewhere — a different commit is a different policy, and the number stops being comparable |
| `RESUME` | `1` continues an interrupted run in `RUN_DIR` rather than repeating the episodes it already scored |
| `DRY_RUN` | `1` prints the two commands it would run and checks what it can without a GPU — that your port is free and your paths exist — then exits. Run this before you book a card |

`SUITE`, `HEALTH_TIMEOUT` and `MODEL_ENV` are documented at the top of `scripts/eval_common.sh`.

On a shared GPU, lower the model's share so the simulator still fits:
`POLICY_OPTS="--opt gpu_memory_utilization=0.25"`. It is an **admission check, not a budget** —
LightNav-0 still takes about 11 GB whatever you set.

## Budget the run by timing a smoke run

A full LightNav-0 run took **6.5 hours on one L20** shared with the simulator. Time a
`MAX_EPISODES=5` smoke run and scale from that rather than from a number here, treating it as a
lower bound: the prefix contains the longer episodes and does not scale linearly. Nothing is printed
between start and end, so use `nohup` or `tmux` and watch `runs/<run>/`.

## An interrupted run continues where it stopped

Each episode saves its score on completion, so the same command with the same `RUN_DIR` and
`RESUME=1` runs only the episodes that never ran:

```bash
RESUME=1 RUN_DIR=<the same directory> MODEL_PATH=<...> bash scripts/eval_janusvln.sh
```

It is refused unless the model, benchmark, episode file, scene root, seed and simulator all match
what wrote those scores — otherwise one success rate would be built out of two measurements.

## Debug the wire without the simulator

```bash
PYTHONPATH=policies python -m insight_policy --policy policies/my_model --model-path <path/to/your/weights> &
$ISAACLAB_DIR/isaaclab.sh -p -m insight_bench check-policy --steps 8
```

`check-policy` reports which waypoint of your plan was picked, the velocity it decodes to, and
whether it reads as an arrival — what the runner would execute at each step. It defaults to
`127.0.0.1:18081`; use `--policy-url` for another port. Missing `model_id` exits `1`. Keep the
unauthenticated server on loopback.

For a baseline, `DRY_RUN=1` prints its interpreter, options and policy directory to pass to
`--policy`: `eval_janusvln.sh` uses `policies/janusvln`, while `eval_uninavid.sh` uses
`policies/uni_navid`.

**"Without the simulator" is not "without the GPU".** Weights load before `/health` answers, to
avoid a cold first `/act`, so a real model already occupies the card by the time you can curl it.
The synthetic black frame checks the connection, not navigation quality; nonsensical output is
expected.

## Sanity-check a run before you believe it

A misintegrated model does not crash. It loads, renders, scores, packs and verifies, and hands you a
plausible number. Four checks catch it, and all four read files the run already wrote:

| Check | Where | What a healthy run looks like |
| :--- | :--- | :--- |
| the model was asked | `trace.json` → `steps[].metadata.policy_inference_time_ms` | a **number** on every step the model was asked for, and `null` on a step served from a queued action. Read the values, not whether the key is there: a queue-based model legitimately reports `null` nine steps in ten, and a waypoint model that re-plans every frame reports a number every step. All `null` means nothing was ever asked |
| it said different things | `steps[].metadata.raw_output` and `steps[].action.executed_delta` | several distinct values per episode. One value repeated is a model answering from nothing |
| the camera moved | `run-result.json` → `metrics.path_length` | metres, not near-zero |
| it chose to stop | `trace.json` → `termination_reason` | `policy_stop` on most episodes. `zero_velocity_stop` is the runner reading a near-zero answer as an arrival, which counts as stopping. All `time_out` means nothing ever decided it had arrived |

`first_frame_warning` in a trace's metadata means that episode's first frame carried no detail — one
bad start pose, or a scene that failed to load.

The JSON Schemas under `src/insight_bench/schemas/v1/` define the run result, the evidence manifest,
the registry, the wire messages and the attestation. **There is no schema for `trace.json`**: its
fields are the ones named in the table above, and the trace is a diagnostic record rather than a wire
contract.

Read `status` before the numbers. `completed` means every episode ran; `partial` means one did not,
or `MAX_EPISODES` cut the suite short, and the script says which. A `failed` *episode* is different
again: it ran fine and did not reach the goal, which is the ordinary outcome here.

For a submission the rule the gate enforces is **at most five episodes may fail to evaluate** — few
enough that dropping them cannot buy a score, since an unevaluated episode leaves the denominator. A
`MAX_EPISODES` run is nowhere near that and is not submittable.

For InteriorGS wrapper and frame checks, see
[Obtain and convert the scenes](scene-conversion.md#check-a-frame-before-you-trust-a-run).
