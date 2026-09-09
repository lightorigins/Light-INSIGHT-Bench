# policies/ — the model side

Everything in this directory runs in **your model's** Python environment, not in
the benchmark's. It depends on nothing but the standard library, `numpy` and
`pillow`, parses under Python 3.8 (several baselines pin py3.8/3.9 venvs), and
never imports `insight_bench`. The two sides meet only over HTTP.

## What a policy is

A policy is one class that answers one question, once per frame: *given this
camera image and this instruction, where do I move next?*

The simulator renders a 480×270 RGB frame at 120° horizontal FOV from a camera
1.0 m off the ground, sends it to your policy, executes the motion that comes
back, renders the next frame, and repeats — up to 300 times, or until your
policy says it has arrived.

## The three methods

```python
from insight_policy import Policy, Step

class MyPolicy(Policy):
    model_id = "my-org/my-model"

    def __init__(self, model_path, **options):
        super().__init__(model_path, **options)   # load weights here

    def reset(self, instruction: str) -> None:
        """New episode. Drop per-episode state, keep the weights."""

    def act(self, rgb) -> Step:
        """One HWC uint8 RGB frame in, one Step out."""
        return Step.primitives(["forward"])

    def finish(self) -> None:
        """Episode over. Release per-episode buffers."""
```

`act` returns exactly one of:

| | |
|---|---|
| `Step.waypoints([(forward_m, lateral_m, yaw_rad), ...])` | explicit displacements; +forward walks along the optical axis, +lateral goes left, +yaw turns left |
| `Step.primitives(["forward", "left", "right"])` | named moves, expanded at this policy's `forward_m` / `turn_deg` |
| `Step.stop()` | "I have arrived" — the pose of this frame is what gets scored |

The two motion answers are handled differently, because models mean different
things by them:

- **A waypoint chunk is a plan.** All of its rows go over the wire in one reply,
  verbatim. The benchmark picks the row it executes — the first with a forward or
  yaw component — and asks your model again on the next frame. Rows are usually
  cumulative poses, so this is the only reading that makes sense.
- **Primitives are separate moves.** `["forward", "left", "left"]` is three of
  them, and the server releases **one per `/act`** so the simulator renders a
  frame for each. Anything larger than one control step is split across replies.

Pass `then_stop=True` for a chunk that ends with a stop. Attach the model's own
text as `raw_output=` — it lands in the trace, and it is what you will read when
an episode goes wrong.

Two things the suite does to a motion before it executes: it drops lateral
velocity, and it clips forward velocity to a non-negative range. A lateral-only
or backwards motion therefore executes as standing still — and the first command
that executes as standing still ends the episode. Turn, then walk.

Two optional hooks: `observe(rgb)` receives frames rendered while a queued
motion drains (only when the policy sets `queue_feed_frames: True`), and
`describe()` adds fields to `/health`.

## Options

Each policy declares every option it accepts in a `defaults` block, and
`--opt key=value` may override those and nothing else — an undeclared key is an
error rather than a silently ignored typo. The base class contributes the knobs
the server itself reads:

| option | meaning |
|---|---|
| `forward_m`, `turn_deg` | size of one `forward` / `left` / `right` |
| `queue_feed_frames` | feed queued-step frames back through `observe()` |
| `on_empty` | `forward` or `stop`, when `act` returns nothing usable |
| `source_hfov_deg` | the rig's horizontal FOV, 120° |
| `center_crop_hfov_deg`, `crop_aspect` | centre-crop into a narrower training FOV |
| `resize` | `(height, width)` after the crop |

## Running one

```bash
python -m insight_policy --policy policies/navid \
    --model-path /weights/navid-7b-r2r-rxr \
    --opt upstream_repo=/code/NaVid-VLN-CE \
    --opt vit_path=/weights/eva_vit_g.pth
# serving on http://127.0.0.1:18081
```

`--policy` names a directory holding a `policy.py` with exactly one `Policy`
subclass; `--host` and `--port` move the socket. The weights load *before* the
port opens, so a `/health` that answers means the model is warm.

Start from `policies/template` — it is a working (if unambitious) policy that
walks forward forever, with the parts to replace marked. Confirm the benchmark
talks to it, then drop your model in.

Next to it are `lightnav0` and the six open-source baselines it is compared
against — `navid`, `uni_navid`, `streamvln`, `internvla_n1`, `janusvln`, and
`embodied_navigator`, which the LightNav-0 paper and the leaderboard call
**TAMP-Nav**. Each carries its own upstream constants as defaults, and each
module docstring records which upstream revision it was ported from and which
`--opt` paths it needs.

Those defaults are the settings the published INSIGHT-Bench rows were produced
with, not neutral ones. In particular `navid` and `streamvln` centre-crop to
their training field of view by default (90° and 79°, both 4:3), because that is
what won the A/B on the eval split and what the published numbers used; the other
four are fed the frame as rendered. Pass `--opt center_crop_hfov_deg=none` to
score a model on the raw frame instead.

## The wire protocol

Four endpoints, JSON in and out, errors reported as HTTP 200 with
`{"success": false, "error": ...}` so the caller can read the reason:

```
GET  /health   -> {"healthy": true, "model_id": ..., ...}
POST /reset    {"instruction", "episode_id"}                       -> {"success": true}
POST /act      {"episode_id", "frame_id", "timestamp_ms",
                "image_jpeg_b64"}                                  -> action + waypoint
POST /finish   {"episode_id"}                                      -> {"success": true}
```

You do not have to implement any of it — `insight_policy.server` does, on the
standard library's `http.server`. It also owns the action queue, the split of a
large motion into the per-step displacements the runner can actually execute,
the optional centre crop, JPEG decoding and the per-episode bookkeeping.
