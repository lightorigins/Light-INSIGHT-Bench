# Changelog

## 0.2.0

The SDK is now one thing: evaluate a navigation model on INSIGHT-Bench in Isaac
Sim, look at the run, and pack it for submission. Everything that served another
purpose was removed rather than kept "just in case", because every extra path
was a thing a reader had to understand before running the one path they wanted.

### The model side moved out of the package

A model is now hosted by `policies/`, a small standalone program with no
dependency on this package: the standard library plus numpy and pillow, parsing
under Python 3.8. That is what makes it installable in a model's own
environment, which is the whole point -- NaVid needs Python 3.8 and torch 2.0,
LightNav-0 needs Python 3.11 and torch 2.10, and Isaac Lab pins torch 2.7. None
of those can share an interpreter.

`insight_bench.policy_server` (FastAPI), `vln_runtime.policy.adapter`
(`VlnPolicyAdapter`, `PolicyDecision`, `PointingOutput`, `WaypointDelta`) and
`vln_runtime.policy.inproc` are gone with it. The HTTP contract on the wire is
unchanged, so a server written against 0.1.0 still works.

### One command per model

`scripts/eval_<model>.sh` starts the model in its own interpreter, waits for it,
runs the evaluation in Isaac Lab's interpreter, renders the page, packs the
evidence and cleans up. Seven models ship with a script; `eval_custom.sh` runs
yours.

### CLI

Six commands: `check-data`, `check-policy`, `run`, `vis`, `pack`, `verify`.
`catalog`, `validate` and `inspect` were removed -- they answered questions
about the registry that a user of one benchmark never asks.

`run` and `check-data` now require `--episodes` and `--scene-root`, because
every published suite needs both and defaulting them only moved the error later.
`run` gained `--max-episodes` for smoke-testing a setup: the result covers part
of the suite and says so in its own adapter metadata, so it cannot be mistaken
for a full measurement. `--adapter` and `--model-id` are gone; a run's model
identity comes from the policy server's `GET /health` and from nowhere else.

`check-data` is new. It reads the episode file, checks it against the digest the
coordinate pins, resolves every episode's scene against the scene root, and
names what is missing -- once per missing asset, not once per episode that
wanted it. It starts no simulator and downloads nothing.

### Benchmarks

The registry holds `insight-bench-v1@1.0.0` and nothing else. The `bench-tiny`
text fixture and the `env-check-sim` coloured-block tier are gone, with their
runners (`deterministic-json-v1`, `isaac-camera-walk-v1`) and the `ModelAdapter`
text protocol they existed for. The only runner is `objnav-http-policy-v1`.

`insight_bench`'s task config now declares its own measures, terminations and
scores instead of importing them from two suites that no longer exist. The term
functions behind them are byte-identical, so scores do not move.

### The report page

`vis` still writes one self-contained offline file, but it says less and shows
more of the run.

The metrics section is gone. It grew to a card per metric key, was cut to six
numbers, and is now removed entirely: a wall of run-level averages is not what
this page is read for, and the two that a reader does want are already in front
of them -- the success rate on the identity line, and every per-cell rate in the
matrix. `run-result.json` is untouched and still records every metric, including
the four the page no longer draws anywhere: `SR@1m/2m/3m`, `spl`,
`distance_to_goal` and the Wilson interval.

The per-suite Groups section is gone too. It restated counting metrics the
episode table already carries.

The episode table follows the internal episode browser these results are
usually read in: `#`, Episode, Scene, Suite, Result, Instr type, Scene class,
NE (m), Frames, with a counted checkbox multi-select filter on Suite, Result,
Instr type and Scene class, each sitting in its own column header. Every column
sorts in three states -- ascending, descending, back to the run's order -- and
every column can be dragged wider or narrower from its right edge. SPL, Path (m), Stopped and Stop step are no longer
columns. Instruction type and scene class stand where that browser puts a
difficulty column, which on this benchmark reads `LR_towards` on every episode;
the two keys that do vary are also the axes of the radars and the matrix above
the table, so filtering on them is what takes a reader from a matrix cell to the
episodes in it.

Relatedly, a column now needs two distinct *present* values to appear at all.
Counting cells instead of values made a missing entry a value of its own, so a
key every scored episode agreed on looked variable as soon as one episode lacked
it -- which is what any run with an errored episode looks like. That is how the
published run came to show a constant column behind a two-option filter.

Replay now covers the whole run. The cap was 200 episodes, which showed a
replay for under a fifth of a published 1097-episode run and read like missing
data; it is now a page-weight budget of 2000, and that run's page is 5.9 MB
with 1096 replays. The cap warning appears only when the cap applies.

The frame panel no longer draws a marker for the `apos`/`opos` pointing tokens.
The 48x27 grid decode behind it was never checked against a pointing policy
server, so the marker could be confidently wrong; the tokens and `raw_output`
are still shown verbatim, which is what the page can honestly say. Playback
also restarts from the beginning when play is pressed at the end of a replay,
instead of stopping again immediately.

### Packaging

One dependency set (numpy, pillow, pydantic, requests) and a `dev` extra. The
`vln`, `server` and `isaac` extras are gone: this package is installed into
Isaac Lab's interpreter and needs all of it there. `docs/` was removed; the
README is the documentation.

### Unchanged on purpose

`contracts.py`, `schemas/v1/*` and the evidence pack format are byte-compatible
with 0.1.0. A run result or evidence pack produced by either version validates
against the other, which is what lets a leaderboard accept both.
