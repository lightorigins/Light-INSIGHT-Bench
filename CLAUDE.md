# Repository guide

## What this is

INSIGHT-Bench, the object-goal navigation benchmark of LightNav-0, and the SDK
that runs it. One product path: evaluate an HTTP policy server on a published
suite in Isaac Sim, render the run, pack the evidence, submit it.

## Two sides, two interpreters

`src/insight_bench/` is the run side. It is installed into **Isaac Lab's**
Python and never imports a model.

`policies/` is the model side. It is a standalone program installed into **the
model's** Python. It must import nothing but the standard library, numpy and
pillow, and must parse under Python 3.8. It must never import `insight_bench`.

They meet over `http://127.0.0.1:18081` and four endpoints: `/health`,
`/reset`, `/act`, `/finish`. That wire is a published contract -- changing a
field name silently changes everyone's scores.

## Layout of the run side

- `contracts.py` / `schemas/v1/` — the wire models and their checked-in JSON
  Schemas. **Byte-frozen**: the leaderboard service pins these digests.
- `registry.py` / `registry/` — manifest validation and the published
  coordinates.
- `runner.py`, `runners/` — dispatch and the one runner, `objnav-http-policy-v1`.
- `vln_runtime/` — episodes, scoring, motion, rollout, traces, and the policy
  HTTP client.
- `simulator/` — the Isaac backend and the runtime attestation.
- `vis.py` / `vis_page.py` — the offline HTML report. No external dependency,
  no CDN, no chart library.
- `evidence.py` — packing, redaction, digests, verification.

## Rules

- Do not add benchmark-name conditionals to Python. Add a versioned manifest.
- Never change the bytes behind an existing coordinate. Publish a new version
  and update every SHA-256 pin.
- Registry values are data, never importable object references. The CLI executes
  only code that shipped in this distribution.
- Treat manifests and evidence inputs as untrusted: keep the relative-path
  validation, the symlink rejection, the deterministic JSON, the digest checks
  and the secret/path redaction.
- Nothing is uploaded or downloaded implicitly, ever.
- No internal hostnames, absolute site paths or internal project names anywhere
  in the tree. `tools/check_independence.py` enforces this, including inside the
  README, which ships in the wheel metadata.

## Verification

```bash
ruff check . && ruff format --check .
mypy
pytest
python tools/export_schemas.py --check
python tools/refresh_registry_index.py --check
python tools/check_independence.py
python -m build && python tools/check_independence.py --dist
```

`policies/` is checked separately, against its own floor:

```bash
uv run --python 3.8 python -m compileall -q policies/
```
