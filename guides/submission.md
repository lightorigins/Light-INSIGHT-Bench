# Adding a row to the INSIGHT-Bench leaderboard

A submission is a pull request that adds **one file** to `submissions/`. That file
carries no scores. It points at an evidence pack, and CI reads the numbers out of
the pack itself, so nothing on the leaderboard depends on a number you typed.

## What you do

**1. Evaluate all 1,097 episodes.** A truncated run cannot earn a row; at most five
episodes may fail to evaluate.

```bash
MODEL_PATH=<path/to/your/checkpoint> bash scripts/eval_<your_model>.sh
```

**2. Build and check the evidence pack.** Every `scripts/eval_*.sh` already does
this and prints the zip. `verify` exiting `0` is the same check CI repeats.

```bash
insight-bench verify runs/<run>/evidence-pack.zip     # must exit 0
sha256sum runs/<run>/evidence-pack.zip
```

**3. Host the pack somewhere public and stable** — a Hugging Face repository, a
GitHub release asset, Zenodo. It must be fetchable over `https` without a login,
and it must stay there: the pack is the evidence for the row.

**4. Open a pull request adding `submissions/<your-method>.json`.**

```json
{
  "method": "MyNav",
  "code": "https://github.com/you/mynav",
  "weights": "https://huggingface.co/you/mynav",
  "torch": "2.5.0",
  "python": "3.11",
  "provenance": "evidence-pack",
  "evidence": {
    "url": "https://huggingface.co/you/mynav/resolve/main/evidence-pack.zip",
    "sha256": "<the sha256sum from step 2>"
  }
}
```

`method`, `code`, `provenance` and `evidence` are required; `weights`, `torch` and
`python` are shown beside your name when present. **A row of this kind must not
carry a `metrics` block** — CI rejects it, because the point of the pack is that
the scores are not self-reported.

**5. Wait for CI, then a maintainer merges.** The `submission` workflow fetches
the pack, proves the digest, runs the packaged verifier and prints the row you
earned in the job summary. When it is green there is nothing left for a
maintainer to re-run. On merge, the `leaderboard` workflow regenerates
`docs/data/leaderboard.json` and the published page follows.

## What CI refuses

| Check | Why it exists |
| :--- | :--- |
| `evidence.url` is `https` and under 4 GiB | nothing else is fetched |
| the pack's sha256 equals the declared one | the row belongs to *that* pack |
| `insight-bench verify` accepts it | the pack's own digests are intact |
| `evidence-manifest.json` sits at the archive root | zipping from the parent buries it |
| the run is `insight-bench-v1@1.0.0` | another coordinate is another benchmark |
| exactly 1,097 episodes, no duplicates, every id in the split | no partial and no invented runs |
| the submission declares no metrics | the scores come from the pack |

## How the numbers are computed

The published convention, reproduced exactly:

* a **per-type cell** is the success rate over the episodes of that type that were
  measurable — an episode that failed to run leaves that cell's denominator;
* **Avg.** is the success rate over the whole split, all 1,097 episodes.

The two agree unless something failed to run. An episode that renders nothing useful
is still scored — the first-frame check writes `first_frame_warning` into its trace
and the run carries on — so a failure here means the episode did not run at all,
which is rarer than it used to be.

Because a dropped episode leaves a denominator, **at most 5 of the 1,097 may fail to
run** before the submission is refused, and the ones that did are listed in the job
summary. Otherwise dropping the hard episodes becomes a way to buy a cell.

Rows marked `published` are numbers reported in the paper and are not recomputed
here.

## What this leaderboard is not

It is **self-reported**. The pinned episode digest proves you ran the published
split unmodified, and every episode's trace is in the pack for spot-checking, but
nothing here proves a model never saw those scenes during training. A controlled
re-run in a hosted environment is a separate service, and it does not exist yet.
