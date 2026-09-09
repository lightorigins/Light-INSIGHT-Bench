"""Verify one leaderboard submission and print the row it earns.

A submission never carries its own scores. It points at an evidence pack; this
script fetches that pack, proves it is the one the submission names, hands it
to the packaged verifier, and computes the leaderboard row from the run result
inside it. Exit 0 means a maintainer can merge without re-running anything.

    python3 tools/verify_submission.py submissions/my-method.json

Published rows -- the paper baselines, which have no pack -- are validated for
shape only and pass through with their reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

BENCHMARK_ID = "insight-bench-v1"
BENCHMARK_VERSION = "1.0.0"
EPISODES = 1097
INSTRUCTION_TYPES = ("Base", "Direction", "Relation", "Extremum", "Ordinal")
SCENE_CLASSES = ("Apartment", "House", "Commercial", "Institution", "Outdoor")
MAX_PACK_BYTES = 4 * 1024**3
# An episode can legitimately fail to run -- one start pose in the split renders a
# near-uniform white frame and the blank-frame guard refuses it. Those episodes leave
# the denominator, which is the convention the published numbers use, so the number
# allowed to leave has to be small enough that dropping episodes cannot buy a score.
MAX_UNEVALUATED = 5
LABELS = Path(__file__).with_name("data") / "episode_labels.json"


class Rejected(Exception):
    """The submission does not earn a row."""


def labels() -> dict[str, list[str]]:
    return json.loads(LABELS.read_text(encoding="utf-8"))["labels"]


def load(path: Path) -> dict[str, Any]:
    """Read a submission file and reject anything the schema does not allow."""
    sub = json.loads(path.read_text(encoding="utf-8"))
    for key in ("method", "code", "provenance"):
        if not isinstance(sub.get(key), str) or not sub[key]:
            raise Rejected(f"{path.name}: '{key}' is required and must be a non-empty string")
    if not sub["code"].startswith("https://"):
        raise Rejected(f"{path.name}: 'code' must be an https URL")

    if sub["provenance"] == "published":
        if "evidence" in sub:
            raise Rejected(f"{path.name}: a published row carries no evidence pack")
        if not isinstance(sub.get("reference"), str):
            raise Rejected(f"{path.name}: a published row must name its 'reference'")
        m = sub.get("metrics")
        if not isinstance(m, dict):
            raise Rejected(f"{path.name}: a published row must carry 'metrics'")
        for block, names in (("instruction", INSTRUCTION_TYPES), ("scene", SCENE_CLASSES)):
            if sorted(m.get(block, {})) != sorted(names):
                raise Rejected(f"{path.name}: metrics.{block} must have exactly {list(names)}")
        if not isinstance(m.get("avg"), (int, float)):
            raise Rejected(f"{path.name}: metrics.avg must be a number")
    elif sub["provenance"] == "evidence-pack":
        if "metrics" in sub:
            raise Rejected(
                f"{path.name}: a verified row declares no metrics -- they are read from the pack"
            )
        ev = sub.get("evidence")
        if not isinstance(ev, dict):
            raise Rejected(f"{path.name}: 'evidence' is required for provenance 'evidence-pack'")
        if not str(ev.get("url", "")).startswith("https://"):
            raise Rejected(f"{path.name}: evidence.url must be an https URL")
        if not isinstance(ev.get("sha256"), str) or len(ev["sha256"]) != 64:
            raise Rejected(f"{path.name}: evidence.sha256 must be 64 hex characters")
    else:
        raise Rejected(f"{path.name}: provenance must be 'published' or 'evidence-pack'")
    return sub


def fetch(url: str, into: Path) -> str:
    """Download the pack, refusing anything larger than the cap. Returns its digest."""
    digest = hashlib.sha256()
    size = 0
    # The URL comes from a submission file, so it is untrusted input -- but it
    # reaches here only after load() has required an https scheme, and urllib
    # will not follow a redirect off http/https/ftp.
    with urllib.request.urlopen(url, timeout=120) as src, into.open("wb") as dst:
        while chunk := src.read(1 << 20):
            size += len(chunk)
            if size > MAX_PACK_BYTES:
                raise Rejected(f"evidence pack exceeds {MAX_PACK_BYTES // 1024**3} GiB")
            digest.update(chunk)
            dst.write(chunk)
    return digest.hexdigest()


def read_run_result(pack: Path) -> dict[str, Any]:
    with zipfile.ZipFile(pack) as z:
        names = set(z.namelist())
        for required in ("evidence-manifest.json", "run-result.json"):
            if required not in names:
                raise Rejected(
                    f"{required} must sit at the archive root -- zip from inside the pack directory"
                )
        return json.loads(z.read("run-result.json"))


def score(run: dict[str, Any], label_map: dict[str, list[str]]) -> dict[str, Any]:
    """Group per-episode success into the two taxonomies.

    The denominator is the episodes that actually ran, matching the convention the
    published numbers use: a per-type cell divides by the episodes of that type that
    were measurable, and Avg is the success rate over the whole split. The two agree
    unless something failed to run, and only a handful may fail before the submission
    is refused -- otherwise dropping the hard episodes becomes a way to buy a cell.
    """
    if run.get("benchmark_id") != BENCHMARK_ID or run.get("benchmark_version") != BENCHMARK_VERSION:
        raise Rejected(
            f"run result is {run.get('benchmark_id')}@{run.get('benchmark_version')}, "
            f"not {BENCHMARK_ID}@{BENCHMARK_VERSION}"
        )
    episodes = run.get("episodes") or []
    if len(episodes) != EPISODES:
        raise Rejected(f"run result has {len(episodes)} episodes, the split has {EPISODES}")

    seen: set[str] = set()
    won: dict[str, float] = {}
    total: dict[str, int] = {}
    unevaluated: list[str] = []
    evaluated = 0
    successes = 0.0
    for e in episodes:
        eid = e.get("episode_id")
        if eid not in label_map:
            raise Rejected(f"episode {eid!r} is not in the split")
        if eid in seen:
            raise Rejected(f"episode {eid!r} appears twice")
        seen.add(eid)
        if e.get("status") == "error" or not e.get("metrics"):
            unevaluated.append(eid)
            continue
        ok = float(e["metrics"].get("success", 0.0)) >= 0.5
        evaluated += 1
        successes += ok
        for key in label_map[eid]:
            total[key] = total.get(key, 0) + 1
            won[key] = won.get(key, 0.0) + ok

    if len(unevaluated) > MAX_UNEVALUATED:
        raise Rejected(
            f"{len(unevaluated)} episodes did not run, at most {MAX_UNEVALUATED} may "
            f"(first: {unevaluated[0]})"
        )
    empty = [k for k in INSTRUCTION_TYPES + SCENE_CLASSES if not total.get(k)]
    if empty:
        raise Rejected(f"no episode ran in {empty[0]}")

    pct = lambda k: round(100.0 * won.get(k, 0.0) / total[k], 1)  # noqa: E731
    return {
        "instruction": {k: pct(k) for k in INSTRUCTION_TYPES},
        "scene": {k: pct(k) for k in SCENE_CLASSES},
        "avg": round(100.0 * successes / EPISODES, 1),
        "evaluated": evaluated,
        "unevaluated": unevaluated,
        "spl": round(float((run.get("metrics") or {}).get("spl", 0.0)), 3),
        "ne_m": round(float((run.get("metrics") or {}).get("distance_to_goal", 0.0)), 2),
    }


def verified_metrics(sub: dict[str, Any], workdir: Path) -> dict[str, Any]:
    """Fetch, prove and read the pack a submission points at."""
    try:
        from insight_bench.evidence import verify_evidence
    except ImportError as exc:  # pragma: no cover - environment problem, not a submission problem
        raise SystemExit(
            "insight_bench is not importable: install the package first (pip install -e .)"
        ) from exc

    pack = workdir / "evidence-pack.zip"
    got = fetch(sub["evidence"]["url"], pack)
    if got != sub["evidence"]["sha256"]:
        raise Rejected(
            f"evidence pack digest is {got}, the submission declares {sub['evidence']['sha256']}"
        )

    report = verify_evidence(pack)
    if not getattr(report, "ok", False):
        raise Rejected(f"insight-bench verify rejected the pack: {report}")

    return score(read_run_result(pack), labels())


def row_for(sub: dict[str, Any], workdir: Path) -> dict[str, Any]:
    metrics = sub["metrics"] if sub["provenance"] == "published" else verified_metrics(sub, workdir)
    row = {
        "method": sub["method"],
        "code": sub["code"],
        "weights": sub.get("weights"),
        "torch": sub.get("torch"),
        "python": sub.get("python"),
        "provenance": sub["provenance"],
        "instruction": metrics["instruction"],
        "scene": metrics["scene"],
        "avg": metrics["avg"],
    }
    if sub["provenance"] == "published":
        row["reference"] = sub["reference"]
    else:
        row["evidence_sha256"] = sub["evidence"]["sha256"]
        for extra in ("evaluated", "spl", "ne_m"):
            if extra in metrics:
                row[extra] = metrics[extra]
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("submission", type=Path, nargs="+")
    args = ap.parse_args(argv)

    failed = False
    for path in args.submission:
        try:
            sub = load(path)
            with tempfile.TemporaryDirectory() as tmp:
                row = row_for(sub, Path(tmp))
        except Rejected as exc:
            print(f"REJECTED {path}: {exc}", file=sys.stderr)
            failed = True
            continue
        print(f"OK {path}  ({row['provenance']})")
        print(json.dumps(row, indent=2, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
