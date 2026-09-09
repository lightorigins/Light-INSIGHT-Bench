"""Rebuild docs/data/leaderboard.json from submissions/.

The leaderboard file is generated, never hand-edited. Every row traces to one
file in submissions/: a published paper row, or a row computed from a verified
evidence pack.

    python3 tools/build_leaderboard.py --check   # CI: fail if the file is stale
    python3 tools/build_leaderboard.py --write   # regenerate it

A verified row already in the file is reused when its evidence digest still
matches, so a rebuild does not re-download packs that have not changed.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

# Below the sys.path line above, which is what makes it importable: this tool
# and verify_submission.py are siblings run as scripts, not a package.
from verify_submission import (
    EPISODES,
    INSTRUCTION_TYPES,
    SCENE_CLASSES,
    Rejected,
    load,
    row_for,
)

ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS = ROOT / "submissions"
TARGET = ROOT / "docs" / "data" / "leaderboard.json"

HEADER = {
    "benchmark": "INSIGHT-Bench",
    "version": "v1",
    "coordinate": "insight-bench-v1@1.0.0",
    "episodes": EPISODES,
    "scenes": 210,
    "metric": "success rate (%)",
    "reference": "LightNav-0",
    "note": (
        "Published measurement. Success rate over the full 1,097-episode split, grouped by "
        "instruction type and by scene class."
    ),
}


def cached(previous: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Verified rows from a previous build, keyed by the pack digest they were computed from."""
    return {
        r["evidence_sha256"]: r
        for r in previous.get("rows", [])
        if r.get("provenance") == "evidence-pack" and r.get("evidence_sha256")
    }


def build(previous: dict[str, Any], *, refresh: bool) -> dict[str, Any]:
    reuse = {} if refresh else cached(previous)
    rows = []
    for path in sorted(SUBMISSIONS.glob("*.json")):
        sub = load(path)
        digest = (sub.get("evidence") or {}).get("sha256")
        if digest and digest in reuse:
            rows.append(reuse[digest])
            continue
        with tempfile.TemporaryDirectory() as tmp:
            rows.append(row_for(sub, Path(tmp)))

    rows.sort(key=lambda r: (-r["avg"], r["method"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i

    doc = dict(HEADER)
    doc["updated"] = previous.get("updated", "")
    doc["instruction_types"] = list(INSTRUCTION_TYPES)
    doc["scene_classes"] = list(SCENE_CLASSES)
    doc["rows"] = rows
    if "reproduction" in previous:
        doc["reproduction"] = previous["reproduction"]
    return doc


def render(doc: dict[str, Any]) -> str:
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="fail if the committed file is stale")
    mode.add_argument("--write", action="store_true", help="regenerate the file in place")
    ap.add_argument(
        "--refresh",
        action="store_true",
        help="re-verify every evidence pack instead of reusing rows with an unchanged digest",
    )
    ap.add_argument("--updated", help="date stamp to record, e.g. 2026-09-08")
    args = ap.parse_args(argv)

    previous = json.loads(TARGET.read_text(encoding="utf-8")) if TARGET.is_file() else {}
    try:
        doc = build(previous, refresh=args.refresh)
    except Rejected as exc:
        print(f"REJECTED {exc}", file=sys.stderr)
        return 1
    if args.updated:
        doc["updated"] = args.updated

    text = render(doc)
    if args.check:
        current = TARGET.read_text(encoding="utf-8") if TARGET.is_file() else ""
        if current != text:
            print(
                f"{TARGET.relative_to(ROOT)} is stale: run "
                "'python3 tools/build_leaderboard.py --write'",
                file=sys.stderr,
            )
            return 1
        print(f"{TARGET.relative_to(ROOT)} is up to date ({len(doc['rows'])} rows)")
        return 0

    TARGET.write_text(text, encoding="utf-8")
    print(f"wrote {TARGET.relative_to(ROOT)} ({len(doc['rows'])} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
