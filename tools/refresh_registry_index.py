"""Recompute the manifest SHA-256 pins in a registry's ``index.json``.

Editing a manifest changes its bytes, and ``Registry.load`` refuses an index
whose pin no longer matches -- which takes every command down, not just
``validate``. This recomputes every pin from the manifest files on disk and
rewrites the index in the same canonical JSON the rest of the registry uses, so
the only difference is the digests that actually changed.

It never touches ``dataset.sha256``: that pins the published episode bytes, not
a file in this repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from insight_bench._json import load_json, sha256_file, stable_json_bytes
from insight_bench._paths import safe_join

DEFAULT_REGISTRY = Path(__file__).parents[1] / "src" / "insight_bench" / "registry"


def refreshed_index_bytes(root: Path) -> bytes:
    """Return the canonical ``index.json`` bytes with every pin recomputed."""
    index = load_json(root / "index.json")
    for entry in index["entries"]:
        entry["sha256"] = sha256_file(safe_join(root, entry["manifest_path"]))
    return stable_json_bytes(index)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help="registry directory holding index.json; defaults to this checkout's",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail instead of writing when a pin is stale",
    )
    args = parser.parse_args(argv)
    index_path = args.registry / "index.json"
    refreshed = refreshed_index_bytes(args.registry)
    if index_path.read_bytes() == refreshed:
        return 0
    if args.check:
        print(f"stale: {index_path}")
        return 1
    index_path.write_bytes(refreshed)
    print(f"updated: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
