"""Dependency gate: prove insight_bench is independent of the operator monorepo.

Scans the source trees and the packaged readme (and, when present, built
distribution archives) for forbidden tokens: the operator harness package name
and internal infrastructure identifiers. Any hit fails the gate.

Usage:
    python tools/check_independence.py            # scan the shipped trees + readme
    python tools/check_independence.py --dist     # additionally scan dist/*
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import zipfile
from pathlib import Path

FORBIDDEN_TOKENS = (
    # Operator repositories external users must never need to import/clone.
    "vln_isaac_benchmark",
    "vln-isaac-benchmark",
    "vln_arena",
    "vln-arena",
    # The names this distribution was published under before it became
    # INSIGHT-Bench. Nothing may reach a reader under a name they cannot look
    # up, and a rename this wide leaves stragglers unless something watches.
    "openvln",
    "objnav-suite",
    "objnav_suite",
    # Internal infrastructure identifiers.
    "vepfs",
    "volces",
    "lrcorp",
    "cr-lr-",
)

_ALLOWED_FILES = {
    Path("tools/check_independence.py"),  # this file names the tokens
    # A frozen record of the schema an older consumer held, published under the
    # old name. Rewriting it would defeat the compatibility check it exists for.
    Path("tests/gate/legacy_v1_run_result.schema.json"),
    # The one place a retired name belongs: this file proves each one is refused
    # rather than quietly resolved, so it has to spell them out.
    Path("tests/gate/test_retired_coordinates_fail_closed.py"),
}


def scan_tree(root: Path) -> list[str]:
    hits: list[str] = []
    for path in sorted(root.rglob("*")):
        # The web types are here because a published page is as public as a
        # published wheel: the leaderboard site ships .html, .css and .js, and a
        # token in one of those reaches a reader exactly the same way.
        if path.suffix not in {
            ".py",
            ".json",
            ".md",
            ".toml",
            ".cfg",
            ".yaml",
            ".yml",
            ".sh",
            ".html",
            ".css",
            ".js",
            ".svg",
        }:
            continue
        if not path.is_file():
            continue
        try:
            relative = path.relative_to(Path.cwd())
        except ValueError:
            relative = path
        if relative in _ALLOWED_FILES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        for token in FORBIDDEN_TOKENS:
            if token in text:
                hits.append(f"{relative}: contains {token!r}")
    return hits


# ``readme = "README.md"`` in pyproject.toml makes that file the distribution's
# long_description, so every byte of it is copied into the wheel's METADATA and
# the sdist's PKG-INFO. A readme lives in none of the scanned trees, which used
# to leave it the one publishable surface the plain gate could not see -- an
# internal path written there reached a published artifact and was caught only
# by --dist, after a build.
PACKAGED_READMES = ("README.md",)


def scan_packaged_readme() -> list[str]:
    hits: list[str] = []
    for name in PACKAGED_READMES:
        path = Path(name)
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        for token in FORBIDDEN_TOKENS:
            if token in text:
                hits.append(f"{name}: contains {token!r}")
    return hits


def scan_dist(dist: Path) -> list[str]:
    hits: list[str] = []
    for archive in sorted(dist.glob("*.whl")) + sorted(dist.glob("*.tar.gz")):
        if archive.suffix == ".whl":
            with zipfile.ZipFile(archive) as bundle:
                members = {name: bundle.read(name) for name in bundle.namelist()}
        else:
            members = {}
            with tarfile.open(archive) as bundle:
                for member in bundle.getmembers():
                    if member.isfile():
                        extracted = bundle.extractfile(member)
                        if extracted is not None:
                            members[member.name] = extracted.read()
        allowed_names = {path.name for path in _ALLOWED_FILES}
        for name, payload in members.items():
            # The same allowlist as the tree scan, matched on the file name: a
            # member path inside an archive carries a version-stamped prefix, so
            # the relative paths the tree scan uses do not appear here.
            if Path(name).name in allowed_names:
                continue
            text = payload.decode("utf-8", errors="replace").lower()
            for token in FORBIDDEN_TOKENS:
                if token in text:
                    hits.append(f"{archive.name}:{name}: contains {token!r}")
    return hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", action="store_true", help="also scan dist/ archives")
    args = parser.parse_args(argv)

    hits: list[str] = []
    # Every directory the sdist ships. policies/ and scripts/ are the ones a
    # user actually reads and edits, and both were written by porting internal
    # code, which is exactly where an internal path gets carried across.
    # Not only what the distribution ships. The site under docs/ is served to
    # the public, guides/ is the reference the readme sends every reader to,
    # submissions/ is published beside it and .github/ is readable by anyone
    # with the repository -- so each is a way a forbidden name reaches a reader,
    # which is what this gate is for.
    for tree in (
        Path("src"),
        Path("policies"),
        Path("scripts"),
        Path("tests"),
        Path("tools"),
        Path("docs"),
        Path("guides"),
        Path("submissions"),
        Path(".github"),
    ):
        if tree.is_dir():
            hits.extend(scan_tree(tree))
    hits.extend(scan_packaged_readme())
    if args.dist:
        dist = Path("dist")
        if not dist.is_dir():
            print("dependency gate: --dist requested but dist/ does not exist", file=sys.stderr)
            return 2
        hits.extend(scan_dist(dist))

    if hits:
        print("dependency gate FAILED:", file=sys.stderr)
        for hit in hits:
            print(f"  {hit}", file=sys.stderr)
        return 1
    print("dependency gate passed: no forbidden tokens found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
