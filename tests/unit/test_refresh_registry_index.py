"""The registry-index refresher, exercised the way the publishing procedure uses it.

Editing a manifest and forgetting its index pin does not degrade the registry,
it disables it: ``Registry.load`` raises, so ``catalog``, ``validate`` and
``run`` all fail. The tool exists so that step cannot be done by hand, and these
tests pin the two properties that make it safe to run -- it changes only the
digests, and it leaves ``dataset.sha256`` alone.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from insight_bench.registry import Registry, RegistryError, builtin_registry_root

REPOSITORY_ROOT = Path(__file__).parents[2]
REFRESH_SCRIPT = REPOSITORY_ROOT / "tools" / "refresh_registry_index.py"

COORDINATE = "insight-bench-v1@1.0.0"
MANIFEST_RELATIVE = "insight-bench-v1/1.0.0/manifest.json"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(REFRESH_SCRIPT), *args],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )


@pytest.fixture
def registry_copy(tmp_path: Path) -> Path:
    root = tmp_path / "registry"
    shutil.copytree(builtin_registry_root(), root)
    return root


def test_the_checked_in_index_pins_the_checked_in_manifests() -> None:
    result = _run("--check")
    assert result.returncode == 0, result.stdout + result.stderr


def test_an_edited_manifest_breaks_the_registry_until_the_index_is_refreshed(
    registry_copy: Path,
) -> None:
    manifest_path = registry_copy / MANIFEST_RELATIVE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_digest = manifest["dataset"]["sha256"]
    manifest["dataset"]["notice"] = (
        manifest["dataset"]["notice"] + " Download: https://example.invalid/x"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RegistryError, match="manifest digest mismatch"):
        Registry.load(registry_copy)
    stale = _run("--registry", str(registry_copy), "--check")
    assert stale.returncode == 1
    assert "stale" in stale.stdout

    refreshed = _run("--registry", str(registry_copy))
    assert refreshed.returncode == 0, refreshed.stdout + refreshed.stderr

    reloaded = Registry.load(registry_copy)
    # The published episode bytes are pinned by dataset.sha256 and must survive
    # untouched: refreshing the index is not a licence to re-pin the dataset.
    assert reloaded.get(COORDINATE).dataset.sha256 == dataset_digest
    assert _run("--registry", str(registry_copy), "--check").returncode == 0


def test_refreshing_an_untouched_registry_changes_no_bytes(registry_copy: Path) -> None:
    before = (registry_copy / "index.json").read_bytes()
    result = _run("--registry", str(registry_copy))
    assert result.returncode == 0, result.stdout + result.stderr
    assert (registry_copy / "index.json").read_bytes() == before
