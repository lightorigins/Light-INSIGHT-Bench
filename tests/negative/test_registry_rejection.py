from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from insight_bench.contracts import RegistryEntry
from insight_bench.registry import Registry, RegistryError, builtin_registry_root


@pytest.mark.negative
def test_registry_contract_rejects_traversal_and_absolute_paths() -> None:
    base = {
        "benchmark_id": "bench-tiny",
        "version": "1.0.0",
        "sha256": "0" * 64,
    }
    for path in ("../manifest.json", "/tmp/manifest.json", r"C:\\manifest.json"):
        with pytest.raises((ValidationError, ValueError)):
            RegistryEntry.model_validate({**base, "manifest_path": path})


@pytest.mark.negative
def test_registry_detects_manifest_tampering(tmp_path: Path) -> None:
    copied = tmp_path / "registry"
    shutil.copytree(builtin_registry_root(), copied)
    manifest = copied / "insight-bench-v1" / "1.0.0" / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["title"] = "tampered"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RegistryError, match="manifest digest mismatch"):
        Registry.load(copied)
