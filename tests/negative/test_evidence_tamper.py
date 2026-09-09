from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from insight_bench.evidence import pack_evidence, verify_evidence
from support import write_run_result


def _pack(tmp_path: Path) -> Path:
    output = tmp_path / "pack"
    pack_evidence(write_run_result(tmp_path / "result.json"), output)
    return output


@pytest.mark.negative
def test_verify_detects_payload_tampering(tmp_path: Path) -> None:
    pack = _pack(tmp_path)
    (pack / "run-result.json").write_text("{}\n", encoding="utf-8")
    report = verify_evidence(pack)
    assert not report.valid
    assert any("SHA-256 mismatch" in error for error in report.errors)


@pytest.mark.negative
def test_verify_detects_unexpected_files(tmp_path: Path) -> None:
    pack = _pack(tmp_path)
    (pack / "injected.txt").write_text("unexpected", encoding="utf-8")
    report = verify_evidence(pack)
    assert not report.valid
    assert "unexpected artifact: injected.txt" in report.errors


@pytest.mark.negative
def test_verify_rejects_symlinks(tmp_path: Path) -> None:
    pack = _pack(tmp_path)
    target = tmp_path / "outside.txt"
    target.write_text("outside", encoding="utf-8")
    try:
        (pack / "linked.txt").symlink_to(target)
    except OSError:
        pytest.skip("symlinks are not supported by this test environment")
    report = verify_evidence(pack)
    assert not report.valid
    assert any("non-regular artifact" in error for error in report.errors)
