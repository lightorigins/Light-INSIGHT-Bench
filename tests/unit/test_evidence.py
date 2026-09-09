from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from insight_bench._json import load_json
from insight_bench.evidence import (
    inspect_evidence,
    pack_evidence,
    verify_evidence,
)
from support import write_run_result


def test_pack_inspect_verify_and_redact(tmp_path: Path) -> None:
    result = write_run_result(tmp_path / "result.json")
    supporting = tmp_path / "private-name.json"
    supporting.write_text(
        json.dumps(
            {
                "api_token": "do-not-share",
                "cwd": "/Users/example/private/run",
                "note": "password=hunter2",
            }
        ),
        encoding="utf-8",
    )
    manifest = pack_evidence(result, tmp_path / "pack", includes=[supporting])
    assert inspect_evidence(tmp_path / "pack") == manifest
    report = verify_evidence(tmp_path / "pack")
    assert report.valid, report.errors
    assert manifest.redactions == ("absolute-path", "inline-secret", "secret-key")
    artifact = load_json(tmp_path / "pack" / "artifacts" / "artifact-001.json")
    assert artifact["api_token"] == "<redacted-secret>"
    assert artifact["cwd"] == "<redacted-path>"
    assert "hunter2" not in artifact["note"]
    assert "private-name" not in (tmp_path / "pack" / "evidence-manifest.json").read_text()


def test_the_pack_carries_the_run_it_was_built_from(tmp_path: Path) -> None:
    """The pack is evidence *for a run*, so it must name that run and no other.

    Previously implicit in the round trip above. It is worth stating now that the
    run result is written by a test rather than produced by a run in the same
    call: nothing else would notice a pack that packed a different file.
    """
    result = write_run_result(tmp_path / "result.json")
    manifest = pack_evidence(result, tmp_path / "pack")
    written = load_json(result)
    assert manifest.run_id == written["run_id"]
    assert manifest.run_fingerprint == written["run_fingerprint"]
    assert [artifact.path for artifact in manifest.artifacts] == ["run-result.json"]
    assert load_json(tmp_path / "pack" / "run-result.json") == written


def test_repeated_pack_has_identical_manifest(tmp_path: Path) -> None:
    result = write_run_result(tmp_path / "result.json")
    first = pack_evidence(result, tmp_path / "pack-a")
    second = pack_evidence(result, tmp_path / "pack-b")
    assert first == second


def _archive_pack(pack: Path, archive: Path) -> Path:
    """Build the upload archive the way the evaluation scripts build it."""
    import zipfile

    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(pack.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(pack).as_posix())
    return archive


def test_verifying_the_upload_archive_matches_verifying_the_directory(tmp_path: Path) -> None:
    result = write_run_result(tmp_path / "result.json")
    pack_evidence(result, tmp_path / "pack")
    archive = _archive_pack(tmp_path / "pack", tmp_path / "evidence-pack.zip")

    from_directory = verify_evidence(tmp_path / "pack")
    from_archive = verify_evidence(archive)

    assert from_directory.valid, from_directory.errors
    assert from_archive.valid, from_archive.errors
    assert from_archive.pack_id == from_directory.pack_id
    assert from_archive.artifacts_checked == from_directory.artifacts_checked


def test_a_tampered_upload_archive_is_rejected(tmp_path: Path) -> None:
    result = write_run_result(tmp_path / "result.json")
    manifest = pack_evidence(result, tmp_path / "pack")
    victim = manifest.artifacts[0].path
    (tmp_path / "pack" / victim).write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    archive = _archive_pack(tmp_path / "pack", tmp_path / "evidence-pack.zip")

    report = verify_evidence(archive)

    assert not report.valid
    assert any(victim in error for error in report.errors)


def test_a_file_that_is_not_an_archive_is_reported_not_crashed(tmp_path: Path) -> None:
    not_an_archive = tmp_path / "evidence-pack.zip"
    not_an_archive.write_text("this is not a zip", encoding="utf-8")

    report = verify_evidence(not_an_archive)

    assert not report.valid
    assert report.errors
