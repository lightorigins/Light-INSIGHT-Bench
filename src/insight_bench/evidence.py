"""Offline evidence packing, inspection, and integrity verification."""

from __future__ import annotations

import mimetypes
import os
import tempfile
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from insight_bench._json import (
    load_json,
    sha256_bytes,
    sha256_file,
    stable_json_bytes,
)
from insight_bench._paths import UnsafePathError, require_regular_file, safe_join
from insight_bench._redaction import redact_data, redact_text
from insight_bench.contracts import EvidenceArtifact, EvidenceManifest, RunResult

MANIFEST_NAME = "evidence-manifest.json"
_TEXT_SUFFIXES = {".csv", ".jsonl", ".log", ".md", ".txt", ".yaml", ".yml"}
_BINARY_SUFFIXES = {".jpeg", ".jpg", ".mp4", ".png", ".webp"}


class EvidenceError(ValueError):
    """Raised for unsafe or invalid evidence operations."""


@dataclass(frozen=True)
class VerificationReport:
    valid: bool
    pack_id: str | None
    errors: tuple[str, ...]
    artifacts_checked: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifacts_checked": self.artifacts_checked,
            "errors": list(self.errors),
            "pack_id": self.pack_id,
            "valid": self.valid,
        }


def pack_evidence(
    run_result: Path,
    output: Path,
    *,
    includes: Iterable[Path] = (),
) -> EvidenceManifest:
    """Create a deterministic local evidence directory without uploading data."""
    try:
        require_regular_file(run_result)
        if output.exists():
            raise EvidenceError("output already exists; choose a new evidence directory")
        output.parent.mkdir(parents=True, exist_ok=True)
        result = RunResult.model_validate(load_json(run_result))
        redactions: set[str] = set()
        sanitized_result = redact_data(result.model_dump(mode="json"), redactions=redactions)
        sanitized_result = RunResult.model_validate(sanitized_result)
        with tempfile.TemporaryDirectory(prefix=".insight-pack-", dir=output.parent) as temporary:
            stage = Path(temporary) / "pack"
            stage.mkdir()
            artifacts: list[EvidenceArtifact] = []
            result_bytes = stable_json_bytes(sanitized_result)
            result_relative = "run-result.json"
            result_path = safe_join(stage, result_relative)
            result_path.write_bytes(result_bytes)
            artifacts.append(
                _artifact(
                    result_relative,
                    result_bytes,
                    role="run-result",
                    sanitization="structured-redaction",
                )
            )
            for index, source in enumerate(includes, start=1):
                require_regular_file(source)
                suffix = source.suffix.lower()
                relative = f"artifacts/artifact-{index:03d}{suffix}"
                payload, sanitization = _sanitize_supporting(source, suffix, redactions)
                destination = safe_join(stage, relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)
                artifacts.append(
                    _artifact(
                        relative,
                        payload,
                        role="supporting",
                        sanitization=sanitization,
                    )
                )
            artifacts.sort(key=lambda item: item.path)
            pack_id = _pack_id(result.run_id, result.run_fingerprint, tuple(artifacts))
            manifest = EvidenceManifest(
                pack_id=pack_id,
                run_id=result.run_id,
                run_fingerprint=result.run_fingerprint,
                artifacts=tuple(artifacts),
                redactions=tuple(sorted(redactions)),
            )
            (stage / MANIFEST_NAME).write_bytes(stable_json_bytes(manifest))
            stage.rename(output)
        return manifest
    except (OSError, ValidationError, UnsafePathError) as exc:
        if isinstance(exc, EvidenceError):
            raise
        raise EvidenceError(redact_text(str(exc))) from exc


def inspect_evidence(root: Path) -> EvidenceManifest:
    """Parse the manifest and validate paths without reading artifact payloads."""
    try:
        manifest_path = root / MANIFEST_NAME
        require_regular_file(manifest_path)
        manifest = EvidenceManifest.model_validate(load_json(manifest_path))
        for artifact in manifest.artifacts:
            safe_join(root, artifact.path)
        return manifest
    except (OSError, ValidationError, UnsafePathError, ValueError) as exc:
        raise EvidenceError(redact_text(str(exc))) from exc


def verify_evidence(pack: Path) -> VerificationReport:
    """Detect altered, missing, extra, non-canonical, and unsafe pack content.

    Accepts either the pack directory or the archive of it that the evaluation
    scripts tell you to upload, so that verifying the artifact you submit is
    the same command as verifying the directory it came from.
    """
    if pack.is_file():
        return _verify_archive(pack)
    return _verify_directory(pack)


def _verify_archive(archive: Path) -> VerificationReport:
    """Verify a packed archive by checking the tree it unpacks to."""
    try:
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch) / "pack"
            with zipfile.ZipFile(archive) as handle:
                # extractall drops absolute paths and parent references, so a
                # crafted archive cannot write outside this scratch directory.
                handle.extractall(root)
            return _verify_directory(root)
    except (OSError, zipfile.BadZipFile) as exc:
        return VerificationReport(False, None, (redact_text(str(exc)),), 0)


def _verify_directory(root: Path) -> VerificationReport:
    errors: list[str] = []
    checked = 0
    try:
        manifest = inspect_evidence(root)
    except EvidenceError as exc:
        return VerificationReport(False, None, (str(exc),), 0)
    manifest_path = root / MANIFEST_NAME
    if manifest_path.read_bytes() != stable_json_bytes(manifest):
        errors.append("evidence manifest is not canonical stable JSON")
    expected_paths = {MANIFEST_NAME, *(artifact.path for artifact in manifest.artifacts)}
    actual_paths: set[str] = set()
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(directory)
        for name in list(directory_names):
            candidate = current / name
            if candidate.is_symlink():
                relative = candidate.relative_to(root).as_posix()
                errors.append(f"symlink directory is not allowed: {relative}")
                directory_names.remove(name)
        for name in file_names:
            candidate = current / name
            relative = candidate.relative_to(root).as_posix()
            actual_paths.add(relative)
            if candidate.is_symlink() or not candidate.is_file():
                errors.append(f"non-regular artifact is not allowed: {relative}")
    missing = sorted(expected_paths - actual_paths)
    extra = sorted(actual_paths - expected_paths)
    errors.extend(f"missing artifact: {path}" for path in missing)
    errors.extend(f"unexpected artifact: {path}" for path in extra)
    for artifact in manifest.artifacts:
        try:
            path = safe_join(root, artifact.path)
            require_regular_file(path)
        except (OSError, UnsafePathError) as exc:
            message = f"unsafe artifact {artifact.path}: {redact_text(str(exc))}"
            if message not in errors:
                errors.append(message)
            continue
        checked += 1
        size = path.stat().st_size
        if size != artifact.size_bytes:
            errors.append(
                f"size mismatch for {artifact.path}: expected {artifact.size_bytes}, got {size}"
            )
        digest = sha256_file(path)
        if digest != artifact.sha256:
            errors.append(f"SHA-256 mismatch for {artifact.path}")
    expected_pack_id = _pack_id(manifest.run_id, manifest.run_fingerprint, manifest.artifacts)
    if expected_pack_id != manifest.pack_id:
        errors.append("pack_id does not match the evidence manifest contents")
    return VerificationReport(not errors, manifest.pack_id, tuple(errors), checked)


def _artifact(
    relative: str,
    payload: bytes,
    *,
    role: str,
    sanitization: str,
) -> EvidenceArtifact:
    media_type = mimetypes.guess_type(relative)[0] or "application/octet-stream"
    return EvidenceArtifact.model_validate(
        {
            "path": relative,
            "sha256": sha256_bytes(payload),
            "size_bytes": len(payload),
            "media_type": media_type,
            "role": role,
            "sanitization": sanitization,
        }
    )


def _sanitize_supporting(source: Path, suffix: str, redactions: set[str]) -> tuple[bytes, str]:
    if suffix == ".json":
        value = redact_data(load_json(source), redactions=redactions)
        return stable_json_bytes(value), "structured-redaction"
    if suffix in _TEXT_SUFFIXES:
        original = source.read_text(encoding="utf-8")
        sanitized = redact_text(original)
        if sanitized != original:
            if "<redacted-path>" in sanitized:
                redactions.add("absolute-path")
            if "<redacted-secret>" in sanitized:
                redactions.add("inline-secret")
        return sanitized.encode(), "text-redaction"
    if suffix in _BINARY_SUFFIXES:
        return source.read_bytes(), "binary-none"
    raise EvidenceError(
        f"unsupported evidence type {suffix or '<none>'!r}; use JSON/text or PNG/JPEG/WebP/MP4"
    )


def _pack_id(run_id: str, run_fingerprint: str, artifacts: tuple[EvidenceArtifact, ...]) -> str:
    return sha256_bytes(
        stable_json_bytes(
            {
                "artifacts": artifacts,
                "run_fingerprint": run_fingerprint,
                "run_id": run_id,
            }
        )
    )
