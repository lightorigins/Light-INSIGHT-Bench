"""Path validation shared by registry and evidence handling."""

from __future__ import annotations

from pathlib import Path, PurePosixPath


class UnsafePathError(ValueError):
    """Raised when an untrusted relative path escapes its declared root."""


def validate_relative_path(value: str) -> str:
    """Validate and normalize a portable, root-relative POSIX path."""
    if not value or "\\" in value or "\x00" in value:
        raise UnsafePathError("path must be a non-empty relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise UnsafePathError(f"unsafe relative path: {value!r}")
    return path.as_posix()


def safe_join(root: Path, relative: str, *, reject_symlinks: bool = True) -> Path:
    """Resolve a validated relative path below *root* without following symlinks."""
    normalized = validate_relative_path(relative)
    root = root.resolve()
    candidate = root.joinpath(*PurePosixPath(normalized).parts)
    if not candidate.resolve(strict=False).is_relative_to(root):
        raise UnsafePathError(f"path escapes root: {relative!r}")
    if reject_symlinks:
        cursor = root
        for part in PurePosixPath(normalized).parts:
            cursor /= part
            if cursor.is_symlink():
                raise UnsafePathError(f"symlinks are not allowed: {relative!r}")
    return candidate


def require_regular_file(path: Path) -> None:
    """Reject missing, non-regular, and symlink inputs."""
    if path.is_symlink() or not path.is_file():
        raise UnsafePathError(f"expected a regular non-symlink file: {path.name}")
