"""Best-effort secret and local path redaction for shareable evidence."""

from __future__ import annotations

import re
from typing import Any

_SECRET_KEY = re.compile(
    r"(?:^|[_-])(api[_-]?key|authorization|credential|password|private[_-]?key|secret|token)(?:$|[_-])",
    re.IGNORECASE,
)
_SECRET_TEXT = re.compile(
    r"(?i)\b(api[_-]?key|authorization|password|secret|token)\b(\s*[=:]\s*)([^\s,;]+)"
)
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+\-/]+=*")
_UNIX_PATH = re.compile(r"(?<![A-Za-z0-9:])/(?:Users|home|private|tmp|var|opt)/[^\s\"'<>]*")
_WINDOWS_PATH = re.compile(r"(?i)(?<![A-Za-z0-9])[A-Z]:\\[^\r\n\"'<>]*")
_PATH_KEY = re.compile(r"(?:^|[_-])(cwd|directory|path|root)(?:$|[_-])", re.IGNORECASE)

REDACTED_SECRET = "<redacted-secret>"
REDACTED_PATH = "<redacted-path>"


def redact_text(value: str) -> str:
    """Remove common inline credentials and host-specific absolute paths."""
    value = _BEARER.sub(REDACTED_SECRET, value)
    value = _SECRET_TEXT.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{REDACTED_SECRET}", value
    )
    value = _UNIX_PATH.sub(REDACTED_PATH, value)
    return _WINDOWS_PATH.sub(REDACTED_PATH, value)


def redact_data(value: Any, *, key: str | None = None, redactions: set[str] | None = None) -> Any:
    """Recursively redact sensitive JSON fields while preserving its shape."""
    found = redactions if redactions is not None else set()
    if key is not None and _SECRET_KEY.search(key):
        found.add("secret-key")
        return REDACTED_SECRET
    if isinstance(value, dict):
        return {
            str(item_key): redact_data(item_value, key=str(item_key), redactions=found)
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [redact_data(item, redactions=found) for item in value]
    if isinstance(value, tuple):
        return [redact_data(item, redactions=found) for item in value]
    if isinstance(value, str):
        if key is not None and _PATH_KEY.search(key) and _looks_absolute(value):
            found.add("absolute-path")
            return REDACTED_PATH
        redacted = redact_text(value)
        if redacted != value:
            if REDACTED_PATH in redacted:
                found.add("absolute-path")
            if REDACTED_SECRET in redacted:
                found.add("inline-secret")
        return redacted
    return value


def _looks_absolute(value: str) -> bool:
    return value.startswith("/") or bool(re.match(r"^[A-Za-z]:[\\/]", value))
