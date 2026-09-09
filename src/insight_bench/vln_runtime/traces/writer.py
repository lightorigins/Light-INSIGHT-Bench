"""Trace persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import RolloutTrace


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return out_path


def write_rollout_trace(path: str | Path, trace: RolloutTrace) -> Path:
    return write_json(path, trace.to_dict())
