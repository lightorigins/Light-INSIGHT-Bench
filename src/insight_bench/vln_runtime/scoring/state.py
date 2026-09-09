"""Offline episode evaluation state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from insight_bench.vln_runtime.episodes import EpisodeSpec
from insight_bench.vln_runtime.traces import RolloutTrace


@dataclass
class EpisodeEvalState:
    """Immutable episode artifacts plus a small memo cache for score terms."""

    episode: EpisodeSpec
    trace: RolloutTrace
    final_measures: dict[str, Any] = field(default_factory=dict)
    task_config: Any | None = None
    cache: dict[tuple[str, tuple[tuple[str, Any], ...]], Any] = field(default_factory=dict)
