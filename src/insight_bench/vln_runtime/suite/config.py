"""Shared config objects for benchmark tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from insight_bench.vln_runtime.measures.manager import MeasureManagerCfg
from insight_bench.vln_runtime.scoring.manager import ScoreManagerCfg
from insight_bench.vln_runtime.terminations import NavigationTerminationsCfg, terminations_from_dict


@dataclass(frozen=True)
class TerrainConfig:
    """Simulator terrain source for a benchmark task."""

    kind: str = "plane"
    prim_path: str = "/World/GroundPlane"
    usd_path: str | None = None
    env_spacing_m: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> TerrainConfig:
        if not payload:
            return cls()
        return cls(
            kind=str(payload.get("kind", "plane")),
            prim_path=str(payload.get("prim_path", "/World/GroundPlane")),
            usd_path=None if payload.get("usd_path") is None else str(payload["usd_path"]),
            env_spacing_m=None
            if payload.get("env_spacing_m") is None
            else float(payload["env_spacing_m"]),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_isaaclab_terrain_kwargs(self) -> dict[str, Any]:
        if self.kind == "plane":
            return {
                "prim_path": self.prim_path,
                "terrain_type": "plane",
            }
        if self.kind != "usd":
            raise ValueError(
                f"unsupported terrain kind for IsaacLab TerrainImporterCfg: {self.kind}"
            )
        if not self.usd_path:
            raise ValueError("terrain.usd_path is required when terrain.kind='usd'")
        payload: dict[str, Any] = {
            "prim_path": self.prim_path,
            "terrain_type": "usd",
            "usd_path": self.usd_path,
            "num_envs": 1,
        }
        if self.env_spacing_m is not None:
            payload["env_spacing"] = self.env_spacing_m
        return payload


@dataclass(frozen=True)
class SensorConfig:
    """Sensor config entry owned by a benchmark task."""

    name: str
    kind: str
    width: int | None = None
    height: int | None = None
    prim_path: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SensorConfig:
        return cls(
            name=str(payload["name"]),
            kind=str(payload.get("kind", payload.get("type", "unknown"))),
            width=None if payload.get("width") is None else int(payload["width"]),
            height=None if payload.get("height") is None else int(payload["height"]),
            prim_path=None if payload.get("prim_path") is None else str(payload["prim_path"]),
            params=dict(payload.get("params") or {}),
        )


@dataclass(frozen=True)
class BenchmarkTaskConfig:
    """One task-level benchmark config.

    Policy/server options stay outside this object. Everything simulator-task
    specific belongs here: terrain, sensors, evidence measures, scoring, and
    task assets.
    """

    name: str
    backend: str
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    sensors: tuple[SensorConfig, ...] = field(default_factory=tuple)
    measures: object = field(default_factory=MeasureManagerCfg.default_navigation)
    terminations: object = field(default_factory=NavigationTerminationsCfg)
    scoring: object = field(default_factory=ScoreManagerCfg.default_navigation)
    assets: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> BenchmarkTaskConfig:
        if not payload:
            return cls(name="insight_bench", backend="camera_walk")
        return cls(
            name=str(payload.get("name", "insight_bench")),
            backend=str(payload.get("backend", "camera_walk")),
            terrain=TerrainConfig.from_dict(payload.get("terrain")),
            sensors=tuple(SensorConfig.from_dict(item) for item in payload.get("sensors", [])),
            measures=MeasureManagerCfg.from_dict(payload.get("measures")),
            terminations=terminations_from_dict(payload.get("terminations")),
            scoring=ScoreManagerCfg.from_dict(payload.get("scoring")),
            assets=dict(payload.get("assets") or {}),
            metadata=dict(payload.get("metadata") or {}),
        )
