"""Simulator-neutral camera pose primitives.

Extracted verbatim from the operator harness's camera-walk backend so that the
pure motion/scoring layers no longer import a module that also hosts simulator
code. The yaw convention is radians, counter-clockwise, normalized to
``(-pi, pi]`` by ``normalize_yaw``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CameraPose:
    x: float
    y: float
    z: float
    yaw: float

    def to_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "yaw": self.yaw,
        }


def normalize_yaw(yaw: float) -> float:
    return math.atan2(math.sin(yaw), math.cos(yaw))
