"""Waypoint-to-motion primitives shared by runners and scoring replay."""

from .pose import CameraPose, normalize_yaw

__all__ = ["CameraPose", "normalize_yaw"]
