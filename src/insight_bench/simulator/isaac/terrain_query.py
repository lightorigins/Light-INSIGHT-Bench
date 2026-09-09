"""Warp-based scene geometry queries for the camera-walk backend.

The camera-walk backend teleports a free camera to kinematically integrated
poses. On its own that cannot follow stairs or ramps (``z`` is frozen) and
flies through walls. This module adds a thin geometry layer that builds a Warp
mesh of the scene once per episode and answers two raycasts against it:

* :meth:`TerrainQuery.ground_height_at` -- downward ray to snap ``z`` to the
  floor surface (terrain following: stairs and ramps).
* :meth:`TerrainQuery.path_blocked` -- horizontal ray to stop forward motion at
  walls and obstacles.

Both raycasts run against the scene's *visual* mesh via
``isaaclab.utils.warp.raycast_mesh``, so no PhysX colliders are required.

Every Isaac / Warp / USD import is inside a function body: importing this
module must stay free on a machine with no simulator, which is what keeps the
public wheel Isaac-free.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

_DOWN = (0.0, 0.0, -1.0)


def _triangulate(face_indices: np.ndarray, face_counts: np.ndarray) -> np.ndarray:
    """Fan-triangulate a USD polygon mesh into ``(M, 3)`` triangle indices."""
    # Fast path: already triangulated (the common HM3D / Matterport case).
    if face_counts.size and bool(np.all(face_counts == 3)):
        return face_indices.reshape(-1, 3)

    tris: list[tuple[int, int, int]] = []
    cursor = 0
    for raw_count in face_counts.tolist():
        count = int(raw_count)
        face = face_indices[cursor : cursor + count]
        cursor += count
        for k in range(1, count - 1):
            tris.append((int(face[0]), int(face[k]), int(face[k + 1])))
    if not tris:
        return np.empty((0, 3), dtype=np.int64)
    return np.asarray(tris, dtype=np.int64)


def _collect_triangle_mesh(
    terrain_prim_path: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Gather and merge every ``UsdGeom.Mesh`` under *terrain_prim_path*.

    Scene USDs contain many sub-meshes, so (unlike Isaac Lab's ``RayCaster``,
    which reads only the first mesh prim) the whole subtree is traversed, each
    mesh's world transform is baked into its vertices, and the result is
    triangulated and concatenated into a single ``(points, triangles)`` pair.
    """
    import isaaclab.sim as sim_utils
    import omni.usd
    from pxr import UsdGeom

    try:
        mesh_prims = sim_utils.get_all_matching_child_prims(
            terrain_prim_path, lambda prim: prim.GetTypeName() == "Mesh"
        )
    except ValueError:
        return None, None

    all_points: list[np.ndarray] = []
    all_tris: list[np.ndarray] = []
    offset = 0
    for prim in mesh_prims:
        mesh = UsdGeom.Mesh(prim)
        raw_points = mesh.GetPointsAttr().Get()
        if raw_points is None or len(raw_points) == 0:
            continue
        points = np.asarray(raw_points, dtype=np.float64)
        # Bake the world transform into the vertices (mirrors RayCaster).
        transform = np.array(omni.usd.get_world_transform_matrix(prim)).T
        points = np.matmul(points, transform[:3, :3].T)
        points += transform[:3, 3]

        face_indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        face_counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        tris = _triangulate(face_indices, face_counts)
        if tris.size == 0:
            continue

        all_points.append(points.astype(np.float32))
        all_tris.append(tris + offset)
        offset += len(points)

    if not all_points:
        return None, None
    return np.concatenate(all_points, axis=0), np.concatenate(all_tris, axis=0)


class TerrainQuery:
    """Owns a Warp mesh of the scene and answers ground / collision raycasts."""

    def __init__(self, mesh: Any, device: str) -> None:
        self._mesh = mesh
        self._device = device

    @classmethod
    def build_from_stage(cls, terrain_prim_path: str, device: str) -> TerrainQuery | None:
        """Build a query from the scene geometry, or ``None`` if no mesh is found.

        Returning ``None`` lets callers fall back to flat kinematic behavior
        (e.g. plane terrains, which have no mesh geometry).
        """
        points, indices = _collect_triangle_mesh(terrain_prim_path)
        if points is None or indices is None or len(points) == 0 or len(indices) == 0:
            return None

        from isaaclab.utils.warp import convert_to_warp_mesh

        mesh = convert_to_warp_mesh(points, indices, device=device)
        return cls(mesh, device)

    def _raycast_distance(
        self,
        start: tuple[float, float, float],
        direction: tuple[float, float, float],
        max_dist: float,
    ) -> float:
        """Hit distance along *direction* from *start* (``inf`` on a miss)."""
        import torch
        from isaaclab.utils.warp import raycast_mesh

        # Isaac Lab's ``raycast_mesh`` reshapes the per-ray distance back to
        # ``ray_starts.shape[:2]`` (despite documenting an ``(N, 3)`` input), so it
        # requires a 3-D ``(N, R, 3)`` tensor. Pass a single ray as ``(1, 1, 3)``; a
        # 2-D ``(1, 3)`` makes it ``view(1, 3)`` a 1-element distance tensor and
        # raise "shape '[1, 3]' is invalid for input of size 1".
        ray_starts = torch.tensor([[list(start)]], dtype=torch.float32, device=self._device)
        ray_dirs = torch.tensor([[list(direction)]], dtype=torch.float32, device=self._device)
        _, dist, _, _ = raycast_mesh(
            ray_starts, ray_dirs, mesh=self._mesh, max_dist=max_dist, return_distance=True
        )
        return float(dist.reshape(-1)[0].item())

    def ground_height_at(
        self,
        x: float,
        y: float,
        *,
        z_ref: float,
        max_step_up_m: float,
        max_step_down_m: float,
        margin_m: float = 0.1,
    ) -> float | None:
        """Floor height at ``(x, y)`` within a window around *z_ref*.

        The downward ray starts above the highest reachable step and only the
        ``[z_ref - max_step_down_m, z_ref + max_step_up_m]`` band is accepted.
        Windowing around the previous floor height is what makes stairs robust:
        it follows the current floor and rejects hits on a ceiling or an
        unrelated upper/lower floor. Returns ``None`` on a miss or an
        out-of-window hit.
        """
        start_z = z_ref + max_step_up_m + margin_m
        max_dist = (max_step_up_m + margin_m) + (max_step_down_m + margin_m)
        dist = self._raycast_distance((x, y, start_z), _DOWN, max_dist)
        if not math.isfinite(dist):
            return None
        hit_z = start_z - dist
        if hit_z < z_ref - max_step_down_m - 1e-3 or hit_z > z_ref + max_step_up_m + 1e-3:
            return None
        return hit_z

    def path_blocked(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        *,
        probe_z: float,
        radius_m: float,
    ) -> float:
        """Allowed fraction ``[0, 1]`` of the horizontal move before a wall.

        Casts a ray at body/eye height (*probe_z*) from the old position toward
        the new one. ``1.0`` means the path is clear; a smaller value clamps the
        move so the camera stops *radius_m* short of the wall.
        """
        dx = x1 - x0
        dy = y1 - y0
        move_len = math.hypot(dx, dy)
        if move_len < 1e-6:
            return 1.0
        direction = (dx / move_len, dy / move_len, 0.0)
        dist = self._raycast_distance((x0, y0, probe_z), direction, move_len + radius_m)
        if not math.isfinite(dist):
            return 1.0
        allowed = max(0.0, dist - radius_m)
        return min(1.0, allowed / move_len)
