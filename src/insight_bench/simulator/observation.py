"""Canonical observation and orientation contract for simulator backends.

The two highest-risk Isaac Sim 6.0.1 incompatibilities (compatibility audit,
2026-08) are normalized away at this boundary:

1. **Quaternion order.** Isaac Lab 2.3 uses WXYZ globally; Isaac Lab 3.x
   switched to XYZW. The SDK's canonical order is **WXYZ** — matching the
   episode schema's ``Pose3D.orientation_wxyz`` — and every backend must
   convert its line-native order here. Line-native quaternions must never
   leak past the backend. Pose readback (command a pose, read it back, check
   yaw error) is Gate G2 in :mod:`insight_bench.simulator.doctor`; its
   tolerance is :data:`G2_YAW_TOLERANCE_RAD`.
2. **Array types.** Camera outputs (rgb / depth / intrinsics) are
   ``torch.Tensor`` on the 5.1 line and ``warp`` arrays on the 6.0 line.
   ``Observation`` is a strict **numpy** contract; :func:`to_numpy` converts
   either source without importing torch or warp.
3. **Buffer reuse.** Both lines hand back views onto a render buffer that is
   overwritten on the next step, so a captured observation would silently
   change value after the fact. ``Observation`` therefore copies every array
   on construction and marks the copy read-only — see
   :func:`_capture_array`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from insight_bench.vln_runtime.motion.pose import CameraPose

CANONICAL_QUAT_ORDER = "wxyz"
"""SDK-wide quaternion element order; backends convert at the boundary."""

G2_YAW_TOLERANCE_RAD = 1e-4
"""Maximum |commanded - read back| yaw accepted by the G2 pose-readback gate."""

Quat = tuple[float, float, float, float]


def quat_xyzw_to_wxyz(quat: Any) -> Quat:
    """Reorder an Isaac Lab 3.x native XYZW quaternion into canonical WXYZ."""
    x, y, z, w = (float(part) for part in quat)
    return (w, x, y, z)


def quat_wxyz_to_xyzw(quat: Any) -> Quat:
    """Reorder a canonical WXYZ quaternion into Isaac Lab 3.x native XYZW."""
    w, x, y, z = (float(part) for part in quat)
    return (x, y, z, w)


def quat_wxyz_from_yaw(yaw: float) -> Quat:
    """Return the canonical WXYZ quaternion for a pure-yaw rotation."""
    half = yaw / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def yaw_from_quat_wxyz(quat: Any) -> float:
    """Extract yaw (radians, (-pi, pi]) from a canonical WXYZ quaternion."""
    w, x, y, z = (float(part) for part in quat)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def to_numpy(value: Any) -> np.ndarray:
    """Convert a backend array (numpy, torch.Tensor, warp array, proxy) to numpy.

    Duck-typed so this module never imports torch or warp: both expose a
    ``.numpy()`` method (CUDA torch tensors need ``.cpu()`` first, handled by
    the fallback). Anything else goes through ``np.asarray``.

    Isaac Lab 3.x is the one source that satisfies none of those. Its sensor
    buffers come back as ``isaaclab.utils.warp.ProxyArray``, which is neither a
    tensor nor a warp array; it publishes two views, ``.warp`` and ``.torch``.
    ``.torch`` is the one taken because it lands on the torch path this function
    already handles, and because it is the view Isaac Lab instruments: with
    ``WARN_ON_TORCH_QUATF_ACCESS=1`` set, reading it on a quaternion buffer
    emits the official "WXYZ in 2.x, XYZW in 3.x" migration warning. Going
    through ``.warp`` instead would take that detector away.

    The unwrap is one hop, not a loop, and it is not guarded: a ``.torch`` that
    raises is a broken proxy, and reporting that is better than falling through
    to ``np.asarray`` and returning a shapeless object array.
    """
    if isinstance(value, np.ndarray):
        return value
    proxy_torch_view = getattr(value, "torch", None)
    if proxy_torch_view is not None:
        value = proxy_torch_view
    numpy_method = getattr(value, "numpy", None)
    if callable(numpy_method):
        try:
            return np.asarray(numpy_method())
        except (RuntimeError, TypeError):
            cpu_method = getattr(value, "cpu", None)
            if callable(cpu_method):
                return np.asarray(cpu_method().numpy())
            raise
    return np.asarray(value)


def _capture_array(array: np.ndarray) -> np.ndarray:
    """Return a private, read-only copy of *array*.

    Backends hand out views onto a reused render buffer; without the copy a
    stored observation would change value when the simulator steps. The
    read-only flag makes the reverse mistake — mutating a captured frame in
    place — fail loudly instead of corrupting evidence.
    """
    captured = array.copy()
    captured.setflags(write=False)
    return captured


@dataclass(frozen=True)
class Observation:
    """One canonical camera observation.

    Strict numpy contract: uint8 ``(H, W, 3)`` rgb, optional float32
    ``(H, W)`` depth, optional float ``(3, 3)`` intrinsics, plus the
    canonical pose and WXYZ orientation. Backends must produce exactly these
    dtypes; silent value-range rescaling is a backend calibration concern
    and deliberately does not happen here.

    Construction takes a snapshot: arrays are copied and frozen read-only,
    and the orientation is normalized to a tuple, so an observation never
    aliases a caller's buffer in either direction.
    """

    rgb: np.ndarray
    pose: CameraPose
    orientation_wxyz: Quat
    depth: np.ndarray | None = None
    intrinsics: np.ndarray | None = None
    frame_id: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.rgb, np.ndarray):
            raise TypeError(f"rgb must be a numpy array, got {type(self.rgb).__name__}")
        if self.rgb.dtype != np.uint8 or self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError(
                f"rgb must be uint8 with shape (H, W, 3), got {self.rgb.dtype} {self.rgb.shape}"
            )
        if self.depth is not None:
            if not isinstance(self.depth, np.ndarray):
                raise TypeError(f"depth must be a numpy array, got {type(self.depth).__name__}")
            if self.depth.dtype != np.float32 or self.depth.ndim != 2:
                raise ValueError(
                    "depth must be float32 with shape (H, W), got "
                    f"{self.depth.dtype} {self.depth.shape}"
                )
            if self.depth.shape != self.rgb.shape[:2]:
                raise ValueError(
                    f"depth shape {self.depth.shape} does not match rgb {self.rgb.shape[:2]}"
                )
        if self.intrinsics is not None:
            if not isinstance(self.intrinsics, np.ndarray):
                raise TypeError(
                    f"intrinsics must be a numpy array, got {type(self.intrinsics).__name__}"
                )
            if self.intrinsics.shape != (3, 3) or not np.issubdtype(
                self.intrinsics.dtype, np.floating
            ):
                raise ValueError(
                    "intrinsics must be a floating (3, 3) matrix, got "
                    f"{self.intrinsics.dtype} {self.intrinsics.shape}"
                )
        if len(self.orientation_wxyz) != 4:
            raise ValueError("orientation_wxyz must have exactly four elements")

        # Snapshot every mutable field; the dataclass is frozen, so assign
        # through object.__setattr__ as dataclasses themselves do.
        object.__setattr__(self, "rgb", _capture_array(self.rgb))
        if self.depth is not None:
            object.__setattr__(self, "depth", _capture_array(self.depth))
        if self.intrinsics is not None:
            object.__setattr__(self, "intrinsics", _capture_array(self.intrinsics))
        object.__setattr__(
            self, "orientation_wxyz", tuple(float(part) for part in self.orientation_wxyz)
        )


def make_observation(
    rgb: Any,
    *,
    pose: CameraPose,
    orientation_wxyz: Any,
    depth: Any | None = None,
    intrinsics: Any | None = None,
    frame_id: int = 0,
) -> Observation:
    """Build a canonical :class:`Observation` from backend-native arrays.

    Arrays are converted with :func:`to_numpy`; dtypes are then validated
    strictly (never rescaled) by ``Observation``. The quaternion must already
    be canonical WXYZ — converting the line-native order is the backend's
    explicit responsibility via :func:`quat_xyzw_to_wxyz`.
    """
    return Observation(
        rgb=to_numpy(rgb),
        pose=pose,
        orientation_wxyz=tuple(float(part) for part in orientation_wxyz),  # type: ignore[arg-type]
        depth=None if depth is None else to_numpy(depth),
        intrinsics=None if intrinsics is None else to_numpy(intrinsics),
        frame_id=frame_id,
    )
