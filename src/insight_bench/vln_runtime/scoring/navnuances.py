"""Continuous-space NavNuances evaluator.

Ports the five per-category protocols from the official graph-based evaluators
(navnuances/evaluation/evaluators) onto our continuous rollout traces:

  DC  first significant displacement vs the annotated reference "forward" vector;
      left/right by 2D cross sign, "around" when the angle exceeds 120 deg.
  LR  towards: final xy within 2 m of obj_center AND obj within the camera's 120°
      horizontal FOV (60° half-angle) at the end;
      past: obj_center's projection falls inside the start->final segment AND the
      final xy is within 3 m of the object.
  VM  final position within 3 m (3D, so floors separate) of the GT path end.
  RR  snap the final position to the nearest nav-graph node; "into" succeeds when
      that node is a valid end, "exit" when it is outside the start region.
  NU  final within 3 m of the closest GT-set path end AND location-nDTW to that GT
      path beats every negative path (margin criterion). Both the agent trace and
      every reference path are arc-length-resampled (0.25 m) before nDTW: the raw
      graph paths are 3-6 sparse nodes while the trace is dense, and comparing
      dense-vs-sparse made the margin depend on node placement luck instead of
      route shape (verified false negatives with the agent stopped 0.15 m from
      the GT end).

The NavNuances protocol scores wherever the predicted trajectory ends; no explicit
stop is required (``stopped`` is still reported for diagnostics).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from insight_bench.vln_runtime.episodes import EpisodeSpec
from insight_bench.vln_runtime.traces.schema import RolloutTrace

ERROR_MARGIN_M = 3.0
TURN_AROUND_DEG = 120.0


@dataclass(frozen=True)
class NavNuancesScore:
    episode_id: str
    capability: str
    subtype: str
    success: bool
    distance_to_goal: float
    path_length: float
    spl: float
    stopped: bool
    stop_step: int
    failure_reason: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "capability": self.capability,
            "subtype": self.subtype,
            "success": self.success,
            "distance_to_goal": self.distance_to_goal,
            "path_length": self.path_length,
            "spl": self.spl,
            "stopped": self.stopped,
            "stop_step": self.stop_step,
            "failure_reason": self.failure_reason,
            "detail": dict(self.detail),
        }


def _xy_dist(a, b) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _d3(a, b) -> float:
    return math.dist([float(v) for v in a[:3]], [float(v) for v in b[:3]])


def _path_length(points) -> float:
    return sum(_d3(p, q) for p, q in zip(points[:-1], points[1:]))


def _angle_deg(u, v) -> float:
    nu, nv = math.hypot(*u), math.hypot(*v)
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    c = max(-1.0, min(1.0, (u[0] * v[0] + u[1] * v[1]) / (nu * nv)))
    return math.degrees(math.acos(c))


# LR_towards also requires the target be within the camera's horizontal FOV at the end:
# 120° full FOV -> 60° half-angle. (The render HFOV is 86°; 120° is the spec value, so this
# is the lenient bound — tighten _LR_FOV_HALF_DEG to ~43 to match the rendered camera.)
_LR_FOV_HALF_DEG = 60.0
# LR_towards success distance: the camera's final xy must be within this many metres of the
# object centre (replaces the earlier "closer than the start" margin, which was too lenient).
_LR_SUCCESS_DIST_M = 2.0


def _yaw_heading_xy(wxyz) -> tuple[float, float]:
    """Horizontal heading unit vector (cos yaw, sin yaw) from a wxyz quaternion (z-yaw)."""
    w, x, y, z = (float(c) for c in wxyz)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return (math.cos(yaw), math.sin(yaw))


_RESAMPLE_STEP_M = 0.25


def _resample_xy(points, step: float = _RESAMPLE_STEP_M):
    """Arc-length resample a polyline to ~``step``-spaced xy points (endpoints kept).

    Stationary jitter (<1 cm) is dropped first. nDTW assumes comparably dense
    sequences; feeding it a 39-step trace against a 4-node graph path makes the
    score depend on where the nodes happen to sit rather than on the route.
    """
    pts: list[tuple[float, float]] = []
    for p in points:
        q = (float(p[0]), float(p[1]))
        if not pts or math.dist(q, pts[-1]) > 0.01:
            pts.append(q)
    if len(pts) <= 1:
        return pts
    out = [pts[0]]
    carry = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        seg = math.dist(a, b)
        t = step - carry
        while t <= seg:
            r = t / seg
            out.append((a[0] + r * (b[0] - a[0]), a[1] + r * (b[1] - a[1])))
            t += step
        carry = seg - (t - step)
    if math.dist(out[-1], pts[-1]) > 1e-6:
        out.append(pts[-1])
    return out


def _ndtw(pred, ref, threshold: float = ERROR_MARGIN_M) -> float:
    """Location-based nDTW (mirrors evaluator_NU's get_locations + ndtw)."""
    if not pred or not ref:
        return 0.0
    inf = float("inf")
    n, m = len(pred), len(ref)
    prev = [inf] * (m + 1)
    prev[0] = 0.0
    for i in range(1, n + 1):
        cur = [inf] * (m + 1)
        for j in range(1, m + 1):
            cost = _xy_dist(pred[i - 1], ref[j - 1])
            cur[j] = cost + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return math.exp(-prev[m] / (threshold * m))


def _eval_dc(nn: dict, start, positions) -> tuple[bool, dict]:
    ref = nn["heading_ref_vec"]
    min_move = float(nn.get("min_move_m", 0.5))
    v1 = None
    for p in positions:
        if _xy_dist(p, start) >= min_move:
            v1 = [p[0] - start[0], p[1] - start[1]]
            break
    if v1 is None:
        return False, {"moved": False}
    is_left = (ref[0] * v1[1] - ref[1] * v1[0]) > 0
    degrees = _angle_deg(ref, v1)
    is_around = degrees > TURN_AROUND_DEG
    subtype = nn["subtype"]
    if subtype == "left":
        ok = is_left and not is_around
    elif subtype == "right":
        ok = (not is_left) and not is_around
    else:  # around
        ok = is_around
    return ok, {"moved": True, "is_left": is_left, "angle_deg": round(degrees, 1)}


def _eval_lr(nn: dict, start, final, final_wxyz=None) -> tuple[bool, dict]:
    obj = nn["obj_center"]
    d_start, d_final = _xy_dist(start, obj), _xy_dist(final, obj)
    if nn["subtype"] == "towards":
        # Outdoor objnav episodes carry an explicit success radius (objects are
        # larger and the collection stop-back is 2.0 m); indoor specs omit the
        # key and keep the historical 2.0 m constant.
        success_dist = float(nn.get("success_dist_m", _LR_SUCCESS_DIST_M))
        near_obj = d_final <= success_dist
        detail = {
            "d_start": round(d_start, 2),
            "d_final": round(d_final, 2),
            "near_obj": bool(near_obj),
            "success_dist_m": success_dist,
        }
        # "towards" also requires the object be within the camera FOV at the end:
        # being within 2 m but ending up facing away does not count.
        if final_wxyz is not None:
            heading = _yaw_heading_xy(final_wxyz)
            dir_to_obj = (obj[0] - final[0], obj[1] - final[1])
            angle = _angle_deg(heading, dir_to_obj)
            detail["angle_to_obj_deg"] = round(angle, 1)
            detail["in_fov"] = bool(angle < _LR_FOV_HALF_DEG)
            return bool(near_obj and detail["in_fov"]), detail
        return bool(near_obj), detail
    # past: obj projects inside the start->final segment (mirrors is_projection_inside_segment)
    seg = [final[0] - start[0], final[1] - start[1]]
    seg_len2 = seg[0] ** 2 + seg[1] ** 2
    if seg_len2 < 1e-9:
        return False, {"d_final": round(d_final, 2), "proj_inside": False}
    t = ((obj[0] - start[0]) * seg[0] + (obj[1] - start[1]) * seg[1]) / seg_len2
    inside = 0.0 <= t <= 1.0
    return inside and d_final < ERROR_MARGIN_M, {
        "d_final": round(d_final, 2),
        "proj_inside": inside,
    }


def _eval_rr(nn: dict, final) -> tuple[bool, dict]:
    nodes = nn["nodes"]
    if not nodes:
        return False, {"nearest_idx": None}
    nearest = min(range(len(nodes)), key=lambda i: _d3(final, nodes[i]))
    if nn["subtype"] == "into":
        ok = nearest in set(nn["valid_end_idx"])
    else:  # exit
        ok = nearest not in set(nn["region_start_idx"])
    return ok, {"nearest_idx": nearest}


def _eval_nu(nn: dict, positions, final) -> tuple[bool, dict]:
    gt_paths = nn["set_paths_loc"][nn["gt_key"]]
    dists = [_d3(final, p[-1]) for p in gt_paths if p]
    if not dists:
        return False, {}
    best = min(range(len(dists)), key=lambda i: dists[i])
    goal_ok = dists[best] <= ERROR_MARGIN_M
    # Resample BOTH sides to the same arc-length granularity before nDTW (see
    # _resample_xy): the margin must compare route shapes, not node placement.
    pred = _resample_xy(positions)
    ndtw_gt = _ndtw(pred, _resample_xy(gt_paths[best]))
    ndtw_neg = 0.0
    for key, paths in nn["set_paths_loc"].items():
        if key == nn["gt_key"]:
            continue
        for p in paths:
            if p:
                ndtw_neg = max(ndtw_neg, _ndtw(pred, _resample_xy(p)))
    margin_ok = ndtw_gt > ndtw_neg
    return goal_ok and margin_ok, {
        "goal_ok": goal_ok,
        "ndtw_gt": round(ndtw_gt, 3),
        "ndtw_neg_max": round(ndtw_neg, 3),
    }


def evaluate_navnuances(episode: EpisodeSpec, trace: RolloutTrace) -> NavNuancesScore:
    nn = dict(episode.metadata.get("navnuances") or {})
    capability = str(nn.get("category", "?"))
    subtype = str(nn.get("subtype", "?"))
    goal = episode.goal_position or episode.start_pose.position
    if not trace.steps:
        return NavNuancesScore(
            episode_id=episode.episode_id,
            capability=capability,
            subtype=subtype,
            success=False,
            distance_to_goal=float("inf"),
            path_length=0.0,
            spl=0.0,
            stopped=False,
            stop_step=int(trace.stop_step),
            failure_reason="empty_trace",
        )

    positions = [step.position for step in trace.steps]
    start = episode.start_pose.position
    final = positions[-1]
    stopped = trace.stop_step >= 0
    path_length = _path_length(positions)
    distance_to_goal = _d3(final, goal)

    if capability == "DC":
        success, detail = _eval_dc(nn, start, positions)
    elif capability == "LR":
        success, detail = _eval_lr(nn, start, final, trace.steps[-1].orientation_wxyz)
    elif capability == "VM":
        success, detail = distance_to_goal < ERROR_MARGIN_M, {}
    elif capability == "RR":
        success, detail = _eval_rr(nn, final)
    elif capability == "NU":
        success, detail = _eval_nu(nn, positions, final)
    else:
        return NavNuancesScore(
            episode_id=episode.episode_id,
            capability=capability,
            subtype=subtype,
            success=False,
            distance_to_goal=distance_to_goal,
            path_length=path_length,
            spl=0.0,
            stopped=stopped,
            stop_step=int(trace.stop_step),
            failure_reason=f"unknown_capability:{capability}",
        )

    straight = _d3(start, goal)
    spl = (
        (straight / max(path_length, straight, 0.01))
        if success and straight > 0
        else (1.0 if success else 0.0)
    )
    return NavNuancesScore(
        episode_id=episode.episode_id,
        capability=capability,
        subtype=subtype,
        success=bool(success),
        distance_to_goal=distance_to_goal,
        path_length=path_length,
        spl=spl,
        stopped=stopped,
        stop_step=int(trace.stop_step),
        failure_reason=None if success else "criterion_not_met",
        detail=detail,
    )
