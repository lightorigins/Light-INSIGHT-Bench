"""Protocol tests for the continuous-space NavNuances evaluator (synthetic traces)."""

from __future__ import annotations

import math

from insight_bench.vln_runtime.episodes import EpisodeSpec
from insight_bench.vln_runtime.scoring.navnuances import evaluate_navnuances
from insight_bench.vln_runtime.traces.schema import RolloutTrace, StepTrace


def _episode(
    category: str, subtype: str, nn_extra: dict, *, start=(0.0, 0.0, 0.0), goal=(4.0, 0.0, 0.0)
) -> EpisodeSpec:
    return EpisodeSpec.from_dict(
        {
            "episode_id": f"test_{category}_{subtype}",
            "instruction": f"test {category} {subtype}",
            "scene": {
                "scene_id": "test",
                "asset_path": "/tmp/none.usd",
                "scene_type": "matterport_usd",
            },
            "start_pose": {"position": list(start), "orientation_wxyz": [1.0, 0.0, 0.0, 0.0]},
            "goal_position": list(goal),
            "goal_radius_m": 3.0,
            "robot": {"robot_id": "camera", "embodiment": "free_camera"},
            "subtasks": [],
            "metadata": {"navnuances": {"category": category, "subtype": subtype, **nn_extra}},
        }
    )


def _trace(episode_id: str, points, stopped: bool = True) -> RolloutTrace:
    return RolloutTrace(
        episode_id=episode_id,
        steps=[StepTrace(step=i, time_s=float(i), position=list(p)) for i, p in enumerate(points)],
        stop_step=len(points) - 1 if stopped else -1,
    )


def _run(ep: EpisodeSpec, points) -> bool:
    return evaluate_navnuances(ep, _trace(ep.episode_id, points)).success


# --- DC: first significant move vs the reference forward vector -------------
def _dc(subtype: str) -> EpisodeSpec:
    return _episode(
        "DC", subtype, {"heading_ref_vec": [1.0, 0.0], "min_move_m": 0.5, "pair_id": "p0"}
    )


def test_dc_left_correct_and_wrong():
    ep = _dc("left")
    assert _run(ep, [(0, 0, 0), (0.4, 0.7, 0)]) is True  # forward-left
    assert _run(ep, [(0, 0, 0), (0.4, -0.7, 0)]) is False  # forward-right
    assert _run(ep, [(0, 0, 0), (0.1, 0.1, 0)]) is False  # never moved >= 0.5 m


def test_dc_around():
    ep = _dc("around")
    assert _run(ep, [(0, 0, 0), (-1.0, 0.1, 0)]) is True  # > 120 deg from forward
    assert _run(ep, [(0, 0, 0), (1.0, 0.1, 0)]) is False


# --- LR: towards (closer than start) / past (projection + 3 m) ---------------
def test_lr_towards():
    ep = _episode("LR", "towards", {"obj_center": [4.0, 0.0, 0.0]})
    assert _run(ep, [(0, 0, 0), (2.0, 0.0, 0)]) is True
    assert _run(ep, [(0, 0, 0), (-2.0, 0.0, 0)]) is False


def test_lr_past():
    ep = _episode("LR", "past", {"obj_center": [4.0, 0.0, 0.0]})
    assert _run(ep, [(0, 0, 0), (5.5, 0.0, 0)]) is True  # beyond, within 3 m
    assert _run(ep, [(0, 0, 0), (2.0, 0.0, 0)]) is False  # stopped before the object
    assert _run(ep, [(0, 0, 0), (8.0, 0.0, 0)]) is False  # beyond but > 3 m away


# --- VM: 3D endpoint within 3 m ----------------------------------------------
def test_vm_endpoint():
    ep = _episode("VM", "up", {}, goal=(4.0, 0.0, 3.0))
    assert _run(ep, [(0, 0, 0), (4.0, 0.0, 3.0)]) is True
    assert _run(ep, [(0, 0, 0), (4.0, 0.0, 0.0)]) is False  # right xy, wrong floor


# --- RR: nearest-node region membership --------------------------------------
def test_rr_into_and_exit():
    nodes = [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 1.0, 0.0], [0.0, 5.0, 0.0]]
    into = _episode("RR", "into", {"nodes": nodes, "valid_end_idx": [1, 2], "region_start_idx": []})
    assert _run(into, [(0, 0, 0), (5.0, 0.4, 0)]) is True
    assert _run(into, [(0, 0, 0), (0.0, 4.8, 0)]) is False
    exit_ep = _episode("RR", "exit", {"nodes": nodes, "valid_end_idx": [], "region_start_idx": [0]})
    assert _run(exit_ep, [(0, 0, 0), (5.0, 0.2, 0)]) is True
    assert _run(exit_ep, [(0, 0, 0), (0.2, 0.0, 0)]) is False


# --- NU: 3 m endpoint + nDTW margin over negatives ----------------------------
_GT = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 2.0, 0.0]]
_NEG = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, -2.0, 0.0]]


def _nu_episode() -> EpisodeSpec:
    return _episode(
        "NU",
        "ordinal_2",
        {
            "set_paths_loc": {"left-2": [_GT], "right-2": [_NEG]},
            "gt_key": "left-2",
        },
        goal=(4.0, 2.0, 0.0),
    )


def _dense(path, step=0.2):
    out = [tuple(path[0])]
    for a, b in zip(path[:-1], path[1:]):
        seg = math.dist(a[:2], b[:2])
        n = max(1, int(seg / step))
        for k in range(1, n + 1):
            r = k / n
            out.append((a[0] + r * (b[0] - a[0]), a[1] + r * (b[1] - a[1]), 0.0))
    return out


def test_nu_gt_vs_negative_route():
    ep = _nu_episode()
    assert _run(ep, _dense(_GT)) is True
    assert _run(ep, _dense(_NEG)) is False


def test_nu_dense_trace_vs_sparse_refs_regression():
    """A dense many-step trace along the GT route must not lose the nDTW margin to a
    sibling path just because the references are sparse graph nodes (the dense-vs-
    sparse mismatch produced false negatives with the agent 0.15 m from the GT end)."""
    ep = _nu_episode()
    wiggly = [
        (x + 0.05 * math.sin(8 * x), y + 0.05 * math.cos(7 * (x + y)), z)
        for x, y, z in _dense(_GT, step=0.1)
    ]
    score = evaluate_navnuances(ep, _trace(ep.episode_id, wiggly))
    assert score.success is True
    assert score.detail["ndtw_gt"] > 0.8
