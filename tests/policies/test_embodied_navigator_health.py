"""Liveness for the one adapter that runs its model in a child process.

Embodied-Navigator launches an upstream HTTP service and talks to it over
loopback, so there are two things that can die and only one of them is a
process this side holds a handle to. The first version of this check asked
``Popen.poll()``, which answers "is the launcher still running" -- and a run
with a live launcher and a dead service behind it produced ten episodes that
each failed on connection refused, after the pre-run health check had passed.
A liveness check that cannot see the thing that actually answers is the same
bug one layer down, so it asks the child the question startup asks.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "policies"))

from embodied_navigator.policy import EmbodiedNavigatorPolicy


class _Proc:
    """The bit of ``subprocess.Popen`` this check touches."""

    def __init__(self, returncode: int | None) -> None:
        self._returncode = returncode

    def poll(self) -> int | None:
        return self._returncode


def _policy(proc: Any, *, port: int) -> EmbodiedNavigatorPolicy:
    """An adapter with its child plumbing set, without loading a model."""
    policy = object.__new__(EmbodiedNavigatorPolicy)
    policy._proc = proc
    policy._base = f"http://127.0.0.1:{port}"
    return policy


def test_no_child_is_not_healthy() -> None:
    assert _policy(None, port=1).healthy() is False


def test_an_exited_child_is_not_healthy() -> None:
    assert _policy(_Proc(1), port=1).healthy() is False


def test_a_live_pid_whose_service_refuses_is_not_healthy() -> None:
    """The case that got through: launcher alive, nothing listening behind it.

    Port 1 is privileged and unbound, so the probe refuses immediately rather
    than waiting out its timeout.
    """
    assert _policy(_Proc(None), port=1).healthy() is False


def test_a_child_that_answers_is_healthy() -> None:
    import http.server
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server's spelling
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args: Any) -> None:
            return

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        policy = _policy(_Proc(None), port=server.server_address[1])
        assert policy.healthy() is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _decoding_policy(monkeypatch: Any, env_action: dict[str, Any]) -> Any:
    """An adapter wired to a canned upstream reply, with no model behind it."""
    from embodied_navigator import policy as module

    policy = object.__new__(EmbodiedNavigatorPolicy)
    policy._proc = _Proc(None)
    policy._base = "http://127.0.0.1:1"
    policy._pose = [0.0, 0.0, 0.0]
    policy._inferences = 0
    policy.step_timeout_s = 1.0
    policy.camera_height_m = 1.0
    policy.hfov_deg = 120.0
    policy.min_wp_m = 0.25
    policy.max_wp_m = 3.0
    policy.horizon_fallback_m = 2.0
    monkeypatch.setattr(
        module, "_post_json", lambda *a, **k: {"env_action": env_action, "model_text": "x"}
    )
    return policy


def test_a_pixel_target_turns_and_advances_in_the_same_waypoint(monkeypatch: Any) -> None:
    """The regression that made this baseline score zero by construction.

    A waypoint list is a plan and the runner executes only its first entry
    before re-planning, so emitting a turn and then an advance meant the
    advance was never reached: 609 consecutive inferences, every one of them
    changing heading, and a path length of exactly 0.000 m. Whatever the first
    waypoint is, it has to carry the forward motion.
    """
    import numpy as np

    # Left of centre and below the horizon: a real target on the floor.
    policy = _decoding_policy(monkeypatch, {"sensor": "rgb_front", "action": [120, 200]})
    step = policy.act(np.zeros((270, 480, 3), dtype=np.uint8))

    first = step.waypoint_deltas[0]
    assert first.forward_m > 0.0, "the executed waypoint must advance, not only turn"
    assert first.yaw_rad > 0.0, "a target left of centre turns left"


def test_a_side_view_choice_is_a_pure_turn(monkeypatch: Any) -> None:
    """Facing a side camera is the one case with nothing to walk towards yet."""
    import numpy as np

    policy = _decoding_policy(monkeypatch, {"sensor": "rgb_left", "action": [10, 10]})
    step = policy.act(np.zeros((270, 480, 3), dtype=np.uint8))

    first = step.waypoint_deltas[0]
    assert first.forward_m == 0.0
    assert first.yaw_rad > 0.0
