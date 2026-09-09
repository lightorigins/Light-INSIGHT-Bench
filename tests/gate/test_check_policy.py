"""`check-policy` is the acceptance command the contract document lacked.

Every failure it reports is one that otherwise surfaces only after a full suite
has run: a server that never says it arrived, a plan whose first waypoint is a
no-op, a forward delta clamped to a fifth of what the model asked for. Each test
here is one of those, seen through the command rather than through a trace.

The server is real. It used to be this SDK's own FastAPI host wrapped around an
in-process policy object; the host now lives in ``policies/`` as a separate
program in a separate interpreter, so what stands in for it here is a stdlib
:mod:`http.server` speaking the four endpoints by hand. That is the honest
substitution anyway: ``check_policy`` speaks HTTP to something it did not write,
and the only thing being scripted is what that something replies.
"""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from insight_bench.cli import main
from insight_bench.policy_check import check_policy

MODEL_ID = "test-lab/check-policy-stub"

Reply = Callable[[int], dict[str, Any]]


# --- the scripted server ------------------------------------------------------


def _waypoint(forward_m: float, lateral_m: float = 0.0, yaw_rad: float = 0.0) -> dict[str, Any]:
    """A waypoint reply, in the shape the wire protocol specifies."""
    return {
        "success": True,
        "action_id": 1,
        "action_name": "WAYPOINT",
        "raw_output": "",
        "waypoint": {
            "cluster_id": 3,
            "waypoints": [
                {"forward_m": forward_m, "lateral_m": lateral_m, "yaw_rad": yaw_rad}
                for _ in range(4)
            ],
        },
        "waypoint_cluster_id": 3,
        "waypoint_horizon": 4,
    }


def _stop() -> dict[str, Any]:
    """An explicit arrival: no waypoints at all, which is legal and common."""
    return {"success": True, "action_id": 0, "action_name": "STOP", "raw_output": "arrived"}


class _PolicyHandler(BaseHTTPRequestHandler):
    """The four endpoints. What ``/act`` answers is the server's ``reply``."""

    server_version = "StubPolicy/1.0"

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # do_GET / do_POST are BaseHTTPRequestHandler's spelling, not this repo's.
    def do_GET(self) -> None:
        if self.path != "/health":
            self._json({"error": "not found"}, status=404)
            return
        health: dict[str, Any] = {"healthy": True, "action_mode": "waypoint"}
        model_id = self.server.model_id  # type: ignore[attr-defined]
        if model_id is not None:
            health["model_id"] = model_id
        self._json(health)

    def do_POST(self) -> None:
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"] or 0)) or b"{}")
        if self.path == "/reset":
            # No frame rides with the reset: the first /act carries it.
            assert "first_frame_jpeg_b64" not in payload
            self.server.instructions.append(payload["instruction"])  # type: ignore[attr-defined]
            self._json({"success": True})
            return
        if self.path == "/act":
            frame_id = int(payload["frame_id"])
            self.server.acted.append(frame_id)  # type: ignore[attr-defined]
            self._json({**self.server.reply(frame_id), "frame_id": frame_id})  # type: ignore[attr-defined]
            return
        if self.path == "/finish":
            self.server.finished.append(payload["episode_id"])  # type: ignore[attr-defined]
            self._json({"success": True})
            return
        self._json({"error": "not found"}, status=404)


@pytest.fixture
def serve():
    """Serve a scripted reply on a real loopback port; `check_policy` speaks HTTP."""
    started: list[tuple[ThreadingHTTPServer, threading.Thread]] = []

    def _serve(reply: Reply, *, model_id: str | None = MODEL_ID) -> ThreadingHTTPServer:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _PolicyHandler)
        server.reply = reply  # type: ignore[attr-defined]
        server.model_id = model_id  # type: ignore[attr-defined]
        server.instructions = []  # type: ignore[attr-defined]
        server.acted = []  # type: ignore[attr-defined]
        server.finished = []  # type: ignore[attr-defined]
        server.url = f"http://127.0.0.1:{server.server_address[1]}"  # type: ignore[attr-defined]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        started.append((server, thread))
        return server

    yield _serve
    for server, thread in started:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _closed_port() -> int:
    """A port nothing is listening on: bound to learn the number, then released."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


# --- what the report says -----------------------------------------------------


def test_a_model_that_never_stops_is_named_as_such(serve) -> None:
    server = serve(lambda frame_id: _waypoint(1.0))
    report = check_policy(server.url, steps=3, instruction="go", timeout_sec=10.0)

    assert report["never_stopped"] is True
    assert report["stopped_at_step"] is None
    assert server.acted == [0, 1, 2]
    # And it shows the clamp while it is at it: 1.0 m asked, 0.25 m executed.
    assert report["steps"][0]["executed_velocity"]["linear_velocity_mps"] == pytest.approx(2.5)


def test_an_arrival_is_reported_with_the_reason_it_was_read_as_one(serve) -> None:
    server = serve(lambda frame_id: _stop() if frame_id else _waypoint(0.25))
    report = check_policy(server.url, steps=5, instruction="go", timeout_sec=10.0)

    assert report["stopped_at_step"] == 1
    assert report["steps"][1]["stop_kind"] == "policy_stop"
    assert report["never_stopped"] is False
    # The plan of a reply that says it arrived is never decoded -- an explicit
    # stop may carry no waypoints at all, as this one does.
    assert report["steps"][1]["executed_velocity"] is None
    assert server.finished == ["check-policy"]


def test_an_all_zero_plan_is_reported_as_the_implicit_stop_it_is(serve) -> None:
    """The failure that looks like motion: a plan that decodes to standing still."""
    server = serve(lambda frame_id: _waypoint(0.0, lateral_m=0.4))
    report = check_policy(server.url, steps=3, instruction="go", timeout_sec=10.0)

    # lateral_m is dropped by the suite, so a purely sideways plan is zero motion.
    assert report["steps"][0]["stop_kind"] == "zero_velocity_stop"
    assert report["stopped_at_step"] == 0


def test_a_server_with_no_model_id_is_visible_before_any_data_exists(serve) -> None:
    """The one gate that otherwise fails after data prep and a full server start."""
    server = serve(lambda frame_id: _waypoint(0.25), model_id=None)
    report = check_policy(server.url, steps=1, instruction="go", timeout_sec=10.0)

    assert report["model_id"] == ""


def test_the_instruction_reaches_the_server_verbatim(serve) -> None:
    server = serve(lambda frame_id: _stop())
    check_policy(server.url, steps=1, instruction="walk to the sofa", timeout_sec=10.0)

    assert server.instructions == ["walk to the sofa"]


# --- the exit codes the command is used by ------------------------------------


def test_a_walking_server_is_a_pass(serve, capsys) -> None:
    # Not stopping is not a failure by itself: the frame is a synthetic black
    # image, so a model that keeps walking on it is behaving normally.
    server = serve(lambda frame_id: _waypoint(0.25))
    assert main(["check-policy", "--policy-url", server.url, "--steps", "3"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["model_id"] == MODEL_ID
    assert report["never_stopped"] is True


def test_expect_stop_turns_a_server_that_never_stops_into_a_failure(serve, capsys) -> None:
    server = serve(lambda frame_id: _waypoint(0.25))
    status = main(["check-policy", "--policy-url", server.url, "--steps", "3", "--expect-stop"])

    assert status == 1
    assert json.loads(capsys.readouterr().out)["never_stopped"] is True


def test_expect_stop_passes_when_the_server_does_stop(serve, capsys) -> None:
    server = serve(lambda frame_id: _stop() if frame_id else _waypoint(0.25))
    status = main(["check-policy", "--policy-url", server.url, "--steps", "3", "--expect-stop"])

    assert status == 0
    assert json.loads(capsys.readouterr().out)["stopped_at_step"] == 1


def test_a_server_that_answers_without_naming_a_model_is_a_failure(serve, capsys) -> None:
    server = serve(lambda frame_id: _waypoint(0.25), model_id=None)
    assert main(["check-policy", "--policy-url", server.url, "--steps", "1"]) == 1
    assert json.loads(capsys.readouterr().out)["model_id"] == ""


def test_an_unreachable_server_fails_the_command_and_says_so(capsys) -> None:
    """Nothing listening: the command cannot answer, and must not exit 0.

    It exits 2 rather than 1 and writes the reason to stderr, which is this
    CLI's shape for "the command errored": exit 1 is reserved for a check that
    ran and returned a negative answer, and here nothing ran at all.
    """
    url = f"http://127.0.0.1:{_closed_port()}"
    status = main(["check-policy", "--policy-url", url, "--steps", "1"])

    assert status == 2
    captured = capsys.readouterr()
    assert captured.out == "", "a report that could not be produced is not printed"
    assert json.loads(captured.err)["error"]
