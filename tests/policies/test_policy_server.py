"""The model-side HTTP protocol, driven end to end against a fake policy.

These tests speak to a real socket on an ephemeral port with ``http.client``,
because the thing under test is the wire contract: what a model's environment
must put on the network for the benchmark to be able to drive it. No model, no
network beyond loopback, no dependency beyond pytest, numpy and pillow.

``policies/`` is not an installed package -- it ships as source for a model's
own environment to copy or point at -- so it is put on the path here the same
way ``python -m insight_policy`` puts it there.
"""

from __future__ import annotations

import base64
import http.client
import io
import json
import math
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "policies"))

from insight_policy import Policy, Step
from insight_policy.server import FramePreprocessor, make_server

RIG_WIDTH = 480
RIG_HEIGHT = 270


class FakePolicy(Policy):
    """Replays a scripted list of Steps, one per act(), and records what it saw."""

    model_id: ClassVar[str] = "fake-org/fake-model"

    defaults: ClassVar[dict[str, Any]] = {"script": None}

    def __init__(self, model_path: str = "/fake/ckpt", **options: Any):
        super().__init__(model_path, **options)
        self.script: list[Any] = list(self.options["script"] or [])
        self.instructions: list[str] = []
        self.acted: list[Any] = []
        self.observed: list[Any] = []
        self.finished = 0

    def reset(self, instruction: str) -> None:
        self.instructions.append(instruction)

    def act(self, rgb: Any) -> Step:
        self.acted.append(rgb)
        if not self.script:
            return Step.stop(raw_output="script exhausted")
        entry = self.script.pop(0)
        if isinstance(entry, BaseException):
            raise entry
        return entry

    def observe(self, rgb: Any) -> None:
        self.observed.append(rgb)

    def finish(self) -> None:
        self.finished += 1

    def describe(self) -> dict[str, Any]:
        return {"steps_left": len(self.script)}


class Client:
    """The four endpoints over a real socket."""

    def __init__(self, port: int):
        self.port = port

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            body = None if payload is None else json.dumps(payload).encode("utf-8")
            headers = {"Content-Type": "application/json"} if body else {}
            conn.request(method, path, body=body, headers=headers)
            response = conn.getresponse()
            return json.loads(response.read().decode("utf-8"))
        finally:
            conn.close()

    def health(self) -> dict:
        return self._request("GET", "/health")

    def reset(self, instruction: str, episode_id: str = "ep-1") -> dict:
        return self._request(
            "POST", "/reset", {"instruction": instruction, "episode_id": episode_id}
        )

    def act(self, frame: Any = None, *, episode_id: str = "ep-1", frame_id: int = 0) -> dict:
        return self._request(
            "POST",
            "/act",
            {
                "episode_id": episode_id,
                "frame_id": frame_id,
                "timestamp_ms": 0,
                "image_jpeg_b64": jpeg_b64(rig_frame() if frame is None else frame),
            },
        )

    def act_raw(self, payload: dict) -> dict:
        return self._request("POST", "/act", payload)

    def finish(self, episode_id: str = "ep-1") -> dict:
        return self._request("POST", "/finish", {"episode_id": episode_id})


def rig_frame(width: int = RIG_WIDTH, height: int = RIG_HEIGHT) -> np.ndarray:
    """A frame the rig could have produced, with structure a crop would move."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    frame[:, :, 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    return frame


def jpeg_b64(frame: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@contextmanager
def serving(policy: Policy):
    httpd = make_server(policy, host="127.0.0.1", port=0)
    thread = threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )
    thread.start()
    try:
        yield Client(httpd.server_address[1])
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


@contextmanager
def served(script: list | None = None, **options: Any):
    policy = FakePolicy(script=script, **options)
    with serving(policy) as client:
        yield client, policy


# ---------------------------------------------------------------------------------
# identity and lifecycle
# ---------------------------------------------------------------------------------


def test_health_reports_the_policy_identity():
    with served() as (client, _):
        health = client.health()

    assert health["healthy"] is True
    assert health["model_id"] == "fake-org/fake-model"
    assert health["model_path"] == "/fake/ckpt"
    # Whatever describe() returned is namespaced, not merged blindly.
    assert health["policy_steps_left"] == 0


def test_act_before_reset_is_refused_with_a_readable_reason():
    with served([Step.primitives(["forward"])]) as (client, policy):
        response = client.act()

    assert response["success"] is False
    assert "reset" in response["error"].lower()
    # The frame never reached the policy: the protocol error came first.
    assert policy.acted == []


def test_act_for_an_episode_that_was_never_reset_names_the_episode():
    with served([Step.primitives(["forward"])]) as (client, _):
        client.reset("go to the sofa", episode_id="ep-1")
        response = client.act(episode_id="ep-other")

    assert response["success"] is False
    assert "ep-other" in response["error"]


def test_finish_releases_the_episode():
    with served([Step.primitives(["forward"])]) as (client, policy):
        client.reset("go to the sofa")
        assert client.finish()["removed"] is True
        assert policy.finished == 1
        assert client.health()["num_active_sessions"] == 0


# ---------------------------------------------------------------------------------
# the action queue
# ---------------------------------------------------------------------------------


def test_queue_releases_one_primitive_per_act():
    chunk = Step.primitives(["forward", "left", "forward"], raw_output="F L F")
    with served([chunk]) as (client, policy):
        client.reset("go to the sofa")
        first = client.act(frame_id=0)
        second = client.act(frame_id=1)
        third = client.act(frame_id=2)

    # One inference produced three motions; each act released exactly one.
    assert len(policy.acted) == 1
    assert [r["waypoint_horizon"] for r in (first, second, third)] == [1, 1, 1]
    assert [len(r["waypoint"]["waypoints"]) for r in (first, second, third)] == [1, 1, 1]
    assert [r["frame_id"] for r in (first, second, third)] == [0, 1, 2]

    # Only the inference frame carries the model's text; the drained ones do not.
    assert first["raw_output"] == "F L F"
    assert second["raw_output"] == ""
    assert first["extra"]["source"] == "inference"
    assert second["extra"]["source"] == "queue"
    # queue_len is measured before this reply's motion is released.
    assert first["extra"]["queue_len"] == 3
    assert second["extra"]["queue_len"] == 2


def test_queued_frames_reach_the_policy_only_when_it_asks_for_them():
    chunk = [Step.primitives(["forward", "forward"])]
    with served(list(chunk), queue_feed_frames=False) as (client, quiet):
        client.reset("go to the sofa")
        client.act()
        client.act()
    with served(list(chunk), queue_feed_frames=True) as (client, hungry):
        client.reset("go to the sofa")
        client.act()
        client.act()

    assert quiet.observed == []
    assert len(hungry.observed) == 1  # the second act's frame, drained from the queue


def test_a_step_that_moves_then_stops_drains_before_it_stops():
    with served([Step.primitives(["forward"], then_stop=True)]) as (client, _):
        client.reset("go to the sofa")
        moved = client.act()
        stopped = client.act()

    assert moved["action_name"] == "WAYPOINT"
    assert stopped["action_name"] == "stop"


def test_an_empty_step_falls_back_to_the_on_empty_setting():
    with served([Step.primitives([])], on_empty="forward") as (client, _):
        client.reset("go to the sofa")
        keeps_going = client.act()
    with served([Step.primitives([])], on_empty="stop") as (client, _):
        client.reset("go to the sofa")
        gives_up = client.act()

    assert keeps_going["action_name"] == "WAYPOINT"
    assert keeps_going["extra"]["on_empty_fallback"] == "forward"
    assert gives_up["action_name"] == "stop"


def test_the_on_empty_fallback_obeys_the_per_step_caps():
    # A policy whose step size is 0.75 m must not have its fallback emitted as
    # one oversized reply: the runner would clip it to 0.25 m and the episode
    # would quietly walk a third of the distance the policy asked for.
    with served([Step.primitives([])], on_empty="forward", forward_m=0.75) as (client, _):
        client.reset("go to the sofa")
        walked = [client.act()["waypoint"]["waypoints"][0]["forward_m"] for _ in range(3)]

    assert walked == pytest.approx([0.25, 0.25, 0.25])
    assert sum(walked) == pytest.approx(0.75)


# ---------------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------------


def test_primitives_become_displacements_at_the_policy_constants():
    script = [Step.primitives(["forward", "left", "right"])]
    with served(script, forward_m=0.25, turn_deg=30.0) as (client, _):
        client.reset("go to the sofa")
        forward = client.act()["waypoint"]["waypoints"][0]
        left = client.act()["waypoint"]["waypoints"][0]
        right = client.act()["waypoint"]["waypoints"][0]

    assert forward == {"forward_m": 0.25, "lateral_m": 0.0, "yaw_rad": 0.0}
    # Positive yaw is counter-clockwise, i.e. a left turn.
    assert left["yaw_rad"] == pytest.approx(math.radians(30.0))
    assert right["yaw_rad"] == pytest.approx(-math.radians(30.0))
    assert left["forward_m"] == 0.0


def test_a_primitive_larger_than_one_step_is_split_across_replies():
    # 0.75 m of forward is three of the 0.25 m the runner can execute per step,
    # and a 45 degree turn is 30 plus 15.
    with served([Step.primitives(["forward"])], forward_m=0.75) as (client, _):
        client.reset("go to the sofa")
        walked = [client.act()["waypoint"]["waypoints"][0]["forward_m"] for _ in range(3)]
    with served([Step.primitives(["left"])], turn_deg=45.0) as (client, _):
        client.reset("go to the sofa")
        turned = [client.act()["waypoint"]["waypoints"][0]["yaw_rad"] for _ in range(2)]

    assert walked == pytest.approx([0.25, 0.25, 0.25])
    assert sum(walked) == pytest.approx(0.75)
    assert turned == pytest.approx([math.radians(30.0), math.radians(15.0)])
    assert sum(turned) == pytest.approx(math.radians(45.0))


def test_an_explicit_waypoint_within_the_caps_survives_the_wire_intact():
    # A policy that asked to turn and translate at once gets exactly that back:
    # the whole motion fits one control step, so there is nothing to decompose.
    step = Step.waypoints([(0.2, 0.05, 0.1)], raw_output="pixel->(240,200)")
    with served([step]) as (client, _):
        client.reset("go to the sofa")
        response = client.act()

    waypoint = response["waypoint"]["waypoints"][0]
    assert waypoint["forward_m"] == pytest.approx(0.2)
    assert waypoint["lateral_m"] == pytest.approx(0.05)
    assert waypoint["yaw_rad"] == pytest.approx(0.1)
    assert response["waypoint_horizon"] == 1
    # A plan is handed over, not queued: the model is asked again next frame.
    assert response["extra"]["queue_len"] == 0


def test_a_waypoint_chunk_arrives_whole_and_is_never_replayed_from_a_queue():
    # A trajectory model answers with a plan: several rows, usually cumulative
    # poses. All of them go over the wire in ONE reply, verbatim, and the runner
    # decides which row to execute and clips it. Releasing them one per /act
    # instead would ask the model for a decision once every H frames and would
    # execute row i -- i steps of motion -- as a single step.
    rows = [(0.0, 0.0, math.radians(18.0 * (i + 1))) for i in range(10)]
    plan = Step.waypoints(rows, raw_output="<act_l0_41><act_l1_53><act_l2_239>")
    with served([plan, plan, plan]) as (client, _):
        client.reset("turn left toward the table")
        first = client.act()
        second = client.act()

    assert first["waypoint_horizon"] == 10
    assert [row["yaw_rad"] for row in first["waypoint"]["waypoints"]] == pytest.approx(
        [math.radians(18.0 * (i + 1)) for i in range(10)]
    )
    # Nothing was held back, and the second frame ran inference again.
    assert first["extra"]["queue_len"] == 0
    assert second["extra"]["source"] == "inference"
    assert second["extra"]["inference_calls"] == 2


def test_an_oversized_waypoint_is_not_rewritten_by_the_server():
    # The per-step cap belongs to the runner, which clips what it executes and
    # records what was asked. A server that quietly split an oversized row would
    # make the trace disagree with the model's own plan.
    step = Step.waypoints([(0.5, 0.0, math.radians(90.0))])
    with served([step]) as (client, _):
        client.reset("go to the sofa")
        response = client.act()

    waypoint = response["waypoint"]["waypoints"][0]
    assert waypoint["forward_m"] == pytest.approx(0.5)
    assert waypoint["yaw_rad"] == pytest.approx(math.radians(90.0))
    assert response["waypoint_horizon"] == 1


def test_a_vocabulary_policy_reports_its_own_cluster_id():
    # The runner reads waypoint_cluster_id straight into the step record, so a
    # policy that decodes from a trajectory vocabulary has to be able to put its
    # real codeword there. Everyone else gets the discrete-action sentinel.
    plan = Step.waypoints([(0.2, 0.0, 0.0)], extras={"cluster_id": 417})
    with served([plan]) as (client, _):
        client.reset("go to the sofa")
        decoded = client.act()
    chunk = Step.primitives(["forward", "forward"])
    with served([chunk]) as (client, _):
        client.reset("go to the sofa")
        plain = client.act()
        drained = client.act()

    assert decoded["waypoint_cluster_id"] == 417
    assert decoded["waypoint"]["cluster_id"] == 417
    # A policy with no vocabulary keeps the sentinel, on the queued reply too.
    assert plain["waypoint_cluster_id"] == -1
    assert drained["waypoint_cluster_id"] == -1


def test_center_crop_narrows_the_field_of_view_by_the_tangent_ratio():
    # crop_w = W * tan(target/2) / tan(source/2) = 480 * tan(45) / tan(60)
    expected_w = round(RIG_WIDTH * math.tan(math.radians(45.0)) / math.tan(math.radians(60.0)))
    assert expected_w == 277
    # 4:3 crop_aspect then fixes the height so the vertical FOV matches too.
    expected_h = round(expected_w / (4 / 3))
    assert expected_h == 208

    preprocessor = FramePreprocessor(
        {"source_hfov_deg": 120.0, "center_crop_hfov_deg": 90.0, "crop_aspect": 4 / 3}
    )
    cropped = preprocessor(rig_frame())
    assert cropped.shape == (expected_h, expected_w, 3)

    # And the policy is what actually receives the cropped frame over the wire.
    with served(
        [Step.primitives(["forward"])],
        center_crop_hfov_deg=90.0,
        crop_aspect=4 / 3,
    ) as (client, policy):
        client.reset("go to the sofa")
        client.act()

    assert policy.acted[0].shape == (expected_h, expected_w, 3)


def test_an_uncropped_policy_sees_the_rig_frame_unchanged():
    with served([Step.primitives(["forward"])]) as (client, policy):
        client.reset("go to the sofa")
        client.act()

    assert policy.acted[0].shape == (RIG_HEIGHT, RIG_WIDTH, 3)


# ---------------------------------------------------------------------------------
# stopping and failure
# ---------------------------------------------------------------------------------


def test_stop_replies_carry_no_waypoint():
    with served([Step.stop(raw_output="stop")]) as (client, _):
        client.reset("go to the sofa")
        response = client.act()

    assert response["success"] is True
    # Lowercase "stop" is what the runner's termination check matches on, and it
    # checks it before it looks for a waypoint, so there is none to send.
    assert response["action_name"] == "stop"
    assert "waypoint" not in response


def test_malformed_base64_is_reported_not_raised():
    with served([Step.primitives(["forward"])]) as (client, policy):
        client.reset("go to the sofa")
        response = client.act_raw(
            {"episode_id": "ep-1", "frame_id": 0, "image_jpeg_b64": "not base64!!"}
        )

    assert response["success"] is False
    assert response["action_name"] == "ERROR"
    assert response["error"]
    assert policy.acted == []


def test_an_empty_image_field_is_reported():
    with served([Step.primitives(["forward"])]) as (client, _):
        client.reset("go to the sofa")
        response = client.act_raw({"episode_id": "ep-1", "frame_id": 3, "image_jpeg_b64": ""})

    assert response["success"] is False
    assert response["frame_id"] == 3


def test_an_exception_inside_the_policy_becomes_success_false():
    boom = RuntimeError("CUDA out of memory")
    with served([boom, Step.primitives(["forward"])]) as (client, _):
        client.reset("go to the sofa")
        failed = client.act(frame_id=7)
        # The server survives it: the next act still works.
        recovered = client.act(frame_id=8)

    assert failed["success"] is False
    assert failed["action_name"] == "ERROR"
    assert "CUDA out of memory" in failed["error"]
    assert failed["frame_id"] == 7
    assert recovered["success"] is True


def test_reset_without_an_instruction_is_refused():
    with served() as (client, policy):
        response = client.reset("   ")

    assert response["success"] is False
    assert policy.instructions == []


def test_an_unknown_route_is_a_404_with_a_body():
    with served() as (client, _):
        response = client._request("GET", "/nope")

    assert response["success"] is False
    assert "/nope" in response["error"]


def test_health_reports_the_policy_and_not_the_server() -> None:
    """A server whose model has gone must not answer `healthy: true`.

    Embodied-Navigator runs its model in a child process. That child died, the
    server stayed up and kept answering, and every reply decoded to no motion:
    two episodes spent their whole 300-action budget with a path length of
    0.00 m while /health said the policy was fine. The runner's only liveness
    signal was the server's own, so nothing refused the run.
    """

    class DeadModel(FakePolicy):
        def healthy(self) -> bool:
            return False

    with serving(DeadModel(script=[])) as client:
        health = client.health()

    assert health["healthy"] is False
    assert health["model_id"] == FakePolicy.model_id


def test_health_is_false_when_asking_the_policy_raises() -> None:
    class Broken(FakePolicy):
        def healthy(self) -> bool:
            raise RuntimeError("the child is unreachable")

    with serving(Broken(script=[])) as client:
        assert client.health()["healthy"] is False


def test_a_policy_that_says_nothing_is_healthy_by_default() -> None:
    with serving(FakePolicy(script=[])) as client:
        assert client.health()["healthy"] is True
