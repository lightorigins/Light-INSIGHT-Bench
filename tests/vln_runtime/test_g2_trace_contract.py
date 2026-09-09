"""The per-episode trace must carry the evidence a G2 acceptance reads.

A RunResult carries metrics and a summary by design -- ADR 0005 refuses opaque metadata blobs --
so the per-step evidence lives in the RolloutTrace sidecar instead. These tests pin what a step
must contain, because the alternative is discovering it an hour into a queued GPU job.

The backend here is a fake, but not a lenient one: it builds real numpy arrays and passes them
through the real ``Observation`` contract, so a step summary that claims uint8 HWC3 rgb is a claim
about an array that actually satisfied that constraint.
"""

from __future__ import annotations

import numpy as np
import pytest

from insight_bench.simulator.observation import make_observation
from insight_bench.vln_runtime.motion.pose import CameraPose
from insight_bench.vln_runtime.rollout.camera_walk import (
    PoseReadbackError,
    _observation_summary,
    _terrain_metadata,
    capture_step_observation,
)
from insight_bench.vln_runtime.traces.schema import RolloutTrace

H, W = 4, 6


class _Backend:
    """Renders real arrays and reports a real terrain outcome."""

    def __init__(
        self, *, query_available: bool, raycast_hit: bool, with_depth: bool = True
    ) -> None:
        self._query_available = query_available
        self._raycast_hit = raycast_hit
        self._with_depth = with_depth
        self.captures = 0

    def capture_observation(self):
        self.captures += 1
        return make_observation(
            np.zeros((H, W, 3), dtype=np.uint8),
            pose=CameraPose(0.0, 0.0, 0.0, 0.0),
            orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            depth=np.zeros((H, W), dtype=np.float32) if self._with_depth else None,
            intrinsics=np.eye(3, dtype=np.float64),
            frame_id=1,
        )

    def last_terrain_query(self):
        return {
            "terrain_query_available": self._query_available,
            "terrain_raycast_hit": self._raycast_hit,
        }


class TestTheObservationSummary:
    def test_it_reports_the_contract_the_acceptance_reads(self):
        backend = _Backend(query_available=True, raycast_hit=True)
        summary = _observation_summary(capture_step_observation(backend), None)
        assert summary["rgb"] == {"dtype": "uint8", "shape": [H, W, 3]}
        assert summary["depth"] == {"dtype": "float32", "shape": [H, W]}
        assert len(summary["intrinsics"]) == 3
        assert all(len(row) == 3 for row in summary["intrinsics"])

    def test_absent_depth_is_reported_absent_not_invented(self):
        backend = _Backend(query_available=True, raycast_hit=True, with_depth=False)
        summary = _observation_summary(capture_step_observation(backend), None)
        assert "depth" not in summary

    def test_a_backend_without_observations_degrades_to_the_frame(self):
        class _FrameOnly:
            pass

        frame = np.zeros((H, W, 3), dtype=np.uint8)
        assert capture_step_observation(_FrameOnly()) is None
        assert _observation_summary(None, frame) == {"rgb": {"dtype": "uint8", "shape": [H, W, 3]}}

    def test_the_summary_does_not_capture_its_own_observation(self):
        """It describes the step's capture, so there can only be one per step.

        Two captures are two readings of a moving camera, and the pose in the record
        would then not be the pose the summary describes.
        """
        backend = _Backend(query_available=True, raycast_hit=True)
        observation = capture_step_observation(backend)
        assert backend.captures == 1

        _observation_summary(observation, None)

        assert backend.captures == 1


class TestAFailedCaptureEndsTheEpisode:
    """The swallow that made a broken run look modest, removed.

    `_observation_summary` used to catch a raising `capture_observation` and record
    `{"error": ...}` while the rollout carried on with commanded poses. On the real L20
    job every readback raised, and the episode still produced a full trace that scored
    perfectly -- against itself.
    """

    def test_a_raising_capture_is_fatal_rather_than_recorded(self):
        class _Broken:
            def capture_observation(self):
                raise RuntimeError("camera exploded")

        with pytest.raises(PoseReadbackError, match="camera exploded"):
            capture_step_observation(_Broken())

    def test_a_capture_returning_none_is_not_a_licence_to_use_the_command(self):
        # None is a protocol violation, not an absence of capability: the method is
        # there, so the caller must not quietly fall back to the commanded pose.
        class _Empty:
            def capture_observation(self):
                return None

        with pytest.raises(PoseReadbackError, match="returned None"):
            capture_step_observation(_Empty())

    def test_the_failure_carries_type_and_message_but_no_traceback(self):
        class _Broken:
            def capture_observation(self):
                raise RuntimeError("/secret/path/to/thing failed")

        # A trace and a log go into a shared artifact directory, and a frame dump carries
        # absolute paths and sometimes locals.
        with pytest.raises(PoseReadbackError) as excinfo:
            capture_step_observation(_Broken())
        message = str(excinfo.value)
        assert "RuntimeError: /secret/path/to/thing failed" in message
        assert "Traceback" not in message and "\n" not in message


class TestTheTerrainReport:
    def test_a_hit_is_reported_as_a_hit(self):
        assert _terrain_metadata(_Backend(query_available=True, raycast_hit=True)) == {
            "terrain_query_available": True,
            "terrain_raycast_hit": True,
        }

    def test_a_raising_query_is_distinguishable_from_a_scene_with_nothing_to_report(self):
        class _Broken:
            def last_terrain_query(self):
                raise RuntimeError("stage closed")

        # An empty dict is what a backend with no terrain surface legitimately yields, so a
        # failure must not produce the same thing.
        assert _terrain_metadata(_Broken()) == {"error": "RuntimeError: stage closed"}

    def test_no_mesh_is_distinguishable_from_a_miss(self):
        """The distinction this exists for.

        A plane terrain has no mesh, so no ray is cast; a mesh terrain can cast one and miss. One
        flag would make "nothing to hit" look like "terrain following failed", and those want
        opposite reactions.
        """
        no_mesh = _terrain_metadata(_Backend(query_available=False, raycast_hit=False))
        missed = _terrain_metadata(_Backend(query_available=True, raycast_hit=False))
        assert no_mesh != missed
        assert no_mesh["terrain_query_available"] is False
        assert missed["terrain_query_available"] is True


class TestTheStepTraceCarriesItThrough:
    """A summary that never reaches the persisted trace is evidence nobody can read."""

    def _trace(self) -> RolloutTrace:
        backend = _Backend(query_available=True, raycast_hit=True)
        record = {
            "step": 0,
            "pose_before": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
            "pose_after": {"x": 1.0, "y": 2.0, "z": 3.0, "yaw": 0.0},
            "measures": {},
            "observation": _observation_summary(capture_step_observation(backend), None),
            "metadata": _terrain_metadata(backend),
        }
        return RolloutTrace.from_camera_walk_records(
            episode_id="ep-0", instruction="go", records=[record]
        )

    def test_position_and_canonical_orientation_are_present(self):
        step = self._trace().steps[-1]
        assert step.position == (1.0, 2.0, 3.0)
        assert len(step.orientation_wxyz) == 4
        # Canonical WXYZ: identity yaw is (1, 0, 0, 0), not (0, 0, 0, 1).
        assert step.orientation_wxyz[0] == pytest.approx(1.0)

    def test_the_observation_contract_survives_into_the_step(self):
        observation = self._trace().steps[-1].observation
        assert observation["rgb"] == {"dtype": "uint8", "shape": [H, W, 3]}
        assert observation["depth"] == {"dtype": "float32", "shape": [H, W]}
        assert len(observation["intrinsics"]) == 3

    def test_the_terrain_report_survives_into_the_step(self):
        metadata = self._trace().steps[-1].metadata
        assert metadata["terrain_raycast_hit"] is True
        assert metadata["terrain_query_available"] is True

    def test_it_round_trips_through_the_persisted_json(self, tmp_path):
        from insight_bench.vln_runtime.traces.writer import write_rollout_trace

        path = write_rollout_trace(tmp_path / "ep-0" / "trace.json", self._trace())
        assert path.name == "trace.json" and path.parent.name == "ep-0"

        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        step = payload["steps"][-1]
        assert step["observation"]["rgb"]["dtype"] == "uint8"
        assert step["observation"]["depth"]["shape"] == [H, W]
        assert step["metadata"]["terrain_raycast_hit"] is True
        assert step["position"] == [1.0, 2.0, 3.0]
