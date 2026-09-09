"""The frame sink: what it names, when it writes, and what it refuses.

The numbering is the point of this file. The rollout counts decisions from 0
and the trace records decision k as step k + 1, so a reader who reconstructed a
filename from a trace index would be off by one on every frame. The sink writes
the trace-step name, the rollout puts that name into the step, and this module
pins that agreement.
"""

from __future__ import annotations

import numpy as np
import pytest

from insight_bench._paths import UnsafePathError
from insight_bench.vln_runtime.episodes import EpisodeSpec
from insight_bench.vln_runtime.traces.frames import JpegFrameSink, frame_stride_for

JPEG_MAGIC = b"\xff\xd8\xff"


def _frame(height: int = 8, width: int = 8, channels: int = 3) -> np.ndarray:
    grid = np.arange(height * width * channels, dtype=np.uint16)
    return ((grid * 37) % 251).astype(np.uint8).reshape(height, width, channels)


def test_the_stride_keeps_every_episode_under_the_frame_cap():
    # 100 steps and 300 steps are the two real suites; both must fit the cap,
    # which is what makes the footprint of a run independent of num_steps.
    assert frame_stride_for(100, max_frames=100) == 1
    assert frame_stride_for(300, max_frames=100) == 3
    assert frame_stride_for(250, max_frames=100) == 3
    assert 250 // frame_stride_for(250, max_frames=100) + 1 <= 100
    # Degenerate inputs still produce a usable stride rather than a ZeroDivisionError.
    assert frame_stride_for(1, max_frames=100) == 1
    assert frame_stride_for(0, max_frames=100) == 1
    assert frame_stride_for(100, max_frames=0) == 100


def test_decision_k_is_written_as_trace_step_k_plus_one(tmp_path):
    sink = JpegFrameSink(root=tmp_path)
    assert sink("ep-1", 0, _frame()) == "frames/step_0001.jpg"
    assert sink("ep-1", 1, _frame()) == "frames/step_0002.jpg"
    written = tmp_path / "ep-1" / "frames" / "step_0001.jpg"
    assert written.read_bytes()[:3] == JPEG_MAGIC
    # Trace step 0 is the pre-action pose. It made no decision and was shown no
    # frame, so its file must never exist for anyone to draw an overlay on.
    assert not (tmp_path / "ep-1" / "frames" / "step_0000.jpg").exists()
    assert sink.written == 2
    assert sink.failed == 0


def test_a_stride_writes_every_nth_decision_and_nothing_between(tmp_path):
    sink = JpegFrameSink(root=tmp_path, stride=3)
    paths = [sink("ep-1", step, _frame()) for step in range(7)]
    assert paths == [
        "frames/step_0001.jpg",
        "",
        "",
        "frames/step_0004.jpg",
        "",
        "",
        "frames/step_0007.jpg",
    ]
    assert sorted(path.name for path in (tmp_path / "ep-1" / "frames").iterdir()) == [
        "step_0001.jpg",
        "step_0004.jpg",
        "step_0007.jpg",
    ]
    assert sink.written == 3


def test_a_frame_the_encoder_rejects_costs_the_picture_and_nothing_else(tmp_path):
    # A frame-only backend can hand back an array that is not RGB. Losing the
    # picture is the correct price; losing the episode is not.
    sink = JpegFrameSink(root=tmp_path)
    assert sink("ep-1", 0, _frame(channels=2)) == ""
    assert sink.failed == 1
    assert "ValueError" in sink.first_failure
    assert not (tmp_path / "ep-1").exists()
    # The first reason is kept, not the last: the interesting one is the first.
    assert sink("ep-1", 1, _frame(channels=1)) == ""
    assert sink.failed == 2
    assert "(8, 8, 2)" in sink.first_failure


def test_a_missing_frame_is_representable_as_no_frame(tmp_path):
    sink = JpegFrameSink(root=tmp_path)
    assert sink("ep-1", 0, None) == ""
    assert sink.written == 0
    assert sink.failed == 0


def test_the_recorded_failure_reason_is_redacted(tmp_path):
    class Unencodable:
        def __array__(self, dtype=None, copy=None):
            raise ValueError("/Users/nobody/scenes/private.usd rendered nothing")

    sink = JpegFrameSink(root=tmp_path)
    assert sink("ep-1", 0, Unencodable()) == ""
    assert "/Users/nobody" not in sink.first_failure
    assert "<redacted-path>" in sink.first_failure


def test_an_episode_id_that_is_not_a_path_segment_is_refused_not_sanitised(tmp_path):
    # EpisodeSpec rejects such an id at parse time, so this is the second lock,
    # on the side that actually touches the filesystem. It refuses rather than
    # counting a failed frame: a dataset choosing where a run writes is not a
    # visualisation problem.
    with pytest.raises(ValueError, match="unsafe episode_id"):
        EpisodeSpec.from_dict(
            {
                "episode_id": "../../evil",
                "scene": {"scene_id": "s"},
                "start_pose": {"position": [0, 0, 0]},
            }
        )
    sink = JpegFrameSink(root=tmp_path / "run")
    (tmp_path / "run").mkdir()
    for hostile in ("../../evil", "/etc/cron.d/x", "a\\b"):
        with pytest.raises(UnsafePathError):
            sink(hostile, 0, _frame())
    assert list((tmp_path / "run").iterdir()) == []
    assert list(tmp_path.iterdir()) == [tmp_path / "run"]
    assert sink.written == 0
