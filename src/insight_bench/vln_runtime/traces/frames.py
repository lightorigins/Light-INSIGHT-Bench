"""Persisting the frame a policy was shown, beside the trace that records it.

`RolloutTrace` has always reserved `steps[i].observation.frame_path` and no
producer has ever filled it. This module is that producer, and it fixes the
meaning of the slot once, here, so nothing downstream has to reconstruct it:

    steps[i].observation.frame_path is the observation frame the policy was
    shown for the decision that step i records.

That is not the same pose as `steps[i].position`. The rollout hands the policy
the frame rendered *before* the action and records the pose reached *after* it,
so the camera that took `steps[i]`'s frame stood at `steps[i - 1].position`.
The half-step is in the trace already; writing the path into the step makes it
readable instead of leaving a reader to derive a filename from an index.

Numbering follows from that and lives only here. The rollout loop counts
decisions from 0; `RolloutTrace.from_camera_walk_records` records decision k as
trace step k + 1, because step 0 is the pre-action pose. So decision k is
written as ``step_{k + 1:04d}.jpg`` and ``step_0000.jpg`` never exists: step 0
made no decision, was shown no frame, and its `frame_path` stays empty.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from insight_bench._paths import safe_join
from insight_bench._redaction import redact_text

FrameSink = Callable[[str, int, Any], str]
"""(episode_id, decision index, RGB frame) -> the episode-relative frame path, or "".

An empty return is the normal "no frame for this step" answer -- strided out,
or unencodable -- and is what lands in the trace, so a step without a frame is
representable rather than pointing at a file that is not there.
"""


def frame_stride_for(num_steps: int, *, max_frames: int) -> int:
    """Persist every ``stride``-th decision, so no episode exceeds *max_frames*.

    A stride, rather than a byte budget spent first-come across the run: the
    footprint is then knowable before the run starts (and refusable, see
    ``VisSession``), and every episode keeps frames. A global budget stops
    writing partway through and starves the tail of the run, which is the part
    a user is triaging.
    """
    return max(1, math.ceil(max(1, int(num_steps)) / max(1, int(max_frames))))


@dataclass
class JpegFrameSink:
    """Writes ``<root>/<episode_id>/frames/step_NNNN.jpg`` and counts what happened.

    One sink per run, not per episode: the episode id arrives with the frame,
    the counters are the run's, and the run manifest reports them once.
    """

    root: Path
    stride: int = 1
    written: int = 0
    failed: int = 0
    first_failure: str = field(default="")

    def __call__(self, episode_id: str, step: int, frame: Any) -> str:
        if frame is None or step % self.stride:
            return ""
        relative = f"frames/step_{step + 1:04d}.jpg"
        # safe_join before anything is encoded, and deliberately outside the
        # try below: an episode id that is not a single path segment is a
        # dataset trying to choose where this run writes, which is refused
        # rather than counted as a frame that failed to encode. EpisodeSpec
        # rejects such ids at parse time too; this is the lock on the side that
        # touches the filesystem.
        destination = safe_join(self.root, f"{episode_id}/{relative}")
        try:
            # In the body: insight_bench.vln_runtime.policy.client imports
            # requests at module scope, and this module is reached from the
            # runners, which must import on a base install.
            from insight_bench.vln_runtime.policy.client import encode_rgb_jpeg

            payload = encode_rgb_jpeg(frame)
        except Exception as exc:  # a backend frame is not guaranteed to be encodable
            # Visualisation never decides whether an episode ran. A backend that
            # renders nothing usable, or a frame of an unexpected shape, costs
            # this step's picture and one line in the manifest.
            self.failed += 1
            if not self.first_failure:
                detail = redact_text(f"{type(exc).__name__}: {exc}")
                self.first_failure = detail or type(exc).__name__
            return ""
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        self.written += 1
        return relative
