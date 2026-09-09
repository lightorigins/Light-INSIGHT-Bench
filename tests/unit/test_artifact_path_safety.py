"""A dataset must not be able to choose where a run writes.

Episode ids name a directory under the artifact root, and for the gated suite the dataset is
user-supplied by design. An id of ``../../x`` or ``/etc/cron.d/x`` would otherwise reach
``write_json``, which does ``mkdir(parents=True, exist_ok=True)`` and then writes -- creating or
overwriting a file anywhere the process can reach.

Two locks, tested separately because either alone would be enough to regress quietly:
parsing refuses the id, and the write path refuses the join.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from insight_bench._paths import UnsafePathError, safe_join
from insight_bench.vln_runtime.episodes.schema import validate_episode_id

MALICIOUS_IDS = [
    "../../../escaped",
    "..",
    "/etc/cron.d/evil",
    "/tmp/evil",
    "a/b",
    "a\\b",
    "",
    ".",
    ".hidden",
    "with space",
    "semi;colon",
    "nul\x00byte",
]

BENIGN_IDS = ["objnav_v2_human_manual_8194nk5LbLH_49", "ep-0", "a.b_c-1", "EP42"]


class TestTheParsingLock:
    @pytest.mark.parametrize("episode_id", MALICIOUS_IDS)
    def test_an_unsafe_id_is_refused_at_parse_time(self, episode_id):
        with pytest.raises(ValueError, match="unsafe episode_id"):
            validate_episode_id(episode_id)

    @pytest.mark.parametrize("episode_id", BENIGN_IDS)
    def test_real_ids_still_parse(self, episode_id):
        # A lock that also refuses the ids the bundled suites actually use is not a fix.
        assert validate_episode_id(episode_id) == episode_id

    def test_every_published_episode_id_is_accepted(self):
        """The lock has to pass the ids the product actually evaluates.

        The objnav suite's episode file is licence-gated and not shipped, so
        this reads the published-format records the runner gate runs against --
        the same publisher, the same id vocabulary, and the ids a real run would
        turn into directory names under the artifact root.
        """
        fixture = (
            Path(__file__).resolve().parents[1] / "fixtures" / "objnav_published_episodes.jsonl"
        )
        records = [json.loads(line) for line in fixture.read_text(encoding="utf-8").splitlines()]
        assert records, "no published records to check the lock against"
        for record in records:
            episode_id = record["episode_id"]
            assert validate_episode_id(episode_id) == episode_id

    def test_an_episode_spec_carrying_a_malicious_id_does_not_load(self):
        from insight_bench.vln_runtime.episodes.schema import EpisodeSpec

        payload = {
            "episode_id": "../../../escaped",
            "instruction": "go",
            "scene": {"scene_id": "s", "asset_path": "/scenes/s.usd"},
            "start_pose": {"position": [0.0, 0.0, 0.0], "yaw_rad": 0.0},
        }
        with pytest.raises(ValueError, match="unsafe episode_id"):
            EpisodeSpec.from_dict(payload)


class TestTheWriteLock:
    @pytest.mark.parametrize("episode_id", ["../../../escaped", "/etc/cron.d/evil", "..", "a\\b"])
    def test_the_join_refuses_to_leave_the_artifact_root(self, tmp_path, episode_id):
        with pytest.raises(UnsafePathError):
            safe_join(tmp_path, f"{episode_id}/trace.json")

    def test_a_benign_id_lands_exactly_where_documented(self, tmp_path):
        assert safe_join(tmp_path, "ep-0/trace.json") == tmp_path / "ep-0" / "trace.json"

    def test_nothing_is_written_outside_the_artifact_root(self, tmp_path):
        """The property that actually matters, checked by looking at the filesystem.

        Not "an exception was raised" -- that is the mechanism. This asserts the outcome: after
        attempting every malicious id, the sibling directory a traversal would land in is still
        empty and the artifact root gained nothing.
        """
        from insight_bench.vln_runtime.traces.writer import write_rollout_trace

        root = tmp_path / "artifacts"
        root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()

        from insight_bench.vln_runtime.traces.schema import RolloutTrace

        trace = RolloutTrace.from_camera_walk_records(
            episode_id="probe",
            instruction="go",
            records=[
                {
                    "step": 0,
                    "pose_before": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
                    "pose_after": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
                    "measures": {},
                }
            ],
        )
        for episode_id in ("../outside/pwned", "../../outside/pwned", "/tmp/pwned"):
            with pytest.raises(UnsafePathError):
                write_rollout_trace(safe_join(root, f"{episode_id}/trace.json"), trace)

        assert list(outside.iterdir()) == [], "a traversal reached outside the artifact root"
        assert list(root.iterdir()) == [], "a refused write still created something"
