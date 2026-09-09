"""``insight-bench vis``: what the page says, and what it refuses to say.

The page itself is asserted structurally rather than against a whole-file
golden. A golden HTML would have to be rewritten for every class name, which
would make the one thing worth pinning -- the escaping, the arithmetic and the
degradation behaviour -- indistinguishable from cosmetics.

Every fixture here is written by these tests. Nothing is copied from any real
run or any real service payload.
"""

from __future__ import annotations

import json
import math
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest

from insight_bench import vis
from insight_bench.cli import main

# The strings an attacker controls. `raw_output` is the worst of them: it is a
# third-party policy server's verbatim reply, echoed into the trace.
HOSTILE_SCRIPT = "</script><script>window.PWNED=1</script>"
HOSTILE_ATTR = '"><img src=x onerror=alert(1)>'
HOSTILE_COMMENT = "--><svg onload=alert(1)>"


def _jpeg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xd8\xff\xd9")


def _step(index: int, *, frame: str = "", **extra: Any) -> dict[str, Any]:
    step: dict[str, Any] = {
        "step": index,
        "time_s": float(index),
        "position": [float(index), float(index) * 0.5, 0.0],
        "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "action": {"selected_waypoint": {"forward_m": 0.4, "lateral_m": 0.0, "yaw_rad": 0.0}},
        "measures": {"distance_to_goal": 5.0 - index},
        "observation": {"frame_path": frame},
        "metadata": {"raw_output": "", "apos_id": None, "apos_xy": None},
    }
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(step.get(key), dict):
            step[key] = {**step[key], **value}
        else:
            step[key] = value
    return step


def _trace(episode_id: str, steps: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    payload = {
        "episode_id": episode_id,
        "instruction": "go to the table",
        "steps": steps,
        "final_measures": {},
        "termination_reason": "stop",
        "stop_step": 3,
        "metadata": {},
    }
    payload.update(extra)
    return payload


def _episode(
    episode_id: str,
    status: str,
    *,
    metrics: dict[str, float] | None = None,
    failure_reason: str = "",
    error: str | None = None,
    labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "episode_id": episode_id,
        "task_id": "objnav",
        "status": status,
        "metrics": metrics or {},
    }
    if error is not None:
        entry["error"] = error
    if status != "error":
        entry["response"] = {
            "schema_version": "1.0.0",
            "request_id": "req-0123456789abcdef",
            "adapter": {"adapter_id": "http-policy", "model_id": "demo/model"},
            "status": "success",
            "output": "stop",
            "metadata": {"failure_reason": failure_reason, "score_labels": labels or {}},
        }
    return entry


def _run_result(episodes: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "run_id": "run-0123456789abcdef",
        "run_fingerprint": "a" * 64,
        "benchmark_id": "insight-bench-v1",
        "benchmark_version": "1.0.0",
        "adapter": {"adapter_id": "http-policy", "model_id": "demo/model"},
        "seed": 0,
        "status": "partial",
        "episodes": episodes,
        "metrics": {"success-rate": 0.25, "spl": 0.4, "distance_to_goal": 1.2},
        "runtime_attestation": {
            "backend_id": "isaac-5.1",
            "backend_status": "official",
            "launcher_id": "native",
            "runtime_kind": "native",
            "os": "linux",
            "driver_version": "570.86.15",
            "publishable": True,
        },
    }
    payload.update(extra)
    return payload


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_run(root: Path, *, manifest: bool = True) -> Path:
    """A four-episode run with every degradation case the page has to survive."""
    root.mkdir(parents=True, exist_ok=True)
    episodes = [
        _episode("ep-error", "error", error=f"policy said {HOSTILE_SCRIPT}"),
        _episode(
            "ep-near-no-stop",
            "failed",
            metrics={"distance_to_goal": 0.5, "stopped": 0.0, "spl": 0.0, "path_length": 2.0},
            failure_reason=HOSTILE_ATTR,
            labels={"capability": "LR_towards"},
        ),
        _episode(
            "ep-passed",
            "passed",
            metrics={"distance_to_goal": 0.8, "stopped": 1.0, "spl": 0.7, "path_length": 5.0},
            labels={"capability": "VC_between"},
        ),
        _episode(
            "ep-nonfinite",
            "failed",
            metrics={"distance_to_goal": 2.5, "stopped": 1.0, "spl": 0.0, "path_length": 9.0},
            labels={"capability": "LR_towards"},
        ),
    ]
    _write(root / "run-result.json", _run_result(episodes))

    sidecars = {
        "ep-error": {"suite": "hm3d", "instr_type": "Direction", "scene_class": "House"},
        "ep-near-no-stop": {
            "suite": "hm3d",
            "instr_type": HOSTILE_COMMENT,
            "scene_class": "Outdoor",
        },
        "ep-passed": {"suite": "outdoor_batch", "instr_type": "Base", "scene_class": "Commercial"},
    }
    for episode_id, extra in sidecars.items():
        _write(
            root / episode_id / "episode.json",
            {
                "episode_id": episode_id,
                "instruction": f"walk to the {HOSTILE_ATTR}",
                "scene_id": "scene56",
                "goal_position": [4.0, 2.0, 0.0],
                "goal_radius_m": 2.0,
                "reference_path": [[0.0, 0.0, 0.0], [4.0, 2.0, 0.0]],
                "difficulty_label": "LR_towards",
                **extra,
            },
        )
    # ep-nonfinite has a sidecar with no category keys at all: it is what lands
    # in the Unknown bucket, and it keeps scene_id constant across the run.
    _write(
        root / "ep-nonfinite" / "episode.json",
        {"episode_id": "ep-nonfinite", "scene_id": "scene56", "suite": "hm3d"},
    )

    _write(
        root / "ep-error" / "trace.json",
        _trace("ep-error", [_step(0), _step(1), _step(2)]),
    )
    _write(
        root / "ep-near-no-stop" / "trace.json",
        _trace(
            "ep-near-no-stop",
            [
                _step(0),
                _step(1, frame="../../etc/passwd"),
                _step(2, frame="/abs/x.jpg"),
                _step(3, frame="a\\b.jpg"),
                _step(4, frame="frames/step_0004.jpg"),
            ],
        ),
    )
    _write(
        root / "ep-passed" / "trace.json",
        _trace(
            "ep-passed",
            [
                _step(0),
                _step(
                    1,
                    frame="frames/step_0001.jpg",
                    metadata={"raw_output": HOSTILE_SCRIPT, "apos_id": 1273, "apos_kind": 0},
                ),
                _step(2, frame="frames/step_0002.jpg", metadata={"apos_xy": [812, 344]}),
                _step(3, frame="frames/step_0003.jpg"),
            ],
        ),
    )
    for index in (1, 2, 3):
        _jpeg(root / "ep-passed" / "frames" / f"step_{index:04d}.jpg")

    # Non-finite values at three depths, exactly as a bare json.dumps writes them.
    nonfinite = json.dumps(
        _trace(
            "ep-nonfinite",
            [_step(0), _step(1), _step(2)],
            final_measures={"nested": {"oracle": float("inf"), "list": [float("nan"), 1.0]}},
        )
    ).replace('"distance_to_goal": 4.0', '"distance_to_goal": -Infinity')
    (root / "ep-nonfinite").mkdir(parents=True, exist_ok=True)
    (root / "ep-nonfinite" / "trace.json").write_text(nonfinite, encoding="utf-8")

    if manifest:
        _write(
            root / "vis-manifest.json",
            {
                "schema": "insight-bench/vis-manifest/1",
                "episode_count": 4,
                "frame_stride": 1,
                "max_frames_per_episode": 100,
                "frames_written": 3,
                "frames_failed": 0,
                "frame_failure_reason": "",
            },
        )
    return root


def _render(root: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    summary = vis.render_run_page(root)
    html = (root / "vis.html").read_text(encoding="utf-8")
    return summary, html, _embedded(html)


def _blob(html: str) -> str:
    """The embedded JSON document, verbatim, as it sits in the file."""
    marker = '<script type="application/json" id="insight-bench-data">'
    start = html.index(marker) + len(marker)
    return html[start : html.index("</script>", start)]


def _embedded(html: str) -> dict[str, Any]:
    payload = json.loads(_blob(html))
    assert isinstance(payload, dict)
    return payload


class _Tags(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[str] = []

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        self.tags.append(tag)


# --------------------------------------------------------------------------- #
# Injection and escaping
# --------------------------------------------------------------------------- #


def test_hostile_strings_cannot_open_a_script_or_escape_an_attribute(tmp_path: Path) -> None:
    _build_run(tmp_path / "run")
    _, html, data = _render(tmp_path / "run")

    parser = _Tags()
    parser.feed(html)
    # Two script elements exist by design: the JSON block and the page's own
    # code. A third would mean a payload broke out of the block.
    assert parser.tags.count("script") == 2
    assert "</script><script>" not in html
    assert "PWNED" in json.dumps(data), "the hostile string must survive as data"
    # Inside the JSON block a payload is inert text; what matters is that none
    # of it appears anywhere markup or code is parsed.
    markup = html.replace(_blob(html), "")
    assert " onerror=" not in markup
    assert " onload=" not in markup
    assert "<foreignObject" not in markup
    for token in ("innerHTML", "eval(", "new Function", "document.write"):
        assert token not in html, token
    # The category value is shown, not dropped: a radar axis carries it verbatim.
    labels = [axis["label"] for axis in data["breakdown"]["instr_type"]]
    assert HOSTILE_COMMENT in labels


def test_every_untrusted_field_reaches_the_page_escaped(tmp_path: Path) -> None:
    _build_run(tmp_path / "run")
    _, html, _ = _render(tmp_path / "run")
    for raw in ("<script", "</script", "<img", "<svg onload"):
        assert raw not in _blob(html)
    assert "\\u003c" in html


def test_the_json_block_survives_a_round_trip_through_json_parse_rules() -> None:
    separators = "\u2028\u2029"
    text = vis._embed_json({"a": "</script>", "b": separators, "c": "&"})
    assert "<" not in text and ">" not in text and "&" not in text
    assert separators not in text
    assert "\\u2028" in text and "\\u2029" in text
    decoded = json.loads(
        text.replace("\\u003c", "<").replace("\\u003e", ">").replace("\\u0026", "&")
    )
    assert decoded == {"a": "</script>", "b": separators, "c": "&"}


# --------------------------------------------------------------------------- #
# Non-finite values
# --------------------------------------------------------------------------- #


def test_non_finite_measures_are_nulled_at_every_depth(tmp_path: Path) -> None:
    root = _build_run(tmp_path / "run")
    summary, html, data = _render(root)
    assert summary["episodes"] == 4
    for token in ("Infinity", "NaN"):
        assert token not in _blob(html), token
    # The per-step `dtg` column this used to inspect is no longer in the payload.
    # What the test is for -- that no `Infinity` token survives into the page at
    # any depth -- is the assertion above; the recursion itself is covered by
    # test_finite_recurses_into_dicts_and_lists.
    assert data["replay"]["ep-nonfinite"]["step"] == [1, 2]


def test_finite_recurses_into_dicts_and_lists() -> None:
    payload = {
        "measures": {"dtg": float("inf")},
        "final_measures": {"nested": [float("nan"), {"deep": float("-inf")}, 2.0]},
        "flag": True,
    }
    assert vis._finite(payload) == {
        "measures": {"dtg": None},
        "final_measures": {"nested": [None, {"deep": None}, 2.0]},
        "flag": True,
    }


# --------------------------------------------------------------------------- #
# Arithmetic
# --------------------------------------------------------------------------- #


def _distanceless_run(root: Path) -> Path:
    """A run whose episodes were scored but never measured a navigation error.

    `vis` renders whatever run-result file it is pointed at, and a result whose
    episodes carry no ``distance_to_goal`` is one it must survive without
    inventing numbers.
    """
    root.mkdir(parents=True, exist_ok=True)
    episodes = [
        _episode("ep-a", "passed", metrics={"exact-match": 1.0}),
        _episode("ep-b", "passed", metrics={"exact-match": 1.0}),
    ]
    _write(
        root / "run-result.json",
        _run_result(episodes, status="completed", metrics={"exact-match": 1.0}),
    )
    return root


def test_a_run_that_measured_no_navigation_error_still_renders(tmp_path: Path) -> None:
    """`vis` renders whatever result it is pointed at, inventing no numbers.

    A benchmark that scores episodes without ever measuring a distance to the
    goal used to be the case that made the SR@Km cards read a measured ``0.0%``
    on a run that scored 2/2; the cards are gone, and what has to hold now is
    that the page renders at all, that the column with nothing in it is absent
    rather than a row of em-dashes, and that the run's own metric keys are
    nowhere on the page because no section shows them any more.
    """
    root = _distanceless_run(tmp_path / "run")
    _, html, data = _render(root)
    assert data["derived"] == {"success_rate": 1.0, "successes": 2, "total": 2}
    assert "distance_to_goal" not in {column["key"] for column in data["columns"]}
    assert "metrics" not in data["run"]
    assert "exact-match" not in html
    # The table is still a table: two episodes, both passed.
    assert [row["status"] for row in data["episodes"]] == ["passed", "passed"]


def test_the_bench_line_is_the_recorded_coordinate_and_never_a_rename(
    tmp_path: Path,
) -> None:
    """The page shows the coordinate the run recorded, not today's.

    A run made before a benchmark was renamed records the retired coordinate,
    and re-labelling it -- on the page, or worse in the file -- would falsify a
    measurement record. So the page joins ``benchmark_id`` and
    ``benchmark_version`` and prints the join: no lookup table, no alias map,
    nothing that could turn a retired coordinate into the current one.
    """
    root = tmp_path / "legacy"
    root.mkdir()
    _write(
        root / "run-result.json",
        _run_result(
            [_episode("ep-a", "passed")],
            benchmark_id="retired-suite-v0",
            benchmark_version="0.4.0",
        ),
    )
    _, _, data = _render(root)
    assert data["run"]["benchmark"] == "retired-suite-v0@0.4.0"

    # A result recording the current coordinate goes through the same one path.
    _, _, data = _render(_build_run(tmp_path / "current"))
    assert data["run"]["benchmark"] == "insight-bench-v1@1.0.0"

    # And an id with no version reads as the id, not as an id@ with nothing after.
    assert vis._coordinate({"benchmark_id": "x", "benchmark_version": ""}) == "x"
    assert vis._coordinate({}) == ""


def test_the_metrics_section_is_gone_from_the_page_and_the_payload(tmp_path: Path) -> None:
    """The METRICS row of cards was removed, and the removal is total.

    The section carried six cards -- ``SR_task`` with a Wilson interval,
    ``SR@1m/2m/3m``, ``SPL`` and ``NE`` -- five of them restating
    ``RunResult.metrics`` under display names and two computing something the
    contract does not carry. The owner found the row not worth its space. So the
    arithmetic behind it went with it, and this asserts that: no card, no
    section, no payload key nobody reads, and no dead helper left in
    :mod:`insight_bench.vis` to make it look retrievable by flipping a flag.

    What this costs is worth naming. SR@1m, SR@2m, SR@3m, SPL, NE, the
    confidence interval and the run's ``metrics`` dict appear nowhere on the page
    now. The overall success rate survives only where the breakdown draws it --
    the dashed radar ring and the matrix's bottom-right total -- so a run
    recorded without ``--vis``, which has no breakdown, shows no success rate at
    all. Every one of those numbers is still in ``run-result.json``, which this
    command reads and never writes.
    """
    from insight_bench.vis_page import PAGE_TEMPLATE

    recorded = {
        "success-rate": 0.5,
        "spl": 0.4,
        "distance_to_goal": 1.2,
        "path_length": 5.6,
        "stop_step": 57.0,
        "stopped": 1.0,
        "success": 0.45,
    }
    root = tmp_path / "run"
    root.mkdir()
    _write(root / "run-result.json", _run_result([_episode("ep-a", "passed")], metrics=recorded))
    _, html, data = _render(root)

    # Nothing builds the section any more.
    for token in ('id="kpi"', "renderKpi", "METRIC_CARDS", "function kpi(", ".kpis {"):
        assert token not in PAGE_TEMPLATE, token
    # And nothing it used to print is on the page, as chrome or as data.
    for token in ("SR_task", "95% CI", "SR@", "Metrics", "success-rate"):
        assert token not in html, token

    # No payload key nobody reads: the metrics echo, the interval, the fixed
    # radii and their threshold list are all gone.
    assert "metrics" not in data["run"]
    assert set(data["derived"]) == {"success_rate", "successes", "total"}
    assert "sr_thresholds_m" not in data
    # `derived.total` is the count on the identity line; the k/n it comes from is
    # the run's own episode summary and one subtraction from it.
    assert data["derived"] == {"success_rate": 1.0, "successes": 1, "total": 1}

    # And no orphaned arithmetic in the module, which would leave the section
    # looking like it is one flag away from coming back.
    for gone in (
        "_wilson_ci",
        "WILSON_Z",
        "_sr_at_thresholds",
        "_sr_key",
        "_stopped",
        "SR_THRESHOLDS_M",
    ):
        assert not hasattr(vis, gone), gone

    # The record is untouched: `vis` reads run-result.json and never writes it,
    # so every key the cards used to show is still on disk beside the page.
    on_disk = json.loads((root / "run-result.json").read_text(encoding="utf-8"))
    assert on_disk["metrics"] == recorded


# --------------------------------------------------------------------------- #
# Frame paths
# --------------------------------------------------------------------------- #


def test_unsafe_and_missing_frame_paths_are_skipped_and_reported(tmp_path: Path) -> None:
    root = _build_run(tmp_path / "run")
    summary, html, data = _render(root)
    for bad in ("../../etc/passwd", "/abs/x.jpg", "a\\b.jpg", "frames/step_0004.jpg"):
        assert bad not in html, bad
    assert any("frame paths failed validation" in line for line in summary["warnings"])
    assert data["replay"]["ep-near-no-stop"].get("frame") is None
    assert summary["frames"] == 3
    assert summary["episodes_with_frames"] == 1


def test_a_frame_reference_is_the_recorded_path_not_one_rebuilt_from_the_index(
    tmp_path: Path,
) -> None:
    root = _build_run(tmp_path / "run")
    _, _, data = _render(root)
    frames = data["replay"]["ep-passed"]["frame"]
    assert frames == [
        "ep-passed/frames/step_0001.jpg",
        "ep-passed/frames/step_0002.jpg",
        "ep-passed/frames/step_0003.jpg",
    ]
    steps = data["replay"]["ep-passed"]["step"]
    assert steps == [1, 2, 3], "step 0 is the pre-action pose and is never a replay state"


def test_a_hostile_episode_id_never_becomes_a_path(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    _write(root / "run-result.json", _run_result([_episode("../../evil", "failed")]))
    summary, html, data = _render(root)
    assert summary["episodes"] == 1
    assert data["episodes"][0]["frames"] is None
    assert not (tmp_path / "evil").exists()
    assert 'src="../../evil' not in html


# --------------------------------------------------------------------------- #
# Degradation
# --------------------------------------------------------------------------- #


def test_a_run_without_vis_still_renders_kpis_a_table_and_a_trajectory(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _build_run(root, manifest=False)
    for episode_id in ("ep-error", "ep-near-no-stop", "ep-passed", "ep-nonfinite"):
        (root / episode_id / "episode.json").unlink()
    summary, html, data = _render(root)
    assert summary["episodes_with_categories"] == 0
    assert data["breakdown"] is None
    assert any("recorded without --vis" in line for line in summary["warnings"])
    assert "needs a run with --vis" in html.lower()
    # The trajectory panel needs nothing but the trace, so replay still exists.
    assert data["replay"]["ep-passed"]["to"]


def test_a_missing_run_result_names_the_paths_it_probed(tmp_path: Path) -> None:
    (tmp_path / "run").mkdir()
    with pytest.raises(vis.VisError, match="probed"):
        vis.render_run_page(tmp_path / "run")


def test_a_run_result_one_level_up_is_found(tmp_path: Path) -> None:
    root = _build_run(tmp_path / "runs")
    (tmp_path / "run-result.json").write_bytes((root / "run-result.json").read_bytes())
    (root / "run-result.json").unlink()
    summary, _, _ = _render(root)
    assert summary["episodes"] == 4


# --------------------------------------------------------------------------- #
# Which run result the page is built from
# --------------------------------------------------------------------------- #


def _named_run(root: Path, name: str, model_id: str) -> Path:
    """A minimal one-episode run result under a chosen file name."""
    payload = _run_result(
        [_episode("ep-a", "passed")],
        adapter={"adapter_id": "http-policy", "model_id": model_id},
    )
    _write(root / name, payload)
    return root / name


def test_a_run_result_under_a_non_default_name_beats_one_in_the_parent(tmp_path: Path) -> None:
    """The silent-substitution case: `--output` is configurable, so a directory
    holding exactly one run result under a chosen name must render *that* run,
    never a neighbouring ``run-result.json`` one level up."""
    _named_run(tmp_path, "run-result.json", "someone-else/model")
    root = tmp_path / "runs"
    root.mkdir()
    _named_run(root, "my-run-result.json", "me/my-model")

    summary, html, data = _render(root)

    assert summary["run_result"] == "my-run-result.json"
    assert data["run"]["source"] == "my-run-result.json"
    assert data["run"]["model_id"] == "me/my-model"
    assert "someone-else/model" not in html


def test_the_default_name_still_wins_over_another_run_result_beside_it(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    _named_run(root, "run-result.json", "default/model")
    _named_run(root, "other-run-result.json", "other/model")
    summary, _, data = _render(root)
    assert summary["run_result"] == "run-result.json"
    assert data["run"]["model_id"] == "default/model"


def test_two_run_results_under_chosen_names_are_refused_rather_than_guessed_at(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    _named_run(root, "a-run.json", "a/model")
    _named_run(root, "b-run.json", "b/model")
    with pytest.raises(vis.VisError, match="ambiguous"):
        vis.render_run_page(root)


def test_the_parent_fallback_warns_on_stdout_and_on_the_page(tmp_path: Path) -> None:
    root = _build_run(tmp_path / "runs")
    (tmp_path / "run-result.json").write_bytes((root / "run-result.json").read_bytes())
    (root / "run-result.json").unlink()
    summary, _, data = _render(root)
    assert any("parent directory" in line for line in summary["warnings"])
    assert summary["warnings"] == data["warnings"][: len(summary["warnings"])]
    assert summary["run_result"] == str(tmp_path / "run-result.json")
    assert data["run"]["source"] == summary["run_result"]


def test_a_run_result_file_can_be_named_directly_and_writes_beside_itself(
    tmp_path: Path,
) -> None:
    root = _build_run(tmp_path / "runs")
    (root / "chosen.json").write_bytes((root / "run-result.json").read_bytes())
    (root / "run-result.json").unlink()
    summary = vis.render_run_page(root / "chosen.json")
    assert summary["run_result"] == "chosen.json"
    assert summary["episodes"] == 4
    assert (root / "vis.html").is_file()


def test_broken_json_beside_a_run_result_does_not_stop_the_good_one(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    (root / "notes.json").write_text("{not json", encoding="utf-8")
    (root / "config.json").write_text('{"unrelated": true}', encoding="utf-8")
    _named_run(root, "my-run.json", "me/my-model")
    summary, _, data = _render(root)
    assert summary["run_result"] == "my-run.json"
    assert data["run"]["model_id"] == "me/my-model"


def test_an_evidence_pack_is_refused_because_verify_would_fail(tmp_path: Path) -> None:
    root = _build_run(tmp_path / "run")
    (root / "evidence-manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(vis.VisError, match="evidence pack"):
        vis.render_run_page(root)


def test_the_run_result_inside_an_evidence_pack_is_refused_too(tmp_path: Path) -> None:
    """Naming the file must not walk past the pack guard: `vis` writes vis.html
    into the directory it reads, and that extra artifact makes `verify` fail."""
    root = _build_run(tmp_path / "pack")
    (root / "evidence-manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(vis.VisError, match="evidence pack"):
        vis.render_run_page(root / "run-result.json")
    assert not (root / "vis.html").exists()


def test_a_directory_that_is_not_one_is_refused(tmp_path: Path) -> None:
    with pytest.raises(vis.VisError, match="not a directory"):
        vis.render_run_page(tmp_path / "nope")


# --------------------------------------------------------------------------- #
# Read-side limits
# --------------------------------------------------------------------------- #


def test_replay_is_capped_while_the_table_and_the_breakdown_stay_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the cap does bite, it bites only the replay -- and it says so.

    The cap is patched down rather than exceeded: the real one is a page-weight
    budget in the thousands, and building that many episode directories to prove
    a slice would cost a minute of test time to assert the same thing.
    """
    monkeypatch.setattr(vis, "VIS_MAX_REPLAY_EPISODES", 20)
    root = tmp_path / "big"
    root.mkdir()
    count = vis.VIS_MAX_REPLAY_EPISODES + 5
    episodes = []
    for index in range(count):
        status = "passed" if index % 2 else "failed"
        episodes.append(_episode(f"ep-{index:04d}", status, metrics={"spl": 0.5}))
        episode_id = f"ep-{index:04d}"
        _write(root / episode_id / "trace.json", _trace(episode_id, [_step(0), _step(1)]))
        _write(
            root / f"ep-{index:04d}" / "episode.json",
            {
                "episode_id": f"ep-{index:04d}",
                "suite": "hm3d",
                "instr_type": "Base",
                "scene_class": "Outdoor",
            },
        )
    _write(root / "run-result.json", _run_result(episodes))

    opened: list[str] = []
    real = vis.load_json

    def counting(path: Path) -> Any:
        opened.append(path.name)
        return real(path)

    monkeypatch.setattr(vis, "load_json", counting)
    summary, html, data = _render(root)
    assert opened.count("trace.json") == vis.VIS_MAX_REPLAY_EPISODES
    assert opened.count("episode.json") == count
    assert len(data["replay"]) == vis.VIS_MAX_REPLAY_EPISODES
    assert len(data["episodes"]) == count
    assert data["breakdown"]["instr_type"][0]["episodes"] == count
    assert any("replay data is inlined for 20 of 25" in line for line in summary["warnings"])
    assert "episodes carry replay" in html


def test_a_run_inside_the_cap_gets_no_cap_warning_at_all(tmp_path: Path) -> None:
    """No cap applied means no sentence about a cap.

    The warning is the page's only account of why an episode has no replay, so
    it has to be absent when the reason does not exist -- a page that warns about
    a limit it did not hit teaches its reader to ignore the warnings.
    """
    root = tmp_path / "small"
    root.mkdir()
    count = 12
    assert count < vis.VIS_MAX_REPLAY_EPISODES
    episodes = []
    for index in range(count):
        episode_id = f"ep-{index:04d}"
        episodes.append(_episode(episode_id, "passed", metrics={"spl": 0.5}))
        _write(root / episode_id / "trace.json", _trace(episode_id, [_step(0), _step(1)]))
    _write(root / "run-result.json", _run_result(episodes))
    summary, _, data = _render(root)
    assert len(data["replay"]) == count
    assert not any("page limit" in line for line in summary["warnings"])
    assert not any("replay data is inlined for" in line for line in summary["warnings"])


def test_replay_is_handed_out_error_then_failed_then_passed(tmp_path: Path) -> None:
    root = tmp_path / "order"
    root.mkdir()
    episodes = [
        _episode("z-error", "error", error="boom"),
        _episode("a-passed", "passed"),
        _episode("m-failed", "failed"),
    ]
    _write(root / "run-result.json", _run_result(episodes))
    for episode_id in ("z-error", "a-passed", "m-failed"):
        _write(root / episode_id / "trace.json", _trace(episode_id, [_step(0), _step(1)]))
    monkey = vis.VIS_MAX_REPLAY_EPISODES
    try:
        vis.VIS_MAX_REPLAY_EPISODES = 2  # type: ignore[misc]
        _, _, data = _render(root)
    finally:
        vis.VIS_MAX_REPLAY_EPISODES = monkey  # type: ignore[misc]
    assert set(data["replay"]) == {"z-error", "m-failed"}


def test_a_long_episode_is_strided_down_to_the_state_cap(tmp_path: Path) -> None:
    root = tmp_path / "long"
    root.mkdir()
    _write(root / "run-result.json", _run_result([_episode("ep-long", "passed")]))
    steps = [_step(index) for index in range(301)]
    _write(root / "ep-long" / "trace.json", _trace("ep-long", steps))
    _, _, data = _render(root)
    entry = data["replay"]["ep-long"]
    assert entry["stride"] == 3
    assert len(entry["step"]) == vis.VIS_MAX_REPLAY_STATES
    assert entry["step"][:3] == [1, 4, 7]
    assert entry["total_steps"] == 301


def test_a_recorded_stride_is_reused_rather_than_recomputed(tmp_path: Path) -> None:
    root = tmp_path / "strided"
    root.mkdir()
    _write(root / "run-result.json", _run_result([_episode("ep-a", "passed")]))
    _write(root / "ep-a" / "trace.json", _trace("ep-a", [_step(index) for index in range(11)]))
    _write(root / "vis-manifest.json", {"frame_stride": 5})
    _, _, data = _render(root)
    assert data["replay"]["ep-a"]["step"] == [1, 6]


def test_a_free_text_value_off_disk_is_clipped() -> None:
    """Unbounded text a run wrote must not decide the page's size.

    This bounded the policy's ``raw_output`` while the frame panel printed it;
    what still reaches the page through it is the manifest's own skip reasons.
    """
    clipped = vis._clip("x" * 5000)

    assert len(clipped) == vis.VIS_MAX_CLIPPED_CHARS + 3
    assert clipped.endswith("...")
    assert vis._clip("short") == "short"


# --------------------------------------------------------------------------- #
# Columns
# --------------------------------------------------------------------------- #


def test_a_column_is_rendered_only_when_its_value_varies(tmp_path: Path) -> None:
    root = _build_run(tmp_path / "run")
    _, _, data = _render(root)
    keys = {column["key"] for column in data["columns"]}
    assert "instr_type" in keys, "three instruction types appear in this run"
    assert "scene_class" in keys, "three scene classes appear in this run"
    assert "scene_id" not in keys, "one scene id everywhere carries no information"


def test_the_episode_table_matches_the_internal_browser_column_for_column() -> None:
    """Order, labels and filters follow an existing internal episode browser.

    A reader who uses both should not be re-learning the table, so the order is
    pinned here rather than left to whatever the tuple happens to say. ``#``,
    ``Episode`` and ``Frames`` belong to the page skeleton; everything between
    them comes from this spec, which is why ``Result`` is no longer hard-coded
    into third place.
    """
    assert [(key, label) for key, label, _, _ in vis._CANDIDATE_COLUMNS] == [
        ("scene_id", "Scene"),
        ("status", "Result"),
        ("instr_type", "Instr type"),
        ("scene_class", "Scene class"),
        ("distance_to_goal", "NE (m)"),
    ]
    # No Suite column: its values are the generator's batch names, which say how
    # an episode was written rather than anything a reader would filter on.
    assert not any(key == "suite" for key, _, _, _ in vis._CANDIDATE_COLUMNS)
    # A filter on every categorical column but the scene, which is
    # high-cardinality here (hundreds of values on a published run) and carries
    # no filter in the browser this follows either. No filter on a number.
    assert {key for key, _, _, on in vis._CANDIDATE_COLUMNS if on} == {
        "status",
        "instr_type",
        "scene_class",
    }
    assert not any(on for _, _, source, on in vis._CANDIDATE_COLUMNS if source == "metric")
    # Instruction type and scene class stand where that browser puts difficulty,
    # which on this benchmark is one string on every row. These two are the axes
    # of the radars and the matrix above the table, so filtering on them is what
    # takes a reader from a matrix cell to the episodes in it.
    assert "capability" not in {key for key, _, _, _ in vis._CANDIDATE_COLUMNS}


def test_the_result_column_stays_even_when_every_episode_agrees(tmp_path: Path) -> None:
    """A run in which everything passed still has to say so.

    Every other column is dropped when it carries one value, because a column of
    one value is a filter dropdown with one option. The result is what the table
    is for, so it is exempt.
    """
    root = tmp_path / "clean"
    root.mkdir()
    episodes = [_episode(f"ep-{index}", "passed", metrics={"spl": 1.0}) for index in range(3)]
    _write(root / "run-result.json", _run_result(episodes))
    _, _, data = _render(root)
    columns = {column["key"]: column for column in data["columns"]}
    assert columns["status"]["label"] == "Result"
    assert columns["status"]["filter"] is True
    # And it is read off the row rather than copied into every cell dict.
    assert "status" not in data["episodes"][0]["cells"]
    assert data["episodes"][0]["status"] == "passed"


def test_each_filterable_column_carries_its_filter_in_its_own_header(tmp_path: Path) -> None:
    """The filter affordance, and where it lives now.

    A checkbox dropdown per categorical column, every value's count taken over
    the whole run so the numbers beside the options do not move as you tick
    boxes, options in descending frequency, and nothing ticked and everything
    ticked as one state -- not filtering -- so ticking the last box clears the
    filter rather than leaving one that excludes nothing. All of that is
    unchanged. What moved is the control: it was a row of four dropdowns above
    the table, and it is now one control in each filterable column's header, so
    a reader narrows the column they are already looking at. The header carries
    the label, which is why the summary carries only the state -- ``all``, or how
    much of the column is getting through, readable without opening it.
    """
    root = _build_run(tmp_path / "run")
    _, html, data = _render(root)
    assert [column["key"] for column in data["columns"] if column["filter"]] == [
        "status",
        "instr_type",
        "scene_class",
    ]
    assert "function checkFilter(host, column, onChange) {" in html
    assert "function tally(column) {" in html
    assert "on.length === 0 || on.length === ticks.length" in html
    # In the header cell it belongs to, not in a row of its own above the table.
    assert "if (column.filter) { filters.push(checkFilter(th, column, applyFilters)); }" in html
    assert 'head.textContent = (on === null ? "all" : on.length + " of " + ticks.length)' in html
    assert "#filters details" not in html
    assert "#filters .opts" not in html
    assert "th .opts label" in html
    # What stays above the table: the id search and the count of what survived,
    # neither of which belongs to any one column.
    assert 'search.placeholder = "search episode id' in html
    assert 'f.count.textContent = visible.length + " of " + DATA.episodes.length + " shown"' in html


def test_every_column_sorts_in_three_states_and_puts_missing_values_last(
    tmp_path: Path,
) -> None:
    """Sorting, which this table deliberately did not have.

    It was left out to match an internal episode browser that has none, on the
    grounds that a header which looks clickable and is not is worse than one
    that does not. The owner asked for sorting, so that alignment no longer
    applies and every header is live: ascending, then descending, then back to
    the order the run result lists its episodes in -- which is what the table
    shows before anything is clicked, so the third click is an undo rather than a
    fourth state to find a way out of.

    Four properties are pinned here rather than left to the DOM. The cycle is
    three states and not four. A missing value sorts last in *both* directions,
    because an episode that recorded no distance is neither the nearest to the
    goal nor the furthest. The position, the frame count and the metrics sort as
    numbers and everything else as text. And the sort moves the rows it already
    has -- 1097 of them on a published run -- rather than rebuilding the table,
    which would drop the filters' hidden flags and the selected row with it.
    """
    root = _build_run(tmp_path / "run")
    _, html, _ = _render(root)
    assert (
        "sort.dir = sort.column === column ? (sort.dir === 1 ? -1 : sort.dir === -1 ? 0 : 1) : 1;"
    ) in html
    assert "if (x === null || y === null) { return x === y ? 0 : (x === null ? 1 : -1); }" in html
    assert 'return column.source === "pos" || column.source === "frames"' in html
    assert "if (!numeric) { return String(value).toLowerCase(); }" in html
    # The reorder: one fragment of existing rows, appended back in the new order.
    assert "var frag = document.createDocumentFragment();" in html
    assert "order.forEach(function (row) { frag.appendChild(row.node); });" in html
    assert "state.body.appendChild(frag);" in html
    # The state is on the header, in a box whose width does not change with the
    # glyph, so a header does not move as its sort state does.
    assert 'th.setAttribute("aria-sort", "none");' in html
    assert "th .ind { flex: none; width: 9px;" in html
    for glyph in ("\\u2191", "\\u2193", "\\u2195"):
        assert glyph in html, glyph


def test_a_column_resizes_from_its_edge_and_the_drag_never_sorts(tmp_path: Path) -> None:
    """Dragging the right edge of a header sets that column's width.

    The obvious bug in a header that both sorts and resizes is that the drag ends
    in a click on the header, which sorts the table the reader was only trying to
    widen. So the guard is on both: a click out of the handle or the filter never
    sorts, and a pointer that moved sets a flag the following click consumes. The
    flag is cleared by every press on a header, so a drag that ends somewhere
    else cannot leave it set and eat the next real click.

    The table is auto-laid-out until the first drag, so the page opens at the
    widths the content asked for; the measured widths are written down at that
    moment, which is what makes each later drag exact rather than a hint the auto
    layout may ignore. Widths are not persisted -- a reload is a reset.
    """
    root = _build_run(tmp_path / "run")
    _, html, _ = _render(root)
    assert "th .grip { position: absolute; top: 0; right: 0; width: 9px;" in html
    assert "cursor: col-resize" in html
    # The affordance: a bar under the pointer, and while dragging.
    assert "th .grip:hover::after, th .grip.on::after { background: var(--accent); }" in html
    assert "var MIN_COLUMN_PX = 46;" in html
    assert "th.style.width = Math.max(MIN_COLUMN_PX, width + (event.clientX - from))" in html
    assert 'table.style.tableLayout = "fixed";' in html
    # The drag never sorts.
    assert 'if (event.target.closest(".flt") || event.target.closest(".grip")) { return; }' in html
    assert "if (state.dragged) { state.dragged = false; return; }" in html
    assert 'th.addEventListener("pointerdown", function () { state.dragged = false; });' in html
    # And a release anywhere ends it, so no drag outlives the pointer.
    assert 'document.addEventListener("pointerup", done);' in html
    assert 'document.removeEventListener("pointermove", move);' in html


def _two_episode_run(root: Path, first: str, second: str) -> None:
    """Two episodes whose instruction type is ``first`` and ``second``."""
    root.mkdir(parents=True, exist_ok=True)
    _write(
        root / "run-result.json",
        _run_result([_episode("ep-a", "passed"), _episode("ep-b", "failed")]),
    )
    for episode_id, instr_type in (("ep-a", first), ("ep-b", second)):
        _write(
            root / episode_id / "episode.json",
            {"episode_id": episode_id, "instr_type": instr_type},
        )


def test_one_distinct_value_drops_the_column_and_two_bring_it_back(tmp_path: Path) -> None:
    _two_episode_run(tmp_path / "same", "Base", "Base")
    _, _, data = _render(tmp_path / "same")
    assert "instr_type" not in {column["key"] for column in data["columns"]}

    _two_episode_run(tmp_path / "differs", "Base", "Direction")
    _, _, data = _render(tmp_path / "differs")
    assert "instr_type" in {column["key"] for column in data["columns"]}


def test_one_value_plus_a_missing_one_is_a_constant_not_a_variable_column(
    tmp_path: Path,
) -> None:
    """A missing value is not a value, so it cannot make a constant look variable.

    This is how the published suite came to ship a Difficulty column: every
    scored episode read ``LR_towards`` and the run's single errored episode
    carried no score labels at all, so counting cells saw two "values" and kept
    a column that printed one string 1096 times behind a two-option filter. An
    errored episode is ordinary, so this has to be decided on present values.
    """
    root = tmp_path / "holed"
    root.mkdir()
    _write(
        root / "run-result.json",
        _run_result(
            [
                _episode("ep-a", "passed"),
                _episode("ep-b", "passed"),
                _episode("ep-err", "error", error="boom"),
            ]
        ),
    )
    for episode_id in ("ep-a", "ep-b"):
        _write(
            root / episode_id / "episode.json",
            {"episode_id": episode_id, "instr_type": "Base", "suite": "hm3d"},
        )
    _, _, data = _render(root)
    keys = {column["key"] for column in data["columns"]}
    assert "instr_type" not in keys, "one instruction type and one blank is a constant"
    assert "suite" not in keys
    # Result is exempt: it is what the table is for, and here it does vary.
    assert "status" in keys


def test_the_single_value_rule_holds_for_every_source_a_column_can_have() -> None:
    """The rule is on the value, not on where the value came from.

    Driven straight at :func:`_variable_columns` so all four source kinds are
    covered, including ``label``, which no shipped column currently uses but
    which the column spec still offers.
    """
    constant = [
        {
            "status": "passed",
            "metrics": {"spl": 0.5},
            "labels": {"capability": "LR_towards"},
            "sidecar": {"suite": "hm3d", "instr_type": "Base"},
        },
        # Same values, then one row that carries none of them.
        {
            "status": "passed",
            "metrics": {"spl": 0.5},
            "labels": {"capability": "LR_towards"},
            "sidecar": {"suite": "hm3d", "instr_type": "Base"},
        },
        {"status": "passed", "metrics": {}, "labels": {}, "sidecar": {}},
    ]
    for source, key in (("metric", "spl"), ("label", "capability"), ("episode", "suite")):
        values = {vis._column_value(row, key, source) for row in constant}
        assert len(values) == 2, (source, values)  # one real value plus a hole
    # A status is never missing, so this one is a plain constant.
    assert {vis._column_value(row, "status", "status") for row in constant} == {"passed"}
    # Both halves of the rule: the three constants-with-a-hole are dropped, and
    # the constant that is exempt is kept.
    assert [column["key"] for column in vis._variable_columns(constant)] == ["status"]
    assert set(vis._ALWAYS_COLUMNS) == {"status"}


# --------------------------------------------------------------------------- #
# Breakdown
# --------------------------------------------------------------------------- #


def test_per_category_counts_sum_to_the_run_totals(tmp_path: Path) -> None:
    root = _build_run(tmp_path / "run")
    _, _, data = _render(root)
    breakdown, derived = data["breakdown"], data["derived"]
    for axis_key in ("instr_type", "scene_class"):
        axes = breakdown[axis_key]
        assert sum(axis["episodes"] for axis in axes) == derived["total"], axis_key
        assert sum(axis["successes"] for axis in axes) == derived["successes"], axis_key
    matrix = breakdown["matrix"]
    cells = [cell for row in matrix["rows"] for cell in row["cells"]]
    assert sum(cell["episodes"] for cell in cells) == derived["total"]
    assert sum(cell["successes"] for cell in cells) == derived["successes"]
    assert sum(cell["episodes"] for cell in matrix["totals"]) == derived["total"]
    assert matrix["grand_total"]["episodes"] == derived["total"]
    assert matrix["grand_total"]["successes"] == derived["successes"]


def test_an_episode_without_category_keys_falls_into_unknown_and_unknown_is_last(
    tmp_path: Path,
) -> None:
    root = _build_run(tmp_path / "run")
    _, _, data = _render(root)
    axes = data["breakdown"]["instr_type"]
    assert axes[-1]["name"] == vis.UNKNOWN_CATEGORY
    assert axes[-1]["episodes"] == 1
    scene = data["breakdown"]["scene_class"]
    assert scene[-1]["name"] == vis.UNKNOWN_CATEGORY


def test_the_retired_category_values_map_onto_the_published_vocabulary() -> None:
    """A run directory from a retired file renders the words a current run does.

    The map is only for the retired ``1.0.0``/``2.0.0`` vocabulary; the current
    values are already their own display name. The targets are therefore the
    published words exactly -- "Institution", not "Institutional", and
    "Extremum", not "Extreme" -- so the two cannot disagree on a page.
    """
    for value, expected in (
        ("小住宅", "Apartment"),
        ("大住宅", "House"),
        ("商业", "Commercial"),
        ("公共机构", "Institution"),
        ("outdoor", "Outdoor"),
        ("extremum", "Extremum"),
    ):
        assert vis.CATEGORY_LABELS[value] == expected
        assert expected in set(vis.SCENE_CLASS_ORDER) | set(vis.INSTR_TYPE_ORDER)
    # An unmapped value is shown as it is, not guessed at and not dropped.
    assert vis.CATEGORY_LABELS.get("未映射", "未映射") == "未映射"
    for current in (*vis.SCENE_CLASS_ORDER, *vis.INSTR_TYPE_ORDER):
        assert vis.CATEGORY_LABELS.get(current, current) == current


def test_the_radar_edge_lifts_so_a_polyline_cannot_leave_the_canvas(tmp_path: Path) -> None:
    root = tmp_path / "high"
    root.mkdir()
    episodes = [_episode(f"ep-{index}", "passed") for index in range(9)]
    episodes.append(_episode("ep-9", "failed"))
    _write(root / "run-result.json", _run_result(episodes))
    for index in range(10):
        _write(
            root / f"ep-{index}" / "episode.json",
            {"episode_id": f"ep-{index}", "instr_type": "Base", "scene_class": "Outdoor"},
        )
    _, _, data = _render(root)
    breakdown = data["breakdown"]
    assert breakdown["edge"] == pytest.approx(0.9)
    for axis in breakdown["instr_type"] + breakdown["scene_class"]:
        assert 0.0 <= axis["frac"] <= 1.0
    assert 0.0 <= breakdown["reference_frac"] <= 1.0


def test_the_radar_edge_stays_at_the_default_when_nothing_exceeds_it(tmp_path: Path) -> None:
    root = tmp_path / "mid"
    root.mkdir()
    episodes = [_episode(f"ep-{index}", "passed" if index < 5 else "failed") for index in range(10)]
    _write(root / "run-result.json", _run_result(episodes))
    for index in range(10):
        _write(
            root / f"ep-{index}" / "episode.json",
            {"episode_id": f"ep-{index}", "instr_type": "Base", "scene_class": "Outdoor"},
        )
    _, _, data = _render(root)
    assert data["breakdown"]["edge"] == pytest.approx(vis.RADAR_EDGE_DEFAULT)


def test_the_diverging_scale_is_centred_and_capped() -> None:
    # The three anchors are the published figure's palette, so a page and
    # assets/insight_bench_matrix.png encode the same rate the same way.
    centre = vis._diverging(0.0)
    assert centre == "#ffffff"
    assert vis._diverging(0.20) == vis._diverging(0.5)
    assert vis._diverging(-0.20) == vis._diverging(-0.5)
    assert vis._diverging(0.20) == "#6a9ddf"
    assert vis._diverging(-0.20) == "#ed8179"
    assert vis._diverging(0.1) not in {centre, "#6a9ddf"}


def test_an_empty_category_cell_shows_no_rate_at_all(tmp_path: Path) -> None:
    root = _build_run(tmp_path / "run")
    _, _, data = _render(root)
    empty = [
        cell
        for row in data["breakdown"]["matrix"]["rows"]
        for cell in row["cells"]
        if cell["episodes"] == 0
    ]
    assert empty, "this fixture has combinations no episode covers"
    assert all(cell["rate"] is None for cell in empty)


def test_the_per_suite_group_summary_is_gone_from_the_payload_and_the_page(
    tmp_path: Path,
) -> None:
    """The Groups section was removed: every number it carried is elsewhere.

    Its SR per suite is one filter away in the episode table, and its SR@Km came
    from the same arithmetic the metric cards reported over the whole run -- both
    of which have since been removed as well. So this asserts the removal is
    total -- no payload key nobody reads, no empty section on the page -- rather
    than merely that the section is hidden.
    """
    from insight_bench.vis_page import PAGE_TEMPLATE

    root = _build_run(tmp_path / "run")
    _, html, data = _render(root)
    assert "groups" not in data
    assert 'id="groups"' not in PAGE_TEMPLATE
    assert "renderGroups" not in PAGE_TEMPLATE
    assert ">Groups<" not in html


# --------------------------------------------------------------------------- #
# The page as a file
# --------------------------------------------------------------------------- #


def test_the_page_references_nothing_remote(tmp_path: Path) -> None:
    root = _build_run(tmp_path / "run")
    _, html, _ = _render(root)
    # The SVG namespace is an identifier, never fetched. Nothing else in the
    # file may look like a URL at all.
    assert re.findall(r"https?://[^\s\"']+", html) == ["http://www.w3.org/2000/svg"]
    for token in ("//cdn", "fonts.googleapis", "xlink:href", "<link ", "@import", "url("):
        assert token not in html, token
    assert "<script src" not in html
    assert 'type="module"' not in html
    assert '<meta charset="utf-8">' in html
    assert 'http-equiv="Content-Security-Policy"' not in html


def test_the_page_contains_the_svg_charts_and_the_frame_panel(tmp_path: Path) -> None:
    root = _build_run(tmp_path / "run")
    _, html, _ = _render(root)
    for token in ("createElementNS", "polygon", "polyline"):
        assert token in html, token


def test_play_from_the_final_state_rewinds_instead_of_stopping(tmp_path: Path) -> None:
    """Play at the end has to mean "again, from the start".

    ``step()`` clamps the cursor at the last state and calls ``play(false)``
    there, so a play request made at the end used to start a timer that
    immediately re-reached the end and switched itself off -- one 250 ms glyph
    flicker per click, for ever. The rewind lives in ``play()`` rather than in
    the button, so the keyboard, the transport button and the click on the image
    all get it.
    """
    root = _build_run(tmp_path / "run")
    _, html, _ = _render(root)
    assert "if (on && entry && state.cursor >= entry.step.length - 1) {" in html
    # And the click on the image is not swallowed: the one overlay drawn over
    # the frame is transparent to the pointer, and the target advertises itself.
    # Nothing is layered over the image any more, so nothing can intercept the
    # click. That used to need `pointer-events: none` on the token overlay.
    assert ".shot img { width: 100%" in html
    assert "aspect-ratio: 16 / 9; cursor: pointer; }" in html
    assert 'shot.addEventListener("click", function () { play(!state.playing); });' in html


def test_the_unverified_pointing_marker_is_gone_and_the_tokens_are_not(
    tmp_path: Path,
) -> None:
    """The frame panel decoded apos/opos into image coordinates with a 48x27 grid
    convention that was never checked against a pointing policy server, so a
    marker could be confidently wrong about where the policy pointed. It is
    removed rather than switched off -- a switch is an invitation to flip it back.

    What replaces it is what the page can honestly say: the tokens themselves,
    verbatim, and the ``raw_output`` line beside them.
    """
    root = _build_run(tmp_path / "run")
    _, html, data = _render(root)
    for gone in (
        "decodePoint",
        "VIS_DRAW_POINT_MARKERS",
        "APOS_COLS",
        "APOS_ROWS",
        "POS_BIN_MAX",
        "getContext",
        "<canvas",
        'createElement("canvas")',
    ):
        assert gone not in html, gone
    # And the token values went with it. They were only ever printed to sit
    # beside the marker; with the marker and its caveat both gone, six bare
    # `apos_id=106`-style numbers over a camera image explain nothing.
    entry = data["replay"]["ep-passed"]
    for absent in ("apos_id", "apos_xy", "opos_id", "opos_xy", "raw", "dtg"):
        assert absent not in entry, absent
    for gone in (
        'parts.push("apos_id=" + aid)',
        "the policy's verbatim reply for this step",
        'lines.push("raw_output " + raw)',
        "trace step ",
    ):
        assert gone not in html, gone
    # The frame panel is now the frame and nothing else: no overlay to print
    # into, so nothing over the image to swallow the click that plays it.
    assert '"tok mono"' not in html


# --------------------------------------------------------------------------- #
# The top-down grid over untrusted coordinates
# --------------------------------------------------------------------------- #


def _teleport_run(root: Path) -> Path:
    """One episode whose position jumps 500 km.

    A unit mix-up, a teleport, or a hostile file all produce it, and
    ``trace.json`` is untrusted. A grid of one line per metre across that span
    is not a slow page, it is a dead tab -- ``paint()`` rebuilds the whole SVG
    on a 250 ms interval while playing.
    """
    root.mkdir(parents=True, exist_ok=True)
    _write(
        root / "run-result.json",
        _run_result([_episode("ep-far", "failed", metrics={"distance_to_goal": 9.0})]),
    )
    _write(
        root / "ep-far" / "trace.json",
        _trace(
            "ep-far",
            [
                _step(0),
                _step(1),
                _step(2, position=[500000.0, 0.0, 0.0]),
                _step(3),
            ],
        ),
    )
    return root


def _grid_caps() -> tuple[int, int]:
    from insight_bench.vis_page import PAGE_TEMPLATE

    match = re.search(r"var GRID_LINES_MAX = (\d+), GRID_LINES_TARGET = (\d+);", PAGE_TEMPLATE)
    assert match, "the grid caps must stay named constants"
    return int(match.group(1)), int(match.group(2))


def test_the_grid_loop_steps_by_the_grid_step_and_is_capped(tmp_path: Path) -> None:
    """The pathological trace renders, and the loop that draws over it is
    bounded in the emitted source: it advances by ``gridStep`` (not one metre)
    and both axes break at ``GRID_LINES_MAX``."""
    root = _teleport_run(tmp_path / "run")
    summary, html, data = _render(root)
    assert summary["episodes"] == 1
    # The coordinate really does reach the page -- the fix is in the drawing,
    # not in silently dropping data.
    assert any(point[0] == 500000.0 for point in data["replay"]["ep-far"]["to"])

    maximum, target = _grid_caps()
    assert (maximum, target) == (40, 12)
    assert html.count("if (drawnX >= GRID_LINES_MAX) { break; }") == 1
    assert html.count("if (drawnY >= GRID_LINES_MAX) { break; }") == 1
    assert "for (i = Math.ceil(minX / gridStep); i * gridStep <= maxX; i += 1)" in html
    assert "for (i = Math.ceil(minY / gridStep); i * gridStep <= maxY; i += 1)" in html
    # The scale bar is one grid step wide and says which, so it stays honest
    # when the step is not one metre.
    assert 'svtext(16 + gridStep * scale, H - 8, gridStep + " m"' in html


@pytest.mark.parametrize("span", [1.0, 4.0, 20.0, 137.0, 1000.0, 500000.0, 1e9])
def test_the_grid_step_keeps_the_line_count_under_the_cap_for_any_span(span: float) -> None:
    """``niceStep`` is executed, not paraphrased: the emitted function is run in
    node and the resulting line count checked against the emitted cap. Skipped
    where no JS engine is installed; the structural test above still guards the
    loop shape in that case."""
    import shutil
    import subprocess

    engine = shutil.which("node") or shutil.which("bun")
    if engine is None:
        pytest.skip("no JS engine on PATH")

    from insight_bench.vis_page import PAGE_TEMPLATE

    maximum, target = _grid_caps()
    start = PAGE_TEMPLATE.index("  function niceStep(span) {")
    end = PAGE_TEMPLATE.index("\n  }\n", start) + len("\n  }\n")
    script = (
        f"var GRID_LINES_TARGET = {target};\n"
        + PAGE_TEMPLATE[start:end]
        + f"\nconsole.log(String(niceStep({span!r})));\n"
    )
    result = subprocess.run([engine, "-e", script], capture_output=True, text=True, check=True)
    step = float(result.stdout.strip())

    assert step >= 1.0, "an indoor episode keeps the metre grid it always had"
    assert math.floor(span / step) + 1 <= maximum, (span, step)
    assert span / step >= 1.0, "a grid so coarse it draws nothing is not a grid"


def test_stdout_stays_a_single_json_object(tmp_path: Path, capsys: Any) -> None:
    root = _build_run(tmp_path / "run")
    assert main(["vis", str(root)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["output"] == "vis.html"
    assert payload["opened_browser"] is False
    assert payload["episodes"] == 4
    assert isinstance(payload["warnings"], list)


def test_the_cli_reports_a_broken_run_directory_on_stderr_with_exit_two(
    tmp_path: Path, capsys: Any
) -> None:
    assert main(["vis", str(tmp_path / "missing")]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["error"]


def test_run_refuses_vis_without_a_run_dir(tmp_path: Path, capsys: Any) -> None:
    # Refused in the CLI, before the registry is read or a simulator is
    # reached: neither the episode file nor the scene root below exists, and
    # the command still fails on the one thing it can already see.
    status = main(
        [
            "run",
            "--episodes",
            str(tmp_path / "episodes.jsonl"),
            "--scene-root",
            str(tmp_path / "scenes"),
            "--output",
            str(tmp_path / "r.json"),
            "--vis",
        ]
    )
    assert status == 2
    assert "--vis needs --run-dir" in json.loads(capsys.readouterr().err)["error"]
    assert not (tmp_path / "r.json").exists()


def test_no_open_is_honoured_even_on_a_tty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _build_run(tmp_path / "run")
    calls: list[str] = []
    monkeypatch.setattr(vis.webbrowser, "open", lambda url: calls.append(url) or True)
    summary = vis.render_run_page(root, open_browser=False)
    assert summary["opened_browser"] is False
    assert calls == []
    summary = vis.render_run_page(root, open_browser=True)
    assert summary["opened_browser"] is True
    assert calls and calls[0].startswith("file://")


def test_a_browser_that_fails_to_open_does_not_fail_the_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _build_run(tmp_path / "run")

    def boom(url: str) -> bool:
        raise OSError("no browser here")

    monkeypatch.setattr(vis.webbrowser, "open", boom)
    assert vis.render_run_page(root, open_browser=True)["opened_browser"] is False


def test_the_pose_column_is_the_ground_plane_pose_with_yaw(tmp_path: Path) -> None:
    root = tmp_path / "pose"
    root.mkdir()
    _write(root / "run-result.json", _run_result([_episode("ep-a", "passed")]))
    turned = _step(1)
    half = math.pi / 4
    turned["orientation_wxyz"] = [math.cos(half), 0.0, 0.0, math.sin(half)]
    _write(root / "ep-a" / "trace.json", _trace("ep-a", [_step(0), turned]))
    _, _, data = _render(root)
    assert data["replay"]["ep-a"]["to"][0] == [1.0, 0.5, pytest.approx(math.pi / 2, abs=1e-3)]
    assert data["replay"]["ep-a"]["from"][0] == [0.0, 0.0, 0.0]


def test_an_episode_with_no_trace_is_told_apart_from_one_past_the_inlining_limit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bare"
    root.mkdir()
    _write(root / "run-result.json", _run_result([_episode("solo", "failed")]))
    summary, html, data = _render(root)
    assert data["episodes"][0]["no_trace"] is True
    assert data["episodes"][0]["frames"] is None
    assert any("no trace.json" in line for line in summary["warnings"])
    assert "no trace.json beside this episode" in html


def test_losses_the_run_counted_but_could_not_write_are_named_on_the_page(
    tmp_path: Path,
) -> None:
    """A frame or sidecar the run failed to write is a warning, not a silence.

    Neither loss fails a run on purpose, so the counters in `vis-manifest.json`
    are the only record of them. Left unread, a run whose sidecars all failed is
    indistinguishable from a run recorded without `--vis`, and a run missing
    half its frames looks like a run the policy stopped early.
    """
    root = _build_run(tmp_path / "run")
    _write(
        root / "vis-manifest.json",
        {
            "schema": "insight-bench/vis-manifest/1",
            "episode_count": 4,
            "frame_stride": 1,
            "max_frames_per_episode": 100,
            "frames_written": 3,
            "frames_failed": 7,
            "frame_failure_reason": "ValueError: unexpected frame shape",
            "sidecars_failed": 2,
            "sidecar_failure_reason": "ValueError: " + HOSTILE_SCRIPT,
        },
    )
    summary, html, _ = _render(root)

    lost = [line for line in summary["warnings"] if "could not write" in line]
    frames = [line for line in lost if "camera frames" in line]
    sidecars = [line for line in lost if "episode.json sidecars" in line]
    assert frames == [
        "the run recorded 7 camera frames it could not write; first reason: "
        "ValueError: unexpected frame shape"
    ]
    assert len(sidecars) == 1
    assert sidecars[0].startswith("the run recorded 2 episode.json sidecars it could not write")
    # The reason is a string from a file under the run directory, so it is
    # untrusted like every other one: it may reach the page only as data.
    assert "</script><script>" not in html


def test_a_manifest_that_lost_nothing_adds_no_warning(tmp_path: Path) -> None:
    root = _build_run(tmp_path / "run")
    summary, _, _ = _render(root)

    assert not [line for line in summary["warnings"] if "could not write" in line]


def test_the_recorded_camera_reaches_the_top_down_view(tmp_path: Path) -> None:
    """`vis-manifest.json`'s camera block is what draws the field-of-view wedge.

    Without it the view degrades to a bare heading arrow, which is correct but
    tells a reader nothing about what the frame beside it could see.
    """
    root = _build_run(tmp_path / "run")
    manifest = json.loads((root / "vis-manifest.json").read_text(encoding="utf-8"))
    _write(root / "vis-manifest.json", {**manifest, "camera": {"camera_hfov_deg": 90.0}})
    _, _, data = _render(root)

    assert data["camera"] == {"camera_hfov_deg": 90.0}


def test_every_published_category_value_has_a_place_in_the_axis_order() -> None:
    """The radar axes are keyed on the published label vocabulary, so they must know it.

    Both charts exist to break a run down by ``metadata.scene_class`` and
    ``metadata.instr_type``. A value missing from these tuples still renders --
    unlisted values sort in after the listed ones -- but the designed order is
    the thing the two charts share, and losing it is how the page stops being
    comparable between runs. This is the check that the tuples track the
    vocabulary the published suite actually ships.
    """
    published_scene_classes = ("Apartment", "House", "Commercial", "Institution", "Outdoor")
    published_instr_types = ("Base", "Direction", "Relation", "Extremum", "Ordinal")
    assert set(published_scene_classes) <= set(vis.SCENE_CLASS_ORDER)
    assert set(published_instr_types) <= set(vis.INSTR_TYPE_ORDER)


def test_no_category_value_or_label_the_page_can_render_is_non_ascii() -> None:
    """The published files carry no Chinese, and neither may the page's own copy."""
    for text in (*vis.SCENE_CLASS_ORDER, *vis.INSTR_TYPE_ORDER, *vis.CATEGORY_LABELS.values()):
        assert text.isascii(), text


def test_both_radars_render_the_published_vocabulary_in_the_designed_order(
    tmp_path: Path,
) -> None:
    """The end-to-end check the axis order was missing.

    ``SCENE_CLASS_ORDER``/``INSTR_TYPE_ORDER`` are matched against the raw
    sidecar value, so a vocabulary change leaves them matching nothing -- and
    nothing fails, because an unlisted value simply sorts in after the listed
    ones. The charts still draw, in alphabetical order, which is a silent
    regression: the designed order is the thing the two charts share so a
    reader comparing them is not re-learning the layout. This renders a run
    carrying all ten published values and pins the order they come out in.
    """
    scene_classes = ("Apartment", "House", "Commercial", "Institution", "Outdoor")
    instr_types = ("Base", "Direction", "Relation", "Extremum", "Ordinal")
    root = tmp_path / "published-vocabulary"
    root.mkdir()
    episodes = [_episode(f"ep-{index}", "passed") for index in range(len(scene_classes))]
    _write(root / "run-result.json", _run_result(episodes))
    # Paired off the alphabetical-order tell: zipping the two tuples puts each
    # chart's designed order in a different alphabetical order from the other.
    for index, (scene_class, instr_type) in enumerate(zip(scene_classes, instr_types, strict=True)):
        _write(
            root / f"ep-{index}" / "episode.json",
            {
                "episode_id": f"ep-{index}",
                "instr_type": instr_type,
                "scene_class": scene_class,
            },
        )
    _, _, data = _render(root)
    breakdown = data["breakdown"]
    assert [axis["name"] for axis in breakdown["scene_class"]] == list(scene_classes)
    assert [axis["name"] for axis in breakdown["instr_type"]] == list(instr_types)
    # Not the fallback: every value was recognised, so none fell to Unknown and
    # none of the axes is a sorted-in leftover.
    assert [axis["label"] for axis in breakdown["scene_class"]] == list(scene_classes)
    assert [axis["label"] for axis in breakdown["instr_type"]] == list(instr_types)
    assert vis.UNKNOWN_CATEGORY not in {axis["name"] for axis in breakdown["scene_class"]}
    assert sorted(scene_classes) != list(scene_classes), "the order must not be alphabetical"
    assert sorted(instr_types) != list(instr_types), "the order must not be alphabetical"


def test_a_truncated_runs_own_caveat_reaches_the_page(tmp_path: Path) -> None:
    """The one sentence saying a run covered part of the suite.

    It was in the run result and printed by the script, and appeared nowhere on
    the page -- so a ten-of-1097 run rendered as an ordinary result. The page is
    what gets screenshotted and outlives the terminal, which is exactly where
    that caveat has to travel.
    """
    root = _build_run(tmp_path / "run")
    result = json.loads((root / "run-result.json").read_text(encoding="utf-8"))
    result["status"] = "partial"
    result["adapter"]["metadata"] = {
        "episode_subset": "10 of 1097 episodes; smoke run, not a full measurement"
    }
    _write(root / "run-result.json", result)

    _, html, data = _render(root)

    assert data["run"]["episode_subset"].startswith("10 of 1097 episodes")
    assert "10 of 1097 episodes; smoke run, not a full measurement" in html
    # Rendered in the status's own colours, beside the status.
    assert 'el("span", "pill " + (run.status || ""), run.episode_subset)' in html


def test_a_full_run_carries_no_such_caveat(tmp_path: Path) -> None:
    root = _build_run(tmp_path / "run")
    _, _html, data = _render(root)

    assert data["run"]["episode_subset"] == ""
