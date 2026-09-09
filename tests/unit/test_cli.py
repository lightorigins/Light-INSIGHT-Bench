"""The six subcommands, driven through :func:`insight_bench.cli.main`.

``check-data``, ``check-policy``, ``run``, ``vis``, ``pack``, ``verify`` -- in
the order a user meets them. ``catalog``, ``validate`` and ``inspect`` are gone
with the fixture benchmarks they listed, and ``--adapter``/``--model-id`` with
the in-process adapters: the model identity of a run is read from the policy
server's ``GET /health`` now, so there is nothing on the command line to set it
with.

Nothing here starts a simulator or opens a socket. ``run`` is driven exactly as
far as the refusals that happen before either -- the two paths a licence-gated
suite cannot ship, and ``--vis`` with nowhere to write -- and ``check-policy``,
which does speak to a server, is covered in ``tests/gate/test_check_policy.py``.
``pack``/``verify``/``vis`` are given a run result built by ``tests/support.py``
rather than produced by a run, which is what the deleted ``bench-tiny`` fixture
benchmark used to be for.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from insight_bench.cli import DEFAULT_BENCHMARK, build_parser, main
from support import write_run_result

#: Real published episode records, as shipped for the scene-binding tests.
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "objnav_published_episodes.jsonl"

COMMANDS = ("check-data", "check-policy", "run", "vis", "pack", "verify")


def _episode_file(tmp_path: Path, count: int = 2) -> tuple[Path, list[dict]]:
    """A user's own episode file: *count* published records, written by this test."""
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    # Two different scene families, so a scene root has to resolve more than one layout.
    picked = [records[0], records[2]][:count]
    path = tmp_path / "episodes.jsonl"
    path.write_text("".join(f"{json.dumps(record)}\n" for record in picked), encoding="utf-8")
    return path, picked


def _scene_root(tmp_path: Path, records: list[dict]) -> Path:
    """The directory the user converted their scenes into, with those scenes in it."""
    root = tmp_path / "scenes"
    root.mkdir(parents=True, exist_ok=True)
    for record in records:
        asset = root / str(record["scene"]["asset"])
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_text("#usda 1.0\n", encoding="utf-8")
    return root


def _json_out(capsys: pytest.CaptureFixture[str]) -> dict:
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, dict)
    return payload


def test_the_command_set_is_exactly_the_six_product_steps() -> None:
    """Stated as a list on purpose: a command reappearing is a product decision.

    ``catalog`` and ``validate`` described a registry a user could choose from;
    there is one product path now and no choice to offer. ``inspect`` read an
    evidence pack without verifying it, which ``verify`` does properly. Read off
    the usage line, which is the same list the user is shown.
    """
    usage = build_parser().format_usage()
    listed = re.search(r"\{([^}]+)\}", usage)
    assert listed is not None, usage
    assert tuple(listed.group(1).split(",")) == COMMANDS


@pytest.mark.parametrize("retired", ["catalog", "validate", "inspect"])
def test_a_retired_command_is_a_usage_error_not_a_silent_no_op(retired: str) -> None:
    with pytest.raises(SystemExit) as caught:
        main([retired])
    assert caught.value.code == 2


def test_check_data_names_every_missing_scene_asset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The command exists so a user learns this in seconds instead of after a startup.

    An empty scene root is the ordinary first run: the episodes parse, and every
    one of them is blocked on a file only the user can put there. Missing assets
    are reported once per *asset*, which is the list of files to go and fetch.
    """
    episodes, records = _episode_file(tmp_path)
    empty = tmp_path / "no-scenes"
    empty.mkdir()

    assert main(["check-data", "--episodes", str(episodes), "--scene-root", str(empty)]) == 1
    report = _json_out(capsys)
    assert report["benchmark"] == DEFAULT_BENCHMARK
    assert report["ready"] is False
    assert report["episodes"]["readable"] is True
    assert report["episodes"]["records"] == len(records)
    assert report["scenes"]["missing_assets"] == sorted(
        str(record["scene"]["asset"]) for record in records
    )
    assert report["scenes"]["episodes_blocked"] == len(records)


def test_check_data_resolves_the_scenes_that_are_present(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With the scenes in place the only problems left are about the file itself.

    Both remaining ones are the point of the command: a file that is not the
    published one, and a file that does not hold the whole coordinate, are each
    enough to make a run not a run of this benchmark. A two-record excerpt is
    both, so ``ready`` stays false and the exit status stays 1.
    """
    episodes, records = _episode_file(tmp_path)
    scene_root = _scene_root(tmp_path, records)

    assert (
        main(
            [
                "check-data",
                "insight-bench-v1@1.0.0",
                "--episodes",
                str(episodes),
                "--scene-root",
                str(scene_root),
            ]
        )
        == 1
    )
    report = _json_out(capsys)
    assert report["benchmark"] == "insight-bench-v1@1.0.0"
    assert report["scenes"]["missing_assets"] == []
    assert report["scenes"]["assets_found"] == report["scenes"]["assets_required"] == len(records)
    assert report["episodes"]["digest_matches"] is False
    assert report["problems"] == [
        "the episode file is not the published one this coordinate pins; "
        "a run on a different file is not a run of this benchmark",
        f"the coordinate is defined over {report['episodes']['expected_records']} episodes "
        f"but the file holds {len(records)}",
    ]


@pytest.mark.parametrize(
    ("command", "argv"),
    [
        ("run", []),
        ("run", ["--episodes", "episodes.jsonl"]),
        ("run", ["--scene-root", "scenes"]),
        ("check-data", []),
    ],
)
def test_the_two_user_supplied_paths_are_required(command: str, argv: list[str]) -> None:
    """Neither the episode file nor the scene root can be defaulted or guessed.

    They are licence-gated downloads this distribution never ships and never
    fetches, so a command that ran without them would be running against
    something else. argparse refuses with the usage exit status, before ``main``
    does any work.
    """
    with pytest.raises(SystemExit) as caught:
        main([command, *argv])
    assert caught.value.code == 2


def test_run_defaults_to_the_published_suite_and_takes_both_run_dir_spellings() -> None:
    """``--artifact-dir`` is kept as an alias, so an existing command line survives."""
    parser = build_parser()
    base = ["run", "--episodes", "episodes.jsonl", "--scene-root", "scenes"]

    defaults = parser.parse_args(base)
    assert defaults.benchmark == DEFAULT_BENCHMARK == "insight-bench-v1@1.0.0"
    assert defaults.artifact_dir is None
    assert defaults.max_episodes is None
    assert defaults.vis is False

    assert parser.parse_args([*base, "--run-dir", "out"]).artifact_dir == Path("out")
    assert parser.parse_args([*base, "--artifact-dir", "out"]).artifact_dir == Path("out")
    assert parser.parse_args([*base, "--max-episodes", "3"]).max_episodes == 3


def test_run_refuses_vis_with_nowhere_to_persist_frames(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Refused at argument level, before a scene root is read or a simulator loaded.

    ``open_vis_session`` refuses too, but only once the runner is about to bring
    Isaac up. Failing here costs the user nothing, which is the whole point.
    """
    episodes, _ = _episode_file(tmp_path)
    assert main(["run", "--episodes", str(episodes), "--scene-root", str(tmp_path), "--vis"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--vis needs --run-dir" in json.loads(captured.err)["error"]


def test_vis_writes_the_page_into_the_run_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``vis`` reads what a run left behind; the page is always ``<run_dir>/vis.html``.

    A run recorded without ``--vis`` has no frames, and the page says so rather
    than failing: that is not a negative answer, so the exit status is 0.
    """
    run_dir = tmp_path / "run"
    write_run_result(run_dir / "run-result.json")

    assert main(["vis", str(run_dir), "--no-open"]) == 0
    summary = _json_out(capsys)
    assert summary["output"] == "vis.html"
    assert summary["run_result"] == "run-result.json"
    assert summary["episodes"] == 2
    assert summary["opened_browser"] is False
    assert (run_dir / "vis.html").is_file()


def test_pack_then_verify_round_trips_and_a_tampered_pack_exits_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = write_run_result(tmp_path / "run-result.json")
    pack = tmp_path / "pack"

    assert main(["pack", str(result), str(pack)]) == 0
    packed = _json_out(capsys)
    assert packed["run_id"] == json.loads(result.read_text(encoding="utf-8"))["run_id"]

    assert main(["verify", str(pack)]) == 0
    assert _json_out(capsys)["valid"] is True

    (pack / "run-result.json").write_text("{}\n", encoding="utf-8")
    assert main(["verify", str(pack)]) == 1
    report = _json_out(capsys)
    assert report["valid"] is False
    assert any("SHA-256 mismatch" in error for error in report["errors"])


def test_the_module_form_runs_the_same_cli(tmp_path: Path) -> None:
    """``python -m insight_bench`` is the only form that works under a wrapped
    interpreter, because the console script lands in that interpreter's ``bin``
    rather than on ``PATH``. Isaac Lab is driven that way, so the README tells a
    reader to use it and this pins that it works -- including that it returns the
    exit status, which module execution discards unless ``__main__`` re-raises it.
    """
    from insight_bench.evidence import pack_evidence

    pack = tmp_path / "pack"
    pack_evidence(write_run_result(tmp_path / "run-result.json"), pack)

    completed = subprocess.run(
        [sys.executable, "-m", "insight_bench", "verify", str(pack)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["valid"] is True
