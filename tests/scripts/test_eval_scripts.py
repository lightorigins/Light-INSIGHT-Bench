"""The eval scripts are the product's front door and nothing exercised them.

Every command in the README goes through one of these, and a shell script fails
at the reader's shell rather than in CI. Two things are checked here without a
GPU, a simulator or a model: that each script parses, and that a dry run
derives the paths and the commands it says it will.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"

#: Every eval script takes MODEL_PATH; eval_custom.sh additionally takes POLICY.
EVAL_SCRIPTS = sorted(
    path.name for path in SCRIPTS.glob("eval_*.sh") if path.name != "eval_common.sh"
)


def _all_scripts() -> list[Path]:
    return sorted(SCRIPTS.glob("*.sh"))


def test_there_are_scripts_to_check() -> None:
    assert len(_all_scripts()) >= 15
    assert len(EVAL_SCRIPTS) >= 8


@pytest.mark.parametrize("script", _all_scripts(), ids=lambda path: path.name)
def test_the_script_parses(script: Path) -> None:
    done = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True, check=False)
    assert done.returncode == 0, f"{script.name}: {done.stderr.strip()}"


@pytest.fixture
def sandbox(tmp_path: Path) -> dict[str, str]:
    """The least on disk that lets a dry run get past its precondition checks."""
    data = tmp_path / "data"
    (data / "insight_bench" / "v1").mkdir(parents=True)
    (data / "insight_bench" / "v1" / "episodes.jsonl").write_text("{}\n", encoding="utf-8")
    (data / "scenes").mkdir()
    isaac = tmp_path / "isaac"
    isaac.mkdir()
    shim = isaac / "isaaclab.sh"
    shim.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    shim.chmod(0o755)
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    # A baseline script needs the model's own checkout, and the dry run now says
    # so rather than letting it fail an hour later with the weights loaded. The
    # sandbox therefore represents a machine whose setup script has been run.
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    return {
        **os.environ,
        "DRY_RUN": "1",
        "EVAL_DIR": str(REPO),
        "DATA_DIR": str(data),
        "ISAACLAB_DIR": str(isaac),
        "MODEL_PATH": str(checkpoint),
        "MODEL_PYTHON": "/usr/bin/python3",
        "RUN_DIR": str(tmp_path / "run"),
        "UPSTREAM_REPO": str(upstream),
    }


def _dry_run(script: str, env: dict[str, str]) -> str:
    done = subprocess.run(
        ["bash", str(SCRIPTS / script)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=REPO,
    )
    assert done.returncode == 0, f"{script}: {done.stdout}\n{done.stderr}"
    return done.stdout


@pytest.mark.parametrize("script", EVAL_SCRIPTS)
def test_a_dry_run_derives_the_published_coordinate_and_data_layout(
    script: str, sandbox: dict[str, str]
) -> None:
    if script == "eval_custom.sh":
        sandbox["POLICY"] = "policies/template"
    output = _dry_run(script, sandbox)

    # The coordinate and the path layout are the two things a reader copies out
    # of the README, and they are derived rather than written down, so a change
    # to either would otherwise show up only on a real run.
    assert "suite=insight-bench-v1@1.0.0" in output
    assert "insight_bench/v1/episodes.jsonl" in output
    assert "would start:" in output and "would run:" in output


@pytest.mark.parametrize("script", EVAL_SCRIPTS)
def test_a_dry_run_keeps_the_two_sides_in_their_own_interpreters(
    script: str, sandbox: dict[str, str]
) -> None:
    if script == "eval_custom.sh":
        sandbox["POLICY"] = "policies/template"
    output = _dry_run(script, sandbox)
    start = next(line for line in output.splitlines() if "would start:" in line)
    run = next(line for line in output.splitlines() if "would run:" in line)

    # The model side runs the model's interpreter and the policy server; the
    # runner side runs Isaac Lab's. Crossing them is the failure this pins.
    assert "/usr/bin/python3 -m insight_policy" in start
    assert "-m insight_bench" not in start
    assert "isaaclab.sh" not in start
    assert "isaaclab.sh -p -m insight_bench run" in run
    assert "--policy-url http://127.0.0.1:" in run


def test_a_dry_run_starts_no_process_and_makes_no_run_directory(sandbox: dict[str, str]) -> None:
    output = _dry_run("eval_lightnav0.sh", sandbox)
    assert "would start:" in output
    assert not Path(sandbox["RUN_DIR"]).exists()


@pytest.mark.parametrize(
    ("drop", "expected"),
    [
        ("MODEL_PATH", "MODEL_PATH"),
        ("ISAACLAB_DIR", "ISAACLAB_DIR"),
    ],
)
def test_a_missing_precondition_names_itself_and_exits_nonzero(
    drop: str, expected: str, sandbox: dict[str, str]
) -> None:
    sandbox[drop] = str(Path(sandbox["EVAL_DIR"]) / "definitely-not-here")
    done = subprocess.run(
        ["bash", str(SCRIPTS / "eval_lightnav0.sh")],
        capture_output=True,
        text=True,
        check=False,
        env=sandbox,
        cwd=REPO,
    )
    assert done.returncode != 0
    assert expected in done.stdout + done.stderr


def test_eval_custom_refuses_without_a_policy_directory(sandbox: dict[str, str]) -> None:
    sandbox.pop("POLICY", None)
    done = subprocess.run(
        ["bash", str(SCRIPTS / "eval_custom.sh")],
        capture_output=True,
        text=True,
        check=False,
        env=sandbox,
        cwd=REPO,
    )
    assert done.returncode == 2
    assert "POLICY" in done.stderr


def test_a_dry_run_refuses_a_missing_upstream_checkout(sandbox: dict[str, str]) -> None:
    """The point of a dry run is to find this before you hold a GPU.

    A baseline needs the model's own code, and pointing at a path that is not
    there used to print two commands, exit 0, and then fail twenty minutes
    later with 7B of weights already loaded.
    """
    sandbox["UPSTREAM_REPO"] = str(Path(sandbox["RUN_DIR"]).parent / "not-cloned")

    done = subprocess.run(
        ["bash", str(SCRIPTS / "eval_janusvln.sh")],
        capture_output=True,
        text=True,
        check=False,
        env=sandbox,
        cwd=REPO,
    )

    assert done.returncode == 1
    assert "UPSTREAM_REPO does not exist" in done.stderr
    # ... and it still shows the commands, so the rest can be eyeballed.
    assert "would run:" in done.stdout


def test_a_dry_run_says_so_when_the_preconditions_it_can_check_pass(
    sandbox: dict[str, str],
) -> None:
    done = subprocess.run(
        ["bash", str(SCRIPTS / "eval_janusvln.sh")],
        capture_output=True,
        text=True,
        check=False,
        env=sandbox,
        cwd=REPO,
    )

    assert done.returncode == 0
    assert "the preconditions this can check all pass" in done.stdout


@pytest.mark.parametrize(
    "script", ["eval_navid.sh", "eval_uninavid.sh", "eval_embodied_navigator.sh"]
)
def test_a_baseline_that_loads_through_upstream_is_told_where_it_is(
    script: str, sandbox: dict[str, str]
) -> None:
    """Three policies need `--opt upstream_repo` and no recipe supplied it.

    They raise "needs --opt upstream_repo=<path>; it has no public default"
    seconds into a run, on the model side, with the GPU already held. The
    script knows the path -- it puts the same one on PYTHONPATH -- so it now
    passes it, and the README recipes work as written.
    """
    output = _dry_run(script, sandbox)
    start = next(line for line in output.splitlines() if "would start:" in line)

    assert f"--opt upstream_repo={sandbox['UPSTREAM_REPO']}" in start


def test_a_caller_can_still_override_the_upstream_option(sandbox: dict[str, str]) -> None:
    sandbox["POLICY_OPTS"] = "--opt upstream_repo=/somewhere/else"
    output = _dry_run("eval_navid.sh", sandbox)
    start = next(line for line in output.splitlines() if "would start:" in line)

    # Both appear; the policy takes the last, which is the caller's.
    assert start.index(sandbox["UPSTREAM_REPO"]) < start.index("/somewhere/else")
