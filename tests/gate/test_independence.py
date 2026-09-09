"""Dependency and honesty gates for the independent evaluation SDK."""

from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: The runtime stack that exists only inside the published Isaac Sim image.
SIMULATOR_MODULES = ("isaacsim", "isaaclab", "omni", "pxr", "carb", "torch")

#: What the model side is not allowed to need. ``policies/`` runs in each
#: model's own interpreter -- several baselines pin Python 3.8 with a
#: transformers 4.31-era dependency set that cannot install pydantic 2 or
#: FastAPI -- so it speaks the wire over ``http.server`` and installs neither
#: the evaluation SDK nor its dependencies.
FORBIDDEN_POLICY_IMPORTS = frozenset({"insight_bench", "fastapi", "pydantic", "requests"})

#: One import of every module a real run loads, including the leaves that are
#: only reached with a simulator attached -- which is where an import of the
#: operator harness would hide, because nothing else exercises them here.
CLOSURE_MODULES = (
    "insight_bench.cli",
    "insight_bench.data_check",
    "insight_bench.policy_check",
    "insight_bench.evidence",
    "insight_bench.vis",
    "insight_bench.runner",
    "insight_bench.runners.insight_bench",
    "insight_bench.vln_runtime.episodes.loader",
    "insight_bench.vln_runtime.episodes.adapters.objnav",
    "insight_bench.vln_runtime.measures.terms",
    "insight_bench.vln_runtime.motion.waypoint_execution",
    "insight_bench.vln_runtime.policy.client",
    "insight_bench.vln_runtime.rollout",
    "insight_bench.vln_runtime.scoring.manager",
    "insight_bench.vln_runtime.scoring.navnuances_terms",
    "insight_bench.vln_runtime.suite.insight_bench",
    "insight_bench.vln_runtime.suite.registry",
    "insight_bench.vln_runtime.terminations.terms",
    "insight_bench.vln_runtime.traces.writer",
    "insight_bench.simulator",
    "insight_bench.simulator.isaac",
    "insight_bench.simulator.isaac.app",
    "insight_bench.simulator.isaac.backend",
    "insight_bench.simulator.isaac.camera_walk",
    "insight_bench.simulator.isaac.terrain_query",
)


def _source_files(tree: Path) -> list[Path]:
    return [
        path
        for path in sorted(tree.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    ]


def _imported_roots(source: str) -> set[str]:
    """Top-level module names a file imports, including inside function bodies.

    Parsed rather than grepped: ``policies/template/policy.py`` names the SDK in
    its docstring precisely to say that it does not import it, and a text match
    cannot tell that apart from an import.
    """
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_source_tree_has_no_forbidden_tokens() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_independence.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_gate_scans_the_readme_pyproject_ships_as_long_description() -> None:
    """A forbidden token in the readme must fail the gate without a build.

    ``readme = "README.md"`` makes the readme the wheel's METADATA and the
    sdist's PKG-INFO, so an internal path there ships to any index the package
    is published to. The readme is in none of the scanned trees, so this used
    to be reachable only by ``--dist`` after ``python -m build``.
    """
    # Assembled, not spelled: this file is itself under a tree the gate scans,
    # and a literal token here would fail it on the fixture rather than the bug.
    token = "ve" + "pfs"
    with tempfile.TemporaryDirectory() as raw:
        tree = Path(raw)
        (tree / "README.md").write_text(
            f"Episodes live at /{token}-internal/somebody/data\n", encoding="utf-8"
        )
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "check_independence.py")],
            cwd=tree,
            capture_output=True,
            text=True,
        )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "README.md" in result.stderr
    assert token in result.stderr


def test_the_gate_scans_the_trees_the_sdist_added() -> None:
    """``policies/`` and ``scripts/`` ship in the sdist, so the gate must see them.

    Both are listed in ``[tool.hatch.build.targets.sdist]`` -- every byte of
    them reaches any index the package is published to -- and both were written
    by porting scripts that ran against internal infrastructure. ``scripts/`` is
    entirely shell, so covering it takes both a tree the scan walks and a suffix
    it does not skip -- missing either leaves the same hole, and a mount path or
    a private registry host is exactly what an evaluation script carries.

    Asserted by planting a token rather than by reading the script's tree list,
    for the same reason the readme clause above does: what has to hold is that
    the gate fails, not that it is spelled a particular way.
    """
    # Assembled, not spelled: this file is under a tree the gate scans.
    token = "ve" + "pfs"
    planted = {
        Path("policies") / "somebody" / "policy.py": f'WEIGHTS = "/{token}-internal/ckpt"\n',
        Path("scripts") / "eval_somebody.sh": f"DATA_DIR=/{token}-internal/data\n",
    }
    for relative, body in planted.items():
        with tempfile.TemporaryDirectory() as raw:
            tree = Path(raw)
            (tree / relative).parent.mkdir(parents=True)
            (tree / relative).write_text(body, encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(ROOT / "tools" / "check_independence.py")],
                cwd=tree,
                capture_output=True,
                text=True,
            )
        report = result.stdout + result.stderr
        assert result.returncode == 1, f"{relative} passed the gate: {report}"
        assert str(relative) in result.stderr, report
        assert token in result.stderr, report


def test_the_model_side_imports_none_of_the_evaluation_stack() -> None:
    """``policies/`` is a separate program in a separate interpreter.

    That separation is what lets a baseline with an ancient pinned environment
    be evaluated at all, and it is load-bearing rather than aesthetic: the SDK
    is never installed next to the model, so an import of it here would not
    degrade, it would fail to start the server. The wire is the only coupling.
    """
    offenders: list[str] = []
    scanned = 0
    for path in _source_files(ROOT / "policies"):
        if path.suffix != ".py":
            continue
        scanned += 1
        roots = _imported_roots(path.read_text(encoding="utf-8"))
        offenders.extend(
            f"{path.relative_to(ROOT)}: imports {name}"
            for name in sorted(roots & FORBIDDEN_POLICY_IMPORTS)
        )
    assert scanned, "policies/ has no modules: the scan would pass vacuously"
    assert not offenders, offenders


def test_importing_the_closure_pulls_no_operator_modules() -> None:
    for name in CLOSURE_MODULES:
        __import__(name)
    # Assembled at runtime so this file itself passes the token scanner.
    forbidden = ("vln_isaac" + "_benchmark", "vln_" + "arena")
    leaked = [name for name in sys.modules if any(token in name for token in forbidden)]
    assert not leaked, f"operator modules leaked into sys.modules: {leaked}"


def test_python_floor_is_3_10() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert 'requires-python = ">=3.10"' in pyproject


def test_the_package_imports_without_the_simulator_runtime() -> None:
    """A plain ``pip install`` has no Isaac Sim, no Isaac Lab and no torch.

    Those live only inside the published runtime image, so every reference to
    them is made inside a function body. What keeps it that way is asserting
    that neither importing the package nor building the whole parser -- which
    ``--help`` does, reaching every subcommand -- puts one in ``sys.modules``.
    """
    script = f"""
import sys

import insight_bench
from insight_bench.cli import main

try:
    main(["--help"])
except SystemExit as exc:
    assert exc.code == 0, exc.code

heavy = set({SIMULATOR_MODULES!r})
loaded = sorted({{name.split(".")[0] for name in sys.modules}} & heavy)
assert not loaded, loaded
print("import stayed light")
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "import stayed light" in result.stdout


def test_backend_policy_is_51_default_60_experimental() -> None:
    from insight_bench.simulator import SIM_BACKENDS, default_backend, get_backend_descriptor

    default = default_backend()
    assert default.backend_id == "isaac-5.1"
    assert default.status == "official"
    assert default.isaac_sim == "5.1.0"

    experimental = get_backend_descriptor("isaac-6.0")
    assert experimental.status == "experimental"
    assert experimental.isaac_sim == "6.0.1"

    # No runtime image digest is published (the image embeds Isaac Sim and is
    # not redistributable); the field is informational and gates nothing.
    assert all(item.image_digest is None for item in SIM_BACKENDS)


def test_isaac_backend_loader_matches_the_machine_it_runs_on() -> None:
    from insight_bench.simulator.base import SimulatorBackend, SimulatorNotAvailableError
    from insight_bench.simulator.isaac import load_isaac_backend
    from insight_bench.simulator.isaac.backend import isaac_import_probe

    available, missing = isaac_import_probe()
    if available and sys.platform.startswith(("linux", "win")):
        # Inside a runtime image the loader must hand back a real backend.
        assert isinstance(load_isaac_backend("isaac-5.1"), SimulatorBackend)
    else:
        # Everywhere else it must say why, not degrade into a stub.
        with pytest.raises(SimulatorNotAvailableError) as excinfo:
            load_isaac_backend("isaac-5.1")
        assert "cannot execute here" in str(excinfo.value)
        assert missing or sys.platform not in {"linux", "win32"}

    # The experimental line has no implementation on any machine.
    with pytest.raises(SimulatorNotAvailableError):
        load_isaac_backend("isaac-6.0")


def test_the_registered_runner_set_is_exactly_the_shipped_one() -> None:
    """Asserting the set is what stops an entry point appearing unnoticed.

    The retired vln-suite-v5 gate runner used to be registered as a skeleton
    that refused to run; the protection worth keeping is not "that one
    refuses" but "the registered set is the one this release documents".
    """
    from insight_bench.runner import available_runners

    assert available_runners() == ("objnav-http-policy-v1",)
    assert not any("vln-suite" in name for name in available_runners())
