"""The model side claims Python 3.8, so something has to hold it to that.

``policies/`` runs inside the model's own interpreter, and the README tells a
reader that any Python 3.8 or newer with numpy and pillow will do. Nothing
enforced it: this repository's own interpreter is 3.11, so a walrus, a match
statement or ``X | None`` in an annotation would pass every other gate here and
fail only on the reader's machine, after they had installed a baseline.

This compiles every module under ``policies/`` with the oldest grammar the
README promises. It checks syntax, not imports, because the baselines import
upstream packages that are not installed here.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

#: The oldest interpreter the README promises for the model side.
OLDEST_SUPPORTED: tuple[int, int] = (3, 8)

POLICIES = Path(__file__).resolve().parents[2] / "policies"


def _policy_sources() -> list[Path]:
    return sorted(path for path in POLICIES.rglob("*.py") if "__pycache__" not in path.parts)


def test_there_are_policy_sources_to_check() -> None:
    # A glob that silently matches nothing would make every test below pass.
    assert len(_policy_sources()) >= 10


@pytest.mark.parametrize("source", _policy_sources(), ids=lambda path: path.name)
def test_the_module_parses_under_the_oldest_supported_grammar(source: Path) -> None:
    text = source.read_text(encoding="utf-8")
    try:
        ast.parse(text, filename=str(source), feature_version=OLDEST_SUPPORTED[1])
    except SyntaxError as error:
        version = ".".join(str(part) for part in OLDEST_SUPPORTED)
        pytest.fail(
            f"{source.relative_to(POLICIES.parent)} needs newer than Python {version}: "
            f"{error.msg} at line {error.lineno}"
        )


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="ast.unparse, used only to report the offending line"
)
def test_no_policy_module_imports_the_runner_package() -> None:
    """The two sides share a port, not an installation."""
    offenders = []
    for source in _policy_sources():
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name.split(".")[0] == "insight_bench" for name in names):
                offenders.append(f"{source.name}:{node.lineno}")
    assert not offenders, f"policies must not import the runner package: {offenders}"
