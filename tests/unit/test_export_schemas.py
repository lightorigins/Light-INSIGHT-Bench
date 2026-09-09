from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[2]
EXPORT_SCRIPT = REPOSITORY_ROOT / "tools" / "export_schemas.py"


def test_checked_in_schema_exports_are_current() -> None:
    result = subprocess.run(
        [sys.executable, str(EXPORT_SCRIPT), "--check"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_schema_check_reports_missing_stale_and_unexpected_files(tmp_path: Path) -> None:
    copied_script = tmp_path / "tools" / "export_schemas.py"
    copied_script.parent.mkdir()
    shutil.copyfile(EXPORT_SCRIPT, copied_script)
    subprocess.run([sys.executable, str(copied_script)], cwd=tmp_path, check=True)

    output = tmp_path / "src" / "insight_bench" / "schemas" / "v1"
    exported = sorted(output.glob("*.schema.json"))
    missing, stale = exported[:2]
    missing.unlink()
    stale.write_bytes(b"{}\n")
    (output / "extra.schema.json").write_bytes(b"{}\n")

    result = subprocess.run(
        [sys.executable, str(copied_script), "--check"],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 1
    assert set(result.stdout.splitlines()) == {
        f"missing: {missing.name}",
        f"stale: {stale.name}",
        "unexpected: extra.schema.json",
    }
