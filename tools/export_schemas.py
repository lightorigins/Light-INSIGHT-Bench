"""Export checked-in JSON Schemas from the public Pydantic contracts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from insight_bench._json import stable_json_bytes
from insight_bench.contracts import PUBLIC_SCHEMAS


def exported_schemas() -> dict[str, bytes]:
    """Return the canonical bytes for every public schema."""
    exports: dict[str, bytes] = {}
    for name, model in sorted(PUBLIC_SCHEMAS.items()):
        schema: dict[str, Any] = model.model_json_schema(
            mode="validation",
            ref_template="#/$defs/{model}",
        )
        schema["$id"] = f"https://www.lightorigins.com/schemas/insight-bench/v1/{name}.schema.json"
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        exports[f"{name}.schema.json"] = stable_json_bytes(schema)
    return exports


def check_exports(output: Path, exports: dict[str, bytes]) -> tuple[str, ...]:
    """Return missing, stale, or unexpected checked-in schema filenames."""
    issues: list[str] = []
    existing = {path.name for path in output.glob("*.schema.json")}
    expected = set(exports)
    issues.extend(f"missing: {name}" for name in sorted(expected - existing))
    issues.extend(f"unexpected: {name}" for name in sorted(existing - expected))
    for name in sorted(existing & expected):
        if (output / name).read_bytes() != exports[name]:
            issues.append(f"stale: {name}")
    return tuple(issues)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail instead of writing when checked-in schemas differ",
    )
    args = parser.parse_args(argv)
    output = Path(__file__).parents[1] / "src" / "insight_bench" / "schemas" / "v1"
    exports = exported_schemas()
    if args.check:
        issues = check_exports(output, exports)
        for issue in issues:
            print(issue)
        return int(bool(issues))
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in exports.items():
        (output / name).write_bytes(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
