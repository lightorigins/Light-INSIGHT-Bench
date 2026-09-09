"""``python -m insight_policy`` -- load one policy directory and serve it.

    python -m insight_policy --policy policies/navid \
        --model-path /weights/navid-7b-r2r-rxr \
        --opt upstream_repo=/code/NaVid-VLN-CE \
        --opt vit_path=/weights/eva_vit_g.pth

``--policy`` names a directory holding a ``policy.py`` that defines exactly one
:class:`~insight_policy.Policy` subclass -- one of the ports shipped here, or
your own copy of ``policies/template``. That file is imported from the path you
give, in this process, with your model's weights: it is your code, running in
your environment, at your explicit instruction. Point it only at a policy you
trust, the same way you would with any script you run.

``--opt key=value`` overrides one of the options that policy declares in its
``defaults`` block, and nothing else -- a key the policy never declared is an
error, so a typo cannot quietly leave the default in place.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

from insight_policy import Policy, coerce_option, resolve_defaults
from insight_policy.server import serve

POLICY_FILENAME = "policy.py"


def load_policy_class(target: Path) -> type[Policy]:
    """Import ``<target>/policy.py`` (or ``target`` itself) and find the Policy."""
    path = target if target.is_file() else target / POLICY_FILENAME
    if not path.is_file():
        raise SystemExit(f"--policy: no {POLICY_FILENAME} at {target}")

    # The policy directory goes on sys.path so a policy can import helpers next
    # to it, and its parent so it can import ones shared across policies (the
    # LLaMA-VID plumbing NaVid and Uni-NaVid both use lives one level up).
    policy_dir = path.parent.resolve()
    for entry in (policy_dir.parent, policy_dir):
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))

    module_name = "insight_hosted_policy_" + policy_dir.name
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"--policy: cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    found = [
        value
        for value in vars(module).values()
        if isinstance(value, type)
        and issubclass(value, Policy)
        and value is not Policy
        and value.__module__ == module_name
    ]
    if not found:
        raise SystemExit(f"{path}: defines no Policy subclass")
    if len(found) > 1:
        names = ", ".join(sorted(cls.__name__ for cls in found))
        raise SystemExit(f"{path}: defines several Policy subclasses ({names}); expected one")
    return found[0]


def parse_options(policy_cls: type[Policy], raw_options: list[str]) -> dict[str, Any]:
    """``["forward_m=0.5"]`` -> ``{"forward_m": 0.5}``, typed by the declared default."""
    declared = resolve_defaults(policy_cls)
    parsed: dict[str, Any] = {}
    for item in raw_options:
        key, sep, value = item.partition("=")
        key = key.strip()
        if not sep or not key:
            raise SystemExit(f"--opt expects key=value, got {item!r}")
        if key not in declared:
            known = ", ".join(sorted(declared))
            raise SystemExit(f"--opt {key}: {policy_cls.__name__} declares no such option; {known}")
        try:
            parsed[key] = coerce_option(key, declared[key], value)
        except ValueError as exc:
            raise SystemExit(f"--opt {item}: {exc}") from exc
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m insight_policy",
        description="Serve one navigation policy over the INSIGHT-Bench HTTP protocol.",
    )
    parser.add_argument(
        "--policy",
        required=True,
        help=f"directory holding a {POLICY_FILENAME} (or the file itself)",
    )
    parser.add_argument("--model-path", required=True, help="checkpoint this policy loads")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=18081, help="bind port (default 18081)")
    parser.add_argument(
        "--opt",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override one option the policy declares; repeatable",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    policy_cls = load_policy_class(Path(args.policy).expanduser())
    options = parse_options(policy_cls, args.opt)

    # Constructing the policy loads the weights. Doing it before the socket is
    # bound is the whole reason /health answering means "ready".
    print(f"[insight_policy] loading {policy_cls.__name__} from {args.model_path}", file=sys.stderr)
    policy = policy_cls(args.model_path, **options)
    serve(policy, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
