"""``insight-bench`` command-line interface.

Six commands, in the order a user meets them: ``check-data`` (is the benchmark
data here?), ``check-policy`` (does my model answer the wire?), ``run``,
``vis``, ``pack``, ``verify``. Every command writes one JSON object to stdout
and nothing else, so its output can be piped straight into a file or a reader.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

from insight_bench._json import stable_json_text
from insight_bench._redaction import redact_text
from insight_bench.evidence import pack_evidence, verify_evidence
from insight_bench.registry import Registry
from insight_bench.runner import RunContext, RunnerError, VisOptions, run_benchmark
from insight_bench.runners._shared import vis_required_bytes
from insight_bench.runners.insight_bench import DEFAULT_POLICY_URL
from insight_bench.vis import render_run_page

#: The suite a bare ``run`` evaluates: the published INSIGHT-Bench split.
DEFAULT_BENCHMARK = "insight-bench-v1@1.0.0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="insight-bench")
    parser.add_argument("--version", action="version", version="%(prog)s 0.2.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_data = subparsers.add_parser(
        "check-data",
        help="check the episode file and the scene root without starting a simulator",
    )
    check_data.add_argument("benchmark", nargs="?", default=DEFAULT_BENCHMARK)
    _data_arguments(check_data)
    _registry_argument(check_data)

    check_policy = subparsers.add_parser(
        "check-policy",
        help="drive a policy server the way a run will, and report what it would execute",
    )
    check_policy.add_argument(
        "--policy-url",
        default=DEFAULT_POLICY_URL,
        help=f"base URL of the policy server to check (default {DEFAULT_POLICY_URL})",
    )
    check_policy.add_argument(
        "--steps", type=int, default=8, help="how many /act calls to make (default 8)"
    )
    check_policy.add_argument(
        "--instruction",
        default="walk to the sofa and stop next to it",
        help="the instruction to send with /reset",
    )
    check_policy.add_argument(
        "--timeout-sec", type=float, default=120.0, help="per-call timeout (default 120)"
    )
    check_policy.add_argument(
        "--expect-stop",
        action="store_true",
        help=(
            "also fail when the server does not stop within --steps. Off by default: the frame "
            "sent is synthetic, so a model that keeps walking is behaving normally"
        ),
    )

    run = subparsers.add_parser("run", help="evaluate a policy server on a benchmark")
    run.add_argument("benchmark", nargs="?", default=DEFAULT_BENCHMARK)
    _data_arguments(run)
    run.add_argument("--output", type=Path, default=Path("run-result.json"))
    run.add_argument("--seed", type=int, default=0)
    run.add_argument(
        "--sim-backend",
        choices=("isaac-5.1", "isaac-6.0"),
        help="select a simulator backend; isaac-5.1 is the official line, isaac-6.0 experimental",
    )
    run.add_argument(
        "--run-dir",
        "--artifact-dir",
        dest="artifact_dir",
        type=Path,
        help=(
            "persist per-episode artifacts here (<run-dir>/<episode_id>/trace.json); "
            "--artifact-dir is accepted as a compatibility alias"
        ),
    )
    run.add_argument(
        "--policy-url",
        default=DEFAULT_POLICY_URL,
        help=(
            f"base URL of the policy server to evaluate (default {DEFAULT_POLICY_URL}); "
            "the model identity of the run is read from its GET /health"
        ),
    )
    run.add_argument(
        "--max-episodes",
        type=int,
        help=(
            "stop after this many episodes. For smoke-testing a setup: the result covers part "
            "of the suite, is labelled as such, and is not a submittable measurement"
        ),
    )
    run.add_argument(
        "--resume",
        action="store_true",
        help=(
            "reuse the episodes already scored in --run-dir instead of running them again, so an "
            "interrupted run continues where it stopped. Refused unless the model, benchmark, "
            "episode file, scene root, seed and simulator all match the run that wrote them"
        ),
    )
    run.add_argument(
        "--vis",
        action="store_true",
        help=(
            "persist replay material -- one JPEG per sampled decision plus a per-episode "
            "episode.json -- into --run-dir, for `insight-bench vis`. At most 100 frames "
            "per episode: a full 1097-episode suite is refused up front unless about "
            f"{_vis_reserve_gb()} GB is free, and typically writes about 3.6 GB across roughly "
            "110,000 files. The trajectory replay on the page needs none of this; only the "
            "camera panel does"
        ),
    )
    _registry_argument(run)

    vis = subparsers.add_parser("vis", help="render a single-page run report from a run directory")
    # A directory or the run-result file itself. --output is configurable, so a
    # directory can hold a run result under any name; naming the file is the
    # unambiguous form, and the summary always reports which file was read.
    vis.add_argument(
        "run_dir",
        metavar="run_dir_or_run_result",
        type=Path,
        help="the run directory, or the run-result JSON file inside it",
    )
    # The page is always <run_dir>/vis.html: frames are resolved relative to it,
    # so an --output anywhere else would produce a page whose every frame fails
    # to load, with nothing on it to say why.
    vis.add_argument(
        "--no-open",
        action="store_true",
        help=(
            "do not open the page. Opening is already off unless stdout is a terminal; it hands "
            "the local file:// URL to the browser you registered and runs no evaluation code"
        ),
    )

    pack = subparsers.add_parser("pack", help="create a local evidence pack")
    pack.add_argument("run_result", type=Path)
    pack.add_argument("output", type=Path)
    pack.add_argument("--include", type=Path, action="append", default=[])

    verify = subparsers.add_parser("verify", help="verify evidence SHA-256 integrity")
    verify.add_argument("pack", type=Path)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "check-data":
            from insight_bench.data_check import check_data

            checked = check_data(
                args.benchmark,
                episodes=args.episode_dataset,
                scene_root=args.scene_root,
                registry=Registry.load(args.registry),
            )
            _emit(checked)
            return 0 if checked["ready"] else 1
        if args.command == "check-policy":
            from insight_bench.policy_check import check_policy

            probed = check_policy(
                args.policy_url,
                steps=args.steps,
                instruction=args.instruction,
                timeout_sec=args.timeout_sec,
            )
            _emit(probed)
            # Only wire-level failures are negative answers. "It did not stop"
            # is not one: the frame is synthetic, so a model that keeps walking
            # on it is behaving normally. Failing on it would train people to
            # ignore this command. `--expect-stop` is for a server that should.
            if not probed["model_id"]:
                return 1
            return 1 if args.expect_stop and probed["never_stopped"] else 0
        if args.command == "run":
            _require_run_dir_for_vis(args)
            result = run_benchmark(
                args.benchmark,
                registry=Registry.load(args.registry),
                seed=args.seed,
                output=args.output,
                context=RunContext(
                    artifact_dir=args.artifact_dir,
                    sim_backend=args.sim_backend,
                    episode_dataset=args.episode_dataset,
                    scene_root=args.scene_root,
                    policy_url=args.policy_url,
                    max_episodes=args.max_episodes,
                    resume=args.resume,
                    vis=VisOptions() if args.vis else None,
                ),
            )
            _emit(result)
            return 0 if result.status == "completed" else 1
        if args.command == "vis":
            # Exit 1 is reserved for "it ran and the answer is negative". A run
            # with no frames still renders a correct page that says so, which is
            # not a negative answer, so `vis` only ever exits 0 or 2.
            _emit(
                render_run_page(
                    args.run_dir,
                    open_browser=not args.no_open and sys.stdout.isatty(),
                )
            )
            return 0
        if args.command == "pack":
            _emit(pack_evidence(args.run_result, args.output, includes=args.include))
            return 0
        if args.command == "verify":
            report = verify_evidence(args.pack)
            _emit(report.to_dict())
            return 0 if report.valid else 1
    except Exception as exc:
        _emit({"error": redact_text(str(exc)) or type(exc).__name__}, stream=sys.stderr)
        return 2
    parser.error(f"unknown command: {args.command}")
    return 2


def _registry_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--registry", type=Path, help="registry root; defaults to bundled registry")


def _data_arguments(parser: argparse.ArgumentParser) -> None:
    """The two paths a licence-gated suite cannot ship, so the user points at them.

    Both are data -- filesystem paths -- never code to import, and nothing here
    ever downloads either of them.
    """
    parser.add_argument(
        "--episodes",
        dest="episode_dataset",
        required=True,
        type=Path,
        help=(
            "the published episode file for this suite; it is checked against the digest "
            "the manifest pins"
        ),
    )
    parser.add_argument(
        "--scene-root",
        required=True,
        type=Path,
        help=(
            "directory the suite's converted scene assets live under, laid out as "
            "<dataset>/<scene_id>/<asset>; scene assets are never downloaded"
        ),
    )


def _vis_reserve_gb() -> str:
    """What ``--vis`` insists is free before a full suite, in GB.

    Derived rather than written out, so the figure in ``--help`` cannot drift
    away from the constants the refusal is actually made of.
    """
    required = vis_required_bytes(
        episode_count=1097,
        num_steps=300,
        max_frames_per_episode=VisOptions().max_frames_per_episode,
    )
    return f"{required / 1e9:.1f}"


def _require_run_dir_for_vis(args: argparse.Namespace) -> None:
    """Refuse ``--vis`` with nowhere to persist to, before anything is launched.

    ``open_vis_session`` refuses too, but only once the runner has resolved the
    manifest and is about to bring a simulator up. Failing here costs nothing.
    """
    if args.vis and args.artifact_dir is None:
        raise RunnerError("--vis needs --run-dir: there is nowhere to persist frames")


def _emit(value: Any, *, stream: TextIO | None = None) -> None:
    (stream or sys.stdout).write(stable_json_text(value))


def run_cli(argv: Sequence[str] | None = None) -> int:
    """The console entry point: :func:`main`, then the guarded Isaac release.

    Everything the CLI does is in ``main``, which is what the tests drive. The
    release lives out here because it is the one step that needs the exit code:
    closing Omniverse Kit hard-exits the process with status 0, so it may run
    only once the output is written *and* the status is a clean 0. Non-zero
    codes return untouched, and a ``main`` that raised never reaches this line
    at all -- both leave Kit to process exit, which reclaims it without
    rewriting the status (see ``insight_bench.simulator.isaac.app``).
    """
    exit_code = main(argv)
    # Imported here, not at module scope, so `import insight_bench.cli` stays as
    # light and as simulator-free as the rest of the package requires.
    from insight_bench.simulator.isaac.app import release_isaac_app_on_success

    return release_isaac_app_on_success(exit_code)


if __name__ == "__main__":
    raise SystemExit(run_cli())
