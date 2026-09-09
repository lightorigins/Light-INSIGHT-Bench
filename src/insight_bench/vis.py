"""Render a single-page HTML report from a run directory.

``insight-bench vis <run_dir_or_run_result>`` reads what a run already left on
disk -- the run result, the per-episode ``trace.json`` files, and the optional
``episode.json`` / ``vis-manifest.json`` sidecars that ``run --vis`` adds -- and
writes ``<run_dir>/vis.html``. The page is one self-contained file that
references the persisted frames relatively, so it opens over ``file://`` and
makes zero network requests.

Four properties are load-bearing and explain most of the shape of this module:

* **The page always says which file it was built from.** ``--output`` is
  configurable, so the run result is not reliably called ``run-result.json``, and
  a reader given a page for the wrong run has no way to notice. So the file that
  was read is named in the stdout summary (``run_result``) and in the page's spec
  block, a run result under any name in the directory the user named beats one
  outside it, an ambiguous directory is refused rather than guessed at, and the
  one remaining fallback -- to the parent directory -- warns when it fires. See
  :func:`_locate_run_result`.

* **Everything read here is untrusted.** Episode ids, instructions, policy
  ``raw_output`` and every metadata value come from a user-supplied dataset or
  from a third-party policy server. They reach the page through exactly one
  channel -- a ``<script type="application/json">`` block escaped by
  :func:`_embed_json` -- and the page assigns them with ``textContent``. The one
  string that becomes an attribute is a frame path, validated with the same
  :func:`~insight_bench._paths.validate_relative_path` / :func:`safe_join` pair
  the registry and evidence paths use.
* **Non-finite measures are normalised on the read side.** The trace writer
  serialises with a bare ``json.dumps`` and the online measures legitimately
  return ``float("inf")`` on a goal-less episode, so a ``trace.json`` on disk may
  hold the literal token ``Infinity`` nested at any depth. ``JSON.parse`` rejects
  it and so does ``allow_nan=False``, so :func:`_finite` recurses.
* **The page shows no run-level metric.** It carried a row of cards -- SR_task,
  SR@1/2/3m, SPL, NE -- re-stating ``RunResult.metrics`` under display names,
  with a Wilson interval and SR at fixed radii computed here because the contract
  does not carry them. The owner removed that section, so the arithmetic behind
  it and the metrics echo in the payload went with it rather than staying as
  numbers nothing on the page can show. ``run`` still prints the dict and
  ``run-result.json`` still records it, untouched by anything here. What this
  module still derives from the episodes is counting: successes over episodes per
  category, which is exactly the ``status == "passed"`` the radars and the matrix
  are built from, so there is no second averaging convention to explain.

Standard library only: this module must import in a ``pip install insight_bench``
environment with no extras.
"""

from __future__ import annotations

import json
import math
import webbrowser
from pathlib import Path
from typing import Any, Final

from insight_bench._json import load_json
from insight_bench._paths import UnsafePathError, safe_join, validate_relative_path
from insight_bench._redaction import redact_data, redact_text
from insight_bench.vis_page import PAGE_TEMPLATE

#: Episodes whose per-step replay data is inlined into the page. Every episode
#: still appears in the table and in every aggregate; this bounds only the
#: per-step arrays, which are what make the file large. The page states the
#: ratio when the limit bites, so it is never mistaken for missing data.
#:
#: The number is a page-weight budget, not a round figure. Frames are referenced
#: by relative path rather than inlined, so an episode's replay costs only its
#: per-step JSON -- measured at about 4 KB per episode on a published run, whose
#: 1097 episodes make a 5.9 MB page. This cap is therefore worth roughly 9 MB,
#: which is about as large as a single file can be and still open promptly over
#: ``file://``. It was 200, which left that same run showing a replay for under
#: a fifth of its episodes: to a reader who did not find the warning explaining
#: the ratio, indistinguishable from missing data.
VIS_MAX_REPLAY_EPISODES: Final = 2000

#: Replay states inlined per episode. ``run --vis`` samples frames at the same
#: cap, and the page reuses the stride the run recorded when there is one.
VIS_MAX_REPLAY_STATES: Final = 100

#: Length at which a free-text value read off disk is truncated for the page.
#: Used for the manifest's own skip reasons, which a run writes and which are
#: therefore of unbounded length. It also bounded the policy's ``raw_output``
#: while the frame panel printed it.
VIS_MAX_CLIPPED_CHARS: Final = 160

#: Half-width, in success-rate points, at which the matrix diverging scale
#: saturates. Written into the page legend: a colour without its centre and its
#: cap cannot be read.
MATRIX_DIVERGING_SPAN: Final = 0.20

#: Fraction of the radar radius the outermost ring sits at, lifted per run when a
#: category exceeds it so a polyline can never leave the canvas.
RADAR_EDGE_DEFAULT: Final = 0.70

#: Display names for the retired category values, mapped onto the vocabulary the
#: published suite now carries. The current values need no entry: they are already
#: the display name, and anything unlisted passes through verbatim -- the page's
#: own copy is English, but a data value is shown as it is, not guessed at and not
#: dropped. This map exists only so a run directory produced against the retired
#: ``1.0.0``/``2.0.0`` files still renders English, and the same words as a
#: current run, rather than putting Chinese on the page.
CATEGORY_LABELS: Final[dict[str, str]] = {
    "小住宅": "Apartment",
    "大住宅": "House",
    "商业": "Commercial",
    "公共机构": "Institution",
    "outdoor": "Outdoor",
    "base": "Base",
    "direction": "Direction",
    "relation": "Relation",
    "extremum": "Extremum",
    "ordinal": "Ordinal",
}

#: Axis and column order for the breakdown. One order for both charts, so a
#: reader comparing them is not re-learning the layout; values not listed here
#: follow in sorted order and ``Unknown`` is always last.
INSTR_TYPE_ORDER: Final = ("Base", "Direction", "Relation", "Extremum", "Ordinal")
SCENE_CLASS_ORDER: Final = ("Apartment", "House", "Commercial", "Institution", "Outdoor")

#: The bucket an episode with no category value falls into. Rendered only when
#: something is actually in it.
UNKNOWN_CATEGORY: Final = "Unknown"

#: Diverging endpoints for the matrix, centred on the run's own success rate:
#: warm below, cool above. These encode data, so they are fixed here rather than
#: taken from the page's chrome tokens. The three are the published figure's own
#: palette, so a page and ``assets/insight_bench_matrix.png`` read the same way.
_MATRIX_CENTRE_RGB: Final = (0xFF, 0xFF, 0xFF)
_MATRIX_LOW_RGB: Final = (0xED, 0x81, 0x79)
_MATRIX_HIGH_RGB: Final = (0x6A, 0x9D, 0xDF)

#: The episode table's columns, in display order, as
#: ``(key, label, source, filterable)``. ``source`` says where the value comes
#: from: the episode's status, a metric, a categorical score label, or the
#: episode sidecar. ``#``, ``Episode`` and ``Frames`` are in the page skeleton;
#: everything here appears only when it varies across the run (see
#: :func:`_variable_columns`), unless it is in :data:`_ALWAYS_COLUMNS`.
#:
#: The order, the labels and which columns carry a filter follow an existing
#: internal episode browser, so a reader who uses both is not re-learning the
#: table. Three of its columns have no counterpart here -- progress, first-failed
#: subtask and primary event -- because no score term this distribution ships
#: computes them; they are absent rather than rendered blank. Its own SPL column
#: is likewise absent from ours because it is absent from its table too, even
#: though its API returns the number.
#:
#: One deliberate divergence: instruction type and scene class stand where that
#: browser puts its difficulty column. Difficulty exists there to let a reader
#: narrow the table to one kind of episode, and on this benchmark it cannot --
#: every episode of the published suite is ``LR_towards``, from the dataset label
#: and from the NavNuances category alike. The two keys that do vary here are
#: instruction type and scene class, and they are the axes of the radars and the
#: matrix directly above the table, so filtering on them is what carries a reader
#: from a matrix cell to the episodes inside it.
#: ``suite`` is deliberately absent. Its values are the generator's own batch
#: names -- ``typed_v3``, ``human_manual``, ``outdoor_batch`` -- which say how an
#: episode came to be written rather than anything about the episode, and the
#: axes a reader wants are the two below.
_CANDIDATE_COLUMNS: Final = (
    ("scene_id", "Scene", "episode", False),
    ("status", "Result", "status", True),
    ("instr_type", "Instr type", "episode", True),
    ("scene_class", "Scene class", "episode", True),
    ("distance_to_goal", "NE (m)", "metric", False),
)

#: Columns kept even when every episode shares one value. The result of an
#: episode is what the table is for, and a run in which everything passed still
#: has to show that rather than drop the column as carrying no information.
_ALWAYS_COLUMNS: Final = frozenset({"status"})

#: Keys copied out of an ``episode.json`` sidecar. An allowlist, not the whole
#: file: the sidecar is written from an allowlist for the same reason, and a
#: future episode field carrying a host path must not reach the page by default.
_EPISODE_KEYS: Final = (
    "instruction",
    "scene_id",
    "goal_position",
    "goal_radius_m",
    "reference_path",
    "suite",
    "difficulty_label",
    "instr_type",
    "scene_class",
)

#: Status groups in the order replay is handed out: the ones you triage first.
_REPLAY_PRIORITY: Final = ("error", "failed", "passed")

#: Per-step arrays the page reads. Six more were carried until the frame panel
#: stopped printing them: ``dtg`` and ``raw`` fed a diagnostic line under the
#: transport, and ``apos_*`` / ``opos_*`` fed a pointing overlay drawn through an
#: unverified decode. All of it stays in ``run-result.json`` and the traces; none
#: of it belongs in the payload of a page that no longer draws it.
_REPLAY_COLUMNS: Final = (
    "step",
    "from",
    "to",
    "frame",
    "waypoint",
)


class VisError(RuntimeError):
    """Raised when a run directory cannot be turned into a page at all."""


def render_run_page(target: Path, *, open_browser: bool = False) -> dict[str, Any]:
    """Write ``<run_dir>/vis.html`` and return the summary to print on stdout.

    ``target`` is either the run directory or the run-result file itself; a file
    names its own run directory, which is the parent it sits in. The output
    location is not configurable on purpose: frames resolve relative to the page,
    so a page written anywhere else would load none of them and give its reader
    nothing to diagnose that with.
    """
    target = Path(target)
    named_file = target.is_file()
    if named_file:
        run_dir = target.parent
    elif target.is_dir():
        run_dir = target
    else:
        raise VisError(f"not a directory or a run-result file: {target}")
    # The guard is on the directory the page would be written into, not on what
    # was named: a pack holds a run-result.json too, so naming that file would
    # otherwise walk straight past this and leave the extra artifact that makes
    # `insight-bench verify` fail.
    if (run_dir / "evidence-manifest.json").is_file():
        raise VisError(
            "this looks like an evidence pack, not a run directory. `vis` writes vis.html "
            "into the directory it reads, and an extra file makes `insight-bench verify` "
            "fail. Point it at the --run-dir the run wrote instead."
        )
    located: list[str] = []
    if named_file:
        result_path = target
    else:
        result_path, located = _locate_run_result(run_dir)
    payload = load_json(result_path)
    if not isinstance(payload, dict):
        raise VisError(f"{result_path.name} is not a JSON object")

    warnings: list[str] = list(located)
    manifest = _load_manifest(run_dir, warnings)
    episodes = _episode_rows(run_dir, payload, warnings)
    replay, traceless = _collect_replay(episodes, manifest, warnings)
    source = _source_label(result_path, run_dir)
    data = _build_payload(payload, episodes, replay, traceless, manifest, warnings, source)

    output = run_dir / "vis.html"
    page = PAGE_TEMPLATE.replace("__INSIGHT_BENCH_DATA__", _embed_json(data))
    output.write_text(page, encoding="utf-8")

    opened = _open_in_browser(output) if open_browser else False
    return {
        "episodes": len(episodes),
        "episodes_with_categories": sum(1 for row in episodes if row["categories"]),
        "episodes_with_frames": sum(1 for entry in replay.values() if entry["frames_present"]),
        "episodes_with_replay": len(replay),
        "frame_stride": data["limits"]["frame_stride"],
        "frames": sum(len(entry["frames_present"]) for entry in replay.values()),
        "opened_browser": opened,
        "output": output.name,
        # Which file the numbers on the page came from. `vis` accepts a
        # directory, so without this the summary cannot distinguish a page built
        # from the run you pointed at from one built from a neighbouring file.
        "run_result": source,
        "warnings": warnings,
    }


# --------------------------------------------------------------------------- #
# Reading the run directory
# --------------------------------------------------------------------------- #


#: Top-level keys every run result carries, used to tell one apart from the
#: other JSON a run directory holds. Deliberately weaker than validating against
#: :class:`~insight_bench.contracts.RunResult`: the whole read side of this module
#: tolerates a partial or hand-edited payload and says on the page what is
#: missing, so recognition must not be stricter than rendering.
_RUN_RESULT_KEYS: Final = ("run_id", "benchmark_id", "episodes")

#: JSON a run directory holds that is never a run result. Sidecars live in the
#: per-episode subdirectories rather than at the top level, but naming them is
#: cheaper than relying on that.
_NOT_A_RUN_RESULT: Final = frozenset({"vis-manifest.json", "episode.json", "trace.json"})


def _locate_run_result(run_dir: Path) -> tuple[Path, list[str]]:
    """The run result for this directory, plus any warning about where it came from.

    Resolution is deliberately biased towards *this* directory, because the cost
    of the alternatives is not symmetric: failing to find a run result is a loud
    error the user can act on, while quietly reading a neighbouring one renders a
    correct-looking page for somebody else's numbers.

    1. ``<run_dir>/run-result.json``, the default ``--output`` name, wins outright.
    2. Otherwise any other top-level ``*.json`` that looks like a run result. One
       such file is the answer -- ``--output`` is configurable, so a run result
       under a chosen name is an ordinary layout, not an oddity. Several are
       ambiguous and refused by name; guessing between them is exactly the silent
       substitution this function exists to prevent.
    3. Only when this directory holds none: ``<run_dir>/../run-result.json``,
       which is what ``--run-dir runs --output run-result.json`` leaves behind.
       That layout is real, so the fallback stays -- but it reads a file outside
       the directory the user named, so it warns, on stdout and on the page.
    """
    default = run_dir / "run-result.json"
    if default.is_file():
        return default, []
    candidates = _run_result_candidates(run_dir)
    if len(candidates) == 1:
        return candidates[0], []
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise VisError(
            f"{run_dir} holds {len(candidates)} files that look like run results ({names}) and "
            "none is named run-result.json, so which run to render is ambiguous. Pass the one "
            "you mean: `insight-bench vis <path-to-run-result.json>`."
        )
    parent = run_dir.parent / "run-result.json"
    if parent.is_file():
        return parent, [
            f"no run result in {run_dir}; rendered {parent} from the parent directory instead. "
            "Check that this is the run you meant -- pass the file directly to be sure."
        ]
    probed = ", ".join(str(path) for path in (default, parent))
    raise VisError(
        f"no run result found; probed {probed} and found no other run result in {run_dir}"
    )


def _run_result_candidates(run_dir: Path) -> list[Path]:
    """Top-level ``*.json`` files in ``run_dir`` whose content is a run result.

    Unreadable and unparseable files are skipped rather than raised on: a run
    directory is untrusted input, and one broken file beside a good run result
    must not be able to stop the good one from rendering.
    """
    found: list[Path] = []
    for path in sorted(run_dir.glob("*.json")):
        if path.name in _NOT_A_RUN_RESULT or not path.is_file():
            continue
        try:
            payload = load_json(path)
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict) and all(key in payload for key in _RUN_RESULT_KEYS):
            found.append(path)
    return found


def _source_label(result_path: Path, run_dir: Path) -> str:
    """How to name the file the page was built from, for stdout and the page.

    A file inside the run directory is named relatively, which is what the reader
    sees beside the page; anything else keeps the path it was actually read from,
    because the whole point of showing it is that it is somewhere unexpected.
    """
    try:
        return result_path.relative_to(run_dir).as_posix()
    except ValueError:
        return str(result_path)


def _load_manifest(run_dir: Path, warnings: list[str]) -> dict[str, Any]:
    path = run_dir / "vis-manifest.json"
    if not path.is_file():
        warnings.append(
            "this run was recorded without --vis, so the page has no camera frames and no "
            "per-episode categories to break the results down by"
        )
        return {}
    try:
        loaded = _finite(load_json(path))
    except (OSError, ValueError) as exc:
        warnings.append(f"vis-manifest.json could not be read: {redact_text(str(exc))}")
        return {}
    if not isinstance(loaded, dict):
        return {}
    _recording_failures(loaded, warnings)
    return loaded


def _recording_failures(manifest: dict[str, Any], warnings: list[str]) -> None:
    """Report what ``run --vis`` counted but could not write.

    The counters exist nowhere else. Neither loss fails the run -- a frame that
    will not encode and a sidecar that will not serialise both cost their
    picture and one line in the manifest, on purpose -- so if the page does not
    read them, a run whose sidecars all failed looks exactly like a run recorded
    without ``--vis`` at all, and a run missing half its frames looks like a run
    the policy stopped early. The reason string is clipped as well as redacted:
    the manifest is read as untrusted input here, like every other file under
    the run directory.
    """
    for count_key, reason_key, subject in (
        ("frames_failed", "frame_failure_reason", "camera frames"),
        ("sidecars_failed", "sidecar_failure_reason", "episode.json sidecars"),
    ):
        count = _int_or_none(manifest.get(count_key)) or 0
        if count < 1:
            continue
        reason = _clip(manifest.get(reason_key))
        detail = f"; first reason: {reason}" if reason else ""
        warnings.append(f"the run recorded {count} {subject} it could not write{detail}")


def _episode_rows(
    run_dir: Path, payload: dict[str, Any], warnings: list[str]
) -> list[dict[str, Any]]:
    """One row per ``EpisodeRunResult``, joined with its ``episode.json`` sidecar.

    Every sidecar is read, not only the ones that get an inlined replay: the
    breakdown's per-cell counts have to sum to the run totals, which needs every
    episode's category keys. A sidecar is a few hundred bytes, so the whole set
    stays well under a megabyte even for the largest published suite.
    """
    raw_episodes = payload.get("episodes")
    if not isinstance(raw_episodes, list) or not raw_episodes:
        raise VisError("run result carries no episodes")
    rows: list[dict[str, Any]] = []
    unreadable = 0
    for index, item in enumerate(raw_episodes):
        if not isinstance(item, dict):
            continue
        episode_id = str(item.get("episode_id", ""))
        response = item.get("response")
        response = response if isinstance(response, dict) else {}
        metadata = response.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        labels = metadata.get("score_labels")
        episode_dir = _episode_dir(run_dir, episode_id)
        sidecar, failed = _load_sidecar(episode_dir)
        unreadable += failed
        rows.append(
            {
                "categories": bool(sidecar.get("instr_type") or sidecar.get("scene_class")),
                "dir": episode_dir,
                "episode_id": episode_id,
                "error": _text(item.get("error")),
                "failure_reason": _text(metadata.get("failure_reason")),
                "index": index + 1,
                "labels": _labels(labels),
                "metrics": _numbers(item.get("metrics")),
                "status": _text(item.get("status")),
                "sidecar": sidecar,
            }
        )
    if unreadable:
        warnings.append(f"{unreadable} episode.json sidecars could not be read and were skipped")
    return rows


def _episode_dir(run_dir: Path, episode_id: str) -> Path | None:
    """The per-episode directory, or ``None`` when the id is not a safe path.

    ``EpisodeSpec`` already forces an episode id to a single slug, but ``vis``
    reads ``run-result.json`` off disk rather than an ``EpisodeSpec``, so on this
    path this is the only check there is.
    """
    try:
        return safe_join(run_dir, episode_id)
    except (UnsafePathError, ValueError):
        return None


def _load_sidecar(episode_dir: Path | None) -> tuple[dict[str, Any], int]:
    if episode_dir is None:
        return {}, 0
    path = episode_dir / "episode.json"
    if not path.is_file():
        return {}, 0
    try:
        loaded = load_json(path)
    except (OSError, ValueError):
        return {}, 1
    if not isinstance(loaded, dict):
        return {}, 1
    sidecar: dict[str, Any] = {}
    for key in _EPISODE_KEYS:
        value = loaded.get(key)
        if value is not None:
            sidecar[key] = redact_data(_finite(value), key=key)
    return sidecar, 0


def _labels(labels: Any) -> dict[str, str]:
    if not isinstance(labels, dict):
        return {}
    return {str(key): _text(str(value)) for key, value in labels.items()}


# --------------------------------------------------------------------------- #
# Replay extraction
# --------------------------------------------------------------------------- #


def _collect_replay(
    episodes: list[dict[str, Any]],
    manifest: dict[str, Any],
    warnings: list[str],
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Per-step arrays for the episodes that get an inlined replay.

    Also the ids that were selected for a replay and had no usable trace beside
    them, because "there is no trace.json here" and "this episode is past the
    inlining limit" are different facts and the page has to say which.
    """
    selected = _replay_selection(episodes)
    if len(selected) < len(episodes):
        warnings.append(
            f"replay data is inlined for {len(selected)} of {len(episodes)} episodes (page limit; "
            "error episodes first, then failed, then passed) -- the table and the breakdown cover "
            "all of them"
        )
    stride_hint = manifest.get("frame_stride")
    replay: dict[str, dict[str, Any]] = {}
    traceless: set[str] = set()
    skipped_frames = 0
    first_reason = ""
    frameless = 0
    for row in selected:
        episode_dir = row["dir"]
        trace_path = None if episode_dir is None else episode_dir / "trace.json"
        if trace_path is None or not trace_path.is_file():
            traceless.add(row["episode_id"])
            continue
        try:
            trace = _finite(load_json(trace_path))
        except (OSError, ValueError) as exc:
            warnings.append(f"trace.json could not be read: {redact_text(str(exc))}")
            traceless.add(row["episode_id"])
            continue
        if not isinstance(trace, dict):
            traceless.add(row["episode_id"])
            continue
        entry, skipped, reason = _replay_entry(row, trace, episode_dir, stride_hint)
        skipped_frames += skipped
        first_reason = first_reason or reason
        if entry is None:
            traceless.add(row["episode_id"])
            continue
        if not entry["frames_present"]:
            frameless += 1
        replay[row["episode_id"]] = entry
    if skipped_frames:
        detail = f": {first_reason}" if first_reason else ""
        warnings.append(f"{skipped_frames} frame paths failed validation and were skipped{detail}")
    if frameless and manifest:
        warnings.append(f"{frameless} episodes with an inlined replay have no usable camera frames")
    if traceless:
        warnings.append(f"{len(traceless)} episodes have no trace.json and cannot be replayed")
    return replay, traceless


def _replay_selection(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Error episodes first, then failed, then passed; each group by episode id."""
    ordered: list[dict[str, Any]] = []
    for status in _REPLAY_PRIORITY:
        ordered.extend(
            sorted(
                (row for row in episodes if row["status"] == status),
                key=lambda row: str(row["episode_id"]),
            )
        )
    covered = {row["episode_id"] for row in ordered}
    ordered.extend(row for row in episodes if row["episode_id"] not in covered)
    return ordered[:VIS_MAX_REPLAY_EPISODES]


def _replay_entry(
    row: dict[str, Any],
    trace: dict[str, Any],
    episode_dir: Path,
    stride_hint: Any,
) -> tuple[dict[str, Any] | None, int, str]:
    """Columnar per-step data for one episode.

    A replay state is a trace step with a decision behind it, so step 0 -- the
    pre-action pose, which no policy ever saw -- is not one. Parallel arrays
    rather than an array of objects, and a column that is empty for the whole
    episode is dropped, which is what keeps a non-pointing policy's pages small.
    """
    steps = trace.get("steps")
    if not isinstance(steps, list) or len(steps) < 2:
        return None, 0, ""
    stride = _stride(len(steps) - 1, stride_hint)
    poses = [_pose(step) for step in steps]
    columns: dict[str, list[Any]] = {name: [] for name in _REPLAY_COLUMNS}
    frames_present: list[int] = []
    skipped = 0
    first_reason = ""
    for index in range(1, len(steps)):
        if (index - 1) % stride:
            continue
        if len(columns["step"]) >= VIS_MAX_REPLAY_STATES:
            break
        step = steps[index] if isinstance(steps[index], dict) else {}
        frame, reason = _frame_reference(step, episode_dir, str(row["episode_id"]))
        if reason:
            skipped += 1
            first_reason = first_reason or reason
        if frame:
            frames_present.append(index)
        columns["step"].append(int(_number(step.get("step"), default=float(index))))
        columns["from"].append(poses[index - 1])
        columns["to"].append(poses[index])
        columns["frame"].append(frame)
        columns["waypoint"].append(_waypoint(step))
    if not columns["step"]:
        return None, skipped, first_reason
    entry: dict[str, Any] = {
        name: values
        for name, values in columns.items()
        if name in {"step", "from", "to"} or any(_present(value) for value in values)
    }
    entry["frames_present"] = frames_present
    entry["instruction"] = _text(trace.get("instruction"))
    entry["stride"] = stride
    entry["termination"] = _text(trace.get("termination_reason"))
    entry["total_steps"] = len(steps)
    hfov = _hfov_deg(steps)
    if hfov is not None:
        entry["hfov_deg"] = hfov
    entry.update(_geometry(row["sidecar"]))
    return entry, skipped, first_reason


def _geometry(sidecar: dict[str, Any]) -> dict[str, Any]:
    """Goal and reference path for the top-down panel, in ground-plane metres.

    From the episode sidecar, so the panel has geometry only when the run wrote
    one; the trajectory itself comes from the trace and is always drawable.
    """
    geometry: dict[str, Any] = {}
    goal = _xy(sidecar.get("goal_position"))
    if goal is not None:
        geometry["goal"] = goal
        geometry["goal_radius_m"] = _number(sidecar.get("goal_radius_m"), default=0.0)
    path = [point for point in map(_xy, sidecar.get("reference_path") or []) if point]
    if len(path) > 1:
        geometry["reference_path"] = path
    return geometry


def _xy(value: Any) -> list[float] | None:
    if not isinstance(value, list | tuple) or len(value) < 2:
        return None
    return [round(_number(value[0], default=0.0), 2), round(_number(value[1], default=0.0), 2)]


def _stride(states: int, hint: Any) -> int:
    """Replay stride: the one the run recorded, else one that fits the cap."""
    recorded = _int_or_none(hint)
    if recorded is not None and recorded >= 1:
        return recorded
    return max(1, math.ceil(states / VIS_MAX_REPLAY_STATES)) if states > 0 else 1


def _frame_reference(step: dict[str, Any], episode_dir: Path, episode_id: str) -> tuple[str, str]:
    """The page-relative frame path for this step, or ``("", reason)``.

    Keyed off ``observation.frame_path`` as recorded, never rebuilt from the step
    index: that field means "the frame the policy was shown for the decision this
    step records", and only the producer knows which file that is.
    """
    observation = step.get("observation")
    raw = observation.get("frame_path") if isinstance(observation, dict) else None
    if not isinstance(raw, str) or not raw:
        return "", ""
    try:
        relative = validate_relative_path(raw)
        target = safe_join(episode_dir, relative)
    except (UnsafePathError, ValueError):
        # The reason names the class of failure and not the rejected string. A
        # path that failed validation is attacker-controlled, and quoting it
        # back into the page would put it in the one place we just kept it out
        # of.
        return "", f"unsafe frame path recorded for episode {episode_id}"
    if not target.is_file():
        return "", f"no such frame file under episode {episode_id}"
    return f"{episode_id}/{relative}", ""


def _hfov_deg(steps: list[Any]) -> float | None:
    """Horizontal field of view from the first step that recorded intrinsics."""
    for step in steps:
        if not isinstance(step, dict):
            continue
        observation = step.get("observation")
        if not isinstance(observation, dict):
            continue
        intrinsics = observation.get("intrinsics")
        rgb = observation.get("rgb")
        shape = rgb.get("shape") if isinstance(rgb, dict) else None
        if not isinstance(intrinsics, list) or not isinstance(shape, list) or len(shape) < 2:
            continue
        try:
            fx = float(intrinsics[0][0])
            width = float(shape[1])
        except (IndexError, TypeError, ValueError):
            continue
        if fx > 0.0 and width > 0.0:
            return round(math.degrees(2.0 * math.atan(width / (2.0 * fx))), 3)
    return None


# --------------------------------------------------------------------------- #
# Aggregates
# --------------------------------------------------------------------------- #


def _build_payload(
    payload: dict[str, Any],
    episodes: list[dict[str, Any]],
    replay: dict[str, dict[str, Any]],
    traceless: set[str],
    manifest: dict[str, Any],
    warnings: list[str],
    source: str,
) -> dict[str, Any]:
    total = len(episodes)
    successes = sum(1 for row in episodes if row["status"] == "passed")
    rate = successes / total if total else 0.0
    adapter = payload.get("adapter")
    adapter = adapter if isinstance(adapter, dict) else {}
    attestation = payload.get("runtime_attestation")
    columns = _variable_columns(episodes)
    strides = {entry["stride"] for entry in replay.values()}
    return {
        "attestation": redact_data(attestation) if isinstance(attestation, dict) else None,
        "breakdown": _breakdown(episodes, rate),
        "camera": _camera(manifest),
        "columns": columns,
        # The run's own episode summary: the count the page prints beside the
        # run id, and the k/n it comes from. The Wilson interval and SR at fixed
        # radii used to be here too, computed for the metric cards; the cards are
        # gone and so is that arithmetic, rather than leaving numbers in the
        # payload that nothing on the page can show.
        "derived": {"success_rate": rate, "successes": successes, "total": total},
        "episodes": [_table_row(row, columns, replay, traceless) for row in episodes],
        "limits": {
            "frame_stride": next(iter(strides)) if len(strides) == 1 else None,
            "max_frames_per_episode": VIS_MAX_REPLAY_STATES,
            "max_replay_episodes": VIS_MAX_REPLAY_EPISODES,
        },
        "replay": {key: _without(entry, "frames_present") for key, entry in replay.items()},
        "run": {
            "adapter_id": _text(adapter.get("adapter_id")),
            "benchmark": _coordinate(payload),
            "model_id": _text(adapter.get("model_id")),
            "run_fingerprint": _text(payload.get("run_fingerprint")),
            "run_id": _text(payload.get("run_id")),
            "seed": _int_or_none(payload.get("seed")),
            # The one sentence saying this run covered part of the suite. It was
            # in the run result and in the terminal and nowhere on the page, so
            # a ten-of-1097 run rendered as an ordinary result -- and the page is
            # the artifact that gets screenshotted and outlives the terminal.
            "episode_subset": _text(
                adapter.get("metadata", {}).get("episode_subset")
                if isinstance(adapter.get("metadata"), dict)
                else ""
            ),
            # Named on the page so a reader can tell which file these numbers
            # are, without having to trust that `vis` picked the obvious one.
            "source": source,
            "status": _text(payload.get("status")),
        },
        "warnings": warnings,
    }


def _coordinate(payload: dict[str, Any]) -> str:
    benchmark = _text(payload.get("benchmark_id"))
    version = _text(payload.get("benchmark_version"))
    return f"{benchmark}@{version}" if benchmark and version else benchmark


def _camera(manifest: dict[str, Any]) -> dict[str, float] | None:
    """The resolved camera block ``run --vis`` recorded, numbers only."""
    camera = manifest.get("camera")
    if not isinstance(camera, dict):
        return None
    numbers = {
        str(key): float(value)
        for key, value in camera.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }
    return numbers or None


def _variable_columns(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Candidate columns that carry more than one distinct value in this run.

    One pass, one rule: a column needs two distinct *present* values to earn its
    place. A column that is empty everywhere, or prints one string on every row,
    tells a reader nothing and contributes a filter whose only option selects
    the whole run. Which columns those are depends on the suite, so it is decided
    from the data rather than hard-coded per benchmark.

    "Present" is the load-bearing word, and it is why this counts values rather
    than cells. Counting cells made a missing value a value of its own, so a key
    that every scored episode agreed on looked variable as soon as one episode
    lacked it -- which is exactly what a run with a single errored episode looks
    like, since an episode that never scored carries no score labels. The
    published suite shipped a whole Difficulty column that way: 1096 rows of
    ``LR_towards``, one blank, and a two-option filter over a constant.
    """
    kept: list[dict[str, Any]] = []
    for key, label, source, filterable in _CANDIDATE_COLUMNS:
        present: set[str] = set()
        for row in episodes:
            value = _column_value(row, key, source)
            if value is not None and value != "":
                present.add(str(value))
            if len(present) > 1:
                break
        if len(present) > 1 or key in _ALWAYS_COLUMNS:
            kept.append({"filter": filterable, "key": key, "label": label, "source": source})
    return kept


def _column_value(row: dict[str, Any], key: str, source: str) -> Any:
    if source == "status":
        return row["status"]
    if source == "metric":
        return row["metrics"].get(key)
    if source == "label":
        return row["labels"].get(key)
    return row["sidecar"].get(key)


def _table_row(
    row: dict[str, Any],
    columns: list[dict[str, Any]],
    replay: dict[str, dict[str, Any]],
    traceless: set[str],
) -> dict[str, Any]:
    entry = replay.get(row["episode_id"])
    return {
        "no_trace": entry is None and row["episode_id"] in traceless,
        # The status column reads the row's own ``status`` rather than a cell, so
        # it is not copied per episode: on a thousand-episode run that is a
        # kilobyte of the same six words.
        "cells": {
            column["key"]: _finite(_column_value(row, column["key"], column["source"]))
            for column in columns
            if column["source"] != "status"
        },
        "episode_id": row["episode_id"],
        "frames": len(entry["frames_present"]) if entry else None,
        "note": row["error"] or row["failure_reason"],
        "replay": entry is not None,
        "status": row["status"],
    }


def _breakdown(episodes: list[dict[str, Any]], overall: float) -> dict[str, Any] | None:
    """Two radars and one matrix over the episode category keys, or ``None``.

    Counting only -- successes over episodes, success being exactly the
    ``status == "passed"`` the headline success rate is built from -- so there is
    no second averaging convention to explain. ``None`` when no episode carries
    category keys at all, which is what a run recorded without ``--vis`` looks
    like; the page then says so rather than drawing an empty shell.
    """
    if not any(row["categories"] for row in episodes):
        return None
    instr = _category_counts(episodes, "instr_type", INSTR_TYPE_ORDER)
    scene = _category_counts(episodes, "scene_class", SCENE_CLASS_ORDER)
    rates = [cell["rate"] for cell in instr + scene if cell["rate"] is not None]
    edge = math.ceil(max([RADAR_EDGE_DEFAULT, *rates]) * 10.0) / 10.0
    return {
        "edge": edge,
        "instr_type": [_radar_axis(cell, edge) for cell in instr],
        "matrix": _matrix(episodes, instr, scene, overall),
        "overall": overall,
        "reference_frac": min(1.0, overall / edge) if edge else 0.0,
        "scene_class": [_radar_axis(cell, edge) for cell in scene],
        "span": MATRIX_DIVERGING_SPAN,
    }


def _category_counts(
    episodes: list[dict[str, Any]], key: str, order: tuple[str, ...]
) -> list[dict[str, Any]]:
    tally: dict[str, list[int]] = {}
    for row in episodes:
        counts = tally.setdefault(_category_name(row, key), [0, 0])
        counts[0] += 1
        counts[1] += row["status"] == "passed"
    names = [name for name in order if name in tally]
    names += sorted(name for name in tally if name not in order and name != UNKNOWN_CATEGORY)
    if UNKNOWN_CATEGORY in tally:
        names.append(UNKNOWN_CATEGORY)
    return [
        {
            "episodes": tally[name][0],
            "label": CATEGORY_LABELS.get(name, name),
            "name": name,
            "rate": (tally[name][1] / tally[name][0]) if tally[name][0] else None,
            "successes": tally[name][1],
        }
        for name in names
    ]


def _radar_axis(cell: dict[str, Any], edge: float) -> dict[str, Any]:
    rate = cell["rate"]
    frac = 0.0 if rate is None or not edge else min(1.0, rate / edge)
    return {**cell, "frac": frac}


def _matrix(
    episodes: list[dict[str, Any]],
    instr: list[dict[str, Any]],
    scene: list[dict[str, Any]],
    overall: float,
) -> dict[str, Any]:
    """Scene class by instruction type, with row, column and grand totals.

    The row and column totals are the per-category counts the radars use, so the
    two charts and the matrix cannot disagree; the grand total is the run total.
    """
    cells: dict[tuple[str, str], list[int]] = {}
    for row in episodes:
        key = (_category_name(row, "scene_class"), _category_name(row, "instr_type"))
        counts = cells.setdefault(key, [0, 0])
        counts[0] += 1
        counts[1] += row["status"] == "passed"
    grand = [
        sum(cell["episodes"] for cell in instr),
        sum(cell["successes"] for cell in instr),
    ]
    return {
        "columns": [{"episodes": cell["episodes"], "label": cell["label"]} for cell in instr],
        "grand_total": _matrix_cell(grand, overall),
        "rows": [
            {
                "cells": [
                    _matrix_cell(cells.get((line["name"], column["name"])), overall)
                    for column in instr
                ],
                "label": line["label"],
                "total": _matrix_cell([line["episodes"], line["successes"]], overall),
            }
            for line in scene
        ],
        "totals": [_matrix_cell([cell["episodes"], cell["successes"]], overall) for cell in instr],
    }


def _category_name(row: dict[str, Any], key: str) -> str:
    value = row["sidecar"].get(key)
    return value if isinstance(value, str) and value else UNKNOWN_CATEGORY


def _matrix_cell(counts: list[int] | None, overall: float) -> dict[str, Any]:
    episodes = counts[0] if counts else 0
    successes = counts[1] if counts else 0
    if not episodes:
        return {"color": _diverging(0.0), "episodes": 0, "rate": None, "successes": 0}
    rate = successes / episodes
    return {
        "color": _diverging(rate - overall),
        "episodes": episodes,
        "rate": rate,
        "successes": successes,
    }


def _diverging(delta: float) -> str:
    """Warm below the run's own success rate, cool above, saturating at the span."""
    position = max(-1.0, min(1.0, delta / MATRIX_DIVERGING_SPAN))
    end = _MATRIX_HIGH_RGB if position >= 0.0 else _MATRIX_LOW_RGB
    mix = abs(position)
    pairs = zip(_MATRIX_CENTRE_RGB, end, strict=True)
    return "#" + "".join(f"{round(a + (b - a) * mix):02x}" for a, b in pairs)


# --------------------------------------------------------------------------- #
# Normalisation, redaction and embedding
# --------------------------------------------------------------------------- #


def _embed_json(payload: object) -> str:
    """Serialize for a ``<script type="application/json">`` block.

    ``<`` alone would end the block early through ``</script`` or open a comment
    with ``<!--``; escaping ``<``, ``>`` and ``&`` is the standard
    belt-and-braces form. U+2028 and U+2029 are JS line terminators that some
    parsers treat as raw newlines inside a string literal.
    """
    text = json.dumps(payload, allow_nan=False, ensure_ascii=False, sort_keys=True)
    return (
        text.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _finite(value: Any) -> Any:
    """Non-finite numbers become ``null``, at any depth.

    ``JSON.parse`` rejects ``Infinity``/``NaN`` and ``allow_nan=False`` raises on
    a nested one just as readily as on a top-level one, and a goal-less episode
    legitimately measures an infinite distance -- so this recurses through
    ``measures``, ``final_measures``, ``subtask_status`` and ``metadata`` instead
    of checking the surface.
    """
    if isinstance(value, dict):
        return {str(key): _finite(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_finite(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float) and not math.isfinite(value):
        return None
    return value


def _numbers(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    # bool is a subclass of int, so a success flag comes through as 0.0/1.0 --
    # which is exactly how the contract stores it.
    return {
        str(key): float(item)
        for key, item in value.items()
        if isinstance(item, int | float) and math.isfinite(item)
    }


def _text(value: Any) -> str:
    return redact_text(value) if isinstance(value, str) else ""


def _clip(value: Any) -> str:
    text = _text(value)
    if len(text) <= VIS_MAX_CLIPPED_CHARS:
        return text
    return text[:VIS_MAX_CLIPPED_CHARS] + "..."


def _number(value: Any, *, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return default
    return float(value) if math.isfinite(value) else default


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    return None


def _round(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def _pose(step: Any) -> list[float]:
    """``[x, y, yaw]``: the ground-plane pose the top-down view draws.

    The rollout tracks a ground-height logical pose (the camera renders above
    it), so x/y is the floor trajectory directly. Yaw comes back out of the
    yaw-only quaternion the trace records.
    """
    if not isinstance(step, dict):
        return [0.0, 0.0, 0.0]
    position = step.get("position")
    position = position if isinstance(position, list) else []
    x = _number(position[0], default=0.0) if len(position) > 0 else 0.0
    y = _number(position[1], default=0.0) if len(position) > 1 else 0.0
    quaternion = step.get("orientation_wxyz")
    yaw = 0.0
    if isinstance(quaternion, list) and len(quaternion) >= 4:
        w = _number(quaternion[0], default=1.0)
        z = _number(quaternion[3], default=0.0)
        yaw = 2.0 * math.atan2(z, w)
    # Centimetres and milliradians: the top-down view is a few hundred pixels
    # wide, so more precision than this is only page weight.
    return [round(x, 2), round(y, 2), round(yaw, 3)]


def _measure(step: dict[str, Any], name: str) -> float | None:
    measures = step.get("measures")
    if not isinstance(measures, dict):
        return None
    value = measures.get(name)
    if isinstance(value, dict):
        value = value.get("value")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def _waypoint(step: dict[str, Any]) -> list[float] | None:
    """The chosen waypoint as body-frame ``[forward_m, lateral_m]``.

    Its ``yaw_rad`` is not carried: the top-down view draws where the waypoint
    is, and a heading at a point 40 cm away is not legible at this scale. What
    is dropped here is in the trace beside the page.
    """
    action = step.get("action")
    waypoint = action.get("selected_waypoint") if isinstance(action, dict) else None
    if not isinstance(waypoint, dict):
        return None
    return [
        round(_number(waypoint.get("forward_m"), default=0.0), 2),
        round(_number(waypoint.get("lateral_m"), default=0.0), 2),
    ]


def _present(value: Any) -> bool:
    return value is not None and value != ""


def _without(entry: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: value for name, value in entry.items() if name != key}


def _open_in_browser(output: Path) -> bool:
    """Hand the finished page to whatever the user registered for ``file://``.

    Deliberately outside the evaluation path: it runs no evaluation code and
    changes no result. It does start a program this distribution does not ship --
    ``xdg-open`` on Linux, ``open`` on macOS -- which is why it is off unless
    stdout is a terminal, switchable off with ``--no-open``, and can never change
    the exit code.
    """
    try:
        return bool(webbrowser.open(output.resolve().as_uri()))
    except Exception:
        return False
