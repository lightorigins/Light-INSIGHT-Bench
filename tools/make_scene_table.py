"""Regenerate guides/scenes.md: which evaluation scene belongs to which scene class.

The classes are a property of the published episode file, not of a table kept
beside it, so this reads them straight out of the file the benchmark pins:

    python3 tools/make_scene_table.py --episodes <DATA_DIR>/insight_bench/v1/episodes.jsonl

Every scene carries exactly one class; the script fails loudly if that ever
stops being true, because a scene in two classes would make the per-class
success rates on the leaderboard mean two different things at once.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

#: What each class means, in the words of the labelling guide the split was built with.
CLASS_NOTES: dict[str, str] = {
    "Apartment": "single-storey flats: no interior staircase between levels",
    "House": (
        "multi-storey or detached homes; an interior staircase is the deciding cue, "
        "with a yard or porch as support. Attics and basements count here too"
    ),
    "Commercial": (
        "supermarkets, convenience stores, restaurants, malls, entertainment venues, "
        "hotels and guesthouses"
    ),
    "Institution": (
        "schools, offices, hospitals, libraries, churches, exhibition halls, historic buildings"
    ),
    "Outdoor": "everything outdoors: streets, open ground and campuses",
}

#: Row order, matching the taxonomy table in the README.
CLASS_ORDER = ("Apartment", "House", "Commercial", "Institution", "Outdoor")

#: Column order, matching the scene-source table in the README.
DATASET_ORDER = ("habitat_gs", "hm3d", "interiorgs", "mp3d")


def collect(episodes: Path) -> tuple[dict[str, dict[str, list[str]]], collections.Counter[str]]:
    """Scene ids per class per source dataset, and episodes per class."""
    scene_class: dict[tuple[str, str], str] = {}
    episode_counts: collections.Counter[str] = collections.Counter()
    with episodes.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            scene = record["scene"]
            key = (str(scene["dataset"]), str(scene["scene_id"]))
            label = str(record.get("metadata", {}).get("scene_class", ""))
            if not label:
                raise SystemExit(f"episode {record.get('episode_id')!r} carries no scene_class")
            previous = scene_class.setdefault(key, label)
            if previous != label:
                raise SystemExit(
                    f"scene {key[1]} is labelled both {previous!r} and {label!r}; "
                    "a scene has exactly one class"
                )
            episode_counts[label] += 1

    grouped: dict[str, dict[str, list[str]]] = {name: {} for name in CLASS_ORDER}
    for (dataset, scene_id), label in scene_class.items():
        grouped.setdefault(label, {}).setdefault(dataset, []).append(scene_id)
    for by_dataset in grouped.values():
        for ids in by_dataset.values():
            ids.sort()
    return grouped, episode_counts


def render(grouped: dict[str, dict[str, list[str]]], episodes: collections.Counter[str]) -> str:
    total_scenes = sum(len(ids) for d in grouped.values() for ids in d.values())
    lines = [
        "# Evaluation scenes by scene class",
        "",
        f"The evaluation split is **{total_scenes} scenes / {sum(episodes.values()):,} episodes**, "
        "and every scene carries exactly one of the five",
        "scene classes. This file lists which. It is generated from the published episode file by",
        "`tools/make_scene_table.py`, so it cannot drift away from what a run is scored against.",
        "",
        "Scene ids are the official ids of their source dataset. The directory layout each one is "
        "expected in is in the",
        "Data section of the [README](../README.md).",
        "",
        "| Scene class | Scenes | Episodes | " + " | ".join(DATASET_ORDER) + " |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in CLASS_ORDER:
        by_dataset = grouped.get(name, {})
        counts = [str(len(by_dataset.get(dataset, []))) for dataset in DATASET_ORDER]
        scenes = sum(len(ids) for ids in by_dataset.values())
        lines.append(f"| {name} | {scenes} | {episodes[name]} | " + " | ".join(counts) + " |")
    totals = [
        str(sum(len(grouped.get(n, {}).get(dataset, [])) for n in CLASS_ORDER))
        for dataset in DATASET_ORDER
    ]
    lines.append(
        f"| **Total** | **{total_scenes}** | **{sum(episodes.values()):,}** | "
        + " | ".join(f"**{value}**" for value in totals)
        + " |"
    )

    for name in CLASS_ORDER:
        by_dataset = grouped.get(name, {})
        scenes = sum(len(ids) for ids in by_dataset.values())
        note = f"{CLASS_NOTES[name]}. {scenes} scenes, {episodes[name]} episodes."
        lines += ["", f"## {name}", "", note, ""]
        for dataset in DATASET_ORDER:
            ids = by_dataset.get(dataset)
            if not ids:
                continue
            lines.append(f"**{dataset}** ({len(ids)})")
            lines.append("")
            lines.append("<pre>" + "  ".join(ids) + "</pre>")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True, help="the published episodes.jsonl")
    parser.add_argument("--output", type=Path, default=Path("guides/scenes.md"))
    args = parser.parse_args(argv)
    grouped, episode_counts = collect(args.episodes)
    args.output.write_text(render(grouped, episode_counts), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
