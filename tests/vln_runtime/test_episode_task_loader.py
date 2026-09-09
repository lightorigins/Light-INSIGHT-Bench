"""`load_episodes_for_task` dispatches on the task, and refuses rather than guessing.

The dispatch exists because an episode file is not self-describing: the published objnav
records are `EpisodeSpec`-shaped but carry a *relative* `scene.asset`, so reading one needs the
scene root the task was configured with, while a converter-produced record carries a resolved
`asset_path` and needs nothing. Reading a file therefore needs to know which task asked. Getting
that wrong does not raise -- it produces episodes that look fine and score against the wrong
thing -- so the interesting tests here are the refusals.

What the objnav loader makes of a real published file is tested next to the loader itself, in
`test_objnav_scene_binding.py`; this file is about the dispatch table.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from insight_bench.vln_runtime.episodes.loader import _TASK_EPISODE_LOADERS, load_episodes_for_task
from insight_bench.vln_runtime.suite.registry import available_task_names


def _write_specs(path: Path, count: int = 2) -> Path:
    records = [
        {
            "episode_id": f"ep-{index}",
            "instruction": "go to the thing",
            "scene": {"scene_id": "s", "asset_path": "/scenes/s.usd"},
            "start_pose": {"position": [0.0, 0.0, 0.0], "yaw_rad": 0.0},
            "goals": [{"position": [1.0, 0.0, 0.0], "success_radius_m": 1.0}],
        }
        for index in range(count)
    ]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    return path


def test_every_registered_task_has_a_loader():
    # A suite that the registry knows about but nothing can load is a suite that fails at the
    # first attempt to run it, on a GPU, an hour in. Fail here instead.
    assert sorted(_TASK_EPISODE_LOADERS) == sorted(available_task_names())
    # And not vacuously: two empty tables would satisfy the equality above while nothing at
    # all could be loaded, so name the suite that has to be in both.
    assert "insight_bench" in _TASK_EPISODE_LOADERS


def test_an_unregistered_task_is_named_not_guessed(tmp_path):
    # The file below is perfectly readable. It is still refused, because "readable" is not the
    # question -- which loader is right for it is, and for an unknown task there is no answer.
    class _Unknown:
        name = "not_a_task"

    with pytest.raises(ValueError, match="no episode dataset loader registered"):
        load_episodes_for_task(_Unknown(), _write_specs(tmp_path / "x.jsonl"))


def test_the_loader_is_importable_without_pulling_in_every_suite():
    # The per-suite loaders are imported inside function bodies: importing the episode package
    # must not drag in every suite config, or the import graph cycles the moment a suite grows a
    # dependency on episode loading.
    import subprocess
    import sys

    done = subprocess.run(
        [
            sys.executable,
            "-c",
            "import insight_bench.vln_runtime.episodes.loader as m, sys; "
            "print(any(k.startswith('insight_bench.vln_runtime.suite.') for k in sys.modules))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert done.stdout.strip() == "False", done.stdout
