<h1 align="center">INSIGHT-Bench</h1>

<div align="center">

![Leaderboard](https://img.shields.io/badge/Leaderboard-coming%20soon-lightgrey?style=flat)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-INSIGHT--Bench-yellow.svg)](https://huggingface.co/datasets/LightOriginsHQ/light-insight-bench)
[![arXiv](https://img.shields.io/badge/arXiv-2608.30935-b31b1b.svg)](https://arxiv.org/abs/2608.30935)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg)](pyproject.toml)

</div>

INSIGHT-Bench evaluates language-guided object-goal navigation from monocular RGB observations in Isaac Sim.
It uses five scene classes and five instruction types to break down navigation performance beyond an aggregate score.

This evaluation release for [LightNav-0](https://github.com/lightorigins/LightNav-0) provides the 1,097-episode split,
an Isaac Lab runner, adapters for seven policies, offline run reports, and evidence packing for submissions.

## 🏡 About

1,097 episodes in 210 held-out scenes. Every episode carries two labels, fixed when it is built, and together they turn
one success rate into a diagnosis.

<div align="center">
<img src="assets/insight-taxonomy-scene.svg" alt="The five scene classes: Apartment, House, Commercial, Institution, Outdoor" width="95%"/>
<br><br>
<img src="assets/insight-taxonomy-instruction.svg" alt="The five instruction types: Base, Direction, Relation, Extremum, Ordinal" width="95%"/>
</div>

**Scene class** is functional layout, independent of the source dataset: compact rooms with frequent doors (Apartment),
longer multi-room topologies (House), open floor plans with repeated instances (Commercial), corridors and repeated
workspaces (Institution), open traversable regions with sparse landmarks (Outdoor). **Instruction type** is the mechanism
that resolves the goal: a uniquely nameable target (`Base`), an egocentric bearing (`Direction`), a unique anchor object
(`Relation`), an argmin or argmax such as the nearest or leftmost (`Extremum`), a ranked instance of an ordered set
(`Ordinal`). Rows of the 5x5 breakdown expose sensitivity to layout, columns isolate the language mechanism, and single
cells expose the interaction the aggregate hides.

| Scene type | Scenes | Episodes | Base | Direction | Relation | Extremum | Ordinal |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Apartment | 120 | 239 | 50 | 50 | 50 | 50 | 39 |
| House | 61 | 216 | 50 | 50 | 31 | 50 | 35 |
| Commercial | 10 | 195 | 42 | 40 | 30 | 50 | 33 |
| Institution | 11 | 219 | 50 | 49 | 32 | 50 | 38 |
| Outdoor | 8 | 228 | 45 | 50 | 50 | 50 | 33 |
| **Total** | **210** | **1,097** | **237** | **239** | **193** | **250** | **178** |

Every policy is driven through one shared deployment protocol. Nothing here is per-model.

| | |
| :--- | :--- |
| Observation | forward-facing monocular RGB, 480x270 — no depth, no odometry, no panorama |
| Field of view | 120° horizontal |
| Camera height | 1.0 m |
| Action budget | 300 actions per episode |
| Success radius | 2.0 m indoor, 3.0 m outdoor |
| Success condition | stop inside the radius **and** the target lies inside the 120° field of view of the final frame |
| Metrics | SR, SPL, terminal NE — printed as `success-rate`, `spl` and `distance_to_goal` |

**NE is `metrics.distance_to_goal` in the run result** — the straight line from where the agent stopped to the target.
A trace carries a different, route-following distance under `route_distance_remaining_m`; do not read that one as NE.

## 💡 Installation

A run needs Linux and an NVIDIA RTX GPU: rendering requires RT cores, so the A100 / A800 / A30 / H100 / H800 / H20 / V100
class cannot be used. Budget about **13 GB of VRAM for the simulator** and whatever your model needs beside it — LightNav-0
takes roughly 11 GB, so a 24 GB card is comfortable. You also need a writable cache: Isaac, Hugging Face and vLLM all write
one, and if your root filesystem is small, point `TMPDIR`, `HF_HOME`, `XDG_CACHE_HOME` and `VLLM_CACHE_ROOT` somewhere with
room before you start.

Anything in `<angle brackets>` from here on is a path only you know: replace it, brackets and all.

### 1. Isaac Sim 5.1.0 + Isaac Lab 2.3.2

```bash
python3.11 -m venv $HOME/isaac-venv && source $HOME/isaac-venv/bin/activate
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
git clone https://github.com/isaac-sim/IsaacLab.git --branch v2.3.2
(cd IsaacLab && ./isaaclab.sh --install none)
export ISAACLAB_DIR=$PWD/IsaacLab
export OMNI_KIT_ACCEPT_EULA=YES     # you are accepting NVIDIA's Omniverse licence
```

`OMNI_KIT_ACCEPT_EULA` is not optional, and it is not only an install-time setting: **every run needs it in the
environment**, including runs by someone who already had Isaac Lab and skipped this step. Without it the first
`import isaacsim` stops at an interactive licence prompt and waits forever, with nothing in any log to say why. Add
`export OMNI_KIT_ALLOW_ROOT=1` as well if you run as root, and keep both in your shell profile.

About 110 GB once the extension cache is populated, and an NVIDIA GPU must be present while it runs.
`ISAAC_DIR=/where/there/is/room bash scripts/setup_isaac_env.sh` does all of this into one directory instead, and prints
the `ISAACLAB_DIR` to export.

Any Isaac Lab 2.2 or newer on Isaac Sim 5.1.0 works; 2.3.2 is the current patch of the line Isaac Sim 5.1 shipped with,
and is what the reference runs behind this repository were produced on, with driver 535.216.03. The runtime you actually
used is recorded in every run result, so a leaderboard row always says which one produced it — as the `isaaclab`
package version rather than the release name, because the package is what a running process can ask: 2.3.2 reports
`0.54.2`, 2.2.1 reports `0.46.3`.

### 2. This repository

```bash
git clone https://github.com/lightorigins/light-insight-bench.git && cd light-insight-bench
export EVAL_DIR=$PWD
# From step 1. If you already had Isaac Lab, export these three yourself:
#   export ISAACLAB_DIR=<path/to/IsaacLab>
#   export OMNI_KIT_ACCEPT_EULA=YES     # every run needs it, not just the install
#   export OMNI_KIT_ALLOW_ROOT=1        # only if you run as root
$ISAACLAB_DIR/isaaclab.sh -p -m pip install -e .
$ISAACLAB_DIR/isaaclab.sh -p -m insight_bench --version
```

That prints `insight-bench 0.2.0`. Both spellings are deliberate and neither is interchangeable:
`insight_bench` with an underscore is the Python package, so it is what `import` and `python -m` take;
`insight-bench` with a hyphen is the command and the distribution, and `insight-bench-v1` is the
benchmark's published coordinate, where a hyphen is the only legal separator. One Isaac Lab serves one checkout: the editable install replaces any previous one, so
two clones sharing an `ISAACLAB_DIR` take turns rather than coexist. Every command below uses the `isaaclab.sh -p -m` form, because the console script the
install creates lives in Isaac Lab's own `bin`, not on your `PATH`.

### 3. Data

The episode package is available on Hugging Face (`pip install -U huggingface_hub` provides the `hf` CLI;
use a recent version). It is not gated and requires no token:

```bash
export DATA_DIR=$EVAL_DIR/data
hf download LightOriginsHQ/light-insight-bench --repo-type dataset --local-dir $DATA_DIR
```

It includes the episode file and **10 converted Habitat-GS scenes**, the only ones whose licence permits
redistribution. Obtain and convert the other **200 scenes** under their own licences:

| Source | Scenes | Where to get them | Licence |
| :--- | ---: | :--- | :--- |
| `habitat_gs` | 10 | already converted, in the episode package. Upstream: <https://huggingface.co/datasets/RukawaY/gs_scenes> | Apache-2.0; attribution required |
| `interiorgs` | 75 | <https://huggingface.co/datasets/spatialverse/SAGE-3D_InteriorGS_usdz> | CC BY-NC 4.0, non-commercial |
| `hm3d` | 73 | <https://matterport.com/habitat-matterport-3d-research-dataset> | academic EULA, access request required |
| `mp3d` | 52 | <https://niessner.github.io/Matterport/> | Matterport3D Terms of Use, signed |
| **Total** | **210** | | |

[guides/scenes.md](guides/scenes.md) lists every scene by id, under the scene class it is labelled with and the
source it comes from.

A converted stage has to be Z-up, `metersPerUnit` 1.0, with a `defaultPrim`, its textures beside it and polygons for
the floor and wall queries — and the episode package ships `scripts/verify_scene.py` to check that, plus the thing no
property can stand in for: that a scene's own episode start poses land inside its geometry.
[Obtain and convert the scenes](guides/scene-conversion.md) covers where each source starts, the InteriorGS download
and the coordinate wrapper it needs, and the directory layout `--scene-root` expects.

Then check what you have, before you need a GPU. This verifies the episode file against its pinned digest and names
every missing scene; it starts no simulator, which also means it cannot tell you a scene renders. `isaaclab.sh -p`
prints a banner ahead of the JSON, so strip everything before the first `{` before piping to `jq`.

```bash
$ISAACLAB_DIR/isaaclab.sh -p -m insight_bench check-data \
  --episodes $DATA_DIR/insight_bench/v1/episodes.jsonl --scene-root $DATA_DIR/scenes
```

## 🧪 Evaluation

### How it works

The simulator runs in Isaac Lab's Python; the model runs the provided `policies/insight_policy` server in its own
environment, with its own torch, CUDA, and weights. Install this SDK only on the simulator side and keep model
dependencies out of Isaac Lab. Implement a `Policy` class on the model side; the server handles local HTTP
(`/health`, `/reset`, `/act`, `/finish`).

The `scripts/eval_*.sh` wrappers start the model, wait for `/health`, evaluate, and stop it through `Policy.close()`.
A policy may start a child model service: Embodied-Navigator uses three processes and two ports. Its eval script
reports and stops both services.

`GET /health` must return a non-empty `model_id`, which identifies the run. It also reports `forward_m`, `turn_deg`,
`queue_feed_frames`, and `preprocess` (including any FOV crop); check these before a full run. The runner rechecks
`healthy` between episodes and stops on `healthy: false`, indicating the model is gone. Policies implement `Policy.healthy()` to report this;
without the hook, health defaults to `true`.

**Each `/act` reply is one 0.1 s control step**, capped at 0.25 m of travel or 30° of yaw:

- `Step.waypoints(...)` sends the full plan. The runner executes the first moving row, clips it to the caps, and
  requests a new plan on the next frame, matching trajectory models' upstream per-frame replanning.
- `Step.primitives(...)` queues discrete moves, releasing one per `/act` at the policy's step size, matching upstream
  discrete-action evaluation.
- `Step.stop()` signals arrival. A model that never stops is scored at its position when the 300-action budget expires.

See the [policy reference](policies/README.md) for interface details, and
[run checks](guides/running-a-model.md#sanity-check-a-run-before-you-believe-it) before interpreting a result.

### LightNav-0

Install the model in its own environment, then evaluate:

```bash
git clone https://github.com/lightorigins/LightNav-0.git && cd LightNav-0
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[vllm]"
hf download LightOriginsHQ/LightNav-0 --local-dir checkpoints/LightNav-0

cd $EVAL_DIR
MODEL_PATH=<path/to/LightNav-0>/checkpoints/LightNav-0 \
MODEL_PYTHON=<path/to/LightNav-0>/.venv/bin/python \
bash scripts/eval_lightnav0.sh
```

`MAX_EPISODES=5` in front of that is a smoke run that finishes in minutes and proves the wire end to end; the
script prints SR, the path of `vis.html` and the path of the evidence zip when it is done.

[Running a model, and trusting the result](guides/running-a-model.md) has the rest: the options every script takes,
how to budget a full run from a smoke run, how to resume an interrupted one, and the four checks worth running before
you believe a number.

### Open-source baselines

Six policies, in the order of the paper's table. Each `setup_*_env.sh` builds that model's own virtual environment, clones
its upstream repository at a pinned commit, and prints the exact weight-download command.

**JanusVLN** — dual implicit memory decoupling semantics from spatiality ([arXiv:2509.22548](https://arxiv.org/abs/2509.22548)). Shared forward stream, unmodified.

```bash
bash scripts/setup_janusvln_env.sh              # weights are on ModelScope, not Hugging Face
modelscope download --model misstl/JanusVLN_Extra --local_dir $EVAL_DIR/checkpoints/JanusVLN
MODEL_PATH=$EVAL_DIR/checkpoints/JanusVLN bash scripts/eval_janusvln.sh
```

**NaVid** — a video VLM that plans the next VLN step from RGB alone ([arXiv:2402.15852](https://arxiv.org/abs/2402.15852)). Centre-cropped to its training field of view.

```bash
bash scripts/setup_navid_env.sh
MODEL_PATH=<path/to/NaVid> \
POLICY_OPTS="--opt vit_path=<path/to/NaVid>/eva_vit_g.pth" \
bash scripts/eval_navid.sh
```

Two weight files, not one. `eva_vit_g.pth` is LLaMA-VID's EVA-CLIP ViT-g tower and is **not in the NaVid weight
repository**; the setup script prints where to get it. The same applies to Uni-NaVid below.

**Uni-NaVid** — one video VLA unifying several embodied navigation tasks ([arXiv:2412.06224](https://arxiv.org/abs/2412.06224)). Shared forward stream, unmodified.

```bash
bash scripts/setup_uninavid_env.sh
MODEL_PATH=<path/to/Uni-NaVid> \
POLICY_OPTS="--opt vit_path=<path/to/Uni-NaVid>/eva_vit_g.pth" \
bash scripts/eval_uninavid.sh
```

**TAMP-Nav**, whose code, weights and script here all say **Embodied-Navigator** — published as
*Embodied-Navigator: Point, Think, Memorize, and Align for Efficient Navigation*
([arXiv:2608.17512](https://arxiv.org/abs/2608.17512)). The paper's tables use the first name and everything you type uses
the second; `en7b-grpo` weights belong to this row. Designed for 4-camera 360° observation with depth-assisted
pixel-to-3D execution; here restricted to the shared forward view, which is why it points at a floor pixel and this
adapter turns that into a bearing plus a ground-plane distance.

```bash
bash scripts/setup_embodied_navigator_env.sh
MODEL_PATH=<path/to/Embodied-Navigator-7B-GRPO> bash scripts/eval_embodied_navigator.sh
```

**InternVLA-N1** — a dual-system navigation foundation model with learned latent plans
([project page](https://internrobotics.github.io/internvla-n1.github.io/)). Driven RGB-only, with a constant depth plane —
which does not mean no depth model: read on.

```bash
bash scripts/setup_internvla_n1_env.sh          # prints the two downloads it needs
MODEL_PATH=$EVAL_DIR/checkpoints/InternVLA-N1/dualvln \
POLICY_OPTS="--opt depth_model_path=$EVAL_DIR/checkpoints/InternVLA-N1/depth_anything_v2_metric_hypersim_small" \
bash scripts/eval_internvla_n1.sh
```

Two downloads, not one. Nothing depth-shaped is fed to this model — the plane is constant — but the checkpoint builds
DepthAnything-V2 at load time and borrows its DINOv2-S tower as System-1's RGB encoder, so `depth_model_path` is required
and the model will not load without it. Point it at the **directory**, which must contain a file named exactly
`depth_anything_v2_metric_hypersim_vits.pth`: the Small variant, not Base or Large. The setup script prints both
`hf download` lines.

**StreamVLN** — streaming VLN with SlowFast context modelling ([arXiv:2507.05240](https://arxiv.org/abs/2507.05240)). Centre-cropped to its training field of view.

```bash
bash scripts/setup_streamvln_env.sh
MODEL_PATH=<path/to/StreamVLN> bash scripts/eval_streamvln.sh
```

| Model | Python | torch | Upstream repo | Weights |
| :--- | :--- | :--- | :--- | :--- |
| JanusVLN | 3.9 | 2.5.0 | [MIV-XJTU/JanusVLN](https://github.com/MIV-XJTU/JanusVLN) | [`misstl/JanusVLN_Extra`](https://modelscope.cn/models/misstl/JanusVLN_Extra) (ModelScope) |
| NaVid | 3.8 | 2.0.1 | [jzhzhang/NaVid-VLN-CE](https://github.com/jzhzhang/NaVid-VLN-CE) | [`Jzzhang/NaVid`](https://huggingface.co/Jzzhang/NaVid) |
| Uni-NaVid | 3.10 | 2.0.1 | [jzhzhang/Uni-NaVid](https://github.com/jzhzhang/Uni-NaVid) | [`Jzzhang/Uni-NaVid`](https://huggingface.co/Jzzhang/Uni-NaVid) |
| TAMP-Nav | 3.10 | 2.6.0 | [ZJU-OmniAI/Embodied-Navigator](https://github.com/ZJU-OmniAI/Embodied-Navigator) | [`UnderTides/Embodied-Navigator-7B-GRPO`](https://huggingface.co/UnderTides/Embodied-Navigator-7B-GRPO) |
| InternVLA-N1 | 3.9 | 2.6.0 | [InternRobotics/InternNav](https://github.com/InternRobotics/InternNav) | [`InternRobotics/InternVLA-N1-DualVLN`](https://huggingface.co/InternRobotics/InternVLA-N1-DualVLN) |
| StreamVLN | 3.9 | 2.1.2 | [InternRobotics/StreamVLN](https://github.com/InternRobotics/StreamVLN) | [`mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3`](https://huggingface.co/mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3) |

### Results

Fine-grained success rate: the first five columns group episodes by instruction type, the next five by scene type. **Bold**
and <u>underlined</u> denote best and second best.

<div align="center">
<img src="assets/insight_radar.png" alt="Success rate by instruction type and by scene type, LightNav-0 against six open-source policies" width="95%"/>
</div>

**Success rate by instruction type and by scene type**, on a shared 0-70 scale. LightNav-0 is the thick blue outline; the
six baselines are thin. The numbers are the table below.

| Method | Base | Direction | Relation | Extremum | Ordinal | Apartment | House | Commercial | Institution | Outdoor | Avg. |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| JanusVLN | 32.5 | 28.5 | <u>28.5</u> | 22.0 | <u>25.8</u> | <u>39.7</u> | 35.2 | <u>26.7</u> | <u>18.3</u> | <u>16.7</u> | <u>27.4</u> |
| NaVid | <u>37.6</u> | <u>29.7</u> | 18.7 | <u>22.4</u> | 24.2 | 37.7 | <u>39.8</u> | 24.6 | <u>18.3</u> | 13.6 | 26.9 |
| Uni-NaVid | 32.9 | 29.3 | 21.8 | 16.8 | 19.7 | 39.3 | 28.7 | 23.1 | 15.5 | 14.0 | 24.3 |
| TAMP-Nav | 18.1 | 14.2 | 17.6 | 10.8 | 20.8 | 26.8 | 18.5 | 15.9 | 9.6 | 8.3 | 16.0 |
| InternVLA-N1 | 18.1 | 13.0 | 9.8 | 8.4 | 7.9 | 12.1 | 19.0 | 12.3 | 7.8 | 7.5 | 11.7 |
| StreamVLN | 15.6 | 9.6 | 14.5 | 8.0 | 10.7 | 19.7 | 15.7 | 12.8 | 4.1 | 5.3 | 11.6 |
| **LightNav-0** | **45.1** | **57.7** | **37.8** | **37.2** | **38.2** | **61.1** | **50.5** | **42.1** | **29.2** | **34.2** | **43.7** |

<div align="center">
<img src="assets/insight_bench_matrix.png" alt="Joint scene-instruction capability matrix of LightNav-0, per-cell success rate over evaluated episodes" width="85%"/>
</div>

**Joint scene–instruction capability matrix of LightNav-0 on the INSIGHT-Bench evaluation split.** The caption below is the published figure's own, describing the published measurement. Your own matrix will put its best cell elsewhere. Each cell reports SR and
successful episodes over evaluated episodes. Warm cells fall below the overall SR of 43.66%, cool cells exceed it;
saturation encodes the magnitude of that deviation on the colorbar scale, and thin borders mark row, column and overall
totals. The bottom-right cell is the aggregate result (479/1,097); the best and worst intersections are House–Direction
(74.0%, 37/50) and Institution–Extremum (18.0%, 9/50).

#### What this repository measures for the same model

The table above is the published measurement. The row below is what this repository measured, running the released
LightNav-0 checkpoint over all 1,097 episodes of the pinned episode file on one L20, with the commands in this README:

| | Base | Direction | Relation | Extremum | Ordinal | Apartment | House | Commercial | Institution | Outdoor | Avg. |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| published | 45.1 | 57.7 | 37.8 | 37.2 | 38.2 | 61.1 | 50.5 | 42.1 | 29.2 | 34.2 | 43.7 |
| this repository | 50.2 | 55.6 | 41.7 | 34.4 | 42.1 | 65.7 | 51.9 | 39.0 | 31.7 | 34.6 | 44.9 |

SPL 0.431, NE 3.91 m, mean stop step 57.3, on episodes whose digest matches the pin.

**Expect your own numbers to move.** The policy samples, so no two runs agree episode for episode.

### Evaluate your own model

Use your model's Python ≥ 3.8 environment with `numpy` and `pillow`; do not install this SDK there.
Copy the working forward-walking template (`cp -r policies/template policies/my_model`) and test it before replacing
its behavior. The [policy reference](policies/README.md#the-three-methods) includes a class example and action details.

Set a real `model_id` (placeholder identities are refused), load weights once in `__init__`, and implement:

- `reset(instruction)`: start the episode and clear history.
- `act(rgb)`: receive a `(270, 480, 3)` uint8 RGB frame at 120° FOV and return a `Step`.
- `finish()`: release per-episode state.

Return `Step.primitives(["forward", "left", "right"])` for moves at `forward_m` / `turn_deg`,
`Step.waypoints([(forward_m, lateral_m, yaw_rad), ...])` for explicit motion, or `Step.stop()` for arrival.
Attach model text as `raw_output`; set `center_crop_hfov_deg` in `defaults` for a narrower training FOV.

```bash
# Debug the wire first, with the simulator out of the picture:
PYTHONPATH=policies python -m insight_policy --policy policies/my_model --model-path <path/to/your/weights> &
$ISAACLAB_DIR/isaaclab.sh -p -m insight_bench check-policy --steps 8

POLICY=policies/my_model MODEL_PYTHON=<path/to/your/venv>/bin/python \
MODEL_PATH=<path/to/your/weights> bash scripts/eval_custom.sh
```

`check-policy` prints what the simulator would execute at each step, so a wiring mistake shows up before you book a
card. What it reports, what it does not prove, and how to point it at a baseline are in
[Running a model](guides/running-a-model.md#debug-the-wire-without-the-simulator).

### Visualize a run

A run keeps camera frames by default; `VIS=0` in front of any `eval_*.sh` turns that off. With frames on a full suite
**refuses to start unless about 7.5 GB is free** where `--run-dir` points, and typically writes 1–4 GB of it. Size the
disk on 7.5 GB, not on what it writes.

```bash
$ISAACLAB_DIR/isaaclab.sh -p -m insight_bench vis runs/<run>
```

<div align="center">
<img src="assets/vis-lightnav0.png" alt="The run report page for a full LightNav-0 evaluation: the two taxonomy radars, the scene-class by instruction-type matrix, and the sortable, filterable episode table beside a trajectory replay" width="92%"/>
</div>

<p align="center"><i>A full LightNav-0 report page, opened from disk with no server. This one is a second run, on scene assets
converted from the public release rather than ours, which is why its numbers sit a little below the table above.</i></p>

That writes one offline HTML page, `runs/<run>/vis.html`: the two taxonomy radars and the 5x5 matrix, a per-episode table
you can sort, filter and resize from its own headers, and a step-by-step replay of every episode that left a trace. The
page needs no server, but it is not a single file — it references the camera frames where they sit in the run directory,
which keeps a 1,097-episode page around 6 MB instead of a gigabyte. Move the page and take `<episode-id>/frames/` with it.

## 📤 Submission

Every `scripts/eval_*.sh` already does this and prints the zip, including every episode trace. Only do it by hand if you
want to build a pack from a run directory you kept:

```bash
cd $EVAL_DIR/runs/<run>
$ISAACLAB_DIR/isaaclab.sh -p -m insight_bench pack run-result.json evidence-pack \
  $(for t in */trace.json; do printf -- '--include %s ' "$t"; done)
$ISAACLAB_DIR/isaaclab.sh -p -m insight_bench verify evidence-pack    # must exit 0
python3 -c "import pathlib,zipfile; p=pathlib.Path('evidence-pack'); z=zipfile.ZipFile('evidence-pack.zip','w',zipfile.ZIP_DEFLATED); [z.write(f, f.relative_to(p).as_posix()) for f in sorted(p.rglob('*')) if f.is_file()]"
```

`--include` takes one path and no globs, which is why the loop is there. **Leave it out and you still get a pack that
verifies** — just a much thinner one, carrying the result and nothing that explains it.

`verify` takes either the pack directory or the `.zip`. The paths you pass do not travel with the result: `pack`
rewrites absolute paths to `<redacted-path>` before anything leaves your machine.

The shape of what you are submitting is not something to reverse-engineer: JSON Schemas for the run result, the evidence
manifest and the rest ship in the package, under `src/insight_bench/schemas/v1/`.

**The archive is built from inside the pack directory**, as above. `evidence-manifest.json` has to sit at the archive
**root**; zipping the directory from its parent buries it one level down and the service rejects the bundle with
`manifest_missing`. Python rather than `zip`, because many minimal images do not ship the `zip` binary.

**Getting the pack onto the leaderboard is a pull request**, and everything above is the half of it that happens on
your machine. The other half — the file to add, every refusal CI applies, how each cell is computed, and what the board
does and does not prove — is [guides/submission.md](guides/submission.md). That file is the contract; this section only gets you a pack it will accept.

`verify` exiting `0` here is not a rehearsal for it: CI runs the same check on the same bytes.

## 🔗 Citation

```bibtex
@misc{lightnav0,
  title  = {LightNav-0: Eliciting VLM Spatial Intelligence for Generalist Embodied Navigation},
  author = {Light Origins Team},
  year   = {2026},
  eprint = {2608.30935},
  archivePrefix = {arXiv},
  url    = {https://arxiv.org/abs/2608.30935}
}
```

## 🙋 Questions

Open an [issue](https://github.com/lightorigins/light-insight-bench/issues). A run that behaves oddly is worth the four
checks in [Sanity-check a run](guides/running-a-model.md#sanity-check-a-run-before-you-believe-it) first — they name the failure in the files the
run already wrote, which is most of a useful bug report.

## 📄 License

This repository is released under the [Apache License 2.0](LICENSE). The scene datasets keep their own licences — InteriorGS
CC BY-NC 4.0, HM3D under an academic EULA, MP3D under the Matterport3D Terms of Use — and none are redistributed here.
