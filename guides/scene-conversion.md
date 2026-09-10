# Obtain and convert the scenes

[Back to Data](../README.md#3-data)

The episode package ships ten converted Habitat-GS scenes. The other 200 you obtain and convert to
USD yourself, under their own licences.

## What a stage has to be

Z-up, `metersPerUnit` 1.0, a `defaultPrim`, textures beside the stage, and polygons for the floor
and wall queries. The episode package's
[`scripts/verify_scene.py`](https://huggingface.co/datasets/LightOriginsHQ/light-insight-bench/blob/main/scripts/verify_scene.py)
checks those, and then the one they cannot stand in for: that this scene's own episode start poses
land inside its geometry. Run it on one scene before converting the rest — valid stage metadata
alone does not establish correct placement.

```bash
pip install usd-core      # a few seconds; Isaac Lab's own interpreter cannot import USD without booting Kit
python3 scripts/verify_scene.py scenes/mp3d/<scan>/<scan>.usd --episodes insight_bench/v1/episodes.jsonl
```

## HM3D and MP3D

Convert the meshes to USD with Omniverse's asset converter, Blender, or glTF↔USD tools; this
repository does not provide a conversion recipe. Our stages came from HM3D's `.glb` and the `.obj`
in MP3D's `matterport_mesh/`, not Habitat's MP3D `.glb`.

## InteriorGS

`interiorgs` has two published forms and only the
[SAGE-3D USDZ conversion](https://huggingface.co/datasets/spatialverse/SAGE-3D_InteriorGS_usdz) is
usable here; the other is gated and forbids redistribution. It is 110 GB for 1,000 scenes, the 75
this suite needs are 7.47 GB, and a partial download is enough. No token is needed.

Read the 75 ids out of the episode file rather than guessing them. **A scene id is not its
filename**: upstream the file is the id's second numeric field, so `interior_0270_840784` lives at
`InteriorGS_usdz/840784.usdz`.

```bash
python3 - "$DATA_DIR/insight_bench/v1/episodes.jsonl" > /tmp/interiorgs.txt <<'IDS'
import json, sys
seen = {}
for line in open(sys.argv[1], encoding="utf-8"):
    scene = json.loads(line)["scene"]
    if scene["dataset"] == "interiorgs":
        seen[scene["asset"].split("/")[1]] = None
for scene_id in seen:
    print(scene_id)
IDS
# One --include, every pattern after it. Repeating the flag keeps only the last
# pattern and quietly downloads a single scene.
hf download spatialverse/SAGE-3D_InteriorGS_usdz --repo-type dataset \
  --local-dir <where/there/is/room> \
  --include $(while read -r id; do printf 'InteriorGS_usdz/%s.usdz ' "${id##*_}"; done < /tmp/interiorgs.txt)
```

Give each scene its own directory named by the scene id, and put the downloaded file in it under
that same name, so `840784.usdz` becomes `interiorgs/interior_0270_840784/interior_0270_840784.usdz`.
The numeric field is then only ever a download detail.

### The wrapper these files need

Each upstream USDZ carries `((-1,0,0,0),(0,0,-1,0),(0,-1,0,0),(0,0,0,1))` on its own `Volume` prim,
and that matrix is its own inverse, so a wrapper repeating it cancels it and the payload lands in
the frame the episodes were authored in. Write one wrapper per scene beside the payload, as
`<scene_id>_zup.usda`, complete with the header — without it the stage fails to open at all:

```
#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "World"
{
    def "scene" (
        prepend references = @./<scene_id>.usdz@
    )
    {
        matrix4d xformOp:transform = ( (-1,0,0,0), (0,0,-1,0), (0,-1,0,0), (0,0,0,1) )
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }
}
```

**Do not copy the wrapper from the scenes the package ships.** Theirs carries
`float3 xformOp:rotateXYZ = (90, 0, 0)` and references a `*_final.usdz`. Applying that to an
upstream `.usdz` puts the scene 180 degrees out about Y, which renders without complaint. The
matrix above is the one for upstream files, and it was checked against all 75 of them.

### Check a frame before you trust a run

Get the wrapper wrong and the scene loads, renders, scores, packs and **verifies** without a
warning while showing the model bright fog: a run's poses come from the episode file, so a wrong
transform moves the scene rather than the agent, and every pose-based check still passes.

`verify_scene.py` above refuses such a stage, because the start poses no longer land inside the
geometry — so run it first. The runner also writes `metadata.first_frame_warning` into that
episode's `trace.json` and carries on, which is a tripwire you have to go and read. For the answer,
compare pixels: render one episode's first frame against the same frame from a run you trust. A
correct wrapper agrees to a correlation above 0.97; a wrong one lands near zero.

## Habitat-GS

The ten bundled scenes are `scene56`–`scene65` of the upstream `val` split of
[Habitat-GS](https://zju3dv.github.io/habitat-gs/), each as a USD referencing a `.usdz`. Upstream
publishes a Gaussian-splat `.gs.ply` and a `.navmesh`; Isaac Sim reads neither, which is why these
ten are the ones we converted. No navmesh ships and nothing in this SDK reads one — the floor and
wall queries build their own mesh from visual geometry.

## Scene layout

`--scene-root` is the common root for the scene paths in the episode file. With the package layout,
exporting `DATA_DIR` is enough; otherwise set `EPISODES` and `SCENE_ROOT`. Every `scripts/eval_*.sh`
reads all three. Symlinks are rejected; hard-link (`cp -al`) to combine scenes on one filesystem
without spending the disk twice.

```
$DATA_DIR/scenes/
  habitat_gs/scene56/scene56.usd
  hm3d/00001-UVdNNRcVyV1/UVdNNRcVyV1.usd
  interiorgs/interior_0270_840784/interior_0270_840784_zup.usda
  mp3d/17DRP5sb8fy/17DRP5sb8fy.usd
```

When the layout is in place, run `check-data` ([Data](../README.md#3-data)): it names every missing
scene without starting a simulator.
