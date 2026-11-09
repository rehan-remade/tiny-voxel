---
name: animated-asset
description: Create a fully animated voxel creature or object for TinyVoxel from a text idea - text-to-image, Hunyuan 3D, C++ voxelizer, headless Blender rig/animate, engine bake. Use when asked to add/generate/create a new creature, animal, animated asset, or voxel model for the game (e.g. "add a corgi", "make a wolf creature", "generate a new animal").
---

# Animated asset pipeline

Turns a one-line idea ("a cheerful corgi") into a creature the engine
animates at runtime. Six stages, each with a QA gate. Run them in order;
never skip a gate — a bad early artifact wastes every later stage.

Two gates are ALWAYS manual user picks — present the material, then stop
and wait for the choice:
  1. concept selection (stage 1): the user picks one of several generated
     concept images;
  2. bucket selection (stage 3): the user picks the voxel resolution from
     a three-bucket comparison sheet.
Every other gate is automated; ask the user only when one fails twice in
a row.

## Asset home & naming

Assets live in `art\<Name>\` at the repo root (PascalCase name, e.g.
`Corgi`), tracked via Git LFS — see `.gitattributes`. Canonical file
set — keep names EXACT, other tooling references them:

```
<Name>_concept.png      stage 1: the user-picked concept image
<Name>_mesh.glb         stage 2 output (Hunyuan GLB; .obj for legacy assets)
<Name>_texture.png      stage 2 albedo/PBR atlas (if delivered separately)
<Name>.vox              stage 3: the user-picked bucket
<Name>_rigged.blend     stage 4 output
<Name>_animated.glb     stage 4 export (all clips)
previews\               QA renders: <Name>_vox_{front,side,top}.png,
                        <Name>_bucket_comparison.png, concept sheet,
                        <Name>_<clip>_preview.mp4, <Name>_gait_demo.mp4
archive\                superseded artifacts: unpicked concept candidates,
                        unpicked bucket .vox variants (never delete)
```

Repo-side files (committed): `tools/creatures/<name>_rules.py`,
`assets/creatures/<name>.cvx2`, `assets/creatures/<name>.def`.

## Stage 1 — concept image (text-to-image)

Model: Seedream 5 Pro, text to image (default).
Alternates if the style won't converge: GPT Image 2, Nano Banana 2.

Inputs, outputs, the proven prompt template and timings are in
[references/generation.md](references/generation.md).

The prompt MUST depict the asset **as voxel art** ("large chunky visible
cubes, MagicaVoxel style, roughly N cubes long") — blocky meshes voxelize
far cleaner than organic sculpts. Include: side three-quarter view, stated
cube proportions, flat colors in clean regions, plain solid light-gray
background, whole model in frame. `image_size` 1024x1024, `output_format`
png. Takes ~2 min.

Generate **4 candidates in one call** (`num_images: 4`) and download them
as `<Name>_concept_candidate_{1..4}.png`. Keep each candidate's URL —
stage 2 needs the picked one.

**Auto QA first:** view each candidate. Reject any that are not blocky
axis-aligned cubes (smooth low-poly fails), lack a plain background, crop
the body, fuse the limbs, or grow thin appendages. If fewer than 2
survive, regenerate the failed slots (strengthen the blocky wording or
switch model) before involving the user.

**USER PICK (mandatory):** build a labeled contact sheet of the surviving
candidates —

```
python tools/pipeline/contact_sheet.py --out <previews>\<Name>_concept_sheet.png --columns 2 "<cand1>.png|candidate 1|1" "<cand2>.png|candidate 2|1" ...
```

— send the sheet (and the individual images) to the user and STOP until
they choose. The pick becomes `<Name>_concept.png`; unpicked candidates
go to `archive\`. If the user rejects all, collect what they dislike,
fold it into the prompt, and rerun the round.

## Stage 2 — image to 3D (Hunyuan)

Model: Hunyuan 3D v3.1 Pro, image to 3D. Input:
`{"input_image_url": <url from stage 1>, "generate_type": "Normal",
"enable_pbr": true}` — pass the stage-1 URL directly, no re-upload.
Long-running: submit, then poll every ~15 s (typical
~2.5 min, budget 10). Download `model_urls.glb` as `<Name>_mesh.glb` (and
the texture).

Notes: default `face_count` 500k makes ~45 MB GLBs; that's fine for the
voxelizer, but `face_count: 100000` halves download time with no voxel
quality loss. If the back of the creature comes out wrong (Hunyuan
hallucinates unseen views — e.g. a nose on the rear), generate a back view
image and pass `back_image_url`.

**QA gate:** view the returned `thumbnail`. Pass = same creature, blocky
geometry preserved, no major back-side hallucination that would survive
voxelization.

## Stage 3 — voxelize (repo tool)

Tool: `tools/voxelizer/` (byte-verified against the drububu web voxelizer
this pipeline used manually: 0.97–0.998 occupancy IoU). Build once:

```
cmake --build build --config Release --target voxelizer
```

Base resolution comes from the project scale rule — **1 voxel = 10 cm,
so N = real-world meters x 10** on the creature's LONGEST dimension:

| creature (example)     | real size          | base N |
|------------------------|--------------------|--------|
| corgi                  | 0.9 m long         | 9      |
| boar                   | 1.7 m long         | 17     |
| stag (antlers up)      | 2.2 m tall         | 22     |
| brown bear (all fours) | 3.0 m long         | 30     |
| humanoid               | see rigging traps: players/bipeds are normalized in Blender, not here |

Voxelize **all three resolution buckets** — bucket 1 (10 cm cells,
`--long-axis N`), bucket 1/2 (5 cm, `2N`), bucket 1/4 (2.5 cm, `4N`;
must stay <= 256). Run from the repo root, one line each (no `^`/`` ` ``
continuations — they differ per shell):

```
build\Release\voxelizer.exe "<assets>\<Name>\<Name>_mesh.glb" --long-axis <N|2N|4N> --out "<assets>\<Name>\<Name>_bucket<1|_half|_quarter>.vox" --previews "<assets>\<Name>\previews"
```

**Facing check first (automated):** view one `_vox_side` preview — the
head must be at the image LEFT (model faces -Y) and `_front` must show
the face. Wrong facing: re-run all three with `--yaw 90|180|270`
(Hunyuan's output facing follows the concept image's angle — a side 3/4
view often lands on +/-X). Silhouette mush at every bucket = the concept
was too organic; back to stage 1.

**USER PICK (mandatory):** build the equal-physical-size comparison (the
per-panel upscales 4/2/1 make cell sizes compare truthfully):

```
python tools/pipeline/contact_sheet.py --out <previews>\<Name>_bucket_comparison.png "<previews>\<Name>_bucket1_vox_side.png|bucket 1 (10cm cells)|4" "<previews>\<Name>_bucket_half_vox_side.png|bucket 1/2 (5cm)|2" "<previews>\<Name>_bucket_quarter_vox_side.png|bucket 1/4 (2.5cm)|1"
```

Send it to the user and STOP until they choose. The pick becomes
`<Name>.vox`; the other two variants move to `archive\`. Record the
bucket — stage 4's rig script scale is `S = 0.1 x bucket` and stage 5
depends on it (see below).

ENGINE CAVEAT: wildlife baking currently requires bucket 1 — the loader
rejects `cellScale != 1` (src/creatures.cpp loadDef guard) until the
sub-voxel wildlife unlock lands. If the user picks 1/2 or 1/4, flag that
dependency before stage 5 (the player avatar path already supports it).

Boss/size variants = same GLB re-voxelized at a larger `--long-axis`;
color variants = in-engine palette swaps. `--fill` for props that need
solid interiors (creatures stay hollow — the drububu-parity default).
`--ascii` prints per-height slices, used to author bone rules in stage 4.

## Stage 4 — rig + animate (headless Blender)

Blender 5.2: `C:\Program Files\Blender Foundation\Blender 5.2\blender.exe`
(not on PATH). Copy the proven build script for the closest body plan from
`tools/creatures/` (`stag_build.py` = quadruped; see the README there) into
the scratchpad as `<name>_build.py`, and write `tools/creatures/<name>_rules.py`
with `bone_for_voxel(x,y,z)` region rules derived from `--ascii` slices.

```
blender -b --factory-startup --python <name>_build.py -- <Name>.vox <outdir> render video export
```

Required clip names: `Idle`, `Walk` (the loader rejects assets without
them); optional: `Gallop`, `Graze`, `Alert`. Same names across animals —
the engine state machine keys on them.

READ [references/rigging-traps.md](references/rigging-traps.md) BEFORE
touching the script — it encodes every hard-won lesson (gait direction is
set by stance sweep, ungulate knee fold directions, pose-bone local frames,
Blender 5.2 API changes, why visual gait judges cannot be trusted).

**QA gate:** numeric, not visual — per-frame grounded-foot counts and the
stance-sweep projection check from the traps file, then render preview
MP4s into `previews\` for the user.

## Stage 5 — bake + integrate

One line:

```
blender -b --factory-startup --python tools/creatures/bake_parts.py -- --vox <Name>.vox --glb <Name>_animated.glb --out assets/creatures/<name>.cvx2 --rules tools/creatures/<name>_rules.py
```

Bake with `--scale <0.1 x bucket>` (bucket 1 = `--scale 0.1`; 1/2 =
`0.05`; 1/4 = `0.025`) so the .cvx2 header's cellScale is right.

Copy `assets/creatures/stag.def` -> `<name>.def`, tune speeds/spawn rules.
The engine auto-loads every `*.def` in `assets/creatures/` — zero engine
code. Clip rate-lock: measure each gait's authored ground speed and set
`<clip>_ref_speed` in the .def in engine units (voxels/s = m/s x 10) —
see the traps file (the x33.1 divisor rule is player-only).

**QA gate:** build + run the game, `/spawn <name>`, verify with window
capture (the verify-visuals memory has the input-synthesis recipe).

## Stage 6 — wrap up

- Everything in its canonical place (checklist above), previews rendered.
- Commit repo files (`<name>_rules.py`, `<name>.cvx2`, `<name>.def`) on a
  branch; Conventional-Commits title, e.g. `feat(creatures): add corgi`.
- One-line summary per stage in the final report: model used, dims,
  voxel count, clips, and any gate that needed a retry.
