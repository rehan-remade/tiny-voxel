# Creature pipeline

Voxel animals are authored as MagicaVoxel `.vox` models, rigged + animated in
headless Blender, then baked to a `.cvx2` **rigid-part** asset the engine
animates at runtime (per-bone rest volumes + per-frame transform tracks —
voxels rotate on screen exactly like the Blender preview; nothing is
re-sampled, so animation cannot flicker). One creature = three files:

```
tools/creatures/<name>_rules.py   voxel->bone assignment rules (authoring truth)
assets/creatures/<name>.cvx2      baked rigid parts + skeleton + clip tracks
assets/creatures/<name>.def       gameplay tuning (speeds, behaviour, spawning)
```

## Adding a new animal

1. **Voxelize** a model to `.vox` at world scale — 1 voxel = 10 cm, so a 1.7 m
   long deer body is 17 voxels. Use the in-repo tool (`tools/voxelizer/`,
   `--long-axis` = target voxel count); the whole text→image→3D→vox flow is
   codified in the `animated-asset` skill (`.claude/skills/animated-asset/`).
2. **Rig + animate**: copy `stag_build.py`, adapt per animal:
   - orientation remap at import if the model doesn't face -Y,
   - `bone_for_voxel(x,y,z)` region rules from ASCII slice analysis
     (keep them in a separate `<name>_rules.py` so the baker can reuse them),
   - skeleton layout (`BONES`, `LEG_X/Y`, `KNEE_Z/HIP_Z`) and gait params.
   Keep the SAME clip names — `Idle / Walk` (required; the loader rejects
   assets without them) and `Gallop / Graze / Alert` (optional; the engine
   state machine keys on them). `WalkToGallop / GallopToWalk` are no longer
   used — the engine crossfades between clips at runtime — but exporting
   them is harmless. Export with `export` to get `<Name>_animated.glb`.
3. **Bake** the rigid parts:
   ```
   blender -b --factory-startup --python tools/creatures/bake_parts.py -- \
     --vox Name.vox --glb Name_animated.glb \
     --out assets/creatures/name.cvx2 --rules tools/creatures/name_rules.py
   ```
   The baker prints per-part dims/origins and the clip table; the loader
   (src/creatures.cpp loadCvx2) validates the layout hard at startup.
4. **Define** gameplay: copy `assets/creatures/stag.def`, tune speeds and
   spawn rules (or dial them live in `/tune creatures` and `/tune save`).
   The engine auto-loads every `*.def` in `assets/creatures/`.

`bake_flipbook.py` (the retired v1 flipbook baker) is kept for reference and
offline QA comparisons only — the engine no longer loads `.cvox` files.

## .cvx2 format

See `docs/creature-animation-roadmap.md`. Engine-local axes: x lateral,
y up, z forward (the facing direction); the origin is the feet centre at
ground level, so a creature's world position needs no anchor correction.
