# Creatures: modular voxel wildlife

Stags (and any future animal) are **data-driven rigid-part voxel puppets**: a
rigged, animated model is baked offline into per-bone rest volumes + per-frame
transform tracks, and the engine poses + renders instances of them around the
player. Adding an animal touches **zero engine code** — drop two files in
`assets/creatures/` (see `tools/creatures/README.md` for the pipeline).

```
tools/creatures/stag_build.py      .vox -> rigged Blender scene -> 7 clips -> GLB
tools/creatures/bake_parts.py      .vox + GLB -> assets/creatures/stag.cvx2
assets/creatures/stag.def          speeds / behaviour / spawn tuning
src/creatures.h/.cpp               loading, posing, spawning, AI, movement
src/shaders/creature.vert/.frag    render pass (lib/creature_trace.glsl shared)
```

## Asset: .cvx2 rigid parts

Each bone's voxels ship once as a rest-pose volume; each clip (Idle / Walk /
Gallop / Graze / Alert / WalkToGallop / GallopToWalk — the standard set every
animal shares) ships as 24 fps tracks of per-part rigid transforms (quat+pos,
creature-local). At runtime the volumes ROTATE — exactly like the Blender
preview — instead of being re-sampled onto a grid per frame, so animation
cannot flicker, and the whole stag is 165 KB (the retired v1 flipbook was
4.3 MB). Creature-local axes: x lateral, y up, z forward; the origin is the
feet centre. 1 cell = 1 world voxel (10 cm) — world scale is baked in.
Format details: `docs/creature-animation-roadmap.md`.

## Rendering: OBB rasterization into the G-buffer

The world renderer is a fullscreen raymarch with **no depth buffer**, and the
primary game view is the map diorama (`renderMap`, mapPass=1). Creatures hook
in as follows:

- `GBuffer` gained a **depth renderbuffer**; `gbuffer.frag` writes
  `gl_FragDepth = t / DEPTH_FAR` (4096) for every pixel. This is the bridge
  between the raymarched terrain and rasterized dynamic objects. (Sampling
  `posMat` from the creature pass instead would be GL 4.3 framebuffer
  feedback UB — the context has no `glTextureBarrier`.)
- Each frame the app poses the visible creatures (`buildPartInstances`
  samples the clip tracks and emits one cell→world affine + inverse per bone
  part) into the parts SSBO (binding 5); the static volume/palette atlas
  lives in binding 4. `drawCreatures` (called at the end of `fillGBuffer` AND
  after the map-pass G-buffer fill) rasterizes one **oriented box per posed
  part** (instanced, vertex-id cube, front-faces culled so it survives a
  camera inside the box). The fragment shader rebuilds the pixel ray with the
  exact raymarch pinhole model (+TAA jitter), DDAs the part's rest volume via
  the shared `lib/creature_trace.glsl`, and writes the same
  `gPosMat/gNormal/gAlbedo` terrain writes, with `gl_FragDepth` under
  `GL_LESS`. Cost scales with rasterized creature pixels, not screen pixels.
- Because creatures live in the G-buffer, the deferred stack shades them for
  free: sun/moon, traced diorama shadows *received* from trees, fog-of-war
  darkening, atmosphere, TAA.
- Material id 38 (`MAT_CREATURE`) has a **white** base colour; the lighting
  pass multiplies the shaded result by the G-buffer albedo (`gAlbedoTex`,
  unit 10) for mat-38 pixels — the .cvox palette (sRGB->linear, 1-based
  indices) survives the deferred stack exactly, and creatures never touch
  the world material table.
- Creatures **cast real shadows**: `creatureSunVis` in `lighting.frag`
  sphere-rejects per creature (bounds computed at load from the actual posed
  track extents), then marches the posed parts along the sun ray with the
  SAME shared trace the G-buffer pass renders with — shadows exactly match
  the visible pose, self-shadowing included (normal-offset start avoids
  acne). A faint contact blob remains for ambient grounding.

Known tradeoffs: motion vectors are camera-only, so fast movers ghost
slightly under TAA — same tradeoff the analytic player character makes
(roadmap PR 4 fixes this with per-part motion vectors). AO rays still don't
see creatures (sun shadows carry the read). Creatures are culled from the
X-ray layer (surface fauna) and against the diorama clip slab / Q-E slice so
they never float over carved terrain.

## Simulation

`CreatureSystem` (a `GameState` member, ticked before `particles.step`) is
transient like `FireSim` — nothing persists. Herds re-derive deterministically
from the world seed on a **192-voxel spawn lattice**: each cell rolls
`creatureCellSeed(seed, cell)` (the appended `SEED_CREATURE` phase —
runtime-only, no GEN_VERSION bump) against `herd_chance x forestDensityAt`
(the exported `biomeAt().treeChance`). Cells activate when the player comes
within `population_radius` (never closer than 100 vox — no on-screen pop-in)
and members despawn beyond `despawn_radius` or outside the resident window
(out-of-window voxels read as air). `recenterWindow` resets the system.

Per-creature FSM: Graze/Idle/Wander (timers, wander targets) -> Alert (face
player inside `alert_dist`) -> Flee (gallop away inside `flee_dist`; the whole
herd spooks together). Gait clips play at `speed / *_ref_speed` so hooves
don't skate.

**Procedural animation layer** (applied at pose time, on top of the tracks):
any clip switch **runtime-crossfades** over 0.25 s (walk<->gallop carries the
stride phase, so the baked transition clips are no longer used); track frames
slerp for sub-frame smoothness; an **alert look-at** rotates the head (full)
and neck (half) toward the player about their baked joint pivots, ramping in
and out; the whole body **pitches to the terrain slope** under it and leans
into turns; and each `*.lower` leg part **snaps vertically to the actual
voxel ground** under its hoof (clamped ±1.5 vox), so slopes don't sink or
float feet. Movement is player-grade: an **oriented three-box body test**
(rear/centre/front along the facing, spanning ~85% of the rest-pose length)
so the long body can't clip into boulders, axis-separated sub-stepped moves
(<=1 voxel per step) with step-up (<=3), gravity (`GRAVITY` from common.h),
ground-snap, and water-as-blocking so wildlife never wades. A stuck detector
(commanded vs actual travel) rerolls the plan instead of grinding into
concave pockets, and an interpenetration self-rescue pops a creature up out
of solid (or silently despawns it) so none is ever left standing in a rock.

## Console + tuning

`/creatures` census, `/spawn stag [n]` force-spawn, `/find stag` warp to the
nearest living one, `/forest` warp to deep forest (spawning is forest-gated).

`/tune creatures` (or Tab from the other /tune pages) opens the live tuning
page: pick a species, drag sliders for population (herd_chance, group sizes,
population target/radius, forest_min), behaviour (alert/flee distances,
wander radius) and speeds. Spawn-shaping edits re-derive the herds within a
second so the density change is visible immediately; `/tune save` patches the
values back into `assets/creatures/<name>.def` (comments preserved), making
them the new shipped defaults. R resets a row to the .def's loaded value.

## Future hooks (not in v1)

Click/observe picking (mapCursorPick is voxel-only — needs a ray-vs-OBB test),
journal entries for fauna (studio plate + SpeciesDef), positional audio,
entity-entity collision, per-creature motion vectors, map-zoom LOD impostors.
