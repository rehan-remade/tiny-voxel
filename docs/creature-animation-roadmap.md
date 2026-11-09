# Creature animation roadmap: flipbook → rigid parts

The v1 creature system (PR #26) bakes animations as voxel *flipbooks*: every
frame is re-sampled onto a fixed axis-aligned grid. That approximation is why
animated stags shimmer — silhouette voxels sit on the fill/empty knife edge of
the re-sampling and pop frame to frame. The fix, and the foundation for
everything below, is rendering **rigid bone parts with runtime transforms**:
voxels genuinely rotate on screen (like the Blender previews), nothing is
re-sampled, and animation data shrinks from megabytes of frames to kilobytes
of bone curves.

Planned PRs, each independently shippable:

1. **feat(creatures): rigid-part skeletal rendering (.cvx2)** ← this PR
   Per-bone rest volumes + skeleton + baked per-part transform tracks
   (`tools/creatures/bake_parts.py`); CPU pose sampler; part-instanced
   renderer (instance SSBO, full-rotation volumes, shared trace include);
   part-aware cast shadows. Stag migrated; flipbook path retired.
2. **feat(creatures): procedural animation layer**
   Alert heads track the player (look-at), body pitch + hoof placement on
   slopes, runtime clip crossfading (replaces the baked WalkToGallop /
   GallopToWalk clips), per-limb layering, lean into turns.
3. **feat(player): rigged voxel avatar**
   The 9-clip voxel humanoid (idle/walk/sprint/crouch/jump/fall/land/punch)
   replaces the analytic box character in the diorama — same render path,
   no special case.
4. **perf(render): creature motion vectors + LOD**
   Per-part previous-frame transforms feed the motion buffer (TAA ghosting
   on movers gone); distant creatures pose/tick at reduced rate.
5. **feat(creatures): second species + clip retargeting**
   Bear via the shared quadruped rig conventions; clips retarget across
   species instead of being re-authored.
6. **feat(physics): part impulses + ragdoll**
   Joint constraints, hit reactions, ragdoll deaths — the "future rigid
   bodies" common.h's canonical GRAVITY was reserved for.
7. **feat(combat): per-part picking, damage, dismemberment**
   Ray-vs-part interaction (observe details now, weak points later), part
   damage states, shed antlers drop as physical items.

## .cvx2 format (v3)

Little-endian; creature-local axes x lateral / y up / z forward, origin at
the feet centre (the authoring armature origin — hooves rest at y=0). All
part/track coordinates are in CELLS; `cellScale` converts cells to world
voxels at runtime (1.0 for wildlife baked at the 10 cm rule; 0.5625 for the
player's 1.8 m / 32-cell model).

```
char[4]  "CVX2"          u32 version = 3
u32      partCount, clipCount, totalFrames
f32      cellScale                     world voxels per cell
u8[1024] palette                       256 x RGBA (indices are 1-based)
part[partCount]:
    char[16] boneName
    i32      parent                    part index, -1 = root
    f32[3]   gridOrigin                rest-pose creature-local corner of cell (0,0,0)
    f32[3]   pivot                     joint the part rotates about (bone head)
    u32[3]   dims                      cells
    u32      voxWordBase               into the trailing word blob
clip[clipCount]:
    char[16] name;  u32 firstFrame, frameCount;  f32 fps;  u32 flags (bit0 loop)
f32      tracks[totalFrames][partCount][7]      quat xyzw + pos xyz:
         the part's creature-local RIGID transform for that frame, applied to
         rest-space points (p' = R q * p + pos)
u32      voxBlob[]                     per-part voxels, 4/word, x-fastest
```

The tracks store flat per-part creature-local transforms (not hierarchy
locals) so playback is a table lookup; the skeleton (`parent`) rides along
for PR 2's IK/look-at, which will re-derive locals where it overrides.
