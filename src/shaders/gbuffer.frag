#version 430 core
// Deferred G-buffer fill. Mirrors the primary raymarch in voxel.frag's main()
// but, instead of shading, writes per-pixel primary-hit attributes to an MRT
// target. No lighting, crosshair, sky, god-rays, or heat paths — this
// is geometry only. The lighting pass (PR3b) consumes these; PR3a uses them
// for the depth/normal/albedo/material debug views.
in vec2 vUV;

layout(location = 0) out vec4 gPosMat;   // xyz = world hit pos, w = material id (-1 = miss/sky)
layout(location = 1) out vec4 gNormal;   // xyz = world normal,  w = brick-step count (heat)
layout(location = 2) out vec4 gAlbedo;   // rgb = surface albedo, w = 1 hit / 0 miss

uniform vec3  camPos;
uniform vec3  camFwd;
uniform vec3  camRight;
uniform vec3  camUp;
uniform vec2  viewport;
uniform float tanHalfFov;
uniform ivec3 gridSize;
uniform ivec3 bricksPerAxis;
uniform ivec2 windowOriginB;  // toroidal brick-window origin (voxel_trace addressing)
uniform float surfaceY;       // approximate world surface Y, for the empty-space skip
uniform float uTime;          // referenced by materials.glsl (biome/animated surfaces)
uniform sampler2D heightTex;  // per-column terrain surface height (materials.glsl)
uniform vec2  jitterNdc;      // sub-pixel TAA jitter in NDC (zero when TAA is off)
// -- Far-field LOD rings (voxel clipmap C1; docs/lod-rings-architecture.md) ---
// RING_N coarser brickmaps (voxel size doubling per level: 2x/4x/8x/16x, so
// extents 410m/819m/1.6km/3.3km) living in TAIL regions of the SAME SSBOs
// (offset uniforms — no arrays-of-SSBO-blocks). The primary ray marches the
// fine window out to DS[0], then CONTINUES level by level, so distant terrain
// and tree impostors stay real 3D voxels out to ~1.2 km. ringOn=0 disables
// (/rings off); ringReadyMask gates individual levels.
const int RING_N = 4;
uniform int   ringOn;
uniform int   ringReadyMask;             // bit r = ring r resident on the GPU
uniform ivec2 ringOriginBs[RING_N];      // per-ring brick-window origins
uniform int   ringBricksYs[RING_N];      // per-ring Y brick counts (XZ fixed 256)
uniform int   ringBmBases[RING_N];       // brickmap tail offsets
uniform int   ringCoarseBases[RING_N];   // coarse-mip tail offsets
// Per-chunk-slot GPU residency (CHUNKS_XZ^2, mirrors World::gpuChunkValid).
// The fine march consults it PER RAY: entering a blank chunk hands off to
// the ring cascade at that exact t — a DIRECTIONAL reveal. (A global scalar
// cap was tried first: one missing chunk anywhere coarsened the whole view
// and started ring marches inside solid terrain in unrelated directions —
// the "floating slab" glitches.)
uniform usampler2D chunkValidTex;
// -- Explored-world map (mapPass == 1; see map.h) ------------------------------
// The M-key diorama re-runs this shader with a clip box (clean cut faces) and
// fog-of-war gating: untouched chunk columns are holes, voxels the player
// hasn't discovered recolor to MAP_UNKNOWN, and unvisited underground reads as
// SOLID mass (Terraria: tunnels you never walked stay filled). All of it is
// uniform-gated so the main view's march is untouched.
uniform int   mapPass;               // 0 off, 1 surface diorama, 2 cave X-ray
uniform float mapFineVpp;            // voxels-per-pixel at the orbit target:
                                     // the fine->ring0 handoff probability
                                     // ramps over 1.35..1.55 (0 = fine always)
uniform vec3  mapClipLo;             // diorama clip box (world voxels)
uniform vec3  mapClipHi;
// Explored masks: a MAP_MASK_CH^2-chunk window into the GLOBAL exploration
// record, centred on the map's pan target (sizes must match map.h
// MapMasks::MASK_CH). Indexed by world chunk coords minus mapMaskOrigin;
// anything outside the mask reads as undiscovered.
uniform ivec2 mapMaskOrigin;
uniform usampler2D  mapExploredTex;  // R8UI:  bit0 surface seen, bit1 any
uniform usampler3D  mapExploredYTex; // R32UI: visited-Y-slice bit words
const int MAP_MASK_CH = 384;

bool maskIn(ivec2 m) {
    return m.x >= 0 && m.y >= 0 && m.x < MAP_MASK_CH && m.y < MAP_MASK_CH;
}

#include "lib/noise.glsl"
#include "lib/voxel_trace.glsl"
#include "lib/materials.glsl"

// Render-only fog-of-war material (must match MAT_MAP_UNKNOWN in common.h).
const uint  MAP_UNKNOWN = 37u;
// Surface skin: within this many voxels of the local terrain height a voxel
// counts as "surface" (gated by the chunk's surface-seen bit, like the trees
// and flora above it); deeper voxels gate per visited Y slice.
const float MAP_SKIN = 12.0;
// Diorama crust: the surface view keeps only this many voxels below the local
// top — the terrain, or the slice cut when it's lower — so the map is a
// floating shell, not a full-depth block. Deeper discoveries stay reachable:
// the crust follows the slice plane down, and the X-ray keeps full depth.
const float MAP_CRUST = 28.0;

// Per-voxel fog-of-war gate. Surface content always renders — the lighting
// pass darkens unexplored regions to near-black with a soft frontier gradient
// (mapExploredAt), which reads better than holes. Below the surface skin,
// voxels in unvisited Y slices still become MAP_UNKNOWN mass — INCLUDING air,
// so undiscovered tunnels stay filled and the slice plane only carves the
// ones the player walked.
uint mapGateVoxel(uint v, ivec3 voxel, ivec2 mcc) {
    float sh = texture(heightTex, (vec2(voxel.xz) + 0.5) /
                       vec2(float(gridSize.x), float(gridSize.z))).r;
    if (float(voxel.y) >= sh - MAP_SKIN) return v;
    // Below the crust everything — rock, putty, water — is carved away.
    if (float(voxel.y) < min(sh, mapClipHi.y) - MAP_CRUST) return 0u;
    if (!maskIn(mcc)) return MAP_UNKNOWN;       // beyond the mask: undiscovered
    int slice = voxel.y >> 4;
    uint w = texelFetch(mapExploredYTex, ivec3(mcc.x, slice >> 5, mcc.y), 0).r;
    return (((w >> uint(slice & 31)) & 1u) != 0u) ? v : MAP_UNKNOWN;
}

// Cave X-ray gate (mapPass == 2): the SOLID is discovered underground SPACE.
// Air (and water) below the surface skin in a visited Y slice becomes a hit —
// the glowing vein; rock and everything undiscovered is pass-through, so the
// march sails through the world and stops exactly on the tunnels the player
// walked. Returns 0 (invisible), 6 (cave water) or MAP_UNKNOWN (cave air).
uint mapCaveVoxel(uint v, ivec3 voxel, ivec2 mcc) {
    if (v != 0u && v != 6u) return 0u;                    // rock: invisible
    float sh = texture(heightTex, (vec2(voxel.xz) + 0.5) /
                       vec2(float(gridSize.x), float(gridSize.z))).r;
    if (float(voxel.y) >= sh - MAP_SKIN) return 0u;       // open air above ground
    if (!maskIn(mcc)) return 0u;                          // beyond the mask
    int slice = voxel.y >> 4;
    uint w = texelFetch(mapExploredYTex, ivec3(mcc.x, slice >> 5, mcc.y), 0).r;
    if (((w >> uint(slice & 31)) & 1u) == 0u) return 0u;  // undiscovered
    return (v == 6u) ? 6u : MAP_UNKNOWN;
}

// Primary-ray switch distances (world vox): fine -> ring0 at DS[0], ring r ->
// r+1 at DS[r+1]; the horizon heightfield owns everything past DS[RING_N].
// Mirrors common.h RING_SWITCH_T.
const float DS[5] = float[5](768.0, 1536.0, 3072.0, 6144.0, 12288.0);

// Hardware depth for the creature OBB pass: hit distance mapped linearly to
// t/DEPTH_FAR (clamped). Terrain beyond DEPTH_FAR saturates at 1.0 — fine,
// creatures are CPU-culled far inside that range. creature.frag mirrors this.
const float DEPTH_FAR = 4096.0;

// World-anchored stochastic switch radius: each ray hands off within an
// overlap BAND around the nominal boundary instead of on an exact circle, so
// the resolution change (and the fine-vs-impostor tree differences) become a
// dithered mix that TAA resolves into a gradual fade. Anchored to the world
// position where the ray crosses the nominal boundary — stable under camera
// translation, so the pattern doesn't crawl as you move (a screen-space
// dither would shimmer against the TAA history).
float switchJitter(vec3 ro, vec3 rd, float dNom, float band) {
    vec3 p = (ro + rd * dNom) * 0.25;      // 4-voxel jitter cells
    uvec3 q = uvec3(ivec3(floor(p)));
    uint h = q.x * 374761393u ^ q.y * 668265263u ^ q.z * 2246822519u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return dNom + (float(h & 1023u) * (1.0 / 1023.0) - 0.5) * band;
}

// Ring brick data lives in its OWN pool SSBO — the fine brickpool sits just
// under the 2^32-byte wall where the driver corrupts sub-range uploads, so
// rings must never extend it. Ring bm entries hold ring-pool GLOBAL indices.
layout(std430, binding = 3) readonly buffer RingPool { uint rbp[]; };
uint ringSampleVoxel(ivec3 v, uint entry) {
    uint bi = entry - 1u;
    ivec3 l = v & (BRICK - 1);
    int linear = l.x + BRICK * (l.y + BRICK * l.z);
    uint word = rbp[bi * 128u + uint(linear >> 2)];
    return (word >> ((uint(linear) & 3u) * 8u)) & 0xFFu;
}

// Ring analogs of bmSlot/brickInWindow (same chunk-contiguous layout at each
// ring's dimensions; must mirror World::ringSlotOfBrick).
int ringSlot(int r, ivec3 brick) {
    int bx = brick.x & 255;
    int bz = brick.z & 255;
    int chunkSlot = (bx >> 3) + 32 * (bz >> 3);
    int local = (bx & 7) + ((bz & 7) << 3) + (brick.y << 6);
    return ringBmBases[r] + chunkSlot * (64 * ringBricksYs[r]) + local;
}
bool brickInRing(int r, ivec3 b) {
    return b.x >= ringOriginBs[r].x && b.x < ringOriginBs[r].x + 256
        && b.z >= ringOriginBs[r].y && b.z < ringOriginBs[r].y + 256
        && b.y >= 0 && b.y < ringBricksYs[r];
}

// Ring march statuses — the cascade must know WHY a march ended, not just
// that it did (conflating them painted coarse content in the wrong bands):
//   TR_HIT    — a real crossed face; outputs valid.
//   TR_CLEAR  — proven empty all the way to maxTW; the band is done.
//   TR_LEFT   — ray left this ring's coverage at tFrontierW; a coarser ring
//               must take over FROM THERE (not from the band start).
//   TR_BUDGET — iteration budget ran out at tFrontierW (treat like TR_LEFT).
const int TR_HIT = 0, TR_CLEAR = 1, TR_LEFT = 2, TR_BUDGET = 3;

// March LOD ring r in RING space (all coords / scale; rd unchanged, so t
// scales linearly). Same DDA + coarse-skip shape as traceRay. Outputs are in
// WORLD units; marching starts at tStartW (finer data owns the inside) and
// content beyond maxTW belongs to the next, coarser ring.
// INVARIANT (the "fabricated slab" fix): a hit is returned ONLY for a face
// the ray actually crossed. If the march STARTS inside solid ring content
// (possible when finer coverage ended early and the ray sank below this
// ring's surface unnoticed), that solid is skipped until the ray has seen
// air once — the camera-facing "surface" at the segment start is not real
// geometry and must never be shaded as one.
int traceRing(int r, vec3 roW, vec3 rd, float tStartW, float maxTW,
              out float tFrontierW, out vec3 hp, out vec3 nrm,
              out uint outMat, out float tOutW, out int stepsOut) {
    float RING_SCALE = float(2 << r);
    float maxT = maxTW / RING_SCALE;
    tFrontierW = tStartW;
    vec3 ro = roW / RING_SCALE;
    vec3 invD = 1.0 / rd;
    vec3 rLo = vec3(float(ringOriginBs[r].x * 8), 0.0, float(ringOriginBs[r].y * 8));
    vec3 rSz = vec3(2048.0, float(ringBricksYs[r] * 8), 2048.0);
    vec3 t0v = (rLo - ro) * invD;
    vec3 t1v = (rLo + rSz - ro) * invD;
    vec3 ts = min(t0v, t1v);
    vec3 tb = max(t0v, t1v);
    float tEnter = max(max(ts.x, ts.y), ts.z);
    float tExit  = min(min(tb.x, tb.y), tb.z);
    if (tExit < 0.0 || tEnter > tExit) return TR_LEFT;   // no coverage on this ray
    // Coverage must include the band START: if this ring's box begins past
    // tStartW (stale AABB mid-recentre), the gap [tStart, tEnter) is
    // unproven — jumping it and returning a farther hit would see through
    // whatever the coarser ring (which does cover the gap) would have drawn.
    if (tEnter * RING_SCALE > tStartW + RING_SCALE) return TR_LEFT;
    float t = max(max(tEnter, 0.0), tStartW / RING_SCALE) + 1e-4;
    if (t > maxT)  return TR_LEFT;                        // degenerate band: no advance
    if (t > tExit) { tFrontierW = tExit * RING_SCALE; return TR_LEFT; }
    bool seenAir = false;   // set once the ray samples any empty space

    ivec3 stp = ivec3((rd.x > 0.0) ? 1 : -1,
                      (rd.y > 0.0) ? 1 : -1,
                      (rd.z > 0.0) ? 1 : -1);
    ivec3 brickLo = ivec3(ringOriginBs[r].x, 0, ringOriginBs[r].y);
    ivec3 bricksPA = ivec3(256, ringBricksYs[r], 256);
    ivec3 brick = clamp(ivec3(floor((ro + rd * t) / 8.0)),
                        brickLo, brickLo + bricksPA - 1);
    vec3 tDeltaB = abs(invD) * 8.0;
    vec3 tMaxB;
    for (int i = 0; i < 3; ++i) {
        float boundary = float(brick[i]) + ((stp[i] > 0) ? 1.0 : 0.0);
        tMaxB[i] = (boundary * 8.0 - ro[i]) * invD[i];
    }
    int outerHitAxis = -1;
    ivec3 cdim = bricksPA >> 2;
    float tCur = t;   // entry t of the brick under consideration (ring units)
    for (int o = 0; o < 512; ++o) {
        tCur = (outerHitAxis < 0) ? t :
            ((outerHitAxis == 0) ? (tMaxB.x - tDeltaB.x)
            : (outerHitAxis == 1) ? (tMaxB.y - tDeltaB.y)
                                  : (tMaxB.z - tDeltaB.z));
        tCur = max(tCur, 0.0);
        if (!brickInRing(r, brick)) {
            tFrontierW = tCur * RING_SCALE;
            return TR_LEFT;
        }
        if (tCur > maxT) return TR_CLEAR;   // empty all the way through the band

        // Coarse skip against the ring's own occupancy mip (tail region).
        ivec3 cellW = brick >> 2;
        ivec3 cellL = ivec3(cellW.x & (cdim.x - 1), cellW.y,
                            cellW.z & (cdim.z - 1));
        if (coarseOcc[ringCoarseBases[r] +
                      cellL.x + cdim.x * (cellL.y + cdim.y * cellL.z)] == 0u) {
            seenAir = true;                 // an empty coarse cell IS air
            ivec3 cmin = cellW << 2;
            vec3 tFar;
            for (int i = 0; i < 3; ++i) {
                int bb = (stp[i] > 0) ? (cmin[i] + 4) : cmin[i];
                tFar[i] = (float(bb * 8) - ro[i]) * invD[i];
            }
            float te = tFar.x; int ea = 0;
            if (tFar.y < te) { te = tFar.y; ea = 1; }
            if (tFar.z < te) { te = tFar.z; ea = 2; }
            if (te > maxT) return TR_CLEAR;   // rest of the band is empty
            vec3 pN = ro + rd * (te + 1e-3);
            ivec3 nb = ivec3(floor(pN / 8.0));
            if (nb != brick) {          // forward progress (see fine march)
                brick = nb;
                for (int i = 0; i < 3; ++i) {
                    float bnd = float(brick[i]) + ((stp[i] > 0) ? 1.0 : 0.0);
                    tMaxB[i] = (bnd * 8.0 - ro[i]) * invD[i];
                }
                outerHitAxis = ea;
                continue;
            }
        }

        float tBrickEntry = tCur;

        uint entry = bm[ringSlot(r, brick)];
        if (entry == 0u) {
            seenAir = true;                 // empty brick: real air crossed
        } else {
            int   hitAxis = outerHitAxis;
            float tHit    = tBrickEntry;
            if ((entry & BM_SOLID) != 0u) {
                // Only a CROSSED face counts. seenAir==false means the march
                // began inside this solid — skip through it (no fabricated
                // camera-facing slab at the segment start).
                if (seenAir && hitAxis >= 0) {
                    outMat = entry & 0xFFu;
                    hp     = (ro + rd * tHit) * RING_SCALE;
                    tOutW  = tHit * RING_SCALE;
                    nrm = (hitAxis == 0) ? vec3(-float(stp.x), 0, 0)
                        : (hitAxis == 1) ? vec3(0, -float(stp.y), 0)
                                         : vec3(0, 0, -float(stp.z));
                    stepsOut = o;
                    return TR_HIT;
                }
            } else {
                vec3 pIn = ro + rd * (tBrickEntry + 1e-4);
                ivec3 voxel = ivec3(floor(pIn));
                voxel = clamp(voxel, brick * 8, brick * 8 + 7);
                vec3 tMaxV;
                for (int i = 0; i < 3; ++i) {
                    float bv = float(voxel[i]) + ((stp[i] > 0) ? 1.0 : 0.0);
                    tMaxV[i] = (bv - ro[i]) * invD[i];
                }
                vec3 tDeltaV = abs(invD);
                for (int s = 0; s < 24; ++s) {
                    if ((voxel >> 3) != brick) break;
                    uint v = ringSampleVoxel(voxel, entry);
                    if (v == 0u) {
                        seenAir = true;     // empty voxel inside a mixed brick
                    } else if (seenAir && hitAxis >= 0) {
                        tHit = (hitAxis == 0) ? (tMaxV.x - tDeltaV.x)
                             : (hitAxis == 1) ? (tMaxV.y - tDeltaV.y)
                                              : (tMaxV.z - tDeltaV.z);
                        outMat = v;
                        hp     = (ro + rd * tHit) * RING_SCALE;
                        tOutW  = tHit * RING_SCALE;
                        nrm = (hitAxis == 0) ? vec3(-float(stp.x), 0, 0)
                            : (hitAxis == 1) ? vec3(0, -float(stp.y), 0)
                                             : vec3(0, 0, -float(stp.z));
                        stepsOut = o;
                        return TR_HIT;
                    }
                    if (tMaxV.x < tMaxV.y && tMaxV.x < tMaxV.z) {
                        voxel.x += stp.x; tMaxV.x += tDeltaV.x; hitAxis = 0;
                    } else if (tMaxV.y < tMaxV.z) {
                        voxel.y += stp.y; tMaxV.y += tDeltaV.y; hitAxis = 1;
                    } else {
                        voxel.z += stp.z; tMaxV.z += tDeltaV.z; hitAxis = 2;
                    }
                }
            }
        }
        if (tMaxB.x < tMaxB.y && tMaxB.x < tMaxB.z) {
            brick.x += stp.x; tMaxB.x += tDeltaB.x; outerHitAxis = 0;
        } else if (tMaxB.y < tMaxB.z) {
            brick.y += stp.y; tMaxB.y += tDeltaB.y; outerHitAxis = 1;
        } else {
            brick.z += stp.z; tMaxB.z += tDeltaB.z; outerHitAxis = 2;
        }
    }
    tFrontierW = tCur * RING_SCALE;
    return TR_BUDGET;
}

void main() {
    // Default = miss/sky sentinel; overwritten on a hit below. Depth defaults
    // to the far plane (statically-written gl_FragDepth must cover ALL paths).
    gPosMat = vec4(0.0, 0.0, 0.0, -1.0);
    gNormal = vec4(0.0);
    gAlbedo = vec4(0.0);
    gl_FragDepth = 1.0;

    vec2 ndc = vUV * 2.0 - 1.0 + jitterNdc;   // sub-pixel jitter for TAA
    float aspect = viewport.x / viewport.y;
    vec3 rd = normalize(camFwd
        + camRight * ndc.x * aspect * tanHalfFov
        + camUp    * ndc.y * tanHalfFov);
    vec3 ro = camPos;

    vec3 invD = 1.0 / rd;
    vec3 winLo = windowLoV();                          // toroidal window world-origin
    vec3 boxLo = winLo, boxHi = winLo + vec3(gridSize);
    if (mapPass != 0) {
        // Diorama: the march box is the map slab ITSELF — it may extend far
        // beyond the resident window (the ring cascade owns the outside; the
        // fine march below only runs where resident). The app keeps the
        // slab's Y span inside [0, GRID_Y].
        boxLo = mapClipLo;
        boxHi = mapClipHi;
    }
    vec3 t0 = (boxLo - ro) * invD;
    vec3 t1 = (boxHi - ro) * invD;
    vec3 ts = min(t0, t1);
    vec3 tb = max(t0, t1);
    float tEnter = max(max(ts.x, ts.y), ts.z);
    float tExit  = min(min(tb.x,  tb.y),  tb.z);
    if (tExit < 0.0 || tEnter > tExit) return;        // misses the grid -> sky

    // Map cut faces: remember which slab face the ray entered through so a
    // march that starts inside solid (the clip plane cutting terrain) shades
    // with the clip-face normal instead of the -rd fallback.
    int enterAxis = -1;
    if (mapPass != 0 && tEnter > 0.0)
        enterAxis = (tEnter == ts.x) ? 0 : (tEnter == ts.y) ? 1 : 2;

    // Empty-space skip above the world (matches the forward path; lossless).
    // +500 clears the tallest highland relief + tree canopies (see voxel.frag).
    float worldTop = surfaceY + 500.0;
    if (ro.y > worldTop) {
        if (rd.y > -1e-4) return;                     // rising / level -> sky
        float tw = (worldTop - ro.y) / rd.y;
        if (tw > tEnter) { tEnter = tw; enterAxis = (mapPass != 0) ? 1 : -1; }
        if (tEnter > tExit) return;                   // grid exits before worldTop
    }

    float t = max(tEnter, 0.0) + 1e-4;
    ivec3 stp = ivec3((rd.x > 0.0) ? 1 : -1,
                      (rd.y > 0.0) ? 1 : -1,
                      (rd.z > 0.0) ? 1 : -1);

    ivec3 brickLo = ivec3(windowOriginB.x, 0, windowOriginB.y);
    vec3 pBrick = (ro + rd * t) / float(BRICK);
    ivec3 brick = clamp(ivec3(floor(pBrick)), brickLo, brickLo + bricksPerAxis - 1);
    vec3 tDeltaB = abs(invD) * float(BRICK);
    vec3 tMaxB;
    for (int i = 0; i < 3; ++i) {
        float boundary = float(brick[i]) + ((stp[i] > 0) ? 1.0 : 0.0);
        tMaxB[i] = (boundary * float(BRICK) - ro[i]) * invD[i];
    }
    int outerHitAxis = enterAxis;   // -1 outside mapPass (or camera inside the box)
    float tBrickEntry = t;
    // Map fog-of-war state for the brick under consideration (mapPass only).
    ivec2 mcc = ivec2(0);

    // Ring handoff gate. D0 sits inside the window's scroll margin, so the
    // primary ray never reaches the streaming frontier (the translucent
    // reveal band this replaces). Disabled when the camera is buried (deep
    // caves / mega-caverns: the ring is pure surface terrain and would slice
    // a solid wall through any open cavern wider than D0).
    float camSurfH = texture(heightTex, camPos.xz /
                             vec2(float(gridSize.x), float(gridSize.z))).r;
    // The map camera is above the diorama by construction — and its XZ can
    // sit outside the window, where the toroidal camSurfH sample is garbage.
    bool ringLive = (ringOn == 1) &&
                    (mapPass != 0 || camPos.y > camSurfH - 8.0);
    // The handoff cap is a HORIZONTAL (XZ) distance converted to along-ray t:
    // data residency is an XZ window, so a vertical ray never leaves the
    // resident column and stays fine at any altitude — with radial caps,
    // flying up past D0 popped the ground below the camera to the ring, and
    // the D0 sphere was visible slicing the terrain from the air.
    float rdXZ = max(length(rd.xz), 1e-4);
    // Fine handoff dithered over a +-16-voxel overlap band (world-anchored).
    // The map keeps fine detail across the WHOLE resident window (no
    // distance cap — its rays hand off to the rings at the window edge).
    float fineCap = (ringLive && mapPass == 0)
        ? switchJitter(camPos, rd, DS[0] / rdXZ, 32.0 / rdXZ)
        : 1e9;
    // Where the rings take over for THIS ray: the cap, or earlier if the ray
    // enters a chunk that is blank on the GPU (see chunkValidTex above) —
    // rays tunnelling through blank chunks would hit the cut-open interiors
    // of the chunks behind them (bare stone far below heightTex, crushed
    // black by the burial term — the artifact the old reveal curtain hid).
    float handoffT = fineCap;
    // WHY the fine march ended (diagnostic channel, ridden out in gNormal.w
    // as reason*2048: 0=cap, 1=blank chunk, 2=window exit, 3=budget).
    int hReason = 3;

    const int MAX_OUTER = 1024;
    ivec3 cdim = bricksPerAxis >> COARSE_BITS;
    float tCurF = max(t, 0.0);
    // Far map: a ray that enters the map slab OUTSIDE the resident window
    // belongs to the rings from its very first step — the fine DDA below
    // would clamp the entry brick into the window and march from the wrong
    // place. (Rays entering inside the window still hand off at the window
    // edge as usual; with the loop skipped, handoffT resolves to the slab
    // entry below.)
    int maxOuter = MAX_OUTER;
    if (mapPass != 0) {
        vec3 pw = ro + rd * (max(tEnter, 0.0) + 1e-3);
        if (pw.x < winLo.x || pw.x >= winLo.x + float(gridSize.x) ||
            pw.z < winLo.z || pw.z >= winLo.z + float(gridSize.z))
            maxOuter = 0;
        // Zoom-driven LOD, dithered: as fine voxels sink below ~1.5 px the
        // rays hand the frame to ring0 STOCHASTICALLY — a world-anchored
        // per-ray coin whose bias ramps with mapFineVpp across the
        // 1.35..1.55 band, so TAA resolves the transition as a gradual
        // cross-fade.
        // A binary flip was a one-frame full-screen pop — and the settle
        // dolly GUARANTEED it mid-open for a band of zoom levels. The coin
        // is anchored at the slab-entry world position (stable under camera
        // motion; a screen-space dither would crawl against the TAA
        // history). X-ray (mapPass==2) never skips: it needs real voxels.
        float pSkip = smoothstep(1.35, 1.55, mapFineVpp);
        if (mapPass == 1 && pSkip > 0.0) {
            uvec3 q = uvec3(ivec3(floor(pw * 0.25)));   // 4-voxel cells
            uint h = q.x * 374761393u ^ q.y * 668265263u ^ q.z * 2246822519u;
            h = (h ^ (h >> 13)) * 1274126177u;
            if (float(h & 1023u) * (1.0 / 1023.0) < pSkip) maxOuter = 0;
        }
    }
    for (int o = 0; o < maxOuter; ++o) {
        // Entry t of the brick under consideration — the running frontier of
        // proven fine coverage. EVERY exit from this loop must hand the
        // rings a truthful frontier (window exit and budget exhaustion
        // included): the slabs/see-through came from exits that silently
        // kept the cap value while fine coverage had ended much earlier.
        tCurF = max((outerHitAxis < 0) ? t :
            ((outerHitAxis == 0) ? (tMaxB.x - tDeltaB.x)
            : (outerHitAxis == 1) ? (tMaxB.y - tDeltaB.y)
                                  : (tMaxB.z - tDeltaB.z)), 0.0);

        if (mapPass != 0) {
            // Entry brick: never march before the clip-plane entry. The slice
            // plane cuts mid-brick (sliceY is an arbitrary float), and the
            // brick-face frontier would otherwise start up to 7 voxels above
            // it — snapping the cut face to brick rows and hitting content
            // outside the box. No-op for every later brick (crossed faces all
            // lie past t) and for the main view.
            tCurF = max(tCurF, t);
            if (tCurF > tExit - 1e-4) break;   // left the diorama box
        }

        if (!brickInWindow(brick)) {          // left the resident window
            handoffT = min(handoffT, tCurF);
            hReason = 2;
            break;
        }

        // Map fog-of-war, per brick — BEFORE the coarse-empty skip, so an
        // unexplored cavern's empty cells stop the ray as unknown mass instead
        // of fast-forwarding through and leaking the cave's shape.
        bool mapHole = false;
        if (mapPass != 0) {
            // Two chunk keys: GPU residency wraps toroidally with the window;
            // the explored mask is a global-coord window around the target.
            ivec2 wcc = ivec2((brick.x >> 3) & 31, (brick.z >> 3) & 31);
            mcc = ivec2(brick.x >> 3, brick.z >> 3) - mapMaskOrigin;
            bool gpuValid = texelFetch(chunkValidTex, wcc, 0).r != 0u;
            // X-ray only: untouched chunk columns can hold no veins, and a
            // chunk whose voxels are blanked mid-stream would read as all-air
            // (one solid glow slab) — skip both wholesale. The surface pass
            // renders unexplored terrain and lets the lighting darken it.
            if (mapPass == 2 &&
                (!maskIn(mcc) ||
                 texelFetch(mapExploredTex, mcc, 0).r == 0u || !gpuValid)) {
                // Wholly unexplored chunk column: a HOLE in the diorama.
                // Fast-forward to the column's XZ exit in one step (columns
                // span full Y, so only X/Z bound them) — the same shape as
                // the coarse-cell skip, including the degenerate-crossing
                // fallback to the plain one-brick step below.
                ivec2 cmin = ivec2((brick.x >> 3) << 3, (brick.z >> 3) << 3);
                float tx = (float(((stp.x > 0) ? cmin.x + 8 : cmin.x) * BRICK) - ro.x) * invD.x;
                float tz = (float(((stp.z > 0) ? cmin.y + 8 : cmin.y) * BRICK) - ro.z) * invD.z;
                float te; int ea;
                if (tx < tz) { te = tx; ea = 0; } else { te = tz; ea = 2; }
                if (!(te < tExit)) break;     // (NaN-safe) nothing more in the box
                vec3 pN = ro + rd * (te + 1e-3);
                ivec3 nb = ivec3(floor(pN / float(BRICK)));
                if (nb != brick) {
                    brick = nb;
                    for (int i = 0; i < 3; ++i) {
                        float bnd = float(brick[i]) + ((stp[i] > 0) ? 1.0 : 0.0);
                        tMaxB[i] = (bnd * float(BRICK) - ro[i]) * invD[i];
                    }
                    outerHitAxis = ea;
                    continue;
                }
                mapHole = true;               // degenerate: step, but stay a hole
            } else if (mapPass == 1 && outerHitAxis >= 0) {
                vec2 bc = (vec2(brick.xz) + 0.5) * float(BRICK);
                float sh = texture(heightTex, bc /
                           vec2(float(gridSize.x), float(gridSize.z))).r;
                float crustBot = min(sh, mapClipHi.y) - MAP_CRUST;
                float bTop = float((brick.y + 1) * BRICK);
                if (bTop < crustBot - 16.0) {
                    // Whole brick below the crust (16-voxel slope guard, since
                    // sh is sampled at the brick centre): carved air — step
                    // over it without descending or stopping. Without this,
                    // the mass stop below painted full-depth putty walls at
                    // the tile sides, crust or no crust.
                    mapHole = true;
                } else if (float(brick.y * BRICK) >= crustBot + 8.0) {
                    // Unvisited Y slice below the surface skin: UNKNOWN MASS.
                    // The ray stops at the crossed brick face whether the
                    // brick holds rock or an undiscovered tunnel's air.
                    // Chunks whose voxels are not yet GPU-resident (blanked
                    // mid-stream) read as air and get the same treatment —
                    // mass, not a see-through void. The margins (skin + brick
                    // relief above, crust + relief below) keep the mass
                    // between the true surface and the crust bottom on
                    // slopes; bricks straddling either boundary fall through
                    // to the per-voxel gate, which cuts exactly.
                    int slice = brick.y >> 1;     // (brick.y * 8) / 16
                    bool sliceSeen = false;
                    if (maskIn(mcc)) {            // OOB texelFetch is undefined
                        uint w = texelFetch(mapExploredYTex,
                                            ivec3(mcc.x, slice >> 5, mcc.y), 0).r;
                        sliceSeen = ((w >> uint(slice & 31)) & 1u) != 0u;
                    }
                    if ((!sliceSeen || !gpuValid) &&
                        bTop < sh - MAP_SKIN - 8.0) {
                        vec3 hp = ro + rd * tCurF;
                        vec3 nrmU = (outerHitAxis == 0) ? vec3(-float(stp.x), 0.0, 0.0)
                                  : (outerHitAxis == 1) ? vec3(0.0, -float(stp.y), 0.0)
                                                        : vec3(0.0, 0.0, -float(stp.z));
                        gPosMat = vec4(hp, float(MAP_UNKNOWN));
                        gNormal = vec4(nrmU, float(o));
                        gAlbedo = vec4(matColor(MAP_UNKNOWN), 1.0);
                        gl_FragDepth = min(tCurF / DEPTH_FAR, 1.0);
                        return;
                    }
                }
            }
        }

        // Directional reveal: hand off to the rings the moment the ray
        // enters a not-yet-uploaded chunk (its own bricks read as air; the
        // danger is what's BEHIND it). Chunk = 8 bricks; CHUNKS_XZ = 32.
        if (ringLive &&
            texelFetch(chunkValidTex,
                       ivec2((brick.x >> 3) & 31, (brick.z >> 3) & 31), 0).r == 0u) {
            handoffT = min(handoffT, tCurF);
            hReason = 1;
            break;
        }

        // Coarse skip (C1): if this whole coarse cell is empty, fast-forward
        // to its far face in one step instead of marching every empty brick —
        // same lossless skip anyHit/traceRay have always used; the primary
        // ray finally gets it now that the cascade makes rays long.
        ivec3 cellW = brick >> COARSE_BITS;
        ivec3 cellL = ivec3(cellW.x & (cdim.x - 1), cellW.y,
                            cellW.z & (cdim.z - 1));
        if (coarseOcc[cellL.x + cdim.x * (cellL.y + cdim.y * cellL.z)] == 0u) {
            ivec3 cmin = cellW << COARSE_BITS;
            vec3 tFar;
            for (int i = 0; i < 3; ++i) {
                int bb = (stp[i] > 0) ? (cmin[i] + COARSE) : cmin[i];
                tFar[i] = (float(bb * BRICK) - ro[i]) * invD[i];
            }
            float te = tFar.x; int ea = 0;
            if (tFar.y < te) { te = tFar.y; ea = 1; }
            if (tFar.z < te) { te = tFar.z; ea = 2; }
            if (te > fineCap) {          // clear through the cap: rings own the rest
                tCurF = fineCap;         // (proven empty up to the cap, not just here)
                hReason = 0;
                break;
            }
            vec3 pN = ro + rd * (te + 1e-3);
            ivec3 nb = ivec3(floor(pN / float(BRICK)));
            if (nb != brick) {          // forward progress: take the fast-forward
                brick = nb;
                for (int i = 0; i < 3; ++i) {
                    float bnd = float(brick[i]) + ((stp[i] > 0) ? 1.0 : 0.0);
                    tMaxB[i] = (bnd * float(BRICK) - ro[i]) * invD[i];
                }
                outerHitAxis = ea;
                continue;
            }
            // Degenerate crossing (corner graze / 0*INF): fall through to the
            // ordinary one-brick DDA step below — guaranteed progress. This
            // stall burned the whole iteration budget on rays with a near-
            // zero direction axis (the screen-column "moving strip").
        }

        if (tCurF > fineCap) { hReason = 0; break; }   // cap: rings own the rest

        uint entry = mapHole ? 0u : bm[bmSlot(brick)];
        if (entry != 0u) {
            tBrickEntry = tCurF;

            int   hitAxis = outerHitAxis;
            float tHit    = tBrickEntry;
            ivec3 voxel;
            uint  mat  = 0u;
            bool  hit  = false;

            if ((entry & BM_SOLID) != 0u) {
                vec3 hp = ro + rd * tBrickEntry + rd * 1e-4;
                voxel = ivec3(floor(hp));
                if (mapPass != 0)   // clip-plane entry point can graze the face
                    voxel = clamp(voxel, brick * BRICK, brick * BRICK + (BRICK - 1));
                mat   = entry & 0xFFu;
                if (mapPass == 1)      mat = mapGateVoxel(mat, voxel, mcc);
                else if (mapPass == 2) mat = mapCaveVoxel(mat, voxel, mcc);
                hit   = (mat != 0u);
            } else {
                vec3 pIn = ro + rd * (tBrickEntry + 1e-4);
                voxel = ivec3(floor(pIn));
                voxel = clamp(voxel, brick * BRICK, brick * BRICK + (BRICK - 1));
                vec3 tMaxV;
                for (int i = 0; i < 3; ++i) {
                    float bvox = float(voxel[i]) + ((stp[i] > 0) ? 1.0 : 0.0);
                    tMaxV[i] = (bvox - ro[i]) * invD[i];
                }
                vec3 tDeltaV = abs(invD);
                for (int s = 0; s < 32; ++s) {
                    if ((voxel >> 3) != brick) break;
                    uint v = sampleVoxel(voxel, entry);
                    if (mapPass == 1)      v = mapGateVoxel(v, voxel, mcc);
                    else if (mapPass == 2) v = mapCaveVoxel(v, voxel, mcc);
                    if (v != 0u) {
                        hit  = true;
                        mat  = v;
                        tHit = (hitAxis == 0) ? (tMaxV.x - tDeltaV.x)
                             : (hitAxis == 1) ? (tMaxV.y - tDeltaV.y)
                             : (hitAxis == 2) ? (tMaxV.z - tDeltaV.z)
                                              : tBrickEntry;
                        break;
                    }
                    if (tMaxV.x < tMaxV.y && tMaxV.x < tMaxV.z) {
                        voxel.x += stp.x; tMaxV.x += tDeltaV.x; hitAxis = 0;
                    } else if (tMaxV.y < tMaxV.z) {
                        voxel.y += stp.y; tMaxV.y += tDeltaV.y; hitAxis = 1;
                    } else {
                        voxel.z += stp.z; tMaxV.z += tDeltaV.z; hitAxis = 2;
                    }
                }
            }

            if (hit) {
                // A brick straddling the diorama box can hit past its face.
                if (mapPass != 0 && tHit > tExit + 1e-3) break;
                vec3 normal = vec3(0.0);
                if      (hitAxis == 0) normal.x = -float(stp.x);
                else if (hitAxis == 1) normal.y = -float(stp.y);
                else if (hitAxis == 2) normal.z = -float(stp.z);
                else                   normal   = -rd;
                vec3 hp = ro + rd * tHit;
                gPosMat = vec4(hp, float(mat));
                gNormal = vec4(normal, float(o));   // w = brick steps, for the heat view
                gAlbedo = vec4(surfaceColor(mat, hp, normal, tHit), 1.0);
                gl_FragDepth = min(tHit / DEPTH_FAR, 1.0);
                return;
            }
        } else if (mapPass == 2 && !mapHole && outerHitAxis >= 0) {
            // X-ray, EMPTY brick: cave space can begin at the brick face
            // itself (tunnel wall on a brick boundary) — test the entry voxel.
            vec3 pIn = ro + rd * (tCurF + 1e-4);
            ivec3 evx = clamp(ivec3(floor(pIn)),
                              brick * BRICK, brick * BRICK + (BRICK - 1));
            uint cv = mapCaveVoxel(0u, evx, mcc);
            if (cv != 0u) {
                vec3 normal = (outerHitAxis == 0) ? vec3(-float(stp.x), 0.0, 0.0)
                            : (outerHitAxis == 1) ? vec3(0.0, -float(stp.y), 0.0)
                                                  : vec3(0.0, 0.0, -float(stp.z));
                gPosMat = vec4(ro + rd * tCurF, float(cv));
                gNormal = vec4(normal, float(o));
                gAlbedo = vec4(matColor(MAP_UNKNOWN), 1.0);
                gl_FragDepth = min(tCurF / DEPTH_FAR, 1.0);
                return;
            }
        }

        if (tMaxB.x < tMaxB.y && tMaxB.x < tMaxB.z) {
            brick.x += stp.x; tMaxB.x += tDeltaB.x; outerHitAxis = 0;
        } else if (tMaxB.y < tMaxB.z) {
            brick.y += stp.y; tMaxB.y += tDeltaB.y; outerHitAxis = 1;
        } else {
            brick.z += stp.z; tMaxB.z += tDeltaB.z; outerHitAxis = 2;
        }
    }
    // Iteration budget exhausted (or any break above): the frontier of
    // PROVEN fine coverage is tCurF, never silently the cap.
    handoffT = min(handoffT, max(tCurF, 0.0));

    // Fine march over: the rings take the ray from wherever fine coverage
    // ACTUALLY ended (handoffT — the cap, a blank chunk, the window edge, or
    // budget exhaustion; conflating these was the root of the slab/see-
    // through artifacts). The cascade advances a monotone per-ray frontier:
    //   HIT    -> done;
    //   CLEAR  -> this band is proven empty; the next ring owns from the
    //             band boundary;
    //   LEFT / BUDGET -> this ring's coverage ended mid-band at tFrontier;
    //             the next (coarser) ring continues FROM THAT t, so stale
    //             ring extents degrade to coarser data instead of gaps.
    // A ring that starts inside solid content skips it (see traceRing) —
    // never a fabricated camera-facing face. The ring flag rides in
    // gPosMat.w as mat + 256 so lighting.frag can gate on it.
    if (ringLive) {
        float tFrontier = handoffT;
        for (int r = 0; r < RING_N; ++r) {
            if ((ringReadyMask & (1 << r)) == 0) continue;   // coarser ring inherits the band
            // Ring-to-ring switches dither too (band scales with the voxels).
            float bandEnd = switchJitter(camPos, rd, DS[r + 1] / rdXZ,
                                         float(64 << r) / rdXZ);
            if (mapPass != 0) bandEnd = min(bandEnd, tExit);   // stay in the slab
            if (bandEnd <= tFrontier) continue;              // band already behind the frontier
            vec3 rhp, rn; uint rmat; float rtW; int rSteps; float tf;
            int st = traceRing(r, ro, rd, tFrontier, bandEnd, tf,
                               rhp, rn, rmat, rtW, rSteps);
            if (st == TR_HIT) {
                gPosMat = vec4(rhp, float(rmat + 256u * uint(r + 1)));
                gNormal = vec4(rn, float(rSteps) + 2048.0 * float(hReason));
                // rmat's top 2 bits are baked AO (rings.cpp) — strip them for
                // the albedo lookup; lighting.frag decodes the full byte.
                gAlbedo = vec4(surfaceColor(rmat & 0x3Fu, rhp, rn, rtW), 1.0);
                gl_FragDepth = min(rtW / DEPTH_FAR, 1.0);
                return;
            }
            if (st == TR_CLEAR) tFrontier = bandEnd;
            else                tFrontier = max(tFrontier, tf);   // LEFT / BUDGET
        }
        gNormal = vec4(tFrontier, 0.0, 0.0, 2048.0 * float(hReason));
        return;
    }
    // Rings off/dead: the frontier is wherever the fine march ended (with
    // rings disabled the cap is infinite, so this is the window exit — the
    // heightfield then begins beyond the window instead of painting through
    // any hole inside it).
    gNormal = vec4(handoffT, 0.0, 0.0, 2048.0 * float(hReason));
    // No hit: miss sentinel already written.
}
