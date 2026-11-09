#include "worldgen/worldgen.h"
#include "world.h"
#include "common.h"
#include "worldgen/internal.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

// =============================================================================
// Far-field LOD rings — the voxel clipmap (C1)
// (docs/lod-rings-architecture.md is the design source of truth).
//
// RING_COUNT coarse brickmaps at 2^(r+1)x voxel scale covering 2^(r+1)x the
// window extent, generated DIRECTLY from the pure worldgen core (internal.h) —
// full-res data does not exist beyond the window, so nothing is downsampled;
// coarse content is re-derived from the same deterministic functions the fine
// world uses:
//   - terrain: surfaceColAt/surfaceMatAt at stride 2^(r+1), FLOOR-quantized
//     heights (a ring never pokes above finer data -> no rim walls at the
//     handoffs; at worst a <=1-coarse-voxel dip that reads as terrain),
//   - oasis water: filled where the shared heightAt/waterY says so,
//   - trees: coarse impostors (trunk column + ~65%-hash-filled canopy
//     ellipsoid, FAT-stamped) at the SAME lattice sites and RNG draws as the
//     fine trees via deriveTreeSite — a tree crossing any handoff keeps its
//     position and size; at ring 3/4 a tree is a 2-4 voxel blob (the LotL far
//     look),
//   - caves / grass / flowers / small flora: skipped (below ring resolution;
//     caves are also the 79% generation-time dominator — skipping them is why
//     each ring costs well under a second, not the window's tens of seconds).
//
// Threading contract (the black-terrain lesson): the parallel phase only
// WRITES DISJOINT per-column staging results; every ring-pool allocation
// happens in the serial merge on the calling (gen) thread via a bump
// allocator over the ring's dedicated slot range. World::allocBrick, the
// free list, and the fine pool vector are never touched.
// =============================================================================

namespace {

constexpr int RING_BY_MAX = BRICKS_Y >> 1;   // ring 0 has the most Y bricks (256)

// Staged fill result for one ring brick column, mirroring worldgen.cpp's
// ColumnResult pattern (parallel compute -> serial merge). Sized for the
// tallest ring; coarser rings use a prefix.
constexpr uint32_t RMIXED = 0xFFFFFFFFu;   // placeholder: data in `mixed`
struct RingCol {
    std::array<uint32_t, RING_BY_MAX> vals;
    std::vector<std::array<uint32_t, BRICK_U32>> mixed;   // by-order
};

// Ring-voxel surface margin around the brick classification band.
constexpr int RMARGIN = 5;

// ---- Baked ring AO (2 bits, upper bits of the ring voxel material byte) ----
// Ring hits have no traced AO (every secondary ray is window-anchored), so
// distant terrain shaded flat: valleys, ridges and canopy undersides all lit
// identically — the single biggest "cartoony far field" cause. Bake a coarse
// AO term at generation time instead: 3 = fully open, 0 = deepest concavity.
// The shader decodes baseMat = byte & 0x3F BEFORE any material logic (all
// real materials are <= 36 < 64, enforced below).
static_assert(MAT_DESERT_WOOD < 64, "ring AO needs every material id below 64");
constexpr uint8_t RAO_SHIFT = 6;
constexpr uint8_t RAO_OPEN  = uint8_t(3u << RAO_SHIFT);   // ao bits = 3 (no occlusion)

// Terrain AO for a voxel whose top sits at world height wy, given the
// neighbourhood average surface height: how far it sits BELOW its
// surroundings (valley floors and lower valley walls darken; ridges and
// mesa tops stay open). Normalization scales with the ring, so near rings
// respond to metre-scale dips and far rings to whole valleys.
inline uint8_t terrainAoBits(float avgH, int wy, int SL) {
    float occl = (avgH - float(wy)) / float(40 << SL);
    int bits = 3 - std::clamp(int(occl * 3.0f), 0, 3);
    return uint8_t(bits << RAO_SHIFT);
}

// Fill one ring brick column of ring r (absolute ring-brick coords; ring
// voxel = world >> (r+1)). Pure — safe from any thread.
void ringFillColumn(int r, int rbx, int rbz, RingCol& col) {
    const int SL = ringScaleLog2(r);
    const int BY = ringBricksY(r);
    const int rx0 = rbx * BRICK, rz0 = rbz * BRICK;
    SurfaceCol sc[BRICK * BRICK];
    int rh[BRICK * BRICK], rtop[BRICK * BRICK];
    int hMin = INT32_MAX, topMax = INT32_MIN;
    for (int lz = 0; lz < BRICK; ++lz)
    for (int lx = 0; lx < BRICK; ++lx) {
        int i = lx + BRICK * lz;
        // Sample at the ring voxel's MIN world corner — the same int fine-grid
        // convention every other height consumer uses (seam-exact by contract).
        int wx = (rx0 + lx) << SL;
        int wz = (rz0 + lz) << SL;
        sc[i] = surfaceColAt(wx, wz);
        rh[i] = sc[i].h >> SL;                        // floor-quantized surface
        int rwy = rh[i];
        // Oasis pool: the fine gen fills air below waterY inside the dish
        // (genOasis); the ring reproduces it from the same shared fields.
        if (sc[i].df.oasisW > 0.0f && sc[i].df.waterY > float(sc[i].h))
            rwy = int(sc[i].df.waterY) >> SL;
        rtop[i] = std::max(rwy, rh[i]);
        hMin   = std::min(hMin, rh[i]);
        topMax = std::max(topMax, rtop[i]);
    }
    // Neighbourhood average surface height for baked AO — brick-column
    // granular (4 heightAt samples per brick column keeps the whole ring's
    // extra cost ~0.1s; AO is a low-frequency valley cue, so brick-level
    // sampling radius + per-voxel depth is exactly the right resolution).
    float avgH;
    {
        int cxW = (rx0 + BRICK / 2) << SL;
        int czW = (rz0 + BRICK / 2) << SL;
        int D   = 24 << SL;
        avgH = 0.25f * float(heightAt(cxW + D, czW) + heightAt(cxW - D, czW)
                           + heightAt(cxW, czW + D) + heightAt(cxW, czW - D));
    }
    col.mixed.clear();
    for (int by = 0; by < BY; ++by) {
        int y0 = by * BRICK, y1 = y0 + BRICK - 1;
        if (y0 >= topMax + RMARGIN) {
            col.vals[by] = BM_EMPTY;
        } else if (y1 < hMin - RMARGIN - 4) {
            // Solid interior: one strata sample per brick (like fillColumn's
            // solid bricks) — strata vary slowly enough that per-voxel noise
            // is invisible at coarse scale from 100+ m away.
            col.vals[by] = BM_SOLID_FLAG | uint32_t(stoneAt(
                (rx0 + BRICK / 2) << SL,
                (y0  + BRICK / 2) << SL,
                (rz0 + BRICK / 2) << SL))
                | uint32_t(terrainAoBits(avgH, (y0 + BRICK / 2) << SL, SL));
        } else {
            // Surface band: per-voxel fill through the shared grammar.
            std::array<uint32_t, BRICK_U32> pd;
            pd.fill(0u);
            bool any = false;
            for (int lz = 0; lz < BRICK; ++lz)
            for (int lx = 0; lx < BRICK; ++lx) {
                int i = lx + BRICK * lz;
                int wx = (rx0 + lx) << SL;
                int wz = (rz0 + lz) << SL;
                for (int ly = 0; ly < BRICK; ++ly) {
                    int ry = y0 + ly;
                    uint8_t mat;
                    if (ry < rh[i]) {
                        // The top filled ring voxel always samples the actual
                        // surface skin (h-1: grass/sand/pavement); below, a
                        // mid-covered fine voxel, so dirt/sandstone bands land
                        // at the right coarse depth.
                        int wy = (ry == rh[i] - 1)
                                   ? sc[i].h - 1
                                   : std::min((ry << SL) + (1 << (SL - 1)),
                                              sc[i].h - 1);
                        mat = surfaceMatAt(wx, wy, wz, sc[i])
                            | terrainAoBits(avgH, (ry + 1) << SL, SL);
                    } else if (ry < rtop[i]) {
                        mat = MAT_WATER | RAO_OPEN;   // pools shade open
                    } else {
                        continue;
                    }
                    int linear = voxInBrick(lx, ly, lz);
                    pd[linear >> 2] |= uint32_t(mat) << ((linear & 3) * 8);
                    any = true;
                }
            }
            if (any) {
                col.vals[by] = RMIXED;
                col.mixed.push_back(pd);
            } else {
                col.vals[by] = BM_EMPTY;
            }
        }
    }
}

// Bump-allocate one brick of ring r (serial merge / stamping phase ONLY).
// Returns the ring-LOCAL slot index, or UINT32_MAX on range exhaustion (the
// caller degrades — never grows).
uint32_t ringAlloc(World::RingLevel& rl, int r, bool& overflow) {
    if (rl.brickCount >= RING_POOL_BRICKS[r]) {
        if (!overflow)
            std::fprintf(stderr, "[ring%d] pool slot range EXHAUSTED (%u bricks)"
                                 " — degrading (remaining impostors dropped)\n",
                         r + 1, RING_POOL_BRICKS[r]);
        overflow = true;
        return UINT32_MAX;
    }
    uint32_t idx = rl.brickCount++;
    rl.pool.resize(size_t(rl.brickCount) * BRICK_U32, 0u);
    return idx;
}

// Write one ring voxel, allocating / expanding bricks on demand. Serial
// phase only. `onlyIfEmpty` = leaves (never eat the trunk or terrain).
void setRingVoxel(World::RingLevel& rl, int r, bool& overflow,
                  int rx, int ry, int rz, uint8_t mat, bool onlyIfEmpty) {
    if (ry < 0 || ry >= ringBricksY(r) * BRICK) return;
    int bx = rx >> BRICK_BITS, by = ry >> BRICK_BITS, bz = rz >> BRICK_BITS;
    if (bx < rl.originB.x || bx >= rl.originB.x + RING_BRICKS_XZ ||
        bz < rl.originB.y || bz >= rl.originB.y + RING_BRICKS_XZ) return;
    int slot = World::ringSlotOfBrick(r, bx, by, bz);
    uint32_t entry = rl.bm[slot];
    uint32_t bi;
    if (entry == BM_EMPTY) {
        bi = ringAlloc(rl, r, overflow);
        if (bi == UINT32_MAX) return;
        rl.bm[slot] = ringPoolBase(r) + bi + 1u;   // ring-pool GLOBAL index
    } else if (entry & BM_SOLID_FLAG) {
        if (onlyIfEmpty) return;                   // solid terrain stays terrain
        bi = ringAlloc(rl, r, overflow);
        if (bi == UINT32_MAX) return;
        uint8_t fill = uint8_t(entry & 0xFFu);
        uint32_t pat = uint32_t(fill) | (uint32_t(fill) << 8)
                     | (uint32_t(fill) << 16) | (uint32_t(fill) << 24);
        for (int i = 0; i < BRICK_U32; ++i)
            rl.pool[size_t(bi) * BRICK_U32 + i] = pat;
        rl.bm[slot] = ringPoolBase(r) + bi + 1u;
    } else {
        bi = entry - 1u - ringPoolBase(r);
    }
    int linear = voxInBrick(rx & BRICK_MASK, ry & BRICK_MASK, rz & BRICK_MASK);
    uint32_t& word = rl.pool[size_t(bi) * BRICK_U32 + (linear >> 2)];
    uint32_t shift = uint32_t(linear & 3) * 8u;
    if (onlyIfEmpty && ((word >> shift) & 0xFFu) != 0u) return;
    word = (word & ~(0xFFu << shift)) | (uint32_t(mat) << shift);
}

// Cheap deterministic per-tree hash stream for impostor lobe VARIATION — an
// INDEPENDENT stream, deliberately NOT the fine tree's mt19937: replaying
// that exactly would cost ~100k discarded draws per tree (the fine canopy
// burns one draw per candidate voxel while filling bulbs), billions per
// coarse ring. Position, height and envelope still match the fine twin
// exactly (they come from the shared deriveTreeSite draws); only the lobe
// arrangement differs, which the handoff distance conceals.
struct TreeHash {
    uint64_t s;
    float f()   { s = mix64(s); return float(s & 0xFFFFFFu) / float(0x1000000); }
    float f11() { return f() * 2.0f - 1.0f; }
};

// Stamp a leaf blob (ellipsoid). The fine bulbs' ~65% popcorn dither applies
// ONLY to the outer shell of larger blobs — the core stays solid, so a lobe
// can never fragment or detach from the branch that feeds it; blobs two
// coarse voxels or smaller stamp entirely solid (they must never thin out).
void stampLeafBlob(World::RingLevel& rl, int r, bool& overflow,
                   float cxW, float cyW, float czW, float radW, float radYW) {
    const int SL = ringScaleLog2(r);
    float scaleF = float(1 << SL);
    radW  = std::max(radW,  scaleF);
    radYW = std::max(radYW, scaleF);
    int rcx = int(cxW) >> SL, rcy = int(cyW) >> SL, rcz = int(czW) >> SL;
    int rrx = std::max(int(radW)  >> SL, 1);
    int rry = std::max(int(radYW) >> SL, 1);
    bool dither = (rrx > 2);
    for (int dz = -rrx; dz <= rrx; ++dz)
    for (int dy = -rry; dy <= rry; ++dy)
    for (int dx = -rrx; dx <= rrx; ++dx) {
        float half = scaleF * 0.5f;
        float wxc = float((rcx + dx) << SL) + half - cxW;
        float wyc = float((rcy + dy) << SL) + half - cyW;
        float wzc = float((rcz + dz) << SL) + half - czW;
        float d = (wxc * wxc) / (radW * radW) + (wyc * wyc) / (radYW * radYW)
                + (wzc * wzc) / (radW * radW);
        if (d > 1.0f) continue;
        // Occupancy field is WORLD-anchored (quantized to the ring-1 voxel
        // lattice) and shared by every ring level — hashing per-level
        // re-rolled which shell voxels survive at each handoff, adding
        // silhouette crawl on top of the resolution change.
        int wqx = ((rcx + dx) << SL) >> 1;
        int wqy = ((rcy + dy) << SL) >> 1;
        int wqz = ((rcz + dz) << SL) >> 1;
        if (dither && d > 0.45f &&        // airy OUTER shell, solid core
            (featureSeed(0x1EAFu, 0x51u, wqx, wqy, wqz) & 1023u) < 358u)
            continue;
        // Baked canopy AO: undersides and the lobe interior sit in leaf
        // shadow, crowns are open sky — the vertical gradient that makes a
        // distant tree read as a lit volume instead of a flat green blob.
        float rel = wyc / radYW;                     // -1 bottom .. +1 top
        uint8_t aoB = (rel > 0.20f) ? 3 : (rel > -0.35f) ? 2 : 1;
        setRingVoxel(rl, r, overflow, rcx + dx, rcy + dy, rcz + dz,
                     uint8_t(MAT_LEAVES | (aoB << RAO_SHIFT)), true);
    }
}

// Stamp a coarse wood segment (1 ring voxel thick) from a to b — the branch
// that CONNECTS a lobe to the trunk. Wood is stamped before/over leaves so
// the connection can never be dithered away.
void stampWoodSeg(World::RingLevel& rl, int r, bool& overflow,
                  glm::vec3 a, glm::vec3 b) {
    const int SL = ringScaleLog2(r);
    float len = glm::length(b - a);
    int steps = std::max(2, int(len / float(1 << SL)) + 1);
    for (int i = 0; i <= steps; ++i) {
        glm::vec3 p = glm::mix(a, b, float(i) / float(steps));
        setRingVoxel(rl, r, overflow,
                     int(p.x) >> SL, int(p.y) >> SL, int(p.z) >> SL,
                     uint8_t(MAT_WOOD | (2u << RAO_SHIFT)), false);   // under canopy
    }
}

// Stamp one tree impostor. Rings 1-2 (20/40 cm voxels): a MULTI-LOBE canopy
// mirroring the fine L-system's structure — 2-3 branch lobes at each of the
// same four trunk heights / polar angles / reach the fine grow() uses, plus
// the apex lobe — so distant trees keep irregular, individual silhouettes
// instead of reading as a field of identical bulbs. Rings 3-4 (80/160 cm):
// lobes are sub-voxel, so one fat ellipsoid remains the right stamp there.
void stampTreeImpostor(World::RingLevel& rl, int r, uint64_t worldSeed,
                       bool& overflow, const TreeSite& s) {
    const int SL = ringScaleLog2(r);
    float L = s.trunkLen;
    int   rx = s.x >> SL, rz = s.z >> SL;
    int   rh = s.h >> SL;
    float bx = float(s.x) + 0.5f, bz = float(s.z) + 0.5f;
    // Trunk: base sunk 1 voxel into ground, rising to ~the apex like the fine
    // trunk does (the old 55% cut was hidden inside the single ellipsoid —
    // with separated lobes it left identical exposed poles). Width honours
    // trunkThick (fine radius 2.5-4 world -> 1-2 ring-1 voxels diameter+).
    int trunkTop = (s.h + int(L * 0.96f)) >> SL;
    // Rasterize the PHYSICAL diameter (2*trunkThick world voxels) at this
    // ring's scale — the old per-ring bit-shift doubled thick trunks' world
    // width between rings 1 and 2.
    int halfW = std::clamp(int(s.trunkThick / float(1 << SL) - 0.25f), 0, 2);
    for (int ry = rh - 1; ry <= trunkTop; ++ry)
        for (int dz = -halfW; dz <= halfW; ++dz)
        for (int dx = -halfW; dx <= halfW; ++dx)
            setRingVoxel(rl, r, overflow, rx + dx, ry, rz + dz,
                         uint8_t(MAT_WOOD | (2u << RAO_SHIFT)), false);

    if (r >= 3) {
        // Final 16x ring: single envelope blob (lobes are sub-voxel there).
        stampLeafBlob(rl, r, overflow, bx, float(s.h) + 0.72f * L, bz,
                      0.60f * L + 3.5f, 0.42f * L + 3.5f);
        return;
    }

    // Rings 1-3: lobed canopy. Structure constants mirror flora.cpp grow()
    // (branch LEVELS/THETAS, reach and shell radii); the per-tree variation
    // comes from the independent hash stream. Every lobe gets a WOOD branch
    // segment back to its origin on the trunk — tips without branches float.
    TreeHash th{ mix64(worldSeed ^ 0xB10B1EAFu ^
                       (uint64_t(uint32_t(s.x)) << 32) ^ uint64_t(uint32_t(s.z))) };
    static const float LEVELS[4] = { 0.40f, 0.58f, 0.74f, 0.90f };
    static const float THETAS[4] = { 1.00f, 0.78f, 0.58f, 0.40f };
    float lobeScale = s.scale;
    for (int lvl = 0; lvl < 4; ++lvl) {
        // Attach-height clamp mirrors flora.cpp grow() EXACTLY (canopy
        // clearance over the character) — diverging here drops the crown at
        // the fine->ring handoff.
        float att = std::max(L * LEVELS[lvl],
                             std::min(L * 0.92f,
                                      TREE_CANOPY_CLEAR + 4.0f * float(lvl)));
        float loY = float(s.h) + att;
        int n = 2 + (int(th.f() * 2.0f) & 1);              // 2-3 lobes per level
        float yawBase = LEVELS[lvl] * 3.7f + th.f() * 1.5f;
        for (int b = 0; b < n; ++b) {
            float yaw   = yawBase + (float(b) / float(n)) * 6.2831853f
                        + th.f11() * 0.25f;
            float theta = THETAS[lvl] + th.f11() * 0.12f;
            float sy    = std::max(std::cos(theta), 0.15f);
            float reach = 0.55f + 0.30f * (1.0f - LEVELS[lvl]);
            float bl    = L * (reach + th.f() * 0.15f);
            float cx = bx + std::cos(yaw) * std::sin(theta) * bl;
            float cy = loY + sy * bl;
            float cz = bz + std::sin(yaw) * std::sin(theta) * bl;
            // Branch first (wood is never dithered away), then foliage.
            stampWoodSeg(rl, r, overflow,
                         glm::vec3(bx, loY, bz), glm::vec3(cx, cy, cz));
            // Fine lobes read as shellR + bulb radius (~3-5 world voxels
            // beyond the shell) — undersizing them left gaps at branch tips.
            float rl2 = (5.5f + th.f() * 2.5f) * lobeScale + 3.5f;
            stampLeafBlob(rl, r, overflow, cx, cy, cz, rl2, rl2 * 0.85f);
        }
    }
    // Apex lobe capping the trunk (the fine tree's crown centreline).
    stampLeafBlob(rl, r, overflow, bx, float(s.h) + L, bz,
                  (6.0f + th.f() * 2.0f) * lobeScale + 3.0f,
                  (5.0f + th.f() * 1.5f) * lobeScale + 3.0f);
}

// Desert flora impostors. The desert's plants ARE its silhouette — without
// them the handoff read as a "plants vanish" circle (ocotillo/saguaro/
// creosote sliced at D0). Site selection REPLAYS flora.cpp's lattices
// draw-for-draw (the SEED_DESERT_SHRUB / SEED_SAGUARO / SEED_OCOTILLO /
// SEED_OASIS_PALM streams). KEEP IN SYNC with flora.cpp: drift only
// misplaces impostors (the fine world is untouched — these are local
// replays, not shared code), but a misplaced impostor pops at the boundary.
// Gates use the pure grammar instead of the surface snapshot (~99% agree).
void stampDesertFlora(World::RingLevel& rl, int r, GenContext& ctx,
                      bool& overflow, int wx0, int wz0, int wx1, int wz1) {
    const int SL = ringScaleLog2(r);
    std::mt19937 rng;
    auto frand = [&]() { return frand01(rng); };
    auto desertSurf = [](int x, int h, int z) {
        SurfaceCol g = surfaceColAt(x, z);
        uint8_t m = surfaceMatAt(x, h - 1, z, g);
        return m == MAT_DESERT_PAVEMENT || m == MAT_SAND_STATIC;
    };
    auto column = [&](int cx, int h, int cz, int topWorld, uint8_t mat, int halfW) {
        int rx = cx >> SL, rz = cz >> SL;
        for (int ry = (h >> SL) - 1; ry <= (topWorld >> SL); ++ry)
            for (int dz = -halfW; dz <= halfW; ++dz)
            for (int dx = -halfW; dx <= halfW; ++dx)
                setRingVoxel(rl, r, overflow, rx + dx, ry, rz + dz, mat, false);
    };

    // Saguaro (rings 1-3): the hero cactus — a solid flesh column.
    // Mirrors flora.cpp genBushes' SEED_SAGUARO block + placeSaguaro's H roll.
    if (r <= 2) {
        constexpr int SAG_CELL = 96;
        for (int gz = floorDiv(wz0, SAG_CELL); gz <= floorDiv(wz1 - 1, SAG_CELL); ++gz)
        for (int gx = floorDiv(wx0, SAG_CELL); gx <= floorDiv(wx1 - 1, SAG_CELL); ++gx) {
            rng.seed(featureSeed(ctx.seed, SEED_SAGUARO, gx, gz));
            int cx = gx * SAG_CELL + 8 + int(frand() * float(SAG_CELL - 16));
            int cz = gz * SAG_CELL + 8 + int(frand() * float(SAG_CELL - 16));
            float dW = biomeDesertW(cx, cz);
            if (dW < 0.35f) continue;
            if (frand() > dW * 0.45f) continue;
            int h = heightAt(cx, cz);
            if (h < 10 || h >= GRID_Y - 60) continue;
            if (!desertSurf(cx, h, cz)) continue;
            { OasisHit oa = oasisAt(cx, cz); if (oa.any && h <= oa.waterY) continue; }
            float roll = frand();
            int H = (roll < 0.65f) ? (26 + int(frand() * 12.0f))
                  : (roll < 0.88f) ? (20 + int(frand() *  8.0f))
                                   : (38 + int(frand() * 14.0f));
            column(cx, h, cz, h + H, uint8_t(MAT_CACTUS_FLESH | RAO_OPEN),
                   (H > 36 && SL == 1) ? 1 : 0);
        }
    }

    // Ocotillo (rings 1-2): thin cane spray with red bloom tips — the red
    // accents the player tracks across the boundary. Site replay of the
    // SEED_OCOTILLO lattice; the stamp is a representative silhouette.
    if (r <= 1) {
        constexpr int OCO_CELL = 34;
        for (int gz = floorDiv(wz0, OCO_CELL); gz <= floorDiv(wz1 - 1, OCO_CELL); ++gz)
        for (int gx = floorDiv(wx0, OCO_CELL); gx <= floorDiv(wx1 - 1, OCO_CELL); ++gx) {
            rng.seed(featureSeed(ctx.seed, SEED_OCOTILLO, gx, gz));
            int cx = gx * OCO_CELL + 4 + int(frand() * float(OCO_CELL - 8));
            int cz = gz * OCO_CELL + 4 + int(frand() * float(OCO_CELL - 8));
            float dW = biomeDesertW(cx, cz);
            if (dW < 0.05f) continue;
            float dens = 0.04f + 0.10f * smoothNoise(cx * 0.01f + 21.0f, cz * 0.01f + 5.0f);
            if (frand() > dW * dens) continue;
            int h = heightAt(cx, cz);
            if (h < 10 || h >= GRID_Y - 30) continue;
            if (!desertSurf(cx, h, cz)) continue;
            { OasisHit oa = oasisAt(cx, cz); if (oa.any && h <= oa.waterY) continue; }
            column(cx, h, cz, h + 16, uint8_t(MAT_DESERT_WOOD | RAO_OPEN), 0);
            int rx = cx >> SL, rz = cz >> SL, rt = (h + 18) >> SL;
            setRingVoxel(rl, r, overflow, rx + 1, rt, rz,     uint8_t(MAT_CACTUS_BLOOM | RAO_OPEN), true);
            setRingVoxel(rl, r, overflow, rx - 1, rt, rz + 1, uint8_t(MAT_CACTUS_BLOOM | RAO_OPEN), true);
        }
    }

    // Creosote matrix (ring 1 only — sub-voxel beyond): the desert's ground
    // texture. Site replay of the SEED_DESERT_SHRUB lattice.
    if (r == 0) {
        constexpr int SHRUB_CELL = 20;
        for (int gz = floorDiv(wz0, SHRUB_CELL); gz <= floorDiv(wz1 - 1, SHRUB_CELL); ++gz)
        for (int gx = floorDiv(wx0, SHRUB_CELL); gx <= floorDiv(wx1 - 1, SHRUB_CELL); ++gx) {
            rng.seed(featureSeed(ctx.seed, SEED_DESERT_SHRUB, gx, gz));
            int cx = gx * SHRUB_CELL + 3 + int(frand() * float(SHRUB_CELL - 6));
            int cz = gz * SHRUB_CELL + 3 + int(frand() * float(SHRUB_CELL - 6));
            float dW = biomeDesertW(cx, cz);
            if (dW < 0.05f) continue;
            float dens = 0.04f + 0.10f * smoothNoise(cx * 0.012f + 8.0f, cz * 0.012f + 3.0f);
            if (frand() > dW * dens) continue;
            int h = heightAt(cx, cz);
            if (h < 10 || h >= GRID_Y - 30) continue;
            if (!desertSurf(cx, h, cz)) continue;
            { OasisHit oa = oasisAt(cx, cz); if (oa.any && h <= oa.waterY) continue; }
            float scale = 0.85f + 0.65f * frand();
            int rx = cx >> SL, rz = cz >> SL, rh = h >> SL;
            int top = (h + int(8.0f * scale)) >> SL;
            for (int ry = rh; ry <= top; ++ry)
                setRingVoxel(rl, r, overflow, rx, ry, rz, uint8_t(MAT_SHRUB_SAGE | RAO_OPEN), false);
            if (scale > 1.1f) {
                setRingVoxel(rl, r, overflow, rx + 1, top, rz, uint8_t(MAT_SHRUB_SAGE | RAO_OPEN), true);
                setRingVoxel(rl, r, overflow, rx, top, rz + 1, uint8_t(MAT_SHRUB_SAGE | RAO_OPEN), true);
            }
        }
    }

    // Oasis palms (rings 1-3): tall landmarks ringing each pool. Site replay
    // of the SEED_OASIS_PALM block (oasisHost/oasisSite are the same pure
    // source of truth the pool itself uses).
    if (r <= 2) {
        for (int gz = floorDiv(wz0, OASIS_CELL); gz <= floorDiv(wz1 - 1, OASIS_CELL); ++gz)
        for (int gx = floorDiv(wx0, OASIS_CELL); gx <= floorDiv(wx1 - 1, OASIS_CELL); ++gx) {
            if (!oasisHost(gx, gz)) continue;
            OasisSite s = oasisSite(gx, gz);
            if (!s.pass) continue;
            rng.seed(featureSeed(ctx.seed, SEED_OASIS_PALM, gx, gz));
            int nPalm = 6 + int(rng() % 4);
            for (int i = 0; i < nPalm; ++i) {
                float ang = frand() * 6.2831853f;
                float uu  = 0.82f + 0.28f * frand();
                int px = int(s.ctrX + std::cos(ang) * uu * s.R);
                int pz = int(s.ctrZ + std::sin(ang) * uu * s.R);
                int ph = heightAt(px, pz);
                if (ph < s.waterY + 1) continue;
                if (!desertSurf(px, ph, pz)) continue;
                // Representative trunk + frond crown (exact height rolls live
                // inside placePalm; a mid-range trunk reads right at 100m+).
                column(px, ph, pz, ph + 56, uint8_t(MAT_WOOD | (2u << RAO_SHIFT)), 0);
                int rx = px >> SL, rz = pz >> SL, rc = (ph + 58) >> SL;
                int rr = std::max(1, 9 >> SL);
                for (int dz = -rr; dz <= rr; ++dz)
                for (int dx = -rr; dx <= rr; ++dx)
                    if (dx * dx + dz * dz <= rr * rr)
                        setRingVoxel(rl, r, overflow, rx + dx, rc, rz + dz,
                                     uint8_t(MAT_LEAVES | RAO_OPEN), true);
            }
        }
    }
}

}  // namespace

// Build (or rebuild) ring r centred on world (cxW, czW) — an EXPLICIT
// snapshot taken by the caller (the recentre worker must not read the live,
// scrolling window origin: torn/racing reads). Gen/background thread; CPU
// staging only — the CALLER publishes ready/dirty on the main thread and
// World::uploadRings swaps the GPU copy (the shader keeps addressing the OLD
// resident data via originGPU meanwhile).
void buildRing(World& w, GenContext& ctx, int r, int cxW, int czW) {
    auto t0 = std::chrono::steady_clock::now();
    const int SL = ringScaleLog2(r);
    const int BY = ringBricksY(r);
    World::RingLevel& rl = w.rings[r];

    // Centre on the snapshot, ring-chunk (8-brick) aligned.
    int rbx = floorDiv((cxW >> SL) - RING_GRID_XZ / 2, BRICK);
    int rbz = floorDiv((czW >> SL) - RING_GRID_XZ / 2, BRICK);
    rl.originB = { floorDiv(rbx, CHUNK_BRICKS) * CHUNK_BRICKS,
                   floorDiv(rbz, CHUNK_BRICKS) * CHUNK_BRICKS };
    // NOTE: `ready` is deliberately NOT cleared — the shader keeps rendering
    // the previous GPU copy (originGPU) during a background rebuild.
    rl.brickCount = 0;
    rl.bm.assign(size_t(ringBmEntries(r)), BM_EMPTY);
    rl.pool.clear();
    rl.coarse.assign(size_t(ringCoarseEntries(r)), 0u);

    // 1) Terrain fill — parallel compute into disjoint staging columns.
    const int NB = RING_BRICKS_XZ;
    std::vector<RingCol> grid(size_t(NB) * NB);
    {
        std::atomic<int> next{ 0 };
        int nT = int(std::max(1u, std::thread::hardware_concurrency()));
        std::vector<std::thread> workers;
        workers.reserve(nT);
        for (int t = 0; t < nT; ++t) {
            workers.emplace_back([&]() {
                int i;
                while ((i = next.fetch_add(1, std::memory_order_relaxed))
                       < NB * NB) {
                    int lx = i & (NB - 1), lz = i >> 8;   // NB == 256
                    ringFillColumn(r, rl.originB.x + lx, rl.originB.y + lz,
                                   grid[i]);
                }
            });
        }
        for (auto& th : workers) th.join();
    }

    // 2) Serial merge: bump-allocate ring slots in fixed (lz, lx, by) order.
    bool overflow = false;
    for (int lz = 0; lz < NB; ++lz)
    for (int lx = 0; lx < NB; ++lx) {
        RingCol& col = grid[size_t(lx) + size_t(NB) * lz];
        size_t mi = 0;
        int bx = rl.originB.x + lx, bz = rl.originB.y + lz;
        for (int by = 0; by < BY; ++by) {
            uint32_t v = col.vals[by];
            if (v == RMIXED) {
                uint32_t bi = ringAlloc(rl, r, overflow);
                if (bi == UINT32_MAX) { v = BM_EMPTY; }
                else {
                    std::memcpy(&rl.pool[size_t(bi) * BRICK_U32],
                                col.mixed[mi].data(), BRICK_U32 * sizeof(uint32_t));
                    v = ringPoolBase(r) + bi + 1u;   // ring-pool GLOBAL index
                }
                ++mi;
            }
            rl.bm[World::ringSlotOfBrick(r, bx, by, bz)] = v;
        }
        col.mixed.clear();
        col.mixed.shrink_to_fit();
    }
    grid.clear(); grid.shrink_to_fit();
    uint32_t terrainBricks = rl.brickCount;

    // 3) Tree impostors over the ring extent — same lattice + draws as
    //    genTrees (deriveTreeSite), gated by the PURE grass test.
    int wx0 = (rl.originB.x * BRICK) << SL;
    int wz0 = (rl.originB.y * BRICK) << SL;
    int wx1 = wx0 + (RING_GRID_XZ << SL);
    int wz1 = wz0 + (RING_GRID_XZ << SL);
    {
        std::mt19937 rng;
        auto pureGrass = [](int x, int h, int z) {
            SurfaceCol g = surfaceColAt(x, z);
            return surfaceMatAt(x, h - 1, z, g) == MAT_GRASS;
        };
        const int M = 96;   // canopy reach margin (mirrors genTrees' MARGIN_TREE scale)
        int kz0 = floorDiv(wz0 - M - TREE_CELL - TREE_ANCHOR, TREE_CELL);
        int kz1 = floorDiv(wz1 - 1 + M - TREE_ANCHOR, TREE_CELL);
        int kx0 = floorDiv(wx0 - M - TREE_CELL - TREE_ANCHOR, TREE_CELL);
        int kx1 = floorDiv(wx1 - 1 + M - TREE_ANCHOR, TREE_CELL);
        int stamped = 0;
        for (int kz = kz0; kz <= kz1 && !overflow; ++kz)
        for (int kx = kx0; kx <= kx1 && !overflow; ++kx) {
            TreeSite s = deriveTreeSite(ctx.seed, TREE_ANCHOR + kx * TREE_CELL,
                                        TREE_ANCHOR + kz * TREE_CELL, rng, pureGrass);
            if (!s.spawn) continue;
            stampTreeImpostor(rl, r, ctx.seed, overflow, s);
            ++stamped;
        }
        // Desert flora impostors (saguaro/ocotillo/creosote/palms) — without
        // them the desert's plants were sliced to nothing at the handoff.
        stampDesertFlora(rl, r, ctx, overflow, wx0, wz0, wx1, wz1);

        auto t1 = std::chrono::steady_clock::now();
        std::printf("[ring%d] built in %.2fs: %u bricks (%u terrain + %u tree/flora), "
                    "%d tree impostors, origin brick (%d,%d)%s\n",
                    r + 1, std::chrono::duration<double>(t1 - t0).count(),
                    rl.brickCount, terrainBricks,
                    rl.brickCount - terrainBricks, stamped,
                    rl.originB.x, rl.originB.y,
                    overflow ? "  [DEGRADED: slot range exhausted]" : "");
    }

    // 4) Coarse occupancy, indexed EXACTLY as the shader reads it
    //    (wrapped X/Z local cell, absolute Y; x + 64*(y + cy*z)).
    const int CY = BY >> COARSE_BITS;
    for (int lz = 0; lz < NB; ++lz)
    for (int by = 0; by < BY; ++by)
    for (int lx = 0; lx < NB; ++lx) {
        int bx = rl.originB.x + lx, bz = rl.originB.y + lz;
        if (rl.bm[World::ringSlotOfBrick(r, bx, by, bz)] == BM_EMPTY) continue;
        int cx = (bx >> COARSE_BITS) & (RING_CBRICKS_XZ - 1);
        int cy = by >> COARSE_BITS;
        int cz = (bz >> COARSE_BITS) & (RING_CBRICKS_XZ - 1);
        rl.coarse[size_t(cx) + RING_CBRICKS_XZ *
                  (size_t(cy) + size_t(CY) * cz)] = 1u;
    }

    // NOTE: ready/dirty are published by the CALLER (main/gen thread) — a
    // worker writing flags the render thread reads is a data race.
}

void buildRings(World& w, GenContext& ctx) {
    // Synchronous path (startup / teleport regen): single-threaded here, so
    // publishing the flags inline is safe.
    int cxW = w.originVoxX() + GRID_XZ / 2;
    int czW = w.originVoxZ() + GRID_XZ / 2;
    for (int r = 0; r < RING_COUNT; ++r) {
        buildRing(w, ctx, r, cxW, czW);
        w.rings[r].ready = true;
        w.rings[r].dirty = true;
    }
}
