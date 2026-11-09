#include "worldgen/worldgen.h"
#include "world.h"
#include "common.h"
#include "worldgen/internal.h"
#include "profiling.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <unordered_set>
#include <vector>



// -- Surface boulders ----------------------------------------------------
// Noise-warped ellipsoids embedded in the ground at random surface points.
// Multi-octave bulge noise gives them lumpy, distinct silhouettes; random
// yaw rotation breaks grid alignment; vertical squash + 25-55% embedding
// makes them look planted, not dropped.
void genBoulders(World& w, GenContext& ctx, const GenBounds& b,
                 const SurfaceSnapshot& snap, ChunkStageBuffer* buf) {
    StageWriter gw(w, b, ctx, buf);
    std::mt19937 rng;   // re-seeded per lattice cell (featureSeed)
    auto frand = [&]() { return frand01(rng); };

    auto placeBoulder = [&](int cx, int cz, float sizeOverride) {
        int h = heightAt(cx, cz);
        if (h < 10 || h >= GRID_Y - 30) return;
        // Only plant on actual grass — skip cave mouths, exposed stone,
        // water, etc. Gate reads the PRE-STAGE snapshot so the decision is
        // identical no matter which chunk replays this boulder.
        if (snap.get(cx, h - 1, cz) != MAT_GRASS) return;

        // Base size with squared bias toward small (most boulders modest,
        // rare erratics big).
        float sizeRoll = frand();
        float base = sizeOverride > 0.0f
                   ? sizeOverride
                   : 2.5f + 14.0f * sizeRoll * sizeRoll;

        // Ellipsoid radii: squashed vertically, slight aspect variation.
        float aX = base * (0.85f + 0.45f * frand());
        float aZ = base * (0.85f + 0.45f * frand());
        float aY = base * (0.55f + 0.30f * frand());

        float embed = 0.25f + 0.30f * frand();   // 25-55% buried
        int   cy    = int(std::round(float(h) - aY * embed));

        float yaw = frand() * 6.2831853f;
        float cs = std::cos(yaw), sn = std::sin(yaw);
        float nseed = frand() * 100.0f;

        int rad = int(std::ceil(std::max({aX, aY, aZ}))) + 1;
        for (int dz = -rad; dz <= rad; ++dz)
        for (int dy = -rad; dy <= rad; ++dy)
        for (int dx = -rad; dx <= rad; ++dx) {
            int x = cx + dx, y = cy + dy, z = cz + dz;
            if (y < 0 || y >= GRID_Y) continue;   // XZ unbounded

            // Rotate around vertical so the ellipsoid isn't grid-aligned.
            float lx = float(dx) * cs - float(dz) * sn;
            float lz = float(dx) * sn + float(dz) * cs;
            float ly = float(dy);

            float ex = lx / aX, ey = ly / aY, ez = lz / aZ;
            float d  = ex*ex + ey*ey + ez*ez;

            // Two-octave bulge: broad knobs + fine detail.
            float big  = smoothNoise3(float(x) * 0.08f + nseed,
                                      float(y) * 0.08f + nseed,
                                      float(z) * 0.08f + nseed);
            float fine = smoothNoise3(float(x) * 0.28f + nseed * 0.5f,
                                      float(y) * 0.28f + nseed * 0.5f,
                                      float(z) * 0.28f + nseed * 0.5f);
            float bulge = (big - 0.5f) * 0.55f + (fine - 0.5f) * 0.18f;

            if (d > 1.0f + bulge) continue;

            uint8_t cur = gw.get(x, y, z);
            if (cur == MAT_WOOD || cur == MAT_LEAVES) continue;  // preserve trees
            gw.set(x, y, z, MAT_STONE);
        }
        if (gw.anchorIn(cx, cz)) ++gw.stats().boulders;
    };

    // Lattice-seeded placement: each 128-voxel cell derives its own RNG
    // stream from (ctx.seed, SEED_BOULDER, cell), so any cell's boulders
    // can be regenerated in isolation (chunk prerequisite). Density matches
    // the old global count (300 over the world) in expectation.
    // Footprint contract: boulder radius <= ~17 + cluster spread ~22, so a
    // chunk must replay cells within ~40 voxels of its border.
    constexpr int   CELL     = 128;
    constexpr float PER_CELL = 300.0f / float((GRID_XZ / CELL) * (GRID_XZ / CELL));
    // Replay every cell whose features could write into (or gate-interact
    // with) the bounds — an unbounded lattice over all of Z^2.
    int c0x = floorDiv(b.x0 - MARGIN_BOULDER, CELL);
    int c1x = floorDiv(b.x1 - 1 + MARGIN_BOULDER, CELL);
    int c0z = floorDiv(b.z0 - MARGIN_BOULDER, CELL);
    int c1z = floorDiv(b.z1 - 1 + MARGIN_BOULDER, CELL);
    for (int cellZ = c0z; cellZ <= c1z; ++cellZ)
    for (int cellX = c0x; cellX <= c1x; ++cellX) {
        rng.seed(featureSeed(ctx.seed, SEED_BOULDER, cellX, cellZ));
        // BIOME density: highland is rocky (more boulders), forest sparse.
        // Scaled at the cell centre so it's replay-consistent and blends.
        float perCell = PER_CELL *
            biomeAt(cellX * CELL + CELL / 2, cellZ * CELL + CELL / 2).boulderScale;
        int n = int(perCell);
        if (frand() < perCell - float(int(perCell))) ++n;
        for (int i = 0; i < n; ++i) {
            int cx = cellX * CELL + int(frand() * CELL);
            int cz = cellZ * CELL + int(frand() * CELL);
            placeBoulder(cx, cz, -1.0f);

            // ~30% chance to drop 1-3 smaller neighbours nearby for a
            // cluster look; small ones can't seed another cluster.
            if ((rng() & 1023) < 300) {
                int extra = 1 + int(rng() % 3);
                for (int j = 0; j < extra; ++j) {
                    int ox = cx + int((frand() - 0.5f) * 22.0f);
                    int oz = cz + int((frand() - 0.5f) * 22.0f);
                    placeBoulder(ox, oz, 2.0f + 3.5f * frand());
                }
            }
        }
    }
}


// -- Bushes / shrubs -----------------------------------------------------
// Real shrubs have a short woody base with thin branches reaching up into
// a compact leafy crown. We build each bush from: a 2-3 voxel trunk, 3-5
// short branches radiating outward, leaf puffs at branch tips, and a
// unifying small leaf puff at the top so it reads as one shape.
void genBushes(World& w, GenContext& ctx, const GenBounds& b,
               const SurfaceSnapshot& snap, ChunkStageBuffer* buf) {
    StageWriter gw(w, b, ctx, buf);
    std::mt19937 rng;   // re-seeded per lattice cell (featureSeed)
    auto frand = [&]() { return frand01(rng); };

    auto trySet = [&](int x, int y, int z, uint8_t mat, bool overwriteLeaves) {
        if (y < 0 || y >= GRID_Y) return;   // XZ unbounded
        uint8_t cur = gw.get(x, y, z);
        if (cur == MAT_WOOD) return;
        if (cur == MAT_LEAVES && !overwriteLeaves) return;
        if (cur != 0 && cur != MAT_LEAVES) return;  // don't disturb stone/dirt/grass
        gw.set(x, y, z, mat);
    };

    auto leafPuff = [&](glm::vec3 c, int r) {
        int r2 = r * r;
        for (int dz = -r; dz <= r; ++dz)
        for (int dy = -r; dy <= r; ++dy)
        for (int dx = -r; dx <= r; ++dx) {
            int dd = dx*dx + dy*dy + dz*dz;
            if (dd > r2) continue;
            // Erode the outer rind so it's not a perfect sphere.
            if (dd > r2 - 2 && (rng() & 3) == 0) continue;
            int x = int(std::round(c.x)) + dx;
            int y = int(std::round(c.y)) + dy;
            int z = int(std::round(c.z)) + dz;
            trySet(x, y, z, MAT_LEAVES, false);
        }
    };

    auto placeBush = [&](int cx, int cz) {
        int h = heightAt(cx, cz);
        if (h < 10 || h >= GRID_Y - 30) return;
        if (snap.get(cx, h - 1, cz) != MAT_GRASS) return;   // pre-stage gate

        // Per-bush scale (squared bias toward small). Most are knee-high
        // clumps; some grow into chest-high shrubs; rare ones approach
        // small saplings.
        float sizeRoll = frand();
        float scale = 0.5f + 2.2f * sizeRoll * sizeRoll;   // 0.5 .. 2.7

        int trunkH = std::max(1, int(std::round(scale * (1.5f + frand()))));
        for (int i = 0; i < trunkH; ++i) {
            trySet(cx, h + i, cz, MAT_WOOD, true);
        }
        glm::vec3 top(cx, h + trunkH, cz);

        int branches = std::max(2, int(std::round(scale * (3.0f + frand() * 2.0f))));
        for (int b = 0; b < branches; ++b) {
            float ang = frand() * 6.2831853f;
            float reach = scale * (1.4f + 1.4f * frand());
            glm::vec3 dir(std::cos(ang) * 0.85f,
                          0.35f + 0.25f * frand(),
                          std::sin(ang) * 0.85f);
            dir = glm::normalize(dir);

            glm::vec3 tip = top;
            int steps = std::max(1, int(std::ceil(reach)));
            for (int i = 1; i <= steps; ++i) {
                tip = top + dir * float(i);
                int x = int(std::round(tip.x));
                int y = int(std::round(tip.y));
                int z = int(std::round(tip.z));
                trySet(x, y, z, MAT_WOOD, true);
            }
            int lr = std::max(1, int(std::round(scale * (0.9f + 0.6f * frand()))));
            leafPuff(tip, lr);
        }

        // Unifying leaf puff just above the trunk top so the bush reads
        // as a single mass. Scales with the bush so big shrubs feel
        // proportionally crowned.
        int puffR = std::max(1, int(std::round(scale * 1.8f)));
        leafPuff(top + glm::vec3(0.0f, 0.6f * scale, 0.0f), puffR);
        if (gw.anchorIn(cx, cz)) ++gw.stats().bushes;
    };

    // Desert creosote — built with the SAME care as the trees' L-system, scaled
    // to a small MULTI-STEMMED shrub. Real creosote sends many thin woody stems
    // up from the base, each forking and spreading into a rounded vase, with
    // small foliage clusters at the MANY branch tips. Fullness comes from the
    // dense BRANCHING (not a procedural volume / blob); airiness comes from
    // partial-fill foliage bulbs and the gaps between branches. Foliage is the
    // muted MAT_SHRUB_SAGE; plants ONLY on desert surface; rooted at y=h.
    auto urand = [&]() { return frand() * 2.0f - 1.0f; };

    // Airy sage clump — the tree's `bulb` trick (skip a fraction so it reads
    // open and the woody stems show through), but muted sage and sparser.
    auto sageBulb = [&](glm::vec3 p, int r) {
        int r2 = r * r;
        int bx = int(std::round(p.x)), by = int(std::round(p.y)), bz = int(std::round(p.z));
        for (int dz = -r; dz <= r; ++dz)
        for (int dy = -r; dy <= r; ++dy)
        for (int dx = -r; dx <= r; ++dx) {
            if (dx*dx + dy*dy + dz*dz > r2) continue;
            if ((rng() & 1023) < 430) continue;          // ~42% skip -> airy
            trySet(bx + dx, by + dy, bz + dz, MAT_SHRUB_SAGE, false);
        }
    };
    // Thin tapered limb between two points (the tree's drawSeg, slimmer). `mat`
    // defaults to ashy desert wood (creosote/ocotillo stems); the palm passes
    // MAT_WOOD for its trunk and MAT_LEAVES for its green frond ribs.
    auto cseg = [&](glm::vec3 a, glm::vec3 c, float r0, float r1, uint8_t mat = MAT_DESERT_WOOD) {
        float L = glm::length(c - a);
        int steps = std::max(2, int(L * 1.5f));
        for (int i = 0; i <= steps; ++i) {
            float t = float(i) / float(steps);
            glm::vec3 p = glm::mix(a, c, t);
            float r = glm::mix(r0, r1, t);
            int ir = std::max(1, int(std::round(r)));
            float r2 = r * r;
            for (int dz = -ir; dz <= ir; ++dz)
            for (int dy = -ir; dy <= ir; ++dy)
            for (int dx = -ir; dx <= ir; ++dx) {
                if (dx*dx + dy*dy + dz*dz > r2) continue;
                trySet(int(std::round(p.x)) + dx, int(std::round(p.y)) + dy,
                       int(std::round(p.z)) + dz, mat, true);
            }
        }
    };
    // Recursive branch: curving woody segment that either forks (spreading
    // up-and-out) or, at the tip, caps with a small sage lobe. Many tips ->
    // a full, airy, organically-branched crown.
    std::function<void(glm::vec3, glm::vec3, float, float, int)> cgrow =
        [&](glm::vec3 origin, glm::vec3 dir, float length, float thick, int depth) {
            if (depth <= 0) return;
            glm::vec3 sideA = glm::normalize(glm::cross(dir, glm::vec3(0, 1, 0.001f)));
            glm::vec3 mid = origin + dir * (length * 0.5f) + sideA * (urand() * length * 0.10f);
            glm::vec3 end = origin + dir * length
                          + sideA * (urand() * length * 0.12f)
                          + glm::vec3(0.0f, 0.0f, 0.0f);
            cseg(origin, mid, thick,         thick * 0.80f);
            cseg(mid,    end, thick * 0.80f, thick * 0.55f);
            if (depth == 1) {                                 // tip: small sage lobe
                sageBulb(end, 2);
                return;
            }
            int n = 2 + int(rng() % 2);                       // fork into 2-3
            float yaw0 = frand() * 6.2831853f;
            for (int c = 0; c < n; ++c) {
                float yaw  = yaw0 + (float(c) / float(n)) * 6.2831853f + urand() * 0.4f;
                float out  = 0.45f + 0.35f * frand();         // how much it splays from parent
                glm::vec3 splay = glm::normalize(glm::vec3(std::cos(yaw), 0.7f, std::sin(yaw)));
                glm::vec3 nd = glm::normalize(glm::mix(dir, splay, out));
                nd.y = std::max(nd.y, 0.2f);
                nd = glm::normalize(nd);
                cgrow(end, nd, length * (0.60f + 0.18f * frand()), thick * 0.62f, depth - 1);
            }
            if ((rng() & 1) == 0) sageBulb(end, 1);           // a little foliage at the fork too
        };

    auto placeCreosote = [&](int cx, int cz) {
        int h = heightAt(cx, cz);
        if (h < 10 || h >= GRID_Y - 30) return;
        uint8_t surf = snap.get(cx, h - 1, cz);
        if (surf != MAT_DESERT_PAVEMENT && surf != MAT_SAND_STATIC) return;  // skip rock/cave/wash
        { OasisHit oa = oasisAt(cx, cz); if (oa.any && h <= oa.waterY) return; }  // not in/under the oasis pool

        float sizeRoll = frand();
        float scale = 0.85f + 0.65f * sizeRoll;               // 0.85..1.5
        glm::vec3 base(float(cx) + 0.5f, float(h), float(cz) + 0.5f);
        trySet(cx, h, cz, MAT_DESERT_WOOD, true);             // weld the very base to the ground

        // MULTI-STEM from the base (creosote's signature): several thin stems
        // splay up-and-out, each forking once into tips with sage lobes.
        int stems = 4 + int(rng() % 4);                       // 4..7
        float yaw0 = frand() * 6.2831853f;
        for (int s = 0; s < stems; ++s) {
            float yaw  = yaw0 + (float(s) / float(stems)) * 6.2831853f + urand() * 0.5f;
            float lean = 0.32f + 0.30f * frand();             // outward splay -> vase
            glm::vec3 dir = glm::normalize(glm::vec3(std::cos(yaw) * lean, 1.0f,
                                                     std::sin(yaw) * lean));
            float len   = (4.5f + 2.5f * frand()) * scale;
            float thick = 0.9f + 0.3f * frand();              // thin
            cgrow(base, dir, len, thick, 2);                  // stem -> fork -> tip lobes
        }
        if (gw.anchorIn(cx, cz)) ++gw.stats().bushes;
    };

    // --- Saguaro (hero cactus) -------------------------------------------
    // A SOLID, non-tapering MAT_CACTUS_FLESH column (the shader ribs it) with a
    // rounded dome top and 0-4 candelabra arms: each arm goes OUT a short stub,
    // through a rounded gooseneck bend, then UP parallel to the trunk. Opaque
    // (unlike the airy foliage plants) so the silhouette + ribs read crisply;
    // only the bloom crown is airy. Rooted at y=h by the first flesh sphere.
    auto fleshSeg = [&](glm::vec3 a, glm::vec3 c, float r) {
        float L = glm::length(c - a);
        int steps = std::max(2, int(L * 1.5f));
        int ir = std::max(1, int(std::round(r)));
        float r2 = r * r;
        for (int i = 0; i <= steps; ++i) {
            glm::vec3 p = glm::mix(a, c, float(i) / float(steps));
            int px = int(std::round(p.x)), py = int(std::round(p.y)), pz = int(std::round(p.z));
            for (int dz = -ir; dz <= ir; ++dz)
            for (int dy = -ir; dy <= ir; ++dy)
            for (int dx = -ir; dx <= ir; ++dx)
                if (dx*dx + dy*dy + dz*dz <= r2)
                    trySet(px + dx, py + dy, pz + dz, MAT_CACTUS_FLESH, false);
        }
    };
    auto fleshBall = [&](glm::vec3 c, float r, bool domeOnly) {
        int ir = std::max(1, int(std::round(r)));
        float r2 = r * r;
        int cxv = int(std::round(c.x)), cyv = int(std::round(c.y)), czv = int(std::round(c.z));
        for (int dz = -ir; dz <= ir; ++dz)
        for (int dy = (domeOnly ? 0 : -ir); dy <= ir; ++dy)
        for (int dx = -ir; dx <= ir; ++dx)
            if (dx*dx + dy*dy + dz*dz <= r2)
                trySet(cxv + dx, cyv + dy, czv + dz, MAT_CACTUS_FLESH, false);
    };
    auto bloomCrown = [&](glm::vec3 c, float r) {
        int n = 6 + int(rng() % 5);                          // 6-10 scattered blossoms
        for (int i = 0; i < n; ++i) {
            float a = frand() * 6.2831853f;
            int bx = int(std::round(c.x + std::cos(a) * r * 0.8f));
            int by = int(std::round(c.y + 0.6f));
            int bz = int(std::round(c.z + std::sin(a) * r * 0.8f));
            for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
                if (dx*dx + dy*dy + dz*dz <= 1)
                    trySet(bx + dx, by + dy, bz + dz, MAT_CACTUS_BLOOM, false);
        }
    };
    auto placeSaguaro = [&](int cx, int cz) {
        int h = heightAt(cx, cz);
        if (h < 10 || h >= GRID_Y - 60) return;
        uint8_t surf = snap.get(cx, h - 1, cz);
        if (surf != MAT_DESERT_PAVEMENT && surf != MAT_SAND_STATIC) return;
        { OasisHit oa = oasisAt(cx, cz); if (oa.any && h <= oa.waterY) return; }  // not in/under the oasis pool
        glm::vec3 base(float(cx) + 0.5f, float(h), float(cz) + 0.5f);

        // Size biased toward the ARMED range so most aren't bare poles; rare
        // giants are landmarks, smaller juveniles (some armless) are the minority.
        float roll = frand();
        int   H = (roll < 0.65f) ? (26 + int(frand() * 12.0f))   // 26-38 common (armed)
                : (roll < 0.88f) ? (20 + int(frand() *  8.0f))   // 20-28 smaller juveniles
                                 : (38 + int(frand() * 14.0f));  // 38-52 giants
        float R = (H > 36) ? 3.0f : 2.0f;
        glm::vec3 top = base + glm::vec3(0.0f, float(H), 0.0f);
        fleshSeg(base, top, R);                              // solid vertical column, rooted at y=h
        fleshBall(top, R + 1.0f, true);                      // rounded dome crown
        if (H < 26) fleshBall(base + glm::vec3(0.0f, 2.0f, 0.0f), R + 1.0f, false);  // juvenile bulge

        bool bloom = smoothNoise3(float(cx) * 0.02f + 3.0f, 11.0f,
                                  float(cz) * 0.02f + 5.0f) > 0.62f;   // regional bloom patches

        // Arm count: most non-juveniles get >= 1 arm (so fewer pole-like ones);
        // only true giants reach 3-4. Irregular yaw so giants aren't radial.
        int maxArms = (H >= 42) ? 4 : (H >= 32 ? 3 : (H >= 24 ? 2 : 1));
        int arms;
        if (H < 24)      arms = (frand() < 0.35f) ? 1 : 0;        // juveniles usually bare
        else if (H < 30) arms = 1 + int(frand() * 2.0f);          // 1-2
        else             arms = 1 + int(frand() * float(maxArms)); // 1..maxArms
        float crownY = base.y + float(H);
        float yaw0 = frand() * 6.2831853f;
        for (int a = 0; a < arms; ++a) {
            float yaw = yaw0 + float(a) * (6.2831853f / float(std::max(arms, 1))) + urand() * 0.9f;
            float fy  = 0.45f + 0.30f * frand();             // attach 0.45-0.75 up the trunk
            glm::vec3 outDir(std::cos(yaw), 0.0f, std::sin(yaw));
            glm::vec3 attach = base + glm::vec3(0.0f, float(H) * fy, 0.0f) + outDir * R;
            // A real saguaro arm: sweeps OUT and DOWN into a U, then curves UP,
            // held well OUT from the trunk (clear gap, not a parallel trident).
            // Quadratic Bezier attach -> via(out+down) -> tip(out+up).
            float outDist = 6.0f + 4.0f * frand();           // horizontal hold-out (the separation)
            float dip     = 1.5f + 1.5f * frand();           // dip below attach at the U bottom
            float topY    = std::max(attach.y + 6.0f, crownY - 2.0f + frand() * 6.0f);  // rise to near crown
            // The rising part stays held OUT at ~full distance so there's a clear
            // gap between arm and trunk (not a tight parallel trident).
            glm::vec3 via = attach + outDir * outDist          + glm::vec3(0.0f, -dip,            0.0f);
            glm::vec3 tip = attach + outDir * (outDist * 0.95f) + glm::vec3(0.0f, topY - attach.y, 0.0f);
            int steps = std::max(12, int((outDist + (topY - attach.y)) * 1.3f));
            for (int sN = 0; sN <= steps; ++sN) {
                float t = float(sN) / float(steps);
                glm::vec3 p = (1.0f - t) * (1.0f - t) * attach
                            + 2.0f * (1.0f - t) * t   * via
                            + t * t                   * tip;   // smooth gooseneck curve
                fleshBall(p, R, false);
            }
            fleshBall(tip, R, true);                          // domed arm tip
            if (bloom) bloomCrown(tip, R);
        }
        if (bloom) bloomCrown(top, R + 1.0f);
        if (gw.anchorIn(cx, cz)) ++gw.stats().bushes;
    };

    // --- Ocotillo ---------------------------------------------------------
    // NOT a cactus: a bottom-pinched SPRAY of many tall, thin, unbranched grey
    // canes springing from one basal caudex, rising vertically then flaring out
    // near the TOP, each capped by a vivid red bloom flame. Tall + narrow (a
    // wine-glass), hollow in the middle, see-through. The red crown-ring is the
    // key species cue, so the bloom is hardened to always read.
    auto bloomTuft = [&](glm::vec3 p) {
        int bx = int(std::round(p.x)), by = int(std::round(p.y)), bz = int(std::round(p.z));
        trySet(bx, by,     bz, MAT_CACTUS_BLOOM, false);     // guaranteed core
        trySet(bx, by + 1, bz, MAT_CACTUS_BLOOM, false);     // guaranteed spire (always reads)
        for (int dy = 0; dy <= 2; ++dy)
        for (int dz = -1; dz <= 1; ++dz)
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dz == 0) continue;
            if ((rng() & 1023) < 341) continue;             // ~33% skip -> airy flame
            if (dy >= 1 && (rng() & 1)) continue;           // thin the upper dabs
            trySet(bx + dx, by + dy, bz + dz, MAT_CACTUS_BLOOM, false);
        }
    };
    auto placeOcotillo = [&](int cx, int cz) {
        int h = heightAt(cx, cz);
        if (h < 10 || h >= GRID_Y - 40) return;
        uint8_t surf = snap.get(cx, h - 1, cz);
        if (surf != MAT_DESERT_PAVEMENT && surf != MAT_SAND_STATIC) return;
        { OasisHit oa = oasisAt(cx, cz); if (oa.any && h <= oa.waterY) return; }  // not in/under the oasis pool
        float scale = 0.85f + 0.55f * frand();
        bool  bloom = frand() < 0.22f;                       // most are bare grey; ~1 in 5 flames red
        glm::vec3 baseP(float(cx) + 0.5f, float(h), float(cz) + 0.5f);
        // Compact tapering woody caudex (3x3 -> plus -> point over 3 voxels): a
        // small swollen base that grounds the cane bundle without the wide flat
        // cross a big disc made.
        for (int dy = 0; dy <= 2; ++dy) {
            int lim = (dy == 0) ? 2 : (dy == 1 ? 1 : 0);     // dd<=lim: 3x3 -> plus -> single
            for (int dz = -1; dz <= 1; ++dz)
            for (int dx = -1; dx <= 1; ++dx)
                if (dx*dx + dz*dz <= lim)
                    trySet(cx + dx, h + dy, cz + dz, MAT_DESERT_WOOD, true);
        }
        glm::vec3 crown(baseP.x, float(h) + 2.0f, baseP.z);
        int canes = 10 + int(rng() % 7);                     // 10..16
        float yaw0 = frand() * 6.2831853f;
        for (int c = 0; c < canes; ++c) {
            float yaw  = yaw0 + (float(c) / float(canes)) * 6.2831853f + urand() * 0.45f;
            float lean = 0.18f + 0.22f * frand();            // STEEP (tall narrow vase), not a flat fan
            float len  = (16.0f + 8.0f * frand()) * scale;
            len *= (1.0f + (0.40f - lean));                  // steeper canes taller -> pointed vase
            glm::vec3 dir = glm::normalize(glm::vec3(std::cos(yaw) * lean, 1.0f, std::sin(yaw) * lean));
            glm::vec3 tip  = crown + dir * len;
            // Quadratic bezier with a STRAIGHT-UP control point: the lower cane
            // rises vertical, only the top flares outward (the real cane arc).
            glm::vec3 ctrl = crown + glm::vec3(0.0f, len * 0.55f, 0.0f);
            int steps = std::max(8, int(len * 1.3f));
            glm::vec3 prev = crown;
            for (int sN = 1; sN <= steps; ++sN) {
                float t = float(sN) / float(steps);
                glm::vec3 p = (1.0f-t)*(1.0f-t)*crown + 2.0f*(1.0f-t)*t*ctrl + t*t*tip;
                cseg(prev, p, 0.9f, 0.9f);                   // thin 1-voxel grey whip
                prev = p;
            }
            if (bloom) bloomTuft(prev);                      // red flame caps the canes (blooming plants only)
        }
        if (gw.anchorIn(cx, cz)) ++gw.stats().bushes;
    };

    // --- Fan palm (Washingtonia) — the OASIS tree -------------------------
    // A bare LEANING trunk (MAT_WOOD only) topped by a radial STAR of distinct
    // drooping GREEN fronds (a CONTINUOUS MAT_LEAVES rib per blade so each reads,
    // tips falling below the hub = a fountain not a wheel), with a dense TAN
    // dead-frond petticoat + hanging beard in MAT_DESERT_WOOD just below the
    // crown. leanYaw biases the lean toward the water. (Placed by the oasis loop.)
    auto placePalm = [&](int cx, int cz, float leanYaw) {
        int h = heightAt(cx, cz);
        if (h < 10 || h >= GRID_Y - 100) return;
        float lean = 0.08f + 0.06f * frand();                // 0.08..0.14 off vertical (gentler on a tall trunk)
        int   H    = 46 + int(frand() * 30.0f);              // trunk 46..76 vox — as tall/taller than the forest trees
        glm::vec3 base(float(cx) + 0.5f, float(h), float(cz) + 0.5f);
        glm::vec3 crown = base + glm::vec3(std::cos(leanYaw) * lean * float(H), float(H), std::sin(leanYaw) * lean * float(H));
        glm::vec3 tctrl = base + glm::vec3(std::cos(leanYaw) * lean * float(H) * 0.3f, float(H) * 0.6f, std::sin(leanYaw) * lean * float(H) * 0.3f);
        // bare leaning trunk (wood ONLY), gentle bezier, slightly thicker base so the tall trunk doesn't read as spindly
        trySet(cx, h, cz, MAT_WOOD, true);                   // weld base to ground
        glm::vec3 prev = base; int ts = std::max(8, int(float(H) * 1.3f));
        for (int i = 1; i <= ts; ++i) {
            float t = float(i) / float(ts);
            glm::vec3 p = (1-t)*(1-t)*base + 2.0f*(1-t)*t*tctrl + t*t*crown;
            cseg(prev, p, 2.0f, 1.2f, MAT_WOOD);
            prev = p;
        }
        // solid apex hub (wood ball) so the crown isn't a hollow pinch
        for (int dz = -3; dz <= 3; ++dz)
        for (int dy = -3; dy <= 3; ++dy)
        for (int dx = -3; dx <= 3; ++dx)
            if (dx*dx + dy*dy + dz*dz <= 7)
                trySet(int(std::round(crown.x))+dx, int(std::round(crown.y))+dy, int(std::round(crown.z))+dz, MAT_WOOD, true);
        // GREEN fronds: distinct radial blades, droop dominant (fountain) — scaled
        // up with the taller trunk so the crown stays in proportion.
        int NF = 12 + int(rng() % 5);                        // 12..16
        float fyaw0 = frand() * 6.2831853f;
        for (int fr = 0; fr < NF; ++fr) {
            float fyaw = fyaw0 + float(fr) * (6.2831853f / float(NF)) + urand() * 0.10f;  // ~even spacing, gaps survive
            float fLen = 19.0f + frand() * 11.0f;            // 19..30
            glm::vec3 out(std::cos(fyaw), 0.0f, std::sin(fyaw));
            glm::vec3 ctrl = crown + out * (fLen * 0.45f) + glm::vec3(0.0f, fLen * 0.22f, 0.0f);          // rise+out
            glm::vec3 tip  = crown + out * (fLen * 0.85f) + glm::vec3(0.0f, -(0.35f + 0.20f * frand()) * fLen, 0.0f);  // tip below hub -> droop
            glm::vec3 pv = crown; int fs = std::max(8, int(fLen * 1.4f));
            for (int s = 1; s <= fs; ++s) {
                float t = float(s) / float(fs);
                glm::vec3 p = (1-t)*(1-t)*crown + 2.0f*(1-t)*t*ctrl + t*t*tip;
                cseg(pv, p, 1.0f, 1.0f, MAT_LEAVES);         // continuous GREEN rib, full length
                if (t > 0.30f && t < 0.75f && (rng() & 1) == 0) {   // airy leaflets flank the rib midspan
                    glm::vec3 sd = glm::normalize(glm::cross(p - pv, glm::vec3(0.0f, 1.0f, 0.001f)));
                    glm::vec3 a = p + sd, bb = p - sd;
                    trySet(int(std::round(a.x)), int(std::round(a.y)), int(std::round(a.z)), MAT_LEAVES, false);
                    trySet(int(std::round(bb.x)), int(std::round(bb.y)), int(std::round(bb.z)), MAT_LEAVES, false);
                }
                pv = p;
            }
        }
        // TAN petticoat: dense dead fronds drooping just BELOW the green crown
        int NP = 18 + int(rng() % 7);                        // 18..24
        for (int pf = 0; pf < NP; ++pf) {
            float pyaw = frand() * 6.2831853f;
            glm::vec3 pout(std::cos(pyaw), 0.0f, std::sin(pyaw));
            float pl = 5.0f + frand() * 4.0f;
            glm::vec3 pstart = crown + glm::vec3(0.0f, -2.0f, 0.0f) + pout * 1.5f;
            glm::vec3 pend   = pstart + pout * (pl * 0.5f) + glm::vec3(0.0f, -pl, 0.0f);
            cseg(pstart, pend, 1.5f, 0.9f, MAT_DESERT_WOOD);
        }
        // BEARD: 1-2 hanging strands below the skirt
        int NB = 1 + int(rng() % 2);
        for (int bd = 0; bd < NB; ++bd) {
            float bl = 6.0f + frand() * 6.0f;
            glm::vec3 bs = crown + glm::vec3((frand()-0.5f)*2.0f, -4.0f, (frand()-0.5f)*2.0f);
            cseg(bs, bs + glm::vec3(0.0f, -bl, 0.0f), 1.0f, 0.8f, MAT_DESERT_WOOD);
        }
        if (gw.anchorIn(cx, cz)) ++gw.stats().bushes;
    };

    // Lattice-seeded placement (see genBoulders). 1100 bushes over the
    // world in expectation. Footprint contract: bush radius <= ~10.
    constexpr int   CELL     = 128;
    constexpr float PER_CELL = 1100.0f / float((GRID_XZ / CELL) * (GRID_XZ / CELL));
    int c0x = floorDiv(b.x0 - MARGIN_BUSH, CELL);
    int c1x = floorDiv(b.x1 - 1 + MARGIN_BUSH, CELL);
    int c0z = floorDiv(b.z0 - MARGIN_BUSH, CELL);
    int c1z = floorDiv(b.z1 - 1 + MARGIN_BUSH, CELL);
    for (int cellZ = c0z; cellZ <= c1z; ++cellZ)
    for (int cellX = c0x; cellX <= c1x; ++cellX) {
        rng.seed(featureSeed(ctx.seed, SEED_BUSH, cellX, cellZ));
        int n = int(PER_CELL);
        if (frand() < PER_CELL - float(int(PER_CELL))) ++n;
        for (int i = 0; i < n; ++i) {
            int cx = cellX * CELL + int(frand() * CELL);
            int cz = cellZ * CELL + int(frand() * CELL);
            placeBush(cx, cz);
        }
    }

    // Desert creosote MATRIX: a jittered FINE grid (one candidate per ~14-voxel
    // cell) gives near-uniform, self-avoiding spacing — real creosote holds
    // itself apart competing for water, and the BARE INTERSPACE is the desert's
    // signature read (the ecological inverse of the meadow's clustering). The
    // jittered grid is the cheap, replay-deterministic Poisson-disk: no
    // cross-cell state, spacing never breaks at lattice lines. Probability is
    // weighted by the desert biome weight so the matrix thins out smoothly
    // toward the highland border instead of stopping at a line.
    constexpr int SHRUB_CELL = 20;                           // wider cells -> more bare ground
    int s0x = floorDiv(b.x0 - MARGIN_BUSH, SHRUB_CELL);
    int s1x = floorDiv(b.x1 - 1 + MARGIN_BUSH, SHRUB_CELL);
    int s0z = floorDiv(b.z0 - MARGIN_BUSH, SHRUB_CELL);
    int s1z = floorDiv(b.z1 - 1 + MARGIN_BUSH, SHRUB_CELL);
    for (int gz = s0z; gz <= s1z; ++gz)
    for (int gx = s0x; gx <= s1x; ++gx) {
        rng.seed(featureSeed(ctx.seed, SEED_DESERT_SHRUB, gx, gz));
        int cx = gx * SHRUB_CELL + 3 + int(frand() * float(SHRUB_CELL - 6));
        int cz = gz * SHRUB_CELL + 3 + int(frand() * float(SHRUB_CELL - 6));
        float dW = biomeDesertW(cx, cz);
        if (dW < 0.05f) continue;                            // cheap reject outside desert
        // Low per-cell occupancy so the BARE INTERSPACE dominates (the desert
        // read) — real creosote holds itself well apart competing for water.
        // ~6-22% occupancy on the 20-vox grid -> mean spacing ~4-8 m, with
        // denser washes (noise high) and barer flats. Multiplied by desert
        // weight so the matrix thins out smoothly toward the highland border.
        float dens = 0.04f + 0.10f * smoothNoise(cx * 0.012f + 8.0f, cz * 0.012f + 3.0f);
        if (frand() > dW * dens) continue;                   // most cells stay empty (bare desert)
        placeCreosote(cx, cz);
    }

    // Saguaro stand: rare HERO landmarks on a sparse jittered lattice, only in
    // firm desert interior (dW high). Coarse cell -> wide spacing.
    constexpr int SAG_CELL = 96;
    int g0x = floorDiv(b.x0 - MARGIN_BUSH, SAG_CELL);
    int g1x = floorDiv(b.x1 - 1 + MARGIN_BUSH, SAG_CELL);
    int g0z = floorDiv(b.z0 - MARGIN_BUSH, SAG_CELL);
    int g1z = floorDiv(b.z1 - 1 + MARGIN_BUSH, SAG_CELL);
    for (int gz = g0z; gz <= g1z; ++gz)
    for (int gx = g0x; gx <= g1x; ++gx) {
        rng.seed(featureSeed(ctx.seed, SEED_SAGUARO, gx, gz));
        int cx = gx * SAG_CELL + 8 + int(frand() * float(SAG_CELL - 16));
        int cz = gz * SAG_CELL + 8 + int(frand() * float(SAG_CELL - 16));
        float dW = biomeDesertW(cx, cz);
        if (dW < 0.35f) continue;                            // interior only
        if (frand() > dW * 0.45f) continue;                  // sparse landmarks
        placeSaguaro(cx, cz);
    }

    // Ocotillo: a sparse accent (own seed + wider cell than the creosote matrix).
    constexpr int OCO_CELL = 34;
    int o0x = floorDiv(b.x0 - MARGIN_BUSH, OCO_CELL);
    int o1x = floorDiv(b.x1 - 1 + MARGIN_BUSH, OCO_CELL);
    int o0z = floorDiv(b.z0 - MARGIN_BUSH, OCO_CELL);
    int o1z = floorDiv(b.z1 - 1 + MARGIN_BUSH, OCO_CELL);
    for (int gz = o0z; gz <= o1z; ++gz)
    for (int gx = o0x; gx <= o1x; ++gx) {
        rng.seed(featureSeed(ctx.seed, SEED_OCOTILLO, gx, gz));
        int cx = gx * OCO_CELL + 4 + int(frand() * float(OCO_CELL - 8));
        int cz = gz * OCO_CELL + 4 + int(frand() * float(OCO_CELL - 8));
        float dW = biomeDesertW(cx, cz);
        if (dW < 0.05f) continue;
        float dens = 0.04f + 0.10f * smoothNoise(cx * 0.01f + 21.0f, cz * 0.01f + 5.0f);
        if (frand() > dW * dens) continue;                   // rare accent
        placeOcotillo(cx, cz);
    }

    // Oasis palms: a tight cluster of fan palms on the DAMP BANK around each
    // oasis pool. Own seed + big margin. NOTE: genBushes runs BEFORE genOasis,
    // so the water doesn't exist yet — gate on heightAt vs the water LEVEL
    // (dry bank), never on MAT_WATER voxels.
    {
        int o0x = floorDiv(b.x0 - MARGIN_OASIS_PALM, OASIS_CELL);
        int o1x = floorDiv(b.x1 - 1 + MARGIN_OASIS_PALM, OASIS_CELL);
        int o0z = floorDiv(b.z0 - MARGIN_OASIS_PALM, OASIS_CELL);
        int o1z = floorDiv(b.z1 - 1 + MARGIN_OASIS_PALM, OASIS_CELL);
        for (int gz = o0z; gz <= o1z; ++gz)
        for (int gx = o0x; gx <= o1x; ++gx) {
            if (!oasisHost(gx, gz)) continue;
            OasisSite s = oasisSite(gx, gz);                  // same source of truth as genOasis/heightAt
            if (!s.pass) continue;
            float ctrX = s.ctrX, ctrZ = s.ctrZ, R = s.R;
            int waterY = s.waterY;                            // rim-min level (bit-identical to the pool)
            rng.seed(featureSeed(ctx.seed, SEED_OASIS_PALM, gx, gz));
            int nPalm = 6 + int(rng() % 4);                  // 6..9 palms ringing the pool
            for (int i = 0; i < nPalm; ++i) {
                float ang = frand() * 6.2831853f;
                float uu  = 0.82f + 0.28f * frand();         // damp annulus (gate keeps only dry-bank ones)
                int px = int(ctrX + std::cos(ang) * uu * R);
                int pz = int(ctrZ + std::sin(ang) * uu * R);
                int ph = heightAt(px, pz);
                if (ph < waterY + 1) continue;               // must be dry bank, not in/under the pool
                uint8_t surf = snap.get(px, ph - 1, pz);
                if (surf != MAT_DESERT_PAVEMENT && surf != MAT_SAND_STATIC) continue;
                float leanYaw = std::atan2(ctrZ - float(pz), ctrX - float(px));  // lean over the water
                placePalm(px, pz, leanYaw);
            }
        }
    }

}


// Recursive L-system trees: curving woody segments fork into 2-3 children;
// leaf clusters spawn only at branch tips. Produces organic trunks and
// limbs rather than a sphere on a stick.
// The forest-tree GRAMMAR: everything one tree writes, generated from its
// TreeSite + the rng stream deriveTreeSite left at draw 7. Writer-agnostic so
// it has exactly two callers: genTrees (StageWriter -> the world) and
// rederiveTreeAt below (scratch map -> exact per-tree voxel ownership — the
// "ask the generator" identity forestry uses). RNG DRAWS ARE THE CONTRACT:
// both callers consume identically because no draw is gated on a world read
// (bulb's occupancy check comes AFTER its skip roll) — reordering draws
// relayouts every forest (GEN_VERSION bump).
template <class SetFn, class GetFn>
static void growForestTree(std::mt19937& rng, const TreeSite& site,
                           SetFn&& setV, GetFn&& getV) {
    auto frand = [&]() { return frand01(rng); };
    auto urand = [&]() { return frand11(rng); };

    auto drawSeg = [&](glm::vec3 a, glm::vec3 b, float r0, float r1) {
        float L = glm::length(b - a);
        int steps = std::max(2, int(L * 1.4f));
        for (int i = 0; i <= steps; ++i) {
            float t = float(i) / steps;
            glm::vec3 p = glm::mix(a, b, t);
            float r = glm::mix(r0, r1, t);
            int ir = std::max(1, int(std::round(r)));
            float r2 = r * r;
            for (int dz = -ir; dz <= ir; ++dz)
            for (int dy = -ir; dy <= ir; ++dy)
            for (int dx = -ir; dx <= ir; ++dx) {
                if (dx*dx + dy*dy + dz*dz > r2) continue;
                int x = int(std::round(p.x)) + dx;
                int y = int(std::round(p.y)) + dy;
                int z = int(std::round(p.z)) + dz;
                if (y < 0 || y >= GRID_Y) continue;   // XZ unbounded
                setV(x, y, z, MAT_WOOD);
            }
        }
    };

    // Two-level canopy: ~5 macro shells (invisible spherical surfaces)
    // tiled with small popcorn bulbs. Each bulb is ~65% filled. Trunk
    // and branches show through gaps between bulbs, matching the airy
    // look of the reference tree.

    auto bulb = [&](glm::vec3 p, int r) {
        int r2 = r * r;
        int cx = int(std::round(p.x));
        int cy = int(std::round(p.y));
        int cz = int(std::round(p.z));
        for (int dz = -r; dz <= r; ++dz)
        for (int dy = -r; dy <= r; ++dy)
        for (int dx = -r; dx <= r; ++dx) {
            int dd = dx*dx + dy*dy + dz*dz;
            if (dd > r2) continue;
            if ((rng() & 1023) < 358) continue;   // ~35% skip = 65% fill
            int x = cx + dx, y = cy + dy, z = cz + dz;
            if (y < 0 || y >= GRID_Y - 1) continue;   // XZ unbounded
            if (getV(x, y, z) != 0) continue;
            setV(x, y, z, MAT_LEAVES);
        }
    };

    auto shellCanopy = [&](glm::vec3 center, float shellR) {
        int surface = 22 + int(rng() % 12);     // 22-33 surface bulbs
        for (int i = 0; i < surface; ++i) {
            float u = frand() * 2.0f - 1.0f;
            float t = frand() * 6.2831853f;
            float s = std::sqrt(std::max(0.0f, 1.0f - u*u));
            glm::vec3 dir(s * std::cos(t), u, s * std::sin(t));
            float r = shellR * (0.75f + 0.35f * frand());
            glm::vec3 pos = center + dir * r;
            bulb(pos, 3 + int(rng() % 3));      // 3-5 voxel bulbs
        }
        // Interior fillers so the trunk intersection isn't visibly hollow.
        int interior = 3 + int(rng() % 3);
        for (int i = 0; i < interior; ++i) {
            float u = frand() * 2.0f - 1.0f;
            float t = frand() * 6.2831853f;
            float s = std::sqrt(std::max(0.0f, 1.0f - u*u));
            glm::vec3 dir(s * std::cos(t), u, s * std::sin(t));
            glm::vec3 pos = center + dir * (shellR * 0.4f * frand());
            bulb(pos, 3 + int(rng() % 2));
        }
    };

    // `scale` is sampled once per tree and carried through the recursion
    // so sub-branches and canopy shells all scale together.
    std::function<void(glm::vec3, glm::vec3, float, float, int, float)> grow =
        [&](glm::vec3 origin, glm::vec3 dir, float length, float thick,
            int depth, float scale) {
            if (depth <= 0) return;

            glm::vec3 end = origin + dir * length;
            glm::vec3 sideA = glm::normalize(glm::cross(dir, glm::vec3(0, 1, 0.001f)));
            glm::vec3 sideB = glm::normalize(glm::cross(dir, sideA));

            if (depth == 3) {
                // Trunk grows ALL THE WAY to `end` (3 curving segments).
                // Branches emerge at 4 levels along the trunk with
                // progressive angles: lower levels reach horizontally
                // (phototropism — they escaped the shade by spreading
                // wide), upper levels stay more vertical (still chasing
                // sky). An apex leaf cluster caps the trunk top so the
                // crown is full from the centerline outward, not a
                // hollow ring like before.
                glm::vec3 q1 = origin + dir * (length * 0.33f)
                             + sideA * (urand() * length * 0.04f);
                glm::vec3 q2 = origin + dir * (length * 0.66f)
                             + sideB * (urand() * length * 0.04f);
                drawSeg(origin, q1, thick,        thick * 0.92f);
                drawSeg(q1,     q2, thick * 0.92f, thick * 0.80f);
                drawSeg(q2,     end, thick * 0.80f, thick * 0.62f);

                // Branch levels: fraction up the trunk, polar angle from
                // vertical (smaller = more vertical). 4 levels distributed
                // from 40% to 92% of trunk height — but the attach height is
                // CLAMPED to at least TREE_CANOPY_CLEAR (+ spacing per level)
                // above the base: crowns sit over the character's head, so
                // walking under a tree is walking, not an obstacle. Small
                // trees keep ~92% as their ceiling and become lollipops.
                static const float LEVELS[4] = { 0.40f, 0.58f, 0.74f, 0.90f };
                static const float THETAS[4] = { 1.00f, 0.78f, 0.58f, 0.40f };

                for (int lvl = 0; lvl < 4; ++lvl) {
                    float att = std::max(length * LEVELS[lvl],
                                         std::min(length * 0.92f,
                                                  TREE_CANOPY_CLEAR + 4.0f * float(lvl)));
                    glm::vec3 bo = origin + dir * att;
                    int n = 2 + int(rng() % 2);   // 2-3 branches per level
                    // Per-level yaw offset so adjacent levels don't stack
                    // their branches in the same compass directions.
                    float yawBase = LEVELS[lvl] * 3.7f + frand() * 1.5f;
                    for (int b = 0; b < n; ++b) {
                        float yaw   = yawBase + (float(b) / float(n)) * 6.2831853f
                                    + urand() * 0.25f;
                        float theta = THETAS[lvl] + urand() * 0.12f;
                        glm::vec3 nd(std::cos(yaw) * std::sin(theta),
                                     std::cos(theta),
                                     std::sin(yaw) * std::sin(theta));
                        nd.y = std::max(nd.y, 0.15f);
                        nd = glm::normalize(nd);
                        // Lower-level branches longer so they reach out
                        // further than upper branches.
                        float reach = 0.55f + 0.30f * (1.0f - LEVELS[lvl]);
                        float bl = length * (reach + frand() * 0.15f);
                        float bt = thick  * (0.45f + 0.10f * (1.0f - LEVELS[lvl]));
                        grow(bo, nd, bl, bt, 2, scale);
                    }
                }

                // Apex lobe — leaf cluster sitting directly on top of
                // the trunk's tip. Without this the crown centerline is
                // hollow and you see sky through the middle of the tree.
                shellCanopy(end, (6.0f + frand() * 2.0f) * scale);
                return;
            }

            // depth == 2 terminal: single curving branch ending in ONE
            // distinct dense leaf lobe. No more sub-branching — that was
            // causing adjacent shells to overlap into one cotton-candy
            // mass. The lobe stays dense (shellCanopy keeps interior
            // fillers); the overall crown stays full because we have 7
            // of these lobes per tree spread around the trunk top.
            glm::vec3 mid = origin + dir * (length * 0.5f)
                          + sideA * (urand() * length * 0.06f)
                          + sideB * (urand() * length * 0.06f);
            drawSeg(origin, mid, thick,         thick * 0.90f);
            drawSeg(mid,    end, thick * 0.90f, thick * 0.65f);
            shellCanopy(end, (5.5f + frand() * 2.5f) * scale);
        };

    glm::vec3 base(float(site.x) + 0.5f, float(site.h), float(site.z) + 0.5f);
    grow(base, glm::vec3(0, 1, 0), site.trunkLen, site.trunkThick, 3, site.scale);
}

void genTrees(World& w, GenContext& ctx, const GenBounds& b,
              const SurfaceSnapshot& snap, ChunkStageBuffer* buf) {
    StageWriter gw(w, b, ctx, buf);
    std::mt19937 rng;   // re-seeded per tree cell (featureSeed)

    // Tighter grid (55 vs 96) + higher spawn chance gives ~3-4× more
    // trees, closing the "empty meadow" gap with the reference. Per-tree
    // scale (squared bias toward small) creates a saplings-to-giants
    // mix so the forest doesn't look like a tree plantation.
    // (TREE_CELL/TREE_ANCHOR + the site derivation live in internal.h,
    // SHARED with the far-field ring impostors.)
    // Unbounded lattice anchored at TREE_ANCHOR + k*TREE_CELL for any integer
    // k (negative included); the cell range derives from the bounds + margin.
    int kz0 = floorDiv(b.z0 - MARGIN_TREE - TREE_CELL - TREE_ANCHOR, TREE_CELL);
    int kz1 = floorDiv(b.z1 - 1 + MARGIN_TREE - TREE_ANCHOR, TREE_CELL);
    int kx0 = floorDiv(b.x0 - MARGIN_TREE - TREE_CELL - TREE_ANCHOR, TREE_CELL);
    int kx1 = floorDiv(b.x1 - 1 + MARGIN_TREE - TREE_ANCHOR, TREE_CELL);
    for (int kz = kz0; kz <= kz1; ++kz) {
        int tz = TREE_ANCHOR + kz * TREE_CELL;
        if (tz + TREE_CELL + MARGIN_TREE <= b.z0 || tz - MARGIN_TREE >= b.z1) continue;
        for (int kx = kx0; kx <= kx1; ++kx) {
            int tx = TREE_ANCHOR + kx * TREE_CELL;
            if (tx + TREE_CELL + MARGIN_TREE <= b.x0 || tx - MARGIN_TREE >= b.x1) continue;
            // Per-cell stream: this tree (and everything about it) can be
            // regenerated knowing only the cell coords + world seed.
            // Footprint contract: canopy radius <= ~80.
            // Site derivation (draws 1..6) is SHARED with the far-field ring
            // impostors (deriveTreeSite, internal.h); growForestTree continues
            // from the SAME rng at draw 7, so this path stays bit-identical.
            TreeSite site = deriveTreeSite(ctx.seed, tx, tz, rng,
                [&](int x, int h, int z) {
                    return snap.get(x, h - 1, z) == MAT_GRASS;   // pre-stage gate
                });
            if (!site.spawn) continue;

            growForestTree(rng, site,
                [&](int x, int y, int z, uint8_t m) { gw.set(x, y, z, m); },
                [&](int x, int y, int z) { return gw.get(x, y, z); });
            if (gw.anchorIn(site.x, site.z)) ++gw.stats().trees;
        }
    }
}

// ---------------------------------------------------------------------------
// "Ask the generator" (forestry tree identity): trees derive bit-identically
// from (world seed, lattice cell) — the property chunk streaming already
// depends on — so a clicked voxel's owning tree is answered by REPLAYING the
// grammar for every site in reach into a scratch map, mirroring StageWriter's
// write semantics (wood overwrites, leaves first-writer) in canonical cell
// order. No storage, no staleness: the lattice IS the registry. Voxels the
// replay over-claims (the generator skipped them for terrain/flora the
// scratch can't see) are filtered by the caller's current-material match.
// The grass gate is the ring impostors' PURE-grammar test — the fine gen's
// pre-stage snapshot no longer exists post-gen (the trunk sits on its own
// grass tile); the ~1% divergence just means an occasional miss, and the
// caller falls back to flood-fill identification.
// ---------------------------------------------------------------------------

bool rederiveTreeAt(const GenContext& ctx, glm::ivec3 click, TreeRederive& out) {
    out = TreeRederive{};
    constexpr int REACH = 96;   // canopy footprint radius bound (MARGIN_TREE)
    auto pureGrass = [](int x, int h, int z) {
        SurfaceCol g = surfaceColAt(x, z);
        return surfaceMatAt(x, h - 1, z, g) == MAT_GRASS;
    };

    // Sites whose footprint can reach the click, in canonical (kz, kx) order.
    // The grammar over-emits heavily (drawSeg spheres revisit each voxel many
    // times), so this runs TWO cheap passes instead of one hash-map scratch —
    // the hash version cost whole seconds on a dense-forest click.
    struct Cell { int tx, tz; TreeSite site; };
    std::vector<Cell> cells;
    {
        std::mt19937 rng;
        int kz0 = floorDiv(click.z - REACH - (TREE_CELL - 1) - TREE_ANCHOR, TREE_CELL);
        int kz1 = floorDiv(click.z + REACH - TREE_ANCHOR, TREE_CELL);
        int kx0 = floorDiv(click.x - REACH - (TREE_CELL - 1) - TREE_ANCHOR, TREE_CELL);
        int kx1 = floorDiv(click.x + REACH - TREE_ANCHOR, TREE_CELL);
        for (int kz = kz0; kz <= kz1; ++kz)
            for (int kx = kx0; kx <= kx1; ++kx) {
                int tx = TREE_ANCHOR + kx * TREE_CELL;
                int tz = TREE_ANCHOR + kz * TREE_CELL;
                TreeSite site = deriveTreeSite(ctx.seed, tx, tz, rng, pureGrass);
                if (!site.spawn) continue;
                // Skipping a site never perturbs the others (per-cell reseed).
                if (std::abs(site.x - click.x) > REACH ||
                    std::abs(site.z - click.z) > REACH) continue;
                cells.push_back({tx, tz, site});
            }
    }
    if (cells.empty()) return false;
    VOX_ZONE_N("flora: rederive tree");

    // Grammar footprint bound: canopy radius <= ~80 horizontally, emissions
    // within [h - 12, h + 152] vertically (branch arcs + apex shells). Both
    // passes use it to SKIP sites that cannot touch the cell(s) in question
    // — in dense forest that turns "regrow ~12 trees per click" into
    // "regrow 1-4", which is the difference between a 60 ms and a ~15 ms
    // click (and the click runs inside the input callback, so it IS the
    // frame hitch).
    constexpr int FOOT = 88;
    auto canTouchClick = [&](const TreeSite& s) {
        return std::abs(s.x - click.x) <= FOOT &&
               std::abs(s.z - click.z) <= FOOT &&
               click.y >= s.h - 12 && click.y <= s.h + 152;
    };

    // Re-arm a fresh rng at grammar draw 7 for one site, then grow it.
    auto growAt = [&](const Cell& c, auto&& setV, auto&& getV) {
        std::mt19937 r;
        TreeSite s = deriveTreeSite(ctx.seed, c.tx, c.tz, r, pureGrass);
        growForestTree(r, s, setV, getV);
    };

    // Pass 1 — who wrote the CLICK voxel? StageWriter semantics collapsed to
    // one cell: final owner = the LAST wood writer if any wood landed there
    // (drawSeg overwrites), else the FIRST leaf writer (bulb skips occupied).
    // Skipping a site that cannot reach the click never changes the answer.
    int lastWood = -1, firstLeaf = -1;
    for (int i = 0; i < int(cells.size()); ++i) {
        if (!canTouchClick(cells[i].site)) continue;
        growAt(cells[i],
            [&](int x, int y, int z, uint8_t m) {
                if (x != click.x || y != click.y || z != click.z) return;
                if (m == MAT_WOOD) lastWood = i;
                else if (firstLeaf < 0 && lastWood < 0) firstLeaf = i;
            },
            [&](int x, int y, int z) -> uint8_t {
                if (x == click.x && y == click.y && z == click.z)
                    return lastWood >= 0 ? MAT_WOOD
                         : firstLeaf >= 0 ? MAT_LEAVES : 0;
                return 0;
            });
    }
    int owner = lastWood >= 0 ? lastWood : firstLeaf;
    if (owner < 0) return false;

    // Pass 2 — the owner's exact voxel set, resolved on a DENSE grid over
    // its footprint (array writes, no hashing). Enumeration here re-centres
    // on the OWNER: every site whose footprint can overlap the owner's box
    // must replay, or interleaved neighbour crowns get claimed as ours (a
    // dense-forest click then "fells" tens of thousands of stolen leaves and
    // the decay sweep inherits the mess).
    const TreeSite& s = cells[size_t(owner)].site;
    std::vector<Cell> replay;
    {
        std::mt19937 rng;
        // Contest radius: a site at distance d can write owner-box cells iff
        // d <= 2*FOOT, but beyond FOOT+22 the overlap is a thin pure-canopy
        // fringe lens — skipping it misattributes at most a few edge leaves
        // (the caller's material match + the decay invariant absorb that),
        // and it halves the replay set: measured 51-58 ms per DENSE-forest
        // identification at 2*FOOT, ~15-20 ms here.
        constexpr int SPAN = FOOT + 22;
        int kz0 = floorDiv(s.z - SPAN - (TREE_CELL - 1) - TREE_ANCHOR, TREE_CELL);
        int kz1 = floorDiv(s.z + SPAN - TREE_ANCHOR, TREE_CELL);
        int kx0 = floorDiv(s.x - SPAN - (TREE_CELL - 1) - TREE_ANCHOR, TREE_CELL);
        int kx1 = floorDiv(s.x + SPAN - TREE_ANCHOR, TREE_CELL);
        for (int kz = kz0; kz <= kz1; ++kz)
            for (int kx = kx0; kx <= kx1; ++kx) {
                int tx = TREE_ANCHOR + kx * TREE_CELL;
                int tz = TREE_ANCHOR + kz * TREE_CELL;
                TreeSite site = deriveTreeSite(ctx.seed, tx, tz, rng, pureGrass);
                if (!site.spawn) continue;
                if (std::abs(site.x - s.x) > SPAN ||
                    std::abs(site.z - s.z) > SPAN) continue;
                replay.push_back({tx, tz, site});
            }
    }
    int ownerR = -1;
    for (int i = 0; i < int(replay.size()); ++i)
        if (replay[i].tx == cells[size_t(owner)].tx &&
            replay[i].tz == cells[size_t(owner)].tz) { ownerR = i; break; }
    if (ownerR < 0) return false;   // cannot happen: the owner is in SPAN

    glm::ivec3 glo(s.x - FOOT, std::max(0, s.h - 12), s.z - FOOT);
    glm::ivec3 ghi(s.x + FOOT, std::min(GRID_Y - 1, s.h + 152), s.z + FOOT);
    glm::ivec3 gd = ghi - glo + 1;
    size_t N = size_t(gd.x) * gd.y * gd.z;
    // Reused across calls. thread_local: the hover worker (forestry.cpp)
    // rederives on a background thread while a click can rederive on the
    // main thread — shared statics would race.
    thread_local std::vector<int8_t>  gOwner;
    thread_local std::vector<uint8_t> gMat;
    gOwner.assign(N, -1);
    gMat.assign(N, 0);
    auto gidx = [&](int x, int y, int z) -> int64_t {
        if (x < glo.x || x > ghi.x || y < glo.y || y > ghi.y ||
            z < glo.z || z > ghi.z) return -1;
        return (int64_t(z - glo.z) * gd.y + (y - glo.y)) * gd.x + (x - glo.x);
    };
    for (int i = 0; i < int(replay.size()); ++i) {
        growAt(replay[i],
            [&](int x, int y, int z, uint8_t m) {
                int64_t g = gidx(x, y, z);
                if (g < 0) return;
                if (m == MAT_WOOD) {                       // wood overwrites
                    gOwner[size_t(g)] = int8_t(i);
                    gMat[size_t(g)] = m;
                } else if (gMat[size_t(g)] == 0) {         // leaves: first writer
                    gOwner[size_t(g)] = int8_t(i);
                    gMat[size_t(g)] = m;
                }
            },
            [&](int x, int y, int z) -> uint8_t {
                int64_t g = gidx(x, y, z);
                return g < 0 ? 0 : gMat[size_t(g)];
            });
    }
    out.base = glm::ivec3(s.x, s.h, s.z);
    out.trunkR = s.trunkThick;
    out.voxels.reserve(8192);
    size_t g = 0;
    for (int z = glo.z; z <= ghi.z; ++z)
        for (int y = glo.y; y <= ghi.y; ++y)
            for (int x = glo.x; x <= ghi.x; ++x, ++g)
                if (gOwner[g] == int8_t(ownerR))
                    out.voxels.push_back({glm::ivec3(x, y, z), gMat[g]});
    out.found = true;
    return true;
}

// ---------------------------------------------------------------------------
// "Ask the generator" (mining rock identity): boulders derive bit-identically
// from (world seed, SEED_BOULDER lattice cell) exactly like trees, so the
// rock under a click is answered by replaying every site in reach — and a
// CONJOINED cluster splits into its true member rocks, which the flood fill
// fundamentally cannot do (it sees one fused stone component and rejects it
// through the terrain caps). Overlap ownership follows StageWriter: boulders
// write MAT_STONE unconditionally, so the LAST covering site in canonical
// (cellZ, cellX, site, child) order owns the voxel.
//
// Two deltas from rederiveTreeAt:
//  - genBoulders draws a cell's sites from ONE sequential stream, and a
//    failed anchor gate consumes none of the seven shape draws — the gate
//    decision at every earlier site shifts the stream for every later one.
//    Trees reseed per site, so gate divergence costs one tree; here one
//    wrong gate garbles the rest of the cell. The oracle therefore consults
//    the LIVE world first: a cluster parent routinely buries its neighbour's
//    anchor in stone (the exact case this exists to split), and live grass
//    is proof of gen-time grass wherever the pure grammar drifts from the
//    gen-time snapshot (which no longer exists). Residual mispredictions
//    (cave-carved anchors) garble the claimed set — the caller's
//    live-material spot check rejects it and falls back to the flood.
//  - The World read makes this MAIN-THREAD ONLY (the tree hover worker
//    stays off World for exactly this reason); rock hover identifies
//    synchronously, so nothing is lost.
// ---------------------------------------------------------------------------

namespace {
struct BoulderSite {
    int   cx = 0, cz = 0, h = 0, cy = 0;
    float aX = 1, aY = 1, aZ = 1;
    float cs = 1, sn = 0, nseed = 0;
};
}  // namespace

// Replicates placeBoulder's write test EXACTLY, including the loop-cube
// clamp: bulge can push the ellipsoid boundary past the cube, but the
// generator never visits there, so neither may we.
static bool boulderCovers(const BoulderSite& s, glm::ivec3 v) {
    if (v.y < 0 || v.y >= GRID_Y) return false;
    int rad = int(std::ceil(std::max({s.aX, s.aY, s.aZ}))) + 1;
    int dx = v.x - s.cx, dy = v.y - s.cy, dz = v.z - s.cz;
    if (std::abs(dx) > rad || std::abs(dy) > rad || std::abs(dz) > rad)
        return false;
    float lx = float(dx) * s.cs - float(dz) * s.sn;
    float lz = float(dx) * s.sn + float(dz) * s.cs;
    float ex = lx / s.aX, ey = float(dy) / s.aY, ez = lz / s.aZ;
    float d  = ex * ex + ey * ey + ez * ez;
    // bulge is bounded to [-0.365, +0.365] (noise in [0,1], 0.55/2 + 0.18/2),
    // so d alone decides outside that band — skipping both noise octaves for
    // the vast majority of cube voxels. Bit-identical to the full test.
    if (d > 1.365f)  return false;
    if (d <= 0.635f) return true;
    float big  = smoothNoise3(float(v.x) * 0.08f + s.nseed,
                              float(v.y) * 0.08f + s.nseed,
                              float(v.z) * 0.08f + s.nseed);
    float fine = smoothNoise3(float(v.x) * 0.28f + s.nseed * 0.5f,
                              float(v.y) * 0.28f + s.nseed * 0.5f,
                              float(v.z) * 0.28f + s.nseed * 0.5f);
    float bulge = (big - 0.5f) * 0.55f + (fine - 0.5f) * 0.18f;
    return d <= 1.0f + bulge;
}

// Every boulder site whose writes can reach [x0..x1]x[z0..z1] (inclusive),
// in canonical order. A DRAW-FOR-DRAW mirror of genBoulders: the cell prefix
// (count round-up), per-site position draws, the cluster roll/count as RAW
// rng() words, the child offset+size draws consumed even for gated children,
// and sizeRoll drawn-and-discarded under a cluster size override. Reordering
// anything here versus genBoulders breaks identity silently — the spot check
// downgrades that to "flood fallback", but conjoined rocks stop splitting.
template <typename Gate>
static void enumBoulderSites(const GenContext& ctx, int x0, int x1, int z0,
                             int z1, Gate&& gate,
                             std::vector<BoulderSite>& out) {
    constexpr int   CELL     = 128;
    constexpr float PER_CELL = 300.0f / float((GRID_XZ / CELL) * (GRID_XZ / CELL));
    std::mt19937 rng;
    auto frand = [&]() { return frand01(rng); };
    auto derive = [&](int cx, int cz, float sizeOverride) {
        int h = heightAt(cx, cz);
        if (h < 10 || h >= GRID_Y - 30) return;   // 0 draws, like the gen
        if (!gate(cx, h, cz)) return;             // 0 draws, like the gen
        float sizeRoll = frand();                 // consumed even when overridden
        float base = sizeOverride > 0.0f
                   ? sizeOverride
                   : 2.5f + 14.0f * sizeRoll * sizeRoll;
        BoulderSite s;
        s.cx = cx; s.cz = cz; s.h = h;
        s.aX = base * (0.85f + 0.45f * frand());
        s.aZ = base * (0.85f + 0.45f * frand());
        s.aY = base * (0.55f + 0.30f * frand());
        float embed = 0.25f + 0.30f * frand();
        s.cy = int(std::round(float(h) - s.aY * embed));
        float yaw = frand() * 6.2831853f;
        s.cs = std::cos(yaw); s.sn = std::sin(yaw);
        s.nseed = frand() * 100.0f;
        out.push_back(s);
    };
    int c0x = floorDiv(x0 - MARGIN_BOULDER, CELL);
    int c1x = floorDiv(x1 + MARGIN_BOULDER, CELL);
    int c0z = floorDiv(z0 - MARGIN_BOULDER, CELL);
    int c1z = floorDiv(z1 + MARGIN_BOULDER, CELL);
    for (int cellZ = c0z; cellZ <= c1z; ++cellZ)
    for (int cellX = c0x; cellX <= c1x; ++cellX) {
        rng.seed(featureSeed(ctx.seed, SEED_BOULDER, cellX, cellZ));
        float perCell = PER_CELL *
            biomeAt(cellX * CELL + CELL / 2, cellZ * CELL + CELL / 2).boulderScale;
        int n = int(perCell);
        if (frand() < perCell - float(int(perCell))) ++n;
        for (int i = 0; i < n; ++i) {
            int cx = cellX * CELL + int(frand() * CELL);
            int cz = cellZ * CELL + int(frand() * CELL);
            derive(cx, cz, -1.0f);
            if ((rng() & 1023) < 300) {
                int extra = 1 + int(rng() % 3);
                for (int j = 0; j < extra; ++j) {
                    int ox = cx + int((frand() - 0.5f) * 22.0f);
                    int oz = cz + int((frand() - 0.5f) * 22.0f);
                    derive(ox, oz, 2.0f + 3.5f * frand());
                }
            }
        }
    }
}

bool rederiveRockAt(const GenContext& ctx, const World& w, glm::ivec3 click,
                    RockRederive& out) {
    out = RockRederive{};
    VOX_ZONE_N("flora: rederive rock");
    // Live grass is proof of gen-time grass (nothing plants grass post-gen);
    // where decoration has since buried the anchor (a parent boulder over a
    // neighbour's tile, the boulder's own root, a tree), the pure grammar
    // answers for the pre-decoration surface — rederiveTreeAt's oracle.
    // CAVE CARVE-OUTS are the divergence that matters: gen's snapshot gate
    // ran post-carve, the grammar is cave-blind, and one phantom site shifts
    // the whole cell's stream. A breached anchor reads as a DEEP air column
    // today, while a mined rock's scar keeps its stone root within the soil
    // clamp depth — probe a few voxels down to tell them apart (a scar pit
    // later flooded by the water sim still mispredicts; the caller's spot
    // check downgrades that cell to the flood).
    auto gate = [&w](int x, int h, int z) {
        uint8_t live = w.getVoxel(x, h - 1, z);
        if (live == MAT_GRASS) return true;
        if (live == MAT_AIR || live == MAT_WATER) {
            bool solidNear = false;
            for (int k = 1; k <= 6 && !solidNear; ++k) {
                uint8_t v = w.getVoxel(x, h - 1 - k, z);
                solidNear = v != MAT_AIR && v != MAT_WATER;
            }
            if (!solidNear || live == MAT_WATER) return false;
        }
        SurfaceCol g = surfaceColAt(x, z);
        return surfaceMatAt(x, h - 1, z, g) == MAT_GRASS;
    };

    // Pass 1 — who wrote the click voxel? Last covering site wins.
    std::vector<BoulderSite> sites;
    enumBoulderSites(ctx, click.x, click.x, click.z, click.z, gate, sites);
    int owner = -1;
    for (int i = 0; i < int(sites.size()); ++i)
        if (boulderCovers(sites[i], click)) owner = i;
    if (owner < 0) return false;
    const BoulderSite os = sites[size_t(owner)];

    // Pass 2 — the owner's OWNED voxels: re-enumerate every site whose
    // writes can reach the owner's cube (re-anchored on the OWNER — a
    // conjoined neighbour may reach voxels the click-centred range missed),
    // and drop any voxel a LATER site also covers: it overwrote us.
    int rad = int(std::ceil(std::max({os.aX, os.aY, os.aZ}))) + 1;
    std::vector<BoulderSite> field;
    enumBoulderSites(ctx, os.cx - rad, os.cx + rad, os.cz - rad, os.cz + rad,
                     gate, field);
    int self = -1;
    for (int i = 0; i < int(field.size()); ++i)
        if (field[i].cx == os.cx && field[i].cz == os.cz &&
            field[i].cy == os.cy && field[i].nseed == os.nseed) { self = i; break; }
    if (self < 0) return false;   // cannot happen: the owner is in range

    out.voxels.reserve(512);
    for (int dz = -rad; dz <= rad; ++dz)
    for (int dy = -rad; dy <= rad; ++dy)
    for (int dx = -rad; dx <= rad; ++dx) {
        glm::ivec3 v(os.cx + dx, os.cy + dy, os.cz + dz);
        if (!boulderCovers(os, v)) continue;
        bool stolen = false;
        for (int i = self + 1; i < int(field.size()) && !stolen; ++i)
            stolen = boulderCovers(field[i], v);
        if (!stolen) out.voxels.push_back(v);
    }
    if (out.voxels.empty()) return false;
    out.anchor = glm::ivec3(os.cx, os.h, os.cz);
    out.found  = true;
    return true;
}


// -- Lakes / ponds -------------------------------------------------------
// Fill terrain depressions below a global LAKE_LEVEL with water. Only
// columns whose surface dips below the level get flooded — high ground
// stays dry — so we get scattered ponds in low spots rather than a
// global ocean. Direct brickmap/brickpool writes (no setVoxel) so we
// don't tag thousands of bricks as "wet" and stall the water sim at
// startup; settled water doesn't need the simulation tracking it.
void genLakes(World& w, GenContext& ctx, const GenBounds& b) {
    (void)ctx;   // no RNG/stats: lakes are a pure per-column function
    constexpr int LAKE_LEVEL = SURFACE_Y - 40;
    auto placeWaterStatic = [&](int x, int y, int z) {
        if (y < 0 || y >= GRID_Y || !w.voxelInWindow(x, z)) return;
        int bx = x >> BRICK_BITS, by = y >> BRICK_BITS, bz = z >> BRICK_BITS;
        int slotIdx = w.slotOfBrick(bx, by, bz);
        uint32_t bm = w.brickmap[slotIdx];

        // Allocate a per-voxel brick if we hit an empty/solid one.
        if (bm == BM_EMPTY || (bm & BM_SOLID_FLAG)) {
            uint8_t fill = (bm == BM_EMPTY) ? 0 : uint8_t(bm & 0xFFu);
            if (fill == MAT_WATER) return;        // already entirely water
            uint32_t bi = w.allocBrick(fill);
            w.brickmap[slotIdx] = bi + 1u;
            bm = bi + 1u;
        }
        uint32_t bi = bm - 1u;
        int linear = voxInBrick(x & BRICK_MASK, y & BRICK_MASK, z & BRICK_MASK);
        int wslot = linear >> 2;
        int wshift = (linear & 3) * 8;
        uint32_t mask = 0xFFu << wshift;
        uint32_t& word = w.brickpool[bi * BRICK_U32 + wslot];
        uint8_t cur = uint8_t((word >> wshift) & 0xFFu);
        if (cur != MAT_AIR) return;               // don't overwrite solids
        word = (word & ~mask) | (uint32_t(MAT_WATER) << wshift);
    };

    // Convert an existing terrain voxel to stone (for the lakebed). Unlike
    // placeWaterStatic this OVERWRITES the current solid, but leaves air and
    // water untouched so we only restyle real terrain.
    auto setBedStone = [&](int x, int y, int z) {
        if (y < 0 || y >= GRID_Y || !w.voxelInWindow(x, z)) return;
        int bx = x >> BRICK_BITS, by = y >> BRICK_BITS, bz = z >> BRICK_BITS;
        int slotIdx = w.slotOfBrick(bx, by, bz);
        uint32_t bm = w.brickmap[slotIdx];
        if (bm == BM_EMPTY || (bm & BM_SOLID_FLAG)) {
            uint8_t fill = (bm == BM_EMPTY) ? 0 : uint8_t(bm & 0xFFu);
            uint32_t bi = w.allocBrick(fill);
            w.brickmap[slotIdx] = bi + 1u;
            bm = bi + 1u;
        }
        uint32_t bi = bm - 1u;
        int linear = voxInBrick(x & BRICK_MASK, y & BRICK_MASK, z & BRICK_MASK);
        int wslot = linear >> 2;
        int wshift = (linear & 3) * 8;
        uint32_t mask = 0xFFu << wshift;
        uint32_t& word = w.brickpool[bi * BRICK_U32 + wslot];
        uint8_t cur = uint8_t((word >> wshift) & 0xFFu);
        if (cur == MAT_AIR || cur == MAT_WATER) return;   // terrain only
        word = (word & ~mask) | (uint32_t(MAT_STONE) << wshift);
    };

    // TEMP DISABLED (lakes were just a water test bed). Static fill only —
    // these bricks are NOT tagged wet, so the sim never touches them and
    // there's no FPS hit. Helpers + LAKE_LEVEL kept intact; flip LAKES_ON to
    // bring them back. Floods every column whose surface dips below
    // LAKE_LEVEL up to that level and stones the top few bed voxels.
    constexpr bool LAKES_ON  = false;
    constexpr int  BED_DEPTH = 3;
    (void)placeWaterStatic; (void)setBedStone; (void)BED_DEPTH;
    if (LAKES_ON) {
        for (int z = b.z0; z < b.z1; ++z)
        for (int x = b.x0; x < b.x1; ++x) {
            int h = heightAt(x, z);
            if (h >= LAKE_LEVEL) continue;
            for (int y = h - 1; y >= 0 && y > h - 1 - BED_DEPTH; --y) {
                setBedStone(x, y, z);
            }
            for (int y = h; y < LAKE_LEVEL; ++y) {
                placeWaterStatic(x, y, z);
            }
        }
    }
}

// Oasis pool: STATIC MAT_WATER in the carved basin, written EXACTLY like
// genLakes' static path — direct brickpool writes, NEVER setVoxel, NEVER
// World::wetBricks — so the pool has ZERO water-sim cost (the wetBricks FPS
// trap). Pure per-column fn of (x,z); runs at the lake stage AFTER terrain fill
// so the basin + air columns exist. Contained by construction: water only where
// the carved surface dips below the flat center-sampled waterY, and the rim
// rises back above it inside the footprint (no desert-wide spill).
void genOasis(World& w, GenContext& ctx, const GenBounds& b) {
    (void)ctx;
    constexpr int BED_DEPTH = 3;
    auto placeWaterStatic = [&](int x, int y, int z) {
        if (y < 0 || y >= GRID_Y || !w.voxelInWindow(x, z)) return;
        int bx = x >> BRICK_BITS, by = y >> BRICK_BITS, bz = z >> BRICK_BITS;
        int slotIdx = w.slotOfBrick(bx, by, bz);
        uint32_t bm = w.brickmap[slotIdx];
        if (bm == BM_EMPTY || (bm & BM_SOLID_FLAG)) {
            uint8_t fill = (bm == BM_EMPTY) ? 0 : uint8_t(bm & 0xFFu);
            if (fill == MAT_WATER) return;
            uint32_t bi = w.allocBrick(fill);
            w.brickmap[slotIdx] = bi + 1u;
            bm = bi + 1u;
        }
        uint32_t bi = bm - 1u;
        int linear = voxInBrick(x & BRICK_MASK, y & BRICK_MASK, z & BRICK_MASK);
        int wslot = linear >> 2;
        int wshift = (linear & 3) * 8;
        uint32_t mask = 0xFFu << wshift;
        uint32_t& word = w.brickpool[bi * BRICK_U32 + wslot];
        uint8_t cur = uint8_t((word >> wshift) & 0xFFu);
        if (cur != MAT_AIR) return;               // only fills air; never clobbers the sand floor/banks
        word = (word & ~mask) | (uint32_t(MAT_WATER) << wshift);
    };
    auto setBedStone = [&](int x, int y, int z) {
        if (y < 0 || y >= GRID_Y || !w.voxelInWindow(x, z)) return;
        int bx = x >> BRICK_BITS, by = y >> BRICK_BITS, bz = z >> BRICK_BITS;
        int slotIdx = w.slotOfBrick(bx, by, bz);
        uint32_t bm = w.brickmap[slotIdx];
        if (bm == BM_EMPTY || (bm & BM_SOLID_FLAG)) {
            uint8_t fill = (bm == BM_EMPTY) ? 0 : uint8_t(bm & 0xFFu);
            uint32_t bi = w.allocBrick(fill);
            w.brickmap[slotIdx] = bi + 1u;
            bm = bi + 1u;
        }
        uint32_t bi = bm - 1u;
        int linear = voxInBrick(x & BRICK_MASK, y & BRICK_MASK, z & BRICK_MASK);
        int wslot = linear >> 2;
        int wshift = (linear & 3) * 8;
        uint32_t mask = 0xFFu << wshift;
        uint32_t& word = w.brickpool[bi * BRICK_U32 + wslot];
        uint8_t cur = uint8_t((word >> wshift) & 0xFFu);
        if (cur == MAT_AIR || cur == MAT_WATER) return;   // terrain only
        word = (word & ~mask) | (uint32_t(MAT_STONE) << wshift);
    };

    for (int z = b.z0; z < b.z1; ++z)
    for (int x = b.x0; x < b.x1; ++x) {
        OasisHit oa = oasisAt(x, z);
        if (!oa.any || oa.w <= 0.0f) continue;    // only INSIDE the warped shore (uw<1) -> water can never escape the rim
        int top = oa.waterY;                      // flat, rim-min level
        int h = heightAt(x, z);                   // carved surface (dig + berm)
        if (h >= top) continue;                   // bank above water -> dry
        for (int y = h - 1; y >= 0 && y > h - 1 - BED_DEPTH; --y) setBedStone(x, y, z);  // stone bed for caustics
        for (int y = h; y < top; ++y) placeWaterStatic(x, y, z);
    }
}


// -- Flowers -------------------------------------------------------------
// Four species, each defined as a stem-height range + a 3D bloom template
// relative to the top of the stem. Per-flower we roll for: stem length,
// optional side leaf, and bloom color from the species palette. That
// means two flowers of the same species rarely look identical.
void genFlowers(World& w, GenContext& ctx, const GenBounds& b,
                const SurfaceSnapshot& snap, ChunkStageBuffer* buf) {
    StageWriter gw(w, b, ctx, buf);
    std::mt19937 rng;   // re-seeded per flower (featureSeed)

    enum Role : uint8_t { R_BLOOM, R_ACCENT };
    struct BloomVox { int dx, dy, dz; Role role; };   // dy = 0 is on top of stem

    // Bell-shaped 3D bloom — used by Poppy and Tulip (color differs).
    static const BloomVox BELL[] = {
        {0,0,0,R_BLOOM}, {1,0,0,R_BLOOM}, {-1,0,0,R_BLOOM},
        {0,0,1,R_BLOOM}, {0,0,-1,R_BLOOM},
        {0,1,0,R_BLOOM},
    };
    // Daisy: flat bicolor — yellow center + white petals on cardinals.
    static const BloomVox DAISY_BLOOM[] = {
        {0,0,0,R_ACCENT},
        {1,0,0,R_BLOOM}, {-1,0,0,R_BLOOM},
        {0,0,1,R_BLOOM}, {0,0,-1,R_BLOOM},
    };
    // Bluebell: small 3D cluster — voxels offset asymmetrically for depth.
    static const BloomVox BLUEBELL_BLOOM[] = {
        {0,0,0,R_BLOOM}, {1,0,0,R_BLOOM},
        {0,0,1,R_BLOOM}, {-1,0,-1,R_BLOOM},
    };

    struct Species {
        const BloomVox* bloom;
        int bloomCount;
        uint8_t colors[3];
        int colorCount;
        uint8_t accent;
        int minStem, maxStem;
    };
    static const Species SPECIES[4] = {
        // bloom, count, colors,            cN, accent,           stem
        { BELL,    6, {MAT_FLOWER_RED,
                       MAT_FLOWER_YELLOW, 0}, 2, MAT_FLOWER_YELLOW, 1, 2 }, // poppy
        { BELL,    6, {MAT_FLOWER_RED,
                       MAT_FLOWER_YELLOW,
                       MAT_FLOWER_WHITE},     3, MAT_FLOWER_YELLOW, 2, 4 }, // tulip
        { DAISY_BLOOM, 5, {MAT_FLOWER_WHITE,
                           0, 0},             1, MAT_FLOWER_YELLOW, 2, 3 }, // daisy
        { BLUEBELL_BLOOM, 4, {MAT_FLOWER_BLUE,
                              0, 0},          1, MAT_FLOWER_BLUE,   2, 3 }, // bluebell
    };

    static const int DIR_DX[4] = { 1, -1,  0,  0 };
    static const int DIR_DZ[4] = { 0,  0,  1, -1 };

    auto placeFlower = [&](int cx, int cz) {
        int h = heightAt(cx, cz);
        if (h < 1 || h >= GRID_Y - 8) return;
        if (snap.get(cx, h - 1, cz) != MAT_GRASS) return;   // pre-stage gate

        const Species& sp = SPECIES[rng() & 3];
        uint8_t bloomMat = sp.colors[rng() % sp.colorCount];

        int stemLen = sp.minStem + int(rng() % (sp.maxStem - sp.minStem + 1));

        // Optional side leaf — single stem-colored voxel sticking out at
        // mid-height. ~50% of flowers (only those tall enough).
        bool hasLeaf  = (stemLen >= 2) && ((rng() & 1) == 0);
        int leafDir   = int(rng() & 3);
        int leafY     = std::max(1, stemLen - 1);

        // Aggregate target cells (max ~16: 4 stem + 1 leaf + 6 bloom).
        struct Cell { int x, y, z; uint8_t mat; };
        Cell cells[16];
        int n = 0;
        for (int i = 0; i < stemLen; ++i) {
            cells[n++] = {cx, h + i, cz, MAT_FLOWER_STEM};
        }
        if (hasLeaf) {
            cells[n++] = {cx + DIR_DX[leafDir], h + leafY,
                          cz + DIR_DZ[leafDir], MAT_FLOWER_STEM};
        }
        for (int i = 0; i < sp.bloomCount; ++i) {
            const BloomVox& v = sp.bloom[i];
            uint8_t m = (v.role == R_ACCENT) ? sp.accent : bloomMat;
            cells[n++] = {cx + v.dx, h + stemLen + v.dy,
                          cz + v.dz, m};
        }

        for (int i = 0; i < n; ++i) {
            const Cell& c = cells[i];
            if (c.y < 0 || c.y >= GRID_Y) return;   // XZ unbounded
            // All-or-nothing template check against the PRE-STAGE snapshot.
            if (snap.get(c.x, c.y, c.z) != MAT_AIR) return;
        }
        for (int i = 0; i < n; ++i) {
            const Cell& c = cells[i];
            gw.set(c.x, c.y, c.z, c.mat);
        }
        if (gw.anchorIn(cx, cz)) ++gw.stats().flowers;
    };

    // Scatter via a pure per-column hash (no sequential draws), then give
    // each spawned flower its own stream for species/stem/leaf rolls.
    // Footprint contract: flower radius <= 2.
    constexpr float DENSITY = 0.00075f;   // ~0.075% of grass surface
    int x0 = b.x0 - MARGIN_FLOWER, x1 = b.x1 + MARGIN_FLOWER;
    int z0 = b.z0 - MARGIN_FLOWER, z1 = b.z1 + MARGIN_FLOWER;
    constexpr float MAX_FLOWER_SCALE = 1.6f;   // meadow; cheap upper-bound reject
    for (int z = z0; z < z1; ++z)
    for (int x = x0; x < x1; ++x) {
        float fc = featureChance(ctx.seed, SEED_FLOWER_SCATTER, x, z);
        if (fc > DENSITY * MAX_FLOWER_SCALE) continue;   // reject before biomeAt
        // BIOME density: meadows bloom, forest/highland are sparse.
        if (fc > DENSITY * biomeAt(x, z).flowerScale) continue;
        rng.seed(featureSeed(ctx.seed, SEED_FLOWER, x, z));
        placeFlower(x, z);
    }
}


// -- Grass tufts ---------------------------------------------------------
// Patches of upright grass blades scattered across the surface. We process
// 8-voxel cells: each cell rolls for "host a cluster?", weighted by a slow
// 2D density-noise field so meadows and bald spots coexist. Each cluster
// is 5-9 narrow vertical columns within radius 2-4 of a per-cell center,
// sharing a base height (triangular distribution favouring short tufts)
// with per-blade jitter. Runs after trees/bushes/boulders so those win
// any footprint conflict.
void genGrass(World& w, GenContext& ctx, const GenBounds& b,
              const SurfaceSnapshot& snap, ChunkStageBuffer* buf) {
    StageWriter gw(w, b, ctx, buf);
    std::mt19937 rng;   // re-seeded per grass cell (featureSeed)
    auto frand = [&]() { return frand01(rng); };

    constexpr int  CELL          = 7;
    constexpr float CELL_PROB    = 0.70f;   // base chance per cell
    constexpr float DENSITY_FREQ = 0.005f;  // ~200-voxel-period density noise

    auto plantBlade = [&](int x, int z, int height) {
        int h = heightAt(x, z);
        if (h < 1 || h >= GRID_Y - height - 1) return;
        // Surface tile under the blade must still be grass (trees/boulders
        // may have overwritten it). Top must be air; we don't want blades
        // poking into bushes, leaves, or wood. Pre-stage snapshot gates.
        if (snap.get(x, h - 1, z) != MAT_GRASS) return;
        for (int dy = 0; dy < height; ++dy) {
            if (snap.get(x, h + dy, z) != MAT_AIR) return;
        }
        for (int dy = 0; dy < height; ++dy) {
            gw.set(x, h + dy, z, MAT_GRASS_BLADE);
        }
        if (gw.anchorIn(x, z)) ++gw.stats().grassBlades;
    };

    int c0x = floorDiv(b.x0 - MARGIN_GRASS, CELL);
    int c1x = floorDiv(b.x1 - 1 + MARGIN_GRASS, CELL);
    int c0z = floorDiv(b.z0 - MARGIN_GRASS, CELL);
    int c1z = floorDiv(b.z1 - 1 + MARGIN_GRASS, CELL);
    for (int cz = c0z; cz <= c1z; ++cz)
    for (int cx = c0x; cx <= c1x; ++cx) {
        // Per-cell stream (footprint contract: cluster radius <= 6).
        rng.seed(featureSeed(ctx.seed, SEED_GRASS, cx, cz));
        // Density noise -> meadow/bald variation. Map [0,1] noise to a
        // [0.15, 1.5] multiplier so deep meadows can exceed base prob.
        float x0 = float(cx * CELL);
        float z0 = float(cz * CELL);
        float dens = smoothNoise(x0 * DENSITY_FREQ, z0 * DENSITY_FREQ);
        float clusterChance = CELL_PROB * (0.15f + 1.35f * dens);

        if (frand() > clusterChance) continue;

        // Cluster center within the cell (use mid-region so footprint
        // mostly stays inside cell bounds; the radius can still spill
        // slightly into neighbours, which softens the grid).
        int center_x = int(x0) + 1 + int(frand() * (CELL - 2));
        int center_z = int(z0) + 1 + int(frand() * (CELL - 2));
        if (gw.anchorIn(center_x, center_z)) ++gw.stats().grassClusters;

        // Anchor height = the tallest blade(s) at cluster center.
        // Outer blades fall off with distance so each cluster has 1-2
        // peaks surrounded by progressively shorter neighbours.
        int anchorH = 5 + int(rng() % 4);   // 5..8

        // Wider footprint with fewer blades; we reject any blade that
        // lands within MIN_SEP of one we already placed so blades have
        // breathing room instead of clumping together.
        int attempts  = 8 + int(rng() % 5);   // 8..12 candidate positions
        int rad       = 2 + int(rng() % 3);   // 2..4
        const int MIN_SEP_SQ = 4;             // distance^2; 2 = "no diagonal touch"

        int placedX[16], placedZ[16];
        int placedN = 0;

        for (int b = 0; b < attempts; ++b) {
            float ang = frand() * 6.2831853f;
            float r   = std::sqrt(frand()) * float(rad);
            int bx = center_x + int(std::round(std::cos(ang) * r));
            int bz = center_z + int(std::round(std::sin(ang) * r));

            // Reject if too close to an already-placed blade.
            bool tooClose = false;
            for (int p = 0; p < placedN; ++p) {
                int dx = bx - placedX[p];
                int dz = bz - placedZ[p];
                if (dx*dx + dz*dz < MIN_SEP_SQ) { tooClose = true; break; }
            }
            if (tooClose) continue;

            // Height falls off with radial distance from cluster center.
            // Closest blades hit anchorH; ~1.3 voxels lost per unit r;
            // ±1 jitter for organic variation. Floor at 2.
            int falloff = int(std::round(r * 1.3f));
            int jitter  = int(rng() % 3) - 1;   // -1, 0, +1
            int h = std::clamp(anchorH - falloff + jitter, 2, anchorH);
            plantBlade(bx, bz, h);

            if (placedN < 16) {
                placedX[placedN] = bx;
                placedZ[placedN] = bz;
                placedN++;
            }
        }
    }
}
