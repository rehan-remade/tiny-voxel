#pragma once
#include <array>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>
#include <glm/glm.hpp>
#include "common.h"
#include "world.h"
#include "worldgen/worldgen.h"   // GenContext, GenStats, CaveLight

// Pure terrain helpers shared across the worldgen translation units.
// -- 3D noise (hash-based, trilinearly interpolated) --------------------------
inline float hash3(int x, int y, int z) {
    uint32_t h = uint32_t(x) * 73856093u
               ^ uint32_t(y) * 19349663u
               ^ uint32_t(z) * 83492791u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h = h ^ (h >> 16);
    return (h & 0xFFFFFFu) / float(0xFFFFFFu);
}
inline float smoothNoise3(float x, float y, float z) {
    int xi = int(std::floor(x)), yi = int(std::floor(y)), zi = int(std::floor(z));
    float fx = x - xi, fy = y - yi, fz = z - zi;
    float ux = fx * fx * (3.0f - 2.0f * fx);
    float uy = fy * fy * (3.0f - 2.0f * fy);
    float uz = fz * fz * (3.0f - 2.0f * fz);
    float c000 = hash3(xi,   yi,   zi  ), c100 = hash3(xi+1, yi,   zi  );
    float c010 = hash3(xi,   yi+1, zi  ), c110 = hash3(xi+1, yi+1, zi  );
    float c001 = hash3(xi,   yi,   zi+1), c101 = hash3(xi+1, yi,   zi+1);
    float c011 = hash3(xi,   yi+1, zi+1), c111 = hash3(xi+1, yi+1, zi+1);
    float a = glm::mix(c000, c100, ux);
    float b = glm::mix(c010, c110, ux);
    float c = glm::mix(c001, c101, ux);
    float d = glm::mix(c011, c111, ux);
    return glm::mix(glm::mix(a, b, uy), glm::mix(c, d, uy), uz);
}

// -- 2D noise -----------------------------------------------------------------
inline float hash2(int x, int z) {
    uint32_t h = uint32_t(x) * 374761393u + uint32_t(z) * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h = h ^ (h >> 16);
    return (h & 0xFFFFFF) / float(0xFFFFFF);
}
inline float smoothNoise(float x, float z) {
    int xi = int(std::floor(x)), zi = int(std::floor(z));
    float xf = x - xi, zf = z - zi;
    float a = hash2(xi,     zi);
    float b = hash2(xi + 1, zi);
    float c = hash2(xi,     zi + 1);
    float d = hash2(xi + 1, zi + 1);
    float u = xf * xf * (3 - 2 * xf);
    float v = zf * zf * (3 - 2 * zf);
    return glm::mix(glm::mix(a, b, u), glm::mix(c, d, u), v);
}
// (SURFACE_Y lives in common.h so the renderer can use it too.)

// Geological strata: each stone has a Gaussian "home depth" weight peaking
// at its preferred depth, plus an independent 3D noise field. The stone with
// the highest (home + noise) wins. Effect: a stone dominates in its home
// zone, but any stone can occasionally win elsewhere — producing veins,
// intrusions, and inclusions of any type at any depth, just like real
// igneous intrusions and xenoliths in sedimentary rock.
inline uint8_t stoneAt(int x, int y, int z) {
    float depth = float(SURFACE_Y - y);   // 0 at surface, positive below
    auto home = [](float d, float center, float spread) {
        float t = (d - center) / spread;
        return std::exp(-t * t);
    };
    float wTop  = home(depth,   60.0f, 240.0f);   // just below surface
    float wGrey = home(depth,  700.0f, 420.0f);
    float wDark = home(depth, 1800.0f, 600.0f);
    float wBed  = home(depth, 3000.0f, 900.0f);

    // Independent two-octave 3D noise per stone. Low octave defines broad
    // pockets/veins; high octave roughens the boundary at fine scale so
    // adjacent stones interleave grain-by-grain (interbedding, intrusion
    // fingers) rather than meeting along clean curves.
    const float Fl = 0.012f;   // low octave: broad shape (~83 vox period)
    const float Fh = 0.085f;   // high octave: fine speckle (~12 vox period)
    auto N = [&](float ox, float oy, float oz, float scaleJitter) {
        float lo = smoothNoise3(x * Fl + ox,  y * Fl + oy,  z * Fl + oz);
        float hi = smoothNoise3(x * Fh + ox*3.1f, y * Fh + oy*3.1f, z * Fh + oz*3.1f);
        return lo * 0.78f + hi * 0.22f + scaleJitter;
    };
    float nTop  = N(11.3f, 31.7f, 53.9f, 0.0f);
    float nGrey = N(71.5f, 23.1f, 17.7f, 0.0f);
    float nDark = N(41.9f, 89.3f, 67.5f, 0.0f);
    float nBed  = N(27.5f, 51.7f, 95.3f, 0.0f);

    // Amplitude tuned so home weight (~1.0 at home, ~0 far away) still
    // dominates inside the zone, but a strong noise peak in another stone
    // can win at a fringe / pocket location.
    const float AMP = 0.55f;
    float vTop  = wTop  + AMP * nTop;
    float vGrey = wGrey + AMP * nGrey;
    float vDark = wDark + AMP * nDark;
    float vBed  = wBed  + AMP * nBed;

    float best = vTop;  uint8_t mat = MAT_STONE;
    if (vGrey > best) { best = vGrey; mat = MAT_GREYSTONE; }
    if (vDark > best) { best = vDark; mat = MAT_DARKSTONE; }
    if (vBed  > best) { best = vBed;  mat = MAT_BEDROCK; }
    return mat;
}

// -- Biomes (POC) ------------------------------------------------------------
// A continuous climate field selects a BLEND of biomes (never a hard id), so
// terrain relief, grass tint, and later flora all transition smoothly across
// borders instead of seaming. Pure function of (x,z) -> deterministic and
// replayable like the rest of worldgen. The grass-tint values here are
// MIRRORED in voxel.frag (biomeGrassTint) so rendered tint lands in the same
// place as the biome's hills; keep the two copies in sync (same hash via
// smoothNoise, same frequency, same weights, same colors).
struct BiomeParams {
    float hillMega;      // relief amplitude on the mega (period 1024) octave
    float hillMacro;     // ... and the macro (period 512) octave
    float baseLift;      // overall surface elevation offset (voxels)
    glm::vec3 grassTint; // base grass color (the shader keeps its own copy)
    float treeChance;    // P(a tree spawns per tree-cell)
    float boulderScale;  // multiplier on boulders-per-cell
    float flowerScale;   // multiplier on flower density
    float rockiness;     // 1 = biome exposes bare rock on high ground
};

// One climate axis for the POC: "moisture", low frequency (~220m period) so
// biomes are large regions you walk across but borders still come within a
// reasonable stroll. CONTRAST-STRETCHED about 0.5: raw value noise clusters
// hard around the midpoint, so without this the field almost never reaches the
// highland/forest ends and the whole world reads as meadow. The stretch (with
// clamp) gives broad pure-biome interiors with blended borders between.
inline float biomeMoisture(int x, int z) {
    float raw = smoothNoise(x * (1.0f / 2200.0f) + 53.7f,
                            z * (1.0f / 2200.0f) + 19.2f);
    return std::clamp(0.5f + (raw - 0.5f) * 2.4f, 0.0f, 1.0f);
}

// Desert occupies the EXTREME-dry tail of the moisture axis (m < ~0.16), with
// highland in the mid-dry band above it. Returns the desert biome weight in
// [0,1] — used by fillColumn to pick the sand/sandstone surface skin. Mirror of
// the wDesert term in biomeAt / the shader's biomeWeights.
inline float biomeDesertW(int x, int z) {
    return glm::smoothstep(0.16f, 0.06f, biomeMoisture(x, z));
}

// Blended biome parameters at (x,z). Three POC biomes along the moisture axis,
// deliberately EXAGGERATED so borders are easy to read in-game (dial back once
// the blend looks right):
//   HIGHLAND (dry) : tall rocky relief, raised, pale olive grass
//   MEADOW   (mid) : gentle rolls, classic green   (== the prior meadow feel)
//   FOREST   (wet) : moderate rolls, slightly lower, darker richer green
inline BiomeParams biomeAt(int x, int z) {
    //                       hillMega hillMacro lift  grassTint                     tree  bldr flwr  rock
    const BiomeParams HIGHLAND{ 300.0f, 150.0f, 45.0f, glm::vec3(0.52f,0.52f,0.28f), 0.04f, 3.2f, 0.2f, 1.0f };
    const BiomeParams MEADOW  {  90.0f,  50.0f,  0.0f, glm::vec3(0.32f,0.62f,0.24f), 0.34f, 1.0f, 1.6f, 0.0f };
    const BiomeParams FOREST  { 150.0f,  75.0f,-14.0f, glm::vec3(0.12f,0.36f,0.18f), 0.94f, 0.4f, 0.3f, 0.0f };
    // DESERT (v1): moderate gentle relief (dunes come later), sits slightly low
    // so washes/oases have a place. No meadow flora (its own desert flora pass
    // comes in a later stage); tan tint feeds blended borders. rockiness 0.30
    // exposes bare sandstone on the rare high desert swell.
    const BiomeParams DESERT  { 140.0f,  60.0f,-25.0f, glm::vec3(0.85f,0.72f,0.48f), 0.0f,  0.0f, 0.0f, 0.30f };

    float m       = biomeMoisture(x, z);
    float wDesert = glm::smoothstep(0.16f, 0.06f, m);                      // extreme DRY tail
    float wHigh   = glm::smoothstep(0.42f, 0.30f, m)                       // dry, but...
                  * glm::smoothstep(0.10f, 0.20f, m);                      // ...fades OUT below desert
    float wForest = glm::smoothstep(0.58f, 0.70f, m);                      // high at the WET end
    float wMeadow = std::max(0.0f, 1.0f - wHigh - wForest - wDesert);
    float inv     = 1.0f / std::max(1e-4f, wHigh + wForest + wMeadow + wDesert);
    wHigh *= inv; wForest *= inv; wMeadow *= inv; wDesert *= inv;

    BiomeParams b;
    b.hillMega     = wHigh*HIGHLAND.hillMega     + wMeadow*MEADOW.hillMega     + wForest*FOREST.hillMega     + wDesert*DESERT.hillMega;
    b.hillMacro    = wHigh*HIGHLAND.hillMacro    + wMeadow*MEADOW.hillMacro    + wForest*FOREST.hillMacro    + wDesert*DESERT.hillMacro;
    b.baseLift     = wHigh*HIGHLAND.baseLift     + wMeadow*MEADOW.baseLift     + wForest*FOREST.baseLift     + wDesert*DESERT.baseLift;
    b.grassTint    = wHigh*HIGHLAND.grassTint    + wMeadow*MEADOW.grassTint    + wForest*FOREST.grassTint    + wDesert*DESERT.grassTint;
    b.treeChance   = wHigh*HIGHLAND.treeChance   + wMeadow*MEADOW.treeChance   + wForest*FOREST.treeChance   + wDesert*DESERT.treeChance;
    b.boulderScale = wHigh*HIGHLAND.boulderScale + wMeadow*MEADOW.boulderScale + wForest*FOREST.boulderScale + wDesert*DESERT.boulderScale;
    b.flowerScale  = wHigh*HIGHLAND.flowerScale  + wMeadow*MEADOW.flowerScale  + wForest*FOREST.flowerScale  + wDesert*DESERT.flowerScale;
    b.rockiness    = wHigh*HIGHLAND.rockiness    + wMeadow*MEADOW.rockiness    + wForest*FOREST.rockiness    + wDesert*DESERT.rockiness;
    return b;
}

inline float smoothstep01(float t) { t = std::clamp(t, 0.0f, 1.0f); return t * t * (3.0f - 2.0f * t); }

// Out-params from the desert height field so fillColumn picks materials that
// match the SHAPE exactly (sand only where there's a dune, etc.) — filled by
// heightAtEx, never recomputed in the per-voxel loop.
struct DesertFields {
    float dW = 0.0f, wErg = 0.0f, duneH = 0.0f;
    float onCap = 0.0f, onCliff = 0.0f, onTalus = 0.0f;   // mesa caprock / cliff / talus weights
    float oasisW = 0.0f, waterY = 0.0f;                   // oasis dish weight + still-water level
};

// Erg-scale wind bearing: a single very-low-freq sample (~600 m period) so dune
// crestlines march PARALLEL across a whole sand-sea and only rotate between
// distant ergs — never a per-cell random direction (which reads as chaos).
inline void desertWind(int x, int z, float& cx, float& cz) {
    float wn = smoothNoise(x * (1.0f / 6000.0f) + 8.3f, z * (1.0f / 6000.0f) + 71.1f);
    float th = wn * 6.2831853f;
    cx = std::cos(th); cz = std::sin(th);
}

// The biome-blended BASE surface (gentle texture + biome relief + lift), with
// NO desert terms. The single shared core every height path routes through, so
// brick-corner samples, per-voxel fill, the snapshot, and the height texture
// can never disagree (no seams by construction).
inline float baseHAt(int x, int z) {
    float mega  = smoothNoise(x * (1.0f / 1024.0f), z * (1.0f / 1024.0f));
    float macro = smoothNoise(x * (1.0f /  512.0f), z * (1.0f /  512.0f));
    float a     = smoothNoise(x * (1.0f /  192.0f), z * (1.0f /  192.0f));
    float b     = smoothNoise(x * (1.0f /   64.0f), z * (1.0f /   64.0f));
    float c     = smoothNoise(x * (1.0f /   24.0f), z * (1.0f /   24.0f));
    float gentle = (a - 0.5f) * 40.0f + (b - 0.5f) * 14.0f + (c - 0.5f) * 5.0f;
    // Micro-relief: high-frequency RIDGED noise (1-|2n-1|) so the ground breaks
    // into clumpy little hummocks/mounds instead of reading as smooth pancake
    // contour steps. Two octaves (~12 and ~6 voxel features), a couple of
    // voxels tall — small enough to keep terrain walkable, big enough that no
    // patch of ground is ever dead flat. This is the cheap anti-pancake win.
    float m1 = smoothNoise(x * (1.0f / 12.0f) + 71.0f, z * (1.0f / 12.0f) + 33.0f);
    float m2 = smoothNoise(x * (1.0f /  6.0f) + 11.0f, z * (1.0f /  6.0f) + 90.0f);
    float ridged = (1.0f - std::fabs(2.0f * m1 - 1.0f)) * 2.4f
                 + (1.0f - std::fabs(2.0f * m2 - 1.0f)) * 1.1f;
    BiomeParams bio = biomeAt(x, z);
    float relief = (mega - 0.5f) * bio.hillMega + (macro - 0.5f) * bio.hillMacro;
    return float(SURFACE_Y) + gentle + relief + ridged + bio.baseLift;
}

// One mesa SITE per lattice cell; scan the 3x3 neighbourhood, keep the TALLEST.
// Real mesas are NOT drums: the footprint is ELONGATED + irregular (rotated,
// squashed, noise-eroded rim), and the wall is STRATIFIED into stepped LEDGES —
// a flat caprock, a TERRACED cliff of ~3 benches, then a wide flared talus
// apron. `frac` is the stepped height fraction (1 at cap -> 0 at base); capY is
// the absolute caprock elevation (sampled at center -> dead-flat top, no tilt,
// bit-identical via baseHAt -> no seam).
struct MesaHit { float capY = 0.0f, frac = 0.0f, wCap = 0.0f, wWall = 0.0f, wTalus = 0.0f; bool any = false; };
inline MesaHit mesaAt(int x, int z) {
    const float INV = 1.0f / 2600.0f;   // mesa lattice — 260 m cells
    MesaHit best;
    int cx0 = int(std::floor(float(x) * INV)), cz0 = int(std::floor(float(z) * INV));
    for (int dz = -1; dz <= 1; ++dz)
    for (int dx = -1; dx <= 1; ++dx) {
        int cx = cx0 + dx, cz = cz0 + dz;
        if (hash2(cx * 131 + 7, cz * 131 + 3) <= 0.86f) continue;        // ~14% of cells host one -> rare landmark
        float R = 110.0f + 110.0f * hash2(cx * 7, cz * 7);               // footprint 110..220 vox
        float ctrX = (float(cx) + 0.22f + 0.56f * hash2(cx, cz))          / INV;
        float ctrZ = (float(cz) + 0.22f + 0.56f * hash2(cx + 99, cz - 99)) / INV;
        float ddx = float(x) - ctrX, ddz = float(z) - ctrZ;
        if (std::fabs(ddx) > R + 170.0f || std::fabs(ddz) > R + 170.0f) continue;   // cheap reject before sqrt
        if (biomeDesertW(int(ctrX), int(ctrZ)) < 0.5f) continue;   // mesas only in FIRM desert (not the grassy border)
        // ELONGATED, ROTATED, NOISE-ERODED footprint (a real mesa plan, not a
        // circle): rotate into the site frame, squash one axis, then warp the
        // radius with two noise octaves for lobes + fingers.
        float ang    = hash2(cx * 3, cz * 9) * 6.2831853f;
        float aspect = 0.55f + 0.85f * hash2(cx * 11, cz * 4);          // 0.55..1.40 elongation
        float ca = std::cos(ang), sa = std::sin(ang);
        float lx = (ddx * ca + ddz * sa) / aspect;
        float lz = (-ddx * sa + ddz * ca) * aspect;
        float d = std::sqrt(lx * lx + lz * lz)
                + (smoothNoise(x * (1.0f / 120.0f) + 200.0f, z * (1.0f / 120.0f) + 90.0f) - 0.5f) * 0.42f * R   // big lobes / embayments
                + (smoothNoise(x * (1.0f /  42.0f) +  11.0f, z * (1.0f /  42.0f) +  3.0f) - 0.5f) * 0.16f * R;  // finer fingers
        float u = d / (R + 60.0f);
        if (u >= 1.15f) continue;
        // CONTINUOUS stratified profile m(u) in [0,1]: flat caprock -> terraced
        // cliff (benches) stepping down to the talus TOP -> a wide scree apron
        // sloping smoothly to 0 so the base EMERGES into the plain (no trench).
        const float TALUS_TOP = 0.18f;                                  // talus height as a fraction of capRise
        float m, wCap = 0.0f, wWall = 0.0f, wTalus = 0.0f;
        if (u < 0.50f) {
            m = 1.0f; wCap = 1.0f;                                       // flat caprock
        } else if (u < 0.82f) {
            float sm = 1.0f - (u - 0.50f) / 0.32f;                       // 1->0 down the cliff
            const float STEPS = 3.0f;                                    // ~3 benches
            float lvl = std::floor(sm * STEPS);
            float frc = sm * STEPS - lvl;
            float stepped = (lvl + smoothstep01((frc - 0.5f) / 0.30f)) / STEPS;  // 0..1 staircase
            m = TALUS_TOP + (1.0f - TALUS_TOP) * stepped;               // cap(1) -> talus top, in steps
            wWall = 1.0f;
        } else {
            m = TALUS_TOP * (1.0f - smoothstep01((u - 0.82f) / 0.33f));  // talus slopes TALUS_TOP -> 0 into the plain
            wTalus = 1.0f;
        }
        float capRise = 140.0f + 90.0f * hash2(cx * 5, cz * 13);         // table height 140..230 vox
        float capY = baseHAt(int(ctrX), int(ctrZ)) + capRise;
        if (!best.any || capY > best.capY) {
            best.capY = capY; best.frac = m; best.wCap = wCap; best.wWall = wWall; best.wTalus = wTalus; best.any = true;
        }
    }
    return best;
}

// OASIS site field — the rare lush reward. ONE pure-(x,z) function feeds the
// basin (heightAt), the static water fill (genOasis), and the palm/green ring
// (genBushes), so they agree by construction. Coarse 3000-vox lattice, ~7% of
// cells host, then thinned by FIRM-desert + no-mesa + low/wet gates so a real
// oasis is several km apart. Helpers split so the per-column hot path pays only
// cheap host-hash + bbox before the heavy gates.
constexpr int   OASIS_CELL  = 2400;    // oasis lattice cell (smaller -> less rare than the old 3000)
constexpr float OASIS_INV   = 1.0f / float(OASIS_CELL);
constexpr float OASIS_DEPTH = 26.0f;   // bowl floor dug below rimMin (vox)
constexpr int   WATER_BELOW_RIM  = 4;  // still-water level below the LOWEST rim sample (covers float->int slop + margin)
constexpr int   MAX_RIM_SPREAD   = 28; // rimMax-rimMin cap: rejects steep slopes where the berm would wall up
constexpr int   OASIS_RIM_SPOKES = 12; // ring samples for rimMin/rimMax
constexpr float OASIS_BERM_BAND  = 0.18f; // berm ramps up over uw in [1-band,1], peaks at the shore, tapers by ~1.25
inline bool oasisHost(int cx, int cz) { return hash2(cx * 191 + 13, cz * 191 + 5) > 0.82f; }  // ~18% of cells host a candidate
inline void oasisCenter(int cx, int cz, float& ctrX, float& ctrZ, float& R) {
    R    = 90.0f + 40.0f * hash2(cx * 7, cz * 7);                       // pool+bank footprint 90..130
    ctrX = (float(cx) + 0.22f + 0.56f * hash2(cx, cz))          / OASIS_INV;
    ctrZ = (float(cz) + 0.22f + 0.56f * hash2(cx + 99, cz - 99)) / OASIS_INV;
}
// One pure (cx,cz) site evaluation: center, footprint, the 12-spoke baseHAt rim
// ring (rimMin = the LOWEST surrounding lip), the basin gates, and the resulting
// flat water level. The SINGLE source of truth shared by oasisAt / genOasis /
// palms / the debug scan, so every path agrees bit-for-bit (no seam). Pure fn of
// the center => constant across the whole oasis. waterY is tied to the MINIMUM
// rim, never the center, so a pool can NEVER perch on a local rise.
struct OasisSite { bool pass = false; float ctrX = 0, ctrZ = 0, R = 0, rimMin = 0; int waterY = 0; };
inline OasisSite oasisSite(int cx, int cz) {
    OasisSite s;
    if (!oasisHost(cx, cz)) return s;
    oasisCenter(cx, cz, s.ctrX, s.ctrZ, s.R);
    if (biomeDesertW(int(s.ctrX), int(s.ctrZ)) < 0.90f) return s;       // DEEP desert interior (not the meadow border)
    if (mesaAt(int(s.ctrX), int(s.ctrZ)).any)            return s;       // no table-in-a-pond
    float rimMin = 1e9f, rimMax = -1e9f, rimSum = 0.0f;                  // lowest/highest/avg surrounding lip
    for (int k = 0; k < OASIS_RIM_SPOKES; ++k) {
        float a  = 6.2831853f * float(k) / float(OASIS_RIM_SPOKES);
        float rx = s.ctrX + std::cos(a) * s.R, rz = s.ctrZ + std::sin(a) * s.R;
        float h  = baseHAt(int(rx), int(rz));
        rimMin = std::min(rimMin, h); rimMax = std::max(rimMax, h); rimSum += h;
    }
    float rimAvg  = rimSum / float(OASIS_RIM_SPOKES);
    float centerH = baseHAt(int(s.ctrX), int(s.ctrZ));
    if (centerH > rimAvg)                        return s;               // center above the avg rim -> a rise/dome, reject (the perched cause)
    if (rimMax - rimMin > float(MAX_RIM_SPREAD)) return s;               // steep/uneven rim -> tall berm wall, reject
    s.rimMin = rimMin;
    s.waterY = int(std::floor(rimMin)) - WATER_BELOW_RIM;
    s.pass   = true;
    return s;
}
struct OasisHit { bool any = false; float ctrX = 0, ctrZ = 0, R = 0, w = 0, bw = 0, rimMin = 0, berm = 0; int waterY = 0; };
inline OasisHit oasisAt(int x, int z) {
    int cx0 = int(std::floor(float(x) * OASIS_INV)), cz0 = int(std::floor(float(z) * OASIS_INV));
    OasisHit best;
    for (int dz = -1; dz <= 1; ++dz)
    for (int dx = -1; dx <= 1; ++dx) {
        int cx = cx0 + dx, cz = cz0 + dz;
        if (!oasisHost(cx, cz)) continue;                              // cheap host hash FIRST
        float ctrX, ctrZ, R; oasisCenter(cx, cz, ctrX, ctrZ, R);
        float ddx = float(x) - ctrX, ddz = float(z) - ctrZ;
        if (std::fabs(ddx) > R * 1.8f || std::fabs(ddz) > R * 1.8f) continue;  // bbox reject BEFORE heavy work
        float u = std::sqrt(ddx * ddx + ddz * ddz) / R;
        if (u >= 1.8f) continue;
        OasisSite s = oasisSite(cx, cz);                               // 12-spoke ring + basin gates (center-constant)
        if (!s.pass) continue;
        // Organic shoreline: warp the radius by noise sampled ON A CIRCLE of the
        // bearing angle, so it is exactly periodic in angle (no +-pi seam) and a
        // pure fn of (x,z)+center. Two lobes: broad + finer fingers.
        float ang  = std::atan2(ddz, ddx);
        float wob  = smoothNoise(ctrX * 0.01f +  6.0f * std::cos(ang) + 31.0f, ctrZ * 0.01f +  6.0f * std::sin(ang) + 17.0f);
        float wob2 = smoothNoise(ctrX * 0.01f + 13.0f * std::cos(ang) +  5.0f, ctrZ * 0.01f + 13.0f * std::sin(ang) +  9.0f);
        float uw = u * (1.0f + 0.30f * (wob - 0.5f)) + 0.12f * (wob2 - 0.5f);
        // Dish w (drives the dig): flat floor -> C1 ramp to 0 at the warped rim.
        // w>0 EXACTLY iff uw<1, so genOasis uses it to bound water inside the shore.
        float w = (uw < 0.55f) ? 1.0f : (uw < 1.0f ? 1.0f - smoothstep01((uw - 0.55f) / 0.45f) : 0.0f);
        // Berm band weight bw: 0 below (1-band), ramps to 1 AT the shore, tapers
        // out by ~1.25 so the bank rises to rimMin then settles back to desert.
        float bw;
        if      (uw < 1.0f - OASIS_BERM_BAND) bw = 0.0f;
        else if (uw < 1.0f)                   bw = smoothstep01((uw - (1.0f - OASIS_BERM_BAND)) / OASIS_BERM_BAND);
        else if (uw < 1.25f)                  bw = 1.0f - smoothstep01((uw - 1.0f) / 0.25f);
        else                                  bw = 0.0f;
        // Berm uses THIS column's baseH (continuous in x,z) -> lifts the carved
        // surface toward rimMin at EVERY azimuth, not just sampled spokes.
        float berm = std::max(0.0f, s.rimMin - baseHAt(x, z)) * bw;
        if (!best.any || w > best.w) {           // keep the DEEPEST dish (deterministic)
            best.any = true; best.ctrX = ctrX; best.ctrZ = ctrZ; best.R = R;
            best.w = w; best.bw = bw; best.rimMin = s.rimMin; best.berm = berm; best.waterY = s.waterY;
        }
    }
    return best;
}

// Additive desert height delta. Everything scales by the desert weight dW so it
// fades to exactly 0 at the highland border (seam-free). A single cheap gate
// (dW<=0.001) makes the non-desert world pay ~nothing.
inline float desertTerms(int x, int z, float baseH, DesertFields* f) {
    float dW = biomeDesertW(x, z);
    if (f) f->dW = dW;
    if (dW <= 0.001f) return 0.0f;                 // cheap gate: most of the world bails here

    // Erg mask: dunes cluster into ~30% of the desert (sand-seas), the rest is
    // flat reg. Primary low-freq lobe + a crenulation octave for organic shores.
    float em = 0.85f * smoothNoise(x * (1.0f / 1700.0f) + 200.5f, z * (1.0f / 1700.0f) + 12.7f)
             + 0.15f * smoothNoise(x * (1.0f /  430.0f) +   5.0f, z * (1.0f /  430.0f) +  9.0f);
    float wErg = smoothstep01((em - 0.50f) / 0.16f);
    if (f) f->wErg = wErg;

    // NO dedicated dune term. Any explicit dune function — periodic sawtooth OR
    // anisotropic FBM — reads as artificially structured ridges (equidistant
    // combs / parallel grooves). The gentle ISOTROPIC base relief (baseHAt's
    // noise octaves) already reads as natural rolling dunes once it's skinned
    // with sand, so the dunes ARE the base terrain. wErg survives only to select
    // the SAND surface material (sand-sea vs gravel reg); it adds no height.
    float dune = 0.0f;
    if (f) f->duneH = 0.0f;


    // Mesa: clamp the column UP toward an absolute flat caprock (dead-flat table
    // over a steep wall over a talus skirt). Additive + C1, so heightAt stays
    // continuous and brick-corner classification sees a real ramp (no float).
    float mesaLift = 0.0f;
    MesaHit ms = mesaAt(x, z);
    if (ms.any) {
        // Stepped lift toward the absolute caprock: frac=1 -> capY (dead-flat
        // top), benches step between, talus a low apron. (capY-baseH may be
        // negative in a high spot -> clamp so the mesa only ever adds.)
        mesaLift = std::max(0.0f, ms.frac * (ms.capY - baseH));
        if (f) { f->onCap = ms.wCap; f->onCliff = ms.wWall; f->onTalus = ms.wTalus; }
    }

    // Oasis basin: the dish weight w (1 floor -> 0 rim) digs the bowl DOWN; the
    // berm lifts the rim band UP to >= rimMin (the lowest surrounding lip), using
    // each column's own baseH so the lift holds at EVERY azimuth -> the still
    // water level (rimMin - WATER_BELOW_RIM) is provably below land on all sides,
    // never perched. Mesa is suppressed by (1-w). Shares the *dW master fade.
    float oasisDip = 0.0f, oasisBerm = 0.0f;
    OasisHit oa = oasisAt(x, z);
    if (oa.any) {
        oasisDip  = -OASIS_DEPTH * oa.w;         // dig the bowl floor
        oasisBerm = oa.berm;                     // raise the rim band toward rimMin
        mesaLift *= (1.0f - oa.w);
        if (f) {
            f->onCap *= (1.0f - oa.w); f->onCliff *= (1.0f - oa.w); f->onTalus *= (1.0f - oa.w);
            f->oasisW = oa.w; f->waterY = float(oa.waterY);
        }
    }

    return (dune + mesaLift + oasisDip + oasisBerm) * dW;     // master fade -> seam-free at the border
}

// Canonical surface height: base core + desert delta. (heightAtEx returns the
// desert fields for fillColumn; identical math.)
inline int heightAt(int x, int z) {
    float baseH = baseHAt(x, z);
    int h = int(baseH + desertTerms(x, z, baseH, nullptr));
    return std::clamp(h, 1, GRID_Y - 1);
}
inline int heightAtEx(int x, int z, DesertFields& f) {
    float baseH = baseHAt(x, z);
    int h = int(baseH + desertTerms(x, z, baseH, &f));
    return std::clamp(h, 1, GRID_Y - 1);
}
constexpr int H_NOISE_MARGIN = 8;

// -- Shared surface-column grammar (fillColumn + the far-field LOD ring) -------
// The per-column surface decision fillColumn makes, factored out so the coarse
// ring generator (rings.cpp) samples the EXACT same material rules as the fine
// world and the two can never drift. CONTRACT: any per-(x,z)-pure surface
// content added to heightAtEx / this grammar automatically appears in the far
// field with zero LOD work; content that is NOT per-column pure (agent-carved
// caves, future rivers) needs its own ring story.
struct SurfaceCol {
    int  h = 0;              // heightAt surface height
    DesertFields df;         // erg/mesa/oasis fields (incl. waterY)
    bool rocky = false;      // bare-rock dither (highland peaks / wind-scoured)
    bool desert = false;     // desert surface-skin dither
};
inline SurfaceCol surfaceColAt(int wx, int wz) {
    SurfaceCol s;
    s.h = heightAtEx(wx, wz, s.df);
    // Highland exposes BARE ROCK on its high ground: rockiness rises with both
    // highland-ness and elevation, with a noise patch boundary so the rock/turf
    // edge isn't a clean contour (flora gates require grass -> peaks self-clear).
    float rockAmt = std::clamp(
        biomeAt(wx, wz).rockiness *
        (float(s.h - SURFACE_Y) / 220.0f - 0.30f) * 2.4f, 0.0f, 1.0f);
    s.rocky = smoothNoise(wx * 0.06f, wz * 0.06f) < rockAmt;
    // Desert skin dithered against the desert weight so the border stipples
    // instead of drawing a hard contour line.
    s.desert = smoothNoise(wx * 0.06f + 113.0f, wz * 0.06f + 47.0f)
             < biomeDesertW(wx, wz);
    return s;
}
// Material at (wx, y, wz) for y < s.h — EXACTLY fillColumn's rules (moved here
// verbatim; see fillColumn for the surface-skin design notes).
inline uint8_t surfaceMatAt(int wx, int y, int wz, const SurfaceCol& s) {
    uint8_t mat = stoneAt(wx, y, wz);
    bool onMesa  = (s.df.onCap > 0.5f) || (s.df.onCliff > 0.05f);
    bool onCap   = (s.df.onCap > 0.5f);
    bool onTalus = (s.df.onTalus > 0.3f) && !onMesa;
    if (onMesa) {
        // Sandstone mesa, INDEPENDENT of the desert/meadow dither so a border
        // mesa never gets a grassy cap. Banded by world-Y in the shader.
        if (s.h - y <= 40 || y > s.h - (onCap ? 4 : 12)) mat = MAT_SANDSTONE;
    } else if (onTalus) {
        if (y == s.h - 1)      mat = MAT_DESERT_PAVEMENT;          // scree surface
        else if (s.h - y <= 8) mat = MAT_SANDSTONE;                // rubble band
    } else if (s.desert) {
        if (s.h - y <= 30) mat = MAT_SANDSTONE;        // shallow warm band
        bool onSand = (s.df.wErg > 0.35f);             // inside a sand-sea (erg)
        if (onSand) {
            // Pale-gold sand skin over the base swells, fixed shallow depth.
            int sandDepth = 8;
            if (y > s.h - sandDepth) mat = MAT_SAND_STATIC;
        } else if (s.rocky) {
            if (y > s.h - 6) mat = MAT_SANDSTONE;      // bare wind-scoured outcrop
        } else if (y == s.h - 1) {
            mat = MAT_DESERT_PAVEMENT;                 // reg gravel surface (default)
        }
    } else if (s.rocky) {
        if (y > s.h - 5) mat = MAT_GREYSTONE;   // bare rock cap
    } else if (y == s.h - 1) mat = MAT_GRASS;
    else if (y > s.h - 5)    mat = MAT_DIRT;
    return mat;
}

// Uniform random in [0,1) / [-1,1). Each phase owns its seeded rng, so pass by ref.
inline float frand01(std::mt19937& rng) { return float(rng() & 0xFFFFFF) / float(0xFFFFFF); }
inline float frand11(std::mt19937& rng) { return frand01(rng) * 2.0f - 1.0f; }

// -- Deterministic per-feature seeding ----------------------------------------
// Every feature (a tree, a boulder cluster, a cave system, a deferred cave
// decoration) derives its own RNG stream from (world seed, phase, key...).
// Nothing shares a sequential stream across features, so any feature can be
// regenerated in isolation — the prerequisite for chunked generation and for
// running feature passes in parallel. The seed itself lives in GenContext
// (worldgen.h); these helpers take it explicitly so they stay pure.

enum : uint32_t {   // phase ids for featureSeed (stable — used in derivations)
    SEED_BOULDER = 1, SEED_BUSH, SEED_TREE, SEED_FLOWER_SCATTER, SEED_FLOWER,
    SEED_GRASS, SEED_CAVE_SITE, SEED_CAVE_SYSTEM, SEED_CAVE_POST,
    SEED_DESERT_SHRUB, SEED_SAGUARO, SEED_OCOTILLO, SEED_OASIS_PALM,
    SEED_CREATURE,   // runtime creature herd cells (no voxels -> no GEN_VERSION)
};

inline uint64_t mix64(uint64_t x) {   // SplitMix64 finalizer
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
inline uint32_t featureSeed(uint64_t worldSeed, uint32_t phase, int a, int b, int c = 0) {
    uint64_t h = worldSeed;
    h = mix64(h ^ uint64_t(phase));
    h = mix64(h ^ uint64_t(uint32_t(a)));
    h = mix64(h ^ uint64_t(uint32_t(b)));
    h = mix64(h ^ uint64_t(uint32_t(c)));
    return uint32_t(h);
}
// featureSeed as a uniform float in [0,1) — for pure scatter decisions
// (e.g. "does a flower spawn at this column?") with no mt19937 needed.
inline float featureChance(uint64_t worldSeed, uint32_t phase, int a, int b, int c = 0) {
    return float(featureSeed(worldSeed, phase, a, b, c) & 0xFFFFFFu) / float(0x1000000);
}

// -- Shared tree-site derivation (genTrees + the far-field LOD ring) ------------
// The first SIX draws of a tree cell's SEED_TREE stream (spawn chance, x/z
// jitter, size roll, trunk length/thickness), factored out so the ring
// generator can re-derive every tree's position and dimensions WITHOUT running
// chunk gen. genTrees passes its own rng — grow() then continues from draw 7
// exactly as before, so the fine path stays bit-identical (contentHash-gated).
// DRAW ORDER IS THE API CONTRACT: reordering these draws relayouts every
// forest (= GEN_VERSION bump, saved edits discarded).
// The grass gate differs by caller: fine gen gates on the pre-stage surface
// SNAPSHOT (order-consistency there is load-bearing); the ring gates on the
// pure grammar (surfaceColAt), which agrees ~99% — the residual divergence
// only mis-places the odd impostor beyond 100 m.
constexpr int TREE_CELL   = 55;   // lattice pitch; anchored at TREE_ANCHOR + k*TREE_CELL
constexpr int TREE_ANCHOR = 16;
// Canopy clearance: the lowest branch attach point is clamped to at least
// this many voxels above the trunk base (unless the tree is so small that
// ~90% of its trunk is lower). The character is 18 voxels tall (P_HEIGHT) —
// with the pathfinder's headroom rule, foliage below ~22 turns every tree
// crown into a wall at head height; lifting the crown lets him walk UNDER
// trees and only route around trunks. Shared by the fine L-system
// (flora.cpp) and the ring impostors (rings.cpp): both must agree or trees
// visibly drop their crowns at the LOD handoff.
constexpr float TREE_CANOPY_CLEAR = 24.0f;
struct TreeSite {
    bool  spawn = false;
    int   x = 0, z = 0, h = 0;          // trunk base (h = surface height there)
    float scale = 0.0f;                 // 0.80..1.55 whole-tree scale
    float trunkLen = 0.0f, trunkThick = 0.0f;
};
template <typename GrassGate>   // bool(int x, int h, int z): grass under the trunk?
inline TreeSite deriveTreeSite(uint64_t seed, int tx, int tz, std::mt19937& rng,
                               GrassGate&& grassTop) {
    TreeSite s;
    rng.seed(featureSeed(seed, SEED_TREE, tx, tz));
    // BIOME density: forest dense, meadow scattered, highland bare. Gated at
    // the cell anchor so every replayer of this cell agrees.
    if (frand01(rng) >= biomeAt(tx, tz).treeChance) return s;          // draw 1
    s.x = tx + int(rng() % std::max(1, TREE_CELL - 10));               // draw 2
    s.z = tz + int(rng() % std::max(1, TREE_CELL - 10));               // draw 3
    s.h = heightAt(s.x, s.z);
    if (s.h < 1 || s.h >= GRID_Y - 240) return s;
    if (!grassTop(s.x, s.h, s.z)) return s;
    // Mild bias toward larger trees: saplings rarer than mediums and giants.
    float sRoll  = frand01(rng);                                       // draw 4
    s.scale      = 0.80f + 0.75f * std::sqrt(sRoll);
    // Longer trunks than the original 22+13: the canopy-clearance clamp
    // (TREE_CANOPY_CLEAR) needs saplings tall enough to hold their crowns
    // over the character's head. Same draw count — the RNG contract holds.
    s.trunkLen   = (28.0f + frand01(rng) * 13.0f) * s.scale;           // draw 5
    s.trunkThick = (2.5f  + frand01(rng) * 1.5f)  * s.scale;           // draw 6
    s.spawn = true;
    return s;
}

// (floorDiv lives in common.h — lattice math over unbounded coords.)

// -- Generation bounds ---------------------------------------------------------
// Voxel-space XZ bounds (full height implied) one generation call covers.
// Coordinates are UNBOUNDED world voxels (6e-3a); the resident window is the
// caller's concern. "Full window" (a window-sized rect) selects the
// monolithic single-pass paths in each stage.
struct GenBounds {
    int x0 = 0, z0 = 0, x1 = GRID_XZ, z1 = GRID_XZ;
    bool isFullWorld() const {
        return x1 - x0 == GRID_XZ && z1 - z0 == GRID_XZ;
    }
};

// -- Buffered chunk-stage writes (order-free, parallel-safe) --------------------
// A chunk regenerates every feature whose footprint can reach it (margin
// replay, in global lattice order); ALL of its writes — in and out of bounds —
// go into its ChunkStageBuffer, and reads consult the buffer FIRST. So:
//  - in-bounds conflicts between features resolve in global feature order
//    (every replayer replays all contenders), exactly like the monolithic
//    pass;
//  - the live world is never touched during compute, so chunks of one stage
//    compute in PARALLEL and in ANY ORDER (workers, streaming, re-entry all
//    get identical content — proven by forward/reverse hash equality);
//  - commit applies only in-bounds entries (disjoint across chunks, so commit
//    order is irrelevant) and folds the per-chunk stat deltas + cave lights
//    into the context in chunk order.
// Reads of PREVIOUS stages hit the live world (stable during a stage — the
// chunk-status ladder completes each stage everywhere before the next).
// Writes are stored as per-BRICK pages (512 data bytes + written bitmask),
// not per-voxel map nodes: margin replay buffers ~16x redundant writes, and
// voxel-granular nodes meant hundreds of millions of heap allocations whose
// lock contention erased the worker parallelism. Pages allocate once per
// touched brick, and — because chunk bounds are brick-aligned — commit
// rejects each out-of-bounds page with ONE test per 512 voxels.
struct StagePage {
    std::array<uint8_t, BRICK_VOX>      data;    // guarded by mask; not zeroed
    std::array<uint64_t, BRICK_VOX/64>  mask{};  // 1 bit per written voxel
};

struct ChunkStageBuffer {
    GenBounds b;
    std::unordered_map<int64_t, StagePage> pages;  // key = packed brick coords
    GenStats stats;                                // per-chunk deltas (commit folds)
    std::vector<CaveLight> lights;                 // per-chunk lights (commit appends)

    // World-coordinate brick key, 21-bit biased XZ fields: unique within
    // +-1M bricks of the origin (the resident window stays far inside that).
    static int64_t pageKey(int bx, int by, int bz) {
        return (int64_t(by) << 42)
             | (int64_t((bx + (1 << 20)) & 0x1FFFFF) << 21)
             |  int64_t((bz + (1 << 20)) & 0x1FFFFF);
    }
    static int keyBY(int64_t k) { return int(k >> 42); }
    static int keyBX(int64_t k) { return int((k >> 21) & 0x1FFFFF) - (1 << 20); }
    static int keyBZ(int64_t k) { return int(k         & 0x1FFFFF) - (1 << 20); }

    // One-entry page cache: feature writes/probes are spatially local, so
    // most accesses revisit the last brick. cachePage == nullptr caches a
    // confirmed ABSENT page (the common case for probes into empty space).
    // unordered_map nodes are pointer-stable, so inserts never invalidate it.
    mutable int64_t    cacheKey  = INT64_MIN;
    mutable StagePage* cachePage = nullptr;

    // Buffer-only probe: true (with the value) iff this buffer holds a write
    // at the position; NO world fallback (callers compose their own). XZ is
    // unbounded — only y is guarded; the window clips at commit/world reads.
    bool tryGet(int x, int y, int z, uint8_t& out) const {
        if (y < 0 || y >= GRID_Y) return false;
        int64_t k = pageKey(x >> BRICK_BITS, y >> BRICK_BITS, z >> BRICK_BITS);
        if (k != cacheKey) {
            auto it = pages.find(k);
            cacheKey  = k;
            cachePage = (it == pages.end())
                      ? nullptr : const_cast<StagePage*>(&it->second);
        }
        if (!cachePage) return false;
        int v = voxInBrick(x & BRICK_MASK, y & BRICK_MASK, z & BRICK_MASK);
        if (!(cachePage->mask[v >> 6] & (1ull << (v & 63)))) return false;
        out = cachePage->data[v];
        return true;
    }
    uint8_t get(const World& w, int x, int y, int z) const {
        uint8_t m;
        return tryGet(x, y, z, m) ? m : w.getVoxel(x, y, z);
    }
    void set(int x, int y, int z, uint8_t m) {
        if (y < 0 || y >= GRID_Y) return;
        int64_t k = pageKey(x >> BRICK_BITS, y >> BRICK_BITS, z >> BRICK_BITS);
        if (k != cacheKey || !cachePage) {
            cacheKey  = k;
            cachePage = &pages[k];
        }
        int v = voxInBrick(x & BRICK_MASK, y & BRICK_MASK, z & BRICK_MASK);
        cachePage->data[v] = m;
        cachePage->mask[v >> 6] |= 1ull << (v & 63);
    }
    // Commit applies in-bounds pages to the world; with `undo`, each
    // overwritten value is recorded first (see UndoBand). worldgen.cpp.
    void commit(World& w, GenContext& ctx, struct UndoBand* undo = nullptr) const;
};

// Rolling pre-decoration overlay for the ladder/streaming drivers (6e).
// Decoration probes CASCADE (a floor probe positions every later write), so
// they must read CARVE-ONLY state — the batch drivers guarantee it globally
// (all decor buffers compute before any commit), but at a moving front the
// strips behind the decor front already hold committed decor. Fix: decor
// commits record each voxel's PRE-commit value here (first writer wins,
// since later writers overwrite decor, not carve state), and decoration's
// world-fallback reads check this overlay first — reconstructing the
// carve-only world exactly within probe reach. Strips are evicted once the
// front moves beyond CAVE_DECO probe reach (~4 strips). A brick belongs to
// exactly one strip (commits clip to brick-aligned strip bounds), so
// per-strip eviction is exact. Read-only during parallel computes; mutated
// only between batches (serial commits + eviction).
struct UndoBand {
    std::unordered_map<int64_t, StagePage> pages;
    std::unordered_map<int, std::vector<int64_t>> stripKeys;  // strip -> keys added

    bool tryGet(int x, int y, int z, uint8_t& out) const {
        if (y < 0 || y >= GRID_Y) return false;
        auto it = pages.find(ChunkStageBuffer::pageKey(
            x >> BRICK_BITS, y >> BRICK_BITS, z >> BRICK_BITS));
        if (it == pages.end()) return false;
        int v = voxInBrick(x & BRICK_MASK, y & BRICK_MASK, z & BRICK_MASK);
        if (!(it->second.mask[v >> 6] & (1ull << (v & 63)))) return false;
        out = it->second.data[v];
        return true;
    }
    // Called by commit BEFORE overwriting (x,y,z): captures the current
    // world value, once per voxel.
    void record(const World& w, int strip, int x, int y, int z) {
        int64_t k = ChunkStageBuffer::pageKey(
            x >> BRICK_BITS, y >> BRICK_BITS, z >> BRICK_BITS);
        auto ins = pages.try_emplace(k);
        if (ins.second) stripKeys[strip].push_back(k);
        StagePage& p = ins.first->second;
        int v = voxInBrick(x & BRICK_MASK, y & BRICK_MASK, z & BRICK_MASK);
        if (p.mask[v >> 6] & (1ull << (v & 63))) return;
        p.data[v] = w.getVoxel(x, y, z);
        p.mask[v >> 6] |= 1ull << (v & 63);
    }
    void evictStrip(int strip) {
        auto it = stripKeys.find(strip);
        if (it == stripKeys.end()) return;
        for (int64_t k : it->second) pages.erase(k);
        stripKeys.erase(it);
    }
};

// Fold one buffer's stat deltas into the context totals (count fields only).
inline void addGenStats(GenStats& dst, const GenStats& src) {
    dst.boulders += src.boulders;   dst.bushes += src.bushes;
    dst.trees += src.trees;         dst.flowers += src.flowers;
    dst.grassClusters += src.grassClusters;
    dst.grassBlades += src.grassBlades;
    dst.crystalsBuilt += src.crystalsBuilt;
    dst.decoBuilt += src.decoBuilt; dst.ivyBuilt += src.ivyBuilt;
}

// Stage writer facade: full-world generation writes straight through (the
// monolithic path, zero-cost); bounded generation REQUIRES a buffer.
struct StageWriter {
    World& w;
    GenBounds b;
    bool full;
    ChunkStageBuffer* buf;     // null in full-world mode
    GenContext& ctx;
    StageWriter(World& w_, const GenBounds& b_, GenContext& ctx_,
                ChunkStageBuffer* buf_)
        : w(w_), b(b_), full(b_.isFullWorld()), buf(full ? nullptr : buf_),
          ctx(ctx_) {}

    uint8_t get(int x, int y, int z) const {
        return buf ? buf->get(w, x, y, z) : w.getVoxel(x, y, z);
    }
    void set(int x, int y, int z, uint8_t m) {
        if (buf) buf->set(x, y, z, m);
        else     w.setVoxelRaw(x, y, z, m);
    }
    // Stats gating: a feature is COUNTED by the chunk that owns its anchor,
    // so chunked and monolithic runs report identical totals.
    bool anchorIn(int x, int z) const {
        return full || (x >= b.x0 && x < b.x1 && z >= b.z0 && z < b.z1);
    }
    GenStats& stats() const { return buf ? buf->stats : ctx.stats; }
};

// -- Pre-stage surface snapshot --------------------------------------------------
// All-or-nothing placement DECISIONS (anchor gates like "standing on grass",
// template checks like "all bloom cells are air") read this snapshot of the
// surface band, captured once per stage BEFORE any chunk of that stage runs.
// Reason: a chunk replaying a margin feature must reach the same accept/
// reject decision as the feature's owner chunk — but live reads into an
// already-processed neighbour see that neighbour's post-stage writes (e.g. a
// tree's own trunk over its grass tile), flipping the gate and cleanly
// cutting the feature at the border. Snapshot gates make decisions a pure
// function of the completed previous stages, identical for every replayer
// (and for the monolithic driver, which uses the same snapshot).
// Per-voxel checks AT WRITE POSITIONS stay live: a chunk's in-bounds state
// evolves in global feature order during its own replay, so those reads are
// order-consistent without snapshotting.
//
// Band: 16 voxels per column starting at heightAt(x,z)-2 — covers every gate
// (tiles at h-1, flower templates to ~h+7, grass columns to h+14).
struct SurfaceSnapshot {
    static constexpr int BAND = 16;
    // GRID_XZ * GRID_XZ * BAND, TOROIDALLY indexed (x & mask) so it works at
    // any window origin and follows a scrolling window like the height
    // texture. Always a CAPTURED (frozen) band — never a live read — which
    // is what makes a margin-replayed feature's gate decision identical for
    // every chunk that draws a slice of it (no border-cut). The in-game
    // streamer captures each chunk's band once at carve-completion (before
    // any flora runs); the offline drivers re-capture the whole window
    // before each stage.
    std::vector<uint8_t> mats;
    int ox = 0, oz = 0;          // window voxel origin (the in-window gate)
    void capture(const World& w);                      // whole window
    void capture(const World& w, const GenBounds& b);  // bounded (strip/chunk)
    uint8_t get(int x, int y, int z) const {
        // OUT OF WINDOW -> 0 (a margin-replayed feature whose anchor lies
        // just beyond the window edge is rejected, NOT wrapped to the
        // opposite edge — the toroidal index is only valid for in-window
        // coords). Index is x&mask so the band follows a scrolling window;
        // at origin 0 this is identical to the old linear snapshot.
        if (mats.empty()) return 0;
        if (x < ox || x >= ox + GRID_XZ || z < oz || z >= oz + GRID_XZ) return 0;
        int y0 = heightAt(x, z) - 2;
        int i = y - y0;
        if (i < 0 || i >= BAND)   // outside the band: immutable this stage
            return 0;             // (callers only probe in-band positions)
        return mats[(size_t(z & (GRID_XZ - 1)) * GRID_XZ
                   + (x & (GRID_XZ - 1))) * BAND + i];
    }
};

// Per-phase replay margins (voxels): how far outside its bounds a chunk must
// re-enumerate lattice cells so every feature that can WRITE into the chunk
// is replayed. With pre-stage snapshot gates, decisions can't be flipped by
// same-stage writes, so margins equal the maximum WRITE reach (plus pad).
// The contentHash harness validates these: a too-small margin makes the
// chunked driver's hash diverge from the monolithic one.
constexpr int MARGIN_BOULDER = 64;   // write reach ~40 (r17 + cluster 22) + pad
constexpr int MARGIN_BUSH    = 32;   // write reach ~10
constexpr int MARGIN_TREE    = 96;   // max horizontal reach ~62 from anchor + pad
constexpr int MARGIN_FLOWER  = 8;    // template +-1
constexpr int MARGIN_GRASS   = 32;   // cluster r4 + cell 7
constexpr int MARGIN_OASIS_PALM = 210; // palm sits up to ~1.1*R(130) from the oasis CENTER + ~30 frond reach + lean + pad

// Worldgen phases (defined across caves.cpp / flora.cpp / worldgen.cpp).
// Each derives all randomness from ctx.seed, reports into ctx.stats, and
// honors `b` as its generation window (margin replay + write clipping).
struct GenContext;
void genCaves   (World& w, GenContext& ctx, const GenBounds& b);
// Split cave entry points for the staged chunk driver: ALL chunks must carve
// before ANY chunk decorates (decoration probes read carved geometry that can
// lie outside the probing chunk).
//
// The cave plan is an UNBOUNDED on-demand lattice (6e-3a): 256-voxel cave
// cells walk their worm systems once and cache; 2048-voxel mega cells
// resolve their cavern proposals once (cap + spacing localized per mega
// cell). ensureCaveRegion walks every cell either reachable from `b` —
// SERIAL: call it before any parallel batch over `b`, so the parallel
// carve/build gathers only ever read the cell maps.
// Walk/cache the cave-plan cells a region needs (and resolve its mega
// cells). `maxCells` bounds how many fresh cells are walked this call: the
// streaming driver passes a small budget so a frontier crossing into new
// cells never blocks the frame on a worm-walk burst; it returns false while
// cells remain (caller retries next frame, holding carve/decor back).
// Batch drivers pass the default (walk everything, one parallel shot).
bool ensureCaveRegion(GenContext& ctx, const GenBounds& b,
                      int maxCells = 1 << 30);
// Carve: bounded calls may stage their SOLID->mixed brick births into a
// CaveCarveStage (compute in parallel — existing-brick writes are chunk-
// disjoint and applied directly — then commit births serially in chunk
// order); without a stage, births commit inline (the serial path).
struct CaveCarveStage {
    GenBounds b;
    std::unordered_map<int, std::array<uint32_t, BRICK_U32>> newBricks;
    // Presorted (slot-ascending) flat copy, built by the COMPUTE worker so
    // the main-thread commit is a linear alloc+memcpy sweep instead of a
    // node-map walk (~3.8K births/chunk in cavernous regions).
    std::vector<std::pair<int, std::array<uint32_t, BRICK_U32>>> sorted;
    void presort() {
        sorted.reserve(newBricks.size());
        for (auto& kv : newBricks) sorted.emplace_back(kv.first, kv.second);
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        newBricks.clear();
    }
};
void genCavesCarve(World& w, GenContext& ctx, const GenBounds& b,
                   CaveCarveStage* stage = nullptr);
void commitCaveCarve(World& w, CaveCarveStage& stage);
// Decoration is chunk-replayable (6c) and order-free (6d): ivy wander is
// capped, every request whose anchor lies within CAVE_DECO_MARGIN of the
// bounds replays in global plan order, each from its recorded seed, with
// stats/lights counted by the anchor's owner only (into the buffer; folded
// at commit). Bounded calls REQUIRE a buffer; decoration buffers must ALL
// compute before ANY commits (the carve-only-world invariant). Full-world
// bounds write straight through (the monolithic path, buf ignored).
// `undo`: pre-decoration overlay for moving-front drivers (see UndoBand);
// batch drivers pass null (the carve-only invariant holds globally there).
void genCavesBuild(World& w, GenContext& ctx, const GenBounds& b,
                   ChunkStageBuffer* buf = nullptr,
                   const UndoBand* undo = nullptr);
// Flora stages: bounded calls REQUIRE a buffer (StageWriter routes through
// it); full-world calls pass null and write straight through.
void genBoulders(World& w, GenContext& ctx, const GenBounds& b,
                 const SurfaceSnapshot& snap, ChunkStageBuffer* buf = nullptr);
void genBushes  (World& w, GenContext& ctx, const GenBounds& b,
                 const SurfaceSnapshot& snap, ChunkStageBuffer* buf = nullptr);
void genTrees   (World& w, GenContext& ctx, const GenBounds& b,
                 const SurfaceSnapshot& snap, ChunkStageBuffer* buf = nullptr);
void genLakes   (World& w, GenContext& ctx, const GenBounds& b);
void genOasis   (World& w, GenContext& ctx, const GenBounds& b);
void genFlowers (World& w, GenContext& ctx, const GenBounds& b,
                 const SurfaceSnapshot& snap, ChunkStageBuffer* buf = nullptr);
void genGrass   (World& w, GenContext& ctx, const GenBounds& b,
                 const SurfaceSnapshot& snap, ChunkStageBuffer* buf = nullptr);
