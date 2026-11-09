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
#include <functional>
#include <random>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// -- Carve ops -----------------------------------------------------------
// Worm walks (phase A) RECORD what to carve instead of writing voxels, so
// the walks parallelize per system and the carving parallelizes spatially.
// Every carve is a write of AIR — overlapping ops are idempotent, so apply
// order doesn't matter and shards never need cross-thread coordination.
struct SliceOp { glm::vec3 pos; float W, H, flatten; bool entrance; };
struct RoomOp  { glm::vec3 c; float Rh, Ht; int floorY; };

// Deferred decoration requests, each carrying its own RNG seed (drawn from
// the requesting stream at plan time) so build results don't depend on how
// many requests ran before them.
struct FormReq    { int x, z, yProbe, reach; float colChance; uint32_t seed; };
struct CrystalReq { glm::ivec3 p; uint32_t seed; };
struct DecoReq    { int x, z, yProbe; float R; uint32_t seed; };
struct IvyReq     { int x, z, yProbe, budget; bool alive; float leafCh, branchCh;
                    uint32_t seed; };
struct MegaProposal { glm::vec3 pos; uint32_t seed; };

// The cave plan is an UNBOUNDED on-demand lattice (6e-3a). Cave systems live
// on a 256-voxel cell lattice: each cell's sites/systems derive purely from
// (seed, cellX, cellZ), walk once, and cache here. Mega-caverns resolve on a
// separate 2048-voxel lattice: each mega cell gathers the proposals landing
// inside it (from systems anchored within WORM_REACH), orders them by
// proposer cell lattice order, and applies the cap + spacing locally — the
// old global cap (18 over the fixed 2048 world) becomes a per-mega-cell cap,
// preserving density. (Spacing across mega-cell borders is not enforced;
// two megas can land closer than 240 across a boundary — acceptable.)
struct CaveCell {                 // one 256-cell's systems, walked once
    std::vector<SliceOp>      slices;
    std::vector<RoomOp>       rooms;
    std::vector<FormReq>      forms;
    std::vector<CrystalReq>   crystals;
    std::vector<DecoReq>      decos;
    std::vector<IvyReq>       ivys;
    std::vector<MegaProposal> megaProps;   // unresolved (resolved per mega cell)
};
struct MegaCellRes {              // one 2048-mega-cell's resolved caverns
    std::vector<RoomOp>     rooms;
    std::vector<FormReq>    forms;
    std::vector<CrystalReq> crystals;
    std::vector<DecoReq>    decos;
    std::vector<IvyReq>     ivys;
};
struct CavePlan {
    std::unordered_map<int64_t, CaveCell>    cells;      // key = chunkKey(cellX, cellZ)
    std::unordered_map<int64_t, MegaCellRes> megaCells;  // key = chunkKey(mcx, mcz)
    bool built = false;     // monolithic decoration build is once-per-window
    int statMaxReach = 0;   // worst measured system reach (contract gauge)
};

namespace {

constexpr int CAVE_ROOF = 5;   // min stone roof over a tunnel (was `ROOF`)

// Step 6b: the worm-reach CONTRACT. Past WORM_TETHER (XZ from the system's
// spawn site) a worm steers home; past the slack it hard-stops (the steer
// makes that backstop nearly unreachable, but it makes the bound PROVABLE).
// Region queries in the streamed world replay every system whose site lies
// within WORM_REACH of their bounds: tether + slack + worst op radius
// (mega Rh 170) + pad. Measured pre-tether reach was 1988 voxels and
// border-censored — i.e. unbounded without this contract.
constexpr float WORM_TETHER       = 1280.0f;
constexpr float WORM_TETHER_SLACK = 384.0f;
constexpr int   WORM_REACH = int(WORM_TETHER + WORM_TETHER_SLACK) + 170 + 64;

// Step 6c: decoration reach contract. Ivy is the widest writer — its surface
// walk is hard-capped at IVY_WANDER XZ voxels from its request anchor (steps
// and hops that would leave the disc are simply not candidates). Everything
// else writes far tighter (formations ~10, crystals ~9, floor patches ~21).
// A chunk replays every decoration request whose anchor lies within
// CAVE_DECO_MARGIN of its bounds; the margin must exceed every writer's
// reach (ivy 64 + leaf fill 1) plus probe pad.
constexpr int IVY_WANDER       = 64;
constexpr int CAVE_DECO_MARGIN = 96;

// Worst slice half-extent in voxels (gallery W<=24, x1.3 edge noise, pad):
// the z-window for binary-searched slice queries against z-sorted cells.
constexpr int SLICE_WMAX = 36;

// Z-extent of an op's voxel footprint (for shard bucketing).
inline void sliceZExtent(const SliceOp& s, int& zmin, int& zmax) {
    int cz = int(std::round(s.pos.z));
    int Wi = int(std::ceil(s.W * 1.3f));
    zmin = cz - Wi; zmax = cz + Wi;
}
inline void roomZExtent(const RoomOp& r, int& zmin, int& zmax) {
    int cz = int(std::round(r.c.z));
    int Ri = int(std::ceil(r.Rh));
    zmin = cz - Ri; zmax = cz + Ri;
}
inline void sliceXExtent(const SliceOp& s, int& xmin, int& xmax) {
    int cx = int(std::round(s.pos.x));
    int Wi = int(std::ceil(s.W * 1.3f));
    xmin = cx - Wi; xmax = cx + Wi;
}
inline void roomXExtent(const RoomOp& r, int& xmin, int& xmax) {
    int cx = int(std::round(r.c.x));
    int Ri = int(std::ceil(r.Rh));
    xmin = cx - Ri; xmax = cx + Ri;
}

// One shard's writer: applies carve ops to the voxel box x in [x0, x1),
// z in [z0, z1). In full-world mode shards are whole brick rows (x spans
// everything); in bounded mode there is a single writer over the window.
// A brick belongs to exactly one shard, so pool writes to existing mixed
// bricks are thread-safe without locks. SOLID->mixed transitions can't call
// allocBrick concurrently; the data is staged in `newBricks` and merged
// serially (in slot order) afterward.
struct ShardWriter {
    World& w;
    int x0, x1, z0, z1;
    std::unordered_map<int, std::array<uint32_t, BRICK_U32>> newBricks;

    void carveAir(int x, int y, int z) {
        if (x < x0 || x >= x1 || z < z0 || z >= z1) return;
        if (y < 2 || y >= GRID_Y - 1) return;   // XZ clip above; y world floor/cap
        int slot = w.slotOfBrick(x >> BRICK_BITS, y >> BRICK_BITS, z >> BRICK_BITS);
        uint32_t bm = w.brickmap[slot];
        if (bm == BM_EMPTY) return;                    // air already
        int linear = voxInBrick(x & BRICK_MASK, y & BRICK_MASK, z & BRICK_MASK);
        int ws = linear >> 2, sh = (linear & 3) * 8;
        if (bm & BM_SOLID_FLAG) {
            uint8_t mat = uint8_t(bm & 0xFFu);
            auto it = newBricks.find(slot);
            if (it == newBricks.end()) {
                auto& arr = newBricks[slot];
                uint32_t pat = uint32_t(mat) | (uint32_t(mat) << 8)
                             | (uint32_t(mat) << 16) | (uint32_t(mat) << 24);
                arr.fill(pat);
                arr[ws] &= ~(0xFFu << sh);
            } else {
                it->second[ws] &= ~(0xFFu << sh);
            }
        } else {
            w.brickpool[size_t(bm - 1u) * BRICK_U32 + ws] &= ~(0xFFu << sh);
        }
    }
};

// Carve one tunnel cross-section (same shape math as the old inline
// carveSlice; see genCaves for the geometric commentary).
void applySlice(ShardWriter& sw, const SliceOp& op) {
    int cx = int(std::round(op.pos.x)), cy = int(std::round(op.pos.y)),
        cz = int(std::round(op.pos.z));
    int Wi = int(std::ceil(op.W * 1.3f));
    int dzLo = std::max(-Wi, sw.z0 - cz), dzHi = std::min(Wi, sw.z1 - 1 - cz);
    int dxLo = std::max(-Wi, sw.x0 - cx), dxHi = std::min(Wi, sw.x1 - 1 - cx);
    for (int dz = dzLo; dz <= dzHi; ++dz)
    for (int dx = dxLo; dx <= dxHi; ++dx) {
        float r = std::sqrt(float(dx*dx + dz*dz));
        float wob = smoothNoise3((cx+dx) * 0.15f, op.pos.y * 0.15f, (cz+dz) * 0.15f) - 0.5f;
        float Weff = op.W * (1.0f + 0.22f * wob);
        if (r > Weff) continue;
        float frac = r / Weff;
        float dome = std::sqrt(std::max(0.0f, 1.0f - frac*frac));
        float flat = (frac < 0.78f) ? 1.0f : (1.0f - (frac - 0.78f) / 0.22f);
        if (flat < 0.0f) flat = 0.0f;
        float ceilH = op.H * (dome + (flat - dome) * op.flatten);
        float bump = 3.0f * (smoothNoise3((cx+dx) * 0.12f, 50.0f, (cz+dz) * 0.12f) - 0.5f);
        int topAbs = cy + int(std::round(ceilH + bump));
        int colSurf = heightAt(cx + dx, cz + dz);
        int roof = (op.entrance && (colSurf - cy) < 17) ? (colSurf + 3)
                                                        : (colSurf - CAVE_ROOF);
        if (topAbs > roof) topAbs = roof;
        int bot = op.entrance ? cy - 3 : cy - 2;
        for (int yy = bot; yy <= topAbs; ++yy) sw.carveAir(cx + dx, yy, cz + dz);
    }
}

// Hollow one cavern room (same shape math as the old inline carveRoom).
void applyRoom(ShardWriter& sw, const RoomOp& op) {
    int cx = int(std::round(op.c.x)), cz = int(std::round(op.c.z));
    int Ri = int(std::ceil(op.Rh));
    int dzLo = std::max(-Ri, sw.z0 - cz), dzHi = std::min(Ri, sw.z1 - 1 - cz);
    int dxLo = std::max(-Ri, sw.x0 - cx), dxHi = std::min(Ri, sw.x1 - 1 - cx);
    for (int dz = dzLo; dz <= dzHi; ++dz)
    for (int dx = dxLo; dx <= dxHi; ++dx) {
        float r = std::sqrt(float(dx*dx + dz*dz));
        float edgeN = smoothNoise3((cx+dx) * 0.045f, op.c.y * 0.045f, (cz+dz) * 0.045f);
        float Reff = op.Rh * (0.76f + 0.36f * edgeN);
        if (r > Reff) continue;
        float frac = r / Reff;
        float fall = frac < 0.62f ? 1.0f : (1.0f - (frac - 0.62f) / 0.38f);
        if (fall < 0.0f) fall = 0.0f;
        float bump = 7.0f * (smoothNoise3((cx+dx) * 0.11f, 77.0f, (cz+dz) * 0.11f) - 0.5f);
        int top = op.floorY + int(op.Ht * fall + bump);
        int colSurf = heightAt(cx + dx, cz + dz);
        if (top > colSurf - CAVE_ROOF) top = colSurf - CAVE_ROOF;
        int fb = int(std::round(2.5f * (smoothNoise3((cx+dx) * 0.08f, 33.0f, (cz+dz) * 0.08f) - 0.5f)));
        for (int yy = op.floorY - 2 + fb; yy <= top; ++yy) sw.carveAir(cx + dx, yy, cz + dz);
    }
}

} // namespace

// -- Caves: agent-carved tunnels -----------------------------------------
// A path agent ("worm") walks underground carving a tunnel cross-section
// with a FLAT FLOOR and an ARCHED CEILING (so you walk on a floor with
// headroom, not float in a round tube). Its pitch is clamped GENTLE so it
// meanders and descends slowly instead of plunging. Worms start at hill
// slopes heading into the hillside, so entrances appear on cliff faces.
// Branches add connectivity; occasional widenings make chambers.
//
// All randomness is RE-SEEDED per feature stream: each cave system, each
// mega resolution, and each deferred decoration request derives its own
// seed (featureSeed / a draw recorded at plan time). Nothing consumes a
// sequential world-wide stream, so any system or request is reproducible
// in isolation — the prerequisite for chunked + parallel + UNBOUNDED
// generation (cells walk lazily, in any order, with identical results).
namespace {

constexpr int CAVE_CELL = 256;    // cave-system site lattice
constexpr int MEGA_CELL = 2048;   // mega-cavern resolution lattice
// Gather reach for resolved mega ops: room Rh <= 170 (x edge noise) + its
// decoration request scatter + CAVE_DECO_MARGIN, padded.
constexpr int MEGA_REACH = 400;
// Cells whose DECORATION REQUESTS can reach bounds: request anchors stay
// within a system's measured reach (<= WORM_REACH), plus the replay margin.
constexpr int CAVE_GATHER_REACH = WORM_REACH + CAVE_DECO_MARGIN;

// Everything one cave system produces during its walk. Voxel carving
// happens later via the recorded ops (applySlice / applyRoom above);
// requests are built in the deferred decoration pass.
struct SystemResult {
    std::vector<SliceOp>      slices;
    std::vector<RoomOp>       rooms;
    std::vector<FormReq>      forms;
    std::vector<CrystalReq>   crystals;
    std::vector<DecoReq>      decos;
    std::vector<IvyReq>       ivys;
    std::vector<MegaProposal> megaProps;
    long long steps = 0;
    bool deep = false;
};

// Plan one COHERENT cavern at c: a big, irregular room with a high,
// mostly-flat ceiling (NOT a hemisphere dome). Size varies a lot — most
// are large, some are huge, and it's dressed with formations afterward.
// PLANS only: records the hollow as a RoomOp (carved later by applyRoom)
// and the decoration requests, all on the caller-provided stream so the
// plan is reproducible from the system/proposal seed alone.
void planRoom(std::mt19937& r, glm::vec3 c, bool mega, SystemResult& out) {
        auto fr = [&]() { return frand01(r); };
        int cx = int(std::round(c.x)), cz = int(std::round(c.z));
        int floorY = int(std::round(c.y)) - 2;
        // Normal cavern, or a deep MEGA cavern ~2.5x the radius: a grand
        // cathedral hall reserved for the deep tier.
        float Rh = mega ? (100.0f + fr() * 70.0f)   // 100-170 (vs 38-60)
                        : (38.0f + fr() * 22.0f);
        float Ht = mega ? (60.0f + fr() * 45.0f)    // 60-105 (vs 28-44)
                        : (28.0f + fr() * 16.0f);
        out.rooms.push_back({ c, Rh, Ht, floorY });

        // Dress the cavern with formations. Each is sized to the LOCAL
        // floor-ceiling gap and scattered (never a dense wall), so the big
        // room stays easy to move through (player is 18 tall, radius 2).
        int nForm = int(Rh * 0.22f) + int(fr() * 6.0f);  // scales with size
        for (int k = 0; k < nForm; ++k) {
            float ang = fr() * 6.2831853f;
            float rad = (0.10f + 0.80f * fr()) * Rh;
            int gcx = cx + int(std::round(std::cos(ang) * rad));
            int gcz = cz + int(std::round(std::sin(ang) * rad));
            // Columns only stand free in the centre (flat full-height
            // ceiling); near the edges they'd merge into a sloped wall.
            float colChance = (rad < 0.45f * Rh) ? 0.30f : 0.0f;
            out.forms.push_back({ gcx, gcz, floorY + 8, 90, colChance, r() });
        }

        // Glow crystals: only in the deep "luminous" band (or any mega
        // cavern, which is always deep). A few clusters scattered on the
        // floor so the cavern lights ITSELF -- the payoff for the descent.
        bool luminous = mega || (floorY < SURFACE_Y - 1400);
        if (luminous) {
            int nCl = mega ? (10 + int(fr() * 9.0f))   // 10-18 in a mega (showpiece)
                           : (2 + int(fr() * 3.0f));   // 2-4 otherwise
            for (int k = 0; k < nCl; ++k) {
                float ang = fr() * 6.2831853f;
                float rad = (0.10f + 0.80f * fr()) * Rh;
                int gcx = cx + int(std::round(std::cos(ang) * rad));
                int gcz = cz + int(std::round(std::sin(ang) * rad));
                out.crystals.push_back({ glm::ivec3(gcx, floorY + 8, gcz), r() });
            }
        }

        // Dirt patches + mushrooms on the floor -- at EVERY depth (including
        // mega-caverns, where they sit warm against the cool crystals).
        // Scattered sites scale with room size; built on the real floor.
        int nDeco = int(Rh * 0.06f) + int(fr() * 2.0f);
        for (int k = 0; k < nDeco; ++k) {
            float ang = fr() * 6.2831853f;
            float rad = (0.05f + 0.85f * fr()) * Rh;
            int gcx = cx + int(std::round(std::cos(ang) * rad));
            int gcz = cz + int(std::round(std::sin(ang) * rad));
            out.decos.push_back({ gcx, gcz, floorY + 8, 0.0f, r() });   // 0 = cavern size
        }

        // Ivy: WIDE per-cavern variation in overgrowth -- some barely touched,
        // some heavily draped. The level scales creeper count, length, leaf
        // density and branching. Living green near the surface, dead brown
        // deeper -- but mega-caverns are forced lush green (crystal-fed).
        float belowSurf = float(SURFACE_Y - floorY);
        float og = mega ? (0.5f + fr() * 0.5f) : fr();   // mega denser; else full range
        float aliveProb = mega ? 0.92f
                        : std::max(0.10f, std::min(0.85f, 1.0f - belowSurf / 2000.0f));
        int   nIvy     = 3 + int(og * Rh * 0.70f);            // many more creepers
        float leafCh   = 0.30f + og * 0.45f;                  // denser leaf swaths
        float branchCh = 0.12f + og * 0.18f;                  // more forking
        for (int k = 0; k < nIvy; ++k) {
            float ang = fr() * 6.2831853f;
            float rad = (0.45f + 0.50f * fr()) * Rh;          // start up near the walls
            int gx = cx + int(std::round(std::cos(ang) * rad));
            int gz = cz + int(std::round(std::sin(ang) * rad));
            int budget = 30 + int(og * 80.0f) + int(fr() * 20.0f);  // longer networks
            bool alive = (fr() < aliveProb);
            out.ivys.push_back({ gx, gz, floorY + int(Ht), budget, alive, leafCh, branchCh,
                                 r() });
        }
    }

void wormWalk(std::mt19937& r, SystemResult& out,
              glm::vec3 pos, glm::vec2 anchor,
              float yaw, float pitch, int length, int depth, bool deep) {
            auto fr = [&]() { return frand01(r); };
            auto ur = [&]() { return frand11(r); };
            float yawRate = 0.0f, pitchRate = 0.0f;
            float W = 7.0f + fr() * 2.0f;     // half-width ~7-9 (14-18 wide)
            float H = 20.0f + fr() * 6.0f;    // ceiling height ~20-26 (headroom)
            int roomsMade = 0;                   // emergent caverns placed by this worm
            int megaMade  = 0;                   // deep mega-caverns (max 1)
            int sinceRoom = 1000;                // steps since last room (spacing)
            // Cross-section the worm is currently carving, lerped toward a
            // target each step so corridor<->gallery transitions taper
            // smoothly instead of popping into a box.
            float curW = W, curH = H, curFlat = 0.0f;
            int galleryLeft = 0;                 // steps left in current gallery (0 = corridor)
            int sinceGallery = 1000;             // spacing between galleries
            float galW = 0.0f, galH = 0.0f;      // this gallery's target dims
            for (int s = 0; s < length; ++s) {
                ++out.steps;
                bool entrance = (depth == 0 && s < 20);   // dig the mouth + dive in
                yawRate   = yawRate   * 0.85f + ur() * 0.06f;   // smooth winding
                pitchRate = pitchRate * 0.80f + ur() * 0.03f;
                yaw += yawRate;

                // Tether (the reach contract): beyond WORM_TETHER from the
                // spawn site, blend the heading back toward home — gently at
                // first, firmly deeper into the slack band — and hard-stop
                // at the slack limit so the bound is provable. Entrance
                // steps are exempt (they're meters from the site).
                if (!entrance) {
                    float ox = pos.x - anchor.x, oz = pos.z - anchor.y;
                    float dHome = std::sqrt(ox * ox + oz * oz);
                    if (dHome > WORM_TETHER + WORM_TETHER_SLACK) break;
                    if (dHome > WORM_TETHER) {
                        float back  = std::atan2(-oz, -ox);
                        float delta = back - yaw;
                        while (delta >  3.14159265f) delta -= 6.2831853f;
                        while (delta < -3.14159265f) delta += 6.2831853f;
                        float t = std::min(1.0f, (dHome - WORM_TETHER) / 160.0f);
                        yaw += delta * (0.10f + 0.30f * t);
                    }
                }

                if (entrance) {
                    // Dive steeply under cover so we don't run along the
                    // surface carving a trench; the mouth is on the hill.
                    pitch = -0.42f;
                } else {
                    // Steady descent so the worm works its way down, plus a
                    // guard that pushes harder near the surface. DEEP-diver
                    // worms commit to a steeper sustained descent so they
                    // reliably reach the mega-cavern tier (a clear route
                    // down); intimate worms stay gentle for winding shallow
                    // exploration.
                    float steady = deep ? 0.020f : 0.010f;
                    float floorP = deep ? -0.40f : -0.20f;
                    // Deep divers never climb (upper clamp 0) so they descend
                    // relentlessly all the way toward bedrock instead of
                    // levelling off; intimate worms may still rise a little.
                    float ceilP  = deep ?  0.00f :  0.12f;
                    pitch += pitchRate - steady;
                    int sH = heightAt(int(pos.x), int(pos.z));
                    if (float(sH) - pos.y < 22.0f) pitch -= 0.06f;
                    if (pitch < floorP) pitch = floorP;
                    if (pitch > ceilP)  pitch = ceilP;
                    // Galleries stay flatter so the hall is level and walkable.
                    if (galleryLeft > 0) pitch *= 0.85f;
                }

                glm::vec3 dir(std::cos(pitch) * std::cos(yaw),
                              std::sin(pitch),
                              std::cos(pitch) * std::sin(yaw));
                pos += dir * 3.0f;

                // XZ is unbounded (the tether is the reach bound); only the
                // world floor stops a worm.
                if (pos.y < 8) break;

                // The entrance mouth is carved a bit wider/taller so it
                // clearly clears the player. Otherwise we carve the current
                // (lerped) cross-section: a tight portrait corridor, or a
                // wide-low gallery during a gallery stretch.
                if (entrance) {
                    out.slices.push_back({ pos, W * 1.15f, H * 1.1f, 0.0f, true });
                } else {
                    bool inGallery = (galleryLeft > 0);
                    float tgtW    = inGallery ? galW : W;
                    float tgtH    = inGallery ? galH : H;
                    float tgtFlat = inGallery ? 1.0f : 0.0f;
                    curW    += (tgtW    - curW)    * 0.12f;   // smooth taper
                    curH    += (tgtH    - curH)    * 0.12f;
                    curFlat += (tgtFlat - curFlat) * 0.12f;
                    if (inGallery) --galleryLeft;
                    out.slices.push_back({ pos, curW, curH, curFlat, false });
                }

                // --- Furnish the passage itself (tubes + galleries) -------
                // Collected along the worm, built in the post-pass (probing
                // real geometry). Spaced so it accents, not carpets.
                if (!entrance) {
                    int px = int(std::round(pos.x)), pz = int(std::round(pos.z));
                    int fyP = int(std::round(pos.y)) + 6;            // probe down for floor
                    int cyP = int(std::round(pos.y + curH));        // ~ceiling for ivy
                    float belowSurf = float(SURFACE_Y) - pos.y;
                    // Floor cover: small dirt/moss patch (bigger in galleries).
                    if ((r() & 63) == 0) {
                        float R = std::max(2.0f, std::min(curW * 0.40f, 6.0f));
                        out.decos.push_back({ px, pz, fyP, R, r() });
                    }
                    // Ivy on the passage walls/ceiling: small networks,
                    // living shallow -> dead-brown deep.
                    if ((r() & 31) == 0) {              // ~4x more passage ivy
                        float aliveP = std::max(0.10f, std::min(0.85f, 1.0f - belowSurf / 2000.0f));
                        out.ivys.push_back({ px, pz, cyP, 18 + int(fr() * 30.0f),
                                             (fr() < aliveP), 0.32f, 0.16f, r() });
                    }
                    // Sparse glow crystals once deep, to light dark passages.
                    if (pos.y < float(SURFACE_Y) - 1400.0f && (r() & 511) < 2) {
                        out.crystals.push_back({ glm::ivec3(px, fyP + 2, pz), r() });
                    }
                }

                // Collect gallery formations (built later in one pass, so
                // following slices can't carve them away). Offset INSIDE the
                // gallery air (0.20-0.60 of half-width) so they land in the
                // room, biased off the dead-centre path. Columns are COMMON
                // here: the flat low ceiling lets drip-pairs bridge, and a
                // flat ceiling means a column connects cleanly anywhere.
                if (galleryLeft > 0 && curW > galW * 0.6f && (r() & 31) < 4) {
                    float side = (r() & 1) ? 1.0f : -1.0f;
                    float off  = curW * (0.20f + 0.40f * fr());
                    int gx = int(std::round(pos.x - std::sin(yaw) * off * side));
                    int gz = int(std::round(pos.z + std::cos(yaw) * off * side));
                    out.forms.push_back({ gx, gz, int(std::round(pos.y)) + 5, 55, 0.55f, r() });
                }

                // EMERGENT caverns: once we're genuinely deep, open a
                // coherent room at an unpredictable point. Capped per worm
                // and spaced out, so a long descent reliably rewards you
                // with a room or two without them being predictable.
                ++sinceRoom;
                ++sinceGallery;
                if (!entrance && galleryLeft == 0) {
                    int sH = heightAt(int(pos.x), int(pos.z));
                    // MEGA cavern: only in the DEEP tier (>1500 below the
                    // global surface), rare and one per worm. The worm only
                    // PROPOSES (a decision on its own stream; the global cap
                    // + spacing are resolved after all worms ran) and behaves
                    // as if the proposal was accepted -- so its later
                    // decisions never depend on other worms.
                    bool wantMega = pos.y < float(SURFACE_Y) - 1500.0f &&
                                    megaMade < 1 &&
                                    sinceRoom > 90 && (r() & 255) < 6;
                    if (wantMega) {
                        out.megaProps.push_back({ pos, r() });
                        ++megaMade;
                        sinceRoom = 0;
                    } else if (float(sH) - pos.y > 50.0f && roomsMade < 2 &&
                               sinceRoom > 80 && (r() & 1023) < 6) {
                        planRoom(r, pos, false, out);
                        ++roomsMade;
                        sinceRoom = 0;
                    }
                }

                // Open a wide-low GALLERY: a hall (≈16:9) where formations
                // can fit, unlike the tight portrait corridor. It's an
                // inline widening of THIS worm (not a new agent), tapered in
                // and out, and spaced away from rooms and other galleries so
                // passages stay varied and don't blob together.
                if (!entrance && galleryLeft == 0 && depth < 2) {
                    int sH = heightAt(int(pos.x), int(pos.z));
                    if (float(sH) - pos.y > 40.0f && sinceGallery > 120 &&
                        sinceRoom > 40 && (r() & 1023) < 9) {
                        galW = 18.0f + fr() * 6.0f;   // 36-48 wide
                        galH = 22.0f + fr() * 8.0f;   // 22-30 tall (flat ceiling
                                                      // clears the 18-voxel player
                                                      // across the whole width)
                        galleryLeft = 30 + int(fr() * 30.0f);
                        sinceGallery = 0;
                    }
                }

                // Branch off a side passage. More branches = more route
                // choices (less "single tube to a dead end"). Branches
                // inherit the parent's deep-dive commitment.
                if (depth < 2 && (r() & 191) == 0) {
                    wormWalk(r, out, pos, anchor, yaw + ur() * 1.5f, pitch + ur() * 0.15f,
                             (length - s) / 2 + 60, depth + 1, deep);
                }
            }
        }

// Everything one cell walk produces, computed PURELY from (seed, cell
// coords) + the pure height field — parallel-safe; appliers fold it into
// the plan/context serially in lattice order.
struct CellWalkOut {
    CaveCell cell;
    std::vector<glm::ivec3> entrances;
    int systems = 0, deep = 0;
    long long steps = 0;
    int reach = 0;
};

// Walk one cave cell's systems (PURE — no plan/ctx mutation). Spawns at
// hill slopes, boring into the hillside + down. Lattice-seeded: each
// 256-voxel cell derives its own site/system seeds, so any cell's systems
// are reproducible in isolation, in any walk order. Site search needs a
// real slope (>= 0.5) for a clean hillside mouth, so flat cells spawn
// nothing — density lands near ~1.09 systems per cell.
void walkCellPure(uint64_t seed, int cellX, int cellZ, CellWalkOut& out) {
    CaveCell& cell = out.cell;
    std::mt19937 rng;
    auto frand = [&]() { return frand01(rng); };

    // ---- Site collection (one rng stream per cell) -----------------------
    // A DENSITY, not a world total. The play-tested feel came from the old
    // FIXED world: 70 systems anchored inside 2048^2, which left every
    // in-window point border-starved — receiving only ~62% of true
    // unbounded coverage (the clipped-disc fraction at the old world's
    // centre: 2048^2 / (pi * 1469^2), with 1469 the measured system reach).
    // The unbounded lattice multiplies the raw per-cell rate by that 0.62
    // so the infinite world reproduces the tuned interior density instead
    // of becoming ~2x cavier everywhere.
    constexpr float PER_CELL = (70.0f / 64.0f) * 0.62f;
    struct Site { uint32_t seed; int sx, sz, h; float yaw; };
    std::vector<Site> sites;
    rng.seed(featureSeed(seed, SEED_CAVE_SITE, cellX, cellZ));
    int n = int(PER_CELL);
    if (frand() < PER_CELL - float(int(PER_CELL))) ++n;
    for (int k = 0; k < n; ++k) {
        // Try many sites within this cell for a sloped mouth. Sloped spots
        // are rare (~2% of the surface), so the budget must be generous or
        // hilly cells fail to spawn their system and cave density drops.
        int sx = -1, sz = -1, h = 0;
        float yaw = 0.0f;
        for (int t = 0; t < 220; ++t) {
            int px = cellX * CAVE_CELL + int(frand() * CAVE_CELL);
            int pz = cellZ * CAVE_CELL + int(frand() * CAVE_CELL);
            // Local slope + uphill direction.
            float gx = float(heightAt(px + 6, pz) - heightAt(px - 6, pz)) / 12.0f;
            float gz = float(heightAt(px, pz + 6) - heightAt(px, pz - 6)) / 12.0f;
            float slope = std::sqrt(gx*gx + gz*gz);
            if (slope < 0.5f) continue;      // need a real hill
            sx = px; sz = pz; h = heightAt(px, pz);
            yaw = std::atan2(gz, gx);        // head INTO the hill (uphill)
            break;
        }
        if (sx < 0) continue;                // no slope in this cell
        sites.push_back({ featureSeed(seed, SEED_CAVE_SYSTEM, cellX, cellZ, k),
                          sx, sz, h, yaw });
    }

    // ---- Walk this cell's systems, concatenating in site (k) order -------
    float maxReach2 = 0.0f;   // worst XZ displacement² of any op from its site
    for (const Site& st : sites) {
        std::mt19937 r(st.seed);
        SystemResult res;
        // ~half the systems are DEEP divers: steeper, longer journeys that
        // reliably plunge to the mega-cavern tier; the rest stay intimate.
        res.deep = (frand01(r) < 0.50f);
        int length = res.deep ? (2000 + int(frand01(r) * 1600.0f))
                              : (700  + int(frand01(r) *  900.0f));
        wormWalk(r, res, glm::vec3(float(st.sx), float(st.h - 2), float(st.sz)),
                 glm::vec2(float(st.sx), float(st.sz)),
                 st.yaw, -0.10f, length, 0, res.deep);
        // Measure this system's carve reach (op distance + op radius) from
        // its spawn site — the contract gauge for region queries.
        float sx = float(st.sx), sz = float(st.sz);
        for (const SliceOp& s : res.slices) {
            float dx = s.pos.x - sx, dz = s.pos.z - sz;
            float d = std::sqrt(dx*dx + dz*dz) + s.W * 1.3f;
            maxReach2 = std::max(maxReach2, d * d);
        }
        for (const RoomOp& rm : res.rooms) {
            float dx = rm.c.x - sx, dz = rm.c.z - sz;
            float d = std::sqrt(dx*dx + dz*dz) + rm.Rh;
            maxReach2 = std::max(maxReach2, d * d);
        }
        for (const MegaProposal& mp : res.megaProps) {
            float dx = mp.pos.x - sx, dz = mp.pos.z - sz;
            float d = std::sqrt(dx*dx + dz*dz) + 170.0f;  // worst mega Rh
            maxReach2 = std::max(maxReach2, d * d);
        }
        out.entrances.push_back(glm::ivec3(st.sx, st.h, st.sz));
        ++out.systems;
        if (res.deep) ++out.deep;
        out.steps += res.steps;
        cell.slices.insert(cell.slices.end(), res.slices.begin(), res.slices.end());
        cell.rooms.insert(cell.rooms.end(), res.rooms.begin(), res.rooms.end());
        cell.forms.insert(cell.forms.end(), res.forms.begin(), res.forms.end());
        cell.crystals.insert(cell.crystals.end(), res.crystals.begin(), res.crystals.end());
        cell.decos.insert(cell.decos.end(), res.decos.begin(), res.decos.end());
        cell.ivys.insert(cell.ivys.end(), res.ivys.begin(), res.ivys.end());
        cell.megaProps.insert(cell.megaProps.end(), res.megaProps.begin(), res.megaProps.end());
    }
    // Sort the cell's slices by z for binary-searched carve queries. SAFE:
    // every carve write is AIR (idempotent), so slice order is free. Request
    // vectors keep their plan order (replay order matters there).
    std::sort(cell.slices.begin(), cell.slices.end(),
              [](const SliceOp& a, const SliceOp& b) { return a.pos.z < b.pos.z; });

    out.reach = int(std::ceil(std::sqrt(maxReach2)));
    // Contract validation: region queries must never miss a system's ops.
    if (out.reach > WORM_REACH)
        std::fprintf(stderr, "[caves] WARNING: cell (%d,%d) reach %d exceeds "
                             "WORM_REACH contract %d\n",
                     cellX, cellZ, out.reach, WORM_REACH);
}

// Fold one pure cell walk into the plan + context (serial; callers apply in
// lattice order so stats/entrances accumulate deterministically).
CaveCell& applyCellWalk(GenContext& ctx, CavePlan& plan, int cellX, int cellZ,
                        CellWalkOut&& o) {
    CaveCell& cell = plan.cells[chunkKey(cellX, cellZ)];
    cell = std::move(o.cell);
    ctx.caveEntrances.insert(ctx.caveEntrances.end(),
                             o.entrances.begin(), o.entrances.end());
    ctx.stats.caveSystems += o.systems;
    ctx.stats.deepSystems += o.deep;
    ctx.stats.wormSteps   += o.steps;
    if (o.reach > plan.statMaxReach) plan.statMaxReach = o.reach;
    ctx.stats.wormMaxReachXZ = plan.statMaxReach;
    ctx.stats.formReqs    += int(cell.forms.size());
    ctx.stats.crystalReqs += int(cell.crystals.size());
    ctx.stats.decoReqs    += int(cell.decos.size());
    ctx.stats.ivyReqs     += int(cell.ivys.size());
    return cell;
}

CaveCell& walkCaveCell(GenContext& ctx, CavePlan& plan, int cellX, int cellZ) {
    auto found = plan.cells.find(chunkKey(cellX, cellZ));
    if (found != plan.cells.end()) return found->second;
    CellWalkOut o;
    walkCellPure(ctx.seed, cellX, cellZ, o);
    return applyCellWalk(ctx, plan, cellX, cellZ, std::move(o));
}

// Resolve one mega cell: gather the proposals landing inside it from every
// system that can reach it, order them by proposer-cell lattice order (the
// old "system order" generalized — identical for every query), then apply
// the cap + spacing greedily WITHIN this cell. Old global cap 18 over the
// fixed 2048 world == cap 18 per 2048 mega cell: density preserved.
MegaCellRes& resolveMegaCell(GenContext& ctx, CavePlan& plan, int mcx, int mcz) {
    int64_t key = chunkKey(mcx, mcz);
    auto found = plan.megaCells.find(key);
    if (found != plan.megaCells.end()) return found->second;

    int x0 = mcx * MEGA_CELL, z0 = mcz * MEGA_CELL;
    // Proposals come from systems anchored within WORM_REACH of this cell.
    int cx0 = floorDiv(x0 - WORM_REACH, CAVE_CELL);
    int cx1 = floorDiv(x0 + MEGA_CELL - 1 + WORM_REACH, CAVE_CELL);
    int cz0 = floorDiv(z0 - WORM_REACH, CAVE_CELL);
    int cz1 = floorDiv(z0 + MEGA_CELL - 1 + WORM_REACH, CAVE_CELL);
    std::vector<MegaProposal> props;
    for (int cz = cz0; cz <= cz1; ++cz)
    for (int cx = cx0; cx <= cx1; ++cx) {
        CaveCell& cell = walkCaveCell(ctx, plan, cx, cz);
        for (const MegaProposal& mp : cell.megaProps)
            if (mp.pos.x >= float(x0) && mp.pos.x < float(x0 + MEGA_CELL) &&
                mp.pos.z >= float(z0) && mp.pos.z < float(z0 + MEGA_CELL))
                props.push_back(mp);
    }

    MegaCellRes& res = plan.megaCells[key];   // create AFTER cell walks
    ctx.stats.megaProposed += int(props.size());
    int made = 0;
    std::vector<glm::vec3> centers;
    for (const MegaProposal& mp : props) {
        if (made >= 18) break;
        bool farEnough = true;
        for (const glm::vec3& mc : centers)
            if (glm::length(mc - mp.pos) < 240.0f) { farEnough = false; break; }
        if (!farEnough) continue;
        std::mt19937 r(mp.seed);
        SystemResult megaOut;
        planRoom(r, mp.pos, true, megaOut);
        res.rooms.insert(res.rooms.end(), megaOut.rooms.begin(), megaOut.rooms.end());
        res.forms.insert(res.forms.end(), megaOut.forms.begin(), megaOut.forms.end());
        res.crystals.insert(res.crystals.end(), megaOut.crystals.begin(), megaOut.crystals.end());
        res.decos.insert(res.decos.end(), megaOut.decos.begin(), megaOut.decos.end());
        res.ivys.insert(res.ivys.end(), megaOut.ivys.begin(), megaOut.ivys.end());
        centers.push_back(mp.pos);
        ctx.megaCaverns.push_back(glm::ivec3(int(mp.pos.x), int(mp.pos.y), int(mp.pos.z)));
        ++made;
        ++ctx.stats.megaCarved;
    }
    ctx.stats.formReqs    += int(res.forms.size());
    ctx.stats.crystalReqs += int(res.crystals.size());
    ctx.stats.decoReqs    += int(res.decos.size());
    ctx.stats.ivyReqs     += int(res.ivys.size());
    return res;
}

// Iterate the WALKED cells / RESOLVED mega cells intersecting b +- reach in
// lattice order ((cz, cx) ascending) — the global replay order every query
// agrees on. Gathers only ever find() — ensureCaveRegion walks first.
template <class Fn>
void forRegionCells(CavePlan& plan, const GenBounds& b, int reach, Fn&& fn) {
    int cx0 = floorDiv(b.x0 - reach, CAVE_CELL), cx1 = floorDiv(b.x1 - 1 + reach, CAVE_CELL);
    int cz0 = floorDiv(b.z0 - reach, CAVE_CELL), cz1 = floorDiv(b.z1 - 1 + reach, CAVE_CELL);
    for (int cz = cz0; cz <= cz1; ++cz)
    for (int cx = cx0; cx <= cx1; ++cx) {
        auto it = plan.cells.find(chunkKey(cx, cz));
        if (it != plan.cells.end()) fn(it->second);
    }
}
template <class Fn>
void forRegionMegas(CavePlan& plan, const GenBounds& b, Fn&& fn) {
    int mx0 = floorDiv(b.x0 - MEGA_REACH, MEGA_CELL), mx1 = floorDiv(b.x1 - 1 + MEGA_REACH, MEGA_CELL);
    int mz0 = floorDiv(b.z0 - MEGA_REACH, MEGA_CELL), mz1 = floorDiv(b.z1 - 1 + MEGA_REACH, MEGA_CELL);
    for (int mz = mz0; mz <= mz1; ++mz)
    for (int mx = mx0; mx <= mx1; ++mx) {
        auto it = plan.megaCells.find(chunkKey(mx, mz));
        if (it != plan.megaCells.end()) fn(it->second);
    }
}

} // namespace

bool ensureCaveRegion(GenContext& ctx, const GenBounds& b, int maxCells) {
    if (!ctx.cavePlan) {
        ctx.cavePlan = std::make_shared<CavePlan>();
        ctx.caveEntrances.clear();
        ctx.caveLights.clear();
        ctx.megaCaverns.clear();
    }
    CavePlan& plan = *ctx.cavePlan;
    auto t0 = std::chrono::steady_clock::now();

    // Collect EVERY missing cell up front — the gather range plus the source
    // ranges of unresolved mega cells (whose proposals come from systems up
    // to WORM_REACH beyond them; this was the serial 0.4 s burst when first
    // entering a new mega region) — in lattice order, deduplicated.
    std::vector<glm::ivec2> missing;
    auto collect = [&](int cx0, int cx1, int cz0, int cz1) {
        for (int cz = cz0; cz <= cz1; ++cz)
        for (int cx = cx0; cx <= cx1; ++cx)
            if (!plan.cells.count(chunkKey(cx, cz)))
                missing.push_back({ cx, cz });
    };
    collect(floorDiv(b.x0 - CAVE_GATHER_REACH, CAVE_CELL),
            floorDiv(b.x1 - 1 + CAVE_GATHER_REACH, CAVE_CELL),
            floorDiv(b.z0 - CAVE_GATHER_REACH, CAVE_CELL),
            floorDiv(b.z1 - 1 + CAVE_GATHER_REACH, CAVE_CELL));
    int mx0 = floorDiv(b.x0 - MEGA_REACH, MEGA_CELL);
    int mx1 = floorDiv(b.x1 - 1 + MEGA_REACH, MEGA_CELL);
    int mz0 = floorDiv(b.z0 - MEGA_REACH, MEGA_CELL);
    int mz1 = floorDiv(b.z1 - 1 + MEGA_REACH, MEGA_CELL);
    for (int mz = mz0; mz <= mz1; ++mz)
    for (int mx = mx0; mx <= mx1; ++mx) {
        if (plan.megaCells.count(chunkKey(mx, mz))) continue;
        int x0 = mx * MEGA_CELL, z0 = mz * MEGA_CELL;
        collect(floorDiv(x0 - WORM_REACH, CAVE_CELL),
                floorDiv(x0 + MEGA_CELL - 1 + WORM_REACH, CAVE_CELL),
                floorDiv(z0 - WORM_REACH, CAVE_CELL),
                floorDiv(z0 + MEGA_CELL - 1 + WORM_REACH, CAVE_CELL));
    }
    std::sort(missing.begin(), missing.end(), [](glm::ivec2 a, glm::ivec2 c) {
        return a.y != c.y ? a.y < c.y : a.x < c.x;
    });
    missing.erase(std::unique(missing.begin(), missing.end(),
                              [](glm::ivec2 a, glm::ivec2 c) {
                                  return a.x == c.x && a.y == c.y;
                              }),
                  missing.end());

    // BUDGET (streaming): walk at most `maxCells` this call so the main
    // thread never blocks on a big worm-walk burst (the 80 ms step spike
    // when the frontier crossed into fresh cave cells). The caller gates
    // carve/decor on the `true` return — until then those chunks just wait,
    // so the frontier (and the reveal curtain) eases in a touch rather than
    // hitching. The unbudgeted startup call (maxCells == INT_MAX) keeps the
    // one-shot parallel fan.
    if (!missing.empty()) {
        // A single cell's worm-walk is ~3-4 ms, so always fan across cores
        // (even the small budgeted slices): 12 cells then costs ~one parallel
        // layer (~4 ms) instead of ~45 ms serial. The budget caps cells per
        // call so a streaming crossing never walks the whole 300-cell burst
        // at once.
        bool partial = int(missing.size()) > maxCells;
        if (partial) missing.resize(maxCells);   // nearest-lattice-order prefix
        std::vector<CellWalkOut> outs(missing.size());
        std::atomic<int> next{0};
        int nThreads = int(std::max(1u, std::thread::hardware_concurrency() - 1));
        std::vector<std::thread> workers;
        for (int t = 0; t < nThreads; ++t)
            workers.emplace_back([&]() {
                int i;
                while ((i = next.fetch_add(1)) < int(missing.size()))
                    walkCellPure(ctx.seed, missing[i].x, missing[i].y, outs[i]);
            });
        for (auto& th : workers) th.join();
        for (size_t i = 0; i < missing.size(); ++i)
            applyCellWalk(ctx, plan, missing[i].x, missing[i].y,
                          std::move(outs[i]));
        if (int(missing.size()) > 32) {
            double dt = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            std::printf("[caves] region walk: +%zu cells in %.2fs (%d systems "
                        "total, max reach %d)\n", missing.size(), dt,
                        ctx.stats.caveSystems, plan.statMaxReach);
        }
        if (partial) return false;   // more cells remain; caller retries
    }

    // All cells for the region are present — resolve its mega cells (their
    // source cells are now cached, so this is cheap) and report ready.
    for (int mz = mz0; mz <= mz1; ++mz)
    for (int mx = mx0; mx <= mx1; ++mx)
        resolveMegaCell(ctx, plan, mx, mz);
    return true;
}

// Carve every plan op that intersects `b`, clipped to `b`. Full-world bounds
// use the parallel brick-row shard layout; a chunk window uses one writer.
// All carve writes are AIR (idempotent), so a voxel carved by ops replayed
// from two neighbouring windows gets the same result as one monolithic pass.
void genCavesCarve(World& w, GenContext& ctx, const GenBounds& b,
                   CaveCarveStage* stage) {
    // Region gather: ops from every walked cell within WORM_REACH plus every
    // resolved mega cell within MEGA_REACH, in lattice order — the global
    // replay order any query reproduces. ensureCaveRegion pre-walked them
    // (gathers only find(); never mutate — safe under parallel computes).
    CavePlan& plan = *ctx.cavePlan;
    if (!b.isFullWorld()) {
        // Bounded window: one writer, ops clipped in x and z. Writes to
        // EXISTING mixed bricks apply directly — bricks are chunk-owned, so
        // parallel bounded carves never collide. SOLID->mixed births either
        // stage into `stage` (parallel compute; commit later in chunk order
        // via commitCaveCarve) or commit inline (the serial path).
        // Ops iterate IN PLACE (no gather vectors — pointer pushes were 40 MB
        // of traffic per chunk), and each cell's slices are z-sorted at walk
        // time, so a binary search skips ~96% of extent tests.
        ShardWriter sw{ w, b.x0, b.x1, b.z0, b.z1, {} };
        int zLo = b.z0 - SLICE_WMAX, zHi = b.z1 + SLICE_WMAX;
        forRegionCells(plan, b, WORM_REACH, [&](const CaveCell& c) {
            auto first = std::lower_bound(
                c.slices.begin(), c.slices.end(), float(zLo),
                [](const SliceOp& s, float z) { return s.pos.z < z; });
            for (auto it = first; it != c.slices.end() && it->pos.z < float(zHi); ++it) {
                int zmin, zmax, xmin, xmax;
                sliceZExtent(*it, zmin, zmax); sliceXExtent(*it, xmin, xmax);
                if (zmax < b.z0 || zmin >= b.z1 || xmax < b.x0 || xmin >= b.x1) continue;
                applySlice(sw, *it);
            }
            for (const RoomOp& r : c.rooms) {
                int zmin, zmax, xmin, xmax;
                roomZExtent(r, zmin, zmax); roomXExtent(r, xmin, xmax);
                if (zmax < b.z0 || zmin >= b.z1 || xmax < b.x0 || xmin >= b.x1) continue;
                applyRoom(sw, r);
            }
        });
        forRegionMegas(plan, b, [&](const MegaCellRes& m) {
            for (const RoomOp& r : m.rooms) {
                int zmin, zmax, xmin, xmax;
                roomZExtent(r, zmin, zmax); roomXExtent(r, xmin, xmax);
                if (zmax < b.z0 || zmin >= b.z1 || xmax < b.x0 || xmin >= b.x1) continue;
                applyRoom(sw, r);
            }
        });
        if (stage) {
            stage->newBricks = std::move(sw.newBricks);
            return;
        }
        CaveCarveStage inlineStage{ b, std::move(sw.newBricks) };
        commitCaveCarve(w, inlineStage);
        return;
    }
    // Full-window path: gather once (one query for the whole window).
    std::vector<const SliceOp*> allSlices;
    std::vector<const RoomOp*>  allRooms;
    forRegionCells(plan, b, WORM_REACH, [&](const CaveCell& c) {
        for (const SliceOp& s : c.slices) allSlices.push_back(&s);
        for (const RoomOp&  r : c.rooms)  allRooms.push_back(&r);
    });
    forRegionMegas(plan, b, [&](const MegaCellRes& m) {
        for (const RoomOp& r : m.rooms) allRooms.push_back(&r);
    });
    auto tStart = std::chrono::steady_clock::now();
    // ---- Phase B: carve, PARALLEL by spatial shard ------------------------
    // Shards are disjoint ranges of whole brick rows (z), so every brick is
    // written by exactly one thread. All carve writes are AIR (idempotent),
    // so op order within a shard is irrelevant. SOLID->mixed brick births are
    // staged per shard and merged serially below, in ascending slot order,
    // keeping pool allocation deterministic regardless of thread count.
    {
        constexpr int SHARDS     = 64;
        constexpr int ROWS_SHARD = BRICKS_XZ / SHARDS;          // brick rows per shard
        constexpr int ZS         = ROWS_SHARD * BRICK;          // voxels per shard in z
        // Bucket op indices by overlapped shard (z relative to the window).
        std::vector<std::vector<int>> sliceBuckets(SHARDS), roomBuckets(SHARDS);
        auto bucket = [&](std::vector<std::vector<int>>& bk, int idx, int zmin, int zmax) {
            int s0 = std::clamp((zmin - b.z0) / ZS, 0, SHARDS - 1);
            int s1 = std::clamp((zmax - b.z0) / ZS, 0, SHARDS - 1);
            for (int s = s0; s <= s1; ++s) bk[s].push_back(idx);
        };
        for (int i = 0; i < int(allSlices.size()); ++i) {
            int zmin, zmax; sliceZExtent(*allSlices[i], zmin, zmax);
            bucket(sliceBuckets, i, zmin, zmax);
        }
        for (int i = 0; i < int(allRooms.size()); ++i) {
            int zmin, zmax; roomZExtent(*allRooms[i], zmin, zmax);
            bucket(roomBuckets, i, zmin, zmax);
        }

        std::vector<ShardWriter*> writers(SHARDS, nullptr);
        std::atomic<int> nextShard{0};
        int nThreads = int(std::max(1u, std::thread::hardware_concurrency()));
        std::vector<std::thread> workers;
        for (int t = 0; t < nThreads; ++t) {
            workers.emplace_back([&]() {
                int s;
                while ((s = nextShard.fetch_add(1)) < SHARDS) {
                    auto* sw = new ShardWriter{ w, b.x0, b.x1,
                                                b.z0 + s * ZS, b.z0 + (s + 1) * ZS, {} };
                    for (int idx : sliceBuckets[s]) applySlice(*sw, *allSlices[idx]);
                    for (int idx : roomBuckets[s])  applyRoom (*sw, *allRooms[idx]);
                    writers[s] = sw;
                }
            });
        }
        for (auto& th : workers) th.join();

        // Serial merge: shard order = ascending z = ascending slot ranges;
        // within a shard, sort by slot. Pool indices are therefore assigned
        // in a fixed global order independent of thread scheduling.
        for (int s = 0; s < SHARDS; ++s) {
            ShardWriter* sw = writers[s];
            if (!sw) continue;
            std::vector<std::pair<int, const std::array<uint32_t, BRICK_U32>*>> items;
            items.reserve(sw->newBricks.size());
            for (const auto& kv : sw->newBricks) items.push_back({ kv.first, &kv.second });
            std::sort(items.begin(), items.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
            for (const auto& it : items) {
                uint32_t bi = w.allocBrickNoFill();
                std::memcpy(&w.brickpool[size_t(bi) * BRICK_U32],
                            it.second->data(), BRICK_U32 * sizeof(uint32_t));
                w.brickmap[it.first] = bi + 1u;
            }
            delete sw;
        }
    }
    double tCarve = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - tStart).count();
    std::printf("[caves] carve %.2fs (parallel)\n", tCarve);
}

// Build every decoration request whose anchor can reach `b`, now that ALL
// carving is finished (chunk-status ladder), so nothing is carved back out
// and probes see final cave geometry. Requests replay in global plan order,
// each re-seeded from its own recorded seed; writes clip to the bounds with
// a buffer overlay so this replayer's own out-of-bounds writes stay
// visible to its later probes. Stats and lights are counted only by the
// anchor's owner. Full-world bounds = the monolithic path (zero-cost
// passthrough, once per world via plan->built).
void genCavesBuild(World& w, GenContext& ctx, const GenBounds& b,
                   ChunkStageBuffer* buf, const UndoBand* undo) {
    CavePlan& plan = *ctx.cavePlan;
    bool full = b.isFullWorld();
    if (full) {
        if (plan.built) return;
        plan.built = true;
        buf = nullptr;          // monolithic path writes straight through
    }
    auto tStart = std::chrono::steady_clock::now();
    auto inMargin = [&](int x, int z) {
        return full || (x >= b.x0 - CAVE_DECO_MARGIN && x < b.x1 + CAVE_DECO_MARGIN &&
                        z >= b.z0 - CAVE_DECO_MARGIN && z < b.z1 + CAVE_DECO_MARGIN);
    };
    auto anchorIn = [&](int x, int z) {
        return full || (x >= b.x0 && x < b.x1 && z >= b.z0 && z < b.z1);
    };
    // Side-effect targets: buffered builds collect stats/lights in the buffer
    // (folded into ctx at commit, so parallel computes never touch ctx).
    GenStats& st = buf ? buf->stats : ctx.stats;
    std::vector<CaveLight>& lightsOut = buf ? buf->lights : ctx.caveLights;
    // Order-free writer (see ChunkStageBuffer in internal.h): bounded builds
    // park every write in the buffer and read it back first, so the live
    // world stays carve-only until all chunks commit; full-world builds
    // write straight through.
    // Read order: own replay buffer -> undo band (pre-decor state behind a
    // moving front) -> live world.
    auto dwGet = [&](int x, int y, int z) -> uint8_t {
        uint8_t m;
        if (buf  && buf->tryGet(x, y, z, m))  return m;
        if (undo && undo->tryGet(x, y, z, m)) return m;
        return w.getVoxel(x, y, z);
    };
    auto dwSet = [&](int x, int y, int z, uint8_t m) {
        if (buf) buf->set(x, y, z, m);
        else     w.setVoxelRaw(x, y, z, m);
    };

    std::mt19937 rng;
    auto frand = [&]() { return frand01(rng); };
    auto urand = [&]() { return frand11(rng); };
    // --- Cave formations (additive solid back into carved air) ----------
    // Place a solid voxel, sampling the strata so the rock matches its
    // surroundings (a tall column even transitions stone with depth).
    auto setSolid = [&](int x, int y, int z) {
        if (y < 2 || y >= GRID_Y - 1) return;
        dwSet(x, y, z, stoneAt(x, y, z));
    };
    // One horizontal slice of a formation: a disc of radius r with a noisy,
    // non-circular edge so the silhouette is irregular blocky rock, not a
    // clean cone/cylinder (matching the reference dripstone).
    auto discSolid = [&](int cx, int y, int cz, float r) {
        if (r < 0.6f) { setSolid(cx, y, cz); return; }   // tip / thin waist
        int ri = int(std::ceil(r)) + 1;
        for (int dz = -ri; dz <= ri; ++dz)
        for (int dx = -ri; dx <= ri; ++dx) {
            float dist = std::sqrt(float(dx*dx + dz*dz));
            float wob = smoothNoise3((cx+dx) * 0.25f, y * 0.18f, (cz+dz) * 0.25f) - 0.5f;
            if (dist <= r * (1.0f + 0.40f * wob)) setSolid(cx + dx, y, cz + dz);
        }
    };
    // Stalagmite: rises from the floor, wide base tapering to a point.
    auto buildStalagmite = [&](int cx, int cz, int yFloor, int H, float baseR) {
        for (int e = 1; e <= 3; ++e) discSolid(cx, yFloor - e, cz, baseR);  // embed base
        for (int yy = 0; yy <= H; ++yy) {
            float level = float(yy) / float(H);            // 0 base .. 1 tip
            discSolid(cx, yFloor + yy, cz, baseR * std::pow(1.0f - level, 0.6f));
        }
    };
    // Stalactite: hangs from the ceiling, wide top tapering to a point. The
    // wide top is pushed a few voxels INTO the ceiling so it always merges
    // with the rock above instead of floating just below it.
    auto buildStalactite = [&](int cx, int cz, int yCeil, int H, float baseR) {
        for (int e = 1; e <= 3; ++e) discSolid(cx, yCeil + e, cz, baseR);  // embed top
        for (int yy = 0; yy <= H; ++yy) {
            float level = float(yy) / float(H);            // 0 ceiling .. 1 tip
            discSolid(cx, yCeil - yy, cz, baseR * std::pow(1.0f - level, 0.6f));
        }
    };
    // Column: floor-to-ceiling hourglass — flared at both ends, thin waist
    // in the middle (the signature dripstone profile). Both flared ends are
    // pushed well into the floor/ceiling so they connect cleanly even where
    // the cave surface is bumpy, instead of ending in mid-air.
    auto buildColumn = [&](int cx, int cz, int yFloor, int yCeil, float waistR, float flareR) {
        int H = yCeil - yFloor;
        if (H < 2) return;
        for (int e = 1; e <= 6; ++e) {
            discSolid(cx, yFloor - e, cz, flareR);   // embed base into floor
            discSolid(cx, yCeil  + e, cz, flareR);   // embed top into ceiling
        }
        for (int yy = 0; yy <= H; ++yy) {
            float level = float(yy) / float(H);
            float edge = std::pow(std::fabs(level - 0.5f) * 2.0f, 1.5f);  // 0 mid, 1 ends
            discSolid(cx, yFloor + yy, cz, waistR + (flareR - waistR) * edge);
        }
    };

    // Probe the ACTUAL carved geometry (not the ideal room formula) so a
    // formation only ever attaches to real rock. This is the fix for
    // floaters: at an intersection another passage may have carved away the
    // ceiling/floor we'd otherwise anchor to, so we look for where solid
    // truly is and bail (return sentinel) if there's nothing to attach to.
    //   floorSurface: scan DOWN for the lowest air voxel sitting on solid.
    //   ceilSurface : scan UP   for the highest air voxel under solid.
    auto floorSurface = [&](int x, int yStart, int z, int reach) -> int {
        for (int y = yStart; y > yStart - reach && y > 2; --y)
            if (dwGet(x, y, z) == MAT_AIR) {
                uint8_t below = dwGet(x, y - 1, z);
                if (below != MAT_AIR && below != MAT_WATER) return y;
            }
        return -1000000;
    };
    auto ceilSurface = [&](int x, int yStart, int z, int reach) -> int {
        for (int y = yStart; y < yStart + reach && y < GRID_Y - 2; ++y)
            if (dwGet(x, y, z) == MAT_AIR) {
                uint8_t above = dwGet(x, y + 1, z);
                if (above != MAT_AIR && above != MAT_WATER) return y;
            }
        return -1000000;
    };
    // A glow-crystal cluster on the floor at (cx,cz,fY): a few thin tapering
    // spires of emissive MAT_CRYSTAL, plus a baked point light so the
    // surrounding cavern is softly lit. Cool, DESATURATED palette (icy
    // blue / teal / soft violet) with slight per-cluster variation -- not
    // neon -- to match the painterly look.
    auto setCrystal = [&](int x, int y, int z) {
        if (y < 2 || y >= GRID_Y - 1) return;
        dwSet(x, y, z, MAT_CRYSTAL);
    };
    // `ownLight` gates the side effects (light registration + stats): only
    // the chunk owning the request's anchor fires them, so margin replays
    // by neighbour chunks never duplicate a cluster's light.
    auto buildCrystalCluster = [&](int cx, int cz, int fY, bool ownLight) {
        float h = frand();
        glm::vec3 col = (h < 0.55f) ? glm::vec3(0.45f, 0.72f, 0.95f)   // icy blue
                      : (h < 0.85f) ? glm::vec3(0.42f, 0.85f, 0.80f)   // teal
                                    : glm::vec3(0.68f, 0.60f, 0.95f);  // soft violet
        int spires = 3 + int(frand() * 5.0f);          // 3-7
        for (int s = 0; s < spires; ++s) {
            int ox = int(std::round(urand() * 4.0f));
            int oz = int(std::round(urand() * 4.0f));
            int hgt = 2 + int(frand() * 5.0f);          // 2-6 tall
            float baseR = 0.8f + frand() * 1.4f;        // thin
            for (int yy = 0; yy <= hgt; ++yy) {
                float r = baseR * (1.0f - 0.85f * float(yy) / float(hgt));  // taper
                int ri = int(std::ceil(r));
                for (int dz = -ri; dz <= ri; ++dz)
                for (int dx = -ri; dx <= ri; ++dx)
                    if (float(dx*dx + dz*dz) <= r * r)
                        setCrystal(cx + ox + dx, fY + yy, cz + oz + dz);
            }
        }
        if (ownLight)
            lightsOut.push_back({ glm::vec3(float(cx), float(fY) + 3.0f, float(cz)),
                                  col, 1.0f });
    };

    // Generic voxel set (any material) for decoration.
    auto setDeco = [&](int x, int y, int z, uint8_t m) {
        if (y < 2 || y >= GRID_Y - 1) return;
        dwSet(x, y, z, m);
    };
    // Place a decoration voxel ONLY into air, so a mushroom growing near a
    // wall/crystal/another structure gets clipped at the surface instead of
    // carving its voxels into solid rock or replacing crystals.
    auto setDecoAir = [&](int x, int y, int z, uint8_t m) {
        if (y < 2 || y >= GRID_Y - 1) return;
        if (dwGet(x, y, z) == MAT_AIR) dwSet(x, y, z, m);
    };
    // Deterministic per-column hash (0..1) for scattering ground cover.
    auto phash = [](int x, int z) -> float {
        uint32_t h = uint32_t(x) * 0x9E3779B1u ^ uint32_t(z) * 0x85EBCA77u;
        h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12;
        return float(h & 0x00FFFFFFu) / float(0x01000000);
    };

    // --- Surface-creeping IVY (climbing-plant tropism walk) ----------------
    // An agent walks through AIR voxels that stay adjacent to rock, so it
    // hugs walls/ceilings and creeps across them. Each step's direction is a
    // tropism blend: momentum (long sweeps) + gravity (droop down walls) +
    // randomness. Lays a vine voxel per step, sprouts leaf clusters, drops
    // the odd dangling tendril, and branches into a network. A visited set
    // stops loops/overlap. (Standard ivy/climbing-plant model.)
    auto isRock = [&](int x, int y, int z) {
        uint8_t m = dwGet(x, y, z);
        return m == MAT_STONE || m == MAT_GREYSTONE || m == MAT_DARKSTONE ||
               m == MAT_BEDROCK || m == MAT_DIRT;
    };
    auto surfaceAir = [&](int x, int y, int z) {
        if (dwGet(x, y, z) != MAT_AIR) return false;
        return isRock(x+1,y,z) || isRock(x-1,y,z) || isRock(x,y+1,z) ||
               isRock(x,y-1,z) || isRock(x,y,z+1) || isRock(x,y,z-1);
    };
    auto vkey = [](int x, int y, int z) -> int64_t {
        return (int64_t(x) << 42) ^ (int64_t(y & 0x1FFFFF) << 21) ^ int64_t(z & 0x1FFFFF);
    };
    std::function<void(int,int,int, int,int, glm::vec3, int, uint8_t, float, float, int,
                       std::unordered_set<int64_t>&)> growIvy =
        [&](int x, int y, int z, int ax, int az, glm::vec3 head, int budget, uint8_t vmat,
            float leafCh, float branchCh, int depth, std::unordered_set<int64_t>& seen) {
        static const int N6[6][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
        // Wander cap (the decoration reach contract): the whole network —
        // branches included — stays within IVY_WANDER XZ of the request
        // anchor (ax, az). Steps/hops beyond the disc aren't candidates.
        auto inDisc = [&](int px, int pz) {
            return std::abs(px - ax) <= IVY_WANDER && std::abs(pz - az) <= IVY_WANDER;
        };
        for (int step = 0; step < budget; ++step) {
            setDecoAir(x, y, z, vmat);
            seen.insert(vkey(x, y, z));
            // Leaf fill: thicken the line into a surface-hugging swath by
            // painting nearby surface-air voxels (denser = more sheet-like).
            if (frand() < leafCh) {
                for (int dz = -1; dz <= 1; ++dz)
                for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx)
                    if (frand() < 0.55f) setDecoAir(x+dx, y+dy, z+dz, vmat);  // air-checked
            }
            // Dangling tendril off the surface.
            if (frand() < 0.05f) {
                int tl = 2 + int(rng() % 4);
                for (int t = 1; t <= tl; ++t) {
                    if (dwGet(x, y - t, z) != MAT_AIR) break;
                    setDecoAir(x, y - t, z, vmat);
                }
            }
            // Pick the next surface-air step via the tropism blend.
            int bestI = -1; float bestS = -1e9f;
            for (int i = 0; i < 6; ++i) {
                int nx = x+N6[i][0], ny = y+N6[i][1], nz = z+N6[i][2];
                if (!inDisc(nx, nz)) continue;          // wander cap
                if (!surfaceAir(nx, ny, nz) || seen.count(vkey(nx, ny, nz))) continue;
                glm::vec3 d((float)N6[i][0], (float)N6[i][1], (float)N6[i][2]);
                float s = 1.0f * glm::dot(d, head)   // momentum
                        + 0.4f * (-d.y)              // gravity droop
                        + 0.7f * frand();            // wander
                if (s > bestS) { bestS = s; bestI = i; }
            }
            if (bestI < 0) {
                // Out of immediate surface to cling to: HOP to a nearby
                // unvisited surface-air cell so the network keeps spreading
                // (this is what stops creepers dying after a few voxels).
                bool hopped = false;
                for (int tries = 0; tries < 14 && !hopped; ++tries) {
                    int hx = x + int(rng()%5) - 2;
                    int hy = y + int(rng()%5) - 2;
                    int hz = z + int(rng()%5) - 2;
                    if ((hx!=x || hy!=y || hz!=z) && inDisc(hx, hz) &&
                        surfaceAir(hx,hy,hz) &&
                        !seen.count(vkey(hx,hy,hz))) { x=hx; y=hy; z=hz; hopped=true; }
                }
                if (hopped) continue;
                int tl = 2 + int(rng() % 4);    // give up -> final dangling tendril
                for (int t = 1; t <= tl; ++t) {
                    if (dwGet(x, y - t, z) != MAT_AIR) break;
                    setDecoAir(x, y - t, z, vmat);
                }
                break;
            }
            glm::vec3 d((float)N6[bestI][0], (float)N6[bestI][1], (float)N6[bestI][2]);
            if (depth < 2 && budget > 12 && frand() < branchCh) {
                glm::vec3 bh = head + glm::vec3(frand()-0.5f, frand()-0.5f, frand()-0.5f) * 1.6f;
                if (glm::length(bh) > 1e-3f)
                    growIvy(x, y, z, ax, az, glm::normalize(bh), budget/2, vmat,
                            leafCh, branchCh * 0.6f, depth + 1, seen);
            }
            x += N6[bestI][0]; y += N6[bestI][1]; z += N6[bestI][2];
            glm::vec3 nh = glm::mix(head, d, 0.5f);
            head = (glm::length(nh) > 1e-3f) ? glm::normalize(nh) : d;
        }
    };
    // One mushroom of half-width `capR` from floor-air (cx,fY,cz): a cream
    // stem under a cap of material `capMat`. `flat`=false -> a rounded red
    // toadstool dome (with white `spots`); `flat`=true -> a squat tan cap
    // (a couple of wide layers, no point, no spots). Caps are one material,
    // shaded smoothly by face normal in the shader (no banding).
    auto buildMushroom = [&](int cx, int fY, int cz, float capR,
                             uint8_t capMat, bool flat, bool spots) {
        int stemH = 2 + int(frand() * 2.0f) + int(capR * 0.35f);
        bool thick = capR > 3.6f;
        for (int i = 0; i < stemH; ++i) {
            setDecoAir(cx, fY + i, cz, MAT_MUSHROOM_STEM);
            if (thick && i < stemH - 1) {
                setDecoAir(cx + 1, fY + i, cz, MAT_MUSHROOM_STEM);
                setDecoAir(cx, fY + i, cz + 1, MAT_MUSHROOM_STEM);
                setDecoAir(cx + 1, fY + i, cz + 1, MAT_MUSHROOM_STEM);
            }
        }
        int capBase = fY + stemH;
        if (flat) {
            // Squat flat cap: a few wide layers, each a bit narrower.
            int flatH = std::max(2, int(std::round(capR * 0.5f)));
            for (int yy = 0; yy < flatH; ++yy) {
                float r = capR * (1.0f - 0.45f * float(yy) / float(std::max(flatH - 1, 1)));
                int ri = int(std::round(r));
                for (int dz = -ri; dz <= ri; ++dz)
                for (int dx = -ri; dx <= ri; ++dx)
                    if (float(dx*dx + dz*dz) <= r * r + 0.25f)
                        setDecoAir(cx + dx, capBase + yy, cz + dz, capMat);
            }
            return;
        }
        // Rounded umbrella dome -- wider than tall at every size.
        int capH = std::max(2, int(std::round(capR * 0.6f)));
        for (int yy = 0; yy <= capH; ++yy) {
            float t = float(yy) / float(capH);
            float r = capR * std::sqrt(std::max(0.0f, 1.0f - t * t));   // ->0 at the peak
            int ri = int(std::round(r));
            for (int dz = -ri; dz <= ri; ++dz)
            for (int dx = -ri; dx <= ri; ++dx) {
                float d2 = float(dx*dx + dz*dz);
                if (d2 > r * r + 0.25f) continue;
                setDecoAir(cx + dx, capBase + yy, cz + dz, capMat);
            }
        }
        if (!spots) return;
        // White spots on the dome FLANKS: dispersed (never touching), never
        // on the peak layer or the rim, never dead-centre.
        int nspot = 2 + int(capR * 0.6f);
        glm::ivec3 sp[10]; int nsp = 0;
        for (int att = 0; att < nspot * 5 && nsp < nspot && nsp < 10; ++att) {
            float a  = frand() * 6.2831853f;
            int   yy = 1 + int(frand() * float(std::max(capH - 1, 1)));  // [1, capH-1]
            if (yy >= capH) yy = capH - 1;
            if (yy < 1) yy = 1;
            float t  = float(yy) / float(capH);
            float r  = capR * std::sqrt(std::max(0.0f, 1.0f - t * t));
            float sr = std::max(1.0f, r * 0.8f);          // out on the flank, not centre
            int sx = int(std::round(std::cos(a) * sr));
            int sz = int(std::round(std::sin(a) * sr));
            bool tooClose = false;
            for (int u = 0; u < nsp; ++u) {
                int ddx = sp[u].x - sx, ddy = sp[u].y - yy, ddz = sp[u].z - sz;
                if (ddx*ddx + ddy*ddy + ddz*ddz < 6) { tooClose = true; break; }  // ~<2.4 apart
            }
            if (tooClose) continue;
            // Only paint a spot where this cap actually has a voxel -- never
            // in air or where the cap was clipped by a wall.
            if (dwGet(cx + sx, capBase + yy, cz + sz) != capMat) continue;
            setDeco(cx + sx, capBase + yy, cz + sz, MAT_MUSHROOM_SPOT);
            sp[nsp++] = glm::ivec3(sx, yy, sz);
        }
    };
    // A floor decoration site at (cx,fY,cz): usually a noisy DIRT patch with
    // a few mushrooms on it; sometimes mushrooms straight on bare stone.
    auto buildFloorPatch = [&](int cx, int fY, int cz, float Rin) {
        bool dirt = (frand() < 0.7f);
        // Rin<=0 -> a big cavern patch; >0 -> that radius (small trail cover).
        float R = (Rin > 0.0f) ? Rin : (7.0f + frand() * 9.0f);
        // Each patch is ONE species (red toadstools or tan flat-caps), so
        // the two kinds cluster together rather than intermingling.
        bool brown = (frand() < 0.42f);
        uint8_t capMat = brown ? MAT_MUSHROOM_CAP2 : MAT_MUSHROOM_CAP;
        if (dirt) {
            int ri = int(std::ceil(R));
            for (int dz = -ri; dz <= ri; ++dz)
            for (int dx = -ri; dx <= ri; ++dx) {
                float dist = std::sqrt(float(dx*dx + dz*dz));
                float wob = smoothNoise3((cx+dx) * 0.25f, fY * 0.2f, (cz+dz) * 0.25f) - 0.5f;
                if (dist > R * (1.0f + 0.4f * wob)) continue;
                // Recolour the top 1-2 floor voxels to dirt -- but ONLY where
                // they're stone, so an overlapping patch can never repaint a
                // neighbour's mushroom (or a crystal) into dirt.
                auto isStone = [](uint8_t m) {
                    return m == MAT_STONE || m == MAT_GREYSTONE ||
                           m == MAT_DARKSTONE || m == MAT_BEDROCK;
                };
                if (isStone(dwGet(cx+dx, fY-1, cz+dz))) setDeco(cx+dx, fY-1, cz+dz, MAT_DIRT);
                if (isStone(dwGet(cx+dx, fY-2, cz+dz))) setDeco(cx+dx, fY-2, cz+dz, MAT_DIRT);
            }
        }
        // Mushrooms with a MIX of sizes (frand^2 -> mostly small, a few
        // big). Spaced by the sum of radii so caps cluster like the
        // reference without merging into a blob, and -- crucially -- so each
        // probe lands on clear ground, never a neighbour's cap (the old
        // stacking bug). Re-probing at spaced columns seats each on its own
        // local floor.
        struct Placed { int x, z; float r; };
        Placed used[8]; int nUsed = 0;
        int target = (dirt ? 2 : 1) + int(frand() * 3.0f);
        for (int attempt = 0; attempt < 30 && nUsed < target && nUsed < 8; ++attempt) {
            int ox = int(std::round(urand() * R * 0.7f));
            int oz = int(std::round(urand() * R * 0.7f));
            float capR = 2.0f + frand() * frand() * 2.6f;   // ~2.0-4.6, biased small
            capR = std::min(capR, R * 0.7f);                 // keep caps inside small patches
            bool tooClose = false;
            for (int u = 0; u < nUsed; ++u) {
                float dd = std::sqrt(float((used[u].x-ox)*(used[u].x-ox) +
                                           (used[u].z-oz)*(used[u].z-oz)));
                if (dd < used[u].r + capR + 2.5f) { tooClose = true; break; }  // clear floor gap
            }
            if (tooClose) continue;
            int mfY = floorSurface(cx + ox, fY + 4, cz + oz, 12);
            if (mfY > -100000) {
                buildMushroom(cx + ox, mfY, cz + oz, capR, capMat, brown, !brown);
                used[nUsed++] = { ox, oz, capR };
            }
        }

        // Ground cover: a low MOSS carpet over the dirt -- dense in the
        // centre, thinning to the rim (bare dirt showing), clumped by noise.
        // Only on dirt voxels, placed into air (so it tucks around mushroom
        // stems). Sparser the deeper we are (drier), matching the shader's
        // green->brown depth tint.
        if (dirt) {
            float belowSurf = float(SURFACE_Y - fY);
            float depthFade = std::max(0.35f, 1.0f - belowSurf / 2500.0f);
            int ri = int(std::ceil(R));
            for (int dz = -ri; dz <= ri; ++dz)
            for (int dx = -ri; dx <= ri; ++dx) {
                if (dwGet(cx+dx, fY-1, cz+dz) != MAT_DIRT) continue;  // only on dirt
                float dist  = std::sqrt(float(dx*dx + dz*dz)) / R;          // 0 centre .. 1 rim
                float clump = smoothNoise3((cx+dx) * 0.18f, fY * 0.1f, (cz+dz) * 0.18f);
                float density = (0.95f - 0.8f * dist) * (0.35f + 0.95f * clump) * depthFade;
                if (phash(cx+dx, cz+dz) < density) {
                    setDecoAir(cx+dx, fY, cz+dz, MAT_CAVE_MOSS);
                    if (phash(cx+dx + 101, cz+dz + 57) < 0.22f)
                        setDecoAir(cx+dx, fY+1, cz+dz, MAT_CAVE_MOSS);    // taller tuft
                }
            }
        }
    };

    // Place one formation at column (gx,gz) sized to the real air gap there,
    // shared by caverns and galleries. Returns nothing if the column has no
    // attachable surfaces (avoids floaters). Columns are only allowed where
    // `allowColumn` is set -- the central, full-height, free-standing part
    // of a room -- never against an edge/wall (physically wrong, and the
    // sloped ceiling there won't meet the column top).
    auto placeFormation = [&](int gx, int gz, int yProbe, int reach, float colChance) {
        int fY = floorSurface(gx, yProbe, gz, reach);
        int cY = ceilSurface (gx, yProbe - reach / 2, gz, reach);
        if (fY < -100000 || cY < -100000) return;        // nothing to attach to
        int gap = cY - fY;
        if (gap < 6) return;
        // A column needs a stalactite + stalagmite to BRIDGE the gap, which
        // only happens where the ceiling is low enough -- common in low
        // galleries, rare in tall caverns. So bridging gets less likely the
        // taller the gap, on top of the per-context base chance.
        float gt = (float(gap) - 14.0f) / 36.0f;
        gt = gt < 0.0f ? 0.0f : (gt > 1.0f ? 1.0f : gt);
        float bridge = colChance * (1.0f - gt);
        float roll = frand();
        if (roll < bridge) {
            float waist = 2.0f + frand() * 3.0f;
            buildColumn(gx, gz, fY, cY, waist, waist + 2.0f + frand() * 3.0f);
        } else if (roll < 0.66f) {
            int H = std::min(int(gap * 0.55f), 6 + int(frand() * 16.0f));
            buildStalagmite(gx, gz, fY, std::max(H, 3), 1.4f + frand() * 2.6f);
        } else {
            int H = std::min(int(gap * 0.50f), 5 + int(frand() * 14.0f));
            buildStalactite(gx, gz, cY, std::max(H, 3), 1.4f + frand() * 2.2f);
        }
    };

    // Replay requests in global plan order — TYPE-MAJOR (all formations,
    // then crystals, then deco, then ivy), each type iterating region cells
    // in lattice order then mega cells, IN PLACE (no gather vectors — this
    // runs per chunk). Requests whose anchor can't reach these bounds skip;
    // each re-seeds from its own recorded seed. Stats are counted by the
    // anchor's owner chunk only.
    auto forBoth = [&](auto&& fn) {
        forRegionCells(plan, b, CAVE_GATHER_REACH,
                       [&](const CaveCell& c) { fn(c); });
        forRegionMegas(plan, b, [&](const MegaCellRes& m) { fn(m); });
    };
    // Formations first.
    forBoth([&](const auto& cell) {
        for (const FormReq& f : cell.forms) {
            if (!inMargin(f.x, f.z)) continue;
            rng.seed(f.seed);
            placeFormation(f.x, f.z, f.yProbe, f.reach, f.colChance);
        }
    });
    // Then the glow-crystal clusters, anchored to the real floor (each owner
    // records a baked light so its cavern self-illuminates).
    forBoth([&](const auto& cell) {
        for (const CrystalReq& c : cell.crystals) {
            if (!inMargin(c.p.x, c.p.z)) continue;
            int fY = floorSurface(c.p.x, c.p.y, c.p.z, 90);
            if (fY > -100000) {
                rng.seed(c.seed);
                bool own = anchorIn(c.p.x, c.p.z);
                buildCrystalCluster(c.p.x, c.p.z, fY, own);
                if (own) ++st.crystalsBuilt;
            }
        }
    });
    // Then dirt patches + mushrooms on the real floor of every cavern.
    forBoth([&](const auto& cell) {
        for (const DecoReq& d : cell.decos) {
            if (!inMargin(d.x, d.z)) continue;
            int fY = floorSurface(d.x, d.yProbe, d.z, 90);
            if (fY > -100000) {
                rng.seed(d.seed);
                buildFloorPatch(d.x, fY, d.z, d.R);
                if (anchorIn(d.x, d.z)) ++st.decoBuilt;
            }
        }
    });
    // Then ivy creepers, each from a real ceiling/upper-wall point, spreading
    // across the surfaces (no floaters; geometry is final here).
    forBoth([&](const auto& cell) {
        for (const IvyReq& v : cell.ivys) {
            if (!inMargin(v.x, v.z)) continue;
            int cY = ceilSurface(v.x, v.yProbe - 25, v.z, 55);
            if (cY < -100000) continue;
            rng.seed(v.seed);
            uint8_t vmat = v.alive ? MAT_VINE : MAT_VINE_DEAD;
            std::unordered_set<int64_t> seen;
            glm::vec3 head = glm::normalize(glm::vec3(frand() - 0.5f, -0.35f, frand() - 0.5f)
                                            + glm::vec3(0.001f, 0.0f, 0.0f));
            growIvy(v.x, cY, v.z, v.x, v.z, head, v.budget, vmat, v.leafCh, v.branchCh,
                    0, seen);
            if (anchorIn(v.x, v.z)) ++st.ivyBuilt;
        }
    });
    if (full) {
        double tBuild = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - tStart).count();
        std::printf("[caves] %d worms (%d deep), %d megas, %d crystals, "
                    "%d deco, %d ivy; build %.2fs\n",
                    ctx.stats.caveSystems, ctx.stats.deepSystems, ctx.stats.megaCarved,
                    ctx.stats.crystalsBuilt,
                    ctx.stats.decoBuilt, ctx.stats.ivyBuilt, tBuild);
    }
}

void commitCaveCarve(World& w, CaveCarveStage& stage) {
    // Births sorted by slot so pool indices assign in a fixed order per
    // chunk; the driver commits chunks in iteration order, making the whole
    // allocation sequence deterministic. (contentHash is allocation-order
    // independent anyway — this just keeps runs byte-comparable.)
    if (stage.sorted.empty()) stage.presort();   // serial callers; the
                                                 // streamer presorts on its
                                                 // compute workers
    for (const auto& it : stage.sorted) {
        uint32_t bi = w.allocBrickNoFill();
        std::memcpy(&w.brickpool[size_t(bi) * BRICK_U32],
                    it.second.data(), BRICK_U32 * sizeof(uint32_t));
        w.brickmap[it.first] = bi + 1u;
    }
    stage.sorted.clear();
    stage.newBricks.clear();
}

// Monolithic entry point: plan the region, carve everything, decorate. The
// chunked drivers instead call ensureCaveRegion + per-chunk genCavesCarve,
// then per-chunk genCavesBuild after ALL carving (the chunk-status ladder).
void genCaves(World& w, GenContext& ctx, const GenBounds& b) {
    ensureCaveRegion(ctx, b);
    genCavesCarve(w, ctx, b);
    if (b.isFullWorld()) genCavesBuild(w, ctx, b);
}
