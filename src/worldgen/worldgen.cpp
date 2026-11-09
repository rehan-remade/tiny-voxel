#include "worldgen/worldgen.h"
#include "world.h"
#include "common.h"
#include "worldgen/internal.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cfloat>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_set>
#include <vector>

namespace {

// -- Parallel bulk-fill scaffolding -------------------------------------------
// The brickmap-fill loop is embarrassingly parallel per (bx, bz) column once
// the per-voxel functions (heightAt/stoneAt/smoothNoise) are pure. The only
// shared mutable state is the
// brickpool, which is grown via w.allocBrick(). To keep generation
// deterministic (cache cares about bit-identical output) we split into:
//   1) Parallel fill: each column emits a ColumnResult holding finalized
//      brickmap values for EMPTY/SOLID bricks plus packed voxel data for
//      MIXED bricks. No global writes.
//   2) Serial merge: walk columns in (bz, bx, by) order -- the original
//      serial order -- and assign pool indices via allocBrick. Pool indices
//      come out in the same order as the single-threaded code.

// Sentinel stored in brickmapVals for mixed bricks; the actual pool data
// lives in ColumnResult::mixedPool at the same by-iteration position. Real
// brickmap values are never 0xFFFFFFFF (BM_SOLID_FLAG=0x80000000 plus an
// 8-bit material; pool indices fit well under 2^31).
constexpr uint32_t BM_MIXED_PLACEHOLDER = 0xFFFFFFFFu;

struct ColumnResult {
    // One entry per by in [0, BRICKS_Y): final brickmap value, or
    // BM_MIXED_PLACEHOLDER if this brick is mixed and its voxel data is in
    // mixedPool (consumed in by-order during the merge).
    std::array<uint32_t, BRICKS_Y> brickmapVals;
    // Packed voxel data for each MIXED brick, in by-order. Reused across
    // stripes so the storage cycles instead of reallocating.
    std::vector<std::array<uint32_t, BRICK_U32>> mixedPool;
};

void fillColumn(int bx, int bz, ColumnResult& col) {
    int x0 = bx * BRICK, z0 = bz * BRICK;
    int hMin = INT32_MAX, hMax = INT32_MIN;
    for (int dz = 0; dz <= BRICK; dz += BRICK)
    for (int dx = 0; dx <= BRICK; dx += BRICK) {
        int h = heightAt(x0 + dx, z0 + dz);
        hMin = std::min(hMin, h);
        hMax = std::max(hMax, h);
    }
    col.mixedPool.clear();
    for (int by = 0; by < BRICKS_Y; ++by) {
        int y0 = by * BRICK;
        int y1 = y0 + BRICK - 1;
        if (y0 >= hMax + H_NOISE_MARGIN) {
            col.brickmapVals[by] = BM_EMPTY;
        } else if (y1 < hMin - H_NOISE_MARGIN - 4) {
            // Below the surface transition: solid stone. Caves are carved as
            // a separate agent-based pass (worms) after the bulk fill.
            col.brickmapVals[by] = BM_SOLID_FLAG | uint32_t(
                stoneAt(x0 + BRICK/2, y0 + BRICK/2, z0 + BRICK/2));
        } else {
            col.brickmapVals[by] = BM_MIXED_PLACEHOLDER;
            col.mixedPool.emplace_back();
            auto& poolData = col.mixedPool.back();
            poolData.fill(0u);
            for (int lz = 0; lz < BRICK; ++lz)
            for (int lx = 0; lx < BRICK; ++lx) {
                int wx = x0 + lx, wz = z0 + lz;
                // Column surface data + the per-voxel material grammar live in
                // internal.h (surfaceColAt / surfaceMatAt) — SHARED with the
                // far-field LOD ring generator so the coarse horizon can never
                // drift from the real surface (see the design-note comments
                // on the grammar there).
                SurfaceCol sc = surfaceColAt(wx, wz);
                int h = sc.h;
                for (int ly = 0; ly < BRICK; ++ly) {
                    int y = y0 + ly;
                    if (y >= h) continue;
                    uint8_t mat = surfaceMatAt(wx, y, wz, sc);
                    int linear = voxInBrick(lx, ly, lz);
                    int wslot  = linear >> 2;
                    int wshift = (linear & 3) * 8;
                    uint32_t mask = 0xFFu << wshift;
                    uint32_t& word = poolData[wslot];
                    word = (word & ~mask) | (uint32_t(mat) << wshift);
                }
            }
        }
    }
}

// Persistent pool of worker threads. Each call to runStripe submits a fresh
// (bz, cols) job, fans out the BRICKS_XZ columns across workers via an atomic
// counter, and blocks until every worker has drained the stripe.
class StripePool {
public:
    StripePool() {
        int N = int(std::max(1u, std::thread::hardware_concurrency()));
        workers_.reserve(N);
        N_ = N;
        for (int i = 0; i < N; ++i) {
            workers_.emplace_back([this] { workerLoop(); });
        }
    }
    ~StripePool() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
            ++generation_;
        }
        cvWork_.notify_all();
        for (auto& w : workers_) w.join();
    }
    int threadCount() const { return N_; }

    void runStripe(int bz, int obx, std::vector<ColumnResult>& cols) {
        nextBx_.store(0, std::memory_order_relaxed);
        finished_.store(0, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lk(m_);
            bz_ = bz;
            obx_ = obx;
            cols_ = &cols;
            ++generation_;
        }
        cvWork_.notify_all();
        std::unique_lock<std::mutex> lk(m_);
        cvDone_.wait(lk, [&] { return finished_.load() == N_; });
    }

private:
    void workerLoop() {
        int seenGen = 0;
        for (;;) {
            std::unique_lock<std::mutex> lk(m_);
            cvWork_.wait(lk, [&] { return stop_ || generation_ > seenGen; });
            if (stop_) return;
            seenGen = generation_;
            int bz = bz_, obx = obx_;
            std::vector<ColumnResult>* cols = cols_;
            lk.unlock();

            for (;;) {
                int bx = nextBx_.fetch_add(1, std::memory_order_relaxed);
                if (bx >= BRICKS_XZ) break;
                fillColumn(obx + bx, bz, (*cols)[bx]);   // world brick coords
            }

            if (finished_.fetch_add(1, std::memory_order_acq_rel) + 1 == N_) {
                std::lock_guard<std::mutex> lk2(m_);
                cvDone_.notify_one();
            }
        }
    }

    int N_ = 0;
    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable cvWork_, cvDone_;
    int generation_ = 0;          // bumped per stripe so workers wake exactly once
    bool stop_ = false;
    int bz_ = 0, obx_ = 0;
    std::vector<ColumnResult>* cols_ = nullptr;
    std::atomic<int> nextBx_{0};
    std::atomic<int> finished_{0};
};

// Persistent worker pool for per-chunk stage computes: fan a batch of jobs
// across workers via an atomic cursor, block until drained. The 6e streaming
// runtime will feed this incrementally (priority by distance to player);
// batch mode is what the generation drivers need today.
class ChunkWorkerPool {
public:
    ChunkWorkerPool() {
        N_ = std::max(1, int(std::thread::hardware_concurrency()) - 2);
        done_.store(N_, std::memory_order_relaxed);   // idle() true pre-dispatch
        workers_.reserve(N_);
        for (int i = 0; i < N_; ++i) workers_.emplace_back([this] { loop(); });
    }
    ~ChunkWorkerPool() {
        { std::lock_guard<std::mutex> lk(m_); stop_ = true; ++gen_; }
        cvWork_.notify_all();
        for (auto& t : workers_) t.join();
    }
    int threads() const { return N_; }
    void runBatch(int count, const std::function<void(int)>& fn) {
        if (count <= 0) return;
        dispatch(count, fn);
        wait();
    }
    // Non-blocking fan-out: returns immediately; poll idle() and call wait()
    // (or just commit once idle) later. One batch in flight at a time.
    void dispatch(int count, std::function<void(int)> fn) {
        if (count <= 0) return;
        fnStore_ = std::move(fn);
        next_.store(0, std::memory_order_relaxed);
        done_.store(0, std::memory_order_relaxed);
        { std::lock_guard<std::mutex> lk(m_); count_ = count; fn_ = &fnStore_; ++gen_; }
        cvWork_.notify_all();
    }
    bool idle() const { return done_.load(std::memory_order_acquire) == N_; }
    void wait() {
        std::unique_lock<std::mutex> lk(m_);
        cvDone_.wait(lk, [&] { return done_.load() == N_; });
    }
private:
    void loop() {
        int seen = 0;
        for (;;) {
            const std::function<void(int)>* fn = nullptr;
            int count = 0;
            {
                std::unique_lock<std::mutex> lk(m_);
                cvWork_.wait(lk, [&] { return stop_ || gen_ > seen; });
                if (stop_) return;
                seen = gen_; fn = fn_; count = count_;
            }
            int i;
            while ((i = next_.fetch_add(1, std::memory_order_relaxed)) < count)
                (*fn)(i);
            if (done_.fetch_add(1, std::memory_order_acq_rel) + 1 == N_) {
                std::lock_guard<std::mutex> lk(m_);
                cvDone_.notify_one();
            }
        }
    }
    int N_ = 0;
    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable cvWork_, cvDone_;
    int gen_ = 0;
    bool stop_ = false;
    int count_ = 0;
    const std::function<void(int)>* fn_ = nullptr;
    std::function<void(int)> fnStore_;    // keeps async dispatch fns alive
    std::atomic<int> next_{0};
    std::atomic<int> done_{0};
};

// One chunk's bulk-fill, staged: compute the 8x8 brick columns into buffers
// (parallel-safe — no world writes), commit with deterministic pool
// allocation (serial, chunk iteration order).
void bulkComputeChunk(const GenBounds& b, std::vector<ColumnResult>& cols) {
    cols.resize(size_t(CHUNK_BRICKS) * CHUNK_BRICKS);
    int bx0 = b.x0 >> BRICK_BITS, bz0 = b.z0 >> BRICK_BITS;
    size_t i = 0;
    for (int bz = bz0; bz < bz0 + CHUNK_BRICKS; ++bz)
    for (int bx = bx0; bx < bx0 + CHUNK_BRICKS; ++bx)
        fillColumn(bx, bz, cols[i++]);
}
void bulkCommitChunk(World& w, const GenBounds& b, std::vector<ColumnResult>& cols) {
    int bx0 = b.x0 >> BRICK_BITS, bz0 = b.z0 >> BRICK_BITS;
    size_t i = 0;
    for (int bz = bz0; bz < bz0 + CHUNK_BRICKS; ++bz)
    for (int bx = bx0; bx < bx0 + CHUNK_BRICKS; ++bx, ++i) {
        ColumnResult& col = cols[i];
        size_t mixIdx = 0;
        for (int by = 0; by < BRICKS_Y; ++by) {
            int slot = w.slotOfBrick(bx, by, bz);
            uint32_t v = col.brickmapVals[by];
            if (v == BM_MIXED_PLACEHOLDER) {
                uint32_t bi = w.allocBrickNoFill();
                std::memcpy(&w.brickpool[size_t(bi) * BRICK_U32],
                            col.mixedPool[mixIdx].data(),
                            BRICK_U32 * sizeof(uint32_t));
                w.brickmap[slot] = bi + 1u;
                ++mixIdx;
            } else {
                w.brickmap[slot] = v;
            }
        }
    }
}

} // namespace

void ChunkStageBuffer::commit(World& w, GenContext& ctx, UndoBand* undo) const {
    // In-bounds pages only — chunk bounds are brick-aligned, so a page is
    // entirely in or out (one test rejects 512 voxels of margin redundancy).
    // Each page already resolved write conflicts in global feature order,
    // and chunks own disjoint in-bounds bricks, so commit order cannot
    // affect voxel content. Stats fold and lights append here so parallel
    // computes never touch the context.
    int bx0 = b.x0 >> BRICK_BITS, bx1 = b.x1 >> BRICK_BITS;
    int bz0 = b.z0 >> BRICK_BITS, bz1 = b.z1 >> BRICK_BITS;
    for (const auto& kv : pages) {
        int bx = keyBX(kv.first), by = keyBY(kv.first), bz = keyBZ(kv.first);
        if (bx < bx0 || bx >= bx1 || bz < bz0 || bz >= bz1) continue;
        int X = bx << BRICK_BITS, Y = by << BRICK_BITS, Z = bz << BRICK_BITS;
        const StagePage& p = kv.second;
        for (int ly = 0; ly < BRICK; ++ly)
        for (int lz = 0; lz < BRICK; ++lz)
        for (int lx = 0; lx < BRICK; ++lx) {
            int v = voxInBrick(lx, ly, lz);
            if (!(p.mask[v >> 6] & (1ull << (v & 63)))) continue;
            if (undo) undo->record(w, b.z0 / CHUNK_VOX, X + lx, Y + ly, Z + lz);
            w.setVoxelRaw(X + lx, Y + ly, Z + lz, p.data[v]);
        }
    }
    addGenStats(ctx.stats, stats);
    ctx.caveLights.insert(ctx.caveLights.end(), lights.begin(), lights.end());
}

void SurfaceSnapshot::capture(const World& w) {
    // Whole resident window, at the world's current origin, stored at
    // toroidal texel indices (x & mask) — identical to a linear fill at
    // origin 0 (the offline drivers' case). Record the origin so get()'s
    // in-window bounds check rejects (rather than wraps) just-outside reads.
    ox = w.originVoxX(); oz = w.originVoxZ();
    constexpr int M = GRID_XZ - 1;
    mats.assign(size_t(GRID_XZ) * GRID_XZ * BAND, 0);
    int nThreads = int(std::max(1u, std::thread::hardware_concurrency()));
    std::vector<std::thread> workers;
    std::atomic<int> nextRow{0};
    for (int t = 0; t < nThreads; ++t) {
        workers.emplace_back([&]() {
            int r;
            while ((r = nextRow.fetch_add(1)) < GRID_XZ) {
                int z = oz + r, rz = z & M;
                for (int rx = 0; rx < GRID_XZ; ++rx) {
                    int x = ox + rx;
                    int y0 = heightAt(x, z) - 2;
                    uint8_t* dst = &mats[(size_t(rz) * GRID_XZ + (x & M)) * BAND];
                    for (int i = 0; i < BAND; ++i)
                        dst[i] = w.getVoxel(x, y0 + i, z);
                }
            }
        });
    }
    for (auto& th : workers) th.join();
}

void SurfaceSnapshot::capture(const World& w, const GenBounds& b) {
    if (mats.empty())
        mats.assign(size_t(GRID_XZ) * GRID_XZ * BAND, 0);
    // The in-window gate tracks the live window origin: bounded captures run
    // strip- or chunk-at-a-time as the frontier (or a ladder front) advances,
    // and the origin is the same for every call within one window placement.
    ox = w.originVoxX(); oz = w.originVoxZ();
    constexpr int M = GRID_XZ - 1;
    for (int z = b.z0; z < b.z1; ++z) {
        for (int x = b.x0; x < b.x1; ++x) {
            int y0 = heightAt(x, z) - 2;
            uint8_t* dst = &mats[(size_t(z & M) * GRID_XZ + (x & M)) * BAND];
            for (int i = 0; i < BAND; ++i)
                dst[i] = w.getVoxel(x, y0 + i, z);
        }
    }
}

void buildHeightmap(GenContext& ctx, int ox, int oz) {
    ctx.heightMap.assign(size_t(GRID_XZ) * GRID_XZ, float(SURFACE_Y));
    // Fill the window [ox, ox+GRID_XZ) x [oz, oz+GRID_XZ) at TOROIDAL texel
    // indices (worldCoord & mask) — matching the REPEAT-wrapped heightTex and
    // updateHeightStrip's incremental writes. At origin (0,0) the residue is
    // the linear index, so this is identical to the old startup fill.
    constexpr int M = GRID_XZ - 1;   // power-of-two mask
    int nThreads = int(std::max(1u, std::thread::hardware_concurrency()));
    std::vector<std::thread> workers;
    std::atomic<int> nextRow{0};
    for (int t = 0; t < nThreads; ++t) {
        workers.emplace_back([&]() {
            int rz;
            while ((rz = nextRow.fetch_add(1)) < GRID_XZ) {
                int wz = oz + rz, tz = wz & M;
                float* row = &ctx.heightMap[size_t(tz) * GRID_XZ];
                for (int rx = 0; rx < GRID_XZ; ++rx) {
                    int wx = ox + rx;
                    row[wx & M] = float(heightAt(wx, wz));
                }
            }
        });
    }
    for (auto& w : workers) w.join();
}

// Aggregate top-of-column surface COLOUR at (x,z) for the far horizon. Reuses
// biomeAt (the single source of truth): the biome's base ground tint, greyed on
// rocky high ground (mirrors fillColumn's rockiness rule), and darkened toward a
// forest canopy where flora density is high. Any future biome's BiomeParams
// (grassTint / rockiness / treeChance) feed this automatically — no per-biome
// far-field code. Packed 0xAABBGGRR for a GL_RGBA8 texture.
static uint32_t packSurfaceColor(const BiomeParams& b, int h, float canopy) {
    glm::vec3 col = b.grassTint;                                   // grass / sand / olive base
    float rockAmt = std::clamp(b.rockiness * (float(h - SURFACE_Y) / 220.0f - 0.30f) * 2.4f,
                               0.0f, 1.0f);                        // bare rock on high ground
    col = glm::mix(col, glm::vec3(0.42f, 0.42f, 0.45f), rockAmt);
    col = glm::mix(col, glm::vec3(0.15f, 0.30f, 0.14f), canopy);   // dense flora -> dark canopy
    col = glm::clamp(col, 0.0f, 1.0f);
    uint32_t r = uint32_t(col.r * 255.0f + 0.5f);
    uint32_t g = uint32_t(col.g * 255.0f + 0.5f);
    uint32_t bl = uint32_t(col.b * 255.0f + 0.5f);
    return r | (g << 8) | (bl << 16) | (0xFFu << 24);
}

void buildFarHeightmap(GenContext& ctx, int cx, int cz) {
    ctx.farHeightMap.assign(size_t(FAR_TEX) * FAR_TEX, float(SURFACE_Y));
    ctx.farColorMap.assign(size_t(FAR_TEX) * FAR_TEX, 0xFF808080u);
    ctx.farOriginX = cx - FAR_EXTENT / 2;          // region min corner (world voxels)
    ctx.farOriginZ = cz - FAR_EXTENT / 2;
    const int ox = ctx.farOriginX, oz = ctx.farOriginZ;
    int nThreads = int(std::max(1u, std::thread::hardware_concurrency()));
    std::vector<std::thread> workers;
    std::atomic<int> nextRow{ 0 };
    for (int t = 0; t < nThreads; ++t) {
        workers.emplace_back([&]() {
            int tz;
            while ((tz = nextRow.fetch_add(1)) < FAR_TEX) {
                int wz = oz + tz * FAR_STRIDE;
                float*    hrow = &ctx.farHeightMap[size_t(tz) * FAR_TEX];
                uint32_t* crow = &ctx.farColorMap[size_t(tz) * FAR_TEX];
                for (int tx = 0; tx < FAR_TEX; ++tx) {
                    int wx = ox + tx * FAR_STRIDE;
                    BiomeParams b = biomeAt(wx, wz);
                    int   h = heightAt(wx, wz);
                    float canopy = std::clamp(b.treeChance * 1.3f, 0.0f, 0.80f);
                    // Raise the far surface to ~tree-canopy height where flora is
                    // dense, with per-cell noise, so a distant forest is a bumpy
                    // raised dark-green CANOPY at tree-top level (not flat ground)
                    // — near 3D trees then dissolve into a canopy at the same
                    // height instead of vanishing. Non-forest (canopy~0) is unbumped.
                    float bump = canopy * 46.0f
                               * (0.55f + 0.65f * smoothNoise(wx * 0.05f + 4.0f, wz * 0.05f + 9.0f));
                    hrow[tx] = float(h) + bump;
                    crow[tx] = packSurfaceColor(b, h, canopy);
                }
            }
        });
    }
    for (auto& w : workers) w.join();
}

// Splat every recorded emissive source into the coarse 3D light volume with a
// smooth quadratic falloff. The volume is a TOROIDAL WINDOW (like heightTex):
// X/Z cell indices wrap by (LV_X-1)/(LV_Z-1), so it follows the resident
// window at origin (ox,oz). Only lights inside the window contribute; the
// shader samples worldPos with REPEAT wrap, so the same world point reads the
// same cell. (A light within ~80 vox of the window edge splats a faint tail
// onto the opposite edge via the wrap, but that edge sits beyond the reveal
// curtain, so it's never seen.) Origin (0,0) — the startup default — is the
// old flat behaviour. Cheap: in-window sources are few.
void buildLightVolume(GenContext& ctx, int ox, int oz) {
    ctx.lightVol.assign(size_t(LV_X) * LV_Y * LV_Z * 3, 0.0f);
    const float R = 80.0f;                 // light reach in voxels
    const int   Rc = int(R / LV_SCALE) + 1;
    for (const CaveLight& L : ctx.caveLights) {
        if (L.pos.x < float(ox) || L.pos.x >= float(ox + GRID_XZ) ||
            L.pos.z < float(oz) || L.pos.z >= float(oz + GRID_XZ)) continue;
        int bx = floorDiv(int(L.pos.x), LV_SCALE);   // base cell (unwrapped)
        int by = int(L.pos.y) / LV_SCALE;
        int bz = floorDiv(int(L.pos.z), LV_SCALE);
        for (int dz = -Rc; dz <= Rc; ++dz)
        for (int dy = -Rc; dy <= Rc; ++dy)
        for (int dx = -Rc; dx <= Rc; ++dx) {
            int vy = by + dy;
            if (vy < 0 || vy >= LV_Y) continue;   // Y is full-height, no wrap
            // World centre of this splat cell (unwrapped — for true falloff).
            float wx = (bx + dx + 0.5f) * LV_SCALE;
            float wy = (vy + 0.5f) * LV_SCALE;
            float wz = (bz + dz + 0.5f) * LV_SCALE;
            float dist = std::sqrt((wx-L.pos.x)*(wx-L.pos.x) + (wy-L.pos.y)*(wy-L.pos.y) + (wz-L.pos.z)*(wz-L.pos.z));
            float fall = 1.0f - dist / R;
            if (fall <= 0.0f) continue;
            fall *= fall;                  // softer core, smooth edge
            int vx = (bx + dx) & (LV_X - 1);   // toroidal store
            int vz = (bz + dz) & (LV_Z - 1);
            size_t idx = (size_t(vz) * LV_Y + vy) * LV_X * 3 + size_t(vx) * 3;
            ctx.lightVol[idx + 0] += L.color.r * L.strength * fall;
            ctx.lightVol[idx + 1] += L.color.g * L.strength * fall;
            ctx.lightVol[idx + 2] += L.color.b * L.strength * fall;
        }
    }
    // Overlapping clusters sum, so clusters packed together would blow out to
    // white. Cap each cell's brightness WITHOUT shifting its hue: scale the
    // RGB down by the largest channel over the cap. Set HIGH so ordinary
    // crystal caverns are untouched (full brightness) and only a pathological
    // pile-up gets bounded -- and it stays cyan rather than washing to white.
    const float CAP = 1.5f;
    for (size_t i = 0; i < ctx.lightVol.size(); i += 3) {
        float m = std::max(ctx.lightVol[i], std::max(ctx.lightVol[i+1], ctx.lightVol[i+2]));
        if (m > CAP) {
            float s = CAP / m;
            ctx.lightVol[i] *= s; ctx.lightVol[i+1] *= s; ctx.lightVol[i+2] *= s;
        }
    }
}

// Exactly rebuild a WORLD LV-cell box (full Y): clear it, resplat EVERY
// in-window light that reaches it, then cap once. Clear+resplat (not add) is
// what keeps this bit-identical to a full buildLightVolume over those cells —
// departed lights vanish, capping is order-independent even for multi-hue
// clusters (the pile-up cap is non-commutative under incremental add). Shared
// by the scroll-strip and new-light paths. R/Rc/CAP must match buildLightVolume.
static void rebuildLightRegion(GenContext& ctx, int ox, int oz,
                               int wcx0, int wcx1, int wcz0, int wcz1) {
    const float R = 80.0f;
    const int   Rc = int(R / LV_SCALE) + 1;
    const float CAP = 1.5f;
    if (wcx1 <= wcx0 || wcz1 <= wcz0) return;
    for (int cz = wcz0; cz < wcz1; ++cz) {
        int vz = cz & (LV_Z - 1);
        for (int cx = wcx0; cx < wcx1; ++cx) {
            int vx = cx & (LV_X - 1);
            for (int y = 0; y < LV_Y; ++y) {
                size_t idx = (size_t(vz) * LV_Y + y) * LV_X * 3 + size_t(vx) * 3;
                ctx.lightVol[idx + 0] = ctx.lightVol[idx + 1] = ctx.lightVol[idx + 2] = 0.0f;
            }
        }
    }
    // Region membership is TOROIDAL: a cell is in-region iff its texture index
    // lands within [rx0,rx0+rxw) (mod LV_X). Splat the FULL [-Rc,Rc] footprint
    // (matching buildLightVolume's edge wrap) and gate writes by that test —
    // clamping dx/dz to the window instead would silently drop wrapped contributions.
    int rx0 = wcx0 & (LV_X - 1), rxw = wcx1 - wcx0;
    int rz0 = wcz0 & (LV_Z - 1), rzw = wcz1 - wcz0;
    for (const CaveLight& L : ctx.caveLights) {
        if (L.pos.x < float(ox) || L.pos.x >= float(ox + GRID_XZ) ||
            L.pos.z < float(oz) || L.pos.z >= float(oz + GRID_XZ)) continue;
        int bx = floorDiv(int(L.pos.x), LV_SCALE);
        int by = int(L.pos.y) / LV_SCALE;
        int bz = floorDiv(int(L.pos.z), LV_SCALE);
        // Toroidal AABB early-out: if this light's footprint arc doesn't overlap
        // the region's arc on either axis, it touches no region cell — skip the
        // whole 13^3 inner loop. The region is usually a narrow band, so the
        // vast majority of in-window lights are rejected here in O(1).
        int _lw = 2 * Rc + 1;
        int _lax = (bx - Rc) & (LV_X - 1), _laz = (bz - Rc) & (LV_Z - 1);
        bool _ox = (((rx0 - _lax) & (LV_X - 1)) < _lw) || (((_lax - rx0) & (LV_X - 1)) < rxw);
        bool _oz = (((rz0 - _laz) & (LV_Z - 1)) < _lw) || (((_laz - rz0) & (LV_Z - 1)) < rzw);
        if (!_ox || !_oz) continue;
        for (int dz = -Rc; dz <= Rc; ++dz) {
            int vz = (bz + dz) & (LV_Z - 1);
            if (((vz - rz0) & (LV_Z - 1)) >= rzw) continue;   // outside region (toroidal)
            for (int dx = -Rc; dx <= Rc; ++dx) {
                int vx = (bx + dx) & (LV_X - 1);
                if (((vx - rx0) & (LV_X - 1)) >= rxw) continue;
                for (int dy = -Rc; dy <= Rc; ++dy) {
                    int vy = by + dy;
                    if (vy < 0 || vy >= LV_Y) continue;
                    float wx = (bx + dx + 0.5f) * LV_SCALE;
                    float wy = (vy + 0.5f) * LV_SCALE;
                    float wz = (bz + dz + 0.5f) * LV_SCALE;
                    float dist = std::sqrt((wx-L.pos.x)*(wx-L.pos.x) + (wy-L.pos.y)*(wy-L.pos.y) + (wz-L.pos.z)*(wz-L.pos.z));
                    float fall = 1.0f - dist / R;
                    if (fall <= 0.0f) continue;
                    fall *= fall;
                    size_t idx = (size_t(vz) * LV_Y + vy) * LV_X * 3 + size_t(vx) * 3;
                    ctx.lightVol[idx + 0] += L.color.r * L.strength * fall;
                    ctx.lightVol[idx + 1] += L.color.g * L.strength * fall;
                    ctx.lightVol[idx + 2] += L.color.b * L.strength * fall;
                }
            }
        }
    }
    for (int cz = wcz0; cz < wcz1; ++cz) {
        int vz = cz & (LV_Z - 1);
        for (int cx = wcx0; cx < wcx1; ++cx) {
            int vx = cx & (LV_X - 1);
            for (int y = 0; y < LV_Y; ++y) {
                size_t i = (size_t(vz) * LV_Y + y) * LV_X * 3 + size_t(vx) * 3;
                float m = std::max(ctx.lightVol[i], std::max(ctx.lightVol[i+1], ctx.lightVol[i+2]));
                if (m > CAP) { float s = CAP / m; ctx.lightVol[i] *= s; ctx.lightVol[i+1] *= s; ctx.lightVol[i+2] *= s; }
            }
        }
    }
}

// Incremental light-volume update for a single one-chunk window scroll. The
// volume is a toroidal window, so on a scroll only a thin strip of cells changes
// world content; a full 48 MB rebuild+upload every scroll (the old path) is
// wasteful. Rebuild only the LEADING band (entered lights, widened interior by
// Rc) and the TRAILING band (width Rc, to drop departed lights' inward reach).
// The two bands are adjacent in toroidal texture space, so the re-upload is one
// contiguous band of width STRIP + 2*Rc, returned via (outXAxis,outOff,outCnt)
// (outCnt may wrap the edge; the renderer splits it).
void buildLightVolumeStrip(GenContext& ctx, int ox, int oz, glm::ivec2 dir,
                           bool& outXAxis, int& outOff, int& outCnt) {
    const float R  = 80.0f;
    const int   Rc = int(R / LV_SCALE) + 1;
    const int STRIP = CHUNK_VOX / LV_SCALE;  // LV cells entering per chunk scroll (4)
    const int ocx = floorDiv(ox, LV_SCALE);  // NEW window LV-cell origin
    const int ocz = floorDiv(oz, LV_SCALE);
    auto rebuild = [&](int wcx0, int wcx1, int wcz0, int wcz1) {
        rebuildLightRegion(ctx, ox, oz, wcx0, wcx1, wcz0, wcz1);
    };
    // A scroll changes the in-window light SET at BOTH edges: the LEADING band
    // gains entered lights (widen interior by Rc so they splat into retained
    // cells), and the TRAILING band must be rebuilt to drop departed lights'
    // inward reach (width Rc). The two bands are adjacent in toroidal texture
    // space, so the re-upload is one contiguous band of width STRIP + 2*Rc.
    if (dir.x != 0) {
        if (dir.x > 0) {
            rebuild(ocx + LV_X - STRIP - Rc, ocx + LV_X, ocz, ocz + LV_Z); // leading
            rebuild(ocx, ocx + Rc,                       ocz, ocz + LV_Z); // trailing
            outOff = (ocx - STRIP - Rc) & (LV_X - 1);
        } else {
            rebuild(ocx, ocx + STRIP + Rc,               ocz, ocz + LV_Z); // leading
            rebuild(ocx + LV_X - Rc, ocx + LV_X,         ocz, ocz + LV_Z); // trailing
            outOff = (ocx - Rc) & (LV_X - 1);
        }
        outXAxis = true;  outCnt = STRIP + 2 * Rc;
    } else {
        if (dir.y > 0) {
            rebuild(ocx, ocx + LV_X, ocz + LV_Z - STRIP - Rc, ocz + LV_Z); // leading
            rebuild(ocx, ocx + LV_X, ocz, ocz + Rc);                       // trailing
            outOff = (ocz - STRIP - Rc) & (LV_Z - 1);
        } else {
            rebuild(ocx, ocx + LV_X, ocz, ocz + STRIP + Rc);               // leading
            rebuild(ocx, ocx + LV_X, ocz + LV_Z - Rc, ocz + LV_Z);         // trailing
            outOff = (ocz - Rc) & (LV_Z - 1);
        }
        outXAxis = false; outCnt = STRIP + 2 * Rc;
    }
}

// Update the volume for lights newly appended to caveLights[from..end)
// (streamed crystal clusters arriving as the cave frontier advances). Computes
// the bounding box of the new lights' footprints (±Rc), then EXACTLY rebuilds
// that box via rebuildLightRegion (clear + resplat all in-window lights + cap)
// — so over-CAP, multi-hue clusters stay bit-identical to a full rebuild, not
// the lossy add-onto-capped path. Returns the WORLD LV-cell box for a partial
// re-upload; an empty box (wbx1<=wbx0) means no new lights are in-window.
void splatNewLights(GenContext& ctx, int ox, int oz, size_t from,
                    int& wbx0, int& wbx1, int& wbz0, int& wbz1) {
    const float R  = 80.0f;
    const int   Rc = int(R / LV_SCALE) + 1;
    wbx0 = INT32_MAX; wbx1 = INT32_MIN; wbz0 = INT32_MAX; wbz1 = INT32_MIN;
    for (size_t li = from; li < ctx.caveLights.size(); ++li) {
        const CaveLight& L = ctx.caveLights[li];
        if (L.pos.x < float(ox) || L.pos.x >= float(ox + GRID_XZ) ||
            L.pos.z < float(oz) || L.pos.z >= float(oz + GRID_XZ)) continue;
        int bx = floorDiv(int(L.pos.x), LV_SCALE);
        int bz = floorDiv(int(L.pos.z), LV_SCALE);
        wbx0 = std::min(wbx0, bx - Rc); wbx1 = std::max(wbx1, bx + Rc + 1);
        wbz0 = std::min(wbz0, bz - Rc); wbz1 = std::max(wbz1, bz + Rc + 1);
    }
    if (wbx1 <= wbx0) return;   // no in-window new lights
    rebuildLightRegion(ctx, ox, oz, wbx0, wbx1, wbz0, wbz1);
}

void pruneCaveLights(const World& w, GenContext& ctx) {
    if (w.dirtyChunks.empty() || ctx.caveLights.empty()) return;
    auto nearEditedChunk = [&](const glm::vec3& p) {
        // A cluster's voxels reach ±8 around the light; check the chunk of
        // the light plus the 8-voxel pad against the edited set.
        int x0 = chunkOfVoxel(int(p.x) - 8), x1 = chunkOfVoxel(int(p.x) + 8);
        int z0 = chunkOfVoxel(int(p.z) - 8), z1 = chunkOfVoxel(int(p.z) + 8);
        for (int cz = z0; cz <= z1; ++cz)
        for (int cx = x0; cx <= x1; ++cx) {
            if (w.dirtyChunks.count(chunkKey(cx, cz))) return true;
        }
        return false;
    };
    auto clusterAlive = [&](const glm::vec3& p) {
        int cx = int(p.x), cy = int(p.y), cz = int(p.z);
        for (int dz = -8; dz <= 8; ++dz)
        for (int dy = -8; dy <= 8; ++dy)
        for (int dx = -8; dx <= 8; ++dx)
            if (w.getVoxel(cx + dx, cy + dy, cz + dz) == MAT_CRYSTAL) return true;
        return false;
    };
    std::vector<CaveLight> kept;
    kept.reserve(ctx.caveLights.size());
    for (const CaveLight& L : ctx.caveLights) {
        if (nearEditedChunk(L.pos) && !clusterAlive(L.pos)) continue;   // mined out
        kept.push_back(L);
    }
    if (kept.size() != ctx.caveLights.size()) {
        std::printf("[worldgen] pruned %d mined-out crystal light(s)\n",
                    int(ctx.caveLights.size() - kept.size()));
        ctx.caveLights.swap(kept);
    }
}

// Parallel bulk fill: per-stripe fan-out into thread-local column results,
// then a serial merge in the original (bz, bx, by) order. The merge is the
// only place that touches the shared brickpool/brickCount, so allocBrick
// assigns pool indices identically to the serial code -> bit-identical
// brickmap/brickpool output -> the disk cache and seed stay deterministic.
static void genBulkFill(World& w) {
    StripePool pool;
    std::printf("Bulk fill: %d worker threads\n", pool.threadCount());
    int obx = w.windowOriginChunk.x * CHUNK_BRICKS;
    int obz = w.windowOriginChunk.y * CHUNK_BRICKS;
    std::vector<ColumnResult> cols(BRICKS_XZ);
    for (int bz = obz; bz < obz + BRICKS_XZ; ++bz) {
        pool.runStripe(bz, obx, cols);
        for (int bx = 0; bx < BRICKS_XZ; ++bx) {
            ColumnResult& col = cols[bx];
            size_t mixIdx = 0;
            for (int by = 0; by < BRICKS_Y; ++by) {
                int slot = w.slotOfBrick(obx + bx, by, bz);
                uint32_t v = col.brickmapVals[by];
                if (v == BM_MIXED_PLACEHOLDER) {
                    uint32_t bi = w.allocBrickNoFill();
                    std::memcpy(&w.brickpool[size_t(bi) * BRICK_U32],
                                col.mixedPool[mixIdx].data(),
                                BRICK_U32 * sizeof(uint32_t));
                    w.brickmap[slot] = bi + 1u;
                    ++mixIdx;
                } else {
                    w.brickmap[slot] = v;
                }
            }
        }
    }
}

// Write the run's stats as key=value lines. Diffing two of these files is
// the validation tool for generation refactors: counts should stay
// statistically similar (exact equality is NOT expected once feature RNG
// streams change — that relayouts the world by design).
static void writeGenStats(const World& w, const GenContext& ctx, const char* path) {
    const GenStats& s = ctx.stats;
    std::FILE* f = std::fopen(path, "w");
    if (!f) { std::fprintf(stderr, "[stats] cannot write %s\n", path); return; }
    for (const auto& p : s.phases)
        std::fprintf(f, "time.%s=%.2f\n", p.name, p.sec);
    std::fprintf(f, "bricks=%u\n",          w.brickCount);
    std::fprintf(f, "boulders=%d\n",        s.boulders);
    std::fprintf(f, "bushes=%d\n",          s.bushes);
    std::fprintf(f, "trees=%d\n",           s.trees);
    std::fprintf(f, "flowers=%d\n",         s.flowers);
    std::fprintf(f, "grass.clusters=%d\n",  s.grassClusters);
    std::fprintf(f, "grass.blades=%d\n",    s.grassBlades);
    std::fprintf(f, "caves.systems=%d\n",   s.caveSystems);
    std::fprintf(f, "caves.deepSystems=%d\n", s.deepSystems);
    std::fprintf(f, "caves.wormSteps=%lld\n", s.wormSteps);
    std::fprintf(f, "caves.wormMaxReachXZ=%d\n", s.wormMaxReachXZ);
    std::fprintf(f, "caves.megaProposed=%d\n", s.megaProposed);
    std::fprintf(f, "caves.megaCarved=%d\n",   s.megaCarved);
    std::fprintf(f, "caves.formReqs=%d\n",     s.formReqs);
    std::fprintf(f, "caves.crystalReqs=%d\n",  s.crystalReqs);
    std::fprintf(f, "caves.crystalsBuilt=%d\n",s.crystalsBuilt);
    std::fprintf(f, "caves.decoReqs=%d\n",     s.decoReqs);
    std::fprintf(f, "caves.decoBuilt=%d\n",    s.decoBuilt);
    std::fprintf(f, "caves.ivyReqs=%d\n",      s.ivyReqs);
    std::fprintf(f, "caves.ivyBuilt=%d\n",     s.ivyBuilt);
    // Semantic voxel-content hash (allocation-order independent): the proof
    // that two generation drivers built the SAME world.
    std::fprintf(f, "world.contentHash=%016llx\n",
                 (unsigned long long)w.contentHash());
    std::fclose(f);
    std::printf("[stats] wrote %s\n", path);
}

void (*g_worldgenPhaseHook)(const char* label) = nullptr;

void generateTerrain(World& w, GenContext& ctx) {
    using clk = std::chrono::steady_clock;
    auto phaseStart = clk::now();
    ctx.stats = GenStats{};
    auto phase = [&](const char* label) {
        auto now = clk::now();
        double dt = std::chrono::duration<double>(now - phaseStart).count();
        std::printf("[worldgen %6.2fs] %s\n", dt, label);
        ctx.stats.phases.push_back({ label, dt });
        if (g_worldgenPhaseHook) g_worldgenPhaseHook(label);   // publish step to the loading screen
        phaseStart = now;
    };

    // Full resident-window bounds, anchored at the window origin (unbounded
    // world coords — origin (0,0) for the default fixed world).
    GenBounds bounds{ w.originVoxX(), w.originVoxZ(),
                      w.originVoxX() + GRID_XZ, w.originVoxZ() + GRID_XZ };

    w.initEmpty();
    // Reserve to the streaming ceiling (6M bricks) up front: address space
    // commits lazily (physical pages only on touch), but the capacity means
    // gen never reallocates the pool — which matters because the streamer
    // reads it from worker threads, and a realloc mid-stream would race.
    w.brickpool.reserve(poolReserveBricks() * BRICK_U32);   // window-scaled: never reallocs mid-gen (race-free)
    phase("init");
    genBulkFill(w);
    phase("bulk_fill");
    genCaves(w, ctx, bounds);
    phase("caves");
    // Each flora stage gates its placements against a fresh PRE-STAGE
    // surface snapshot (see SurfaceSnapshot in internal.h) so decisions are
    // reproducible by any chunked replayer — the monolithic driver uses the
    // identical semantics so both build the same world.
    SurfaceSnapshot snap;
    snap.capture(w);
    genBoulders(w, ctx, bounds, snap);
    phase("boulders");
    snap.capture(w);
    genBushes(w, ctx, bounds, snap);
    phase("bushes");
    snap.capture(w);
    genTrees(w, ctx, bounds, snap);
    phase("trees");
    genLakes(w, ctx, bounds);
    genOasis(w, ctx, bounds);
    phase("lakes");
    snap.capture(w);
    genFlowers(w, ctx, bounds, snap);
    phase("flowers");
    snap.capture(w);
    genGrass(w, ctx, bounds, snap);
    phase("grass");
    buildRings(w, ctx);        // far-field LOD rings (coarse horizon)
    phase("far_field");

    std::printf("Terrain: %u bricks (%.1f MB pool)\n",
                w.brickCount, w.brickpool.size() * 4.0 / (1024.0 * 1024.0));
    std::printf("[stats] flora: boulders=%d bushes=%d trees=%d flowers=%d "
                "grass=%d/%d\n",
                ctx.stats.boulders, ctx.stats.bushes, ctx.stats.trees,
                ctx.stats.flowers, ctx.stats.grassClusters, ctx.stats.grassBlades);
    writeGenStats(w, ctx, "worldgen_stats.txt");
}

// LADDER driver (6e-1): generates the world strip-by-strip through a moving
// stage pipeline — the exact order a streaming frontier will use. Per-stage
// compute fronts advance together, separated by enough strips that no stage
// reads state a later front has already modified where it matters:
//
//   master step t:  bulk(t)  carve(t-2)  decor(t-7)
//                   boulders(t-11)  bushes(t-15)  trees(t-19)
//                   lakes(t-23)     flowers(t-27) grass(t-31)
//
// Gap rules: carve trails bulk by 2 (needs only its own strip); decor trails
// carve by 5 > 4 strips = 256 vox >= its 224-vox probe reach; flora stages
// trail each other by 4 so each stage's SNAPSHOT capture front (running 2
// strips ahead of its compute front) reads rows whose previous stage has
// committed but whose own stage has not — gate state identical to the batch
// drivers' whole-world pre-stage capture. Commits are clipped to strips, so
// a captured row can never contain same-stage writes.
//
// KNOWN v1 GAP (deliberate, measured before fixed): decoration probes
// CASCADE (a floor probe positions later writes), and within ~4 strips
// behind the decor front the live world already holds committed neighbour
// decor instead of the carve-only state the batch driver guarantees. This
// is the 6c divergence class (~0.015%); the undo band (doc 7/6e) removes it.
static void generateTerrainLadder(World& w, GenContext& ctx) {
    using clk = std::chrono::steady_clock;
    auto tStart = clk::now();
    ctx.stats = GenStats{};

    const int NS = GRID_XZ / CHUNK_VOX;   // strips; also chunks per strip
    const int OX = w.originVoxX(), OZ = w.originVoxZ();
    ChunkWorkerPool pool;
    w.initEmpty();
    // Reserve to the streaming ceiling (6M bricks) up front: address space
    // commits lazily (physical pages only on touch), but the capacity means
    // gen never reallocates the pool — which matters because the streamer
    // reads it from worker threads, and a realloc mid-stream would race.
    w.brickpool.reserve(poolReserveBricks() * BRICK_U32);   // window-scaled: never reallocs mid-gen (race-free)
    std::printf("[worldgen] LADDER driver: %d strips of %d chunks at origin "
                "(%d,%d), %d workers\n", NS, NS, OX, OZ, pool.threads());
    ensureCaveRegion(ctx, GenBounds{ OX, OZ, OX + GRID_XZ, OZ + GRID_XZ });

    auto chunkBounds = [&](int s, int i) {
        return GenBounds{ OX + i * CHUNK_VOX, OZ + s * CHUNK_VOX,
                          OX + (i + 1) * CHUNK_VOX, OZ + (s + 1) * CHUNK_VOX };
    };
    auto stripBounds = [&](int s) {
        return GenBounds{ OX, OZ + s * CHUNK_VOX,
                          OX + GRID_XZ, OZ + (s + 1) * CHUNK_VOX };
    };

    auto runBulk = [&](int s) {
        std::vector<std::vector<ColumnResult>> cols(NS);
        pool.runBatch(NS, [&](int i) { bulkComputeChunk(chunkBounds(s, i), cols[i]); });
        for (int i = 0; i < NS; ++i) bulkCommitChunk(w, chunkBounds(s, i), cols[i]);
    };
    auto runCarve = [&](int s) {
        std::vector<CaveCarveStage> stages(NS);
        pool.runBatch(NS, [&](int i) {
            stages[i] = CaveCarveStage{ chunkBounds(s, i), {} };
            genCavesCarve(w, ctx, stages[i].b, &stages[i]);
        });
        for (auto& st : stages) commitCaveCarve(w, st);
    };
    // Decoration carries the undo band: computes read pre-decor state for
    // the strips behind the front, commits record what they overwrite, and
    // strips evict once the front passes beyond probe reach (224 vox < 4
    // strips; keep 5 for slack).
    UndoBand undo;
    auto runDecor = [&](int s) {
        std::vector<ChunkStageBuffer> bufs(NS);
        pool.runBatch(NS, [&](int i) {
            bufs[i].b = chunkBounds(s, i);
            genCavesBuild(w, ctx, bufs[i].b, &bufs[i], &undo);
        });
        for (const auto& bf : bufs) bf.commit(w, ctx, &undo);
        undo.evictStrip(OZ / CHUNK_VOX + s - 5);   // world strip index (commit keys)
    };
    auto runFlora = [&](int s, auto&& fn) {
        std::vector<ChunkStageBuffer> bufs(NS);
        pool.runBatch(NS, [&](int i) {
            bufs[i].b = chunkBounds(s, i);
            fn(bufs[i].b, &bufs[i]);
        });
        for (const auto& bf : bufs) bf.commit(w, ctx);
    };

    SurfaceSnapshot snapBoulder, snapBush, snapTree, snapFlower, snapGrass;

    constexpr int OFF_BULK = 0,  OFF_CARVE = 2,  OFF_DECOR = 7;
    constexpr int OFF_BOULDER = 11, OFF_BUSH = 15, OFF_TREE = 19;
    constexpr int OFF_LAKE = 23,    OFF_FLOWER = 27, OFF_GRASS = 31;
    auto in = [&](int s) { return s >= 0 && s < NS; };

    for (int t = 0; t < NS + OFF_GRASS; ++t) {
        if (in(t - OFF_BULK))  runBulk(t - OFF_BULK);
        if (in(t - OFF_CARVE)) runCarve(t - OFF_CARVE);
        if (in(t - OFF_DECOR)) runDecor(t - OFF_DECOR);

        if (in(t - OFF_BOULDER + 2))
            snapBoulder.capture(w, stripBounds(t - OFF_BOULDER + 2));
        if (in(t - OFF_BOULDER))
            runFlora(t - OFF_BOULDER, [&](const GenBounds& b, ChunkStageBuffer* sb) {
                genBoulders(w, ctx, b, snapBoulder, sb); });

        if (in(t - OFF_BUSH + 2))
            snapBush.capture(w, stripBounds(t - OFF_BUSH + 2));
        if (in(t - OFF_BUSH))
            runFlora(t - OFF_BUSH, [&](const GenBounds& b, ChunkStageBuffer* sb) {
                genBushes(w, ctx, b, snapBush, sb); });

        if (in(t - OFF_TREE + 2))
            snapTree.capture(w, stripBounds(t - OFF_TREE + 2));
        if (in(t - OFF_TREE))
            runFlora(t - OFF_TREE, [&](const GenBounds& b, ChunkStageBuffer* sb) {
                genTrees(w, ctx, b, snapTree, sb); });

        if (in(t - OFF_LAKE))
            for (int i = 0; i < NS; ++i) genLakes(w, ctx, chunkBounds(t - OFF_LAKE, i));
            for (int i = 0; i < NS; ++i) genOasis(w, ctx, chunkBounds(t - OFF_LAKE, i));

        if (in(t - OFF_FLOWER + 2))
            snapFlower.capture(w, stripBounds(t - OFF_FLOWER + 2));
        if (in(t - OFF_FLOWER))
            runFlora(t - OFF_FLOWER, [&](const GenBounds& b, ChunkStageBuffer* sb) {
                genFlowers(w, ctx, b, snapFlower, sb); });

        if (in(t - OFF_GRASS + 2))
            snapGrass.capture(w, stripBounds(t - OFF_GRASS + 2));
        if (in(t - OFF_GRASS))
            runFlora(t - OFF_GRASS, [&](const GenBounds& b, ChunkStageBuffer* sb) {
                genGrass(w, ctx, b, snapGrass, sb); });
    }

    double total = std::chrono::duration<double>(clk::now() - tStart).count();
    ctx.stats.phases.push_back({ "ladder_total", total });
    std::printf("[worldgen] ladder complete in %.2fs\n", total);
    std::printf("Terrain: %u bricks (%.1f MB pool)\n",
                w.brickCount, w.brickpool.size() * 4.0 / (1024.0 * 1024.0));
    std::printf("[stats] flora: boulders=%d bushes=%d trees=%d flowers=%d "
                "grass=%d/%d\n",
                ctx.stats.boulders, ctx.stats.bushes, ctx.stats.trees,
                ctx.stats.flowers, ctx.stats.grassClusters, ctx.stats.grassBlades);
    writeGenStats(w, ctx, "worldgen_stats.txt");
}

// Chunk-by-chunk validation driver. Builds the same world as generateTerrain
// through the bounded phase API: STAGE-OUTER, CHUNK-INNER — every chunk
// completes stage N before any chunk runs stage N+1, so cross-chunk reads of
// previous stages always see final state (the chunk-status ladder that true
// streaming will also use). Serial within each stage for deterministic pool
// allocation; this driver optimizes for being OBVIOUSLY correct, not fast.
void generateTerrainChunked(World& w, GenContext& ctx) {
    using clk = std::chrono::steady_clock;
    auto phaseStart = clk::now();
    ctx.stats = GenStats{};
    auto phase = [&](const char* label) {
        auto now = clk::now();
        double dt = std::chrono::duration<double>(now - phaseStart).count();
        std::printf("[worldgen %6.2fs] %s\n", dt, label);
        ctx.stats.phases.push_back({ label, dt });
        if (g_worldgenPhaseHook) g_worldgenPhaseHook(label);   // publish step to the loading screen
        phaseStart = now;
    };

    // VOXEL_CHUNKED_GEN=2 / --chunked-gen=2 iterates chunks in REVERSE — the
    // order-independence proof: both orders must produce the IDENTICAL
    // contentHash now that every stage is order-free (flora + decoration:
    // buffered compute-then-commit; carve: idempotent air, chunk-owned
    // bricks; bulk: pure). =3 selects the LADDER driver (streaming order).
    bool reverseOrder = false;
    if (const char* v = std::getenv("VOXEL_CHUNKED_GEN")) {
        if (v[0] == '3') { generateTerrainLadder(w, ctx); return; }
        reverseOrder = (v[0] == '2');
    }

    const int NC = GRID_XZ / CHUNK_VOX;          // chunks per axis
    const int OX = w.originVoxX(), OZ = w.originVoxZ();
    std::vector<GenBounds> order;
    order.reserve(size_t(NC) * NC);
    for (int cz = OZ; cz < OZ + GRID_XZ; cz += CHUNK_VOX)
    for (int cx = OX; cx < OX + GRID_XZ; cx += CHUNK_VOX)
        order.push_back({ cx, cz, cx + CHUNK_VOX, cz + CHUNK_VOX });
    if (reverseOrder) std::reverse(order.begin(), order.end());

    ChunkWorkerPool pool;

    // Stripe runner: parallel-compute one row's worth of chunks, then commit
    // them serially in iteration order — bounds peak buffer memory while
    // keeping pool allocation deterministic. Safe for bulk/carve/flora
    // because buffered reads make chunk content independent of what other
    // chunks have committed; decoration must NOT stripe (see below).
    auto forStripes = [&](const char* label, auto&& compute, auto&& commit) {
        const size_t S = size_t(NC);
        double tCompute = 0.0, tCommit = 0.0;
        for (size_t s0 = 0; s0 < order.size(); s0 += S) {
            int cnt = int(std::min(S, order.size() - s0));
            auto t0 = std::chrono::steady_clock::now();
            pool.runBatch(cnt, [&](int i) { compute(i, order[s0 + i]); });
            auto t1 = std::chrono::steady_clock::now();
            for (int i = 0; i < cnt; ++i) commit(i, order[s0 + i]);
            auto t2 = std::chrono::steady_clock::now();
            tCompute += std::chrono::duration<double>(t1 - t0).count();
            tCommit  += std::chrono::duration<double>(t2 - t1).count();
        }
        std::printf("[chunked] %s: compute %.2fs commit %.2fs\n",
                    label, tCompute, tCommit);
    };

    w.initEmpty();
    // Reserve to the streaming ceiling (6M bricks) up front: address space
    // commits lazily (physical pages only on touch), but the capacity means
    // gen never reallocates the pool — which matters because the streamer
    // reads it from worker threads, and a realloc mid-stream would race.
    w.brickpool.reserve(poolReserveBricks() * BRICK_U32);   // window-scaled: never reallocs mid-gen (race-free)
    phase("init");
    std::printf("[worldgen] CHUNKED driver: %dx%d chunks of %d voxels "
                "(%s order, %d workers)\n",
                NC, NC, CHUNK_VOX, reverseOrder ? "REVERSE" : "forward",
                pool.threads());

    {   // Bulk fill: pure per-column compute, staged commits.
        std::vector<std::vector<ColumnResult>> cols(NC);
        forStripes("bulk",
                   [&](int i, const GenBounds& b) { bulkComputeChunk(b, cols[i]); },
                   [&](int i, const GenBounds& b) { bulkCommitChunk(w, b, cols[i]); });
    }
    phase("bulk_fill");

    ensureCaveRegion(ctx, GenBounds{ OX, OZ, OX + GRID_XZ, OZ + GRID_XZ });
    {   // Carve: existing-brick writes are chunk-disjoint (direct, parallel-
        // safe); SOLID->mixed births stage per chunk and commit serially.
        std::vector<CaveCarveStage> stages(NC);
        forStripes("carve",
                   [&](int i, const GenBounds& b) {
                       stages[i] = CaveCarveStage{ b, {} };
                       genCavesCarve(w, ctx, b, &stages[i]);
                   },
                   [&](int i, const GenBounds&) { commitCaveCarve(w, stages[i]); });
    }
    {   // Decoration: ALL chunks compute against the carve-only world, THEN
        // all commit (the order-free invariant) — batch, never striped.
        std::vector<ChunkStageBuffer> bufs(order.size());
        auto t0 = std::chrono::steady_clock::now();
        pool.runBatch(int(order.size()), [&](int i) {
            bufs[i].b = order[i];
            genCavesBuild(w, ctx, order[i], &bufs[i]);
        });
        auto t1 = std::chrono::steady_clock::now();
        size_t nWrites = 0;
        for (const ChunkStageBuffer& dbuf : bufs) {
            nWrites += dbuf.pages.size();
            dbuf.commit(w, ctx);
        }
        auto t2 = std::chrono::steady_clock::now();
        std::printf("[chunked] decor: compute %.2fs commit %.2fs (%zu pages)\n",
                    std::chrono::duration<double>(t1 - t0).count(),
                    std::chrono::duration<double>(t2 - t1).count(), nWrites);
    }
    phase("caves");

    // One pre-stage snapshot per flora stage (NOT per chunk): captured after
    // the previous stage completed for ALL chunks, exactly matching the
    // monolithic driver's snapshot.
    SurfaceSnapshot snap;
    auto floraStage = [&](const char* label, auto&& fn) {
        snap.capture(w);
        std::vector<ChunkStageBuffer> bufs(NC);
        forStripes(label,
                   [&](int i, const GenBounds& b) {
                       bufs[i] = ChunkStageBuffer{};
                       bufs[i].b = b;
                       fn(b, &bufs[i]);
                   },
                   [&](int i, const GenBounds&) { bufs[i].commit(w, ctx); });
        phase(label);
    };
    floraStage("boulders", [&](const GenBounds& b, ChunkStageBuffer* s) {
        genBoulders(w, ctx, b, snap, s); });
    floraStage("bushes", [&](const GenBounds& b, ChunkStageBuffer* s) {
        genBushes(w, ctx, b, snap, s); });
    floraStage("trees", [&](const GenBounds& b, ChunkStageBuffer* s) {
        genTrees(w, ctx, b, snap, s); });
    for (const GenBounds& b : order) genLakes(w, ctx, b);
    for (const GenBounds& b : order) genOasis(w, ctx, b);
    phase("lakes");
    floraStage("flowers", [&](const GenBounds& b, ChunkStageBuffer* s) {
        genFlowers(w, ctx, b, snap, s); });
    floraStage("grass", [&](const GenBounds& b, ChunkStageBuffer* s) {
        genGrass(w, ctx, b, snap, s); });
    buildRings(w, ctx);        // far-field LOD rings (coarse horizon)
    phase("far_field");

    std::printf("Terrain: %u bricks (%.1f MB pool)\n",
                w.brickCount, w.brickpool.size() * 4.0 / (1024.0 * 1024.0));
    std::printf("[stats] flora: boulders=%d bushes=%d trees=%d flowers=%d "
                "grass=%d/%d\n",
                ctx.stats.boulders, ctx.stats.bushes, ctx.stats.trees,
                ctx.stats.flowers, ctx.stats.grassClusters, ctx.stats.grassBlades);
    writeGenStats(w, ctx, "worldgen_stats.txt");
}

int terrainHeightAt(int x, int z) { return heightAt(x, z); }

float forestDensityAt(int x, int z) { return biomeAt(x, z).treeChance; }

uint32_t creatureCellSeed(uint64_t worldSeed, int cellX, int cellZ) {
    return featureSeed(worldSeed, SEED_CREATURE, cellX, cellZ);
}

// Dev-console nav: nearest oasis / mesa SITE to a world XZ. oasisSite/mesaAt are
// pure (x,z) lattice functions, so we just scan the cells around the player and
// keep the closest hosting+passing one. Search ~25 cells out (~60 km) — far more
// than the resident window; cheap as a one-shot command.
bool nearestOasis(int px, int pz, int& ox, int& oy, int& oz) {
    int pcx = int(std::floor(float(px) * OASIS_INV));
    int pcz = int(std::floor(float(pz) * OASIS_INV));
    double best = 1e30; bool found = false; float bcx = 0, bcz = 0; int bwy = 0;
    for (int dz = -25; dz <= 25; ++dz)
    for (int dx = -25; dx <= 25; ++dx) {
        OasisSite s = oasisSite(pcx + dx, pcz + dz);
        if (!s.pass) continue;
        double d = double(s.ctrX - px) * (s.ctrX - px) + double(s.ctrZ - pz) * (s.ctrZ - pz);
        if (d < best) { best = d; found = true; bcx = s.ctrX; bcz = s.ctrZ; bwy = s.waterY; }
    }
    if (!found) return false;
    ox = int(bcx); oy = bwy + 25; oz = int(bcz);   // a vantage just above the pool
    return true;
}
bool nearestMesa(int px, int pz, int& mx, int& my, int& mz) {
    const float INV = 1.0f / 2600.0f;              // mirrors mesaAt's lattice
    int pcx = int(std::floor(float(px) * INV));
    int pcz = int(std::floor(float(pz) * INV));
    double best = 1e30; bool found = false; float bcx = 0, bcz = 0, bcapY = 0;
    for (int dz = -25; dz <= 25; ++dz)
    for (int dx = -25; dx <= 25; ++dx) {
        int cx = pcx + dx, cz = pcz + dz;
        if (hash2(cx * 131 + 7, cz * 131 + 3) <= 0.86f) continue;          // host hash
        float ctrX = (float(cx) + 0.22f + 0.56f * hash2(cx, cz))          / INV;
        float ctrZ = (float(cz) + 0.22f + 0.56f * hash2(cx + 99, cz - 99)) / INV;
        if (biomeDesertW(int(ctrX), int(ctrZ)) < 0.5f) continue;           // firm desert only
        double d = double(ctrX - px) * (ctrX - px) + double(ctrZ - pz) * (ctrZ - pz);
        if (d < best) {
            MesaHit mh = mesaAt(int(ctrX), int(ctrZ));                     // capY at the center
            best = d; found = true; bcx = ctrX; bcz = ctrZ;
            bcapY = mh.any ? mh.capY : (baseHAt(int(ctrX), int(ctrZ)) + 185.0f);
        }
    }
    if (!found) return false;
    mx = int(bcx); my = int(bcapY) + 3; mz = int(bcz);   // stand on the caprock
    return true;
}

// =============================================================================
// ChunkStreamer (6e-3b): the chunk-status ladder as a runtime scheduler.
// =============================================================================
struct ChunkStreamerImpl {
    World& w;
    GenContext& ctx;
    ChunkWorkerPool pool;
    SurfaceSnapshot snap;                          // live-bound (no captures)
    std::unordered_map<int64_t, uint8_t> stage;    // absent == CS_NONE
    std::vector<int64_t> uploads;                  // chunks ready for GPU push
    std::function<bool(int64_t)> skip;             // evicted-cache guard

    // The in-flight ASYNC batch: computes run on the pool while frames keep
    // rendering; commits drain a few chunks per frame once compute is done.
    struct Flight {
        uint8_t tgt = CS_NONE;
        std::vector<glm::ivec2> chunks;
        std::vector<std::vector<ColumnResult>> cols;   // CS_BULK
        std::vector<CaveCarveStage> carve;             // CS_CARVE
        std::vector<ChunkStageBuffer> bufs;            // decor + flora
        bool computing = false;
        int committed = 0;
    } flight;

    GenBounds bndOf(glm::ivec2 c) const {
        return GenBounds{ c.x * CHUNK_VOX, c.y * CHUNK_VOX,
                          (c.x + 1) * CHUNK_VOX, (c.y + 1) * CHUNK_VOX };
    }
    bool chunkInWindow(int cx, int cz) const {
        glm::ivec2 o = w.windowOriginChunk;
        const int NC = GRID_XZ / CHUNK_VOX;
        return cx >= o.x && cx < o.x + NC && cz >= o.y && cz < o.y + NC;
    }

    ChunkStreamerImpl(World& w_, GenContext& ctx_) : w(w_), ctx(ctx_) {}
    uint8_t get(int cx, int cz) const {
        auto it = stage.find(chunkKey(cx, cz));
        return it == stage.end() ? uint8_t(CS_NONE) : it->second;
    }
    void set(int cx, int cz, uint8_t s) {
        int64_t k = chunkKey(cx, cz);
        if (s == CS_NONE) stage.erase(k); else stage[k] = s;
    }
    // Dependencies for advancing TO each stage: neighbours within `radius`
    // (chunks, Chebyshev) must be at >= `minStage`. The radius covers the
    // stage's margin replay; minStage is CARVE for every flora stage because
    // a flora gate reads the frozen SURFACE SNAPSHOT, which a chunk captures
    // at its CARVE-completion — so a neighbour must be carved (captured)
    // before this chunk gates a feature whose anchor lives there. (Reading
    // the live world instead used to let the first chunk that commits a tree
    // overwrite its own grass tile, flipping the gate for the neighbours
    // that should draw the rest → border-cut trees/rocks. The snapshot makes
    // the decision identical for every replayer.) Decoration also needs
    // CARVE for its carved-geometry probes.
    static void reqFor(uint8_t target, int& radius, uint8_t& minStage) {
        switch (target) {
            case CS_DECOR:   radius = 2; minStage = CS_CARVE; break;
            case CS_BOULDER: radius = 1; minStage = CS_CARVE; break;
            case CS_BUSH:    radius = 1; minStage = CS_CARVE; break;
            case CS_TREE:    radius = 2; minStage = CS_CARVE; break;
            case CS_FLOWER:  radius = 1; minStage = CS_CARVE; break;
            case CS_GRASS:   radius = 1; minStage = CS_CARVE; break;
            default:         radius = 0; minStage = CS_NONE;  break;
        }
    }
    bool depsMet(int cx, int cz, uint8_t target) const {
        int r; uint8_t need;
        reqFor(target, r, need);
        for (int dz = -r; dz <= r; ++dz)
        for (int dx = -r; dx <= r; ++dx) {
            if (dx == 0 && dz == 0) continue;
            if (get(cx + dx, cz + dz) < need) return false;
        }
        return true;
    }
};

ChunkStreamer::ChunkStreamer(World& w, GenContext& ctx)
    : impl(new ChunkStreamerImpl(w, ctx)) {}
ChunkStreamer::~ChunkStreamer() { delete impl; }

void ChunkStreamer::seedAllFinal() {
    const int NC = GRID_XZ / CHUNK_VOX;
    glm::ivec2 o = impl->w.windowOriginChunk;
    for (int cz = o.y; cz < o.y + NC; ++cz)
    for (int cx = o.x; cx < o.x + NC; ++cx)
        impl->stage[chunkKey(cx, cz)] = CS_FINAL;
}

void ChunkStreamer::reset() {
    // Finish (and discard) any in-flight compute so no worker is reading the
    // world during the imminent wipe, then forget all chunk/upload state.
    if (impl->flight.computing) impl->pool.wait();
    impl->flight = ChunkStreamerImpl::Flight{};
    impl->stage.clear();
    impl->uploads.clear();
}
void ChunkStreamer::setSkip(std::function<bool(int64_t)> skip) {
    impl->skip = std::move(skip);
}
uint8_t ChunkStreamer::stageOf(int ccx, int ccz) const { return impl->get(ccx, ccz); }
void ChunkStreamer::markFinal(int ccx, int ccz) { impl->set(ccx, ccz, CS_FINAL); }
void ChunkStreamer::clear(int ccx, int ccz) { impl->set(ccx, ccz, CS_NONE); }
int ChunkStreamer::pendingInWindow() const {
    const int NC = GRID_XZ / CHUNK_VOX;
    glm::ivec2 o = impl->w.windowOriginChunk;
    int nPend = 0;
    for (int cz = o.y; cz < o.y + NC; ++cz)
    for (int cx = o.x; cx < o.x + NC; ++cx)
        if (impl->get(cx, cz) < CS_FINAL) ++nPend;
    return nPend;
}
int ChunkStreamer::drainUploads(int64_t* out, int maxCount) {
    ChunkStreamerImpl& S = *impl;
    int nOut = 0;
    while (nOut < maxCount && !S.uploads.empty()) {
        out[nOut++] = S.uploads.front();
        S.uploads.erase(S.uploads.begin());
    }
    return nOut;
}

int ChunkStreamer::step(glm::ivec2 pc, EditDelta& d) {
    (void)d;
    ChunkStreamerImpl& S = *impl;
    World& w = S.w;
    const int NC = GRID_XZ / CHUNK_VOX;
    glm::ivec2 o = w.windowOriginChunk;

    // -- Progress the in-flight async batch first ---------------------------
    // Compute runs on the pool while frames render; once idle, commits drain
    // a few chunks per frame so the main thread never pays a whole batch.
    auto& F = S.flight;
    if (F.tgt != CS_NONE) {
        if (F.computing) {
            if (!S.pool.idle()) return 0;     // still computing in background
            F.computing = false;
        }
        // Commits allocate/write thousands of bricks per chunk in cavernous
        // regions, and per-chunk cost varies wildly (a flat-plains chunk vs a
        // honeycombed one). A FIXED per-frame count therefore either starves
        // the frontier (too few) or hitches (a burst of heavy chunks). Spend
        // a TIME budget instead: commit chunks until ~3 ms is used, so the
        // per-frame cost stays flat regardless of chunk density while the
        // frontier still advances as fast as the hardware allows.
        constexpr double COMMIT_MS = 3.0;
        using cclk = std::chrono::steady_clock;
        auto cT0 = cclk::now();
        int n = int(F.chunks.size());
        int done = 0;
        // Free chunk i's staging the instant it is consumed. A DECOR/flora
        // batch stages thousands of per-brick StagePages per chunk; destructing
        // all 32 chunks' maps at once in `F = Flight{}` was a ~38 ms free-storm
        // (one spike per batch). Releasing per chunk spreads that teardown
        // across the time-budgeted commit frames instead.
        auto freeStaging = [&](int i) {
            switch (F.tgt) {
            case CS_BULK:  F.cols[i]  = {};                break;
            case CS_CARVE: F.carve[i] = CaveCarveStage{};  break;
            default:       F.bufs[i]  = ChunkStageBuffer{}; break;
            }
        };
        for (; F.committed < n; ++F.committed) {
            // Always commit at least one (so progress is guaranteed); then
            // stop once the time budget is spent.
            if (done > 0 && std::chrono::duration<double, std::milli>(
                                cclk::now() - cT0).count() >= COMMIT_MS) break;
            int i = F.committed;
            int cx = F.chunks[i].x, cz = F.chunks[i].y;
            // Scrolled out or cleared mid-flight: its slots may belong to an
            // entering chunk now — discard the stale result.
            if (!S.chunkInWindow(cx, cz) ||
                S.get(cx, cz) != uint8_t(F.tgt - 1)) { freeStaging(i); continue; }
            switch (F.tgt) {
            case CS_BULK:  bulkCommitChunk(w, S.bndOf(F.chunks[i]), F.cols[i]); break;
            case CS_CARVE: commitCaveCarve(w, F.carve[i]); break;
            default:       F.bufs[i].commit(w, S.ctx); break;
            }
            // Freeze this chunk's post-carve surface band into the snapshot
            // the flora stages gate against. Captured ONCE here (carve done,
            // before any flora) so every chunk that margin-replays a feature
            // reads the identical surface — no border cuts. The flora deps
            // (minStage CARVE) guarantee neighbours are captured first.
            if (F.tgt == CS_CARVE)
                S.snap.capture(w, S.bndOf(F.chunks[i]));
            S.set(cx, cz, F.tgt);
            // GPU milestones: ground at BULK — the window-edge rim
            // (including the permanently sub-final edge bands) must read as
            // terrain, not a void gash — and the dressed chunk at FINAL.
            // With the 2-deep pipeline, the bare-ground band is only the
            // outermost ~3 chunks of the window.
            if (F.tgt == CS_BULK || F.tgt == CS_FINAL)
                S.uploads.push_back(chunkKey(cx, cz));
            ++done;
            freeStaging(i);   // release this chunk's staging now (see above)
        }
        if (F.committed < n) return done;
        F = ChunkStreamerImpl::Flight{};       // batch complete (staging already freed)
        // fall through: select + dispatch the next batch this same frame
    }

    // Eligible chunks; the batch = the target stage of the NEAREST eligible
    // chunk (finish terrain around the player before pushing the frontier).
    struct Cand { int cx, cz, dist; };
    std::vector<Cand> cands;
    int bestDist = INT32_MAX;
    uint8_t bestTgt = 0;
    for (int cz = o.y; cz < o.y + NC; ++cz)
    for (int cx = o.x; cx < o.x + NC; ++cx) {
        uint8_t s = S.get(cx, cz);
        if (s >= CS_FINAL) continue;
        if (S.skip && S.skip(chunkKey(cx, cz))) continue;   // evicted-cached
        uint8_t tgt = uint8_t(s + 1);
        if (!S.depsMet(cx, cz, tgt)) continue;
        int dist = std::max(std::abs(cx - pc.x), std::abs(cz - pc.y));
        cands.push_back({ cx, cz, dist });
        if (dist < bestDist) {
            bestDist = dist;
            bestTgt  = tgt;
        }
    }
    if (cands.empty()) return 0;
    // Keep only candidates advancing to bestTgt, nearest first, cap batch.
    std::vector<Cand> batch;
    for (const Cand& c : cands)
        if (uint8_t(S.get(c.cx, c.cz) + 1) == bestTgt) batch.push_back(c);
    std::sort(batch.begin(), batch.end(),
              [](const Cand& a, const Cand& b) { return a.dist < b.dist; });
    constexpr int BATCH = 32;   // a full entering strip's worth per compute
    if (int(batch.size()) > BATCH) batch.resize(BATCH);

    int n = int(batch.size());
    GenContext& ctx = S.ctx;

    // Lakes are a no-op while LAKES_ON is false; the oasis pool fills here too.
    if (bestTgt == CS_LAKE) {
        for (int i = 0; i < n; ++i) {
            genLakes(w, ctx, S.bndOf({ batch[i].cx, batch[i].cz }));
            genOasis(w, ctx, S.bndOf({ batch[i].cx, batch[i].cz }));
            S.set(batch[i].cx, batch[i].cz, CS_LAKE);
        }
        return n;
    }

    GenBounds u = S.bndOf({ batch[0].cx, batch[0].cz });
    for (int i = 1; i < n; ++i) {
        GenBounds bi = S.bndOf({ batch[i].cx, batch[i].cz });
        u.x0 = std::min(u.x0, bi.x0); u.x1 = std::max(u.x1, bi.x1);
        u.z0 = std::min(u.z0, bi.z0); u.z1 = std::max(u.z1, bi.z1);
    }

    // Carve/decor need the cave plan for the region. Walk it INCREMENTALLY
    // (budgeted) so a crossing into fresh cells never blocks the frame on a
    // worm-walk burst; until it is fully ready, hold this batch back (return
    // without dispatching — next frame retries). The frontier just pauses a
    // beat (the curtain eases in slightly) instead of hitching.
    if (bestTgt == CS_CARVE || bestTgt == CS_DECOR) {
        constexpr int CELL_BUDGET = 12;   // ~3 ms of serial worm-walking
        if (!ensureCaveRegion(ctx, u, CELL_BUDGET))
            return 0;
    }

    // Stage the batch and DISPATCH its compute to the pool — no waiting.
    // Computes only read the live world and write their own staging
    // (buffers / carve births / column results); commits next frames.
    F.tgt = bestTgt;
    F.chunks.resize(n);
    for (int i = 0; i < n; ++i) F.chunks[i] = { batch[i].cx, batch[i].cz };

    ChunkStreamerImpl* P = impl;
    switch (bestTgt) {
    case CS_BULK:
        F.cols.assign(n, {});
        S.pool.dispatch(n, [P](int i) {
            bulkComputeChunk(P->bndOf(P->flight.chunks[i]), P->flight.cols[i]);
        });
        break;
    case CS_CARVE:
        F.carve.assign(n, {});
        S.pool.dispatch(n, [P](int i) {
            P->flight.carve[i].b = P->bndOf(P->flight.chunks[i]);
            genCavesCarve(P->w, P->ctx, P->flight.carve[i].b, &P->flight.carve[i]);
            P->flight.carve[i].presort();   // commit becomes a linear sweep
        });
        break;
    case CS_DECOR:
        F.bufs.assign(n, {});
        S.pool.dispatch(n, [P](int i) {
            P->flight.bufs[i].b = P->bndOf(P->flight.chunks[i]);
            genCavesBuild(P->w, P->ctx, P->flight.bufs[i].b, &P->flight.bufs[i]);
        });
        break;
    default: {   // the five flora stages, live-gated (seam-class accepted)
        F.bufs.assign(n, {});
        uint8_t tgt = bestTgt;
        S.pool.dispatch(n, [P, tgt](int i) {
            ChunkStageBuffer& buf = P->flight.bufs[i];
            buf.b = P->bndOf(P->flight.chunks[i]);
            switch (tgt) {
            case CS_BOULDER: genBoulders(P->w, P->ctx, buf.b, P->snap, &buf); break;
            case CS_BUSH:    genBushes  (P->w, P->ctx, buf.b, P->snap, &buf); break;
            case CS_TREE:    genTrees   (P->w, P->ctx, buf.b, P->snap, &buf); break;
            case CS_FLOWER:  genFlowers (P->w, P->ctx, buf.b, P->snap, &buf); break;
            case CS_GRASS:   genGrass   (P->w, P->ctx, buf.b, P->snap, &buf); break;
            }
        });
        break;
    }
    }
    F.computing = true;
    return 0;
}
