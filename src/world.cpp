#include "world.h"
#include "materials.h"
#include "profiling.h"

#include <glad/gl.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

namespace {

inline uint8_t readVoxFromBrick(const std::vector<uint32_t>& pool, uint32_t bi,
                                int lx, int ly, int lz) {
    int linear = voxInBrick(lx, ly, lz);
    int slot = linear >> 2;
    int shift = (linear & 3) * 8;
    return uint8_t((pool[bi * BRICK_U32 + slot] >> shift) & 0xFFu);
}
inline void writeVoxInBrick(std::vector<uint32_t>& pool, uint32_t bi,
                            int lx, int ly, int lz, uint8_t v) {
    int linear = voxInBrick(lx, ly, lz);
    int slot = linear >> 2;
    int shift = (linear & 3) * 8;
    uint32_t mask = 0xFFu << shift;
    uint32_t& w = pool[bi * BRICK_U32 + slot];
    w = (w & ~mask) | (uint32_t(v) << shift);
}

} // namespace

void World::initEmpty() {
    brickmap.assign(size_t(BRICKS_XZ) * BRICKS_XZ * BRICKS_Y, BM_EMPTY);
    brickpool.clear();
    brickCount = 0;
    brickHighWater = 0;
    freeBricks.clear();
    for (auto& r : rings) {    // rings rebuild wholesale after regen (buildRings)
        r.ready = false;
        r.brickCount = 0;
    }
    gpuChunkValid.assign(size_t(CHUNKS_XZ) * CHUNKS_XZ, 0u);
}

void World::buildCoarse() {
    coarse.assign(size_t(CBRICKS_XZ) * CBRICKS_XZ * CBRICKS_Y, 0u);
    for (int bz = 0; bz < BRICKS_XZ; ++bz)
    for (int by = 0; by < BRICKS_Y;  ++by)
    for (int bx = 0; bx < BRICKS_XZ; ++bx) {
        if (brickmap[bmIdx(bx, by, bz)] != BM_EMPTY)
            coarse[cbmIdx(bx >> COARSE_BITS, by >> COARSE_BITS, bz >> COARSE_BITS)] = 1u;
    }
}

// Re-derive the coarse-occupancy cells covered by chunk (ccx,ccz) from the CPU
// brickmap and re-upload the mip. buildCoarse() only runs once at init, but the
// streaming window swaps chunks in/out at runtime — without this resync a newly
// streamed-in feature in a previously-empty coarse cell would be skipped by
// anyHit (shadow/AO), leaking light through it. Mirrors buildCoarse's exact
// per-cell test (brickmap[bmIdx] != BM_EMPTY) so the two can never disagree.
void World::refreshCoarseForChunk(int ccx, int ccz) {
    if (coarse.empty() || coarseSSBO == 0) return;
    const int lcx = ccx & (CHUNKS_XZ - 1);          // local (wrapped) chunk column
    const int lcz = ccz & (CHUNKS_XZ - 1);
    const int CPC = CHUNK_BRICKS >> COARSE_BITS;     // coarse cells per chunk axis (2)
    const int cx0 = lcx * CPC, cz0 = lcz * CPC;
    bool changed = false;
    for (int dz = 0; dz < CPC; ++dz)
    for (int cy = 0; cy < CBRICKS_Y; ++cy)
    for (int dx = 0; dx < CPC; ++dx) {
        const int cx = cx0 + dx, cz = cz0 + dz;
        uint32_t occ = 0u;
        for (int bz = 0; bz < COARSE && !occ; ++bz)
        for (int by = 0; by < COARSE && !occ; ++by)
        for (int bx = 0; bx < COARSE && !occ; ++bx)
            if (brickmap[bmIdx((cx << COARSE_BITS) + bx,
                               (cy << COARSE_BITS) + by,
                               (cz << COARSE_BITS) + bz)] != BM_EMPTY) occ = 1u;
        uint32_t& cell = coarse[cbmIdx(cx, cy, cz)];
        if (cell != occ) { cell = occ; changed = true; }
    }
    if (changed) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, coarseSSBO);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                        coarse.size() * sizeof(uint32_t), coarse.data());
    }
}

uint32_t World::allocBrick(uint8_t fill) {
    uint32_t idx;
    if (!freeBricks.empty()) {
        idx = freeBricks.back();
        freeBricks.pop_back();
    } else {
        idx = brickHighWater++;
        if (size_t(idx + 1) * BRICK_U32 > brickpool.size()) {
            // Growing past CAPACITY reallocates — which would race the async
            // streaming computes reading the pool. The startup reserve makes
            // this unreachable in normal play; scream if it ever isn't.
            if (size_t(idx + 1) * BRICK_U32 > brickpool.capacity())
                std::fprintf(stderr, "[world] brickpool grew past its reserve "
                                     "(%u bricks) — async reads may race!\n",
                             idx + 1);
            brickpool.resize(size_t(idx + 1) * BRICK_U32);
        }
    }
    ++brickCount;
    uint32_t pat = uint32_t(fill)
                 | (uint32_t(fill) << 8)
                 | (uint32_t(fill) << 16)
                 | (uint32_t(fill) << 24);
    for (int i = 0; i < BRICK_U32; ++i) brickpool[size_t(idx) * BRICK_U32 + i] = pat;
    return idx;
}

uint32_t World::allocBrickNoFill() {
    uint32_t idx;
    if (!freeBricks.empty()) {
        idx = freeBricks.back();
        freeBricks.pop_back();
    } else {
        idx = brickHighWater++;
        if (size_t(idx + 1) * BRICK_U32 > brickpool.size()) {
            if (size_t(idx + 1) * BRICK_U32 > brickpool.capacity())
                std::fprintf(stderr, "[world] brickpool grew past its reserve "
                                     "(%u bricks) — async reads may race!\n",
                             idx + 1);
            brickpool.resize(size_t(idx + 1) * BRICK_U32);
        }
    }
    ++brickCount;
    return idx;
}

void World::freeBrick(uint32_t bi) {
    freeBricks.push_back(bi);
    --brickCount;
}

uint64_t World::contentHash() const {
    // FNV-1a per brick-row (fixed bz), rows hashed in parallel, then the row
    // hashes are combined sequentially in z order — fully deterministic and
    // ~20x faster than a single-threaded pass over the ~2 GB pool.
    auto fnv = [](uint64_t h, const uint8_t* p, size_t n) {
        for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 0x100000001B3ull; }
        return h;
    };
    std::vector<uint64_t> rowHash(BRICKS_XZ, 0);
    std::atomic<int> nextRow{0};
    int nThreads = int(std::max(1u, std::thread::hardware_concurrency()));
    std::vector<std::thread> workers;
    for (int t = 0; t < nThreads; ++t) {
        workers.emplace_back([&]() {
            int bz;
            while ((bz = nextRow.fetch_add(1)) < BRICKS_XZ) {
                uint64_t h = 0xCBF29CE484222325ull;
                for (int by = 0; by < BRICKS_Y; ++by)
                for (int bx = 0; bx < BRICKS_XZ; ++bx) {
                    uint32_t bm = brickmap[bmIdx(bx, by, bz)];
                    if (bm == BM_EMPTY) {
                        uint8_t tag = 0x00;
                        h = fnv(h, &tag, 1);
                    } else if (bm & BM_SOLID_FLAG) {
                        uint8_t tag[2] = { 0x01, uint8_t(bm & 0xFFu) };
                        h = fnv(h, tag, 2);
                    } else {
                        uint8_t tag = 0x02;
                        h = fnv(h, &tag, 1);
                        h = fnv(h, (const uint8_t*)&brickpool[size_t(bm - 1u) * BRICK_U32],
                                BRICK_U32 * sizeof(uint32_t));
                    }
                }
                rowHash[bz] = h;
            }
        });
    }
    for (auto& w : workers) w.join();
    uint64_t H = 0xCBF29CE484222325ull;
    for (int bz = 0; bz < BRICKS_XZ; ++bz) {
        H ^= rowHash[bz];
        H *= 0x100000001B3ull;
        H ^= H >> 32;
    }
    return H;
}

uint64_t World::regionHash(int x0, int z0, int x1, int z1) const {
    auto fnv = [](uint64_t h, const uint8_t* p, size_t n) {
        for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 0x100000001B3ull; }
        return h;
    };
    uint64_t h = 0xCBF29CE484222325ull;
    int bx0 = x0 >> BRICK_BITS, bx1 = (x1 - 1) >> BRICK_BITS;
    int bz0 = z0 >> BRICK_BITS, bz1 = (z1 - 1) >> BRICK_BITS;
    for (int bz = bz0; bz <= bz1; ++bz)
    for (int by = 0; by < BRICKS_Y; ++by)
    for (int bx = bx0; bx <= bx1; ++bx) {
        uint32_t bm = brickmap[slotOfBrick(bx, by, bz)];
        if (bm == BM_EMPTY) {
            uint8_t tag = 0x00;
            h = fnv(h, &tag, 1);
        } else if (bm & BM_SOLID_FLAG) {
            uint8_t tag[2] = { 0x01, uint8_t(bm & 0xFFu) };
            h = fnv(h, tag, 2);
        } else {
            uint8_t tag = 0x02;
            h = fnv(h, &tag, 1);
            h = fnv(h, (const uint8_t*)&brickpool[size_t(bm - 1u) * BRICK_U32],
                    BRICK_U32 * sizeof(uint32_t));
        }
    }
    return h;
}

void World::uploadChunk(int ccx, int ccz) {
    if (!brickmapSSBO || !brickpoolSSBO) return;
    if (!gpuChunkValid.empty()) gpuChunkValid[chunkSlotIdx(ccx, ccz)] = 1u;
    // Chunk-contiguous layout: the whole column is ONE brickmap range.
    constexpr int PER_CHUNK = CHUNK_BRICKS * CHUNK_BRICKS * BRICKS_Y;
    int base = slotOfBrick(ccx * CHUNK_BRICKS, 0, ccz * CHUNK_BRICKS);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, brickmapSSBO);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER,
                    GLintptr(base) * sizeof(uint32_t),
                    PER_CHUNK * sizeof(uint32_t), &brickmap[base]);
    refreshCoarseForChunk(ccx, ccz);   // keep the occupancy mip in sync (shadow/AO)
    std::vector<int64_t> mixed;
    for (int i = 0; i < PER_CHUNK; ++i) {
        uint32_t bm = brickmap[base + i];
        if (bm != BM_EMPTY && !(bm & BM_SOLID_FLAG))
            mixed.push_back(int64_t(bm - 1u));
    }
    if (mixed.empty()) return;
    ensurePoolGPU();
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, brickpoolSSBO);
    std::sort(mixed.begin(), mixed.end());
    size_t i = 0;
    while (i < mixed.size()) {
        size_t j = i + 1;
        while (j < mixed.size() && mixed[j] == mixed[j - 1] + 1) ++j;
        glBufferSubData(GL_SHADER_STORAGE_BUFFER,
                        mixed[i] * BRICK_U32 * sizeof(uint32_t),
                        (mixed[j - 1] - mixed[i] + 1) * BRICK_U32 * sizeof(uint32_t),
                        &brickpool[size_t(mixed[i]) * BRICK_U32]);
        i = j;
    }
}

void World::gpuBlankChunk(int ccx, int ccz) {
    if (!brickmapSSBO) return;
    if (!gpuChunkValid.empty()) gpuChunkValid[chunkSlotIdx(ccx, ccz)] = 0u;
    constexpr int PER_CHUNK = CHUNK_BRICKS * CHUNK_BRICKS * BRICKS_Y;
    int base = slotOfBrick(ccx * CHUNK_BRICKS, 0, ccz * CHUNK_BRICKS);
    const uint32_t zero = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, brickmapSSBO);
    glClearBufferSubData(GL_SHADER_STORAGE_BUFFER, GL_R32UI,
                         GLintptr(base) * sizeof(uint32_t),
                         PER_CHUNK * sizeof(uint32_t),
                         GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
}

uint8_t World::getVoxel(int x, int y, int z) const {
    if (y < 0 || y >= GRID_Y || !voxelInWindow(x, z)) return 0;
    uint32_t bm = brickmap[slotOfBrick(x >> BRICK_BITS, y >> BRICK_BITS, z >> BRICK_BITS)];
    if (bm == BM_EMPTY) return 0;
    if (bm & BM_SOLID_FLAG) return uint8_t(bm & 0xFFu);
    return readVoxFromBrick(brickpool, bm - 1,
                            x & BRICK_MASK, y & BRICK_MASK, z & BRICK_MASK);
}

void World::setVoxelRaw(int x, int y, int z, uint8_t val) {
    // Worldgen fast path: identical write semantics to setVoxel but no
    // EditDelta tracking (generation uploads everything afterward via
    // uploadInitial) and no wetBricks registration (worldgen water uses its
    // own static-fill path in genLakes precisely to AVOID waking the sim).
    // Runtime/gameplay edits must keep using setVoxel.
    if (y < 0 || y >= GRID_Y || !voxelInWindow(x, z)) return;
    int slotIdx = slotOfBrick(x >> BRICK_BITS, y >> BRICK_BITS, z >> BRICK_BITS);
    uint32_t bm = brickmap[slotIdx];
    if (bm == BM_EMPTY || (bm & BM_SOLID_FLAG)) {
        uint8_t fill = (bm == BM_EMPTY) ? 0 : uint8_t(bm & 0xFFu);
        if (fill == val) return;
        uint32_t bi = allocBrick(fill);
        brickmap[slotIdx] = bi + 1u;
        writeVoxInBrick(brickpool, bi, x & BRICK_MASK, y & BRICK_MASK, z & BRICK_MASK, val);
    } else {
        writeVoxInBrick(brickpool, bm - 1u, x & BRICK_MASK, y & BRICK_MASK, z & BRICK_MASK, val);
    }
}

void World::setVoxel(int x, int y, int z, uint8_t val, EditDelta& d) {
    if (y < 0 || y >= GRID_Y || !voxelInWindow(x, z)) return;
    dirtyChunks.insert(chunkKey(chunkOfVoxel(x), chunkOfVoxel(z)));
    dirtySinceSave.insert(chunkKey(chunkOfVoxel(x), chunkOfVoxel(z)));
    editsSinceSave = true;
    int slotIdx = slotOfBrick(x >> BRICK_BITS, y >> BRICK_BITS, z >> BRICK_BITS);
    if (val == 6 /*MAT_WATER*/) wetBricks.insert(slotIdx);
    if (val == MAT_SAND)        fallingBricks.insert(slotIdx);   // placed powder wakes the granular sim

    // Any edit can destabilise adjacent powder: clearing a voxel removes the
    // support under the grain above (and opens a slump path for diagonal
    // neighbours), so re-wake powder in the cell's neighbourhood or digging
    // into a sand pile would leave the rest hanging like a solid.
    {
        static const int OFF[10][3] = {
            { 0, 1, 0}, { 0,-1, 0},                          // above / below
            { 1, 0, 0}, {-1, 0, 0}, { 0, 0, 1}, { 0, 0,-1},  // 4 laterals
            { 1, 1, 0}, {-1, 1, 0}, { 0, 1, 1}, { 0, 1,-1},  // 4 diagonal-above (can slump in)
        };
        for (auto& o : OFF) {
            int nx = x + o[0], ny = y + o[1], nz = z + o[2];
            if (ny < 0 || ny >= GRID_Y || !voxelInWindow(nx, nz)) continue;
            if (isPowder(getVoxel(nx, ny, nz)))
                fallingBricks.insert(slotOfBrick(nx >> BRICK_BITS, ny >> BRICK_BITS, nz >> BRICK_BITS));
        }
    }
    uint32_t bm = brickmap[slotIdx];

    if (bm == BM_EMPTY || (bm & BM_SOLID_FLAG)) {
        uint8_t fill = (bm == BM_EMPTY) ? 0 : uint8_t(bm & 0xFFu);
        if (fill == val) return;
        uint32_t bi = allocBrick(fill);
        brickmap[slotIdx] = bi + 1u;
        writeVoxInBrick(brickpool, bi, x & BRICK_MASK, y & BRICK_MASK, z & BRICK_MASK, val);
        d.dirtyBrickmap.insert(slotIdx);
        d.dirtyPool.insert(bi);
        // A newly-allocated brick in a previously-empty coarse cell must mark
        // the cell occupied, or shadow/AO rays would skip the new geometry.
        // (Solid-brick cells are already occupied; worldgen builds coarse after.)
        if (bm == BM_EMPTY && !coarse.empty()) {
            int lbx = (x >> BRICK_BITS) & (BRICKS_XZ - 1);   // wrapped local brick coords
            int lby =  y >> BRICK_BITS;                       // (Y is not wrapped)
            int lbz = (z >> BRICK_BITS) & (BRICKS_XZ - 1);
            int ci = cbmIdx(lbx >> COARSE_BITS, lby >> COARSE_BITS, lbz >> COARSE_BITS);
            if (coarse[ci] == 0u) { coarse[ci] = 1u; d.dirtyCoarse.insert(ci); }
        }
    } else {
        uint32_t bi = bm - 1u;
        writeVoxInBrick(brickpool, bi, x & BRICK_MASK, y & BRICK_MASK, z & BRICK_MASK, val);
        d.dirtyPool.insert(bi);
    }
}

// Transient sim write: same voxel mutation + EditDelta collection as setVoxel,
// but WITHOUT marking dirtyChunks/editsSinceSave (physics doesn't persist for
// now) and without auto-waking any active set (the sim manages its own).
void World::setVoxelSim(int x, int y, int z, uint8_t val, EditDelta& d) {
    if (y < 0 || y >= GRID_Y || !voxelInWindow(x, z)) return;
    int slotIdx = slotOfBrick(x >> BRICK_BITS, y >> BRICK_BITS, z >> BRICK_BITS);
    uint32_t bm = brickmap[slotIdx];
    if (bm == BM_EMPTY || (bm & BM_SOLID_FLAG)) {
        uint8_t fill = (bm == BM_EMPTY) ? 0 : uint8_t(bm & 0xFFu);
        if (fill == val) return;
        uint32_t bi = allocBrick(fill);
        brickmap[slotIdx] = bi + 1u;
        writeVoxInBrick(brickpool, bi, x & BRICK_MASK, y & BRICK_MASK, z & BRICK_MASK, val);
        d.dirtyBrickmap.insert(slotIdx);
        d.dirtyPool.insert(bi);
        if (bm == BM_EMPTY && !coarse.empty()) {
            int lbx = (x >> BRICK_BITS) & (BRICKS_XZ - 1);
            int lby =  y >> BRICK_BITS;
            int lbz = (z >> BRICK_BITS) & (BRICKS_XZ - 1);
            int ci = cbmIdx(lbx >> COARSE_BITS, lby >> COARSE_BITS, lbz >> COARSE_BITS);
            if (coarse[ci] == 0u) { coarse[ci] = 1u; d.dirtyCoarse.insert(ci); }
        }
    } else {
        uint32_t bi = bm - 1u;
        writeVoxInBrick(brickpool, bi, x & BRICK_MASK, y & BRICK_MASK, z & BRICK_MASK, val);
        d.dirtyPool.insert(bi);
    }
}

// Granular-settle write: setVoxelSim plus a save re-mark. Sand that keeps
// slumping AFTER an autosave would otherwise be frozen mid-slump forever —
// the carry path reuses the stale compressed bytes verbatim — so if the
// chunk is ALREADY save-dirty, re-mark it (and re-arm the save trigger) so
// the next save re-extracts the live, settled state. Farm writes must stay
// on setVoxelSim: crops are transient by contract (farm.h) and must never
// bake into the edit-save.
void World::setVoxelSettle(int x, int y, int z, uint8_t val, EditDelta& d) {
    setVoxelSim(x, y, z, val, d);
    int64_t ck = chunkKey(chunkOfVoxel(x), chunkOfVoxel(z));
    if (dirtyChunks.count(ck)) {
        dirtySinceSave.insert(ck);
        editsSinceSave = true;
    }
}

// Persist a terraform edit (tilled farmland): the sim-style brick write plus the
// save marks, with no active-set wake. Unconditionally marks the chunk dirty so
// the bed survives a reload (unlike setVoxelSim, which never persists).
void World::setVoxelPersist(int x, int y, int z, uint8_t val, EditDelta& d) {
    if (y < 0 || y >= GRID_Y || !voxelInWindow(x, z)) return;
    setVoxelSim(x, y, z, val, d);
    int64_t ck = chunkKey(chunkOfVoxel(x), chunkOfVoxel(z));
    dirtyChunks.insert(ck);
    dirtySinceSave.insert(ck);
    editsSinceSave = true;
}

int World::editSphere(glm::ivec3 center, int radius, uint8_t value, EditDelta& d) {
    glm::ivec3 wlo(originVoxX(), 0, originVoxZ());
    glm::ivec3 whi = wlo + glm::ivec3(GRID_XZ - 1, GRID_Y - 1, GRID_XZ - 1);
    glm::ivec3 lo = glm::clamp(center - radius, wlo, whi);
    glm::ivec3 hi = glm::clamp(center + radius, wlo, whi);
    int r2 = radius * radius;
    int changed = 0;
    for (int z = lo.z; z <= hi.z; ++z)
    for (int y = lo.y; y <= hi.y; ++y)
    for (int x = lo.x; x <= hi.x; ++x) {
        int dx = x - center.x, dy = y - center.y, dz = z - center.z;
        if (dx*dx + dy*dy + dz*dz > r2) continue;
        uint8_t prev = getVoxel(x, y, z);
        if (value != 0 && prev != 0) continue;       // place: empty cells only
        if (value == 0 && prev == 0) continue;       // dig: solid only
        setVoxel(x, y, z, value, d);
        ++changed;
    }
    return changed;
}

void World::cleanupOrphans(glm::ivec3 lo, glm::ivec3 hi, EditDelta& d) {
    glm::ivec3 wlo(originVoxX(), 0, originVoxZ());
    glm::ivec3 whi = wlo + glm::ivec3(GRID_XZ - 1, GRID_Y - 1, GRID_XZ - 1);
    glm::ivec3 elo = glm::clamp(lo - 1, wlo, whi);
    glm::ivec3 ehi = glm::clamp(hi + 1, wlo, whi);
    for (int z = elo.z; z <= ehi.z; ++z)
    for (int y = elo.y; y <= ehi.y; ++y)
    for (int x = elo.x; x <= ehi.x; ++x) {
        if (getVoxel(x, y, z) == 0) continue;
        int n = 0;
        if (getVoxel(x + 1, y, z) != 0) ++n;
        if (getVoxel(x - 1, y, z) != 0) ++n;
        if (getVoxel(x, y + 1, z) != 0) ++n;
        if (getVoxel(x, y - 1, z) != 0) ++n;
        if (getVoxel(x, y, z + 1) != 0) ++n;
        if (getVoxel(x, y, z - 1) != 0) ++n;
        if (n <= 1) setVoxel(x, y, z, 0, d);
    }
}

RayHit World::raycast(glm::vec3 ro, glm::vec3 rd, float maxDist) const {
    RayHit r{false, {}, {}, 0.0f};
    glm::vec3 invD = 1.0f / rd;
    glm::vec3 wlo(float(originVoxX()), 0.0f, float(originVoxZ()));
    glm::vec3 whi = wlo + glm::vec3((float)GRID_XZ, (float)GRID_Y, (float)GRID_XZ);
    glm::vec3 t0 = (wlo - ro) * invD;
    glm::vec3 t1 = (whi - ro) * invD;
    glm::vec3 ts = glm::min(t0, t1);
    glm::vec3 tb = glm::max(t0, t1);
    float tEnter = std::max({ts.x, ts.y, ts.z});
    float tExit  = std::min({tb.x, tb.y, tb.z});
    if (tExit < 0.0f || tEnter > tExit) return r;

    float t = std::max(tEnter, 0.0f) + 1e-4f;
    glm::vec3 p = ro + rd * t;
    glm::ivec3 v = glm::clamp(glm::ivec3(glm::floor(p)),
                              glm::ivec3(wlo), glm::ivec3(whi) - 1);
    glm::ivec3 stp((rd.x > 0) ? 1 : -1, (rd.y > 0) ? 1 : -1, (rd.z > 0) ? 1 : -1);
    glm::vec3 tDelta = glm::abs(invD);
    glm::vec3 tMax;
    for (int i = 0; i < 3; ++i) {
        float boundary = float(v[i]) + ((stp[i] > 0) ? 1.0f : 0.0f);
        tMax[i] = (boundary - ro[i]) * invD[i];
    }
    int hitAxis = -1;
    int maxIter = int(maxDist) * 3 + 64;
    for (int i = 0; i < maxIter; ++i) {
        if (v.y < 0 || v.y >= GRID_Y || !voxelInWindow(v.x, v.z)) break;
        if (getVoxel(v.x, v.y, v.z) != 0) {
            r.hit = true;
            r.voxel = v;
            r.normal = glm::ivec3(0);
            if      (hitAxis == 0) r.normal.x = -stp.x;
            else if (hitAxis == 1) r.normal.y = -stp.y;
            else if (hitAxis == 2) r.normal.z = -stp.z;
            r.dist = (hitAxis == 0) ? (tMax.x - tDelta.x)
                   : (hitAxis == 1) ? (tMax.y - tDelta.y)
                   : (hitAxis == 2) ? (tMax.z - tDelta.z)
                                    : t;
            if (r.dist > maxDist) { r.hit = false; }
            return r;
        }
        if (tMax.x < tMax.y && tMax.x < tMax.z) {
            v.x += stp.x; tMax.x += tDelta.x; hitAxis = 0;
        } else if (tMax.y < tMax.z) {
            v.y += stp.y; tMax.y += tDelta.y; hitAxis = 1;
        } else {
            v.z += stp.z; tMax.z += tDelta.z; hitAxis = 2;
        }
        if (tMax.x > maxDist && tMax.y > maxDist && tMax.z > maxDist) break;
    }
    return r;
}

void World::uploadInitial() {
    if (brickmapSSBO == 0)  glGenBuffers(1, &brickmapSSBO);
    if (brickpoolSSBO == 0) glGenBuffers(1, &brickpoolSSBO);
    if (coarseSSBO == 0)    glGenBuffers(1, &coarseSSBO);

    buildCoarse();   // build the occupancy mip from the finished brickmap

    // The brickmap SSBO carries the FINE window at [0, brickmap.size()) and
    // every LOD ring's brickmap in TAIL regions after it (the shader indexes
    // ring r via the ringBmBase[r] uniform — one buffer, pure offset math, no
    // arrays-of-SSBO-blocks). Ring tails are left undefined until
    // uploadRings() fills them; the shader never reads a ring while its
    // cascade entry is off.
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, brickmapSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 ringBmOffset(RING_COUNT) * sizeof(uint32_t),
                 nullptr, GL_DYNAMIC_DRAW);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                    brickmap.size() * sizeof(uint32_t), brickmap.data());

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, brickpoolSSBO);
    // PREALLOCATE to the streaming ceiling (the same bound the CPU brickpool
    // reserves: max(highWater*1.5, 6M bricks)). Streaming grows the live
    // brick count several-fold across cave-density regimes; sizing the GPU
    // buffer to the ceiling up front means ensurePoolGPU's grow-by-copy
    // (a ~one-time multi-hundred-ms stall when first hitting a dense region)
    // essentially never fires. The cost is committing that VRAM at load
    // rather than mid-session — VRAM the dense regions need regardless, just
    // surfaced at startup instead of as a stutter. (A render-distance setting
    // is the future knob for scaling this down on smaller GPUs.)
    // HARD WALL: this buffer must stay under 2^32 BYTES — above 4 GiB the
    // driver silently corrupted sub-range uploads (streamed chunks lost
    // surface bricks -> black-terraces desync). The LOD rings therefore live
    // in their OWN pool SSBO (binding 3), never in this one.
    constexpr size_t POOL_BRICKS_MAX = ((size_t(1) << 32) / (BRICK_U32 * sizeof(uint32_t))) - 8;
    size_t actual = size_t(brickHighWater) * BRICK_U32 * sizeof(uint32_t);
    size_t ceilBricks = std::max<size_t>(size_t(brickHighWater) * 3 / 2,
                                         poolReserveBricks());
    ceilBricks = std::min(ceilBricks, POOL_BRICKS_MAX);
    size_t ceil = ceilBricks * BRICK_U32 * sizeof(uint32_t);
    glBufferData(GL_SHADER_STORAGE_BUFFER, ceil, nullptr, GL_DYNAMIC_DRAW);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, actual, brickpool.data());

    // Coarse mip: fine cells at [0, coarse.size()), ring cells in tail
    // regions (shader offset uniforms ringCoarseBase[r]), same one-buffer scheme.
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, coarseSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 ringCoarseOffset(RING_COUNT) * sizeof(uint32_t),
                 nullptr, GL_DYNAMIC_DRAW);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                    coarse.size() * sizeof(uint32_t), coarse.data());

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, brickmapSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, brickpoolSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, coarseSSBO);

    gpuChunkValid.assign(size_t(CHUNKS_XZ) * CHUNKS_XZ, 1u);   // whole window resident

    while (uploadRings()) {}   // startup/teleport: drain every dirty ring now
}

// Upload the FIRST dirty, ready LOD ring: brickmap + coarse into the tail
// regions of the shared SSBOs, brick data into the rings' own pool SSBO at
// each ring's slot base. One ring per call (hitch amortization — see world.h);
// returns whether one was uploaded. GL thread only.
bool World::uploadRings() {
    if (brickmapSSBO == 0) return false;
    if (ringPoolSSBO == 0) {
        glGenBuffers(1, &ringPoolSSBO);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ringPoolSSBO);
        glBufferData(GL_SHADER_STORAGE_BUFFER,
                     size_t(RING_POOL_TOTAL) * BRICK_U32 * sizeof(uint32_t),
                     nullptr, GL_DYNAMIC_DRAW);   // 1.54 GB, well under the 4 GiB wall
    }
    for (int r = 0; r < RING_COUNT; ++r) {
        RingLevel& rl = rings[r];
        if (!rl.ready || !rl.dirty || rl.bm.empty()) continue;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, brickmapSSBO);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER,
                        ringBmOffset(r) * sizeof(uint32_t),
                        rl.bm.size() * sizeof(uint32_t), rl.bm.data());
        if (rl.brickCount > 0) {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, ringPoolSSBO);
            glBufferSubData(GL_SHADER_STORAGE_BUFFER,
                            size_t(ringPoolBase(r)) * BRICK_U32 * sizeof(uint32_t),
                            size_t(rl.brickCount) * BRICK_U32 * sizeof(uint32_t),
                            rl.pool.data());
        }
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, coarseSSBO);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER,
                        ringCoarseOffset(r) * sizeof(uint32_t),
                        rl.coarse.size() * sizeof(uint32_t), rl.coarse.data());
        rl.originGPU = rl.originB;   // shader addressing now matches resident data
        rl.dirty = false;
        std::printf("[ring%d] uploaded: %u bricks (%.0f MB), origin (%d,%d)\n",
                    r + 1, rl.brickCount,
                    rl.brickCount * double(BRICK_U32) * 4.0 / (1024.0 * 1024.0),
                    rl.originB.x, rl.originB.y);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ringPoolSSBO);
        return true;                 // ONE ring per call (hitch amortization)
    }
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ringPoolSSBO);
    return false;                    // nothing dirty
}

void World::uploadDelta(const EditDelta& d) {
    VOX_ZONE_N("uploadDelta");
    // Coalesce dirty indices into contiguous runs so chunk-sized deltas
    // (evict/reload touch 32K brickmap slots per chunk) become a handful of
    // glBufferSubData calls instead of one per index.
    auto runs = [](auto begin, auto end, auto&& emit) {
        std::vector<int64_t> v(begin, end);
        std::sort(v.begin(), v.end());
        size_t i = 0;
        while (i < v.size()) {
            size_t j = i + 1;
            while (j < v.size() && v[j] == v[j - 1] + 1) ++j;
            emit(v[i], v[j - 1] - v[i] + 1);   // (first, count)
            i = j;
        }
    };
    if (!d.dirtyBrickmap.empty()) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, brickmapSSBO);
        runs(d.dirtyBrickmap.begin(), d.dirtyBrickmap.end(),
             [&](int64_t first, int64_t count) {
            glBufferSubData(GL_SHADER_STORAGE_BUFFER,
                            first * sizeof(uint32_t),
                            count * sizeof(uint32_t),
                            &brickmap[size_t(first)]);
        });
    }
    if (!d.dirtyPool.empty()) {
        ensurePoolGPU();
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, brickpoolSSBO);
        runs(d.dirtyPool.begin(), d.dirtyPool.end(),
             [&](int64_t first, int64_t count) {
            glBufferSubData(GL_SHADER_STORAGE_BUFFER,
                            first * BRICK_U32 * sizeof(uint32_t),
                            count * BRICK_U32 * sizeof(uint32_t),
                            &brickpool[size_t(first) * BRICK_U32]);
        });
    }
    if (!d.dirtyCoarse.empty()) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, coarseSSBO);
        for (int ci : d.dirtyCoarse)
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, ci * sizeof(uint32_t),
                            sizeof(uint32_t), &coarse[ci]);
    }
}

void World::ensurePoolGPU() {
    // Regional cave density swings the live brick count several-fold (flat
    // plains ~1.4M bricks per window, cavernous hills ~5M), so the GPU pool
    // must grow during play. Growing via GPU-side COPY (~2 ms) instead of a
    // full CPU re-upload (the old path: a 100+ ms hitch per 32 MB step)
    // keeps VRAM proportional to actual content — no fixed worst-case
    // allocation needed on smaller cards.
    GLint64 sz = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, brickpoolSSBO);
    glGetBufferParameteri64v(GL_SHADER_STORAGE_BUFFER, GL_BUFFER_SIZE, &sz);
    size_t needed = size_t(brickHighWater) * BRICK_U32 * sizeof(uint32_t);
    if (size_t(sz) >= needed) return;
    // Generous step (50%): the grow itself costs a VRAM alloc + GPU copy
    // (~one large hitch), so make it rare across density regimes.
    size_t bumped = needed + std::max<size_t>(needed / 2, size_t(64) * 1024 * 1024);
    unsigned int newBuf = 0;
    glGenBuffers(1, &newBuf);
    glBindBuffer(GL_COPY_WRITE_BUFFER, newBuf);
    while (glGetError() != GL_NO_ERROR) {}      // clear pending errors
    glBufferData(GL_COPY_WRITE_BUFFER, bumped, nullptr, GL_DYNAMIC_DRAW);
    if (glGetError() != GL_NO_ERROR) {
        std::fprintf(stderr, "[world] brickpool grow to %.1f MB failed; "
                             "new bricks may not render\n",
                     bumped / (1024.0 * 1024.0));
        glDeleteBuffers(1, &newBuf);
        return;
    }
    glBindBuffer(GL_COPY_READ_BUFFER, brickpoolSSBO);
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, sz);
    glDeleteBuffers(1, &brickpoolSSBO);
    brickpoolSSBO = newBuf;
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, brickpoolSSBO);
    std::printf("[world] brickpool GPU grew to %.0f MB (copy)\n",
                bumped / (1024.0 * 1024.0));
}
