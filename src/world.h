#pragma once
#include "common.h"
#include <vector>
#include <unordered_set>
#include <cstdint>
#include <glm/glm.hpp>

struct EditDelta {
    std::unordered_set<int>      dirtyBrickmap;  // brickmap slots whose entry changed
    std::unordered_set<uint32_t> dirtyPool;      // pool indices whose contents changed
    std::unordered_set<int>      dirtyCoarse;    // coarse-occupancy cells that flipped to set
};

struct RayHit {
    bool       hit;
    glm::ivec3 voxel;
    glm::ivec3 normal;
    float      dist;
};

// Sparse voxel world. World XZ coordinates are unbounded; the CPU brickmap
// and brickpool hold the RESIDENT WINDOW — GRID_XZ^2 voxels anchored at
// `windowOriginChunk` — addressed toroidally (world brick & mask), exactly
// mirroring the GPU window. Anything outside the window reads as air and
// rejects writes. World mutation happens through setVoxel/editSphere;
// callers pass an EditDelta to collect dirty bricks, then uploadDelta().
struct World {
    std::vector<uint32_t> brickmap;    // BRICKS_XZ * BRICKS_Y * BRICKS_XZ entries
    std::vector<uint32_t> brickpool;   // BRICK_U32 uints per allocated brick
    // Slot allocator: bricks are uniform fixed-size slots, so freed slots
    // recycle with zero fragmentation. CPU slot index == GPU slot index
    // forever (uploadDelta stays 1:1). brickCount = LIVE bricks;
    // brickHighWater = slots ever touched (the upload extent).
    uint32_t              brickCount = 0;
    uint32_t              brickHighWater = 0;
    std::vector<uint32_t> freeBricks;

    // Resident-window anchor, in CHUNK coords (so scrolls are whole chunks).
    // (0,0) until streaming moves it.
    glm::ivec2 windowOriginChunk{0, 0};

    int originVoxX() const { return windowOriginChunk.x * CHUNK_VOX; }
    int originVoxZ() const { return windowOriginChunk.y * CHUNK_VOX; }
    bool voxelInWindow(int x, int z) const {
        int ox = originVoxX(), oz = originVoxZ();
        return x >= ox && x < ox + GRID_XZ && z >= oz && z < oz + GRID_XZ;
    }
    bool brickInWindow(int bx, int by, int bz) const {
        int obx = windowOriginChunk.x * CHUNK_BRICKS;
        int obz = windowOriginChunk.y * CHUNK_BRICKS;
        return by >= 0 && by < BRICKS_Y &&
               bx >= obx && bx < obx + BRICKS_XZ &&
               bz >= obz && bz < obz + BRICKS_XZ;
    }
    // Toroidal slot of a WORLD brick coordinate (caller checks the window).
    // CHUNK-CONTIGUOUS layout (6e-3): each chunk column owns one contiguous
    // 32768-slot range, so streaming syncs a whole chunk with ONE GPU call
    // (upload, blank) instead of 4096 scattered ranges. Mirrored by the
    // shader's bmSlot().
    int slotOfBrick(int bx, int by, int bz) const {
        int cx = (bx >> CHUNK_BRICK_BITS) & (CHUNKS_XZ - 1);
        int cz = (bz >> CHUNK_BRICK_BITS) & (CHUNKS_XZ - 1);
        int local = (bx & (CHUNK_BRICKS - 1))
                  + CHUNK_BRICKS * (bz & (CHUNK_BRICKS - 1))
                  + CHUNK_BRICKS * CHUNK_BRICKS * by;
        return (cx + CHUNKS_XZ * cz) * (CHUNK_BRICKS * CHUNK_BRICKS * BRICKS_Y)
             + local;
    }
    // World brick coords of a window slot, resolved against the origin.
    glm::ivec3 brickOfSlot(int slot) const {
        constexpr int PER_CHUNK = CHUNK_BRICKS * CHUNK_BRICKS * BRICKS_Y;
        int chunkSlot = slot / PER_CHUNK, local = slot % PER_CHUNK;
        int sx = (chunkSlot & (CHUNKS_XZ - 1)) * CHUNK_BRICKS
               + (local & (CHUNK_BRICKS - 1));
        int sz = (chunkSlot / CHUNKS_XZ) * CHUNK_BRICKS
               + ((local / CHUNK_BRICKS) & (CHUNK_BRICKS - 1));
        int sy = local / (CHUNK_BRICKS * CHUNK_BRICKS);
        int obx = windowOriginChunk.x * CHUNK_BRICKS;
        int obz = windowOriginChunk.y * CHUNK_BRICKS;
        return { obx + ((sx - obx) & (BRICKS_XZ - 1)), sy,
                 obz + ((sz - obz) & (BRICKS_XZ - 1)) };
    }

    // GLuint, kept as unsigned int so this header doesn't drag in glad.
    unsigned int brickmapSSBO = 0;
    unsigned int brickpoolSSBO = 0;

    // Coarse occupancy mip (see common.h): one uint per coarse cell, 1 = occupied.
    std::vector<uint32_t> coarse;
    unsigned int coarseSSBO = 0;
    void buildCoarse();   // (re)build occupancy from the brickmap (before uploadInitial)
    void refreshCoarseForChunk(int ccx, int ccz);  // resync coarse cells on stream-in

    // -- Far-field LOD rings (voxel clipmap C1; see common.h) ------------------
    // Built wholesale by buildRings/buildRing (rings.cpp) on the gen thread at
    // startup / teleport regen / background recentre; ring brickmaps + coarse
    // mips upload into TAIL regions of the fine SSBOs, ring brick data into
    // the rings' OWN pool SSBO (binding 3 — never the fine pool, which sits
    // at the 4 GiB driver wall). No free lists — each ring rebuilds from
    // scratch with a bump allocator; mixed entries hold ring-pool GLOBAL
    // indices + 1 (ringPoolBase(r) + local).
    struct RingLevel {
        std::vector<uint32_t> bm;      // ringBmEntries(r), brickmap encoding
        std::vector<uint32_t> pool;    // brick data (uploaded at ringPoolBase(r))
        std::vector<uint32_t> coarse;  // occupancy, shader-order indexed
        uint32_t   brickCount = 0;     // bump-allocator high-water (ring-local)
        glm::ivec2 originB{0, 0};      // ring-brick origin (ring vox = world >> (r+1))
        glm::ivec2 originGPU{0, 0};    // origin of the data ON the GPU (set at upload
                                       // so a background rebuild never desyncs
                                       // shader addressing from resident data)
        bool       ready = false;      // gates this ring in the shader cascade
        bool       dirty = false;      // CPU rebuilt since last upload
    };
    RingLevel rings[RING_COUNT];
    unsigned int ringPoolSSBO = 0;
    // Upload AT MOST ONE dirty ring per call (a full ring is ~100-350 MB of
    // SubData; several in one frame stalls the driver against in-flight
    // reads). Returns true if a ring was uploaded — callers that must drain
    // everything (startup) loop; the per-frame pump calls it once. GL thread.
    bool uploadRings();
    // Tail offsets of ring r's regions inside the shared SSBOs.
    static size_t ringBmOffset(int r) {
        size_t off = size_t(BRICKS_XZ) * BRICKS_XZ * BRICKS_Y;
        for (int i = 0; i < r; ++i) off += size_t(ringBmEntries(i));
        return off;
    }
    static size_t ringCoarseOffset(int r) {
        size_t off = size_t(CBRICKS_XZ) * CBRICKS_XZ * CBRICKS_Y;
        for (int i = 0; i < r; ++i) off += size_t(ringCoarseEntries(i));
        return off;
    }
    // Toroidal slot of a RING brick coord — mirrors the shader's ringSlot()
    // (the same chunk-contiguous layout as slotOfBrick, at ring r's dims).
    static int ringSlotOfBrick(int r, int bx, int by, int bz) {
        int lx = bx & (RING_BRICKS_XZ - 1);
        int lz = bz & (RING_BRICKS_XZ - 1);
        int chunkSlot = (lx >> CHUNK_BRICK_BITS)
                      + (RING_BRICKS_XZ >> CHUNK_BRICK_BITS) * (lz >> CHUNK_BRICK_BITS);
        int local = (lx & (CHUNK_BRICKS - 1))
                  + ((lz & (CHUNK_BRICKS - 1)) << CHUNK_BRICK_BITS)
                  + (by << (2 * CHUNK_BRICK_BITS));
        return chunkSlot * (CHUNK_BRICKS * CHUNK_BRICKS * ringBricksY(r)) + local;
    }

    // Brickmap slot indices that contain at least one water voxel. Used by
    // the water simulation to scope which bricks need ticking each frame.
    std::unordered_set<int> wetBricks;

    // Same, for the fire automaton: bricks holding a MAT_FIRE/MAT_EMBER voxel.
    // Drains to empty as fire burns out (see fire.cpp / FireSim).
    std::unordered_set<int> burnBricks;

    // Same, for the granular (falling-sand) sim: bricks with unsettled powder.
    // Woken when a powder voxel is placed/disturbed; drains as it settles.
    std::unordered_set<int> fallingBricks;

    // Chunk columns touched by gameplay edits (setVoxel; worldgen's
    // setVoxelRaw deliberately does NOT mark — procgen is reproduced from
    // seed, only player changes persist). Keys are chunkKey(ccx, ccz) in
    // WORLD chunk coords (unbounded). Loaded edit-chunks are re-inserted on
    // load so a later save keeps them.
    std::unordered_set<int64_t> dirtyChunks;
    // Chunks edited since the LAST save (subset of dirtyChunks; cleared by
    // saveWorldEdits). dirtyChunks is cumulative — it lists everything the
    // save file must RETAIN, and grows for the life of a world — so the
    // save recompresses only this subset and carries every other chunk's
    // previous compressed bytes verbatim. Without the split, a long-lived
    // world's autosave recompressed hundreds of chunks every 60 s (whole
    // seconds of one-frame hitch).
    std::unordered_set<int64_t> dirtySinceSave;
    bool editsSinceSave = false;   // cleared by saveWorldEdits

    void initEmpty();
    void uploadInitial();                         // creates SSBOs and uploads CPU data
    void uploadDelta(const EditDelta& d);         // sub-uploads dirty bricks/slots
    // Push one chunk column's brickmap entries + mixed bricks to the GPU
    // (streaming milestone sync). One contiguous brickmap range + coalesced
    // pool ranges.
    void uploadChunk(int ccx, int ccz);
    // Zero one chunk column's GPU brickmap range (entering strips must show
    // air, not the leaving strip's stale content, until their upload lands).
    void gpuBlankChunk(int ccx, int ccz);
    // Per-chunk-slot "the GPU holds renderable content" flags: cleared by
    // gpuBlankChunk, set by uploadChunk/uploadInitial. THE signal for the
    // fine-march reveal cap: a blanked-not-yet-uploaded chunk is air on the
    // GPU (rays tunnel into cut-open interiors behind it), while a chunk
    // merely awaiting a milestone RE-upload still shows valid earlier-stage
    // content and must NOT collapse the cap (that painted coarse-ring slabs
    // over perfectly good terrain). Indexed by wrapped chunk coords.
    std::vector<uint8_t> gpuChunkValid;
    static int chunkSlotIdx(int ccx, int ccz) {
        return (ccx & (CHUNKS_XZ - 1)) + CHUNKS_XZ * (ccz & (CHUNKS_XZ - 1));
    }
    // Grow the GPU brick pool (GPU-side copy, ~ms) when highWater outruns it.
    void ensurePoolGPU();

    uint32_t allocBrick(uint8_t fill);
    // Same, but skips the fill: for callers that memcpy the whole brick
    // immediately (chunk commits/reloads — thousands per streamed chunk).
    uint32_t allocBrickNoFill();
    // Recycle a pool slot (chunk overlay replacing generated bricks; later:
    // chunk eviction). The slot's stale data is never sampled — no brickmap
    // entry points at it once freed.
    void     freeBrick(uint32_t bi);

    uint8_t getVoxel(int x, int y, int z) const;
    void    setVoxel(int x, int y, int z, uint8_t v, EditDelta& d);
    // Worldgen-only: same write as setVoxel but no delta/wetBricks tracking.
    void    setVoxelRaw(int x, int y, int z, uint8_t v);
    // Sim write (water/granular/fire-style automata): updates voxels + collects
    // the EditDelta for the GPU sub-upload, but does NOT mark dirtyChunks — so
    // transient physics never bloats the edit-save. (Persistence is a one-call
    // switch to setVoxel later.) Does not auto-wake any active set; the sim
    // manages its own.
    void    setVoxelSim(int x, int y, int z, uint8_t v, EditDelta& d);
    // setVoxelSim + save re-mark for the granular sim: settled sand must
    // supersede a stale mid-slump autosave snapshot (farm crops must NOT
    // use this — they are transient by contract).
    void    setVoxelSettle(int x, int y, int z, uint8_t v, EditDelta& d);
    // setVoxelSim + the save marks (dirtyChunks/dirtySinceSave/editsSinceSave)
    // but WITHOUT waking any active sim set — a terraform edit that must persist
    // (tilled farmland) without being treated as granular/water. Crops still use
    // plain setVoxelSim (FarmState persists them, not the edit-save).
    void    setVoxelPersist(int x, int y, int z, uint8_t v, EditDelta& d);

    // Semantic content hash: covers every voxel's MATERIAL in fixed slot
    // order, independent of brickpool allocation order. Two worlds with
    // identical voxels hash equal even if their pools were filled in a
    // different order — the validation tool proving that different
    // generation drivers (monolithic vs chunk-by-chunk) build the same world.
    uint64_t contentHash() const;

    // Semantic hash of a WORLD-coordinate voxel region (brick-aligned reads,
    // iterated in world order) — comparable across different window ORIGINS,
    // unlike contentHash whose slot order is window-relative. The region
    // must lie inside the resident window. The 6e-3 overlap gate: two
    // windows at different origins must hash their shared region equal.
    uint64_t regionHash(int x0, int z0, int x1, int z1) const;

    // Returns number of voxel cells changed. Skips no-ops.
    int  editSphere(glm::ivec3 center, int radius, uint8_t value, EditDelta& d);

    // After a dig, sweep a slightly larger box and remove voxels with <=1
    // neighbour. Cleans the spherical-brush "fuzz" left at the edge of pits.
    void cleanupOrphans(glm::ivec3 lo, glm::ivec3 hi, EditDelta& d);

    RayHit raycast(glm::vec3 ro, glm::vec3 rd, float maxDist) const;
};

// Helpers used by worldgen and others. Must match World::slotOfBrick's
// chunk-contiguous layout (this is the origin-0, in-range variant).
inline int bmIdx(int bx, int by, int bz) {
    int local = (bx & (CHUNK_BRICKS - 1))
              + CHUNK_BRICKS * (bz & (CHUNK_BRICKS - 1))
              + CHUNK_BRICKS * CHUNK_BRICKS * by;
    return ((bx >> CHUNK_BRICK_BITS) + CHUNKS_XZ * (bz >> CHUNK_BRICK_BITS))
         * (CHUNK_BRICKS * CHUNK_BRICKS * BRICKS_Y) + local;
}
inline int cbmIdx(int cx, int cy, int cz) {
    return cx + CBRICKS_XZ * (cy + CBRICKS_Y * cz);
}
inline int voxInBrick(int lx, int ly, int lz) {
    return lx + BRICK * (ly + BRICK * lz);
}
