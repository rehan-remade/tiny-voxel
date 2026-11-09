#include "worldio.h"
#include "world.h"
#include "common.h"
#include "worldgen/worldgen.h"
#include "profiling.h"

#include <miniz.h>

#include <algorithm>
#include <cstdio>
#include <unordered_map>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

constexpr char     MAGIC[4] = {'T', 'V', 'X', 'E'};   // E = edits
constexpr uint32_t VERSION  = 1u;

#pragma pack(push, 1)
struct Header {
    char     magic[4];
    uint32_t version;
    uint64_t worldSeed;     // edits only apply to the world they were made in
    uint32_t genVersion;    // worldgen logic version at save time
    uint32_t chunkVox;      // CHUNK_VOX at save time
    uint32_t chunkCount;
};
struct ChunkHeader {
    int32_t  ccx, ccz;
    uint32_t rawBytes;      // uncompressed blob size
    uint32_t compBytes;     // zlib stream size
};
#pragma pack(pop)

// One chunk's blob layout (before compression):
//   CHUNK_BRICKS * BRICKS_Y * CHUNK_BRICKS uint32 entries, iterated
//   (by, bzLocal, bxLocal): BM_EMPTY, BM_SOLID_FLAG|mat, or 1-based index
//   into the mixed-brick payload array that follows (BRICK_U32 u32s each,
//   in entry order).
constexpr size_t CHUNK_ENTRIES = size_t(CHUNK_BRICKS) * CHUNK_BRICKS * BRICKS_Y;

size_t zlibCompressInto(const void* src, size_t srcBytes, std::vector<uint8_t>& out) {
    mz_ulong bound = compressBound(mz_ulong(srcBytes));
    out.resize(bound);
    mz_ulong destLen = bound;
    int rc = mz_compress(out.data(), &destLen,
                         (const unsigned char*)src, mz_ulong(srcBytes));
    if (rc != MZ_OK) { out.clear(); return 0; }
    out.resize(destLen);
    return destLen;
}
bool zlibDecompressInto(const void* comp, size_t compBytes, void* dst, size_t dstBytes) {
    mz_ulong outLen = mz_ulong(dstBytes);
    int rc = mz_uncompress((unsigned char*)dst, &outLen,
                           (const unsigned char*)comp, mz_ulong(compBytes));
    return rc == MZ_OK && outLen == dstBytes;
}

// Serialize one chunk column from the live world into `blob`.
void extractChunk(const World& w, int ccx, int ccz, std::vector<uint32_t>& blob) {
    blob.clear();
    blob.resize(CHUNK_ENTRIES);
    std::vector<uint32_t> bricks;        // mixed payloads, appended after entries
    size_t e = 0;
    int bx0 = ccx * CHUNK_BRICKS, bz0 = ccz * CHUNK_BRICKS;
    for (int by = 0; by < BRICKS_Y; ++by)
    for (int lz = 0; lz < CHUNK_BRICKS; ++lz)
    for (int lx = 0; lx < CHUNK_BRICKS; ++lx, ++e) {
        uint32_t bm = w.brickmap[w.slotOfBrick(bx0 + lx, by, bz0 + lz)];
        if (bm == BM_EMPTY || (bm & BM_SOLID_FLAG)) {
            // A solid crop brick (crops are sparse, so this is defensive) saves
            // as air — crops never bake into the edit-save (FarmState owns them).
            blob[e] = ((bm & BM_SOLID_FLAG) && isCropMaterial(uint8_t(bm & 0xFFu)))
                    ? BM_EMPTY : bm;
        } else {
            const uint32_t* src = &w.brickpool[size_t(bm - 1u) * BRICK_U32];
            size_t base = bricks.size();
            bricks.insert(bricks.end(), src, src + BRICK_U32);
            // Strip TRANSIENT crop voxels: FarmState re-stamps them, so baking
            // them here would resurrect un-harvestable ghost pumpkins on reload.
            for (size_t wi = base; wi < bricks.size(); ++wi) {
                uint32_t word = bricks[wi];
                for (int b = 0; b < 4; ++b)
                    if (isCropMaterial(uint8_t((word >> (b * 8)) & 0xFFu)))
                        word &= ~(uint32_t(0xFFu) << (b * 8));
                bricks[wi] = word;
            }
            blob[e] = uint32_t(bricks.size() / BRICK_U32);   // 1-based local index
        }
    }
    blob.insert(blob.end(), bricks.begin(), bricks.end());
}

// Overlay one serialized chunk onto the live world, replacing the generated
// content of that column wholesale. Replaced mixed bricks are returned to
// the slot allocator's free list. `wakeWater` registers the chunk's water
// with the sim â€” pass true only for EDITED chunks; bulk-waking static lake
// water tanks the simulation.
void overlayChunk(World& w, int ccx, int ccz, const std::vector<uint32_t>& blob,
                  bool wakeWater) {
    const uint32_t* payload = blob.data() + CHUNK_ENTRIES;
    size_t payloadCount = (blob.size() - CHUNK_ENTRIES) / BRICK_U32;
    size_t e = 0;
    int bx0 = ccx * CHUNK_BRICKS, bz0 = ccz * CHUNK_BRICKS;
    for (int by = 0; by < BRICKS_Y; ++by)
    for (int lz = 0; lz < CHUNK_BRICKS; ++lz)
    for (int lx = 0; lx < CHUNK_BRICKS; ++lx, ++e) {
        int slot = w.slotOfBrick(bx0 + lx, by, bz0 + lz);
        uint32_t v = blob[e];
        uint32_t old = w.brickmap[slot];
        if (v == BM_EMPTY || (v & BM_SOLID_FLAG)) {
            if (old != BM_EMPTY && !(old & BM_SOLID_FLAG)) w.freeBrick(old - 1u);
            w.brickmap[slot] = v;
            if (wakeWater &&
                (v & BM_SOLID_FLAG) && uint8_t(v & 0xFFu) == MAT_WATER)
                w.wetBricks.insert(slot);
        } else {
            if (v == 0 || v > payloadCount) continue;        // corrupt entry: skip
            if (old != BM_EMPTY && !(old & BM_SOLID_FLAG)) w.freeBrick(old - 1u);
            uint32_t bi = w.allocBrickNoFill();
            const uint32_t* src = payload + size_t(v - 1u) * BRICK_U32;
            std::memcpy(&w.brickpool[size_t(bi) * BRICK_U32],
                        src, BRICK_U32 * sizeof(uint32_t));
            w.brickmap[slot] = bi + 1u;
            // Wake persisted water: wetBricks isn't saved, so without this a
            // player's poured water stays frozen after a relaunch. Scoped to
            // edited chunks only â€” cheap. (When static lakes return, a dirty
            // lakeside chunk would wake its slice of lake too; revisit then.)
            if (wakeWater)
            for (int i = 0; i < BRICK_U32; ++i) {
                uint32_t word = src[i];
                if ((( word        & 0xFFu) == MAT_WATER) ||
                    (((word >> 8 ) & 0xFFu) == MAT_WATER) ||
                    (((word >> 16) & 0xFFu) == MAT_WATER) ||
                    (((word >> 24) & 0xFFu) == MAT_WATER)) {
                    w.wetBricks.insert(slot);
                    break;
                }
            }
        }
    }
}

} // namespace

void extractChunkBlob(const World& w, int ccx, int ccz,
                      std::vector<uint32_t>& blob) {
    extractChunk(w, ccx, ccz, blob);
}

bool evictChunk(World& w, ChunkCache& cache, int ccx, int ccz, EditDelta* d) {
    int64_t key = chunkKey(ccx, ccz);
    if (cache.has(key)) return false;            // already evicted

    // Serialize CURRENT content (generated + any edits) into the cache â€”
    // RAW: compression is deferred to compactChunkCache so a window scroll
    // never pays 32 zlib passes in one frame.
    ChunkCache::Entry e;
    extractChunk(w, ccx, ccz, e.raw);
    e.rawBytes = uint32_t(e.raw.size() * sizeof(uint32_t));
    {
        std::lock_guard<std::mutex> lk(cache.mtx);   // vs background compressor
        cache.totalRaw += e.rawBytes;
        cache.entries.emplace(key, std::move(e));
    }

    // Blank the column: free mixed bricks, clear brickmap, drop wet slots.
    bool wet = !w.wetBricks.empty();   // hoist: 32K hash erases otherwise
    int bx0 = ccx * CHUNK_BRICKS, bz0 = ccz * CHUNK_BRICKS;
    for (int by = 0; by < BRICKS_Y; ++by)
    for (int lz = 0; lz < CHUNK_BRICKS; ++lz)
    for (int lx = 0; lx < CHUNK_BRICKS; ++lx) {
        int slot = w.slotOfBrick(bx0 + lx, by, bz0 + lz);
        uint32_t bm = w.brickmap[slot];
        if (bm != BM_EMPTY && !(bm & BM_SOLID_FLAG)) w.freeBrick(bm - 1u);
        w.brickmap[slot] = BM_EMPTY;
        if (wet) w.wetBricks.erase(slot);
        if (d) d->dirtyBrickmap.insert(slot);
    }
    return true;
}

bool compressOneFromCache(ChunkCache& cache) {
    // Phase 1 (locked): find a raw-but-uncompressed entry, COPY its blob out.
    // Copying (not moving) leaves the entry serviceable for a concurrent
    // reload while we compress without the lock.
    int64_t key = 0;
    std::vector<uint32_t> raw;
    uint32_t rawBytes = 0;
    {
        std::lock_guard<std::mutex> lk(cache.mtx);
        for (auto& kv : cache.entries) {
            if (!kv.second.raw.empty() && kv.second.comp.empty()) {
                key = kv.first;
                raw = kv.second.raw;            // ~1.6 MB copy, ~0.2 ms
                rawBytes = kv.second.rawBytes;
                break;
            }
        }
        if (raw.empty()) return false;
    }
    // Phase 2 (UNLOCKED): the expensive zlib pass.
    std::vector<uint8_t> comp;
    if (!zlibCompressInto(raw.data(), rawBytes, comp)) return true;
    // Phase 3 (locked): publish, unless the entry was reloaded/erased meanwhile.
    {
        std::lock_guard<std::mutex> lk(cache.mtx);
        auto it = cache.entries.find(key);
        if (it != cache.entries.end() && !it->second.raw.empty() &&
            it->second.comp.empty()) {
            cache.totalComp += comp.size();
            cache.totalRaw  -= it->second.raw.size() * sizeof(uint32_t);
            it->second.comp = std::move(comp);
            it->second.raw.clear();
            it->second.raw.shrink_to_fit();
        }
    }
    return true;
}

void compactChunkCache(ChunkCache& cache, int maxN) {
    if (cache.totalRaw == 0) return;
    int compacted = 0;
    for (auto& kv : cache.entries) {
        if (compacted >= maxN) break;
        ChunkCache::Entry& e = kv.second;
        if (e.raw.empty()) continue;
        if (zlibCompressInto(e.raw.data(), e.rawBytes, e.comp)) {
            cache.totalRaw  -= e.raw.size() * sizeof(uint32_t);
            cache.totalComp += e.comp.size();
            e.raw.clear();
            e.raw.shrink_to_fit();
            ++compacted;
        }
    }
}

void dropChunk(World& w, int ccx, int ccz, EditDelta* d) {
    bool wet = !w.wetBricks.empty();
    int bx0 = ccx * CHUNK_BRICKS, bz0 = ccz * CHUNK_BRICKS;
    for (int by = 0; by < BRICKS_Y; ++by)
    for (int lz = 0; lz < CHUNK_BRICKS; ++lz)
    for (int lx = 0; lx < CHUNK_BRICKS; ++lx) {
        int slot = w.slotOfBrick(bx0 + lx, by, bz0 + lz);
        uint32_t bm = w.brickmap[slot];
        if (bm != BM_EMPTY && !(bm & BM_SOLID_FLAG)) w.freeBrick(bm - 1u);
        w.brickmap[slot] = BM_EMPTY;
        if (wet) w.wetBricks.erase(slot);
        if (d) d->dirtyBrickmap.insert(slot);
    }
}

bool reloadChunk(World& w, ChunkCache& cache, int ccx, int ccz, EditDelta* d) {
    int64_t key = chunkKey(ccx, ccz);
    // Pull the blob out + erase the entry under the lock (vs the background
    // compressor). Decompress inside the lock too — reload is at most once
    // per frame and rare (re-approaching cached terrain), so the brief hold
    // can't matter; only the bg thread ever waits on it.
    std::vector<uint32_t> blob;
    {
        std::lock_guard<std::mutex> lk(cache.mtx);
        auto it = cache.entries.find(key);
        if (it == cache.entries.end()) return false;
        if (!it->second.raw.empty()) {
            blob = std::move(it->second.raw);        // not yet compressed
        } else {
            blob.resize(it->second.rawBytes / sizeof(uint32_t));
            if (!zlibDecompressInto(it->second.comp.data(), it->second.comp.size(),
                                    blob.data(), it->second.rawBytes))
                return false;
        }
        cache.totalComp -= it->second.comp.size();
        cache.totalRaw  -= it->second.raw.size() * sizeof(uint32_t);
        cache.entries.erase(it);
    }
    // Wake water only for edited chunks; reloaded static lake water must
    // stay out of the simulation's working set.
    overlayChunk(w, ccx, ccz, blob, w.dirtyChunks.count(key) != 0);

    // Mark the whole column for GPU upload: every brickmap slot, plus the
    // pool slots its mixed bricks landed in.
    int bx0 = ccx * CHUNK_BRICKS, bz0 = ccz * CHUNK_BRICKS;
    for (int by = 0; by < BRICKS_Y; ++by)
    for (int lz = 0; lz < CHUNK_BRICKS; ++lz)
    for (int lx = 0; lx < CHUNK_BRICKS; ++lx) {
        int slot = w.slotOfBrick(bx0 + lx, by, bz0 + lz);
        uint32_t bm = w.brickmap[slot];
        if (d) d->dirtyBrickmap.insert(slot);
        if (bm != BM_EMPTY && !(bm & BM_SOLID_FLAG)) if (d) d->dirtyPool.insert(bm - 1u);
    }
    return true;
}

// Read an existing edits file's chunk blobs (compressed, verbatim), keyed by
// chunk. Used by saveWorldEdits as the fallback source for dirty chunks that
// can no longer be materialized locally — see the aliasing note there.
static void readEditsBlobs(const GenContext& ctx, const std::string& path,
                           std::unordered_map<int64_t,
                               std::pair<ChunkHeader, std::vector<uint8_t>>>& out) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return;
    Header h{};
    if (std::fread(&h, sizeof(h), 1, f) != 1 ||
        std::memcmp(h.magic, MAGIC, 4) != 0 || h.version != VERSION ||
        h.chunkVox != uint32_t(CHUNK_VOX) || h.worldSeed != ctx.seed ||
        h.genVersion != GEN_VERSION) {
        std::fclose(f);
        return;
    }
    for (uint32_t i = 0; i < h.chunkCount; ++i) {
        ChunkHeader ch{};
        if (std::fread(&ch, sizeof(ch), 1, f) != 1) break;
        std::vector<uint8_t> comp(ch.compBytes);
        if (std::fread(comp.data(), 1, comp.size(), f) != comp.size()) break;
        out[chunkKey(ch.ccx, ch.ccz)] = { ch, std::move(comp) };
    }
    std::fclose(f);
}

bool saveWorldEdits(const World& w, const GenContext& ctx, const std::string& path,
                    const ChunkCache* cache) {
    VOX_ZONE_N("saveWorldEdits");
    // Stable order (sorted chunk keys) so identical edit-states produce
    // identical files.
    std::vector<int64_t> ids(w.dirtyChunks.begin(), w.dirtyChunks.end());
    std::sort(ids.begin(), ids.end());

    // Previous save's blobs: the ONLY safe source for a dirty chunk that is
    // neither in the cache nor inside the resident window. extractChunk's
    // slotOfBrick masks toroidally, so extracting an OUT-OF-WINDOW chunk
    // silently serializes whatever distant chunk aliases to the same slot —
    // which then overlays as foreign terrain at the original coords forever
    // (and re-marks itself dirty: self-perpetuating corruption). Reached via
    // teleports and long flights whose auto-save lands far from the edits.
    std::unordered_map<int64_t,
        std::pair<ChunkHeader, std::vector<uint8_t>>> prev;
    readEditsBlobs(ctx, path, prev);

    const std::string tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "[worldio] cannot open %s for write\n", tmp.c_str());
        return false;
    }
    Header h{};
    std::memcpy(h.magic, MAGIC, 4);
    h.version    = VERSION;
    h.worldSeed  = ctx.seed;
    h.genVersion = GEN_VERSION;
    h.chunkVox   = CHUNK_VOX;
    h.chunkCount = uint32_t(ids.size());   // patched after the loop (skips)
    bool ok = std::fwrite(&h, sizeof(h), 1, f) == 1;

    std::vector<uint32_t> blob;
    std::vector<uint8_t>  comp;
    size_t totalComp = 0;
    uint32_t written = 0, carried = 0, dropped = 0;
    for (int64_t id : ids) {
        if (!ok) break;
        int ccx = chunkKeyX(id), ccz = chunkKeyZ(id);
        // Not edited since the last save -> the previous file's compressed
        // bytes are still authoritative: carry them VERBATIM. This is the
        // autosave's hot path — dirtyChunks is cumulative for the life of a
        // world, and recompressing all of it every 60 s cost whole seconds
        // in one frame once a playtest world had a few hundred edited
        // chunks. Only the dirtySinceSave subset pays extract+compress.
        if (!w.dirtySinceSave.count(id)) {
            if (auto it = prev.find(id); it != prev.end()) {
                ok = std::fwrite(&it->second.first, sizeof(ChunkHeader), 1, f) == 1 &&
                     std::fwrite(it->second.second.data(), 1,
                                 it->second.second.size(),
                                 f) == it->second.second.size();
                totalComp += it->second.second.size();
                ++carried;
                continue;
            }
            // No prior blob (e.g. save file deleted mid-session): fall
            // through to the live/cache paths below.
        }
        // A dirty chunk that is currently EVICTED has nothing in the live
        // world; its cache entry holds the authoritative (already
        // compressed, same blob format) content.
        if (cache) {
            auto it = cache->entries.find(id);
            if (it != cache->entries.end()) {
                // Raw (not-yet-compacted) entries compress on the fly.
                const std::vector<uint8_t>* blobC = &it->second.comp;
                if (!it->second.raw.empty()) {
                    if (!zlibCompressInto(it->second.raw.data(),
                                          it->second.rawBytes, comp)) {
                        ok = false; break;
                    }
                    blobC = &comp;
                }
                ChunkHeader ch{ ccx, ccz, it->second.rawBytes,
                                uint32_t(blobC->size()) };
                ok = std::fwrite(&ch, sizeof(ch), 1, f) == 1 &&
                     std::fwrite(blobC->data(), 1, blobC->size(),
                                 f) == blobC->size();
                totalComp += blobC->size();
                ++written;
                continue;
            }
        }
        // Live extraction is only valid for chunks INSIDE the resident
        // window (see the aliasing note above readEditsBlobs). Out-of-window
        // chunks with no cache entry keep their previous saved blob.
        if (w.brickInWindow(ccx * CHUNK_BRICKS, 0, ccz * CHUNK_BRICKS)) {
            VOX_ZONE_N("chunk compress");
            extractChunk(w, ccx, ccz, blob);
            size_t rawBytes = blob.size() * sizeof(uint32_t);
            if (!zlibCompressInto(blob.data(), rawBytes, comp)) { ok = false; break; }
            ChunkHeader ch{ ccx, ccz, uint32_t(rawBytes), uint32_t(comp.size()) };
            ok = std::fwrite(&ch, sizeof(ch), 1, f) == 1 &&
                 std::fwrite(comp.data(), 1, comp.size(), f) == comp.size();
            totalComp += comp.size();
            ++written;
        } else if (auto it = prev.find(id); it != prev.end()) {
            ok = std::fwrite(&it->second.first, sizeof(ChunkHeader), 1, f) == 1 &&
                 std::fwrite(it->second.second.data(), 1, it->second.second.size(),
                             f) == it->second.second.size();
            totalComp += it->second.second.size();
            ++carried;
        } else {
            // Nothing trustworthy to write: better a lost edit than foreign
            // terrain pasted over it. Loud, so the producer path is findable.
            std::fprintf(stderr, "[worldio] dirty chunk (%d,%d) is out-of-window "
                                 "with no cache/prior blob — DROPPED\n", ccx, ccz);
            ++dropped;
        }
    }
    if (ok && written + carried != h.chunkCount) {
        h.chunkCount = written + carried;          // patch the real count in
        ok = std::fseek(f, 0, SEEK_SET) == 0 &&
             std::fwrite(&h, sizeof(h), 1, f) == 1;
    }
    std::fclose(f);
    if (!ok) { std::remove(tmp.c_str()); return false; }
    if (carried || dropped)
        std::printf("[worldio] save: %u live, %u carried from prior file, %u dropped\n",
                    written - 0u, carried, dropped);

    // Atomic-ish replace: never leave a torn file at `path`.
    std::remove(path.c_str());
    if (std::rename(tmp.c_str(), path.c_str()) != 0) return false;
    std::printf("[worldio] saved %zu edited chunk(s) to %s (%.1f KB; "
                "%u recompressed, %u carried)\n",
                ids.size(), path.c_str(), totalComp / 1024.0, written, carried);
    const_cast<World&>(w).editsSinceSave = false;
    const_cast<World&>(w).dirtySinceSave.clear();
    return true;
}

bool loadWorldEdits(World& w, const GenContext& ctx, const std::string& path) {
    VOX_ZONE_N("loadWorldEdits");
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    Header h{};
    if (std::fread(&h, sizeof(h), 1, f) != 1 ||
        std::memcmp(h.magic, MAGIC, 4) != 0 || h.version != VERSION ||
        h.chunkVox != uint32_t(CHUNK_VOX)) {
        std::fprintf(stderr, "[worldio] %s: unrecognized edits file, ignoring\n",
                     path.c_str());
        std::fclose(f);
        return false;
    }
    if (h.worldSeed != ctx.seed) {
        std::fprintf(stderr, "[worldio] %s belongs to another world seed, ignoring\n",
                     path.c_str());
        std::fclose(f);
        return false;
    }
    if (h.genVersion != GEN_VERSION) {
        // Stale terrain version: these edits were authored against a heightmap
        // that no longer exists. Overlaying them would paste old-terrain
        // chunks onto the new surface -> flat-topped mesas / vertical
        // chunk-aligned seams. DISCARD the whole file instead (the chunks
        // regenerate fresh and seamless).
        std::fprintf(stderr, "[worldio] %s saved under gen v%u (now v%u): "
                             "discarding stale edits so terrain stays seamless\n",
                     path.c_str(), h.genVersion, GEN_VERSION);
        std::fclose(f);
        return false;
    }

    std::vector<uint8_t>  comp;
    std::vector<uint32_t> blob;
    uint32_t applied = 0;
    for (uint32_t i = 0; i < h.chunkCount; ++i) {
        ChunkHeader ch{};
        if (std::fread(&ch, sizeof(ch), 1, f) != 1) break;
        comp.resize(ch.compBytes);
        if (std::fread(comp.data(), 1, comp.size(), f) != comp.size()) break;
        // Only overlay chunks inside the resident window (at startup the
        // window holds the spawn region; off-window edits stay in the file
        // and apply when streaming brings their chunks in).
        if (!w.brickInWindow(ch.ccx * CHUNK_BRICKS, 0, ch.ccz * CHUNK_BRICKS))
            continue;
        if (ch.rawBytes < CHUNK_ENTRIES * sizeof(uint32_t)) continue;
        blob.resize(ch.rawBytes / sizeof(uint32_t));
        if (!zlibDecompressInto(comp.data(), comp.size(), blob.data(), ch.rawBytes))
            continue;
        overlayChunk(w, ch.ccx, ch.ccz, blob, /*wakeWater=*/true);
        // Re-mark dirty so the next save retains this chunk.
        w.dirtyChunks.insert(chunkKey(ch.ccx, ch.ccz));
        ++applied;
    }
    std::fclose(f);
    w.editsSinceSave = false;
    std::printf("[worldio] applied %u edited chunk(s) from %s\n", applied, path.c_str());
    return applied > 0;
}
