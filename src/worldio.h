#pragma once
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
struct World;
struct GenContext;
struct EditDelta;

// Player-edit persistence (the procgen-era replacement for the old whole-world
// cache): terrain is ALWAYS regenerated from the seed at startup â€” generation
// is deterministic and chunk-isolated, proven by World::contentHash â€” and only
// the chunks a player has edited are stored, as a region file of compressed
// per-chunk brick data. Saves scale with edits, not world size.
//
// saveWorldEdits: writes every dirty chunk to `path` (atomically via a temp
//                 file). Clears World::editsSinceSave. True on success.
//                 A dirty chunk that is currently EVICTED serves its blob
//                 from `cache` (the live world holds nothing for it).
// loadWorldEdits: overlays saved chunks onto the freshly generated world and
//                 re-marks them dirty (so the next save retains them). False
//                 (and world untouched) if the file is missing or belongs to
//                 a different world seed. A generator-version mismatch warns
//                 but still applies â€” player edits outrank fresh terrain.

// SESSION-SCOPED chunk cache (6e-2): evicted chunks park their full content
// here, compressed, and restore byte-identically on re-entry. Deliberately
// NOT persisted â€” generation is deterministic in streaming order (the ladder
// proof), so a relaunch rebuilds identical chunks from the seed and edits;
// the cache only carries content across an evict/reload within one session.
// An entry's presence == "this chunk is currently evicted".
struct ChunkCache {
    struct Entry {
        std::vector<uint32_t> raw;       // uncompressed blob (fresh evictions)
        std::vector<uint8_t>  comp;      // zlib blob once compacted
        uint32_t rawBytes = 0;
    };
    std::unordered_map<int64_t, Entry> entries;   // key = chunkKey(ccx, ccz)
    size_t totalComp = 0, totalRaw = 0;            // gauges: resident bytes
    // Guards `entries`/totals against the background compressor thread.
    // Held only for O(1) map ops + a blob copy (sub-ms) — NEVER during zlib.
    std::mutex mtx;
    bool has(int64_t key) const { return entries.count(key) != 0; }
};

// Compress ONE raw cache entry (zlib). Designed to run on a dedicated
// background thread: the cache mutex is held only to copy the raw blob out
// and to store the result, never across the compression itself, so a
// window scroll's evict/reload never blocks on zlib (that was a 40 ms
// main-thread hitch). Returns true if it compressed something, false if no
// raw entries remain (caller should sleep briefly then retry).
bool compressOneFromCache(ChunkCache& cache);

// Legacy synchronous compactor (still used by the offline evict-check gate).
void compactChunkCache(ChunkCache& cache, int maxN);

// Serialize one chunk column into the edits-file blob format (read-only —
// parallel-safe; the scroll path extracts a whole strip across cores).
void extractChunkBlob(const World& w, int ccx, int ccz,
                      std::vector<uint32_t>& blob);

// Evict one chunk column: serialize its CURRENT content (including any
// edits) into the cache, return its mixed bricks to the slot allocator,
// blank its brickmap entries, drop its wetBricks slots, and mark the slots
// in `d` for GPU upload. False if the chunk is already evicted.
bool evictChunk(World& w, ChunkCache& cache, int ccx, int ccz, EditDelta* d = nullptr);

// Reload one evicted chunk from the cache, byte-identical: restores bricks
// (reusing freed slots), re-marks `d`, and wakes water ONLY for edited
// chunks (static lake water must stay out of the sim's wetBricks set).
// Erases the cache entry. False if the chunk is not in the cache.
bool reloadChunk(World& w, ChunkCache& cache, int ccx, int ccz, EditDelta* d = nullptr);

// Blank one chunk column WITHOUT caching it â€” for partially generated
// frontier chunks leaving the window (their content must never resurrect
// as final; they regenerate from the seed on return).
void dropChunk(World& w, int ccx, int ccz, EditDelta* d = nullptr);

bool saveWorldEdits(const World& world, const GenContext& ctx,
                    const std::string& path, const ChunkCache* cache = nullptr);
bool loadWorldEdits(World& world, const GenContext& ctx, const std::string& path);
