#pragma once
#include <glm/glm.hpp>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>
#include <string>
#include "common.h"

struct World;
struct CavePlan;   // full cave plan (ops + decoration requests); caves.cpp

// --- Emissive cave lighting (glow crystals, later fungi/lava) ---------------
// Each emissive feature records a point light during generation. A coarse 3D
// RGB "light volume" is splatted from these and sampled in the shader so the
// glow softly illuminates the surrounding rock (a handheld lantern can't fill
// a mega-cavern; the cavern lights ITSELF). The source list is tiny, so we
// persist it in a sidecar and rebuild the volume every startup (cache or fresh)
// just like the heightmap.
struct CaveLight { glm::vec3 pos; glm::vec3 color; float strength; };

constexpr int LV_SCALE = 16;                 // voxels per light-volume cell
constexpr int LV_X = GRID_XZ / LV_SCALE;     // 128
constexpr int LV_Y = GRID_Y  / LV_SCALE;     // 256
constexpr int LV_Z = GRID_XZ / LV_SCALE;     // 128

// -- Generation stats (golden-stats harness) -----------------------------------
// Every phase reports what it actually placed. Printed at the end of
// generateTerrain and written to worldgen_stats.txt so refactors of the
// generation pipeline can be validated against a baseline run by diffing
// counts (statistical similarity) rather than eyeballing the world.
struct GenStats {
    // flora
    int boulders = 0, bushes = 0, trees = 0, flowers = 0;
    int grassClusters = 0, grassBlades = 0;
    // caves
    int caveSystems = 0, deepSystems = 0, megaProposed = 0, megaCarved = 0;
    long long wormSteps = 0;
    // Max XZ displacement of any carve op (incl. its radius) from its
    // system's spawn site. THE sizing input for step 6b's region queries:
    // a region must replay every system whose spawn cell lies within this
    // distance of its bounds.
    int wormMaxReachXZ = 0;
    int formReqs = 0, crystalReqs = 0, decoReqs = 0, ivyReqs = 0;
    int crystalsBuilt = 0, decoBuilt = 0, ivyBuilt = 0;
    // timings, in phase order
    struct PhaseTime { const char* name; double sec; };
    std::vector<PhaseTime> phases;
};

// The default seed of the (currently only) world. Becomes player-chosen /
// saved per-world once worlds are streamed.
constexpr uint64_t DEFAULT_WORLD_SEED = 0xA11CE5EEDull;

// Bump when worldgen output changes meaningfully (terrain height, strata,
// feature placement). Stamped into the player-edits file; on load a mismatch
// means those edits were authored against terrain that no longer exists, so
// they are DISCARDED rather than overlaid — pasting old-terrain chunks onto a
// new heightmap produces flat-topped mesas / vertical chunk-aligned seams.
// v2: regional terrain character (hills / valleys / flat meadow in heightAt).
// v3: biome POC — climate field drives terrain relief + grass tint (3 biomes).
// v4: biome field tuned — contrast-stretched, smaller regions, distinct tints.
// v5: per-biome flora — forest dense trees, highland boulders, meadow flowers.
// v6: highland exposes bare grey rock on high ground (vegetation self-clears).
// v7: desert biome — dry-tail climate split, gravel-pavement surface, shallow
//     banded sandstone strata, warm-tan tint (dunes/mesas/flora are fast-follow).
// v8: desert dunes (asymmetric wind slip-faces in clustered ergs) + flat-topped
//     sandstone mesas; sand/sandstone/pavement surface grammar.
// v9: desert oasis — basin + static water pool + fan palms + lush bank ring.
// v10: dunes ARE the base terrain — removed the dedicated dune height term
//      (periodic & FBM both read as artificial ridges); ergs only pick the sand
//      surface now, the swells come from baseHAt's natural isotropic relief.
// v11: oasis basin redesign — water level tied to the MINIMUM rim (12-spoke
//      baseHAt ring), basin-gated placement (genuine lows only), a continuous
//      lift-to-rimMin berm + warped shore => provably contained, organic pools.
// v12: tall fan palms (trunk 46..76, crown scaled to match the forest trees).
// v13 (base): trees raised — longer trunks + branch attach heights clamped to
//      TREE_CANOPY_CLEAR so the character walks UNDER crowns.
// v14: + micro-relief hummocks in baseHAt (this merge combines both, so it
//      differs from either lineage's v13 — bump to force a clean regen).
constexpr uint32_t GEN_VERSION = 14u;

// -- Generation context --------------------------------------------------------
// Everything one generated world derives from + produces, in one object:
// the seed in, the registries/maps out. No worldgen global state — a context
// can be created per world (and later: generation can run per-chunk against
// the same context, since all derivation is (seed, coords)-pure).
struct GenContext {
    uint64_t seed = DEFAULT_WORLD_SEED;

    // Cave entrance (worm-start) surface positions. Dev console /cave warps
    // to the nearest one.
    std::vector<glm::ivec3> caveEntrances;
    // Mega-cavern centres (deep grand halls). Dev console /mega warps to one.
    // Persisted in the world.lights sidecar, so it survives cached loads.
    std::vector<glm::ivec3> megaCaverns;
    // Emissive sources (crystal clusters). Persisted in the sidecar; the
    // light volume below is rebuilt from them every startup.
    std::vector<CaveLight>  caveLights;

    // Far-field LoD heightfield (Phase A horizon): FAR_TEX*FAR_TEX floats of
    // heightAt() over a FAR_EXTENT-wide region centred on (farOriginX,farOriginZ)
    // (the region's min corner). Cheap (no caves/flora); rebuilt at startup +
    // teleport. Uploaded to the GPU and raymarched in the g-buffer miss branch.
    std::vector<float> farHeightMap;
    // Paired far-field SURFACE COLOR (packed RGBA8, FAR_TEX^2): the aggregate
    // top-of-column colour derived from the SAME gen logic the near world uses
    // (biome grassTint + rocky-peak grey + flora-density canopy tint), so any
    // future biome/vegetation is captured automatically. Rendered chunky.
    std::vector<uint32_t> farColorMap;
    int farOriginX = 0, farOriginZ = 0;   // world-voxel min corner of the far region

    // Per-column terrain surface height, GRID_XZ*GRID_XZ floats (row-major
    // z*XZ+x). Uploaded to the GPU as a texture so the shader can compute
    // each point's depth below the LOCAL surface (drives cave darkness /
    // skylight). Rebuilt deterministically every startup (cache or fresh).
    std::vector<float> heightMap;
    // Coarse emissive light volume, LV_X*LV_Y*LV_Z * 3 (RGB), x fastest.
    std::vector<float> lightVol;

    // Cached cave plan (worm walks -> carve ops + decoration requests),
    // derived once from `seed` and reused by every bounded generation call.
    std::shared_ptr<CavePlan> cavePlan;

    GenStats stats;
};

// Optional progress hook: if set, called with each phase label as generation
// crosses a phase boundary (e.g. "init", "bulk_fill", "caves", ...). The loading
// screen installs this to publish the current step. Plain fn pointer so worldgen
// stays decoupled from the app; must be cheap + thread-safe (gen runs on a
// background thread while the main thread renders the loading screen).
extern void (*g_worldgenPhaseHook)(const char* label);

void generateTerrain(World& w, GenContext& ctx);
// Validation driver: generates the SAME world as generateTerrain but
// chunk-by-chunk through the bounded phase API (stage-outer, chunk-inner).
// Slower; exists to prove chunk isolation — World::contentHash() of both
// drivers must match. Enabled at startup via the VOXEL_CHUNKED_GEN env var.
void generateTerrainChunked(World& w, GenContext& ctx);

// Fill ctx.heightMap for the resident window whose voxel origin is (ox,oz),
// at toroidal texel indices (matching the REPEAT-wrapped heightTex). Origin
// (0,0) — the startup default — is the plain linear fill.
void buildHeightmap(GenContext& ctx, int ox = 0, int oz = 0);
// Fill ctx.farHeightMap: a coarse FAR_TEX^2 heightfield (heightAt only, no
// caves/flora) over the FAR_EXTENT-wide region CENTRED on world (cx,cz). Sets
// ctx.farOriginX/Z to the region's min corner. Multithreaded; cheap.
void buildFarHeightmap(GenContext& ctx, int cx, int cz);
// Build far-field LOD rings (rings.cpp; docs/lod-rings-architecture.md):
// coarse 2^(r+1)x-voxel brickmaps centred on the resident window — terrain +
// oasis water + tree impostors from the pure gen core. Fill w.rings[r]
// (uploaded by World::uploadRings via uploadInitial, or after a background
// recentre rebuild). Gen/background thread; no GL. Both terrain drivers call
// buildRings after the flora stages; the recentre rebuilds single rings.
// buildRing centres ring r on the EXPLICIT world-voxel snapshot (cxW, czW)
// and does NOT publish ready/dirty (the caller does, on the main thread).
void buildRing(World& w, GenContext& ctx, int r, int cxW, int czW);
void buildRings(World& w, GenContext& ctx);
// Splat ctx.caveLights into ctx.lightVol as a TOROIDAL WINDOW at voxel
// origin (ox,oz) — only in-window lights contribute; X/Z wrap to match the
// REPEAT-sampled lightVol texture. Origin (0,0) is the startup default.
void buildLightVolume(GenContext& ctx, int ox = 0, int oz = 0);
// Incremental version: rebuild only the strip that changed on a single
// one-chunk window scroll (`dir`, one axis), instead of the whole 48 MB
// volume. Returns the texture sub-rect to re-upload in (outXAxis, outOff,
// outCnt); outCnt may wrap the toroidal edge. Identical result to a full
// buildLightVolume for every cell whose value changes.
void buildLightVolumeStrip(GenContext& ctx, int ox, int oz, glm::ivec2 dir,
                           bool& outXAxis, int& outOff, int& outCnt);
// Incrementally splat ONLY caveLights[from..end) (newly streamed clusters)
// into the volume, returning the WORLD LV-cell box written for a partial
// re-upload (empty if no in-window new lights). Avoids the full rebuild when
// the cave frontier appends lights as you walk.
void splatNewLights(GenContext& ctx, int ox, int oz, size_t from,
                    int& wbx0, int& wbx1, int& wbz0, int& wbz1);

// -- Streaming chunk generation (6e-3b) -----------------------------------------
// The CHUNK-STATUS LADDER as a runtime scheduler: every chunk has a stage,
// and a chunk may advance to stage s only when every chunk within that
// stage's dependency radius is at >= s-1 (out-of-window neighbours count as
// NONE, so the window's leading rim is a naturally graded pipeline — the
// frontier finalizes only once its margin neighbourhood exists, in any
// scroll direction). step() advances a parallel batch per call. In-game
// frontier reads accept seam-class divergence (gates/probes read live final
// neighbours instead of stage-exact state); content is pinned by the chunk
// cache once generated, and the offline drivers remain the exact oracle.
enum : uint8_t {
    CS_NONE = 0, CS_BULK, CS_CARVE, CS_DECOR, CS_BOULDER,
    CS_BUSH, CS_TREE, CS_LAKE, CS_FLOWER, CS_GRASS,
    CS_FINAL = CS_GRASS,
};
// Pure terrain height (world coords, unbounded) — heightAt for non-worldgen
// callers (height-texture strip updates on window scroll).
int terrainHeightAt(int x, int z);
// Pure blended tree density in [0,1] at a world column (biomeAt().treeChance):
// FOREST ~0.94, MEADOW ~0.34, HIGHLAND ~0.04, DESERT 0. The runtime "is this
// forest?" query for creature spawning — no world state, thread-safe.
float forestDensityAt(int x, int z);
// Deterministic RNG seed for a creature herd-lattice cell (featureSeed with
// the stable SEED_CREATURE phase). Exported so gameplay code never includes
// worldgen internals.
uint32_t creatureCellSeed(uint64_t worldSeed, int cellX, int cellZ);
// Dev-console nav: nearest oasis / mesa SITE to a world XZ. Out coords are a
// teleport target (oasis = a vantage above the pool; mesa = atop the caprock).
// Return false if none found within the search radius.
bool nearestOasis(int px, int pz, int& ox, int& oy, int& oz);
bool nearestMesa (int px, int pz, int& mx, int& my, int& mz);

struct EditDelta;
struct ChunkStreamerImpl;
class ChunkStreamer {
public:
    ChunkStreamer(World& w, GenContext& ctx);
    ~ChunkStreamer();
    void seedAllFinal();                     // after batch startup generation
    // Drain any in-flight async batch (wait for the worker pool to go idle)
    // and forget all chunk state — required before a teleport regenerates
    // the world under the streamer (the pool must not be reading the world
    // initEmpty() is about to wipe). Caller follows with seedAllFinal().
    void reset();
    // Chunks the scheduler must never touch (currently-evicted cache entries
    // — they reload byte-identical instead of regenerating).
    void setSkip(std::function<bool(int64_t)> skip);
    uint8_t stageOf(int ccx, int ccz) const;
    void markFinal(int ccx, int ccz);        // chunk-cache reload
    void clear(int ccx, int ccz);            // eviction / partial drop
    // Advance up to one parallel batch of same-stage chunks (nearest the
    // player first). Returns chunks advanced; dirty slots collect in `d`.
    int step(glm::ivec2 playerChunk, EditDelta& d);
    // Chunks whose GPU image is stale (bulk/carve/final milestones); the app
    // drains a few per frame through World::uploadChunk.
    int drainUploads(int64_t* out, int maxCount);
    int pendingInWindow() const;             // gauge: chunks not yet FINAL
private:
    ChunkStreamerImpl* impl;
};

// "Ask the generator" (forestry tree identity): forest trees derive
// bit-identically from (world seed, lattice cell) — the same property chunk
// streaming depends on — so the owning tree of a clicked voxel is recovered
// by replaying the tree grammar for every site in reach. `voxels` holds the
// owner's ENTIRE emitted set (as-emitted materials, wood/leaves); callers
// intersect it with the current world (a voxel whose material no longer
// matches was dug/burned/claimed by terrain and drops out). Returns false
// when no replayed tree emitted the clicked voxel — the odd pure-grammar
// gate divergence or non-forest wood — and the caller should fall back to
// flood-fill identification. Implemented in worldgen/flora.cpp.
struct TreeRederive {
    bool found = false;
    glm::ivec3 base{0};       // trunk base (site position, h = surface)
    float trunkR = 2.0f;      // site trunk thickness (stump/arrival sizing)
    std::vector<std::pair<glm::ivec3, uint8_t>> voxels;
};
bool rederiveTreeAt(const GenContext& ctx, glm::ivec3 click, TreeRederive& out);

// Exact boulder identity from the generator (same registry idea, SEED_BOULDER
// lattice): replays every boulder site in reach of `click` in canonical order
// and partitions overlap voxels by StageWriter's last-writer rule, so each
// rock of a CONJOINED cluster is answered separately. `w` feeds the anchor
// grass-gate oracle (live grass, else pure grammar) — genBoulders draws a
// whole cell from one sequential stream, so the gate must be right or the
// cell's later sites derive garbage; callers must spot-check the claimed set
// against live materials and fall back to flood identification. Reads the
// World: MAIN THREAD ONLY (the tree hover worker path stays pure).
// Implemented in worldgen/flora.cpp.
struct RockRederive {
    bool found = false;
    glm::ivec3 anchor{0};      // site anchor (x, h = surface height, z)
    std::vector<glm::ivec3> voxels;   // voxels OWNED by this rock (last-writer)
};
bool rederiveRockAt(const GenContext& ctx, const World& w, glm::ivec3 click,
                    RockRederive& out);

// After overlaying saved player edits onto a fresh world: drop plan-derived
// lights whose crystal cluster was mined out in an edited chunk (otherwise a
// mined cavern would glow again every launch). Caller rebuilds the volume.
// Player-PLACED crystal lights are re-derived in step 8 (lights-from-voxels);
// until then they don't survive a restart.
void pruneCaveLights(const World& w, GenContext& ctx);
