#pragma once
// Forestry: RuneScape-style tree chopping over the diorama cursor. Trees are
// baked voxels with no entity identity, so a click recovers one by flood fill
// (collectTree). Felling clears the tree's voxels (a stump stays behind) and
// hands the felled set to a transient rigid voxel model that hinges over at
// the stump through the creature part pipeline; on impact it shatters into
// debris items — physical voxel chunks that bounce, settle, hover, and
// magnetize to the player. Small wood clusters (a stump, fresh or from an
// old save), charred skeletons, and stranded leaf blobs classify as stumps
// and get a "clear it" action instead.
//
// The chop SESSION (menu -> walk to tree -> swing loop) lives in app.cpp with
// the rest of the input/pathfinding state; this module owns the mechanics
// that only need the World: identification, felling, fall physics, debris.
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include "creatures.h"   // CreatureModel, CreaturePartInstance, CreatureBound

struct World;
struct ParticleSystem;
struct GenContext;

// A boulder recovered as ONE rock: identity comes from the generator
// (rederiveRockAt splits conjoined clusters by last-writer ownership), with
// a bounded flood fill as the fallback whose caps double as the
// "is this actually a boulder?" test — a click on exposed strata or a stone
// hillside floods past them and is refused.
struct RockInfo {
    bool valid = false;
    glm::ivec3 base{0};               // bottom-slab centroid (stand-ring anchor)
    float radius = 2.0f;              // XZ footprint radius (arrival ring sizing)
    std::vector<glm::ivec3> stone;    // the boulder's MAT_STONE voxels
    glm::ivec3 lo{0}, hi{0};
};

// A connected tree, cactus, or stump recovered from the voxel world.
struct TreeInfo {
    bool valid   = false;
    bool isStump = false;             // stump-shaped or charred — not fellable
    bool isCactus = false;            // flesh trunk + bloom foliage cluster
    glm::ivec3 base{0};               // trunk base (grounded-cluster centroid)
    int   cutY   = 0;                 // fell plane: wood below it near the base stays
    float trunkR = 2.0f;              // base-slab radius (arrival ring sizing)
    float stumpR = 3.5f;              // stump keep-radius (tight for small trees
                                      // — a knee-high fork must fell, not stay)
    int   charN  = 0;                 // charred voxels in the wood set
    std::vector<glm::ivec3> wood;     // wood-like voxels (see woodMats)
    std::vector<uint8_t>    woodMats; // parallel: MAT_WOOD/MAT_DESERT_WOOD/MAT_CHAR
    std::vector<glm::ivec3> leaves;   // this tree's attached MAT_LEAVES voxels
    glm::ivec3 lo{0}, hi{0};          // bounds over wood + leaves
};

// One debris item: a voxel chunk that flies to the player.
struct DebrisItem {
    glm::vec3 pos{0}, vel{0};
    uint8_t mat   = 0;                // MAT_WOOD or MAT_LEAVES
    uint8_t state = 0;                // 0 airborne, 1 settled (hover), 2 magnetized
    float age = 0.0f;
    float hoverY = 0.0f;              // settled hover base
    float spin = 0.0f;                // yaw phase
    float magAt = 0.4f;               // age when the magnet engages (staggered
                                      // per item so pickups stream, not clump)
    // Last RENDERED transform (real motion vectors for the TAA/motion pass).
    glm::vec3 prevPos{0};
    float prevA = 0.0f;
    bool  hasPrev = false;
};

// A felled tree mid-fall: a transient rigid voxel model (part 0 wood, part 1
// leaves; cellScale 1, model-local space = world - pivot) hinging at the
// stump top. Instances are rebuilt per frame; prevXf carries last frame's
// rows so the motion/TAA pass sees real motion vectors.
struct FallingTree {
    CreatureModel model;
    glm::vec3 pivot{0};               // world hinge (stump top centre)
    glm::vec3 dir{0};                 // horizontal fall direction (unit)
    float angle = 0.0f, angVel = 0.0f;
    float trunkLen = 8.0f;
    glm::vec3 boundCentroid{0};       // model-local bound centre (rotation-invariant r)
    float boundRadius = 8.0f;
    std::vector<glm::ivec3> fellWood; // original world voxel positions (pre-fall)
    std::vector<uint8_t>    fellMats;
    std::vector<glm::ivec3> fellLeaf;
    std::vector<uint8_t>    fellLeafMats;  // leaves vs cactus blooms
    // Contact probes: sampled voxels of the whole model (trunk AND crown) in
    // pivot-local space — the fall crumbles the moment ANY of them sweeps
    // into solid ground, never clipping in first.
    std::vector<glm::vec3> probes;
    std::vector<float> prevXf;        // 12 floats/part: last frame's rows
    // Per-part FILE-LOCAL word bases, snapshotted before the first atlas
    // append (appendModelToAtlas rebases wordBase in place; a live fall
    // re-appended during atlas compaction must rebase from these).
    std::vector<uint32_t> localWordBase;
    bool done = false;
};

// One amortized leaf-support sweep. The invariant it enforces: a MAT_LEAVES
// voxel survives only if its 26-conn leaf-graph distance to wood-like
// support (wood / desert wood / char / stump cut face) is <= LEAF_SUPPORT_N.
// Seeded from removed voxels (fell, uproot, fire's burned-away leaves); the
// flood covers the WHOLE touched leaf component — no bbox escape hatch —
// with per-frame budgets so mega-canopies crumble over ~a second instead of
// hitching one frame. Ungrounded floating wood stubs met along the way
// dissolve too.
struct LeafDecayJob {
    int phase = 0;                          // 0 gather, 1 relax, 2 remove
    std::deque<glm::ivec3> seeds;           // removed voxels: expand to leaves
    std::deque<glm::ivec4> frontier;        // candidate leaf positions (w =
                                            // steps from a seed: the gather
                                            // horizon — see forestry.cpp)
    std::unordered_set<int64_t> queued;     // ever-pushed keys: push-time
                                            // dedup (26x duplicate fanout)
    std::unordered_map<int64_t, int> idxOf; // leaf voxel -> region index
    std::vector<glm::ivec3> voxels;         // the gathered leaf region
    std::vector<int16_t>    dist;           // -1 unknown; else steps from wood
    std::deque<int>         bfs;            // relax queue (support seeds first)
    std::unordered_map<int64_t, bool> woodVerdict;  // wood voxel -> grounded?
    std::vector<glm::ivec3> deadWood;       // ungrounded stubs to dissolve
    size_t cursor = 0, cursorW = 0;         // phase-2 progress
    int removed = 0;                        // stats for the timber log line
};

// -- Hover highlight ---------------------------------------------------------
// Identifying the tree under the cursor costs ~10-40 ms (generator replay) —
// fine per click, far too much per frame — so hover identity runs on a
// WORKER THREAD: the main thread requests the voxel under the cursor, the
// worker replays the generator (rederiveTreeAt is pure — world seed +
// stateless terrain functions; its grid scratch is thread_local) and builds
// a 0/1 byte mask over the tree's bbox, and the main thread uploads it as a
// small 3D texture the lighting pass tints. A frame or two of latency is
// imperceptible for a hover cue.
struct TreeHoverResult {
    uint64_t   req = 0;              // request id this answers (stale-drop)
    bool       found = false;
    bool       isCactus = false;     // spot-check against flesh, not wood
    glm::ivec3 lo{0};                // mask origin (world voxels)
    glm::ivec3 dims{0};
    std::vector<uint8_t> mask;       // dims.x*y*z bytes (x-fastest), 1 = tree
    int        woodN = 0;            // emitted wood count
    int        woodH = 0;            // emitted wood height (stump gate)
    // A spread of emitted wood voxels the MAIN thread spot-checks against
    // the live world. The registry path is pure generator replay; the flood
    // fallback (non-registry flora) DOES read the live World off-thread —
    // an accepted streaming-compute-style race (see treeHoverMain), which is
    // exactly why adoption is gated by this spot-check: mostly gone (or
    // torn) => don't advertise it as clickable.
    std::vector<glm::ivec3> woodSample;
};

struct TreeHover {
    std::thread worker;
    std::mutex mtx;
    std::condition_variable cv;
    bool quit = false;
    bool hasReq = false;
    uint64_t reqId = 0;
    glm::ivec3 reqVox{0};
    const GenContext* gen = nullptr;
    const World* world = nullptr;    // flood fallback for non-registry flora
    bool hasResult = false;
    TreeHoverResult result;
    ~TreeHover();
};

// Ask the worker for the tree containing `vox`. Coalescing: a newer request
// replaces an unserviced one; a stale in-flight result is dropped on
// completion. Starts the thread on first use.
void treeHoverRequest(TreeHover& th, const GenContext& gen, const World& w,
                      glm::ivec3 vox);
// Fetch the newest finished result (returns true once per completion).
bool treeHoverPoll(TreeHover& th, TreeHoverResult& out);

struct Forestry {
    // Each live fall costs one creature-bound slot (traced shadow), so the
    // count is hard-capped; past it a fell shatters in place instead.
    static constexpr int MAX_FALLS = 4;
    std::vector<FallingTree> falls;
    std::vector<DebrisItem>  items;
    std::deque<LeafDecayJob> decayJobs;
    TreeHover hover;
    int collected[256] = {};          // absorbed items per material (inventory later)
    // Shared static models, appended to the base creature atlas at startup.
    CreatureModel woodChunk, leafChunk, cactusChunk, stoneChunk, axeModel,
                  pickModel;
    std::mt19937 rng{0xF0357EEDu};
};

// Build the wood-chunk / leaf-chunk / axe models (call once, then append each
// to the atlas blob with appendModelToAtlas before the initial GPU upload).
void forestryBuildStaticModels(Forestry& f);

// Identify the tree containing `seed` (a wood-like or leaf voxel). Asks the
// GENERATOR first (rederiveTreeAt: trees derive bit-identically from the
// world seed + lattice cell, so membership is exact — the sparse object
// registry in its purest form), intersected with the current world so
// dug/burned voxels drop out. Falls back to flood-fill + ownership
// heuristics for wood the generator can't answer for (palms/desert flora,
// gate divergence, old-save drift). Returns false when there's no tree
// there or it crosses the resident-window edge.
bool collectTree(const World& w, const GenContext& gen, glm::ivec3 seed,
                 TreeInfo& out);

// Identify the boulder containing `seed` (a MAT_STONE voxel). Asks the
// GENERATOR first (rederiveRockAt: boulders derive bit-identically from the
// world seed + SEED_BOULDER lattice, with overlap voxels partitioned by
// StageWriter's last-writer rule) — so each rock of a CONJOINED cluster, and
// giant erratics the flood's terrain caps refuse, answer as individually
// minable rocks. The claimed set is clamped at the soil line (the deep root
// stays as the quarry scar) and intersected with live MAT_STONE. Falls back
// to the bounded flood fill (whose size/extent caps double as the
// "boulder, not terrain" test) when the replay is distrusted — gate
// divergence over cave-carved anchors, seam-class drift. Returns false for
// strata/hillside stone, sub-ROCK_MIN pebbles, or a rock crossing the
// resident-window edge.
bool collectRock(const World& w, const GenContext& gen, glm::ivec3 seed,
                 RockInfo& out);

// Crumble: clears the boulder's voxels from the world and bursts them
// straight into magnetizing stone-chunk debris + dust (the uprootStump
// model — rocks have no hinge to fall on, they just come apart).
void mineRock(Forestry& f, World& w, const RockInfo& r, ParticleSystem& ps);

// Fell: clears the tree's voxels from the world (leaving the stump), spawns
// the falling rigid model tipping AWAY from `awayFrom`, and appends its
// volume to atlasDyn (grow-only — the caller re-uploads the atlas SSBO).
// Returns false when there's nothing to fell OR the fall roster is full —
// the caller should fall back to uprootStump.
bool fellTree(Forestry& f, World& w, const TreeInfo& t, glm::vec3 awayFrom,
              ParticleSystem& ps, std::vector<uint32_t>& atlasDyn);

// Clear a cluster in place: removes it and bursts the voxels straight into
// debris items (char burns to dust). No longer player-facing — kept as the
// fell fallback for a full fall roster.
void uprootStump(Forestry& f, World& w, const TreeInfo& t, ParticleSystem& ps);

// Physics tick: fall hinge + impact shatter, item bounce/settle/magnet/absorb.
// Returns true when a fall retired this tick (the caller then compacts the
// dynamic atlas: base blob + the still-live falls).
bool forestryTick(Forestry& f, const World& w, glm::vec3 playerChest,
                  ParticleSystem& ps, float dt);

// Queue a leaf-support sweep seeded by removed voxel positions (their leaf
// neighbours start the flood). fellTree/uprootStump queue themselves; fire
// queues the leaves it burns to air.
void forestryQueueLeafDecay(Forestry& f, const std::vector<glm::ivec3>& removed);
// Amortized decay work: call once per unpaused frame (after forestryTick).
void forestryDecayTick(Forestry& f, World& w, ParticleSystem& ps);

// Atlas compaction: rebuild `atlasDyn` as `atlasBase` + every still-live
// fall's model (their stamped offsets are refreshed). Bounds atlas growth
// under sustained chopping; call when a fall retires, then re-upload.
void forestryCompactAtlas(Forestry& f, const std::vector<uint32_t>& atlasBase,
                          std::vector<uint32_t>& atlasDyn);

// Render-list building. Falls carry CreatureBound entries (traced shadows) —
// append these BEFORE the player avatar's own bound (lighting.frag assumes
// the player is the LAST bound). Items are instances only: no bounds, so the
// per-pixel shadow loop never marches 300 chunks. Returns bounds appended.
int  appendFallInstances(Forestry& f, std::vector<CreaturePartInstance>& parts,
                         CreatureBound* bounds, int maxBounds);
void appendItemInstances(Forestry& f, std::vector<CreaturePartInstance>& parts);

// The equipped tools: one instance riding the player's right-forearm
// instance (its rows already fold in the avatar's pose + yaw + cellScale).
CreaturePartInstance makeAttachedAxe(const Forestry& f,
                                     const CreaturePartInstance& hand,
                                     glm::ivec3 handDims);
CreaturePartInstance makeAttachedPick(const Forestry& f,
                                      const CreaturePartInstance& hand,
                                      glm::ivec3 handDims);
