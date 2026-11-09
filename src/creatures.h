#pragma once
// Wildlife: data-driven voxel creatures (stags first). Each species is a pair
// of files in assets/creatures/ — a .cvx2 rigid-part asset baked offline from
// a rigged GLB (tools/creatures/bake_parts.py: per-bone rest volumes + per-
// frame transform tracks) and a .def gameplay tuning file. Creatures are
// transient like FireSim: never saved, herds re-derive deterministically from
// the world seed on a spawn lattice, so the same forests hold the same herds
// every session.
//
// Creature-local axes (the .cvx2 convention): x lateral, y up, z forward,
// origin at the feet centre. World facing follows the player convention
// fwd = (cos yaw, 0, sin yaw).
#include "common.h"
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>
#include <glm/glm.hpp>

struct World;

// One animation clip inside a .cvx2 asset.
struct CreatureClip {
    char     name[17] = {};
    uint32_t first = 0, count = 0;   // frame range (global track-frame indices)
    float    fps  = 24.0f;
    bool     loop = false;
};

// One rigid bone part: a rest-pose voxel volume in creature-local space
// (x lateral, y up, z forward, origin at the feet centre).
struct CreaturePart {
    char       name[17] = {};
    int        parent = -1;          // part index (skeleton; used by PR 2's IK)
    glm::vec3  gridOrigin{0};        // creature-local corner of cell (0,0,0), rest pose
    glm::vec3  pivot{0};             // joint the part rotates about (bone head, rest pose)
    glm::ivec3 dims{0};              // cells (1 cell = cellScale world voxels)
    uint32_t   wordBase = 0;         // into vox[] (file-local; GPU-absolute after atlas build)
};

// A baked rigid-part asset (tools/creatures/bake_parts.py; format in
// docs/creature-animation-roadmap.md). Animation = per-frame per-part rigid
// transforms applied to the rest volumes at runtime — voxels rotate on
// screen instead of being re-sampled, so animation cannot flicker.
struct CreatureModel {
    uint32_t   palette[256] = {};    // RGBA8 (indices are 1-based)
    std::vector<CreaturePart> parts;
    std::vector<CreatureClip> clips;
    float      cellScale = 1.0f;     // world voxels per cell (player: 0.5625;
                                     // wildlife buckets: 1, 0.5, 0.25)
    uint32_t   totalFrames = 0;
    std::vector<float>    tracks;    // totalFrames x parts x 7 (quat xyzw, pos xyz)
    std::vector<uint32_t> vox;       // part volumes, 4 voxels/word, x-fastest
    float      boundRadius = 0.0f;   // creature-local sphere over all rest parts
    uint32_t   palBase = 0;          // GPU atlas word offset (stamped at atlas build)
    int clipIndex(const char* name) const;
};

// One posed part for the renderer: cell-space -> world affine + inverse and
// volume addressing. Matches the std430 record in lib/creature_trace.glsl.
struct CreaturePartInstance {
    glm::vec4 rowX, rowY, rowZ;      // world = rows . (cell, 1)
    glm::vec4 invX, invY, invZ;      // cell  = inv  . (world, 1)
    glm::vec4 prevX, prevY, prevZ;   // LAST FRAME's cell -> world (motion vectors)
    glm::vec4 dimsPal;               // xyz part dims, w palette word base
    glm::vec4 meta;                  // x vox word base, y 1 = terrain-material
                                     // volume (forestry falls), zw unused
    glm::vec4 anchor;                // terrain-material mode: xyz = the part's
                                     // PRE-FALL world lo corner (painterly
                                     // detail anchors to anchor + cell so the
                                     // pattern rides the moving model).
                                     // Appended LAST; the std430 mirror in
                                     // lib/creature_trace.glsl must match.
};

// Per-species tuning from assets/creatures/<name>.def. Distances in world
// voxels (10 cm), speeds in voxels/second.
struct CreatureDef {
    std::string   name;
    CreatureModel model;
    float walkSpeed = 3.2f,  walkRef = 3.2f;     // *Ref: zero-foot-skate speed
    float gallopSpeed = 42.0f, gallopRef = 42.0f;
    float turnSpeed = 2.5f;                       // rad/s yaw slew
    float alertDist = 90.0f, fleeDist = 45.0f, calmDist = 130.0f;
    float wanderR = 80.0f;
    float grazeTime = 14.0f, idleTime = 6.0f;     // mean calm-state seconds
    int   spawnForest = 1;
    float herdChance = 0.18f;                     // per-lattice-cell herd roll
    // Whole-number knobs stored as float so the /tune panel's slider pattern
    // (raw float pointer per row) edits them directly; use sites round.
    float groupMin = 2, groupMax = 4;
    float populationTarget = 8;
    float populationRadius = 500.0f, despawnRadius = 700.0f;
    float forestMin = 0.55f;   // min forestDensityAt to qualify as habitat
    float collideHalf = 3.0f, collideHeight = 14.0f;  // nav box (not render box)
    float bodyLen = 0.0f;      // world voxels, derived at load from the rest-
                               // pose part extents (cells x cellScale): the
                               // oriented 3-box body test spans it
    // Resolved clip indices (looked up once at load; -1 if missing). The
    // baked WalkToGallop/GallopToWalk transitions are loaded but unused —
    // runtime crossfading replaced them (kept for tooling QA).
    int clipIdle = -1, clipWalk = -1, clipGallop = -1, clipGraze = -1,
        clipAlert = -1, clipWalkToGallop = -1, clipGallopToWalk = -1;
    // Resolved part indices for the procedural layer (-1 if absent).
    int partHead = -1, partNeck = -1;
    std::vector<int> hoofParts;   // "*.lower" leg parts (ground-snap targets)
};

// ---- /tune creatures page -------------------------------------------------
// One row per live-tunable CreatureDef knob. `key` doubles as the .def file
// key (/tune save patches the values back into assets/creatures/<name>.def).
// `snap` = whole-number knob; `respawn` = editing re-derives the herd lattice
// so the change shows up in the world within a second.
struct CreatureTuneRow {
    const char* label;   // panel row label (14 chars fit the column)
    const char* key;     // .def key
    float mn, mx;
    bool  snap, respawn;
};
constexpr CreatureTuneRow CREATURE_TUNE[] = {
    { "herd.chance",  "herd_chance",         0.0f,    1.0f,  false, true  },
    { "group.min",    "group_min",           1.0f,    8.0f,  true,  true  },
    { "group.max",    "group_max",           1.0f,   12.0f,  true,  true  },
    { "pop.target",   "population_target",   0.0f,   40.0f,  true,  true  },
    { "pop.radius",   "population_radius", 150.0f,  900.0f,  false, true  },
    { "despawn.r",    "despawn_radius",    200.0f, 1200.0f,  false, false },
    { "forest.min",   "forest_min",          0.0f,    0.95f, false, true  },
    { "walk.speed",   "walk_speed",          0.5f,   20.0f,  false, false },
    { "gallop.speed", "gallop_speed",        5.0f,   90.0f,  false, false },
    { "alert.dist",   "alert_dist",         20.0f,  200.0f,  false, false },
    { "flee.dist",    "flee_dist",          10.0f,  120.0f,  false, false },
    { "wander.r",     "wander_r",           10.0f,  200.0f,  false, false },
};
constexpr int CREATURE_TUNE_N = int(sizeof(CREATURE_TUNE) / sizeof(CREATURE_TUNE[0]));

inline float* creatureTuneField(CreatureDef& d, int row) {
    switch (row) {
        case 0:  return &d.herdChance;
        case 1:  return &d.groupMin;
        case 2:  return &d.groupMax;
        case 3:  return &d.populationTarget;
        case 4:  return &d.populationRadius;
        case 5:  return &d.despawnRadius;
        case 6:  return &d.forestMin;
        case 7:  return &d.walkSpeed;
        case 8:  return &d.gallopSpeed;
        case 9:  return &d.alertDist;
        case 10: return &d.fleeDist;
        case 11: return &d.wanderR;
    }
    return nullptr;
}

enum CreatureState : uint8_t {
    CRE_IDLE, CRE_GRAZE, CRE_WANDER, CRE_ALERT, CRE_FLEE,
};

struct Creature {
    uint8_t   type = 0;              // index into CreatureSystem::defs
    CreatureState st = CRE_IDLE;
    glm::vec3 pos{0};                // feet centre, world voxels (float)
    float     velY = 0.0f;
    float     speed = 0.0f;          // current horizontal speed (smoothed)
    float     yaw = 0.0f, targetYaw = 0.0f;
    glm::vec2 moveTarget{0};         // XZ destination (wander / flee)
    float     stateT = 0.0f;         // seconds left in the current calm state
    float     fleeRepickT = 0.0f;
    int       clip = 0;
    float     clipT = 0.0f;          // seconds into the clip (at 1x rate)
    // Runtime crossfade: any clip switch blends FROM the pose snapshot below
    // (whatever was actually on screen — so a switch mid-fade cannot pop)
    // over CRE_FADE seconds. Replaces the baked gait-transition clips.
    float     fadeT = 1e9f;          // seconds since the switch (>= fade = done)
    std::vector<float> fadePose;     // last displayed base pose (7 floats/part)
    // Last frame's world rows per part (12 floats each) — the motion pass
    // reprojects creature pixels through these instead of the camera-only
    // path, killing TAA ghosting on movers. Identity-keyed per creature, so
    // visibility-order changes between frames can't mismatch parts.
    std::vector<float> prevXf;
    // Pose-rate LOD: far creatures re-pose every 3rd frame from this cache.
    // lodCenter freezes the shadow-bound centre with the cached geometry
    // (the sim keeps ticking pos while the pose is frozen).
    std::vector<CreaturePartInstance> lodCache;
    glm::vec3 lodCenter{0};
    uint32_t  lastPoseFrame = 0;
    // Frame this creature last made it into the instance buffer at all —
    // prevXf is only "last frame's rows" when builds were consecutive.
    uint32_t  lastBuiltFrame = 0;
    // Procedural layer state (smoothed per frame in tickCreature).
    float     lookW = 0.0f;          // alert head look-at weight 0..1
    float     pitch = 0.0f;          // terrain-follow body pitch (radians)
    float     yawRate = 0.0f;        // smoothed turn rate (lean into turns)
    int64_t   cell = 0;              // spawn-lattice cell key (herd identity)
    bool      onGround = true;
    float     stuckT = 0.0f;         // seconds of commanded-but-blocked motion
};

// Per-creature summary for the lighting pass (shadow-loop bounds + blob).
struct CreatureBound {
    glm::vec3 center;   // world
    float     radius;
    int       firstPart, partCount;   // range in the part-instance list
};

// Load a .cvx2 asset directly (the player avatar loads outside the wildlife
// def system — it must never spawn, tune, or census as a creature).
bool loadCreatureModel(const char* path, CreatureModel& m);
// Append one model's palette + volumes to a GPU atlas blob, stamping the
// absolute offsets into the model (same layout CreatureSystem::buildGpuAtlas
// produces per species).
void appendModelToAtlas(CreatureModel& m, std::vector<uint32_t>& blob);
// Pose ONE standalone model (the player avatar): pure clip playback with the
// same crossfade-from-snapshot the creatures use — no procedural layer — and
// append its part instances. Returns the shadow bound.
CreatureBound appendModelInstances(const CreatureModel& m,
                                   glm::vec3 pos, float yaw,
                                   int clip, float clipT, float fadeT,
                                   std::vector<float>& fadePose,
                                   std::vector<float>& prevXf,
                                   std::vector<CreaturePartInstance>& out);

struct CreatureSystem {
    std::vector<CreatureDef> defs;
    // As-loaded values of every CREATURE_TUNE row per species — the /tune
    // panel's R-reset targets (the .def file's numbers, not code defaults).
    std::vector<std::vector<float>> tuneDefaults;
    std::vector<Creature>    all;
    std::unordered_set<int64_t> activeCells;   // lattice cells with a live herd
    double   scanT = 0.0;                      // spawn-scan throttle
    uint32_t rng = 0x9E3779B9u;                // behaviour rng (not determinism-bearing)
    bool     atlasBuilt = false;               // buildGpuAtlas stamps offsets once
    uint32_t poseFrame = 0;                    // buildPartInstances call counter (LOD)

    // Herd spawn lattice pitch, world voxels (~19 m: a few cells per forest).
    static constexpr int CELL = 192;

    // Scan assets/creatures/*.def (+ their .cvx2). Returns species loaded.
    int  loadAssets(const char* dir);
    // Drop every creature + active cell (window teleport / regen).
    void reset();
    // Per-frame tick: spawning, state machines, terrain-following movement.
    // viewPos: the active camera anchor (diorama orbit camera when the map
    // is open, else the player eye) — despawn keeps creatures alive near
    // EITHER anchor so watched herds never vanish on screen.
    void update(const World& w, uint64_t worldSeed, glm::vec3 playerPos,
                glm::vec3 viewPos, float dt);
    // Pose the nearest visible creatures (<= maxCreatures, culled against
    // maxDist and the diorama clip slab): samples each one's clip track and
    // appends one CreaturePartInstance per bone part to `parts`, plus a
    // per-creature bounds record to `bounds`. Returns the creature count.
    // (The World and player position feed the procedural layer: hoof ground
    // snap and the alert head look-at.)
    // (Non-const: refreshes each posed creature's crossfade pose snapshot.)
    int  buildPartInstances(std::vector<CreaturePartInstance>& parts,
                            CreatureBound* bounds, int maxCreatures,
                            const World& w, glm::vec3 playerPos,
                            glm::vec3 camPos, float maxDist,
                            glm::vec3 clipLo, glm::vec3 clipHi);
    // Concatenate every model's palette + part volumes into one word buffer
    // for the GPU atlas, stamping ABSOLUTE word offsets into the models
    // (palBase, per-part wordBase). Call once after loadAssets.
    void buildGpuAtlas(std::vector<uint32_t>& blob);
    int  countOfType(uint8_t t) const;
    // Dev-console force spawn: n creatures of the named species scattered
    // just outside camDist of pos. Returns how many actually placed.
    int  debugSpawn(const World& w, const char* species, glm::vec3 pos, int n);
    // /tune save: patch each species' current CREATURE_TUNE values back into
    // its assets/creatures/<name>.def (comments and other keys preserved).
    // Returns how many files were rewritten.
    int  saveDefs(const char* dir) const;
};
