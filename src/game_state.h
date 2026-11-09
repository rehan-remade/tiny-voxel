#pragma once
// =============================================================================
// Global game state. Single owner; passed by reference to anyone who needs it.
//
// APP-LAYER ONLY -- includers: app.cpp and commands.cpp, nothing else (the
// build enforces the whitelist: cmake/check_game_state_includers.cmake).
// Module code keeps taking narrow parameters (World&, its own struct, PODs --
// see farm.h, creatures.h, render.h FrameView) so module headers never grow
// dependencies on each other through this aggregate. It exists for exactly
// two consumers that legitimately span every subsystem: the app glue loop
// and the console command handlers (commands.cpp).
// =============================================================================

#include "common.h"
#include "world.h"
#include "worldgen/worldgen.h"   // GenContext, ChunkStreamer
#include "player.h"
#include "particles.h"
#include "forestry.h"            // Forestry, TreeInfo (chop sessions)
#include "materials.h"           // MP_COUNT etc. (tweaks.h sizes depend on it)
#include "render.h"
#include "inventory.h"
#include "fire.h"
#include "creatures.h"
#include "map.h"
#include "farm.h"
#include "worldio.h"             // ChunkCache
#include "console.h"
#include "tweaks.h"              // TWEAK_N / MATTUNE_N array sizes

#include <glm/glm.hpp>

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

// Matches GLFW's own `typedef struct GLFWcursor GLFWcursor;` — repeating an
// identical typedef is legal, and it spares this header the GL include-order
// dance for one opaque pointer (the forestry hover hand cursor).
typedef struct GLFWcursor GLFWcursor;

// Demo director (/orbit, /hud): creator tools for recording footage. The
// orbit is an eased yaw spin of the diorama camera; framing moves (pitch/
// dist/height) are GLIDES that write the same cam.* tweak[] slots the wheel
// and the /tune sliders drive — one source of truth, so /tune save can bake
// a framing you like. Ticked in app.cpp's map-motion block (frozen while
// paused), set from commands.cpp.
struct CamGlide {
    bool  active = false;
    float from = 0.0f, to = 0.0f;   // tweak-value endpoints
    float t = 0.0f, dur = 0.0f;     // elapsed / total seconds (smoothstepped)
};
struct DemoCam {
    bool  on = false;          // yaw spin running (rate eases in/out)
    float degPerSec = 6.0f;    // signed target rate; 6 = one lap a minute
    float spinNow   = 0.0f;    // eased actual rate (deg/s)
    CamGlide pitch, dist, height;   // -> TW_CAM_PITCH / _DIST / _HEIGHT
};

// A "collect flyer": the harvested item, rendered through the creature-part
// pipeline, flying from the plot into the hotbar so you SEE it enter your
// bag. Grade sets the size; `style` picks the species' flight theatrics
// (see updateCollectFlyers in app.cpp).
enum FlyStyle {
    FLY_ARC = 0,   // pumpkin: loft + swoop (the classic)
    FLY_YANK,      // carrot/radish: starts BURIED, pulled out with a dirt
                   // burst, backflips at the apex, then swoops to the player
    FLY_POP,       // tomato: each fruit pops off the bush, staggered
    FLY_ROLL,      // cabbage: bounces & rolls along the ground, then leaps in
    FLY_TUMBLE,    // corn: ears tumble end-over-end on a flat fast arc
};
struct CollectFlyer {
    glm::vec3 start{0.0f};   // world spawn (fruit centre / crop root)
    float     t   = 0.0f;    // 0..1 flight progress (after delay)
    float     dur = 0.72f;   // seconds
    float     delay = 0.0f;  // stagger before the flight starts (multi-fruit)
    float     delay0 = 0.0f; // initial delay (drives the waiting fruit's
                             // drop-to-the-soil settle; the plant is gone)
    ItemId    item = 0;
    uint16_t  data = 0;      // grade
    int       grade = 0;
    int       species = 0;   // CropSpecies — picks the baked produce model
    int       style = FLY_ARC;
    uint32_t  rng = 0;       // per-flyer variation (spin direction/rate)
    float     groundY = 0.0f;// soil surface (yank emerge / roll bounces)
    bool      burst = false; // one-shot dirt/leaf burst already fired
};

struct GameState {
    World          world;
    GenContext     gen;       // worldgen seed + derived registries/maps
    Player         player;
    ParticleSystem particles;
    Renderer       renderer;
    Hotbar         hotbar;
    FireSim        fireSim;   // transient fire automaton state (cast spell)
    CreatureSystem creatures; // wildlife (transient; herds re-derive from seed)
    Forestry       forestry;  // tree chopping: falls + debris items (forestry.h)
    std::vector<CreaturePartInstance> creatureParts; // per-frame posed parts (fv points here)

    // Player voxel avatar (creature roadmap PR 3): the rigged 9-clip
    // humanoid rendered through the same part pipeline — replaces the
    // analytic box character in the diorama (and casts real shadows).
    CreatureModel playerModel;
    bool  playerAvatar = false;
    int   pClip = -1, pcIdle = -1, pcWalk = -1, pcSprint = -1,
          pcJump = -1, pcFall = -1, pcChop = -1, pcMine = -1,
          pcHarvest = -1, pcCrouchIdle = -1, pcCrouchWalk = -1;
    float pClipT = 0.0f, pFadeT = 1e9f;
    std::vector<float> pFadePose;
    std::vector<float> pPrevXf;   // last frame's part rows (motion vectors)
    int   playerHandPart = -1;    // "armR.lower" part index (axe attachment)

    // Creature/prop GPU atlas blobs. atlasBase = the startup upload
    // (wildlife + player + forestry chunk/axe models). atlasDyn = atlasBase
    // plus every LIVE falling tree's volume, grown append-only (the appender
    // restamps offsets, so a model must never pass through it twice) and
    // reset to atlasBase once no falls remain.
    std::vector<uint32_t> atlasBase, atlasDyn;

    // Forestry context menu (RuneScape "Choose Option") + chop session.
    bool       ctxOpen = false;
    glm::vec2  ctxPos{0.0f};      // framebuffer coords, top-left of the panel
    int        ctxHover = -1;
    int        ctxKind = 0;       // 0 tree (chop), 1 rock (mine), 2 crop (harvest)
    TreeInfo   ctxTree;
    RockInfo   ctxRock;
    glm::ivec2 ctxCell{0};        // ripe-crop plot origin (ctxKind==2)
    // Hover highlight (tickTreeHover): the tree under the cursor, identified
    // on the forestry worker thread and rendered as a warm rim by the
    // lighting pass. hovMask mirrors the GPU 3D texture for containment
    // tests; hovStrength eases in/out.
    bool       hovActive = false;
    glm::ivec3 hovLo{0}, hovDims{0};
    std::vector<uint8_t> hovMask;
    float      hovStrength = 0.0f;
    bool       hovHandWant = false;  // tree/menu wants the pointing hand
    glm::ivec3 hovPickVox{INT32_MIN, 0, 0};   // last pick that drove a request
    // OS pointing-hand cursor: ONE owner applies it (main loop, after
    // tickTreeHover) from hovHandWant + the god-harvest hoverRipe, change-
    // triggered — two independent setters used to stomp each other.
    GLFWcursor* handCursor = nullptr;
    bool        handShown  = false;  // hand currently applied to the window
    int        chopPhase = 0;     // 0 none, 1 walking to target, 2 swinging
    int        chopKind = 0;      // 0 tree (chop), 1 rock (mine), 2 crop (harvest)
    TreeInfo   chopTree;
    RockInfo   chopRock;
    glm::ivec2 chopCell{0};       // session crop plot origin (chopKind==2)
    // Per-16-sector XZ surface reach of the session rock (mining only): the
    // stand ring, the arrival gate, and the strike re-aim all measure from
    // the rock FACE on the approach side, not the global max radius — one
    // scalar parks the player a metre off the narrow side of a rotated
    // squashed ellipsoid ("we mine quite far away from the rock").
    float      mineSectorR[16] = {};
    glm::ivec3 chopTargetVox{0};  // swing target (dust + liveness checks)
    glm::vec3  chopStand{0.0f};   // standable tile ringing the trunk
    int        swingsDone = 0, swingsNeed = 0;
    float      chopSwingT = 0.0f; // drives the Chop clip + the impact beat
    int        chopReplans = 0;
    // LMB click-vs-drag: a press arms a click; >4 px displacement from the
    // press anchor turns it into an orbit drag (no time limit — slow clicks
    // must stay clicks).
    double     pressX = 0, pressY = 0;
    bool       clickArmed = false;
    // Forestry -> satchel bridge: collected[] totals already banked into the
    // hotbar, per material (the drain gives only the per-frame delta).
    int        forestryDrained[256] = {};

    // Input.
    double lastMouseX = 0, lastMouseY = 0;
    bool   firstMouse = true;
    bool   leftDown = false, rightDown = false;

    // Gameplay.
    // Boot gate: false until world generation finishes. The loading screen
    // pumps glfwPollEvents while the gen thread writes the world, and the
    // input callbacks are already installed — every world-touching input
    // verb must sit behind this flag or a boot-screen click races the
    // generator. (The old mapView.open check did this job by accident.)
    bool   worldReady  = false;
    bool   devMode     = true;   // god mode (instant mine, infinite items) for fast iteration
    float  slopeComp   = 1.0f;   // cos(slope): scales horiz speed so along-slope speed is constant
    float  crouchT     = 0.0f;   // smoothed crouch amount 0..1 (drives height + eye)
    bool   paused      = false;
    glm::vec3 spawnPos{0.0f};    // initial spawn â€” /tp spawn warps the player back here
    bool  showGpuTimers = false; // dev: per-pass GPU timing overlay (/gpu)
    bool  taaEnabled = true;     // temporal anti-aliasing (/taa), on by default
    bool  ringsOn    = true;     // far-field LOD ring 1 (/rings), on by default
    int   colorMode  = 0;        // /srgb: 0 legacy output curve, 1 sRGB experiment
    int   slopeShade = 0;        // /slopeshade: macro terrain slope diffuse normals (parked; opt-in)
    int   fogMode    = 0;        // /fog: 0 off (DEFAULT), 1 auto (dawn/dusk rhythm), 2 heavy (test)
    // /tune: live visual-tuning panel. Two pages — lighting (tweak[]/uTweak)
    // and materials (matTune[]/uMatTune, per-material cell params).
    float tweak[TWEAK_N] = {};
    float matTune[MATTUNE_N] = {};
    bool  tuneOpen = false;
    int   tunePage = 0;          // 0 = lighting, 1 = materials, 2 = creatures
    int   tuneSel  = 0;          // selected row within the page
    int   tuneMatIdx = 0;        // selected material (index into MAT_LIST)
    int   tuneSpeciesIdx = 0;    // selected species (index into creatures.defs)
    double creatureRetuneAt = 0; // debounced herd re-derive after spawn-knob edits
    bool  tuneDrag = false;      // LMB held on a slider bar
    // -- World clock (/time) --------------------------------------------------
    // timeOfDay in [0,1): 0 midnight, .25 sunrise, .5 noon, .75 sunset. It
    // advances every frame at timeSpeed x the base 20-minute day (main-loop
    // "Day / night cycle" block); 0 pauses. The old dev-only T-key 60x
    // fast-forward is now "/time speed 60".
    float timeOfDay = 0.28f;     // start just after sunrise
    float timeSpeed = 1.0f;      // day-cycle multiplier (/time speed 0..1000)
    // BASE in-game seconds for a crop to fully ripen (all FARM_STAGE_MAX
    // stages; each species scales this by its CROP_DEFS growScale). Growth
    // advances in in-game time (dt x timeSpeed), so at x1 this is real
    // seconds; /croptime <days> sets it to days x DAY_LENGTH. Session-only
    // (like timeSpeed, not persisted). Default = the old fixed 45s pacing.
    float cropGrowSecs = FARM_STAGE_SECS * FARM_STAGE_MAX;
    Console console;             // dev console (slash opens, Esc closes; dev mode only)

    // Explored-world map (see map.h). The masks are the GPU-ready
    // explored bits for a MASK_CH^2-chunk square that follows the map's pan
    // target (mapMaskOrigin, recentred with hysteresis), rebuilt when the
    // marks or the origin change; the stamp keys the renderer's lazy
    // re-upload.
    MapExplored mapExplored;
    MapView     mapView;
    MapMasks    mapMasks;
    uint64_t    mapMasksVersion = ~0ull;  // MapExplored::version last built
    glm::ivec2  mapMaskOrigin{INT32_MIN, INT32_MIN};   // desired mask origin
    glm::ivec2  mapMasksOrigin{INT32_MIN, INT32_MIN};  // origin last built
    uint64_t    mapMasksStamp = 0;
    // Minimap scale (world voxels per pixel). 3.0 puts the orbit camera's
    // default ground footprint INSIDE the disc, so the view cone reads.
    float       minimapZoom = 3.0f;

    // Demo director (/orbit, /hud): recording tools. hudHidden drops the
    // PASSIVE chrome only (minimap, hotbar tray, DEV badge) — UI you
    // summon on purpose (console, context menu, farm bar, satchel) stays,
    // and minimapHit treats the hidden minimap as not there.
    DemoCam demoCam;
    bool    hudHidden = false;

    // Click-to-move (the diorama IS the game): the right button picks a
    // ground destination through the map camera; the player steers there
    // with the normal walking physics. walkAnimT is the character's stride
    // phase (advances with actual horizontal speed).
    glm::vec3 moveTarget{0.0f};    // where the ROUTE ends (stud marker)
    glm::vec3 movePlanDest{0.0f};  // what was CLICKED (fresh-compare + replans)
    bool      moveActive = false;
    float     moveStallT = 0.0f;   // seconds of no progress -> replan/give up
    std::vector<glm::vec3> movePath;  // A* route (feet waypoints, target last)
    int       moveWp = 0;             // waypoint being steered toward
    float     repathT = 0.0f;         // hold-steer replan throttle (s)
    int       repathTries = 0;        // stall replans on this route so far
    float     walkAnimT  = 0.0f;

    // Farming (Farm Mode, G): toolbar + crops + the pending walk-to-plot job.
    FarmState farm;

    // Inventory satchel (E): a bag panel that floats over the dimmed
    // diorama. Row 0 of the bag is the hotbar. invHover/invTidyHot are the
    // per-frame cursor pick for the panel's highlight + footer.
    bool invOpen     = false;
    int  invHover    = -1;
    bool invTidyHot  = false;

    // Held-item name label: shows briefly above the hotbar when the selected
    // slot (or its contents) changes, then fades.
    int   hotbarPrevSel  = -1;
    int   hotbarPrevItem = -1;
    float hotbarNameT    = 0.0f;

    // Harvest collection: a pickup message ("+1 Giant Pumpkin") that overrides
    // the hotbar name label briefly.
    char       pickupMsg[32] = {};
    float      pickupMsgT     = 0.0f;

    // God-harvest hover affordance: the ripe crop under the cursor lights up
    // (its voxel bbox, gated on crop materials in the lighting pass) + a warm
    // "divine touch" ground ring blooms on pick.
    bool       hoverRipe = false;
    glm::ivec3 hoverLo{0}, hoverHi{0};
    float      ringT   = 0.0f;      // press-ring life 1->0
    glm::vec3  ringPos{0.0f};       // ring centre (cell centre, ground)

    // Harvest: baked produce volumes (in the creature atlas), ONE per species
    // per grade (0 small / 1 medium / 2 giant) so harvested produce flies as
    // its OWN size/shape — a yanked carrot reveals the whole buried root, a
    // corn ear is a husked cob — instead of a scaled generic blob. Plus the
    // live "collect flyers" that carry harvested produce into the bag.
    CreatureModel            produceModels[CROP_SPECIES_N][3];  // [species][grade]
    std::vector<CollectFlyer> collectFlyers;

    // Chunk streaming (6e-2/6e-3b): session-scoped cache of evicted chunks,
    // the chunk-status-ladder scheduler, and the master toggle (/stream).
    ChunkCache chunkCache;
    ChunkStreamer* streamer = nullptr;   // owned by main()
    bool streamOn = true;
    std::vector<int64_t> gpuQueue;       // chunks awaiting uploadChunk (budgeted)

    // Far-field LOD ring recentre: when the window drifts from the ring's
    // build centre, the ring is rebuilt on this background thread (pure
    // worldgen, CPU staging only) and uploaded on completion; the GPU keeps
    // rendering the old ring until the swap (World::ringOriginGPU).
    std::thread       ringRebuild;
    std::atomic<bool> ringRebuildDone{false};
    bool              ringRebuilding = false;
    uint32_t          ringRebuildMask = 0;   // rings the in-flight rebuild covers
    std::string editsPath;               // region save file (set in main)

    // Window.
    int   savedX = 0, savedY = 0, savedW = 1280, savedH = 800;
    bool  isFullscreen = false;
    bool  wasMaximized = false;   // OS-maximized before F11; re-applied on exit
};
