#pragma once
// Farming vertical slice (Farm Mode, toggled with G in the diorama).
//
// The loop: TILL a plot -> PLANT a seed -> watch it GROW through discrete
// stages -> HARVEST the ripe crop. A click issues a JOB: the character walks
// to the plot (reusing the click-to-move A* route) and performs the tool
// action on arrival. Growth is time-accelerated so the whole loop is
// playable in under a minute — this slice exists to answer "is it fun?",
// not to be balanced or persistent.
//
// Deliberately NOT in this slice: persistence (crops vanish when the resident
// window scrolls or the game reloads — every world write goes through
// setVoxelSim, so nothing touches the edit-save), watering/moisture, and
// terrain leveling (till only accepts flat-enough ground).
//
// Scale: 1 voxel = 10cm. A plot CELL is FARM_CELL voxels square (~0.8m), snapped
// to a world grid so adjacent tills form a continuous field. Crops root at the
// cell centre and stay within the cell footprint.
#include <climits>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <glm/glm.hpp>
#include "common.h"

struct World;
struct EditDelta;

enum FarmTool { FT_TILL = 0, FT_PLANT, FT_HARVEST, FT_COUNT };

// Crop species: what a plot can grow. Each species has its own stage shapes
// (cropVoxels), growth pacing, flesh material, and harvest theatrics (the
// collect-flyer style lives app-side; see collectProduce in app.cpp).
enum CropSpecies {
    CROP_PUMPKIN = 0,
    CROP_CARROT,      // root: buried, yanked out of the soil on harvest
    CROP_TOMATO,      // bush: several fruits, plucked one by one
    CROP_CABBAGE,     // head: rolls off the plot when cut
    CROP_CORN,        // stalk: ears pop off, stalk bursts into confetti
    CROP_RADISH,      // root: fast grower, small cheeky yank
    CROP_SPECIES_N
};

struct CropSpeciesDef {
    const char* name;       // display name ("Carrot")
    uint8_t     flesh;      // MAT_* of the produce body (dust + fruit voxels)
    float       growScale;  // per-species ripen-time multiplier (radish < corn)
    bool        root;       // true = grows buried (two-tug harvest, dirt burst)
};
extern const CropSpeciesDef CROP_DEFS[CROP_SPECIES_N];

// One planted crop. Its voxels are a pure function of (species, genome, stage):
// the generator (cropVoxels) is deterministic, so shape is stable across
// regrows and every plant from the same tile-seed looks individual.
struct Crop {
    glm::ivec3 root{0};      // farmland voxel the plant sits ON (voxels grow at root.y+1+)
    uint8_t    species = 0;  // CropSpecies — what was planted here
    uint32_t   genome = 0;   // per-plant RNG seed -> vine arms, fruit shape
    float      age   = 0.0f; // seconds since planted
    int        stage = 0;    // 0 sprout, 1 vine, 2 flowering, 3 mature
    // Ripe size tier (0 small, 1 medium, 2 giant), from a genome roll BIASED by
    // free room around the plant (see cropFreeNeighbours): packed fields keep
    // the baseline variety, spacing skews toward giants (each fills its own
    // plot, staying a distinct fruit). Re-evaluated at every stage change, so
    // planting/harvesting a neighbour can stunt or fatten it.
    int        size = 1;
    // Free-neighbour bitmask (8 compass plots) frozen at plant time — the vines
    // lean toward these open directions as they grow (telegraphs the room).
    uint8_t    freeDirs = 0;
    // Voxels this crop currently owns (for exact clear on regrow / harvest,
    // so neighbours are never nuked). Rebuilt every stage change.
    std::vector<glm::ivec3> occupied;
};

struct FarmState {
    bool     mode = false;         // Farm Mode active (G)
    int      tool = FT_TILL;       // selected tool (1/2/3 or toolbar click)
    int      seedSel = CROP_PUMPKIN;  // species the Plant tool sows (seed bar)
    std::vector<Crop> crops;

    // Hold-and-drag painting: RMB held sweeps the active tool across cells,
    // acting once per cell entered (lastActedCell dedups the drag).
    bool       painting = false;
    glm::ivec2 lastActedCell{INT_MIN, INT_MIN};

    // Live cursor footprint (updated each frame for the preview overlay).
    bool       cursorValid = false;   // is there a hit under the cursor at all
    bool       cursorOk    = false;    // is the current tool valid on that cell
    glm::ivec2 cursorCell{0};
    int        cursorY = 0;
};

// Plot geometry. A cell is FARM_CELL^2 voxels, snapped to the FARM_CELL grid.
constexpr int FARM_CELL = 8;
inline glm::ivec2 farmCellOrigin(int wx, int wz) {
    auto snap = [](int v) { return (v >= 0 ? v / FARM_CELL : (v - FARM_CELL + 1) / FARM_CELL) * FARM_CELL; };
    return glm::ivec2(snap(wx), snap(wz));
}

// Growth pacing (accelerated for the feel test): seconds per stage.
constexpr float FARM_STAGE_SECS = 15.0f;
constexpr int   FARM_STAGE_MAX  = 3;

// ---- Tool actions. Each mutates the world through `d` (setVoxelSim, so no
// edit-save bloat); the caller does ONE uploadDelta. All read/write only the
// resident window and no-op safely off-window. Return true if anything changed.

// TILL: LEVEL the plot to a single bed height (fill the low columns with soil,
// shave the high ones) and cap it with MAT_FARMLAND — so tilling a slope carves
// a flat terrace. Refuses only a cliff/wall or a tree/crop over the cell.
bool farmTill(World& w, FarmState& fs, glm::ivec2 cellOrigin, int y, EditDelta& d);

// PLANT: seed a crop of the SELECTED species (fs.seedSel) at the cell centre
// if it is tilled farmland and empty.
bool farmPlant(World& w, FarmState& fs, glm::ivec2 cellOrigin, int y, EditDelta& d);

// Species of the crop rooted in this cell (CropSpecies), or -1 if none.
int farmCropSpeciesAt(const FarmState& fs, glm::ivec2 cellOrigin);

// Is there a RIPE (fully grown) crop rooted in this cell? (cheap; for the
// pluck cursor test — only ripe crops are harvestable.)
bool farmHasRipeCropAt(const FarmState& fs, glm::ivec2 cellOrigin);

// The voxel bounding box (lo..hi inclusive) of the ripe crop rooted in this
// cell — for the hover highlight. False if no ripe crop there.
bool farmRipeCropBounds(const FarmState& fs, glm::ivec2 cellOrigin,
                        glm::ivec3& lo, glm::ivec3& hi);

// HARVEST: pluck the RIPE crop rooted in this cell — removes its voxels and
// copies the plucked Crop to `out` (species/size/root/genome intact, so the
// caller can regenerate its fruit positions via cropVoxels for the pickup
// theatrics). No-op (false) if the cell has no crop or the crop isn't ripe
// yet. The caller awards the produce item(s).
bool farmHarvest(World& w, FarmState& fs, glm::ivec2 cellOrigin,
                 Crop& out, EditDelta& d);

// Validity query for the cursor preview (green/red) — cheap, world reads only.
// For FT_TILL, *outTargetY (when non-null) receives the level the bed will
// flatten to, so the preview slab can sit at bed height.
bool farmToolValid(const World& w, const FarmState& fs, int tool,
                   glm::ivec2 cellOrigin, int y, int* outTargetY = nullptr);

// Per-frame growth: advance every crop's age by `dt` (IN-GAME seconds, already
// scaled by /time speed) and re-mesh on a stage change. `stageSecs` = BASE
// in-game seconds per growth stage (runtime, set via /croptime); each species
// scales it by its CROP_DEFS growScale. One batched delta.
void farmTick(World& w, FarmState& fs, float dt, float stageSecs, EditDelta& d);

// Deterministic crop-shape generator: append (voxel, material) for the crop's
// CURRENT stage. Pure function of (root, species, genome, stage, size).
void cropVoxels(const Crop& c, std::vector<std::pair<glm::ivec3, uint8_t>>& out);

// ---- Persistence. FarmState is the source of truth for crops (their voxels are
// transient and re-stamped by farmTick when resident). The sidecar is seed-keyed
// like map.dat. `nowUnix` is wall-clock seconds: save stamps it; load fast-
// forwards every crop's age by (now - saved) so crops GROW WHILE AWAY (Stardew-
// style). Load leaves occupied empty — farmTick re-stamps on the next resident
// tick. Both return false on I/O error or seed mismatch (farm left untouched).
bool saveFarmState(const FarmState& fs, const std::string& path,
                   uint64_t worldSeed, int64_t nowUnix);
bool loadFarmState(FarmState& fs, const std::string& path,
                   uint64_t worldSeed, int64_t nowUnix);
