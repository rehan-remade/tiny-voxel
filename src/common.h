#pragma once
#include <cstdint>
#include <glm/glm.hpp>

// Voxel world layout. Two-level brickmap: BRICKS^3 macro cells, each of BRICK^3
// voxels, giving an effective grid of GRID^3 voxels.
constexpr int BRICK_BITS = 3;
constexpr int BRICK      = 1 << BRICK_BITS;      // 8
constexpr int BRICK_MASK = BRICK - 1;
constexpr int BRICK_VOX  = BRICK * BRICK * BRICK; // 512 voxels per brick
constexpr int BRICK_U32  = BRICK_VOX / 4;         // 128 uints per brick (4 voxels packed per uint)

// Non-cubic world: X/Z bounded by BRICKS_XZ * BRICK, Y by BRICKS_Y * BRICK.
// Deeper-than-wide so geological descent has room. At 10cm/voxel: XZ=2048
// (~205m window -> ~102m view radius from the centred player), Y=4096 (~410m).
// BRICKS_XZ MUST stay a power of two — toroidal '& (mask-1)' wrapping is used
// everywhere. Raising it widens the resident window => further render distance
// (the haze fade + reveal curtain key off GRID_XZ/2, so they follow it). Still
// blocked on: (1) ~86s full-window regen at 4096 (needs streamed startup) and
// (2) GPU-upload THROUGHPUT — at 4096 the budgeted/LIFO upload can't keep
// revealed chunks current, so the curtain opens on stale recycled bricks (the
// "jumbled trees" desync); needs reveal-gated-on-upload + FIFO + scaled budget.
constexpr int BRICKS_XZ  = 256;
constexpr int BRICKS_Y   = 512;
constexpr int GRID_XZ    = BRICKS_XZ * BRICK;     // 2048
constexpr int GRID_Y     = BRICKS_Y  * BRICK;     // 4096

// CPU brickpool reserve in BRICKS. Sized to the WINDOW so the brickpool vector
// NEVER reallocates mid-generation/streaming: a realloc moves the backing store
// while async gen workers AND the GPU pool upload read it, corrupting bricks
// into the "black terrain / bedrock cubes" GPU desync. Scales with window AREA
// (~8M bricks per 2048^2 window covers the densest cavern regions), with an 8M
// floor. At GRID_XZ=4096 this is ~32M (the 4x window genuinely needs ~20M+).
inline size_t poolReserveBricks() {
    double r = double(GRID_XZ) / 2048.0;
    size_t scaled = size_t(8000000.0 * r * r);
    return scaled > 8000000u ? scaled : size_t(8000000u);
}

// Far-field horizon heightfield: the SILHOUETTE layer beyond the last LOD
// ring (past RING_SWITCH_T[4] = 1.23 km), generated from heightAt() ONLY (no
// caves/flora) and raymarched in the lighting MISS branch. FAR_TEX texels at
// FAR_STRIDE voxels each => FAR_EXTENT voxels across (radius FAR_EXTENT/2).
// 1024 * 64 = 65536 vox = ~6.5 km across (~3.3 km radius) at 10cm/voxel.
// CLAMP-wrapped + recentred on the window at build/teleport time. It still
// marches from near the camera (not from the ring boundary) so it also
// self-heals ring warm-up gaps after a teleport.
constexpr int FAR_TEX    = 1024;
constexpr int FAR_STRIDE = 64;
constexpr int FAR_EXTENT = FAR_TEX * FAR_STRIDE;  // 65536 voxels

// Chunk columns: the unit of generation windows, edit-dirty tracking, and
// per-chunk persistence. Full height, brick-aligned in XZ.
constexpr int CHUNK_VOX    = 64;                  // voxels per chunk side (8 bricks)
constexpr int CHUNKS_XZ    = GRID_XZ / CHUNK_VOX; // 32 x 32 chunks resident
constexpr int CHUNK_BITS   = 6;                   // log2(CHUNK_VOX)
constexpr int CHUNK_BRICKS = CHUNK_VOX / BRICK;   // 8
constexpr int CHUNK_BRICK_BITS = 3;               // log2(CHUNK_BRICKS)

// World XZ coordinates are UNBOUNDED ints; the resident window covers
// GRID_XZ^2 voxels of them. Chunk coords are floor(x / CHUNK_VOX) — the
// arithmetic shift keeps that exact for negatives.
inline int     chunkOfVoxel(int v)        { return v >> CHUNK_BITS; }
inline int64_t chunkKey(int ccx, int ccz) {
    return (int64_t(ccx) << 32) | int64_t(uint32_t(ccz));
}
inline int chunkKeyX(int64_t k) { return int(k >> 32); }
inline int chunkKeyZ(int64_t k) { return int(int32_t(uint32_t(k & 0xFFFFFFFFull))); }

// Floor division for lattice math over UNBOUNDED coordinates: cell indices
// of negative coords must round toward -inf, not toward zero.
inline int floorDiv(int a, int b) {
    return (a >= 0) ? a / b : -((-a + b - 1) / b);
}

// Coarse occupancy mip: one flag per COARSE^3 block of bricks (a 32-voxel cell).
// Conservative — set when any brick in the cell is non-empty — so long shadow/AO
// rays can skip large empty spans. Only ever gains cells (bricks are never freed).
constexpr int COARSE_BITS = 2;                        // 4 bricks per coarse cell / axis
constexpr int COARSE      = 1 << COARSE_BITS;         // 4
constexpr int CBRICKS_XZ  = BRICKS_XZ >> COARSE_BITS; // 64
constexpr int CBRICKS_Y   = BRICKS_Y  >> COARSE_BITS; // 128

// -- Far-field LOD rings (voxel clipmap C1; docs/lod-rings-architecture.md) ----
// RING_COUNT concentric, coarser brickmaps beyond the fine window. Ring r has
// voxels 2^(r+1) x the fine size (20/40/80/160 cm), the same 256^2 XZ brick
// budget => extents 410 m / 819 m / 1.6 km / 3.3 km across, and Y bricks
// halved per level (same world height). All generated DIRECTLY from the pure
// worldgen core (rings.cpp): floor-quantized terrain (a ring never pokes above
// finer data -> no rim walls at handoffs), tree impostors at the SAME lattice
// sites / RNG draws as fine trees, oasis water from the shared fields. Ring
// brickmaps + coarse mips ride as TAIL regions of the fine SSBOs (offset
// uniforms); ring brick DATA lives in its OWN pool SSBO (binding 3) -- the
// fine pool sits just under the 2^32-BYTE driver wall (doc Part III) and must
// never grow. Rings are filled by a bump allocator on the gen thread only --
// never World::allocBrick -- and DEGRADE (drop impostors) on range exhaustion.
constexpr int RING_COUNT      = 4;
constexpr int RING_BRICKS_XZ  = 256;                       // per ring (= BRICKS_XZ)
constexpr int RING_GRID_XZ    = RING_BRICKS_XZ * BRICK;    // 2048 ring vox per ring
constexpr int RING_CBRICKS_XZ = RING_BRICKS_XZ >> COARSE_BITS;   // 64
constexpr int ringScaleLog2(int r)  { return r + 1; }      // 2x, 4x, 8x, 16x
constexpr int ringBricksY(int r)    { return BRICKS_Y >> ringScaleLog2(r); }
constexpr int ringBmEntries(int r)  { return RING_BRICKS_XZ * RING_BRICKS_XZ * ringBricksY(r); }
constexpr int ringCoarseEntries(int r) {
    return RING_CBRICKS_XZ * RING_CBRICKS_XZ * (ringBricksY(r) >> COARSE_BITS);
}
// Per-ring pool budgets (bricks). Rings 3/4 cover 4x/16x the tree-cell area of
// ring 1, so they get double budgets (adversarial-review amendment C2).
constexpr uint32_t RING_POOL_BRICKS[RING_COUNT] = { 500000u, 500000u, 1000000u, 1000000u };
constexpr uint32_t ringPoolBase(int r) {           // prefix sum into the ring pool SSBO
    uint32_t b = 0;
    for (int i = 0; i < r; ++i) b += RING_POOL_BRICKS[i];
    return b;
}
constexpr uint32_t RING_POOL_TOTAL = ringPoolBase(RING_COUNT);   // 3M bricks = 1.54 GB
// Primary-ray switch distances, world voxels: fine hands off to ring 0 at
// D=768 (76.8 m, inside the window's LEAD=14 scroll margin so the ray never
// reaches the streaming frontier), each ring doubles, and the horizon
// heightfield takes over past 12288 (1.23 km). Each D sits well inside its
// ring's half-extent, so valid data exists on both sides of every switch.
// gbuffer.frag mirrors these as its DS[] constants.
constexpr float RING_SWITCH_T[RING_COUNT + 1] = { 768.0f, 1536.0f, 3072.0f, 6144.0f, 12288.0f };

// Brickmap entry encoding (one uint32 per brick):
//   0                 -> fully empty, no allocation
//   0x80000000 | mat  -> fully solid of material `mat`, no allocation
//   else              -> (pool index + 1)
constexpr uint32_t BM_EMPTY      = 0u;
constexpr uint32_t BM_SOLID_FLAG = 0x80000000u;

constexpr int   SURFACE_Y = 3200;                 // approximate global surface height
// World scale, anchored to the player: the 18-voxel player is 1.8 m tall, so
// one voxel is 10 cm. All player-facing distances derive from this constant.
constexpr float VOXEL_CM  = 10.0f;

// Canonical game gravity, in voxels/s^2 — the SINGLE source of truth for every
// physical fall (player, falling sand, future rigid bodies). 400 vox/s^2 =
// 40 m/s^2 ~ 4g: deliberately exaggerated above real 9.8 m/s^2 for a snappy,
// weighty feel (tuned originally for jump arcs). Anything that falls integrates
// THIS so the world and the player obey one gravity.
constexpr float GRAVITY          = 400.0f;
constexpr float GRAVITY_TERMINAL = 320.0f;   // max fall speed (vox/s) — caps tunneling + runaway

// Material IDs.
enum : uint8_t {
    MAT_AIR       = 0,
    MAT_DIRT      = 1,
    MAT_GRASS     = 2,
    MAT_STONE     = 3,    // topstone (shallow)
    MAT_WOOD      = 4,
    MAT_LEAVES    = 5,
    MAT_WATER     = 6,
    MAT_GREYSTONE = 7,    // mid-depth stratum
    MAT_DARKSTONE = 8,    // deep stratum
    MAT_BEDROCK   = 9,    // near-bottom stratum
    MAT_GRASS_BLADE = 10, // above-ground grass tufts
    MAT_FLOWER_STEM   = 11,
    MAT_FLOWER_RED    = 12,
    MAT_FLOWER_YELLOW = 13,
    MAT_FLOWER_BLUE   = 14,
    MAT_FLOWER_WHITE  = 15,
    MAT_CRYSTAL       = 16, // luminescent glow crystal (emissive; lights caverns)
    MAT_MUSHROOM_STEM = 17, // cream stem
    MAT_MUSHROOM_CAP  = 18, // bright red cap (upper dome)
    MAT_MUSHROOM_RIM  = 19, // darker red cap underside/rim
    MAT_MUSHROOM_SPOT = 20, // white cap spots
    MAT_MUSHROOM_CAP2 = 21, // light brown flat-cap mushroom
    MAT_CAVE_MOSS     = 22, // low ground-cover moss on cave dirt patches
    MAT_VINE          = 23, // living ivy creeping on walls/ceilings (green)
    MAT_VINE_DEAD     = 24, // dead ivy (brown)
    MAT_FIRE          = 25, // active flame: strong emissive + shader flicker
    MAT_EMBER         = 26, // dying flame / coal: dull-red glow, does not spread
    MAT_CHAR          = 27, // blackened burnt wood (final residue, faint heat)
    MAT_ASH           = 28, // inert grey residue (reserved; v1 burns soft fuel to air)
    MAT_SAND          = 29, // granular: falls + slumps (physics Stage 1)
    // Desert biome (v1). MAT_SAND_STATIC is SOLID surface sand — distinct from
    // the MAT_SAND powder so the vast desert surface is never fed to the
    // granular sim / wetBricks (that active-set cost would tank FPS).
    MAT_SAND_STATIC     = 30, // static dune/sand surface skin
    MAT_SANDSTONE       = 31, // banded desert bedrock / mesa rock / outcrops
    MAT_DESERT_PAVEMENT = 32, // reg gravel plain — the default desert ground
    MAT_CACTUS_FLESH    = 33, // saguaro/cactus body
    MAT_CACTUS_BLOOM    = 34, // cactus flower accent (vivid, not emissive)
    MAT_SHRUB_SAGE      = 35, // dry desert shrub foliage (muted silver-sage)
    MAT_DESERT_WOOD     = 36, // ashy grey-brown desert stems (ocotillo/creosote)
    // Render-only sentinel: never placed in the world. The explored-world map
    // (map.h) recolors voxels the player hasn't discovered to this flat putty
    // tone — its id must stay in sync with MAP_UNKNOWN in gbuffer.frag.
    MAT_MAP_UNKNOWN     = 37,
    // Render-only: creature G-buffer hits (creature.frag). Never placed in the
    // world; albedo comes from the creature's own baked palette, this id only
    // selects shading params (matte fur) in the lighting pass.
    MAT_CREATURE        = 38,
    // Farming (near-field, player-placed — NEVER emitted into the LOD rings, so
    // these ids are free to sit alongside terrain without the <64 ring-AO cap
    // mattering; kept low anyway for tidiness). See farm.h.
    MAT_FARMLAND        = 39, // tilled dark soil (furrowed rows)
    MAT_CROP_VINE       = 40, // crop stem / sprawling vine (green)
    MAT_CROP_LEAF       = 41, // broad crop leaf (brighter blue-green)
    MAT_PUMPKIN         = 42, // ripe pumpkin flesh (ribbed orange)
    MAT_CROP_FLOWER     = 43, // crop blossom accent (yellow)
    // Sawn stump cross-section, stamped by a fell (forestry.cpp) as
    // concentric growth rings: pale sapwood alternating with darker ring
    // bands. Neither is wood-like for the tree flood fill.
    MAT_CUT_WOOD        = 44, // cut face: pale cream sapwood
    MAT_CUT_RING        = 45, // cut face: darker cream growth ring
    // Vegetable flesh (farming, transient like ids 40..43 — see farm.h). One
    // id per species; greens/blossoms reuse MAT_CROP_VINE/LEAF/FLOWER.
    MAT_CARROT          = 46, // carrot root (vivid red-orange, ridged)
    MAT_TOMATO          = 47, // tomato fruit (glossy red)
    MAT_CABBAGE         = 48, // cabbage head (pale wrapped green)
    MAT_CORN            = 49, // corn ear (golden kernel rows)
    MAT_RADISH          = 50, // radish bulb (pink-magenta)
};
constexpr int MAT_COUNT = 256;          // size of the table; shader-visible array

// Crop voxels (vine/leaf/pumpkin/flower 40..43 + vegetable flesh 46..50) are
// TRANSIENT: FarmState is their source of truth and re-stamps them when
// resident, so they are stripped from the world edit-save (worldio
// extractChunk) — otherwise a saved chunk bakes in un-harvestable ghost
// produce that FarmState no longer owns. (44/45 between the ranges are the
// PERSISTENT sawn-stump faces — keep them out.)
inline bool isCropMaterial(uint8_t m) {
    return (m >= MAT_CROP_VINE && m <= MAT_CROP_FLOWER) ||
           (m >= MAT_CARROT   && m <= MAT_RADISH);
}

// Real seconds for a full in-game day at /time speed 1 (a 20-minute day/night
// cycle). Shared by the sun clock (app.cpp) and in-game-time gameplay pacing
// (crop growth: /croptime N days = N * DAY_LENGTH in-game seconds to ripen).
constexpr float DAY_LENGTH = 1200.0f;

// Map view camera FOV (tan of half-angle, ~33deg): mild perspective, diorama-
// flat. Shared by the renderer's diorama passes (render.cpp) and MapView::pan
// (map.cpp), whose pixel->world scale must match what the camera projects.
constexpr float MAP_TAN_HALF_FOV = 0.30f;

// RuneScape-style minimap layout (framebuffer px, top-right of the diorama).
// Shared by the click hit-test (app.cpp minimapHit) and minimap.frag, whose
// pixel math duplicates these numbers and must stay in lockstep.
constexpr float MM_MARGIN = 14.0f;              // gap to the window corner
constexpr float MM_R_MAP  = 118.0f;             // terrain disc radius
constexpr float MM_RING   = 18.0f;              // stone ring thickness
constexpr float MM_R_OUT  = MM_R_MAP + MM_RING; // full frame radius
constexpr float MM_ORB_R  = 17.0f;              // free-cam orb radius
constexpr float MM_ORB_D  = 0.70710678f;        // orb rides the lower-left
                                                // diagonal at MM_R_OUT + 7
