#include "materials.h"
#include "common.h"

// digTime here is "seconds per swing while pointed at this material in
// playtest mode" (dev mode uses a fixed fast interval). Rebalanced so the
// hierarchy goes leaves -> grass/dirt -> wood -> topstone -> deeper stones.
// Columns: color, digTime, smoothness, metallic, emissive.
const MatProps g_materials[256] = {
    /* MAT_AIR    */ {{0.00f, 0.00f, 0.00f}, 0.00f, 0.00f, 0.0f, 0.0f},
    /* MAT_DIRT   */ {{0.45f, 0.30f, 0.20f}, 0.12f, 0.00f, 0.0f, 0.0f},
    /* MAT_GRASS  */ {{0.30f, 0.65f, 0.28f}, 0.12f, 0.05f, 0.0f, 0.0f},
    /* MAT_STONE  */ {{0.62f, 0.58f, 0.50f}, 3.50f, 0.12f, 0.0f, 0.0f},   // topstone -- bare hands hurt
    /* MAT_WOOD   */ {{0.32f, 0.20f, 0.10f}, 0.30f, 0.08f, 0.0f, 0.0f},   // medium
    /* MAT_LEAVES */ {{0.18f, 0.45f, 0.18f}, 0.05f, 0.22f, 0.0f, 0.0f},   // very soft; waxy sheen
    /* MAT_WATER  */ {{0.15f, 0.35f, 0.65f}, 0.05f, 0.90f, 0.0f, 0.0f},   // (water has its own surface path)
    /* MAT_GREYSTONE */ {{0.40f, 0.43f, 0.48f},  7.0f, 0.14f, 0.0f, 0.0f},
    /* MAT_DARKSTONE */ {{0.28f, 0.30f, 0.36f}, 14.0f, 0.16f, 0.0f, 0.0f},
    /* MAT_BEDROCK   */ {{0.15f, 0.15f, 0.18f}, 30.0f, 0.10f, 0.0f, 0.0f},  // basically refuses hands
    /* MAT_GRASS_BLADE */ {{0.46f, 0.78f, 0.30f}, 0.05f, 0.06f, 0.0f, 0.0f}, // bright, slightly yellow-green tufts
    /* MAT_FLOWER_STEM */ {{0.30f, 0.55f, 0.20f}, 0.02f, 0.08f, 0.0f, 0.0f},
    /* MAT_FLOWER_RED  */ {{0.88f, 0.22f, 0.22f}, 0.02f, 0.12f, 0.0f, 0.0f},
    /* MAT_FLOWER_YELLOW */ {{0.96f, 0.88f, 0.30f}, 0.02f, 0.12f, 0.0f, 0.0f},
    /* MAT_FLOWER_BLUE */ {{0.32f, 0.42f, 0.92f}, 0.02f, 0.12f, 0.0f, 0.0f},
    /* MAT_FLOWER_WHITE */ {{0.95f, 0.95f, 0.92f}, 0.02f, 0.12f, 0.0f, 0.0f},
    /* MAT_CRYSTAL */ {{0.62f, 0.90f, 1.05f}, 6.0f, 0.70f, 0.0f, 1.5f},   // icy glow (special-cased emissive; data here for PR11)
    /* MAT_MUSHROOM_STEM */ {{0.90f, 0.85f, 0.72f}, 0.04f, 0.10f, 0.0f, 0.0f}, // cream
    /* MAT_MUSHROOM_CAP  */ {{0.68f, 0.23f, 0.19f}, 0.04f, 0.18f, 0.0f, 0.0f}, // warm red; moist cap sheen
    /* MAT_MUSHROOM_RIM  */ {{0.40f, 0.10f, 0.09f}, 0.04f, 0.12f, 0.0f, 0.0f}, // dark red (underside)
    /* MAT_MUSHROOM_SPOT */ {{0.93f, 0.90f, 0.80f}, 0.04f, 0.12f, 0.0f, 0.0f}, // warm white spots
    /* MAT_MUSHROOM_CAP2 */ {{0.70f, 0.47f, 0.29f}, 0.04f, 0.14f, 0.0f, 0.0f}, // light brown flat cap
    /* MAT_CAVE_MOSS */ {{0.34f, 0.46f, 0.18f}, 0.04f, 0.05f, 0.0f, 0.0f}, // olive ground-cover moss
    /* MAT_VINE */      {{0.28f, 0.44f, 0.16f}, 0.04f, 0.10f, 0.0f, 0.0f}, // living ivy (green)
    /* MAT_VINE_DEAD */ {{0.36f, 0.27f, 0.14f}, 0.04f, 0.04f, 0.0f, 0.0f}, // dead ivy (brown)
    /* MAT_FIRE  */ {{1.00f, 0.55f, 0.16f}, 0.02f, 0.00f, 0.0f, 3.2f},  // warm orange flame, strong glow
    /* MAT_EMBER */ {{0.95f, 0.35f, 0.10f}, 0.02f, 0.05f, 0.0f, 1.1f},  // dull-red dying coal
    /* MAT_CHAR  */ {{0.10f, 0.09f, 0.085f}, 0.06f, 0.02f, 0.0f, 0.15f}, // blackened burnt wood
    /* MAT_ASH   */ {{0.46f, 0.45f, 0.43f}, 0.04f, 0.02f, 0.0f, 0.0f},   // inert grey residue
    // color, digTime, smoothness, metallic, emissive, density, state, angleOfRepose, strength
    /* MAT_SAND */ {{0.82f, 0.70f, 0.46f}, 0.10f, 0.04f, 0.0f, 0.0f, 1.6f, MS_POWDER, 33.0f, 0.05f}, // granular
    // Desert biome (v1) — all MS_SOLID (physics fields default). Base albedos;
    // the painterly variation (ripple, banding, ribs) is in materials.glsl.
    /* MAT_SAND_STATIC     */ {{0.89f, 0.75f, 0.53f}, 0.10f, 0.03f, 0.0f, 0.0f}, // warm pale gold (NOT orange)
    /* MAT_SANDSTONE       */ {{0.78f, 0.55f, 0.38f}, 2.00f, 0.10f, 0.0f, 0.0f}, // banded warm rock
    /* MAT_DESERT_PAVEMENT */ {{0.78f, 0.68f, 0.52f}, 0.50f, 0.05f, 0.0f, 0.0f}, // reg gravel
    /* MAT_CACTUS_FLESH    */ {{0.31f, 0.42f, 0.29f}, 0.06f, 0.10f, 0.0f, 0.0f}, // desaturated blue-green
    /* MAT_CACTUS_BLOOM    */ {{0.85f, 0.26f, 0.12f}, 0.02f, 0.10f, 0.0f, 0.0f}, // vivid bloom accent
    /* MAT_SHRUB_SAGE      */ {{0.42f, 0.47f, 0.28f}, 0.05f, 0.04f, 0.0f, 0.0f}, // muted dry shrub foliage
    /* MAT_DESERT_WOOD     */ {{0.44f, 0.40f, 0.31f}, 0.30f, 0.04f, 0.0f, 0.0f}, // ashy grey-brown stems
    /* MAT_MAP_UNKNOWN     */ {{0.42f, 0.40f, 0.36f}, 0.00f, 0.00f, 0.0f, 0.0f}, // map fog-of-war mass (render-only)
    /* MAT_CREATURE        */ {{1.00f, 1.00f, 1.00f}, 0.00f, 0.03f, 0.0f, 0.0f}, // creature hits: WHITE base — lighting.frag multiplies the shaded result by the G-buffer albedo (the .cvox palette), so the white base makes that substitution exact
    // Farming (near-field only). Base albedos; surfaceColor() in materials.glsl
    // adds the furrow rows / rib / leaf variation.
    /* MAT_FARMLAND    */ {{0.24f, 0.16f, 0.10f}, 0.10f, 0.02f, 0.0f, 0.0f}, // dark tilled loam
    /* MAT_CROP_VINE   */ {{0.30f, 0.46f, 0.18f}, 0.03f, 0.06f, 0.0f, 0.0f}, // green stem/vine
    /* MAT_CROP_LEAF   */ {{0.34f, 0.55f, 0.24f}, 0.03f, 0.08f, 0.0f, 0.0f}, // broad leaf
    /* MAT_PUMPKIN     */ {{0.86f, 0.44f, 0.12f}, 0.20f, 0.10f, 0.0f, 0.0f}, // ripe orange flesh
    /* MAT_CROP_FLOWER */ {{0.97f, 0.82f, 0.22f}, 0.02f, 0.12f, 0.0f, 0.0f}, // yellow blossom
    /* MAT_CUT_WOOD    */ {{0.76f, 0.63f, 0.42f}, 0.30f, 0.10f, 0.0f, 0.0f}, // sawn stump face: pale cream sapwood
    /* MAT_CUT_RING    */ {{0.55f, 0.42f, 0.26f}, 0.28f, 0.08f, 0.0f, 0.0f}, // sawn stump face: darker growth ring
    // Vegetable flesh (one id per crop species; see farm.h CROP_DEFS).
    /* MAT_CARROT      */ {{0.88f, 0.35f, 0.08f}, 0.10f, 0.08f, 0.0f, 0.0f}, // vivid red-orange root
    /* MAT_TOMATO      */ {{0.80f, 0.16f, 0.10f}, 0.08f, 0.35f, 0.0f, 0.0f}, // glossy red fruit (waxy sheen)
    /* MAT_CABBAGE     */ {{0.52f, 0.68f, 0.38f}, 0.12f, 0.10f, 0.0f, 0.0f}, // pale wrapped-leaf green
    /* MAT_CORN        */ {{0.92f, 0.78f, 0.22f}, 0.12f, 0.18f, 0.0f, 0.0f}, // golden kernel rows
    /* MAT_RADISH      */ {{0.80f, 0.20f, 0.45f}, 0.08f, 0.14f, 0.0f, 0.0f}, // pink-magenta bulb
    // Remaining 206 entries default-initialise to {{0,0,0}, 0, 0, 0, 0}.
};
