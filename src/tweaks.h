#pragma once
// Live visual-tuning (/tune): three pages (Tab cycles).
//   /tune lighting  — sun / ambient / colour / atmosphere / camera constants,
//                     delivered to the shaders via uTweak[] (TWEAKS below).
//   /tune materials — per-material voxel-cell texture (uMatTune[], MAT_LIST +
//                     MAT_PARAMS below): select a material, dial its cells.
//   /tune creatures — per-species gameplay knobs (CREATURE_TUNE, creatures.h):
//                     select a species, dial population/behaviour; /tune save
//                     patches assets/creatures/<name>.def in place.
// The VALUES here are the shipped defaults (single source of truth). Tune in
// game, /tune save -> tune.cfg, then bake the numbers back here.

struct TweakDef {
    const char* name;   // shown in the panel + used as the tune.cfg key
    float mn, mx, def;
};

// ---- LIGHTING page: global scalars in uTweak[] (index order == TW_* in
// lib/tweaks.glsl; keep in sync). ------------------------------------------
// clang-format off
constexpr TweakDef TWEAKS[] = {
    // name              min    max    default
    { "sun.strength",    0.4f,  2.5f,  1.40f },  //  0 direct sun multiplier
    { "amb.sky",         0.3f,  2.5f,  1.00f },  //  1 sky-dome ambient scale
    { "amb.ground",      0.3f,  2.5f,  1.00f },  //  2 ground-bounce ambient scale
    { "shadow.floor",    0.0f,  0.7f,  0.00f },  //  3 sun kept in full shadow
    { "sun.wrap",        0.0f,  0.6f,  0.00f },  //  4 diffuse wrap width
    { "ao.floor",        0.0f,  1.0f,  0.30f },  //  5 darkest fine AO
    { "ao.on.sun",       0.0f,  1.0f,  0.15f },  //  6 AO contact strength on sun
    { "col.saturation",  0.8f,  1.8f,  1.30f },  //  7 tonemap saturation
    { "col.aces",        0.0f,  1.0f,  0.35f },  //  8 tonemap ACES mix
    { "atmo.density",    0.0f, 20.0f,  6.00f },  //  9 haze extinction (x1e-5)
    { "atmo.blue",       0.0f,  1.0f,  0.55f },  // 10 in-scatter sky coupling
    { "atmo.warm",       0.0f,  1.0f,  0.65f },  // 11 sunward glow strength
    { "fog.max",         0.0f,  6.0f,  1.60f },  // 12 auto-fog peak (x1e-3)
    { "fog.height",     30.0f, 300.f, 90.00f },  // 13 fog scale height (vox)
    { "cloud.shadow",    0.0f,  0.9f,  0.55f },  // 14 cloud shadow depth
    { "ring.ao.floor",   0.3f,  1.0f,  0.55f },  // 15 darkest baked ring AO
    { "albedo.clump",    0.0f,  0.35f, 0.12f },  // 16 terrain colour mottling depth
    { "albedo.size",     8.0f, 120.f, 40.00f },  // 17 clump feature size (voxels)
    { "cam.pitch",       0.25f, 1.40f, 0.55f },  // 18 follow-cam elevation (rad)
    { "cam.dist",       120.0f, 1200.f, 638.f }, // 19 follow-cam orbit distance (vox)
    { "cam.height",      0.0f, 60.0f, 14.00f },  // 20 follow-cam look-at height (vox)
    { "moon.strength",   0.0f,  3.0f,  1.00f },  // 21 direct moonlight multiplier
};
// clang-format on
constexpr int TWEAK_N   = int(sizeof(TWEAKS) / sizeof(TWEAKS[0]));
constexpr int TWEAK_GPU = 24;   // uTweak[] array size (keep in sync w/ lib/tweaks.glsl)
static_assert(TWEAK_N <= TWEAK_GPU, "grow uTweak[] in lib/tweaks.glsl");

constexpr int TW_CAM_PITCH  = 18;
constexpr int TW_CAM_DIST   = 19;
constexpr int TW_CAM_HEIGHT = 20;
constexpr int TW_MOON_STRENGTH = 21;   // CPU reads it for the moon-vs-sun pick

// ---- MATERIALS page: per-material voxel-cell texture ----------------------
// Each surface material gets MP_COUNT params, delivered as uMatTune[id*MP+p]
// (MT() in lib/tweaks.glsl), applied by applyMatCell() in lib/materials.glsl.
// Indexed by material id (< MTUNE_MATS). MAT_LIST is the selector order.
enum MatParam { MP_VAR = 0, MP_CELL, MP_ASPECT, MP_SEAM, MP_HUE, MP_SPECK, MP_COUNT };
constexpr int MTUNE_MATS = 64;
constexpr int MATTUNE_N  = MTUNE_MATS * MP_COUNT;   // uMatTune[] size

struct MatParamDef { const char* name; float mn, mx; };
// clang-format off
constexpr MatParamDef MAT_PARAMS[MP_COUNT] = {
    { "variation", 0.0f,  0.8f },   // MP_VAR    cell value contrast (0 = off)
    { "cell.size", 2.0f, 16.0f },   // MP_CELL   voxels per cell
    { "aspect",    0.3f,  4.0f },   // MP_ASPECT cell Y-stretch (>1 = tall, e.g. bark)
    { "seam",      0.0f,  0.9f },   // MP_SEAM   cell-border darkening (visible grooves)
    { "hue",       0.0f,  0.6f },   // MP_HUE    warm/cool drift across cells
    { "speck",     0.0f,  0.4f },   // MP_SPECK  fine per-voxel stipple on top
};
struct MatEntry { int id; const char* name; };
constexpr MatEntry MAT_LIST[] = {
    {  2, "grass"       }, { 10, "grass blade" }, {  4, "wood"       }, {  5, "leaves"     },
    {  1, "dirt"        }, {  3, "topstone"    }, {  7, "greystone"  }, {  8, "darkstone"  },
    {  9, "bedrock"     }, { 29, "sand"        }, { 30, "sand skin"  }, { 31, "sandstone"  },
    { 32, "pavement"    }, { 33, "cactus"      }, { 35, "sage shrub" }, { 36, "desert wood"},
    { 22, "cave moss"   }, { 23, "vine"        }, { 42, "pumpkin"     }, { 46, "carrot"     },
    { 47, "tomato"      }, { 48, "cabbage"     }, { 49, "corn"        }, { 50, "radish"     },
};
// clang-format on
constexpr int MAT_LIST_N = int(sizeof(MAT_LIST) / sizeof(MAT_LIST[0]));

// Seeded per-material defaults (material id -> the 6 params). Materials not
// listed default to MAT_NEUTRAL (variation 0 = no cell texture, but sane
// params ready to dial up). Baked from the tuned grass/wood/leaf look.
struct MatSeed { int id; float p[MP_COUNT]; };
//                                    VAR   CELL  ASPCT SEAM  HUE   SPECK
constexpr float MAT_NEUTRAL[MP_COUNT] = { 0.0f, 5.0f, 1.0f, 0.30f, 0.0f, 0.0f };
// clang-format off
constexpr MatSeed MAT_SEEDS[] = {
    {  2, { 0.22f, 4.5f,  1.0f, 0.80f, 0.30f, 0.10f } },  // grass
    { 10, { 0.22f, 4.5f,  1.0f, 0.80f, 0.30f, 0.05f } },  // grass blade
    {  4, { 0.60f, 3.4f,  2.0f, 0.80f, 0.00f, 0.00f } },  // wood — tall bark cells
    {  5, { 0.60f, 4.8f,  1.0f, 0.80f, 0.18f, 0.00f } },  // leaves
    { 30, { 0.12f, 16.0f, 1.0f, 0.30f, 0.00f, 0.04f } },  // sand skin
    { 32, { 0.24f, 5.0f,  1.0f, 0.30f, 0.00f, 0.04f } },  // desert pavement
    { 33, { 0.56f, 4.8f,  1.0f, 0.30f, 0.00f, 0.12f } },  // cactus
    { 42, { 0.16f, 5.0f,  2.2f, 0.45f, 0.00f, 0.04f } },  // pumpkin — vertical ribs
    { 46, { 0.14f, 3.0f,  3.0f, 0.40f, 0.00f, 0.05f } },  // carrot — tall ridge lines
    { 48, { 0.28f, 4.0f,  1.0f, 0.60f, 0.10f, 0.00f } },  // cabbage — wrapped leaf plates
    { 49, { 0.10f, 2.0f,  1.2f, 0.50f, 0.00f, 0.10f } },  // corn — tight kernel grid
};
// clang-format on
constexpr int MAT_SEEDS_N = int(sizeof(MAT_SEEDS) / sizeof(MAT_SEEDS[0]));

// Fill a MATTUNE_N array with the shipped per-material defaults (neutral for
// everything, then the seeded materials). Shared by app.cpp init + /tune reset.
inline void fillMatTuneDefaults(float* out) {
    for (int m = 0; m < MTUNE_MATS; ++m)
        for (int p = 0; p < MP_COUNT; ++p)
            out[m * MP_COUNT + p] = MAT_NEUTRAL[p];
    for (int s = 0; s < MAT_SEEDS_N; ++s)
        for (int p = 0; p < MP_COUNT; ++p)
            out[MAT_SEEDS[s].id * MP_COUNT + p] = MAT_SEEDS[s].p[p];
}

// /tune panel layout — shared by drawTunePanel + the mouse hit-test (app.cpp)
// + tune.frag. Left-anchored, vertically centred; row 0 = header.
constexpr int   TUNE_COLS   = 36;
constexpr int   TUNE_MAXROWS = TWEAK_N + 2;   // lighting page is the tallest
constexpr float TUNE_CW     = 16.0f;
constexpr float TUNE_CH     = 16.0f;
constexpr float TUNE_MARGIN = 12.0f;
constexpr int   TUNE_BAR_C0 = 15;
constexpr int   TUNE_BAR_W  = 12;
