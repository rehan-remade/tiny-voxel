// Live tuning uniforms (/tune) — the in-game slider panel's values.
// INDEX ORDER MUST MATCH src/tweaks.h (the C++ table is the source of truth
// for names / ranges / shipped defaults). Include this BEFORE any lib that
// references TW_* (sky, atmosphere, tonemap, lighting).
uniform float uTweak[24];   // LIGHTING page (cam tweaks 18-20 are CPU-side)
// MATERIALS page: per-material voxel-cell params, id*MP_COUNT + param.
// MP_COUNT = 6, MTUNE_MATS = 64 (src/tweaks.h). MT(mat, param) fetches one.
uniform float uMatTune[384];   // 64 materials * 6 params
#define MP_VAR    0
#define MP_CELL   1
#define MP_ASPECT 2
#define MP_SEAM   3
#define MP_HUE    4
#define MP_SPECK  5
#define MT(mat, p) uMatTune[int(mat) * 6 + (p)]

#define TW_SUN_STR    uTweak[0]   // direct sun multiplier
#define TW_AMB_SKY    uTweak[1]   // sky-dome ambient scale
#define TW_AMB_GRD    uTweak[2]   // ground-bounce ambient scale
#define TW_SHAD_FLOOR uTweak[3]   // sun kept in full shadow
#define TW_SUN_WRAP   uTweak[4]   // diffuse wrap width
#define TW_AO_FLOOR   uTweak[5]   // darkest fine AO
#define TW_AO_SUN     uTweak[6]   // AO contact strength on the sun term
#define TW_SATURATION uTweak[7]   // tonemap saturation
#define TW_ACES_MIX   uTweak[8]   // tonemap ACES mix
#define TW_ATMO_DENS  uTweak[9]   // haze extinction (x1e-5)
#define TW_ATMO_BLUE  uTweak[10]  // in-scatter sky coupling
#define TW_ATMO_WARM  uTweak[11]  // sunward glow strength
#define TW_FOG_MAX    uTweak[12]  // auto-fog peak (x1e-3; CPU-side rhythm)
#define TW_FOG_H      uTweak[13]  // fog scale height (voxels)
#define TW_CLOUD_SHAD uTweak[14]  // cloud shadow depth
#define TW_RING_AO    uTweak[15]  // darkest baked ring AO
#define TW_CLUMP_STR  uTweak[16]  // terrain albedo mottling depth
#define TW_CLUMP_SIZE uTweak[17]  // clump feature size (voxels)
// 18-20: cam.pitch/dist/height (CPU-side, not read by shaders).
#define TW_MOON_STR   uTweak[21]  // direct moonlight multiplier
