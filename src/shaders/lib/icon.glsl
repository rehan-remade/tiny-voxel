// Inventory icon raytracers — ONE copy shared by hud.frag and bag.frag (the
// two used to carry hand-synced duplicates of the pumpkin; keep everything
// icon-shaped in here). All icons are live true-isometric voxel DDA renders:
// real chunky voxels in the world's 3-face lighting, cell-centred +Y-UP pixel
// coords in, one flat colour per voxel face out. No time dependence anywhere,
// so icons are stable frame to frame.
// Unlike the world libs (which document deps and rely on the top-level shader
// including them in order), this lib pulls in noise.glsl ITSELF: its consumers
// are leaf UI shaders that need no other lib, so one include gets them
// everything and there is no order ritual to get wrong.
#include "lib/noise.glsl"

// ---------------------------------------------------------------------------
// Shared isometric camera (matches the diorama's classic iso reading).
// ---------------------------------------------------------------------------
const vec3 ICON_R = vec3(-1.0, 0.0,  1.0) * 0.70710678;   // screen right
const vec3 ICON_U = vec3(-1.0, 2.0, -1.0) * 0.40824829;   // screen up
const vec3 ICON_D = vec3(-1.0,-1.0, -1.0) * 0.57735027;   // view dir into scene

// The world's 3-face voxel lighting: bright top, mid screen-left, dark
// screen-right. `nrm` is the axis flag vec3 the DDA tracks.
float iconFaceShade(vec3 nrm) {
    return (nrm.y > 0.5) ? 1.20      // top face   (+Y)
         : (nrm.x > 0.5) ? 0.78      // +X -> screen-left
         : (nrm.z > 0.5) ? 0.55      // +Z -> screen-right
         :                 0.42;
}

// ---------------------------------------------------------------------------
// Produce icons (ICON_BLOB) — whole-vegetable voxel icons, one shape per
// species (dispatched on the item's material id), mirroring the baked
// collect-flyer models. Occupancy: 0 air / 1 flesh / 2 greens / 3 accent.
//
// SYNC CONTRACT: the shapes, grade constants (carrot L/maxR, tomato/radish
// rr, cabbage r/ry, corn h) and accent colours here are hand-mirrored from
// buildProduceModel in app.cpp (which in turn mirrors cropVoxels' ripe
// shapes in farm.cpp). Any retune there must be repeated here AND in
// produceBounds below, or the icon silently drifts from the flyer.
// ---------------------------------------------------------------------------
int produceOcc(int mat, ivec3 v, int grade) {
    if (mat == 46) {          // carrot: tapered root, point down, fronds up
        int   L    = (grade == 2) ? 9 : (grade == 1) ? 6 : 4;
        float maxR = (grade == 2) ? 1.5 : (grade == 1) ? 1.1 : 0.9;
        if (v.y >= 0 && v.y < L && abs(v.x) <= 1 && abs(v.z) <= 1) {
            float rr = maxR * float(v.y + 1) / float(L);
            if (float(v.x * v.x + v.z * v.z) <= rr * rr + 0.01) return 1;
        }
        if (v.x == 0 && v.z == 0 && (v.y == L || v.y == L + 1)) return 2;
        if (v.y == L && v.z == 0 && abs(v.x) == 1) return 2;
        return 0;
    }
    if (mat == 47) {          // tomato: oblate fruit + stem nub
        int   r  = (grade == 2) ? 2 : 1;
        float rr = (grade == 0) ? 1.2 : (grade == 1) ? 1.45 : 2.4;
        float ey = float(v.y) / 0.85;
        if (float(v.x * v.x + v.z * v.z) + ey * ey <= rr * rr) return 1;
        if (v.x == 0 && v.z == 0 && v.y == r + 1) return 2;
        return 0;
    }
    if (mat == 48) {          // cabbage: flattened wrapped head
        int r  = grade + 1;
        int ry = max(1, r - 1);
        // Wrap leaves FIRST: their cells lie inside the flesh ellipsoid, so
        // tested after it they'd be unreachable (the baked model overwrites
        // flesh with leaf at these same spots; grade 0 has no wrap leaves).
        if (grade >= 1 && v.y == ry &&
            ((abs(v.x) == r && v.z == 0) || (v.x == 0 && v.z == -r)))
            return 2;         // loose wrap leaves
        if (v.y >= 0 && v.y <= 2 * ry) {
            float ex = float(v.x) / (float(r) + 0.5);
            float ez = float(v.z) / (float(r) + 0.5);
            float ey = (float(v.y) - float(ry)) / (float(ry) + 0.5);
            if (ex * ex + ez * ez + ey * ey <= 1.0) return 1;
        }
        return 0;
    }
    if (mat == 49) {          // corn: husked cob
        int h = (grade == 2) ? 6 : (grade == 1) ? 4 : 3;
        if (v.y >= 1 && v.y <= h) {
            if (v.x == 0 && v.z == 0) return 1;
            if (grade >= 1 && abs(v.x) + abs(v.z) == 1) return 1;
        }
        if (v.x == 0 && v.z == 0 && v.y == h + 1) return 1;   // rounded tip
        if (v.y == 0 && abs(v.x) + abs(v.z) <= 1) return 3;   // husk cup
        return 0;
    }
    if (mat == 50) {          // radish: bulb + cream taproot + leaf tuft
        int   r  = (grade == 2) ? 2 : 1;
        float rr = (grade == 0) ? 1.2 : (grade == 1) ? 1.45 : 2.4;
        ivec3 c  = v - ivec3(0, r + 1, 0);
        if (float(c.x * c.x + c.y * c.y + c.z * c.z) <= rr * rr) return 1;
        if (v.x == 0 && v.z == 0 && v.y == 0) return 3;
        if (v.x == 0 && v.z == 0 && (v.y == 2 * r + 2 || v.y == 2 * r + 3)) return 2;
        return 0;
    }
    // Default (42): pumpkin — the ripe field fruit, mirrors cropVoxels.
    int r = grade + 1;
    int h = r + 1;
    if (v.y >= 0 && v.y <= h) {
        float ex = float(v.x) / (float(r) + 0.7);
        float ez = float(v.z) / (float(r) + 0.7);
        float ey = (float(v.y) - float(h) * 0.5) / (float(h) * 0.62);
        if (ex * ex + ez * ez + ey * ey <= 1.0) return 1;   // flesh
    }
    if (v.x == 0 && v.z == 0 && v.y == h + 1) return 2;     // stem
    return 0;
}

void produceBounds(int mat, int grade, out vec3 bLo, out vec3 bHi) {
    if (mat == 46) {
        int L = (grade == 2) ? 9 : (grade == 1) ? 6 : 4;
        bLo = vec3(-1.0, 0.0, -1.0);
        bHi = vec3(2.0, float(L + 2), 2.0);
    } else if (mat == 47) {
        int r = (grade == 2) ? 2 : 1;
        bLo = vec3(float(-r), float(-r), float(-r));
        bHi = vec3(float(r + 1), float(r + 2), float(r + 1));
    } else if (mat == 48) {
        int r  = grade + 1;
        int ry = max(1, r - 1);
        bLo = vec3(float(-r), 0.0, float(-r));
        bHi = vec3(float(r + 1), float(2 * ry + 1), float(r + 1));
    } else if (mat == 49) {
        int h = (grade == 2) ? 6 : (grade == 1) ? 4 : 3;
        bLo = vec3(-1.0, 0.0, -1.0);
        bHi = vec3(2.0, float(h + 2), 2.0);
    } else if (mat == 50) {
        int r = (grade == 2) ? 2 : 1;
        bLo = vec3(float(-r), 0.0, float(-r));
        bHi = vec3(float(r + 1), float(2 * r + 4), float(r + 1));
    } else {
        int r = grade + 1;
        int h = r + 1;
        bLo = vec3(float(-r), 0.0, float(-r));
        bHi = vec3(float(r + 1), float(h + 2), float(r + 1));
    }
}

bool produceIcon(vec2 s, int mat, int grade, float cellHalf, vec3 tint, out vec3 outCol) {
    grade = clamp(grade, 0, 2);
    vec3 bLo, bHi;
    produceBounds(mat, grade, bLo, bHi);
    vec3  C   = 0.5 * (bLo + bHi);
    vec3  e   = 0.5 * (bHi - bLo);
    float extent = max(dot(e, abs(ICON_U)), dot(e, abs(ICON_R)));
    float grF    = (grade >= 2) ? 0.96 : (grade == 1) ? 0.88 : 0.80;
    float ppv    = (cellHalf - 5.0) * grF / extent;
    vec3 O = C + (s.x / ppv) * ICON_R + (s.y / ppv) * ICON_U - ICON_D * (extent * 4.0 + 4.0);
    vec3  t0 = (bLo - O) / ICON_D, t1 = (bHi - O) / ICON_D;
    vec3  ts = min(t0, t1), tb = max(t0, t1);
    float tEnter = max(max(ts.x, ts.y), ts.z);
    float tExit  = min(min(tb.x, tb.y), tb.z);
    if (tEnter > tExit || tExit < 0.0) return false;
    vec3  p      = O + ICON_D * (tEnter + 1e-4);
    ivec3 v      = ivec3(floor(p));
    vec3  tDelta = abs(1.0 / ICON_D);
    vec3  tMax   = (vec3(v) - O) / ICON_D;
    vec3  nrm    = (ts.x >= ts.y && ts.x >= ts.z) ? vec3(1, 0, 0)
                 : (ts.y >= ts.z)                  ? vec3(0, 1, 0)
                 :                                   vec3(0, 0, 1);
    ivec3 lo = ivec3(floor(bLo)) - 1;
    ivec3 hi = ivec3(ceil(bHi))  + 1;
    for (int i = 0; i < 40; ++i) {
        if (any(lessThan(v, lo)) || any(greaterThan(v, hi))) return false;
        int m = produceOcc(mat, v, grade);
        if (m != 0) {
            // Accents MUST match buildProduceModel's palette (app.cpp) — the
            // icon mirrors the collect-flyer model the player just caught.
            vec3 base = (m == 2) ? vec3(0.28, 0.42, 0.18)                 // greens
                      : (m == 3) ? ((mat == 49) ? vec3(0.72, 0.78, 0.52)  // husk
                                                : vec3(0.92, 0.88, 0.78)) // cream
                                 : tint;
            outCol = base * iconFaceShade(nrm);
            return true;
        }
        if (tMax.x < tMax.y && tMax.x < tMax.z) {
            v.x -= 1; tMax.x += tDelta.x; nrm = vec3(1, 0, 0);
        } else if (tMax.y < tMax.z) {
            v.y -= 1; tMax.y += tDelta.y; nrm = vec3(0, 1, 0);
        } else {
            v.z -= 1; tMax.z += tDelta.z; nrm = vec3(0, 0, 1);
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Material block (ICON_CUBE) — a chunky ICON_B^3 voxel chip of the material,
// corners hash-nibbled so the silhouette reads as stacked voxels. Seeded per
// material: every material chips differently, but stably.
// ---------------------------------------------------------------------------
const int ICON_B = 6;   // block edge length in voxels

bool blockOcc(ivec3 v, int mat) {
    if (any(lessThan(v, ivec3(0))) || any(greaterThanEqual(v, ivec3(ICON_B))))
        return false;
    if (mat == 6) return true;   // water: liquids don't chip
    // About half the TOP corners lose a 4-voxel Manhattan wedge. Only the top
    // (the block sits flat), and never single-voxel nibbles — at cell size
    // those read as stray pixels, not chips.
    for (int c = 0; c < 4; ++c) {
        ivec3 corner = ivec3((c & 1) * (ICON_B - 1), ICON_B - 1,
                             ((c >> 1) & 1) * (ICON_B - 1));
        int   d      = abs(v.x - corner.x) + abs(v.y - corner.y) + abs(v.z - corner.z);
        if (d < 2 && vhash(ivec3(mat, c * 7 + 1, mat * 3 + c)) > 0.55) return false;
    }
    return true;
}

// Mirror of lighting.glsl's applyMatCell with the shipped MAT_SEEDS params
// passed as literals (icons keep the baked defaults; /tune adjustments don't
// re-tint the hotbar, which is fine at 6 voxels). Groups the chip's voxels
// into coherent Worley cells: bark plates, leaf clumps, grass tufts.
void iconMatCell(ivec3 vox, float var, float cellSz, float aspect, float seam,
                 float hueA, float speck, inout vec3 base) {
    vec3 vc = vec3(vox);
    if (var > 0.001) {
        vec3 p = vec3(vc.x, vc.y / aspect, vc.z);
        vec3 cell = stoneCells3((p + 0.5) / cellSz);
        float chunk  = cell.z - 0.5;                       // per-cell shade
        float jitter = (vhash(vox) - 0.5) * 0.35;          // within-cell grain
        float structured = floor((chunk + jitter) * 4.0 + 0.5) / 4.0;  // quantized
        float flow = (vnoise3(p * 0.035) - 0.5) * 0.60
                   + (vnoise3(p * 0.11 + 9.0) - 0.5) * 0.28;
        float t  = structured + flow;
        float sm = 1.0 - smoothstep(0.04, 0.15 + 0.35 * seam, cell.y);
        float g  = (1.0 + t) * (1.0 - seam * sm);          // multiplier ~1
        base *= 1.0 + var * (g - 1.0);
    }
    if (hueA > 0.001) {
        float h = vnoise3(vc * 0.03 + 12.0) - 0.5;         // low-freq warm/cool drift
        base.r *= 1.0 + hueA * 1.6 * h;
        base.b *= 1.0 - hueA * 1.3 * h;
    }
    if (speck > 0.001) {
        float sv = vhash(vox + ivec3(5, 9, 2)) - 0.5;      // fine per-voxel stipple
        base *= 1.0 + speck * 1.6 * sv;
    }
}

// Per-voxel colour of the chip — mirrors surfaceColor()'s close-range look
// (same base colours and noise recipes), evaluated in a per-material patch of
// the shared noise fields so every material shows its own stable pattern.
// `vL` is the chip-local voxel, `nrm` the hit face's axis flag (grass paints
// by face exactly like the world: green tops, capped dirt sides).
vec3 iconVoxColor(int mat, ivec3 vL, vec3 nrm, vec3 tint) {
    ivec3 v  = vL + mat * ivec3(13, 7, 29);   // each material its own patch
    vec3  vc = vec3(v);
    float jv = vhash(v);
    vec3  col;
    if (mat == 1) {                 // dirt: gritty hash clods, warm/cool drift
        float fine   = jv - 0.5;
        float coarse = vhash(v / 2) - 0.5;
        col = vec3(0.32, 0.20, 0.12) * (1.0 + 0.62 * (fine * 0.78 + coarse * 0.45));
        float warm = vhash(v / 3 + ivec3(50, 20, 70)) - 0.5;
        col.r *= 1.0 + 0.10 * warm;
        col.b *= 1.0 - 0.10 * warm;
    } else if (mat == 2) {          // ground turf: green tops, capped dirt sides
        vec3 grass = vec3(0.32, 0.62, 0.24) * (0.95 + 0.10 * jv);   // meadow tint
        vec3 dirt  = vec3(0.36, 0.24, 0.16)
                   * (0.85 + 0.45 * vhash(v + ivec3(13, 17, 5)));
        if      (nrm.y > 0.5)       col = grass;            // any top face
        else if (vL.y == ICON_B - 1) col = grass * 0.80;     // side: grass cap row
        else if (vL.y == ICON_B - 2 &&
                 vhash(ivec3(v.x, 0, v.z)) > 0.62)
                                    col = grass * 0.66;     // ragged hang-down
        else                        col = dirt;
        iconMatCell(v, 0.22, 4.5, 1.0, 0.80, 0.30, 0.10, col);
    } else if (mat == 3 || mat == 7 || mat == 8 || mat == 9) {   // stone family
        vec3 baseCol; float contrast; vec3 mineralCol;
        if (mat == 3) {             // topstone: warm beige limestone
            baseCol = vec3(0.60, 0.55, 0.47); contrast = 0.38;
            mineralCol = vec3(0.74, 0.70, 0.60);
        } else if (mat == 7) {      // greystone: cool blue-grey slate
            baseCol = vec3(0.40, 0.43, 0.48); contrast = 0.40;
            mineralCol = vec3(0.58, 0.62, 0.68);
        } else if (mat == 8) {      // darkstone: bluish charcoal
            baseCol = vec3(0.36, 0.38, 0.45); contrast = 0.42;
            mineralCol = vec3(0.50, 0.54, 0.62);
        } else {                    // bedrock: dark basalt
            baseCol = vec3(0.27, 0.27, 0.32); contrast = 0.44;
            mineralCol = vec3(0.85, 0.74, 0.48);
        }
        col = baseCol * stoneGrain(v, contrast);
        float warmth = vnoise3(vc * 0.045 + 50.0) - 0.5;
        col.r *= 1.0 + 0.13 * warmth;
        col.b *= 1.0 - 0.13 * warmth;
        float mineral = vhash(v + ivec3(19, 5, 23));
        col = mix(col, mineralCol, smoothstep(0.95, 0.99, mineral) * 0.6);
    } else if (mat == 4) {          // wood: a sawn log, not a dirt-brown box.
        // Icon-scale adaptation: world trunks are bark on every face, but at 6
        // voxels bark-all-over is confusable with dirt (both mid-brown mottle).
        // The item comes from chopping, so the TOP face shows pale end grain
        // with growth rings; sides keep the world's bark cells over per-column
        // streaks so the grain runs visibly up the block.
        if (nrm.y > 0.5) {
            vec2  rc   = vec2(vL.x, vL.z) - vec2(float(ICON_B - 1) * 0.5);
            float ring = mod(floor(length(rc) * 1.8 + 0.3), 2.0);
            col = mix(vec3(0.62, 0.47, 0.28), vec3(0.48, 0.35, 0.19), ring)
                * (0.94 + 0.12 * jv);
        } else {
            col = vec3(0.35, 0.23, 0.11)
                * (0.80 + 0.34 * vhash(ivec3(v.x, -9, v.z)));   // bark streaks
            iconMatCell(v, 0.60, 3.4, 2.0, 0.80, 0.00, 0.00, col);
        }
    } else if (mat == 5) {          // canopy: leaf clumps, hue drift
        col = vec3(0.24, 0.46, 0.22) * 1.10;
        iconMatCell(v, 0.60, 4.8, 1.0, 0.80, 0.18, 0.00, col);
        // Icon-scale sun-catch: the world canopy pops because noon sun rakes
        // its top; the icon's flat 1.20 top shade underplays that, so bright
        // leaf clusters carry it instead.
        if (nrm.y > 0.5) {
            col *= 1.10;
            if (vhash(v + ivec3(3, 31, 7)) > 0.72)
                col = mix(col, vec3(0.52, 0.72, 0.30), 0.55);
        }
    } else if (mat == 6) {          // water: calm block — pristine, glinting
        col = tint * (0.94 + 0.08 * jv);
        col *= mix(0.80, 1.02, float(vL.y) / float(ICON_B - 1));   // depth falloff
        if (nrm.y > 0.5 && vhash(v + ivec3(3, 17, 29)) > 0.78)
            col = mix(col, vec3(0.72, 0.90, 0.98), 0.6);           // ripple glints
    } else if (mat == 10) {         // grass tufts: per-column warm/cool blades
        float jc = vhash(ivec3(v.x, 0, v.z));
        col = mix(vec3(0.46, 0.58, 0.26), vec3(0.26, 0.48, 0.30), jc)
            * (0.88 + 0.20 * jv);
        iconMatCell(v, 0.22, 4.5, 1.0, 0.80, 0.30, 0.05, col);
    } else if (mat == 16) {         // glow crystal: the world's saturated cyan
        // gem (shadeHit hard-codes this colour; the palette entry is unused by
        // the world renderer, so tint would mismatch what you mined). Sparkle
        // stands in for the emissive glow the icon can't have.
        col = vec3(0.40, 0.85, 1.05) * (0.93 + 0.08 * jv);
        if (vhash(v + ivec3(9, 27, 3)) > 0.80) col *= 1.35;
    } else if (mat == 29 || mat == 30) {   // sand: pale gold, wind ripple.
        // Powder sand (29) borrows the settled-skin look on purpose: both ids
        // read "Sand" in the bag, and the skin is the sand the player sees.
        col = vec3(0.89, 0.75, 0.53);
        col *= 1.0 + 0.05 * sin(float(v.x + v.z) * 0.55) + 0.05 * (jv - 0.5);
        iconMatCell(v, 0.12, 16.0, 1.0, 0.30, 0.00, 0.04, col);
    } else if (mat == 31) {         // sandstone: layer-cake bands
        // World bands are 3 voxels; the chip banding is /2 so three strata
        // show inside 6 voxels (the icon-scale read of "banded rock").
        float bt = vhash(ivec3(7, vL.y / 2 + mat, 13));
        col = mix(vec3(0.61, 0.32, 0.19), vec3(0.89, 0.77, 0.60), bt);
        col *= stoneGrain(v, 0.28);
        float warmth = vnoise3(vc * 0.045 + 70.0) - 0.5;
        col.r *= 1.0 + 0.10 * warmth;
        col.b *= 1.0 - 0.10 * warmth;
    } else if (mat == 32) {         // reg gravel: speckle + varnish pebbles
        float fine   = jv - 0.5;
        float coarse = vhash(v / 2) - 0.5;
        col = vec3(0.78, 0.68, 0.52) * (1.0 + 0.28 * (fine * 0.6 + coarse * 0.4));
        float pebble = vhash(v + ivec3(33, 9, 41));
        col = mix(col, vec3(0.30, 0.24, 0.18), smoothstep(0.93, 0.985, pebble) * 0.65);
        iconMatCell(v, 0.24, 5.0, 1.0, 0.30, 0.00, 0.04, col);
    } else if (mat == 33) {         // cactus flesh: waxy ribbed blue-green
        float rib = 0.5 + 0.5 * sin(float(v.x + v.z) * 2.2);
        col = vec3(0.31, 0.42, 0.29) * (0.86 + 0.18 * rib) * (0.92 + 0.12 * jv);
        iconMatCell(v, 0.56, 4.8, 1.0, 0.30, 0.00, 0.12, col);
    } else if (mat == 35) {         // sage shrub: dusty olive mix
        col = mix(vec3(0.27, 0.31, 0.19), vec3(0.56, 0.58, 0.40), jv);
    } else if (mat == 36) {         // desert wood: ashy grey-brown grain
        col = vec3(0.44, 0.40, 0.31) * (0.82 + 0.30 * jv);
    } else if (mat == 39) {         // farmland: dark loam, ridge/furrow columns
        float row = float(vL.x & 1);
        col = vec3(0.24, 0.16, 0.10) * (0.82 + 0.30 * row) * (0.88 + 0.24 * jv);
    } else if (mat == 42) {         // pumpkin flesh: ribbed orange
        col = vec3(0.86, 0.44, 0.12) * (0.90 + 0.12 * jv);
        iconMatCell(v, 0.16, 5.0, 2.2, 0.45, 0.00, 0.04, col);
    } else {                        // default: palette colour with per-voxel life
        col = tint * (0.86 + 0.26 * jv);
        float warm = vhash(v / 2 + ivec3(11, 4, 27)) - 0.5;
        col.r *= 1.0 + 0.08 * warm;
        col.b *= 1.0 - 0.08 * warm;
    }
    return col;
}

bool materialBlockIcon(vec2 s, int mat, float cellHalf, vec3 tint, out vec3 outCol) {
    vec3  bLo = vec3(0.0);
    vec3  bHi = vec3(float(ICON_B));
    vec3  C   = 0.5 * (bLo + bHi);
    vec3  e   = 0.5 * (bHi - bLo);
    float extent = max(dot(e, abs(ICON_U)), dot(e, abs(ICON_R)));
    float ppv    = (cellHalf - 5.0) * 0.94 / extent;
    vec3 O = C + (s.x / ppv) * ICON_R + (s.y / ppv) * ICON_U - ICON_D * (extent * 4.0 + 4.0);
    vec3  t0 = (bLo - O) / ICON_D, t1 = (bHi - O) / ICON_D;
    vec3  ts = min(t0, t1), tb = max(t0, t1);
    float tEnter = max(max(ts.x, ts.y), ts.z);
    float tExit  = min(min(tb.x, tb.y), tb.z);
    if (tEnter > tExit || tExit < 0.0) return false;
    vec3  p      = O + ICON_D * (tEnter + 1e-4);
    ivec3 v      = ivec3(floor(p));
    vec3  tDelta = abs(1.0 / ICON_D);
    vec3  tMax   = (vec3(v) - O) / ICON_D;
    vec3  nrm    = (ts.x >= ts.y && ts.x >= ts.z) ? vec3(1, 0, 0)
                 : (ts.y >= ts.z)                  ? vec3(0, 1, 0)
                 :                                   vec3(0, 0, 1);
    ivec3 lo = ivec3(floor(bLo)) - 1;
    ivec3 hi = ivec3(ceil(bHi))  + 1;
    for (int i = 0; i < 40; ++i) {
        if (any(lessThan(v, lo)) || any(greaterThan(v, hi))) return false;
        if (blockOcc(v, mat)) {
            outCol = iconVoxColor(mat, v, nrm, tint) * iconFaceShade(nrm);
            return true;
        }
        if (tMax.x < tMax.y && tMax.x < tMax.z) {
            v.x -= 1; tMax.x += tDelta.x; nrm = vec3(1, 0, 0);
        } else if (tMax.y < tMax.z) {
            v.y -= 1; tMax.y += tDelta.y; nrm = vec3(0, 1, 0);
        } else {
            v.z -= 1; tMax.z += tDelta.z; nrm = vec3(0, 0, 1);
        }
    }
    return false;
}
