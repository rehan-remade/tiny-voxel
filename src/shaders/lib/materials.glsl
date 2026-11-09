// Material palette UBO + per-voxel procedural surface colouring.
// Depends on noise.glsl and the surfaceY/heightTex/uTime uniforms.

layout(std140, binding = 0) uniform Materials {
    vec4 matColors[256];   // rgb = albedo
    vec4 matProps[256];    // x = smoothness, y = metallic, z = emissive
};
vec3  matColor(uint v)      { return matColors[v].rgb; }
float matSmoothness(uint v) { return matProps[v].x; }
float matMetallic(uint v)   { return matProps[v].y; }
float matEmissive(uint v)   { return matProps[v].z; }
// Biome weights (x=highland, y=meadow, z=forest, w=desert) at a world XZ —
// mirrors biomeAt's climate field on the CPU EXACTLY (same hash, frequency,
// stretch, smoothstep thresholds, normalization). Shared by every biome-tinted
// material. MUST stay bit-for-bit in sync with internal.h:biomeAt or rendered
// tint drifts from the terrain it sits on.
vec4 biomeWeights(vec2 wxz) {
    float raw     = bSmoothNoise(wxz.x * (1.0/2200.0) + 53.7, wxz.y * (1.0/2200.0) + 19.2);
    float m       = clamp(0.5 + (raw - 0.5) * 2.4, 0.0, 1.0);   // mirror biomeMoisture
    float wDesert = smoothstep(0.16, 0.06, m);                  // extreme DRY tail
    float wHigh   = smoothstep(0.42, 0.30, m) * smoothstep(0.10, 0.20, m);
    float wForest = smoothstep(0.58, 0.70, m);
    float wMeadow = max(0.0, 1.0 - wHigh - wForest - wDesert);
    float inv     = 1.0 / max(1e-4, wHigh + wForest + wMeadow + wDesert);
    return vec4(wHigh, wMeadow, wForest, wDesert) * inv;
}
// Ground-turf biome tint (the moderate, pre-amplification values).
vec3 biomeGrassTint(vec2 wxz) {
    vec4 w = biomeWeights(wxz);
    return w.x * vec3(0.52, 0.52, 0.28)    // highland: dry olive
         + w.y * vec3(0.32, 0.62, 0.24)    // meadow:   bright green
         + w.z * vec3(0.12, 0.36, 0.18)    // forest:   deep green
         + w.w * vec3(0.85, 0.72, 0.48);   // desert:   warm tan (dry-grass borders)
}
// Multiplicative tint nudging FOLIAGE (leaves, grass tufts) toward the biome's
// character so tree crowns and tufts read with their biome, not one global
// green. Render-only; no gen impact.
vec3 biomeFoliageTint(vec2 wxz) {
    vec4 w = biomeWeights(wxz);
    return w.x * vec3(1.55, 1.08, 0.26)    // highland: extreme golden/dry
         + w.y * vec3(1.00, 1.00, 1.00)    // meadow:   neutral
         + w.z * vec3(0.20, 0.58, 0.34)    // forest:   deep dark green
         + w.w * vec3(1.12, 1.02, 0.66);   // desert:   warm desaturated
}
// (stoneGrainSized / stoneGrain moved to lib/noise.glsl — they are pure
// functions and the inventory icons in lib/icon.glsl share them now.)
vec3 surfaceColor(uint mat, vec3 hitPos, vec3 normal, float tHit) {
    // Distance-attenuate the per-voxel surface noise to kill sub-pixel
    // shimmer/moire on faraway terrain. Up close (tHit < 80) noise is at
    // full strength; beyond ~250 voxels it's effectively gone and the
    // material reads as its flat base tone. No global desaturation -- only
    // the high-frequency variation is dampened, preserving close-up texture.
    float nfade = 1.0 - smoothstep(80.0, 250.0, tHit);

    if (mat == 1u) {
        // Dirt: pure HASH jitter (random, no direction) -- value noise was
        // giving it wood grain. Fine per-voxel speckle + a coarse per-2-voxel
        // clump + per-3-voxel warm/cool drift, all hash-based so there's no
        // structure, just gritty random clods. High contrast = very jittery.
        ivec3 vox = ivec3(floor(hitPos - normal * 0.5));
        float fine   = vhash(vox) - 0.5;                 // per-voxel
        float coarse = vhash(vox / 2) - 0.5;             // per 2x2x2 clump
        float t = fine * 0.78 + coarse * 0.45;
        vec3 col = vec3(0.32, 0.20, 0.12) * (1.0 + 0.62 * t);
        float warm = vhash(vox / 3 + ivec3(50, 20, 70)) - 0.5;
        col.r *= 1.0 + 0.10 * warm;
        col.b *= 1.0 - 0.10 * warm;
        return col;
    }
    if (mat == 2u) {
        ivec3 vox = ivec3(floor(hitPos));
        vec3 vc = vec3(vox) + 0.5;

        // Ground turf: biome tint (moderate) from the climate field, plus a
        // faint local drift so a single biome doesn't read perfectly flat.
        vec3 base = biomeGrassTint(vc.xz);
        float drift = (vnoise2(vc.xz * 0.045 + vec2(31.3, 17.7)) - 0.5);
        base *= 1.0 + 0.08 * drift;

        float j = mix(0.5, blockNoise(vc + vec3(7,3,11), 1.0), nfade);
        base *= (0.95 + 0.10 * j);

        bool isGrassFace  = (normal.y > 0.5);
        bool isDirtBottom = (normal.y < -0.5);

        // Decide what this face would look like up close.
        vec3 faceCol;
        if (isGrassFace) {
            faceCol = base;
        } else if (isDirtBottom) {
            vec3 dirtCol = vec3(0.36, 0.24, 0.16);
            float a = mix(0.5, blockNoise(vc + vec3(13,17,5), 1.0), nfade);
            faceCol = dirtCol * (0.85 + 0.45 * a);
        } else {
            // Side face: dirt strip with a grass cap at the top.
            vec3 dirtCol = vec3(0.36, 0.24, 0.16);
            float a = mix(0.5, blockNoise(vc + vec3(13,17,5), 1.0), nfade);
            vec3 dirt = dirtCol * (0.85 + 0.45 * a);
            float fy = fract(hitPos.y);
            if (fy > 0.84) {
                faceCol = base;
            } else if (fy > 0.70 && blockNoise(vec3(vox.x, 0, vox.z), 1.0) > 0.62) {
                faceCol = base * 0.80;
            } else {
                faceCol = dirt;
            }
        }

        // At distance, fade toward the grass top color regardless of face.
        // Slopes/hills mix grass and dirt at sub-pixel scale up close, which
        // creates a strong moire/shimmer aliasing pattern at range. Pulling
        // everything to the biome's grass tone at far distances kills that
        // pattern -- distant hills read as smooth green like an aerial photo.
        return mix(base, faceCol, nfade);
    }
    // Stone family: the VOXEL is the paint unit. Each voxel face is one flat
    // color, varied per-voxel by stoneGrain (a quantized palette with gentle
    // patch clustering) plus a subtle per-voxel warm/cool shift. No sub-voxel
    // detail at all -- depth comes from the blocky geometry + AO + lighting,
    // exactly like the reference. A rare whole voxel reads as a brighter
    // mineral block (still per-voxel, so it stays a clean cube).
    if (mat == 3u || mat == 7u || mat == 8u || mat == 9u) {
        ivec3 vox = ivec3(floor(hitPos - normal * 0.5));   // the solid voxel
        vec3 baseCol; float contrast; vec3 mineralCol;
        if (mat == 3u) {        // topstone: warm beige limestone
            baseCol = vec3(0.60, 0.55, 0.47); contrast = 0.38;
            mineralCol = vec3(0.74, 0.70, 0.60);
        } else if (mat == 7u) { // greystone: cool blue-grey slate
            baseCol = vec3(0.40, 0.43, 0.48); contrast = 0.40;
            mineralCol = vec3(0.58, 0.62, 0.68);
        } else if (mat == 8u) { // darkstone: bluish charcoal
            baseCol = vec3(0.36, 0.38, 0.45); contrast = 0.42;
            mineralCol = vec3(0.50, 0.54, 0.62);
        } else {                // bedrock: dark basalt (kept moody but lit-able)
            baseCol = vec3(0.27, 0.27, 0.32); contrast = 0.44;
            mineralCol = vec3(0.85, 0.74, 0.48);
        }
        vec3 col = baseCol * stoneGrain(vox, contrast);
        // Per-REGION hue drift (low-frequency): some areas warm reddish, others
        // cooler yellow-tan. This is the warm/cool spread the reference has and
        // a flat monochrome orange lacks -- a big part of its "variation".
        float warmth = (vnoise3(vec3(vox) * 0.045 + 50.0) - 0.5);
        col.r *= 1.0 + 0.13 * warmth;
        col.b *= 1.0 - 0.13 * warmth;
        // Rare whole-voxel mineral block (per-voxel, so it's a clean cube).
        float mineral = vhash(vox + ivec3(19, 5, 23));
        col = mix(col, mineralCol, smoothstep(0.95, 0.99, mineral) * 0.6);
        return col;
    }
    if (mat == 4u) {
        // FLAT per-voxel bark base — no sub-voxel noise. The Voronoi-cell
        // quantized variation (patch clumps + discrete tone steps, exactly the
        // stone family's stoneGrain) is applied per-voxel in lighting.glsl
        // (wood.var). One solid colour per cube, like rock.
        return vec3(0.33, 0.22, 0.12);
    }
    if (mat == 5u) {
        // FLAT per-voxel canopy base (biome-tinted, low-frequency only) — no
        // sub-voxel noise. The quantized leaf value+hue steps come from the
        // Voronoi stoneGrain in lighting.glsl (leaf.var): each cube one solid
        // green, clustered into patches, like the reference.
        return vec3(0.24, 0.46, 0.22) * biomeFoliageTint(hitPos.xz);
    }
    if (mat == 10u) {
        // Grass blade. Per-column hue variation (yellow-green <-> blue-green)
        // gives the meadow noticeable color noise instead of one flat tone;
        // a tiny per-voxel jitter breaks the column itself so each blade has
        // micro-variation along its height.
        // DISTANCE-INDEPENDENT column/voxel variation (no nfade): the blades
        // are the diorama's dominant surface, and fading their variation out
        // at range left them a flat neon slab. Per-column + per-voxel, so each
        // tuft is one homogeneous colour. Palette MUTED off the old neon
        // yellow-green — the extra field-scale hue/value spread now comes from
        // the shared grass zones in lighting.glsl (grass.var, extended to mat 10).
        ivec3 vox     = ivec3(floor(hitPos));
        float jColumn = vhash(ivec3(vox.x, 0, vox.z));
        float jVoxel  = vhash(vox);
        vec3 warm = vec3(0.46, 0.58, 0.26);   // muted olive yellow-green
        vec3 cool = vec3(0.26, 0.48, 0.30);   // muted natural green
        vec3 base = mix(warm, cool, jColumn);
        base *= 0.88 + 0.20 * jVoxel;
        return base * biomeFoliageTint(hitPos.xz);   // tufts match their biome
    }
    if (mat == 11u) {
        // Flower stem: same family as grass blades, slightly cooler/darker so
        // the stem reads as a distinct plant part next to the bloom.
        float jv = mix(0.5, vhash(ivec3(floor(hitPos))), nfade);
        vec3 base = vec3(0.30, 0.55, 0.20);
        return base * (0.88 + 0.18 * jv);
    }
    if (mat == 12u) {        // poppy red
        float jv = mix(0.5, vhash(ivec3(floor(hitPos))), nfade);
        return vec3(0.92, 0.20, 0.20) * (0.92 + 0.12 * jv);
    }
    if (mat == 13u) {        // dandelion yellow
        float jv = mix(0.5, vhash(ivec3(floor(hitPos))), nfade);
        return vec3(0.98, 0.90, 0.28) * (0.92 + 0.10 * jv);
    }
    if (mat == 14u) {        // cornflower blue
        float jv = mix(0.5, vhash(ivec3(floor(hitPos))), nfade);
        return vec3(0.32, 0.44, 0.95) * (0.92 + 0.12 * jv);
    }
    if (mat == 15u) {        // chamomile white
        float jv = mix(0.5, vhash(ivec3(floor(hitPos))), nfade);
        return vec3(0.96, 0.96, 0.92) * (0.94 + 0.08 * jv);
    }
    if (mat == 18u || mat == 21u) {   // mushroom cap (red dome / brown flat)
        // Top faces bright, underside dark -- a continuous shade across the cap
        // (no material banding). Plus a faint per-voxel jitter for life.
        float jv = mix(0.5, vhash(ivec3(floor(hitPos))), nfade);
        float up = clamp(normal.y * 0.5 + 0.5, 0.0, 1.0);   // 0 underside .. 1 top
        return matColor(mat) * mix(0.66, 1.06, up) * (0.94 + 0.10 * jv);
    }
    if (mat >= 17u && mat <= 20u) {   // mushroom stem / rim / spot
        float jv = mix(0.5, vhash(ivec3(floor(hitPos))), nfade);
        return matColor(mat) * (0.90 + 0.16 * jv);
    }
    if (mat == 22u) {   // cave moss ground-cover
        // Per-voxel green variation (yellow-green <-> olive), drying to brown
        // with depth so the shallow "living" band is lush and the deep is drier.
        ivec3 vox = ivec3(floor(hitPos - normal * 0.5));
        float jv = vhash(vox);
        vec3 bright = vec3(0.44, 0.58, 0.20);   // yellow-green
        vec3 olive  = vec3(0.22, 0.34, 0.13);   // darker olive
        vec3 c = mix(olive, bright, jv);
        float depth = clamp((surfaceY - hitPos.y) / 1900.0, 0.0, 1.0);
        c = mix(c, c * vec3(1.15, 0.82, 0.62), depth * 0.55);  // browner/drier deep
        return c;
    }
    if (mat == 23u) {   // living ivy: mottled green with brighter leaf voxels
        ivec3 vox = ivec3(floor(hitPos - normal * 0.5));
        float jv = vhash(vox);
        vec3 stem = vec3(0.24, 0.38, 0.14);
        vec3 leaf = vec3(0.46, 0.60, 0.22);
        vec3 c = mix(stem, leaf, smoothstep(0.5, 0.9, jv));
        return c * (0.82 + 0.30 * jv);
    }
    if (mat == 24u) {   // dead ivy: dry brown
        float jv = mix(0.5, vhash(ivec3(floor(hitPos))), nfade);
        return vec3(0.36, 0.27, 0.14) * (0.82 + 0.30 * jv);
    }
    if (mat == 30u) {   // MAT_SAND_STATIC: warm pale gold dune/sand surface
        ivec3 vox = ivec3(floor(hitPos - normal * 0.5));
        vec3 base = vec3(0.89, 0.75, 0.53);                 // pale warm gold, never orange
        // Wind ripple: gentle parallel corrugation in world XZ; faint per-voxel grain.
        float ripple = sin((hitPos.x + hitPos.z) * 0.55);
        float jv     = vhash(vox) - 0.5;
        base *= 1.0 + (0.05 * ripple + 0.05 * jv) * nfade;
        // Up-facing crests blow lighter; steep faces sit a touch deeper (a cheap
        // aspect cue; the full warm/cool shadow split lands in lighting.glsl, S2).
        base *= mix(0.93, 1.05, clamp(normal.y, 0.0, 1.0));
        return base;
    }
    if (mat == 31u) {   // MAT_SANDSTONE: warm banded "layer cake" rock
        ivec3 vox = ivec3(floor(hitPos - normal * 0.5));
        float band = floor(hitPos.y / 3.0);                 // bands keyed on ABSOLUTE world Y
        float bt   = vhash(ivec3(7, int(band), 13));        // per-band tone
        vec3 light = vec3(0.89, 0.77, 0.60);                // cream/buff
        vec3 rust  = vec3(0.61, 0.32, 0.19);                // rust terracotta
        vec3 base  = mix(rust, light, bt);
        base *= stoneGrain(vox, 0.28);                      // sit in the stone family
        float warmth = (vnoise3(vec3(vox) * 0.045 + 70.0) - 0.5);
        base.r *= 1.0 + 0.10 * warmth;
        base.b *= 1.0 - 0.10 * warmth;
        return base;
    }
    if (mat == 32u) {   // MAT_DESERT_PAVEMENT: reg gravel — default desert ground
        ivec3 vox = ivec3(floor(hitPos - normal * 0.5));
        float fine   = vhash(vox) - 0.5;
        float coarse = vhash(vox / 2) - 0.5;
        vec3 col = vec3(0.78, 0.68, 0.52) * (1.0 + 0.28 * (fine * 0.6 + coarse * 0.4) * nfade);
        float pebble = vhash(vox + ivec3(33, 9, 41));       // sparse dark desert-varnish flecks
        col = mix(col, vec3(0.30, 0.24, 0.18), smoothstep(0.93, 0.985, pebble) * 0.65);
        return col;
    }
    if (mat == 33u) {   // MAT_CACTUS_FLESH: desaturated blue-green, waxy, ribbed
        float jv  = mix(0.5, vhash(ivec3(floor(hitPos))), nfade);
        float rib = 0.5 + 0.5 * sin((hitPos.x + hitPos.z) * 2.2);   // vertical ribbing
        vec3 base = vec3(0.31, 0.42, 0.29) * (0.86 + 0.18 * rib);
        return base * (0.92 + 0.12 * jv);
    }
    if (mat == 34u) {   // MAT_CACTUS_BLOOM: vivid accent (color, not glow)
        float jv = mix(0.5, vhash(ivec3(floor(hitPos))), nfade);
        return vec3(0.85, 0.26, 0.12) * (0.92 + 0.12 * jv);
    }
    if (mat == 35u) {   // MAT_SHRUB_SAGE: dry desert scrub — muted, dusty, low-sat
        ivec3 vox = ivec3(floor(hitPos - normal * 0.5));
        float jv = vhash(vox);
        vec3 dark = vec3(0.27, 0.31, 0.19);                 // shadowed interior olive
        vec3 lite = vec3(0.56, 0.58, 0.40);                 // dusty sage highlight
        vec3 c = mix(dark, lite, jv);
        c *= 0.90 + 0.12 * clamp(normal.y, 0.0, 1.0);       // tops catch a touch more
        return c;
    }
    if (mat == 36u) {   // MAT_DESERT_WOOD: ashy grey-brown desert stems (not tree-bark brown)
        ivec3 vox = ivec3(floor(hitPos - normal * 0.5));
        float jv = vhash(vox);
        vec3 c = vec3(0.44, 0.40, 0.31) * (0.82 + 0.30 * jv);  // muted, slight per-voxel grain
        c.g *= 1.0 + 0.04 * (jv - 0.5);                     // faint grey-green tinge
        return c;
    }
    if (mat == 39u) {   // MAT_FARMLAND: dark tilled loam with furrow rows
        ivec3 vox = ivec3(floor(hitPos - normal * 0.5));
        vec3 base = vec3(0.24, 0.16, 0.10);
        float row = 0.5 + 0.5 * sin(hitPos.x * 3.14159);    // corrugated rows along X
        base *= 0.82 + 0.30 * row;                          // ridge bright / furrow dark
        base *= 0.88 + 0.24 * vhash(vox);                   // per-voxel clod grain
        base *= mix(0.80, 1.06, clamp(normal.y, 0.0, 1.0)); // tops catch light
        return base;
    }
    if (mat == 40u) {   // MAT_CROP_VINE: green stem/vine
        ivec3 vox = ivec3(floor(hitPos - normal * 0.5));
        float jv = vhash(vox);
        return vec3(0.30, 0.46, 0.18) * (0.80 + 0.34 * jv);
    }
    if (mat == 41u) {   // MAT_CROP_LEAF: broad leaf, brighter blue-green
        ivec3 vox = ivec3(floor(hitPos - normal * 0.5));
        float jv = vhash(vox);
        vec3 c = mix(vec3(0.26, 0.44, 0.20), vec3(0.42, 0.62, 0.28), jv);
        c *= mix(0.84, 1.08, clamp(normal.y, 0.0, 1.0));    // top faces catch light
        return c;
    }
    if (mat == 42u) {   // MAT_PUMPKIN: orange flesh — flat PER-VOXEL colour (no
                        // sub-voxel pattern); rib/cell texture comes from
                        // applyMatCell (/tune materials -> "pumpkin").
        ivec3 vox = ivec3(floor(hitPos - normal * 0.5));
        float jv = vhash(vox);
        return vec3(0.86, 0.44, 0.12) * (0.90 + 0.12 * jv);
    }
    if (mat == 43u) {   // MAT_CROP_FLOWER: yellow blossom
        float jv = mix(0.5, vhash(ivec3(floor(hitPos))), nfade);
        return vec3(0.97, 0.82, 0.22) * (0.90 + 0.12 * jv);
    }
    if (mat == 46u) {   // MAT_CARROT: vivid red-orange root shoulder
        ivec3 vox = ivec3(floor(hitPos - normal * 0.5));
        float jv = vhash(vox);
        return vec3(0.88, 0.35, 0.08) * (0.88 + 0.16 * jv);
    }
    if (mat == 47u) {   // MAT_TOMATO: glossy red fruit, tops catch the light
        ivec3 vox = ivec3(floor(hitPos - normal * 0.5));
        float jv = vhash(vox);
        vec3 c = vec3(0.80, 0.16, 0.10) * (0.90 + 0.14 * jv);
        c *= mix(0.92, 1.12, clamp(normal.y, 0.0, 1.0));
        return c;
    }
    if (mat == 48u) {   // MAT_CABBAGE: pale wrapped-leaf green, veined swirl
        ivec3 vox = ivec3(floor(hitPos - normal * 0.5));
        float jv = vhash(vox);
        float vein = 0.5 + 0.5 * sin(hitPos.x * 1.7 + hitPos.y * 2.3 + hitPos.z * 1.1);
        vec3 c = mix(vec3(0.40, 0.56, 0.30), vec3(0.62, 0.78, 0.44), jv);
        c *= 0.88 + 0.18 * vein;
        return c;
    }
    if (mat == 49u) {   // MAT_CORN: golden kernel rows (voxel-parity checker)
        ivec3 vox = ivec3(floor(hitPos - normal * 0.5));
        float jv = vhash(vox);
        float k = float((vox.x + vox.y + vox.z) & 1);
        return vec3(0.92, 0.78, 0.22) * (0.80 + 0.20 * k) * (0.92 + 0.12 * jv);
    }
    if (mat == 50u) {   // MAT_RADISH: pink-magenta bulb
        ivec3 vox = ivec3(floor(hitPos - normal * 0.5));
        float jv = vhash(vox);
        vec3 c = vec3(0.80, 0.20, 0.45) * (0.88 + 0.16 * jv);
        c.r *= 1.0 + 0.06 * (jv - 0.5);
        return c;
    }
    return matColor(mat);
}
