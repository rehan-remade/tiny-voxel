// Lighting + shading for the voxel renderer: AO, soft sun shadows, and the
// big shadeHit() surface shader (including the day/night sun+moon model).
// Consumed by the deferred lighting pass (lighting.frag) on the map diorama
// path. Depends on noise/voxel_trace/materials/sky/tonemap and the scene
// uniforms, so it must be included after those.

float computeAO(ivec3 hitVoxel, vec3 normal, vec3 hitPos) {
    ivec3 N = ivec3(round(normal));
    ivec3 T1, T2; faceTangents(N, T1, T2);
    ivec3 base = hitVoxel + N;
    float ao[4];
    int idx = 0;
    for (int j = -1; j <= 1; j += 2)
    for (int i = -1; i <= 1; i += 2) {
        bool s1 = getVoxelAt(base + T1 * i) != 0u;
        bool s2 = getVoxelAt(base + T2 * j) != 0u;
        bool cd = getVoxelAt(base + T1 * i + T2 * j) != 0u;
        float v = (s1 && s2) ? 0.0
                : (3.0 - (s1 ? 1.0 : 0.0) - (s2 ? 1.0 : 0.0) - (cd ? 1.0 : 0.0));
        ao[idx++] = v / 3.0;
    }
    vec2 uv = vec2(dot(hitPos - vec3(hitVoxel), vec3(T1)),
                   dot(hitPos - vec3(hitVoxel), vec3(T2)));
    uv = clamp(uv, 0.001, 0.999);
    float lo = mix(ao[0], ao[1], uv.x);
    float hi = mix(ao[2], ao[3], uv.x);
    return mix(lo, hi, uv.y);
}

// 4-tap soft shadow. Each tap perturbs the sun direction within a small cone
// and the visibility is averaged, so shadow boundaries fade in over a few
// voxels of penumbra instead of cutting hard.
float softShadow(vec3 origin, vec3 normal, vec3 sd, vec3 jitterKey) {
    vec3 up = abs(sd.y) > 0.9 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
    vec3 t1 = normalize(cross(sd, up));
    vec3 t2 = cross(sd, t1);
    const float CONE = 0.020;   // ~1.1° half-angle
    float occ = 0.0;
    for (int i = 0; i < 4; i++) {
        vec2 fixedOff = vec2(
            (i == 0 || i == 2) ?  1.0 : -1.0,
            (i == 0 || i == 1) ?  1.0 : -1.0);
        // jitterKey == origin for static terrain; moving surfaces (forestry
        // falls) pass an object-anchored key so the dither doesn't re-roll.
        vec2 jitter = (vec2(
            hash31(jitterKey * 17.3 + float(i) * 7.0),
            hash31(jitterKey * 23.1 + float(i) * 13.0)
        ) - 0.5) * 0.5;
        vec3 sdir = normalize(sd + (t1 * (fixedOff.x + jitter.x)
                                  + t2 * (fixedOff.y + jitter.y)) * CONE);
        occ += anyHit(origin + normal * 0.5, sdir, 220.0) ? 1.0 : 0.0;
    }
    return 1.0 - occ * 0.25;
}
float softShadow(vec3 origin, vec3 normal, vec3 sd) {
    return softShadow(origin, normal, sd, origin);
}

// Per-material voxel-cell texture (/tune materials, uMatTune via MT()). Groups
// tiny voxels into coherent CELLS (Worley/Voronoi) so a cluster reads as one
// bark plate / leaf clump / grass tuft while the world stays tiny-voxel — the
// "discretize the look above the geometry" idea. All strictly per-voxel
// homogeneous (colour constant across a voxel face). Params (MT(mat, ...)):
//   MP_VAR    value contrast between cells (0 = material untouched)
//   MP_CELL   cell size in voxels
//   MP_ASPECT cell Y-stretch (>1 = tall cells, e.g. vertical bark grooves)
//   MP_SEAM   darkening on cell borders (how visible the grooves are)
//   MP_HUE    warm<->cool drift across cells
//   MP_SPECK  fine per-voxel stipple on top
void applyMatCell(uint mat, ivec3 vox, inout vec3 base) {
    if (mat >= uint(64)) return;
    float var   = MT(mat, MP_VAR);
    float hueA  = MT(mat, MP_HUE);
    float speck = MT(mat, MP_SPECK);
    if (var < 0.001 && hueA < 0.001 && speck < 0.001) return;   // material untouched
    vec3 vc = vec3(vox);
    if (var > 0.001) {
        float cellSz = max(MT(mat, MP_CELL), 1.0);
        float aspect = max(MT(mat, MP_ASPECT), 0.05);
        float seam   = MT(mat, MP_SEAM);
        // Anisotropic cells: squash Y before the Worley lookup so cells stretch
        // vertically when aspect>1 (bark grooves run up the trunk).
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

// (terrainFog is GONE. Geometry never blends toward the sky/background any
// more: the primary ray hands off from fine voxels to LOD ring 1's coarse
// voxels in gbuffer.frag — two OPAQUE representations of the same content —
// so trees stay crisp to the handoff instead of dissolving translucent. The
// reveal curtain's frontier-hiding job is covered by the cascade fall-through:
// an unstreamed fine chunk shows ring terrain, not fog.)

// NOTE: AO and sun/moon visibility are computed per-pixel by lighting.frag
// (baked ring AO, traced softShadow for fine in-window diorama hits, the
// heightfield march for far ring-pool hits) and handed in as aoIn /
// shadowIn / moonShadowIn.
// `ringHit` = this hit is LOD-ring coarse terrain (beyond the fine window):
// EVERY window-anchored input (heightTex burial, lightVol, water-neighbour
// getVoxelAt, any anyHit/softShadow secondary ray) MUST gate on it — they all
// wrap toroidally and return other-side-of-window garbage out there. One flag,
// checked structurally at each input, per the design review.
// `normal` = the GEOMETRIC face normal: material colour (grass tops!),
// AO probes, water-face logic — anything that identifies WHICH face this is.
// `shadeN` = the normal for DIFFUSE light only (/slopeshade may tilt it
// toward the terrain slope; equal to `normal` when off). Conflating them
// painted distant grass-block SIDES green: surfaceColor classifies top-vs-
// side by normal.y, and the slope-blended normal read as "top".
// `detailPos` / `voxel`: where the painterly SURFACE PATTERN samples from.
// Equal to hitPos / its behind-the-face voxel for static terrain; forestry's
// falling trees pass their object-anchored pre-fall coordinates so the
// pattern rides the moving model. Everything that LIGHTS the point (burial,
// light volume, lantern, translucency rays) keeps using the real hitPos.
// `creaturePix`: creature-pass terrain-material hit — its voxels are no
// longer in the brick world, so world-ray extras (leaf translucency) take
// their ring-style fallback.
vec3 shadeHit(uint mat, ivec3 voxel, vec3 normal, vec3 shadeN, vec3 hitPos, float tHit, vec3 rd, bool ringHit, float aoIn, float shadowIn, float moonShadowIn, vec3 detailPos, bool creaturePix) {
    if (mat == 16u) {
        // Glow crystal: self-emissive (independent of lantern/sun). Per-voxel
        // facet variation (brightness + slight hue jitter + per-voxel pulse
        // phase) so a cluster reads as distinct crystalline shards, not one
        // smooth blob. Saturated cyan kept OFF pure white so it stays gem-like.
        ivec3 cv = ivec3(floor(hitPos - normal * 0.5));
        float v = vhash(cv);
        vec3 col = vec3(0.40, 0.85, 1.05);
        col *= 0.93 + 0.08 * v;                 // very faint per-voxel facet
        col.r += (v - 0.5) * 0.02;              // barely-there hue jitter
        col.g += (0.5 - v) * 0.006;
        col *= 0.96 + 0.04 * normal.y;          // top faces a hair brighter (form)
        // Pulse drifts only over LARGE distances (low spatial frequency), so
        // neighbouring crystals stay in sync -- none looks darker than the next.
        // Gentle amplitude keeps it a soft breath, not a flicker.
        float pulse = 0.92 + 0.08 * sin(uTime * 1.6 + (hitPos.x + hitPos.z) * 0.015);
        return col * pulse;
    }
    if (mat == 25u) {
        // Active flame: self-lit and animated, but ONE flat colour per voxel
        // (the painterly grid look). The flicker phase comes from the INTEGER
        // voxel coord, never the sub-voxel hit point, so a face never shows an
        // internal gradient; neighbouring voxels are phase-shifted so the fire
        // still shimmers and ripples across the grid. Skips the generic sun/AO
        // path -- fire makes its own light, it isn't lit or shadowed.
        ivec3 cv = ivec3(floor(hitPos - normal * 0.5));
        float v  = vhash(cv);
        float ph = v * 6.2831 + float(cv.x) * 1.7 + float(cv.y) * 2.3 + float(cv.z) * 1.3;
        float fl = 0.75 + 0.40 * sin(uTime * 13.0 + ph)
                        + 0.20 * sin(uTime * 27.0 + ph * 1.7);
        fl = clamp(fl, 0.35, 1.8);
        // hotter at the flicker peaks: deep orange -> bright yellow core
        vec3 flame = mix(vec3(1.00, 0.38, 0.08), vec3(1.00, 0.82, 0.34),
                         clamp(fl - 0.7, 0.0, 1.0));
        return flame * fl * 2.8;
    }
    if (mat == 26u) {
        // Dying ember: dull red-orange with a slow gentle breath, much dimmer.
        ivec3 cv = ivec3(floor(hitPos - normal * 0.5));
        float v  = vhash(cv);
        float gl = 0.65 + 0.35 * sin(uTime * 3.5 + v * 6.2831);
        return vec3(0.95, 0.30, 0.08) * gl * 1.2;
    }
    if (mat == 6u) {
        // Water has three visual states detected from neighbour context:
        //  - POOL TOP   : settled water surface (air above, support below).
        //                 Fresnel-reflective with sky reflection + light teal.
        //  - FALLING    : water voxel with air directly below it. Renders as
        //                 bright sparkly cyan droplet — the "live" water.
        //  - SUBMERGED  : water with water above; deep teal interior tint
        //                 (rarely visible directly).
        // Ring water is always a settled pool surface (the ring generator only
        // fills oasis pools); its neighbours can't be probed (toroidal wrap).
        uint above = ringHit ? 0u : getVoxelAt(voxel + ivec3(0,  1, 0));
        uint below = ringHit ? 6u : getVoxelAt(voxel + ivec3(0, -1, 0));
        bool isFalling      = (below == 0u);
        bool isPoolSurface  = (above == 0u && !isFalling);
        bool isSubmerged    = (above == 6u);

        // Falling droplet: brighter, more emissive, with animated sparkle.
        if (isFalling) {
            vec3 base = vec3(0.45, 0.78, 0.95);
            float sparkle = blockNoise(hitPos + vec3(uTime * 2.0, uTime * 1.3, uTime * 0.7), 18.0);
            float glint   = smoothstep(0.78, 0.98, sparkle);
            vec3 col = base * (0.85 + 0.30 * glint);
            // Edge-facing faces a touch brighter — looks like wet shine.
            float face = max(max(abs(normal.x), abs(normal.z)), normal.y);
            col += vec3(0.45, 0.55, 0.65) * glint * 0.40 * face;
            return col;
        }

        // Pool surface: a FACETED grid of voxel-cell tiles. Each surface cell
        // gets its own flat, slowly-scrolling normal, so every tile reflects a
        // slightly different slice of sky and catches the sun at a different
        // angle. That per-tile variation IS the rippling-mosaic + sun-glitter
        // look from the reference — not one smooth mirror plane.
        if (isPoolSurface && normal.y > 0.5) {
            vec2 cc = vec2(floor(hitPos.xz)) + 0.5;     // this tile's centre

            // Per-cell flat tilt from two scrolling octaves. Low freq makes the
            // bright tiles cluster and drift; high freq adds fine sparkle.
            // Sampling at the cell centre keeps the value FLAT across the tile.
            float aX = (vnoise2(cc * 0.060 + vec2(uTime * 0.16, 0.0)) - 0.5)
                     + (vnoise2(cc * 0.190 + vec2(uTime * 0.42, 0.0)) - 0.5) * 0.5
                     + (vnoise2(cc * 0.620 + vec2(uTime * 0.70, 0.0)) - 0.5) * 0.40;
            float aZ = (vnoise2(cc * 0.060 + vec2(0.0, uTime * 0.13)) - 0.5)
                     + (vnoise2(cc * 0.190 + vec2(0.0, uTime * 0.37)) - 0.5) * 0.5
                     + (vnoise2(cc * 0.620 + vec2(0.0, uTime * 0.62)) - 0.5) * 0.40;
            vec3 N = normalize(vec3(aX * 0.42, 1.0, aZ * 0.42));

            vec3 R = reflect(rd, N);
            R.y = max(R.y, 0.02);                            // don't look under horizon
            vec3 reflectColor = skyColor(R) * 1.12;

            // Stylised low fixed reflection so the bottom stays visible from any
            // angle (no grazing-angle mirror).
            float fresnel = 0.12;

            // Refraction + parallax walk to the bed: gives genuine see-through
            // depth in shallows, fading to scatter colour in deep water.
            vec3 rdRefr = refract(rd, N, 1.0 / 1.33);
            if (dot(rdRefr, rdRefr) < 1e-4) rdRefr = vec3(0.0, -1.0, 0.0);
            rdRefr = normalize(mix(rd, rdRefr, 0.30));

            vec3 bedCol = vec3(0.04, 0.20, 0.24);
            float waterPath = 0.0;
            if (!ringHit) {   // ring pools: default deep colour (no bed walk out-of-window)
                vec3 p = hitPos + rdRefr * 0.6;    // start just inside water
                for (int i = 0; i < 96; ++i) {
                    uint v = getVoxelAt(ivec3(floor(p)));
                    if (v == 6u) { waterPath += 0.5; p += rdRefr * 0.5; continue; }
                    if (v == 0u) break;            // ray exited water into air
                    vec3 raw = surfaceColor(v, p, vec3(0.0, 1.0, 0.0), tHit);
                    // Moon share keeps the see-through-shallows read at night
                    // (~the moonlit shore's fraction of day; matches the 0.30
                    // moon weight on the bed caustics).
                    bedCol = raw * (0.05 + 0.85 * dayLight
                                    + 0.22 * moonLight * TW_MOON_STR);
                    break;
                }
            } else {
                waterPath = 8.0;                   // distant pool: read as deep
            }
            // Aquamarine Beer-Lambert: absorb red fastest and blue faster than
            // green, so the deep colour lands on teal rather than navy.
            // Beer-Lambert with a path OFFSET so depth obscures the bed
            // MONOTONICALLY: the +5 offset gives the 1-voxel shallows their
            // blue cast (no bare-rock-through-glass), and from there it only
            // ever gets MORE obscured with depth — never the inversion where
            // deep water shows more rock than shallow. Stronger absorption than
            // before so the deep centre reads as near-solid blue.
            vec3 absorb   = vec3(0.12, 0.06, 0.08);
            vec3 transmit = exp(-absorb * (waterPath + 5.0));
            vec3 scatter  = vec3(0.07, 0.37, 0.50) * (0.55 + 0.45 * dayLight);
            vec3 waterCol = bedCol * transmit + scatter * (1.0 - transmit);

            vec3 col = mix(waterCol, reflectColor, fresnel);

            // Two-band per-cell shimmer (flat per tile). The broad SWELL drifts
            // large brighter regions; the FINE band is high-frequency so
            // adjacent tiles differ strongly — that's the busy mosaic of small
            // square patches that shimmy around in the reference.
            float swell = 0.5 * vnoise2(cc * 0.075 + vec2(-uTime * 0.11,  uTime * 0.09))
                        + 0.5 * vnoise2(cc * 0.210 + vec2( uTime * 0.31, -uTime * 0.27));
            float fine  = 0.6 * vnoise2(cc * 0.55 + vec2( uTime * 0.60, -uTime * 0.45))
                        + 0.4 * vnoise2(cc * 1.15 + vec2(-uTime * 0.85,  uTime * 0.75));
            col *= 0.80 + 0.24 * swell + 0.24 * fine;
            col += vec3(0.03, 0.11, 0.11) * smoothstep(0.58, 1.0, fine) * dayLight;

            // Light shadow on the water — same cast as land, toward the
            // DOMINANT body (sun by day, moon by night). The sample point is
            // displaced horizontally by the per-cell surface tilt, so the
            // shadow's edge weaves tile-by-tile and ripples with the water.
            // Ring pools: no shadow ray (out-of-window), analytic full light.
            bool  moonRules = (moonLight * TW_MOON_STR > dayLight);
            vec3  domDir = moonRules ? moonDir : sunDir;
            float domStr = moonRules ? moonLight * TW_MOON_STR : dayLight;
            vec3  shOrigin = hitPos + vec3(aX, 0.0, aZ) * 2.5;
            float shadow   = (ringHit || domStr < 0.03) ? 1.0
                           : softShadow(shOrigin, vec3(0.0, 1.0, 0.0), domDir);
            // The darkening fades with the light's strength: at the twilight
            // handoff neither body meaningfully lights the pool, and a shadow
            // toward a horizon-grazing body must not dim the whole surface.
            // The x10 ramp saturates by domStr 0.1 — below dayLight's value
            // for any sun above the horizon (0.12), so every daytime frame
            // keeps the original full-depth water shadow.
            col *= mix(0.60, 1.0, mix(1.0, shadow, clamp(domStr * 10.0, 0.0, 1.0)));

            // Glitter: thin sparse streaks, NOT a glossy sheet. A tight
            // specular (high power) lights only tiles whose facet points almost
            // exactly at the body's reflection, and we gate hard on the
            // brightest fine tiles so it scatters into a thin trail — the warm
            // sun path by day, the classic cool moon-trail at night. No base
            // floor -> the rest of the surface stays matte. Killed in shadow.
            vec3 reflL    = reflect(-domDir, N);
            float spec    = pow(max(dot(-rd, reflL), 0.0), 90.0);
            float sparkle = smoothstep(0.62, 0.97, fine);
            vec3 glintCol = moonRules ? vec3(0.75, 0.85, 1.00) : vec3(1.0, 0.98, 0.90);
            col += glintCol * spec * sparkle * 2.6 * domStr * shadow;

            return col;
        }

        // Submerged body or side face of pool seen from above water:
        // deeper teal tint. Rarely the dominant fragment but matters at
        // pool edges and shoreline.
        vec3 sideTint = vec3(0.16, 0.38, 0.52);
        if (isSubmerged) sideTint *= 0.78;
        // Dominant-body direct light (sun by day, moon by night), same as the
        // pool surface above.
        bool  moonRules2 = (moonLight * TW_MOON_STR > dayLight);
        vec3  ld2  = moonRules2 ? moonDir : sunDir;
        float str2 = moonRules2 ? moonLight * TW_MOON_STR * 0.6 : dayLight;
        vec3  lc2  = moonRules2 ? MOON_TINT : vec3(1.0, 0.95, 0.85);
        float diff2 = max(dot(normal, ld2), 0.0);
        float shadow2 = (diff2 > 0.0 && str2 > 0.03 && !ringHit)
                      ? softShadow(hitPos, normal, ld2) : 1.0;
        vec3 ambient2 = mix(vec3(0.06,0.10,0.20), vec3(0.20,0.30,0.42), dayLight);
        vec3 col = sideTint * (ambient2 + lc2 * str2 * diff2 * shadow2 * 0.5);
        return col;
    }
    vec3 base = surfaceColor(mat, detailPos, normal, tHit);
    // Meter-scale, WORLD-ANCHORED albedo clumps on terrain materials — the
    // accordion killer that works WITH the voxel look instead of fighting it:
    // the terrace stripes read as stripes because every voxel in a band is
    // the identical colour; irregular mottling (mineral patches, coarse sand,
    // dust) cuts across the contour rows so the eye stops tracking them.
    // World-anchored and applied to fine AND ring hits alike, so the D0 seam
    // agrees and nothing swims under camera motion. Never flora/water/wood.
    if (TW_CLUMP_STR > 0.001 &&
        (mat == 1u || mat == 2u || mat == 3u || mat == 7u || mat == 8u ||
         mat == 9u || mat == 29u || mat == 30u || mat == 31u || mat == 32u)) {
        float inv = 1.0 / max(TW_CLUMP_SIZE, 4.0);
        float n = vnoise2(hitPos.xz * inv)
                + 0.5 * vnoise2(hitPos.xz * (inv * 2.7) + 31.7);
        base *= 1.0 + TW_CLUMP_STR * (n * 1.3333 - 1.0);
    }

    // Per-material voxel-cell texture (/tune materials): every surface material
    // can carry its own Voronoi cell pattern — value plates, cell borders, an
    // aspect stretch, hue drift, fine stipple — dialled per material. Applied
    // to the shaded base here (fine AND ring hits both route through here).
    // The caller computes the behind-the-face voxel (object-anchored for
    // moving creature-pass surfaces) — sample the cell pattern from it.
    applyMatCell(mat, voxel, base);

    // Desert biome weight at this surface point. Drives the warm-sun /
    // cool-violet-shadow split and the dust haze below — sampled per-hit so it
    // only touches desert geometry and fades smoothly across the border. One
    // value-noise eval (mirrors biomeAt's climate field).
    float desertW = biomeWeights(hitPos.xz).w;

    // Cave darkness from the point's depth below the LOCAL terrain surface
    // (sampled from the height texture) — not a global constant. This makes
    // deep caves genuinely dark while low valleys (which sit below the global
    // surface but are open to sky) stay bright, and it's a per-POINT value so
    // brightness never pops as the camera moves in/out of a cave. Bright
    // within a few voxels of the surface, dark by ~70 down.
    // Ring hits are open surface terrain by construction (the ring generator
    // fills no caves); heightTex would wrap toroidally out there -> gate.
    float surfH = texture(heightTex, vec2(hitPos.x, hitPos.z) /
                                     vec2(float(gridSize.x), float(gridSize.z))).r;
    float localDepth = ringHit ? 0.0 : (surfH - hitPos.y);
    float burial = clamp(smoothstep(1.0, 18.0, localDepth), 0.0, 1.0);
    float skyVis = mix(1.0, 0.015, burial);

    // Wrap width 0 = classic hard 90-degree cutoff (shipped default).
    float diff = clamp((dot(shadeN, sunDir) + TW_SUN_WRAP)
                       / (1.0 + TW_SUN_WRAP), 0.0, 1.0);
    // Map hits arrive with ringHit set and carry their traced/heightfield
    // sun visibility in shadowIn (computed per-pixel in lighting.frag); a
    // non-ring hit would take the single-shot 4-tap cone march, skipped
    // when the point is too deep for the sun to matter.
    float shadow = ringHit ? shadowIn                      // analytic (no rays out-of-window)
                 : ((diff > 0.0 && skyVis > 0.10)
                      ? softShadow(hitPos, normal, sunDir) : 1.0);
    shadow = mix(TW_SHAD_FLOOR, 1.0, shadow);   // 0 floor = true black shadow
    // Ring hits carry their baked AO in aoIn; a non-ring hit would take the
    // analytic 4-neighbour AO (single-shot, no denoiser).
    float ao = ringHit ? aoIn                              // analytic (no neighbour probes)
             : computeAO(voxel, normal, hitPos);
    // Stronger contact darkening: floor lowered so crevices and concave corners
    // read genuinely deep. AO modulates ALL light (sun, ambient, lamp, lantern)
    // below, so it shapes the stone instead of being washed out by the lamp.
    ao = mix(TW_AO_FLOOR, 1.0, ao);

    // Hemispherical ambient: upward-facing surfaces pick up cool sky tint,
    // downward-facing surfaces pick up warm ground/leaf bounce, side faces
    // get a smooth blend. The single biggest thing that makes the world
    // feel like it's outdoors instead of in flat studio lighting.
    // Raised ~35% (2026-07 lighting pass): with AO/burial no longer crushing
    // the direct sun, the shadowed sides of hills were the remaining darkness
    // — outdoor shadow is lit by the whole sky dome and should stay readable.
    vec3 skyAmb = mix(vec3(0.08,0.13,0.22), vec3(0.27,0.36,0.52), dayLight)
                * skyVis * TW_AMB_SKY;
    skyAmb += vec3(0.04, 0.06, 0.13) * moonLight * TW_MOON_STR * skyVis;   // faint moonlit lift
    vec3 grdAmb = mix(vec3(0.05,0.06,0.08), vec3(0.21,0.16,0.10), dayLight)
                * skyVis * TW_AMB_GRD;
    // DESERT keystone: push the sky/shadow fill toward a strong COOL blue-violet
    // sky dome and the ground bounce toward WARM sand — so a shadowed dune face
    // / mesa wall reads cool while the sunlit face reads warm. This complementary
    // split is what sculpts the terrain into form at golden hour (the meadow has
    // no such split). Blended by desertW; green biomes untouched.
    {
        // The /tune ambient scales must govern BOTH palettes — this desert
        // override replaces the green-biome ambient by desertW, so without
        // the multipliers here the sliders visibly did nothing on dunes.
        vec3 skyAmbDesert = mix(vec3(0.10,0.12,0.22), vec3(0.30,0.37,0.60), dayLight)
                          * skyVis * TW_AMB_SKY;
        vec3 grdAmbDesert = mix(vec3(0.06,0.05,0.04), vec3(0.30,0.20,0.10), dayLight)
                          * skyVis * TW_AMB_GRD;
        skyAmb = mix(skyAmb, skyAmbDesert, desertW);
        grdAmb = mix(grdAmb, grdAmbDesert, desertW);
    }
    float skyW = clamp(shadeN.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 ambient = mix(grdAmb, skyAmb, skyW);
    // (Cave darkness comes from the per-point skyVis burial term above —
    // no camera-driven hack.)
    // Sun colour warms to orange as it nears the horizon (sunrise/sunset),
    // white-gold when high — this is what makes golden-hour light feel warm.
    vec3 sunTint = mix(vec3(1.0, 0.46, 0.20), vec3(1.0, 0.95, 0.85),
                       smoothstep(-0.02, 0.30, sunDir.y));
    // Desert sun runs warmer (honey-amber even when high) to deepen the
    // warm/cool split against the cool-violet shadows above.
    sunTint = mix(sunTint, mix(vec3(1.0, 0.50, 0.18), vec3(1.0, 0.88, 0.60),
                               smoothstep(-0.02, 0.30, sunDir.y)), desertW);
    // Direct-sun gating: on the TRACED paths (the rings' heightfield sun
    // march, the map's traced shadow) the shadow term already measures
    // whether sunlight reaches this point, so the heightmap burial term
    // (skyVis) must NOT also multiply the sun — that double-count crushed
    // every cliff face below its own crest and dimmed whole hillsides. A
    // cave stays dark because its sun ray hits the roof (shadow -> 0) and
    // its ambient still scales by skyVis. skyVis only gates the sun on the
    // non-ring fallback, whose shadow-march skip relies on it for a
    // pop-free fade.
    float sunGate = ringHit ? 1.0 : skyVis;
    // 1.4 sun strength: the sun's orbit tops out at ~57 deg elevation
    // (app.cpp TILT — an art choice that keeps shadows alive at noon), so
    // flat ground never sees more than ~0.84 of the sun's raw intensity.
    // Compensate here so peak daylight actually reaches full brightness —
    // "even at noon the map looks dull" was this, compounding with the
    // old AO-on-sun and burial double-counts.
    vec3 sunCol  = sunTint * TW_SUN_STR * dayLight * sunGate;
    // AO is ambient/sky occlusion: it shapes the sky + bounce light, NOT the
    // direct sun — the shadow ray answers the sun question. Multiplying both
    // (the old `(ambient + sun) * ao`) under-exposed every sunlit surface
    // near any geometry: a big part of "the map reads dark even at noon". A
    // light contact term is kept on the sun so crevices don't glow under
    // grazing light. Ring hits keep FULL AO on the sun: their baked AO
    // stands in for canopy self-shadow the heightfield march can't see.
    // (AO still never touches the local lights added below — a deep cave's
    // AO must not crush the player's own lantern.)
    float sunAo = ringHit ? ao : mix(1.0 - TW_AO_SUN, 1.0, ao);
    vec3 light = ambient * ao + sunCol * diff * shadow * sunAo;

    // MOONLIGHT: a dim cool directional term that takes over once the sun is
    // down, so night reads as moonlit navy rather than going black. Ring hits
    // carry the heightfield/traced moon visibility in moonShadowIn
    // (lighting.frag); a non-ring hit would get the sun's 4-tap cone march,
    // gated to actually-moonlit pixels.
    float moonDiff = max(dot(shadeN, moonDir), 0.0);
    float moonShadow = ringHit
                     ? moonShadowIn
                     : ((moonLight > 0.02 && moonDiff > 0.0 && skyVis > 0.10)
                        ? softShadow(hitPos, normal, moonDir) : 1.0);
    // sunAo applies to the moon exactly as to the sun (AO is direction-free):
    // ring hits keep full baked AO as the canopy self-shadow stand-in the
    // heightfield moon march can't see; fine hits keep the light contact term.
    light += MOON_TINT * TW_MOON_STR * moonLight * moonDiff * sunGate * 0.6
           * moonShadow * sunAo;

    // Emissive glow (crystals etc.): trilinearly sample the baked light volume
    // at this point and add it in, so deep caverns are softly lit from within
    // by their own crystals -- the payoff a handheld lantern can't provide.
    // (Window-anchored: REPEAT-wraps out-of-window -> gated for ring hits.)
    if (!ringHit)
        light += texture(lightVol, vec3(hitPos.x / float(gridSize.x),
                                        hitPos.y / float(gridSize.y),
                                        hitPos.z / float(gridSize.z))).rgb;

    // Leaf translucency (sub-surface scattering): when the camera is looking
    // roughly toward the sun and this leaf has the sun on the *back* side,
    // light transmits through it, adding a forward-scattered warm-green glow.
    // This is what gives canopies that rich "lit-from-behind" look in the
    // reference — without it, leaves shade identically to stone and feel flat.
    // Ring canopies skip the ray (it would clip against the near window and
    // return garbage — the whole distant forest would bloom); they get a
    // fixed mild transmittance instead so far foliage keeps a hint of it.
    if (mat == 5u) {
        float forwardSun = max(dot(rd, sunDir), 0.0);
        if (forwardSun > 0.05) {
            // Falling-crown pixels (creaturePix) take the same fixed mild
            // transmittance as ring canopies: their voxels left the brick
            // world at the fell, so the back-lit ray would report open sky
            // and bloom the whole crown.
            float backLit = (ringHit || creaturePix) ? 0.35
                          : (anyHit(hitPos + rd * 1.5, sunDir, 220.0) ? 0.0 : 1.0);
            vec3 transTint = vec3(1.30, 1.45, 0.85);
            light += transTint * backLit * forwardSun * forwardSun * 0.85 * dayLight;
        }
        // Moonlit canopies get the same lit-from-behind pass at moon scale —
        // a cool silver-green rim when looking toward the moon through leaves.
        float forwardMoon = max(dot(rd, moonDir), 0.0);
        if (moonLight > 0.02 && forwardMoon > 0.05) {
            float backLitM = ringHit ? 0.35
                           : (anyHit(hitPos + rd * 1.5, moonDir, 220.0) ? 0.0 : 1.0);
            light += vec3(0.55, 0.75, 0.80) * backLitM * forwardMoon * forwardMoon
                   * 0.35 * moonLight * TW_MOON_STR;
        }
    }

    // Fire lights: each burning cluster casts a warm pool onto nearby surfaces
    // (like a campfire). Same falloff shape as the lantern; added to `light` so
    // it tints the surface albedo and reads at night. Skipped when uFireN == 0.
    for (int i = 0; i < uFireN; ++i) {
        vec3  L    = uFireLights[i].xyz - hitPos;
        float dist = length(L);
        vec3  Ldir = L / max(dist, 0.001);
        float str  = uFireLights[i].w;
        float pool = 1.0 / (1.0 + dist * dist * 0.0020);          // ~few-metre reach
        float cut  = 1.0 - smoothstep(45.0, 95.0, dist);          // soft cutoff
        float face = 0.45 + 0.55 * max(dot(normal, Ldir), 0.0);   // wrap fill
        float flk  = 0.85 + 0.15 * sin(uTime * 9.0 + float(i) * 2.3);
        light += vec3(1.0, 0.46, 0.16) * (pool * str * face * cut * flk);
    }

    // Resolve albedo × total incident light. AO is already folded into the
    // sky+sun term above; local lights are intentionally not sky-occluded.
    vec3 col = base * light;

    // Self-illumination: emissive materials add their own colour after lighting
    // (not sky-occluded). Crystals are still special-cased above; this covers
    // any other emissive material. NOTE: smoothness/metallic feed PR11's
    // ray-traced reflections — there is intentionally NO analytic specular
    // highlight here: a sharp Blinn-Phong glint over the denoised soft shadow
    // traced a faint rim along every block edge.
    col += base * matEmissive(mat);

    // Caustics: voxels with water directly above are lake-bed surfaces.
    // Two animated noise fields multiplied + thresholded give bright
    // squiggly light patterns dancing on the floor, like sun focused
    // through wavy water.
    if (!ringHit && normal.y > 0.5 && getVoxelAt(voxel + ivec3(0, 1, 0)) == 6u) {
        float c1 = blockNoise(vec3(hitPos.x + uTime * 0.45, 0.0, hitPos.z + uTime * 0.27), 0.45);
        float c2 = blockNoise(vec3(hitPos.x - uTime * 0.32, 0.0, hitPos.z + uTime * 0.58), 0.70);
        float caustic = smoothstep(0.35, 0.85, c1 * c2 * 2.0);
        col += vec3(0.45, 0.70, 0.85) * caustic
             * (dayLight * 0.85 + moonLight * TW_MOON_STR * 0.30);
    }

    // Desert dust haze: distant geometry desaturates toward a warm dusty band,
    // giving the open plain depth and making a far mesa/landmark read as "walk
    // toward it". Layered before the reveal-curtain fog; gated by skyVis so cave
    // interiors aren't washed warm.
    if (desertW > 0.01 && mapPass == 0) {   // map diorama: no aerial haze
        float dust = clamp(tHit / 850.0, 0.0, 1.0);
        dust = dust * dust;
        vec3 dustCol = vec3(0.86, 0.74, 0.56) * (0.35 + 0.65 * dayLight);
        col = mix(col, dustCol, dust * 0.5 * desertW * skyVis);
    }
    // (No distance fog: geometry stays opaque all the way out — the fine->ring
    // handoff in gbuffer.frag replaced the old dissolve. Depth cueing, if
    // wanted, is COLOUR-only like the desert dust above, never alpha.)

    return col;
}
