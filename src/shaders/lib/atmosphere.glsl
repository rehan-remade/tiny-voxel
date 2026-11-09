// Unified analytic atmosphere.
//   Step 1: DISTANCE SCATTERING — per-channel extinction + directional,
//           sky-coupled in-scatter (cool blue away from the sun, warming
//           through the forward-scatter phase lobe toward it).
//   Step 2: HEIGHT FOG — closed-form exponential valley mist with a CPU
//           dawn/dusk rhythm (fogDensity/fogFloor uniforms from app.cpp).
//   Step 3: LIGHT x AIR — the fog and the haze's sun-glow are SHADOWED:
//           sunlight visibility (terrain via the heightfield march + the
//           cloud layer) is sampled along the view path, so mist is bright
//           where sun strikes it and dim in a mountain's or cloud's shade,
//           and the sunward glow forms real shafts instead of a uniform lobe.
// COLOUR-only on every opaque hit class (fine / ring / horizon) + the sky's
// fog tail — geometry stays opaque (the translucent terrain-fog artifact
// class must never return). Deterministic -> TAA-stable.
// Depends on lib/sky.glsl (baseSkyColor, cloudShadow) + the far-heightfield
// and sunDir/dayLight uniforms (declared by the including shader).

// Cheap forward-scatter lobe (Henyey-Greenstein-shaped): how much the air
// between eye and surface glows when the view runs toward the sun.
float atmoPhase(vec3 rd) {
    float mu = max(dot(rd, sunDir), 0.0);
    return pow(mu, 8.0);
}

float farHeightAt(vec2 wxz) {
    vec2 uv = (wxz - farOrigin) / (farStride * farTexels);   // 0..1 over the region
    return texture(farHeightTex, clamp(uv, 0.0, 1.0)).r;
}

// Budgeted directional-light visibility for a world point: a short march over
// the far heightfield toward the light (~10 texture taps, exponentially spaced
// so a nearby ridge casts a crisp edge and a distant mountain still registers,
// out to ~600m). Deterministic (no jitter) -> temporally stable under TAA.
// `lift` raises the start point above the hit: ring terrain is floor-
// QUANTIZED below the continuous heightfield, so an unlifted march
// self-shadows flat ground. Sun and moon both march through here.
float farVisDir(vec3 p, float lift, vec3 ld) {
    if (farTexels < 0.5 || ld.y < 0.02) return 1.0;
    vec3 ro = vec3(p.x, p.y + lift, p.z);
    float vis = 1.0;
    float t = farStride;
    for (int i = 0; i < 10; ++i) {
        vec3 q = ro + ld * t;
        float h = farHeightAt(q.xz);
        // Penumbra from how deep the light ray dips below terrain vs distance.
        vis = min(vis, 1.0 - clamp((h - q.y) / (0.15 * t), 0.0, 1.0));
        if (vis < 0.02) break;
        t *= 1.65;
    }
    // Keep a floor: heightfield shadows are an approximation — full black
    // reads as a hole at 500m, a deep dim reads as terrain in shade.
    return 0.25 + 0.75 * vis;
}
float farSunVis(vec3 p, float lift) { return farVisDir(p, lift, sunDir); }

// ---- Height fog -------------------------------------------------------------
uniform float fogDensity;   // extinction at the fog floor (0 = clear air)
uniform float fogFloor;     // world Y where the fog is thickest

#define FOG_H TW_FOG_H      // scale height: fog is 1/e as dense FOG_H/10 m up

float fogOpticalDepth(vec3 ro, vec3 rd, float dist) {
    // integral of fogDensity * exp(-(y(t) - fogFloor)/FOG_H) dt over [0, dist]
    // (clamped so a camera far below the floor can't overflow the exp).
    float a  = max((ro.y - fogFloor) / FOG_H, -8.0);
    float mu = rd.y / FOG_H;
    float od = (abs(mu) < 1e-5)
             ? exp(-a) * dist
             : (exp(-a) - exp(-min(a + mu * dist, 8.0) )) / mu;
    return fogDensity * max(od, 0.0);
}

// Sunlight reaching the AIR: ONE near-field sample of the heightfield sun
// march x the broad cloud factor. Deliberately smooth and local — nearby
// valley/ridge shading and slow weather only. Two failed versions taught us
// the constraint: sparse point samples of visibility kilometres along the
// ray CANNOT reconstruct coherent light shafts (3 samples painted cloud
// octaves as vertical rain curtains; at low sun the distant heightfield
// marches went blotchy per-azimuth). Coherent crepuscular shafts would need
// a pass that integrates the whole path (no such pass exists today — the
// single-sample locality here is a deliberate cost bound). The air only
// needs "am I in a shaded valley / under a cloud bank", which one sample
// within ~60m answers smoothly.
float airVisDir(vec3 ro, vec3 rd, float dist, vec3 ld) {
    vec3 p = ro + rd * (min(dist, 1200.0) * 0.5);
    return farVisDir(p, 12.0, ld) * cloudShadowSoftDir(p, ld);
}
float airSunVis(vec3 ro, vec3 rd, float dist) {
    return airVisDir(ro, rd, dist, sunDir);
}

vec3 fogColor(vec3 rd, float sunVis, float moonVis) {
    // Bright mist, warmed by the sun phase — but only where sunlight actually
    // REACHES the air (sunVis): dawn fog looking sunward goes gold in the lit
    // bands and stays cool pearl in terrain/cloud shade. Shadowed mist dims
    // toward a cool grey rather than glowing on its own.
    vec3 c = mix(vec3(0.72, 0.78, 0.87), vec3(1.05, 0.88, 0.62),
                 atmoPhase(rd) * 0.75 * sunVis);
    c *= mix(0.42, 1.0, sunVis);
    // Moonlit mist: the sun treatment at moon scale — a cool silver bank,
    // brighter toward the moon's bearing, dim in a ridge's/cloud's shade
    // (moonVis), so night fog reads as moonlit rather than self-luminous grey.
    float phaseM = pow(max(dot(rd, moonDir), 0.0), 8.0);
    vec3 night = MOON_TINT * (0.40 + 0.60 * phaseM)
               * mix(0.42, 1.0, moonVis) * moonLight * TW_MOON_STR;
    return c * (0.10 + 0.90 * dayLight) + night;
}

// Fog on SKY pixels: the mist bank must not end at the last terrain hit —
// low view rays keep accumulating fog "to infinity" (capped at the horizon
// melt distance), so a fog sea meets the sky seamlessly.
vec3 applyFogSky(vec3 col, vec3 rd) {
    if (fogDensity <= 0.00001) return col;
    float f = 1.0 - exp(-fogOpticalDepth(camPos, rd, 30000.0));
    if (f < 0.002) return col;
    float mv = (moonLight > 0.01) ? airVisDir(camPos, rd, 24000.0, moonDir) : 0.0;
    return mix(col, fogColor(rd, airSunVis(camPos, rd, 24000.0), mv), f);
}

vec3 applyAtmosphere(vec3 col, vec3 rd, float dist) {
    // dist is in VOXELS (10cm each — the aerialPerspective units lesson).
    // Rayleigh-shaped extinction, art-directed magnitude: untouched at 100m,
    // ~35% air at 800m, ~65% by 2km, converged to sky by the heightfield's
    // 2.4-3km melt band. (2.5e-5 and 4.2e-5 both read as too-late arrival.)
    vec3 SIGMA = vec3(0.60, 0.85, 1.15) * (TW_ATMO_DENS * 1.0e-5);
    vec3 T = exp(-SIGMA * dist);
    // Light scattered INTO the path: base cool sky-blue blended with the live
    // sky in this view direction, warmed toward the sun by the phase lobe.
    // The WARM (sun) part is gated by mid-path sun visibility, so the sunward
    // glow breaks into shafts where ridges or clouds shade the air — the base
    // blue skylight component stays unshadowed (the upper air is always lit).
    // Night air goes dark (clear starry distance), not milky.
    float phase = atmoPhase(rd) * TW_ATMO_WARM;
    if (phase > 0.01) phase *= airSunVis(camPos, rd, dist);   // near-field, smooth
    vec3 skyIn = mix(vec3(0.58, 0.70, 0.88), baseSkyColor(rd), TW_ATMO_BLUE);
    vec3 inCol = mix(skyIn, vec3(1.0, 0.82, 0.58), phase);
    inCol *= (0.06 + 0.94 * dayLight);
    // Moonlit air: the same phase-lobe treatment toward the moon at moon
    // scale — distance picks up a faint cool silver, and the moonward haze
    // breaks into shade where ridges/clouds block the moon. Kept well below
    // the day magnitudes so clear night distance stays dark, not milky.
    if (moonLight > 0.01) {
        float phaseM = pow(max(dot(rd, moonDir), 0.0), 8.0) * TW_ATMO_WARM;
        if (phaseM > 0.01) phaseM *= airVisDir(camPos, rd, dist, moonDir);
        inCol += MOON_TINT * (0.08 + 0.30 * phaseM) * moonLight * TW_MOON_STR;
    }
    col = col * T + inCol * (1.0 - T);
    // Height fog on top of the clear-air scatter (fog is the denser, lower,
    // time-of-day layer; scatter is the permanent kilometre-scale air).
    if (fogDensity > 0.00001) {
        float f = 1.0 - exp(-fogOpticalDepth(camPos, rd, dist));
        if (f > 0.002) {
            float mv = (moonLight > 0.01) ? airVisDir(camPos, rd, dist, moonDir) : 0.0;
            col = mix(col, fogColor(rd, airSunVis(camPos, rd, dist), mv), f);
        }
    }
    return col;
}
