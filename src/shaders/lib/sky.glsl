// Procedural sky + day/night phase + stars + clouds. Depends on noise.glsl
// and the sunDir/moonDir/dayLight/moonLight/uTime uniforms.

// The ONE moonlight colour: every pass that adds a direct-moon term (surface
// diffuse, water glitter, reflections, horizon, air) tints with this, so a
// moonlit scene stays a single coherent cool cast. Scaled by TW_MOON_STR.
const vec3 MOON_TINT = vec3(0.40, 0.52, 0.85);

// Day/night phase weights from the sun's elevation. `dusk` is the thin warm
// band as the sun crosses the horizon (sunrise AND sunset); the three sum ~1.
void skyPhase(out float dayW, out float duskW, out float nightW) {
    float se = sunDir.y;
    dayW   = smoothstep(-0.04, 0.20, se);
    nightW = smoothstep( 0.04, -0.16, se);
    duskW  = max(1.0 - dayW - nightW, 0.0);
}
// Twinkling star at a hashed 3D cell. Sparse, crisp points; brighter stars are
// rarer. uTime drives a gentle per-star twinkle.
float starField(vec3 dir) {
    vec3 p = dir * 230.0;
    vec3 i = floor(p), f = fract(p);
    float h = hash31(i);
    if (h < 0.965) return 0.0;
    vec3 sp = vec3(hash31(i + 1.3), hash31(i + 2.7), hash31(i + 4.1));
    float d = length(f - sp);
    float core = smoothstep(0.16, 0.0, d);
    float tw   = 0.55 + 0.45 * sin(uTime * 2.6 + h * 90.0);
    return core * (0.35 + (h - 0.965) / 0.035) * tw;
}
vec3 baseSkyColor(vec3 rd) {
    float up = clamp(rd.y * 0.5 + 0.5, 0.0, 1.0);   // 0 nadir .. 1 zenith
    float dayW, duskW, nightW; skyPhase(dayW, duskW, nightW);

    // Desert sky: blend the day+dusk palette toward a desert look by the CAMERA's
    // desert weight (the same normalized climate field the ground's per-hit
    // desertW uses, so sky + terrain commit to "desert" in lockstep). A pure
    // function of camPos.xz -> the dome cross-fades smoothly as you walk and is
    // frozen under pure rotation (no shimmer); Dsky=0 keeps the meadow sky
    // byte-identical, and night is left untouched.
    float Dsky = smoothstep(0.12, 0.85, biomeWeights(vec2(camPos.x, camPos.z)).w);

    vec3 dayHoriz   = mix(vec3(0.66, 0.82, 1.00), vec3(0.90, 0.84, 0.72), Dsky);  // cool haze -> warm pale dust
    vec3 dayZen     = mix(vec3(0.22, 0.48, 0.92), vec3(0.15, 0.40, 0.86), Dsky);  // -> deep dry cobalt
    vec3 duskHoriz  = mix(vec3(0.98, 0.49, 0.26), vec3(1.00, 0.60, 0.34), Dsky);  // -> amber-rose golden hour
    vec3 duskZen    = mix(vec3(0.16, 0.18, 0.40), vec3(0.22, 0.19, 0.34), Dsky);  // -> dusty mauve
    vec3 nightHoriz = vec3(0.035, 0.05, 0.11);   // night unchanged (clear cold desert dome)
    vec3 nightZen   = vec3(0.010, 0.015, 0.045);

    vec3 horiz = dayW*dayHoriz + duskW*duskHoriz + nightW*nightHoriz;
    vec3 zen   = dayW*dayZen   + duskW*duskZen   + nightW*nightZen;
    vec3 col = mix(horiz, zen, up);

    // Directional sunrise/sunset fire: a warm band concentrated toward the
    // sun's compass bearing, hugging the horizon, strongest at low sun.
    vec2 rdAz  = normalize(vec2(rd.x, rd.z) + 1e-5);
    vec2 sunAz = normalize(vec2(sunDir.x, sunDir.z) + 1e-5);
    float toSun = max(dot(rdAz, sunAz), 0.0);
    float band  = exp(-abs(rd.y) * 5.5);
    col += mix(vec3(1.0, 0.42, 0.14), vec3(1.0, 0.50, 0.24), Dsky)   // amber-rose over desert
         * duskW * pow(toSun, 2.5) * band * (0.9 - 0.30 * Dsky);     // trimmed so it doesn't double-warm the warm desert horizon
    return col;
}
// The cloud field over a world XZ position — ONE definition shared by the
// sky dome (visible clouds) and cloudShadow (their shadows on the ground),
// so a shadow patch always sits under the cloud that casts it.
float cloudCoverAt(vec2 worldXZ) {
    vec2 cuv = worldXZ * 0.0033 + vec2(uTime * 0.06, uTime * 0.022);
    float n = vnoise2(cuv) * 0.55 + vnoise2(cuv * 2.1 + 17.0) * 0.27
            + vnoise2(cuv * 4.3 + 31.0) * 0.13 + vnoise2(cuv * 8.7 + 47.0) * 0.05;
    return smoothstep(0.50, 0.78, n);
}

// Light dimming at a world point from the cloud layer: project along the
// light direction to the cloud plane and sample the SAME field the sky
// renders. Soft big moving patches of dim — the "weather even on a clear
// day" cue. Directional so the sun channel AND the moon channel each get
// clouds along their own body's bearing.
float cloudShadowDir(vec3 p, vec3 ld) {
    if (ld.y < 0.03) return 1.0;                   // grazing light: shadow is off-plane
    // Cloud plane anchored to the WORLD (surface + 220), not the camera: a
    // camera-relative plane made the shadow field slide across the ground
    // whenever the map camera zoomed.
    float tc = max((surfaceY + 220.0 - p.y) / ld.y, 0.0);
    return 1.0 - TW_CLOUD_SHAD * cloudCoverAt(p.xz + ld.xz * tc);
}
float cloudShadow(vec3 p) { return cloudShadowDir(p, sunDir); }

// SOFT cloud dimming for the AIR (fog / haze in-scatter): first noise octave
// only, gentler curve, weaker depth. A kilometres-long air path averages over
// small cloud puffs — point-sampling the full sharp field along view rays
// painted its fine octaves as dark vertical "rain curtain" streaks across
// the fog and sky. Surfaces keep the sharp cloudShadow above; air must only
// ever see broad, slow weather.
float cloudShadowSoftDir(vec3 p, vec3 ld) {
    if (ld.y < 0.03) return 1.0;
    float tc = max((surfaceY + 220.0 - p.y) / ld.y, 0.0);   // world-anchored (see cloudShadowDir)
    vec2 cuv = (p.xz + ld.xz * tc) * 0.0033
             + vec2(uTime * 0.06, uTime * 0.022);
    float cover = smoothstep(0.42, 0.90, vnoise2(cuv));
    return 1.0 - TW_CLOUD_SHAD * 0.64 * cover;   // air: ~2/3 of ground depth
}
float cloudShadowSoft(vec3 p) { return cloudShadowSoftDir(p, sunDir); }

vec3 skyColor(vec3 rd) {
    vec3 col = baseSkyColor(rd);
    float dayW, duskW, nightW; skyPhase(dayW, duskW, nightW);

    // STARS + a faint tilted milky-way band, fading in with night, hidden
    // right at the horizon.
    if (nightW > 0.001 && rd.y > -0.04) {
        float hf = smoothstep(-0.02, 0.10, rd.y);
        float s  = starField(rd);
        float mwBand = exp(-pow(dot(rd, normalize(vec3(0.40, 0.22, -0.89))), 2.0) * 7.0);
        float mw = mwBand * (0.10 + 0.16 * vnoise2(rd.xz * 6.0 + rd.y * 4.0));
        col += (vec3(0.92, 0.95, 1.0) * s + vec3(0.16, 0.20, 0.30) * mw) * nightW * hf;
    }

    // CLOUDS (plane projection), lit by phase: warm at golden hour, silver-on-
    // slate by moonlight at night. The plane is WORLD-anchored like the
    // shadows (cloudShadow) so a patch always sits under the cloud that casts
    // it; a camera above the deck (high map zoom-out) sees clear sky, which
    // is physically what being above the layer means.
    float cloudT = (surfaceY + 220.0 - camPos.y) / rd.y;
    if (rd.y > 0.04 && cloudT > 0.0) {
        float cover = cloudCoverAt(camPos.xz + rd.xz * cloudT);
        float sunAmt = max(dot(rd, sunDir), 0.0);
        vec3 cloudCol = mix(vec3(1.0), vec3(1.0, 0.62, 0.34), pow(sunAmt, 4.0) * (dayW + duskW));
        float moonAmt = max(dot(rd, moonDir), 0.0);
        vec3 nightCloud = vec3(0.15, 0.17, 0.24) + vec3(0.45) * pow(moonAmt, 4.0) * moonLight;
        cloudCol = mix(cloudCol, nightCloud, nightW);
        cloudCol *= mix(vec3(0.30, 0.35, 0.45), vec3(1.0), dayW);
        float horizonFade = smoothstep(0.04, 0.22, rd.y);
        col = mix(col, cloudCol, cover * horizonFade * 0.85);
    }

    // SUN: a soft disc that whitens high and swells deep-orange near the
    // horizon, with a tight bloom and a broad warm halo.
    float sd    = max(dot(rd, sunDir), 0.0);
    float sunUp = smoothstep(-0.12, 0.02, sunDir.y);
    float Dsun  = smoothstep(0.12, 0.85, biomeWeights(vec2(camPos.x, camPos.z)).w);
    vec3  sunCol = mix(vec3(1.0, 0.40, 0.14),
                       mix(vec3(1.0, 0.96, 0.84), vec3(1.0, 0.95, 0.80), Dsun),  // harsher dry-air near-white noon
                       smoothstep(-0.04, 0.28, sunDir.y));
    float discInner = mix(0.99955, 0.99900, duskW);       // softer + bigger low
    float sunDisc = smoothstep(discInner - 0.0007, discInner, sd);
    col = mix(col, sunCol, sunDisc * sunUp);
    col += sunCol * pow(sd, 1200.0) * 3.0 * sunUp;        // tight bloom
    col += vec3(1.0, 0.70, 0.40) * pow(sd, 6.0) * 0.10 * (dayW + duskW);

    // MOON: pale disc with crater mottling, crisp glow + soft halo, up at night.
    float md     = max(dot(rd, moonDir), 0.0);
    float moonUp = smoothstep(-0.06, 0.06, moonDir.y);
    if (moonUp > 0.001) {
        float moonDisc = smoothstep(0.99950, 0.99974, md);
        float cr = 0.80 + 0.20 * vnoise2(rd.xz * 64.0 + rd.y * 40.0);
        vec3  moonCol = vec3(0.86, 0.89, 0.96) * cr;
        col = mix(col, moonCol, moonDisc * moonUp);
        col += vec3(0.70, 0.78, 0.95) * pow(md, 900.0) * 1.1 * moonUp;
        col += vec3(0.35, 0.42, 0.60) * pow(md, 6.0) * 0.06 * moonUp * nightW;
    }
    return col;
}
