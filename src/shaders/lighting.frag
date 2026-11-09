#version 430 core
// Deferred lighting pass. Reads the G-buffer (gbuffer.frag's output) and runs
// exactly the same shading the forward voxel.frag used to do inline — only the
// inputs now come from textures instead of a per-pixel raymarch. Reconstructs
// the ray, hit position, normal, material, and voxel, then calls shadeHit().
in vec2 vUV;
out vec4 FragColor;

uniform sampler2D gPosMat;   // xyz world hit pos, w material id (-1 = miss/sky)
uniform sampler2D gNormal;   // xyz world normal, w brick-step count (heat)

uniform vec3  camPos;
uniform vec3  camFwd;
uniform vec3  camRight;
uniform vec3  camUp;
uniform vec2  viewport;
uniform float tanHalfFov;
uniform ivec3 gridSize;
uniform ivec3 bricksPerAxis;
uniform ivec2 windowOriginB; // toroidal brick-window origin (voxel_trace addressing)
uniform vec3  sunDir;
uniform float dayLight;
uniform vec3  moonDir;       // opposite the sun; up at night (day/night model)
uniform float moonLight;     // 0..1 cool moonlight strength (peaks at midnight)
uniform float uTime;
uniform float surfaceY;
uniform sampler2D heightTex;
uniform sampler3D lightVol;
// Dynamic fire lights: warm point lights aggregated from burning voxels on the
// CPU (fire.cpp / app.cpp). xyz = world position, w = strength. uFireN == 0
// (no fire) skips the loop entirely.
uniform int   uFireN;
uniform vec4  uFireLights[24];
// Rigid creature parts (lib/creature_trace.glsl): wildlife isn't in the
// brick SSBOs, so world shadow rays can't see it. The lighting pass marches
// the posed part volumes along the sun ray for REAL cast + self shadows
// (creatureSunVis), plus a faint contact blob. uCreN == 0 skips everything.
#include "lib/creature_trace.glsl"
uniform int   uCreN;
uniform vec4  uCreBound[32];    // xyz world bounding centre, w radius
uniform ivec2 uCreRange[32];    // x first part instance, y part count
                                // (32 = FrameView::MAX_CREATURES)
// Bounds [x, x+y) skip the CONTACT BLOB only (falling trees: a tree-sized
// bound would paint a giant dark disc; their traced shadows still run).
uniform ivec2 uCreBlobSkip;
// G-buffer albedo: creature pixels (mat 38) shade with a WHITE material base
// and multiply this in afterwards — the baked palette colour.
uniform sampler2D gAlbedoTex;

// Any-hit through every posed creature near ro along one direction:
// per-creature sphere reject, then march each bone part with the same trace
// the G-buffer pass renders with — occlusion exactly matches the visible
// pose. The soft wrapper (creatureSunVis, after the lib includes) averages
// this over the same jittered sun cone terrain shadows use.
bool creatureAnyHit(vec3 ro, vec3 sd) {
    for (int i = 0; i < uCreN; ++i) {
        vec3 rel = uCreBound[i].xyz - ro;
        float r  = uCreBound[i].w + 1.0;
        float b  = dot(rel, sd);
        // Ray-sphere reject: origin outside and closest approach beyond r.
        if (dot(rel, rel) - b * b > r * r) continue;
        if (b < 0.0 && dot(rel, rel) > r * r) continue;
        for (int pi = 0; pi < uCreRange[i].y; ++pi) {
            float t; vec3 nn; uint ci; vec3 cc;
            if (creTracePart(uCreRange[i].x + pi, ro, sd, 1e9,
                             0, vec3(0.0), vec3(0.0), t, nn, ci, cc))
                return true;
        }
    }
    return false;
}

// Declared here (not with the map block below): playerGhost reads it, and
// GLSL wants the declaration before first use.
uniform int   uPlayerAvatar; // 1 = the rigged voxel avatar renders through the
                             // creature pass; skip the analytic box character
                             // (the stud marker + trail below stay)

// Closest-hit through the AVATAR's parts only (the appended last bound when
// uPlayerAvatar == 1) — the buried-marker tint and the X-ray ghost draw a
// pose-accurate voxel silhouette of the real character, not the retired
// analytic box figure. No clip slab: the ghost must show through the Q/E
// slice exactly like it shows through terrain (find yourself anywhere).
bool playerGhost(vec3 ro, vec3 rd, out float tHit, out vec3 col) {
    tHit = 1e9;
    if (uPlayerAvatar == 0 || uCreN == 0) return false;
    int i = uCreN - 1;
    vec3 rel = uCreBound[i].xyz - ro;
    float r  = uCreBound[i].w + 1.0;
    float b  = dot(rel, rd);
    if (dot(rel, rel) - b * b > r * r) return false;
    if (b < 0.0 && dot(rel, rel) > r * r) return false;
    for (int pi = 0; pi < uCreRange[i].y; ++pi) {
        float t; vec3 nn; uint ci; vec3 cc;
        if (creTracePart(uCreRange[i].x + pi, ro, rd, tHit,
                         0, vec3(0.0), vec3(0.0), t, nn, ci, cc) && t < tHit) {
            tHit = t;
            col  = cc;
        }
    }
    return tHit < 1e8;
}
// Explored-world map: 1 = shade the surface diorama (forces the
// ringHit analytic path — flat sun+ambient, no window-anchored reads — so cut
// interiors don't burial-darken to black), 2 = shade the cave X-ray layer
// (discovered underground space as depth-tinted glow). Both draw the analytic
// player pin and transparent misses. Also read by lib/lighting.glsl.
uniform int   mapPass;
uniform vec3  mapPlayerPos;  // character feet (world voxels)
uniform float mapPlayerYaw;  // facing (player fwd = cos/sin yaw)
uniform float mapWalkPhase;  // stride phase; negative = idle pose
uniform vec3  mapMoveTarget; // click-to-move destination (marker)
uniform int   mapMoveOn;
// Farm Mode cursor footprint: a translucent slab over the plot cell under the
// cursor, tinted green (valid for the active tool) or red (invalid).
uniform int   mapFarmOn;     // 1 = draw the footprint
uniform int   mapFarmOk;     // 1 = tool valid here (green) else red
uniform vec3  mapFarmMin;    // cell min corner (world voxels: x, topY, z)
uniform float mapFarmSize;   // cell side length (voxels)
// God-harvest: the hovered ripe crop glows (its voxel bbox, gated on crop
// materials), and a warm "divine touch" ring blooms on the ground on pick.
uniform int   mapHoverOn;
uniform vec3  mapHoverLo;    // hovered crop bbox (world voxels)
uniform vec3  mapHoverHi;
uniform int   mapRingOn;
uniform vec3  mapRingPos;    // ring centre (ground)
uniform float mapRingT;      // 1 -> 0 (life)
uniform float mapCover;      // surface-layer coverage (1.0: the diorama is the frame)
uniform float mapGhost;      // X-ray dims the surface by BRIGHTNESS (~0.12), not
                             // alpha: single renderer, nothing behind to show through
uniform float mapStandoff;   // orbit camera-to-target distance:
                             // atmosphere accrues only beyond it, so zoom != weather
uniform int  mapPathN;       // click-to-move route trail (0 = none)
uniform vec4 mapPath[32];    // x,y = world XZ; z = world Y; w = cumulative length
// Forestry hover highlight (diorama): a 0/1 R8 mask over the hovered tree's
// bbox. Gated by material in the apply block so the generator replay's
// over-emit (cells that are really terrain) can never tint the ground.
uniform int       uHoverOn;
uniform vec3      uHoverLo;         // mask origin, world voxels
uniform vec3      uHoverDims;
uniform float     uHoverStrength;   // eased 0..1
uniform sampler3D uHoverMask;

bool hoverMaskAt(vec3 wp) {
    vec3 t = (wp - uHoverLo) / uHoverDims;
    if (any(lessThan(t, vec3(0.0))) || any(greaterThanEqual(t, vec3(1.0))))
        return false;
    return texture(uHoverMask, t).r > 0.5;
}
uniform usampler2D mapExploredTex;   // R8UI chunk flags (bit0 surface seen)
uniform sampler2D  mapExploredSoftTex; // prefiltered ramp: bit0 box-blurred 5x5
                                       // chunks (buildMasks), bilinear R8,
                                       // border color 0 = darkness beyond
uniform ivec2 mapMaskOrigin;         // mask texel (0,0) in world chunk coords
const int MAP_MASK_CH = 384;         // chunks per mask side (= MapMasks::MASK_CH)

// (mapExploredTex — the raw integer flags — is kept declared for unit-8
// binding discipline and future per-flag reads; the brightness ramp itself
// now comes from the prefiltered soft texture below.)
// Wide fog-of-war falloff (~5-chunk / 320-voxel ramp at the frontier): ONE
// bilinear tap of the pre-blurred field. buildMasks bakes the old 25-tap box
// into the R8 texture; hardware bilinear between texel (= chunk) centres
// reproduces the old mapExploredAt interpolation exactly — this replaced a
// 100-texelFetch reconstruction that ran on EVERY map hit pixel.
float mapExploredSoft(vec2 wxz) {
    vec2 uv = (wxz / 64.0 - vec2(mapMaskOrigin)) / float(MAP_MASK_CH);
    return texture(mapExploredSoftTex, uv).r;
}
uniform vec2  jitterNdc;      // sub-pixel TAA jitter in NDC (must match gbuffer.frag)
// Far-field LoD heightfield: sampled by the atmosphere's sun-shadow marches
// (farSunVis: terrain-cast shadows + shadowed fog/haze under the live sun).
uniform sampler2D farHeightTex;
uniform vec2  farOrigin;      // world-voxel MIN corner of the far region
uniform float farStride;      // world voxels per far texel
uniform float farTexels;      // FAR_TEX (0 = disabled)

#include "lib/tweaks.glsl"
#include "lib/noise.glsl"
#include "lib/voxel_trace.glsl"
#include "lib/materials.glsl"
#include "lib/sky.glsl"
#include "lib/atmosphere.glsl"
#include "lib/tonemap.glsl"
#include "lib/lighting.glsl"

// Creature sun visibility with the SAME 4-tap jittered cone softShadow uses
// for terrain, so a stag's shadow edge and the tree's shadow edge beside it
// carry an identical penumbra — one scene, one look. Each tap is a full
// any-hit through the posed parts (sphere-rejected, so the common case is
// four dot products per creature).
float creatureSunVis(vec3 p, vec3 n, vec3 sd, vec3 jitterKey) {
    if (uCreN == 0) return 1.0;
    vec3 ro = p + n * 0.75 + sd * 0.75;
    vec3 up = abs(sd.y) > 0.9 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
    vec3 t1 = normalize(cross(sd, up));
    vec3 t2 = cross(sd, t1);
    const float CONE = 0.020;   // matches softShadow (~1.1° half-angle)
    float occ = 0.0;
    for (int i = 0; i < 4; i++) {
        vec2 fixedOff = vec2(
            (i == 0 || i == 2) ?  1.0 : -1.0,
            (i == 0 || i == 1) ?  1.0 : -1.0);
        // Jitter keyed to jitterKey, not ro: for a MOVING surface (falling
        // trees) the caller passes an object-anchored key so the 4-tap
        // dither is frozen on the model instead of re-rolling every frame
        // (per-frame 25% shadow quanta the map TAA can never converge).
        vec2 jitter = (vec2(
            hash31(jitterKey * 17.3 + float(i) * 7.0),
            hash31(jitterKey * 23.1 + float(i) * 13.0)
        ) - 0.5) * 0.5;
        vec3 sdir = normalize(sd + (t1 * (fixedOff.x + jitter.x)
                                  + t2 * (fixedOff.y + jitter.y)) * CONE);
        occ += creatureAnyHit(ro, sdir) ? 1.0 : 0.0;
    }
    return 1.0 - occ * 0.25;
}

// Click-to-move route trail: gold marching dashes along the mapPath
// polyline (w = distance-to-DESTINATION, so the pattern anchors to the
// fixed end and stays put while the character advances or waypoints are
// consumed). Returns 0..1 dash intensity at a world position. yUp/yDn
// bound the vertical window around the route chord (canopy above the path
// must not catch the ground trail); pass 0/0 to skip the gate (the X-ray
// cast sits below the surface-anchored route).
float trailGlow(vec2 xz, float yRef, float yUp, float yDn) {
    if (mapPathN < 2) return 0.0;
    float dBest = 1e9, lBest = 0.0, dyBest = 0.0;
    for (int i = 0; i + 1 < mapPathN; ++i) {
        vec2 a = mapPath[i].xy, b = mapPath[i + 1].xy;
        vec2 ab = b - a;
        float len2 = max(dot(ab, ab), 1e-6);
        float s = clamp(dot(xz - a, ab) / len2, 0.0, 1.0);
        float dd = length(xz - (a + ab * s));
        float yq = mix(mapPath[i].z, mapPath[i + 1].z, s);
        float dy = yRef - yq;
        if (yUp > 0.0 && (dy > yUp || dy < -yDn)) continue;
        if (dd < dBest) {
            dBest  = dd;
            dyBest = abs(dy);
            lBest  = mix(mapPath[i].w, mapPath[i + 1].w, s);
        }
    }
    // Soft vertical falloff inside the hard gate: a cliff face beside the
    // route fades out instead of wearing a full-height gold stripe.
    float vFade = 1.0 - smoothstep(6.0, 14.0, dyBest);
    float core  = 1.0 - smoothstep(0.55, 1.05, dBest);
    // Dashes hop in discrete half-period steps: the map TAA neighbour
    // clamp smears continuous sub-pixel motion; frame-flips stay crisp
    // (same law as the character's quantized bob).
    float ph   = floor(uTime * 2.0) * 0.5;
    float dash = step(fract(lBest * 0.10 + ph), 0.62);
    return core * dash * vFade;
}

// (farHeightAt + farSunVis live in lib/atmosphere.glsl — the fog and
// in-scatter sample sunlight along the ray, so the atmosphere owns them.)
uniform int slopeShade;   // /slopeshade: macro terrain slope normals for diffuse

// --- Slope shading (the "accordion" fix) -----------------------------------
// Six-direction face normals make a hillside strobe between "all treads lit"
// and "all risers dark" as the viewpoint moves (low sun in front: every riser
// ~0, every tread ~full). The cure is a SECOND normal used only for diffuse:
// the macro slope of the terrain, while geometry/silhouettes keep the exact
// voxel faces. Reconstructed in SCREEN space from the G-buffer positions, so
// its resolution automatically matches the pixel footprint at every distance
// (a fixed world-space gradient — the rejected stride-64 attempt — shows its
// own texel lattice as patchwork facets). Strict bilateral gates: neighbours
// must be hits of the SAME ring level, terrain materials, and similar depth,
// else the pixel keeps its face normal (silhouettes/cliff edges untouched).
// (v1 reconstructed the slope from neighbouring G-buffer positions in screen
// space — its depth-rejection gate refused nearly every sample on grazing
// slopes, exactly where the accordion is worst. v2 samples the height fields
// directly in WORLD space: per-voxel heightTex for fine hits — full window
// resolution, so no stride-64 patchwork — and the far heightfield for rings.)
float slopeHeightW(vec2 wxz) {
    return texture(heightTex, wxz / vec2(float(gridSize.x), float(gridSize.z))).r;
}
vec3 slopeShadeNormal(vec3 p0, vec3 geoN, float tHit, uint matFull) {
    // Engage from a couple of metres — the accordion is worst on the hill
    // RIGHT in front of the player. Never fully replaces the facet.
    float wgt = smoothstep(16.0, 90.0, tHit) * 0.85;
    if (wgt < 0.01) return geoN;
    bool ring = (matFull >= 256u);
    vec3 n2;
    if (ring) {
        float e = farStride;                     // stride-64 far field
        n2 = normalize(vec3(
            farHeightAt(p0.xz - vec2(e, 0.0)) - farHeightAt(p0.xz + vec2(e, 0.0)),
            2.0 * e,
            farHeightAt(p0.xz - vec2(0.0, e)) - farHeightAt(p0.xz + vec2(0.0, e))));
    } else {
        float e = 3.0;                           // +-3 voxels: the LOCAL slope
        n2 = normalize(vec3(
            slopeHeightW(p0.xz - vec2(e, 0.0)) - slopeHeightW(p0.xz + vec2(e, 0.0)),
            2.0 * e,
            slopeHeightW(p0.xz - vec2(0.0, e)) - slopeHeightW(p0.xz + vec2(0.0, e))));
    }
    // A sharp DROP beside the hit (cliff edge, dug pit) makes the gradient
    // lie about the local surface — reject when the slope disagrees hard
    // with the face (also keeps true cliff walls crisp).
    wgt *= clamp(dot(n2, geoN) * 1.4, 0.0, 1.0);
    return normalize(mix(geoN, n2, wgt));
}

// (aerialPerspective was replaced by applyAtmosphere — lib/atmosphere.glsl:
// real extinction + directional sky-coupled in-scatter instead of a flat
// luminance/haze mix. Same invariants: colour-only, every opaque hit class.)

void main() {
    vec2 ndc = vUV * 2.0 - 1.0 + jitterNdc;   // match gbuffer's jittered ray
    float aspect = viewport.x / viewport.y;
    vec3 rd = normalize(camFwd
        + camRight * ndc.x * aspect * tanHalfFov
        + camUp    * ndc.y * tanHalfFov);

    vec4  pm    = texture(gPosMat, vUV);
    bool  miss  = (pm.w < 0.0);
    vec3  hitPos = pm.xyz;
    float tHit  = miss ? 1e9 : length(hitPos - camPos);

    // The player character + click-to-move destination: analytic AABB stacks
    // like the old diamond pin, depth-tested against the diorama hit and
    // folded into pinT/pinCol so every downstream branch (sky-front,
    // terrain-front, buried ghost, X-ray bleed) keeps working unchanged.
    // The character is a blocky humanoid at the player's feet, oriented by
    // yaw, with a QUANTIZED two-frame walk cycle — analytic geometry has no
    // motion vectors, and discrete pose flips stay crisp under the map TAA
    // where any smooth sub-pixel drift would smear. UI-colored (no tonemap).
    float pinT = 1e9;
    vec3  pinCol = vec3(1.0, 0.38, 0.10);
    bool  pinIsStud = false;   // stud hits stay solid even with the avatar on
    if (mapPass != 0) {
        // FIXED world size (player height, 18 voxels): the character is a
        // thing in the world, not a UI pin — it must not scale with zoom.
        // With the rigged voxel avatar active the boxes are DEAD: they don't
        // trace at all (any box surface the avatar's animated limbs don't
        // cover exactly would tint through as the old character). The
        // buried-marker tint and the X-ray ghost trace the avatar's actual
        // parts instead (playerGhost below); the stud keeps its boxes.
        vec3 hitN = vec3(0.0), hitQ = vec3(0.0), hitC = vec3(1.0);
        float cs = cos(mapPlayerYaw), sn = sin(mapPlayerYaw);
        if (uPlayerAvatar == 0) {
        // Ray in character-local space (+x = facing, +z = character right).
        vec3 roW = camPos - mapPlayerPos;
        vec3 roL = vec3(cs * roW.x + sn * roW.z, roW.y, -sn * roW.x + cs * roW.z);
        vec3 rdL = vec3(cs * rd.x  + sn * rd.z,  rd.y,  -sn * rd.x  + cs * rd.z);
        float sw = (mapWalkPhase < 0.0) ? 0.0
                 : (fract(mapWalkPhase) < 0.5 ? 1.0 : -1.0);
        const vec3 SHIRT = vec3(1.00, 0.38, 0.10);   // the identity orange
        const vec3 PANTS = vec3(0.24, 0.27, 0.37);
        const vec3 SKIN  = vec3(0.95, 0.75, 0.58);
        vec3 bc[6]; vec3 bh[6]; vec3 bcol[6];
        bc[0] = vec3(0.0, 11.1, 0.0);         bh[0] = vec3(1.7, 3.1, 2.8);   bcol[0] = SHIRT;        // torso
        bc[1] = vec3(0.0, 16.2, 0.0);         bh[1] = vec3(1.9, 2.0, 1.9);   bcol[1] = SKIN;         // head
        bc[2] = vec3( sw * 1.5, 4.0, -1.55);  bh[2] = vec3(1.25, 4.0, 1.2);  bcol[2] = PANTS;        // legs
        bc[3] = vec3(-sw * 1.5, 4.0,  1.55);  bh[3] = vec3(1.25, 4.0, 1.2);  bcol[3] = PANTS;
        bc[4] = vec3(-sw * 1.7, 10.4, -3.75); bh[4] = vec3(0.85, 2.7, 0.85); bcol[4] = SHIRT * 0.85; // arms
        bc[5] = vec3( sw * 1.7, 10.4,  3.75); bh[5] = vec3(0.85, 2.7, 0.85); bcol[5] = SHIRT * 0.85;
        vec3 invL = 1.0 / rdL;
        for (int i = 0; i < 6; ++i) {
            vec3 c  = bc[i], he = bh[i];
            vec3 ta = (c - he - roL) * invL;
            vec3 tb = (c + he - roL) * invL;
            vec3 t1 = min(ta, tb), t2 = max(ta, tb);
            float tn = max(max(t1.x, t1.y), t1.z);
            float tf = min(min(t2.x, t2.y), t2.z);
            if (tn < tf && tn > 0.0 && tn < pinT) {
                pinT = tn;
                vec3 nL = (tn == t1.x) ? vec3(-sign(rdL.x), 0.0, 0.0)
                        : (tn == t1.y) ? vec3(0.0, -sign(rdL.y), 0.0)
                                       : vec3(0.0, 0.0, -sign(rdL.z));
                // Normal back in WORLD axes so shading doesn't spin with yaw.
                hitN = vec3(cs * nL.x - sn * nL.z, nL.y, sn * nL.x + cs * nL.z);
                hitQ = abs(roL + rdL * tn - c) / he - abs(nL);
                hitC = bcol[i];
            }
        }
        }
        // Destination stud: a gold slab + post on the clicked spot, blinking
        // as a discrete size flip (same TAA reasoning as the walk cycle).
        if (mapMoveOn == 1) {
            // World-fixed like the character (no zoom scaling). Hovers ~2.5
            // voxels: the pick resolves against FINE ground, but at far zoom
            // ring0 carries the frame and a ring cell (which samples one
            // fine column per 2x2) can stand a few voxels proud of the
            // clicked column — without the hover the stud buries and
            // degrades to its 30% ghost.
            float pulse = (fract(uTime * 1.1) < 0.5) ? 1.0 : 1.18;
            vec3 inv = 1.0 / rd;
            for (int i = 0; i < 2; ++i) {
                vec3 c  = mapMoveTarget + vec3(0.0, (i == 0) ? 2.85 : 4.2, 0.0);
                vec3 he = (i == 0) ? vec3(2.4, 0.35, 2.4) * pulse
                                   : vec3(0.7, 1.05, 0.7);
                vec3 ta = (c - he - camPos) * inv;
                vec3 tb = (c + he - camPos) * inv;
                vec3 t1 = min(ta, tb), t2 = max(ta, tb);
                float tn = max(max(t1.x, t1.y), t1.z);
                float tf = min(min(t2.x, t2.y), t2.z);
                if (tn < tf && tn > 0.0 && tn < pinT) {
                    pinT = tn;
                    pinIsStud = true;
                    hitN = (tn == t1.x) ? vec3(-sign(rd.x), 0.0, 0.0)
                         : (tn == t1.y) ? vec3(0.0, -sign(rd.y), 0.0)
                                        : vec3(0.0, 0.0, -sign(rd.z));
                    hitQ = abs(camPos + rd * tn - c) / he - abs(hitN);
                    hitC = (i == 0) ? vec3(1.0, 0.85, 0.30) : vec3(1.0, 0.95, 0.55);
                }
            }
        }
        if (pinT < 1e8) {
            // Face-keyed flat light (bright top / mid X / dark Z / dim
            // underside), matching how the diorama's own voxels shade, with
            // darkened box borders so each part reads as blocks.
            float face = hitN.y >  0.5 ? 1.00
                       : hitN.y < -0.5 ? 0.42
                       : abs(hitN.x) > 0.5 ? 0.78 : 0.60;
            face *= 1.0 - 0.30 * smoothstep(0.55, 1.0,
                                            max(hitQ.x, max(hitQ.y, hitQ.z)));
            pinCol = hitC * face;
        }
    }

    if (miss) {
        if (mapPass != 0) {
            if (pinT < 1e8 && (uPlayerAvatar == 0 || pinIsStud)) {
                FragColor = vec4(pinCol, 1.0); return;
            }
            // Full-screen world view: a miss is the LIVE sky, run through
            // the same height-fog tail as terrain so a fog sea meets the
            // sky instead of stopping at the slab edge. The surface pass
            // owns every pixel (coverage 1); in X-ray it dims to a ghost
            // like the terrain does, and the X-ray re-march keeps alpha 0
            // so its blend leaves the pass-1 image where there are no veins.
            vec3 sc = skyColor(rd);
            if (mapPass == 1) sc = applyFogSky(sc, rd);
            FragColor = vec4(present(sc) * mapGhost,
                             (mapPass == 1) ? mapCover : 0.0);
            return;
        }
        // Studio plate miss: true sky (sun/moon/clouds/stars).
        vec3 col = skyColor(rd);
        col = applyFogSky(col, rd);   // a fog sea must meet the sky, not end at terrain
        FragColor = vec4(present(col), 1.0);
        return;
    }

    // LOD-ring hits ride in as material + 256 (gbuffer.frag). They shade
    // through the SAME shadeHit as fine hits — real per-face sun/ambient —
    // but with every window-anchored input neutralized (no AO/shadow rays,
    // no heightTex burial, no lightVol: those all wrap toroidally and read
    // other-side-of-window garbage for out-of-window positions).
    uint  matFull = uint(pm.w + 0.5);
    // ringData = an actual ring-pool hit (carries packed AO, below). ringHit
    // additionally covers every MAP surface hit: the map shades everything
    // through the ring path — analytic flat sun+ambient with every
    // window-anchored input neutralized — exactly what a fog-of-war diorama
    // wants (cut-open interiors must not burial-darken).
    bool  ringData = (matFull >= 256u);
    bool  ringHit  = ringData || (mapPass == 1);
    // Ring voxels carry 2 bits of BAKED AO in the top of the material byte
    // (rings.cpp: terrain concavity + canopy undersides — the traced-AO
    // stand-in for out-of-window hits). Decode BEFORE any material logic:
    // every material test below must see the base id. Fine hits (< 256)
    // use the full byte untouched (map fine hits included: MAT_MAP_UNKNOWN
    // is 37 and must not lose bits).
    uint  packedM = matFull & 0xFFu;
    uint  mat     = ringData ? (packedM & 0x3Fu) : packedM;
    float ringAO  = ringData ? mix(TW_RING_AO, 1.0, float(packedM >> 6u) / 3.0) : 1.0;
    vec4 nrm    = texture(gNormal, vUV);
    vec3 normal = nrm.xyz;

    // With the avatar active the analytic character never draws solid (the
    // avatar's own voxels are in the G-buffer); the stud always does.
    if (mapPass != 0 && pinT < tHit && (uPlayerAvatar == 0 || pinIsStud)) {
        FragColor = vec4(pinCol, 1.0); return;
    }

    // Cave X-ray layer: the discovered underground SPACE rendered as SOLID
    // casts — sun-shaded like real geometry with a rim so the tubes keep
    // their volume, tinted amber (shallow) through blue (deep); water pockets
    // teal. Everything else is transparent (the ghosted surface + the live
    // game behind). UI-colored — no tonemap.
    if (mapPass == 2) {
        float depth01 = clamp((surfaceY - hitPos.y) / 2400.0, 0.0, 1.0);
        vec3 glow = (mat == 6u)
                  ? vec3(0.25, 0.80, 0.90)
                  : mix(vec3(1.00, 0.62, 0.22), vec3(0.35, 0.55, 1.00), depth01);
        // The X-ray keeps its own fixed studio light: it is a READOUT, not a
        // scene — caves must be findable at midnight, and the live sun dips
        // below the horizon where lambert would flatten the whole layer.
        const vec3 xraySun = normalize(vec3(-0.45, 0.80, 0.25));
        float lam = max(dot(normal, xraySun), 0.0);
        float ndv = max(dot(normal, -rd), 0.0);
        float rim = pow(1.0 - ndv, 2.0);
        float shade = 0.45 + 0.40 * lam + 0.35 * rim;
        shade *= 0.90 + 0.20 * vhash(ivec3(floor(hitPos - normal * 0.5)));
        vec3 xcol = glow * shade;
        // The player stands INSIDE the discovered-air cast, so in X-ray the
        // cast front always occludes the marker — without this ghost the
        // marker vanishes in exactly the view meant for finding yourself
        // underground. pinCol keeps its face shading, so the diamond's form
        // reads through the tunnel wall.
        // The route reads through the X-ray like the character does: no
        // vertical gate (the trail is surface-anchored, the cast is deep).
        float tg = trailGlow(hitPos.xz, hitPos.y, 0.0, 0.0);
        if (tg > 0.0) xcol = mix(xcol, vec3(1.0, 0.80, 0.25), tg * 0.55);
        if (uPlayerAvatar == 1) {
            // Avatar ghost: the character's real voxels through the cast
            // (palette colour lifted toward the identity orange so the
            // silhouette stays findable against the glow).
            float tAv; vec3 gc;
            if (playerGhost(camPos, rd, tAv, gc))
                xcol = mix(xcol, mix(gc, vec3(1.0, 0.55, 0.20), 0.35), 0.55);
            if (pinIsStud && pinT < 1e8) xcol = mix(xcol, pinCol, 0.55);
        } else if (pinT < 1e8) xcol = mix(xcol, pinCol, 0.55);
        FragColor = vec4(xcol, 1.0);
        return;
    }

    ivec3 voxel = ivec3(floor(hitPos - normal * 0.5));   // the solid voxel behind the face
    // Slope-shaded normal for DIFFUSE only (see slopeShadeNormal above): the
    // geometric face normal still drives the voxel id, the reflection Fresnel
    // below, and the G-buffer. (A world-space variant from the stride-64
    // heightfield was tried and rejected — patchwork facets.)
    // Fog-of-war brightness for this hit, computed EARLY (a single bilinear
    // tap since the mask prefilter) so deep-unexplored pixels can skip the
    // whole lighting stack below: at the 0.05 darkness floor, slope shading,
    // cloud shadows, shadeHit's noise fields, reflections and atmosphere all
    // crush into a couple of 8-bit output levels — a flat cheap shade is
    // indistinguishable there, and the march becomes the only cost left.
    float mapE = (mapPass == 1) ? mapExploredSoft(hitPos.xz) : 1.0;
    // Cheap darkness shade, ALWAYS computed (a few multiplies): the
    // deep-unexplored early-out uses it alone, and a smooth band above zero
    // cross-fades it with the full model so no shading-model contour tracks
    // the frontier iso-line (which crawls as new exploration lands).
    // Day/night aware so the band can't glow at night or flatten at noon.
    vec3 cheapDarkCol = ((mat == 37u) ? vec3(0.52, 0.47, 0.40) : matColor(mat))
                      * (0.55 + 0.35 * max(normal.y, 0.0))
                      * (0.30 + 0.90 * dayLight);
    vec3 col;
    if (mapPass == 1 && mapE <= 0.0) {
        // Exactly-zero explored field (mask texels quantize to 0): the vast
        // majority of the dark region, and the actual perf win.
        col = cheapDarkCol;
    } else {   // ---- full shading path (everything until the fog-of-war tail)
    bool terrainMat = (mat == 1u || mat == 2u || mat == 3u || mat == 7u ||
                       mat == 8u || mat == 9u || mat == 29u || mat == 30u ||
                       mat == 31u || mat == 32u);
    vec3 shadeN = (slopeShade == 1 && terrainMat)
                ? slopeShadeNormal(hitPos, normal, tHit, matFull) : normal;

    // Creature-pass pixels come in two flavours (creature.frag): palette mode
    // (albedo a == 1.0) shades the WHITE material-38 base and multiplies the
    // baked palette colour back in; TERRAIN-MATERIAL mode (a = realMat/255,
    // forestry's falling trees) shades through the exact material path the
    // standing voxels used. For the latter, recover the OBJECT-STABLE
    // position: cell = inv * hitPos rides the model rigidly, and anchor is
    // the part's pre-fall world lo corner — detailPos/detailVox reproduce
    // the standing tree's painterly pattern exactly (no step change at the
    // fell, no boiling as it rotates), and they key the shadow-tap jitter so
    // the 4-tap dither is frozen in object space instead of re-rolling.
    uint shadeMat = mat;
    vec4 creAlb = vec4(1.0);
    bool creTerrain = false;
    vec3 detailPos = hitPos;
    ivec3 detailVox = voxel;
    if (mat == 38u) {
        creAlb = texture(gAlbedoTex, vUV);
        if (creAlb.a < 0.9) {
            creTerrain = true;
            shadeMat = uint(round(creAlb.a * 255.0));
            CrePart cp = creParts[int(nrm.w + 0.5)];
            vec3 cell = vec3(dot(cp.invX.xyz, hitPos) + cp.invX.w,
                             dot(cp.invY.xyz, hitPos) + cp.invY.w,
                             dot(cp.invZ.xyz, hitPos) + cp.invZ.w);
            // Rigid cellScale-1 transform: inv rows rotate the normal back
            // to the part's ORIGINAL axis-aligned frame, so the behind-the-
            // face voxel snap stays stable on the rotated model.
            vec3 cn = vec3(dot(cp.invX.xyz, normal),
                           dot(cp.invY.xyz, normal),
                           dot(cp.invZ.xyz, normal));
            detailPos = cp.anchor.xyz + cell;
            detailVox = ivec3(floor(detailPos - cn * 0.5));
        }
    }
    // Ring hits: baked AO + heightfield sun shadow (lift scales with the
    // ring's voxel size — coarser rings sit further below the continuous
    // heightfield). The moon channel gets the same heightfield march along
    // moonDir, so distant valleys fall into moon-shade like near ones.
    float ringScaleV = ringHit ? float(1 << ((matFull >> 8u))) : 2.0;
    vec3 aos;
    if (ringHit) {
        // Sun visibility for the analytic path. The DIORAMA gets real shadows:
        // its near-field is FINE, in-window voxels (ringData == false), so a
        // proper shadow ray casts trees / plants / relief onto the ground —
        // exactly what was missing. Far ring-pool hits (ringData) can't ray
        // (window-anchored), so they keep the heightfield sun march.
        float sv = (mapPass == 1 && !ringData)
                 ? softShadow(hitPos, normal, sunDir, detailPos)
                 : farSunVis(hitPos, ringScaleV * 1.5 + 2.0);
        // The moon channel mirrors the sun's diorama split exactly: fine
        // in-window map hits trace a real moon softShadow (trees cast moon
        // shadows on the night diorama, matching the creatures'), far
        // ring-pool hits keep the heightfield march.
        float mv = (moonLight > 0.02)
                 ? ((mapPass == 1 && !ringData)
                    ? softShadow(hitPos, normal, moonDir)
                    : farVisDir(hitPos, ringScaleV * 1.5 + 2.0, moonDir)) : 1.0;
        aos = vec3(ringAO, sv, mv);
    } else {
        // Plate fine hits: shadeHit computes the analytic AO + 4-tap soft
        // shadow itself; these inputs are unused on that path.
        aos = vec3(1.0);
    }
    // Cloud shadows: the sky's cloud layer dims the light over the ground
    // under it — big soft patches drifting with the same field the sky
    // renders. Sun and moon channels each project along their own body's
    // bearing, identically for fine and ring hits (and inside the D0 seam
    // blend), so the boundary can't show a lighting step. Traced shadow rays
    // can't see clouds, hence the explicit terms. The map surface pass gets
    // them too — the diorama lives under the real sky now; only the X-ray
    // keeps its flat readable glow. The moon term is gated so daylight
    // frames never pay for it.
    if (mapPass != 2) {
        aos.g *= cloudShadow(hitPos);
        if (moonLight > 0.01) aos.b *= cloudShadowDir(hitPos, moonDir);
    }

    // Creatures cast (and self-cast) shadows: march their flipbook volumes
    // along the light ray. One site covers every path — map traced shadows
    // and ring hits (cheap distance reject).
    // At night the same cone march runs along the moon ray, so a stag still
    // throws a shadow under the moon. Both rays share the object-anchored
    // jitter key so a falling tree's shadow dither stays frozen in its
    // object space (world-keyed jitter re-rolls per frame on movers).
    if (uCreN > 0 && mapPass != 2) {
        aos.g = min(aos.g, creatureSunVis(hitPos, normal, sunDir, detailPos));
        if (moonLight > 0.01)
            aos.b = min(aos.b, creatureSunVis(hitPos, normal, moonDir, detailPos));
    }

    col = shadeHit(shadeMat, detailVox, normal, shadeN, hitPos, tHit, rd,
                   ringHit, aos.r, aos.g, aos.b, detailPos, creTerrain);
    if (mat == 38u && !creTerrain) col *= creAlb.rgb;

    // Map unknown mass (MAT_MAP_UNKNOWN): stylized flat earth-wall shading in
    // place of the generic path, which leaves sun-averted side walls near
    // black. Warm putty, brighter tops, and a faint per-column stripe so tall
    // cut walls read like packed earth instead of a void.
    if (mapPass == 1 && mat == 37u) {
        float lit = 0.60 + 0.34 * max(normal.y, 0.0)
                  + 0.14 * max(dot(normal, sunDir), 0.0);
        float stripe = vhash(ivec3(voxel.x, 0, voxel.z));   // per-column grain
        col = vec3(0.52, 0.47, 0.40) * lit * (0.92 + 0.08 * stripe);
    }

    // Faint contact blob under each creature — ambient grounding where the
    // traced creature shadow is weak (high sun = small footprint; AO rays
    // can't see creatures at all). Subtle on purpose: creatureSunVis carries
    // the read now. Feet sit ~0.45r below the bounding centre (creatures.cpp).
    if (uCreN > 0 && mat != 38u) {
        for (int ci = 0; ci < uCreN; ++ci) {
            if (ci >= uCreBlobSkip.x && ci < uCreBlobSkip.x + uCreBlobSkip.y)
                continue;
            float r  = uCreBound[ci].w;
            vec3  ft = uCreBound[ci].xyz - vec3(0.0, r * 0.45, 0.0);
            float dy = hitPos.y - ft.y;
            if (dy < -3.0 || dy > 3.5) continue;
            // Footprint tracks the BODY, not the full bound (which is
            // height-dominated for an antlered head): ~1/3 of the posed
            // reach ends up around the old hand-tuned disc size.
            float rr = r * 0.34;
            float dd = length(hitPos.xz - ft.xz);
            col *= mix(0.80, 1.0, smoothstep(rr * 0.25, rr, dd));
        }
    }

    // Atmosphere (distance scatter + height fog) applies to the map surface
    // pass: the far reaches of the slab haze out with distance. X-ray stays
    // clear — depth tint IS its distance cue and haze would murk the glow.
    // The camera's orbit standoff is subtracted first: zoom must not change
    // the weather — only WORLD distance beyond the follow target accrues haze.
    if (mapPass != 2)
        col = applyAtmosphere(col, rd, max(tHit - mapStandoff, 0.0));
    }   // ---- end full shading path (the cheap-dark branch rejoins here)
    // Cross-fade the last stretch of the frontier into the cheap model:
    // water shimmer, cloud shadows and the sun terms fade out INTO the
    // darkness instead of stopping dead on the early-out boundary.
    if (mapPass == 1 && mapE > 0.0)
        col = mix(cheapDarkCol, col, smoothstep(0.0, 0.06, mapE));
    // Map fog of war: unexplored terrain shows up DARK (not as holes), with
    // a soft one-chunk gradient at the frontier of what's about to open up.
    if (mapPass == 1) {
        // No smoothstep re-sharpening: the wide field IS the ramp.
        col *= mix(0.05, 1.0, mapE);
        // X-ray ghosting: dim the surface layer itself (the buried-pin tint
        // below stays AFTER this, so the marker reads through the gloom).
        col *= mapGhost;
        // Click-to-move trail: gold dashes marching along the planned route,
        // painted on the ground. Drawn AFTER the darkness and the ghost dim
        // so the route reads into unexplored gloom (it is UI, like the
        // stud). The window is asymmetric: tight above (16 < HEADROOM keeps
        // low canopy from catching the trail) and generous below (chords
        // lerp over descents). GROUND ONLY: the avatar and creatures walk
        // right on the chord, and without the material gate the marching
        // dashes crawl across their bodies.
        if (mat != 38u) {
            float tg = trailGlow(hitPos.xz, hitPos.y, 16.0, 24.0);
            if (tg > 0.0) col = mix(col, vec3(1.0, 0.80, 0.25), tg * 0.85);
        }
        // Forestry hover highlight: SELECTION OUTLINE. Draw the iso-contour
        // of the tree mask's screen-space coverage: membership sampled over
        // a small ring, and where coverage crosses ~50% — the silhouette
        // boundary — a warm line lands on both sides of the edge. Interior
        // canopy gaps keep high coverage, so the fluffy crown stays quiet
        // (the old per-pixel fill lift + 4-tap rim read as blotchy glow
        // there). No fill: outline only, like a classic selection.
        if (uHoverOn == 1 && !ringData && uHoverStrength > 0.001) {
            vec3 bLo = uHoverLo - 4.0;
            vec3 bHi = uHoverLo + uHoverDims + 4.0;
            if (all(greaterThan(hitPos, bLo)) && all(lessThan(hitPos, bHi))) {
                vec2 px = 1.0 / viewport;
                float cov = 0.0;
                float covC = 0.0;   // centre-tap membership (thin-feature trace)
                for (int i = 0; i < 9; ++i) {
                    vec2 o = (i == 0) ? vec2(0.0)
                           : 2.6 * px * vec2(cos(float(i - 1) * 0.7853982),
                                             sin(float(i - 1) * 0.7853982));
                    vec4 npm = texture(gPosMat, vUV + o);
                    uint nm = npm.w < 0.0 ? 0u : (uint(npm.w + 0.5) & 0xFFu);
                    // +rd*0.5 steps off the hit face into the voxel itself
                    // (the mask holds voxels, not faces). Mats: wood, leaves,
                    // desert wood, cactus flesh/bloom, sage tufts, and stone
                    // (minable boulders).
                    bool t = (nm == 3u || nm == 4u || nm == 5u || nm == 36u ||
                              nm == 33u || nm == 34u || nm == 35u) &&
                             hoverMaskAt(npm.xyz + rd * 0.5);
                    if (i == 0 && t) covC = 1.0;
                    cov += t ? (1.0 / 9.0) : 0.0;
                }
                // Iso-contour of the silhouette, PLUS a direct trace for
                // thin features: a 1-2 voxel branch never reaches 50%
                // neighbourhood coverage, so without the second term the
                // outline hugged trunks and skipped twigs.
                float line = smoothstep(0.55, 0.95, 1.0 - abs(cov * 2.0 - 1.0));
                line = max(line, covC * smoothstep(0.75, 0.45, cov));
                float breathe = 0.86 + 0.14 * sin(uTime * 2.8);
                col = mix(col, vec3(1.00, 0.94, 0.55),
                          line * 0.85 * uHoverStrength * breathe);
            }
        }
        // Buried-marker ghost: reaching here with a marker hit means terrain
        // occludes it (player under an overhang / in a cave), so tint its
        // pixels faintly — you can always find yourself on the map. With the
        // avatar on, the ghost is the avatar's OWN voxels traced behind the
        // terrain hit (+0.5 keeps the visible avatar's surface pixels from
        // self-tinting); the box figure never tints — anywhere its chunky
        // silhouette peeked past the animated limbs it read as the old
        // character bleeding through the new one.
        if (uPlayerAvatar == 1) {
            float tAv; vec3 gc;
            if (playerGhost(camPos, rd, tAv, gc) && tAv > tHit + 0.5)
                col = mix(col, gc, 0.30);
            if (pinIsStud && pinT < 1e8 && pinT > tHit)
                col = mix(col, pinCol, 0.30);
        } else if (pinT < 1e8)
            col = mix(col, pinCol, 0.30);

        // Farm Mode footprint: a translucent slab painted over the plot cell
        // under the cursor. Green = the active tool is valid here, red = not.
        // A brighter rim makes the plot square read as a discrete tile. Drawn
        // as UI (after the darkness/ghost), clipped to sit on/near the ground.
        if (mapFarmOn == 1) {
            vec3 flo = mapFarmMin + vec3(0.0, -0.5, 0.0);
            vec3 fhi = mapFarmMin + vec3(mapFarmSize, 0.6, mapFarmSize);
            vec3 finv = 1.0 / rd;
            vec3 fa = (flo - camPos) * finv, fb = (fhi - camPos) * finv;
            vec3 f1 = min(fa, fb), f2 = max(fa, fb);
            float ftn = max(max(f1.x, f1.y), f1.z);
            float ftf = min(min(f2.x, f2.y), f2.z);
            if (ftn < ftf && ftf > 0.0 && ftn < tHit + 2.0) {
                vec3  fhp = camPos + rd * max(ftn, 0.0);
                vec2  fuv = (fhp.xz - mapFarmMin.xz) / mapFarmSize;   // 0..1
                float edge = min(min(fuv.x, 1.0 - fuv.x), min(fuv.y, 1.0 - fuv.y));
                float rim  = 1.0 - smoothstep(0.0, 0.07, edge);
                vec3  tint = (mapFarmOk == 1) ? vec3(0.20, 1.10, 0.35)
                                              : vec3(1.10, 0.28, 0.24);
                col = mix(col, tint, 0.26 + 0.5 * rim);
            }
        }

        // God-harvest hover glow: brighten + warm-tint the hovered ripe crop's
        // own voxels (its bbox sits above the farmland, so terrain stays put),
        // with a gentle breathe — the "this is harvestable" affordance.
        if (mapHoverOn == 1 &&
            all(greaterThanEqual(hitPos, mapHoverLo - 0.5)) &&
            all(lessThanEqual(hitPos, mapHoverHi + 0.5))) {
            float breathe = 0.82 + 0.18 * sin(uTime * 2.6);
            col *= 1.0 + 0.30 * breathe;
            col = mix(col, vec3(1.0, 0.92, 0.62), 0.26 * breathe);
        }

    }

    // Alpha carries the diorama COVERAGE consumed by mapview.frag's
    // composite: mapCover = 1 normally, ~0.1 for the ghosted surface under
    // the X-ray. (The X-ray pass itself, mapPass == 2, writes 0.)
    FragColor = vec4(present(col), (mapPass == 1) ? mapCover : 0.0);
}
