#version 430 core
in vec2 vUV;
out vec4 FragColor;

// RuneScape-style minimap (drawMinimap): a circular top-down TRUE render of
// the world in the top-right corner while the diorama is up. Every disc pixel
// casts a short vertical ray down the voxel column (the shared brick DDA in
// lib/voxel_trace.glsl, started just above the heightTex ground) and shades
// the hit with the renderer's own surfaceColor — so trees, water, digs and
// edits are all live, per-voxel crisp. The map rotates with the orbit camera
// (screen-up = view direction, RS-style); a compass N stud rides the ring at
// world north (+Z). Overlaid: the camera's frustum wedge, the player arrow,
// the click-to-move flag, creature dots, and the game's fog-of-war ramp.
//
// Geometry must match the click hit-test in app.cpp (minimapHit) and the
// layout constants in common.h: disc radius MM_R_MAP inside an MM_RING px
// stone ring, MM_MARGIN px off the top-right corner; the free-cam orb rides
// the ring's lower-left diagonal at MM_R_OUT + 7, radius MM_ORB_R.

uniform vec2  viewport;
uniform vec3  camPos;        // map orbit camera (world voxels)
uniform vec3  camFwd;        // normalized look direction
uniform float camDist;       // camera-to-target standoff (world voxels)
uniform float voxPerPx;      // minimap zoom: world voxels per screen pixel
uniform float tanHalfH;      // horizontal FOV tan (frustum wedge spread)
uniform vec2  playerXZ;      // player feet (world voxels)
uniform float playerYaw;     // facing: fwd = (cos yaw, sin yaw)
uniform vec2  moveTargetXZ;  // click-to-move route end
uniform int   moveOn;
uniform int   freeCamOn;     // orb state
uniform float dayLight;      // 0 night .. 1 noon: terrain dims after dark
uniform int   creN;          // creature dots (yellow, RS npc style)
uniform vec2  creXZ[32];
uniform ivec2 maskOrigin;    // explored-mask texel (0,0) in world chunk coords
uniform sampler2D exploredSoft;  // prefiltered fog-of-war ramp (R8 bilinear)

// Scene uniforms consumed by the included libs (names fixed by them).
uniform ivec3 gridSize;
uniform ivec3 bricksPerAxis;
uniform ivec2 windowOriginB;
uniform float surfaceY;
uniform sampler2D heightTex;

#include "lib/noise.glsl"
#include "lib/voxel_trace.glsl"
#include "lib/materials.glsl"

// Layout (px). MUST match common.h MM_* and app.cpp minimapHit.
const float MARGIN = 14.0;
const float R_MAP  = 118.0;
const float RING   = 18.0;
const float R_OUT  = R_MAP + RING;
const float ORB_R  = 17.0;
const vec2  ORB_DIR = vec2(-0.70710678, 0.70710678);   // lower-left diagonal
const int   MASK_CH = 384;      // explored-mask side, chunks (map.h)
const float CHUNK   = 64.0;     // CHUNK_VOX

// Map basis (filled in main): screen-right / screen-up in world XZ, and the
// disc centre in both spaces.
vec2 gRight, gFwd, gCenterW;

// World XZ -> disc-local px (x right, y DOWN — matching lc in main).
vec2 lcOf(vec2 wxz) {
    vec2 d = (wxz - gCenterW) / voxPerPx;
    return vec2(dot(d, gRight), -dot(d, gFwd));
}

// Top-down terrain color of one world column: vertical ray from just above
// the ground field, shaded flat-lit. `ground` returns the heightTex sample
// for the hillshade.
vec3 terrainAt(vec2 wxz, out float ground) {
    ground = 0.0;
    // Beyond the resident window there is no voxel data (the diorama shows
    // LOD rings out there; the minimap shows the unknown): dark parchment.
    vec3 winLo = windowLoV();
    const vec3 UNKNOWN = vec3(0.050, 0.042, 0.034);
    if (wxz.x < winLo.x + 1.0 || wxz.x >= winLo.x + float(gridSize.x) - 1.0 ||
        wxz.y < winLo.z + 1.0 || wxz.y >= winLo.z + float(gridSize.z) - 1.0)
        return UNKNOWN;
    float h = texture(heightTex, wxz / vec2(gridSize.x, gridSize.z)).r;
    ground = h;
    // Start just above the tallest flora (trees top out well under +120) —
    // the coarse-skip makes the empty descent a handful of steps. The tiny
    // XZ tilt keeps the DDA's inverse-direction math off exact-zero axes.
    vec3 ro = vec3(wxz.x, min(h + 120.0, float(gridSize.y) - 2.0), wxz.y);
    vec3 rd = normalize(vec3(2.0e-5, -1.0, 3.0e-5));
    vec3 hp, hn; uint hm;
    if (!traceRay(ro, rd, ro.y + 4.0, hp, hn, hm))
        return UNKNOWN;   // fully carved column (pit/cave mouth)

    if (hm == 6u) {
        // Water: Beer-Lambert depth tint over the actual bed material —
        // shallow shores read teal, deep pools navy (lighting.glsl's recipe,
        // reduced to a depth count).
        ivec3 v = ivec3(floor(hp - hn * 0.5));
        int depth = 0; uint bed = 0u;
        for (int i = 1; i <= 24; ++i) {
            uint b = getVoxelAt(v - ivec3(0, i, 0));
            if (b != 6u) { bed = b; break; }
            depth++;
        }
        vec3 bedCol = matColor(bed) * 0.75;
        vec3 transmit = exp(-vec3(0.12, 0.06, 0.08) * (float(depth) + 4.0));
        return bedCol * transmit + vec3(0.07, 0.37, 0.50) * (1.0 - transmit);
    }
    // tHit 100: most per-voxel noise kept — the tiny-voxel speckle IS the
    // map's texture — but the harshest sub-pixel shimmer softened.
    vec3 col = surfaceColor(hm, hp, vec3(0.0, 1.0, 0.0), 100.0);
    // Hillshade: light from screen up-left (a hand-drawn map's constant sun).
    vec2 lightW = (-0.7071 * gRight) + (0.7071 * gFwd);
    float h2 = texture(heightTex, (wxz + lightW * 3.0) / vec2(gridSize.x, gridSize.z)).r;
    col *= clamp(1.0 + (h - h2) * 0.05, 0.72, 1.24);
    return col;
}

// 5x5 bitmap 'N' for the compass stud, bit 4 = leftmost column.
int nRow(int r) {
    return (r == 0) ? 17 : (r == 1) ? 25 : (r == 2) ? 21 : (r == 3) ? 19 : 17;
}

void main() {
    vec2 pxTop = vec2(gl_FragCoord.x, viewport.y - gl_FragCoord.y);
    vec2 C     = vec2(viewport.x - MARGIN - R_OUT, MARGIN + R_OUT);
    vec2 orbC  = C + (R_OUT + 7.0) * ORB_DIR;
    vec2 lc    = pxTop - C;          // disc-local, y DOWN
    float r    = length(lc);
    float rOrb = length(pxTop - orbC);
    if (r > R_OUT + 7.0 && rOrb > ORB_R + 6.0) discard;

    // Rotate with the orbit camera: screen-up = the direction the camera
    // looks across the ground (matches MapView::pan's fwdXZ).
    gFwd     = normalize(camFwd.xz);
    gRight   = vec2(-gFwd.y, gFwd.x);
    // camPos + camFwd*camDist is the orbit target; its XZ is the map centre.
    gCenterW = camPos.xz + camFwd.xz * camDist;

    // Bevel light for all the "metal/stone" chrome: from screen up-left.
    vec2 lightS = vec2(-0.7071, -0.7071);   // up-left in y-down coords

    vec4 outC = vec4(0.0);

    if (r < R_MAP) {
        // -- Terrain disc ----------------------------------------------------
        vec2 wxz = gCenterW + voxPerPx * (lc.x * gRight - lc.y * gFwd);
        float ground;
        vec3 col = terrainAt(wxz, ground);

        // Fog of war: the game's explored record, same ramp as the diorama.
        vec2 uvE = (wxz / CHUNK - vec2(maskOrigin)) / float(MASK_CH);
        float soft = texture(exploredSoft, uvE).r;
        col *= mix(0.06, 1.0, soft);
        // Night dims the land, not the chrome (you can always read the map).
        col *= mix(0.45, 1.0, clamp(dayLight, 0.0, 1.0));

        // -- Camera view cone: the orbit camera's viewport on the ground -----
        // Flashlight read: land OUTSIDE the cone dims, the two boundary rays
        // draw bright — explicit at any zoom (no distance fades; at close
        // zoom the apex sits off-disc and only the rays cross the map).
        {
            vec2 camLc = lcOf(camPos.xz);            // apex (below centre)
            vec2 v  = lc - camLc;
            float vl = max(length(v), 1e-3);
            vec2 toC = -camLc / max(length(camLc), 1e-3);
            float ca = dot(v / vl, toC);
            float halfC = inversesqrt(1.0 + tanHalfH * tanHalfH); // cos(halfFov)
            float inside = smoothstep(halfC - 0.008, halfC + 0.008, ca);
            col *= mix(0.68, 1.0, inside);
            float edge = 1.0 - smoothstep(0.003, 0.012, abs(ca - halfC));
            col = mix(col, vec3(1.0, 0.96, 0.80), edge * 0.55);
        }

        // -- Click-to-move flag (RS red destination flag) ---------------------
        if (moveOn == 1) {
            vec2 f = lc - lcOf(moveTargetXZ);
            if (length(lcOf(moveTargetXZ)) < R_MAP - 2.0) {
                bool pole = f.x >= -0.5 && f.x < 1.5 && f.y > -10.0 && f.y <= 1.0;
                bool flag = f.x >= 1.5 && f.x < 8.5 && f.y > -10.0 && f.y <= -5.0;
                if (pole) col = vec3(0.10, 0.08, 0.06);
                if (flag) {
                    bool rim = f.x > 7.5 || f.y > -5.7 || f.y < -9.3;
                    col = rim ? vec3(0.45, 0.06, 0.05) : vec3(0.86, 0.13, 0.10);
                }
            }
        }

        // -- Creature dots (RS yellow npc dots) --------------------------------
        for (int i = 0; i < creN; ++i) {
            vec2 d = lcOf(creXZ[i]);
            if (length(d) >= R_MAP - 3.0) continue;
            float dd = length(lc - d);
            if (dd < 2.8) col = vec3(0.12, 0.10, 0.05);
            if (dd < 1.7) col = vec3(0.96, 0.83, 0.18);
        }

        // -- Player arrow (white, black-rimmed, points where the player faces)
        {
            vec2 pLc = lcOf(playerXZ);
            // Off-map (free-cam panned away): clamp to the rim as a dot.
            if (length(pLc) > R_MAP - 6.0) {
                vec2 rim = pLc / length(pLc) * (R_MAP - 6.0);
                float dd = length(lc - rim);
                if (dd < 4.0) col = vec3(0.10);
                if (dd < 2.6) col = vec3(0.97);
            } else {
                vec2 pf = vec2(cos(playerYaw), sin(playerYaw));   // world facing
                vec2 fDir = normalize(vec2(dot(pf, gRight), -dot(pf, gFwd)));
                vec2 perp = vec2(-fDir.y, fDir.x);
                vec2 q = lc - pLc;
                float a = dot(q, fDir), b = dot(q, perp);
                // Notched triangle: tip a=+7, wings back to a=-5.5.
                float wHalf = (7.0 - a) * 0.52;
                bool inTri  = a > -5.5 - 1.5 && a < 7.0 + 1.5 &&
                              abs(b) < wHalf + 1.5 && a > -5.5 + abs(b) * 0.8 - 2.4;
                bool inCore = a > -5.5 && a < 7.0 &&
                              abs(b) < (7.0 - a) * 0.42 && a > -5.5 + abs(b) * 0.8;
                if (inTri)  col = vec3(0.08);
                if (inCore) col = vec3(0.97);
            }
        }

        // Inner shadow hugging the ring (RS's recessed-glass look).
        col *= 1.0 - 0.38 * smoothstep(R_MAP - 9.0, R_MAP, r);
        outC = vec4(col, 1.0);
    } else if (r < R_OUT) {
        // -- Stone ring frame: chunky quantized blocks, gold trim -------------
        float ang = atan(lc.y, lc.x);
        float a01 = fract(ang / 6.2831853);
        const float SEGS = 40.0;
        float segF = a01 * SEGS;
        float m    = fract(segF);                     // within-block coord
        int   seg  = int(floor(segF));
        // Per-block tone, quantized to discrete steps (tiny-voxel stone).
        float tone = 0.78 + 0.40 * floor(vhash(ivec3(seg, 7, 3)) * 3.0) / 3.0;
        // Mortar seams between blocks.
        float seam = 1.0 - 0.45 * (1.0 - smoothstep(0.00, 0.08, min(m, 1.0 - m)));
        // Radial bevel: outer half catches the up-left light, inner half dips.
        float rn   = (r - R_MAP) / RING;
        vec2  nrm  = lc / max(r, 1e-3);
        float lit  = 0.5 + 0.5 * dot(nrm, lightS);
        float bevel = mix(0.78, 1.05 + 0.30 * lit, smoothstep(0.15, 0.55, rn))
                    * (1.0 - 0.25 * smoothstep(0.80, 1.0, rn));
        vec3 col = vec3(0.30, 0.25, 0.19) * tone * seam * bevel;
        // Gold inner rim + dark outer lip.
        if (r < R_MAP + 2.5)
            col = vec3(0.78, 0.60, 0.24) * (0.75 + 0.5 * lit);
        if (r > R_OUT - 2.0)
            col = vec3(0.05, 0.042, 0.032);

        // Cardinal studs rotate with the world (N red-crowned, with a letter).
        vec2 dirs[4] = vec2[4](vec2(0.0, 1.0), vec2(1.0, 0.0),
                               vec2(0.0, -1.0), vec2(-1.0, 0.0));
        for (int i = 0; i < 4; ++i) {
            vec2 sd = vec2(dot(dirs[i], gRight), -dot(dirs[i], gFwd));
            vec2 pC = normalize(sd) * (R_MAP + RING * 0.52);
            vec2 q  = lc - pC;
            float dia = abs(q.x) + abs(q.y);             // diamond stud
            float sz  = (i == 0) ? 8.5 : 5.0;
            if (dia < sz + 1.3) col = vec3(0.07, 0.055, 0.04);
            if (dia < sz)
                col = (i == 0 ? vec3(0.88, 0.20, 0.12) : vec3(0.80, 0.62, 0.26))
                    * (0.8 + 0.45 * lit);
            if (i == 0) {
                // 5x5 'N' punched dark into the north stud.
                ivec2 g = ivec2(floor(q + vec2(2.5)));
                if (g.x >= 0 && g.x < 5 && g.y >= 0 && g.y < 5 &&
                    ((nRow(g.y) >> (4 - g.x)) & 1) != 0)
                    col = vec3(0.10, 0.05, 0.03);
            }
        }
        outC = vec4(col, 1.0);
    } else {
        // Drop shadow just outside the ring.
        float sh = 1.0 - smoothstep(0.0, 7.0, r - R_OUT);
        outC = vec4(vec3(0.01), 0.45 * sh);
    }

    // -- Free-cam orb button (over everything) ---------------------------------
    if (rOrb < ORB_R + 6.0) {
        vec2 q = pxTop - orbC;
        float lit = 0.5 + 0.5 * dot(q / max(rOrb, 1e-3), lightS);
        if (rOrb < ORB_R) {
            vec3 col;
            if (rOrb > ORB_R - 2.5) {                    // stone/gold rim
                col = (freeCamOn == 1 ? vec3(0.85, 0.65, 0.25)
                                      : vec3(0.36, 0.31, 0.25))
                    * (0.7 + 0.55 * lit);
            } else {                                     // glass interior
                col = (freeCamOn == 1) ? vec3(0.17, 0.11, 0.03)
                                       : vec3(0.045, 0.055, 0.075);
                col *= 1.0 - 0.30 * smoothstep(ORB_R - 9.0, ORB_R - 2.5, rOrb);
                // Camcorder icon: body block + lens wedge pointing right.
                vec3 ink = (freeCamOn == 1) ? vec3(1.0, 0.85, 0.45)
                                            : vec3(0.82, 0.87, 0.92);
                bool body = abs(q.x + 2.5) < 4.5 && abs(q.y) < 4.0 &&
                            !(abs(q.x + 2.5) > 3.4 && abs(q.y) > 2.9);
                bool lens = q.x >= 2.0 && q.x < 6.0 &&
                            abs(q.y) < (q.x - 1.0) * 0.85;
                if (body || lens) col = ink;
                // Red "recording" pip while free-cam is live.
                if (freeCamOn == 1 && length(q - vec2(-4.0, -5.6)) < 1.6)
                    col = vec3(0.95, 0.15, 0.10);
            }
            outC = vec4(col, 0.97);
        } else {
            // Amber halo when on; soft shadow when off.
            float halo = 1.0 - smoothstep(0.0, 6.0, rOrb - ORB_R);
            vec4 glow = (freeCamOn == 1)
                      ? vec4(1.0, 0.72, 0.25, 0.30 * halo)
                      : vec4(0.0, 0.0, 0.0, 0.35 * halo);
            outC = vec4(mix(outC.rgb, glow.rgb, glow.a),
                        max(outC.a, glow.a));
        }
    }

    if (outC.a <= 0.004) discard;
    FragColor = outC;
}
