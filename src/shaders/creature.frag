#version 430 core
// Rigid creature parts: fragment DDA over the rasterized part box. Rebuilds
// the pixel's primary ray (same pinhole + jitter as gbuffer.frag), marches
// the part's rest volume in cell space via the shared trace include, and
// writes the SAME G-buffer attributes terrain writes — the deferred stack
// (lighting, traced shadows, fog-of-war, atmosphere, TAA) shades creatures
// with no extra code. Depth-tests against terrain via the G-buffer depth
// attachment: gbuffer.frag writes gl_FragDepth = t/DEPTH_FAR, this pass does
// the same under GL_LESS.

layout(location = 0) out vec4 gPosMat;   // xyz world hit pos, w material id
layout(location = 1) out vec4 gNormal;   // xyz world normal,  w part index (motion pass)
layout(location = 2) out vec4 gAlbedo;   // rgb albedo (from the .cvx2 palette)

uniform vec3  camPos;
uniform vec3  camFwd;
uniform vec3  camRight;
uniform vec3  camUp;
uniform vec2  viewport;
uniform float tanHalfFov;
uniform vec2  jitterNdc;
// Diorama clip slab (mapPass == 1): terrain outside it is carved away, so
// creature voxels clip against the same box (the Q/E slice cuts through a
// stag exactly like it cuts through a tree).
uniform int   mapPass;
uniform vec3  mapClipLo;
uniform vec3  mapClipHi;

#include "lib/creature_trace.glsl"

flat in int vInst;

// Must match gbuffer.frag's depth mapping (see DEPTH_FAR there).
const float DEPTH_FAR = 4096.0;
const float MAT_CREATURE = 38.0;       // common.h MAT_CREATURE

void main() {
    vec2 ndc = (gl_FragCoord.xy / viewport) * 2.0 - 1.0 + jitterNdc;
    float aspect = viewport.x / viewport.y;
    vec3 rd = normalize(camFwd
        + camRight * ndc.x * aspect * tanHalfFov
        + camUp    * ndc.y * tanHalfFov);

    float t; vec3 nrm; uint ci; vec3 col;
    if (!creTracePart(vInst, camPos, rd, 1e9,
                      mapPass == 1 ? 1 : 0, mapClipLo, mapClipHi,
                      t, nrm, ci, col))
        discard;
    gPosMat = vec4(camPos + rd * t, MAT_CREATURE);
    // w = part instance index: the motion pass reprojects creature pixels
    // through that part's previous-frame transform (real motion vectors).
    gNormal = vec4(nrm, float(vInst));
    // Two albedo modes, keyed on the instance's meta.y:
    //  - palette (creatures/props): rgb = baked palette colour, a = 1.0; the
    //    lighting pass shades a white base and multiplies this in.
    //  - TERRAIN MATERIAL (falling trees): the volume's bytes are world
    //    material ids; a = mat/255 tells the lighting pass to shade the
    //    pixel through the exact terrain path, so a felled tree looks
    //    identical to its standing voxels (no step change at the fell).
    bool terrainMat = creParts[vInst].meta.y > 0.5;
    gAlbedo = terrainMat ? vec4(1.0, 1.0, 1.0, float(ci) / 255.0)
                         : vec4(col, 1.0);
    gl_FragDepth = clamp(t / DEPTH_FAR, 0.0, 1.0);
}
