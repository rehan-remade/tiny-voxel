#version 430 core
// Rigid creature parts: one instanced box per posed bone part. Expands a
// unit cube (vertex-id lookup, no VBO) through the part's cell->world affine
// and projects with the SAME pinhole model the raymarch passes use (fwd/
// right/up basis + tanHalfFov + TAA jitter), so the fragment DDA and terrain
// rays agree exactly. BACK faces are drawn (front-culled) so a camera inside
// a box still rasterizes it; depth comes from gl_FragDepth in the fragment.

uniform vec3  camPos;
uniform vec3  camFwd;
uniform vec3  camRight;
uniform vec3  camUp;
uniform vec2  viewport;
uniform float tanHalfFov;
uniform vec2  jitterNdc;

#include "lib/creature_trace.glsl"

flat out int vInst;

// 36 corner ids of a unit cube (2 tris x 6 faces), CCW outward.
const int CID[36] = int[36](
    0,2,3, 0,3,1,   4,5,7, 4,7,6,     // -z, +z
    0,1,5, 0,5,4,   2,6,7, 2,7,3,     // -y, +y
    0,4,6, 0,6,2,   1,3,7, 1,7,5);    // -x, +x

void main() {
    int inst = gl_InstanceID;
    vInst = inst;
    CrePart p = creParts[inst];
    int cid = CID[gl_VertexID];
    vec3 cell = vec3(float(cid & 1), float((cid >> 1) & 1), float((cid >> 2) & 1))
              * p.dimsPal.xyz;
    vec3 world = vec3(dot(p.rowX.xyz, cell) + p.rowX.w,
                      dot(p.rowY.xyz, cell) + p.rowY.w,
                      dot(p.rowZ.xyz, cell) + p.rowZ.w);

    vec3 d = world - camPos;
    float vx = dot(d, camRight), vy = dot(d, camUp), vz = dot(d, camFwd);
    float aspect = viewport.x / viewport.y;
    // ndc = v.xy/(vz*axes) - jitter; premultiplied by w = vz for the raster.
    // z stays 0: real depth is written per-fragment from the DDA hit t.
    gl_Position = vec4(vx / (aspect * tanHalfFov) - jitterNdc.x * vz,
                       vy / tanHalfFov          - jitterNdc.y * vz,
                       0.0, vz);
}
