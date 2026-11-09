#version 430 core
// Screen-space motion vectors for the temporal stack (TAA in PR5, denoiser in
// PR8). The world is static, so motion is pure camera reprojection: where a
// pixel's surface point was on screen LAST frame vs. this frame. Reads the
// G-buffer world position and projects it through the previous frame's camera.
// motion = thisFrameUV - prevFrameUV, in [0,1] UV units.
in vec2 vUV;
out vec2 FragMotion;

uniform sampler2D gPosMat;     // xyz world hit pos, w material id (-1 = miss/sky)
uniform sampler2D gNormal;     // w = part instance index for creature pixels

// Creature parts (binding 5): each record carries THIS frame's inverse and
// LAST frame's forward transform, so a creature pixel reprojects through the
// part's actual motion instead of the static-world assumption.
#include "lib/creature_trace.glsl"

// Current camera (only needed to rebuild the view ray for sky pixels).
uniform vec3  camPos;
uniform vec3  camFwd;
uniform vec3  camRight;
uniform vec3  camUp;
uniform float tanHalfFov;
uniform vec2  viewport;

// Previous frame's camera.
uniform vec3  prevPos;
uniform vec3  prevFwd;
uniform vec3  prevRight;
uniform vec3  prevUp;
uniform float prevTanHalfFov;
uniform float prevAspect;
uniform int   prevValid;       // 0 on the first frame -> motion is zero

// Project a world point into a camera's UV. Inverse of the ray generation:
// ndc = (dot(d,right)/dot(d,fwd)) / (aspect*thf).
vec2 projectPoint(vec3 p, vec3 cpos, vec3 cf, vec3 cr, vec3 cu, float thf, float aspect) {
    vec3 d = p - cpos;
    float z = dot(d, cf);
    if (z <= 1e-4) return vec2(-1.0);          // behind the camera -> off-screen
    float ndcx = (dot(d, cr) / z) / (aspect * thf);
    float ndcy = (dot(d, cu) / z) / thf;
    return vec2(ndcx, ndcy) * 0.5 + 0.5;
}
// Same, for a pure direction (sky at infinity: rotation only, ignore translation).
vec2 projectDir(vec3 dir, vec3 cf, vec3 cr, vec3 cu, float thf, float aspect) {
    float z = dot(dir, cf);
    if (z <= 1e-4) return vec2(-1.0);
    float ndcx = (dot(dir, cr) / z) / (aspect * thf);
    float ndcy = (dot(dir, cu) / z) / thf;
    return vec2(ndcx, ndcy) * 0.5 + 0.5;
}

void main() {
    if (prevValid == 0) { FragMotion = vec2(0.0); return; }

    float aspect = viewport.x / viewport.y;
    vec4 pm = texture(gPosMat, vUV);

    vec2 prevUV;
    if (pm.w < 0.0) {
        // Sky / miss: reproject the view ray as a direction.
        vec2 ndc = vUV * 2.0 - 1.0;
        vec3 rd = normalize(camFwd + camRight * ndc.x * aspect * tanHalfFov
                                   + camUp    * ndc.y * tanHalfFov);
        prevUV = projectDir(rd, prevFwd, prevRight, prevUp, prevTanHalfFov, prevAspect);
    } else if (pm.w > 37.5 && pm.w < 38.5) {
        // Creature pixel (MAT_CREATURE): the surface MOVED. Take the hit
        // back to part cell space with this frame's inverse, then forward
        // through LAST frame's transform — where that voxel actually was.
        int idx = int(texture(gNormal, vUV).w + 0.5);
        CrePart cp = creParts[idx];
        vec3 cell = vec3(dot(cp.invX.xyz, pm.xyz) + cp.invX.w,
                         dot(cp.invY.xyz, pm.xyz) + cp.invY.w,
                         dot(cp.invZ.xyz, pm.xyz) + cp.invZ.w);
        vec3 prevP = vec3(dot(cp.prevX.xyz, cell) + cp.prevX.w,
                          dot(cp.prevY.xyz, cell) + cp.prevY.w,
                          dot(cp.prevZ.xyz, cell) + cp.prevZ.w);
        prevUV = projectPoint(prevP, prevPos, prevFwd, prevRight, prevUp, prevTanHalfFov, prevAspect);
    } else {
        prevUV = projectPoint(pm.xyz, prevPos, prevFwd, prevRight, prevUp, prevTanHalfFov, prevAspect);
    }
    FragMotion = vUV - prevUV;   // this frame's UV is just vUV
}
