// Rigid creature parts: posed rest-pose volumes traced in cell space.
// Instance records are rebuilt on the CPU each frame (CreaturePartInstance
// in src/creatures.h — layouts must match this std430 struct exactly);
// volumes + palettes live in the static atlas (binding 4, uploaded once).
// Shared by creature.vert/.frag (G-buffer pass) and lighting.frag (creature
// cast shadows), so the two always agree on the geometry.
layout(std430, binding = 4) readonly buffer CreatureVox   { uint    creVox[]; };
struct CrePart {
    vec4 rowX, rowY, rowZ;      // cell -> world affine rows (rigid, x cellScale)
    vec4 invX, invY, invZ;      // world -> cell affine rows
    vec4 prevX, prevY, prevZ;   // LAST frame's cell -> world (motion vectors)
    vec4 dimsPal;               // xyz part dims (cells), w palette word base
    vec4 meta;                  // x vox word base, y 1 = terrain-material volume
    vec4 anchor;                // terrain-material: xyz pre-fall world lo corner
};
layout(std430, binding = 5) readonly buffer CreatureParts { CrePart creParts[]; };

float creSrgbToLin(float c) {
    return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4);
}

// Palette lookup (.vox colour indices are 1-based), sRGB -> linear.
vec3 crePalette(CrePart p, uint ci) {
    uint w = creVox[uint(p.dimsPal.w) + ci - 1u];
    vec3 c = vec3(float(w & 0xFFu), float((w >> 8) & 0xFFu),
                  float((w >> 16) & 0xFFu)) / 255.0;
    return vec3(creSrgbToLin(c.r), creSrgbToLin(c.g), creSrgbToLin(c.b));
}

// March one posed part along a world ray. The transform is rigid, so cell-
// space t == world t. When clipOn != 0, solid cells whose WORLD position
// falls outside [clipLo, clipHi] are skipped and the march continues — the
// diorama's Q/E slice carves through a body like it carves through terrain.
bool creTracePart(int idx, vec3 ro, vec3 rd, float tCap,
                  int clipOn, vec3 clipLo, vec3 clipHi,
                  out float tHit, out vec3 nrm, out uint ciOut, out vec3 colOut) {
    CrePart p = creParts[idx];
    vec3 o   = vec3(dot(p.invX.xyz, ro), dot(p.invY.xyz, ro), dot(p.invZ.xyz, ro))
             + vec3(p.invX.w, p.invY.w, p.invZ.w);
    vec3 dir = vec3(dot(p.invX.xyz, rd), dot(p.invY.xyz, rd), dot(p.invZ.xyz, rd));
    vec3 dims = p.dimsPal.xyz;
    vec3 invD = 1.0 / dir;
    vec3 t0 = (vec3(0.0) - o) * invD, t1 = (dims - o) * invD;
    vec3 ts = min(t0, t1), tb = max(t0, t1);
    float tEnter = max(max(ts.x, ts.y), ts.z);
    float tExit  = min(min(tb.x, tb.y), min(tb.z, tCap));
    if (tExit <= 0.0 || tEnter > tExit) return false;
    float t = max(tEnter, 0.0);
    ivec3 gi = ivec3(dims + 0.5);
    ivec3 cell = clamp(ivec3(floor(o + dir * (t + 1e-4))), ivec3(0), gi - 1);
    ivec3 stp = ivec3(sign(dir));
    vec3 tDelta = abs(invD);
    vec3 tMaxV;
    for (int i = 0; i < 3; ++i) {
        float edge = float(cell[i]) + ((dir[i] > 0.0) ? 1.0 : 0.0);
        tMaxV[i] = (edge - o[i]) * invD[i];              // dir==0 -> inf, never stepped
    }
    int axis = (tEnter <= 0.0) ? -1
             : (tEnter == ts.x) ? 0 : (tEnter == ts.y) ? 1 : 2;
    // Budget = the exact worst-case cell count of a box crossing (dx+dy+dz),
    // so no loader-accepted part can out-run the march (a fixed cap silently
    // punched silhouette holes on large parts at grazing angles).
    int budget = gi.x + gi.y + gi.z;
    for (int s = 0; s < budget; ++s) {
        if (any(lessThan(cell, ivec3(0))) || any(greaterThanEqual(cell, gi))) break;
        int lin = cell.x + gi.x * (cell.y + gi.y * cell.z);
        uint word = creVox[uint(p.meta.x) + uint(lin >> 2)];
        uint v = (word >> uint((lin & 3) * 8)) & 0xFFu;
        if (v != 0u) {
            vec3 hp = ro + rd * t;
            if (clipOn == 0 || (all(greaterThanEqual(hp, clipLo)) &&
                                all(lessThanEqual(hp, clipHi)))) {
                vec3 nl = vec3(0.0);
                if (axis >= 0) nl[axis] = -float(stp[axis]);
                else           nl = -dir;                // degenerate: started inside
                nrm = normalize(vec3(dot(p.rowX.xyz, nl),
                                     dot(p.rowY.xyz, nl),
                                     dot(p.rowZ.xyz, nl)));
                ciOut  = v;
                colOut = crePalette(p, v);
                tHit   = t;
                return true;
            }
        }
        if (tMaxV.x < tMaxV.y && tMaxV.x < tMaxV.z)      { axis = 0; cell.x += stp.x; t = tMaxV.x; tMaxV.x += tDelta.x; }
        else if (tMaxV.y < tMaxV.z)                      { axis = 1; cell.y += stp.y; t = tMaxV.y; tMaxV.y += tDelta.y; }
        else                                             { axis = 2; cell.z += stp.z; t = tMaxV.z; tMaxV.z += tDelta.z; }
        if (t > tExit + 1e-3) break;
    }
    return false;
}
