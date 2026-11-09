// Voxel storage layout + DDA traversal: brickmap/brickpool SSBOs, brick
// constants, voxel sampling, and the anyHit shadow/occlusion ray. References
// the gridSize/bricksPerAxis uniforms declared by the including shader.

layout(std430, binding = 0) readonly buffer Brickmap  { uint bm[];  };
layout(std430, binding = 1) readonly buffer Brickpool { uint bp[];  };
layout(std430, binding = 2) readonly buffer Coarse    { uint coarseOcc[]; };  // 1 = cell occupied
const int BRICK = 8;
const uint BM_SOLID = 0x80000000u;
const int COARSE_BITS = 2;          // 4 bricks per coarse cell / axis (matches common.h)
const int COARSE      = 1 << COARSE_BITS;
uint sampleVoxel(ivec3 v, uint brickEntry) {
    uint bi = brickEntry - 1u;
    ivec3 l = v & (BRICK - 1);
    int linear = l.x + BRICK * (l.y + BRICK * l.z);
    uint word = bp[bi * 128u + uint(linear >> 2)];
    return (word >> ((uint(linear) & 3u) * 8u)) & 0xFFu;
}
// Window-space helpers. World-space brick coords wrap into the toroidal
// brickmap; anything outside the window's world-region reads as air.
vec3 windowLoV() {
    return vec3(float(windowOriginB.x * BRICK), 0.0, float(windowOriginB.y * BRICK));
}
// CHUNK-CONTIGUOUS slot layout (must mirror World::slotOfBrick): each 8x8
// brick chunk column owns one contiguous 32768-slot range, so the CPU can
// sync/blank a whole chunk with one buffer call while streaming.
int bmSlot(ivec3 brick) {
    int bx = brick.x & (bricksPerAxis.x - 1);
    int bz = brick.z & (bricksPerAxis.z - 1);
    int chunkSlot = (bx >> 3) + (bricksPerAxis.x >> 3) * (bz >> 3);
    int local = (bx & 7) + ((bz & 7) << 3) + (brick.y << 6);
    return chunkSlot * (64 * bricksPerAxis.y) + local;
}
bool brickInWindow(ivec3 brick) {
    return brick.x >= windowOriginB.x && brick.x < windowOriginB.x + bricksPerAxis.x
        && brick.z >= windowOriginB.y && brick.z < windowOriginB.y + bricksPerAxis.z
        && brick.y >= 0 && brick.y < bricksPerAxis.y;
}
uint getVoxelAt(ivec3 v) {
    ivec3 b = v >> 3;            // arithmetic shift: floor for negatives too
    if (!brickInWindow(b)) return 0u;
    uint entry = bm[bmSlot(b)];
    if (entry == 0u) return 0u;
    if ((entry & BM_SOLID) != 0u) return entry & 0xFFu;
    return sampleVoxel(v, entry);
}
void faceTangents(ivec3 N, out ivec3 T1, out ivec3 T2) {
    if      (abs(N.x) > 0) { T1 = ivec3(0,1,0); T2 = ivec3(0,0,1); }
    else if (abs(N.y) > 0) { T1 = ivec3(1,0,0); T2 = ivec3(0,0,1); }
    else                   { T1 = ivec3(1,0,0); T2 = ivec3(0,1,0); }
}
bool anyHit(vec3 ro, vec3 rd, float maxT) {
    vec3 invD = 1.0 / rd;
    vec3 winLo = windowLoV();
    vec3 t0 = (winLo - ro) * invD;
    vec3 t1 = (winLo + vec3(gridSize) - ro) * invD;
    vec3 ts = min(t0, t1);
    vec3 tb = max(t0, t1);
    float tEnter = max(max(ts.x, ts.y), ts.z);
    float tExit  = min(min(tb.x, tb.y), tb.z);
    if (tExit < 0.0 || tEnter > tExit) return false;
    float t = max(tEnter, 0.0) + 1e-4;
    if (t > maxT) return false;

    ivec3 stp = ivec3((rd.x > 0.0) ? 1 : -1,
                      (rd.y > 0.0) ? 1 : -1,
                      (rd.z > 0.0) ? 1 : -1);
    ivec3 brickLo = ivec3(windowOriginB.x, 0, windowOriginB.y);
    vec3 pBrick = (ro + rd * t) / float(BRICK);
    ivec3 brick = clamp(ivec3(floor(pBrick)), brickLo, brickLo + bricksPerAxis - 1);
    vec3 tDeltaB = abs(invD) * float(BRICK);
    vec3 tMaxB;
    for (int i = 0; i < 3; ++i) {
        float boundary = float(brick[i]) + ((stp[i] > 0) ? 1.0 : 0.0);
        tMaxB[i] = (boundary * float(BRICK) - ro[i]) * invD[i];
    }
    int outerHitAxis = -1;
    ivec3 cdim = bricksPerAxis >> COARSE_BITS;
    for (int o = 0; o < 256; ++o) {
        if (!brickInWindow(brick)) return false;

        // Coarse skip: if this whole coarse cell is empty, fast-forward to its
        // far face in one step instead of marching every empty brick inside it.
        // Occupancy is indexed by WRAPPED window-local brick coords (mirrors
        // World::buildCoarse / bmSlot's wrapping); the cell geometry below stays
        // in world space so the ray math is unchanged.
        ivec3 cellW = brick >> COARSE_BITS;                       // world coarse-cell coords
        ivec3 cellL = ivec3(cellW.x & (cdim.x - 1), cellW.y,
                            cellW.z & (cdim.z - 1));              // wrapped local -> occupancy index
        if (coarseOcc[cellL.x + cdim.x * (cellL.y + cdim.y * cellL.z)] == 0u) {
            ivec3 cmin = cellW << COARSE_BITS;       // the cell's min brick corner (world)
            vec3 tFar;
            for (int i = 0; i < 3; ++i) {
                int bb = (stp[i] > 0) ? (cmin[i] + COARSE) : cmin[i];
                tFar[i] = (float(bb * BRICK) - ro[i]) * invD[i];
            }
            float tExit = tFar.x; int exitAxis = 0;
            if (tFar.y < tExit) { tExit = tFar.y; exitAxis = 1; }
            if (tFar.z < tExit) { tExit = tFar.z; exitAxis = 2; }
            if (tExit > maxT) return false;
            vec3 pNext = ro + rd * (tExit + 1e-3);
            ivec3 nb = ivec3(floor(pNext / float(BRICK)));
            if (nb != brick) {          // forward-progress guard (degenerate axis)
                brick = nb;
                for (int i = 0; i < 3; ++i) {
                    float bnd = float(brick[i]) + ((stp[i] > 0) ? 1.0 : 0.0);
                    tMaxB[i] = (bnd * float(BRICK) - ro[i]) * invD[i];
                }
                outerHitAxis = exitAxis;
                continue;
            }
        }

        float tBrickEntry = (outerHitAxis < 0) ? t :
            ((outerHitAxis == 0) ? (tMaxB.x - tDeltaB.x)
            : (outerHitAxis == 1) ? (tMaxB.y - tDeltaB.y)
                                  : (tMaxB.z - tDeltaB.z));
        if (tBrickEntry > maxT) return false;

        uint entry = bm[bmSlot(brick)];
        if (entry != 0u) {
            if ((entry & BM_SOLID) != 0u) return true;
            vec3 pIn = ro + rd * (tBrickEntry + 1e-4);
            ivec3 voxel = ivec3(floor(pIn));
            voxel = clamp(voxel, brick * BRICK, brick * BRICK + (BRICK - 1));
            vec3 tMaxV;
            for (int i = 0; i < 3; ++i) {
                float bv = float(voxel[i]) + ((stp[i] > 0) ? 1.0 : 0.0);
                tMaxV[i] = (bv - ro[i]) * invD[i];
            }
            vec3 tDeltaV = abs(invD);
            for (int s = 0; s < 24; ++s) {
                if ((voxel >> 3) != brick) break;
                if (sampleVoxel(voxel, entry) != 0u) return true;
                if (tMaxV.x < tMaxV.y && tMaxV.x < tMaxV.z) {
                    voxel.x += stp.x; tMaxV.x += tDeltaV.x;
                } else if (tMaxV.y < tMaxV.z) {
                    voxel.y += stp.y; tMaxV.y += tDeltaV.y;
                } else {
                    voxel.z += stp.z; tMaxV.z += tDeltaV.z;
                }
            }
        }
        if (tMaxB.x < tMaxB.y && tMaxB.x < tMaxB.z) {
            brick.x += stp.x; tMaxB.x += tDeltaB.x; outerHitAxis = 0;
        } else if (tMaxB.y < tMaxB.z) {
            brick.y += stp.y; tMaxB.y += tDeltaB.y; outerHitAxis = 1;
        } else {
            brick.z += stp.z; tMaxB.z += tDeltaB.z; outerHitAxis = 2;
        }
    }
    return false;
}

// Trace a ray and return its first solid hit (position, geometric face normal,
// material). Same windowed DDA + coarse-skip as anyHit, but records the hit
// instead of just reporting occlusion. Used by the reflection pass to find what
// a smooth surface reflects. Returns false (and leaves outputs unset) on a miss.
bool traceRay(vec3 ro, vec3 rd, float maxT, out vec3 hitPos, out vec3 hitNormal, out uint hitMat) {
    hitPos = ro; hitNormal = vec3(0.0); hitMat = 0u;
    vec3 invD = 1.0 / rd;
    vec3 winLo = windowLoV();
    vec3 t0 = (winLo - ro) * invD;
    vec3 t1 = (winLo + vec3(gridSize) - ro) * invD;
    vec3 ts = min(t0, t1);
    vec3 tb = max(t0, t1);
    float tEnter = max(max(ts.x, ts.y), ts.z);
    float tExit  = min(min(tb.x, tb.y), tb.z);
    if (tExit < 0.0 || tEnter > tExit) return false;
    float t = max(tEnter, 0.0) + 1e-4;
    if (t > maxT) return false;

    ivec3 stp = ivec3((rd.x > 0.0) ? 1 : -1,
                      (rd.y > 0.0) ? 1 : -1,
                      (rd.z > 0.0) ? 1 : -1);
    ivec3 brickLo = ivec3(windowOriginB.x, 0, windowOriginB.y);
    vec3 pBrick = (ro + rd * t) / float(BRICK);
    ivec3 brick = clamp(ivec3(floor(pBrick)), brickLo, brickLo + bricksPerAxis - 1);
    vec3 tDeltaB = abs(invD) * float(BRICK);
    vec3 tMaxB;
    for (int i = 0; i < 3; ++i) {
        float boundary = float(brick[i]) + ((stp[i] > 0) ? 1.0 : 0.0);
        tMaxB[i] = (boundary * float(BRICK) - ro[i]) * invD[i];
    }
    int outerHitAxis = -1;
    ivec3 cdim = bricksPerAxis >> COARSE_BITS;
    for (int o = 0; o < 256; ++o) {
        if (!brickInWindow(brick)) return false;

        ivec3 cellW = brick >> COARSE_BITS;
        ivec3 cellL = ivec3(cellW.x & (cdim.x - 1), cellW.y, cellW.z & (cdim.z - 1));
        if (coarseOcc[cellL.x + cdim.x * (cellL.y + cdim.y * cellL.z)] == 0u) {
            ivec3 cmin = cellW << COARSE_BITS;
            vec3 tFar;
            for (int i = 0; i < 3; ++i) {
                int bb = (stp[i] > 0) ? (cmin[i] + COARSE) : cmin[i];
                tFar[i] = (float(bb * BRICK) - ro[i]) * invD[i];
            }
            float te = tFar.x; int ea = 0;
            if (tFar.y < te) { te = tFar.y; ea = 1; }
            if (tFar.z < te) { te = tFar.z; ea = 2; }
            if (te > maxT) return false;
            vec3 pN = ro + rd * (te + 1e-3);
            ivec3 nb = ivec3(floor(pN / float(BRICK)));
            if (nb != brick) {          // forward-progress guard (degenerate axis)
                brick = nb;
                for (int i = 0; i < 3; ++i) {
                    float bnd = float(brick[i]) + ((stp[i] > 0) ? 1.0 : 0.0);
                    tMaxB[i] = (bnd * float(BRICK) - ro[i]) * invD[i];
                }
                outerHitAxis = ea;
                continue;
            }
        }

        float tBrickEntry = (outerHitAxis < 0) ? t :
            ((outerHitAxis == 0) ? (tMaxB.x - tDeltaB.x)
            : (outerHitAxis == 1) ? (tMaxB.y - tDeltaB.y)
                                  : (tMaxB.z - tDeltaB.z));
        if (tBrickEntry > maxT) return false;

        uint entry = bm[bmSlot(brick)];
        if (entry != 0u) {
            int   hitAxis = outerHitAxis;
            float tHit    = max(tBrickEntry, 0.0);
            if ((entry & BM_SOLID) != 0u) {
                hitMat    = entry & 0xFFu;
                hitPos    = ro + rd * tHit;
                hitNormal = (hitAxis == 0) ? vec3(-float(stp.x), 0, 0)
                          : (hitAxis == 1) ? vec3(0, -float(stp.y), 0)
                          : (hitAxis == 2) ? vec3(0, 0, -float(stp.z)) : -rd;
                return true;
            }
            vec3 pIn = ro + rd * (tBrickEntry + 1e-4);
            ivec3 voxel = ivec3(floor(pIn));
            voxel = clamp(voxel, brick * BRICK, brick * BRICK + (BRICK - 1));
            vec3 tMaxV;
            for (int i = 0; i < 3; ++i) {
                float bv = float(voxel[i]) + ((stp[i] > 0) ? 1.0 : 0.0);
                tMaxV[i] = (bv - ro[i]) * invD[i];
            }
            vec3 tDeltaV = abs(invD);
            for (int s = 0; s < 24; ++s) {
                if ((voxel >> 3) != brick) break;
                uint v = sampleVoxel(voxel, entry);
                if (v != 0u) {
                    tHit = (hitAxis == 0) ? (tMaxV.x - tDeltaV.x)
                         : (hitAxis == 1) ? (tMaxV.y - tDeltaV.y)
                         : (hitAxis == 2) ? (tMaxV.z - tDeltaV.z) : tBrickEntry;
                    hitMat    = v;
                    hitPos    = ro + rd * tHit;
                    hitNormal = (hitAxis == 0) ? vec3(-float(stp.x), 0, 0)
                              : (hitAxis == 1) ? vec3(0, -float(stp.y), 0)
                              : (hitAxis == 2) ? vec3(0, 0, -float(stp.z)) : -rd;
                    return true;
                }
                if (tMaxV.x < tMaxV.y && tMaxV.x < tMaxV.z) {
                    voxel.x += stp.x; tMaxV.x += tDeltaV.x; hitAxis = 0;
                } else if (tMaxV.y < tMaxV.z) {
                    voxel.y += stp.y; tMaxV.y += tDeltaV.y; hitAxis = 1;
                } else {
                    voxel.z += stp.z; tMaxV.z += tDeltaV.z; hitAxis = 2;
                }
            }
        }
        if (tMaxB.x < tMaxB.y && tMaxB.x < tMaxB.z) {
            brick.x += stp.x; tMaxB.x += tDeltaB.x; outerHitAxis = 0;
        } else if (tMaxB.y < tMaxB.z) {
            brick.y += stp.y; tMaxB.y += tDeltaB.y; outerHitAxis = 1;
        } else {
            brick.z += stp.z; tMaxB.z += tDeltaB.z; outerHitAxis = 2;
        }
    }
    return false;
}
