#include "player.h"
#include "world.h"
#include "common.h"

#include <cmath>

// isBlockingForPlayer moved to player.h: the pathfinder (path.cpp) and the
// collision box must agree on what blocks.

glm::vec3 Player::eye()  const { return pos + glm::vec3(0.0f, eyeOff, 0.0f); }

bool overlapsSolid(const World& w, glm::vec3 feet, float height) {
    // Check EVERY voxel the player's AABB overlaps -- NOT a few sample points.
    // Point sampling left ~4-voxel vertical gaps, so thin slabs/leaves between
    // samples were missed: the player walked through thin slabs and got stuck
    // inside tree canopies when jumping. A full scan is exact and still cheap.
    // `height` defaults to standing; a crouching player passes a shorter box.
    int x0 = int(std::floor(feet.x - P_RADIUS)), x1 = int(std::floor(feet.x + P_RADIUS));
    int z0 = int(std::floor(feet.z - P_RADIUS)), z1 = int(std::floor(feet.z + P_RADIUS));
    int y0 = int(std::floor(feet.y + 0.5f));            // just above feet (skip floor block)
    int y1 = int(std::floor(feet.y + height - 0.5f));   // up to the head
    for (int y = y0; y <= y1; ++y)
    for (int x = x0; x <= x1; ++x)
    for (int z = z0; z <= z1; ++z)
        if (isBlockingForPlayer(w.getVoxel(x, y, z))) return true;
    return false;
}

bool isOnGround(const World& w, glm::vec3 feet) {
    int y = int(std::floor(feet.y - 0.5f));
    if (y < 0) return false;
    // Full footprint (every column under the player), so a narrow ledge/slab
    // anywhere beneath the body counts as ground rather than slipping a gap.
    int x0 = int(std::floor(feet.x - P_RADIUS)), x1 = int(std::floor(feet.x + P_RADIUS));
    int z0 = int(std::floor(feet.z - P_RADIUS)), z1 = int(std::floor(feet.z + P_RADIUS));
    for (int x = x0; x <= x1; ++x)
    for (int z = z0; z <= z1; ++z) {
        if (isBlockingForPlayer(w.getVoxel(x, y, z))) return true;
    }
    return false;
}

// Any body voxel in water → drives swim physics. Sample roughly the same
// points as overlapsSolid so partial immersion (waist-deep) still counts.
bool isInWater(const World& w, glm::vec3 feet) {
    static const float xs[5] = {0.0f, -P_RADIUS,  P_RADIUS, -P_RADIUS,  P_RADIUS};
    static const float zs[5] = {0.0f, -P_RADIUS, -P_RADIUS,  P_RADIUS,  P_RADIUS};
    static const float hs[5] = {1.0f, P_HEIGHT*0.25f, P_HEIGHT*0.50f,
                                P_HEIGHT*0.75f, P_HEIGHT - 2.0f};
    for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 5; ++j) {
        if (w.getVoxel(int(std::floor(feet.x + xs[i])),
                       int(std::floor(feet.y + hs[j])),
                       int(std::floor(feet.z + zs[i]))) == MAT_WATER)
            return true;
    }
    return false;
}

