#include "water.h"
#include "world.h"
#include "common.h"

#include <algorithm>
#include <vector>

void simulateWater(World& w, EditDelta& d, std::mt19937& rng) {
    if (w.wetBricks.empty()) return;

    // Snapshot every water voxel currently in any wet brick, so a single tick
    // doesn't double-process water it just moved.
    std::vector<glm::ivec3> cells;
    cells.reserve(4096);
    std::vector<int> dryBricks;     // wet bricks with no water left after scan

    for (int slot : w.wetBricks) {
        glm::ivec3 b = w.brickOfSlot(slot);   // window-resolved WORLD brick
        int bx = b.x, by = b.y, bz = b.z;
        bool any = false;
        for (int lz = 0; lz < BRICK; ++lz)
        for (int ly = 0; ly < BRICK; ++ly)
        for (int lx = 0; lx < BRICK; ++lx) {
            int x = bx * BRICK + lx;
            int y = by * BRICK + ly;
            int z = bz * BRICK + lz;
            if (w.getVoxel(x, y, z) == MAT_WATER) {
                cells.push_back({x, y, z});
                any = true;
            }
        }
        if (!any) dryBricks.push_back(slot);
    }
    for (int s : dryBricks) w.wetBricks.erase(s);
    if (cells.empty()) return;

    // Shuffle to avoid directional bias.
    std::shuffle(cells.begin(), cells.end(), rng);

    constexpr int CAP = 12000;
    int n = std::min((int)cells.size(), CAP);

    auto markWet = [&](int x, int y, int z) {
        w.wetBricks.insert(w.slotOfBrick(x >> BRICK_BITS, y >> BRICK_BITS, z >> BRICK_BITS));
    };

    static const int DIRS[4][2] = {{1,0}, {-1,0}, {0,1}, {0,-1}};

    for (int i = 0; i < n; ++i) {
        glm::ivec3 p = cells[i];
        if (w.getVoxel(p.x, p.y, p.z) != MAT_WATER) continue; // already moved this tick

        // 1. Fall straight down if possible.
        if (p.y > 0 && w.getVoxel(p.x, p.y - 1, p.z) == 0) {
            w.setVoxel(p.x, p.y,     p.z, 0,         d);
            w.setVoxel(p.x, p.y - 1, p.z, MAT_WATER, d);
            markWet(p.x, p.y - 1, p.z);
            continue;
        }

        // 2. Spread sideways toward a lower neighbour (one whose cell-below
        //    is empty), so water on slopes drains downhill quickly.
        int start = int(rng() & 3);
        bool moved = false;
        for (int k = 0; k < 4; ++k) {
            int di = (start + k) & 3;
            int nx = p.x + DIRS[di][0];
            int nz = p.z + DIRS[di][1];
            if (!w.voxelInWindow(nx, nz)) continue;
            if (w.getVoxel(nx, p.y,     nz) != 0) continue;
            if (w.getVoxel(nx, p.y - 1, nz) != 0) continue;     // not draining-lower
            w.setVoxel(p.x, p.y, p.z, 0,         d);
            w.setVoxel(nx,  p.y, nz, MAT_WATER, d);
            markWet(nx, p.y, nz);
            moved = true;
            break;
        }
        // No flat-plane creep: without a "level" system that would just
        // produce endless brownian shuffling. Water settles when it can't
        // descend any further.
        (void)moved;
    }
}
