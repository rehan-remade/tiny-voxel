#include "granular.h"
#include "world.h"
#include "materials.h"
#include <algorithm>
#include <unordered_set>
#include <vector>
#include <glm/glm.hpp>

namespace {
// Max powder cells stepped per tick. High enough that ordinary piles process in
// ONE tick (so the full pass runs and settled grains DRAIN out of the active
// set). If a pile exceeds this, the truncation path below keeps only the
// unprocessed frontier + actual movers, so it still drains progressively rather
// than locking the whole mass in the set forever.
constexpr int CAP        = 64000;
constexpr int FALL_SPEED = 5;       // cells/tick a grain drops (uniform -> cohesive; fast settle)
}

void simulateGranular(World& w, EditDelta& d, std::mt19937& rng) {
    if (w.fallingBricks.empty()) return;

    // 1. Snapshot every powder cell across the active bricks (so we never
    //    iterate a set we're mutating); drop bricks that hold no powder.
    std::vector<glm::ivec3> cells;
    cells.reserve(2048);
    std::vector<int> dead;
    for (int slot : w.fallingBricks) {
        glm::ivec3 b = w.brickOfSlot(slot);
        int X = b.x * BRICK, Y = b.y * BRICK, Z = b.z * BRICK;
        bool any = false;
        for (int lz = 0; lz < BRICK; ++lz)
        for (int ly = 0; ly < BRICK; ++ly)
        for (int lx = 0; lx < BRICK; ++lx) {
            if (isPowder(w.getVoxel(X + lx, Y + ly, Z + lz))) {
                cells.push_back({ X + lx, Y + ly, Z + lz });
                any = true;
            }
        }
        if (!any) dead.push_back(slot);
    }
    for (int s : dead) w.fallingBricks.erase(s);
    if (cells.empty()) return;

    // 2. Order: shuffle for lateral fairness, then stable-sort BOTTOM-UP. The
    //    bottom-up pass is the key to cohesion — by the time a grain is handled
    //    the one below it has already dropped and left a gap, so the whole
    //    column falls as a unit in ONE tick (no hovering top, fast settle).
    std::shuffle(cells.begin(), cells.end(), rng);
    std::stable_sort(cells.begin(), cells.end(),
                     [](const glm::ivec3& a, const glm::ivec3& b){ return a.y < b.y; });
    int n = std::min(int(cells.size()), CAP);

    std::unordered_set<int> woken;          // next-tick active set (settled bricks drop out)
    auto wake = [&](int x, int y, int z) {
        if (y < 0 || y >= GRID_Y || !w.voxelInWindow(x, z)) return;
        woken.insert(w.slotOfBrick(x >> BRICK_BITS, y >> BRICK_BITS, z >> BRICK_BITS));
    };
    static const int DIAG[4][2] = { {1,0}, {-1,0}, {0,1}, {0,-1} };

    for (int i = 0; i < n; ++i) {
        glm::ivec3 p = cells[i];
        uint8_t m = w.getVoxel(p.x, p.y, p.z);
        if (!isPowder(m)) continue;          // already moved this tick

        // (a) Fall straight down at the uniform speed, sweeping up to FALL_SPEED
        //     cells and stopping at the first obstacle (no tunneling).
        if (p.y > 0 && w.getVoxel(p.x, p.y - 1, p.z) == MAT_AIR) {
            int step = 0;
            while (step < FALL_SPEED && (p.y - step - 1) >= 0 &&
                   w.getVoxel(p.x, p.y - step - 1, p.z) == MAT_AIR)
                ++step;
            int ny = p.y - step;
            w.setVoxelSettle(p.x, p.y, p.z, MAT_AIR, d);
            w.setVoxelSettle(p.x, ny,  p.z, m,       d);
            wake(p.x, ny, p.z);                              // keep falling next tick
            wake(p.x, p.y + 1, p.z);                         // a grain above can now fall
            continue;
        }

        // (b) Can't fall: slump to a random OPEN down-diagonal (pile relaxes to
        //     its repose angle).
        int r = int(rng() & 3);
        bool moved = false;
        for (int k = 0; k < 4 && !moved; ++k) {
            const int* dd = DIAG[(r + k) & 3];
            int nx = p.x + dd[0], nz = p.z + dd[1];
            if (p.y <= 0)                                continue;
            if (!w.voxelInWindow(nx, nz))               continue;
            if (w.getVoxel(nx, p.y,     nz) != MAT_AIR) continue;   // side must be clear
            if (w.getVoxel(nx, p.y - 1, nz) != MAT_AIR) continue;   // and the move descends
            w.setVoxelSettle(p.x, p.y,     p.z, MAT_AIR, d);
            w.setVoxelSettle(nx, p.y - 1, nz, m,        d);
            wake(nx, p.y - 1, nz);
            wake(p.x, p.y + 1, p.z);                                // disturb the column above
            for (int j = 0; j < 4; ++j)                             // and lateral neighbours
                wake(p.x + DIAG[j][0], p.y, p.z + DIAG[j][1]);
            moved = true;
        }
        (void)moved;                          // rest: not woken -> its brick drains out
    }

    // Next-tick active set. `woken` already holds the neighbourhood of every
    // grain that MOVED; processed grains that rested are intentionally absent
    // so a settled pile DRAINS to nothing. If we hit the cap, the cells beyond
    // `n` never got a turn (bottom-up: these are the highest, freshest grains) —
    // keep their bricks so the unstable frontier isn't stranded.
    for (int i = n; i < int(cells.size()); ++i) {
        const glm::ivec3& c = cells[i];
        woken.insert(w.slotOfBrick(c.x >> BRICK_BITS, c.y >> BRICK_BITS, c.z >> BRICK_BITS));
    }
    w.fallingBricks.swap(woken);
}
