#include "fire.h"
#include "world.h"
#include <algorithm>
#include <vector>

namespace {
constexpr int     FIRE_CAP    = 6000;   // max cells processed per tick (< water's 12000)
constexpr int     FIRE_RADIUS = 48;     // horizontal spread limit from the cast point (voxels)
constexpr int     FIRE_R2     = FIRE_RADIUS * FIRE_RADIUS;
constexpr uint8_t EMBER_LIFE  = 2;      // ticks an ember glows before dying
constexpr uint8_t WAS_WOOD    = 0x80;   // high bit of the life value
constexpr uint8_t TICK_MASK   = 0x7F;

// Small discrete plants (grass tufts, flowers) ignite WHOLE in one go — no
// probabilistic creep — so a touched tuft never leaves stray single voxels.
bool burnsFully(uint8_t m) {
    return m == MAT_GRASS_BLADE
        || m == MAT_FLOWER_STEM || m == MAT_FLOWER_RED || m == MAT_FLOWER_YELLOW
        || m == MAT_FLOWER_BLUE || m == MAT_FLOWER_WHITE
        || m == MAT_CACTUS_BLOOM;   // small flower cluster -> lights whole, no stray bloom voxels
}

// Seed one voxel into the FIRE state, recording its fuel class + life, and the
// XZ of the cast that started this fire (so spread can be radius-bounded).
void lightVoxel(World& w, FireSim& f, int x, int y, int z, uint8_t orig,
                int64_t originXZ, EditDelta& d) {
    w.setVoxel(x, y, z, MAT_FIRE, d);
    int64_t kk = FireSim::key(x, y, z);
    f.life[kk]   = (fuelLife(orig) & TICK_MASK)
                 | ((orig == MAT_WOOD || orig == MAT_DESERT_WOOD) ? WAS_WOOD : 0);  // both leave a charred skeleton
    f.origin[kk] = originXZ;
    w.burnBricks.insert(w.slotOfBrick(x >> BRICK_BITS, y >> BRICK_BITS, z >> BRICK_BITS));
}
}  // namespace

bool isFlammable(uint8_t m) {
    // Grass ground + all vegetation. MAT_GRASS (the continuous terrain skin)
    // burns too — that's what makes a sweeping grass fire — but spread is capped
    // to FIRE_RADIUS of the cast point (see simulateFire), so it's a contained
    // blast, not an endless world-spanning sheet.
    switch (m) {
    case MAT_GRASS:        case MAT_WOOD:         case MAT_LEAVES:
    case MAT_GRASS_BLADE:  case MAT_FLOWER_STEM:  case MAT_FLOWER_RED:
    case MAT_FLOWER_YELLOW:case MAT_FLOWER_BLUE:  case MAT_FLOWER_WHITE:
    case MAT_CAVE_MOSS:    case MAT_VINE:         case MAT_VINE_DEAD:
    // Desert flora: dry shrubs (sage) + woody stems (creosote/ocotillo/palm
    // skirt) + the delicate cactus bloom all burn. The cactus BODY
    // (MAT_CACTUS_FLESH) does NOT — it's water-filled — and sand / sandstone /
    // gravel pavement never burn (they're not listed).
    case MAT_SHRUB_SAGE:   case MAT_DESERT_WOOD:  case MAT_CACTUS_BLOOM:
        return true;
    default:
        return false;
    }
}

uint8_t fuelLife(uint8_t m) {
    if (m == MAT_WOOD || m == MAT_DESERT_WOOD) return 10;          // logs / dry stems smoulder longest
    if (m == MAT_LEAVES || m == MAT_VINE || m == MAT_VINE_DEAD || m == MAT_SHRUB_SAGE) return 5;
    return 3;                                                      // grass / flowers / blooms flash
}

bool igniteVoxel(World& w, FireSim& f, int x, int y, int z, EditDelta& d) {
    uint8_t m = w.getVoxel(x, y, z);
    if (!isFlammable(m)) return false;
    lightVoxel(w, f, x, y, z, m, FireSim::packXZ(x, z), d);   // this cast is the origin
    return true;
}

void simulateFire(World& w, FireSim& f, EditDelta& d,
                  std::vector<glm::ivec3>* leafGone) {
    if (w.burnBricks.empty()) return;

    // 1. Snapshot every fire/ember cell across the active bricks (so we never
    //    iterate a set we're mutating), and prune bricks with none left.
    std::vector<glm::ivec3> cells;
    cells.reserve(2048);
    std::vector<int> dead;
    for (int slot : w.burnBricks) {
        glm::ivec3 b = w.brickOfSlot(slot);
        int X = b.x * BRICK, Y = b.y * BRICK, Z = b.z * BRICK;
        bool any = false;
        for (int lz = 0; lz < BRICK; ++lz)
        for (int ly = 0; ly < BRICK; ++ly)
        for (int lx = 0; lx < BRICK; ++lx) {
            uint8_t m = w.getVoxel(X + lx, Y + ly, Z + lz);
            if (m == MAT_FIRE || m == MAT_EMBER) {
                cells.push_back({ X + lx, Y + ly, Z + lz });
                any = true;
            }
        }
        if (!any) dead.push_back(slot);
    }
    for (int s : dead) w.burnBricks.erase(s);
    if (cells.empty()) return;

    std::shuffle(cells.begin(), cells.end(), f.rng);
    int n = std::min(int(cells.size()), FIRE_CAP);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);

    for (int i = 0; i < n; ++i) {
        glm::ivec3 p = cells[i];
        uint8_t m = w.getVoxel(p.x, p.y, p.z);
        if (m != MAT_FIRE && m != MAT_EMBER) continue;     // changed under us
        int64_t kk = FireSim::key(p.x, p.y, p.z);
        auto it = f.life.find(kk);
        uint8_t lv    = (it != f.life.end()) ? it->second : 0;
        uint8_t ticks = lv & TICK_MASK;
        uint8_t wood  = lv & WAS_WOOD;

        // 2. Spread — only an active flame ignites neighbours (embers can't),
        //    which halves late-stage pressure and makes the burn visibly die.
        //    26-neighbourhood: canopy leaves are often only DIAGONALLY linked,
        //    so face-only (6) spread strands single leaf voxels — diagonals fix
        //    that and make the burn fuller and faster.
        if (m == MAT_FIRE) {
            // Origin of this fire (the cast XZ), inherited down the chain.
            auto oit = f.origin.find(kk);
            int64_t originXZ = (oit != f.origin.end()) ? oit->second
                                                       : FireSim::packXZ(p.x, p.z);
            int ox = int(int32_t(uint32_t(originXZ)));
            int oz = int(int32_t(uint32_t(originXZ >> 32)));
            for (int dy = -1; dy <= 1; ++dy)
            for (int dz = -1; dz <= 1; ++dz)
            for (int dx = -1; dx <= 1; ++dx) {
                if (!dx && !dy && !dz) continue;
                int nx = p.x + dx, ny = p.y + dy, nz = p.z + dz;
                if (ny < 0 || ny >= GRID_Y) continue;
                if (!w.voxelInWindow(nx, nz))    continue;
                // Radius bound (HORIZONTAL, from the cast point): caps the
                // ground/forest sweep so fire can't run across the world, while
                // leaving vertical climbing (tree canopies) unconstrained.
                int hx = nx - ox, hz = nz - oz;
                if (hx * hx + hz * hz > FIRE_R2) continue;
                uint8_t nm = w.getVoxel(nx, ny, nz);
                if (!isFlammable(nm)) continue;
                float chance;
                if (burnsFully(nm)) {
                    chance = 1.0f;                          // whole tuft/flower lights
                } else {
                    chance = (dy > 0) ? 0.50f : 0.30f;      // rises faster than it creeps
                    if (nm == MAT_LEAVES || nm == MAT_GRASS) chance *= 1.6f;
                }
                if (u01(f.rng) < chance)
                    lightVoxel(w, f, nx, ny, nz, nm, originXZ, d);
            }
        }

        // 3. Decay — one-way; this is what drains the active set.
        if (ticks > 0) --ticks;
        if (ticks > 0) {
            f.life[kk] = ticks | wood;
        } else if (m == MAT_FIRE) {
            w.setVoxel(p.x, p.y, p.z, MAT_EMBER, d);        // flame -> coal (still ticking)
            f.life[kk] = (EMBER_LIFE & TICK_MASK) | wood;   // origin kept (still in set)
        } else {                                           // ember spent -> residue, leaves the set
            w.setVoxel(p.x, p.y, p.z, wood ? MAT_CHAR : MAT_AIR, d);
            // Foliage burned clean away: the canopy above/around may have
            // just lost its support chain — hand the position to forestry's
            // leaf-decay sweep. (Wood becomes solid CHAR and keeps
            // supporting; no seed needed.)
            if (!wood && leafGone) leafGone->push_back(glm::ivec3(p));
            f.life.erase(kk);
            f.origin.erase(kk);
        }
    }
}

void douseAllFire(World& w, FireSim& f, EditDelta& d,
                  std::vector<glm::ivec3>* removed) {
    for (int slot : w.burnBricks) {
        glm::ivec3 b = w.brickOfSlot(slot);
        int X = b.x * BRICK, Y = b.y * BRICK, Z = b.z * BRICK;
        for (int lz = 0; lz < BRICK; ++lz)
        for (int ly = 0; ly < BRICK; ++ly)
        for (int lx = 0; lx < BRICK; ++lx) {
            uint8_t m = w.getVoxel(X + lx, Y + ly, Z + lz);
            if (m == MAT_FIRE || m == MAT_EMBER) {
                w.setVoxel(X + lx, Y + ly, Z + lz, MAT_AIR, d);
                if (removed)
                    removed->push_back(glm::ivec3(X + lx, Y + ly, Z + lz));
            }
        }
    }
    w.burnBricks.clear();
    f.life.clear();
    f.origin.clear();
}
