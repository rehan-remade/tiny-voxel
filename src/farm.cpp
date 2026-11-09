#include "farm.h"
#include "world.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>

// =============================================================================
// Material classes for tilling. A column's "top" is the first voxel that is
// neither air/water nor thin flora (grass tufts, flowers, weeds we mow when
// tilling). Ground is tillable; a blocker (tree, cactus, existing crop) makes
// the whole cell invalid.
// =============================================================================
namespace {

bool isGround(uint8_t m) {
    switch (m) {
        case MAT_DIRT: case MAT_GRASS: case MAT_STONE: case MAT_GREYSTONE:
        case MAT_DARKSTONE: case MAT_SAND: case MAT_SAND_STATIC:
        case MAT_SANDSTONE: case MAT_DESERT_PAVEMENT: case MAT_FARMLAND:
            return true;
        default: return false;
    }
}
bool isThinFlora(uint8_t m) {
    switch (m) {
        case MAT_GRASS_BLADE: case MAT_FLOWER_STEM: case MAT_FLOWER_RED:
        case MAT_FLOWER_YELLOW: case MAT_FLOWER_BLUE: case MAT_FLOWER_WHITE:
        case MAT_SHRUB_SAGE: case MAT_CACTUS_BLOOM: case MAT_VINE:
        case MAT_VINE_DEAD:
            return true;
        default: return false;
    }
}
constexpr int BLOCKED = INT_MIN;

// First ground voxel Y in a column near yHint, mowing thin flora on the way
// down. Returns BLOCKED if a tree/crop/solid blocker caps the column, or
// INT_MIN+1 ("none") if no ground was found in range.
constexpr int NO_GROUND = INT_MIN + 1;
int columnTop(const World& w, int x, int z, int yHint) {
    // Wide window: a tilled cell may span a slope, so a column's ground can sit
    // well above/below the picked hint. Must comfortably cover the max spread
    // we still accept (a cell-width cliff).
    int hi = yHint + 16, lo = yHint - 16;
    for (int y = hi; y >= lo; --y) {
        uint8_t v = w.getVoxel(x, y, z);
        if (v == 0 || v == MAT_WATER || isThinFlora(v)) continue;
        if (isGround(v)) return y;
        return BLOCKED;                 // wood/leaves/cactus/crop caps the column
    }
    return NO_GROUND;
}

// splitmix32: deterministic per-crop RNG.
struct Rng {
    uint32_t s;
    explicit Rng(uint32_t seed) : s(seed) {}
    uint32_t next() {
        s += 0x9E3779B9u;
        uint32_t z = s;
        z = (z ^ (z >> 16)) * 0x85EBCA6Bu;
        z = (z ^ (z >> 13)) * 0xC2B2AE35u;
        return z ^ (z >> 16);
    }
    int   range(int n) { return int(next() % uint32_t(n)); }
    float frac()       { return float(next() & 0xFFFFFFu) / float(0x1000000); }  // 0..1
};

uint32_t cellGenome(int cx, int cz) {
    uint32_t h = uint32_t(cx) * 0x27D4EB2Fu ^ uint32_t(cz) * 0x165667B1u;
    h = (h ^ (h >> 15)) * 0x2C1B3C6Du;
    h = (h ^ (h >> 12)) * 0x297A2D39u;
    return h ^ (h >> 15);
}

// 8 compass directions on the ground plane.
const glm::ivec2 DIRS[8] = {
    {1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {-1, 1}, {1, -1}, {-1, -1}
};

}  // namespace

// Species registry. growScale multiplies the base /croptime pacing (radish is
// the quick win, corn the season-long wait); root species grow their body
// BURIED (the field shows only greens + a shoulder) and get the two-tug
// yank-from-the-soil harvest.
const CropSpeciesDef CROP_DEFS[CROP_SPECIES_N] = {
    //  name       flesh        grow  root
    { "Pumpkin", MAT_PUMPKIN, 1.00f, false },
    { "Carrot",  MAT_CARROT,  0.75f, true  },
    { "Tomato",  MAT_TOMATO,  1.15f, false },
    { "Cabbage", MAT_CABBAGE, 0.90f, false },
    { "Corn",    MAT_CORN,    1.30f, false },
    { "Radish",  MAT_RADISH,  0.50f, true  },
};

// =============================================================================
// Deterministic crop generator, one silhouette language per species:
//   pumpkin — sprawling vine crown, then one plump centred fruit (unchanged);
//   carrot/radish — the body grows BURIED: the field shows a frond/leaf
//     rosette and a flesh "shoulder" breaking the soil (the harvest yank
//     reveals the rest);
//   tomato — an upright staked bush that hangs discrete fruits;
//   cabbage — a flat rosette that hearts up into a wrapped head;
//   corn — a tall stalk with alternating blades, golden ears, and a tassel.
// Pure function of (root, species, genome, stage, size); everything stays
// ~inside the cell.
// =============================================================================
namespace {
using VoxList = std::vector<std::pair<glm::ivec3, uint8_t>>;

// Low ring of ground leaves around the plant (shared skirt look).
void leafSkirt(VoxList& out, Rng& rng, glm::ivec3 base, int radius, int n) {
    for (int i = 0; i < n; ++i) {
        glm::ivec2 d = DIRS[rng.range(8)];
        out.push_back({ base + glm::ivec3(d.x * radius, 0, d.y * radius),
                        MAT_CROP_LEAF });
    }
}

// The classic pumpkin stages — EXACTLY the shapes this generator has always
// produced (regrows of pre-species saves must not reshuffle a field).
void pumpkinVoxels(const Crop& c, Rng& rng, glm::ivec3 base, VoxList& out) {
    int stage = c.stage;
    if (stage == FARM_STAGE_MAX) {
        int r = (c.size == 2) ? 3 : (c.size == 1) ? 2 : 1;   // 7 / 5 / 3 wide
        int h = r + 1;                           // plump, a touch taller than half
        // Big fruit dead-centre; only the small tier gets a little nudge.
        glm::ivec3 fc = (c.size >= 2) ? base
                                      : base + glm::ivec3(rng.range(3) - 1, 0,
                                                          rng.range(3) - 1);
        for (int dx = -r; dx <= r; ++dx)
            for (int dz = -r; dz <= r; ++dz)
                for (int dy = 0; dy <= h; ++dy) {
                    float ex = float(dx) / (r + 0.7f);
                    float ez = float(dz) / (r + 0.7f);
                    float ey = (float(dy) - h * 0.5f) / (h * 0.62f);
                    if (ex * ex + ez * ez + ey * ey <= 1.0f)
                        out.push_back({ fc + glm::ivec3(dx, dy, dz), MAT_PUMPKIN });
                }
        out.push_back({ fc + glm::ivec3(0, h + 1, 0), MAT_CROP_VINE });
        leafSkirt(out, rng, base, r + 1, 4 + rng.range(3));   // 4..6
        return;
    }

    // Growing: crown + sprawling vine arms.
    int crownH = (stage == 0) ? 1 : 2;
    for (int y = 0; y < crownH; ++y)
        out.push_back({ base + glm::ivec3(0, y, 0), MAT_CROP_VINE });
    if (stage >= 1)
        out.push_back({ base + glm::ivec3(0, crownH, 0), MAT_CROP_LEAF });

    if (stage == 0) return;   // sprout: just the little crown

    int numArms = 2 + rng.range(3);            // 2..4
    int armLen  = (stage == 1) ? 2 : 3;        // reach further once flowering
    // Directions to prefer: the plant leans its vines toward open plots.
    int pref[8], np = 0;
    for (int i = 0; i < 8; ++i) if (c.freeDirs & (1u << i)) pref[np++] = i;
    glm::ivec3 flowerTip = base;
    for (int a = 0; a < numArms; ++a) {
        // One roll drives both the lean decision and the index (stable shape).
        uint32_t pick = rng.next();
        int di = (np > 0 && (pick & 3u) != 0u) ? pref[(pick >> 2) % np]
                                               : int((pick >> 2) % 8u);
        glm::ivec2 d = DIRS[di];
        glm::ivec3 tip = base;
        for (int s = 1; s <= armLen; ++s) {
            glm::ivec3 v = base + glm::ivec3(d.x * s, 0, d.y * s);
            out.push_back({ v, MAT_CROP_VINE });
            tip = v;
            if (s >= armLen - 1)
                out.push_back({ v + glm::ivec3(0, 1, 0), MAT_CROP_LEAF });
        }
        if (a == 0) flowerTip = tip;
    }

    // Blossom at the first arm's tip (flowering stage; becomes fruit next).
    if (stage == 2)
        out.push_back({ flowerTip + glm::ivec3(0, 1, 0), MAT_CROP_FLOWER });
}

// Root vegetables (carrot / radish): a frond rosette; when ripe, a flesh
// shoulder breaks the soil — small teaser for the yank. Radish keeps a lower,
// rounder tuft; carrot fronds stand taller.
void rootVoxels(const Crop& c, Rng& rng, glm::ivec3 base, VoxList& out) {
    uint8_t flesh   = CROP_DEFS[c.species].flesh;
    bool    carrot  = (c.species == CROP_CARROT);
    int     stage   = c.stage;

    if (stage == FARM_STAGE_MAX) {
        // Shoulder: the buried body's crown at ground level, sized by grade.
        if (c.size == 0) {
            out.push_back({ base, flesh });
        } else if (c.size == 1) {
            out.push_back({ base, flesh });
            for (int i = 0; i < 4; ++i)
                out.push_back({ base + glm::ivec3(DIRS[i].x, 0, DIRS[i].y), flesh });
        } else {
            for (int dx = -1; dx <= 1; ++dx)
                for (int dz = -1; dz <= 1; ++dz)
                    out.push_back({ base + glm::ivec3(dx, 0, dz), flesh });
            // A giant bulges out of the soil.
            out.push_back({ base + glm::ivec3(0, 1, 0), flesh });
            for (int i = 0; i < 4; ++i)
                if (rng.frac() < 0.5f)
                    out.push_back({ base + glm::ivec3(DIRS[i].x, 1, DIRS[i].y), flesh });
        }
        // Greens on top of the shoulder.
        int tuftY = (c.size == 2) ? 2 : 1;
        int tuftH = carrot ? 2 : 1;
        for (int y = 0; y < tuftH; ++y)
            out.push_back({ base + glm::ivec3(0, tuftY + y, 0), MAT_CROP_VINE });
        int fronds = (carrot ? 4 : 3) + rng.range(2);
        for (int i = 0; i < fronds; ++i) {
            glm::ivec2 d = DIRS[rng.range(8)];
            out.push_back({ base + glm::ivec3(d.x, tuftY + tuftH - 1, d.y),
                            MAT_CROP_LEAF });
        }
        return;
    }

    // Growing: sprout -> spreading rosette (all greens; the body is buried).
    out.push_back({ base, MAT_CROP_VINE });
    if (stage == 0) return;
    int ring = (stage == 1) ? 4 : 8;           // orthos, then all 8
    for (int i = 0; i < ring; ++i)
        out.push_back({ base + glm::ivec3(DIRS[i].x, 0, DIRS[i].y), MAT_CROP_LEAF });
    if (stage >= 2) {
        out.push_back({ base + glm::ivec3(0, 1, 0),
                        carrot ? MAT_CROP_VINE : MAT_CROP_LEAF });
        if (carrot)   // carrots bolt a little flower crown before ripening
            out.push_back({ base + glm::ivec3(0, 2, 0), MAT_CROP_FLOWER });
    }
}

// Tomato: an upright bush. Ripe, it hangs 2/4/6 discrete fruits on the
// outside — each one is a collect flyer at harvest.
void tomatoVoxels(const Crop& c, Rng& rng, glm::ivec3 base, VoxList& out) {
    int stage = c.stage;
    if (stage < FARM_STAGE_MAX) {
        int stemH = 1 + stage;                 // 1 / 2 / 3
        for (int y = 0; y < stemH; ++y)
            out.push_back({ base + glm::ivec3(0, y, 0), MAT_CROP_VINE });
        if (stage >= 1) {
            out.push_back({ base + glm::ivec3(0, stemH, 0), MAT_CROP_LEAF });
            for (int i = 0; i < 2 + stage; ++i) {
                glm::ivec2 d = DIRS[rng.range(8)];
                out.push_back({ base + glm::ivec3(d.x, stemH - 1, d.y),
                                MAT_CROP_LEAF });
            }
        }
        if (stage == 2)   // blossoms where the fruit will hang
            for (int i = 0; i < 2; ++i) {
                glm::ivec2 d = DIRS[rng.range(4)];
                out.push_back({ base + glm::ivec3(d.x, 1, d.y), MAT_CROP_FLOWER });
            }
        return;
    }

    // Ripe bush. Fruits are pushed FIRST: stampCrop places voxels in list
    // order into air only (first-wins), and at reach 1 a fruit's cell
    // coincides with the bush frame's vine/leaf ring — fruits must claim
    // their cells or the frame silently hides them (a small tomato could
    // ripen with ZERO visible fruit while the harvest still paid out).
    int bushH  = (c.size == 2) ? 4 : 3;
    int nFruit = 2 + 2 * c.size;   // 2/4/6, each on its own (dir, height) spot
    int reach  = (c.size == 2) ? 2 : 1;   // giants hang one voxel further out
    int spot0  = rng.range(8);
    for (int i = 0; i < nFruit; ++i) {
        glm::ivec2 d = DIRS[(spot0 + i * 3) % 8];        // stride 3 visits all 8
        int fy = 1 + ((i & 1) != 0 ? 1 : 0);             // alternate two rows
        out.push_back({ base + glm::ivec3(d.x * reach, fy, d.y * reach),
                        MAT_TOMATO });
    }
    // Frame: stem + leaf crown + the vine/leaf ring (skips cells a fruit took).
    for (int y = 0; y < bushH; ++y)
        out.push_back({ base + glm::ivec3(0, y, 0), MAT_CROP_VINE });
    out.push_back({ base + glm::ivec3(0, bushH, 0), MAT_CROP_LEAF });
    for (int i = 0; i < 4; ++i) {
        glm::ivec2 d = DIRS[i];
        out.push_back({ base + glm::ivec3(d.x, bushH - 1, d.y), MAT_CROP_LEAF });
        out.push_back({ base + glm::ivec3(d.x, 1, d.y), MAT_CROP_VINE });
    }
    leafSkirt(out, rng, base, 2, 2 + rng.range(2));
}

// Cabbage: flat rosette hearts up into one wrapped head.
void cabbageVoxels(const Crop& c, Rng& rng, glm::ivec3 base, VoxList& out) {
    int stage = c.stage;
    if (stage < FARM_STAGE_MAX) {
        out.push_back({ base, MAT_CROP_VINE });
        if (stage == 0) return;
        int ring = (stage == 1) ? 4 : 8;
        for (int i = 0; i < ring; ++i)
            out.push_back({ base + glm::ivec3(DIRS[i].x, 0, DIRS[i].y),
                            MAT_CROP_LEAF });
        if (stage >= 2)   // the heart starts to close
            out.push_back({ base + glm::ivec3(0, 1, 0), MAT_CABBAGE });
        return;
    }

    // Ripe: a flattened head (wider than tall) with a leaf wrap.
    int r  = (c.size == 2) ? 3 : (c.size == 1) ? 2 : 1;   // 7 / 5 / 3 wide
    int ry = std::max(1, r - 1);
    for (int dx = -r; dx <= r; ++dx)
        for (int dz = -r; dz <= r; ++dz)
            for (int dy = 0; dy <= 2 * ry; ++dy) {
                float ex = float(dx) / (r + 0.5f);
                float ez = float(dz) / (r + 0.5f);
                float ey = (float(dy) - ry) / (ry + 0.5f);
                if (ex * ex + ez * ez + ey * ey <= 1.0f)
                    out.push_back({ base + glm::ivec3(dx, dy, dz), MAT_CABBAGE });
            }
    // Loose outer leaves curling up the sides.
    for (int i = 0; i < 3 + rng.range(3); ++i) {
        glm::ivec2 d = DIRS[rng.range(8)];
        out.push_back({ base + glm::ivec3(d.x * r, 1 + rng.range(ry), d.y * r),
                        MAT_CROP_LEAF });
    }
    leafSkirt(out, rng, base, r + 1, 3 + rng.range(3));
}

// Corn: a tall stalk, alternating blades, 1/2/3 golden ears, tassel on top.
void cornVoxels(const Crop& c, Rng& rng, glm::ivec3 base, VoxList& out) {
    int stage = c.stage;
    int stalkH = (stage == 0) ? 1 : (stage == 1) ? 3
               : (stage == 2) ? 5 : 5 + c.size;          // ripe: 5/6/7
    for (int y = 0; y < stalkH; ++y)
        out.push_back({ base + glm::ivec3(0, y, 0), MAT_CROP_VINE });
    if (stage == 0) return;

    if (stage == FARM_STAGE_MAX) {
        // Ears FIRST: stampCrop places voxels in list order into air only
        // (first-wins), and half of all d0 rolls put a leaf blade on an ear
        // cell — the ears must claim their columns or they render stunted.
        // Ears: 2-tall golden columns hugging the stalk, one husk leaf below.
        int nEars = 1 + c.size;
        int d0 = rng.range(4);
        for (int e = 0; e < nEars; ++e) {
            glm::ivec2 d = DIRS[(d0 + e) % 4];
            int ey = 2 + e;                              // stagger up the stalk
            out.push_back({ base + glm::ivec3(d.x, ey,     d.y), MAT_CORN });
            out.push_back({ base + glm::ivec3(d.x, ey + 1, d.y), MAT_CORN });
            out.push_back({ base + glm::ivec3(d.x, ey - 1, d.y), MAT_CROP_LEAF });
        }
    }

    // Alternating leaf blades up the stalk (skip cells an ear claimed).
    for (int y = 1; y < stalkH - 1; ++y) {
        glm::ivec2 d = DIRS[y & 3];
        out.push_back({ base + glm::ivec3(d.x, y, d.y), MAT_CROP_LEAF });
    }
    if (stage >= 2)   // tassel
        out.push_back({ base + glm::ivec3(0, stalkH, 0), MAT_CROP_FLOWER });
}
}  // namespace

void cropVoxels(const Crop& c, std::vector<std::pair<glm::ivec3, uint8_t>>& out) {
    Rng rng(c.genome);
    glm::ivec3 base = c.root + glm::ivec3(0, 1, 0);   // grows above the farmland
    switch (c.species) {
        case CROP_CARROT:
        case CROP_RADISH:  rootVoxels(c, rng, base, out);    break;
        case CROP_TOMATO:  tomatoVoxels(c, rng, base, out);  break;
        case CROP_CABBAGE: cabbageVoxels(c, rng, base, out); break;
        case CROP_CORN:    cornVoxels(c, rng, base, out);    break;
        default:           pumpkinVoxels(c, rng, base, out); break;
    }
}

// =============================================================================
// Stamp a crop's CURRENT stage: clear the voxels it owned, then place the new
// stage into empty air only (never overwriting terrain or a neighbour), and
// record the new footprint. All writes go through setVoxelSim (transient).
// =============================================================================
static void stampCrop(World& w, Crop& c, EditDelta& d) {
    // Clear ONLY our own crop voxels — after a scroll/reload the old footprint
    // coords may now hold regenerated terrain, and blanking those would gouge
    // holes in the ground.
    for (const glm::ivec3& v : c.occupied)
        if (w.voxelInWindow(v.x, v.z) && isCropMaterial(w.getVoxel(v.x, v.y, v.z)))
            w.setVoxelSim(v.x, v.y, v.z, 0, d);
    c.occupied.clear();

    std::vector<std::pair<glm::ivec3, uint8_t>> vox;
    cropVoxels(c, vox);
    for (auto& pm : vox) {
        const glm::ivec3& v = pm.first;
        if (!w.voxelInWindow(v.x, v.z)) continue;
        if (w.getVoxel(v.x, v.y, v.z) != 0) continue;   // only into air
        w.setVoxelSim(v.x, v.y, v.z, pm.second, d);
        c.occupied.push_back(v);
    }
}

// Find the MAT_FARMLAND voxel Y in a column (a tilled bed, even if a crop now
// stands on it), or INT_MIN if none. Used for field continuity.
static int farmlandY(const World& w, int x, int z, int yHint) {
    for (int y = yHint + 16; y >= yHint - 16; --y)
        if (w.getVoxel(x, y, z) == MAT_FARMLAND) return y;
    return INT_MIN;
}

// How far a bed may sit above/below a new plot's own average ground before we
// refuse to extend the terrace onto it (start a new step instead).
constexpr int TERRACE_SPAN = 5;

// =============================================================================
// Plot plan shared by the preview + the till action. Scans every column's
// ground and rejects a cliff/wall or any tree/crop over the cell. The bed
// height is FIELD-AWARE: if an orthogonally-adjacent plot is already tilled,
// the new plot INHERITS that bed's level so a contiguous field stays perfectly
// co-planar (no plot floats above its neighbour). A plot with no tilled
// neighbour founds a fresh terrace at its own average ground height. Extending
// a field onto ground that has risen/fallen past TERRACE_SPAN is refused —
// that boundary is where a new terrace step belongs.
// =============================================================================
static bool cellPlan(const World& w, glm::ivec2 o, int yHint, int* targetY) {
    long long sum = 0;
    int n = 0, lo = INT_MAX, hi = INT_MIN;
    for (int dz = 0; dz < FARM_CELL; ++dz)
        for (int dx = 0; dx < FARM_CELL; ++dx) {
            int x = o.x + dx, z = o.y + dz;
            if (!w.voxelInWindow(x, z)) return false;
            int t = columnTop(w, x, z, yHint);
            if (t == BLOCKED || t == NO_GROUND) return false;   // tree/crop/no floor
            sum += t; ++n;
            lo = std::min(lo, t);
            hi = std::max(hi, t);
        }
    if (n == 0) return false;
    if (hi - lo > FARM_CELL) return false;   // effectively a cliff/wall
    int localAvg = int((sum + n / 2) / n);   // rounded average ground

    // Field continuity: adopt the average bed height of adjacent tilled plots.
    const glm::ivec2 nb[4] = {
        { o.x + FARM_CELL, o.y }, { o.x - FARM_CELL, o.y },
        { o.x, o.y + FARM_CELL }, { o.x, o.y - FARM_CELL }
    };
    long long nbSum = 0; int nbN = 0;
    for (const glm::ivec2& c : nb) {
        int cx = c.x + FARM_CELL / 2, cz = c.y + FARM_CELL / 2;
        if (!w.voxelInWindow(cx, cz)) continue;
        int fy = farmlandY(w, cx, cz, yHint);
        if (fy != INT_MIN) { nbSum += fy; ++nbN; }
    }
    int target = (nbN > 0) ? int((nbSum + nbN / 2) / nbN) : localAvg;

    // Don't drag a terrace too far from the ground the new plot actually sits
    // on — beyond that, the land wants a fresh step.
    if (std::abs(target - localAvg) > TERRACE_SPAN) return false;
    if (targetY) *targetY = target;
    return true;
}

bool farmTill(World& w, FarmState&, glm::ivec2 o, int y, EditDelta& d) {
    int target;
    if (!cellPlan(w, o, y, &target)) return false;
    bool any = false;
    for (int dz = 0; dz < FARM_CELL; ++dz)
        for (int dx = 0; dx < FARM_CELL; ++dx) {
            int x = o.x + dx, z = o.y + dz;
            if (!w.voxelInWindow(x, z)) continue;
            int t = columnTop(w, x, z, y);
            if (t == BLOCKED || t == NO_GROUND) continue;
            // Tilling terraforms the ground, so it PERSISTS (setVoxelPersist):
            // the bed survives a reload, and crops re-stamp on top of it.
            // 1. Clear everything (terrain shave + floating flora) above the
            //    bed level — no break on air gaps, so nothing is left hovering.
            for (int yy = target + 1; yy <= t + 16; ++yy)
                if (w.getVoxel(x, yy, z) != 0) { w.setVoxelPersist(x, yy, z, 0, d); any = true; }
            // 2. Fill low columns up to the bed level with soil.
            for (int yy = t + 1; yy <= target; ++yy) { w.setVoxelPersist(x, yy, z, MAT_DIRT, d); any = true; }
            // 3. Cap the bed with farmland.
            if (w.getVoxel(x, target, z) != MAT_FARMLAND) {
                w.setVoxelPersist(x, target, z, MAT_FARMLAND, d);
                any = true;
            }
        }
    return any;
}

static int cropIndexAt(const FarmState& fs, glm::ivec2 o) {
    for (int i = 0; i < int(fs.crops.size()); ++i)
        if (farmCellOrigin(fs.crops[i].root.x, fs.crops[i].root.z) == o)
            return i;
    return -1;
}

// Spacing -> vigor. Count the FREE plots among the 8 compass neighbours: tilled
// farmland, on the same terrace level, and not already planted — the room this
// plant has to sprawl. Optionally returns the free-direction bitmask (DIRS
// order) for the vine lean.
static int cropFreeNeighbours(const World& w, const FarmState& fs,
                              const Crop& c, uint8_t* dirsOut) {
    glm::ivec2 cell = farmCellOrigin(c.root.x, c.root.z);
    int count = 0; uint8_t mask = 0;
    for (int i = 0; i < 8; ++i) {
        glm::ivec2 nc = cell + DIRS[i] * FARM_CELL;
        int cx = nc.x + FARM_CELL / 2, cz = nc.y + FARM_CELL / 2;
        if (!w.voxelInWindow(cx, cz)) continue;
        int fy = farmlandY(w, cx, cz, c.root.y);
        if (fy == INT_MIN) continue;                          // not tilled
        if (std::abs(fy - c.root.y) > TERRACE_SPAN) continue;  // other terrace
        if (cropIndexAt(fs, nc) >= 0) continue;               // occupied
        ++count; mask |= uint8_t(1u << i);
    }
    if (dirsOut) *dirsOut = mask;
    return count;
}

// Ripe size tier from a stable genome roll BIASED by free room. Packed field
// (0 free) => baseline variety (mostly medium, occasional small/giant); each
// free neighbour nudges the odds up, so an open plot reliably grows a giant.
// Kept with RNG in the mix so there's still a lucky-giant thrill.
//   0 small, 1 medium, 2 giant. (A rare multi-plot "colossal" is parked as a
//   future depth feature — see design notes.)
static int cropSize(uint32_t genome, int freeN) {
    float roll  = float(genome & 0xFFFFu) / 65535.0f;   // stable 0..1 per plant
    float score = roll + 0.12f * float(freeN);          // +0 .. +0.96
    if (score >= 0.92f) return 2;   // giant
    if (score >= 0.15f) return 1;   // medium
    return 0;                       // small
}

bool farmPlant(World& w, FarmState& fs, glm::ivec2 o, int y, EditDelta& d) {
    int cx = o.x + FARM_CELL / 2, cz = o.y + FARM_CELL / 2;
    if (!w.voxelInWindow(cx, cz)) return false;
    int t = columnTop(w, cx, cz, y);
    if (t == BLOCKED || t == NO_GROUND) return false;
    if (w.getVoxel(cx, t, cz) != MAT_FARMLAND) return false;   // must be tilled
    if (cropIndexAt(fs, o) >= 0) return false;                 // already planted

    Crop c;
    c.root    = glm::ivec3(cx, t, cz);
    c.species = uint8_t(std::clamp(fs.seedSel, 0, CROP_SPECIES_N - 1));
    c.genome  = cellGenome(cx, cz);
    c.age     = 0.0f;
    c.stage   = 0;
    // Freeze the free-direction mask for the vine lean; seed the size tier from
    // current room (re-evaluated each stage as neighbours come and go).
    int freeN = cropFreeNeighbours(w, fs, c, &c.freeDirs);
    c.size = cropSize(c.genome, freeN);
    stampCrop(w, c, d);
    fs.crops.push_back(std::move(c));
    return true;
}

bool farmHasRipeCropAt(const FarmState& fs, glm::ivec2 o) {
    int idx = cropIndexAt(fs, o);
    return idx >= 0 && fs.crops[idx].stage == FARM_STAGE_MAX;
}

int farmCropSpeciesAt(const FarmState& fs, glm::ivec2 o) {
    int idx = cropIndexAt(fs, o);
    return idx >= 0 ? int(fs.crops[idx].species) : -1;
}

bool farmRipeCropBounds(const FarmState& fs, glm::ivec2 o,
                        glm::ivec3& lo, glm::ivec3& hi) {
    int idx = cropIndexAt(fs, o);
    if (idx < 0 || fs.crops[idx].stage != FARM_STAGE_MAX) return false;
    const Crop& c = fs.crops[idx];
    lo = hi = c.root;
    for (const glm::ivec3& v : c.occupied) {
        lo = glm::min(lo, v);
        hi = glm::max(hi, v);
    }
    return true;
}

bool farmHarvest(World& w, FarmState& fs, glm::ivec2 o,
                 Crop& out, EditDelta& d) {
    int idx = cropIndexAt(fs, o);
    if (idx < 0) return false;
    Crop& c = fs.crops[idx];
    if (c.stage != FARM_STAGE_MAX) return false;   // only RIPE crops are pluckable
    out = c;                                        // copy BEFORE the erase
    for (const glm::ivec3& v : c.occupied)
        if (w.voxelInWindow(v.x, v.z))
            w.setVoxelSim(v.x, v.y, v.z, 0, d);
    fs.crops.erase(fs.crops.begin() + idx);
    return true;
}

bool farmToolValid(const World& w, const FarmState& fs, int tool,
                   glm::ivec2 o, int y, int* outTargetY) {
    if (tool == FT_TILL)
        return cellPlan(w, o, y, outTargetY);
    if (tool == FT_PLANT) {
        int cx = o.x + FARM_CELL / 2, cz = o.y + FARM_CELL / 2;
        if (!w.voxelInWindow(cx, cz)) return false;
        int t = columnTop(w, cx, cz, y);
        if (t == BLOCKED || t == NO_GROUND) return false;
        return w.getVoxel(cx, t, cz) == MAT_FARMLAND && cropIndexAt(fs, o) < 0;
    }
    if (tool == FT_HARVEST)
        return cropIndexAt(fs, o) >= 0;
    return false;
}

void farmTick(World& w, FarmState& fs, float dt, float stageSecs, EditDelta& d) {
    stageSecs = std::max(stageSecs, 0.001f);   // guard /croptime 0 (div-by-zero)
    for (Crop& c : fs.crops) {
        if (!w.voxelInWindow(c.root.x, c.root.z)) continue;   // scrolled away
        c.age += dt;
        // Species pacing: a radish wins the sprint, corn takes the season.
        float ss = stageSecs * CROP_DEFS[c.species].growScale;
        int ns = std::min(FARM_STAGE_MAX, int(c.age / ss));
        // "Absent" = not stamped in THIS window: no footprint yet (fresh sidecar
        // load) or our first OWNED voxel is gone (the plot streamed out and back,
        // or was evicted). Test an actually-placed voxel, NEVER a fixed offset —
        // cropVoxels doesn't always fill root+1 (a small fruit's centre nudges
        // diagonally), which made those crops re-stamp every single frame.
        bool absent = c.occupied.empty() ||
                      !isCropMaterial(w.getVoxel(c.occupied[0].x, c.occupied[0].y,
                                                 c.occupied[0].z));
        if (ns != c.stage || absent) {
            if (absent) {
                // The plot regenerated (streamed back in / freshly loaded): the
                // TILLED BED is a world-terrain edit that doesn't survive chunk
                // regen the way FarmState crops do, so it reverts to grass. Re-
                // till it (mows the regrown flora, re-caps farmland) BEFORE the
                // crop restamps on top — otherwise the pumpkin floats on grass.
                farmTill(w, fs, farmCellOrigin(c.root.x, c.root.z), c.root.y, d);
            }
            if (ns != c.stage) {
                c.stage = ns;
                // Re-evaluate size against CURRENT room: a neighbour planted
                // since this crop sprouted stunts it; a harvested neighbour lets
                // it swell. Also where an offline-grown crop (loaded at its saved
                // stage) jumps to its new stage and its size recomputes — so
                // offline growth yields the grade online growth would.
                c.size = cropSize(c.genome, cropFreeNeighbours(w, fs, c, nullptr));
            }
            stampCrop(w, c, d);
        }
    }
}

// =============================================================================
// Persistence: a seed-keyed sidecar (farm.dat), same tmp-write+rename contract
// as saveWorldEdits / MapExplored::save. occupied is NOT stored (rebuilt by
// stampCrop on the next resident tick); savedUnix drives offline growth.
// =============================================================================
namespace {
struct FarmHeader {
    char     magic[4];      // "TVFM"
    uint32_t version;
    uint64_t worldSeed;
    int64_t  savedUnix;     // wall-clock at save (offline-growth anchor)
    uint32_t cropCount;
};
struct CropRec {
    int32_t  rootX, rootY, rootZ;
    uint32_t genome;
    float    age;
    int32_t  stage;
    int32_t  size;
    uint32_t freeDirs;
    uint32_t species;    // CropSpecies (v2+; a v1 file is all pumpkins)
};
// v1 record (pre-species) — layout kept for the legacy loader below.
struct CropRecV1 {
    int32_t  rootX, rootY, rootZ;
    uint32_t genome;
    float    age;
    int32_t  stage;
    int32_t  size;
    uint32_t freeDirs;
};
constexpr uint32_t FARM_SAVE_VERSION = 2;
}  // namespace

bool saveFarmState(const FarmState& fs, const std::string& path,
                   uint64_t worldSeed, int64_t nowUnix) {
    const std::string tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return false;
    FarmHeader h{};
    std::memcpy(h.magic, "TVFM", 4);
    h.version   = FARM_SAVE_VERSION;
    h.worldSeed = worldSeed;
    h.savedUnix = nowUnix;
    h.cropCount = uint32_t(fs.crops.size());
    bool ok = std::fwrite(&h, sizeof(h), 1, f) == 1;
    for (const Crop& c : fs.crops) {
        if (!ok) break;
        CropRec r{ c.root.x, c.root.y, c.root.z, c.genome, c.age,
                   c.stage, c.size, c.freeDirs, c.species };
        ok = std::fwrite(&r, sizeof(r), 1, f) == 1;
    }
    ok = (std::fclose(f) == 0) && ok;   // fclose flushes: a full disk fails here
    if (!ok) { std::remove(tmp.c_str()); return false; }
    std::remove(path.c_str());
    return std::rename(tmp.c_str(), path.c_str()) == 0;
}

bool loadFarmState(FarmState& fs, const std::string& path,
                   uint64_t worldSeed, int64_t nowUnix) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        // Recover an orphaned tmp: a save that removed farm.dat but died before
        // the rename left the fresh data only in farm.dat.tmp.
        f = std::fopen((path + ".tmp").c_str(), "rb");
        if (!f) return false;
    }
    FarmHeader h{};
    if (std::fread(&h, sizeof(h), 1, f) != 1 ||
        std::memcmp(h.magic, "TVFM", 4) != 0 ||
        (h.version != 1 && h.version != FARM_SAVE_VERSION) ||
        h.worldSeed != worldSeed) {
        std::fclose(f);
        return false;
    }
    // Offline growth: fast-forward every crop by the real seconds since the save
    // (clamped against a backwards clock).
    double elapsed = double(nowUnix) - double(h.savedUnix);
    if (elapsed < 0.0) elapsed = 0.0;
    fs.crops.clear();
    uint32_t got = 0;
    for (uint32_t i = 0; i < h.cropCount; ++i) {
        CropRec r{};
        if (h.version == 1) {
            // v1 (pre-species): same layout minus the trailing species field —
            // every legacy crop is a pumpkin.
            CropRecV1 r1{};
            if (std::fread(&r1, sizeof(r1), 1, f) != 1) break;
            r = CropRec{ r1.rootX, r1.rootY, r1.rootZ, r1.genome, r1.age,
                         r1.stage, r1.size, r1.freeDirs, CROP_PUMPKIN };
        } else {
            if (std::fread(&r, sizeof(r), 1, f) != 1) break;
        }
        ++got;
        Crop c;
        c.root     = glm::ivec3(r.rootX, r.rootY, r.rootZ);
        c.species  = uint8_t(std::min(r.species, uint32_t(CROP_SPECIES_N - 1)));
        c.genome   = r.genome;
        c.age      = r.age + float(elapsed);
        // Keep the SAVED stage: farmTick advances it on the first resident tick
        // and recomputes size against CURRENT neighbours there, so an offline-
        // grown crop harvests at the grade online growth would have produced
        // (recomputing stage here would freeze the stale plant-time size tier).
        c.stage    = std::min(FARM_STAGE_MAX, std::max(0, r.stage));
        c.size     = r.size;
        c.freeDirs = uint8_t(r.freeDirs);
        // occupied left empty — farmTick re-stamps when the plot is resident.
        fs.crops.push_back(std::move(c));
    }
    std::fclose(f);
    if (got != h.cropCount)
        std::fprintf(stderr, "[farm] WARNING: %s truncated — %u of %u crops\n",
                     path.c_str(), got, h.cropCount);
    std::printf("[farm] loaded %u crops, +%.0fs offline growth\n", got, elapsed);
    return true;
}
