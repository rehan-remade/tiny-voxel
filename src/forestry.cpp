#include "forestry.h"
#include "world.h"
#include "common.h"
#include "materials.h"
#include "particles.h"
#include "profiling.h"
#include "player.h"      // isBlockingForPlayer
#include "worldgen/worldgen.h"   // rederiveTreeAt (exact tree identity)
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <deque>
#include <functional>
#include <unordered_map>
#include <unordered_set>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Wood-like: what the tree flood fill walks. Char is included so burned
// skeletons stay clearable now that dig/place is gone (they classify as
// stumps and burst to dust).
static bool isWoodMat(uint8_t m) {
    return m == MAT_WOOD || m == MAT_DESERT_WOOD || m == MAT_CHAR;
}
// The sawn stump face (either ring band) — counts as rooted support for the
// leaf-decay invariant, never as fellable wood.
static bool isCutFace(uint8_t m) {
    return m == MAT_CUT_WOOD || m == MAT_CUT_RING;
}
static bool isLeafMat(uint8_t m) { return m == MAT_LEAVES; }
// Cacti fell as their own species: flesh trunk, blooms for foliage.
static bool isCactusSolid(uint8_t m) { return m == MAT_CACTUS_FLESH; }
// Foliage = leaves + cactus blooms + sage tufts. Blooms grow on BOTH cacti
// and ocotillo (the shared-material edge case) — a clicked bloom belongs to
// whatever solid it walks to, and a fell always takes its own foliage along.
// Sage is creosote/desert-tree tip foliage: without it the fell left tufts
// floating and the hover outline hugged bare stems.
static bool isFoliage(uint8_t m) {
    return m == MAT_LEAVES || m == MAT_CACTUS_BLOOM || m == MAT_SHRUB_SAGE;
}

// Flood-fill visited key. 21 bits per axis (two's-complement masked) — only
// collides for voxels ~2M apart, far past the fill's 192-voxel extent cap.
static int64_t voxKey(glm::ivec3 v) {
    return (int64_t(v.x) & 0x1FFFFF) |
           ((int64_t(v.y) & 0x1FFFFF) << 21) |
           ((int64_t(v.z) & 0x1FFFFF) << 42);
}

static const glm::ivec3 N6[6] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};

// The creature palette decodes sRGB -> linear (crePalette), and terrain
// albedo is the linear matProps colour — so encode linear -> sRGB here and
// the felled tree keeps the standing tree's brightness.
static uint32_t packSrgb(glm::vec3 lin) {
    auto enc = [](float c) {
        c = glm::clamp(c, 0.0f, 1.0f);
        float s = (c <= 0.0031308f) ? c * 12.92f
                                    : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
        return uint32_t(std::round(s * 255.0f));
    };
    return enc(lin.r) | (enc(lin.g) << 8) | (enc(lin.b) << 16) | (255u << 24);
}

static uint32_t posHash(glm::ivec3 v) {
    uint32_t h = uint32_t(v.x) * 0x8DA6B343u ^ uint32_t(v.y) * 0xD8163841u ^
                 uint32_t(v.z) * 0xCB1AB31Fu;
    h ^= h >> 13; h *= 0x9E3779B1u; h ^= h >> 16;
    return h;
}

// Solid for item bounces / fall impact (leaves handled by the callers).
static bool solidAt(const World& w, glm::vec3 p) {
    uint8_t v = w.getVoxel(int(std::floor(p.x)), int(std::floor(p.y)),
                           int(std::floor(p.z)));
    return isBlockingForPlayer(v);
}

// ---------------------------------------------------------------------------
// Static models: wood/leaf chunk (debris items) + the equipped axe
// ---------------------------------------------------------------------------

// One-part model builder: `cells` maps linear index (x-fastest, the
// creature_trace layout) to a 1-based palette index (0 = empty).
static void buildOnePart(CreatureModel& m, const char* name, glm::ivec3 dims,
                         float cellScale, const std::vector<uint8_t>& cells) {
    m = CreatureModel{};
    m.cellScale = cellScale;
    CreaturePart p;
    std::snprintf(p.name, sizeof(p.name), "%s", name);
    p.parent = -1;
    p.gridOrigin = glm::vec3(0);
    p.pivot = glm::vec3(0);
    p.dims = dims;
    p.wordBase = 0;
    m.parts.push_back(p);
    size_t n = size_t(dims.x) * dims.y * dims.z;
    m.vox.assign((n + 3) / 4, 0u);
    for (size_t i = 0; i < n && i < cells.size(); ++i)
        if (cells[i])
            m.vox[i >> 2] |= uint32_t(cells[i]) << ((i & 3) * 8);
}

void forestryBuildStaticModels(Forestry& f) {
    // Chunk lump shared by wood + leaves: 3^3 with the corners knocked off.
    auto lumpCells = [](int salt) {
        std::vector<uint8_t> c(27, 0);
        for (int z = 0; z < 3; ++z)
            for (int y = 0; y < 3; ++y)
                for (int x = 0; x < 3; ++x) {
                    if ((x != 1) + (y != 1) + (z != 1) == 3) continue;
                    c[x + 3 * (y + 3 * z)] =
                        uint8_t(1 + (posHash({x + salt, y, z}) & 3));
                }
        return c;
    };
    {
        glm::vec3 base = matProps(MAT_WOOD).color;
        buildOnePart(f.woodChunk, "chunk", {3, 3, 3}, 0.6f, lumpCells(7));
        f.woodChunk.palette[0] = packSrgb(base * 0.78f);
        f.woodChunk.palette[1] = packSrgb(base * 0.95f);
        f.woodChunk.palette[2] = packSrgb(base * 1.12f);
        f.woodChunk.palette[3] = packSrgb(base * 1.30f);
        glm::vec3 cac = matProps(MAT_CACTUS_FLESH).color;
        buildOnePart(f.cactusChunk, "chunk", {3, 3, 3}, 0.6f, lumpCells(11));
        f.cactusChunk.palette[0] = packSrgb(cac * 0.72f);
        f.cactusChunk.palette[1] = packSrgb(cac * 0.92f);
        f.cactusChunk.palette[2] = packSrgb(cac * 1.10f);
        f.cactusChunk.palette[3] = packSrgb(cac * 1.28f);
    }
    {
        glm::vec3 base = matProps(MAT_LEAVES).color;
        buildOnePart(f.leafChunk, "chunk", {3, 3, 3}, 0.6f, lumpCells(3));
        f.leafChunk.palette[0] = packSrgb(base * 0.70f);
        f.leafChunk.palette[1] = packSrgb(base * 0.90f);
        f.leafChunk.palette[2] = packSrgb(base * 1.08f);
        f.leafChunk.palette[3] = packSrgb(base * 1.28f);
    }
    {
        glm::vec3 base = matProps(MAT_STONE).color;
        buildOnePart(f.stoneChunk, "chunk", {3, 3, 3}, 0.6f, lumpCells(17));
        f.stoneChunk.palette[0] = packSrgb(base * 0.72f);
        f.stoneChunk.palette[1] = packSrgb(base * 0.90f);
        f.stoneChunk.palette[2] = packSrgb(base * 1.05f);
        f.stoneChunk.palette[3] = packSrgb(base * 1.22f);
    }
    // Axe: handle up the +y column, head at the top swinging out +x.
    // Axe cells render at the PLAYER's cell scale (the instance inherits the
    // hand transform), so 8 handle cells ~ 45 cm — a hatchet.
    {
        glm::ivec3 d(5, 8, 1);
        std::vector<uint8_t> c(size_t(5) * 8, 0);
        auto set = [&](int x, int y, uint8_t v) { c[x + 5 * y] = v; };
        for (int y = 0; y < 8; ++y) set(1, y, 1);          // handle (brown)
        for (int y = 5; y < 8; ++y) {                      // head (steel)
            set(2, y, 2);
            set(3, y, 2);
        }
        set(4, 5, 3); set(4, 6, 3); set(4, 7, 3);          // cutting edge (bright)
        set(0, 6, 2);                                      // small poll (back)
        buildOnePart(f.axeModel, "axe", d, 0.5625f, c);
        f.axeModel.palette[0] = packSrgb({0.34f, 0.21f, 0.10f});
        f.axeModel.palette[1] = packSrgb({0.46f, 0.49f, 0.54f});
        f.axeModel.palette[2] = packSrgb({0.80f, 0.83f, 0.88f});
    }
    // Pickaxe: same handle, a cross-bar head at the top with drooping tips —
    // the two-point silhouette reads "pick" even at a handful of pixels.
    {
        glm::ivec3 d(7, 8, 1);
        std::vector<uint8_t> c(size_t(7) * 8, 0);
        auto set = [&](int x, int y, uint8_t v) { c[x + 7 * y] = v; };
        for (int y = 0; y < 8; ++y) set(3, y, 1);          // handle (brown)
        for (int x = 1; x < 6; ++x) set(x, 7, 2);          // head bar (iron)
        set(1, 6, 2); set(5, 6, 2);                        // shoulders
        set(0, 5, 3); set(6, 5, 3);                        // drooping tips (bright)
        buildOnePart(f.pickModel, "pick", d, 0.5625f, c);
        f.pickModel.palette[0] = packSrgb({0.34f, 0.21f, 0.10f});
        f.pickModel.palette[1] = packSrgb({0.40f, 0.42f, 0.47f});
        f.pickModel.palette[2] = packSrgb({0.78f, 0.81f, 0.87f});
    }
}

// ---------------------------------------------------------------------------
// Tree identification (flood fill)
// ---------------------------------------------------------------------------

static constexpr int   WOOD_CAP     = 40000;   // safety: never fell megastructures
static constexpr int   LEAF_CAP     = 80000;
static constexpr int   EXTENT_CAP   = 192;     // worldgen trees stay well inside
static constexpr float LEAF_XZ_CAP  = 85.0f;   // flora.cpp footprint contract ~80
static constexpr float WOOD_XZ_CAP  = 60.0f;   // own branch reach tops out ~50
static constexpr int   STUMP_MAX_H  = 8;       // wood this flat is a stump (trunks >= ~22)
static constexpr int   STUMP_H      = 7;       // trunk voxels the fell leaves (~70 cm)

// The resident window: edits outside it are silently dropped and reads come
// back air, so a tree touching the border would half-collect / half-fell.
static bool insideWindow(const World& w, glm::ivec3 lo, glm::ivec3 hi) {
    int ox = w.originVoxX(), oz = w.originVoxZ();
    return lo.x - 2 >= ox && hi.x + 2 < ox + GRID_XZ &&
           lo.z - 2 >= oz && hi.z + 2 < oz + GRID_XZ &&
           lo.y - 2 >= 0  && hi.y + 2 < GRID_Y;
}

// Flood-fill identification (the FALLBACK path): grounded-cluster wood
// partition + leaf-ownership BFS. Heuristic by nature — kept for wood the
// generator can't answer for (palms/desert flora, pure-grammar gate
// divergence, old-save drift).
static bool collectTreeFlood(const World& w, glm::ivec3 seed, TreeInfo& out) {
    VOX_ZONE_N("forestry: collect flood");
    out = TreeInfo{};
    uint8_t sm = w.getVoxel(seed.x, seed.y, seed.z);

    // Clicked leaves: walk the leaf mass (26-conn — canopy bulbs are often
    // only diagonally linked, see fire.cpp) to the nearest wood, then treat
    // the click as a click on that tree. A woodless blob isn't a tree.
    if (isFoliage(sm)) {
        std::unordered_set<int64_t> lseen{voxKey(seed)};
        std::deque<glm::ivec3> lq{seed};
        glm::ivec3 found(INT32_MIN);
        int walked = 0;
        while (!lq.empty() && found.x == INT32_MIN && ++walked < 6000) {
            glm::ivec3 v = lq.front(); lq.pop_front();
            for (int dz = -1; dz <= 1 && found.x == INT32_MIN; ++dz)
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (!dx && !dy && !dz) continue;
                        glm::ivec3 n = v + glm::ivec3(dx, dy, dz);
                        uint8_t m = w.getVoxel(n.x, n.y, n.z);
                        if (isWoodMat(m) || isCactusSolid(m)) { found = n; break; }
                        if (isFoliage(m) && lseen.insert(voxKey(n)).second)
                            lq.push_back(n);
                    }
        }
        if (found.x == INT32_MIN) return false;
        seed = found;
        sm = w.getVoxel(seed.x, seed.y, seed.z);
    }
    if (!isWoodMat(sm) && !isCactusSolid(sm)) return false;
    const bool cactus = isCactusSolid(sm);
    // Per-cluster solid test: a cactus flood never crosses into wood and
    // vice versa (they only share BLOOMS, which are foliage).
    auto solidFor = [&](uint8_t m) {
        return cactus ? isCactusSolid(m) : isWoodMat(m);
    };

    // Wood: 26-connectivity (tapered L-system capsules touch diagonally),
    // tracking BFS depth and "grounded" voxels (wood sitting on solid
    // non-wood — a rooted trunk base). The flood is bounded to the clicked
    // column's NEIGHBOURHOOD: in dense forest the branch-touching wood
    // network can span the whole biome, and flooding (or bailing on) all of
    // it would make every tree in the patch unchoppable. The per-tree
    // partition below only needs the local grounded bases anyway.
    constexpr float WOOD_SEED_RADIUS = 70.0f;
    struct Node { glm::ivec3 v; int d; };
    std::unordered_set<int64_t> seen{voxKey(seed)};
    std::deque<Node> q{{seed, 0}};
    std::vector<glm::ivec3> comp;
    std::vector<Node> grounded;
    glm::ivec3 lo = seed, hi = seed;
    while (!q.empty()) {
        Node nd = q.front(); q.pop_front();
        comp.push_back(nd.v);
        lo = glm::min(lo, nd.v); hi = glm::max(hi, nd.v);
        if (int(comp.size()) > WOOD_CAP) return false;
        if (glm::any(glm::greaterThan(hi - lo, glm::ivec3(EXTENT_CAP))))
            return false;
        uint8_t below = w.getVoxel(nd.v.x, nd.v.y - 1, nd.v.z);
        if (below != MAT_AIR && !solidFor(below) && !isFoliage(below) &&
            isBlockingForPlayer(below))
            grounded.push_back(nd);
        for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    if (!dx && !dy && !dz) continue;
                    glm::ivec3 n = nd.v + glm::ivec3(dx, dy, dz);
                    if (glm::distance(glm::vec2(n.x, n.z),
                                      glm::vec2(seed.x, seed.z)) >
                        WOOD_SEED_RADIUS) continue;
                    if (!solidFor(w.getVoxel(n.x, n.y, n.z))) continue;
                    if (seen.insert(voxKey(n)).second) q.push_back({n, nd.d + 1});
                }
    }

    // Which trunk is OURS? The grounded voxel closest (in graph steps) to the
    // click follows the clicked branch down to its own base. Cluster the
    // grounded set by XZ (per-tree root discs are <= ~13 wide, trees >= ~11
    // apart) — more than one cluster means this component MERGED with a
    // neighbour through touching branch wood, and the fell must not take the
    // neighbour with it: wood is then kept by nearest-base partition.
    glm::ivec3 rootV = comp[0];
    int rootIdx = -1;
    for (auto& c : comp) if (c.y < rootV.y) rootV = c;   // fallback: lowest
    if (!grounded.empty()) {
        int best = 0;
        for (int i = 1; i < int(grounded.size()); ++i)
            if (grounded[i].d < grounded[best].d) best = i;
        rootV = grounded[best].v;
        rootIdx = best;
    }
    // Root clusters by GROUND CONNECTIVITY (union nearby grounded voxels),
    // not by a fixed radius: one trunk's root disc is contiguous, but two
    // separate plants' base welds never touch. The old 9-voxel radius was
    // tuned for forest giants and MERGED neighbouring desert plants (their
    // jittered lattices put bases 4-8 apart) into one fellable "tree".
    std::vector<int> parent(grounded.size());
    for (int i = 0; i < int(parent.size()); ++i) parent[i] = i;
    std::function<int(int)> findRoot = [&](int i) {
        while (parent[i] != i) { parent[i] = parent[parent[i]]; i = parent[i]; }
        return i;
    };
    for (int i = 0; i < int(grounded.size()); ++i)
        for (int j = i + 1; j < int(grounded.size()); ++j) {
            glm::ivec3 a = grounded[i].v, b = grounded[j].v;
            if (std::abs(a.x - b.x) <= 3 && std::abs(a.z - b.z) <= 3 &&
                std::abs(a.y - b.y) <= 3)
                parent[findRoot(i)] = findRoot(j);
        }
    struct Cluster { glm::vec2 c; int n; };
    std::vector<Cluster> clusters;
    std::unordered_map<int, int> clusterIdx;   // union root -> cluster index
    std::vector<int> clusterOfG(grounded.size(), -1);
    for (int i = 0; i < int(grounded.size()); ++i) {
        int r = findRoot(i);
        auto it = clusterIdx.find(r);
        int ci;
        if (it == clusterIdx.end()) {
            ci = int(clusters.size());
            clusterIdx.emplace(r, ci);
            clusters.push_back({glm::vec2(0.0f), 0});
        } else {
            ci = it->second;
        }
        Cluster& cl = clusters[ci];
        cl.c = (cl.c * float(cl.n) + glm::vec2(grounded[i].v.x, grounded[i].v.z))
             / float(cl.n + 1);
        cl.n += 1;
        clusterOfG[i] = ci;
    }
    int ours = (rootIdx >= 0) ? clusterOfG[rootIdx]
             : clusters.empty() ? -1 : 0;
    glm::vec2 ourBase = (ours >= 0) ? clusters[ours].c
                                    : glm::vec2(rootV.x, rootV.z);
    bool multi = clusters.size() > 1;

    out.wood.reserve(comp.size());
    for (auto& v : comp) {
        glm::vec2 p(v.x, v.z);
        if (glm::distance(p, ourBase) > WOOD_XZ_CAP) continue;
        if (multi) {
            float dOurs = glm::distance(p, ourBase);
            bool nearer = true;
            for (int i = 0; i < int(clusters.size()) && nearer; ++i)
                if (i != ours && glm::distance(p, clusters[i].c) < dOurs)
                    nearer = false;
            if (!nearer) continue;
        }
        out.wood.push_back(v);
    }
    if (out.wood.empty()) return false;

    // Base / cut plane / classification from the KEPT set.
    lo = out.wood[0]; hi = out.wood[0];
    for (auto& v : out.wood) { lo = glm::min(lo, v); hi = glm::max(hi, v); }
    int minY = lo.y;
    glm::dvec3 acc(0); int nAcc = 0;
    for (auto& v : out.wood)
        if (v.y <= minY + 1) { acc += glm::dvec3(v.x, v.y, v.z); ++nAcc; }
    out.base = glm::ivec3(int(std::round(acc.x / nAcc)), minY,
                          int(std::round(acc.z / nAcc)));
    out.isCactus = cactus;
    // Stump SCALES with the tree: quarter height capped at STUMP_H. A fixed
    // 7-voxel stump left most of a knee-high Joshua tree standing after its
    // "fell". Cacti keep only a low nub.
    int woodH = hi.y - minY + 1;
    out.cutY = minY + (cactus ? 2 : std::clamp(woodH / 4, 2, STUMP_H));
    out.trunkR = 2.0f;
    for (auto& v : out.wood)
        if (v.y <= minY + 2)
            out.trunkR = std::max(out.trunkR,
                glm::distance(glm::vec2(v.x, v.z),
                              glm::vec2(out.base.x, out.base.z)));
    out.trunkR = std::min(out.trunkR, 9.0f);
    // Small trees keep only a TIGHT stump disc: desert trees fork at knee
    // height, and a trunkR-wide keep radius (inflated by the ground-level
    // fork) preserved the whole fork as "stump".
    out.stumpR = cactus ? 1.6f
               : (woodH <= 16 ? 2.0f : out.trunkR + 1.5f);

    out.woodMats.reserve(out.wood.size());
    for (auto& v : out.wood) {
        uint8_t m = w.getVoxel(v.x, v.y, v.z);
        out.woodMats.push_back(m);
        if (m == MAT_CHAR) ++out.charN;
    }
    // Stump-shaped (flat — real trunks run >= ~22 tall), tiny, or mostly
    // charred: uproot/clear semantics rather than a fell.
    out.isStump = (hi.y - minY + 1) <= STUMP_MAX_H ||
                  int(out.wood.size()) < 40 ||
                  out.charN * 2 > int(out.wood.size());

    // Leaves: neighbouring crowns interpenetrate freely, so raw connectivity
    // would hand one click the canopy of half a forest. Ownership is settled
    // by a multi-source BFS over the connected leaf mass: every leaf touching
    // wood seeds the flood labelled by WHOSE wood it touches (ours vs anyone
    // else's — dropped merge-clusters included), labels spread outward by
    // graph distance (ties and the truncation frontier go foreign, so a
    // neighbour is never stripped), and only OUR leaves fell. Leaves whose
    // region never meets wood at all — a floating blob tangled into our crown
    // — count as ours: leaving leaves hanging is the one unacceptable outcome.
    {
        std::unordered_set<int64_t> ourWood;
        for (auto& v : out.wood) ourWood.insert(voxKey(v));

        // Pass 1: the connected leaf mass around our wood, horizon-bounded
        // (a deep forest's canopy is one mass — never flood it all).
        constexpr int LEAF_HORIZON = 48;
        struct LNode { glm::ivec3 v; int d; };
        std::vector<glm::ivec3> mass;
        std::vector<int> depth;
        std::unordered_map<int64_t, int> idxOf;
        std::deque<LNode> mq;
        auto pushLeaf = [&](glm::ivec3 n, int dpt) {
            if (int(mass.size()) > LEAF_CAP) return;
            if (!isFoliage(w.getVoxel(n.x, n.y, n.z))) return;
            if (glm::distance(glm::vec2(n.x, n.z), ourBase) >
                LEAF_XZ_CAP + 40.0f) return;
            int64_t k = voxKey(n);
            if (idxOf.count(k)) return;
            idxOf.emplace(k, int(mass.size()));
            mass.push_back(n);
            depth.push_back(dpt);
            if (dpt < LEAF_HORIZON) mq.push_back({n, dpt});
        };
        for (auto& v : out.wood)
            for (int dz = -1; dz <= 1; ++dz)
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (!dx && !dy && !dz) continue;
                        pushLeaf(v + glm::ivec3(dx, dy, dz), 0);
                    }
        while (!mq.empty()) {
            LNode nd = mq.front(); mq.pop_front();
            for (int dz = -1; dz <= 1; ++dz)
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (!dx && !dy && !dz) continue;
                        pushLeaf(nd.v + glm::ivec3(dx, dy, dz), nd.d + 1);
                    }
        }

        // Pass 2: seed ownership labels from wood adjacency. Foreign seeds
        // enqueue FIRST so equidistant ties resolve foreign.
        std::vector<int8_t> label(mass.size(), -1);   // -1 ?, 0 ours, 1 foreign
        std::deque<int> oursQ, foreignQ;
        for (int i = 0; i < int(mass.size()); ++i) {
            bool nearOurs = false, nearForeign = false;
            for (int dz = -1; dz <= 1; ++dz)
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (!dx && !dy && !dz) continue;
                        glm::ivec3 n = mass[i] + glm::ivec3(dx, dy, dz);
                        uint8_t nm = w.getVoxel(n.x, n.y, n.z);
                        // Ownership crosses species: a bloom mass wedged
                        // between a cactus and an ocotillo is contested by
                        // BOTH solids, whoever it belongs to.
                        if (!isWoodMat(nm) && !isCactusSolid(nm)) continue;
                        if (ourWood.count(voxKey(n))) nearOurs = true;
                        else                          nearForeign = true;
                    }
            // Horizon-frontier leaves fence the truncation: whatever lies
            // beyond wasn't explored, so nothing inside may claim through it.
            if (nearForeign || depth[i] >= LEAF_HORIZON) {
                label[i] = 1;
                foreignQ.push_back(i);
            } else if (nearOurs) {
                label[i] = 0;
                oursQ.push_back(i);
            }
        }
        std::deque<int> bfs;
        for (int i : foreignQ) bfs.push_back(i);
        for (int i : oursQ)    bfs.push_back(i);
        while (!bfs.empty()) {
            int i = bfs.front(); bfs.pop_front();
            for (int dz = -1; dz <= 1; ++dz)
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (!dx && !dy && !dz) continue;
                        auto it = idxOf.find(voxKey(mass[i] + glm::ivec3(dx, dy, dz)));
                        if (it == idxOf.end() || label[it->second] >= 0) continue;
                        label[it->second] = label[i];
                        bfs.push_back(it->second);
                    }
        }
        // Only clearly-OURS leaves ride the falling model. Ambiguous and
        // never-met-wood blobs are the support-decay engine's problem now:
        // they crumble into auto-collected items moments after the fell
        // instead of being yanked off a neighbour (or left hanging).
        for (int i = 0; i < int(mass.size()); ++i)
            if (label[i] == 0) {
                out.leaves.push_back(mass[i]);
                lo = glm::min(lo, mass[i]);
                hi = glm::max(hi, mass[i]);
            }
    }
    out.lo = lo; out.hi = hi;
    out.valid = insideWindow(w, lo, hi);
    return out.valid;
}

// Exact identity, asked of the GENERATOR (rederiveTreeAt, worldgen/flora.cpp):
// forest trees derive bit-identically from (seed, lattice cell), so the
// replay returns the clicked tree's true emitted voxel set — no ownership
// guessing, no caps, no flood cost. Membership intersects with the CURRENT
// world: a voxel whose material no longer matches what the generator emitted
// (dug, burned to air, claimed by terrain in the replay's over-emit) simply
// drops out.
enum class ExactResult { NoTree, OutOfWindow, Ok };
static ExactResult collectTreeExact(const World& w, const GenContext& gen,
                                    glm::ivec3 seedVox, TreeInfo& out) {
    VOX_ZONE_N("forestry: collect exact");
    TreeRederive rd;
    if (!rederiveTreeAt(gen, seedVox, rd)) return ExactResult::NoTree;
    out = TreeInfo{};
    glm::ivec3 lo(INT32_MAX), hi(INT32_MIN);
    int minWoodY = INT32_MAX, maxWoodY = INT32_MIN;
    for (auto& vm : rd.voxels) {
        const glm::ivec3& v = vm.first;
        uint8_t cur = w.getVoxel(v.x, v.y, v.z);
        if (vm.second == MAT_WOOD) {
            if (!isWoodMat(cur)) continue;
            out.wood.push_back(v);
            out.woodMats.push_back(cur);
            if (cur == MAT_CHAR) ++out.charN;
            minWoodY = std::min(minWoodY, v.y);
            maxWoodY = std::max(maxWoodY, v.y);
        } else {
            if (!isLeafMat(cur)) continue;
            out.leaves.push_back(v);
        }
        lo = glm::min(lo, v);
        hi = glm::max(hi, v);
    }
    if (out.wood.empty()) return ExactResult::NoTree;
    out.base   = rd.base;
    out.trunkR = glm::clamp(rd.trunkR, 2.0f, 9.0f);
    // Same scaled-stump rule as the flood path, from the WOOD-ONLY bbox and
    // anchored at the surviving wood's base (hi.y spans the canopy too —
    // leaf-inclusive height let a burned-to-stub registry tree with an
    // intact crown dodge stump classification and take the wide stump disc).
    int woodH = maxWoodY - minWoodY + 1;
    out.cutY   = minWoodY + std::clamp(woodH / 4, 2, STUMP_H);
    out.stumpR = (woodH <= 16) ? 2.0f : out.trunkR + 1.5f;
    out.lo = lo; out.hi = hi;
    // Same classification as the flood path: flat / tiny / mostly-charred
    // remnants are stumps, not fellable trees.
    out.isStump = woodH <= STUMP_MAX_H ||
                  int(out.wood.size()) < 40 ||
                  out.charN * 2 > int(out.wood.size());
    out.valid = insideWindow(w, lo, hi);
    return out.valid ? ExactResult::Ok : ExactResult::OutOfWindow;
}

bool collectTree(const World& w, const GenContext& gen, glm::ivec3 seedVox,
                 TreeInfo& out) {
    auto t0 = std::chrono::steady_clock::now();
    auto ms = [&]() {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t0).count();
    };
    ExactResult ex = collectTreeExact(w, gen, seedVox, out);
    if (ex == ExactResult::Ok) {
        std::printf("[forestry] identity: generator (%zu wood, %zu leaves) in %.1f ms\n",
                    out.wood.size(), out.leaves.size(), ms());
        return true;
    }
    // A window-edge tree fails the SAME insideWindow test on the flood path
    // — don't pay the full flood a second time to learn nothing.
    if (ex == ExactResult::OutOfWindow) return false;
    if (collectTreeFlood(w, seedVox, out)) {
        std::printf("[forestry] identity: flood fallback (%zu wood, %zu leaves) in %.1f ms\n",
                    out.wood.size(), out.leaves.size(), ms());
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Boulder identification (generator replay first, bounded flood fallback)
// ---------------------------------------------------------------------------

// genBoulders (worldgen/flora.cpp) writes compact MAT_STONE ellipsoids onto
// grass — but a boulder's buried root reaches h - aY*(embed+1), which on
// thin-soil ground fuses seamlessly with the MAT_STONE strata. A naive flood
// would leak through the root into the hillside and refuse almost every
// mid-size rock, so membership is bounded at the SOIL LINE: a voxel belongs
// only if it sits above terrainHeightAt(x,z) - ROCK_ROOT_K for its own
// column (the generator's decoration-free ground truth). The deep root stays
// in the ground and becomes the quarry scar. Exposed-strata skins still
// qualify per-column, so a click on a stone hillside crawls the skin until
// the extent cap refuses it — that's the intended "you mine boulders, not
// mountainsides" gate (the hover latch caches the rejection).
static constexpr int ROCK_CAP        = 12000;  // FLOOD path: is-this-a-boulder test
static constexpr int ROCK_EXTENT_CAP = 44;     // FLOOD path: per-axis bbox extent
static constexpr int ROCK_MIN        = 10;     // pebbles aren't a session
// GEN path: exact identity needs no terrain caps — the flood's caps double as
// its "is this actually a boulder?" test, but a replayed rock IS a boulder by
// construction, which is what makes giant erratics and fused cluster members
// minable at all. This is pure insanity guarding: a single boulder tops out
// ~26k voxels, so anything bigger is a garbled replay the spot check missed.
static constexpr int ROCK_GEN_CAP    = 30000;
// Buried margin kept below the soil line. The surface grammar (internal.h
// fillColumn) puts dirt at (h-5, h-1) and the stone strata top at h-5, so
// K=4 admits dirt-zone voxels only — the flood can't reach strata by
// construction — while absorbing the few voxels of heightAt-vs-carved-ground
// noise that would otherwise orphan rocks sitting in local dips.
static constexpr int ROCK_ROOT_K     = 4;

// Shared result assembly: base = bottom-slab centroid (base.y == lo.y — the
// strike-height math in app.cpp leans on that), radius = max XZ reach.
static void rockFromComp(std::vector<glm::ivec3>&& comp, glm::ivec3 lo,
                         glm::ivec3 hi, RockInfo& out) {
    glm::dvec3 bc(0);
    int bn = 0;
    for (auto& v : comp)
        if (v.y <= lo.y + 1) { bc += glm::dvec3(v); ++bn; }
    glm::ivec3 base(int(std::round(bc.x / bn)), lo.y, int(std::round(bc.z / bn)));
    float r2 = 1.0f;
    for (auto& v : comp)
        r2 = std::max(r2, glm::distance(glm::vec2(v.x, v.z),
                                        glm::vec2(base.x, base.z)));
    out.valid  = true;
    out.base   = base;
    out.radius = r2;
    out.stone  = std::move(comp);
    out.lo = lo; out.hi = hi;
}

bool collectRock(const World& w, const GenContext& gen, glm::ivec3 seed,
                 RockInfo& out) {
    VOX_ZONE_N("forestry: collect rock");
    out = RockInfo{};
    if (w.getVoxel(seed.x, seed.y, seed.z) != MAT_STONE) return false;
    auto t0 = std::chrono::steady_clock::now();
    auto ms = [&]() {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t0).count();
    };

    // Per-column soil line, cached (heightAt is a stack of noise octaves).
    std::unordered_map<int64_t, int> surf;
    auto soilY = [&](int x, int z) {
        int64_t k = (int64_t(x) << 32) ^ (int64_t(z) & 0xFFFFFFFFll);
        auto it = surf.find(k);
        if (it != surf.end()) return it->second;
        int h = terrainHeightAt(x, z);
        surf.emplace(k, h);
        return h;
    };

    // Ask the GENERATOR first (rederiveRockAt): exact per-boulder membership
    // splits conjoined clusters, which the flood below fuses into one
    // component and rejects through the terrain caps. The claimed set is
    // clamped at the soil line (the deep root stays in the ground as the
    // quarry scar — same contract as the flood) and intersected with live
    // MAT_STONE; a garbled replay (gate divergence over a cave-carved
    // anchor, a seam-class tree block) fails the liveness ratio or loses
    // the clicked voxel, and we fall through to the flood.
    {
        RockRederive rd;
        if (rederiveRockAt(gen, w, seed, rd)) {
            std::vector<glm::ivec3> keep;
            keep.reserve(rd.voxels.size());
            int clamped = 0;
            bool haveSeed = false;
            for (auto& v : rd.voxels) {
                if (v.y < soilY(v.x, v.z) - ROCK_ROOT_K) continue;
                ++clamped;
                if (w.getVoxel(v.x, v.y, v.z) != MAT_STONE) continue;
                haveSeed = haveSeed || v == seed;
                keep.push_back(v);
            }
            if (haveSeed && int(keep.size()) >= ROCK_MIN &&
                int(keep.size()) <= ROCK_GEN_CAP &&
                keep.size() * 10 >= size_t(clamped) * 6) {
                glm::ivec3 lo = keep[0], hi = keep[0];
                for (auto& v : keep) {
                    lo = glm::min(lo, v);
                    hi = glm::max(hi, v);
                }
                if (!insideWindow(w, lo, hi)) return false;
                rockFromComp(std::move(keep), lo, hi, out);
                std::printf("[forestry] rock identity (gen): %zu stone, base "
                            "%d,%d,%d r %.1f anchor %d,%d in %.1f ms\n",
                            out.stone.size(), out.base.x, out.base.y,
                            out.base.z, out.radius, rd.anchor.x, rd.anchor.z,
                            ms());
                return true;
            }
            std::printf("[forestry] rock rederive distrusted (%zu claimed, "
                        "%d clamped, %zu alive, seed %d) -- flood fallback\n",
                        rd.voxels.size(), clamped, keep.size(), int(haveSeed));
        }
    }

    auto inBoulder = [&](glm::ivec3 v) {
        return v.y >= soilY(v.x, v.z) - ROCK_ROOT_K &&
               w.getVoxel(v.x, v.y, v.z) == MAT_STONE;
    };
    // (No print: this fires per hovered voxel while sweeping a scar field
    // or a strata skin — the hovPickVox latch caches the rejection, but the
    // first touch of every voxel would still spam a line.)
    if (!inBoulder(seed)) return false;

    std::unordered_set<int64_t> seen{voxKey(seed)};
    std::deque<glm::ivec3> q{seed};
    std::vector<glm::ivec3> comp;
    glm::ivec3 lo = seed, hi = seed;
    while (!q.empty()) {
        glm::ivec3 v = q.front(); q.pop_front();
        comp.push_back(v);
        lo = glm::min(lo, v); hi = glm::max(hi, v);
        if (int(comp.size()) > ROCK_CAP) {
            std::printf("[forestry] rock reject: volume cap (terrain, not a boulder)\n");
            return false;
        }
        if (glm::any(glm::greaterThan(hi - lo, glm::ivec3(ROCK_EXTENT_CAP)))) {
            std::printf("[forestry] rock reject: extent cap %d,%d,%d (terrain)\n",
                        hi.x - lo.x, hi.y - lo.y, hi.z - lo.z);
            return false;
        }
        for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    if (!dx && !dy && !dz) continue;
                    glm::ivec3 n = v + glm::ivec3(dx, dy, dz);
                    if (!inBoulder(n)) continue;
                    if (seen.insert(voxKey(n)).second) q.push_back(n);
                }
    }
    if (int(comp.size()) < ROCK_MIN) {
        std::printf("[forestry] rock reject: only %zu voxels\n", comp.size());
        return false;
    }
    if (!insideWindow(w, lo, hi)) return false;

    rockFromComp(std::move(comp), lo, hi, out);
    std::printf("[forestry] rock identity (flood): %zu stone, base %d,%d,%d "
                "r %.1f in %.1f ms\n",
                out.stone.size(), out.base.x, out.base.y, out.base.z,
                out.radius, ms());
    return true;
}

// ---------------------------------------------------------------------------
// Hover highlight worker
// ---------------------------------------------------------------------------

// Build the result's mask + wood stats from a (position, material) span.
template <typename GetPos, typename GetMat>
static bool hoverMaskFrom(TreeHoverResult& r, size_t n, GetPos pos, GetMat mat) {
    if (n == 0) return false;
    glm::ivec3 lo = pos(0), hi = lo;
    for (size_t i = 1; i < n; ++i) {
        lo = glm::min(lo, pos(i));
        hi = glm::max(hi, pos(i));
    }
    glm::ivec3 d = hi - lo + 1;
    if (d.x > 200 || d.y > 200 || d.z > 200) return false;
    r.lo = lo;
    r.dims = d;
    r.mask.assign(size_t(d.x) * d.y * d.z, 0);
    int minWy = INT32_MAX, maxWy = INT32_MIN;
    for (size_t i = 0; i < n; ++i) {
        glm::ivec3 c = pos(i) - lo;
        // 255, not 1: the mask uploads as normalized R8 — the shader
        // thresholds at 0.5 and 1/255 would never pass.
        r.mask[size_t(c.x) + size_t(d.x) * (size_t(c.y) + size_t(d.y) * c.z)] = 255;
        uint8_t m = mat(i);
        if (isWoodMat(m) || isCactusSolid(m)) {
            ++r.woodN;
            minWy = std::min(minWy, pos(i).y);
            maxWy = std::max(maxWy, pos(i).y);
            if (int(r.woodSample.size()) < 64 && (r.woodN % 7) == 0)
                r.woodSample.push_back(pos(i));
        }
    }
    r.woodH = (r.woodN > 0) ? maxWy - minWy + 1 : 0;
    return true;
}

static void treeHoverMain(TreeHover* th) {
    for (;;) {
        glm::ivec3 vox;
        const GenContext* gen = nullptr;
        const World* world = nullptr;
        uint64_t id = 0;
        {
            std::unique_lock<std::mutex> lk(th->mtx);
            th->cv.wait(lk, [&] { return th->quit || th->hasReq; });
            if (th->quit) return;
            th->hasReq = false;
            vox = th->reqVox;
            gen = th->gen;
            world = th->world;
            id  = th->reqId;
        }
        VOX_ZONE_N("hover identify");
        TreeHoverResult r;
        r.req = id;
        TreeRederive rd;
        if (rederiveTreeAt(*gen, vox, rd) && !rd.voxels.empty()) {
            r.found = hoverMaskFrom(r, rd.voxels.size(),
                                    [&](size_t i) { return rd.voxels[i].first; },
                                    [&](size_t i) { return rd.voxels[i].second; });
        }
        // Non-registry flora (palms, desert trees, cacti): flood-identify.
        // The worker reads the live World here — same accepted risk profile
        // as the async streaming computes (brickpool is reserve-allocated and
        // never reallocates in play; a torn read at worst smudges one hover
        // mask for a frame). The MAIN thread still does the authoritative
        // spot-check before adopting.
        if (!r.found && world) {
            TreeInfo ti;
            if (collectTreeFlood(*world, vox, ti) && !ti.isStump) {
                std::vector<glm::ivec3> all = ti.wood;
                all.insert(all.end(), ti.leaves.begin(), ti.leaves.end());
                r.found = hoverMaskFrom(r, all.size(),
                    [&](size_t i) { return all[i]; },
                    [&](size_t i) {
                        return i < ti.wood.size()
                             ? (i < ti.woodMats.size() ? ti.woodMats[i]
                                                       : uint8_t(MAT_WOOD))
                             : uint8_t(MAT_LEAVES);
                    });
                if (r.found) r.isCactus = ti.isCactus;
            }
        }
        {
            std::lock_guard<std::mutex> lk(th->mtx);
            if (id == th->reqId) {          // else a newer request superseded us
                th->result = std::move(r);
                th->hasResult = true;
            }
        }
    }
}

TreeHover::~TreeHover() {
    {
        std::lock_guard<std::mutex> lk(mtx);
        quit = true;
    }
    cv.notify_all();
    if (worker.joinable()) worker.join();
}

void treeHoverRequest(TreeHover& th, const GenContext& gen, const World& w,
                      glm::ivec3 vox) {
    {
        std::lock_guard<std::mutex> lk(th.mtx);
        th.gen = &gen;
        th.world = &w;
        th.reqVox = vox;
        ++th.reqId;
        th.hasReq = true;
        th.hasResult = false;               // an in-flight answer is stale now
        if (!th.worker.joinable())
            th.worker = std::thread(treeHoverMain, &th);
    }
    th.cv.notify_one();
}

bool treeHoverPoll(TreeHover& th, TreeHoverResult& out) {
    std::lock_guard<std::mutex> lk(th.mtx);
    if (!th.hasResult) return false;
    th.hasResult = false;
    out = std::move(th.result);
    return true;
}

// ---------------------------------------------------------------------------
// Felling
// ---------------------------------------------------------------------------

// Wood below the cut plane AND hugging the trunk column stays as the stump;
// everything else (low branches included) fells.
static bool isStumpVoxel(const TreeInfo& t, glm::ivec3 v) {
    if (v.y >= t.cutY) return false;
    glm::vec2 d(float(v.x - t.base.x), float(v.z - t.base.z));
    return glm::dot(d, d) <= t.stumpR * t.stumpR;
}

// Pack voxel sets into model parts. Model-local space = world - pivot. The
// volume's bytes are WORLD MATERIAL IDS, not palette indices — the fall
// instances render in terrain-material mode (meta.y = 1), so the lighting
// pass shades the moving tree through the exact path its standing voxels
// used and the fell causes no step change in appearance.
//
// The set is GRID-SPLIT into tight sub-box parts instead of one union bbox:
// a big crown's union is ~85-90% empty space, so tight boxes cut the
// per-pixel DDA and the 4-tap shadow trace by several x (escaping rays slab-
// reject most boxes) and shrink the atlas footprint an order of magnitude.
// Capped per set so the lighting shadow loop's per-bound part walk stays
// bounded (~one extra creature's worth per fall).
static void addFellParts(CreatureModel& m, const char* name,
                         const std::vector<glm::ivec3>& cells,
                         const uint8_t* mats, uint8_t fallbackMat,
                         glm::vec3 pivot) {
    if (cells.empty()) return;
    glm::ivec3 setLo = cells[0];
    for (auto& v : cells) setLo = glm::min(setLo, v);
    constexpr int PART_CAP = 10;   // per set: wood + leaves <= 20 per fall
    int S = 32;
    std::unordered_map<int64_t, std::vector<int>> buckets;
    for (;; S *= 2) {
        buckets.clear();
        for (int i = 0; i < int(cells.size()); ++i) {
            glm::ivec3 b = (cells[i] - setLo) / S;
            buckets[voxKey(b)].push_back(i);
        }
        if (int(buckets.size()) <= PART_CAP) break;
    }
    int part = 0;
    for (auto& [key, idxs] : buckets) {
        glm::ivec3 lo = cells[idxs[0]], hi = lo;
        for (int i : idxs) {
            lo = glm::min(lo, cells[i]);
            hi = glm::max(hi, cells[i]);
        }
        glm::ivec3 d = hi - lo + 1;
        CreaturePart p;
        std::snprintf(p.name, sizeof(p.name), "%s%d", name, part++);
        p.parent = -1;
        p.gridOrigin = glm::vec3(lo) - pivot;
        p.pivot = glm::vec3(0);
        p.dims = d;
        p.wordBase = uint32_t(m.vox.size());
        size_t n = size_t(d.x) * d.y * d.z;
        m.vox.resize(m.vox.size() + (n + 3) / 4, 0u);
        for (int i : idxs) {
            glm::ivec3 c = cells[i] - lo;
            size_t lin = size_t(c.x) +
                         size_t(d.x) * (size_t(c.y) + size_t(d.y) * c.z);
            uint8_t pi = mats ? mats[i] : fallbackMat;
            m.vox[p.wordBase + (lin >> 2)] |= uint32_t(pi) << ((lin & 3) * 8);
        }
        m.parts.push_back(p);
    }
}

bool fellTree(Forestry& f, World& w, const TreeInfo& t, glm::vec3 awayFrom,
              ParticleSystem& ps, std::vector<uint32_t>& atlasDyn) {
    VOX_ZONE_N("forestry: fell");
    if (!t.valid || t.isStump) return false;
    if (int(f.falls.size()) >= Forestry::MAX_FALLS) return false;

    FallingTree ft;
    ft.pivot = glm::vec3(t.base.x + 0.5f, float(t.cutY), t.base.z + 0.5f);

    // Fall away from the chopper; degenerate (standing on the trunk) picks +x.
    glm::vec2 away(ft.pivot.x - awayFrom.x, ft.pivot.z - awayFrom.z);
    float alen = glm::length(away);
    away = (alen > 2.0f) ? away / alen : glm::vec2(1, 0);
    ft.dir = glm::vec3(away.x, 0, away.y);

    // Split the wood, clear the world (single batched delta), keep the stump
    // — its top face becomes the pale sawn cross-section. Every voxel is
    // re-read and filtered against the CURRENT world: the TreeInfo may be
    // seconds old (collected when the chop began — re-collecting on the
    // final swing was a big part of the fell hitch), and fire or edits may
    // have mutated parts of the tree since.
    EditDelta d;
    int topY = t.cutY;
    std::vector<glm::ivec3> cutFace;
    for (size_t i = 0; i < t.wood.size(); ++i) {
        glm::ivec3 v = t.wood[i];
        uint8_t cur = w.getVoxel(v.x, v.y, v.z);
        bool okSolid = t.isCactus ? isCactusSolid(cur) : isWoodMat(cur);
        if (!okSolid) continue;                    // burned/dug since collect
        if (isStumpVoxel(t, v)) {
            // Cacti keep their flesh nub as-is: growth rings on a cactus
            // would be nonsense.
            if (!t.isCactus && v.y == t.cutY - 1) cutFace.push_back(v);
            continue;
        }
        ft.fellWood.push_back(v);
        ft.fellMats.push_back(cur);
        topY = std::max(topY, v.y);
        w.setVoxel(v.x, v.y, v.z, MAT_AIR, d);
    }
    if (ft.fellWood.empty()) return false;
    for (auto& v : t.leaves) {
        if (!isFoliage(w.getVoxel(v.x, v.y, v.z))) continue;
        ft.fellLeaf.push_back(v);
        ft.fellLeafMats.push_back(w.getVoxel(v.x, v.y, v.z));
        w.setVoxel(v.x, v.y, v.z, MAT_AIR, d);
    }
    // Growth rings: alternate pale sapwood / darker bands by integer radius
    // — measured from the CUT FACE's own centroid, not t.base. The base is
    // the generator's site anchor, and the trunk CURVES as it grows
    // (flora.cpp jitters every segment), so by cut height the wood disc sits
    // up to a voxel off that axis and anchor-centred rings always looked one
    // block off-centre.
    if (!cutFace.empty()) {
        glm::dvec2 acc(0.0);
        for (auto& v : cutFace) acc += glm::dvec2(v.x + 0.5, v.z + 0.5);
        glm::vec2 fc(acc / double(cutFace.size()));
        for (auto& v : cutFace) {
            float rr = glm::distance(glm::vec2(v.x + 0.5f, v.z + 0.5f), fc);
            uint8_t face = (int(rr) & 1) ? MAT_CUT_RING : MAT_CUT_WOOD;
            w.setVoxel(v.x, v.y, v.z, face, d);
        }
    }
    w.uploadDelta(d);
    // No leaf may be left hanging, whatever the ownership pass decided: the
    // support-decay sweep floods outward from everything we removed and
    // crumbles any canopy that lost its wood (amortized over a few frames).
    {
        std::vector<glm::ivec3> gone = ft.fellWood;
        gone.insert(gone.end(), ft.fellLeaf.begin(), ft.fellLeaf.end());
        forestryQueueLeafDecay(f, gone);
    }

    ft.trunkLen = std::max(4.0f, float(topY + 1) - ft.pivot.y);

    // The transient rigid model: grid-split tight parts so the per-pixel DDA
    // never marches the union's empty space. Volumes carry world MATERIAL
    // ids (rendered in terrain-material mode — see addFellParts).
    ft.model.cellScale = 1.0f;
    addFellParts(ft.model, "wood", ft.fellWood, ft.fellMats.data(), MAT_WOOD,
                 ft.pivot);
    addFellParts(ft.model, "leaf", ft.fellLeaf,
                 ft.fellLeafMats.empty() ? nullptr : ft.fellLeafMats.data(),
                 MAT_LEAVES, ft.pivot);
    // Snapshot file-local word bases: atlas compaction re-appends live falls,
    // and appendModelToAtlas rebases wordBase in place each time.
    for (auto& p : ft.model.parts) ft.localWordBase.push_back(p.wordBase);

    // Ground-contact probes: sample the WHOLE model (trunk and crown, in
    // pivot-local space). The fall crumbles the moment any of them sweeps
    // into solid ground — the model never visibly clips in.
    auto addProbes = [&](const std::vector<glm::ivec3>& cells, size_t target) {
        size_t stride = std::max<size_t>(1, cells.size() / target);
        for (size_t i = 0; i < cells.size(); i += stride) {
            glm::vec3 local = glm::vec3(cells[i]) + 0.5f - ft.pivot;
            if (glm::length(local) > 3.5f) ft.probes.push_back(local);
        }
    };
    addProbes(ft.fellWood, 48);
    addProbes(ft.fellLeaf, 32);
    if (ft.probes.empty())
        ft.probes.push_back(glm::vec3(0.0f, ft.trunkLen, 0.0f));

    // Rotation-invariant bound: centre of the local bbox, radius to its
    // farthest corner (+margin) — valid at every fall angle.
    glm::vec3 blo(1e9f), bhi(-1e9f);
    for (auto& p : ft.model.parts) {
        blo = glm::min(blo, p.gridOrigin);
        bhi = glm::max(bhi, p.gridOrigin + glm::vec3(p.dims));
    }
    ft.boundCentroid = (blo + bhi) * 0.5f;
    ft.boundRadius = glm::length(bhi - blo) * 0.5f + 2.0f;

    // A nudge so sin(angle) has something to work with (the axe bite).
    ft.angle = 0.05f;
    ft.angVel = 0.12f;

    appendModelToAtlas(ft.model, atlasDyn);   // grow-only; caller re-uploads
    ps.spawnDust(ft.pivot + glm::vec3(0, 1, 0), MAT_WOOD, 6);
    f.falls.push_back(std::move(ft));
    return true;
}

// ---------------------------------------------------------------------------
// Debris items
// ---------------------------------------------------------------------------

static constexpr int   ITEM_CAP        = 320;
static constexpr int   WOOD_ITEM_MAX   = 220;
static constexpr int   LEAF_ITEM_MAX   = 80;
static constexpr float MAGNET_RADIUS   = 30.0f;   // 3 m
static constexpr float ABSORB_RADIUS   = 3.2f;
static constexpr float ITEM_LIFE       = 150.0f;

static void spawnItem(Forestry& f, const World& w, glm::vec3 pos, uint8_t mat,
                      glm::vec3 vel) {
    if (int(f.items.size()) >= ITEM_CAP) {
        // Wood is loot, leaves are ambience: a full pool evicts its oldest
        // leaf before it ever drops a wood item.
        if (isFoliage(mat)) return;
        int oldest = -1;
        float oldestAge = -1.0f;
        for (int i = 0; i < int(f.items.size()); ++i)
            if (isFoliage(f.items[i].mat) && f.items[i].age > oldestAge) {
                oldestAge = f.items[i].age;
                oldest = i;
            }
        if (oldest < 0) {
            // Pool saturated with LOOT (a rock crumble during a fell's
            // flight window): the source voxels are already gone, so a
            // dropped spawn is silently destroyed yield. Skip the flight
            // and bank the credit directly.
            ++f.collected[mat];
            return;
        }
        f.items[oldest] = f.items.back();
        f.items.pop_back();
    }
    // Nudge out of solids: on a hillside a fallen trunk can land deep inside
    // the slope, and a buried item is an invisible (lost-looking) item.
    for (int i = 0; i < 24 && solidAt(w, pos); ++i) pos.y += 1.0f;
    DebrisItem it;
    it.pos = pos;
    it.vel = vel;
    it.mat = mat;
    std::uniform_real_distribution<float> ph(0.0f, 6.283f);
    std::uniform_real_distribution<float> mg(0.25f, 0.85f);
    it.spin = ph(f.rng);
    // Everything auto-collects: after a brief scatter each item turns
    // magnetized no matter how far the player is (walking the pickup line
    // with click-to-move was jank). Staggered so the stream reads as a
    // stream, not a clump.
    it.magAt = mg(f.rng);
    f.items.push_back(it);
}

// Shatter a felled set into items + dust. `R` maps pre-fall world offsets
// (about pivot) to their fallen positions (identity for uproots). Char burns
// to dust rather than dropping wood.
static void shatterIntoItems(Forestry& f, const World& w, ParticleSystem& ps,
                             glm::vec3 pivot, const glm::mat3& R,
                             const std::vector<glm::ivec3>& wood,
                             const uint8_t* mats,
                             const std::vector<glm::ivec3>& leaves,
                             const uint8_t* leafMats = nullptr) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    auto place = [&](glm::ivec3 v) {
        return pivot + R * (glm::vec3(v) + 0.5f - pivot);
    };
    auto burst = [&](glm::vec3 p) {
        glm::vec3 radial = p - pivot;
        radial.y = 0.0f;
        float rl = glm::length(radial);
        radial = (rl > 0.5f) ? radial / rl : glm::vec3(0);
        return radial * (2.0f + 4.0f * (u(f.rng) * 0.5f + 0.5f)) +
               glm::vec3(u(f.rng) * 5.0f, 10.0f + 9.0f * (u(f.rng) * 0.5f + 0.5f),
                         u(f.rng) * 5.0f);
    };
    size_t wStride = std::max<size_t>(1, wood.size() / WOOD_ITEM_MAX);
    for (size_t i = 0; i < wood.size(); i += wStride) {
        uint8_t m = mats ? mats[i] : MAT_WOOD;
        glm::vec3 p = place(wood[i]);
        if (m == MAT_CHAR) ps.spawnDust(p, MAT_CHAR, 2);
        else               spawnItem(f, w, p, m, burst(p));
    }
    size_t lStride = std::max<size_t>(1, leaves.size() / LEAF_ITEM_MAX);
    for (size_t i = 0; i < leaves.size(); i += lStride)
        spawnItem(f, w, place(leaves[i]),
                  leafMats ? leafMats[i] : MAT_LEAVES,
                  burst(place(leaves[i])));
    // Dust along the wreck: leaves mostly poof.
    if (!leaves.empty()) {
        size_t dStride = std::max<size_t>(1, leaves.size() / 40);
        for (size_t i = 0; i < leaves.size(); i += dStride)
            ps.spawnDust(place(leaves[i]),
                         leafMats ? leafMats[i] : MAT_LEAVES, 3);
    }
    if (!wood.empty()) {
        size_t wdStride = std::max<size_t>(1, wood.size() / 14);
        for (size_t i = 0; i < wood.size(); i += wdStride)
            ps.spawnDust(place(wood[i]), mats ? mats[i] : MAT_WOOD, 3);
    }
}

void uprootStump(Forestry& f, World& w, const TreeInfo& t, ParticleSystem& ps) {
    VOX_ZONE_N("forestry: uproot");
    if (!t.valid) return;
    // Re-read every snapshot voxel against the CURRENT world — the click-time
    // TreeInfo can be seconds stale (fire chars wood, a neighbour's fell or
    // the decay sweep removes leaves). Only SURVIVORS are removed, seeded,
    // and shattered: feeding the stale lists to shatterIntoItems spawned
    // full wood items for charred trunks and leaf items for foliage that had
    // already burned away.
    EditDelta d;
    std::vector<glm::ivec3> wood, leaves;
    std::vector<uint8_t>    woodMats, leafMats;
    wood.reserve(t.wood.size());
    woodMats.reserve(t.wood.size());
    for (auto& v : t.wood) {
        uint8_t m = w.getVoxel(v.x, v.y, v.z);
        if (isWoodMat(m) || isCactusSolid(m)) {
            w.setVoxel(v.x, v.y, v.z, MAT_AIR, d);
            wood.push_back(v);
            woodMats.push_back(m);
        }
    }
    leaves.reserve(t.leaves.size());
    leafMats.reserve(t.leaves.size());
    for (auto& v : t.leaves) {
        uint8_t m = w.getVoxel(v.x, v.y, v.z);
        if (isFoliage(m)) {
            w.setVoxel(v.x, v.y, v.z, MAT_AIR, d);
            leaves.push_back(v);
            leafMats.push_back(m);
        }
    }
    w.uploadDelta(d);
    {
        std::vector<glm::ivec3> gone = wood;
        gone.insert(gone.end(), leaves.begin(), leaves.end());
        forestryQueueLeafDecay(f, gone);
    }
    glm::vec3 pivot(t.base.x + 0.5f, t.base.y + 0.5f, t.base.z + 0.5f);
    shatterIntoItems(f, w, ps, pivot, glm::mat3(1.0f), wood,
                     woodMats.empty() ? nullptr : woodMats.data(), leaves,
                     leafMats.empty() ? nullptr : leafMats.data());
    ps.spawnDust(pivot, MAT_DIRT, 8);
}

void mineRock(Forestry& f, World& w, const RockInfo& r, ParticleSystem& ps) {
    VOX_ZONE_N("forestry: mine rock");
    if (!r.valid) return;
    // Survivors only, like uprootStump: the click-time roster can be swings
    // stale, and the world edit AND the loot must agree — shattering the
    // snapshot would spawn stone items (and satchel credit) for voxels that
    // were already gone at crumble time.
    EditDelta d;
    std::unordered_set<int64_t> gone;
    std::vector<glm::ivec3> survivors;
    survivors.reserve(r.stone.size());
    for (auto& v : r.stone)
        if (w.getVoxel(v.x, v.y, v.z) == MAT_STONE) {
            w.setVoxel(v.x, v.y, v.z, MAT_AIR, d);
            gone.insert(voxKey(v));
            survivors.push_back(v);
        }
    // Stranded-crumb sweep: a removal can leave small unsupported stone blobs
    // hovering — a garbled-replay bite out of a neighbour, a ledge overhang
    // the soil clamp let through. Flood each remaining above-soil stone blob
    // touching the removed shell (small cap: cap-hit == big == grounded rock,
    // leave it); a blob with no solid footing pops into debris items instead
    // of floating forever. Mirrors the leaf-decay "no hanging leaves" rule.
    {
        constexpr int CRUMB_CAP = 400;
        std::unordered_set<int64_t> visited;
        std::unordered_map<int64_t, int> surf;   // heightAt is a noise stack
        auto above = [&](glm::ivec3 v) {
            int64_t k = (int64_t(v.x) << 32) ^ (int64_t(v.z) & 0xFFFFFFFFll);
            auto it = surf.find(k);
            int h = it != surf.end() ? it->second : terrainHeightAt(v.x, v.z);
            if (it == surf.end()) surf.emplace(k, h);
            return v.y >= h - ROCK_ROOT_K;
        };
        auto crumbStone = [&](glm::ivec3 v) {
            return !gone.count(voxKey(v)) && above(v) &&
                   w.getVoxel(v.x, v.y, v.z) == MAT_STONE;
        };
        for (auto& rv : survivors) {
            // Interior voxels can't touch a crumb: any 26-neighbour of a
            // voxel whose SIX face neighbours are all removed is itself
            // 26-adjacent to one of those face neighbours — seeding from
            // the face boundary is complete, and skips ~4/5 of a big rock.
            bool interior = true;
            static const glm::ivec3 F[6] = {{1,0,0}, {-1,0,0}, {0,1,0},
                                            {0,-1,0}, {0,0,1}, {0,0,-1}};
            for (int i = 0; i < 6 && interior; ++i)
                interior = gone.count(voxKey(rv + F[i])) != 0;
            if (interior) continue;
            for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                glm::ivec3 s0 = rv + glm::ivec3(dx, dy, dz);
                if (!crumbStone(s0) || visited.count(voxKey(s0))) continue;
                std::vector<glm::ivec3> blob{s0};
                std::deque<glm::ivec3> q{s0};
                visited.insert(voxKey(s0));
                bool grounded = false, big = false;
                while (!q.empty() && !big) {
                    glm::ivec3 v = q.front(); q.pop_front();
                    uint8_t below = w.getVoxel(v.x, v.y - 1, v.z);
                    if (below != MAT_AIR && below != MAT_STONE &&
                        isBlockingForPlayer(below))
                        grounded = true;   // sits on dirt/strata, not on kin
                    if (below == MAT_STONE && !above({v.x, v.y - 1, v.z}))
                        grounded = true;   // rooted into sub-soil stone
                    for (int z2 = -1; z2 <= 1; ++z2)
                    for (int y2 = -1; y2 <= 1; ++y2)
                    for (int x2 = -1; x2 <= 1; ++x2) {
                        glm::ivec3 n = v + glm::ivec3(x2, y2, z2);
                        if (!crumbStone(n) || visited.count(voxKey(n))) continue;
                        visited.insert(voxKey(n));
                        blob.push_back(n);
                        q.push_back(n);
                        if (int(blob.size()) > CRUMB_CAP) { big = true; break; }
                    }
                }
                if (big || grounded) continue;
                for (size_t i = 0; i < blob.size(); ++i) {
                    auto& v = blob[i];
                    w.setVoxel(v.x, v.y, v.z, MAT_AIR, d);
                    gone.insert(voxKey(v));
                    if (i % 6 == 0)
                        spawnItem(f, w, glm::vec3(v) + 0.5f, MAT_STONE,
                                  glm::vec3(0, 12.0f, 0));
                }
                if (!blob.empty())
                    ps.spawnDust(glm::vec3(blob[0]) + 0.5f, MAT_STONE, 3);
            }
        }
    }
    w.uploadDelta(d);
    // A boulder can be load-bearing: sage rooted on its flank, a felled
    // tree's branch stub the decay sweep grounded VIA the stone. Seed the
    // same invariant every other remover seeds.
    forestryQueueLeafDecay(f, survivors);
    // Crumble about the blob centre so chunks burst outward on all sides
    // (the base sits on the ground — bursting about it looks like a geyser).
    glm::vec3 pivot = (glm::vec3(r.lo) + glm::vec3(r.hi)) * 0.5f + 0.5f;
    std::vector<uint8_t> mats(survivors.size(), MAT_STONE);
    static const std::vector<glm::ivec3> noLeaves;
    shatterIntoItems(f, w, ps, pivot, glm::mat3(1.0f), survivors, mats.data(),
                     noLeaves);
    ps.spawnDust(glm::vec3(r.base) + glm::vec3(0.5f, 1.0f, 0.5f), MAT_STONE, 8);
}

// ---------------------------------------------------------------------------
// Leaf-support decay (the "no hanging leaves" invariant)
//
// A MAT_LEAVES voxel survives only if its 26-conn leaf-graph distance to
// wood-like support is <= LEAF_SUPPORT_N. Enforced as amortized jobs seeded
// by whatever got removed (fell, uproot, fire) — the flood covers the WHOLE
// touched leaf component with no bbox escape hatch, so remnants bridged
// through the removed crown, LEAF_CAP truncation leftovers, and worldgen's
// quasi-floating popcorn bulbs all crumble once an edit disturbs them.
// Calibration (flora.cpp): forest crowns keep leaves within ~19-30 graph
// steps of branch wood; oasis palm fronds are continuous ~36-step leaf ribs
// — N = 48 clears both with margin. Ungrounded floating WOOD stubs (our own
// branches dropped by the merge partition) dissolve too: they'd otherwise
// anchor a floating mini-canopy forever.
// ---------------------------------------------------------------------------

static constexpr int LEAF_SUPPORT_N      = 48;
// The gather budget is denominated in WORLD READS (the dominant cost): each
// adopted voxel bills its full 26-neighbour sweep (~27 getVoxel calls) and a
// fresh wood-stub flood bills its component the same way. Counting "voxel
// visits" undercounted the real work ~27x — a nominally budgeted frame did
// ~650k scattered reads during a mega-canopy decay.
static constexpr int DECAY_GATHER_BUDGET = 160000;  // ~world reads / frame
static constexpr int DECAY_RELAX_BUDGET  = 24000;   // BFS relaxes / frame
static constexpr int DECAY_REMOVE_BUDGET = 4000;    // removals / frame
static constexpr int DECAY_REGION_CAP    = 200000;  // abort: merged mega-mass
static constexpr int DECAY_STUB_CAP      = 512;     // wood-stub flood bound

void forestryQueueLeafDecay(Forestry& f, const std::vector<glm::ivec3>& removed) {
    if (removed.empty()) return;
    LeafDecayJob job;
    job.seeds.insert(job.seeds.end(), removed.begin(), removed.end());
    f.decayJobs.push_back(std::move(job));
}

// Is this wood voxel part of a grounded (or big, or sawn) component? Caches
// verdicts per job; a freshly-discovered UNGROUNDED stub's voxels are queued
// for dissolution and never count as support.
static bool woodSupports(LeafDecayJob& job, const World& w, glm::ivec3 v) {
    auto it = job.woodVerdict.find(voxKey(v));
    if (it != job.woodVerdict.end()) return it->second;
    std::vector<glm::ivec3> comp{v};
    std::unordered_set<int64_t> seen{voxKey(v)};
    std::deque<glm::ivec3> q{v};
    bool grounded = false;
    while (!q.empty() && int(comp.size()) <= DECAY_STUB_CAP) {
        glm::ivec3 c = q.front(); q.pop_front();
        uint8_t below = w.getVoxel(c.x, c.y - 1, c.z);
        if (below != MAT_AIR && !isWoodMat(below) && !isLeafMat(below) &&
            isBlockingForPlayer(below))
            grounded = true;
        for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    if (!dx && !dy && !dz) continue;
                    glm::ivec3 n = c + glm::ivec3(dx, dy, dz);
                    uint8_t m = w.getVoxel(n.x, n.y, n.z);
                    if (isCutFace(m)) grounded = true;        // a stump: rooted
                    if (!isWoodMat(m)) continue;
                    if (seen.insert(voxKey(n)).second) {
                        comp.push_back(n);
                        q.push_back(n);
                    }
                }
    }
    if (int(comp.size()) > DECAY_STUB_CAP) grounded = true;   // big = a real tree
    for (auto& c : comp) job.woodVerdict.emplace(voxKey(c), grounded);
    if (!grounded)
        job.deadWood.insert(job.deadWood.end(), comp.begin(), comp.end());
    return grounded;
}

void forestryDecayTick(Forestry& f, World& w, ParticleSystem& ps) {
    if (f.decayJobs.empty()) return;
    VOX_ZONE_N("forestry: decay tick");
    LeafDecayJob& job = f.decayJobs.front();

    if (job.phase == 0) {
        // Gather: expand seeds to their leaf neighbours, then flood the leaf
        // region, spotting support (and floating stubs) along the way. The
        // flood is HORIZON-BOUNDED: only leaves within ~N steps of a removed
        // voxel can have lost their support path (a support chain is <= N
        // long), so anything farther is untouched — without this bound a
        // dense forest's connected canopy is one mega-region and the job
        // aborts at the cap. Leaves AT the horizon count as supported (their
        // support, if any, lies out in unexplored canopy — conservative,
        // never eats a healthy crown).
        constexpr int GATHER_HORIZON = LEAF_SUPPORT_N + 2;
        int budget = DECAY_GATHER_BUDGET;
        while (budget > 0 && !job.seeds.empty()) {
            glm::ivec3 s = job.seeds.front(); job.seeds.pop_front();
            for (int dz = -1; dz <= 1; ++dz)
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (!dx && !dy && !dz) continue;
                        glm::ivec3 n = s + glm::ivec3(dx, dy, dz);
                        uint8_t m = w.getVoxel(n.x, n.y, n.z);
                        if (isFoliage(m)) {
                            // Push-time dedup: without it every interior
                            // leaf enters the frontier up to 26x and the
                            // budget is spent popping duplicates.
                            if (!job.idxOf.count(voxKey(n)) &&
                                job.queued.insert(voxKey(n)).second)
                                job.frontier.push_back(glm::ivec4(n, 1));
                        } else if (isWoodMat(m)) {
                            // A removal can sever WOOD too (a doused or
                            // dug-out trunk band): probe the surviving
                            // neighbour component — ungrounded stubs queue
                            // into deadWood and dissolve in phase 2. (The
                            // STUB_CAP bound applies: a >512-voxel floating
                            // tree reads as grounded, by design.)
                            size_t pre = job.woodVerdict.size();
                            woodSupports(job, w, n);
                            budget -= 27 * int(job.woodVerdict.size() - pre);
                        }
                    }
            budget -= 27;                       // the neighbour sweep above
        }
        while (budget > 0 && !job.frontier.empty()) {
            glm::ivec4 pd = job.frontier.front(); job.frontier.pop_front();
            glm::ivec3 p(pd.x, pd.y, pd.z);
            budget -= 1;                        // the liveness re-check read
            if (!isFoliage(w.getVoxel(p.x, p.y, p.z))) continue;
            int64_t k = voxKey(p);
            if (job.idxOf.count(k)) continue;
            budget -= 26;                       // this voxel's neighbour sweep
            int idx = int(job.voxels.size());
            job.idxOf.emplace(k, idx);
            job.voxels.push_back(p);
            job.dist.push_back(-1);
            bool supported = false;
            // Ground-rooted sage is its own plant (desert groundcover), not
            // canopy: sitting on solid ground counts as support.
            if (w.getVoxel(p.x, p.y, p.z) == MAT_SHRUB_SAGE) {
                uint8_t below = w.getVoxel(p.x, p.y - 1, p.z);
                if (below != MAT_AIR && !isFoliage(below) &&
                    isBlockingForPlayer(below))
                    supported = true;
            }
            if (pd.w >= GATHER_HORIZON) {
                supported = true;   // horizon fence: assume support beyond
            } else if (!supported) {
                for (int dz = -1; dz <= 1; ++dz)
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            if (!dx && !dy && !dz) continue;
                            glm::ivec3 n = p + glm::ivec3(dx, dy, dz);
                            uint8_t m = w.getVoxel(n.x, n.y, n.z);
                            if (isFoliage(m)) {
                                if (!job.idxOf.count(voxKey(n)) &&
                                    job.queued.insert(voxKey(n)).second)
                                    job.frontier.push_back(
                                        glm::ivec4(n, pd.w + 1));
                            } else if (isCutFace(m) || isCactusSolid(m)) {
                                // Cut faces are rooted by construction;
                                // cactus flesh supports its blooms.
                                supported = true;
                            } else if (isWoodMat(m)) {
                                // A FRESH stub flood is real work: bill its
                                // component reads (cached verdicts are free).
                                size_t pre = job.woodVerdict.size();
                                if (woodSupports(job, w, n)) supported = true;
                                budget -= 27 * int(job.woodVerdict.size() - pre);
                            }
                        }
            }
            if (supported) {
                job.dist[idx] = 1;
                job.bfs.push_back(idx);
            }
            if (int(job.voxels.size()) > DECAY_REGION_CAP) {
                std::printf("[forestry] decay aborted: leaf region > %d\n",
                            DECAY_REGION_CAP);
                f.decayJobs.pop_front();
                return;
            }
        }
        if (job.seeds.empty() && job.frontier.empty()) job.phase = 1;
        return;
    }

    if (job.phase == 1) {
        // Relax: multi-source BFS from the supported seeds.
        int budget = DECAY_RELAX_BUDGET;
        while (budget > 0 && !job.bfs.empty()) {
            int i = job.bfs.front(); job.bfs.pop_front();
            --budget;
            int dnext = job.dist[i] + 1;
            if (dnext > LEAF_SUPPORT_N) continue;
            glm::ivec3 v = job.voxels[i];
            for (int dz = -1; dz <= 1; ++dz)
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (!dx && !dy && !dz) continue;
                        auto it = job.idxOf.find(voxKey(v + glm::ivec3(dx, dy, dz)));
                        if (it == job.idxOf.end() || job.dist[it->second] >= 0)
                            continue;
                        job.dist[it->second] = int16_t(dnext);
                        job.bfs.push_back(it->second);
                    }
        }
        if (job.bfs.empty()) job.phase = 2;
        return;
    }

    // Phase 2: crumble everything unsupported (plus floating wood stubs).
    // Budgeted so a mega-canopy dissolves as a wave, not a hitch.
    EditDelta d;
    bool touched = false;
    int budget = DECAY_REMOVE_BUDGET;
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    while (budget > 0 && job.cursorW < job.deadWood.size()) {
        glm::ivec3 v = job.deadWood[job.cursorW++];
        --budget;
        uint8_t m = w.getVoxel(v.x, v.y, v.z);
        if (!isWoodMat(m)) continue;
        w.setVoxel(v.x, v.y, v.z, MAT_AIR, d);
        touched = true;
        glm::vec3 p = glm::vec3(v) + 0.5f;
        if (m == MAT_CHAR)                ps.spawnDust(p, MAT_CHAR, 2);
        else if ((job.cursorW & 7) == 0)  spawnItem(f, w, p, m,
                       glm::vec3(u(f.rng) * 4.0f, 8.0f + u(f.rng) * 4.0f,
                                 u(f.rng) * 4.0f));
        else if ((job.cursorW & 3) == 0)  ps.spawnDust(p, m, 2);
    }
    while (budget > 0 && job.cursor < job.voxels.size()) {
        size_t i = job.cursor++;
        --budget;
        if (job.dist[i] >= 0 && job.dist[i] <= LEAF_SUPPORT_N) continue;
        glm::ivec3 v = job.voxels[i];
        uint8_t fm = w.getVoxel(v.x, v.y, v.z);
        if (!isFoliage(fm)) continue;
        w.setVoxel(v.x, v.y, v.z, MAT_AIR, d);
        touched = true;
        ++job.removed;
        glm::vec3 p = glm::vec3(v) + 0.5f;
        // Decay is ambience, shatter is loot: throttle its items so a decay
        // wave can't starve a concurrent fell's wood drops out of ITEM_CAP.
        if (job.removed % 24 == 0 && int(f.items.size()) < 260)
            spawnItem(f, w, p, fm,
                      glm::vec3(u(f.rng) * 4.0f, 6.0f + u(f.rng) * 4.0f,
                                u(f.rng) * 4.0f));
        else if (job.removed % 5 == 0)
            ps.spawnDust(p, fm, 2);
    }
    if (touched) w.uploadDelta(d);
    if (job.cursor >= job.voxels.size() && job.cursorW >= job.deadWood.size()) {
        if (job.removed > 0)
            std::printf("[forestry] decay: %d unsupported leaves crumbled\n",
                        job.removed);
        // A dissolved stub may itself have been the sole support of foliage
        // (or of wood above it) OUTSIDE this job's gathered region — chain a
        // follow-up sweep seeded from the removed wood. Terminates: every
        // pass strictly removes voxels.
        std::vector<glm::ivec3> chain = std::move(job.deadWood);
        f.decayJobs.pop_front();
        if (!chain.empty()) forestryQueueLeafDecay(f, chain);
    }
}

// ---------------------------------------------------------------------------
// Physics tick
// ---------------------------------------------------------------------------

static glm::mat3 fallRot(const FallingTree& ft) {
    glm::vec3 axis = glm::cross(glm::vec3(0, 1, 0), ft.dir);
    float len = glm::length(axis);
    if (len < 1e-4f) return glm::mat3(1.0f);
    return glm::mat3_cast(glm::angleAxis(ft.angle, axis / len));
}

void forestryCompactAtlas(Forestry& f, const std::vector<uint32_t>& atlasBase,
                          std::vector<uint32_t>& atlasDyn) {
    VOX_ZONE_N("forestry: compact atlas");
    atlasDyn = atlasBase;
    for (auto& ft : f.falls) {
        // appendModelToAtlas rebases wordBase in place; restore file-local
        // bases first or a re-appended live fall double-offsets.
        for (size_t i = 0; i < ft.model.parts.size(); ++i)
            ft.model.parts[i].wordBase = ft.localWordBase[i];
        appendModelToAtlas(ft.model, atlasDyn);
    }
}

bool forestryTick(Forestry& f, const World& w, glm::vec3 playerChest,
                  ParticleSystem& ps, float dt) {
    VOX_ZONE_N("forestry: tick");
    dt = std::min(dt, 0.05f);   // physics never sees a hitch-sized step

    // -- falling trees -------------------------------------------------------
    // Substepped hinge integration with whole-model contact probes: the tree
    // crumbles the moment ANY sampled voxel (trunk or crown) sweeps into
    // solid ground — never visibly clipping in first. Leaves, grass and
    // flowers give way (the crown crashes through a neighbour's canopy
    // instead of shattering mid-air).
    bool retired = false;
    for (auto& ft : f.falls) {
        if (ft.done) continue;
        bool impact = false;
        float remaining = dt;
        while (remaining > 1e-5f && !impact) {
            // Cap each substep so the swept angle stays under ~0.045 rad —
            // a probe can't tunnel through the ground between checks.
            float sub = std::min(remaining, 0.016f);
            if (ft.angVel > 0.8f) sub = std::min(sub, 0.045f / ft.angVel);
            ft.angVel += 5.5f * std::sin(std::max(ft.angle, 0.05f)) * sub;
            ft.angle += ft.angVel * sub;
            remaining -= sub;
            if (ft.angle >= 1.5707f) {   // flat: nothing left to fall onto
                ft.angle = 1.5707f;
                impact = true;
                break;
            }
            if (ft.angle < 0.12f) continue;   // still separating from the stump
            glm::mat3 R = fallRot(ft);
            for (auto& lp : ft.probes) {
                glm::vec3 p = ft.pivot + R * lp;
                uint8_t v = w.getVoxel(int(std::floor(p.x)),
                                       int(std::floor(p.y)),
                                       int(std::floor(p.z)));
                if (!isFoliage(v) && isBlockingForPlayer(v)) {
                    impact = true;
                    break;
                }
            }
        }
        if (impact) {
            shatterIntoItems(f, w, ps, ft.pivot, fallRot(ft), ft.fellWood,
                             ft.fellMats.data(), ft.fellLeaf,
                             ft.fellLeafMats.empty() ? nullptr
                                                     : ft.fellLeafMats.data());
            ft.done = true;
            retired = true;
        }
    }
    if (retired)
        f.falls.erase(std::remove_if(f.falls.begin(), f.falls.end(),
                                     [](const FallingTree& t) { return t.done; }),
                      f.falls.end());

    // -- debris items --------------------------------------------------------
    for (size_t i = 0; i < f.items.size();) {
        DebrisItem& it = f.items[i];
        it.age += dt;
        bool gone = it.age > ITEM_LIFE;

        if (!gone && it.state != 2 &&
            (it.age >= it.magAt ||
             glm::distance(it.pos, playerChest) < MAGNET_RADIUS))
            it.state = 2;

        if (!gone) switch (it.state) {
            case 0: {   // airborne: gravity + axis-separated world collision
                it.vel.y = std::max(it.vel.y - GRAVITY * dt, -GRAVITY_TERMINAL);
                // Substepped per-axis move (<= 1 voxel per probe): a terminal-
                // velocity chunk covers ~16 voxels a tick and would tunnel
                // through thin canopies with an endpoint-only test.
                auto moveAxis = [&](int axis) {
                    float dlt = it.vel[axis] * dt;
                    int steps = std::max(1, int(std::ceil(std::abs(dlt))));
                    float step = dlt / float(steps);
                    for (int st = 0; st < steps; ++st) {
                        glm::vec3 np = it.pos;
                        np[axis] += step;
                        if (solidAt(w, np)) return false;
                        it.pos = np;
                    }
                    return true;
                };
                if (!moveAxis(0)) it.vel.x *= -0.4f;
                if (!moveAxis(2)) it.vel.z *= -0.4f;
                if (!moveAxis(1)) {
                    if (it.vel.y < -26.0f) {           // lively bounce
                        it.vel.y *= -0.35f;
                        it.vel.x *= 0.6f;
                        it.vel.z *= 0.6f;
                    } else if (it.vel.y < 0.0f) {      // came to rest: hover
                        it.state = 1;
                        it.hoverY = it.pos.y;
                        it.vel = glm::vec3(0);
                    } else {
                        it.vel.y = 0.0f;               // bonked a ceiling
                    }
                }
                if (it.pos.y < -64.0f) gone = true;    // fell out of the world
                break;
            }
            case 1: {   // settled: hover-bob + slow spin (rendered from spin/age)
                it.pos.y = it.hoverY + 0.35f * std::sin(it.age * 2.2f + it.spin);
                if (!solidAt(w, {it.pos.x, it.hoverY - 1.4f, it.pos.z}))
                    it.state = 0;                      // ground dug away: fall
                break;
            }
            case 2: {   // magnetized: home in, no collision, absorb on contact
                glm::vec3 dvec = playerChest - it.pos;
                float dist = glm::length(dvec);
                if (dist < ABSORB_RADIUS) {
                    ps.spawnDust(it.pos, it.mat, 2);
                    ++f.collected[it.mat];
                    gone = true;
                    break;
                }
                float sp = std::min(95.0f, glm::length(it.vel) + 480.0f * dt);
                it.vel = dvec / dist * sp;
                it.pos += it.vel * dt;
                break;
            }
        }

        if (gone) {
            f.items[i] = f.items.back();
            f.items.pop_back();
        } else {
            ++i;
        }
    }
    return retired;
}

// ---------------------------------------------------------------------------
// Render-list building
// ---------------------------------------------------------------------------

static void setRows(CreaturePartInstance& inst, const glm::mat3& M, glm::vec3 T) {
    // world = M * cell + T; rows store the matrix ROWS.
    inst.rowX = glm::vec4(M[0][0], M[1][0], M[2][0], T.x);
    inst.rowY = glm::vec4(M[0][1], M[1][1], M[2][1], T.y);
    inst.rowZ = glm::vec4(M[0][2], M[1][2], M[2][2], T.z);
    glm::mat3 Mi = glm::inverse(M);
    glm::vec3 Ti = -(Mi * T);
    inst.invX = glm::vec4(Mi[0][0], Mi[1][0], Mi[2][0], Ti.x);
    inst.invY = glm::vec4(Mi[0][1], Mi[1][1], Mi[2][1], Ti.y);
    inst.invZ = glm::vec4(Mi[0][2], Mi[1][2], Mi[2][2], Ti.z);
    inst.prevX = inst.rowX; inst.prevY = inst.rowY; inst.prevZ = inst.rowZ;
    inst.anchor = glm::vec4(0.0f);   // palette mode default; falls overwrite
}

int appendFallInstances(Forestry& f, std::vector<CreaturePartInstance>& parts,
                        CreatureBound* bounds, int maxBounds) {
    int nb = 0;
    for (auto& ft : f.falls) {
        if (nb >= maxBounds) break;
        int first = int(parts.size());
        glm::mat3 R = fallRot(ft);
        for (auto& p : ft.model.parts) {
            CreaturePartInstance inst;
            setRows(inst, R, R * p.gridOrigin + ft.pivot);
            inst.dimsPal = glm::vec4(glm::vec3(p.dims), float(ft.model.palBase));
            // meta.y = 1: terrain-material volume — the bytes are world
            // material ids and the lighting pass shades them like standing
            // voxels (no step change at the fell). See creature.frag.
            inst.meta = glm::vec4(float(p.wordBase), 1.0f, 0, 0);
            // The part's PRE-FALL world lo corner: the lighting pass anchors
            // the painterly per-voxel detail to anchor + cell, so the
            // pattern rides the rotating model rigidly (and matches the
            // standing tree exactly) instead of boiling in world space.
            inst.anchor = glm::vec4(glm::round(p.gridOrigin + ft.pivot), 0.0f);
            parts.push_back(inst);
        }
        int count = int(parts.size()) - first;
        // Real motion vectors: stamp last frame's rows, refresh the cache.
        bool have = ft.prevXf.size() == size_t(count) * 12;
        if (!have) ft.prevXf.resize(size_t(count) * 12);
        for (int i = 0; i < count; ++i) {
            CreaturePartInstance& inst = parts[size_t(first) + i];
            float* c = ft.prevXf.data() + size_t(i) * 12;
            if (have) {
                inst.prevX = glm::vec4(c[0], c[1], c[2], c[3]);
                inst.prevY = glm::vec4(c[4], c[5], c[6], c[7]);
                inst.prevZ = glm::vec4(c[8], c[9], c[10], c[11]);
            }
            c[0] = inst.rowX.x; c[1] = inst.rowX.y; c[2]  = inst.rowX.z; c[3]  = inst.rowX.w;
            c[4] = inst.rowY.x; c[5] = inst.rowY.y; c[6]  = inst.rowY.z; c[7]  = inst.rowY.w;
            c[8] = inst.rowZ.x; c[9] = inst.rowZ.y; c[10] = inst.rowZ.z; c[11] = inst.rowZ.w;
        }
        bounds[nb].center = ft.pivot + R * ft.boundCentroid;
        bounds[nb].radius = ft.boundRadius;
        bounds[nb].firstPart = first;
        bounds[nb].partCount = count;
        ++nb;
    }
    return nb;
}

void appendItemInstances(Forestry& f, std::vector<CreaturePartInstance>& parts) {
    for (auto& it : f.items) {
        const CreatureModel& m =
            (it.mat == MAT_LEAVES || it.mat == MAT_SHRUB_SAGE) ? f.leafChunk
          : (it.mat == MAT_CACTUS_FLESH || it.mat == MAT_CACTUS_BLOOM)
              ? f.cactusChunk
          : (it.mat == MAT_STONE) ? f.stoneChunk
                                  : f.woodChunk;
        const CreaturePart& p = m.parts[0];
        float rate = (it.state == 2) ? 9.0f : (it.state == 0) ? 6.0f : 1.6f;
        float a = it.spin + it.age * rate;
        auto itemRows = [&](glm::vec3 pos, float ang, glm::mat3& M, glm::vec3& T) {
            float cy = std::cos(ang), sy = std::sin(ang);
            glm::mat3 Ry(glm::vec3(cy, 0, -sy), glm::vec3(0, 1, 0),
                         glm::vec3(sy, 0, cy));
            M = Ry * m.cellScale;
            T = pos - M * (glm::vec3(p.dims) * 0.5f);
        };
        glm::mat3 M; glm::vec3 T;
        itemRows(it.pos, a, M, T);
        CreaturePartInstance inst;
        setRows(inst, M, T);
        // Real motion vectors: rebuild LAST frame's transform (magnetized
        // items are the fastest movers on screen — zero-motion prev rows
        // would smear them under the map TAA).
        if (it.hasPrev) {
            glm::mat3 Mp; glm::vec3 Tp;
            itemRows(it.prevPos, it.prevA, Mp, Tp);
            inst.prevX = glm::vec4(Mp[0][0], Mp[1][0], Mp[2][0], Tp.x);
            inst.prevY = glm::vec4(Mp[0][1], Mp[1][1], Mp[2][1], Tp.y);
            inst.prevZ = glm::vec4(Mp[0][2], Mp[1][2], Mp[2][2], Tp.z);
        }
        it.prevPos = it.pos;
        it.prevA = a;
        it.hasPrev = true;
        inst.dimsPal = glm::vec4(glm::vec3(p.dims), float(m.palBase));
        inst.meta = glm::vec4(float(p.wordBase), 0, 0, 0);
        parts.push_back(inst);
    }
}

// Shared attach math for hand tools. Tool basis in forearm-cell space: the
// handle continues past the fist (hand-local -y, the hand being the bottom
// of the lower-arm box) so an overhead windup points the head up and the
// strike drives it forward; the business face points creature-forward (+z).
static CreaturePartInstance makeAttachedTool(const CreatureModel& tool,
                                             glm::vec3 grip,
                                             const CreaturePartInstance& hand,
                                             glm::ivec3 handDims) {
    // Hand instance rows -> matrix + translation (includes the avatar's
    // cellScale, so the tool renders at player scale automatically).
    glm::mat3 A;
    A[0] = glm::vec3(hand.rowX.x, hand.rowY.x, hand.rowZ.x);
    A[1] = glm::vec3(hand.rowX.y, hand.rowY.y, hand.rowZ.y);
    A[2] = glm::vec3(hand.rowX.z, hand.rowY.z, hand.rowZ.z);
    glm::vec3 t(hand.rowX.w, hand.rowY.w, hand.rowZ.w);

    glm::mat3 Mloc(glm::vec3(0, 0, 1),    // tool +x -> hand +z
                   glm::vec3(0, -1, 0),   // tool +y (handle/head) -> hand -y
                   glm::vec3(1, 0, 0));   // right-handed completion
    glm::mat3 R = A * Mloc;

    const CreaturePart& ap = tool.parts[0];
    glm::vec3 attach(handDims.x * 0.5f, 0.6f, handDims.z * 0.5f);  // the fist
    glm::vec3 T = (A * attach + t) - R * grip;

    CreaturePartInstance inst;
    setRows(inst, R, T);
    // Motion vectors: the hand instance carries ITS last-frame rows — compose
    // the tool's previous transform from them the same way (a mid-swing tool
    // head moves half a metre a frame; zero motion would ghost it).
    {
        glm::mat3 Ap;
        Ap[0] = glm::vec3(hand.prevX.x, hand.prevY.x, hand.prevZ.x);
        Ap[1] = glm::vec3(hand.prevX.y, hand.prevY.y, hand.prevZ.y);
        Ap[2] = glm::vec3(hand.prevX.z, hand.prevY.z, hand.prevZ.z);
        glm::vec3 tp(hand.prevX.w, hand.prevY.w, hand.prevZ.w);
        glm::mat3 Rp = Ap * Mloc;
        glm::vec3 Tp = (Ap * attach + tp) - Rp * grip;
        inst.prevX = glm::vec4(Rp[0][0], Rp[1][0], Rp[2][0], Tp.x);
        inst.prevY = glm::vec4(Rp[0][1], Rp[1][1], Rp[2][1], Tp.y);
        inst.prevZ = glm::vec4(Rp[0][2], Rp[1][2], Rp[2][2], Tp.z);
    }
    inst.dimsPal = glm::vec4(glm::vec3(ap.dims), float(tool.palBase));
    inst.meta = glm::vec4(float(ap.wordBase), 0, 0, 0);
    return inst;
}

CreaturePartInstance makeAttachedAxe(const Forestry& f,
                                     const CreaturePartInstance& hand,
                                     glm::ivec3 handDims) {
    return makeAttachedTool(f.axeModel, {1.5f, 1.2f, 0.5f}, hand, handDims);
}

CreaturePartInstance makeAttachedPick(const Forestry& f,
                                      const CreaturePartInstance& hand,
                                      glm::ivec3 handDims) {
    return makeAttachedTool(f.pickModel, {3.5f, 1.2f, 0.5f}, hand, handDims);
}
