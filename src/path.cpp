#include "path.h"
#include "world.h"
#include "player.h"

#include <cmath>
#include <cstdint>
#include <queue>
#include <unordered_map>

// Tunables. MAX_POPS caps the search so a click across a maze of trunks
// degrades to "walk as far as possible toward it" instead of a frame hitch;
// the 1.15 heuristic weight keeps open-field searches expanding in a tight
// corridor along the line (bounded suboptimality — routes may be up to 15%
// longer than perfect, never wrong).
static constexpr int   MAX_POPS   = 8000;
static constexpr float H_WEIGHT   = 1.15f;
static constexpr int   MAX_DROP   = 8;      // walkable step DOWN (fall a bit)
static constexpr int   HEADROOM   = 18;     // ceil(P_HEIGHT)
static constexpr int   GOAL_YBAND = 12;     // arrive only near the clicked LEVEL

static inline int64_t colKey(int x, int z) {
    return (int64_t(x) << 32) ^ (int64_t(z) & 0xFFFFFFFFll);
}

// How a column is traversed: on foot (feetY on a floor) or swimming at the
// approach level (no floor within drop reach, but water to float in).
struct ColHit {
    int  feetY = INT32_MIN;
    bool swim  = false;
    bool ok() const { return feetY != INT32_MIN; }
};

// The body is 5x5 COLUMNS wide (P_RADIUS=2.0; overlapsSolid scans feet+-2),
// not one: a floor is standable only if the torso band is also clear on the
// 8 ring columns at +-2. The band starts at +4 so step-height terrain lips
// under the body don't read as walls (auto-step absorbs those); worldgen
// obstacles (trunks radius >= 2, boulders) are far taller than 4 and are
// caught. This is what keeps A* from threading gaps the body cannot fit.
static const int RING_X[8] = {2, -2, 0, 0, 2, 2, -2, -2};
static const int RING_Z[8] = {0, 0, 2, -2, 2, -2, 2, -2};

static bool bodyClear(const World& w, int x, int z, int g, int headroom) {
    for (int i = 1; i <= headroom; ++i)
        if (isBlockingForPlayer(w.getVoxel(x, g + i, z))) return false;
    for (int r = 0; r < 8; ++r) {
        int rx = x + RING_X[r], rz = z + RING_Z[r];
        if (!w.voxelInWindow(rx, rz)) return false;
        for (int i = 4; i <= headroom; ++i)
            if (isBlockingForPlayer(w.getVoxel(rx, g + i, rz))) return false;
    }
    return true;
}

// Traversal for column (x,z) approached with feet at refY. On-foot floors
// prefer the level closest to refY (paths hug terrain); drops must clear the
// DEPARTURE height too (headroom + k) or the body wedges under the overhang
// on the way down. When no floor is reachable but the approach level is in
// water, the column is swimmable at that level.
static ColHit resolveMove(const World& w, int x, int z, int refY) {
    if (!w.voxelInWindow(x, z)) return {};
    auto walkableAt = [&](int g, int headroom) {
        if (g < 0 || g + headroom + 1 >= GRID_Y) return false;
        if (!isBlockingForPlayer(w.getVoxel(x, g, z))) return false;
        return bodyClear(w, x, z, g, headroom);
    };
    int base = refY - 1;   // the floor under the approach feet
    for (int k = 0; k <= MAX_DROP; ++k) {
        if (k <= int(P_STEP) && walkableAt(base + k, HEADROOM))
            return {base + k + 1, false};
        if (k > 0 && walkableAt(base - k, HEADROOM + k))
            return {base - k + 1, false};
    }
    // Swim: float across at the approach level (deep water has no floor
    // within MAX_DROP). Water is non-blocking, so bodyClear still verifies
    // the air/water band and the ring for over-water branches and cliffs.
    int wy = std::max(refY - 1, 0);
    if (w.getVoxel(x, wy, z) == MAT_WATER &&
        bodyClear(w, x, z, wy, HEADROOM))
        return {refY, true};
    return {};
}

// Memoized wrapper: A* touches each column up to 8 times as a neighbor and
// the ring test is ~130 voxel reads — cache per (column, refY band).
using ColCache = std::unordered_map<uint64_t, ColHit>;
static ColHit resolveCached(const World& w, ColCache& cache,
                            int x, int z, int refY) {
    uint64_t key = uint64_t(colKey(x, z)) * 0x9E3779B97F4A7C15ull
                 ^ uint64_t(uint32_t(refY >> 2));
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    ColHit h = resolveMove(w, x, z, refY);
    cache.emplace(key, h);
    return h;
}

// Walkability line test for smoothing: supercover-ish — sample at 0.5-voxel
// spacing so no column the segment enters is skipped, and when a step
// crosses BOTH an x and a z column boundary, require the two orthogonal
// corner columns too (the same no-corner-cut rule A* uses).
static bool walkLine(const World& w, ColCache& cache, glm::vec3 a, glm::vec3 b) {
    glm::vec2 d(b.x - a.x, b.z - a.z);
    float len = glm::length(d);
    if (len < 1e-3f) return true;
    glm::vec2 stepDir = d / len;
    int   n    = int(std::ceil(len * 2.0f));
    float refY = a.y;
    int   px = int(std::floor(a.x)), pz = int(std::floor(a.z));
    for (int i = 1; i <= n; ++i) {
        float t = std::min(float(i) * 0.5f, len);
        glm::vec2 q(a.x + stepDir.x * t, a.z + stepDir.y * t);
        int cx = int(std::floor(q.x)), cz = int(std::floor(q.y));
        if (cx == px && cz == pz) continue;
        if (cx != px && cz != pz) {          // diagonal transition
            if (!resolveCached(w, cache, px, cz, int(std::round(refY))).ok() ||
                !resolveCached(w, cache, cx, pz, int(std::round(refY))).ok())
                return false;
        }
        ColHit h = resolveCached(w, cache, cx, cz, int(std::round(refY)));
        if (!h.ok()) return false;
        refY = float(h.feetY);
        px = cx; pz = cz;
    }
    return true;
}

bool findPath(const World& w, glm::vec3 from, glm::vec3 to,
              std::vector<glm::vec3>& out) {
    out.clear();

    struct Node { float g; int fromX, fromZ, feetY; bool swim, closed; };
    std::unordered_map<int64_t, Node> nodes;
    ColCache cache;
    // Open list with lazy deletion: stale entries are skipped when their
    // recorded g no longer matches the node's best.
    struct Open { float f, g; int x, z; };
    struct Cmp { bool operator()(const Open& a, const Open& b) const { return a.f > b.f; } };
    std::priority_queue<Open, std::vector<Open>, Cmp> open;

    int sx = int(std::floor(from.x)), sz = int(std::floor(from.z));
    int tx = int(std::floor(to.x)),   tz = int(std::floor(to.z));
    int ty = int(std::round(to.y));
    ColHit start = resolveMove(w, sx, sz, int(std::round(from.y)));
    int sy = start.ok() ? start.feetY : int(std::round(from.y));
    auto heur = [&](int x, int z) {
        float dx = float(tx - x), dz = float(tz - z);
        return std::sqrt(dx * dx + dz * dz);
    };

    nodes[colKey(sx, sz)] = {0.0f, sx, sz, sy, start.swim, false};
    open.push({H_WEIGHT * heur(sx, sz), 0.0f, sx, sz});

    // Track the best approach so a capped/failed search still walks partway.
    int   bestX = sx, bestZ = sz;
    float bestH = heur(sx, sz);
    bool  reached = false;

    static const int DX[8] = {1, -1, 0, 0, 1, 1, -1, -1};
    static const int DZ[8] = {0, 0, 1, -1, 1, -1, 1, -1};

    for (int pops = 0; pops < MAX_POPS && !open.empty(); ++pops) {
        Open cur = open.top(); open.pop();
        auto& cnRef = nodes[colKey(cur.x, cur.z)];
        if (cnRef.closed || cur.g > cnRef.g + 1e-4f) continue;   // stale entry
        cnRef.closed = true;
        // Copy before expanding: inserting neighbors can rehash the map and
        // dangle any reference into it.
        Node cn = cnRef;

        float h = heur(cur.x, cur.z);
        if (h < bestH) { bestH = h; bestX = cur.x; bestZ = cur.z; }
        // Arrive only near the clicked LEVEL: the surface directly over an
        // X-ray cave click is the wrong floor — that's a close approach,
        // not a success.
        if (cur.x == tx && cur.z == tz &&
            std::abs(cn.feetY - ty) <= GOAL_YBAND) { reached = true; break; }

        // Resolve the 4 orthogonal neighbors once; diagonals reuse them for
        // the corner-cut rule instead of re-resolving.
        ColHit ortho[4];
        for (int i = 0; i < 4; ++i)
            ortho[i] = resolveCached(w, cache, cur.x + DX[i], cur.z + DZ[i],
                                     cn.feetY);
        for (int i = 0; i < 8; ++i) {
            int nx = cur.x + DX[i], nz = cur.z + DZ[i];
            ColHit nh;
            if (i < 4) {
                nh = ortho[i];
            } else {
                // Diagonal: both flanking orthogonals must pass (a
                // wider-than-one-voxel body cannot cut the corner block).
                int ox = (DX[i] > 0) ? 0 : 1;        // matching +-x ortho
                int oz = (DZ[i] > 0) ? 2 : 3;        // matching +-z ortho
                if (!ortho[ox].ok() || !ortho[oz].ok()) continue;
                nh = resolveCached(w, cache, nx, nz, cn.feetY);
            }
            if (!nh.ok()) continue;
            int dy = nh.feetY - cn.feetY;
            float cost = (i < 4) ? 1.0f : 1.41421356f;
            if (dy > 0) cost += 0.45f * float(dy);          // climbing is work
            else if (dy < 0) cost += 0.10f * float(-dy);    // dropping, mildly
            if (nh.swim) cost *= 1.8f;                      // swimming is slow

            float g2 = cn.g + cost;
            auto it = nodes.find(colKey(nx, nz));
            if (it == nodes.end() || (!it->second.closed && g2 < it->second.g - 1e-4f)) {
                nodes[colKey(nx, nz)] = {g2, cur.x, cur.z, nh.feetY, nh.swim, false};
                open.push({g2 + H_WEIGHT * heur(nx, nz), g2, nx, nz});
            }
        }
    }

    // Reconstruct to the target if reached, else to the closest approach.
    int cx = reached ? tx : bestX, cz = reached ? tz : bestZ;
    if (cx == sx && cz == sz) return false;   // nowhere to go at all
    std::vector<glm::vec3> rev;
    while (!(cx == sx && cz == sz)) {
        const Node& n = nodes.at(colKey(cx, cz));
        rev.push_back(glm::vec3(float(cx) + 0.5f, float(n.feetY), float(cz) + 0.5f));
        cx = n.fromX; cz = n.fromZ;
        if ((int)rev.size() > MAX_POPS) break;   // paranoia vs parent cycles
    }

    // Line-of-sight smoothing: keep the farthest waypoint each anchor can
    // walk to in a straight line — open fields collapse to a handful of
    // waypoints, and steering stops zig-zagging on voxel-grid diagonals.
    // Chords are also capped by Y drift: a straight line over rolling
    // terrain must not arc far from the walked ground (the trail is drawn
    // against the chord, and steering Y is meaningless mid-chord).
    std::vector<glm::vec3> fwd(rev.rbegin(), rev.rend());
    glm::vec3 anchor(float(sx) + 0.5f, float(sy), float(sz) + 0.5f);
    size_t i = 0;
    while (i < fwd.size()) {
        size_t j = std::min(i + 48, fwd.size() - 1);   // bounded lookahead
        size_t keep = i;
        for (; j > i; --j) {
            if (std::abs(fwd[j].y - anchor.y) > 12.0f) continue;
            if (walkLine(w, cache, anchor, fwd[j])) { keep = j; break; }
        }
        out.push_back(fwd[keep]);
        anchor = fwd[keep];
        i = keep + 1;
    }
    return !out.empty();
}
