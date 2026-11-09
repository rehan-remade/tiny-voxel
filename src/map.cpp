#include "map.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

// Reveal tuning. Surface reveal is a wide proximity disc (you can see far in
// the open); underground reveal is a tight sphere (lantern reach).
constexpr int   SURF_R_CHUNKS = 10;    // ~64 m radius of surface fill
constexpr int   UG_R          = 40;    // ~4 m underground reveal sphere
constexpr float CAVE_DEPTH    = 25.0f; // this deep below the surface = in a cave
constexpr float MARK_PERIOD   = 0.25f; // seconds between marking sweeps

constexpr char     MAGIC[4] = {'T', 'V', 'X', 'M'};   // M = map
constexpr uint32_t VERSION  = 1u;

#pragma pack(push, 1)
struct Header {
    char     magic[4];
    uint32_t version;
    uint64_t worldSeed;   // exploration only applies to the world it happened in
    uint32_t chunkVox;    // CHUNK_VOX at save time
    uint32_t ysliceVox;   // MapExplored::YSLICE_VOX at save time
    uint32_t ysliceCount; // MapExplored::YSLICES — pins ChunkRec's on-disk size
                          // (a GRID_Y change would otherwise misparse old files)
    uint32_t chunkCount;
};
struct ChunkRec {
    int32_t  ccx, ccz;
    uint8_t  surfaceSeen;
    uint32_t ymask[MapExplored::YSLICE_U32];
};
#pragma pack(pop)

} // namespace

void MapExplored::update(const glm::vec3& eye, float surfH, float dt) {
    markAcc += dt;
    if (markAcc < MARK_PERIOD) return;
    markAcc = 0.0f;

    int px = int(std::floor(eye.x));
    int py = int(std::floor(eye.y));
    int pz = int(std::floor(eye.z));
    bool changed = false;

    // Surface fill: only while sky-plausible (not in a cave), so tunnelling
    // under unexplored terrain never reveals its surface.
    if (surfH - eye.y < CAVE_DEPTH) {
        int pcx = chunkOfVoxel(px), pcz = chunkOfVoxel(pz);
        for (int dz = -SURF_R_CHUNKS; dz <= SURF_R_CHUNKS; ++dz)
        for (int dx = -SURF_R_CHUNKS; dx <= SURF_R_CHUNKS; ++dx) {
            if (dx * dx + dz * dz > SURF_R_CHUNKS * SURF_R_CHUNKS) continue;
            Entry& e = chunks[chunkKey(pcx + dx, pcz + dz)];
            if (!e.surfaceSeen) { e.surfaceSeen = 1; changed = true; }
        }
    }

    // Underground reveal: mark the visited 16-voxel Y slices of every chunk
    // column the reveal sphere touches. Runs on the surface too (walking
    // reveals the first few voxels of ground under your path — Terraria-ish),
    // where the marked above-ground slices are simply never consulted.
    int s0 = std::clamp((py - UG_R) >> YSLICE_BITS, 0, YSLICES - 1);
    int s1 = std::clamp((py + UG_R) >> YSLICE_BITS, 0, YSLICES - 1);
    int c0x = chunkOfVoxel(px - UG_R), c1x = chunkOfVoxel(px + UG_R);
    int c0z = chunkOfVoxel(pz - UG_R), c1z = chunkOfVoxel(pz + UG_R);
    for (int cz = c0z; cz <= c1z; ++cz)
    for (int cx = c0x; cx <= c1x; ++cx) {
        Entry& e = chunks[chunkKey(cx, cz)];
        for (int s = s0; s <= s1; ++s) {
            uint32_t& w   = e.ymask[s >> 5];
            uint32_t  bit = 1u << (s & 31);
            if (!(w & bit)) {
                w |= bit;
                changed = true;
                minSliceSeen = std::min(minSliceSeen, s);
            }
        }
    }

    if (changed) { ++version; dirtySinceSave = true; }
}

void MapExplored::buildMasks(glm::ivec2 maskOriginChunk, MapMasks& out) const {
    // Inverted loop vs the old window-local build: iterate the (sparse)
    // exploration record and scatter into the mask, instead of probing the
    // hashmap once per mask cell — the mask is now MASK_CH^2 = 36k cells.
    std::fill(out.flags.begin(), out.flags.end(), uint8_t(0));
    std::fill(out.ymask.begin(), out.ymask.end(), 0u);
    constexpr int N = MapMasks::MASK_CH;
    for (const auto& kv : chunks) {
        int lx = chunkKeyX(kv.first) - maskOriginChunk.x;
        int lz = chunkKeyZ(kv.first) - maskOriginChunk.y;
        if (lx < 0 || lx >= N || lz < 0 || lz >= N) continue;
        const Entry& e = kv.second;
        bool any = e.surfaceSeen != 0;
        for (int w = 0; w < YSLICE_U32; ++w) {
            out.ymask[size_t(lx) + size_t(N) * (size_t(w) + YSLICE_U32 * size_t(lz))]
                = e.ymask[w];
            any = any || e.ymask[w] != 0;
        }
        out.flags[size_t(lx) + size_t(N) * size_t(lz)] =
            uint8_t((e.surfaceSeen ? 1u : 0u) | (any ? 2u : 0u));
    }

    // Prefilter the fog-of-war ramp: 5x5 box blur of bit0, separable
    // (horizontal then vertical), out-of-mask = unexplored (0). The shader
    // used to reconstruct this ramp with 100 texelFetches per map pixel;
    // now it is one bilinear tap of this field. The blur radius IS the
    // documented ~5-chunk frontier ramp — change them together.
    std::vector<uint8_t> rowSum(size_t(N) * N, 0);
    for (int z = 0; z < N; ++z)
        for (int x = 0; x < N; ++x) {
            int s = 0;
            for (int dx = -2; dx <= 2; ++dx) {
                int xx = x + dx;
                if (xx >= 0 && xx < N)
                    s += out.flags[size_t(xx) + size_t(N) * size_t(z)] & 1;
            }
            rowSum[size_t(x) + size_t(N) * size_t(z)] = uint8_t(s);
        }
    for (int z = 0; z < N; ++z)
        for (int x = 0; x < N; ++x) {
            int s = 0;
            for (int dz = -2; dz <= 2; ++dz) {
                int zz = z + dz;
                if (zz >= 0 && zz < N)
                    s += rowSum[size_t(x) + size_t(N) * size_t(zz)];
            }
            out.soft[size_t(x) + size_t(N) * size_t(z)] =
                uint8_t((s * 255 + 12) / 25);
        }
}

bool MapExplored::save(const std::string& path, uint64_t worldSeed) {
    // Stable order (sorted keys) so identical exploration produces identical
    // files — same contract as saveWorldEdits.
    std::vector<int64_t> ids;
    ids.reserve(chunks.size());
    for (const auto& kv : chunks) ids.push_back(kv.first);
    std::sort(ids.begin(), ids.end());

    const std::string tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "[map] cannot open %s for write\n", tmp.c_str());
        return false;
    }
    Header h{};
    std::memcpy(h.magic, MAGIC, 4);
    h.version    = VERSION;
    h.worldSeed   = worldSeed;
    h.chunkVox    = CHUNK_VOX;
    h.ysliceVox   = YSLICE_VOX;
    h.ysliceCount = YSLICES;
    h.chunkCount  = uint32_t(ids.size());
    bool ok = std::fwrite(&h, sizeof(h), 1, f) == 1;
    for (int64_t id : ids) {
        if (!ok) break;
        const Entry& e = chunks.at(id);
        ChunkRec rec{};
        rec.ccx = chunkKeyX(id);
        rec.ccz = chunkKeyZ(id);
        rec.surfaceSeen = e.surfaceSeen;
        std::memcpy(rec.ymask, e.ymask, sizeof rec.ymask);
        ok = std::fwrite(&rec, sizeof(rec), 1, f) == 1;
    }
    // fclose flushes the stdio buffer — a full disk fails HERE, not in the
    // buffered fwrites above; ignoring it would rename a torn file over the
    // good one.
    ok = (std::fclose(f) == 0) && ok;
    if (!ok) { std::remove(tmp.c_str()); return false; }

    std::remove(path.c_str());
    if (std::rename(tmp.c_str(), path.c_str()) != 0) return false;
    dirtySinceSave = false;
    return true;
}

bool MapExplored::load(const std::string& path, uint64_t worldSeed) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    Header h{};
    if (std::fread(&h, sizeof(h), 1, f) != 1 ||
        std::memcmp(h.magic, MAGIC, 4) != 0 || h.version != VERSION ||
        h.chunkVox != uint32_t(CHUNK_VOX) ||
        h.ysliceVox != uint32_t(YSLICE_VOX) ||
        h.ysliceCount != uint32_t(YSLICES)) {
        std::fprintf(stderr, "[map] %s: unrecognized map file, ignoring\n",
                     path.c_str());
        std::fclose(f);
        return false;
    }
    if (h.worldSeed != worldSeed) {
        std::fprintf(stderr, "[map] %s belongs to another world seed, ignoring\n",
                     path.c_str());
        std::fclose(f);
        return false;
    }
    chunks.clear();
    minSliceSeen = YSLICES;
    uint32_t got = 0;
    for (uint32_t i = 0; i < h.chunkCount; ++i) {
        ChunkRec rec{};
        if (std::fread(&rec, sizeof(rec), 1, f) != 1) break;
        ++got;
        Entry e;
        e.surfaceSeen = rec.surfaceSeen;
        std::memcpy(e.ymask, rec.ymask, sizeof e.ymask);
        for (int w = 0; w < YSLICE_U32; ++w) {
            if (!e.ymask[w]) continue;
            for (int b = 0; b < 32; ++b)
                if (e.ymask[w] & (1u << b))
                    minSliceSeen = std::min(minSliceSeen, w * 32 + b);
        }
        chunks[chunkKey(rec.ccx, rec.ccz)] = e;
    }
    std::fclose(f);
    if (got != h.chunkCount)
        std::fprintf(stderr, "[map] %s: truncated (read %u of %u records); "
                             "keeping the partial coverage\n",
                     path.c_str(), got, h.chunkCount);
    ++version;   // masks are stale relative to whatever was built before
    return !chunks.empty();
}

void MapView::pan(float dx, float dy, float viewportH) {
    // Drag the world with the cursor: the grabbed point stays under it. The
    // pixel->world scale is the vertical FOV projected to the EASED camera
    // distance (viewDist — plain `dist` would slip during the open settle);
    // up-screen motion lands on the tilted ground plane foreshortened by
    // sin(pitch), so dy gets the inverse (clamped — a near-horizon camera
    // would otherwise send the target flying).
    float s = 2.0f * viewDist() * MAP_TAN_HALF_FOV / std::max(viewportH, 1.0f);
    glm::vec2 fwdXZ(-std::sin(yaw), -std::cos(yaw));   // camera->target, horizontal
    glm::vec2 rightXZ(-fwdXZ.y, fwdXZ.x);
    glm::vec2 d = rightXZ * (-dx * s)
                + fwdXZ  * (dy * s / std::max(std::sin(pitch), 0.35f));
    // Pans accumulate into the look-around OFFSET, not the target: the target
    // is composed as followPos + panOffset every frame (follow()), so a
    // direct write here would be stomped a frame later.
    panOffset.x = glm::clamp(panOffset.x + d.x, -PAN_R, PAN_R);
    panOffset.y = glm::clamp(panOffset.y + d.y, -PAN_R, PAN_R);
}

void MapView::follow(const glm::vec3& playerPos, bool panHeld, float dt) {
    // Critically-damped-ish exponential chase: fast enough that walking never
    // outruns the frame, slow enough that jumps and step-ups read as glides.
    followPos += (playerPos - followPos) * (1.0f - std::exp(-6.0f * dt));
    // Character-locked camera: outside free-cam the offset always eases home
    // (it can only be nonzero right after leaving free-cam). In free-cam the
    // look-around stays parked wherever the grab left it.
    if (!freeCam && !panHeld) {
        panOffset *= std::exp(-2.5f * dt);
        // Snap only once the remainder is sub-pixel at any zoom (a 0.5-voxel
        // snap reads as a visible pop when zoomed close).
        if (glm::dot(panOffset, panOffset) < 0.0025f) panOffset = glm::vec2(0.0f);
    }
    target = followPos + glm::vec3(panOffset.x, 0.0f, panOffset.y);
}

void MapView::camera(glm::vec3& pos, glm::vec3& fwd) const {
    glm::vec3 dir(std::cos(pitch) * std::sin(yaw),
                  std::sin(pitch),
                  std::cos(pitch) * std::cos(yaw));
    pos = target + dir * viewDist();
    fwd = -dir;
}
