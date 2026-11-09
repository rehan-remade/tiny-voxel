#include "creatures.h"
#include "world.h"
#include "player.h"            // isBlockingForPlayer (shared passability)
#include "worldgen/worldgen.h" // terrainHeightAt / forestDensityAt / creatureCellSeed
#include <glm/gtc/quaternion.hpp>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <filesystem>

// ---------------------------------------------------------------------------
// Assets
// ---------------------------------------------------------------------------

int CreatureModel::clipIndex(const char* name) const {
    for (size_t i = 0; i < clips.size(); ++i)
        if (std::strcmp(clips[i].name, name) == 0) return int(i);
    return -1;
}

// .cvx2 rigid-part asset (tools/creatures/bake_parts.py writes it; layout in
// docs/creature-animation-roadmap.md).
static bool loadCvx2(const char* path, CreatureModel& m) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "[creatures] missing %s\n", path); return false; }
    auto fail = [&](const char* why) {
        std::fprintf(stderr, "[creatures] %s: %s\n", path, why);
        std::fclose(f);
        return false;
    };
    char magic[4]; uint32_t ver = 0;
    if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "CVX2", 4) != 0)
        return fail("bad magic (expected CVX2 — re-bake with bake_parts.py)");
    std::fread(&ver, 4, 1, f);
    if (ver != 3) return fail("unsupported version (re-bake with bake_parts.py)");
    uint32_t nParts = 0, nClips = 0, total = 0;
    std::fread(&nParts, 4, 1, f);
    std::fread(&nClips, 4, 1, f);
    std::fread(&total,  4, 1, f);
    std::fread(&m.cellScale, 4, 1, f);
    if (nParts == 0 || nParts > 64 || nClips == 0 || nClips > 64 ||
        total == 0 || total > 65536 ||
        !(m.cellScale > 0.05f && m.cellScale <= 4.0f))
        return fail("implausible header");
    std::fread(m.palette, 4, 256, f);
    m.totalFrames = total;
    m.parts.resize(nParts);
    uint64_t wordsNeeded = 0;
    for (auto& p : m.parts) {
        char name[16]; int32_t parent; float origin[3], pivot[3]; uint32_t dims[3], wb;
        std::fread(name, 1, 16, f);
        std::fread(&parent, 4, 1, f);
        std::fread(origin, 4, 3, f);
        std::fread(pivot,  4, 3, f);
        std::fread(dims,   4, 3, f);
        if (std::fread(&wb, 4, 1, f) != 1) return fail("truncated part table");
        std::memcpy(p.name, name, 16); p.name[16] = 0;
        p.parent = parent;
        p.gridOrigin = { origin[0], origin[1], origin[2] };
        p.pivot      = { pivot[0], pivot[1], pivot[2] };
        p.dims = { int(dims[0]), int(dims[1]), int(dims[2]) };
        p.wordBase = wb;
        if (p.dims.x <= 0 || p.dims.x > 64 || p.dims.y <= 0 || p.dims.y > 64 ||
            p.dims.z <= 0 || p.dims.z > 64 || p.parent >= int(nParts))
            return fail("implausible part");
        uint64_t words = (uint64_t(p.dims.x) * p.dims.y * p.dims.z + 3) / 4;
        if (p.wordBase != wordsNeeded) return fail("part word base mismatch");
        wordsNeeded += words;
    }
    m.clips.resize(nClips);
    for (auto& c : m.clips) {
        char name[16]; uint32_t first, count, flags; float fps;
        std::fread(name, 1, 16, f);
        std::fread(&first, 4, 1, f);
        std::fread(&count, 4, 1, f);
        std::fread(&fps,   4, 1, f);
        if (std::fread(&flags, 4, 1, f) != 1) return fail("truncated clip table");
        std::memcpy(c.name, name, 16); c.name[16] = 0;
        c.first = first; c.count = count; c.fps = fps; c.loop = (flags & 1u) != 0;
        if (uint64_t(c.first) + uint64_t(c.count) > uint64_t(total) ||
            c.count == 0 || !(c.fps > 0.5f && c.fps <= 240.0f))
            return fail("clip out of range");
    }
    size_t nTrack = size_t(total) * nParts * 7;
    m.tracks.resize(nTrack);
    if (std::fread(m.tracks.data(), 4, nTrack, f) != nTrack)
        return fail("truncated tracks");
    m.vox.resize(size_t(wordsNeeded));
    if (std::fread(m.vox.data(), 4, m.vox.size(), f) != m.vox.size())
        return fail("truncated voxels");
    std::fclose(f);
    // Exact posed bound: the farthest any part-box corner gets from the feet
    // origin across EVERY track frame (the rest-pose sum-of-norms guess both
    // overshot ~30% — bloating the shadow cull + contact blob — and wasn't
    // guaranteed to hold once tracks translate parts). ~50k transforms, once.
    // Cell units here; converted to world voxels after the scan.
    m.boundRadius = 0.0f;
    for (uint32_t fr = 0; fr < m.totalFrames; ++fr) {
        const float* row = m.tracks.data() + size_t(fr) * m.parts.size() * 7;
        for (size_t pi = 0; pi < m.parts.size(); ++pi) {
            const float* t7 = row + pi * 7;
            glm::quat q(t7[3], t7[0], t7[1], t7[2]);
            glm::mat3 R = glm::mat3_cast(q);
            glm::vec3 t(t7[4], t7[5], t7[6]);
            const CreaturePart& p = m.parts[pi];
            for (int c = 0; c < 8; ++c) {
                glm::vec3 corner = p.gridOrigin +
                    glm::vec3((c & 1) ? p.dims.x : 0, (c & 2) ? p.dims.y : 0,
                              (c & 4) ? p.dims.z : 0);
                m.boundRadius = std::max(m.boundRadius,
                                         glm::length(R * corner + t));
            }
        }
    }
    m.boundRadius *= m.cellScale;   // world voxels
    return true;
}

bool loadCreatureModel(const char* path, CreatureModel& m) {
    return loadCvx2(path, m);
}

// Tiny key = value parser ('#' comments), matching the repo's hand-rolled-IO style.
static bool loadDef(const char* path, CreatureDef& d) {
    FILE* f = std::fopen(path, "r");
    if (!f) return false;
    char line[256], model[128] = {};
    while (std::fgets(line, sizeof line, f)) {
        char key[64], val[128];
        if (char* h = std::strchr(line, '#')) *h = 0;
        if (std::sscanf(line, " %63[a-zA-Z_] = %127s", key, val) != 2) continue;
        float x = float(std::atof(val));
        if      (!std::strcmp(key, "model"))             std::snprintf(model, sizeof model, "%s", val);
        else if (!std::strcmp(key, "walk_speed"))        d.walkSpeed = x;
        else if (!std::strcmp(key, "walk_ref_speed"))    d.walkRef = x;
        else if (!std::strcmp(key, "gallop_speed"))      d.gallopSpeed = x;
        else if (!std::strcmp(key, "gallop_ref_speed"))  d.gallopRef = x;
        else if (!std::strcmp(key, "turn_speed"))        d.turnSpeed = x;
        else if (!std::strcmp(key, "alert_dist"))        d.alertDist = x;
        else if (!std::strcmp(key, "flee_dist"))         d.fleeDist = x;
        else if (!std::strcmp(key, "calm_dist"))         d.calmDist = x;
        else if (!std::strcmp(key, "wander_r"))          d.wanderR = x;
        else if (!std::strcmp(key, "graze_time"))        d.grazeTime = x;
        else if (!std::strcmp(key, "idle_time"))         d.idleTime = x;
        else if (!std::strcmp(key, "spawn_forest"))      d.spawnForest = int(x);
        else if (!std::strcmp(key, "herd_chance"))       d.herdChance = x;
        else if (!std::strcmp(key, "group_min"))         d.groupMin = x;
        else if (!std::strcmp(key, "group_max"))         d.groupMax = x;
        else if (!std::strcmp(key, "population_target")) d.populationTarget = x;
        else if (!std::strcmp(key, "population_radius")) d.populationRadius = x;
        else if (!std::strcmp(key, "despawn_radius"))    d.despawnRadius = x;
        else if (!std::strcmp(key, "forest_min"))        d.forestMin = x;
        else if (!std::strcmp(key, "collide_half"))      d.collideHalf = x;
        else if (!std::strcmp(key, "collide_height"))    d.collideHeight = x;
    }
    std::fclose(f);
    if (!model[0]) { std::fprintf(stderr, "[creatures] %s: no model key\n", path); return false; }
    std::string dir(path);
    size_t slash = dir.find_last_of("/\\");
    dir = (slash == std::string::npos) ? "" : dir.substr(0, slash + 1);
    if (!loadCvx2((dir + model).c_str(), d.model)) return false;
    // The procedural layer (hoof snap, look-at, body pitch) and the nav
    // collision convert cell units to world voxels through cellScale, so the
    // asset pipeline's sub-voxel buckets (1, 1/2, 1/4 world voxels per cell)
    // all work. Anything off-bucket is an authoring mistake — reject it
    // loudly instead of shipping a creature at a size nobody chose.
    static constexpr float BUCKETS[] = { 1.0f, 0.5f, 0.25f };
    bool bucketOk = false;
    for (float b : BUCKETS) bucketOk |= std::abs(d.model.cellScale - b) < 1e-4f;
    if (!bucketOk) {
        std::fprintf(stderr, "[creatures] %s: cellScale %.3f unsupported for "
                     "wildlife (bake with --scale 0.1, 0.05, or 0.025)\n",
                     path, d.model.cellScale);
        return false;
    }
    return true;
}

int CreatureSystem::loadAssets(const char* dir) {
    namespace fs = std::filesystem;
    std::vector<std::string> paths;
    std::error_code ec;
    for (auto& e : fs::directory_iterator(dir, ec))
        if (e.path().extension() == ".def") paths.push_back(e.path().string());
    std::sort(paths.begin(), paths.end());   // deterministic type indices
    for (auto& p : paths) {
        CreatureDef d;
        auto stem = fs::path(p).stem().string();
        d.name = stem;
        if (!loadDef(p.c_str(), d)) continue;
        const CreatureModel& m = d.model;
        d.clipIdle         = m.clipIndex("Idle");
        d.clipWalk         = m.clipIndex("Walk");
        d.clipGallop       = m.clipIndex("Gallop");
        d.clipGraze        = m.clipIndex("Graze");
        d.clipAlert        = m.clipIndex("Alert");
        d.clipWalkToGallop = m.clipIndex("WalkToGallop");
        d.clipGallopToWalk = m.clipIndex("GallopToWalk");
        if (d.clipIdle < 0 || d.clipWalk < 0) {
            std::fprintf(stderr, "[creatures] %s: needs Idle + Walk clips\n", p.c_str());
            continue;
        }
        // Procedural-layer part lookups: the alert look-at drives head/neck,
        // the ground snap nudges the "*.lower" leg parts.
        for (size_t pi = 0; pi < m.parts.size(); ++pi) {
            if (!std::strcmp(m.parts[pi].name, "head")) d.partHead = int(pi);
            if (!std::strcmp(m.parts[pi].name, "neck")) d.partNeck = int(pi);
            const char* dot = std::strrchr(m.parts[pi].name, '.');
            if (dot && !std::strcmp(dot, ".lower")) d.hoofParts.push_back(int(pi));
        }
        // Body span for the oriented collision test: the rest-pose forward
        // extent over all parts (with a margin — legs reach past the torso).
        // Part extents are in cells; bodyLen is consumed in world voxels
        // (against collideHalf and groundYAt spans), so convert here.
        float zLo = 1e9f, zHi = -1e9f;
        for (auto& part : m.parts) {
            zLo = std::min(zLo, part.gridOrigin.z);
            zHi = std::max(zHi, part.gridOrigin.z + float(part.dims.z));
        }
        d.bodyLen = (zHi - zLo) * 0.85f * m.cellScale;
        // Snapshot the as-loaded tune values — /tune's R-reset restores these.
        std::vector<float> snap(CREATURE_TUNE_N);
        for (int i = 0; i < CREATURE_TUNE_N; ++i) snap[i] = *creatureTuneField(d, i);
        tuneDefaults.push_back(std::move(snap));
        std::printf("[creatures] %s: %zu parts, %u track frames, %zu clips\n",
                    stem.c_str(), m.parts.size(), m.totalFrames, m.clips.size());
        defs.push_back(std::move(d));   // move LAST — m aliases d.model
    }
    return int(defs.size());
}

void appendModelToAtlas(CreatureModel& m, std::vector<uint32_t>& blob) {
    m.palBase = uint32_t(blob.size());
    for (int i = 0; i < 256; ++i) blob.push_back(m.palette[i]);
    uint32_t voxBase = uint32_t(blob.size());
    for (auto& p : m.parts) p.wordBase += voxBase;       // file-local -> absolute
    blob.insert(blob.end(), m.vox.begin(), m.vox.end());
}

void CreatureSystem::buildGpuAtlas(std::vector<uint32_t>& blob) {
    blob.clear();
    if (atlasBuilt) {
        std::fprintf(stderr, "[creatures] buildGpuAtlas called twice\n");
        return;
    }
    atlasBuilt = true;
    for (auto& d : defs) appendModelToAtlas(d.model, blob);
}

// ---------------------------------------------------------------------------
// Terrain queries (creature-sized; water counts as blocking so wildlife
// never wades into pools)
// ---------------------------------------------------------------------------

static bool blockingForCreature(uint8_t v) {
    return isBlockingForPlayer(v) || v == MAT_WATER;
}

static bool boxClear(const World& w, glm::vec3 feet, float half, float height) {
    int x0 = int(std::floor(feet.x - half)), x1 = int(std::floor(feet.x + half));
    int z0 = int(std::floor(feet.z - half)), z1 = int(std::floor(feet.z + half));
    int y0 = int(std::floor(feet.y + 0.01f));
    int y1 = int(std::floor(feet.y + height - 0.01f));
    for (int y = y0; y <= y1; ++y)
        for (int z = z0; z <= z1; ++z)
            for (int x = x0; x <= x1; ++x)
                if (blockingForCreature(w.getVoxel(x, y, z))) return false;
    return true;
}

// Top of the first blocking voxel scanning DOWN a column; -1 if none in range.
static float groundYAt(const World& w, float fx, float fz, float yFrom, float yTo) {
    int x = int(std::floor(fx)), z = int(std::floor(fz));
    for (int y = int(std::floor(yFrom)); y >= int(std::floor(yTo)); --y)
        if (blockingForCreature(w.getVoxel(x, y, z))) return float(y + 1);
    return -1.0f;
}

// Standable ground at a column: the blocking top must not be water (water
// "blocks" the body box so wildlife never wades — but its surface must not
// count as footing either, or stags stroll across ponds).
static bool groundIsStandable(const World& w, float fx, float fz, float gy) {
    return w.getVoxel(int(std::floor(fx)), int(std::floor(gy)) - 1,
                      int(std::floor(fz))) != MAT_WATER;
}

// Oriented body test: three collision boxes along the facing axis (rear,
// centre, front). A single ±collideHalf box let the long visual body clip
// into boulders and trunks; this spans bodyLen so the nose and haunches
// collide too, while staying narrow enough to thread the forest.
static bool bodyClear(const World& w, const CreatureDef& d, glm::vec3 feet, float yaw) {
    glm::vec3 f(std::cos(yaw), 0.0f, std::sin(yaw));
    float off = std::max(0.0f, d.bodyLen * 0.5f - d.collideHalf);
    return boxClear(w, feet,           d.collideHalf, d.collideHeight)
        && boxClear(w, feet + f * off, d.collideHalf, d.collideHeight)
        && boxClear(w, feet - f * off, d.collideHalf, d.collideHeight);
}

static uint32_t xorshift(uint32_t& s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return s;
}
static float rand01(uint32_t& s) { return float(xorshift(s) & 0xFFFFFF) / float(0x1000000); }

// ---------------------------------------------------------------------------
// Spawning: deterministic herd lattice. Each CELL^2 column rolls once from
// (worldSeed, SEED_CREATURE, cell) — the same forests hold the same herds
// every session; instances are transient and re-derive on approach.
// ---------------------------------------------------------------------------

void CreatureSystem::reset() {
    all.clear();
    activeCells.clear();
    scanT = 0.0;
}

int CreatureSystem::countOfType(uint8_t t) const {
    int n = 0;
    for (auto& c : all) n += (c.type == t);
    return n;
}

// Returns how many creatures were actually placed (the caller only marks the
// lattice cell as consumed when something spawned — failed placement retries
// on a later scan, so streaming gaps and busy sites don't burn cells forever).
static int trySpawnHerd(CreatureSystem& sys, const World& w, uint64_t seed,
                        int cx, int cz, int64_t key, glm::vec3 playerPos) {
    int placedTotal = 0;
    // Type choice + all placement randomness from the cell's own stream.
    for (uint8_t t = 0; t < sys.defs.size(); ++t) {
        const CreatureDef& d = sys.defs[t];
        uint32_t s = creatureCellSeed(seed, cx, cz) ^ (0x517CC1B7u * (t + 1));
        int bx = cx * CreatureSystem::CELL + CreatureSystem::CELL / 2;
        int bz = cz * CreatureSystem::CELL + CreatureSystem::CELL / 2;
        float density = forestDensityAt(bx, bz);
        if (d.spawnForest && density < d.forestMin) continue;
        if (rand01(s) >= d.herdChance * density) continue;
        if (sys.countOfType(t) >= int(d.populationTarget)) continue;

        int lo = int(d.groupMin), hi = std::max(lo, int(d.groupMax));
        int size = lo + int(rand01(s) * float(hi - lo + 1));
        glm::vec2 site(bx + (rand01(s) - 0.5f) * (CreatureSystem::CELL / 2),
                       bz + (rand01(s) - 0.5f) * (CreatureSystem::CELL / 2));
        for (int i = 0; i < size; ++i) {
            // Scatter members around the site. Dense forest fails the box
            // test often (trunks everywhere), so give each member several
            // tries at a widening radius.
            for (int att = 0; att < 10; ++att) {
                float ang = rand01(s) * 6.2832f;
                float rad = 4.0f + rand01(s) * (14.0f + 2.0f * att);
                float fx = site.x + std::cos(ang) * rad;
                float fz = site.y + std::sin(ang) * rad;
                if (!w.voxelInWindow(int(fx), int(fz))) continue;
                // Members scatter from the cell centre — never let one pop
                // into existence in front of the camera (the cell-distance
                // guard alone leaves a ~96-voxel loophole).
                glm::vec2 toP(playerPos.x - fx, playerPos.z - fz);
                if (glm::dot(toP, toP) < 70.0f * 70.0f) continue;
                int th = terrainHeightAt(int(fx), int(fz));
                // Scan from just above the terrain surface (well under tree
                // crowns, TREE_CANOPY_CLEAR=24) so nobody spawns on a canopy.
                float gy = groundYAt(w, fx, fz, float(th + 4), float(th - 20));
                if (gy < 0 || !groundIsStandable(w, fx, fz, gy)) continue;
                glm::vec3 feet(fx, gy, fz);
                float yaw = rand01(s) * 6.2832f;
                if (!bodyClear(w, d, feet, yaw)) continue;   // full body, at this facing
                Creature c;
                c.type = t;
                c.pos = feet;
                c.yaw = c.targetYaw = yaw;
                c.cell = key;
                c.st = (rand01(s) < 0.6f) ? CRE_GRAZE : CRE_IDLE;
                c.stateT = 3.0f + rand01(s) * 8.0f;
                c.clip = (c.st == CRE_GRAZE && d.clipGraze >= 0) ? d.clipGraze : d.clipIdle;
                const CreatureClip& cl = d.model.clips[c.clip];
                c.clipT = rand01(s) * float(cl.count) / cl.fps;   // desync the herd
                sys.all.push_back(c);
                ++placedTotal;
                break;
            }
        }
    }
    return placedTotal;
}

// ---------------------------------------------------------------------------
// Behaviour + locomotion
// ---------------------------------------------------------------------------

static float yawTowards(float yaw, float target, float maxStep) {
    float dl = std::remainder(target - yaw, 6.28318530f);
    return yaw + std::clamp(dl, -maxStep, maxStep);
}

static int clipForState(const Creature& c, const CreatureDef& d) {
    switch (c.st) {
        case CRE_FLEE:   return d.clipGallop >= 0 ? d.clipGallop : d.clipWalk;
        case CRE_WANDER: return d.clipWalk;
        case CRE_GRAZE:  return d.clipGraze >= 0 ? d.clipGraze : d.clipIdle;
        case CRE_ALERT:  return d.clipAlert >= 0 ? d.clipAlert : d.clipIdle;
        default:         return d.clipIdle;
    }
}

static void enterCalm(Creature& c, const CreatureDef& d, uint32_t& rng) {
    float r = rand01(rng);
    if (r < 0.55f) { c.st = CRE_GRAZE; c.stateT = d.grazeTime * (0.5f + rand01(rng)); }
    else if (r < 0.8f) { c.st = CRE_IDLE; c.stateT = d.idleTime * (0.5f + rand01(rng)); }
    else {
        c.st = CRE_WANDER;
        float ang = rand01(rng) * 6.2832f, rad = 10.0f + rand01(rng) * d.wanderR;
        c.moveTarget = glm::vec2(c.pos.x + std::cos(ang) * rad,
                                 c.pos.z + std::sin(ang) * rad);
        c.stateT = 25.0f;   // wander timeout
    }
}

static void tickCreature(CreatureSystem& sys, Creature& c, const World& w,
                         glm::vec3 playerPos, float dt) {
    const CreatureDef& d = sys.defs[c.type];
    uint32_t& rng = sys.rng;
    glm::vec2 toPlayer(playerPos.x - c.pos.x, playerPos.z - c.pos.z);
    float dp = glm::length(toPlayer);

    // Self-rescue: if the centre box is somehow inside solid (bad spawn, an
    // edit landing on the creature), pop up out of it — or despawn silently
    // when buried too deep to fix. Never leave a stag standing in a rock.
    if (!boxClear(w, c.pos, d.collideHalf, d.collideHeight)) {
        int lift = 1;
        for (; lift <= 4; ++lift)
            if (boxClear(w, c.pos + glm::vec3(0, float(lift), 0),
                         d.collideHalf, d.collideHeight)) break;
        if (lift <= 4) c.pos.y += float(lift);
        else { c.pos.y = -10.0f; return; }   // update() culls y < 1
    }

    // -- state transitions ----------------------------------------------------
    if (c.st != CRE_FLEE && dp < d.fleeDist) {
        // Spook the whole herd: deer bolt together.
        for (auto& o : sys.all)
            if (o.cell == c.cell && o.type == c.type && o.st != CRE_FLEE) {
                o.st = CRE_FLEE;
                o.fleeRepickT = 0.0f;
            }
    }
    switch (c.st) {
        case CRE_IDLE:
        case CRE_GRAZE:
            if (dp < d.alertDist) { c.st = CRE_ALERT; c.stateT = 0.0f; break; }
            c.stateT -= dt;
            if (c.stateT <= 0.0f) enterCalm(c, d, rng);
            break;
        case CRE_WANDER: {
            if (dp < d.alertDist) { c.st = CRE_ALERT; c.stateT = 0.0f; break; }
            glm::vec2 dv = c.moveTarget - glm::vec2(c.pos.x, c.pos.z);
            c.stateT -= dt;
            if (glm::length(dv) < 3.0f || c.stateT <= 0.0f) enterCalm(c, d, rng);
            break;
        }
        case CRE_ALERT:
            c.targetYaw = std::atan2(toPlayer.y, toPlayer.x);   // face the player
            if (dp > d.calmDist) {
                c.stateT += dt;
                if (c.stateT > 2.0f) enterCalm(c, d, rng);
            } else c.stateT = 0.0f;
            break;
        case CRE_FLEE:
            c.fleeRepickT -= dt;
            if (c.fleeRepickT <= 0.0f) {
                c.fleeRepickT = 0.6f;
                glm::vec2 away = (dp > 1.0f) ? -toPlayer / dp
                                             : glm::vec2(std::cos(c.yaw), std::sin(c.yaw));
                float j = (rand01(rng) - 0.5f) * 1.0f;
                float ca = std::cos(j), sa = std::sin(j);
                away = glm::vec2(away.x * ca - away.y * sa, away.x * sa + away.y * ca);
                c.moveTarget = glm::vec2(c.pos.x, c.pos.z) + away * 150.0f;
            }
            if (dp > d.calmDist) { c.st = CRE_ALERT; c.stateT = 0.0f; }
            break;
    }

    // -- steering + gait ------------------------------------------------------
    float targetSpeed = 0.0f;
    if (c.st == CRE_WANDER) targetSpeed = d.walkSpeed;
    if (c.st == CRE_FLEE)   targetSpeed = d.gallopSpeed;
    if (c.st == CRE_WANDER || c.st == CRE_FLEE) {
        glm::vec2 dv = c.moveTarget - glm::vec2(c.pos.x, c.pos.z);
        if (glm::length(dv) > 0.5f) c.targetYaw = std::atan2(dv.y, dv.x);
    }
    float oldYaw = c.yaw;
    c.yaw = yawTowards(c.yaw, c.targetYaw, d.turnSpeed * dt);
    // Smoothed turn rate feeds the lean-into-turns roll at pose time.
    float yr = std::remainder(c.yaw - oldYaw, 6.28318530f) / std::max(dt, 1e-4f);
    c.yawRate += (yr - c.yawRate) * std::min(1.0f, 8.0f * dt);
    float accel = std::max(12.0f, targetSpeed);
    c.speed += std::clamp(targetSpeed - c.speed, -accel * 2.0f * dt, accel * dt);
    if (c.speed < 0.05f) c.speed = 0.0f;

    // -- move (axis-separated, sub-stepped, with step-up — like tickPlayer) ---
    if (c.speed > 0.0f) {
        glm::vec2 dir(std::cos(c.yaw), std::sin(c.yaw));
        float dist = c.speed * dt;
        glm::vec3 before = c.pos;
        // Sub-step <= 1 voxel so a gallop frame can't lodge the body a full
        // stride deep into something before the collision test sees it.
        int nStep = std::max(1, int(std::ceil(dist)));
        float step = dist / float(nStep);
        for (int ss = 0; ss < nStep; ++ss) {
            auto tryAxis = [&](glm::vec3 delta) -> bool {
                glm::vec3 np = c.pos + delta;
                // Never step onto water: probe the destination column's
                // footing before committing the move.
                float wg = groundYAt(w, np.x, np.z, np.y + 3.5f, np.y - 6.0f);
                if (wg >= 0.0f && !groundIsStandable(w, np.x, np.z, wg)) return false;
                for (int lift = 0; lift <= 3; ++lift) {
                    glm::vec3 sp = np + glm::vec3(0, float(lift), 0);
                    if (bodyClear(w, d, sp, c.yaw)) { c.pos = sp; return true; }
                }
                return false;
            };
            bool okX = tryAxis({dir.x * step, 0, 0});
            bool okZ = tryAxis({0, 0, dir.y * step});
            if (!okX && !okZ) break;
        }
        // Stuck detection: commanded travel vs actual. Blind steering can
        // wedge a long body into concave pockets the axis moves never
        // resolve — reroll the plan instead of grinding against the wall.
        glm::vec2 moved(c.pos.x - before.x, c.pos.z - before.z);
        if (glm::length(moved) < dist * 0.3f) c.stuckT += dt;
        else c.stuckT = std::max(0.0f, c.stuckT - 2.0f * dt);
        if (c.stuckT > 1.0f) {
            c.stuckT = 0.0f;
            c.speed *= 0.3f;
            if (c.st == CRE_FLEE) {
                // Hard deflection — anywhere but into the wall again. Route
                // it through moveTarget: steering re-derives targetYaw from
                // there every frame.
                float ny = c.yaw + (rand01(rng) < 0.5f ? 1.0f : -1.0f)
                                 * (1.6f + rand01(rng) * 1.0f);
                c.moveTarget = glm::vec2(c.pos.x + std::cos(ny) * 80.0f,
                                         c.pos.z + std::sin(ny) * 80.0f);
                c.fleeRepickT = 0.8f;   // hold the new heading briefly
            } else {
                // Back out the way it came, roughly.
                float back = c.yaw + 3.14159f + (rand01(rng) - 0.5f) * 1.2f;
                c.st = CRE_WANDER;
                c.moveTarget = glm::vec2(c.pos.x + std::cos(back) * 30.0f,
                                         c.pos.z + std::sin(back) * 30.0f);
                c.stateT = 15.0f;
            }
        }
    } else c.stuckT = 0.0f;

    // -- gravity / ground follow ----------------------------------------------
    if (c.onGround) {
        float gy = groundYAt(w, c.pos.x, c.pos.z, c.pos.y + 0.5f, c.pos.y - 4.5f);
        if (gy >= 0.0f && gy <= c.pos.y + 1.05f) { c.pos.y = gy; c.velY = 0.0f; }
        else c.onGround = false;   // walked off a ledge
    }
    if (!c.onGround) {
        c.velY = std::max(c.velY - GRAVITY * dt, -GRAVITY_TERMINAL);
        float ny = c.pos.y + c.velY * dt;
        float gy = groundYAt(w, c.pos.x, c.pos.z, c.pos.y + 0.5f, ny - 1.0f);
        if (c.velY <= 0.0f && gy >= 0.0f && ny <= gy) {
            c.pos.y = gy; c.velY = 0.0f; c.onGround = true;
        } else c.pos.y = ny;
    }

    // -- clip machine: any switch runtime-CROSSFADES from the old pose (the
    // baked WalkToGallop/GallopToWalk clips are no longer needed) -------------
    const CreatureModel& m = d.model;
    int want = clipForState(c, d);
    if (want != c.clip) {
        // The crossfade source is c.fadePose — the last pose actually shown
        // (kept fresh by buildPartInstances) — so switching mid-fade blends
        // from the on-screen pose instead of popping to the interrupted clip.
        c.fadeT = 0.0f;
        const CreatureClip& from = m.clips[c.clip];
        const CreatureClip& to   = m.clips[want];
        bool gaitSwap = (c.clip == d.clipWalk && want == d.clipGallop) ||
                        (c.clip == d.clipGallop && want == d.clipWalk);
        if (gaitSwap) {
            // Carry the stride PHASE across the gait change so planted legs
            // blend into the matching part of the new cycle.
            float phase = std::fmod(c.clipT * from.fps, float(from.count))
                        / float(from.count);
            c.clipT = phase * float(to.count) / to.fps;
        } else {
            c.clipT = to.loop ? rand01(rng) * float(to.count) / to.fps : 0.0f;
        }
        c.clip = want;
    }
    c.fadeT += dt;
    // Foot-skate lock: gait clips play at (ground speed / authored speed).
    float rate = 1.0f;
    if (c.clip == d.clipWalk   && d.walkRef   > 0.01f) rate = c.speed / d.walkRef;
    if (c.clip == d.clipGallop && d.gallopRef > 0.01f) rate = c.speed / d.gallopRef;
    c.clipT += dt * std::clamp(rate, 0.3f, 2.5f);

    // -- procedural-layer state (applied at pose time in buildPartInstances) --
    // Alert look-at weight: ramp in while facing the player, out otherwise.
    float lookTgt = (c.st == CRE_ALERT) ? 1.0f : 0.0f;
    c.lookW += std::clamp(lookTgt - c.lookW, -3.0f * dt, 3.0f * dt);
    // Terrain-follow pitch: ground slope under the body along the facing.
    {
        glm::vec2 fw(std::cos(c.yaw), std::sin(c.yaw));
        float half = std::max(4.0f, d.bodyLen * 0.5f);
        float hF = groundYAt(w, c.pos.x + fw.x * half, c.pos.z + fw.y * half,
                             c.pos.y + 6.0f, c.pos.y - 8.0f);
        float hR = groundYAt(w, c.pos.x - fw.x * half, c.pos.z - fw.y * half,
                             c.pos.y + 6.0f, c.pos.y - 8.0f);
        float tgt = (hF >= 0.0f && hR >= 0.0f)
                  ? std::clamp(std::atan2(hF - hR, 2.0f * half), -0.35f, 0.35f)
                  : 0.0f;
        c.pitch += (tgt - c.pitch) * std::min(1.0f, 6.0f * dt);
    }
}

void CreatureSystem::update(const World& w, uint64_t worldSeed,
                            glm::vec3 playerPos, glm::vec3 viewPos, float dt) {
    if (defs.empty()) return;

    // Despawn: out of range or outside the resident window (world reads air
    // out there — a creature would fall through it). Range is measured from
    // the player AND the view anchor (the diorama's orbit camera when the
    // map is open): the camera pans far from the body, and a herd you are
    // WATCHING must never vanish on screen just because the player's feet
    // are 700 voxels away.
    float maxDespawn = 0.0f;
    for (auto& d : defs) maxDespawn = std::max(maxDespawn, d.despawnRadius);
    for (size_t i = 0; i < all.size();) {
        Creature& c = all[i];
        glm::vec2 dv(playerPos.x - c.pos.x, playerPos.z - c.pos.z);
        glm::vec2 cv(viewPos.x  - c.pos.x, viewPos.z  - c.pos.z);
        float nearest = std::min(glm::length(dv), glm::length(cv));
        bool out = nearest > defs[c.type].despawnRadius
                || !w.voxelInWindow(int(c.pos.x), int(c.pos.z))
                || c.pos.y < 1.0f;
        if (out) {
            int64_t cell = c.cell;
            all[i] = all.back();
            all.pop_back();
            bool alive = false;
            for (auto& o : all) alive |= (o.cell == cell);
            if (!alive) activeCells.erase(cell);
        } else ++i;
    }

    // Spawn scan on a throttle: roll every lattice cell in range once.
    scanT -= dt;
    if (scanT <= 0.0) {
        scanT = 0.7;
        float maxR = 0.0f;
        for (auto& d : defs) maxR = std::max(maxR, d.populationRadius);
        int c0x = floorDiv(int(playerPos.x - maxR), CELL);
        int c1x = floorDiv(int(playerPos.x + maxR), CELL);
        int c0z = floorDiv(int(playerPos.z - maxR), CELL);
        int c1z = floorDiv(int(playerPos.z + maxR), CELL);
        for (int cz = c0z; cz <= c1z; ++cz)
            for (int cx = c0x; cx <= c1x; ++cx) {
                int64_t key = chunkKey(cx, cz);
                int bx = cx * CELL + CELL / 2, bz = cz * CELL + CELL / 2;
                glm::vec2 dv(playerPos.x - bx, playerPos.z - bz);
                float dist = glm::length(dv);
                if (dist > maxR) {
                    if (activeCells.count(key)) {
                        bool alive = false;
                        for (auto& o : all) alive |= (o.cell == key);
                        if (!alive) activeCells.erase(key);
                    }
                    continue;
                }
                if (dist < 100.0f) continue;           // never pop in on-screen next to the player
                if (activeCells.count(key)) continue;
                if (!w.voxelInWindow(bx, bz)) continue;
                // Mark the cell consumed only when a herd actually placed:
                // streaming gaps, busy sites, and the population cap all
                // retry on a later scan instead of burning the cell. (A
                // failed herd ROLL re-rolls too — deterministic arithmetic,
                // same result, negligible cost.)
                if (trySpawnHerd(*this, w, worldSeed, cx, cz, key, playerPos) > 0)
                    activeCells.insert(key);
            }
    }

    for (auto& c : all) tickCreature(*this, c, w, playerPos, dt);
}

int CreatureSystem::debugSpawn(const World& w, const char* species,
                               glm::vec3 pos, int n) {
    int type = -1;
    for (size_t i = 0; i < defs.size(); ++i)
        if (defs[i].name == species) { type = int(i); break; }
    if (type < 0) return -1;
    const CreatureDef& d = defs[type];
    int placed = 0;
    for (int i = 0; i < n; ++i) {
        Creature c;
        c.type = uint8_t(type);
        c.st = CRE_IDLE;
        c.stateT = 2.0f;
        c.clip = d.clipIdle;
        c.yaw = c.targetYaw = rand01(rng) * 6.2832f;
        bool ok = false;
        // Pass 0: the nice spawn — open ring, standable, body clear of
        // obstacles. Pass 1: hug the player and drop the body-clearance
        // test — overlapping a bush beats not spawning (the sim's unstuck
        // logic walks it free). Natural herd placement stays strict; this
        // relaxation is /spawn-only.
        for (int pass = 0; pass < 2 && !ok; ++pass) {
            for (int att = 0; att < 24 && !ok; ++att) {
                float ang = rand01(rng) * 6.2832f;
                float rad = pass == 0 ? 25.0f + rand01(rng) * 30.0f
                                      :  6.0f + rand01(rng) * 24.0f;
                float fx = pos.x + std::cos(ang) * rad;
                float fz = pos.z + std::sin(ang) * rad;
                if (!w.voxelInWindow(int(fx), int(fz))) continue;
                int th = terrainHeightAt(int(fx), int(fz));
                float gy = groundYAt(w, fx, fz, float(th + 4), float(th - 20));
                if (gy < 0 || !groundIsStandable(w, fx, fz, gy)) continue;
                glm::vec3 feet(fx, gy, fz);
                if (pass == 0 && !bodyClear(w, d, feet, c.yaw)) continue;
                c.pos = feet;
                ok = true;
            }
        }
        // Last resort: at the player's own feet. /spawn is a debug command —
        // it must NEVER come back empty-handed.
        if (!ok) {
            float gy = groundYAt(w, pos.x, pos.z, pos.y + 4.0f, pos.y - 20.0f);
            c.pos = glm::vec3(pos.x, gy >= 0.0f ? gy : pos.y, pos.z);
        }
        c.cell = chunkKey(floorDiv(int(c.pos.x), CELL),
                          floorDiv(int(c.pos.z), CELL));
        all.push_back(c);
        ++placed;
    }
    return placed;
}

// /tune save: rewrite each species' .def with the live CREATURE_TUNE values.
// Line-oriented patch — comments, ordering, and non-tuned keys all survive;
// tuned keys missing from the file are appended under a marker.
int CreatureSystem::saveDefs(const char* dir) const {
    int written = 0;
    for (auto& d : defs) {
        std::string path = std::string(dir) + "/" + d.name + ".def";
        FILE* f = std::fopen(path.c_str(), "r");
        if (!f) continue;
        std::vector<std::string> lines;
        char buf[512];
        while (std::fgets(buf, sizeof buf, f)) lines.push_back(buf);
        std::fclose(f);

        CreatureDef& dm = const_cast<CreatureDef&>(d);   // read-only field access
        bool patched[CREATURE_TUNE_N] = {};
        auto valueOf = [&](int row) {
            char v[48];
            if (CREATURE_TUNE[row].snap)
                std::snprintf(v, sizeof v, "%d", int(*creatureTuneField(dm, row)));
            else
                std::snprintf(v, sizeof v, "%.3f", *creatureTuneField(dm, row));
            return std::string(v);
        };
        for (auto& line : lines) {
            char key[64], val[128];
            std::string noComment = line.substr(0, line.find('#'));
            if (std::sscanf(noComment.c_str(), " %63[a-zA-Z_] = %127s", key, val) != 2)
                continue;
            for (int r = 0; r < CREATURE_TUNE_N; ++r) {
                if (std::strcmp(key, CREATURE_TUNE[r].key) != 0) continue;
                size_t hash = line.find('#');
                std::string comment = (hash == std::string::npos)
                                    ? "\n" : "   " + line.substr(hash);
                char head[192];
                std::snprintf(head, sizeof head, "%-17s = %s", key, valueOf(r).c_str());
                line = std::string(head) + comment;
                patched[r] = true;
                break;
            }
        }
        bool anyMissing = false;
        for (int r = 0; r < CREATURE_TUNE_N; ++r) anyMissing |= !patched[r];
        if (anyMissing) {
            lines.push_back("\n# added by /tune save\n");
            for (int r = 0; r < CREATURE_TUNE_N; ++r)
                if (!patched[r]) {
                    char head[192];
                    std::snprintf(head, sizeof head, "%-17s = %s\n",
                                  CREATURE_TUNE[r].key, valueOf(r).c_str());
                    lines.push_back(head);
                }
        }
        f = std::fopen(path.c_str(), "w");
        if (!f) continue;
        for (auto& line : lines) std::fputs(line.c_str(), f);
        std::fclose(f);
        ++written;
    }
    return written;
}

// ---------------------------------------------------------------------------
// Render list
// ---------------------------------------------------------------------------

// Sample one clip at time t (seconds at 1x) into per-part rigid transforms,
// slerping between the two neighbouring track frames — with rigid parts the
// motion can be as smooth as the Blender preview instead of 24 fps steps.
static void samplePose(const CreatureModel& m, int clip, float t,
                       glm::quat* q, glm::vec3* p) {
    const CreatureClip& cl = m.clips[clip];
    float f = t * cl.fps;
    int   n = int(cl.count);
    int   fi, fj;
    float a = f - std::floor(f);
    if (cl.loop) {
        fi = int(f) % n;
        fj = (fi + 1) % n;
    } else {
        fi = std::min(int(f), n - 1);
        fj = std::min(fi + 1, n - 1);
    }
    const float* ri = m.tracks.data() + (size_t(cl.first + fi) * m.parts.size()) * 7;
    const float* rj = m.tracks.data() + (size_t(cl.first + fj) * m.parts.size()) * 7;
    for (size_t i = 0; i < m.parts.size(); ++i) {
        const float* a7 = ri + i * 7;
        const float* b7 = rj + i * 7;
        glm::quat qa(a7[3], a7[0], a7[1], a7[2]);
        glm::quat qb(b7[3], b7[0], b7[1], b7[2]);
        q[i] = glm::slerp(qa, qb, a);
        p[i] = glm::mix(glm::vec3(a7[4], a7[5], a7[6]),
                        glm::vec3(b7[4], b7[5], b7[6]), a);
    }
}

// Shortest-arc rotation from one direction to another, angle-clamped.
static glm::quat rotationTo(glm::vec3 from, glm::vec3 to, float maxAngle) {
    from = glm::normalize(from);
    to   = glm::normalize(to);
    float d = std::clamp(glm::dot(from, to), -1.0f, 1.0f);
    float angle = std::min(std::acos(d), maxAngle);
    if (angle < 1e-3f) return glm::quat(1, 0, 0, 0);
    glm::vec3 axis = glm::cross(from, to);
    float len = glm::length(axis);
    if (len < 1e-5f) return glm::quat(1, 0, 0, 0);   // opposed: skip, clamp hides it
    return glm::angleAxis(angle, axis / len);
}

static constexpr float CRE_FADE = 0.25f;   // clip crossfade seconds

// Compose a model's posed parts (creature-local q/p per part) into world
// instance records. cellScale folds a uniform scale into the affine: rows are
// s*(Ry*Rp), so the world size of a cell is cellScale voxels — the shader's
// DDA parameterizes by the WORLD ray and needs no change (t stays world t).
static void composeModelParts(const CreatureModel& m, glm::vec3 pos, float yaw,
                              const glm::quat* q, const glm::vec3* p,
                              std::vector<CreaturePartInstance>& out) {
    float cy = std::cos(yaw), sy = std::sin(yaw);
    // Creature local: x lateral, y up, z forward; world fwd = (cos yaw, 0, sin yaw).
    glm::mat3 Ry(glm::vec3(sy, 0, -cy),          // column 0 = local +x in world
                 glm::vec3(0, 1, 0),
                 glm::vec3(cy, 0, sy));
    float s = m.cellScale;
    for (size_t pi = 0; pi < m.parts.size(); ++pi) {
        const CreaturePart& part = m.parts[pi];
        glm::mat3 Rp = glm::mat3_cast(q[pi]);
        glm::mat3 M  = Ry * Rp;                      // orthonormal
        glm::mat3 L  = M * s;
        glm::vec3 T  = pos + (Ry * (Rp * part.gridOrigin + p[pi])) * s;
        glm::mat3 Li = glm::transpose(M) * (1.0f / s);
        glm::vec3 Ti = -(Li * T);
        CreaturePartInstance inst;
        inst.rowX = glm::vec4(L[0][0], L[1][0], L[2][0], T.x);
        inst.rowY = glm::vec4(L[0][1], L[1][1], L[2][1], T.y);
        inst.rowZ = glm::vec4(L[0][2], L[1][2], L[2][2], T.z);
        inst.invX = glm::vec4(Li[0][0], Li[1][0], Li[2][0], Ti.x);
        inst.invY = glm::vec4(Li[0][1], Li[1][1], Li[2][1], Ti.y);
        inst.invZ = glm::vec4(Li[0][2], Li[1][2], Li[2][2], Ti.z);
        inst.dimsPal = glm::vec4(glm::vec3(part.dims), float(m.palBase));
        inst.meta    = glm::vec4(float(part.wordBase), 0, 0, 0);
        inst.anchor  = glm::vec4(0.0f);   // palette mode: unused
        out.push_back(inst);
    }
}

// Stamp last frame's world rows into instances [first, end) from `cache`
// (12 floats/part), then refresh the cache with THIS frame's rows. First
// frame (empty cache) — or a stale cache (`fresh` false: the model was NOT
// displayed last frame, so its pixels are disocclusions and "where it was
// N frames ago" would be a bogus motion vector) — falls back to current
// rows: zero motion, no garbage.
static void stampPrevRows(std::vector<CreaturePartInstance>& out, int first,
                          std::vector<float>& cache, bool fresh) {
    int n = int(out.size()) - first;
    bool have = fresh && (cache.size() == size_t(n) * 12);
    if (!have) cache.resize(size_t(n) * 12);
    for (int i = 0; i < n; ++i) {
        CreaturePartInstance& inst = out[size_t(first) + i];
        float* c = cache.data() + size_t(i) * 12;
        if (have) {
            inst.prevX = glm::vec4(c[0], c[1], c[2],  c[3]);
            inst.prevY = glm::vec4(c[4], c[5], c[6],  c[7]);
            inst.prevZ = glm::vec4(c[8], c[9], c[10], c[11]);
        } else {
            inst.prevX = inst.rowX; inst.prevY = inst.rowY; inst.prevZ = inst.rowZ;
        }
        c[0] = inst.rowX.x; c[1]  = inst.rowX.y; c[2]  = inst.rowX.z; c[3]  = inst.rowX.w;
        c[4] = inst.rowY.x; c[5]  = inst.rowY.y; c[6]  = inst.rowY.z; c[7]  = inst.rowY.w;
        c[8] = inst.rowZ.x; c[9]  = inst.rowZ.y; c[10] = inst.rowZ.z; c[11] = inst.rowZ.w;
    }
}

CreatureBound appendModelInstances(const CreatureModel& m,
                                   glm::vec3 pos, float yaw,
                                   int clip, float clipT, float fadeT,
                                   std::vector<float>& fadePose,
                                   std::vector<float>& prevXf,
                                   std::vector<CreaturePartInstance>& out) {
    glm::quat q[64];
    glm::vec3 p[64];
    size_t np = m.parts.size();
    samplePose(m, clip, clipT, q, p);
    if (fadeT < CRE_FADE && fadePose.size() == np * 7) {
        float wgt = fadeT / CRE_FADE;
        wgt = wgt * wgt * (3.0f - 2.0f * wgt);
        for (size_t i = 0; i < np; ++i) {
            const float* s7 = fadePose.data() + i * 7;
            glm::quat qs(s7[3], s7[0], s7[1], s7[2]);
            q[i] = glm::slerp(qs, q[i], wgt);
            p[i] = glm::mix(glm::vec3(s7[4], s7[5], s7[6]), p[i], wgt);
        }
    }
    fadePose.resize(np * 7);
    for (size_t i = 0; i < np; ++i) {
        float* s7 = fadePose.data() + i * 7;
        s7[0] = q[i].x; s7[1] = q[i].y; s7[2] = q[i].z; s7[3] = q[i].w;
        s7[4] = p[i].x; s7[5] = p[i].y; s7[6] = p[i].z;
    }
    CreatureBound b;
    b.firstPart = int(out.size());
    composeModelParts(m, pos, yaw, q, p, out);
    // The caller clears prevXf on frames the avatar isn't appended, so a
    // non-empty cache here is always exactly one frame old.
    stampPrevRows(out, b.firstPart, prevXf, true);
    b.partCount = int(out.size()) - b.firstPart;
    b.center = pos + glm::vec3(0, m.boundRadius * 0.45f, 0);
    b.radius = m.boundRadius + 1.5f;
    return b;
}

int CreatureSystem::buildPartInstances(std::vector<CreaturePartInstance>& parts,
                                       CreatureBound* bounds, int maxCreatures,
                                       const World& w, glm::vec3 playerPos,
                                       glm::vec3 camPos, float maxDist,
                                       glm::vec3 clipLo, glm::vec3 clipHi) {
    parts.clear();
    struct Cand { float d2; int i; };
    std::vector<Cand> cand;
    cand.reserve(all.size());
    for (int i = 0; i < int(all.size()); ++i) {
        const Creature& c = all[i];
        glm::vec3 dv = c.pos - camPos;
        float d2 = glm::dot(dv, dv);
        if (d2 > maxDist * maxDist) continue;
        // Diorama clip slab: whole-instance cull on the feet; partially
        // sliced bodies stay listed and the shader clips their voxels
        // against the same box the terrain marches (the Q/E cut carves
        // through a stag like it carves through a tree).
        if (c.pos.x < clipLo.x || c.pos.x > clipHi.x ||
            c.pos.z < clipLo.z || c.pos.z > clipHi.z ||
            c.pos.y > clipHi.y || c.pos.y < clipLo.y) continue;
        cand.push_back({d2, i});
    }
    if (int(cand.size()) > maxCreatures) {
        std::nth_element(cand.begin(), cand.begin() + maxCreatures, cand.end(),
                         [](const Cand& a, const Cand& b) { return a.d2 < b.d2; });
        cand.resize(maxCreatures);
    }
    int n = 0;
    glm::quat q[64];
    glm::vec3 p[64];
    ++poseFrame;
    for (auto& cd : cand) {
        Creature& cm = all[cd.i];
        const Creature& c = cm;
        const CreatureDef& d = defs[c.type];
        const CreatureModel& m = d.model;
        size_t np = m.parts.size();

        // -- pose-rate LOD: far creatures re-pose every 3rd frame, replaying
        // the cached instance block otherwise (sampling, procedural layer,
        // and matrix work all skipped; the 3-frame pose jump is sub-pixel at
        // that distance, and prev==cur rows read as zero motion — correct
        // for TAA at range). Walkers only: a galloping fleer covers ~2 vox
        // per replay window, which the panned map camera resolves as judder
        // (and would out-run the shadow bound), so movers pose fresh.
        if (cd.d2 > 400.0f * 400.0f && cm.lodCache.size() == np &&
            poseFrame - cm.lastPoseFrame < 3 &&
            c.speed < d.walkSpeed * 1.5f) {
            bounds[n].firstPart = int(parts.size());
            parts.insert(parts.end(), cm.lodCache.begin(), cm.lodCache.end());
            // A replayed pose was ALSO what last frame displayed, so the
            // surface didn't move: prev must equal cur (the cache still
            // carries the fresh frame's real motion rows — overwrite).
            for (size_t i = parts.size() - np; i < parts.size(); ++i) {
                parts[i].prevX = parts[i].rowX;
                parts[i].prevY = parts[i].rowY;
                parts[i].prevZ = parts[i].rowZ;
            }
            bounds[n].partCount = int(np);
            // The bound must describe the FROZEN geometry, not the live
            // sim position that kept ticking underneath it.
            bounds[n].center = cm.lodCenter;
            bounds[n].radius = m.boundRadius + 2.5f;
            cm.lastBuiltFrame = poseFrame;
            ++n;
            continue;
        }

        // -- base pose: current clip, crossfaded from the last SHOWN pose ---
        samplePose(m, c.clip, c.clipT, q, p);
        if (c.fadeT < CRE_FADE && c.fadePose.size() == np * 7) {
            float wgt = c.fadeT / CRE_FADE;
            wgt = wgt * wgt * (3.0f - 2.0f * wgt);       // smoothstep
            for (size_t i = 0; i < np; ++i) {
                const float* s7 = c.fadePose.data() + i * 7;
                glm::quat qs(s7[3], s7[0], s7[1], s7[2]);
                glm::vec3 ps(s7[4], s7[5], s7[6]);
                q[i] = glm::slerp(qs, q[i], wgt);
                p[i] = glm::mix(ps, p[i], wgt);
            }
        }
        // Refresh the snapshot with what this frame will display (base pose
        // only — the procedural layers below re-derive every frame anyway).
        cm.fadePose.resize(np * 7);
        for (size_t i = 0; i < np; ++i) {
            float* s7 = cm.fadePose.data() + i * 7;
            s7[0] = q[i].x; s7[1] = q[i].y; s7[2] = q[i].z; s7[3] = q[i].w;
            s7[4] = p[i].x; s7[5] = p[i].y; s7[6] = p[i].z;
        }
        // Authored hoof clearance BEFORE body pitch/lean: the ground snap
        // must preserve the clip's swing lift (snapping to absolute ground
        // glued walking feet to the floor — the exact foot-skate the rate
        // lock exists to prevent). Pose math is in cells; the lift is
        // compared against ground heights below, so store it in world voxels.
        float s = m.cellScale;
        float hoofLift[8];
        int   nHoof = int(d.hoofParts.size());
        for (int h = 0; h < nHoof && h < 8; ++h) {
            const CreaturePart& part = m.parts[d.hoofParts[h]];
            glm::vec3 bcRest = part.gridOrigin +
                glm::vec3(part.dims.x * 0.5f, 0.0f, part.dims.z * 0.5f);
            hoofLift[h] = (q[d.hoofParts[h]] * bcRest + p[d.hoofParts[h]]).y * s;
        }

        // -- whole-body terrain pitch + turn lean (about the feet origin) ---
        // Positive roll about local +z tips toward the anatomical right, so
        // a right turn (yawRate > 0) leans with POSITIVE sign.
        float lean = std::clamp(c.yawRate * 0.10f, -0.12f, 0.12f);
        if (std::abs(c.pitch) > 1e-3f || std::abs(lean) > 1e-3f) {
            glm::quat body = glm::angleAxis(lean, glm::vec3(0, 0, 1))
                           * glm::angleAxis(-c.pitch, glm::vec3(1, 0, 0));
            for (size_t i = 0; i < np; ++i) {
                q[i] = body * q[i];
                p[i] = body * p[i];
            }
        }

        // -- alert look-at: head (full) + neck (half) rotate toward the
        // player about their joint pivots ------------------------------------
        if (d.partHead >= 0 && c.lookW > 0.01f) {
            float cy0 = std::cos(c.yaw), sy0 = std::sin(c.yaw);
            glm::vec3 Xw0(sy0, 0, -cy0), Zw0(cy0, 0, sy0);
            glm::vec3 relW = (playerPos + glm::vec3(0, 15, 0)) - c.pos;
            // The pivot chain below (head/neck joints, p[] write-backs) is
            // all cell-space, so bring the world-voxel target into cells too
            // — mixing the two skews the aim direction whenever s != 1.
            glm::vec3 tgtL(glm::dot(relW, Xw0), relW.y, glm::dot(relW, Zw0));
            tgtL /= s;
            const CreaturePart& hp = m.parts[d.partHead];
            glm::vec3 headPivot = q[d.partHead] * hp.pivot + p[d.partHead];
            glm::vec3 headFwd   = q[d.partHead] * glm::vec3(0, 0, 1);
            glm::quat delta = rotationTo(headFwd, tgtL - headPivot, 1.0f);
            delta = glm::slerp(glm::quat(1, 0, 0, 0), delta, c.lookW);
            if (d.partNeck >= 0) {
                glm::quat half = glm::slerp(glm::quat(1, 0, 0, 0), delta, 0.5f);
                const CreaturePart& npt = m.parts[d.partNeck];
                glm::vec3 neckPivot = q[d.partNeck] * npt.pivot + p[d.partNeck];
                q[d.partNeck] = half * q[d.partNeck];
                p[d.partNeck] = neckPivot - q[d.partNeck] * npt.pivot;
                // the head's joint rides the rotated neck
                headPivot = neckPivot + half * (headPivot - neckPivot);
            }
            q[d.partHead] = delta * q[d.partHead];
            p[d.partHead] = headPivot - q[d.partHead] * hp.pivot;
        }

        // -- hoof ground snap: nudge each "*.lower" part vertically so the
        // foot meets the actual voxel ground (pitch rotates about the body
        // centre, so hooves would otherwise sink/hover on slopes) ------------
        if (c.onGround) {
            float cy0 = std::cos(c.yaw), sy0 = std::sin(c.yaw);
            glm::vec3 Xw0(sy0, 0, -cy0), Zw0(cy0, 0, sy0);
            for (int h = 0; h < nHoof && h < 8; ++h) {
                int hi = d.hoofParts[h];
                const CreaturePart& part = m.parts[hi];
                glm::vec3 bcRest = part.gridOrigin +
                    glm::vec3(part.dims.x * 0.5f, 0.0f, part.dims.z * 0.5f);
                glm::vec3 bcL = (q[hi] * bcRest + p[hi]) * s;   // -> world voxels
                glm::vec3 bcW = c.pos + Xw0 * bcL.x + glm::vec3(0, bcL.y, 0) + Zw0 * bcL.z;
                float gy = groundYAt(w, bcW.x, bcW.z, c.pos.y + 4.0f, c.pos.y - 6.0f);
                if (gy < 0.0f) continue;
                // Target = ground + the clip's AUTHORED clearance (swing lift
                // and graze-dig survive); only the pitch/terrain residual is
                // corrected, so flat ground yields dy ~ 0. The clamp and the
                // dead-band are world-voxel limits on the VISIBLE correction;
                // the write-back into p[] is cell-space.
                float target = gy + std::max(hoofLift[h], 0.0f);
                float dy = std::clamp(target - bcW.y, -1.5f, 1.5f);
                if (std::abs(dy) > 0.05f) p[hi].y += dy / s;
            }
        }

        // -- compose world matrices ------------------------------------------
        bounds[n].firstPart = int(parts.size());
        composeModelParts(m, c.pos, c.yaw, q, p, parts);
        stampPrevRows(parts, bounds[n].firstPart, cm.prevXf,
                      cm.lastBuiltFrame + 1 == poseFrame);
        cm.lodCache.assign(parts.begin() + bounds[n].firstPart, parts.end());
        cm.lastPoseFrame  = poseFrame;
        cm.lastBuiltFrame = poseFrame;
        bounds[n].partCount = int(parts.size()) - bounds[n].firstPart;
        // Slight radius pad: the procedural layer (look-at, hoof snap) can
        // push a part a touch past the track-measured bound.
        bounds[n].center = c.pos + glm::vec3(0, m.boundRadius * 0.45f, 0);
        bounds[n].radius = m.boundRadius + 2.5f;
        cm.lodCenter = bounds[n].center;
        ++n;
    }
    return n;
}
