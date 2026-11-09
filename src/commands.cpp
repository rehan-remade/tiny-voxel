// In-game dev console commands: the registry (ONE table driving dispatch,
// generated /help, and tab completion), its handlers, and the completer.
// Split out of app.cpp: command handlers legitimately span every subsystem,
// so this TU (and app.cpp -- nobody else) includes the full GameState
// aggregate. All commands begin with '/'.

#include "commands.h"
#include "game_state.h"
#include "audio.h"               // cmdMusic (not transitive via game_state.h)

#include <GLFW/glfw3.h>          // glfwGetTime (cmdTune's retune debounce)
#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ============================================================================
// Console command registry. ONE table (CONSOLE_CMDS, below the handlers)
// drives dispatch, the generated /help, and tab completion, so the help text
// can never drift from the code again (it had: /fp /reflect /douse /stream
// and others were implemented but unlisted).
// Handlers keep the old return convention: KEEP_OPEN for info/query/error
// output the player should read, CLOSE for action commands that hand back
// to the game.
// ============================================================================
static constexpr bool KEEP_OPEN = true;
static constexpr bool CLOSE     = false;

// /help sections, in display order.
enum CmdGroup : uint8_t { G_WARP, G_WORLD, G_RENDER, G_PANEL, G_DEMO, G_DEBUG, G_N };
static const char* const CMD_GROUP_NAME[G_N] =
    { "warp:", "world:", "render:", "panels:", "demo:", "debug:" };

struct ConsoleCmd {
    const char* name;      // primary name, no slash ("tp")
    const char* aliases;   // space-separated extra names, or nullptr
    CmdGroup    group;     // /help section
    const char* args;      // arg summary for the usage line, "" if none
    const char* help;      // one-liner; "  " + help must fit 72 cols
    const char* argWords;  // tab-completable arg keywords (space-separated),
                           // nullptr if none; dynamic sources (species names)
                           // are special-cased in the completer
    bool (*fn)(GameState& s, const std::vector<std::string>& tok);
};

// Shared by every warp command. If the destination is outside the resident
// window, recenter the window there first (regenerate + gate) — otherwise
// the per-frame clamp would just snap the player back to the window edge.
static void consoleTeleport(GameState& s, float x, float y, float z) {
    const int NC = GRID_XZ / CHUNK_VOX;
    int dcx = floorDiv(int(std::floor(x)), CHUNK_VOX);
    int dcz = floorDiv(int(std::floor(z)), CHUNK_VOX);
    glm::ivec2 o = s.world.windowOriginChunk;
    if (dcx < o.x || dcx >= o.x + NC || dcz < o.y || dcz >= o.y + NC)
        recenterWindow(s, int(std::floor(x)), int(std::floor(z)));
    s.player.pos = glm::vec3(x, y, z);
    s.player.vel = glm::vec3(0.0f);
    char buf[96];
    std::snprintf(buf, sizeof(buf), "teleported to %.1f %.1f %.1f", x, y, z);
    s.console.print(buf);
}

// cmdHelp iterates CONSOLE_CMDS, which in turn stores &cmdHelp — so it alone
// is declared ahead of the table (defined just after it).
static bool cmdHelp(GameState& s, const std::vector<std::string>& tok);
static bool cmdCreatures(GameState& s, const std::vector<std::string>& tok) {
    char buf[128];
    for (size_t t = 0; t < s.creatures.defs.size(); ++t) {
        float best = 1e9f;
        for (auto& c : s.creatures.all)
            if (c.type == t)
                best = std::min(best, glm::length(glm::vec2(
                    c.pos.x - s.player.pos.x, c.pos.z - s.player.pos.z)));
        int cnt = s.creatures.countOfType(uint8_t(t));
        if (cnt > 0)
            std::snprintf(buf, sizeof(buf), "  %s: %d alive, nearest %.0f vox",
                          s.creatures.defs[t].name.c_str(), cnt, best);
        else
            std::snprintf(buf, sizeof(buf), "  %s: none alive",
                          s.creatures.defs[t].name.c_str());
        s.console.print(buf);
    }
    std::snprintf(buf, sizeof(buf), "  active herd cells: %zu",
                  s.creatures.activeCells.size());
    s.console.print(buf);
    return KEEP_OPEN;
}

static bool cmdSpawn(GameState& s, const std::vector<std::string>& tok) {
    if (tok.size() < 2) { s.console.print("usage: /spawn <species> [n]"); return KEEP_OPEN; }
    // Right after boot the .cvx2 assets may still be loading — without
    // this check a valid name reads as an "unknown species" typo.
    if (s.creatures.defs.empty()) {
        s.console.print("creature assets still loading - try again in a moment");
        return KEEP_OPEN;
    }
    int n = tok.size() > 2 ? std::max(1, std::atoi(tok[2].c_str())) : 1;
    int got = s.creatures.debugSpawn(s.world, tok[1].c_str(), s.player.pos, n);
    char buf[96];
    if (got < 0) std::snprintf(buf, sizeof(buf), "unknown species '%s'", tok[1].c_str());
    else         std::snprintf(buf, sizeof(buf), "spawned %d %s", got, tok[1].c_str());
    s.console.print(buf);
    return KEEP_OPEN;
}

static bool cmdFind(GameState& s, const std::vector<std::string>& tok) {
    // Warp to the nearest living creature of a species — lands just
    // outside its flee radius (approaching from your current side), so
    // it alerts and faces you instead of bolting off-screen on arrival.
    if (s.creatures.defs.empty()) { s.console.print("no species loaded"); return KEEP_OPEN; }
    std::string want = tok.size() > 1 ? tok[1]
                     : s.creatures.defs.size() == 1 ? s.creatures.defs[0].name
                     : "";
    int type = -1;
    for (size_t i = 0; i < s.creatures.defs.size(); ++i)
        if (s.creatures.defs[i].name == want) { type = int(i); break; }
    if (type < 0) { s.console.print("usage: /find <species> (e.g. /find stag)"); return KEEP_OPEN; }
    const Creature* best = nullptr;
    float bestD2 = 1e18f;
    for (auto& c : s.creatures.all) {
        if (c.type != type) continue;
        glm::vec2 dv(c.pos.x - s.player.pos.x, c.pos.z - s.player.pos.z);
        float d2 = glm::dot(dv, dv);
        if (d2 < bestD2) { bestD2 = d2; best = &c; }
    }
    if (!best) {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "no %s alive nearby (try /forest, they spawn there)",
                      want.c_str());
        s.console.print(buf);
        return KEEP_OPEN;
    }
    // Stand-off point on your side of the animal, snapped to the ground.
    glm::vec2 away(s.player.pos.x - best->pos.x, s.player.pos.z - best->pos.z);
    float len = glm::length(away);
    away = (len > 1.0f) ? away / len : glm::vec2(1.0f, 0.0f);
    float standoff = s.creatures.defs[type].fleeDist + 10.0f;
    float dx = best->pos.x + away.x * standoff;
    float dz = best->pos.z + away.y * standoff;
    consoleTeleport(s, dx, float(terrainHeightAt(int(dx), int(dz)) + 3), dz);
    return CLOSE;
}

static bool cmdForest(GameState& s, const std::vector<std::string>& tok) {
    // Ring-scan the pure biome field for deep forest and warp there.
    int px = int(s.player.pos.x), pz = int(s.player.pos.z);
    for (int r = 120; r < 6000; r += 60) {
        int steps = std::max(8, r / 30);
        for (int i = 0; i < steps; ++i) {
            float a = float(i) * 6.2832f / float(steps);
            int x = px + int(std::cos(a) * r), z = pz + int(std::sin(a) * r);
            if (forestDensityAt(x, z) > 0.85f) {
                consoleTeleport(s, float(x), float(terrainHeightAt(x, z) + 24), float(z));
                return CLOSE;
            }
        }
    }
    s.console.print("no deep forest within 600 m");
    return KEEP_OPEN;
}

static bool cmdPos(GameState& s, const std::vector<std::string>& tok) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "pos %.1f %.1f %.1f",
                  s.player.pos.x, s.player.pos.y, s.player.pos.z);
    s.console.print(buf);
    return KEEP_OPEN;
}

static bool cmdGpu(GameState& s, const std::vector<std::string>& tok) {
    s.showGpuTimers = !s.showGpuTimers;
    s.console.print(std::string("GPU pass timers ") +
                    (s.showGpuTimers ? "ON (top-left overlay)" : "OFF"));
    return CLOSE;   // close so the overlay isn't hidden behind the console
}

static bool cmdTaa(GameState& s, const std::vector<std::string>& tok) {
    s.taaEnabled = !s.taaEnabled;
    s.console.print(std::string("temporal AA ") + (s.taaEnabled ? "ON" : "OFF"));
    return CLOSE;   // close so you can see the difference
}

static bool cmdRings(GameState& s, const std::vector<std::string>& tok) {
    s.ringsOn = !s.ringsOn;
    s.console.print(std::string("far-field LOD ring ") + (s.ringsOn ? "ON" : "OFF"));
    return CLOSE;   // close so you can see the horizon change
}

// SEAM: the /tune PANEL helpers (tuneRowCount/tuneRowValue/creatureTuneEdited)
// live in app.cpp and co-own this feature — both sides mutate the same
// TWEAKS/MAT_LIST/CREATURE_TUNE tables, s.tweak/s.matTune, and the
// s.creatureRetuneAt debounce. Change one, check the other.
static bool cmdTune(GameState& s, const std::vector<std::string>& tok) {
    if (tok.size() >= 2 && tok[1] == "save") {
        std::FILE* f = std::fopen("tune.cfg", "w");
        if (f) {
            for (int i = 0; i < TWEAK_N; ++i)
                std::fprintf(f, "%s %.4f\n", TWEAKS[i].name, s.tweak[i]);
            // Materials: "mat.<name>.<param>" keys, only for non-neutral rows.
            for (int m = 0; m < MAT_LIST_N; ++m) {
                int id = MAT_LIST[m].id;
                for (int p = 0; p < MP_COUNT; ++p)
                    std::fprintf(f, "mat.%s.%s %.4f\n", MAT_LIST[m].name,
                                 MAT_PARAMS[p].name, s.matTune[id * MP_COUNT + p]);
            }
            std::fclose(f);
            // Creature knobs live in their own .def files, not tune.cfg —
            // patch the current values back in place (comments survive).
            int nDefs = s.creatures.saveDefs("assets/creatures");
            char msg[96];
            std::snprintf(msg, sizeof msg,
                          "tune.cfg saved + %d creature .def file(s) updated", nDefs);
            s.console.print(msg);
        } else s.console.print("could not write tune.cfg");
        return KEEP_OPEN;
    }
    if (tok.size() >= 2 && tok[1] == "reset") {
        for (int i = 0; i < TWEAK_N; ++i) s.tweak[i] = TWEAKS[i].def;
        fillMatTuneDefaults(s.matTune);
        for (size_t sp = 0; sp < s.creatures.defs.size(); ++sp)
            for (int i = 0; i < CREATURE_TUNE_N; ++i)
                *creatureTuneField(s.creatures.defs[sp], i) =
                    s.creatures.tuneDefaults[sp][i];
        s.creatureRetuneAt = glfwGetTime() + 0.6;
        s.console.print("tuning reset to shipped defaults");
        return KEEP_OPEN;
    }
    if (tok.size() >= 2 && tok[1] == "lighting")  { s.tunePage = 0; s.tuneSel = 0; s.tuneOpen = true; s.console.print("tune: LIGHTING page"); return CLOSE; }
    if (tok.size() >= 2 && tok[1] == "materials") { s.tunePage = 1; s.tuneSel = 0; s.tuneOpen = true; s.console.print("tune: MATERIALS page"); return CLOSE; }
    if (tok.size() >= 2 && tok[1] == "creatures") { s.tunePage = 2; s.tuneSel = 0; s.tuneOpen = true; s.console.print("tune: CREATURES page"); return CLOSE; }
    s.tuneOpen = !s.tuneOpen;
    s.console.print(s.tuneOpen
        ? "tune ON: TAB=page, up/down select, left/right adjust, R reset, /tune save"
        : "tune panel OFF");
    return CLOSE;
}

static bool cmdFog(GameState& s, const std::vector<std::string>& tok) {
    s.fogMode = (s.fogMode + 1) % 3;
    s.console.print(s.fogMode == 0 ? "fog: OFF"
                  : s.fogMode == 1 ? "fog: AUTO (thick at dawn/dusk, burns off by noon)"
                                   : "fog: HEAVY (test density, ignores time of day)");
    return CLOSE;
}

static bool cmdSlopeshade(GameState& s, const std::vector<std::string>& tok) {
    s.slopeShade = 1 - s.slopeShade;
    s.console.print(s.slopeShade ? "slope shading: ON (terrain diffuse uses macro slope)"
                                 : "slope shading: OFF (pure voxel face normals)");
    return CLOSE;   // close so you can compare the hillside
}

static bool cmdSrgb(GameState& s, const std::vector<std::string>& tok) {
    s.colorMode = 1 - s.colorMode;
    s.console.print(s.colorMode
        ? "colour: sRGB-encode EXPERIMENT (linear -> ACES -> sRGB; palettes not yet retuned)"
        : "colour: LEGACY output curve (the shipped, visually-authored look)");
    return CLOSE;   // close so you can compare the scene
}

static bool cmdMusic(GameState& s, const std::vector<std::string>& tok) {
    if (Audio::ambiencePlaying()) {
        Audio::stopAmbience();
        s.console.print("music OFF");
    } else {
        Audio::startAmbience("assets/audio/ambient.mp3", 0.55f);
        s.console.print(Audio::ambiencePlaying()
            ? "music ON (looping assets/audio/ambient.mp3)"
            : "music: no track at assets/audio/ambient.mp3");
    }
    return CLOSE;
}

static bool cmdDouse(GameState& s, const std::vector<std::string>& tok) {
    EditDelta d;
    // Every doused voxel just became AIR — seed the leaf-support sweep with
    // them (same contract as simulateFire's burn-out path), or a crown whose
    // support was mid-burn is stranded floating.
    std::vector<glm::ivec3> gone;
    douseAllFire(s.world, s.fireSim, d, &gone);
    s.world.uploadDelta(d);
    forestryQueueLeafDecay(s.forestry, gone);
    s.console.print("doused all fire");
    return CLOSE;
}

// Per-crop growth pacing, in IN-GAME days (obeys /time speed). Set growth to a
// few days + crank /time speed for a growing-crops timelapse. Session-only.
static bool cmdCropTime(GameState& s, const std::vector<std::string>& tok) {
    char buf[160];   // the info line is ~120 bytes (the em dash is 3 in UTF-8)
    if (tok.size() < 2) {
        std::snprintf(buf, sizeof buf,
                      "crops ripen in %.2f in-game days (%.0fs at x1, before "
                      "each species' pace) — pair with /time speed for a timelapse",
                      s.cropGrowSecs / DAY_LENGTH, s.cropGrowSecs);
        s.console.print(buf);
        return KEEP_OPEN;
    }
    char* end = nullptr;
    float days = std::strtof(tok[1].c_str(), &end);
    if (end == tok[1].c_str() || *end != '\0' || !(days > 0.0f) || days > 10000.0f) {
        s.console.print("usage: /croptime <days>  (base in-game days for a crop "
                        "to ripen, >0)");
        return KEEP_OPEN;
    }
    s.cropGrowSecs = days * DAY_LENGTH;
    std::snprintf(buf, sizeof buf,
                  "crops now ripen in %g in-game days (%.0fs at x1; radishes "
                  "sprint, corn takes its time)",
                  days, s.cropGrowSecs);
    s.console.print(buf);
    return CLOSE;
}

// Dev: till + sow plots just east of the player — the aim-free way to drive
// the farming pipeline (same spirit as /chop /mine). "/plant carrot" sows one
// plot; "/plant all" sows one plot of EVERY species in a row, so a demo field
// (or a growth timelapse with /croptime + /time speed) is one command.
static bool cmdPlant(GameState& s, const std::vector<std::string>& tok) {
    if (tok.size() < 2) {
        s.console.print("usage: /plant <pumpkin|carrot|tomato|cabbage|corn|radish|all>");
        return KEEP_OPEN;
    }
    auto lower = [](char c) { return (c >= 'A' && c <= 'Z') ? char(c - 'A' + 'a') : c; };
    std::string w = tok[1];
    for (char& c : w) c = lower(c);   // whole arg case-insensitive, incl. "all"
    int one = -1;
    bool all = (w == "all");
    if (!all) {
        // Prefix match ("/plant cab" works).
        for (int sp = 0; sp < CROP_SPECIES_N && one < 0; ++sp) {
            const char* nm = CROP_DEFS[sp].name;
            size_t k = 0;
            while (k < w.size() && nm[k] && w[k] == lower(nm[k])) ++k;
            if (k == w.size() && k > 0) one = sp;
        }
        if (one < 0) {
            s.console.print("unknown crop - pumpkin carrot tomato cabbage corn "
                            "radish, or all");
            return KEEP_OPEN;
        }
    }
    glm::ivec2 baseCell = farmCellOrigin(int(std::floor(s.player.pos.x)),
                                         int(std::floor(s.player.pos.z)));
    int y = int(std::floor(s.player.pos.y));
    int savedSel = s.farm.seedSel;
    int nWant = all ? CROP_SPECIES_N : 1, planted = 0;
    // March a 2-row band east of the player, one EMPTY cell between plots:
    // adjacent plots inherit each other's terrace level and cellPlan refuses
    // the row after a few cells of dune slope (TERRACE_SPAN) — spaced plots
    // each found their own terrace, and the free room grows show-off giants.
    // A blocked cell just moves the band along (desert scrub blocks plenty).
    for (int i = 0, k = 0; i < nWant && k < nWant * 4; ++k) {
        glm::ivec2 cell = baseCell +
                          glm::ivec2((1 + k / 2) * 2 * FARM_CELL,
                                     (k & 1) * 2 * FARM_CELL);
        EditDelta d;
        // Till, then gate on the PLANT: farmTill's return means "wrote any
        // voxel", which is false on an intact already-tilled empty bed —
        // exactly the cell a replant wants. farmPlant itself rejects
        // anything that isn't a free farmland cap.
        farmTill(s.world, s.farm, cell, y, d);
        s.farm.seedSel = all ? i : one;
        bool ok = farmPlant(s.world, s.farm, cell, y, d);
        s.world.uploadDelta(d);
        if (ok) { ++i; ++planted; }
    }
    s.farm.seedSel = savedSel;
    char buf[96];
    std::snprintf(buf, sizeof buf,
                  "planted %d plot%s east of you (pace: /croptime + /time speed)",
                  planted, planted == 1 ? "" : "s");
    s.console.print(buf);
    return CLOSE;
}

static bool cmdTp(GameState& s, const std::vector<std::string>& tok) {
    if (tok.size() >= 2 && tok[1] == "spawn") {
        consoleTeleport(s, s.spawnPos.x, s.spawnPos.y, s.spawnPos.z);
        return CLOSE;
    }
    if (tok.size() >= 4) {
        consoleTeleport(s, std::strtof(tok[1].c_str(), nullptr),
                        std::strtof(tok[2].c_str(), nullptr),
                        std::strtof(tok[3].c_str(), nullptr));
        return CLOSE;
    }
    s.console.print("usage: /tp <x> <y> <z> | /tp spawn");
    return KEEP_OPEN;                 // bad usage: stay so they see it
}

static bool cmdCave(GameState& s, const std::vector<std::string>& tok) {
    // Warp to a cave entrance (worm-start on a hillside). With no args,
    // the nearest one; with an index, that specific entrance.
    if (s.gen.caveEntrances.empty()) {
        s.console.print("no cave entrances (regenerate world)");
        return KEEP_OPEN;
    }
    int idx;
    if (tok.size() >= 2) {
        idx = std::atoi(tok[1].c_str());
        if (idx < 0 || idx >= int(s.gen.caveEntrances.size())) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "entrance 0..%d",
                          int(s.gen.caveEntrances.size()) - 1);
            s.console.print(buf);
            return KEEP_OPEN;
        }
    } else {
        // Nearest entrance to the player.
        idx = 0;
        float best = 1e30f;
        for (int i = 0; i < int(s.gen.caveEntrances.size()); ++i) {
            glm::vec3 e(s.gen.caveEntrances[i]);
            float d = glm::distance(glm::vec2(e.x, e.z),
                                    glm::vec2(s.player.pos.x, s.player.pos.z));
            if (d < best) { best = d; idx = i; }
        }
    }
    glm::ivec3 e = s.gen.caveEntrances[idx];
    consoleTeleport(s, float(e.x) + 0.5f, float(e.y) + 2.0f, float(e.z) + 0.5f);
    char buf[96];
    std::snprintf(buf, sizeof(buf), "warped to cave entrance %d of %d",
                  idx, int(s.gen.caveEntrances.size()));
    s.console.print(buf);
    return CLOSE;
}

static bool cmdMega(GameState& s, const std::vector<std::string>& tok) {
    // Warp into a deep mega-cavern. No args = nearest; index = that one.
    if (s.gen.megaCaverns.empty()) {
        s.console.print("no mega-caverns (regenerate world)");
        return KEEP_OPEN;
    }
    int idx;
    if (tok.size() >= 2) {
        idx = std::atoi(tok[1].c_str());
        if (idx < 0 || idx >= int(s.gen.megaCaverns.size())) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "mega-cavern 0..%d",
                          int(s.gen.megaCaverns.size()) - 1);
            s.console.print(buf);
            return KEEP_OPEN;
        }
    } else {
        idx = 0;
        float best = 1e30f;
        for (int i = 0; i < int(s.gen.megaCaverns.size()); ++i) {
            glm::vec3 e(s.gen.megaCaverns[i]);
            float d = glm::distance(e, s.player.pos);
            if (d < best) { best = d; idx = i; }
        }
    }
    glm::ivec3 e = s.gen.megaCaverns[idx];
    consoleTeleport(s, float(e.x) + 0.5f, float(e.y) + 3.0f, float(e.z) + 0.5f);
    char buf[96];
    std::snprintf(buf, sizeof(buf), "warped to mega-cavern %d of %d",
                  idx, int(s.gen.megaCaverns.size()));
    s.console.print(buf);
    return CLOSE;
}

static bool cmdOasis(GameState& s, const std::vector<std::string>& tok) {
    // Warp to the nearest desert oasis (a vantage just above the pool).
    int ox, oy, oz;
    if (!nearestOasis(int(std::floor(s.player.pos.x)),
                      int(std::floor(s.player.pos.z)), ox, oy, oz)) {
        s.console.print("no oasis within range (head into deeper desert)");
        return KEEP_OPEN;
    }
    consoleTeleport(s, float(ox) + 0.5f, float(oy), float(oz) + 0.5f);
    s.console.print("warped to nearest oasis");
    return CLOSE;
}

static bool cmdMesa(GameState& s, const std::vector<std::string>& tok) {
    // Warp to the nearest desert mesa (atop the caprock).
    int mx, my, mz;
    if (!nearestMesa(int(std::floor(s.player.pos.x)),
                     int(std::floor(s.player.pos.z)), mx, my, mz)) {
        s.console.print("no mesa within range (head into deeper desert)");
        return KEEP_OPEN;
    }
    consoleTeleport(s, float(mx) + 0.5f, float(my), float(mz) + 0.5f);
    s.console.print("warped to nearest mesa");
    return CLOSE;
}

static bool cmdStream(GameState& s, const std::vector<std::string>& tok) {
    // Master streaming toggle (6e-3b): window scroll + frontier ladder.
    s.streamOn = !s.streamOn;
    s.console.print(std::string("streaming ") + (s.streamOn ? "ON" : "OFF"));
    return CLOSE;
}

static bool cmdTime(GameState& s, const std::vector<std::string>& tok) {
    // World clock (replaces the old dev T-key fast-forward). Anchors:
    // 0 midnight, .25 sunrise, .5 noon, .75 sunset. Queries KEEP_OPEN;
    // setters CLOSE so you watch the sky change.
    auto clockStr = [](float tod) {          // 0.28 -> "06:43"
        int mins = int(tod * 1440.0f + 0.5f) % 1440;
        char b[8];
        std::snprintf(b, sizeof b, "%02d:%02d", mins / 60, mins % 60);
        return std::string(b);
    };
    char buf[96];
    if (tok.size() == 1) {
        std::snprintf(buf, sizeof buf, "time %s (%.2f)  speed x%g%s",
                      clockStr(s.timeOfDay).c_str(), s.timeOfDay,
                      s.timeSpeed, s.timeSpeed == 0.0f ? "  PAUSED" : "");
        s.console.print(buf);
        return KEEP_OPEN;
    }
    const std::string& sub = tok[1];
    if (sub == "speed") {
        if (tok.size() == 2) {
            std::snprintf(buf, sizeof buf,
                          "time speed x%g (1 = 20-min day, 0 = paused)",
                          s.timeSpeed);
            s.console.print(buf);
            return KEEP_OPEN;
        }
        char* end = nullptr;
        float v = std::strtof(tok[2].c_str(), &end);
        // !(v >= 0) also rejects NaN.
        if (end == tok[2].c_str() || *end != '\0' || !(v >= 0.0f) || v > 1000.0f) {
            s.console.print("bad speed '" + tok[2] +
                            "' (want 0..1000; 1 = normal, 60 = old fast-forward)");
            return KEEP_OPEN;
        }
        s.timeSpeed = v;
        std::snprintf(buf, sizeof buf, "time speed set to x%g%s", v,
                      v == 0.0f ? " (paused)" : "");
        s.console.print(buf);
        return CLOSE;
    }
    if (sub == "pause") {
        s.timeSpeed = 0.0f;
        s.console.print("time paused (/time resume to run)");
        return CLOSE;
    }
    if (sub == "resume") {
        s.timeSpeed = 1.0f;
        s.console.print("time running at x1");
        return CLOSE;
    }
    // Set the clock: preset name, HH:MM, or a 0..1 fraction. A fraction
    // must contain a dot — a bare "12" is ambiguous between an hour and
    // a fraction, so it teaches instead of guessing.
    static const struct { const char* name; float tod; } TIME_PRESETS[] = {
        { "midnight", 0.00f }, { "sunrise", 0.25f }, { "dawn", 0.25f },
        { "morning",  0.28f },                        // the boot default
        { "noon",     0.50f }, { "midday",  0.50f },
        { "sunset",   0.75f }, { "dusk",    0.75f },
    };
    float tod = -1.0f;
    for (auto& p : TIME_PRESETS)
        if (sub == p.name) { tod = p.tod; break; }
    if (tod < 0.0f && sub.find(':') != std::string::npos) {
        int h = 0, m = 0, n = -1;
        if (std::sscanf(sub.c_str(), "%d:%d%n", &h, &m, &n) == 2 &&
            n == int(sub.size()) && h >= 0 && h <= 23 && m >= 0 && m <= 59) {
            tod = float(h * 60 + m) / 1440.0f;
        } else {
            s.console.print("bad time '" + sub + "' (want HH:MM, 00:00..23:59)");
            return KEEP_OPEN;
        }
    }
    if (tod < 0.0f && sub.find('.') != std::string::npos) {
        char* end = nullptr;
        float v = std::strtof(sub.c_str(), &end);
        if (end == sub.c_str() || *end != '\0' || !(v >= 0.0f) || v > 1.0f) {
            s.console.print("bad time '" + sub + "' (fraction must be 0..1)");
            return KEEP_OPEN;
        }
        tod = v - std::floor(v);             // 1.0 wraps to 0
    }
    if (tod < 0.0f) {
        s.console.print("bad time '" + sub + "' (want a preset, HH:MM, or 0..1)");
        s.console.print("usage: /time [preset|HH:MM|0..1] | speed <n> | pause | resume");
        s.console.print("presets: midnight sunrise dawn morning noon midday sunset dusk");
        return KEEP_OPEN;
    }
    s.timeOfDay = tod;
    std::snprintf(buf, sizeof buf, "time set to %s (%.2f)",
                  clockStr(tod).c_str(), tod);
    s.console.print(buf);
    return CLOSE;
}

// -- forestry ----------------------------------------------------------------
// /chop and /fell share the tree hunt: nearest STANDING tree via a top-down
// column scan for wood, tried nearest-first — desert bushes classify as
// stumps, and a bush next to the player must not shadow the tree behind it.
static bool cmdChopFell(GameState& s, const std::vector<std::string>& tok) {
    std::string cmd = tok[0];
    if (!cmd.empty() && cmd[0] == '/') cmd = cmd.substr(1);
    const bool instant = (cmd == "fell");

    glm::vec3 pp = s.player.pos;
    struct Cand { float d2; glm::ivec3 v; };
    std::vector<Cand> cands;
    for (int dz = -80; dz <= 80; dz += 2)
        for (int dx = -80; dx <= 80; dx += 2) {
            int x = int(std::floor(pp.x)) + dx;
            int z = int(std::floor(pp.z)) + dz;
            for (int dy = 60; dy >= -30; dy -= 2) {
                int y = int(std::floor(pp.y)) + dy;
                uint8_t m = s.world.getVoxel(x, y, z);
                if (m == MAT_WOOD || m == MAT_DESERT_WOOD ||
                    m == MAT_CACTUS_FLESH) {
                    cands.push_back({float(dx * dx + dz * dz), {x, y, z}});
                    break;
                }
            }
        }
    std::sort(cands.begin(), cands.end(),
              [](const Cand& a, const Cand& b) { return a.d2 < b.d2; });
    TreeInfo t;
    bool found = false;
    int tried = 0;
    // A failed candidate's whole cluster is dead — skip its bbox, or a
    // single wide stump eats every try before the next real tree.
    std::vector<std::pair<glm::ivec3, glm::ivec3>> failed;
    for (auto& c : cands) {
        bool covered = false;
        for (auto& fb : failed)
            if (glm::all(glm::greaterThanEqual(c.v, fb.first)) &&
                glm::all(glm::lessThanEqual(c.v, fb.second))) {
                covered = true;
                break;
            }
        if (covered) continue;
        if (++tried > 12) break;
        TreeInfo ti;
        if (collectTree(s.world, s.gen, c.v, ti) && !ti.isStump) {
            t = std::move(ti);
            found = true;
            break;
        }
        if (ti.valid) failed.push_back({ti.lo, ti.hi});
    }
    if (!found) {
        s.console.print("no standing tree nearby");
        return KEEP_OPEN;
    }
    if (instant) {
        clearTreeHover(s);
        if (fellTree(s.forestry, s.world, t, s.player.pos, s.particles,
                     s.atlasDyn)) {
            s.renderer.uploadCreatureAssets(s.atlasDyn);
            s.console.print("timber!");
            return CLOSE;
        }
        uprootStump(s.forestry, s.world, t, s.particles);
        s.console.print("cleared (fall roster full)");
        return CLOSE;
    }
    if (startChopSession(s, t)) {
        s.console.print("chopping...");
        return CLOSE;
    }
    s.console.print("no standable spot by that tree");
    return KEEP_OPEN;
}

// /mine: nearest minable rock via a top-down MAT_STONE column scan. Strata
// outcrops and scar floors reject in collectRock; a rejected candidate only
// suppresses a ±2 box (rejects carry no bbox), so cap the tries.
static bool cmdMine(GameState& s, const std::vector<std::string>&) {
    glm::vec3 pp = s.player.pos;
    struct Cand { float d2; glm::ivec3 v; };
    std::vector<Cand> cands;
    for (int dz = -80; dz <= 80; dz += 2)
        for (int dx = -80; dx <= 80; dx += 2) {
            int x = int(std::floor(pp.x)) + dx;
            int z = int(std::floor(pp.z)) + dz;
            for (int dy = 60; dy >= -30; dy -= 2) {
                int y = int(std::floor(pp.y)) + dy;
                if (s.world.getVoxel(x, y, z) != MAT_STONE) continue;
                // Only EXPOSED, above-soil stone may burn a try: on grass
                // nearly every column has buried strata topstone at h-5,
                // and scar-floor roots read as stone too — without these
                // gates the 12-try budget dies within ~8 voxels of the
                // player while a boulder stands in plain view. Air above
                // is O(1) and kills the strata flood; the soil line (a
                // noise stack) only runs for the few that pass.
                if (s.world.getVoxel(x, y + 1, z) == MAT_AIR &&
                    y >= terrainHeightAt(x, z) - 4)
                    cands.push_back({float(dx * dx + dz * dz), {x, y, z}});
                break;   // deeper stone in this column is buried anyway
            }
        }
    std::sort(cands.begin(), cands.end(),
              [](const Cand& a, const Cand& b) { return a.d2 < b.d2; });
    RockInfo r;
    bool found = false;
    int tried = 0;
    std::vector<std::pair<glm::ivec3, glm::ivec3>> failed;
    for (auto& c : cands) {
        bool covered = false;
        for (auto& fb : failed)
            if (glm::all(glm::greaterThanEqual(c.v, fb.first)) &&
                glm::all(glm::lessThanEqual(c.v, fb.second))) {
                covered = true;
                break;
            }
        if (covered) continue;
        if (++tried > 12) break;
        RockInfo ri;
        if (collectRock(s.world, s.gen, c.v, ri)) {
            r = std::move(ri);
            found = true;
            break;
        }
        failed.push_back({c.v - 2, c.v + 2});
    }
    if (!found) {
        s.console.print("no minable rock nearby");
        return KEEP_OPEN;
    }
    if (startMineSession(s, r)) {
        s.console.print("mining...");
        return CLOSE;
    }
    s.console.print("no standable spot by that rock");
    return KEEP_OPEN;
}

// Walk to + pluck the nearest RIPE crop — /chop and /mine's farming sibling,
// driving the full pipeline (route, tug(s), the species' collect flyer).
static bool cmdHarvest(GameState& s, const std::vector<std::string>&) {
    glm::vec3 pp = s.player.pos;
    float best = 1e18f;
    glm::ivec2 bestCell{0};
    bool found = false;
    for (const Crop& c : s.farm.crops) {
        if (c.stage != FARM_STAGE_MAX) continue;
        float dx = float(c.root.x) - pp.x, dz = float(c.root.z) - pp.z;
        float d2 = dx * dx + dz * dz;
        if (d2 < best) {
            best = d2;
            bestCell = farmCellOrigin(c.root.x, c.root.z);
            found = true;
        }
    }
    if (!found) {
        s.console.print("no ripe crop anywhere - /plant one and let it grow");
        return KEEP_OPEN;
    }
    if (startHarvestSessionAt(s, bestCell.x, bestCell.y)) {
        s.console.print("harvesting...");
        return CLOSE;
    }
    s.console.print("no standable spot by that plot");
    return KEEP_OPEN;
}

// -- demo director -----------------------------------------------------------
// /orbit: a slow cinematic spin of the diorama orbit camera, for recording
// demo footage. The spin rate eases in/out (no first-frame jerk on camera
// starts/cuts); framing moves are smoothstepped GLIDES that write the same
// cam.* tweaks the wheel and the /tune sliders drive, so nothing fights and
// /tune save can bake a framing you like. Ticked in app.cpp's map-motion
// block — it freezes with pause and only moves while the diorama is up.
static bool cmdOrbit(GameState& s, const std::vector<std::string>& tok) {
    DemoCam& dc = s.demoCam;
    char buf[120];
    auto num = [](const std::string& w, float& out) {
        char* end = nullptr;
        float v = std::strtof(w.c_str(), &end);
        if (end == w.c_str() || *end != '\0' || v != v) return false;   // v!=v: NaN
        out = v;
        return true;
    };

    // Framing glides: /orbit pitch|dist|height <value> [sec]. The value is
    // clamped to the tweak's own slider range; pitch speaks degrees on the
    // command line (the tweak stores radians).
    if (tok.size() >= 2 &&
        (tok[1] == "pitch" || tok[1] == "dist" || tok[1] == "height")) {
        CamGlide* g;
        int  tw;
        bool deg = false;
        if      (tok[1] == "pitch") { g = &dc.pitch;  tw = TW_CAM_PITCH; deg = true; }
        else if (tok[1] == "dist")  { g = &dc.dist;   tw = TW_CAM_DIST; }
        else                        { g = &dc.height; tw = TW_CAM_HEIGHT; }
        float scale = deg ? glm::degrees(1.0f) : 1.0f;   // print in CLI units
        float v;
        if (tok.size() < 3 || !num(tok[2], v)) {
            std::snprintf(buf, sizeof buf,
                          "usage: /orbit %s <%.0f..%.0f%s> [sec]   (now %.0f)",
                          tok[1].c_str(), TWEAKS[tw].mn * scale,
                          TWEAKS[tw].mx * scale, deg ? " deg" : " vox",
                          s.tweak[tw] * scale);
            s.console.print(buf);
            return KEEP_OPEN;
        }
        if (deg) v = glm::radians(v);
        v = glm::clamp(v, TWEAKS[tw].mn, TWEAKS[tw].mx);
        float sec = 2.0f;                       // default: a gentle reframe
        if (tok.size() >= 4 &&
            (!num(tok[3], sec) || sec < 0.0f || sec > 300.0f)) {
            s.console.print("bad glide time '" + tok[3] + "' (seconds, 0..300)");
            return KEEP_OPEN;
        }
        g->from = s.tweak[tw];
        g->to   = v;
        g->t    = 0.0f;
        g->dur  = std::max(sec, 0.01f);         // 0 = effectively instant
        g->active = true;
        std::snprintf(buf, sizeof buf, "camera %s: %.0f -> %.0f over %gs",
                      tok[1].c_str(), g->from * scale, v * scale, sec);
        s.console.print(buf);
        return CLOSE;
    }

    if (tok.size() >= 2 && tok[1] == "status") {
        std::snprintf(buf, sizeof buf, "orbit %s  speed %g deg/s (%.0fs/rev)",
                      dc.on ? "ON" : "off", dc.degPerSec,
                      360.0f / std::max(std::fabs(dc.degPerSec), 0.01f));
        s.console.print(buf);
        bool gliding = dc.pitch.active || dc.dist.active || dc.height.active;
        std::snprintf(buf, sizeof buf,
                      "framing: pitch %.0f deg  dist %.0f  height %.0f%s",
                      glm::degrees(s.tweak[TW_CAM_PITCH]), s.tweak[TW_CAM_DIST],
                      s.tweak[TW_CAM_HEIGHT], gliding ? "  (gliding)" : "");
        s.console.print(buf);
        return KEEP_OPEN;
    }

    if (tok.size() >= 2 && tok[1] == "off") {
        // Full stop for a botched take: the spin eases out, active glides
        // freeze where they are (their tweak keeps the mid-glide value).
        dc.on = false;
        dc.pitch.active = dc.dist.active = dc.height.active = false;
        s.console.print("orbit stopping (spin eases out, glides frozen)");
        return CLOSE;
    }

    // Bare /orbit toggles at the last speed; /orbit <deg/s> sets and starts.
    if (tok.size() >= 2) {
        float v;
        if (!num(tok[1], v) || std::fabs(v) > 90.0f) {
            s.console.print("usage: /orbit [-90..90 deg/s | off | status]   (negative = clockwise)");
            s.console.print("       /orbit pitch|dist|height <value> [sec]");
            return KEEP_OPEN;
        }
        if (v == 0.0f) {
            dc.on = false;
            s.console.print("orbit stopping (eases out)");
            return CLOSE;
        }
        dc.degPerSec = v;
        dc.on = true;
    } else {
        dc.on = !dc.on;
    }
    if (!dc.on) {
        s.console.print("orbit stopping (eases out)");
        return CLOSE;
    }
    std::snprintf(buf, sizeof buf, "orbiting at %g deg/s (%.0fs/rev)",
                  dc.degPerSec,
                  360.0f / std::max(std::fabs(dc.degPerSec), 0.01f));
    s.console.print(buf);
    return CLOSE;
}

// /freecam: chrome-independent twin of the minimap orb. The orb click is the
// ONLY other freeCam toggle, and /hud hides the orb — without this command a
// recording with hidden HUD could enter free-cam and never leave it (RMB
// pans instead of ordering walks, with no on-screen indicator of why).
// Mirrors the orb click's semantics: toggle + drop any chop session.
static bool cmdFreecam(GameState& s, const std::vector<std::string>&) {
    s.mapView.freeCam = !s.mapView.freeCam;
    cancelChop(s);
    std::string msg = s.mapView.freeCam
        ? "free-cam ON: right-drag pans the view, walking stops"
        : "free-cam OFF: camera follows the character";
    s.console.print(msg);
    return CLOSE;
}

// /hud: hide the passive screen chrome for a clean recording. Summoned UI
// (console, context menu, farm bar, satchel) stays usable — only the always-
// on overlays go: minimap, hotbar tray, DEV badge, name label. The
// hidden minimap also stops eating clicks/wheel (minimapHit gates on this);
// /freecam covers the orb it takes away.
static bool cmdHud(GameState& s, const std::vector<std::string>&) {
    s.hudHidden = !s.hudHidden;
    s.console.print(s.hudHidden
        ? "HUD hidden (minimap, hotbar, badges) - /hud brings it back"
        : "HUD restored");
    return CLOSE;
}

static bool cmdLogs(GameState& s, const std::vector<std::string>&) {
    char buf[112];
    std::snprintf(buf, sizeof(buf),
                  "  wood %d  leaves %d  cactus %d  blooms %d  stone %d",
                  s.forestry.collected[MAT_WOOD] +
                  s.forestry.collected[MAT_DESERT_WOOD],
                  s.forestry.collected[MAT_LEAVES] +
                  s.forestry.collected[MAT_SHRUB_SAGE],
                  s.forestry.collected[MAT_CACTUS_FLESH],
                  s.forestry.collected[MAT_CACTUS_BLOOM],
                  s.forestry.collected[MAT_STONE]);
    s.console.print(buf);
    return KEEP_OPEN;
}

// ----------------------------------------------------------------------------
// The registry. Grouped for /help; order within a group is display order.
// Every name and alias is unique, so row order never changes which handler
// runs. argWords feeds tab completion (nullptr = no static arg keywords).
static const ConsoleCmd CONSOLE_CMDS[] = {
    // -- warp -----------------------------------------------------------------
    { "tp",        nullptr,                 G_WARP,  "<x> <y> <z> | spawn",
      "teleport to coords (/tp spawn = back to spawn)",  "spawn",  cmdTp },
    { "pos",       nullptr,                 G_WARP,  "",
      "print current position",                          nullptr,  cmdPos },
    { "find",      nullptr,                 G_WARP,  "<species>",
      "warp to the nearest living creature (e.g. /find stag)", nullptr, cmdFind },
    { "forest",    nullptr,                 G_WARP,  "",
      "warp to the nearest deep forest",                 nullptr,  cmdForest },
    { "cave",      nullptr,                 G_WARP,  "[n]",
      "warp to nearest (or nth) cave entrance",          nullptr,  cmdCave },
    { "mega",      "megacavern megacanyon", G_WARP,  "[n]",
      "warp into nearest (or nth) mega-cavern",          nullptr,  cmdMega },
    { "oasis",     nullptr,                 G_WARP,  "",
      "warp to nearest desert oasis",                    nullptr,  cmdOasis },
    { "mesa",      nullptr,                 G_WARP,  "",
      "warp to nearest desert mesa",                     nullptr,  cmdMesa },
    // -- world ----------------------------------------------------------------
    { "time",      nullptr,                 G_WORLD,
      "[preset|HH:MM|0..1] | speed <n> | pause | resume",
      "set/inspect time of day and cycle speed (replaces T key)",
      "midnight sunrise dawn morning noon midday sunset dusk speed pause resume",
      cmdTime },
    { "croptime",  nullptr,                 G_WORLD, "<days>",
      "base in-game days for crops to ripen (timelapse; obeys /time speed)",
      nullptr, cmdCropTime },
    { "plant",     nullptr,                 G_WORLD, "<crop|all>",
      "till + sow plot(s) east of you (e.g. /plant all)",
      "pumpkin carrot tomato cabbage corn radish all", cmdPlant },
    { "spawn",     nullptr,                 G_WORLD, "<species> [n]",
      "force-spawn creatures nearby (e.g. /spawn stag 3)", nullptr, cmdSpawn },
    { "creatures", nullptr,                 G_WORLD, "",
      "wildlife census (counts + nearest)",              nullptr,  cmdCreatures },
    { "chop",      nullptr,                 G_WORLD, "",
      "walk to + chop the nearest tree (full pipeline)", nullptr,  cmdChopFell },
    { "fell",      nullptr,                 G_WORLD, "",
      "instantly fell the nearest tree (skip the walk)", nullptr,  cmdChopFell },
    { "mine",      nullptr,                 G_WORLD, "",
      "walk to + mine the nearest boulder (full pipeline)", nullptr, cmdMine },
    { "harvest",   nullptr,                 G_WORLD, "",
      "walk to + pluck the nearest ripe crop (full pipeline)", nullptr,
      cmdHarvest },
    { "logs",      nullptr,                 G_WORLD, "",
      "what the magnet has collected so far",            nullptr,  cmdLogs },
    { "douse",     nullptr,                 G_WORLD, "",
      "extinguish all burning voxels",                   nullptr,  cmdDouse },
    { "music",     nullptr,                 G_WORLD, "",
      "toggle ambient music (assets/audio/ambient.mp3)", nullptr,  cmdMusic },
    { "stream",    nullptr,                 G_WORLD, "",
      "toggle chunk streaming (window scroll + frontier)", nullptr, cmdStream },
    // -- render ---------------------------------------------------------------
    { "taa",       nullptr,                 G_RENDER, "",
      "toggle temporal anti-aliasing",                   nullptr,  cmdTaa },
    { "rings",     nullptr,                 G_RENDER, "",
      "toggle far-field LOD ring (distant 3D terrain)",  nullptr,  cmdRings },
    { "srgb",      nullptr,                 G_RENDER, "",
      "A/B the output colour pipeline (legacy vs sRGB)", nullptr,  cmdSrgb },
    { "slopeshade", nullptr,                G_RENDER, "",
      "toggle macro terrain slope shading (accordion fix)", nullptr, cmdSlopeshade },
    { "fog",       nullptr,                 G_RENDER, "",
      "cycle height fog: auto -> heavy -> off",          nullptr,  cmdFog },
    // -- panels ---------------------------------------------------------------
    { "tune",      nullptr,                 G_PANEL, "[page|save|reset]",
      "live tuning; pages: lighting/materials/creatures",
      "lighting materials creatures save reset",                   cmdTune },
    { "help",      nullptr,                 G_PANEL, "[command]",
      "this index; /help <name> shows one command",      nullptr,  cmdHelp },
    // -- demo (recording / creator tools) ---------------------------------------
    { "orbit",     nullptr,                 G_DEMO,
      "[deg/s|off|status] | pitch|dist|height <v> [sec]",
      "cinematic camera orbit + framing glides (recording)",
      "off status pitch dist height",                              cmdOrbit },
    { "hud",       nullptr,                 G_DEMO, "",
      "hide/show passive HUD chrome for clean footage",  nullptr,  cmdHud },
    { "freecam",   nullptr,                 G_DEMO, "",
      "toggle free-cam look-around (the minimap orb, sans HUD)", nullptr,
      cmdFreecam },
    // -- debug ----------------------------------------------------------------
    { "gpu",       nullptr,                 G_DEBUG, "",
      "toggle per-pass GPU timing overlay",              nullptr,  cmdGpu },
};

// Resolve a typed command word (leading slash already stripped) to its row:
// exact match on the primary name, then the space-separated alias list.
static const ConsoleCmd* findConsoleCmd(const std::string& cmd) {
    for (const ConsoleCmd& c : CONSOLE_CMDS) {
        if (cmd == c.name) return &c;
        for (const char* p = c.aliases; p && *p; ) {
            const char* e = p;
            while (*e && *e != ' ') ++e;
            if (size_t(e - p) == cmd.size() &&
                std::strncmp(p, cmd.c_str(), cmd.size()) == 0) return &c;
            p = *e ? e + 1 : e;
        }
    }
    return nullptr;
}

// /help: a grouped index that fits the panel whole (the old hand-written
// list was 28 lines into a 13-visible-line window — the top always scrolled
// away — and 7 commands + 2 aliases had drifted out of it). /help <name>
// gives one command's usage line, aliases included.
static bool cmdHelp(GameState& s, const std::vector<std::string>& tok) {
    if (tok.size() >= 2) {
        std::string want = tok[1];
        if (!want.empty() && want[0] == '/') want = want.substr(1);
        const ConsoleCmd* c = findConsoleCmd(want);
        if (!c) {
            s.console.print("no such command: " + want +
                            " (bare /help lists everything)");
            return KEEP_OPEN;
        }
        std::string u = std::string("usage: /") + c->name;
        if (c->args[0]) { u += ' '; u += c->args; }
        s.console.print(u);
        if (c->aliases) {
            std::string al = "  aliases:";
            for (const char* p = c->aliases; *p; ) {
                const char* e = p;
                while (*e && *e != ' ') ++e;
                al += " /"; al.append(p, e);
                p = *e ? e + 1 : e;
            }
            s.console.print(al);
        }
        s.console.print(std::string("  ") + c->help);
        return KEEP_OPEN;
    }
    // Bare /help: one packed line per group, flushed to a continuation line
    // (names column stays aligned) before it would cross the 72-col panel.
    s.console.print("commands (/help <name> for usage):");
    for (int g = 0; g < G_N; ++g) {
        char head[16];
        std::snprintf(head, sizeof head, "  %-8s", CMD_GROUP_NAME[g]);
        std::string line = head;
        for (const ConsoleCmd& c : CONSOLE_CMDS) {
            if (c.group != g) continue;
            std::string item = std::string(" /") + c.name;
            if (line.size() + item.size() > size_t(Console::COLS)) {
                s.console.print(line);
                line.assign(10, ' ');
            }
            line += item;
        }
        s.console.print(line);
    }
    return KEEP_OPEN;
}

// ----------------------------------------------------------------------------
// Tab completion (bash/cmd hybrid), driven by the same registry as dispatch
// and /help. Editing is append-only — the cursor is always at the end — so
// completion always targets the input's final token. First Tab extends to
// the longest common prefix of the matches (bash); when no extension is
// possible, repeated Tab cycles the candidates in place (cmd / Minecraft),
// Shift+Tab backwards. A unique match completes whole + a trailing space.
// Any real edit (type/backspace/recall/submit) cancels the cycle — Console's
// mutators call resetTab() so no call site can forget.
void consoleTabComplete(GameState& s, bool back) {
    Console& con = s.console;

    // Split [head][token]: token is everything after the last space (head
    // keeps that space). No space yet = completing the command word.
    size_t sp = con.input.find_last_of(' ');
    std::string head  = (sp == std::string::npos) ? std::string()
                                                  : con.input.substr(0, sp + 1);
    std::string typed = (sp == std::string::npos) ? con.input
                                                  : con.input.substr(sp + 1);

    // While cycling, matches are filtered by the token as it stood when the
    // cycle began (tabBase) — the completion just written into `input` must
    // not narrow the next step. `slash` preserves a typed leading '/'.
    bool cycling = (con.tabIdx >= 0);
    std::string raw = cycling ? con.tabBase : typed;
    std::string slash, prefix = raw;
    if (head.empty() && !prefix.empty() && prefix[0] == '/') {
        slash = "/";
        prefix = prefix.substr(1);
    }

    // Gather candidates (bare names, never slashed).
    std::vector<std::string> cand;
    auto addWords = [&](const char* w) {            // space-separated list
        while (w && *w) {
            const char* e = w;
            while (*e && *e != ' ') ++e;
            cand.emplace_back(w, e);
            w = *e ? e + 1 : e;
        }
    };
    if (head.empty()) {
        for (const ConsoleCmd& c : CONSOLE_CMDS) {
            cand.push_back(c.name);
            addWords(c.aliases);
        }
    } else {
        std::string word = con.input.substr(0, con.input.find(' '));
        if (!word.empty() && word[0] == '/') word = word.substr(1);
        const ConsoleCmd* c = findConsoleCmd(word);
        if (!c) return;
        addWords(c->argWords);
        // Dynamic sources the static table can't hold:
        if (!std::strcmp(c->name, "spawn") || !std::strcmp(c->name, "find"))
            for (auto& d : s.creatures.defs) cand.push_back(d.name);
        if (!std::strcmp(c->name, "help"))
            for (const ConsoleCmd& cc : CONSOLE_CMDS) {
                cand.push_back(cc.name);
                addWords(cc.aliases);
            }
    }

    // Filter by prefix; sort so the cycle order is stable across presses.
    std::vector<std::string> m;
    for (auto& x : cand)
        if (x.size() >= prefix.size() &&
            x.compare(0, prefix.size(), prefix) == 0) m.push_back(x);
    if (m.empty()) return;                          // nothing: silent no-op
    std::sort(m.begin(), m.end());
    m.erase(std::unique(m.begin(), m.end()), m.end());

    // Rewrite the final token; skip a completion that would not fit whole
    // (the render cursor invariant caps input at MAX_INPUT).
    auto setInput = [&](const std::string& word, bool space) {
        std::string next = head + slash + word + (space ? " " : "");
        if (int(next.size()) > Console::MAX_INPUT) return;
        con.input = next;
    };

    if (m.size() == 1) {                            // unique: done, arg next
        setInput(m[0], true);
        con.resetTab();
        return;
    }
    std::string lcp = m[0];                         // longest common prefix
    for (auto& x : m) {
        size_t i = 0;
        while (i < lcp.size() && i < x.size() && lcp[i] == x[i]) ++i;
        lcp.resize(i);
    }
    if (!cycling && lcp.size() > prefix.size()) {   // bash step: extend
        setInput(lcp, false);
        return;
    }
    if (!cycling) {                                 // cmd step: start cycling
        con.tabBase = raw;
        con.tabIdx = back ? int(m.size()) - 1 : 0;
    } else {
        int n = int(m.size());
        con.tabIdx = (con.tabIdx + (back ? -1 : 1) + n) % n;
    }
    setInput(m[size_t(con.tabIdx)], false);
}

bool execConsoleCommand(GameState& s, const std::string& line) {
    // Tokenize on whitespace.
    std::vector<std::string> tok;
    {
        std::string cur;
        for (char c : line) {
            if (c == ' ' || c == '\t') { if (!cur.empty()) { tok.push_back(cur); cur.clear(); } }
            else cur.push_back(c);
        }
        if (!cur.empty()) tok.push_back(cur);
    }
    if (tok.empty()) return KEEP_OPEN;
    std::string cmd = tok[0];
    if (!cmd.empty() && cmd[0] == '/') cmd = cmd.substr(1);

    if (const ConsoleCmd* c = findConsoleCmd(cmd)) return c->fn(s, tok);
    s.console.print("unknown command: " + cmd + " (try /help)");
    return KEEP_OPEN;
}
