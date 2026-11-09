// Top-level glue: window creation, input, gameplay tick, frame loop.
// Owns a single GameState and dispatches to world / player / particles / render.

#include "common.h"
#include "world.h"
#include "worldgen/worldgen.h"
#include "player.h"
#include "particles.h"
#include "materials.h"
#include "render.h"
#include "font8x8.h"
#include "water.h"
#include "fire.h"
#include "granular.h"
#include "worldio.h"
#include "console.h"
#include "audio.h"
#include "map.h"
#include "path.h"
#include "farm.h"
#include "creatures.h"
#include "forestry.h"

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <ctime>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <io.h>      // _dup2, _fileno -- for tee-ing stderr into the log file

#include "profiling.h"
#include "profiling_gpu.h"   // after <glad/gl.h>

// Global game state: struct GameState lives in game_state.h (app-layer-only
// aggregate -- see its banner; commands.cpp is the only other includer).
#include "game_state.h"

static GameState* g_state = nullptr;

// Simple startup profiler -- prints elapsed-from-start and delta-from-prev
// at each milestone so we can see where the time actually goes.
struct StartupTimer {
    using clock = std::chrono::steady_clock;
    clock::time_point start = clock::now();
    clock::time_point prev  = start;
    void lap(const char* label) {
        auto now = clock::now();
        double total = std::chrono::duration<double>(now - start).count();
        double delta = std::chrono::duration<double>(now - prev ).count();
        std::printf("[startup %6.2fs  +%5.2fs] %s\n", total, delta, label);
        prev = now;
    }
};

#include "commands.h"

// Wrapped toroidal sample of the per-column terrain height. heightMap is a
// GRID_XZ^2 window — index with the residue (matching buildHeightmap), never
// raw world coords (large / negative coords would index out of bounds).
static float surfaceHeightAt(const GameState& s, int wx, int wz) {
    if (s.gen.heightMap.empty()) return float(SURFACE_Y);
    return s.gen.heightMap[size_t(wx & (GRID_XZ - 1))
                         + size_t(wz & (GRID_XZ - 1)) * GRID_XZ];
}

// =============================================================================
// GLFW callbacks
// =============================================================================
// /tune page helpers — one source for both pages (lighting = TWEAKS/tweak[];
// materials = MAT_PARAMS/matTune[] for the selected material, with an extra
// row 0 = material selector). Selectable-row index == GameState::tuneSel.
// SEAM: cmdTune in commands.cpp co-owns this feature — both sides mutate the
// same TWEAKS/MAT_LIST/CREATURE_TUNE tables, s.tweak/s.matTune, and the
// s.creatureRetuneAt debounce. Change one, check the other.
static int tuneRowCount(const GameState& s) {
    if (s.tunePage == 0) return TWEAK_N;
    if (s.tunePage == 1) return MP_COUNT + 1;               // selector + params
    return CREATURE_TUNE_N + 1;                             // selector + knobs
}
// Pointer to the value for a selectable row + its range/default. Returns null
// for a selector row (row 0 on the materials/creatures pages), which has no
// scalar value (left/right cycles the material/species instead).
static float* tuneRowValue(GameState& s, int row, float& mn, float& mx, float& def) {
    if (s.tunePage == 0) {
        if (row < 0 || row >= TWEAK_N) return nullptr;
        mn = TWEAKS[row].mn; mx = TWEAKS[row].mx; def = TWEAKS[row].def;
        return &s.tweak[row];
    }
    if (s.tunePage == 2) {
        if (row <= 0 || row > CREATURE_TUNE_N || s.creatures.defs.empty())
            return nullptr;
        int i = row - 1;
        int sp = s.tuneSpeciesIdx % int(s.creatures.defs.size());
        mn = CREATURE_TUNE[i].mn; mx = CREATURE_TUNE[i].mx;
        def = s.creatures.tuneDefaults[sp][i];              // as loaded from the .def
        return creatureTuneField(s.creatures.defs[sp], i);
    }
    if (row <= 0 || row > MP_COUNT) return nullptr;         // 0 = selector
    int p = row - 1, id = MAT_LIST[s.tuneMatIdx].id;
    mn = MAT_PARAMS[p].mn; mx = MAT_PARAMS[p].mx;
    def = MAT_NEUTRAL[p];
    for (int i = 0; i < MAT_SEEDS_N; ++i)
        if (MAT_SEEDS[i].id == id) def = MAT_SEEDS[i].p[p];
    return &s.matTune[id * MP_COUNT + p];
}

// After ANY write to a creatures-page row: snap whole-number knobs, keep the
// group range sane, and — for knobs that shape spawning — schedule a debounced
// herd re-derive so the world reflects the new numbers within a second.
static void creatureTuneEdited(GameState& s) {
    if (s.tunePage != 2 || s.creatures.defs.empty()) return;
    int row = s.tuneSel - 1;
    if (row < 0 || row >= CREATURE_TUNE_N) return;
    CreatureDef& d = s.creatures.defs[s.tuneSpeciesIdx % int(s.creatures.defs.size())];
    float* v = creatureTuneField(d, row);
    if (v && CREATURE_TUNE[row].snap) *v = std::round(*v);
    if (d.groupMax < d.groupMin) d.groupMax = d.groupMin;
    if (d.despawnRadius < d.populationRadius + 50.0f)
        d.despawnRadius = d.populationRadius + 50.0f;       // despawn must outreach spawn
    if (CREATURE_TUNE[row].respawn) s.creatureRetuneAt = glfwGetTime() + 0.6;
}

// /tune mouse interaction: map a cursor position (window coords, top-left
// origin) onto the panel grid drawn by drawTunePanel/tune.frag. `press`
// selects the row under the cursor; press OR drag inside (or near) the bar
// cells scrubs the value. Layout constants live in tweaks.h — one source.
static void tunePointer(double cx, double cy, bool press) {
    auto& s = *g_state;
    GLFWwindow* win = glfwGetCurrentContext();
    int winW, winH, fbW, fbH;
    glfwGetWindowSize(win, &winW, &winH);
    glfwGetFramebufferSize(win, &fbW, &fbH);
    float px = float(cx) * (winW ? float(fbW) / float(winW) : 1.0f);
    float py = float(cy) * (winH ? float(fbH) / float(winH) : 1.0f);
    int   nRows  = tuneRowCount(s);
    float panelH = float(nRows + 2) * TUNE_CH;                 // + header + hint
    float xL = px - TUNE_MARGIN;
    float yL = py - (float(fbH) - panelH) * 0.5f;   // top-origin, row 0 = header
    int row = int(std::floor(yL / TUNE_CH)) - 1;    // -1: skip the header row
    if (press) {
        if (xL < 0.0f || xL >= TUNE_COLS * TUNE_CW || row < 0 || row >= nRows) {
            s.tuneDrag = false;                     // clicked outside the rows
            return;
        }
        s.tuneSel = row;
    }
    // Selector rows: click left / right half cycles the material / species.
    if ((s.tunePage == 1 || s.tunePage == 2) && s.tuneSel == 0) {
        if (press) {
            int n = (s.tunePage == 1) ? MAT_LIST_N : int(s.creatures.defs.size());
            int& idx = (s.tunePage == 1) ? s.tuneMatIdx : s.tuneSpeciesIdx;
            if (n > 0) {
                float mid = TUNE_COLS * TUNE_CW * 0.5f;
                idx = (xL < mid) ? (idx + n - 1) % n : (idx + 1) % n;
            }
            s.tuneDrag = false;
        }
        return;
    }
    float mn, mx, def; float* v = tuneRowValue(s, s.tuneSel, mn, mx, def);
    if (!v) return;
    float barX0 = TUNE_BAR_C0 * TUNE_CW, barW = TUNE_BAR_W * TUNE_CW;
    if (press && (xL < barX0 - 8.0f || xL > barX0 + barW + 8.0f)) return;
    float t = std::clamp((xL - barX0) / barW, 0.0f, 1.0f);
    *v = mn + t * (mx - mn);
    creatureTuneEdited(s);
}

static void cursorCb(GLFWwindow* win, double x, double y) {
    auto& s = *g_state;
    if (s.firstMouse) { s.lastMouseX = x; s.lastMouseY = y; s.firstMouse = false; return; }
    float dx = float(x - s.lastMouseX);
    float dy = float(y - s.lastMouseY);
    s.lastMouseX = x; s.lastMouseY = y;
    // An active /tune slider scrub owns the cursor even over the map (the
    // map branch below would otherwise eat the drag and sliders could only
    // be click-set in the game's primary mode).
    if (s.tuneOpen && s.tuneDrag) {
        tunePointer(x, y, /*press=*/false);
        return;
    }
    // Diorama: left-drag orbits; right-drag pans ONLY in free-cam (in play
    // mode the right button is the move command, handled in the main loop).
    if (s.leftDown) {
        // Click-vs-drag dead zone: orbit only engages once the cursor
        // strays >4 px from the press anchor (displacement, not summed
        // tremor — a slow steady click must stay a click).
        if (s.clickArmed) {
            if (std::max(std::abs(x - s.pressX), std::abs(y - s.pressY)) <= 4.0)
                return;
            s.clickArmed = false;   // it's a drag
            s.ctxOpen = false;
        }
        // Orbit: yaw is free live state; PITCH writes the cam.pitch tweak
        // (the camera reads it every frame), so dragging in play and the
        // /tune slider drive the same value instead of fighting.
        s.mapView.yaw += dx * 0.008f;
        float& cp = s.tweak[TW_CAM_PITCH];
        cp = glm::clamp(cp + dy * 0.006f,
                        TWEAKS[TW_CAM_PITCH].mn, TWEAKS[TW_CAM_PITCH].mx);
    } else if (s.rightDown && s.mapView.freeCam) {
        int ww = 0, wh = 0;                // cursor deltas are window coords
        glfwGetWindowSize(win, &ww, &wh);
        s.mapView.pan(dx, dy, float(wh));
    }
}

static void charCb(GLFWwindow*, unsigned int cp) {
    auto& s = *g_state;
    if (s.console.open) s.console.onChar(cp);
}

static void keyCb(GLFWwindow* win, int key, int /*sc*/, int action, int mods) {
    auto& s = *g_state;
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;

    // Console keeps the cursor captured (mouse-look is gated off while open),
    // so opening/closing it never changes cursor mode â€” only pause does.
    auto closeConsole = [&]() {
        s.console.open = false;
        s.console.clearInput();
    };

    // /tune panel: Tab switches page, arrows navigate/adjust (shift = fine),
    // R resets the row, Esc closes. Swallows its keys so they can't double-act.
    if (s.tuneOpen && !s.console.open) {
        int nRows = tuneRowCount(s);
        if (key == GLFW_KEY_TAB) {
            if (action == GLFW_PRESS) { s.tunePage = (s.tunePage + 1) % 3; s.tuneSel = 0; }
            return;   // lighting -> materials -> creatures (swallow repeats)
        }
        if (key == GLFW_KEY_UP)    { s.tuneSel = (s.tuneSel + nRows - 1) % nRows; return; }
        if (key == GLFW_KEY_DOWN)  { s.tuneSel = (s.tuneSel + 1) % nRows; return; }
        // Selector rows: left/right cycles the material / species.
        if ((s.tunePage == 1 || s.tunePage == 2) && s.tuneSel == 0) {
            int n = (s.tunePage == 1) ? MAT_LIST_N : int(s.creatures.defs.size());
            int& idx = (s.tunePage == 1) ? s.tuneMatIdx : s.tuneSpeciesIdx;
            if (n > 0) {
                if (key == GLFW_KEY_LEFT)  { idx = (idx + n - 1) % n; return; }
                if (key == GLFW_KEY_RIGHT) { idx = (idx + 1) % n; return; }
            }
            if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) { s.tuneOpen = false; return; }
            return;   // swallow other keys on the selector row
        }
        float mn, mx, def; float* v = tuneRowValue(s, s.tuneSel, mn, mx, def);
        float step = v ? (mx - mn) * ((mods & GLFW_MOD_SHIFT) ? 0.01f : 0.05f) : 0.0f;
        // Whole-number creature knobs: a key press must move a full unit or
        // the post-edit snap rounds the fractional step straight back.
        if (v && s.tunePage == 2 && s.tuneSel >= 1 &&
            CREATURE_TUNE[s.tuneSel - 1].snap)
            step = std::max(step, 1.0f);
        if (v && key == GLFW_KEY_LEFT)  { *v = std::max(mn, *v - step); creatureTuneEdited(s); return; }
        if (v && key == GLFW_KEY_RIGHT) { *v = std::min(mx, *v + step); creatureTuneEdited(s); return; }
        if (key == GLFW_KEY_R) {         // R = reset row, shift+R = reset the page
            if (mods & GLFW_MOD_SHIFT) {
                if (s.tunePage == 0) for (int i = 0; i < TWEAK_N; ++i) s.tweak[i] = TWEAKS[i].def;
                else {
                    int rows = tuneRowCount(s) - 1;
                    for (int p = 0; p < rows; ++p) {
                        float a,b,dd; float* pv = tuneRowValue(s, 1 + p, a, b, dd);
                        if (pv) *pv = dd;
                    }
                    if (s.tunePage == 2) s.creatureRetuneAt = glfwGetTime() + 0.6;
                }
            } else if (v) { *v = def; creatureTuneEdited(s); }
            return;
        }
        if (v && key == GLFW_KEY_HOME) { *v = def; creatureTuneEdited(s); return; }
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) { s.tuneOpen = false; return; }
    }

    // Esc: close the console if it's open; otherwise toggle pause. Handled
    // here (not polled) so a single Esc press can't both close the console
    // AND pause in the same frame.
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        if (s.console.open) { closeConsole(); return; }
        // Satchel closes on Esc BEFORE it would reach the pause menu.
        if (s.invOpen) { s.invOpen = false; s.leftDown = s.rightDown = false; return; }
        if (s.ctxOpen)      { s.ctxOpen = false; return; }
        // The diorama IS the game, so Esc pauses over it like anywhere else.
        s.paused = !s.paused;
        s.leftDown = s.rightDown = false;
        s.clickArmed = false;   // a press that straddles a pause is not a click
        return;
    }

    // Tab flips the diorama between the surface view and the cave X-ray.
    if (key == GLFW_KEY_TAB && action == GLFW_PRESS && s.worldReady &&
        !s.console.open && !s.paused) {
        s.mapView.xray = !s.mapView.xray;
        return;
    }

    // Inventory satchel (E): floats over the dimmed diorama (Esc
    // closes it — handled in the Escape block above). While open, its click
    // capture (mouseBtnCb) and the loop's !invOpen gates suppress world input
    // (including the diorama's Q/E slice raise, which shares this key). Not over
    // the console/pause (avoids overlapping modal panels).
    if (key == GLFW_KEY_E && action == GLFW_PRESS && s.worldReady &&
        !s.console.open && !s.paused) {
        s.invOpen = !s.invOpen;
        s.leftDown = s.rightDown = false;
        s.clickArmed = false;   // a press that straddles the toggle is not a click
        return;
    }

    // Farm Mode (G): the diorama farming toolbar. Toggling on always resets to
    // the Till tool; a pending job is dropped when leaving. Entering it also
    // dismisses any forestry interaction in flight — the two cursor modes
    // must never race for the same clicks.
    if (key == GLFW_KEY_G && action == GLFW_PRESS && s.worldReady &&
        !s.console.open && !s.paused) {
        s.farm.mode = !s.farm.mode;
        s.clickArmed = false;   // a press that straddles the toggle is not a click
        if (s.farm.mode) {
            s.farm.tool = FT_TILL;
            s.ctxOpen = false;
            cancelChop(s);
        } else {
            s.farm.painting = false;
            s.farm.cursorValid = false;
        }
        return;
    }
    // In Farm Mode, 1/2/3 select the tool (Till/Plant/Harvest). Re-pressing 2
    // with the Plant tool already armed cycles the seed choice (the seed bar
    // above the toolbar is also clickable).
    if (s.farm.mode && !s.console.open && !s.paused &&
        action == GLFW_PRESS && key >= GLFW_KEY_1 && key <= GLFW_KEY_3) {
        int t = key - GLFW_KEY_1;
        if (t == FT_PLANT && s.farm.tool == FT_PLANT)
            s.farm.seedSel = (s.farm.seedSel + 1) % CROP_SPECIES_N;
        s.farm.tool = t;
        return;
    }

    // Hotbar slot select (1-8): active when NOT in Farm Mode (there 1-3 are
    // the farm tools, handled above) and not browsing the satchel.
    if (!s.console.open && !s.paused && !s.invOpen && !s.farm.mode &&
        key >= GLFW_KEY_1 && key <= GLFW_KEY_8) {
        s.hotbar.selected = key - GLFW_KEY_1;
        return;
    }

    // Slash opens the console (dev mode only), Minecraft-style. The matching
    // char event then appends the leading '/' into the input line, so the
    // user sees "/" already typed. We don't prefill here to avoid doubling.
    if (key == GLFW_KEY_SLASH && action == GLFW_PRESS && !s.console.open) {
        if (!s.devMode || s.paused) return;
        s.console.open = true;
        s.console.clearInput();
        // First-ever open (nothing printed yet): teach the two things you
        // can't discover by typing.
        if (s.console.scrollback.empty())
            s.console.print("type /help for commands; up/down history, tab completes");
        s.leftDown = s.rightDown = false;
        return;
    }

    if (!s.console.open) return;

    if (key == GLFW_KEY_BACKSPACE) {
        s.console.backspace();
    } else if (key == GLFW_KEY_UP) {          // recall older command (repeats
        s.console.histPrev();                 // when held — keyCb admits
    } else if (key == GLFW_KEY_DOWN) {        // GLFW_REPEAT). Down walks back
        s.console.histNext();                 // toward the stashed draft.
    } else if (key == GLFW_KEY_TAB) {         // complete command / argument
        consoleTabComplete(s, (mods & GLFW_MOD_SHIFT) != 0);
    } else if (key == GLFW_KEY_ENTER || key == GLFW_KEY_KP_ENTER) {
        std::string line = s.console.takeInput();
        bool keepOpen = true;
        if (!line.empty()) {
            // Record before echoing: the scrollback copy gets a "> " prefix
            // and hard-wraps at 72 cols, so it can't serve as history. Even
            // failed commands are pushed — up-arrow-and-fix-the-typo is the
            // whole point. The file rewrite happens on every change so the
            // command that crashes the game is still recallable next run.
            if (s.console.pushHistory(line))
                s.console.saveHistory("console_history.txt");
            s.console.print("> " + line);
            keepOpen = execConsoleCommand(s, line);
        }
        // Action commands close + return to game; info/query commands keep
        // the console open so their output stays visible.
        if (!keepOpen) closeConsole();
    }
}

// The RuneScape-style minimap (top-right while the diorama is up). Geometry
// must match minimap.frag: a terrain disc of MM_R_MAP px inside an MM_RING px
// stone ring, MM_MARGIN px off the corner, the free-cam orb riding the ring's
// lower-left diagonal at MM_R_OUT + 7 (constants in common.h). Hit-testing
// happens in FRAMEBUFFER pixels like tunePointer.
// Returns 0 = miss, 1 = terrain disc (outWorld = the clicked world XZ,
// inverting the shader's rotate-with-camera transform), 2 = free-cam orb,
// 3 = ring frame (swallow only).
static int minimapHit(GLFWwindow* win, double cx, double cy, glm::vec2* outWorld) {
    auto& s = *g_state;
    if (s.hudHidden) return 0;   // /hud: a hidden minimap must not eat input
    int winW, winH, fbW, fbH;
    glfwGetWindowSize(win, &winW, &winH);
    glfwGetFramebufferSize(win, &fbW, &fbH);
    float px = float(cx) * (winW ? float(fbW) / float(winW) : 1.0f);
    float py = float(cy) * (winH ? float(fbH) / float(winH) : 1.0f);
    glm::vec2 p(px, py);
    glm::vec2 C(float(fbW) - MM_MARGIN - MM_R_OUT, MM_MARGIN + MM_R_OUT);
    glm::vec2 orbC = C + (MM_R_OUT + 7.0f) * glm::vec2(-MM_ORB_D, MM_ORB_D);
    if (glm::distance(p, orbC) <= MM_ORB_R) return 2;
    glm::vec2 lc = p - C;                    // disc-local, y DOWN (top-origin)
    float r = glm::length(lc);
    if (r <= MM_R_MAP) {
        if (outWorld) {
            // Same basis the shader uses: screen-up = camera view direction.
            float yaw = s.mapView.yaw;
            glm::vec2 fwdXZ(-std::sin(yaw), -std::cos(yaw));
            glm::vec2 rightXZ(-fwdXZ.y, fwdXZ.x);
            glm::vec2 center(s.mapView.target.x, s.mapView.target.z);
            *outWorld = center + s.minimapZoom * (lc.x * rightXZ - lc.y * fwdXZ);
        }
        return 1;
    }
    return (r <= MM_R_OUT + 2.0f) ? 3 : 0;
}

// A minimap click orders a walk, RS-style. The disc is a straight top-down
// view, so the destination is the clicked column's top surface (scanned down
// from just above the terrain height — canopy or water surface is what the
// pixel shows, same contract as mapCursorPick). Unexplored columns are
// ignored: walking into the dark is ordered on the big world, not through
// the minimap. Mirrors the fresh-click branch of the RMB world pick.
static void minimapMoveOrder(GameState& s, glm::vec2 wxz) {
    if (s.player.fly) return;                // fly steers with WASD
    int wx = int(std::floor(wxz.x)), wz = int(std::floor(wxz.y));
    if (!s.world.voxelInWindow(wx, wz)) return;
    auto it = s.mapExplored.chunks.find(
        chunkKey(chunkOfVoxel(wx), chunkOfVoxel(wz)));
    if (it == s.mapExplored.chunks.end() || !it->second.surfaceSeen) return;
    float surfH = surfaceHeightAt(s, wx, wz);
    glm::vec3 dest(float(wx) + 0.5f, surfH + 1.0f, float(wz) + 0.5f);
    int yTop = std::min(int(surfH) + 120, GRID_Y - 2);
    int yBot = std::max(int(surfH) - 64, 1);
    for (int y = yTop; y >= yBot; --y) {
        uint8_t v = s.world.getVoxel(wx, y, wz);
        if (isBlockingForPlayer(v) || v == MAT_WATER) {
            dest.y = float(y) + 1.0f;        // stand ON the column top
            break;
        }
    }
    s.moveActive   = findPath(s.world, s.player.pos, dest, s.movePath);
    s.moveWp       = 0;
    s.movePlanDest = dest;
    s.repathT      = 0.30f;
    s.moveStallT   = 0.0f;
    s.repathTries  = 0;
    // The stud marks where the route actually ends — a blocked click walks
    // as close as it can get.
    s.moveTarget   = s.moveActive ? s.movePath.back() : dest;
}

// Cursor ray through the map camera (shared by click-to-move picking and
// the hover affordances). False when the window is degenerate.
// NOTE: uses the CALLBACK-tracked cursor (s.lastMouseX/Y), not
// glfwGetCursorPos — the latter polls the live OS cursor, which for a click
// is the same thing, but the tracked value is what the click event actually
// pointed at (and it keeps the interaction drivable by synthetic input).
static bool mapCursorRay(GameState& s, GLFWwindow* win, glm::vec3& ro, glm::vec3& rd) {
    double mx = s.lastMouseX, my = s.lastMouseY;
    int ww = 0, wh = 0; glfwGetWindowSize(win, &ww, &wh);
    if (ww <= 0 || wh <= 0) return false;
    glm::vec3 camPos, camFwd;
    s.mapView.camera(camPos, camFwd);
    glm::vec3 cf = glm::normalize(camFwd);
    glm::vec3 cr = glm::normalize(glm::cross(cf, glm::vec3(0, 1, 0)));
    glm::vec3 cu = glm::cross(cr, cf);
    float ndcX = (float(mx) + 0.5f) / float(ww) * 2.0f - 1.0f;
    float ndcY = 1.0f - (float(my) + 0.5f) / float(wh) * 2.0f;
    float aspect = float(ww) / float(wh);
    ro = camPos;
    rd = glm::normalize(cf + cr * (ndcX * aspect * MAP_TAN_HALF_FOV)
                           + cu * (ndcY * MAP_TAN_HALF_FOV));
    return true;
}

// Diorama cursor pick: a CPU voxel march that applies the SAME fog-of-war
// gates the shader renders with (mapGateVoxel / mapCaveVoxel in
// gbuffer.frag), so clicks land on what the pixel actually shows — never on
// the raw world hiding behind it. Per mode:
//  - Above the Q/E slice plane (and below the slab floor) is carved air.
//  - SURFACE: the top MAP_SKIN voxels are real; below them an unvisited
//    16-voxel Y slice is SOLID unknown mass (undiscovered tunnels render
//    filled — the click stops on the mass exactly like the eye does), and
//    below the crust is carved. Unexplored columns are dark but VISIBLE,
//    so they stay clickable — walking into the dark is the game.
//  - X-RAY: rock is glass; the pixel shows the first DISCOVERED cave
//    air/water along the ray, and the returned voxel is that cave's FLOOR
//    (the spot the character can stand on). Wholly-unexplored columns are
//    holes and never picked.
static RayHit mapCursorPick(GameState& s, GLFWwindow* win, float maxDist) {
    RayHit miss{};
    miss.hit = false;
    glm::vec3 ro, rd;
    if (!mapCursorRay(s, win, ro, rd)) return miss;
    bool xray = s.mapView.xray;
    float sliceY = s.mapView.sliceY;
    // Slab floor, same formula as the framing block (mapClipLo.y).
    float yBot = glm::clamp(
        std::min(float(s.mapExplored.minSliceSeen * MapExplored::YSLICE_VOX - 32),
                 float(SURFACE_Y - 500)),
        0.0f, float(SURFACE_Y - 100));

    // Clip the ray to resident window XZ x [yBot, sliceY] — the exact volume
    // the map march sees (out-of-window has no CPU voxels anyway).
    glm::vec3 invD = 1.0f / rd;
    glm::vec3 wlo(float(s.world.originVoxX()), yBot, float(s.world.originVoxZ()));
    glm::vec3 whi = glm::vec3(wlo.x + float(GRID_XZ), sliceY, wlo.z + float(GRID_XZ));
    glm::vec3 ta = (wlo - ro) * invD, tb = (whi - ro) * invD;
    glm::vec3 ts = glm::min(ta, tb), te = glm::max(ta, tb);
    float tEnter = std::max({ts.x, ts.y, ts.z});
    float tExit  = std::min({te.x, te.y, te.z, maxDist});
    if (tExit < 0.0f || tEnter > tExit) return miss;

    float t = std::max(tEnter, 0.0f) + 1e-4f;
    glm::vec3 p = ro + rd * t;
    glm::ivec3 v = glm::clamp(glm::ivec3(glm::floor(p)),
                              glm::ivec3(wlo), glm::ivec3(glm::floor(whi)) - 1);
    glm::ivec3 stp((rd.x > 0) ? 1 : -1, (rd.y > 0) ? 1 : -1, (rd.z > 0) ? 1 : -1);
    glm::vec3 tDelta = glm::abs(invD);
    glm::vec3 tMax;
    for (int i = 0; i < 3; ++i) {
        float boundary = float(v[i]) + ((stp[i] > 0) ? 1.0f : 0.0f);
        tMax[i] = (boundary - ro[i]) * invD[i];
    }

    // Per-column / per-chunk gate caches (the DDA crosses each many times).
    int colX = INT32_MIN, colZ = INT32_MIN;
    float sh = 0.0f;
    int chX = INT32_MIN, chZ = INT32_MIN;
    const MapExplored::Entry* entry = nullptr;
    auto visited = [&](int y) {
        if (!entry) return false;
        int sl = y >> MapExplored::YSLICE_BITS;
        return (entry->ymask[sl >> 5] & (1u << (sl & 31))) != 0u;
    };

    int maxIter = int(tExit - t) * 3 + 64;
    for (int i = 0; i < maxIter; ++i) {
        if (float(v.y) < yBot || float(v.y) >= sliceY ||
            !s.world.voxelInWindow(v.x, v.z)) break;
        if (v.x != colX || v.z != colZ) {
            colX = v.x; colZ = v.z;
            sh = surfaceHeightAt(s, v.x, v.z);
            int cx = chunkOfVoxel(v.x), cz = chunkOfVoxel(v.z);
            if (cx != chX || cz != chZ) {
                chX = cx; chZ = cz;
                auto it = s.mapExplored.chunks.find(chunkKey(cx, cz));
                entry = (it != s.mapExplored.chunks.end()) ? &it->second : nullptr;
            }
        }
        uint8_t vox = s.world.getVoxel(v.x, v.y, v.z);
        bool skin = float(v.y) >= sh - 12.0f;           // MAP_SKIN
        if (!xray) {
            float crustBot = std::min(sh, sliceY) - 28.0f;   // MAP_CRUST
            bool solid =
                (skin && vox != 0) ||                        // real surface skin
                (!skin && float(v.y) >= crustBot &&
                 (!visited(v.y) || vox != 0));               // unknown mass / real
            if (solid) { miss.hit = true; miss.voxel = v; return miss; }
        } else if (entry && !skin && (vox == 0 || vox == MAT_WATER) &&
                   visited(v.y)) {
            // Struck the discovered-space cast: stand on its floor.
            int y = v.y;
            while (y - 1 >= int(yBot)) {
                uint8_t below = s.world.getVoxel(v.x, y - 1, v.z);
                if (below != 0 && below != MAT_WATER) break;
                --y;
            }
            if (y - 1 < int(yBot)) return miss;
            miss.hit = true;
            miss.voxel = glm::ivec3(v.x, y - 1, v.z);
            return miss;
        }
        if (tMax.x < tMax.y && tMax.x < tMax.z)      { v.x += stp.x; tMax.x += tDelta.x; }
        else if (tMax.y < tMax.z)                    { v.y += stp.y; tMax.y += tDelta.y; }
        else                                         { v.z += stp.z; tMax.z += tDelta.z; }
        if (tMax.x > tExit && tMax.y > tExit && tMax.z > tExit) break;
    }
    return miss;
}

// =============================================================================
// Forestry interaction: the context menu and the chop session.
// =============================================================================

// Per-16-sector max XZ surface reach of a rock about its base column, with
// each sector widened to its neighbours' max (a stand tile's ±2 body ring
// must clear the adjacent sectors' bulges too). Empty sectors inherit the
// widened neighbour pass; floor 1.0.
static void rockSectorReach(const RockInfo& r, float out[16]) {
    float raw[16];
    for (int i = 0; i < 16; ++i) raw[i] = 1.0f;
    for (auto& v : r.stone) {
        float dx = float(v.x) - float(r.base.x);
        float dz = float(v.z) - float(r.base.z);
        float d = std::sqrt(dx * dx + dz * dz);
        float ang = std::atan2(dz, dx);
        int a = int(std::floor(ang / 0.3926991f + 0.5f)) & 15;
        raw[a] = std::max(raw[a], d);
    }
    for (int i = 0; i < 16; ++i)
        out[i] = std::max({raw[(i + 15) & 15], raw[i], raw[(i + 1) & 15]});
}

// The stone voxel to swing at: pick-strike height, nearest the given spot.
static glm::ivec3 rockStrikeVoxel(const RockInfo& r, glm::vec2 fromXZ) {
    float strikeY = float(r.base.y) +
                    glm::clamp(float(r.hi.y - r.base.y) * 0.6f, 1.0f, 5.0f);
    glm::ivec3 tgt = r.stone[0];
    float best = 1e9f;
    for (auto& v : r.stone) {
        float d = std::abs(float(v.y) - strikeY) * 2.0f +
                  glm::distance(glm::vec2(v.x, v.z), fromXZ);
        if (d < best) { best = d; tgt = v; }
    }
    return tgt;
}

// Find a standable tile ringing a trunk or boulder. The target column itself
// is never walkable, and aiming A* at it burns the full search budget on
// every click — so pick the ring tile nearest the player and path THERE.
static bool findGatherStand(GameState& s, glm::ivec3 base, float coreR,
                            glm::vec3& out, const float* sectorR = nullptr) {
    glm::vec3 pp = s.player.pos;
    float bestD = 1e9f;
    bool found = false;
    for (int ring = 0; ring < 3 && !found; ++ring) {
        // Body radius 2 + margin: the tile's own ±2 ring must clear the
        // target or A* can never actually finish on it. With sectorR the
        // ring hugs the surface in EACH direction instead of parking at the
        // global max reach on every side.
        for (int a = 0; a < 16; ++a) {
            float rad = (sectorR ? sectorR[a] : coreR) +
                        3.5f + float(ring) * 1.5f;
            float ang = float(a) * 0.3926991f;
            int x = int(std::floor(base.x + 0.5f + std::cos(ang) * rad));
            int z = int(std::floor(base.z + 0.5f + std::sin(ang) * rad));
            // Scan deep below the base too: a stranded leaf blob (or a tree
            // on a bank) can hang well above the ground you'd stand on.
            for (int y = base.y + 8; y >= base.y - 40; --y) {
                if (!isBlockingForPlayer(s.world.getVoxel(x, y, z))) continue;
                bool clear = true;                      // feet at y+1, headroom
                for (int h = 1; h <= int(P_HEIGHT) && clear; ++h)
                    if (isBlockingForPlayer(s.world.getVoxel(x, y + h, z)))
                        clear = false;
                if (clear) {
                    glm::vec3 cand(x + 0.5f, float(y + 1), z + 0.5f);
                    // Prefer tiles at the TARGET's level: a clifftop lip
                    // right beside a valley tree is often nearer the player
                    // but unreachable-by-foot / useless to swing from.
                    float d = glm::distance(glm::vec2(cand.x, cand.z),
                                            glm::vec2(pp.x, pp.z)) +
                              6.0f * std::abs(cand.y - float(base.y));
                    if (d < bestD) { bestD = d; out = cand; found = true; }
                }
                break;                                  // topmost ground only
            }
        }
    }
    return found;
}

bool startChopSession(GameState& s, const TreeInfo& t) {
    if (s.player.fly) return false;
    glm::vec3 stand;
    if (!t.valid || !findGatherStand(s, t.base, t.trunkR, stand)) return false;
    // Swing target: trunk wood hugging the cut plane — dust lands there and
    // its liveness gates every swing.
    glm::ivec3 tgt = t.wood.empty() ? t.base : t.wood[0];
    float best = 1e9f;
    for (auto& v : t.wood) {
        float d = std::abs(float(v.y) - float(t.cutY)) * 2.0f +
                  glm::distance(glm::vec2(v.x, v.z),
                                glm::vec2(t.base.x, t.base.z));
        if (d < best) { best = d; tgt = v; }
    }
    s.chopKind      = 0;
    s.chopTree      = t;
    s.chopTargetVox = tgt;
    s.chopStand     = stand;
    s.swingsDone    = 0;
    s.swingsNeed    = t.isStump ? 3
                    : std::clamp(4 + int(t.wood.size()) / 300, 4, 10);
    s.chopSwingT    = 0.0f;
    s.chopReplans   = 0;
    // Route with the exact click-to-move plumbing — the stall handler replans
    // toward movePlanDest, which must be OUR stand tile, not a stale click.
    s.moveActive   = findPath(s.world, s.player.pos, stand, s.movePath);
    s.moveWp       = 0;
    s.movePlanDest = stand;
    s.moveTarget   = s.moveActive ? s.movePath.back() : stand;
    s.moveStallT   = 0.0f;
    s.repathT      = 0.30f;
    s.repathTries  = 0;
    s.chopPhase    = 1;
    std::printf("[forestry] session: base %d,%d,%d wood %zu leaves %zu "
                "stump %d charN %d trunkR %.1f stand %.0f,%.0f,%.0f route %d "
                "swings %d\n",
                t.base.x, t.base.y, t.base.z, t.wood.size(), t.leaves.size(),
                int(t.isStump), t.charN, t.trunkR, stand.x, stand.y, stand.z,
                int(s.moveActive), s.swingsNeed);
    return true;
}

bool startMineSession(GameState& s, const RockInfo& r) {
    if (s.player.fly) return false;
    if (!r.valid) return false;
    // Commit the sector table only once the session is definitely starting:
    // a failed start (no standable ring tile) must not clobber the LIVE
    // session's arrival gate with another rock's reach.
    float secR[16];
    rockSectorReach(r, secR);
    glm::vec3 stand;
    if (!findGatherStand(s, r.base, r.radius, stand, secR))
        return false;
    std::copy(std::begin(secR), std::end(secR), std::begin(s.mineSectorR));
    // Swing target: the stone voxel facing the stand at pick-strike height
    // (a little above the base slab) — dust lands there and its liveness
    // gates every swing. Re-aimed on arrival: the walk may legitimately
    // stop on a different side of the rock.
    glm::ivec3 tgt = rockStrikeVoxel(r, glm::vec2(stand.x, stand.z));
    s.chopKind      = 1;
    s.chopRock      = r;
    s.chopTargetVox = tgt;
    s.chopStand     = stand;
    s.swingsDone    = 0;
    s.swingsNeed    = std::clamp(4 + int(r.stone.size()) / 400, 4, 10);
    s.chopSwingT    = 0.0f;
    s.chopReplans   = 0;
    s.moveActive   = findPath(s.world, s.player.pos, stand, s.movePath);
    s.moveWp       = 0;
    s.movePlanDest = stand;
    s.moveTarget   = s.moveActive ? s.movePath.back() : stand;
    s.moveStallT   = 0.0f;
    s.repathT      = 0.30f;
    s.repathTries  = 0;
    s.chopPhase    = 1;
    std::printf("[forestry] mine session: base %d,%d,%d stone %zu radius %.1f "
                "stand %.0f,%.0f,%.0f route %d swings %d\n",
                r.base.x, r.base.y, r.base.z, r.stone.size(), r.radius,
                stand.x, stand.y, stand.z, int(s.moveActive), s.swingsNeed);
    return true;
}

// Walk to a ripe crop's plot and PLUCK it — mirrors the mine session but ends
// after one or two Harvest-clip tugs (roots are stuck), firing collectProduce
// on the final beat. chopKind 2. Crop liveness is farmHasRipeCropAt, not a
// voxel material.
static bool startHarvestSession(GameState& s, glm::ivec2 cell) {
    if (s.player.fly) return false;
    if (!farmHasRipeCropAt(s.farm, cell)) return false;
    glm::ivec3 lo, hi;
    if (!farmRipeCropBounds(s.farm, cell, lo, hi)) return false;
    // Stand ring around the plot centre; lo.y is the crop's ground level.
    glm::ivec3 base(cell.x + FARM_CELL / 2, lo.y, cell.y + FARM_CELL / 2);
    glm::vec3 stand;
    if (!findGatherStand(s, base, float(FARM_CELL) * 0.5f, stand)) return false;
    s.chopKind      = 2;
    s.chopCell      = cell;
    s.chopTargetVox = lo;          // crop root; liveness uses farmHasRipeCropAt
    s.chopStand     = stand;
    s.swingsDone    = 0;
    // Root vegetables are STUCK: the first tug strains and showers soil, the
    // second gives. Everything else pops off in one pluck.
    {
        int spc = farmCropSpeciesAt(s.farm, cell);
        s.swingsNeed = (spc >= 0 && CROP_DEFS[spc].root) ? 2 : 1;
    }
    s.chopSwingT    = 0.0f;
    s.chopReplans   = 0;
    s.moveActive   = findPath(s.world, s.player.pos, stand, s.movePath);
    s.moveWp       = 0;
    s.movePlanDest = stand;
    s.moveTarget   = s.moveActive ? s.movePath.back() : stand;
    s.moveStallT   = 0.0f;
    s.repathT      = 0.30f;
    s.repathTries  = 0;
    s.chopPhase    = 1;
    std::printf("[farm] harvest session: cell %d,%d base %d,%d,%d "
                "stand %.0f,%.0f,%.0f route %d\n",
                cell.x, cell.y, base.x, base.y, base.z,
                stand.x, stand.y, stand.z, int(s.moveActive));
    return true;
}

// Console back-edge (/harvest — see commands.h): plain ints keep glm out of
// that header.
bool startHarvestSessionAt(GameState& s, int cellX, int cellZ) {
    return startHarvestSession(s, glm::ivec2(cellX, cellZ));
}

void cancelChop(GameState& s) {
    if (s.chopPhase == 1) s.moveActive = false;
    s.chopPhase = 0;
}

// Cursor in FRAMEBUFFER pixels — the space the menu shader draws in (window
// and framebuffer sizes differ under DPI scaling; same convention as
// camButtonHit).
static glm::vec2 cursorFbPos(GLFWwindow* win, double cx, double cy) {
    int winW = 0, winH = 0, fbW = 0, fbH = 0;
    glfwGetWindowSize(win, &winW, &winH);
    glfwGetFramebufferSize(win, &fbW, &fbH);
    return glm::vec2(float(cx) * (winW ? float(fbW) / float(winW) : 1.0f),
                     float(cy) * (winH ? float(fbH) / float(winH) : 1.0f));
}

// Context-menu geometry (mirrors ctxmenu.frag / FrameView CTX_* constants).
// ctxPos and the hit test both live in framebuffer pixels.
static int ctxRowHit(const GameState& s, GLFWwindow* win, double cx, double cy) {
    glm::vec2 p = cursorFbPos(win, cx, cy);
    float lx = p.x - s.ctxPos.x;
    float ly = p.y - s.ctxPos.y;
    if (lx < 0.0f || lx > FrameView::CTX_W) return -1;
    ly -= FrameView::CTX_HDR;
    if (ly < 0.0f) return -1;
    int row = int(ly / FrameView::CTX_ROW);
    return (row >= 0 && row < 2) ? row : -1;
}

// A resolved LMB CLICK over the diorama (not a drag): activate the open menu,
// or pick a tree under the cursor and open one.
static void handleMapClick(GameState& s, GLFWwindow* win) {
    double cx = s.lastMouseX, cy = s.lastMouseY;   // callback-tracked cursor
    // Fly / free-cam / cave X-ray first: those modes get no forestry
    // interactions at all — in X-ray the surface trees render as ghosts —
    // (a stale open menu must not fire an action either).
    if (s.player.fly || s.mapView.freeCam || s.mapView.xray) {
        s.ctxOpen = false;
        return;
    }
    if (s.ctxOpen) {
        int row = ctxRowHit(s, win, cx, cy);
        s.ctxOpen = false;
        if (row == 0) {
            if      (s.ctxKind == 1) startMineSession(s, s.ctxRock);
            else if (s.ctxKind == 2) startHarvestSession(s, s.ctxCell);
            else                     startChopSession(s, s.ctxTree);
        }
        return;                       // row 1 = Cancel; outside = close
    }
    RayHit h = mapCursorPick(s, win, 6000.0f);
    if (!h.hit) return;
    uint8_t m = s.world.getVoxel(h.voxel.x, h.voxel.y, h.voxel.z);
    glm::ivec2 cropCell = farmCellOrigin(h.voxel.x, h.voxel.z);
    if (farmHasRipeCropAt(s.farm, cropCell)) {
        s.ctxCell = cropCell;                                   // ripe crop -> Harvest
        s.ctxKind = 2;
    } else if (m == MAT_STONE) {
        RockInfo r;
        if (!collectRock(s.world, s.gen, h.voxel, r)) return;   // strata: no menu
        s.ctxRock = std::move(r);
        s.ctxKind = 1;
    } else {
        if (m != MAT_WOOD && m != MAT_DESERT_WOOD && m != MAT_CHAR &&
            m != MAT_LEAVES && m != MAT_CACTUS_FLESH && m != MAT_CACTUS_BLOOM &&
            m != MAT_SHRUB_SAGE)
            return;
        TreeInfo t;
        if (!collectTree(s.world, s.gen, h.voxel, t)) return;
        if (t.isStump) return;  // stumps/char clusters: no interaction (for now)
        s.ctxTree = t;
        s.ctxKind = 0;
    }
    // Anchor so the ACTION row sits under the cursor (one less mouse move on
    // repeat chops, the OSRS way), clamped to the framebuffer.
    int fbW = 0, fbH = 0;
    glfwGetFramebufferSize(win, &fbW, &fbH);
    glm::vec2 fb = cursorFbPos(win, cx, cy);
    float px = fb.x - FrameView::CTX_W * 0.5f;
    float py = fb.y - FrameView::CTX_HDR - FrameView::CTX_ROW * 0.5f;
    float mh = FrameView::CTX_HDR + FrameView::CTX_ROW * 2.0f;
    s.ctxPos = glm::vec2(
        glm::clamp(px, 4.0f, std::max(4.0f, float(fbW) - FrameView::CTX_W - 4.0f)),
        glm::clamp(py, 4.0f, std::max(4.0f, float(fbH) - mh - 4.0f)));
    s.ctxHover = ctxRowHit(s, win, cx, cy);
    s.ctxOpen = true;
}

// Is this voxel inside the current hover mask?
static bool hovContains(const GameState& s, glm::ivec3 v) {
    if (!s.hovActive) return false;
    glm::ivec3 c = v - s.hovLo;
    if (c.x < 0 || c.y < 0 || c.z < 0 ||
        c.x >= s.hovDims.x || c.y >= s.hovDims.y || c.z >= s.hovDims.z)
        return false;
    return s.hovMask[size_t(c.x) + size_t(s.hovDims.x) *
                     (size_t(c.y) + size_t(s.hovDims.y) * c.z)] != 0;
}

// Tree hover: pick under the cursor each frame (sub-ms), identify the tree on
// the forestry worker thread (~10-40 ms happens OFF-frame), ease a highlight
// in when the cursor sits on a fellable tree, and swap to the pointing-hand
// cursor. RS rule: the affordance must never cost a frame.
static void tickTreeHover(GameState& s, GLFWwindow* win, float dt) {
    bool suppress = s.mapView.freeCam || s.mapView.xray ||
                    s.player.fly || s.farm.mode || s.leftDown || s.rightDown ||
                    s.paused || s.console.open ||
                    s.invOpen || s.tuneOpen ||
                    (s.ctxOpen && s.ctxKind == 2);   // crop Harvest menu: no tree glow
    bool want = false;
    if (!suppress) {
        RayHit h = mapCursorPick(s, win, 6000.0f);
        if (h.hit) {
            uint8_t m = s.world.getVoxel(h.voxel.x, h.voxel.y, h.voxel.z);
            bool treeMat = m == MAT_WOOD || m == MAT_DESERT_WOOD ||
                           m == MAT_LEAVES || m == MAT_CACTUS_FLESH ||
                           m == MAT_CACTUS_BLOOM || m == MAT_SHRUB_SAGE;
            if (treeMat) {
                if (hovContains(s, h.voxel)) {
                    want = true;                    // still on the same tree
                } else if (h.voxel != s.hovPickVox) {
                    s.hovPickVox = h.voxel;         // coalesced async request
                    treeHoverRequest(s.forestry.hover, s.gen, s.world, h.voxel);
                }
            } else if (m == MAT_STONE) {
                // Boulders identify synchronously: generator replay + bounded
                // flood, at most once per hovered voxel — the hovPickVox
                // latch also caches the REJECTION for strata / hillside
                // stone, so sweeping the cursor across a mountain face
                // doesn't re-identify every frame. Not while a menu is up:
                // the ray passing beside the panel must not steal the mask
                // from the rock/tree the menu was opened on.
                if (hovContains(s, h.voxel)) {
                    want = true;                    // still on the same rock
                } else if (h.voxel != s.hovPickVox && !s.ctxOpen) {
                    s.hovPickVox = h.voxel;
                    RockInfo r;
                    if (collectRock(s.world, s.gen, h.voxel, r)) {
                        s.hovActive = true;
                        s.hovLo   = r.lo;
                        s.hovDims = r.hi - r.lo + 1;
                        s.hovMask.assign(size_t(s.hovDims.x) * s.hovDims.y *
                                         s.hovDims.z, 0);
                        for (auto& v : r.stone) {
                            glm::ivec3 c = v - s.hovLo;
                            s.hovMask[size_t(c.x) + size_t(s.hovDims.x) *
                                      (size_t(c.y) + size_t(s.hovDims.y) * c.z)]
                                = 255;
                        }
                        s.renderer.updateHoverMask(s.hovDims, s.hovMask.data());
                        want = true;
                    }
                }
            }
        }
    }
    // While a TREE/ROCK menu is up the cursor sits over the panel: hold the
    // highlight. A crop (Harvest) menu owns no tree — don't re-light the last one.
    if (s.ctxOpen && s.ctxKind != 2 && s.hovActive) want = true;

    // Adopt a finished identification: gate out stumps/flat remnants and
    // spot-check a sample of its wood against the LIVE world (the worker's
    // flood fallback reads World off-thread — an accepted race; this gate
    // rejects torn or stale results) — a mostly-gone tree isn't clickable.
    TreeHoverResult r;
    if (treeHoverPoll(s.forestry.hover, r)) {
        bool ok = r.found && r.woodN >= 40 && r.woodH > 8;
        if (ok && !r.woodSample.empty()) {
            int alive = 0;
            for (auto& v : r.woodSample) {
                uint8_t m = s.world.getVoxel(v.x, v.y, v.z);
                if (r.isCactus ? (m == MAT_CACTUS_FLESH)
                               : (m == MAT_WOOD || m == MAT_DESERT_WOOD))
                    ++alive;
            }
            ok = alive * 10 >= int(r.woodSample.size()) * 3;
        }
        // Stale-answer gate: adopt only if the answer covers the voxel that
        // drove the LATEST request (hovPickVox). A tree answer landing after
        // the cursor moved onto a rock (sync identify re-latched hovPickVox)
        // or onto another tree would otherwise steal the mask — and then
        // dead-latch the affordance on the old voxel, since hovContains
        // misses and the latch suppresses re-identification.
        if (ok) {
            glm::ivec3 c = s.hovPickVox - r.lo;
            bool covers = c.x >= 0 && c.y >= 0 && c.z >= 0 &&
                          c.x < r.dims.x && c.y < r.dims.y && c.z < r.dims.z &&
                          r.mask[size_t(c.x) + size_t(r.dims.x) *
                                 (size_t(c.y) + size_t(r.dims.y) * c.z)] != 0;
            ok = covers;
        }
        if (ok) {
            s.hovActive = true;
            s.hovLo   = r.lo;
            s.hovDims = r.dims;
            s.hovMask = std::move(r.mask);
            s.renderer.updateHoverMask(r.dims, s.hovMask.data());
        }
    }

    float target = (want && s.hovActive) ? 1.0f : 0.0f;
    float step = dt / (target > 0.5f ? 0.10f : 0.16f);
    s.hovStrength = glm::clamp(s.hovStrength +
                               glm::clamp(target - s.hovStrength, -step, step),
                               0.0f, 1.0f);
    // Cursor affordance: record the desire only — the OS pointing hand has
    // ONE owner (the apply block after this tick), shared with the
    // god-harvest ripe-crop hover. Setting it here every frame stomped the
    // crop hand the frame after its change-triggered set.
    s.hovHandWant = (want && s.hovStrength > 0.25f) || s.ctxOpen;
}

// Drop the highlight (the world under it changed wholesale: a fell, an
// uproot — cheap and unconditional beats a staleness proof).
void clearTreeHover(GameState& s) {
    s.hovActive = false;
    s.hovStrength = 0.0f;
    s.hovPickVox = glm::ivec3(INT32_MIN, 0, 0);
}

// Farm Mode toolbar hit-test (bottom-centre): the tool bar, plus the SEED bar
// floating above it while the Plant tool is armed. Returns the tool slot 0..2
// under the cursor or -1; *seedOut (may be null) gets the seed slot
// (CropSpecies) or -1. Geometry MUST match farmbar.frag / drawFarmBar
// (tool bar: 36 cols of 12/slot; seed bar: 8 cols per species).
static int farmBarHit(GLFWwindow* win, double cx, double cy, int tool,
                      int* seedOut = nullptr) {
    if (seedOut) *seedOut = -1;
    int winW, winH, fbW, fbH;
    glfwGetWindowSize(win, &winW, &winH);
    glfwGetFramebufferSize(win, &fbW, &fbH);
    float px = float(cx) * (winW ? float(fbW) / float(winW) : 1.0f);
    float py = float(cy) * (winH ? float(fbH) / float(winH) : 1.0f);
    const float CW = 16.0f, PAD = 8.0f, MARGIN = 164.0f, H = 32.0f;   // above the hotbar tray
    const float W = 36.0f * CW + 2.0f * PAD;
    float x0 = (float(fbW) - W) * 0.5f;
    float y0 = float(fbH) - H - MARGIN;
    if (px >= x0 && px < x0 + W && py >= y0 && py < y0 + H) {
        float tx = px - x0 - PAD;
        if (tx < 0.0f || tx >= 36.0f * CW) return -1;
        return int(tx / CW) / 12;
    }
    // Seed bar: one 8-col slot per species, 6 px above the tool bar.
    if (tool == FT_PLANT && seedOut) {
        const float SW = float(CROP_SPECIES_N) * 8.0f * CW + 2.0f * PAD;
        float sx0 = (float(fbW) - SW) * 0.5f;
        float sy0 = y0 - H - 6.0f;
        if (px >= sx0 && px < sx0 + SW && py >= sy0 && py < sy0 + H) {
            float tx = px - sx0 - PAD;
            if (tx >= 0.0f && tx < float(CROP_SPECIES_N) * 8.0f * CW)
                *seedOut = int(tx / CW) / 8;
        }
    }
    return -1;
}

// Is the cursor over ANY farm-bar surface (tool bar or live seed bar)?
static bool farmBarOver(GLFWwindow* win, double cx, double cy, int tool) {
    int seed = -1;
    return farmBarHit(win, cx, cy, tool, &seed) >= 0 || seed >= 0;
}

// Inventory satchel hit-test. Returns the bag slot 0..BAG-1 under the cursor
// (or -1), and sets `tidy` when over the Tidy button. Geometry MUST match
// bag.frag (top-origin framebuffer pixels).
static int bagHitTest(GLFWwindow* win, double cx, double cy, bool& tidy) {
    tidy = false;
    int winW, winH, fbW, fbH;
    glfwGetWindowSize(win, &winW, &winH);
    glfwGetFramebufferSize(win, &fbW, &fbH);
    float px = float(cx) * (winW ? float(fbW) / float(winW) : 1.0f);
    float py = float(cy) * (winH ? float(fbH) / float(winH) : 1.0f);
    const float CELL = 64.0f, GAP = 8.0f, PAD = 22.0f, HEADER = 34.0f,
                FOOTER = 30.0f, TIDYW = 88.0f, TIDYH = 24.0f;
    const int   COLS = 8, ROWS = 3;
    const float GRIDW = COLS * CELL + (COLS - 1) * GAP;
    const float GRIDH = ROWS * CELL + (ROWS - 1) * GAP;
    const float CARDW = GRIDW + 2.0f * PAD;
    const float CARDH = PAD + HEADER + GRIDH + FOOTER + PAD;
    float cardX = std::floor((float(fbW) - CARDW) * 0.5f);
    float cardY = std::floor((float(fbH) - CARDH) * 0.5f);
    float tx = cardX + CARDW - PAD - TIDYW;
    float ty = cardY + PAD + (HEADER - TIDYH) * 0.5f;
    if (px >= tx && px < tx + TIDYW && py >= ty && py < ty + TIDYH) { tidy = true; return -1; }
    float gridX = cardX + PAD, gridY = cardY + PAD + HEADER;
    float rx = px - gridX, ry = py - gridY;
    if (rx < 0.0f || ry < 0.0f || rx >= GRIDW || ry >= GRIDH) return -1;
    float pitch = CELL + GAP;
    int col = int(rx / pitch), vrow = int(ry / pitch);
    if (col < 0 || col >= COLS || vrow < 0 || vrow >= ROWS) return -1;
    if (rx - col * pitch >= CELL || ry - vrow * pitch >= CELL) return -1;   // in a gap
    int row = (ROWS - 1) - vrow;   // row 0 = hotbar = bottom
    return row * COLS + col;
}

// =============================================================================
// Baked produce volumes — the whole vegetable that flies into the bag on
// harvest, one model per (species, grade) so a small radish flies chunky-and-
// small and a giant carrot flies as the whole glorious root the soil was
// hiding. Rendered through the creature-part pipeline (draws in the diorama);
// baked ONCE into the creature atlas at load, instanced per flyer.
// =============================================================================

// The atlas palette is sRGB bytes (the shader does sRGB->linear). Derive the
// colours from the SAME material table the world renders, so a flyer matches
// its in-field crop instead of reading over-saturated.
static uint32_t produceSrgb8(const glm::vec3& lin) {
    auto enc = [](float c) {
        c = std::clamp(c, 0.0f, 1.0f);
        float s = (c <= 0.0031308f) ? 12.92f * c : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
        return uint32_t(std::lround(std::clamp(s, 0.0f, 1.0f) * 255.0f));
    };
    return enc(lin.r) | (enc(lin.g) << 8) | (enc(lin.b) << 16) | (255u << 24);
}

static void produceBegin(CreatureModel& m, int W, int H, int D) {
    m = CreatureModel{};
    m.cellScale = 1.0f;
    CreaturePart p{};
    p.dims = glm::ivec3(W, H, D);
    p.wordBase = 0;                    // file-local; rebased in appendModelToAtlas
    m.parts.push_back(p);
    m.vox.assign(size_t((W * H * D + 3) / 4), 0u);
}

static void produceSet(CreatureModel& m, int x, int y, int z, uint8_t v) {
    const glm::ivec3 d = m.parts[0].dims;
    if (x < 0 || y < 0 || z < 0 || x >= d.x || y >= d.y || z >= d.z) return;
    int lin = x + d.x * (y + d.y * z);
    uint32_t& w  = m.vox[size_t(lin >> 2)];
    int       sh = (lin & 3) * 8;
    w = (w & ~(0xFFu << sh)) | (uint32_t(v) << sh);
}

// Build the (species, grade) produce volume. Palette: 1 = flesh, 2 = greens,
// 3 = accent (corn husk / radish taproot tip).
// SYNC CONTRACT: shapes/constants/accent colours are hand-mirrored by the
// icon raytracer (shaders/lib/icon.glsl produceOcc + produceBounds) across
// the C++/GLSL boundary — retunes here must be repeated there.
static void buildProduceModel(CreatureModel& m, int species, int grade) {
    const glm::vec3 flesh = g_materials[CROP_DEFS[species].flesh].color;
    const glm::vec3 vine  = g_materials[MAT_CROP_VINE].color;
    const glm::vec3 leafG = g_materials[MAT_CROP_LEAF].color;
    grade = std::clamp(grade, 0, 2);

    switch (species) {
        case CROP_CARROT: {
            // The reveal: a long tapered root, point down, frond crown on top.
            const int L = (grade == 2) ? 9 : (grade == 1) ? 6 : 4;
            const float maxR = (grade == 2) ? 1.5f : (grade == 1) ? 1.1f : 0.9f;
            produceBegin(m, 3, L + 2, 3);
            m.palette[0] = produceSrgb8(flesh);
            m.palette[1] = produceSrgb8(leafG);
            for (int y = 0; y < L; ++y) {
                float rr = maxR * float(y + 1) / float(L);   // y0 = the tip
                for (int dx = -1; dx <= 1; ++dx)
                    for (int dz = -1; dz <= 1; ++dz)
                        if (float(dx * dx + dz * dz) <= rr * rr + 0.01f)
                            produceSet(m, 1 + dx, y, 1 + dz, 1);
            }
            produceSet(m, 1, L, 1, 2);                    // frond crown
            produceSet(m, 1, L + 1, 1, 2);
            produceSet(m, 0, L, 1, 2);
            produceSet(m, 2, L, 1, 2);
            break;
        }
        case CROP_TOMATO: {
            // A plump oblate fruit + stem nub (grade mostly drives COUNT; the
            // giant tier also plumps up).
            const int r = (grade == 2) ? 2 : 1;
            const float rr = (grade == 0) ? 1.2f : (grade == 1) ? 1.45f : 2.4f;
            produceBegin(m, 2 * r + 1, 2 * r + 2, 2 * r + 1);
            m.palette[0] = produceSrgb8(flesh);
            m.palette[1] = produceSrgb8(vine);
            for (int dx = -r; dx <= r; ++dx)
                for (int dz = -r; dz <= r; ++dz)
                    for (int dy = -r; dy <= r; ++dy) {
                        float ey = float(dy) / 0.85f;   // oblate: squashed poles
                        if (float(dx * dx + dz * dz) + ey * ey <= rr * rr)
                            produceSet(m, r + dx, r + dy, r + dz, 1);
                    }
            produceSet(m, r, 2 * r + 1, r, 2);            // stem nub
            break;
        }
        case CROP_CABBAGE: {
            // A wrapped head, wider than tall, with loose outer leaves.
            // Tight box: content tops out at y = 2*ry (see the pumpkin note).
            const int r  = grade + 1;
            const int ry = std::max(1, r - 1);
            produceBegin(m, 2 * r + 1, 2 * ry + 1, 2 * r + 1);
            m.palette[0] = produceSrgb8(flesh);
            m.palette[1] = produceSrgb8(leafG);
            for (int dx = -r; dx <= r; ++dx)
                for (int dz = -r; dz <= r; ++dz)
                    for (int dy = 0; dy <= 2 * ry; ++dy) {
                        float ex = float(dx) / (r + 0.5f);
                        float ez = float(dz) / (r + 0.5f);
                        float ey = (float(dy) - ry) / (ry + 0.5f);
                        if (ex * ex + ez * ez + ey * ey <= 1.0f)
                            produceSet(m, r + dx, dy, r + dz, 1);
                    }
            if (grade >= 1) {                             // loose wrap leaves
                produceSet(m, 0,     ry, r, 2);
                produceSet(m, 2 * r, ry, r, 2);
                produceSet(m, r, ry, 0, 2);
            }
            break;
        }
        case CROP_CORN: {
            // A husked cob: golden kernel column in a leaf cup.
            const int h = (grade == 2) ? 6 : (grade == 1) ? 4 : 3;
            produceBegin(m, 3, h + 2, 3);
            m.palette[0] = produceSrgb8(flesh);
            m.palette[1] = produceSrgb8(leafG);
            m.palette[2] = produceSrgb8(glm::vec3(0.72f, 0.78f, 0.52f));  // pale husk
            for (int y = 1; y <= h; ++y) {
                produceSet(m, 1, y, 1, 1);                // core
                if (grade >= 1) {                          // plump plus-cross cob
                    produceSet(m, 0, y, 1, 1);
                    produceSet(m, 2, y, 1, 1);
                    produceSet(m, 1, y, 0, 1);
                    produceSet(m, 1, y, 2, 1);
                }
            }
            produceSet(m, 1, h + 1, 1, 1);                // rounded tip
            produceSet(m, 1, 0, 1, 3);                    // husk cup
            produceSet(m, 0, 0, 1, 3);
            produceSet(m, 2, 0, 1, 3);
            produceSet(m, 1, 0, 0, 3);
            produceSet(m, 1, 0, 2, 3);
            break;
        }
        case CROP_RADISH: {
            // A round bulb with a cream taproot tip below and a leaf tuft above.
            const int r = (grade == 2) ? 2 : 1;
            const float rr = (grade == 0) ? 1.2f : (grade == 1) ? 1.45f : 2.4f;
            produceBegin(m, 2 * r + 1, 2 * r + 4, 2 * r + 1);
            m.palette[0] = produceSrgb8(flesh);
            m.palette[1] = produceSrgb8(leafG);
            m.palette[2] = produceSrgb8(glm::vec3(0.92f, 0.88f, 0.78f));  // cream tip
            for (int dx = -r; dx <= r; ++dx)
                for (int dz = -r; dz <= r; ++dz)
                    for (int dy = -r; dy <= r; ++dy)
                        if (float(dx * dx + dz * dz + dy * dy) <= rr * rr)
                            produceSet(m, r + dx, r + 1 + dy, r + dz, 1);
            produceSet(m, r, 0, r, 3);                    // taproot tip
            produceSet(m, r, 2 * r + 2, r, 2);            // leaf tuft
            produceSet(m, r, 2 * r + 3, r, 2);
            break;
        }
        default: {
            // Pumpkin — voxel-for-voxel the SAME ellipsoid cropVoxels grows in
            // the ground (minus the leaf skirt), orange gourd + green stem.
            // Box height is TIGHT (fruit 0..h + stem at h+1): the flyer is
            // positioned by box centre, so slack pushes the fruit down.
            const int r = grade + 1;
            const int h = r + 1;                 // plump: a touch taller than half
            produceBegin(m, 2 * r + 1, h + 2, 2 * r + 1);
            m.palette[0] = produceSrgb8(flesh);
            m.palette[1] = produceSrgb8(vine);
            for (int dx = -r; dx <= r; ++dx)
                for (int dz = -r; dz <= r; ++dz)
                    for (int dy = 0; dy <= h; ++dy) {
                        float ex = float(dx) / (r + 0.7f);
                        float ez = float(dz) / (r + 0.7f);
                        float ey = (float(dy) - h * 0.5f) / (h * 0.62f);
                        if (ex * ex + ez * ez + ey * ey <= 1.0f)
                            produceSet(m, r + dx, dy, r + dz, 1);
                    }
            produceSet(m, r, h + 1, r, 2);       // stem
            break;
        }
    }
}

// Pluck the ripe crop in `cell`: award its whole graded produce and launch the
// species' COLLECT-FLYER theatrics — roots get yanked out of the soil, tomato
// fruits pop off one by one, the cabbage head rolls, corn ears tumble, the
// pumpkin does its classic loft-and-swoop. Items bank on ARRIVAL (the flyer
// update loop's landing branch).
// The generic "collect a produce object" path — reused by the walk-to-plot
// pluck session and the god-mode farm tool.
static bool collectProduce(GameState& s, glm::ivec2 cell) {
    EditDelta d;
    Crop c;
    if (!farmHarvest(s.world, s.farm, cell, c, d)) return false;
    s.world.uploadDelta(d);
    const ItemId item = itemForCrop(c.species);
    const glm::vec3 rootC(float(c.root.x) + 0.5f, float(c.root.y) + 1.0f,
                          float(c.root.z) + 0.5f);        // soil surface centre
    // Foliage always rips loose.
    s.particles.spawnDust(rootC + glm::vec3(0.0f, 1.5f, 0.0f), MAT_CROP_LEAF, 6);

    int n = 0;
    auto push = [&](glm::vec3 start, int style, float delay, float dur) {
        CollectFlyer fl;
        fl.start   = start;
        fl.item    = item;
        fl.data    = uint16_t(c.size);
        fl.grade   = c.size;
        fl.species = c.species;
        fl.style   = style;
        fl.delay   = delay;
        fl.delay0  = delay;
        fl.dur     = dur;
        fl.rng     = c.genome ^ (uint32_t(n++) * 0x9E3779B9u);
        fl.groundY = float(c.root.y) + 1.0f;
        s.collectFlyers.push_back(fl);
    };

    switch (c.species) {
        case CROP_CARROT:
        case CROP_RADISH:
            // The whole buried root gets YANKED out (starts underground; the
            // flyer update fires the dirt burst the moment it breaks the soil).
            push(rootC, FLY_YANK, 0.0f, c.species == CROP_RADISH ? 0.9f : 1.1f);
            break;
        case CROP_TOMATO:
        case CROP_CORN: {
            // One flyer per fruit/ear, plucked off the plant in sequence, each
            // from its ACTUAL spot on the bush/stalk: cropVoxels regenerates
            // the pure shape (with materials), then only voxels the stamp
            // actually placed (c.occupied — terrain or a neighbour may have
            // blocked some) count. What you saw hanging is what you harvest.
            const uint8_t flesh = CROP_DEFS[c.species].flesh;
            std::vector<std::pair<glm::ivec3, uint8_t>> vox;
            cropVoxels(c, vox);
            auto stamped = [&](const glm::ivec3& v) {
                for (const glm::ivec3& o : c.occupied)
                    if (o == v) return true;
                return false;
            };
            std::vector<glm::ivec3> spots;
            for (auto& pm : vox) {
                if (pm.second != flesh || !stamped(pm.first)) continue;
                bool dup = false;                // corn ears are 2-tall columns
                for (auto& sp : spots)
                    if (sp.x == pm.first.x && sp.z == pm.first.z) { dup = true; break; }
                if (!dup) spots.push_back(pm.first);
            }
            if (spots.empty()) spots.push_back(c.root + glm::ivec3(0, 2, 0));
            const bool corn = (c.species == CROP_CORN);
            for (size_t i = 0; i < spots.size(); ++i)
                push(glm::vec3(spots[i]) + glm::vec3(0.5f),
                     corn ? FLY_TUMBLE : FLY_POP, 0.14f * float(i),
                     corn ? 0.85f : 0.62f);
            if (corn)   // the stripped stalk collapses into leaf confetti
                s.particles.spawnDust(rootC + glm::vec3(0.0f, 3.0f, 0.0f),
                                      MAT_CROP_LEAF, 10);
            break;
        }
        case CROP_CABBAGE:
            // The cut head rolls off the plot, bouncing, then leaps into the bag.
            push(rootC + glm::vec3(0.0f, 1.0f, 0.0f), FLY_ROLL, 0.0f, 1.35f);
            break;
        default:
            // Pumpkin: the classic loft-and-swoop.
            push(rootC + glm::vec3(0.0f, 2.0f, 0.0f), FLY_ARC, 0.0f, 0.72f);
            break;
    }
    return true;
}

// Perform a farm tool on one plot cell (god-mode: acts at range, no walking).
// Called from the hold-drag paint loop. Batches its own upload.
static void farmActCell(GameState& s, int tool, glm::ivec2 cell, int y) {
    if (tool == FT_HARVEST) { collectProduce(s, cell); return; }
    EditDelta d;
    if (tool == FT_TILL)  farmTill(s.world, s.farm, cell, y, d);
    else                  farmPlant(s.world, s.farm, cell, y, d);
    s.world.uploadDelta(d);
}

static void mouseBtnCb(GLFWwindow* win, int btn, int action, int) {
    auto& s = *g_state;
    // Boot gate: the loading screen pumps events while the gen thread is
    // still writing the world. Both click paths below pick against the
    // world (handleMapClick / the RMB move order), so nothing may pass
    // until generation finishes. No press can arm during load, so skipping
    // the release bookkeeping here is safe too.
    if (!s.worldReady) return;
    // Releases FIRST, before any modal gate: a release swallowed by pause/
    // console latches the button state — and a latched rightDown is
    // a character that chases the cursor with no button held.
    if (action == GLFW_RELEASE) {
        if (btn == GLFW_MOUSE_BUTTON_LEFT) {
            // A press that never left the 4px dead zone is a CLICK — the
            // diorama's tree-pick / context-menu interaction.
            bool wasClick = s.clickArmed && !s.paused &&
                            !s.console.open && !s.tuneOpen;
            s.leftDown = false;
            s.tuneDrag = false;
            s.clickArmed = false;
            if (wasClick) handleMapClick(s, win);
        }
        if (btn == GLFW_MOUSE_BUTTON_RIGHT) {
            s.rightDown = false;
            s.farm.painting = false;   // end hold-drag painting
        }
        return;
    }
    if (s.paused || s.console.open) return;
    if (s.tuneOpen) {
        // LMB = select / drag a slider. Gameplay clicks are fully
        // suppressed while the panel is open.
        if (btn == GLFW_MOUSE_BUTTON_LEFT) {
            double cx, cy; glfwGetCursorPos(win, &cx, &cy);
            s.tuneDrag = true;
            tunePointer(cx, cy, /*press=*/true);
        }
        return;
    }
    // Inventory satchel: while open it OWNS the pointer (world clicks are
    // suppressed, so a click never leaks to orbit / move / dig). LMB tidies or
    // brings a stack to hand; a click on empty space just does nothing (the
    // panel stays open — Esc/I closes).
    if (s.invOpen) {
        if (btn == GLFW_MOUSE_BUTTON_LEFT) {
            double cx, cy; glfwGetCursorPos(win, &cx, &cy);
            bool tidy = false;
            int slot = bagHitTest(win, cx, cy, tidy);
            if (tidy) {
                s.hotbar.tidy();
            } else if (slot >= 0) {
                if (slot < Hotbar::COLS) {
                    s.hotbar.selected = slot;               // arm this hotbar slot
                } else {
                    // Bring a storage stack to hand: swap it into the armed slot.
                    std::swap(s.hotbar.bag.slots[slot],
                              s.hotbar.bag.slots[s.hotbar.selected]);
                }
            }
        }
        s.leftDown = s.rightDown = false;
        return;
    }
    // Minimap (top right): LEFT on the orb toggles free-cam; LEFT on the
    // terrain disc orders a walk there (or, in free-cam, pans the camera
    // there). ANY press over the minimap footprint is swallowed — and the
    // drag flags reset on a handled LEFT — so orbit drags and world move
    // orders can't fire through the HUD.
    {
        double cx, cy;
        glfwGetCursorPos(win, &cx, &cy);
        glm::vec2 wxz;
        int hit = minimapHit(win, cx, cy, &wxz);
        if (hit != 0) {
            if (btn == GLFW_MOUSE_BUTTON_LEFT) {
                if (hit == 2) {
                    s.mapView.freeCam = !s.mapView.freeCam;
                } else if (hit == 1) {
                    if (s.mapView.freeCam) {
                        // Jump the look-around offset so the camera target
                        // lands on the clicked spot (same fence as pan()).
                        glm::vec2 off = wxz - glm::vec2(s.mapView.followPos.x,
                                                        s.mapView.followPos.z);
                        s.mapView.panOffset = glm::clamp(off,
                            glm::vec2(-MapView::PAN_R), glm::vec2(MapView::PAN_R));
                    } else {
                        minimapMoveOrder(s, wxz);
                    }
                }
                s.leftDown = s.rightDown = false;
                s.ctxOpen = false;   // the button eats the press — no stale menu
                cancelChop(s);       // free-cam mode drops the session
            }
            return;
        }
    }
    // Farm Mode input: LMB on a toolbar slot selects the tool; RMB issues a
    // farm job at the cursor cell (walk there, then act) instead of a plain
    // move. LMB off the toolbar still orbits the camera (falls through).
    // Forestry never races these clicks: entering Farm Mode dismissed any
    // open menu/chop, and the clickArmed forestry click below is skipped.
    if (s.farm.mode && action == GLFW_PRESS) {
        double cx, cy; glfwGetCursorPos(win, &cx, &cy);
        if (btn == GLFW_MOUSE_BUTTON_LEFT) {
            int seed = -1;
            int slot = farmBarHit(win, cx, cy, s.farm.tool, &seed);
            if (slot >= 0) { s.farm.tool = slot; return; }
            if (seed >= 0) { s.farm.seedSel = seed; return; }
        }
        if (btn == GLFW_MOUSE_BUTTON_RIGHT && !s.mapView.freeCam &&
            !s.player.fly) {
            if (!farmBarOver(win, cx, cy, s.farm.tool)) {
                // Begin hold-drag painting; the main loop acts on each new cell
                // (including this one, since lastActedCell is reset).
                s.farm.painting = true;
                s.farm.lastActedCell = glm::ivec2(INT_MIN, INT_MIN);
            }
            return;   // RMB in Farm Mode never falls through to a plain move
        }
    }
    // (RMB instant-harvest removed: harvesting a ripe crop in normal play is now
    // the LMB "Choose Option" -> Harvest walk-to-plot session. Farm Mode's
    // toolbar harvest is untouched. RMB here falls through to the normal order.)
    if (btn == GLFW_MOUSE_BUTTON_LEFT) {
        s.leftDown = true;
        if (!s.farm.mode) {
            s.pressX = s.lastMouseX;   // callback-tracked press anchor
            s.pressY = s.lastMouseY;
            s.clickArmed = true;
        }
    }
    if (btn == GLFW_MOUSE_BUTTON_RIGHT) {
        s.rightDown = true;
        // A fresh move order dismisses the menu and abandons the chop.
        if (!s.mapView.freeCam) {
            s.ctxOpen = false;
            cancelChop(s);
        }
    }
}

static void scrollCb(GLFWwindow* win, double /*xoff*/, double yoff) {
    auto& s = *g_state;
    if (s.paused || s.console.open) return;
    if (s.tuneOpen) {                          // wheel = fine-adjust selected row
        if ((s.tunePage == 1 || s.tunePage == 2) && s.tuneSel == 0) {
            int n = (s.tunePage == 1) ? MAT_LIST_N : int(s.creatures.defs.size());
            int& idx = (s.tunePage == 1) ? s.tuneMatIdx : s.tuneSpeciesIdx;
            if (n > 0) idx = (idx + (yoff > 0 ? 1 : n - 1)) % n;
            return;
        }
        float mn, mx, def; float* v = tuneRowValue(s, s.tuneSel, mn, mx, def);
        if (v) {
            float d = float(yoff) * (mx - mn) * 0.01f;
            if (s.tunePage == 2 && s.tuneSel >= 1 && CREATURE_TUNE[s.tuneSel - 1].snap)
                d = (yoff > 0) ? std::max(d, 1.0f) : std::min(d, -1.0f);
            *v = std::clamp(*v + d, mn, mx);
            creatureTuneEdited(s);
        }
        return;
    }
    // Scroll zooms. Over the minimap the wheel zooms the MINIMAP scale
    // instead of the orbit camera (RS minimap zoom).
    double cx, cy;
    glfwGetCursorPos(win, &cx, &cy);
    if (minimapHit(win, cx, cy, nullptr) != 0) {
        s.minimapZoom = glm::clamp(
            s.minimapZoom * std::pow(1.12f, float(-yoff)), 1.0f, 5.0f);
        return;
    }
    // Otherwise the wheel adjusts the cam.dist tweak itself, so wheel + the
    // /tune slider drive the SAME value (no
    // fight — the camera reads cam.dist every frame).
    float& cd = s.tweak[TW_CAM_DIST];
    cd = glm::clamp(cd * std::pow(0.9f, float(yoff)),
                    TWEAKS[TW_CAM_DIST].mn, TWEAKS[TW_CAM_DIST].mx);
}

// =============================================================================
// Tick: player physics. Skipped while paused.
// Horizontal move feel: exponential approach rates (per second) toward the
// target velocity, instead of snapping. High on the ground for responsiveness,
// low in the air so a jump keeps its momentum (you can only nudge trajectory).
// Decel (no input) a touch higher than accel for a crisp but non-instant stop.
static constexpr float GROUND_ACCEL = 16.0f;
static constexpr float GROUND_DECEL = 18.0f;
static constexpr float AIR_ACCEL    = 4.0f;
static constexpr float AIR_DECEL    = 0.8f;

// =============================================================================
static void tickPlayer(GameState& s, GLFWwindow* win, float dt) {
    Player& p = s.player;

    bool sprint = glfwGetKey(win, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS;
    bool crouch = !p.fly && (glfwGetKey(win, GLFW_KEY_LEFT_SHIFT)  == GLFW_PRESS ||
                             glfwGetKey(win, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

    // Crouch (sneak): blend toward a shorter hitbox + lower eye. When releasing,
    // refuse to stand if the standing-height box would overlap (you're under a
    // low gap) -- stay crouched until there's headroom, like Minecraft.
    float crouchTarget = crouch ? 1.0f : 0.0f;
    if (crouchTarget < s.crouchT && overlapsSolid(s.world, p.pos, P_HEIGHT))
        crouchTarget = s.crouchT;                       // blocked from rising -> hold
    s.crouchT += (crouchTarget - s.crouchT) * std::min(1.0f, dt * 12.0f);
    if (s.crouchT < 0.001f) s.crouchT = 0.0f;
    float curHeight = P_HEIGHT - s.crouchT * (P_HEIGHT - CROUCH_HEIGHT);
    p.eyeOff        = P_EYE    - s.crouchT * (P_EYE    - CROUCH_EYE);

    // Crouch slows you to ~30% of walk and overrides sprint; else sprint logic.
    float speed = WALK_SPEED * (crouch ? CROUCH_MULT : (sprint ? SPRINT_MULT : 1.0f));

    if (p.fly) {
        glm::vec3 m(0);
        // Diorama fly is horizontal + Space/Ctrl vertical (a top-down camera
        // has no useful 3D look direction to fly along). The WASD basis is
        // the MAP camera's yaw, so movement is screen-relative (W flies away
        // from the camera): forward = -orbit dir = (-sin yaw, 0, -cos yaw).
        glm::vec3 baseF(-std::sin(s.mapView.yaw), 0.0f, -std::cos(s.mapView.yaw));
        glm::vec3 baseR = glm::normalize(glm::cross(baseF, glm::vec3(0, 1, 0)));
        if (glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS) m += baseF;
        if (glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS) m -= baseF;
        if (glfwGetKey(win, GLFW_KEY_D) == GLFW_PRESS) m += baseR;
        if (glfwGetKey(win, GLFW_KEY_A) == GLFW_PRESS) m -= baseR;
        if (glfwGetKey(win, GLFW_KEY_SPACE) == GLFW_PRESS) m.y += 1.0f;
        if (glfwGetKey(win, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) m.y -= 1.0f;
        if (m.x != 0.0f || m.z != 0.0f)
            p.yaw = std::atan2(m.z, m.x);   // face travel (fwd = cos/sin yaw)
        if (glm::length(m) > 0) p.pos += glm::normalize(m) * speed * 2.0f * dt;
        // Frontier gate (see the walking path for rationale).
        p.pos.x = std::clamp(p.pos.x, float(s.world.originVoxX()) + 4.0f,
                             float(s.world.originVoxX() + GRID_XZ) - 4.0f);
        p.pos.z = std::clamp(p.pos.z, float(s.world.originVoxZ()) + 4.0f,
                             float(s.world.originVoxZ() + GRID_XZ) - 4.0f);
        return;
    }

    // Click-to-move steering: follow the planned route (path.cpp) waypoint
    // by waypoint; physics (accel/friction, auto-step, slides) still owns
    // the motion — the path only picks the direction. Arrival is an XZ
    // ring; the stall bail-out lives in the main loop where actual
    // post-collision velocity is known.
    glm::vec3 wish(0);
    if (s.moveActive) {
        while (s.moveWp < int(s.movePath.size())) {
            const glm::vec3& wp = s.movePath[s.moveWp];
            glm::vec2 d(wp.x - p.pos.x, wp.z - p.pos.z);
            float distXZ = glm::length(d);
            bool last = (s.moveWp + 1 == int(s.movePath.size()));
            // Generous ring (2.2 > worst body-to-waypoint standoff when a
            // waypoint hugs an obstacle corner), and skip-ahead: physics
            // slides can carry the body past a waypoint — never walk BACK
            // to one when the next is already closer.
            if (distXZ < (last ? 2.0f : 2.2f)) { ++s.moveWp; continue; }
            if (!last) {
                const glm::vec3& nw = s.movePath[s.moveWp + 1];
                if (glm::length(glm::vec2(nw.x - p.pos.x, nw.z - p.pos.z))
                        < distXZ) { ++s.moveWp; continue; }
            }
            d /= distXZ;
            wish = glm::vec3(d.x, 0.0f, d.y);
            p.yaw = std::atan2(d.y, d.x);      // face travel (fwd = cos/sin yaw)
            break;
        }
        if (s.moveWp >= int(s.movePath.size())) s.moveActive = false;
    }

    bool inWater = isInWater(s.world, p.pos);

    if (inWater) {
        // Swim: slower horizontal, drag-eased Y, paddle controls. Sprint
        // still multiplies â€” water slows you down vs. land but holding
        // Ctrl is faster than not.
        float swimSpeed = speed * SWIM_SPEED_MULT;
        p.vel.x = wish.x * swimSpeed;
        p.vel.z = wish.z * swimSpeed;
        p.vel.y -= GRAVITY * WATER_GRAVITY_MULT * dt;
        if (glfwGetKey(win, GLFW_KEY_SPACE) == GLFW_PRESS)
            p.vel.y += SWIM_THRUST * dt;
        if (glfwGetKey(win, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
            p.vel.y -= SWIM_THRUST * 0.7f * dt;
        // Vertical drag â€” keeps motion calm and "floaty" instead of ballistic.
        p.vel.y *= std::exp(-WATER_DRAG * dt);
        // Clamp terminal sink + rise speeds so the player doesn't accelerate
        // forever while holding a key.
        p.vel.y = std::max(p.vel.y, -90.0f);
        p.vel.y = std::min(p.vel.y,  110.0f);
    } else {
        // Horizontal movement via acceleration + friction rather than snapping
        // velocity straight to the target. Holding a key ramps up; releasing
        // coasts to a stop; in the air the rates are low so a jump keeps its
        // momentum and you can only nudge your trajectory. This is what kills
        // the instant-on/instant-off "robotic" feel. cos(slope) (measured last
        // frame) keeps along-ramp speed equal to flat walking.
        glm::vec2 vel2(p.vel.x, p.vel.z);
        glm::vec2 targetV = glm::vec2(wish.x, wish.z) * (speed * s.slopeComp);
        bool moving = glm::length(wish) > 0.0f;
        float rate = p.onGround ? (moving ? GROUND_ACCEL : GROUND_DECEL)
                                : (moving ? AIR_ACCEL    : AIR_DECEL);
        vel2 += (targetV - vel2) * (1.0f - std::exp(-rate * dt));
        if (!moving && p.onGround && glm::length(vel2) < 0.5f) vel2 = glm::vec2(0.0f);
        p.vel.x = vel2.x;
        p.vel.z = vel2.y;

        p.vel.y -= GRAVITY * dt;
        if (glfwGetKey(win, GLFW_KEY_SPACE) == GLFW_PRESS && p.onGround) {
            p.vel.y = JUMP_VEL;
        }
    }

    auto trySlideOrStep = [&](glm::vec3& np, glm::vec3 prev, int axis, float& vAxis) {
        if (!overlapsSolid(s.world, np, curHeight)) return;
        if (p.vel.y > 1.0f) {
            if (axis == 0) np.x = prev.x;
            else            np.z = prev.z;
            vAxis = 0.0f;
            return;
        }
        for (int lift = 1; lift <= int(P_STEP); ++lift) {
            glm::vec3 t = np;
            t.y += float(lift);
            if (!overlapsSolid(s.world, t, curHeight)) { np = t; return; }
        }
        if (axis == 0) np.x = prev.x;
        else            np.z = prev.z;
        vAxis = 0.0f;
    };

    float yBefore = p.pos.y;
    float xBefore = p.pos.x, zBefore = p.pos.z;
    bool  wasGround = p.onGround;   // grounded state coming into this frame
    glm::vec3 np = p.pos;
    glm::vec3 prev = np;
    np.x += p.vel.x * dt;
    trySlideOrStep(np, prev, 0, p.vel.x);
    prev = np;
    np.z += p.vel.z * dt;
    trySlideOrStep(np, prev, 2, p.vel.z);
    prev = np;
    // Swept Y move: advance in <=2-voxel increments and stop at the first
    // collision, instead of testing only the endpoint. An endpoint-only test
    // tunnels through floors when a fall (or a hitch frame) moves the player
    // further than the body height in one tick.
    {
        float dy = p.vel.y * dt;
        int   nSteps = std::max(1, int(std::ceil(std::fabs(dy) / 2.0f)));
        float step = dy / float(nSteps);
        for (int i = 0; i < nSteps; ++i) {
            glm::vec3 t = np; t.y += step;
            if (overlapsSolid(s.world, t, curHeight)) { p.vel.y = 0.0f; break; }
            np = t;
        }
    }
    // -- Window frontier gate ------------------------------------------------
    // The player must stay inside the resident window (the world beyond it
    // isn't generated/loaded). Soft clamp a few voxels in from the edge â€”
    // an invisible wall, which also replaces the old "walk off the world
    // edge and fall forever" behavior. Streaming will keep the window
    // centered on the player so this gate should rarely engage.
    {
        float lo = 4.0f;
        float gx0 = float(s.world.originVoxX()) + lo;
        float gz0 = float(s.world.originVoxZ()) + lo;
        float gx1 = float(s.world.originVoxX() + GRID_XZ) - lo;
        float gz1 = float(s.world.originVoxZ() + GRID_XZ) - lo;
        if (np.x < gx0) { np.x = gx0; p.vel.x = std::max(p.vel.x, 0.0f); }
        if (np.x > gx1) { np.x = gx1; p.vel.x = std::min(p.vel.x, 0.0f); }
        if (np.z < gz0) { np.z = gz0; p.vel.z = std::max(p.vel.z, 0.0f); }
        if (np.z > gz1) { np.z = gz1; p.vel.z = std::min(p.vel.z, 0.0f); }
    }
    p.pos = np;
    p.onGround = isOnGround(s.world, p.pos);
    if (p.onGround && p.vel.y < 0) p.vel.y = 0;

    // -- Auto step-DOWN ----------------------------------------------------
    // Mirror of the auto step-UP in trySlideOrStep. If we were walking on the
    // ground and just ran off a ledge no taller than a step (not jumping or
    // rising), snap straight down onto the ground below in ONE frame instead
    // of briefly free-falling. That single-frame drop is then caught by the
    // step-smoothing below, so the camera glides down -- symmetric with up.
    if (wasGround && !p.onGround && p.vel.y <= 0.0f) {
        glm::vec3 t = p.pos;
        float dropped = 0.0f;
        while (dropped < float(P_STEP)) {
            glm::vec3 lower = t; lower.y -= 0.25f;
            if (overlapsSolid(s.world, lower, curHeight)) break;   // ground just below -> rest here
            t = lower; dropped += 0.25f;
        }
        if (dropped > 0.0f && isOnGround(s.world, t)) {
            p.pos = t; p.onGround = true; p.vel.y = 0.0f;
        }
    }

    float dyUp = p.pos.y - yBefore;

    // -- Slope speed compensation -----------------------------------------
    // We just moved `horiz` horizontally and climbed/descended `dyUp`. On a
    // ramp dyUp/horiz = tan(slope), so cos(slope) = 1/sqrt(1+tan^2) is the
    // factor that makes along-slope speed equal flat speed. Apply it (smoothed)
    // to next frame's horizontal speed. Eases back to 1 in the air / when idle.
    {
        float dxh = p.pos.x - xBefore, dzh = p.pos.z - zBefore;
        float horiz = std::sqrt(dxh*dxh + dzh*dzh);
        float targetComp = 1.0f;
        if (p.onGround && horiz > 1e-3f) {
            float ratio = std::fabs(dyUp) / horiz;            // tan(slope)
            targetComp = 1.0f / std::sqrt(1.0f + ratio*ratio); // cos(slope)
            if (targetComp < 0.55f) targetComp = 0.55f;        // don't crawl on cliffs
        }
        s.slopeComp += (targetComp - s.slopeComp) * std::min(1.0f, dt * 12.0f);
    }
}

// =============================================================================
// Loading gate (6f): show a clean screen while the world isn't ready, so the
// player never stares at a black/frozen window during startup gen or watches
// the world assemble after a teleport.
// =============================================================================
static void drawLoadingVeil(const char* title) {
    GLFWwindow* win = glfwGetCurrentContext();
    if (!win) return;
    int fbw, fbh; glfwGetFramebufferSize(win, &fbw, &fbh);
    glViewport(0, 0, fbw, fbh);
    glClearColor(0.60f, 0.72f, 0.85f, 1.0f);   // calm sky — a "loading" wash
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glfwSwapBuffers(win);
    glfwSetWindowTitle(win, title);
    glfwPollEvents();                           // keep the OS from greying us out
}

// ---- Animated loading screen (threaded gen + per-phase progress) ------------
// World-gen runs on a background thread (pure CPU, GL-free) while the main thread
// renders loading.frag each frame and polls the running step published by the
// worldgen phase hook. All GL stays on the main thread; the thread is joined
// before any GPU upload. See loading.frag for the visuals.
namespace {
struct LoadStep { const char* label; const char* text; float weight; };
// In worldgen phase order. `weight` ~ wall-time share (caves/trees dominate) for
// a smooth, monotone progress estimate; `text` is the player-facing line.
const LoadStep kLoadSteps[] = {
    { "init",      "Unrolling a fresh page",   0.02f },
    { "bulk_fill", "Raising the land",         0.10f },
    { "caves",     "Carving hidden passages",  0.42f },
    { "boulders",  "Scattering stones",        0.03f },
    { "bushes",    "Seeding the undergrowth",  0.05f },
    { "trees",     "Growing the forests",      0.21f },
    { "lakes",     "Filling lakes and oases",  0.05f },
    { "flowers",   "Sprinkling wildflowers",   0.05f },
    { "grass",     "Combing in the grass",     0.07f },
    { "far_field", "Painting the horizon",     0.04f },
};
constexpr int kNumSteps = int(sizeof(kLoadSteps) / sizeof(kLoadSteps[0]));
std::atomic<int> g_loadStepIdx{ 0 };   // index of the RUNNING phase (0..kNumSteps-1)

// phase() fires when a phase FINISHES, so the running phase is the NEXT one.
void loadingPhaseHook(const char* label) {
    for (int i = 0; i < kNumSteps; ++i)
        if (std::strcmp(label, kLoadSteps[i].label) == 0) {
            g_loadStepIdx.store(i + 1 < kNumSteps ? i + 1 : kNumSteps - 1,
                                std::memory_order_relaxed);
            return;
        }
    // Unknown sub-stage label (e.g. a cave-decor step): keep the current line.
}
// Pack an ASCII string into 12 ints (4 codes/int, LSB = first char), matching
// the layout loading.frag / console.frag expect. Returns the length.
void packLoadStr(const char* s, int out[12], int& len) {
    for (int i = 0; i < 12; ++i) out[i] = 0;
    int n = 0;
    for (; s[n] && n < 48; ++n)
        out[n >> 2] |= (int((unsigned char)s[n]) & 0xFF) << ((n & 3) * 8);
    len = n;
}
}  // namespace

// Teleport re-gate (6f): jump the resident window to centre on (destX,destZ),
// regenerate it there, and re-upload — all behind the loading veil — so a
// teleport lands the player in a FINISHED world instead of bare frontier.
// The window is otherwise clamped to the player, so this is the only way to
// reach terrain far from where you are.
// NOT static: exported to commands.cpp (consoleTeleport) -- the console
// module's single sanctioned app.cpp back-edge; see commands.h.
void recenterWindow(GameState& s, int destX, int destZ) {
    if (!s.streamer) return;
    // A teleport mid-interaction: the chop session's stand tile and route,
    // the open menu, and the hover mask all reference the OLD window — drop
    // them before the world underneath is regenerated.
    s.ctxOpen = false;
    cancelChop(s);
    clearTreeHover(s);
    // A live background ring rebuild reads gen state + writes the ring
    // vectors this regen is about to replace — drain it first.
    if (s.ringRebuilding) {
        s.ringRebuild.join();
        s.ringRebuilding = false;
        s.ringRebuildDone.store(false, std::memory_order_relaxed);
    }
    drawLoadingVeil("tiny voxels  —  arriving…");

    const int NC = GRID_XZ / CHUNK_VOX;
    s.streamer->reset();                         // drain async pool BEFORE wipe
    // Flush edits to disk + drop the cache under the lock (the background
    // compressor also touches `entries`). Edits survive on disk and reload
    // for the new region after gen.
    {
        std::lock_guard<std::mutex> lk(s.chunkCache.mtx);
        if (s.world.editsSinceSave)
            saveWorldEdits(s.world, s.gen, s.editsPath, &s.chunkCache);
        s.chunkCache.entries.clear();            // edits are on disk now
        s.chunkCache.totalRaw = s.chunkCache.totalComp = 0;
    }
    s.world.windowOriginChunk = { floorDiv(destX, CHUNK_VOX) - NC / 2,
                                  floorDiv(destZ, CHUNK_VOX) - NC / 2 };
    s.world.wetBricks.clear();
    s.creatures.reset();                         // stale herds hold old-window coords
    s.gen.cavePlan.reset();                      // fresh region's cave plan
    s.gpuQueue.clear();

    generateTerrainChunked(s.world, s.gen);      // origin-aware regen
    s.world.brickpool.reserve(                   // keep the streaming reserve
        std::max<size_t>(size_t(s.world.brickHighWater) * 3 / 2,
                         poolReserveBricks()) * BRICK_U32);
    s.world.uploadInitial();
    int ox = s.world.originVoxX(), oz = s.world.originVoxZ();
    buildHeightmap(s.gen, ox, oz);
    s.renderer.updateHeightRect(0, 0, GRID_XZ, GRID_XZ, s.gen.heightMap.data());
    buildFarHeightmap(s.gen, destX, destZ);          // re-centre the far horizon on the teleport
    s.renderer.uploadFarHeightmap(s.gen.farHeightMap, s.gen.farColorMap);
    s.streamer->seedAllFinal();
    if (loadWorldEdits(s.world, s.gen, s.editsPath))
        pruneCaveLights(s.world, s.gen);
    buildLightVolume(s.gen, ox, oz);             // origin-aware glow at the dest
    s.renderer.updateLightVolume(s.gen.lightVol);
}

// =============================================================================
// =============================================================================
// Evict/reload round-trip check (6e-2 validation gate)
// =============================================================================
// Semantic hash of one chunk column, same tag scheme as World::contentHash
// (material content only, independent of pool slot assignment).
static uint64_t chunkSemanticHash(const World& w, int ccx, int ccz) {
    auto fnv = [](uint64_t h, const uint8_t* p, size_t n) {
        for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 0x100000001B3ull; }
        return h;
    };
    uint64_t h = 0xCBF29CE484222325ull;
    int bx0 = ccx * CHUNK_BRICKS, bz0 = ccz * CHUNK_BRICKS;
    for (int by = 0; by < BRICKS_Y; ++by)
    for (int lz = 0; lz < CHUNK_BRICKS; ++lz)
    for (int lx = 0; lx < CHUNK_BRICKS; ++lx) {
        uint32_t bm = w.brickmap[w.slotOfBrick(bx0 + lx, by, bz0 + lz)];
        if (bm == BM_EMPTY) {
            uint8_t tag = 0x00;
            h = fnv(h, &tag, 1);
        } else if (bm & BM_SOLID_FLAG) {
            uint8_t tag[2] = { 0x01, uint8_t(bm & 0xFFu) };
            h = fnv(h, tag, 2);
        } else {
            uint8_t tag = 0x02;
            h = fnv(h, &tag, 1);
            h = fnv(h, (const uint8_t*)&w.brickpool[size_t(bm - 1u) * BRICK_U32],
                    BRICK_U32 * sizeof(uint32_t));
        }
    }
    return h;
}

// Round-trip N chunks through evict + reload and verify the world is
// byte-identical (per-chunk + global hashes) and the slot allocator fully
// reused the freed bricks (brickCount AND brickHighWater unchanged).
static void runEvictCheck(GameState& state, int n) {
    World& w = state.world;
    const int NC = GRID_XZ / CHUNK_VOX;
    n = std::clamp(n, 1, NC * NC);
    // Deterministic spread sample of chunk columns.
    std::vector<int> ids(size_t(NC) * NC);
    for (int i = 0; i < NC * NC; ++i) ids[i] = i;
    std::mt19937 rng(0xE71C7u);
    std::shuffle(ids.begin(), ids.end(), rng);
    ids.resize(n);

    uint64_t g0  = w.contentHash();
    uint32_t c0  = w.brickCount, hw0 = w.brickHighWater;
    size_t free0 = w.freeBricks.size(), wet0 = w.wetBricks.size();
    std::vector<uint64_t> h0(n);
    for (int i = 0; i < n; ++i)
        h0[i] = chunkSemanticHash(w, ids[i] % NC, ids[i] / NC);

    EditDelta d;   // no GPU yet at check time; collected and discarded
    int evicted = 0;
    for (int i = 0; i < n; ++i)
        evicted += evictChunk(w, state.chunkCache, ids[i] % NC, ids[i] / NC, &d) ? 1 : 0;
    std::printf("[evict-check] evicted %d/%d chunks: bricks %u->%u, "
                "free list %zu->%zu, cache %.1f MB raw\n",
                evicted, n, c0, w.brickCount, free0, w.freeBricks.size(),
                state.chunkCache.totalRaw / (1024.0 * 1024.0));

    int reloaded = 0;
    for (int i = 0; i < n; ++i)
        reloaded += reloadChunk(w, state.chunkCache, ids[i] % NC, ids[i] / NC, &d) ? 1 : 0;

    int hashMismatch = 0;
    for (int i = 0; i < n; ++i)
        if (chunkSemanticHash(w, ids[i] % NC, ids[i] / NC) != h0[i]) ++hashMismatch;
    uint64_t g1 = w.contentHash();
    bool pass = evicted == n && reloaded == n && hashMismatch == 0 &&
                g1 == g0 && w.brickCount == c0 && w.brickHighWater == hw0 &&
                w.wetBricks.size() == wet0 && state.chunkCache.entries.empty();
    std::printf("[evict-check] %s: %d chunks round-tripped; chunk hashes %s, "
                "global %016llx %s, bricks %u/%u hw %u/%u wet %zu/%zu\n",
                pass ? "PASS" : "FAIL", n,
                hashMismatch ? "DIVERGED" : "identical",
                (unsigned long long)g1, g1 == g0 ? "==" : "!=",
                w.brickCount, c0, w.brickHighWater, hw0,
                w.wetBricks.size(), wet0);
}

// =============================================================================
// Window scroll (6e-3b): the resident window follows the player.
// =============================================================================
// Recompute the height texture's texels for the strip that just ENTERED the
// window. REPEAT wrap means the entering strip's texels are exactly the
// leaving strip's, so this is an in-place overwrite (the CPU heightMap is
// kept in sync at the same toroidal indices).
// Ad-hoc parallel fan for scroll-time work (the gen pool may be mid-batch).
template <class Fn>
static void scrollFan(int total, Fn&& fn) {
    int nT = std::max(1, int(std::thread::hardware_concurrency()) - 2);
    std::atomic<int> next{ 0 };
    std::vector<std::thread> ts;
    ts.reserve(nT);
    for (int t = 0; t < nT; ++t)
        ts.emplace_back([&]() {
            int r;
            while ((r = next.fetch_add(1)) < total) fn(r);
        });
    for (auto& th : ts) th.join();
}

static void updateHeightStrip(GameState& s, glm::ivec2 dir) {
    World& w = s.world;
    const int M = GRID_XZ - 1;   // power-of-two mask
    int ox = w.originVoxX(), oz = w.originVoxZ();
    std::vector<float>& hm = s.gen.heightMap;
    // ~131K multi-octave height evaluations per strip (~10 ms serial - a
    // visible scroll spike); fan the rows across cores.
    auto fan = [](int total, auto&& fn) { scrollFan(total, fn); };
    if (dir.x != 0) {
        int wx0 = (dir.x > 0) ? ox + GRID_XZ - CHUNK_VOX : ox;
        int tx0 = wx0 & M;
        std::vector<float> data(size_t(GRID_XZ) * CHUNK_VOX);
        fan(GRID_XZ, [&](int t) {                     // texel row
            int wz = oz + ((t - oz) & M);             // window z landing on row t
            for (int lx = 0; lx < CHUNK_VOX; ++lx) {
                float h = float(terrainHeightAt(wx0 + lx, wz));
                data[size_t(t) * CHUNK_VOX + lx] = h;
                if (!hm.empty()) hm[size_t(t) * GRID_XZ + tx0 + lx] = h;
            }
        });
        s.renderer.updateHeightRect(tx0, 0, CHUNK_VOX, GRID_XZ, data.data());
    } else {
        int wz0 = (dir.y > 0) ? oz + GRID_XZ - CHUNK_VOX : oz;
        int tz0 = wz0 & M;
        std::vector<float> data(size_t(CHUNK_VOX) * GRID_XZ);
        fan(CHUNK_VOX, [&](int lz) {
            for (int t = 0; t < GRID_XZ; ++t) {       // texel column
                int wx = ox + ((t - ox) & M);
                float h = float(terrainHeightAt(wx, wz0 + lz));
                data[size_t(lz) * GRID_XZ + t] = h;
                if (!hm.empty()) hm[size_t(tz0 + lz) * GRID_XZ + t] = h;
            }
        });
        s.renderer.updateHeightRect(0, tz0, GRID_XZ, CHUNK_VOX, data.data());
    }
}

// Move the window origin one chunk: the trailing strip leaves (final chunks
// cache; partial frontier chunks DROP â€” they regenerate from seed), the
// leading strip enters (cache hits restore byte-identical immediately;
// misses stream in through the chunk-status ladder).
static void scrollWindow(GameState& s, ChunkStreamer& streamer, glm::ivec2 dir) {
    World& w = s.world;
    const int NC = GRID_XZ / CHUNK_VOX;
    glm::ivec2 o = w.windowOriginChunk;
    // GPU note: the leaving strip needs no GPU work (out-of-window bricks
    // are never sampled), and the entering strip's stale slot content is
    // tolerated for the few frames until its uploadChunk lands (it is the
    // far horizon edge). EditDelta is deliberately NOT used here â€” a strip
    // is a million dirty-slot inserts.
    // Leaving strip: only EDITED chunks are cached (extract + serialize) —
    // generation is deterministic, so an unedited chunk regenerates
    // byte-identical on return (proven by the overlap gate), exactly like
    // first-visit streaming. That makes the strip almost pure blank+free
    // (parallel, chunk-disjoint), instead of 32 extracts/scroll (the 40 ms
    // hitch). `cache` flags the rare edited columns.
    constexpr int PER_CHUNK = CHUNK_BRICKS * CHUNK_BRICKS * BRICKS_Y;
    struct Leave {
        int cx, cz; bool cache;
        std::vector<uint32_t> blob, frees;
    };
    std::vector<Leave> leaving;
    leaving.reserve(NC);
    for (int i = 0; i < NC; ++i) {
        int cx = dir.x ? (dir.x > 0 ? o.x : o.x + NC - 1) : o.x + i;
        int cz = dir.x ? o.y + i : (dir.y > 0 ? o.y : o.y + NC - 1);
        int64_t k = chunkKey(cx, cz);
        if (!s.chunkCache.has(k)) {
            bool edited = w.dirtyChunks.count(k) != 0 &&
                          streamer.stageOf(cx, cz) == CS_FINAL;
            leaving.push_back({ cx, cz, edited, {}, {} });
        }
        streamer.clear(cx, cz);
    }
    bool wet = !w.wetBricks.empty();
    scrollFan(int(leaving.size()), [&](int i) {
        Leave& L = leaving[i];
        if (L.cache) extractChunkBlob(w, L.cx, L.cz, L.blob);
        int base = w.slotOfBrick(L.cx * CHUNK_BRICKS, 0, L.cz * CHUNK_BRICKS);
        L.frees.reserve(4096);
        for (int k = 0; k < PER_CHUNK; ++k) {
            uint32_t bm = w.brickmap[base + k];
            if (bm != BM_EMPTY && !(bm & BM_SOLID_FLAG))
                L.frees.push_back(bm - 1u);
            w.brickmap[base + k] = BM_EMPTY;
        }
    });
    for (Leave& L : leaving) {
        if (L.cache) {
            ChunkCache::Entry e;
            e.rawBytes = uint32_t(L.blob.size() * sizeof(uint32_t));
            e.raw = std::move(L.blob);
            std::lock_guard<std::mutex> lk(s.chunkCache.mtx);  // vs bg compressor
            s.chunkCache.totalRaw += e.rawBytes;
            s.chunkCache.entries.emplace(chunkKey(L.cx, L.cz), std::move(e));
        }
        for (uint32_t bi : L.frees) w.freeBrick(bi);
        if (wet) {
            int base = w.slotOfBrick(L.cx * CHUNK_BRICKS, 0,
                                     L.cz * CHUNK_BRICKS);
            for (int k = 0; k < PER_CHUNK; ++k) w.wetBricks.erase(base + k);
        }
    }
    w.windowOriginChunk = o + dir;
    glm::ivec2 no = w.windowOriginChunk;
    for (int i = 0; i < NC; ++i) {
        int cx = dir.x ? (dir.x > 0 ? no.x + NC - 1 : no.x) : no.x + i;
        int cz = dir.x ? no.y + i : (dir.y > 0 ? no.y + NC - 1 : no.y);
        // The entering slots still hold the leaving strip's content on the
        // GPU — blank them NOW (one clear per chunk) so the horizon shows
        // air instead of 2 km-old ghost terrain until uploads land.
        w.gpuBlankChunk(cx, cz);
        if (s.chunkCache.has(chunkKey(cx, cz))) {
            reloadChunk(w, s.chunkCache, cx, cz);
            streamer.markFinal(cx, cz);
            s.gpuQueue.push_back(chunkKey(cx, cz));
        }
    }
    updateHeightStrip(s, dir);
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char** argv) {
    // --chunked-gen[=2] selects the chunk-by-chunk validation worldgen driver
    // (=2 iterates chunks in REVERSE for the order-independence proof). An
    // ARGUMENT, not just an env var: env propagation through PowerShell/cmd
    // process starts proved flaky during validation runs; an argv flag is
    // unambiguous. Internally it sets the env var the worldgen code reads.
    int evictCheckN = 0;     // --evict-check[=N]: round-trip gate after gen
    bool overlapCheck = false;   // --overlap-check: shifted-window gate, then exit
    bool tpTest = false;         // --tp-test: teleport-recenter gate, report, exit
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--chunked-gen", 13) == 0) {
            const char* mode = (argv[i][13] == '=') ? argv[i] + 14 : "1";
            _putenv_s("VOXEL_CHUNKED_GEN", mode);
        } else if (std::strncmp(argv[i], "--evict-check", 13) == 0) {
            evictCheckN = (argv[i][13] == '=') ? std::atoi(argv[i] + 14) : 64;
        } else if (std::strcmp(argv[i], "--overlap-check") == 0) {
            overlapCheck = true;
            _putenv_s("VOXEL_CHUNKED_GEN", "3");   // ladder at both origins
        } else if (std::strcmp(argv[i], "--tp-test") == 0) {
            tpTest = true;
        }
    }
    // Redirect stdout + stderr to game.log.txt for the whole run so we have
    // a persistent record of everything printed. _IONBF (unbuffered) is
    // important on Windows: MSVC silently downgrades _IOLBF to block buffering
    // when the stream isn't a TTY, leaving us with an empty file mid-run.
    if (!std::freopen("game.log.txt", "w", stdout)) {
        std::fprintf(stderr, "warning: could not open game.log.txt\n");
    }
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    _dup2(_fileno(stdout), _fileno(stderr));   // stderr -> same file as stdout
    for (int i = 1; i < argc; ++i) std::printf("[argv] %s\n", argv[i]);
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    std::printf("=== voxel session ===\n");
    std::fflush(stdout);

    StartupTimer T;
    if (!glfwInit()) { std::fprintf(stderr, "glfwInit failed\n"); return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* win = glfwCreateWindow(1280, 800, "tiny voxels", nullptr, nullptr);
    if (!win) { std::fprintf(stderr, "window create failed\n"); return 1; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    if (!gladLoadGL((GLADloadfunc)glfwGetProcAddress)) {
        std::fprintf(stderr, "glad load failed\n"); return 1;
    }
    VOX_GPU_CONTEXT;
    T.lap("window + GL context");

    GameState state;
    for (int i = 0; i < TWEAK_N; ++i) state.tweak[i] = TWEAKS[i].def;
    fillMatTuneDefaults(state.matTune);
    {   // tune.cfg: /tune save's output — lighting keys "<name>" and material
        // keys "mat.<mat>.<param>", clamped to range. Silently absent in a
        // clean checkout (the baked defaults are the shipped look).
        std::FILE* f = std::fopen("tune.cfg", "r");
        if (f) {
            char name[80]; float v;
            int loaded = 0;
            while (std::fscanf(f, "%79s %f", name, &v) == 2) {
                bool hit = false;
                for (int i = 0; i < TWEAK_N && !hit; ++i)
                    if (std::strcmp(name, TWEAKS[i].name) == 0) {
                        state.tweak[i] = std::clamp(v, TWEAKS[i].mn, TWEAKS[i].mx);
                        hit = true;
                    }
                if (!hit && std::strncmp(name, "mat.", 4) == 0) {
                    // parse mat.<matname>.<paramname>
                    for (int m = 0; m < MAT_LIST_N && !hit; ++m)
                    for (int p = 0; p < MP_COUNT && !hit; ++p) {
                        char key[80];
                        std::snprintf(key, sizeof(key), "mat.%s.%s",
                                      MAT_LIST[m].name, MAT_PARAMS[p].name);
                        if (std::strcmp(name, key) == 0) {
                            state.matTune[MAT_LIST[m].id * MP_COUNT + p] =
                                std::clamp(v, MAT_PARAMS[p].mn, MAT_PARAMS[p].mx);
                            hit = true;
                        }
                    }
                }
                if (hit) ++loaded;
            }
            std::fclose(f);
            std::printf("[tune] loaded %d value(s) from tune.cfg\n", loaded);
        }
    }
    // Console command memory: recall survives relaunch. Same tolerant CWD
    // sidecar pattern as tune.cfg above — silently absent in a clean checkout.
    state.console.loadHistory("console_history.txt");
    g_state = &state;

    glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    glfwSetCursorPosCallback(win, cursorCb);
    glfwSetMouseButtonCallback(win, mouseBtnCb);
    glfwSetScrollCallback(win, scrollCb);
    glfwSetCharCallback(win, charCb);
    glfwSetKeyCallback(win, keyCb);

    // The world is ALWAYS regenerated from the seed (deterministic and
    // chunk-isolated, proven by World::contentHash); only player edits
    // persist, as a region file of edited chunks overlaid after gen. The
    // old whole-world cache (world.dat + source-mtime invalidation) and the
    // cave-lights sidecar are gone â€” saves scale with edits, not world size.
    const std::string editsPath = "region_0_0.dat";
    state.editsPath = editsPath;
    std::printf("Generating terrain (%dx%dx%d effective resolution)...\n",
                GRID_XZ, GRID_Y, GRID_XZ);
    // DEFAULT driver = the parallel chunked driver (6e-3a): buffered margin
    // replay is the CANONICAL world semantics â€” chunked fwd == reverse ==
    // ladder == shifted-window overlap, all proven by hash. The monolithic
    // write-through driver (--chunked-gen=0) is a legacy perf baseline; its
    // decoration probes see earlier requests' writes, which diverges from
    // the buffered semantics now that cave systems reach in from outside
    // the window (~0.01% of voxels, counts identical).
    const char* genMode = std::getenv("VOXEL_CHUNKED_GEN");
    bool gen0 = (genMode && genMode[0] == '0');
    auto runGen = [&] {
        if (gen0) generateTerrain(state.world, state.gen);
        else      generateTerrainChunked(state.world, state.gen);
    };

    // 6f spawn gate. Headless/gate modes (overlap/evict/walk/tp tests) run gen
    // SYNCHRONOUSLY with no window work — their determinism + exit paths depend
    // on it. Normal startup runs gen on a BACKGROUND thread (pure CPU, GL-free)
    // and animates loading.frag here on the main thread, polling the running
    // phase the worldgen hook publishes. The thread is joined before any GL.
    bool animatedLoad = (!overlapCheck && evictCheckN == 0);
    GLuint loadProg = animatedLoad
        ? loadStandaloneProgram("fullscreen.vert", "loading.frag", "loading") : 0;
    if (animatedLoad && loadProg) {
        GLuint loadVAO = 0; glGenVertexArrays(1, &loadVAO);
        int fontPacked[256];
        for (int g = 0; g < 128; ++g) {
            int lo = 0, hi = 0;
            for (int r = 0; r < 4; ++r) lo |= int(FONT8X8[g][r])     << (r * 8);
            for (int r = 0; r < 4; ++r) hi |= int(FONT8X8[g][r + 4]) << (r * 8);
            fontPacked[g * 2] = lo; fontPacked[g * 2 + 1] = hi;
        }
        int titleC[12], titleL, subC[12], subL;
        packLoadStr("tiny voxels", titleC, titleL);
        packLoadStr("", subC, subL);   // (subtitle slot kept free for a future tagline)
        const GLint lVp = glGetUniformLocation(loadProg, "viewport"),
                    lPr = glGetUniformLocation(loadProg, "progress"),
                    lFn = glGetUniformLocation(loadProg, "font"),
                    lTC = glGetUniformLocation(loadProg, "titleChars"),
                    lTL = glGetUniformLocation(loadProg, "titleLen"),
                    lSC = glGetUniformLocation(loadProg, "subChars"),
                    lSL = glGetUniformLocation(loadProg, "subLen"),
                    lStC = glGetUniformLocation(loadProg, "stepChars"),
                    lStL = glGetUniformLocation(loadProg, "stepLen");
        float cum[kNumSteps + 1]; cum[0] = 0.0f;
        for (int i = 0; i < kNumSteps; ++i) cum[i + 1] = cum[i] + kLoadSteps[i].weight;

        auto drawLoad = [&](float prog, const char* step) {
            int sc[12], sl; packLoadStr(step, sc, sl);
            int fbw, fbh; glfwGetFramebufferSize(win, &fbw, &fbh);
            glViewport(0, 0, fbw, fbh);
            glUseProgram(loadProg);
            glUniform2f(lVp, float(fbw), float(fbh));
            glUniform1f(lPr, prog);
            glUniform1iv(lFn, 256, fontPacked);
            glUniform1iv(lTC, 12, titleC); glUniform1i(lTL, titleL);
            glUniform1iv(lSC, 12, subC);   glUniform1i(lSL, subL);
            glUniform1iv(lStC, 12, sc);    glUniform1i(lStL, sl);
            glBindVertexArray(loadVAO);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            glfwSwapBuffers(win);
            glfwPollEvents();
        };

        g_loadStepIdx.store(0);
        g_worldgenPhaseHook = loadingPhaseHook;
        std::atomic<bool> genDone{ false };
        std::thread genThread([&] { runGen(); genDone.store(true, std::memory_order_release); });

        int lastIdx = -1; double idxT0 = glfwGetTime(); float prog = 0.0f;
        while (!genDone.load(std::memory_order_acquire)) {
            double now = glfwGetTime();
            int idx = g_loadStepIdx.load(std::memory_order_relaxed);
            if (idx != lastIdx) { lastIdx = idx; idxT0 = now; }
            float dtP = float(now - idxT0);
            float target = cum[idx] + (cum[idx + 1] - cum[idx]) * (1.0f - std::exp(-dtP * 0.4f));
            if (target > prog) prog = target;                 // never run backwards
            int dots = int(now * 2.0) % 4;
            char step[80];
            std::snprintf(step, sizeof(step), "%s%.*s", kLoadSteps[idx].text, dots, "...");
            drawLoad(prog, step);
        }
        genThread.join();
        g_worldgenPhaseHook = nullptr;
        drawLoad(1.0f, "Stepping into the world");          // final 100% frame
        glDeleteVertexArrays(1, &loadVAO);
        glDeleteProgram(loadProg);
    } else {
        if (animatedLoad)   // shader failed to compile -> graceful fallback
            drawLoadingVeil("tiny voxels  —  generating world…");
        runGen();
    }
    state.worldReady = true;   // input verbs may touch the world from here on
    T.lap("world generated");
    if (overlapCheck) {
        // The 6e-3a unbounded-determinism gate: regenerate the window at a
        // SHIFTED origin and compare the shared world region (shrunk by a
        // 4-chunk rim from both windows' edges â€” edge chunks are provisional
        // by contract; their probes read air beyond the window). Pure-gen
        // comparison: runs before player edits overlay. Exits after.
        uint64_t hA = state.world.regionHash(768, 768, 1792, 1792);
        std::printf("[overlap-check] window A origin (0,0): region [768,1792)^2 "
                    "= %016llx\n", (unsigned long long)hA);
        state.world.windowOriginChunk = { 8, 8 };   // window [512, 2560)^2
        generateTerrainChunked(state.world, state.gen);
        uint64_t hB = state.world.regionHash(768, 768, 1792, 1792);
        std::printf("[overlap-check] window B origin (8,8): region [768,1792)^2 "
                    "= %016llx\n", (unsigned long long)hB);
        std::printf("[overlap-check] %s\n", hA == hB ? "PASS" : "FAIL");
        std::exit(hA == hB ? 0 : 1);
    }
    if (loadWorldEdits(state.world, state.gen, editsPath)) {
        // Mined-out crystal clusters in edited chunks must not glow again.
        pruneCaveLights(state.world, state.gen);
        T.lap("player edits applied");
    }
    // Farm crops persist in their own seed-keyed sidecar; their voxels re-stamp
    // on the first resident farmTick, and offline growth advances each crop's
    // age by the real time elapsed since the save.
    loadFarmState(state.farm, "farm.dat", state.gen.seed, int64_t(std::time(nullptr)));
    if (evictCheckN > 0) {
        runEvictCheck(state, evictCheckN);
        T.lap("evict-check");
    }

    // Streaming scheduler (6e-3b): the start window arrives FINAL from the
    // batch driver; the streamer takes over from here (window scroll +
    // frontier ladder). Evicted-but-cached chunks are off-limits to the
    // scheduler â€” they reload byte-identical instead of regenerating.
    ChunkStreamer streamer(state.world, state.gen);
    streamer.seedAllFinal();
    streamer.setSkip([&state](int64_t key) { return state.chunkCache.has(key); });
    state.streamer = &streamer;
    // ASYNC streaming computes read brickpool from worker threads, so the
    // vector must never REALLOCATE during play. Regional cave density swings
    // the live count several-fold (flat ~1.4M bricks/window, cavernous
    // ~5M), so reserve deep: address space commits lazily, physical pages
    // only on touch. allocBrick warns if it is ever exceeded.
    state.world.brickpool.reserve(
        std::max<size_t>(size_t(state.world.brickHighWater) * 3 / 2,
                         poolReserveBricks()) * BRICK_U32);

    // Background cache compressor: evicted chunks are cached RAW (cheap) on
    // the main thread during a scroll; zlib (the old 40 ms/frame hitch) now
    // runs here, off the render thread. Spins down to a 2 ms idle poll when
    // there is no raw backlog.
    std::atomic<bool> stopCompressor{false};
    std::thread cacheCompressor([&state, &stopCompressor]() {
        while (!stopCompressor.load(std::memory_order_relaxed)) {
            if (!compressOneFromCache(state.chunkCache))
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    });

    if (tpTest) {
        // 6f gate check: teleport-recenter far away (twice, to the SAME
        // origin) and verify the window comes back FINAL with real ground,
        // and that regen is deterministic (same origin -> same content).
        auto probe = [&](int X, int Z) {
            recenterWindow(state, X, Z);
            int gy = terrainHeightAt(X, Z);
            bool ground = state.world.getVoxel(X, gy - 1, Z) != 0;
            uint64_t hh = state.world.contentHash();
            // Glow check: crystal lights for this region must exist AND the
            // light volume must hold energy (the lightVol-windowing fix).
            double lvEnergy = 0.0;
            for (float v : state.gen.lightVol) lvEnergy += v;
            std::printf("[tp-test] @ (%d,%d): origin (%d,%d) pending=%d ground=%d "
                        "hash=%016llx  lights=%zu lightVol=%.0f\n", X, Z,
                        state.world.windowOriginChunk.x,
                        state.world.windowOriginChunk.y,
                        streamer.pendingInWindow(), ground ? 1 : 0,
                        (unsigned long long)hh,
                        state.gen.caveLights.size(), lvEnergy);
            return hh;
        };
        uint64_t a1 = probe(60000, 60000);
        probe(-40000, 12000);              // somewhere else (clears state)
        uint64_t a2 = probe(60000, 60000); // back: must reproduce a1
        std::printf("[tp-test] %s: far-teleport regen deterministic (%016llx %s)\n",
                    a1 == a2 ? "PASS" : "FAIL", (unsigned long long)a1,
                    a1 == a2 ? "==" : "!=");
        stopCompressor.store(true, std::memory_order_relaxed);
        cacheCompressor.join();
        glfwDestroyWindow(win);
        glfwTerminate();
        return a1 == a2 ? 0 : 1;
    }

    // Explored-world map (M): fog-of-war coverage persists across sessions,
    // keyed by world seed.
    if (state.mapExplored.load("map.dat", state.gen.seed))
        std::printf("map: %zu explored chunk column(s) restored\n",
                    state.mapExplored.chunks.size());

    // Spawn feet just above terrain at world center.
    {
        int cx = GRID_XZ / 2, cz = GRID_XZ / 2;
        int top = 0;
        for (int y = GRID_Y - 1; y >= 0; --y) {
            if (state.world.getVoxel(cx, y, cz) != 0) { top = y; break; }
        }
        state.player.pos = glm::vec3(float(cx) + 0.5f, float(top) + 1.0f, float(cz) + 0.5f);
        state.spawnPos   = state.player.pos;
    }

    state.world.uploadInitial();
    T.lap("world uploaded to GPU");
    // Per-column surface heights â†’ uploaded as a texture in renderer.init() so
    // the shader can measure each point's depth below the local surface.
    buildHeightmap(state.gen);
    buildLightVolume(state.gen);   // splat emissive sources into the 3D light volume
    T.lap("heightmap + light volume built");
    state.renderer.init(state.gen.heightMap, state.gen.lightVol);
    T.lap("renderer init");
    // Wildlife: species defs + baked rigid-part assets -> one GPU atlas.
    // Herds spawn lazily around the player during play (creatures.update).
    // The player avatar loads OUTSIDE the def system (it must never spawn,
    // tune, or census as a creature) but shares the volume atlas.
    state.creatures.loadAssets("assets/creatures");
    state.playerAvatar =
        loadCreatureModel("assets/player/player.cvx2", state.playerModel);
    {
        std::vector<uint32_t> atlas;
        state.creatures.buildGpuAtlas(atlas);
        // Bake the harvest produce volumes (species x grade) into the same
        // static atlas (each palBase + part wordBase stamped absolute here).
        for (int sp = 0; sp < CROP_SPECIES_N; ++sp)
            for (int g = 0; g < 3; ++g) {
                buildProduceModel(state.produceModels[sp][g], sp, g);
                appendModelToAtlas(state.produceModels[sp][g], atlas);
            }
        if (state.playerAvatar) {
            appendModelToAtlas(state.playerModel, atlas);
            state.pcIdle   = state.playerModel.clipIndex("Idle");
            state.pcWalk   = state.playerModel.clipIndex("Walk");
            state.pcSprint = state.playerModel.clipIndex("Sprint");
            state.pcJump   = state.playerModel.clipIndex("Jump");
            state.pcFall   = state.playerModel.clipIndex("Fall");
            // Chop (authored for forestry); the older Punch swing stands in
            // if the asset predates it (non-looping — chopSwingT is fmod'd
            // into the clip, so a clamped clip still cycles).
            state.pcChop   = state.playerModel.clipIndex("Chop");
            if (state.pcChop < 0) state.pcChop = state.playerModel.clipIndex("Punch");
            // Mine (authored for mining); Chop stands in on older assets.
            state.pcMine   = state.playerModel.clipIndex("Mine");
            if (state.pcMine < 0) state.pcMine = state.pcChop;
            // Harvest (authored for farming); Chop stands in on older assets.
            state.pcHarvest = state.playerModel.clipIndex("Harvest");
            if (state.pcHarvest < 0) state.pcHarvest = state.pcChop;
            for (int i = 0; i < int(state.playerModel.parts.size()); ++i)
                if (!std::strcmp(state.playerModel.parts[i].name, "armR.lower"))
                    state.playerHandPart = i;
            state.pcCrouchIdle = state.playerModel.clipIndex("CrouchIdle");
            state.pcCrouchWalk = state.playerModel.clipIndex("CrouchWalk");
            state.pClip    = state.pcIdle;
            if (state.pcIdle < 0 || state.pcWalk < 0) state.playerAvatar = false;
            std::printf("[player] avatar: %zu parts, %zu clips (%s)\n",
                        state.playerModel.parts.size(),
                        state.playerModel.clips.size(),
                        state.playerAvatar ? "active" : "missing Idle/Walk — analytic fallback");
        } else {
            std::printf("[player] no assets/player/player.cvx2 — analytic character\n");
        }
        // Forestry props share the atlas: debris chunk models + the axe.
        forestryBuildStaticModels(state.forestry);
        appendModelToAtlas(state.forestry.woodChunk, atlas);
        appendModelToAtlas(state.forestry.leafChunk, atlas);
        appendModelToAtlas(state.forestry.cactusChunk, atlas);
        appendModelToAtlas(state.forestry.stoneChunk, atlas);
        appendModelToAtlas(state.forestry.axeModel, atlas);
        appendModelToAtlas(state.forestry.pickModel, atlas);
        state.atlasBase = atlas;
        state.atlasDyn  = std::move(atlas);
        state.renderer.uploadCreatureAssets(state.atlasDyn);
    }
    T.lap("creature assets");

    // Far-field horizon (render-distance Phase A): a coarse heightfield centred
    // on the resident window, raymarched in the g-buffer miss branch for km-scale
    // terrain silhouettes beyond the voxel window. Cheap (heightAt only).
    buildFarHeightmap(state.gen, state.world.originVoxX() + GRID_XZ / 2,
                                 state.world.originVoxZ() + GRID_XZ / 2);
    state.renderer.uploadFarHeightmap(state.gen.farHeightMap, state.gen.farColorMap);
    T.lap("far heightfield");

    // Build the item registry (materials auto-register + rich item defs) BEFORE
    // any giveItem — Container::add reads per-item maxStack from it.
    initItems();

    // Starter palette in the SATCHEL storage (not the hotbar — that stays clear
    // for what you collect). Full stacks of the common building materials so the
    // player can build/test immediately; mining + harvest add more.
    state.hotbar.giveToStorage(MAT_DIRT,   999);
    state.hotbar.giveToStorage(MAT_GRASS,  999);
    state.hotbar.giveToStorage(MAT_STONE,  999);
    state.hotbar.giveToStorage(MAT_WOOD,   999);
    state.hotbar.giveToStorage(MAT_LEAVES, 999);
    state.hotbar.giveToStorage(MAT_WATER,  999);
    state.hotbar.giveToStorage(MAT_SAND,   999);   // physics test: place + watch it fall/pile

    std::printf("Controls: RMB click/hold: walk there, LMB drag: orbit, wheel: zoom,\n"
                "          minimap (top right): LMB walk there, wheel zooms the map,\n"
                "          camera orb on its ring toggles FREE CAM (RMB drags the\n"
                "          view; minimap clicks pan it), Q/E slice, TAB cave x-ray,\n"
                "          Space jump, Ctrl sprint, F fly (dev:\n"
                "          WASD+Space/Ctrl), F11 fullscreen, ESC pause\n");

    // Audio: init the engine. Music is OFF by default — toggle the looping
    // ambience bed in the dev console with /music (it loads
    // assets/audio/ambient.mp3). Kept opt-in so it's a deliberate test, not
    // something shipped on by default.
    Audio::init();

    // The diorama IS the game: open it on the player from the first frame.
    state.mapView.openAt(state.player.pos, false);

    T.lap("ready, entering main loop");
    double lastT = glfwGetTime();
    double startT = lastT;

    bool prevF11 = false, prevF = false, prevF3 = false;
    bool prevSpace = false;
    double lastSpacePress = -1.0;

    double nextAmbient = 0.0;
    double nextWaterTick = 0.0;
    std::mt19937 waterRng{0xF1AB1Du};
    double nextFireTick = 0.0;
    double nextGranularTick = 0.0;
    std::mt19937 granularRng{0x5A4Du};

    double fpsAccumT = 0.0;
    int    fpsAccumN = 0;
    double fpsAccumDt = 0.0;
    double fpsAccumMin = 0.0;
    double fpsAccumMax = 0.0;

    while (!glfwWindowShouldClose(win)) {
        double now = glfwGetTime();
        float dtRaw = float(now - lastT);
        lastT = now;
        // Clamp the simulation timestep. A hitch frame (shader hot-reload
        // recompile, window drag) otherwise integrates a
        // huge gravity step and the endpoint-only Y collision tunnels the
        // player straight through the floor into the caves below.
        float dt = std::min(dtRaw, 0.05f);
        VOX_ZONE_BEGIN(zPoll, "input poll");
        glfwPollEvents();
        VOX_ZONE_END(zPoll);
        // FPS counter: print to stderr once per second.
        fpsAccumT  += dtRaw;
        fpsAccumDt += dtRaw;
        if (fpsAccumN == 0 || dtRaw < fpsAccumMin) fpsAccumMin = dtRaw;
        if (dtRaw > fpsAccumMax) fpsAccumMax = dtRaw;
        fpsAccumN += 1;
        if (fpsAccumT >= 1.0) {
            double avgMs = (fpsAccumDt / fpsAccumN) * 1000.0;
            double minMs = fpsAccumMin * 1000.0;
            double maxMs = fpsAccumMax * 1000.0;
            std::fprintf(stderr, "[fps] %5.1f  avg %5.2fms  min %5.2fms  max %5.2fms"
                         "  bricks %u live / %u hw / %zu free\n",
                         fpsAccumN / fpsAccumT, avgMs, minMs, maxMs,
                         state.world.brickCount, state.world.brickHighWater,
                         state.world.freeBricks.size());
            if (!state.chunkCache.entries.empty() || streamer.pendingInWindow() > 0)
                std::fprintf(stderr, "[stream] pending=%d evicted=%zu cache "
                             "%.1f+%.1f MB  origin (%d,%d)\n",
                             streamer.pendingInWindow(),
                             state.chunkCache.entries.size(),
                             state.chunkCache.totalComp / (1024.0 * 1024.0),
                             state.chunkCache.totalRaw  / (1024.0 * 1024.0),
                             state.world.windowOriginChunk.x,
                             state.world.windowOriginChunk.y);
            fpsAccumT = 0.0; fpsAccumN = 0; fpsAccumDt = 0.0;
            fpsAccumMin = 0.0; fpsAccumMax = 0.0;
        }


        // -- Streaming (6e-3b): the window follows the player ----------------
        if (state.streamOn) {
            const int NC   = GRID_XZ / CHUNK_VOX;
            const int LEAD = 14;    // keep the player this far from any edge
            int pcx = floorDiv(int(std::floor(state.player.pos.x)), CHUNK_VOX);
            int pcz = floorDiv(int(std::floor(state.player.pos.z)), CHUNK_VOX);
            glm::ivec2 o = state.world.windowOriginChunk;

            // 1. Auto-scroll, one axis-step per frame, to keep the player
            //    >= LEAD chunks from every window edge.
            glm::ivec2 dir{ 0, 0 };
            if      (pcx - o.x < LEAD)       dir = { -1, 0 };
            else if (o.x + NC - pcx <= LEAD) dir = { +1, 0 };
            else if (pcz - o.y < LEAD)       dir = { 0, -1 };
            else if (o.y + NC - pcz <= LEAD) dir = { 0, +1 };
            if (dir.x || dir.y) scrollWindow(state, streamer, dir);

            // 1b. Far-field rings follow the window: when the window centre
            // has drifted past ring r's threshold, rebuild the drifted rings
            // on a background thread (~0.7s each, pure worldgen + CPU
            // staging, from a CENTRE SNAPSHOT taken here — the worker never
            // reads the live scrolling origin) and publish + swap them on
            // completion, on THIS thread. The shader keeps rendering the OLD
            // rings (originGPU) until the swap, so there is no gap mid-
            // rebuild. Thresholds are (GRID_XZ/8) << r = 256/512/1024/2048:
            // ring r's spare coverage beyond its owned band is only
            // 512 << r, and the player's ±128 off-centre allowance plus
            // movement during the rebuild must fit inside it — the old
            // quarter-window trigger consumed the entire margin and let ring
            // AABBs fall short of their bands (screen-space wedges of
            // coarser content).
            if (state.ringRebuilding &&
                state.ringRebuildDone.load(std::memory_order_acquire)) {
                state.ringRebuild.join();
                state.ringRebuilding = false;
                state.ringRebuildDone.store(false, std::memory_order_relaxed);
                for (int r = 0; r < RING_COUNT; ++r)          // publish on main
                    if (state.ringRebuildMask & (1u << r))
                        state.world.rings[r].dirty = true;
                state.ringRebuildMask = 0;
            }
            state.world.uploadRings();   // amortized: at most one dirty ring per frame
            if (!state.ringRebuilding) {
                int wcx = state.world.originVoxX() + GRID_XZ / 2;
                int wcz = state.world.originVoxZ() + GRID_XZ / 2;
                uint32_t rebuildMask = 0;
                for (int r = 0; r < RING_COUNT; ++r) {
                    const World::RingLevel& rl = state.world.rings[r];
                    if (!rl.ready || rl.dirty) continue;   // pending upload: wait
                    int sl  = ringScaleLog2(r);
                    int rcx = ((rl.originGPU.x + RING_BRICKS_XZ / 2) * BRICK) << sl;
                    int rcz = ((rl.originGPU.y + RING_BRICKS_XZ / 2) * BRICK) << sl;
                    if (std::abs(wcx - rcx) > (GRID_XZ / 8) << r ||
                        std::abs(wcz - rcz) > (GRID_XZ / 8) << r)
                        rebuildMask |= 1u << r;
                }
                if (rebuildMask) {
                    state.ringRebuildMask = rebuildMask;
                    state.ringRebuilding = true;
                    state.ringRebuild = std::thread([&state, rebuildMask, wcx, wcz]() {
                        for (int r = 0; r < RING_COUNT; ++r)
                            if (rebuildMask & (1u << r))
                                buildRing(state.world, state.gen, r, wcx, wcz);
                        state.ringRebuildDone.store(true, std::memory_order_release);
                    });
                }
            }

            // 2. Re-entry: reload the nearest in-window cached (edited) chunk
            //    (1/frame). Far cached chunks wait until approached.
            {
                o = state.world.windowOriginChunk;
                int bestRD = INT32_MAX, bestRX = 0, bestRZ = 0;
                for (const auto& kv : state.chunkCache.entries) {
                    int cx = chunkKeyX(kv.first), cz = chunkKeyZ(kv.first);
                    if (cx < o.x || cx >= o.x + NC ||
                        cz < o.y || cz >= o.y + NC) continue;
                    int dist = std::max(std::abs(cx - pcx), std::abs(cz - pcz));
                    if (dist < bestRD) { bestRD = dist; bestRX = cx; bestRZ = cz; }
                }
                if (bestRD != INT32_MAX) {
                    reloadChunk(state.world, state.chunkCache, bestRX, bestRZ);
                    streamer.markFinal(bestRX, bestRZ);
                    state.gpuQueue.push_back(chunkKey(bestRX, bestRZ));
                }
            }

            // 3. Advance the frontier ladder one batch; push milestone chunks
            //    to the GPU (budgeted — bursting many buffer writes per frame
            //    risks driver sync stalls against the in-flight raymarch).
            //    ALL pending uploads drain into gpuQueue first so the per-ray
            //    reveal (chunkValidTex) sees the TRUE un-uploaded set — a
            //    backlog hidden inside the streamer rendered as coarse-ring
            //    stripes where fine content should be. Uploads then run
            //    NEAREST-FIRST (the old LIFO pop buried mid-path chunks
            //    indefinitely during sustained flight): what the player can
            //    see fixes itself first, and the reveal cap retires fastest.
            {
                EditDelta scratch;
                streamer.step({ pcx, pcz }, scratch);
                int64_t up[64];
                int n;
                while ((n = streamer.drainUploads(up, 64)) > 0)
                    state.gpuQueue.insert(state.gpuQueue.end(), up, up + n);
                for (int i = 0; i < 8 && !state.gpuQueue.empty(); ++i) {
                    size_t best = 0; float bestD = 1e30f;
                    for (size_t j = 0; j < state.gpuQueue.size(); ++j) {
                        float cx = float(chunkKeyX(state.gpuQueue[j]) * CHUNK_VOX)
                                 + CHUNK_VOX * 0.5f;
                        float cz = float(chunkKeyZ(state.gpuQueue[j]) * CHUNK_VOX)
                                 + CHUNK_VOX * 0.5f;
                        float dx = cx - state.player.pos.x;
                        float dz = cz - state.player.pos.z;
                        float d = dx * dx + dz * dz;
                        if (d < bestD) { bestD = d; best = j; }
                    }
                    int64_t k = state.gpuQueue[best];
                    state.gpuQueue[best] = state.gpuQueue.back();
                    state.gpuQueue.pop_back();
                    state.world.uploadChunk(chunkKeyX(k), chunkKeyZ(k));
                }
            }

            // 4. Don't let the player walk into ungenerated frontier: revert
            //    to the last position whose chunk has (visible) ground —
            //    which means GPU-RESIDENT ground, not merely CPU-generated
            //    (a walkable-but-blank-on-GPU chunk put the LOD handoff
            //    inside the player's own chunk).
            {
                static glm::vec3 lastSafe = state.player.pos;
                int ccx = floorDiv(int(std::floor(state.player.pos.x)), CHUNK_VOX);
                int ccz = floorDiv(int(std::floor(state.player.pos.z)), CHUNK_VOX);
                bool gpuOk = state.world.gpuChunkValid.empty() ||
                             state.world.gpuChunkValid[World::chunkSlotIdx(ccx, ccz)];
                if (streamer.stageOf(ccx, ccz) >= CS_BULK && gpuOk)
                    lastSafe = state.player.pos;
                else {
                    state.player.pos.x = lastSafe.x;
                    state.player.pos.z = lastSafe.z;
                }
            }

            // 5. Cave glow follows the window. The lightVol is a toroidal
            //    window (like heightTex), 48 MB total — rebuilding + re-uploading
            //    ALL of it every scroll was a ~75 ms hitch (~2/s while walking).
            //    Instead update INCREMENTALLY: a one-chunk scroll rebuilds only
            //    the entering/leaving strip (buildLightVolumeStrip), and newly
            //    streamed crystal clusters splat only their local box
            //    (splatNewLights) — both upload just their sub-rect. A full
            //    rebuild (which also prunes out-of-window lights) is the rare
            //    fallback: init, teleport, an edit, or the periodic prune.
            {
                static glm::ivec2 lastLO{ INT32_MIN, INT32_MIN };
                static size_t     lastLC = SIZE_MAX;
                static size_t     lastPruned = 0;
                glm::ivec2 lo = state.world.windowOriginChunk;
                int ox = state.world.originVoxX(), oz = state.world.originVoxZ();
                auto& L = state.gen.caveLights;
                bool first   = (lastLO.x == INT32_MIN);
                glm::ivec2 d = first ? glm::ivec2(0) : (lo - lastLO);
                bool scroll1 = (std::abs(d.x) + std::abs(d.y) == 1);
                bool grew    = (!first && L.size() > lastLC);     // streamed appends
                bool shrank  = (!first && L.size() < lastLC);     // mined/edited elsewhere
                // The incremental paths keep (but ignore) out-of-window lights;
                // force an occasional full rebuild to prune them so the list
                // can't grow without bound on a long traversal.
                bool tooMany = (!first && L.size() > lastPruned * 2 + 8192);
                if (!tooMany && !first && scroll1 && !grew && !shrank) {
                    // One-chunk scroll, no light change: rebuild + upload ONLY the
                    // entering strip (~few ms vs ~75 for the whole 48 MB volume).
                    bool xAxis; int off, cnt;
                    buildLightVolumeStrip(state.gen, ox, oz, d, xAxis, off, cnt);
                    state.renderer.updateLightVolumeRect(state.gen.lightVol, xAxis, off, cnt);
                    lastLO = lo;
                } else if (!tooMany && !first && grew && d.x == 0 && d.y == 0) {
                    // New streamed crystal cluster(s) appended, no scroll: splat
                    // ONLY the new lights and upload their local box.
                    int bx0, bx1, bz0, bz1;
                    splatNewLights(state.gen, ox, oz, lastLC, bx0, bx1, bz0, bz1);
                    state.renderer.updateLightVolumeBox(state.gen.lightVol, bx0, bx1, bz0, bz1);
                    lastLC = L.size();
                } else if (tooMany || first || grew || shrank || d.x || d.y) {
                    // Fallback full rebuild: init, scroll+new-lights same frame, a
                    // teleport/multi-chunk jump, or a prune/edit shrank the list.
                    // Prune out-of-window lights here (the only all-iterating path).
                    L.erase(std::remove_if(L.begin(), L.end(), [&](const CaveLight& c) {
                        return c.pos.x < float(ox) || c.pos.x >= float(ox + GRID_XZ) ||
                               c.pos.z < float(oz) || c.pos.z >= float(oz + GRID_XZ);
                    }), L.end());
                    buildLightVolume(state.gen, ox, oz);
                    state.renderer.updateLightVolume(state.gen.lightVol);
                    lastLO = lo; lastLC = L.size(); lastPruned = L.size();
                }
            }
        }

        // -- Day / night cycle ------------------------------------------------
        // timeOfDay advances every frame at /time's speed multiplier (0 =
        // paused, 60 = the old dev fast-forward). The sun and moon ride a
        // shared tilted great-circle orbit 180 degrees apart, so the moon
        // rises exactly as the sun sets and neither body ever just blinks
        // out. Even at the /time speed cap (x1000) with dt clamped to 0.05 s
        // the largest per-frame step is ~0.042 day, so the single-floor wrap
        // here and the fog block's 0.6/0.4 day-crossing window stay sound.
        state.timeOfDay += dt * (state.timeSpeed / DAY_LENGTH);   // DAY_LENGTH in common.h
        state.timeOfDay -= std::floor(state.timeOfDay);   // wrap [0,1)

        // ang = 0 at sunrise (sun on the east horizon), pi/2 at noon, pi at
        // sunset, 3pi/2 at midnight (sun straight down -> moon overhead).
        float ang = (state.timeOfDay - 0.25f) * 6.2831853f;
        constexpr float TILT = 0.58f;                  // ~33deg orbit tilt: an arc, not the zenith
        float ca = std::cos(ang), sa = std::sin(ang);
        glm::vec3 sunDir  = glm::normalize(glm::vec3(ca, sa * std::cos(TILT),
                                                         sa * std::sin(TILT)));
        glm::vec3 moonDir = -sunDir;
        // dayLight ~0 once the sun is below the horizon; moonLight is a dim cool
        // term peaking at midnight, gated by (1-dayLight) so dawn/dusk don't
        // double up.
        float dayLight  = glm::clamp(sunDir.y * 1.5f + 0.12f, 0.0f, 1.0f);
        float moonLight = glm::clamp(moonDir.y, 0.0f, 1.0f) * (1.0f - dayLight);

        // Esc (pause / close-console) is handled in keyCb so a single press
        // can't both close the console and pause.

        // Fullscreen toggle. Borderless, not exclusive: an undecorated
        // window laid over the monitor never touches the display mode, so
        // it can't drop the panel into a mode it rejects (high-refresh
        // ultrawides with DSC are the usual casualty of GLFW's
        // closest-mode match), and losing focus to another app -- OBS on a
        // second monitor -- doesn't auto-iconify and flap the video mode
        // the way a glfwSetWindowMonitor fullscreen window does. Targets
        // the monitor holding the window, not whichever one is primary.
        bool f11 = glfwGetKey(win, GLFW_KEY_F11) == GLFW_PRESS;
        if (f11 && !prevF11) {
            if (!state.isFullscreen) {
                // Resolve the target monitor before touching the window at
                // all, so the zero-monitor abort (RDP drop, DP deep sleep)
                // is a true no-op. The overlap uses the current rect -- for
                // a maximized window that's still the monitor the user
                // sees it on. Primary is only the fallback when the window
                // sits on none, and can itself be null with no monitors.
                int px, py, pw, ph;
                glfwGetWindowPos(win, &px, &py);
                glfwGetWindowSize(win, &pw, &ph);
                GLFWmonitor* mon = glfwGetPrimaryMonitor();
                int nMon = 0;
                GLFWmonitor** mons = glfwGetMonitors(&nMon);
                long long best = 0;
                for (int i = 0; i < nMon; ++i) {
                    const GLFWvidmode* m = glfwGetVideoMode(mons[i]);
                    if (!m) continue;
                    int mx, my;
                    glfwGetMonitorPos(mons[i], &mx, &my);
                    long long ow = std::min(px + pw, mx + m->width) -
                                   std::max(px, mx);
                    long long oh = std::min(py + ph, my + m->height) -
                                   std::max(py, my);
                    long long area = std::max(0LL, ow) * std::max(0LL, oh);
                    if (area > best) { best = area; mon = mons[i]; }
                }
                const GLFWvidmode* mode = mon ? glfwGetVideoMode(mon) : nullptr;
                if (mode) {
                    // Drop OS-maximize before saving geometry: WS_MAXIMIZE
                    // survives the style / pos / size calls below, and
                    // Windows clamps a *maximized* undecorated window to
                    // the work area (taskbar showing) on the next frame
                    // re-evaluation. Restoring also makes the saved
                    // geometry the true windowed rect, not the maximized
                    // one; the flag is replayed on exit.
                    state.wasMaximized =
                        glfwGetWindowAttrib(win, GLFW_MAXIMIZED) == GLFW_TRUE;
                    if (state.wasMaximized) glfwRestoreWindow(win);
                    glfwGetWindowPos(win, &state.savedX, &state.savedY);
                    glfwGetWindowSize(win, &state.savedW, &state.savedH);
                    int mx, my;
                    glfwGetMonitorPos(mon, &mx, &my);
                    glfwSetWindowAttrib(win, GLFW_DECORATED, GLFW_FALSE);
                    glfwSetWindowPos(win, mx, my);
                    glfwSetWindowSize(win, mode->width, mode->height);
                    state.isFullscreen = true;
                }
            } else {
                glfwSetWindowAttrib(win, GLFW_DECORATED, GLFW_TRUE);
                // The layout may have changed while fullscreen (monitor
                // unplugged, RDP): if the saved rect no longer touches any
                // monitor, recenter it on the primary work area rather
                // than stranding the window off the virtual desktop.
                bool onScreen = false;
                int nMon = 0;
                GLFWmonitor** mons = glfwGetMonitors(&nMon);
                for (int i = 0; i < nMon && !onScreen; ++i) {
                    const GLFWvidmode* m = glfwGetVideoMode(mons[i]);
                    if (!m) continue;
                    int mx, my;
                    glfwGetMonitorPos(mons[i], &mx, &my);
                    onScreen = state.savedX < mx + m->width  &&
                               state.savedX + state.savedW > mx &&
                               state.savedY < my + m->height &&
                               state.savedY + state.savedH > my;
                }
                GLFWmonitor* prim = glfwGetPrimaryMonitor();
                if (!onScreen && prim) {
                    // Clamp and center the OUTER rect: saved coords are
                    // client-area, so fold in the just-re-enabled frame or
                    // a work-area-tall window puts its caption above the
                    // screen top -- unreachable with the mouse, the exact
                    // stranding this branch exists to prevent.
                    int wx, wy, ww, wh;
                    glfwGetMonitorWorkarea(prim, &wx, &wy, &ww, &wh);
                    int fl, ft, fr, fb;
                    glfwGetWindowFrameSize(win, &fl, &ft, &fr, &fb);
                    int availW = std::max(1, ww - fl - fr);
                    int availH = std::max(1, wh - ft - fb);
                    state.savedW = std::min(state.savedW, availW);
                    state.savedH = std::min(state.savedH, availH);
                    state.savedX = wx + fl + (availW - state.savedW) / 2;
                    state.savedY = wy + ft + (availH - state.savedH) / 2;
                }
                glfwSetWindowPos(win, state.savedX, state.savedY);
                glfwSetWindowSize(win, state.savedW, state.savedH);
                if (state.wasMaximized) glfwMaximizeWindow(win);
                state.isFullscreen = false;
            }
            glfwSwapInterval(1);
        }
        prevF11 = f11;

        // Map motion + slice plane: the settle-in eases while open; holding
        // Q/E lowers/raises the horizontal cut so explored tunnels appear.
        // Frozen while paused: a paused world must not glide.
        if (!state.paused) {
            state.mapView.animT = std::min(1.0f, state.mapView.animT + dt * 3.0f);
            // Follow-camera: the orbit anchor chases the walking player; a
            // free-cam look-around (right-drag) parks until the mode ends.
            state.mapView.follow(state.player.pos,
                                 state.rightDown && state.mapView.freeCam, dt);
            // Demo director (/orbit): eased cinematic yaw spin + framing
            // glides. The glides land in the cam.* tweaks BEFORE the framing
            // lines below read them, so /orbit, the wheel and /tune all
            // drive the SAME values (an active glide simply wins the frame).
            {
                DemoCam& dc = state.demoCam;
                float want = dc.on ? dc.degPerSec : 0.0f;
                dc.spinNow += (want - dc.spinNow) * (1.0f - std::exp(-1.6f * dt));
                if (!dc.on && std::fabs(dc.spinNow) < 0.02f) dc.spinNow = 0.0f;
                // remainder() re-centres yaw each frame so hour-long takes
                // never push float trig into visible precision loss.
                state.mapView.yaw = std::remainder(
                    state.mapView.yaw + glm::radians(dc.spinNow) * dt,
                    6.2831853f);
                auto glide = [&](CamGlide& g, int tw) {
                    if (!g.active) return;
                    g.t = std::min(g.t + dt, g.dur);
                    float k = g.t / g.dur;
                    k = k * k * (3.0f - 2.0f * k);   // smoothstep ease in/out
                    state.tweak[tw] = glm::mix(g.from, g.to, k);
                    if (g.t >= g.dur) g.active = false;
                };
                glide(dc.pitch,  TW_CAM_PITCH);
                glide(dc.dist,   TW_CAM_DIST);
                glide(dc.height, TW_CAM_HEIGHT);
            }
            // Camera framing is live-tunable via /tune (cam.pitch/dist/height):
            // drag a slider and the diorama reframes on the spot; /tune save
            // bakes the sweet spot. Pitch + distance drive the "see sky vs see
            // in front" balance; height lifts the look-at above the feet.
            state.mapView.pitch = state.tweak[TW_CAM_PITCH];
            state.mapView.dist  = state.tweak[TW_CAM_DIST];
            state.mapView.target.y += state.tweak[TW_CAM_HEIGHT];
            // Character stride phase: ~12 voxels per step at any speed, so
            // the walk cycle tracks actual motion (sprint steps faster).
            state.walkAnimT += glm::length(glm::vec2(state.player.vel.x,
                                                     state.player.vel.z))
                             * dt * (1.0f / 24.0f);
            // Console can sit over the map; typed q/e must not move the slice.
            // E also opens the satchel, so the slice raise pauses while it's up.
            if (!state.console.open && !state.invOpen) {
                float rate = 420.0f * dt;
                if (glfwGetKey(win, GLFW_KEY_Q) == GLFW_PRESS) state.mapView.sliceY -= rate;
                if (glfwGetKey(win, GLFW_KEY_E) == GLFW_PRESS) state.mapView.sliceY += rate;
            }
        }

        // Explored-world map marking: proximity fog-of-war reveal around the
        // player (self-throttled; see map.cpp for the reveal shape).
        {
            glm::vec3 pp = state.player.pos;
            float surfH = surfaceHeightAt(state, int(std::floor(pp.x)),
                                                 int(std::floor(pp.z)));
            state.mapExplored.update(state.player.eye(), surfH, dt);
        }

        // Tree hover affordance: runs OUTSIDE the gameplay gate so the
        // highlight can fade out while a menu/pause sits over the diorama
        // (its own suppress list keeps it from picking through overlays).
        tickTreeHover(state, win, float(dt));

        // OS pointing hand: ONE owner for both hover affordances (forestry
        // tree/menu via hovHandWant, god-harvest ripe crop via hoverRipe),
        // change-triggered. Two independent per-frame setters used to stomp
        // each other — the crop hand flashed for a single frame. hoverRipe
        // is last frame's value here (recomputed below); one frame of hover
        // latency is imperceptible.
        {
            // The console freezes the inPlay block that recomputes
            // hoverRipe, so gate it here too or a frozen-true ripe hover
            // pins the hand for the whole console session.
            bool hand = !state.paused &&
                        !state.invOpen && !state.console.open &&
                        (state.hovHandWant || state.hoverRipe);
            if (hand != state.handShown) {
                if (!state.handCursor)
                    state.handCursor =
                        glfwCreateStandardCursor(GLFW_POINTING_HAND_CURSOR);
                glfwSetCursor(win, hand ? state.handCursor : nullptr);
                state.handShown = hand;
            }
        }

        // Gameplay gate: the diorama is the game — movement (click-to-move),
        // the fly/dev toggles, and destination picking all run here.
        bool inPlay = !state.paused && !state.console.open;
        if (inPlay) {
            // Dev-mode toggle: F3 flips between god-mode (instant mine, infinite
            // items) and the real-rules playtest experience.
            bool f3 = glfwGetKey(win, GLFW_KEY_F3) == GLFW_PRESS;
            if (f3 && !prevF3) {
                state.devMode = !state.devMode;
                std::printf("dev mode %s\n", state.devMode ? "ON" : "OFF");
            }
            prevF3 = f3;

            // Fly toggle: F always, or double-tap Space while in dev mode
            // (Minecraft-creative style â€” quick escape from a deep cave).
            bool fkey = glfwGetKey(win, GLFW_KEY_F) == GLFW_PRESS;
            if (fkey && !prevF) {
                state.player.fly = !state.player.fly;
                state.player.vel = glm::vec3(0);
                state.moveActive = false;   // a foot route is meaningless now
                cancelChop(state);
            }
            prevF = fkey;

            bool space = glfwGetKey(win, GLFW_KEY_SPACE) == GLFW_PRESS;
            if (space && !prevSpace) {
                if (state.devMode && lastSpacePress > 0.0 &&
                    (now - lastSpacePress) < 0.30) {
                    state.player.fly = !state.player.fly;
                    state.player.vel = glm::vec3(0);
                    state.moveActive = false;
                    cancelChop(state);
                    std::printf("fly %s\n", state.player.fly ? "ON" : "OFF");
                    lastSpacePress = -1.0;  // consume so a third tap doesn't re-toggle
                } else {
                    lastSpacePress = now;
                }
            }
            prevSpace = space;

            // Click-to-move: while the right button is held (and free-cam
            // isn't repurposing it), pick the ground under the cursor
            // through the map camera every frame — press sets, hold steers.
            // mapCursorPick clips to the rendered slab (slice plane, X-ray
            // crust, fog holes), and the CPU raycast only sees the resident
            // window, so clicks past it simply don't land (the diorama's
            // own "walk range"). Holds drifting over the minimap don't
            // pick through it.
            if (!state.mapView.freeCam && state.rightDown &&
                !state.player.fly && !state.invOpen) {   // fly steers with WASD; routes are for feet
                double mx, my; glfwGetCursorPos(win, &mx, &my);
                if (minimapHit(win, mx, my, nullptr) == 0) {
                    RayHit h = mapCursorPick(state, win, 6000.0f);
                    if (h.hit) {
                        // Stand ON the clicked voxel: destination is its top.
                        glm::vec3 dest(float(h.voxel.x) + 0.5f,
                                       float(h.voxel.y) + 1.0f,
                                       float(h.voxel.z) + 0.5f);
                        // Plan a route around trunks and too-low canopy
                        // (A*, path.cpp). A fresh click plans immediately; a
                        // held steer replans on a throttle or when the
                        // cursor has moved meaningfully. Freshness compares
                        // against the last PLANNED dest, never the route end
                        // (a partial route's end differs from the click every
                        // frame — comparing it would bypass the throttle and
                        // run a full search per frame). XZ only: the Y of a
                        // canopy pick is not steering intent.
                        bool fresh = !state.moveActive ||
                            glm::distance(glm::vec2(dest.x, dest.z),
                                          glm::vec2(state.movePlanDest.x,
                                                    state.movePlanDest.z)) > 6.0f;
                        if (fresh || state.repathT <= 0.0f) {
                            state.moveActive =
                                findPath(state.world, state.player.pos, dest,
                                         state.movePath);
                            state.moveWp       = 0;
                            state.movePlanDest = dest;
                            state.repathT      = 0.30f;
                            if (fresh) {
                                // Throttled hold-replans keep the stall
                                // clock running, or a wedge under a held
                                // button never trips the bail-out.
                                state.moveStallT  = 0.0f;
                                state.repathTries = 0;
                            }
                            // The stud marks where the route actually ends —
                            // a blocked click walks as close as it can get.
                            state.moveTarget = state.moveActive
                                             ? state.movePath.back() : dest;
                        }
                    }
                }
            }

            // Farm Mode footprint preview + hold-drag painting: pick the cell
            // under the cursor every frame, test the active tool (green/red),
            // and — while RMB is held — act on each new valid cell entered.
            if (state.farm.mode && !state.invOpen) {
                double mx, my; glfwGetCursorPos(win, &mx, &my);
                bool overHud = minimapHit(win, mx, my, nullptr) != 0 ||
                               farmBarOver(win, mx, my, state.farm.tool);
                RayHit fh = overHud ? RayHit{}
                                    : mapCursorPick(state, win, 6000.0f);
                if (fh.hit) {
                    state.farm.cursorValid = true;
                    state.farm.cursorCell  = farmCellOrigin(fh.voxel.x, fh.voxel.z);
                    int ty = fh.voxel.y;   // TILL fills this with the bed level
                    state.farm.cursorOk    = farmToolValid(
                        state.world, state.farm, state.farm.tool,
                        state.farm.cursorCell, fh.voxel.y, &ty);
                    state.farm.cursorY     = ty;
                } else {
                    state.farm.cursorValid = false;
                }
                if (state.farm.painting && state.farm.cursorValid &&
                    state.farm.cursorOk &&
                    state.farm.cursorCell != state.farm.lastActedCell) {
                    farmActCell(state, state.farm.tool,
                                state.farm.cursorCell, state.farm.cursorY);
                    state.farm.lastActedCell = state.farm.cursorCell;
                }
            }

            // God-harvest hover: the ripe crop under the cursor lights up —
            // the "this is harvestable" affordance (and the negative case:
            // unripe crops don't light). Runs in normal play AND Farm Mode
            // (gather is contextual). The pointing-hand cursor itself is
            // applied by the shared owner after tickTreeHover — this block
            // only records hoverRipe.
            if (!state.invOpen) {
                double mx, my; glfwGetCursorPos(win, &mx, &my);
                bool overHud = minimapHit(win, mx, my, nullptr) != 0 ||
                               (state.farm.mode &&
                                farmBarOver(win, mx, my, state.farm.tool));
                bool ripe = false;
                if (!overHud) {
                    RayHit h = mapCursorPick(state, win, 6000.0f);
                    if (h.hit) {
                        glm::ivec2 cell = farmCellOrigin(h.voxel.x, h.voxel.z);
                        glm::ivec3 lo, hi;
                        if (farmRipeCropBounds(state.farm, cell, lo, hi)) {
                            state.hoverLo = lo; state.hoverHi = hi; ripe = true;
                        }
                    }
                }
                state.hoverRipe = ripe;
            } else {
                state.hoverRipe = false;
            }
            state.ringT = std::max(0.0f, state.ringT - float(dt) * 2.6f);

            VOX_ZONE_BEGIN(zPlayer, "player tick");
            tickPlayer(state, win, dt);
            VOX_ZONE_END(zPlayer);

            // -- Forestry chop session ------------------------------------
            // Walk to the trunk, face it, swing on the clip's impact beat;
            // the final swing fells FROM THE CLICK-TIME SNAPSHOT — fellTree/
            // uprootStump re-read every voxel against the current world, so
            // drift since the click (fire, other fells) filters per voxel
            // without re-running the whole identification.
            if (state.chopPhase != 0) {
                glm::ivec3 tv = state.chopTargetVox;
                uint8_t tm = state.world.getVoxel(tv.x, tv.y, tv.z);
                // A crop session whose final tug already collected is legally
                // cropless — it must stay "alive" through the rest of the clip
                // so the stand-up recover plays out (the swing block ends it).
                bool alive = state.chopKind == 1
                           ? tm == MAT_STONE
                           : state.chopKind == 2
                           ? (farmHasRipeCropAt(state.farm, state.chopCell) ||
                              state.swingsDone >= state.swingsNeed)
                           : tm == MAT_WOOD || tm == MAT_DESERT_WOOD ||
                             tm == MAT_CHAR || tm == MAT_CACTUS_FLESH;
                if (!alive || state.player.fly) {
                    std::printf("[forestry] cancel: alive=%d fly=%d (target mat %d)\n",
                                int(alive), int(state.player.fly), int(tm));
                    cancelChop(state);
                }
            }
            // The session target: trunk base + radius for trees, boulder
            // base + footprint radius for rocks (same walk/swing machinery).
            glm::ivec3 gatherBase =
                  state.chopKind == 1 ? state.chopRock.base
                : state.chopKind == 2 ? glm::ivec3(state.chopCell.x + FARM_CELL / 2, 0,
                                                   state.chopCell.y + FARM_CELL / 2)
                : state.chopTree.base;
            float gatherCoreR =
                  state.chopKind == 1 ? state.chopRock.radius
                : state.chopKind == 2 ? float(FARM_CELL) * 0.5f
                : state.chopTree.trunkR;
            if (state.chopPhase == 1) {   // walking to the target
                glm::vec2 d(state.player.pos.x - (gatherBase.x + 0.5f),
                            state.player.pos.z - (gatherBase.z + 0.5f));
                // Ring sized to where a body can actually STAND (stand ring
                // is coreR+3.5, and A* may finish a couple voxels short of
                // the exact tile) — plus accept "route done, next to the
                // stand tile" outright. Rocks measure from the SURFACE on
                // the approach side (per-sector reach): the global radius
                // let the walk stop with metres of air between pick and
                // stone on the narrow side of a rotated ellipsoid.
                float arriveR;
                if (state.chopKind == 1) {
                    float ang = std::atan2(d.y, d.x);
                    int sec = int(std::floor(ang / 0.3926991f + 0.5f)) & 15;
                    arriveR = state.mineSectorR[sec] + 4.5f;
                } else {
                    arriveR = std::max(11.0f, gatherCoreR + 7.0f);
                }
                // Y gate against the STAND tile (its height is known and
                // walkable) — gating on base.y strands sessions whose stand
                // sits on a bank, or under a floating leaf blob.
                bool okY = std::abs(state.player.pos.y - state.chopStand.y) <= 4.5f;
                bool nearStand = glm::distance(
                    glm::vec2(state.player.pos.x, state.player.pos.z),
                    glm::vec2(state.chopStand.x, state.chopStand.z)) < 4.0f;
                bool arrived = okY && (glm::dot(d, d) < arriveR * arriveR ||
                                       (!state.moveActive && nearStand));
                if (arrived) {
                    state.moveActive = false;
                    state.chopPhase = 2;
                    state.chopSwingT = 0.0f;
                    // Swing at the face we actually stopped at, not the one
                    // planned from the stand tile.
                    if (state.chopKind == 1 && !state.chopRock.stone.empty())
                        state.chopTargetVox = rockStrikeVoxel(
                            state.chopRock,
                            glm::vec2(state.player.pos.x, state.player.pos.z));
                    std::printf("[forestry] arrived — %s\n",
                                state.chopKind == 1 ? "mining" : "chopping");
                } else if (!state.moveActive) {
                    // Route finished short (partial A*, or the stall handler
                    // gave up mid-descent): replan fresh from HERE — long
                    // cross-country routes legitimately arrive in several
                    // legs — then give up rather than hang in Walking.
                    if (++state.chopReplans > 3 ||
                        !findPath(state.world, state.player.pos,
                                  state.chopStand, state.movePath)) {
                        std::printf("[forestry] cancel: walk gave up (replans %d)\n",
                                    state.chopReplans);
                        cancelChop(state);
                    } else {
                        state.moveActive = true;
                        state.moveWp = 0;
                        state.moveTarget = state.movePath.back();
                        state.moveStallT = 0.0f;
                        state.repathTries = 0;
                    }
                }
            }
            if (state.chopPhase == 2) {   // swinging
                if (isInWater(state.world, state.player.pos)) {
                    cancelChop(state);
                } else {
                    // Square up to the target.
                    glm::vec3 bp(gatherBase.x + 0.5f, 0, gatherBase.z + 0.5f);
                    float wantYaw = std::atan2(bp.z - state.player.pos.z,
                                               bp.x - state.player.pos.x);
                    float dy = wantYaw - state.player.yaw;
                    while (dy >  3.14159265f) dy -= 6.2831853f;
                    while (dy < -3.14159265f) dy += 6.2831853f;
                    state.player.yaw += glm::clamp(dy, -10.0f * float(dt),
                                                        10.0f * float(dt));
                    // Swing beat: impact at 55% of the swing cycle (where the
                    // authored clip's tool lands; the avatar clip is slaved to
                    // chopSwingT below so effects and pose never drift).
                    int gatherClip = state.chopKind == 1 ? state.pcMine
                                   : state.chopKind == 2 ? state.pcHarvest
                                                         : state.pcChop;
                    float clipLen = state.chopKind == 1 ? 1.0f : 0.8333f;
                    if (state.playerAvatar && gatherClip >= 0) {
                        const CreatureClip& cl =
                            state.playerModel.clips[gatherClip];
                        clipLen = float(cl.count) / cl.fps;
                    }
                    // Harvest's pluck lands at frame 14/24 (~0.583); chop/mine 55%.
                    float impactT = clipLen * (state.chopKind == 2 ? 0.583f : 0.55f);
                    float prevT = state.chopSwingT;
                    state.chopSwingT += float(dt);
                    if (state.chopKind == 2) {
                        // Crop pluck. Root vegetables take TWO tugs — the first
                        // strains and showers soil ("it's stuck!"), the second
                        // gives and fires the collect; everything else pops on
                        // the first beat. The one-shot Harvest clip replays per
                        // tug (the pose is fmod'd into it below) and finishes
                        // its stand-up recover before ending — cancelling at
                        // impact would snap the gardener back to Idle mid-motion.
                        int prevTug = int(std::floor((prevT - impactT) / clipLen));
                        int curTug  = int(std::floor((state.chopSwingT - impactT) / clipLen));
                        if (curTug > prevTug && state.swingsDone < state.swingsNeed) {
                            glm::vec3 at = glm::vec3(state.chopTargetVox) + glm::vec3(0.5f);
                            int spc = farmCropSpeciesAt(state.farm, state.chopCell);
                            bool root = spc >= 0 && CROP_DEFS[spc].root;
                            if (++state.swingsDone < state.swingsNeed) {
                                // Strain tug: soil + leaves shake loose.
                                state.particles.spawnDust(at, MAT_FARMLAND, 7);
                                state.particles.spawnDust(
                                    at + glm::vec3(0.0f, 1.0f, 0.0f), MAT_CROP_LEAF, 4);
                                std::printf("[farm] TUG %d/%d cell %d,%d\n",
                                            state.swingsDone, state.swingsNeed,
                                            state.chopCell.x, state.chopCell.y);
                            } else {
                                state.particles.spawnDust(
                                    at, spc >= 0 ? CROP_DEFS[spc].flesh
                                                 : uint8_t(MAT_PUMPKIN), 5);
                                collectProduce(state, state.chopCell);
                                std::printf("[farm] PLUCK cell %d,%d%s\n",
                                            state.chopCell.x, state.chopCell.y,
                                            root ? " (yank!)" : "");
                            }
                        }
                        if (state.swingsDone >= state.swingsNeed &&
                            state.chopSwingT >= clipLen * float(state.swingsNeed))
                            cancelChop(state);
                    } else {
                    int prevSwing = int(std::floor((prevT - impactT) / clipLen));
                    int curSwing  = int(std::floor((state.chopSwingT - impactT) / clipLen));
                    if (curSwing > prevSwing) {
                        glm::ivec3 tv = state.chopTargetVox;
                        uint8_t tm = state.world.getVoxel(tv.x, tv.y, tv.z);
                        state.particles.spawnDust(glm::vec3(tv) + glm::vec3(0.5f),
                                                  tm ? tm : (state.chopKind == 1
                                                             ? MAT_STONE
                                                             : MAT_WOOD), 6);
                        if (++state.swingsDone >= state.swingsNeed &&
                            state.chopKind == 1) {
                            // Crumble. mineRock re-reads per voxel, so drift
                            // since the click filters exactly like the fell.
                            clearTreeHover(state);
                            mineRock(state.forestry, state.world,
                                     state.chopRock, state.particles);
                            std::printf("[forestry] CRUMBLE: %zu stone\n",
                                        state.chopRock.stone.size());
                            cancelChop(state);
                        } else if (state.swingsDone >= state.swingsNeed) {
                            // Timber. Fell from the snapshot (stumps,
                            // charred skeletons, leaf blobs, and a full fall
                            // roster all fall back to the direct shatter).
                            // Reuse the session's click-time TreeInfo:
                            // fellTree re-reads every voxel against the
                            // current world, so drift since the collect is
                            // filtered per voxel — re-running the whole
                            // flood-fill here was a big slice of the fell
                            // hitch.
                            clearTreeHover(state);
                            if (fellTree(state.forestry, state.world,
                                         state.chopTree, state.player.pos,
                                         state.particles, state.atlasDyn)) {
                                state.renderer.uploadCreatureAssets(state.atlasDyn);
                                std::printf("[forestry] TIMBER: %zu wood %zu leaves\n",
                                            state.chopTree.wood.size(),
                                            state.chopTree.leaves.size());
                            } else {
                                // Fall roster full / nothing above the cut:
                                // clear in place instead.
                                uprootStump(state.forestry, state.world,
                                            state.chopTree, state.particles);
                                std::printf("[forestry] cleared in place: %zu wood\n",
                                            state.chopTree.wood.size());
                            }
                            cancelChop(state);
                        }
                    }
                    }   // end: crop one-shot vs tree/rock multi-swing
                }
            }

            // Stall handling: if the world stops the walk anyway (physics
            // wedge the path didn't predict, lost window data), replan from
            // HERE up to twice — the wedged spot becomes an obstacle seen
            // from a new angle — then give up rather than jog in place.
            state.repathT -= dt;
            if (state.moveActive) {
                float hv = glm::length(glm::vec2(state.player.vel.x,
                                                 state.player.vel.z));
                state.moveStallT = (hv < WALK_SPEED * 0.15f)
                                 ? state.moveStallT + dt : 0.0f;
                if (state.moveStallT > 0.6f) {
                    state.moveStallT = 0.0f;
                    // Replan toward what was CLICKED (a partial route's end
                    // may itself be the unreachable spot we wedged short of).
                    if (++state.repathTries > 2 ||
                        !findPath(state.world, state.player.pos,
                                  state.movePlanDest, state.movePath)) {
                        state.moveActive = false;
                    } else {
                        state.moveWp = 0;
                        state.moveTarget = state.movePath.back();
                    }
                }
            }

            // Crop growth: advance every planted crop through its stages
            // (accelerated) and rewrite its voxels on a stage change. One
            // batched upload for all crops this frame.
            if (!state.farm.crops.empty()) {
                EditDelta gd;
                // Growth rides the in-game clock (obeys /time speed) at the
                // runtime per-crop pacing (/croptime). Age stays in in-game
                // seconds — at x1 that equals real seconds, so the offline path
                // (real elapsed) needs no conversion.
                farmTick(state.world, state.farm, float(dt) * state.timeSpeed,
                         state.cropGrowSecs / float(FARM_STAGE_MAX), gd);
                state.world.uploadDelta(gd);
            }

        }

        // Wildlife: herd spawning + state machines + terrain-following motion.
        // Transient like fire/water; freezes with the pause menu.
        // Spawn-shaping /tune edits re-derive the herds once the sliders have
        // been quiet for a beat (herds despawn + respawn under the new rules).
        if (state.creatureRetuneAt > 0 && glfwGetTime() >= state.creatureRetuneAt) {
            state.creatureRetuneAt = 0;
            state.creatures.reset();
        }
        if (!state.paused) {
            // Second despawn anchor: the map's orbit camera (a watched herd
            // must not vanish because the BODY is far away).
            glm::vec3 viewPos = state.player.pos;
            {
                glm::vec3 vf;
                state.mapView.camera(viewPos, vf);
            }
            VOX_ZONE_BEGIN(zCreatures, "creatures");
            state.creatures.update(state.world, state.gen.seed,
                                   state.player.pos, viewPos, dt);
            VOX_ZONE_END(zCreatures);
        }

        // Forestry physics: falling trees + debris items (magnet pickup).
        // Freezes with the pause menu like the wildlife. Once the last fall
        // retires, the dynamic atlas resets to the base blob (it only ever
        // GROWS while falls are live — see GameState::atlasDyn).
        if (!state.paused) {
            glm::vec3 chest = state.player.pos + glm::vec3(0.0f, 10.0f, 0.0f);
            bool retired = forestryTick(state.forestry, state.world, chest,
                                        state.particles, float(dt));
            if (retired) {
                // Compact on EVERY retire (base + still-live falls), not just
                // when the roster empties — a sustained chop chain would
                // otherwise grow the dynamic atlas without bound.
                forestryCompactAtlas(state.forestry, state.atlasBase,
                                     state.atlasDyn);
                state.renderer.uploadCreatureAssets(state.atlasDyn);
            }
            // Amortized leaf-support decay (the "no hanging leaves"
            // invariant): a few budgeted slices per frame, crumbling any
            // canopy the latest fell/burn left unsupported.
            forestryDecayTick(state.forestry, state.world, state.particles);
            // Bank forestry pickups: absorbed debris tallies into
            // forestry.collected[]; drain the per-material deltas into the
            // satchel so chopped wood/leaves/cactus land in the bag. Only
            // advance drained[] by what the bag ACCEPTED — a full bag's
            // leftover stays pending and retries as space frees, instead of
            // being silently written off.
            for (int m = 1; m < 256; ++m) {
                int d = state.forestry.collected[m] - state.forestryDrained[m];
                if (d > 0) {
                    int leftover = state.hotbar.giveItem(uint8_t(m), d);
                    state.forestryDrained[m] += d - leftover;
                }
            }
        }

        // A stall or replan can leave the chop's walk dead while the session
        // still says Walking — the replan logic above owns that; here we only
        // keep the context menu glued to reality (modal opens close it).
        if (state.ctxOpen &&
            (state.paused || state.console.open ||
             state.mapView.xray || state.invOpen || state.tuneOpen))
            state.ctxOpen = false;
        if (state.ctxOpen)
            state.ctxHover = ctxRowHit(state, win, state.lastMouseX,
                                       state.lastMouseY);

        state.particles.step(state.paused ? 0.0f : dt);

        // Per-section timers: when a frame is slow, dump the breakdown.
        auto secNow = []() { return glfwGetTime(); };
        double tSec0 = secNow();

        // Water simulation tick. ~7 Hz feels right -- fast enough that flowing
        // looks alive, slow enough that placing a block of water doesn't
        // explode outward in one frame.
        double tWater = 0.0;
        if (!state.paused && now >= nextWaterTick) {
            VOX_ZONE_N("water sim");
            double w0 = secNow();
            EditDelta wd;
            simulateWater(state.world, wd, waterRng);
            state.world.uploadDelta(wd);
            tWater = secNow() - w0;
            nextWaterTick = now + 0.04;
        }

        // Fire automaton tick (~7 Hz): brisk, watchable spread; it's a one-way
        // state machine so the active set drains on its own. One batched
        // uploadDelta per tick, same path as water.
        if (!state.paused && now >= nextFireTick) {
            VOX_ZONE_N("fire sim");
            EditDelta fd;
            // Burned-away foliage seeds the leaf-support decay sweep: a fire
            // eating a band through a crown must not strand the rest floating.
            std::vector<glm::ivec3> leafGone;
            simulateFire(state.world, state.fireSim, fd, &leafGone);
            state.world.uploadDelta(fd);
            forestryQueueLeafDecay(state.forestry, leafGone);
            // Rising embers off the flames — budgeted (≈1 in 4 active bricks,
            // capped) so they never flood the shared particle buffer.
            int emberBudget = 18;
            for (int slot : state.world.burnBricks) {
                if (emberBudget <= 0) break;
                if ((state.fireSim.rng() & 3) != 0) continue;
                glm::ivec3 b = state.world.brickOfSlot(slot);
                state.particles.spawnEmber(glm::vec3(
                    b.x * BRICK + BRICK * 0.5f,
                    b.y * BRICK + BRICK * 0.7f,
                    b.z * BRICK + BRICK * 0.5f));
                --emberBudget;
            }
            nextFireTick = now + 0.15;
        }

        // Granular (falling-sand) tick — same cadence + batched-upload pattern
        // as water; the active set drains as powder settles, so it sleeps when
        // nothing is moving. Transient (setVoxelSim → no save impact for now).
        if (!state.paused && now >= nextGranularTick) {
            VOX_ZONE_N("granular sim");
            EditDelta gd;
            simulateGranular(state.world, gd, granularRng);
            state.world.uploadDelta(gd);
            nextGranularTick = now + 0.04;
        }

        // Build per-frame view bundle for the renderer.
        int w, h;
        glfwGetFramebufferSize(win, &w, &h);

        // (The dig/place aim raycast left with that system — forestry owns
        // cursor interactions now.)
        VOX_ZONE_BEGIN(zFrameView, "FrameView build");
        FrameView fv;
        fv.width  = w;
        fv.height = h;
        fv.farOrigin = glm::vec2(float(state.gen.farOriginX), float(state.gen.farOriginZ));
        fv.sunDir = sunDir;
        fv.dayLight = dayLight;
        fv.moonDir = moonDir;
        fv.moonLight = moonLight;
        fv.hotbar   = &state.hotbar;
        // Held-item name label: fire on a change of selected slot OR its item,
        // then fade out. Quick fade-in, ~1.6s hold, ~0.4s fade.
        {
            int sel = state.hotbar.selected;
            int it  = int(state.hotbar.selectedStack().item);
            if (sel != state.hotbarPrevSel || it != state.hotbarPrevItem) {
                if (it != 0) state.hotbarNameT = 2.0f;
                state.hotbarPrevSel  = sel;
                state.hotbarPrevItem = it;
            }
            state.hotbarNameT = std::max(0.0f, state.hotbarNameT - float(dt));
            float t = state.hotbarNameT;
            float fade = std::min(std::min(1.0f, (2.0f - t) / 0.1f),
                                  std::min(1.0f, t / 0.4f));
            const ItemStack& ss = state.hotbar.selectedStack();
            fv.hotbarName     = ss.isEmpty() ? nullptr : itemDisplayName(ss.item, ss.data);
            fv.hotbarNameFade = ss.isEmpty() ? 0.0f : fade;
            // A fresh harvest pickup message overrides the label ("+1 Pumpkin").
            state.pickupMsgT = std::max(0.0f, state.pickupMsgT - float(dt));
            if (state.pickupMsgT > 0.0f) {
                float pt = state.pickupMsgT;
                fv.hotbarName     = state.pickupMsg;
                fv.hotbarNameFade = std::min(std::min(1.0f, (1.8f - pt) / 0.1f),
                                             std::min(1.0f, pt / 0.5f));
            }
        }
        // Inventory satchel: per-frame cursor pick for the highlight + footer.
        if (state.invOpen) {
            double mx, my; glfwGetCursorPos(win, &mx, &my);
            bool tidy = false;
            state.invHover   = bagHitTest(win, mx, my, tidy);
            state.invTidyHot = tidy;
        } else {
            state.invHover = -1;
            state.invTidyHot = false;
        }
        fv.invOpen    = state.invOpen ? 1 : 0;
        fv.invHover   = state.invHover;
        fv.invTidyHot = state.invTidyHot ? 1 : 0;
        // Tree hover highlight (mask uploaded by tickTreeHover).
        fv.hoverOn       = state.hovActive ? 1 : 0;
        fv.hoverLo       = glm::vec3(state.hovLo);
        fv.hoverDims     = glm::vec3(state.hovDims);
        fv.hoverStrength = state.hovStrength;
        // Forestry context menu ("Choose Option").
        fv.ctxOpen = state.ctxOpen;
        if (state.ctxOpen) {
            fv.ctxPos   = state.ctxPos;
            fv.ctxRowN  = 2;
            fv.ctxHover = state.ctxHover;
            std::snprintf(fv.ctxText[0], sizeof(fv.ctxText[0]), "Choose Option");
            // Crop menus name the vegetable ("Harvest Carrot").
            char harvestLbl[24] = "Harvest";
            if (state.ctxKind == 2) {
                int spc = farmCropSpeciesAt(state.farm, state.ctxCell);
                if (spc >= 0)
                    std::snprintf(harvestLbl, sizeof harvestLbl, "Harvest %s",
                                  CROP_DEFS[spc].name);
            }
            std::snprintf(fv.ctxText[1], sizeof(fv.ctxText[1]), "%s",
                          state.ctxKind == 2      ? harvestLbl
                        : state.ctxKind == 1      ? "Mine Rock"
                        : state.ctxTree.isCactus  ? "Chop down Cactus"
                                                  : "Chop down Tree");
            std::snprintf(fv.ctxText[2], sizeof(fv.ctxText[2]), "Cancel");
            fv.ctxIcon[0] = state.ctxKind == 2 ? 2    // spade (harvest)
                          : state.ctxKind == 1 ? 3    // pick
                                               : 1;   // axe
            fv.ctxIcon[1] = 0;
        }
        fv.paused = state.paused;
        fv.devMode = state.devMode;
        fv.hideHud = state.hudHidden;
        // Aggregate burning bricks into a handful of warm point lights so fire
        // casts light onto its surroundings (esp. at night). Coarse-bucketed so
        // a whole burning tree is ~1-3 lights, not hundreds; burnBricks is
        // radius-bounded, so this is cheap. (Linear cluster search, cap 24.)
        fv.fireLightN = 0;
        if (!state.world.burnBricks.empty()) {
            constexpr int FLMAX = FrameView::MAX_FIRE_LIGHTS;
            constexpr int CELL  = 24;                 // voxels per cluster (~2.4 m)
            glm::vec3 acc[FLMAX]; int cnt[FLMAX] = {}; int64_t keyOf[FLMAX]; int nL = 0;
            for (int slot : state.world.burnBricks) {
                glm::ivec3 b = state.world.brickOfSlot(slot);
                int X = b.x * BRICK, Y = b.y * BRICK, Z = b.z * BRICK;
                int64_t key = (int64_t(floorDiv(X, CELL)) & 0xFFFFF)
                            | ((int64_t(floorDiv(Y, CELL)) & 0xFFFFF) << 20)
                            | ((int64_t(floorDiv(Z, CELL)) & 0xFFFFF) << 40);
                int idx = -1;
                for (int i = 0; i < nL; ++i) if (keyOf[i] == key) { idx = i; break; }
                if (idx < 0) { if (nL >= FLMAX) continue; idx = nL++; keyOf[idx] = key;
                               acc[idx] = glm::vec3(0); cnt[idx] = 0; }
                acc[idx] += glm::vec3(X + BRICK * 0.5f, Y + BRICK * 0.7f, Z + BRICK * 0.5f);
                ++cnt[idx];
            }
            for (int i = 0; i < nL; ++i) {
                glm::vec3 p = acc[i] / float(cnt[i]);
                float str = glm::clamp(cnt[i] / 6.0f, 0.35f, 1.6f);   // denser fire = brighter
                fv.fireLights[i] = glm::vec4(p, str);
            }
            fv.fireLightN = nL;
        }
        fv.showGpuTimers = state.showGpuTimers;
        fv.taa = state.taaEnabled;
        fv.ringsOn = state.ringsOn;
        fv.colorMode = state.colorMode;
        fv.slopeShade = state.slopeShade;
        std::memcpy(fv.tweak, state.tweak, sizeof(state.tweak));
        std::memcpy(fv.matTune, state.matTune, sizeof(state.matTune));
        fv.tuneOpen = state.tuneOpen ? 1 : 0;
        fv.tunePage = state.tunePage;
        fv.tuneSel  = state.tuneSel;
        fv.tuneMatIdx = state.tuneMatIdx;
        // Creatures page: hand the renderer the selected species' live values.
        if (state.tunePage == 2 && !state.creatures.defs.empty()) {
            state.tuneSpeciesIdx %= int(state.creatures.defs.size());
            CreatureDef& td = state.creatures.defs[state.tuneSpeciesIdx];
            std::snprintf(fv.tuneSpecies, sizeof(fv.tuneSpecies), "%s",
                          td.name.c_str());
            for (int i = 0; i < CREATURE_TUNE_N; ++i)
                fv.tuneCreature[i] = *creatureTuneField(td, i);
        }
        // Height-fog rhythm: thickest when the sun is LOW but up (dawn/dusk),
        // burning off toward noon; a per-day roll makes some mornings misty
        // and others crystal. HEAVY ignores the clock for tuning.
        {
            float lowSun = glm::clamp(sunDir.y / 0.28f, 0.0f, 1.0f);
            float band   = lowSun * (1.0f - lowSun) * 4.0f;   // peaks at elev ~8 deg
            static int   dayIdx  = 0;                         // wraps of the clock
            static float prevTod = -1.0f;
            // (A backwards /time jump across this window can bump dayIdx —
            // harmless: it only rerolls the morning mist.)
            if (prevTod > 0.6f && state.timeOfDay < 0.4f) ++dayIdx;
            prevTod = state.timeOfDay;
            float hs   = std::sin(float(dayIdx) * 12.9898f + 4.1414f) * 43758.5453f;
            float roll = 0.25f + 0.75f * (hs - std::floor(hs));
            float fogMax = state.tweak[12] * 1e-3f;      // "fog.max" (tweaks.h)
            float dens   = (state.fogMode == 0) ? 0.0f
                         : (state.fogMode == 2) ? fogMax * 1.4f
                         : fogMax * band * roll;
            fv.fogDensity = dens;
            fv.fogFloor   = float(SURFACE_Y) - 24.0f;
        }
        fv.console = &state.console;
        // Explored-world map: camera, clip slab, and the target-centred
        // explored masks (rebuilt when the marks or the mask origin change).
        fv.map.xray = state.mapView.xray;
        {
            // Full-screen world view: the map is no longer a floating square
            // tile. The slab spans everything the ring cascade can draw
            // (DS[4] = 12288 is the cascade's own horizon), misses render
            // sky, and the whole screen becomes the diorama world.
            constexpr float MAP_VIEW_R = 12288.0f;
            // Pan range: the ring cascade renders far beyond the resident
            // window, so the fence is ring COVERAGE around the player (ring2
            // spans +-8192), not the window. Fine voxels where resident,
            // coarser ring data beyond — a map losing detail away from you.
            // pan() clamps the look-around offset to MapView::PAN_R; this
            // clamp is the belt-and-suspenders on the composed target (the
            // smoothed follow anchor lags the raw player by a few voxels).
            state.mapView.target.x = glm::clamp(state.mapView.target.x,
                state.player.pos.x - MapView::PAN_R, state.player.pos.x + MapView::PAN_R);
            state.mapView.target.z = glm::clamp(state.mapView.target.z,
                state.player.pos.z - MapView::PAN_R, state.player.pos.z + MapView::PAN_R);
            // Slab floor: just below the deepest explored slice, with a
            // sensible minimum thickness when only the surface is known.
            float yBot = glm::clamp(
                std::min(float(state.mapExplored.minSliceSeen *
                               MapExplored::YSLICE_VOX - 32),
                         float(SURFACE_Y - 500)),
                0.0f, float(SURFACE_Y - 100));
            state.mapView.sliceY = glm::clamp(state.mapView.sliceY,
                                              yBot + 32.0f, float(GRID_Y));
            state.mapView.camera(fv.map.camPos, fv.map.camFwd);
            // Orbit standoff for zoom-independent atmosphere (see lighting.frag).
            fv.map.camDist = glm::length(fv.map.camPos - state.mapView.target);
            // Slab centred on the PLAYER (the data centre), not the pan
            // target: the camera can never leave it in XZ (pan fence +
            // max zoom < MAP_VIEW_R), and the world simply ends in sky at
            // the cascade's horizon.
            fv.map.clipLo = glm::vec3(state.player.pos.x - MAP_VIEW_R, yBot,
                                      state.player.pos.z - MAP_VIEW_R);
            fv.map.clipHi = glm::vec3(state.player.pos.x + MAP_VIEW_R,
                                      state.mapView.sliceY,
                                      state.player.pos.z + MAP_VIEW_R);
            fv.map.playerPos = state.player.pos;
            fv.map.playerYaw = state.player.yaw;
            // Walk cycle only while actually moving; negative = idle pose.
            float mvHoriz = glm::length(glm::vec2(state.player.vel.x,
                                                  state.player.vel.z));
            fv.map.walkPhase  = (mvHoriz > 4.0f) ? state.walkAnimT : -1.0f;
            fv.map.freeCam    = state.mapView.freeCam ? 1 : 0;
            fv.map.mmVoxPerPx = state.minimapZoom;
            fv.map.moveTarget = state.moveTarget;
            fv.map.moveActive = state.moveActive ? 1 : 0;
            // Farm Mode toolbar + cursor footprint overlay.
            fv.map.farmOn    = state.farm.mode ? 1 : 0;
            fv.map.farmTool  = state.farm.tool;
            fv.map.farmSeed  = state.farm.seedSel;
            fv.map.farmCurOn = (state.farm.mode && state.farm.cursorValid) ? 1 : 0;
            fv.map.farmOk    = state.farm.cursorOk ? 1 : 0;
            fv.map.farmMin   = glm::vec3(float(state.farm.cursorCell.x),
                                         float(state.farm.cursorY + 1),
                                         float(state.farm.cursorCell.y));
            fv.map.farmSize  = float(FARM_CELL);
            // God-harvest hover glow + press ring.
            fv.map.hoverOn = state.hoverRipe ? 1 : 0;
            fv.map.hoverLo = glm::vec3(state.hoverLo);
            fv.map.hoverHi = glm::vec3(state.hoverHi) + glm::vec3(1.0f);   // inclusive->[lo,hi+1)
            fv.map.ringOn  = (state.ringT > 0.0f) ? 1 : 0;
            fv.map.ringPos = state.ringPos;
            fv.map.ringT   = state.ringT;
            // Route trail for the shader: the character, then the remaining
            // waypoints (decimated to the array cap, destination always
            // kept), each entry (x, z, y, cumulative XZ length) so the
            // dashes march at a constant world rate along the line.
            fv.map.pathN = 0;
            if (state.moveActive && state.moveWp < int(state.movePath.size())) {
                // Select points: the character, then the remaining waypoints
                // strided down to the cap — but never striding across a big
                // vertical move (the trail draws straight chords, and a
                // chord arcing over relief loses its ground) — destination
                // always last.
                glm::vec3 pts[MapFrame::PATH_MAX];
                int n = 0;
                pts[n++] = state.player.pos;
                int remain = int(state.movePath.size()) - state.moveWp;
                int stride = std::max(1, (remain + MapFrame::PATH_MAX - 3)
                                             / (MapFrame::PATH_MAX - 2));
                for (int i = state.moveWp;
                     i < int(state.movePath.size()) - 1 &&
                     n < MapFrame::PATH_MAX - 1; ++i) {
                    if ((i - state.moveWp) % stride == 0 ||
                        std::abs(state.movePath[i].y - pts[n - 1].y) > 12.0f)
                        pts[n++] = state.movePath[i];
                }
                pts[n++] = state.movePath.back();
                // w = distance-to-DESTINATION along the polyline: the dash
                // pattern anchors to the fixed end, so it holds still while
                // the character advances and waypoints are consumed.
                fv.map.pathN = n;
                float w2 = 0.0f;
                fv.map.path[n - 1] = glm::vec4(pts[n - 1].x, pts[n - 1].z,
                                               pts[n - 1].y, 0.0f);
                for (int i = n - 2; i >= 0; --i) {
                    w2 += glm::length(glm::vec2(pts[i + 1].x - pts[i].x,
                                                pts[i + 1].z - pts[i].z));
                    fv.map.path[i] = glm::vec4(pts[i].x, pts[i].z, pts[i].y, w2);
                }
            }
            // Explored mask: a MASK_CH^2-chunk square centred on the PLAYER
            // — the same centre as the slab, so it covers every visible
            // point at any pan. Recentred with hysteresis so rebuilds and
            // uploads stay rare.
            glm::ivec2 tch(chunkOfVoxel(int(std::floor(state.player.pos.x))),
                           chunkOfVoxel(int(std::floor(state.player.pos.z))));
            glm::ivec2 ctr = state.mapMaskOrigin
                           + glm::ivec2(MapMasks::MASK_CH / 2);
            if (state.mapMaskOrigin.x == INT32_MIN ||
                std::abs(tch.x - ctr.x) > 16 || std::abs(tch.y - ctr.y) > 16)
                state.mapMaskOrigin = tch - glm::ivec2(MapMasks::MASK_CH / 2);
            if (state.mapMasksVersion != state.mapExplored.version ||
                state.mapMasksOrigin  != state.mapMaskOrigin) {
                state.mapExplored.buildMasks(state.mapMaskOrigin,
                                             state.mapMasks);
                state.mapMasksVersion = state.mapExplored.version;
                state.mapMasksOrigin  = state.mapMaskOrigin;
                ++state.mapMasksStamp;
            }
            fv.map.masks        = &state.mapMasks;
            fv.map.maskOrigin   = state.mapMaskOrigin;
            fv.map.masksVersion = state.mapMasksStamp;
        }

        // Visible wildlife: pose the nearest creatures (nearest-first, capped,
        // culled against the diorama clip slab) into per-part instance
        // records. The vector is app-owned scratch; fv carries a pointer.
        {
            glm::vec3 creCam = fv.map.camPos;
            CreatureBound bounds[FrameView::MAX_CREATURES];
            // Slot budget: the bound arrays are exactly MAX_CREATURES wide
            // (mirrored by uCreBound[32] in lighting.frag) — reserve one for
            // the player and one per live falling tree BEFORE the wildlife
            // fill, or a busy herd + a fell writes past the end.
            int nFalls = int(state.forestry.falls.size());
            int maxCre = FrameView::MAX_CREATURES -
                         (state.playerAvatar ? 1 : 0) - nFalls;
            fv.creatureN = state.creatures.buildPartInstances(
                state.creatureParts, bounds, std::max(0, maxCre),
                state.world, state.player.pos,
                creCam, 1200.0f, fv.map.clipLo, fv.map.clipHi);
            for (int i = 0; i < fv.creatureN; ++i) {
                fv.creatureBound[i] = glm::vec4(bounds[i].center, bounds[i].radius);
                fv.creatureRange[i] = glm::ivec2(bounds[i].firstPart, bounds[i].partCount);
            }
            // Falling trees: rigid 2-part models through the same pipeline,
            // with bounds for their traced cast shadows. They must sit
            // BEFORE the player (lighting.frag treats the LAST bound as the
            // avatar) and skip the contact-blob loop (a tree-sized bound
            // would paint a giant dark disc — creBlobSkip covers them).
            {
                CreatureBound fb[Forestry::MAX_FALLS];
                int room = FrameView::MAX_CREATURES - fv.creatureN -
                           (state.playerAvatar ? 1 : 0);
                int nb = appendFallInstances(state.forestry, state.creatureParts,
                                             fb, std::max(0, std::min(room, Forestry::MAX_FALLS)));
                fv.creBlobSkip = glm::ivec2(fv.creatureN, nb);
                for (int i = 0; i < nb; ++i) {
                    fv.creatureBound[fv.creatureN] =
                        glm::vec4(fb[i].center, fb[i].radius);
                    fv.creatureRange[fv.creatureN] =
                        glm::ivec2(fb[i].firstPart, fb[i].partCount);
                    ++fv.creatureN;
                }
            }
            // Player avatar rides the same pipeline (same G-buffer pass, same
            // traced shadows — the player finally casts one).
            if (state.playerAvatar) {
                float hs = glm::length(glm::vec2(state.player.vel.x,
                                                 state.player.vel.z));
                bool swim = isInWater(state.world, state.player.pos);
                bool mining   = state.chopPhase == 2 && state.chopKind == 1 &&
                                state.pcMine >= 0 && !swim &&
                                state.player.onGround;
                bool chopping = state.chopPhase == 2 && state.chopKind == 0 &&
                                state.pcChop >= 0 && !swim &&
                                state.player.onGround;
                bool harvesting = state.chopPhase == 2 && state.chopKind == 2 &&
                                  state.pcHarvest >= 0 && !swim &&
                                  state.player.onGround;
                int want = state.pcIdle;
                if (mining)
                    want = state.pcMine;
                else if (chopping)
                    want = state.pcChop;
                else if (harvesting)
                    want = state.pcHarvest;
                else if (swim)
                    // No Swim clip yet: a slow walk cycle reads as paddling
                    // (and beats freezing on the last Fall frame underwater).
                    want = (hs > 4.0f) ? state.pcWalk : state.pcIdle;
                else if (!state.player.onGround && state.pcFall >= 0)
                    want = (state.player.vel.y > 20.0f && state.pcJump >= 0)
                         ? state.pcJump : state.pcFall;
                // Crouch (Shift sneak): crouchT covers both the key being held
                // and "stuck crouched under a low gap", so the pose follows
                // the hitbox, not just the input.
                else if (state.crouchT > 0.5f && state.pcCrouchIdle >= 0 &&
                         state.pcCrouchWalk >= 0)
                    want = (hs > 4.0f) ? state.pcCrouchWalk : state.pcCrouchIdle;
                else if (hs > WALK_SPEED * 1.1f && state.pcSprint >= 0)
                    want = state.pcSprint;
                else if (hs > 4.0f) want = state.pcWalk;
                if (want != state.pClip) {
                    state.pClip = want; state.pClipT = 0.0f; state.pFadeT = 0.0f;
                }
                // Stride rate-lock: the clips are authored at human cadence,
                // the game moves Minecraft-fast (walk 48/14 = 3.43x — the
                // clamp must clear it or default walking foot-skates).
                float rate = 1.0f;
                if (state.pClip == state.pcWalk)   rate = swim ? 0.6f : hs / 14.0f;
                if (state.pClip == state.pcSprint) rate = hs / 42.0f;
                // 6.0: CrouchWalk's authored stance-sweep measured in the rig
                // (0.182 m/s) scaled by the Walk anchor (0.423 m/s ↔ 14).
                if (state.pClip == state.pcCrouchWalk) rate = hs / 6.0f;
                if (!state.paused) {
                    state.pClipT += dt * std::clamp(rate, 0.4f, 4.0f);
                    state.pFadeT += dt;
                }
                // The swing pose is slaved to the session's swing clock so
                // the impact dust lands exactly on the tool strike (55%
                // beat) — the fmod re-loops a non-looping fallback (Punch).
                if ((chopping && state.pClip == state.pcChop) ||
                    (mining && state.pClip == state.pcMine) ||
                    (harvesting && state.pClip == state.pcHarvest)) {
                    const CreatureClip& cl = state.playerModel.clips[state.pClip];
                    state.pClipT = std::fmod(state.chopSwingT,
                                             float(cl.count) / cl.fps);
                }
                CreatureBound b = appendModelInstances(
                    state.playerModel, state.player.pos, state.player.yaw,
                    state.pClip, state.pClipT, state.pFadeT, state.pFadePose,
                    state.pPrevXf, state.creatureParts);
                fv.creatureBound[fv.creatureN] = glm::vec4(b.center, b.radius);
                fv.creatureRange[fv.creatureN] = glm::ivec2(b.firstPart, b.partCount);
                ++fv.creatureN;
                fv.playerAvatarOn = 1;
                // The equipped tool rides the right forearm while swinging —
                // one extra instance AFTER the player's range (drawn, but
                // outside every bound: no shadow-loop cost).
                if ((chopping || mining) && state.playerHandPart >= 0) {
                    const CreaturePartInstance& hand =
                        state.creatureParts[size_t(b.firstPart) +
                                            state.playerHandPart];
                    glm::ivec3 hd =
                        state.playerModel.parts[state.playerHandPart].dims;
                    state.creatureParts.push_back(
                        mining ? makeAttachedPick(state.forestry, hand, hd)
                               : makeAttachedAxe(state.forestry, hand, hd));
                }
            } else {
                // Not drawn this frame -> the prev-row cache goes stale;
                // clear it so the next append starts from zero motion.
                state.pPrevXf.clear();
            }
            // Debris items: instance-only voxel chunks (hover/spin/magnet).
            appendItemInstances(state.forestry, state.creatureParts);

            // Harvest collect flyers: each produce object flies from its plot
            // to the PLAYER with its species' theatrics — roots yank out of the
            // soil and backflip, tomato fruits pop off in sequence, the cabbage
            // head rolls and bounces, corn ears tumble, pumpkins loft-and-swoop.
            // Drawn through the creature-part pipeline (so they render in the
            // diorama). The item BANKS into the bag and the "+1" label fires on
            // ARRIVAL — you see it land, then the count goes up. Append one
            // part per live flyer BEFORE the parts pointer is published below.
            if (!state.collectFlyers.empty()) {
                // Flyers ride gameplay time: frozen while paused (like the
                // particles and the avatar clip clock), else the whole harvest
                // finishes behind the pause menu.
                const float fdt = state.paused ? 0.0f : float(dt);
                // Target the player's chest; recomputed each frame so the arc
                // tracks the avatar if it moves.
                glm::vec3 playerPt = state.player.pos + glm::vec3(0.0f, 10.0f, 0.0f);
                // Rodrigues axis-angle rotation (unit axis).
                auto rotAxis = [](glm::vec3 a, float ang) {
                    float c = std::cos(ang), s = std::sin(ang);
                    glm::mat3 K(0.0f,  a.z, -a.y,      // columns of the cross-
                               -a.z, 0.0f,  a.x,       // product matrix (K v = a x v)
                                a.y, -a.x, 0.0f);
                    return glm::mat3(1.0f) * c + K * s +
                           glm::outerProduct(a, a) * (1.0f - c);
                };
                for (size_t i = 0; i < state.collectFlyers.size();) {
                    CollectFlyer& fl = state.collectFlyers[i];
                    const CreatureModel& pm =
                        state.produceModels[std::clamp(fl.species, 0,
                                                       CROP_SPECIES_N - 1)]
                                           [std::clamp(fl.grade, 0, 2)];
                    glm::vec3 dims = glm::vec3(pm.parts[0].dims);
                    // Emit one posed instance: world = T * R * S about the
                    // model centre (flyers rotate mid-air, so full affine rows).
                    auto emit = [&](glm::vec3 pos, const glm::mat3& R,
                                    float sxz, float syv) {
                        glm::mat3 RS = R;
                        RS[0] *= sxz; RS[1] *= syv; RS[2] *= sxz;
                        glm::vec3 T = pos - RS * (dims * 0.5f);
                        glm::mat3 Minv = glm::inverse(RS);
                        glm::vec3 iT = -(Minv * T);
                        CreaturePartInstance q{};
                        q.rowX = glm::vec4(RS[0][0], RS[1][0], RS[2][0], T.x);
                        q.rowY = glm::vec4(RS[0][1], RS[1][1], RS[2][1], T.y);
                        q.rowZ = glm::vec4(RS[0][2], RS[1][2], RS[2][2], T.z);
                        q.invX = glm::vec4(Minv[0][0], Minv[1][0], Minv[2][0], iT.x);
                        q.invY = glm::vec4(Minv[0][1], Minv[1][1], Minv[2][1], iT.y);
                        q.invZ = glm::vec4(Minv[0][2], Minv[1][2], Minv[2][2], iT.z);
                        q.prevX = q.rowX; q.prevY = q.rowY; q.prevZ = q.rowZ;
                        q.dimsPal = glm::vec4(dims, float(pm.palBase));
                        q.meta    = glm::vec4(float(pm.parts[0].wordBase), 0, 0, 0);
                        state.creatureParts.push_back(q);
                    };
                    // Stagger: multi-fruit harvests pluck one fruit at a time.
                    // The plant itself vanished when the harvest fired, so a
                    // waiting fruit doesn't float in mid-air — it DROPS to the
                    // soil with a quick eased fall and rests there until its
                    // turn (the bush collapses, the fruits tumble, then hop to
                    // you one by one).
                    if (fl.delay > 0.0f) {
                        fl.delay -= fdt;
                        float restY = fl.groundY + dims.y * 0.5f;
                        if (fl.delay > 0.0f) {
                            float w = (fl.delay0 > 0.0f)
                                ? std::clamp((fl.delay0 - fl.delay) / 0.22f,
                                             0.0f, 1.0f)
                                : 1.0f;
                            float y = glm::mix(fl.start.y, restY, w * w);
                            float squash = 0.18f * glm::smoothstep(0.85f, 1.0f, w);
                            emit(glm::vec3(fl.start.x, y, fl.start.z),
                                 glm::mat3(1.0f), 1.0f + squash * 0.6f,
                                 1.0f - squash);
                            ++i;
                            continue;
                        }
                        fl.start.y = restY;   // launch from where it rests
                    }
                    fl.t += fdt / fl.dur;
                    if (fl.t >= 1.0f) {
                        // Bank on arrival. A FULL bag must not destroy the
                        // produce: the flyer holds at the chest and retries
                        // until a slot frees (mirrors the forestry drain's
                        // leftover handling above).
                        if (state.hotbar.giveItem(fl.item, 1, fl.data) == 0) {
                            std::snprintf(state.pickupMsg, sizeof(state.pickupMsg),
                                          "+1 %s", itemDisplayName(fl.item, fl.data));
                            state.pickupMsgT = 1.8f;
                            state.collectFlyers.erase(state.collectFlyers.begin() + i);
                            continue;
                        }
                        fl.t = 1.0f;
                        if (state.pickupMsgT <= 0.0f) {
                            std::snprintf(state.pickupMsg, sizeof(state.pickupMsg),
                                          "Bag full!");
                            state.pickupMsgT = 1.2f;
                        }
                        emit(playerPt, glm::mat3(1.0f), 1.0f, 1.0f);
                        ++i;
                        continue;
                    }
                    const float te = fl.t;
                    const float rf = float(fl.rng & 0xFFFFu) / 65535.0f; // 0..1
                    const float spinDir = (fl.rng & 0x10000u) ? 1.0f : -1.0f;
                    glm::vec3 pos = fl.start;
                    glm::mat3 R(1.0f);
                    float sxz = 1.0f, syv = 1.0f;

                    switch (fl.style) {
                    case FLY_YANK: {
                        // Phase A: dragged straight up out of the soil — slow
                        // strain, then it GIVES (cubic ease); the terrain hides
                        // whatever is still buried. The dirt bursts the moment
                        // the root breaks the surface.
                        const float pullT = 0.32f;
                        const float buriedY = fl.groundY - dims.y * 0.55f;
                        glm::vec3 outPt(fl.start.x, fl.groundY + dims.y * 0.5f + 1.5f,
                                        fl.start.z);
                        if (te < pullT) {
                            float u = te / pullT;
                            float e = u * u * u;
                            pos = glm::vec3(fl.start.x,
                                            buriedY + (outPt.y - buriedY) * e,
                                            fl.start.z);
                            R = rotAxis(glm::vec3(0, 1, 0),
                                        0.20f * std::sin(u * 18.0f) * (1.0f - u));
                            syv = 1.0f + 0.22f * u;          // stretched by the pull
                            sxz = 1.0f - 0.10f * u;
                            if (!fl.burst && e > 0.35f) {
                                fl.burst = true;
                                glm::vec3 g(fl.start.x, fl.groundY + 0.5f, fl.start.z);
                                state.particles.spawnDust(g, MAT_FARMLAND, 10);
                                state.particles.spawnDust(g, MAT_DIRT, 6);
                            }
                        } else {
                            // Phase B: airborne — one proud backflip, then swoop.
                            float v = (te - pullT) / (1.0f - pullT);
                            glm::vec3 mid = (outPt + playerPt) * 0.5f +
                                            glm::vec3(0, 16.0f, 0);
                            pos = (1 - v) * (1 - v) * outPt +
                                  2 * (1 - v) * v * mid + v * v * playerPt;
                            glm::vec3 fwd = playerPt - outPt; fwd.y = 0.0f;
                            glm::vec3 axis = glm::dot(fwd, fwd) > 1e-6f
                                ? glm::normalize(glm::vec3(-fwd.z, 0.0f, fwd.x))
                                : glm::vec3(1, 0, 0);
                            R = rotAxis(axis, spinDir * v * 6.2831853f) *
                                rotAxis(glm::vec3(0, 1, 0), spinDir * v * 3.1f);
                            float p   = std::min(v / 0.4f, 1.0f);
                            float wob = std::pow(1.0f - p, 0.75f) *
                                        std::cos(p * 15.708f);
                            syv = 1.0f + 0.26f * wob;        // release boing
                            sxz = 1.0f - 0.14f * wob;
                        }
                        break;
                    }
                    case FLY_POP: {
                        // The fruit pips off the bush — a quick inflate, a small
                        // outward kick, then a low swoop to the bag.
                        glm::vec3 kick(std::cos(rf * 6.2831853f), 0.0f,
                                       std::sin(rf * 6.2831853f));
                        glm::vec3 mid = glm::mix(fl.start, playerPt, 0.35f) +
                                        kick * 3.0f + glm::vec3(0, 7.0f, 0);
                        pos = (1 - te) * (1 - te) * fl.start +
                              2 * (1 - te) * te * mid + te * te * playerPt;
                        R = rotAxis(glm::vec3(0, 1, 0),
                                    spinDir * te * 6.2831853f);
                        float p = std::min(te / 0.18f, 1.0f);   // detach "pip"
                        sxz = 1.0f + 0.20f * (1.0f - p);
                        syv = 1.0f - 0.14f * (1.0f - p);
                        break;
                    }
                    case FLY_ROLL: {
                        // The head bounces & rolls along the ground toward the
                        // player's feet (three decaying hops, contact squash),
                        // then leaps up into the bag. The "ground" under the
                        // roll blends from the plot's soil to the player's
                        // feet height, so a harvest across a terrace step
                        // doesn't roll through the hill or hover over a dip.
                        const float rollT = 0.80f;
                        glm::vec3 a(fl.start.x, 0.0f, fl.start.z);
                        glm::vec3 b(state.player.pos.x, 0.0f, state.player.pos.z);
                        float dist = glm::distance(a, b);
                        glm::vec3 dirH = dist > 1e-3f ? (b - a) / dist
                                                      : glm::vec3(1, 0, 0);
                        glm::vec3 axis(-dirH.z, 0.0f, dirH.x);
                        float rollR   = std::max(dims.x * 0.5f, 1.0f);
                        float fullAng = dist / rollR;          // circumference roll
                        if (te < rollT) {
                            float u = te / rollT;
                            glm::vec3 xz = glm::mix(a, b, u);
                            float gy = glm::mix(fl.groundY, state.player.pos.y, u);
                            float ph = u * 3.0f;               // three hops
                            int   k  = std::min(2, int(ph));
                            float f  = ph - float(k);
                            float h  = 4.5f * std::pow(0.5f, float(k));
                            pos = glm::vec3(xz.x,
                                            gy + dims.y * 0.5f +
                                                h * 4.0f * f * (1.0f - f),
                                            xz.z);
                            R = rotAxis(axis, -fullAng * u);   // forward roll
                            float edge   = std::min(f, 1.0f - f);
                            float squash = std::max(0.0f, 1.0f - edge * 5.0f);
                            syv = 1.0f - 0.24f * squash;       // contact splat
                            sxz = 1.0f + 0.14f * squash;
                        } else {
                            float v = (te - rollT) / (1.0f - rollT);
                            glm::vec3 from(b.x,
                                           state.player.pos.y + dims.y * 0.5f,
                                           b.z);
                            pos = glm::mix(from, playerPt, v);
                            pos.y += 5.0f * v * (1.0f - v);    // the final leap
                            R = rotAxis(axis, -(fullAng + v * 2.0f));
                            syv = 1.0f + 0.15f * (1.0f - v);
                            sxz = 1.0f - 0.08f * (1.0f - v);
                        }
                        break;
                    }
                    case FLY_TUMBLE: {
                        // Ears tumble end-over-end on a flat, fast arc.
                        glm::vec3 mid = (fl.start + playerPt) * 0.5f +
                                        glm::vec3(0, 9.0f, 0);
                        pos = (1 - te) * (1 - te) * fl.start +
                              2 * (1 - te) * te * mid + te * te * playerPt;
                        glm::vec3 fwd = playerPt - fl.start; fwd.y = 0.0f;
                        glm::vec3 axis = glm::dot(fwd, fwd) > 1e-6f
                            ? glm::normalize(glm::vec3(-fwd.z, 0.0f, fwd.x))
                            : glm::vec3(1, 0, 0);
                        R = rotAxis(axis, spinDir * te * (2.0f + rf) * 6.2831853f);
                        break;
                    }
                    default: {
                        // FLY_ARC — the classic pumpkin loft-and-swoop with the
                        // decaying launch boing.
                        glm::vec3 mid = (fl.start + playerPt) * 0.5f +
                                        glm::vec3(0, 18.0f, 0);
                        pos = (1 - te) * (1 - te) * fl.start +
                              2 * (1 - te) * te * mid + te * te * playerPt;
                        float p   = std::min(te / 0.32f, 1.0f);
                        float wob = std::pow(1.0f - p, 0.75f) *
                                    std::cos(p * 15.708f);   // ~2.5 cycles
                        syv = 1.0f + 0.32f * wob;             // tall/short
                        sxz = 1.0f - 0.18f * wob;             // narrow/wide
                        break;
                    }
                    }
                    emit(pos, R, sxz, syv);
                    ++i;
                }
            }

            fv.creatureParts = state.creatureParts.data();
            fv.creaturePartN = int(state.creatureParts.size());
        }

        fv.time   = float(now - startT);
        VOX_ZONE_END(zFrameView);

        // Dev-mode readout in window title: live FPS + streaming gauges +
        // position. Throttle to ~5 Hz so we aren't thrashing the window
        // manager. FPS is EMA-smoothed off the raw frame dt so it reads as a
        // steady number, and we also carry the worst frame seen in the
        // window so transient hitches are visible (they vanish from an
        // instantaneous reading).
        {
            static double titleNext = 0.0;
            static bool   titleIsDev = false;
            static double fpsEMA = 0.0;
            static double worstMs = 0.0;
            double instMs = dtRaw * 1000.0;
            fpsEMA = (fpsEMA == 0.0) ? (1.0 / std::max(dtRaw, 1e-4f))
                                     : fpsEMA * 0.9 + (1.0 / std::max(dtRaw, 1e-4f)) * 0.1;
            worstMs = std::max(worstMs, instMs);
            if (state.devMode) {
                if (now >= titleNext) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "tiny voxels  [dev]  %.0f fps (worst %.0fms)  "
                        "pend %d  pos %.0f %.0f %.0f",
                        fpsEMA, worstMs,
                        state.streamOn ? streamer.pendingInWindow() : 0,
                        state.player.pos.x, state.player.pos.y, state.player.pos.z);
                    glfwSetWindowTitle(win, buf);
                    titleNext = now + 0.20;
                    worstMs = 0.0;           // reset per readout window
                    titleIsDev = true;
                }
            } else if (titleIsDev) {
                glfwSetWindowTitle(win, "tiny voxels");
                titleIsDev = false;
            }
        }

        // The renderer's toroidal-window origin mirrors the world's.
        state.renderer.windowOriginB = { state.world.windowOriginChunk.x * CHUNK_BRICKS,
                                         state.world.windowOriginChunk.y * CHUNK_BRICKS };
        double tR0 = secNow();
        state.renderer.renderFrame(state.world, state.particles, fv);
        double tRender = secNow() - tR0;
        double tS0 = secNow();
        VOX_ZONE_BEGIN(zSwap, "swap");
        glfwSwapBuffers(win);
        VOX_ZONE_END(zSwap);
        double tSwap = secNow() - tS0;
        VOX_FRAME_MARK;
        VOX_GPU_COLLECT;
        double tFrame = secNow() - tSec0;
        if (tFrame > 0.030) {
            std::fprintf(stderr,
                "[slow %.1fms] water %.1f  render %.1f  swap %.1f\n",
                tFrame*1000, tWater*1000, tRender*1000, tSwap*1000);
        }

        // Autosave: every 60s, only if something changed since the last save.
        // Cheap â€” extracts + compresses only the dirty chunks.
        static double lastAutosave = now;
        if (state.world.editsSinceSave && now - lastAutosave > 60.0) {
            VOX_ZONE_N("autosave");
            saveWorldEdits(state.world, state.gen, editsPath, &state.chunkCache);
            lastAutosave = now;
        }
        // Same cadence for the explored-map sidecar (a few KB, uncompressed).
        static double lastMapSave = now;
        if (state.mapExplored.dirtySinceSave && now - lastMapSave > 60.0) {
            state.mapExplored.save("map.dat", state.gen.seed);
            lastMapSave = now;
        }
        // Farm sidecar: crops change via growth (which isn't a world edit), so
        // it saves on its own cadence; the timestamp anchors offline growth.
        static double lastFarmSave = now;
        if (now - lastFarmSave > 60.0) {
            saveFarmState(state.farm, "farm.dat", state.gen.seed,
                          int64_t(std::time(nullptr)));
            lastFarmSave = now;
        }
    }

    // Stop the background cache compressor before touching the cache for the
    // final save (save reads entries; compressor mutates them).
    stopCompressor.store(true, std::memory_order_relaxed);
    cacheCompressor.join();
    if (state.ringRebuilding) state.ringRebuild.join();   // drain the ring worker

    // Final save so quitting never loses edits (or exploration).
    if (state.world.editsSinceSave)
        saveWorldEdits(state.world, state.gen, editsPath, &state.chunkCache);
    if (state.mapExplored.dirtySinceSave)
        state.mapExplored.save("map.dat", state.gen.seed);
    saveFarmState(state.farm, "farm.dat", state.gen.seed, int64_t(std::time(nullptr)));

    Audio::shutdown();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
