#pragma once
#include <string>

// Console command module (commands.cpp): the registry-driven dispatcher for
// the in-game dev console, split out of app.cpp. Deliberately thin: the
// handlers act on the whole live GameState, so this header forward-declares
// it and NEVER includes game_state.h -- keep it that way (the aggregate is
// app-layer-only; see game_state.h's banner).
struct GameState;

// Execute a submitted console command line. Returns true if the console
// should STAY OPEN afterwards (info/query/error output the player should
// read), false if it should close and return to the game (action commands
// like /tp).
bool execConsoleCommand(GameState& s, const std::string& line);

// Tab completion for the console input line (bash/cmd hybrid; Shift+Tab
// cycles matches backwards). Wired to GLFW_KEY_TAB in app.cpp's keyCb.
void consoleTabComplete(GameState& s, bool back);

// -- Imported from app.cpp: deliberate back-edges, capped at TWO groups ------
// consoleTeleport needs the window-recenter path that lives with the
// streaming code in app.cpp.
void recenterWindow(GameState& s, int destX, int destZ);
// /chop, /fell, /mine and /harvest drive the gather session + hover state,
// which live with the input/animation glue in app.cpp. If a THIRD group ever
// wants to join this section, stop and extract a proper module instead.
struct TreeInfo;
struct RockInfo;
bool startChopSession(GameState& s, const TreeInfo& t);
bool startMineSession(GameState& s, const RockInfo& r);
// Plain ints (a farm cell origin) keep this header glm-free.
bool startHarvestSessionAt(GameState& s, int cellX, int cellZ);
void cancelChop(GameState& s);
void clearTreeHover(GameState& s);
