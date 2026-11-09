#pragma once
#include <string>
#include <vector>

// In-game dev console: scrollback + an editable input line + command memory.
// Rendering and command execution live elsewhere (render.cpp draws it;
// commands.cpp holds the registry + handlers and executes submitted lines;
// app.cpp's keyCb feeds input and persists the history to
// console_history.txt, tune.cfg-style sidecar). This struct is the text
// model + editing operations, including bash-style up/down history.
struct Console {
    static constexpr int COLS = 72;       // visible character columns
    static constexpr int ROWS = 14;       // total rows incl. the input line
    static constexpr int VISIBLE_LINES = ROWS - 1;
    // Longest storable input line: the renderer draws "> " + input + a
    // blinking cursor into COLS cells, so the cursor vanishes once the line
    // hits 72 (render.cpp). onChar enforces this while typing; history
    // recall/load must clamp too — assigning `input` bypasses onChar.
    static constexpr int MAX_INPUT = COLS - 3;
    static constexpr int HIST_MAX  = 64;  // history cap, in RAM and on disk

    bool open = false;
    std::string input;
    std::vector<std::string> scrollback;  // oldest first; we show the tail

    // -- command memory (bash-style Up/Down recall) -------------------------
    // Submitted lines, oldest..newest, stored raw (pre-"> " echo, pre-wrap).
    // histPos == -1 means live edit; otherwise it indexes history while the
    // in-progress draft waits in histStash until Down walks past the newest
    // entry. Editing a recalled line never rewrites history (bash default).
    std::vector<std::string> history;
    int histPos = -1;
    std::string histStash;

    // -- tab-completion cycle state ------------------------------------------
    // The completer itself lives in app.cpp next to the command registry
    // (it needs the command table + live species names); Console just holds
    // the in-progress cycle so every input mutation can cancel it.
    std::string tabBase;   // the final token as typed when the cycle began
    int tabIdx = -1;       // -1 = not cycling; else index into the matches

    void resetTab() { tabIdx = -1; tabBase.clear(); }

    void print(const std::string& line);  // push an output line (wraps long)
    void onChar(unsigned int cp);          // append a printable char
    void backspace();
    void clearInput() { input.clear(); histPos = -1; histStash.clear(); resetTab(); }
    std::string takeInput();               // returns + clears the input line

    bool pushHistory(const std::string& line); // on submit; false = no change
    void histPrev();                           // Up: recall older
    void histNext();                           // Down: newer / restore draft
    void loadHistory(const char* path);        // tolerant; absent file is fine
    void saveHistory(const char* path) const;  // rewrite the whole file
};
