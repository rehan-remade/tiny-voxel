#include "console.h"
#include <cstdio>   // history file I/O (fopen/fgets/fprintf)

void Console::print(const std::string& line) {
    // Hard-wrap to COLS so long output stays inside the panel.
    if (line.empty()) { scrollback.emplace_back(); }
    for (size_t i = 0; i < line.size(); i += COLS) {
        scrollback.push_back(line.substr(i, COLS));
    }
    // Keep scrollback bounded.
    const size_t MAX = 200;
    if (scrollback.size() > MAX)
        scrollback.erase(scrollback.begin(), scrollback.end() - MAX);
}

void Console::onChar(unsigned int cp) {
    if (cp < 0x20 || cp > 0x7E) return;              // printable ASCII only
    if (input.size() >= size_t(MAX_INPUT)) return;   // "> " + cursor must fit
    input.push_back(char(cp));
    resetTab();                   // a real edit ends any completion cycle
}

void Console::backspace() {
    if (input.empty()) return;
    input.pop_back();
    resetTab();                   // a real edit ends any completion cycle
}

std::string Console::takeInput() {
    std::string s = input;
    input.clear();
    histPos = -1;                 // submitting always exits history browsing
    histStash.clear();
    resetTab();
    return s;
}

// -- Command memory -----------------------------------------------------------

bool Console::pushHistory(const std::string& line) {
    histPos = -1;
    histStash.clear();
    if (line.empty()) return false;
    // Bash-style ignoredups: adjacent repeats collapse so mashing the same
    // command doesn't flood the buffer; non-adjacent duplicates are kept.
    if (!history.empty() && history.back() == line) return false;
    history.push_back(line.substr(0, MAX_INPUT));
    if (history.size() > size_t(HIST_MAX)) history.erase(history.begin());
    return true;                  // history changed: caller rewrites the file
}

void Console::histPrev() {
    if (history.empty()) return;
    if (histPos < 0) {            // entering browse mode: stash the draft
        histStash = input;
        histPos = int(history.size());
    }
    if (histPos > 0) --histPos;   // clamp at oldest (no wrap)
    input = history[size_t(histPos)];
    resetTab();                   // recall replaces the line: cycle is stale
}

void Console::histNext() {
    if (histPos < 0) return;      // not browsing
    ++histPos;
    if (histPos >= int(history.size())) {   // past newest: restore the draft
        input = histStash;
        histStash.clear();
        histPos = -1;
    } else {
        input = history[size_t(histPos)];
    }
    resetTab();                   // recall replaces the line: cycle is stale
}

void Console::loadHistory(const char* path) {
    std::FILE* f = std::fopen(path, "r");
    if (!f) return;               // silently absent, like tune.cfg
    char buf[256];
    while (std::fgets(buf, sizeof buf, f)) {
        // Same filter as onChar (printable ASCII only) — it also strips the
        // \r\n, and a hand-edited over-long line degrades to clamped pieces
        // instead of crashing anything.
        std::string s;
        for (char* p = buf; *p; ++p)
            if (*p >= 0x20 && *p <= 0x7E) s.push_back(*p);
        if (s.empty()) continue;
        if (s.size() > size_t(MAX_INPUT)) s.resize(MAX_INPUT);
        history.push_back(s);
    }
    std::fclose(f);
    if (history.size() > size_t(HIST_MAX))   // hand-edited file may exceed
        history.erase(history.begin(), history.end() - HIST_MAX);
}

void Console::saveHistory(const char* path) const {
    // Plain rewrite of a tiny convenience file (tune.cfg-style, not the
    // tmp+rename dance map.dat does): worst crash case loses recall, never
    // game data. It IS rewritten on every submit, precisely so the command
    // that crashed the game is still there to recall after relaunch.
    std::FILE* f = std::fopen(path, "w");
    if (!f) return;
    for (const auto& h : history) std::fprintf(f, "%s\n", h.c_str());
    // fclose flushes the stdio buffer — a full disk fails HERE, not in the
    // buffered fprintfs above (same lesson as MapExplored::save).
    if (std::fclose(f) != 0) std::remove(path);   // don't leave a torn file
}
