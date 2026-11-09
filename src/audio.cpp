// Streamed ambience / music playback on a miniaudio engine.
#define NOMINMAX                // windows.h via miniaudio: keep std::min/max
#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_ENCODING          // we never WRITE audio files (decoding stays on
                                // for streamed ambience: WAV/MP3/FLAC)
#include "miniaudio.h"

#include "audio.h"

#include <cstdio>

namespace {

ma_engine g_engine;
bool      g_ok = false;

ma_sound g_amb{};            // streamed looping ambience / music bed
bool     g_ambReady = false;

}  // namespace

namespace Audio {

bool init() {
    if (ma_engine_init(nullptr, &g_engine) != MA_SUCCESS) {
        std::fprintf(stderr, "[audio] engine init failed; running silent\n");
        return false;
    }
    g_ok = true;
    return true;
}

void startAmbience(const char* path, float volume) {
    if (!g_ok) return;
    if (g_ambReady) { ma_sound_uninit(&g_amb); g_ambReady = false; }
    // STREAM from disk (don't decode the whole track into RAM) and skip
    // spatialization (it's a 2D bed, not a positioned source).
    ma_uint32 flags = MA_SOUND_FLAG_STREAM | MA_SOUND_FLAG_NO_SPATIALIZATION;
    if (ma_sound_init_from_file(&g_engine, path, flags, nullptr, nullptr, &g_amb)
        != MA_SUCCESS) {
        std::fprintf(stderr, "[audio] ambience '%s' not loaded "
                     "(missing/unsupported) — staying silent\n", path);
        return;
    }
    ma_sound_set_looping(&g_amb, MA_TRUE);
    ma_sound_set_volume(&g_amb, volume);
    ma_sound_start(&g_amb);
    g_ambReady = true;
    std::printf("[audio] ambience playing: %s\n", path);
}

void setAmbienceVolume(float v) {
    if (g_ambReady) ma_sound_set_volume(&g_amb, v);
}

void stopAmbience() {
    if (g_ambReady) { ma_sound_uninit(&g_amb); g_ambReady = false; }
}

bool ambiencePlaying() { return g_ambReady; }

void shutdown() {
    if (!g_ok) return;
    if (g_ambReady) { ma_sound_uninit(&g_amb); g_ambReady = false; }
    ma_engine_uninit(&g_engine);
    g_ok = false;
}

}  // namespace Audio
