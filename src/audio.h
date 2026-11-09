#pragma once
// Minimal audio playback (miniaudio backend): a streamed, looping ambience /
// music bed (/music). No one-shot sound effects are wired up right now.
namespace Audio {

bool init();        // safe to call once at startup; failure = silent game
void shutdown();

// Streamed, looping ambience / music bed. Safe to call with a missing or
// unsupported file — it logs and stays silent (harmless to wire up before any
// asset exists). Calling again swaps the track. Decodes WAV / MP3 / FLAC
// (Suno exports MP3, so that works out of the box).
void startAmbience(const char* path, float volume = 0.6f);
void setAmbienceVolume(float v);
void stopAmbience();
bool ambiencePlaying();   // true once a track is loaded + playing

}  // namespace Audio
