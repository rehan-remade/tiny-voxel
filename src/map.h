#pragma once
// The explored-world map: fog-of-war exploration tracking, its
// persistence sidecar (map.dat), and the orbit-camera state for the diorama
// view (the game's one and only view). Rendering lives in render.cpp (Renderer::renderMap), which raymarches
// the LIVE resident window through the shared g-buffer/lighting shaders (the
// LOD ring cascade carries the diorama beyond the window) and gates every
// voxel against the masks built here.
//
// Exploration model (Terraria semantics, translated to 3D):
//  - SURFACE: walking in the open reveals a disc of chunk columns around the
//    player (Minecraft map fill). Revealed columns render their true surface
//    skin (top MAP_SKIN voxels + everything above it: trees, flora, water).
//  - UNDERGROUND: a small sphere around the player marks visited 16-voxel
//    Y slices per chunk column. Below the surface skin, voxels in unvisited
//    slices render as flat MAT_MAP_UNKNOWN mass — unvisited tunnels stay
//    FILLED, and the map's slice plane carves down to the ones you walked.
//  - Chunk columns with no exploration at all are holes in the diorama.
#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include <glm/glm.hpp>
#include "common.h"

// GPU-ready explored bits: a MASK_CH^2-chunk square window into the unbounded
// exploration record, centred on the map's pan target (NOT the resident
// window — the map renders far beyond it via the LOD rings). Indexed by
// GLOBAL chunk coords minus the mask origin; out-of-mask reads as unexplored.
// Rebuilt from MapExplored whenever the marks or the mask origin change,
// uploaded by the renderer as two integer textures. MASK_CH must match
// MAP_MASK_CH in gbuffer.frag / lighting.frag.
struct MapMasks {
    // 384 chunks = 24576 voxels: covers the full-screen world slab
    // (player +- MAP_VIEW_R = 12288, see the framing block in app.cpp).
    static constexpr int MASK_CH    = 384;
    static constexpr int YSLICE_U32 = (GRID_Y / 16) / 32;   // 8 words / column
    // bit0 = surface seen, bit1 = any exploration.
    // ymask layout for glTexImage3D(MASK_CH x, 8 words, MASK_CH z):
    // index = x + MASK_CH * (word + YSLICE_U32 * z).
    // Heap-backed: ~1.2 MB, far too big for GameState's stack slot.
    std::vector<uint8_t>  flags;
    std::vector<uint32_t> ymask;
    // Prefiltered fog-of-war brightness: bit0 box-blurred over 5x5 chunks
    // and stored normalized (R8). ONE hardware-bilinear tap of this field
    // replaces the shader's old 100-texelFetch ramp on every map pixel.
    std::vector<uint8_t>  soft;
    MapMasks()
        : flags(size_t(MASK_CH) * MASK_CH, 0),
          ymask(size_t(MASK_CH) * MASK_CH * YSLICE_U32, 0),
          soft(size_t(MASK_CH) * MASK_CH, 0) {}
};

struct MapExplored {
    static constexpr int YSLICE_BITS = 4;                     // 16-voxel Y slices
    static constexpr int YSLICE_VOX  = 1 << YSLICE_BITS;
    static constexpr int YSLICES     = GRID_Y / YSLICE_VOX;   // 256
    static constexpr int YSLICE_U32  = YSLICES / 32;          // 8
    static_assert(YSLICE_U32 == MapMasks::YSLICE_U32, "mask layouts must agree");

    struct Entry {
        uint8_t  surfaceSeen = 0;
        uint32_t ymask[YSLICE_U32] = {};
    };
    // Keyed by chunkKey(ccx, ccz) in unbounded WORLD chunk coords — exploration
    // outlives the resident window (which destroys voxel data on scroll).
    std::unordered_map<int64_t, Entry> chunks;

    uint64_t version = 0;            // bumps on any new mark (mask rebuild key)
    bool     dirtySinceSave = false;
    int      minSliceSeen = YSLICES; // lowest visited slice -> diorama bottom

    // Marking driver, called once per frame from the game loop (self-throttled).
    // `surfH` is the local terrain height at the player column (toroidal
    // heightMap sample); below CAVE_DEPTH under it counts as underground.
    void update(const glm::vec3& eye, float surfH, float dt);

    // Assemble the GPU masks for a MASK_CH^2-chunk square starting at
    // maskOriginChunk (world chunk coords). O(explored chunks) + a clear.
    void buildMasks(glm::ivec2 maskOriginChunk, MapMasks& out) const;

    // Versioned sidecar (map.dat). Seed-keyed: another world's exploration is
    // ignored (fresh map), never a crash. Deliberately NOT GEN_VERSION-gated —
    // exploration is a location record, and heights/colors re-derive live.
    bool save(const std::string& path, uint64_t worldSeed);
    bool load(const std::string& path, uint64_t worldSeed);

private:
    float markAcc = 1e9f;   // large so the first update() marks immediately
};

// Orbit-camera + control state for the open map. Pure state; the app feeds it
// input from the GLFW callbacks and clamps the pan against the ring-covered
// world around the player (see the framing block in app.cpp).
struct MapView {
    // Pan fence: ring coverage around the player (ring2 spans +-8192), so a
    // look-around offset never orbits the camera off the rendered world.
    static constexpr float PAN_R = 5632.0f;

    bool  xray  = false;         // Tab: cave X-ray (ghost surface + glow veins)
    float animT = 0.0f;          // eased 0..1 settle-in after opening
    float yaw   = 0.85f;         // orbit azimuth (radians)
    float pitch = 0.85f;         // orbit elevation (radians above horizon)
    float dist  = 760.0f;        // orbit distance (voxels)
    glm::vec3 target{0.0f};      // orbit target (world voxels)
    float sliceY = float(GRID_Y);// horizontal cut plane (Q/E)

    // Follow-camera state: the orbit anchor tracks the walking player
    // (smoothed). The camera is CHARACTER-LOCKED unless freeCam is on: only
    // there does right-drag pan accumulate a look-around offset; leaving
    // free-cam eases the offset home. target = followPos + offset, composed
    // once per frame in follow().
    bool      freeCam = false;   // HUD toggle: right button pans, not moves
    glm::vec3 followPos{0.0f};   // smoothed player position (world voxels)
    glm::vec2 panOffset{0.0f};   // look-around offset (XZ, world voxels)

    // Recentre on the player. Underground opens straight into the X-ray so
    // the cave you're standing in is immediately visible.
    void openAt(const glm::vec3& playerPos, bool underground) {
        target    = playerPos;
        followPos = playerPos;
        panOffset = glm::vec2(0.0f);
        xray      = underground;
        sliceY    = float(GRID_Y);
        animT     = 0.0f;
    }
    void drag(float dx, float dy) {
        yaw   += dx * 0.008f;
        pitch  = glm::clamp(pitch + dy * 0.006f, 0.12f, 1.50f);
    }
    // Grab-the-world pan (primary drag): needs the window height in pixels to
    // convert the cursor delta into world units at the orbit distance.
    void pan(float dx, float dy, float viewportH);
    // Per-frame follow: smooth followPos toward the player, ease the pan
    // offset home once the grab has been idle, compose target. `panHeld` is
    // the live pan button so an active grab never eases out from under the
    // cursor.
    void follow(const glm::vec3& playerPos, bool panHeld, float dt);
    void zoom(float steps) {
        dist = glm::clamp(dist * std::pow(0.88f, steps), 90.0f, 2800.0f);
    }
    // The distance the camera ACTUALLY sits at: `dist` plus the settle-in
    // dolly while animT eases. pan() must use this too, or grabs made during
    // the open animation slip out from under the cursor.
    float viewDist() const {
        float ease = 1.0f - (1.0f - animT) * (1.0f - animT);   // ease-out
        return dist * (1.0f + 0.35f * (1.0f - ease));
    }
    // Orbit camera; the settle-in eases the distance so opening has motion.
    void camera(glm::vec3& pos, glm::vec3& fwd) const;
};
