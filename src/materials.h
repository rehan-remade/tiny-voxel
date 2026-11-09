#pragma once
#include <cstdint>
#include <glm/glm.hpp>

// Physical state of a material, for the physics automata (Stage 0 foundation).
enum MatState : uint8_t {
    MS_SOLID  = 0,   // holds its shape; subject to structural collapse (later)
    MS_POWDER = 1,   // granular: falls + slumps toward its angle of repose
    MS_LIQUID = 2,   // flows (the water sim)
};

// Per-material properties. Single source of truth -- shaders, gameplay,
// particles, and the physics sims all read from this table. Add a new material
// by adding a row. The physics fields carry sensible DEFAULTS (water-relative
// density 1, solid, full strength), so existing rows that omit them stay valid.
struct MatProps {
    glm::vec3 color;
    float     digTime;     // seconds per swing to dig this material
    // Surface response (PR10). Drives the sun specular highlight; smoothness/
    // metallic are otherwise RESERVED (the ray-traced reflection consumer was
    // removed with the first-person view). Defaults (0) = matte, no glow.
    float     smoothness;  // 0 rough/matte .. 1 mirror-smooth (specular sharpness + strength)
    float     metallic;    // 0 dielectric (white highlight) .. 1 metal (albedo-tinted highlight)
    float     emissive;    // 0 .. N self-illumination added on top of lighting
    // -- Physics (Stage 0). CPU-only; NOT packed into the shader material UBO. --
    float     density       = 1.0f;     // relative mass, water = 1.0 (sink/swap, collapse load)
    uint8_t   state         = MS_SOLID; // MatState
    float     angleOfRepose = 0.0f;     // degrees; powders slump no steeper (0 = grid-natural ~45)
    float     strength      = 1.0f;     // cohesion: resistance to collapse (Stage 3+)
};

extern const MatProps g_materials[256];

inline const MatProps& matProps(uint8_t m) { return g_materials[m]; }
inline bool isPowder(uint8_t m) { return g_materials[m].state == MS_POWDER; }
inline bool isLiquid(uint8_t m) { return g_materials[m].state == MS_LIQUID; }
