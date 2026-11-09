#pragma once
#include <glm/glm.hpp>
#include "common.h"   // canonical GRAVITY / GRAVITY_TERMINAL (one game gravity)

constexpr float P_HEIGHT    = 18.0f;
constexpr float P_RADIUS    = 2.0f;
constexpr float P_EYE       = 16.2f;
constexpr float P_STEP      = 3.0f;
constexpr float WALK_SPEED  = 48.0f;
constexpr float SPRINT_MULT = 1.3f;   // sprint is +30% over walk (Minecraft-like)
// Crouch (sneak): ~30% of walk speed (MC's 1.3/4.3) and a shorter hitbox
// (1.5/1.8 of standing) so you fit under low gaps; the eye lowers to match.
constexpr float CROUCH_MULT   = 0.30f;
constexpr float CROUCH_HEIGHT = P_HEIGHT * (1.5f / 1.8f);   // 15.0
constexpr float CROUCH_EYE    = P_EYE    * (1.5f / 1.8f);   // 13.5
constexpr float JUMP_VEL    = 130.0f;
// GRAVITY is now canonical in common.h (shared by player + sand + physics).

// Swim tuning. Buoyancy ≈ 0.18 of normal gravity (sinks slowly), Space adds
// upward thrust, Ctrl adds downward thrust. Strong drag keeps things calm.
constexpr float WATER_GRAVITY_MULT = 0.18f;
constexpr float SWIM_THRUST        = 280.0f;
constexpr float WATER_DRAG         = 3.5f;     // exp(-DRAG * dt) per tick
constexpr float SWIM_SPEED_MULT    = 0.55f;

struct Player {
    glm::vec3 pos{0};
    glm::vec3 vel{0};
    float yaw = -1.57f;     // facing (fwd = cos/sin yaw); drives the avatar
    bool onGround = false;
    bool fly = false;
    float eyeOff = P_EYE;   // current eye height above feet (lowers when crouching)

    glm::vec3 eye() const;  // head position (exploration reveal, water checks)
};

// Voxels the player walks through. Grass blades and flowers are decorative-
// only; without this the player gets snagged on every tuft and petal.
// Water is passable too -- swim physics in app.cpp handle the in-water state.
// (Shared here so the pathfinder and the collision box agree on what blocks.)
inline bool isBlockingForPlayer(uint8_t v) {
    return v != MAT_AIR
        && v != MAT_WATER
        && v != MAT_GRASS_BLADE
        && v != MAT_FLOWER_STEM
        && v != MAT_FLOWER_RED
        && v != MAT_FLOWER_YELLOW
        && v != MAT_FLOWER_BLUE
        && v != MAT_FLOWER_WHITE
        // Mushrooms are decorative -- walk through them like grass/flowers so
        // they don't block the descent.
        && v != MAT_MUSHROOM_STEM
        && v != MAT_MUSHROOM_CAP
        && v != MAT_MUSHROOM_RIM
        && v != MAT_MUSHROOM_SPOT
        && v != MAT_MUSHROOM_CAP2
        && v != MAT_CAVE_MOSS
        && v != MAT_VINE
        && v != MAT_VINE_DEAD
        // Flames are passable -- you can walk through fire (it'll be a hazard
        // later, not a wall). Char/ash stay solid (real burnt material).
        && v != MAT_FIRE
        && v != MAT_EMBER;
}

struct World;
bool overlapsSolid   (const World& w, glm::vec3 feet, float height = P_HEIGHT);
bool isOnGround      (const World& w, glm::vec3 feet);
bool isInWater       (const World& w, glm::vec3 feet);
