#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <random>
#include <cstdint>

struct Particle {
    glm::vec3 pos, vel, color;
    float life, maxLife;
    bool  ambient = false;     // ambient particles drift, ignore gravity
};

struct ParticleSystem {
    static constexpr int MAX = 1024;
    std::vector<Particle> particles;
    std::mt19937 rng{0xC0FFEEu};

    void spawnDust(glm::vec3 pos, uint8_t mat, int count);
    void spawnAmbient(glm::vec3 around, int count, bool firefly);
    void spawnEmber(glm::vec3 pos);   // one warm rising ember from a flame
    int  ambientCount() const;
    void step(float dt);
};
