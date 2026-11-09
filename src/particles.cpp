#include "particles.h"
#include "materials.h"

#include <algorithm>

void ParticleSystem::spawnDust(glm::vec3 pos, uint8_t mat, int count) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    glm::vec3 col = matProps(mat).color * (0.55f + 0.20f * u01(rng));
    for (int i = 0; i < count && (int)particles.size() < MAX; ++i) {
        Particle p;
        p.pos = pos + glm::vec3(u(rng), u(rng), u(rng)) * 1.5f;
        p.vel = glm::vec3(u(rng) * 30.0f,
                          20.0f + u01(rng) * 40.0f,
                          u(rng) * 30.0f);
        p.color = col;
        p.maxLife = 0.6f + u01(rng) * 0.5f;
        p.life = p.maxLife;
        particles.push_back(p);
    }
}

void ParticleSystem::spawnAmbient(glm::vec3 around, int count, bool firefly) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    glm::vec3 base = firefly
        ? glm::vec3(1.00f, 0.85f, 0.42f)   // firefly yellow
        : glm::vec3(0.85f, 0.88f, 0.95f);  // dust mote pale
    for (int i = 0; i < count && (int)particles.size() < MAX; ++i) {
        Particle p;
        p.pos = around;
        p.pos.x += u(rng) * 220.0f;
        p.pos.z += u(rng) * 220.0f;
        p.pos.y += 24.0f + u01(rng) * 100.0f;
        p.vel = glm::vec3(u(rng) * 4.0f, u(rng) * 1.5f, u(rng) * 4.0f);
        p.color = base * (0.65f + u01(rng) * 0.35f);
        p.maxLife = 14.0f + u01(rng) * 10.0f;
        p.life = p.maxLife;
        p.ambient = true;
        particles.push_back(p);
    }
}

// A warm ember that floats UP off a flame (ambient = no gravity) and fades
// fast. Short-lived, so it's excluded from ambientCount (which budgets the
// long-lived dust/firefly motes) by the maxLife test below.
void ParticleSystem::spawnEmber(glm::vec3 pos) {
    if ((int)particles.size() >= MAX) return;
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    Particle p;
    p.pos     = pos + glm::vec3(u(rng), u01(rng) * 2.0f, u(rng)) * 2.0f;
    p.vel     = glm::vec3(u(rng) * 5.0f, 9.0f + u01(rng) * 12.0f, u(rng) * 5.0f);
    p.color   = glm::vec3(1.0f, 0.45f + 0.30f * u01(rng), 0.13f);  // orange -> yellow
    p.maxLife = 0.7f + u01(rng) * 0.9f;
    p.life    = p.maxLife;
    p.ambient = true;     // hot air rises: no gravity, gentle upward drift
    particles.push_back(p);
}

int ParticleSystem::ambientCount() const {
    int n = 0;
    // Only the long-lived dust/firefly motes; transient embers (short maxLife)
    // are ambient too but must not crowd out the ambient-mote budget.
    for (const auto& p : particles) if (p.ambient && p.maxLife > 5.0f) ++n;
    return n;
}

void ParticleSystem::step(float dt) {
    constexpr float G = 240.0f;
    for (auto& p : particles) {
        if (!p.ambient) p.vel.y -= G * dt;
        p.pos += p.vel * dt;
        p.life -= dt;
    }
    particles.erase(
        std::remove_if(particles.begin(), particles.end(),
                       [](const Particle& p){ return p.life <= 0.0f; }),
        particles.end());
}
