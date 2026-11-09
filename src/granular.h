#pragma once
#include <random>

struct World;
struct EditDelta;

// Granular (falling-sand) automaton — Stage 1 physics. Powder voxels
// (MatState MS_POWDER) fall and pile. Processed BOTTOM-UP so a column drops as
// a cohesive unit each tick (each grain falls into the gap the one below just
// vacated) — no per-grain velocity, because independent accelerating grains
// disperse a sand mass (the bottom races ahead of the top). A bounded ACTIVE
// SET over World::fallingBricks, mirroring the water sim: snapshot -> step ->
// the caller does ONE batched uploadDelta. Settled piles drop out of the set so
// it drains to zero and sleeps. Transient (no persistence yet).
void simulateGranular(World& w, EditDelta& d, std::mt19937& rng);
