#pragma once
#include <random>
struct World;
struct EditDelta;

// Single tick of water cellular automaton. Iterates over all wet bricks and
// for each water cell, attempts (1) fall down, (2) sideways spread to a lower
// neighbour, (3) sideways spread on a flat plane with limited probability.
// Water cells move (not duplicate), so total water mass is conserved.
void simulateWater(World& w, EditDelta& d, std::mt19937& rng);
