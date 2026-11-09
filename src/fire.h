#pragma once
#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>
#include <glm/glm.hpp>
#include "common.h"

struct World;
struct EditDelta;

// Runtime fire automaton. Unlike water (which MOVES voxels and conserves mass,
// so its active set can creep forever), fire is a strict ONE-WAY state machine:
//
//     flammable fuel --ignite--> MAT_FIRE --life--> MAT_EMBER --life--> CHAR / air
//
// A voxel ignites exactly once, every burning voxel has a finite life counting
// to extinction, and MAT_EMBER does NOT spread. So the burning set is bounded
// by the reachable fuel and DRAINS every tick — no FPS cliff like a static
// water fill. Entirely transient: never saved, never marks dirtyChunks.
struct FireSim {
    // packed world voxel coord -> packed life:
    //   bit 7    = fuel was wood (leaves a charred residue instead of air)
    //   bits 0-6 = ticks remaining in the CURRENT phase (phase = the voxel's
    //              material, read live via getVoxel)
    std::unordered_map<int64_t, uint8_t> life;
    // voxel key -> packed XZ of the CAST that started this fire. Spread is
    // bounded to a horizontal radius of that point, so grass/forest fire is a
    // contained blast (impressive) instead of a world-spanning sheet (endless).
    // Inherited on ignition, dropped when the voxel burns out.
    std::unordered_map<int64_t, int64_t> origin;
    std::mt19937 rng{0xF1AE5EEDu};

    static int64_t packXZ(int x, int z) {
        return int64_t(uint32_t(x)) | (int64_t(uint32_t(z)) << 32);
    }

    // 26 bits X, 26 bits Z, 12 bits Y — unique for any two in-window voxels
    // (the resident window is only GRID_XZ wide, far inside this range).
    static int64_t key(int x, int y, int z) {
        return ( int64_t(x) & 0x3FFFFFF)
             | ((int64_t(z) & 0x3FFFFFF) << 26)
             | ((int64_t(y) & 0xFFF)     << 52);
    }
};

bool    isFlammable(uint8_t m);          // can this material catch fire?
uint8_t fuelLife(uint8_t m);             // ticks of MAT_FIRE before it decays

// Ignite a single voxel (the cast seed). Returns false if it isn't flammable.
bool igniteVoxel(World& w, FireSim& f, int x, int y, int z, EditDelta& d);

// One automaton tick: spread active flame to flammable neighbours, then decay
// every burning voxel one step along the state machine. All voxel writes go
// through `d` for a single batched uploadDelta by the caller.
// `leafGone` (nullable): positions whose burned-out residue was AIR and whose
// fuel was foliage — the caller seeds forestry's leaf-support decay with them
// (a burn that eats a band through a crown must not strand the rest floating).
void simulateFire(World& w, FireSim& f, EditDelta& d,
                  std::vector<glm::ivec3>* leafGone = nullptr);

// Extinguish everything immediately (panic / cleanup): turn all fire/ember to
// air and clear the active set. Writes through `d`. `removed` (nullable):
// every doused position — each one just became AIR, so the caller must seed
// forestry's leaf-support decay with them exactly like simulateFire's
// burn-out path, or a crown whose support was burning when /douse ran is
// stranded floating.
void douseAllFire(World& w, FireSim& f, EditDelta& d,
                  std::vector<glm::ivec3>* removed = nullptr);
