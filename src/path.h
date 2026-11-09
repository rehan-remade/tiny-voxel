#pragma once
// Click-to-move pathfinding for the diorama: A* over walkable ground columns
// of the resident voxel window. A column is walkable when it has ground to
// stand on AND standing headroom above it (P_HEIGHT), so tree trunks and
// too-low canopies read as obstacles instead of things to wedge against.
// The result is line-of-sight smoothed, so open ground costs few waypoints.
#include <vector>
#include <glm/glm.hpp>

struct World;

// Route from `from` (feet) to `to` (feet). Fills `out` with waypoint feet
// positions (voxel-center XZ), excluding the start, ending at the closest
// REACHABLE point to `to` (the destination itself when it can be reached).
// Returns false when no first step exists (fully boxed in / target column
// unreachable and no partial progress possible).
bool findPath(const World& w, glm::vec3 from, glm::vec3 to,
              std::vector<glm::vec3>& out);
