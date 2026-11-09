"""Exact voxel->bone assignment for Player.vox (mirrors player_build.py).
Passed to bake_parts.py via --rules. The player model is 20x9x32 vox
(x width, y depth, z height) at 1.8 m normalized scale — bake with
--scale 0.05625 (1.8/32), not the 10 cm wildlife rule."""


def bone_for_voxel(x, y, z):
    if z >= 23:
        return "head"
    if (x <= 4 or x >= 15) and z >= 12:                # arms hang outboard
        side = "L" if x >= 10 else "R"
        return f"arm{side}.upper" if z >= 15 else f"arm{side}.lower"
    if z >= 17:
        return "chest"
    if z >= 13:
        return "spine"
    if z >= 11:
        return "hips"
    side = "L" if x >= 10 else "R"                     # legs, split at x 9/10
    if z <= 1:
        return f"foot{side}"
    return f"leg{side}.lower" if z <= 5 else f"leg{side}.upper"
