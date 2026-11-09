"""Exact voxel->bone assignment for Corgi.vox (mirrors corgi_build.py).
Bucket-1/2 asset, user-cleaned model: 9x18x14 cells, 5 cm per cell
(rig scale S = 0.05). Passed to bake_parts.py via --rules so baked parts
match the authored rigid weights. y == 17 is the protruding stub tail."""


def bone_for_voxel(x, y, z):
    if z >= 12 or (z == 11 and (x <= 1 or x >= 6)):    # upright ears
        return "earL" if x >= 5 else "earR"            # +X = animal's left
    if (z >= 8 and y <= 8) or (z == 7 and y <= 7):     # oversized head + chin
        return "head"
    if z == 0:                                         # paws (legs are 2 cells)
        side = "L" if x >= 5 else "R"
        return f"F{side}.lower" if y <= 7 else f"H{side}.lower"
    if z == 1 and (x <= 1 or x >= 6):                  # leg shanks at the body edge
        side = "L" if x >= 5 else "R"
        return f"F{side}.upper" if y <= 7 else f"H{side}.upper"
    if y >= 17:                                        # stub tail nub
        return "tail"
    return "chest" if y <= 9 else "hips"
