"""Exact voxel->bone assignment for Stag.vox (mirrors stag_build.py).
Passed to bake_flipbook.py via --rules so baked parts match the authored
rigid weights instead of the vertex-vote approximation."""


def bone_for_voxel(x, y, z):
    if z <= 6 and y <= 6 and (x <= 3 or x >= 6):       # front legs
        side = "L" if x >= 5 else "R"                   # +X = animal's left
        return f"F{side}.lower" if z <= 3 else f"F{side}.upper"
    if z <= 6 and y >= 13 and (x <= 3 or x >= 6):       # hind legs
        side = "L" if x >= 5 else "R"
        return f"H{side}.lower" if z <= 3 else f"H{side}.upper"
    if z >= 14 or (z == 13 and y <= 4):                 # skull, muzzle, ears, antlers
        return "head"
    if z in (12, 13) or (z == 11 and y <= 4):
        return "neck"
    return "chest" if y <= 8 else "hips"
