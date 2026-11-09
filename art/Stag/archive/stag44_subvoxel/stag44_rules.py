"""Voxel->bone assignment for the 44-tall Stag44.vox (25x33x44, S=0.05,
cellScale 0.5). Mirrors stag44_build.py; derived from fresh slice analysis
(legs z<=12 in four clusters, belly shell starts z13; knee at cell 8 =
KNEE_Z 0.40 m; head cut z>=28 with jaw dip z26-27 y<=8; neck z24-27 with
throat front z22-23 y<=8; chest/hips split y<=16)."""


def bone_for_voxel(x, y, z):
    if z <= 12 and (x <= 10 or x >= 13):               # legs (four clusters)
        side = "L" if x >= 12 else "R"                  # +X = animal's left
        part = "lower" if z <= 7 else "upper"           # knee joint at cell 8
        return f"{'F' if y <= 18 else 'H'}{side}.{part}"
    if z >= 28 or (z >= 26 and y <= 8):                 # skull, muzzle, antlers
        return "head"
    if z >= 24 or (z >= 22 and y <= 8):                 # neck + throat front
        return "neck"
    return "chest" if y <= 16 else "hips"
