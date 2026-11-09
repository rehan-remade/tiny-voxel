"""Authored zero-skate ground speed of a .cvx2 Walk clip, in WORLD voxels/s.
Planted-hoof method: bottom-center of each *.lower part per frame; during
stance (y within 0.5 cell of that hoof's min) the hoof sweeps backward in
creature-local z; speed = median(-dz/dt) * cellScale. Compare to the .def's
walk_ref_speed (world voxels/s, cellScale-independent by design)."""
import struct, sys
import numpy as np

def load(path):
    f = open(path, "rb")
    assert f.read(4) == b"CVX2"
    ver, = struct.unpack("<I", f.read(4))
    nparts, nclips, total = struct.unpack("<III", f.read(12))
    cellscale, = struct.unpack("<f", f.read(4))
    f.read(1024)
    parts = []
    for _ in range(nparts):
        name = f.read(16).rstrip(b"\0").decode()
        parent, = struct.unpack("<i", f.read(4))
        origin = struct.unpack("<3f", f.read(12))
        pivot = struct.unpack("<3f", f.read(12))
        dims = struct.unpack("<3I", f.read(12))
        wb, = struct.unpack("<I", f.read(4))
        parts.append((name, origin, dims))
    clips = {}
    for _ in range(nclips):
        name = f.read(16).rstrip(b"\0").decode()
        first, count, fps, flags = struct.unpack("<IIfI", f.read(16))
        clips[name] = (first, count, fps)
    tracks = np.frombuffer(f.read(total * nparts * 7 * 4), dtype=np.float32)
    tracks = tracks.reshape(total, nparts, 7)
    return cellscale, parts, clips, tracks

def rot(q, v):  # q = (x,y,z,w)
    x, y, z, w = q
    u = np.array([x, y, z]); v = np.asarray(v, dtype=np.float64)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)

for path in sys.argv[1:]:
    cs, parts, clips, tracks = load(path)
    first, count, fps = clips["Walk"]
    speeds = []
    for pi, (name, origin, dims) in enumerate(parts):
        if not name.endswith(".lower"):
            continue
        rest = np.array([origin[0] + dims[0] * 0.5, origin[1], origin[2] + dims[2] * 0.5])
        pos = np.array([rot(tracks[first + fr, pi, 0:4], rest) + tracks[first + fr, pi, 4:7]
                        for fr in range(count)])
        ymin = pos[:, 1].min()
        planted = pos[:, 1] < ymin + 0.5
        dz = []
        for fr in range(count):
            nx = (fr + 1) % count
            if planted[fr] and planted[nx]:
                dz.append(pos[nx, 2] - pos[fr, 2])
        v = -np.median(dz) * fps * cs      # cells/frame -> world voxels/s
        speeds.append(v)
        print(f"  {name:9s} stance sweep {v:5.2f} vox/s ({sum(planted)}/{count} planted)")
    print(f"{path}: cellScale {cs}  Walk authored ground speed = "
          f"{np.median(speeds):.2f} world voxels/s (def walk_ref_speed = 3.2)")
