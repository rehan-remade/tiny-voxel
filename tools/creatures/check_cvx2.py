"""Check a .cvx2: per-clip paw-bottom heights (world voxels, engine math).
Replicates composeModelParts: world = (q * (gridOrigin + corner) + p) * cellScale.
Paw plane should be ~0 on stance frames of every clip."""
import struct, sys
import numpy as np

def quat_rot(q, v):  # q = (x,y,z,w)
    x, y, z, w = q
    u = np.array([x, y, z]); s = w
    return 2*np.dot(u, v)*u + (s*s - np.dot(u, u))*v + 2*s*np.cross(u, v)

path = sys.argv[1]
f = open(path, 'rb')
assert f.read(4) == b"CVX2"
ver, = struct.unpack("<I", f.read(4))
nparts, nclips, ntracks = struct.unpack("<III", f.read(12))
cellScale, = struct.unpack("<f", f.read(4))
f.read(256*4)  # palette
parts = []
for _ in range(nparts):
    name = f.read(16).rstrip(b"\0").decode()
    parent, = struct.unpack("<i", f.read(4))
    origin = struct.unpack("<3f", f.read(12))
    pivot = struct.unpack("<3f", f.read(12))
    dims = struct.unpack("<3I", f.read(12))
    wb, = struct.unpack("<I", f.read(4))
    parts.append((name, np.array(origin), np.array(dims)))
clips = []
for _ in range(nclips):
    cname = f.read(16).rstrip(b"\0").decode()
    first, count, cfps, loop = struct.unpack("<IIfI", f.read(16))
    clips.append((cname, first, count))
tracks = np.frombuffer(f.read(ntracks*nparts*7*4), dtype=np.float32).reshape(ntracks, nparts, 7)

print(f"{path}: cellScale={cellScale} parts={nparts} clips={[c[0] for c in clips]}")
lowers = [i for i, p in enumerate(parts) if p[0].endswith(".lower")]
for cname, first, count in clips:
    mins = []
    for fr in range(first, first+count):
        row = tracks[fr]
        frame_min = 9e9
        for i in lowers:
            name, origin, dims = parts[i]
            q = row[i, :4]; p = row[i, 4:7]
            for cx in (0, dims[0]):
                for cz in (0, dims[2]):
                    corner = origin + np.array([cx, 0.0, cz])
                    y = (quat_rot(q, corner) + p)[1] * cellScale
                    frame_min = min(frame_min, y)
        mins.append(frame_min)
    mins = np.array(mins)
    print(f"  {cname:8s} paw-plane min-y over {count:3d} frames: "
          f"min={mins.min():+.3f} mean={mins.mean():+.3f} max={mins.max():+.3f} (world voxels)")
# whole-model min at frame 0 of first clip
row = tracks[clips[0][1]]
lo = 9e9
for i, (name, origin, dims) in enumerate(parts):
    q = row[i, :4]; p = row[i, 4:7]
    for cx in (0, dims[0]):
        for cy in (0, dims[1]):
            for cz in (0, dims[2]):
                y = (quat_rot(q, origin + np.array([cx, cy, cz], float)) + p)[1] * cellScale
                lo = min(lo, y)
print(f"  whole-model min-y (frame 0 of {clips[0][0]}): {lo:+.3f}")
