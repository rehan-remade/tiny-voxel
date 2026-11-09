"""Bake an animated creature GLB into a .cvox voxel-flipbook for the engine.

The GLB must be a rigid-weighted voxel-lattice mesh (the kind stag_build.py
authors: every vertex sits on the 10 cm voxel lattice and is weighted 100 %
to exactly one bone). For each animation frame every bone's rigid transform
is applied by INVERSE nearest-neighbour resampling of that bone's rest
voxels, so rotated limbs never develop resampling holes.

Run headless:
  blender -b --factory-startup --python bake_flipbook.py -- \
      --vox Stag.vox --glb Stag_animated.glb --out stag.cvox \
      [--rules stag_rules.py] [--step 1] [--debug-vox-dir DIR]

--rules points at a python file exposing bone_for_voxel(x, y, z) -> bone
name (exact authoring rules). Without it, voxels vote by the dominant
vertex-group of the mesh vertices on their 8 lattice corners (good enough
for foreign assets; joints may differ by a voxel).

.cvox format (little-endian), engine-local axes x=lateral, y=up, z=forward:
  char[4]  magic "CVOX"
  u32      version = 1
  u32      gx, gy, gz          fixed frame-grid dims (cells, 1 cell = 1 world voxel)
  f32[3]   anchor              cell coords of the creature origin (feet centre)
  u32      clipCount
  u32      totalFrames
  u8[1024] palette             256 x RGBA (index 0 = empty)
  clip[clipCount] { char[16] name; u32 firstFrame; u32 frameCount; f32 fps; u32 flags(bit0=loop) }
  u8[totalFrames * gx*gy*gz]   palette indices, frame-major, idx = x + gx*(y + gy*z)
"""
import importlib.util
import os
import struct
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

bpy.ops.wm.read_factory_settings(use_empty=True)

argv = sys.argv[sys.argv.index("--") + 1:]


def arg(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


VOX_PATH = arg("--vox")
GLB_PATH = arg("--glb")
OUT_PATH = arg("--out")
RULES_PATH = arg("--rules")
STEP = int(arg("--step", "1"))
DEBUG_VOX_DIR = arg("--debug-vox-dir")

S = 0.1  # 1 voxel = 10 cm (project scale, VOXEL_CM in src/common.h)

# Clips that loop seamlessly (their last authored frame repeats the first,
# so the duplicate is dropped at bake time). Everything else is a one-shot.
LOOP_CLIPS = {"Idle", "Walk", "Gallop", "Graze", "Alert"}


# ---- vox load (same RIFF-ish parser as stag_build.py) ------------------------
def load_vox(path):
    buf = open(path, "rb").read()
    off, size, vox, pal = 8, None, [], None
    while off < len(buf):
        cid = buf[off:off + 4]
        content, children = struct.unpack_from("<ii", buf, off + 4)
        off += 12
        c = off
        if cid == b"SIZE":
            size = struct.unpack_from("<iii", buf, c)
        elif cid == b"XYZI":
            n, = struct.unpack_from("<i", buf, c)
            vox = [struct.unpack_from("<4B", buf, c + 4 + i * 4) for i in range(n)]
        elif cid == b"RGBA":
            pal = [struct.unpack_from("<4B", buf, c + i * 4) for i in range(256)]
        off = c + content
        if cid != b"MAIN":
            off += children
    return size, vox, pal


size, voxels, palette = load_vox(VOX_PATH)
print(f"vox: {size} voxels={len(voxels)}")

# ---- import glb ---------------------------------------------------------------
bpy.ops.import_scene.gltf(filepath=GLB_PATH)
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
mesh_obj = next(o for o in bpy.data.objects
                if o.type == "MESH" and o.vertex_groups)
mesh = mesh_obj.data
bone_names = [b.name for b in arm.data.bones]
print(f"glb: mesh={mesh_obj.name} verts={len(mesh.vertices)} bones={len(bone_names)}")

# ---- recover the voxel lattice from the rest mesh ------------------------------
# Mesh verts sit on a 10 cm lattice at (voxcorner - center_offset) * S, where
# the authoring centre offset may be half-integer (e.g. CY=8.5). Recover the
# per-axis FRACTIONAL part first, then align integer mins with the .vox mins.
verts_local = np.array([v.co[:] for v in mesh.vertices], dtype=np.float64)
lat_f = verts_local / S
frac = np.median(lat_f - np.floor(lat_f), axis=0)
frac[frac > 0.75] -= 1.0                              # wrap 0.99.. to -0.01
lat = np.rint(lat_f - frac).astype(np.int64)
resid = np.abs(lat_f - frac - lat).max()
if resid > 0.05:
    raise SystemExit(f"mesh verts are not on the voxel lattice (max residual {resid:.3f} cells)")
vox_arr = np.array([(x, y, z) for x, y, z, _ in voxels], dtype=np.int64)
col_arr = np.array([ci for _, _, _, ci in voxels], dtype=np.uint8)
corner_off = vox_arr.min(axis=0) - lat.min(axis=0)   # vert lattice -> vox corner coords
print(f"lattice residual {resid:.4f} cells, frac {frac.round(3)}, corner offset {corner_off}")

# ---- voxel -> bone -------------------------------------------------------------
if RULES_PATH:
    spec = importlib.util.spec_from_file_location("rules", RULES_PATH)
    rules = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rules)
    vox_bone = [rules.bone_for_voxel(x, y, z) for x, y, z in vox_arr]
    print(f"bone assignment: rules file {os.path.basename(RULES_PATH)}")
else:
    # Dominant vertex-group per vert, then majority vote over each voxel's
    # corner verts; unresolved (interior) voxels flood from assigned neighbours.
    vert_bone = {}
    gname = {g.index: g.name for g in mesh_obj.vertex_groups}
    for v in mesh.vertices:
        if v.groups:
            g = max(v.groups, key=lambda g: g.weight)
            vert_bone[tuple(lat[v.index] + corner_off)] = gname[g.group]
    vox_bone = []
    from collections import Counter, deque
    occ_set = {tuple(v) for v in vox_arr}
    for x, y, z in vox_arr:
        votes = Counter()
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    b = vert_bone.get((x + dx, y + dy, z + dz))
                    if b:
                        votes[b] += 1
        vox_bone.append(votes.most_common(1)[0][0] if votes else None)
    # flood-fill unassigned interiors
    idx_of = {tuple(v): i for i, v in enumerate(vox_arr)}
    todo = deque(i for i, b in enumerate(vox_bone) if b is None)
    while todo:
        i = todo.popleft()
        x, y, z = vox_arr[i]
        for d in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
            j = idx_of.get((x+d[0], y+d[1], z+d[2]))
            if j is not None and vox_bone[j]:
                vox_bone[i] = vox_bone[j]
                break
        else:
            todo.append(i)
    print("bone assignment: vertex-weight vote")

missing = {b for b in vox_bone if b not in bone_names}
if missing:
    raise SystemExit(f"voxels assigned to unknown bones: {missing}")

# Group voxels per bone. Parents claim first in authoring, so bake children
# (leg lowers etc.) LAST: at joint overlaps the extremity wins the cell.
by_bone = {}
for i, b in enumerate(vox_bone):
    by_bone.setdefault(b, []).append(i)
order = sorted(by_bone.keys(), key=lambda b: len(arm.data.bones[b].parent_recursive))
print("parts:", {b: len(ix) for b, ix in ((b, by_bone[b]) for b in order)})

# Rest voxel centres in ARMATURE space (undo any importer object transforms;
# mesh and rig share it, so relative space is the authoring space: Z-up).
mesh_to_arm = arm.matrix_world.inverted() @ mesh_obj.matrix_world
M = np.array(mesh_to_arm)
centers_mesh = (vox_arr + 0.5 - corner_off + frac) * S     # mesh-local
centers_arm = centers_mesh @ M[:3, :3].T + M[:3, 3]        # armature space

# Engine axes: authoring space is Z-up with the animal facing -Y.
# engine = (x_bl, z_bl, -y_bl)  ->  x lateral, y up, z forward. Det +1.
def to_engine(p):
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)

def from_engine(p):
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], axis=-1)

rest_mat = {b: np.array(arm.data.bones[b].matrix_local.inverted())
            for b in order}

clips = []   # (name, [deform-matrix-per-frame dict], fps, loop)
scene = bpy.context.scene
fps = scene.render.fps

for act in bpy.data.actions:
    name = act.name
    if not arm.animation_data:
        arm.animation_data_create()
    arm.animation_data.action = act
    if hasattr(arm.animation_data, "action_slot") and len(act.slots):
        arm.animation_data.action_slot = act.slots[0]   # Blender 5.x slotted actions
    f0, f1 = int(act.frame_range[0]), int(act.frame_range[1])
    loop = name in LOOP_CLIPS
    if loop and f1 > f0:
        f1 -= 1          # last frame duplicates the first in cyclic clips
    frames = []
    for f in range(f0, f1 + 1, STEP):
        scene.frame_set(f)
        bpy.context.view_layer.update()
        D = {}
        for b in order:
            pose = np.array(arm.pose.bones[b].matrix)
            D[b] = pose @ rest_mat[b]        # rest armature-space -> posed
        frames.append(D)
    clips.append((name, frames, fps / STEP, loop))
    print(f"clip {name}: {len(frames)} baked frames (src {f0}..{f1}, loop={loop})")

# ---- global grid bounds over all posed frames ----------------------------------
lo = np.array([1e9] * 3)
hi = np.array([-1e9] * 3)
for _, frames, _, _ in clips:
    for D in frames:
        for b in order:
            ix = by_bone[b]
            c = centers_arm[ix]
            p = c @ D[b][:3, :3].T + D[b][:3, 3]
            e = to_engine(p)
            lo = np.minimum(lo, e.min(axis=0))
            hi = np.maximum(hi, e.max(axis=0))
lo -= S * 0.5
hi += S * 0.5
gdim = np.ceil((hi - lo) / S).astype(int)
gx, gy, gz = int(gdim[0]), int(gdim[1]), int(gdim[2])
# anchor: engine origin (feet centre on the ground under the body) in cells.
# x/z centre of the grid footprint, y at world 0 (hoof contact height).
anchor = (np.array([ (0.0 - lo[0]), (0.0 - lo[1]), (0.0 - lo[2]) ]) / S)
print(f"grid {gx}x{gy}x{gz} cells, anchor {anchor.round(2)}, aabb {lo.round(2)}..{hi.round(2)}")

# ---- bake frames (inverse NN resample per rigid part) ---------------------------
# Per-part rest occupancy grid for O(1) inverse lookups.
part_grid = {}
for b in order:
    ix = by_bone[b]
    g = np.zeros(size, dtype=np.uint8)
    v = vox_arr[ix]
    g[v[:, 0], v[:, 1], v[:, 2]] = col_arr[ix]
    part_grid[b] = g

frame_blobs = []
counts = []
for name, frames, _, _ in clips:
    for D in frames:
        out = np.zeros((gx, gy, gz), dtype=np.uint8)
        for b in order:
            ix = by_bone[b]
            c = centers_arm[ix]
            Db = D[b]
            p = c @ Db[:3, :3].T + Db[:3, 3]
            e = to_engine(p)
            clo = np.maximum(np.floor((e.min(axis=0) - lo) / S).astype(int) - 1, 0)
            chi = np.minimum(np.ceil((e.max(axis=0) - lo) / S).astype(int) + 1,
                             [gx - 1, gy - 1, gz - 1])
            xs = np.arange(clo[0], chi[0] + 1)
            ys = np.arange(clo[1], chi[1] + 1)
            zs = np.arange(clo[2], chi[2] + 1)
            if not (len(xs) and len(ys) and len(zs)):
                continue
            gxs, gys, gzs = np.meshgrid(xs, ys, zs, indexing="ij")
            cell_centers = np.stack([gxs, gys, gzs], axis=-1).reshape(-1, 3)
            world = lo + (cell_centers + 0.5) * S
            pb = from_engine(world)                       # blender armature space
            inv = np.linalg.inv(Db)
            rest = pb @ inv[:3, :3].T + inv[:3, 3]        # rest armature space
            inv_m = np.linalg.inv(M)
            rest_mesh = rest @ inv_m[:3, :3].T + inv_m[:3, 3]
            vc = np.floor(rest_mesh / S - frac).astype(int) + corner_off
            ok = ((vc >= 0) & (vc < np.array(size))).all(axis=1)
            vals = np.zeros(len(vc), dtype=np.uint8)
            vals[ok] = part_grid[b][vc[ok, 0], vc[ok, 1], vc[ok, 2]]
            hit = vals > 0
            cc = cell_centers[hit]
            out[cc[:, 0], cc[:, 1], cc[:, 2]] = vals[hit]
        frame_blobs.append(out.transpose(2, 1, 0).tobytes())  # x-fastest
        counts.append(int((out > 0).sum()))

cmin, cmax = min(counts), max(counts)
print(f"baked {len(frame_blobs)} frames, voxel counts {cmin}..{cmax} (rest {len(voxels)})")
if cmin < len(voxels) * 0.7:
    print("WARNING: some frames lost >30% of voxels — check transforms")

# ---- optional debug .vox dumps ---------------------------------------------------
if DEBUG_VOX_DIR:
    os.makedirs(DEBUG_VOX_DIR, exist_ok=True)
    fi = 0
    for name, frames, _, _ in clips:
        mid = fi + len(frames) // 2
        blob = frame_blobs[mid]
        a = np.frombuffer(blob, dtype=np.uint8).reshape(gz, gy, gx).transpose(2, 1, 0)
        xyzi = b""
        n = 0
        for x, y, z in zip(*np.nonzero(a)):
            # engine (x,y,z) -> magica (x, z_depth=forward, y_up): keep it viewable
            xyzi += struct.pack("<4B", int(x), int(a.shape[2] - 1 - z) if False else int(z), int(y), int(a[x, y, z]))
            n += 1
        pal_bytes = b"".join(struct.pack("<4B", *palette[i]) for i in range(256))
        def chunk(cid, content, children=b""):
            return cid + struct.pack("<ii", len(content), len(children)) + content + children
        inner = chunk(b"SIZE", struct.pack("<iii", gx, gz, gy)) \
              + chunk(b"XYZI", struct.pack("<i", n) + xyzi) \
              + chunk(b"RGBA", pal_bytes)
        data = b"VOX " + struct.pack("<i", 150) + chunk(b"MAIN", b"", inner)
        path = os.path.join(DEBUG_VOX_DIR, f"debug_{name}_f{len(frames)//2}.vox")
        open(path, "wb").write(data)
        fi += len(frames)
    print(f"debug .vox dumps in {DEBUG_VOX_DIR}")

# ---- write .cvox ------------------------------------------------------------------
with open(OUT_PATH, "wb") as f:
    f.write(b"CVOX")
    f.write(struct.pack("<I", 1))
    f.write(struct.pack("<III", gx, gy, gz))
    f.write(struct.pack("<3f", *anchor))
    total = sum(len(fr) for _, fr, _, _ in clips)
    f.write(struct.pack("<II", len(clips), total))
    for i in range(256):
        f.write(struct.pack("<4B", *(palette[i] if palette else (200, 200, 200, 255))))
    first = 0
    for name, frames, cfps, loop in clips:
        nb = name.encode()[:16]
        f.write(nb + b"\0" * (16 - len(nb)))
        f.write(struct.pack("<IIfI", first, len(frames), cfps, 1 if loop else 0))
        first += len(frames)
    for blob in frame_blobs:
        f.write(blob)
sz = os.path.getsize(OUT_PATH)
print(f"wrote {OUT_PATH}: {total} frames, {sz/1e6:.2f} MB")
