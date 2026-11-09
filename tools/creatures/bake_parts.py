"""Bake an animated creature GLB into a .cvx2 rigid-part asset for the engine.

Unlike bake_flipbook.py (which re-samples every animation frame onto one
axis-aligned grid, causing silhouette voxels to pop frame-to-frame), this
baker keeps each bone's voxels as a REST-POSE volume and exports the bone's
rigid transform per frame. The engine rotates the volumes at runtime, so the
voxels animate exactly like the Blender preview — no re-sampling, no flicker
— and animation data shrinks from per-frame grids to per-frame transforms.

Run headless:
  blender -b --factory-startup --python bake_parts.py -- \
      --vox Stag.vox --glb Stag_animated.glb --out stag.cvx2 \
      --rules stag_rules.py [--step 1]

Format: see docs/creature-animation-roadmap.md. Creature-local axes are
x lateral / y up / z forward with the origin at the feet centre (armature
origin); tracks store each part's creature-local rigid transform per frame
(quat + pos applied to rest-space points).
"""
import importlib.util
import os
import struct
import sys

import bpy
import numpy as np
from mathutils import Matrix

bpy.ops.wm.read_factory_settings(use_empty=True)

argv = sys.argv[sys.argv.index("--") + 1:]


def arg(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


VOX_PATH = arg("--vox")
GLB_PATH = arg("--glb")
OUT_PATH = arg("--out")
RULES_PATH = arg("--rules")
STEP = int(arg("--step", "1"))

# Metres per MODEL voxel. Wildlife uses the world rule (0.1 = 10 cm); the
# player model is normalized to 1.8 m over 32 voxels (pass --scale 0.05625).
# The ratio S/0.1 ships in the header as cellScale — world voxels per cell.
S = float(arg("--scale", "0.1"))
LOOP_CLIPS = {"Idle", "Walk", "Gallop", "Graze", "Alert",
              "Sprint", "CrouchIdle", "CrouchWalk", "Chop", "Mine"}


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
# Model-specific preprocessing hook: a rules module may expose
# remap_voxels(size, voxels) to reorient a source model (e.g. the bear faces
# -X and rotates +90 deg about Z so every creature faces -Y).
if RULES_PATH:
    _spec = importlib.util.spec_from_file_location("rules_pre", RULES_PATH)
    _rp = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_rp)
    if hasattr(_rp, "remap_voxels"):
        size, voxels = _rp.remap_voxels(size, voxels)
        print(f"remapped: {size}")
print(f"vox: {size} voxels={len(voxels)}")

bpy.ops.import_scene.gltf(filepath=GLB_PATH)
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
mesh_obj = next(o for o in bpy.data.objects if o.type == "MESH" and o.vertex_groups)
mesh = mesh_obj.data
bone_names = [b.name for b in arm.data.bones]
print(f"glb: mesh={mesh_obj.name} bones={len(bone_names)}")

# ---- voxel lattice recovery (same as bake_flipbook.py) -------------------------
verts_local = np.array([v.co[:] for v in mesh.vertices], dtype=np.float64)
lat_f = verts_local / S
frac = np.median(lat_f - np.floor(lat_f), axis=0)
frac[frac > 0.75] -= 1.0
lat = np.rint(lat_f - frac).astype(np.int64)
resid = np.abs(lat_f - frac - lat).max()
if resid > 0.05:
    raise SystemExit(f"mesh verts are not on the voxel lattice (residual {resid:.3f})")
vox_arr = np.array([(x, y, z) for x, y, z, _ in voxels], dtype=np.int64)
col_arr = np.array([ci for _, _, _, ci in voxels], dtype=np.uint8)
corner_off = vox_arr.min(axis=0) - lat.min(axis=0)

# ---- voxel -> bone --------------------------------------------------------------
if RULES_PATH:
    spec = importlib.util.spec_from_file_location("rules", RULES_PATH)
    rules = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rules)
    vox_bone = [rules.bone_for_voxel(x, y, z) for x, y, z in vox_arr]
else:
    raise SystemExit("--rules is required (vertex-vote fallback not ported yet)")
missing = {b for b in vox_bone if b not in bone_names}
if missing:
    raise SystemExit(f"voxels assigned to unknown bones: {missing}")

by_bone = {}
for i, b in enumerate(vox_bone):
    by_bone.setdefault(b, []).append(i)
# Deterministic part order: skeleton-hierarchy depth then name (parents first).
part_bones = sorted(by_bone.keys(),
                    key=lambda b: (len(arm.data.bones[b].parent_recursive), b))
part_index = {b: i for i, b in enumerate(part_bones)}
print("parts:", {b: len(by_bone[b]) for b in part_bones})

# ---- joint overlap ---------------------------------------------------------------
# Every voxel belongs to exactly ONE bone, so when a joint bends, the seam
# between a part and its skeleton neighbour opens a visible gap (legs detach
# from the torso mid-stride). Plug it by duplicating the 1-voxel shell ACROSS
# each joint: any voxel 6-adjacent to a parent/child part is copied into that
# part too. Straight pose: the copies coincide (invisible). Bent joint: each
# copy swings with its part and bridges the seam from both sides.
def _skel_parent(b):
    p = arm.data.bones[b].parent
    while p is not None and p.name not in part_index:
        p = p.parent                      # skip voxel-less helper bones
    return p.name if p else None


adjacent = {b: set() for b in part_bones}
for b in part_bones:
    p = _skel_parent(b)
    if p:
        adjacent[b].add(p)
        adjacent[p].add(b)
vox_index = {(int(x), int(y), int(z)): i for i, (x, y, z) in enumerate(vox_arr)}
extra = {b: set() for b in part_bones}
for i, (x, y, z) in enumerate(vox_arr):
    b = vox_bone[i]
    for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                       (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        j = vox_index.get((int(x) + dx, int(y) + dy, int(z) + dz))
        if j is None:
            continue
        b2 = vox_bone[j]
        if b2 != b and b2 in adjacent[b]:
            extra[b2].add(i)              # i sits on b2's seam: copy it over
if any(extra.values()):
    print("joint overlap:", {b: len(s) for b, s in extra.items() if s})

mesh_to_arm = arm.matrix_world.inverted() @ mesh_obj.matrix_world
M = np.array(mesh_to_arm)

# Engine axis map (blender armature Z-up, facing -Y  ->  engine y-up, +z fwd).
A = np.array([[1.0, 0.0,  0.0],
              [0.0, 0.0,  1.0],
              [0.0, -1.0, 0.0]])   # rows: eng.x = bl.x, eng.y = bl.z, eng.z = -bl.y


def to_engine_pts(p):
    return p @ A.T


# ---- part volumes (rest pose, tight AABB, engine cell indexing) -----------------
# Vox coords: x width, y length (blender -Y forward), z up. Engine cell
# (i,j,k) of a part maps back to vox cell (mnx+i, mxy-k, mnz+j): x is shared,
# engine y = vox z (up), engine z flips vox y (forward).
parts = []
vox_blob_words = []
for b in part_bones:
    ix = by_bone[b] + sorted(extra[b] - set(by_bone[b]))
    v = vox_arr[ix]
    mn = v.min(axis=0)
    mx = v.max(axis=0)
    gx = int(mx[0] - mn[0] + 1)
    gy = int(mx[2] - mn[2] + 1)          # engine up = vox z
    gz = int(mx[1] - mn[1] + 1)          # engine fwd = -vox y
    grid = np.zeros((gx, gy, gz), dtype=np.uint8)
    for (x, y, z), ci in zip(v, col_arr[ix]):
        grid[x - mn[0], z - mn[2], mx[1] - y] = ci
    # gridOrigin: engine coords of cell (0,0,0)'s min corner. Compute from the
    # vox AABB's 8 corners through mesh -> armature -> engine, take the min.
    corners = []
    for cx in (mn[0], mx[0] + 1):
        for cy in (mn[1], mx[1] + 1):
            for cz in (mn[2], mx[2] + 1):
                pm = (np.array([cx, cy, cz], dtype=np.float64) - corner_off + frac) * S
                pa = pm @ M[:3, :3].T + M[:3, 3]
                corners.append(pa)
    eng = to_engine_pts(np.array(corners))
    origin = eng.min(axis=0) / S          # engine units are voxels at runtime
    words = np.zeros(( (gx * gy * gz + 3) // 4, ), dtype=np.uint32)
    flat = grid.transpose(2, 1, 0).reshape(-1)      # x-fastest
    for i, ci in enumerate(flat):
        if ci:
            words[i >> 2] |= np.uint32(int(ci)) << np.uint32((i & 3) * 8)
    parent = arm.data.bones[b].parent
    while parent is not None and parent.name not in part_index:
        parent = parent.parent            # skip voxel-less helper bones (root)
    # Pivot = the bone head in creature-local rest space: the joint the part
    # rotates about (PR 2's look-at / IK layer rotates parts around these).
    head = np.array(arm.data.bones[b].head_local, dtype=np.float64)
    pivot = (head @ A.T) / S
    parts.append({
        "bone": b,
        "parent": part_index[parent.name] if parent else -1,
        "origin": origin,
        "pivot": pivot,
        "dims": (gx, gy, gz),
        "wordBase": sum(len(w) for w in vox_blob_words),
    })
    vox_blob_words.append(words)
    print(f"  part {b:10s} dims {gx}x{gy}x{gz} origin {origin.round(2)} parent {parts[-1]['parent']}")

rest_inv = {b: np.array(arm.data.bones[b].matrix_local.inverted()) for b in part_bones}

# ---- clips + tracks --------------------------------------------------------------
scene = bpy.context.scene
fps = scene.render.fps
clips = []
tracks = []   # per frame: partCount x 7 floats

for act in bpy.data.actions:
    name = act.name
    if not arm.animation_data:
        arm.animation_data_create()
    arm.animation_data.action = act
    if hasattr(arm.animation_data, "action_slot") and len(act.slots):
        arm.animation_data.action_slot = act.slots[0]
    f0, f1 = int(act.frame_range[0]), int(act.frame_range[1])
    loop = name in LOOP_CLIPS
    if loop and f1 > f0:
        f1 -= 1
    first = len(tracks)
    for f in range(f0, f1 + 1, STEP):
        scene.frame_set(f)
        bpy.context.view_layer.update()
        row = np.zeros((len(part_bones), 7), dtype=np.float32)
        for pi, b in enumerate(part_bones):
            pose = np.array(arm.pose.bones[b].matrix)
            D = pose @ rest_inv[b]                    # rest arm-space -> posed
            R_eng = A @ D[:3, :3] @ A.T
            t_eng = A @ D[:3, 3] / S                  # voxel units
            q = Matrix(R_eng.tolist()).to_quaternion()
            row[pi] = (q.x, q.y, q.z, q.w, t_eng[0], t_eng[1], t_eng[2])
        tracks.append(row)
    clips.append((name, first, len(tracks) - first, fps / STEP, loop))
    print(f"clip {name}: {len(tracks) - first} frames (loop={loop})")

# ---- write .cvx2 -----------------------------------------------------------------
with open(OUT_PATH, "wb") as f:
    f.write(b"CVX2")
    f.write(struct.pack("<I", 3))
    f.write(struct.pack("<III", len(parts), len(clips), len(tracks)))
    f.write(struct.pack("<f", S / 0.1))   # cellScale: world voxels per cell
    for i in range(256):
        f.write(struct.pack("<4B", *(palette[i] if palette else (200, 200, 200, 255))))
    for p in parts:
        nb = p["bone"].encode()[:16]
        f.write(nb + b"\0" * (16 - len(nb)))
        f.write(struct.pack("<i", p["parent"]))
        f.write(struct.pack("<3f", *p["origin"]))
        f.write(struct.pack("<3f", *p["pivot"]))
        f.write(struct.pack("<3I", *p["dims"]))
        f.write(struct.pack("<I", p["wordBase"]))
    for name, first, count, cfps, loop in clips:
        nb = name.encode()[:16]
        f.write(nb + b"\0" * (16 - len(nb)))
        f.write(struct.pack("<IIfI", first, count, cfps, 1 if loop else 0))
    for row in tracks:
        f.write(row.tobytes())
    for words in vox_blob_words:
        f.write(words.tobytes())
sz = os.path.getsize(OUT_PATH)
nvox = int(sum((np.frombuffer(w.tobytes(), dtype=np.uint8) > 0).sum() for w in vox_blob_words))
print(f"wrote {OUT_PATH}: {len(parts)} parts ({nvox} voxels), "
      f"{len(tracks)} track frames, {sz/1e3:.1f} KB")
