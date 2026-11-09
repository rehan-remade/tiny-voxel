"""Build a rigged stag from Stag.vox with three animation clips:
Walk (elegant), Graze, Alert. Run headless:
blender -b --factory-startup --python stag_build2.py -- <vox> <outdir> [render] [video] [export]
"""
import math
import os
import struct
import sys

import bpy
from mathutils import Vector

bpy.ops.wm.read_factory_settings(use_empty=True)  # no default cube/light/cam

argv = sys.argv[sys.argv.index("--") + 1:]
VOX_PATH = argv[0]
OUT_DIR = argv[1]
DO_RENDER = "render" in argv[2:]
DO_VIDEO = "video" in argv[2:]
DO_EXPORT = "export" in argv[2:]

S = 0.05             # 44-tall fine bake: 1 cell = 5 cm (cellScale 0.5)
CX, CY = 12.0, 16.0  # voxel-space center (x on body midline, y mid-length)
FPS = 24

# ---- walk tuning (elegant, not marching) -----------------------------------
# Marching = high knee lift + metronome symmetry. Deer elegance = long slow
# stride, LOW delicate hoof arc, soft heel-peel at push-off, quiet body.
CYC = 36             # frames per stride (1.5 s: stately)
STRIDE_F = 0.30      # m, front hoof fore/aft travel
STRIDE_H = 0.28      # m, hind hoof fore/aft travel
STEP_H_F = 0.075     # m, front swing lift (low!)
STEP_H_H = 0.060     # m, hind swing lift
STANCE = 0.62        # fraction of cycle the hoof is planted
PEEL = 0.18          # last fraction of stance: heel peels off ground
PEEL_H = 0.022       # m, heel lift at end of peel
BODY_DROP = 0.025    # m, standing crouch so legs keep a natural micro-bend
BOB_AMP = 0.014      # m, vertical body bob (2x per cycle)
SURGE_AMP = 0.010    # m, fore-aft body sway (2x per cycle)
ROLL_AMP = 2.0       # deg, body roll toward stance side
PITCH_AMP = 1.0      # deg, body pitch rock
NECK_NOD = 3.5       # deg, neck nod (2x per cycle)
HEAD_NOD = 3.0       # deg, head counter-nod (keeps the head serene)
# lateral-sequence walk; contact at t = 1 - offset, so offsets are reversed:
PHASE = {"HL": 0.00, "FL": 0.75, "HR": 0.50, "FR": 0.25}
# leg IK orientation (verified): local-X sign that tilts a bone tail to -Y
LX_SIGN = -1.0
# IK bend branch. Ungulate anatomy: the front carpus apex points FORWARD
# (hoof curls back beneath it, like a horse folding its leg); the hind hock
# apex points BACKWARD. Verified visually after the flip.
KNEE_S = {"F": 1.0, "H": -1.0}


# ---- vox load ----------------------------------------------------------------
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


def srgb_to_linear(c):
    c /= 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


# ---- bone assignment rules (voxel coords; from slice analysis of Stag44) -----
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


# claim order: parents first so shared boundary verts stay body-side
BONE_ORDER = ["hips", "chest", "neck", "head",
              "FL.upper", "FR.upper", "HL.upper", "HR.upper",
              "FL.lower", "FR.lower", "HL.lower", "HR.lower"]

# ---- build mesh ---------------------------------------------------------------
size, voxels, palette = load_vox(VOX_PATH)
occ = {(x, y, z): ci for x, y, z, ci in voxels}

verts, vmap, faces, face_bone, face_col = [], {}, [], [], []

FACE_DIRS = [
    ((1, 0, 0),  [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),
    ((-1, 0, 0), [(0, 1, 0), (0, 0, 0), (0, 0, 1), (0, 1, 1)]),
    ((0, 1, 0),  [(1, 1, 0), (0, 1, 0), (0, 1, 1), (1, 1, 1)]),
    ((0, -1, 0), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),
    ((0, 0, 1),  [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),
    ((0, 0, -1), [(0, 1, 0), (1, 1, 0), (1, 0, 0), (0, 0, 0)]),
]


def vidx(px, py, pz):
    key = (px, py, pz)
    if key not in vmap:
        vmap[key] = len(verts)
        verts.append(((px - CX) * S, (py - CY) * S, pz * S))
    return vmap[key]


for (x, y, z), ci in occ.items():
    bone = bone_for_voxel(x, y, z)
    r, g, b, a = palette[ci - 1] if palette else (200, 200, 200, 255)
    col = (srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b), 1.0)
    for (dx, dy, dz), corners in FACE_DIRS:
        if (x + dx, y + dy, z + dz) in occ:
            continue
        faces.append(tuple(vidx(x + cx_, y + cy_, z + cz_) for cx_, cy_, cz_ in corners))
        face_bone.append(bone)
        face_col.append(col)

mesh = bpy.data.meshes.new("Stag")
mesh.from_pydata(verts, [], faces)
mesh.update()

ca = mesh.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="CORNER")
li = 0
for poly, col in zip(mesh.polygons, face_col):
    for _ in range(poly.loop_total):
        ca.data[li].color = col
        li += 1

obj = bpy.data.objects.new("Stag", mesh)
bpy.context.collection.objects.link(obj)

mat = bpy.data.materials.new("StagVox")
mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
vc = mat.node_tree.nodes.new("ShaderNodeVertexColor")
vc.layer_name = "Col"
mat.node_tree.links.new(vc.outputs["Color"], bsdf.inputs["Base Color"])
bsdf.inputs["Roughness"].default_value = 0.85
mesh.materials.append(mat)

# ---- armature -------------------------------------------------------------------
arm_data = bpy.data.armatures.new("StagRig")
arm = bpy.data.objects.new("StagRig", arm_data)
bpy.context.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="EDIT")

LEG_X = {"FL": 0.20, "FR": -0.20, "HL": 0.25, "HR": -0.25}
LEG_Y = {"F": -0.30, "H": 0.70}
KNEE_Z, HIP_Z = 0.40, 0.70   # high hock/carpus, short thigh + long shank

BONES = {
    "root":  ((0, 0.95, 0.0), (0, 0.60, 0.0), None, False),
    "hips":  ((0, 0.70, 0.90), (0, 0.15, 0.92), "root", False),
    "chest": ((0, 0.15, 0.92), (0, -0.30, 0.95), "hips", True),
    "neck":  ((0, -0.25, 1.10), (0, -0.50, 1.45), "chest", False),
    "head":  ((0, -0.50, 1.45), (0, -0.82, 1.50), "neck", True),
}
for leg in ("FL", "FR", "HL", "HR"):
    lx, ly = LEG_X[leg], LEG_Y[leg[0]]
    parent = "chest" if leg[0] == "F" else "hips"
    BONES[f"{leg}.upper"] = ((lx, ly, HIP_Z), (lx, ly, KNEE_Z), parent, False)
    BONES[f"{leg}.lower"] = ((lx, ly, KNEE_Z), (lx, ly, 0.0), f"{leg}.upper", True)

ebs = arm_data.edit_bones
for name in ["root"] + BONE_ORDER:
    h, t, parent, connect = BONES[name]
    b = ebs.new(name)
    b.head, b.tail = Vector(h), Vector(t)
    if parent:
        b.parent = ebs[parent]
        b.use_connect = connect
bpy.ops.object.mode_set(mode="OBJECT")

# ---- weights (rigid, first-claim in parent-first order) ------------------------
vgroups = {n: obj.vertex_groups.new(name=n) for n in BONE_ORDER}
claimed = set()
by_bone = {n: [] for n in BONE_ORDER}
for f_i, poly in enumerate(mesh.polygons):
    by_bone[face_bone[f_i]].extend(poly.vertices)
for name in BONE_ORDER:
    fresh = [v for v in set(by_bone[name]) if v not in claimed]
    if fresh:
        vgroups[name].add(fresh, 1.0, "REPLACE")
        claimed.update(fresh)

obj.parent = arm
mod = obj.modifiers.new("Armature", "ARMATURE")
mod.object = arm

# ---- animation helpers -----------------------------------------------------------
scene = bpy.context.scene
scene.render.fps = FPS

pb = arm.pose.bones
for b in pb:
    b.rotation_mode = "XYZ"

L1 = HIP_Z - KNEE_Z
L2 = KNEE_Z


def clear_pose():
    for b in pb:
        b.location = (0, 0, 0)
        b.rotation_euler = (0, 0, 0)


def smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def foot_traj(p, stride, steph, stance=STANCE, peel=PEEL, peelh=PEEL_H):
    """Hoof path in sagittal plane; p=0 contact. Returns (fwd, height)."""
    if p < stance:
        u = p / stance
        z = 0.0
        if u > 1.0 - peel:                    # heel peels before liftoff
            w = (u - (1.0 - peel)) / peel
            z = peelh * w * w
        return stride * (0.5 - u), z
    u = (p - stance) / (1.0 - stance)
    return stride * (-0.5 + smoothstep(u)), steph * math.sin(math.pi * u ** 0.85)


def solve_leg(dx, zf, leg, hip_h, delta=0.0):
    """2-bone sagittal IK -> (upper_x_deg, lower_x_deg).
    delta: parent-induced pitch of the leg's rest direction (rad, forward+),
    subtracted so body pitch doesn't drag the hooves with it."""
    dz = hip_h - zf
    D = math.hypot(dx, dz)
    D = max(abs(L1 - L2) + 1e-3, min(D, L1 + L2 - 1e-3))
    theta = math.atan2(dx, dz)
    a1 = math.acos(max(-1, min(1, (L1*L1 + D*D - L2*L2) / (2*L1*D))))
    a2 = math.acos(max(-1, min(1, (L1*L1 + L2*L2 - D*D) / (2*L1*L2))))
    s = KNEE_S[leg[0]]
    return (math.degrees(LX_SIGN * (theta - delta + s*a1)),
            math.degrees(LX_SIGN * (-s*(math.pi - a2))))


def key_rot(bone, frame, xyz_deg):
    b = pb[bone]
    b.rotation_euler = [math.radians(a) for a in xyz_deg]
    b.keyframe_insert("rotation_euler", frame=frame)


def key_loc(bone, frame, loc):
    b = pb[bone]
    b.location = loc
    b.keyframe_insert("location", frame=frame)


# Pose-bone location is in the bone's LOCAL rest frame; for the hips bone
# local Z points along world -Z (measured), so express intent in WORLD space
# and convert. dy: world +Y = toward tail. dz: world +Z = up.
HIPS_W2L = None  # resolved lazily; edit-bones must exist first


def key_hips_world(frame, dy, dz):
    global HIPS_W2L
    if HIPS_W2L is None:
        HIPS_W2L = pb["hips"].bone.matrix_local.to_3x3().inverted()
    key_loc("hips", frame, tuple(HIPS_W2L @ Vector((0.0, dy, dz))))


def key_stand_legs(frame, hip_h=HIP_Z - BODY_DROP):
    for leg in ("FL", "FR", "HL", "HR"):
        ux, lx = solve_leg(0.0, 0.0, leg, hip_h)
        key_rot(f"{leg}.upper", frame, (ux, 0, 0))
        key_rot(f"{leg}.lower", frame, (lx, 0, 0))


def new_action(name):
    act = bpy.data.actions.new(name)
    if arm.animation_data is None:
        arm.animation_data_create()
    arm.animation_data.action = act
    clear_pose()
    return act


def make_cyclic(act):
    for layer in act.layers:
        for strip in layer.strips:
            for slot in act.slots:
                cb = strip.channelbag(slot)
                if cb:
                    for fc in cb.fcurves:
                        if not any(m.type == "CYCLES" for m in fc.modifiers):
                            fc.modifiers.new("CYCLES")


# ---- neck/head sign self-calibration ----------------------------------------------
def muzzle():
    bpy.context.view_layer.update()
    return (arm.matrix_world @ pb["head"].tail).copy()


clear_pose()
REST_MZ = muzzle()


def probe_sign(bone, axis, want):
    """Sign of a +20deg rotation on axis ('x'/'z') that moves the muzzle in
    +want direction ('down' => -z, 'left' => +x)."""
    clear_pose()
    e = [0.0, 0.0, 0.0]
    e["xyz".index(axis)] = math.radians(20)
    pb[bone].rotation_euler = e
    d = muzzle() - REST_MZ
    clear_pose()
    v = -d.z if want == "down" else d.x
    return 1.0 if v > 0 else -1.0


N_D = probe_sign("neck", "x", "down")   # neck pitch-down sign
H_D = probe_sign("head", "x", "down")   # head pitch-down sign
C_D = probe_sign("chest", "x", "down")  # chest pitch-down sign
HP_D = probe_sign("hips", "x", "down")  # whole-body nose-down sign (via hips)
N_Y = probe_sign("neck", "z", "left")   # neck yaw-left sign
H_Y = probe_sign("head", "z", "left")   # head yaw-left sign
print(f"CAL signs: neckD={N_D} headD={H_D} chestD={C_D} neckY={N_Y} headY={H_Y}")

# Graze-down pose: the neck plunges but the HEAD COUNTER-ROTATES so the face
# stays near-vertical (muzzle pointing down-forward, antlers swept back over
# the shoulders). Head fold is solved from a muzzle-direction target, never a
# fold ratio — a ratio tips the antlers forward like a charging ram.
GRAZE_CHEST = 12.0
HEAD_REST_ELEV = 9.0     # deg, head bone elevation at rest (from geometry)
MUZ_ELEV = -65.0         # deg, muzzle points 65deg below horizontal when down
TARGET_MUZ_Z = 0.32      # grass-top height at voxel scale


def graze_head_for(neck_t):
    return HEAD_REST_ELEV - MUZ_ELEV - GRAZE_CHEST - neck_t  # downward-positive


def graze_muzzle_z(t):
    clear_pose()
    pb["chest"].rotation_euler = (math.radians(C_D * GRAZE_CHEST), 0, 0)
    pb["neck"].rotation_euler = (math.radians(N_D * t), 0, 0)
    pb["head"].rotation_euler = (math.radians(H_D * graze_head_for(t)), 0, 0)
    return muzzle().z


lo, hi = 40.0, 105.0
for _ in range(40):
    mid = 0.5 * (lo + hi)
    if graze_muzzle_z(mid) > TARGET_MUZ_Z:
        lo = mid
    else:
        hi = mid
GRAZE_NECK = 0.5 * (lo + hi)
GRAZE_HEAD = graze_head_for(GRAZE_NECK)
print(f"CAL graze: neck={GRAZE_NECK:.1f} head={GRAZE_HEAD:.1f} "
      f"muzzle_z@down={graze_muzzle_z(GRAZE_NECK):.3f}")
clear_pose()

# ---- unified gait machinery (walk, gallop, and the transitions between) ------------
GCYC = 12                 # frames per gallop stride (0.5 s ~ 2 strides/s)
STANCE_G = 0.32
PHASE_G = {"HL": 0.00, "HR": 0.88, "FL": 0.68, "FR": 0.56}
STRIDE_G = {"F": 0.44, "H": 0.46}
STEP_G = {"F": 0.16, "H": 0.20}     # hinds tuck high under the belly
BODY_DROP_G = 0.065       # deep running crouch; legs never straighten at speed
BOB_G = 0.045
SURGE_G = 0.03
GPITCH = 5.5              # deg nose-up amplitude
GNECK = 5.0
LEGS = ("FL", "FR", "HL", "HR")


def hump(g, c, w):
    """Raised-cosine bump at cycle position c, half-width w (wraps)."""
    d = abs(g - c)
    d = min(d, 1.0 - d)
    return 0.5 * (1 + math.cos(math.pi * d / w)) if d < w else 0.0


def walk_channels(t):
    bob = -BOB_AMP * math.cos(4 * math.pi * (t + 0.05))
    surge = SURGE_AMP * math.sin(4 * math.pi * t + 0.9)
    roll = ROLL_AMP * math.sin(2 * math.pi * t)
    pitch = PITCH_AMP * math.sin(4 * math.pi * t + 0.7)
    nod = NECK_NOD * math.sin(4 * math.pi * t + 1.2)
    return {"dy": -surge, "dz": -BODY_DROP + bob,
            "hips_rx": pitch, "hips_ry": roll,
            "chest_rx": -pitch * 0.6, "chest_ry": -roll * 0.5,
            "neck_rx": N_D * nod,
            "head_rx": -H_D * HEAD_NOD * math.sin(4 * math.pi * t + 1.2)}


def gallop_channels(g):
    # body low over the loaded fronts (g~0.45), high in flight (g~0.95);
    # explicit croup compression while the hinds load (the hips pivot sits at
    # the hind roots, so nose-up pitch alone never drops the croup)
    bob = (-BOB_G * math.cos(2 * math.pi * (g - 0.45))
           - 0.035 * hump(g, 0.15, 0.28))
    surge = SURGE_G * math.sin(2 * math.pi * (g - 0.1))
    # nose-up while the hinds load, nose DIVES into the front catch
    nose_up = GPITCH * math.cos(2 * math.pi * (g - 0.12))
    neck = GNECK * math.cos(2 * math.pi * (g - 0.5))
    return {"dy": -surge, "dz": -BODY_DROP_G + bob,
            "hips_rx": -HP_D * nose_up, "hips_ry": 0.0,
            "chest_rx": C_D * 0.45 * nose_up, "chest_ry": 0.0,
            "neck_rx": N_D * neck, "head_rx": -H_D * 0.6 * neck}


def walk_foot(leg, t):
    stride = STRIDE_F if leg[0] == "F" else STRIDE_H
    steph = STEP_H_F if leg[0] == "F" else STEP_H_H
    return foot_traj((t + PHASE[leg]) % 1.0, stride, steph)


def gallop_foot(leg, g):
    return foot_traj((g + PHASE_G[leg]) % 1.0, STRIDE_G[leg[0]], STEP_G[leg[0]],
                     stance=STANCE_G, peel=0.25, peelh=0.03)


def key_gait_frame(f, phi, w):
    """Key one frame of a gait: w=0 pure walk, w=1 pure gallop, else a blend.
    Body channels and foot POSITIONS blend (positions, not parameters, so a
    mid-stance foot never teleports when the stance fraction morphs). The body
    is keyed first, then the scene evaluates at f, then the leg roots come
    from the parents' pose matrices — immune to animation-eval stomping."""
    cw = walk_channels(phi)
    cg = gallop_channels(phi)

    def L(k):
        return (1 - w) * cw[k] + w * cg[k]

    key_hips_world(f, L("dy"), L("dz"))
    key_rot("hips", f, (L("hips_rx"), L("hips_ry"), 0))
    key_rot("chest", f, (L("chest_rx"), L("chest_ry"), 0))
    key_rot("neck", f, (L("neck_rx"), 0, 0))
    key_rot("head", f, (L("head_rx"), 0, 0))
    scene.frame_set(f)
    bpy.context.view_layer.update()
    for leg in LEGS:
        par = pb["chest" if leg[0] == "F" else "hips"]
        M = par.matrix @ par.bone.matrix_local.inverted()   # rest -> posed
        root = arm.matrix_world @ (M @ Vector(BONES[f"{leg}.upper"][0]))
        v = M.to_3x3() @ Vector((0, 0, -1))
        delta = math.atan2(-v.y, -v.z)          # parent-induced rest tilt
        fw = walk_foot(leg, phi)
        fg = gallop_foot(leg, phi)
        fdx = (1 - w) * fw[0] + w * fg[0]
        zf = (1 - w) * fw[1] + w * fg[1]
        dx = fdx + (root.y - LEG_Y[leg[0]])     # compensate root fore/aft drift
        ux, lx = solve_leg(dx, zf, leg, root.z, delta)
        key_rot(f"{leg}.upper", f, (ux, 0, 0))
        key_rot(f"{leg}.lower", f, (lx, 0, 0))


# ---- clip 1: Walk -------------------------------------------------------------------
walk = new_action("Walk")
for f in range(1, CYC + 2, 2):
    key_gait_frame(f, ((f - 1) % CYC) / CYC, 0.0)
make_cyclic(walk)

# walk guard: planted hooves must sweep +Y (tailward) AND touch the ground
dg = bpy.context.evaluated_depsgraph_get()
ok = True
for leg in ("FL", "HL"):
    ys, zmin = [], 9.9
    for f in range(1, CYC + 1):
        p = (((f - 1) / CYC) + PHASE[leg]) % 1.0
        scene.frame_set(f)
        dg.update()
        ae = arm.evaluated_get(dg)
        tip = ae.matrix_world @ ae.pose.bones[f"{leg}.lower"].tail
        zmin = min(zmin, tip.z)
        if p < STANCE * 0.9:
            ys.append((p, tip.y))
    ys.sort()
    if ys[-1][1] <= ys[0][1] or zmin > 0.03:
        ok = False
    print(f"WALK GUARD {leg}: sweep {ys[0][1]:+.3f}->{ys[-1][1]:+.3f} minZ={zmin:.3f}")
print("DIRECTION GUARD:", "OK forward+grounded" if ok else "FAIL")

# ---- clip 2: Graze (96f loop: vigilant half-raise -> down -> nibble/shift -> raise)
graze = new_action("Graze")
GLEN = 96
UPFRAC = 0.55   # loop seam pose: head 55% raised (vigilance check)


def graze_pose(fr, frac, nib=0.0, yaw=0.0):
    """frac 0=neutral standing, 1=muzzle in grass. nib=extra head peck deg."""
    key_rot("chest", fr, (C_D * GRAZE_CHEST * frac, 0, 0))
    key_rot("neck", fr, (N_D * GRAZE_NECK * frac, 0, N_Y * yaw))
    key_rot("head", fr, (H_D * (GRAZE_HEAD * frac + nib), 0, H_Y * yaw * 0.5))


GT = [(1, 1 - UPFRAC, 0, 0), (5, 1 - UPFRAC, 0, 0), (14, 1, 0, 0),
      (20, 1, 0, 0), (22, 1, 6, 0), (24, 1, 0, 0), (27, 1, 6, 0),
      (29, 1, 0, 0), (32, 1, 6, 0), (34, 1, 0, 0),
      (40, 1, 0, -9),                                    # new patch, right
      (44, 1, 6, -9), (46, 1, 0, -9), (49, 1, 6, -9), (51, 1, 0, -9),
      (58, 1, 0, 7),                                     # new patch, left
      (62, 1, 6, 7), (64, 1, 0, 7), (67, 1, 6, 7), (69, 1, 0, 7),
      (76, 1, 0, 0),
      (88, 1 - UPFRAC, 0, 0), (GLEN, 1 - UPFRAC, 0, 0)]
for fr, frac, nib, yaw in GT:
    graze_pose(fr, frac, nib, yaw)
key_stand_legs(1)
for fr, rl in ((1, 0.0), (30, 1.0), (60, -1.0), (GLEN, 0.0)):   # weight sway
    key_hips_world(fr, 0, -BODY_DROP)
    key_rot("hips", fr, (0, rl, 0))
make_cyclic(graze)

# ---- clip 3: Alert (120f loop: snap up -> freeze -> scan L/R -> ease down) ---------
alert = new_action("Alert")
ALEN = 120
A_NECK, A_HEAD, A_CHEST = 3.0, 2.0, 1.0    # deg up: posture hint only, no crane
A_YAW_N, A_YAW_H = 20.0, 13.0              # deg — the head TURN is the alert


def alert_pose(fr, up, yaw):
    key_rot("chest", fr, (-C_D * A_CHEST * up, 0, 0))
    key_rot("neck", fr, (-N_D * A_NECK * up, 0, N_Y * A_YAW_N * yaw))
    key_rot("head", fr, (-H_D * A_HEAD * up, 0, H_Y * A_YAW_H * yaw))


AT = [(1, 0, 0), (4, 0, 0), (8, 1, 1.0),      # SNAP: head whips left (alarm)
      (40, 1, 1.0),                           # frozen stare
      (52, 1, -1.0), (88, 1, -1.0),           # sweep to the right, hold
      (100, 1, 0), (110, 0.3, 0), (ALEN, 0, 0)]
for fr, up, yaw in AT:
    alert_pose(fr, up, yaw)
key_stand_legs(1)
key_hips_world(1, 0, -BODY_DROP)
key_hips_world(4, 0, -BODY_DROP)
key_hips_world(8, 0.018, -BODY_DROP - 0.004)              # weight rocks back
for f in range(12, 100, 4):                                # breathing micro-bob
    key_hips_world(f, 0.018, -BODY_DROP - 0.004
                   + 0.0025 * math.sin(2 * math.pi * (f - 8) / 22))
key_hips_world(104, 0.018, -BODY_DROP - 0.004)
key_hips_world(ALEN, 0, -BODY_DROP)
make_cyclic(alert)

# ---- clip 4: Gallop (transverse: HL, HR, FL, FR contacts, then suspension) ---------
gallop = new_action("Gallop")
for f in range(1, GCYC + 2):
    key_gait_frame(f, ((f - 1) % GCYC) / GCYC, 1.0)
make_cyclic(gallop)

# gallop gait report: grounded-hoof count per frame (expect a 0-count window)
dg = bpy.context.evaluated_depsgraph_get()
counts = []
for f in range(1, GCYC + 1):
    scene.frame_set(f)
    dg.update()
    ae = arm.evaluated_get(dg)
    zs = {l: (ae.matrix_world @ ae.pose.bones[f"{l}.lower"].tail).z for l in LEGS}
    n = sum(1 for l in LEGS if zs[l] < 0.05)
    counts.append(n)
print("GALLOP grounded hooves/frame:", counts,
      "| suspension:", "YES" if 0 in counts else "NO")

# ---- clip 5: Idle (144f: breathing, a glance left, a soft sniff, weight sway) ------
idle = new_action("Idle")
ILEN = 144
key_stand_legs(1)
for f in range(1, ILEN + 2, 6):                 # breathing, 2 s period
    ph = 2 * math.pi * (f - 1) / 48.0
    key_hips_world(f, 0, -BODY_DROP + 0.004 * math.sin(ph))
    key_rot("chest", f, (C_D * 0.7 * math.sin(ph + 0.5), 0, 0))
for fr, rl in ((1, 0.0), (45, 0.8), (100, -0.8), (ILEN + 1, 0.0)):
    key_rot("hips", fr, (0, rl, 0))
NT = [(1, 0, 0), (28, 0, 0), (36, 8, 0), (60, 8, 0), (70, 0, 0),
      (88, 0, 4), (104, 0, 4), (112, 0, 0), (ILEN + 1, 0, 0)]
for fr, yaw, dip in NT:
    key_rot("neck", fr, (N_D * dip * 0.6, 0, N_Y * yaw * 0.7))
    key_rot("head", fr, (H_D * dip * 0.4, 0, H_Y * yaw * 0.5))
make_cyclic(idle)


# ---- clips 6/7: gait transitions ----------------------------------------------------
def build_transition(name, frames, w_of_u):
    """Non-looping blend clip. A master phase clock advances at a cadence that
    morphs between the two gaits; the phase total is rescaled to land exactly
    on a cycle boundary, so frame 1 == source pose at phase 0 and the last
    frame == target pose at phase 0 — the clips chain seamlessly."""
    act = new_action(name)
    phis = [0.0]
    for i in range(1, frames):
        w_mid = w_of_u((i - 0.5) / (frames - 1))
        phis.append(phis[-1] + (1 - w_mid) / CYC + w_mid / GCYC)
    target = max(1.0, round(phis[-1]))
    phis = [p * target / phis[-1] for p in phis]
    for f in range(1, frames + 1):
        key_gait_frame(f, phis[f - 1] % 1.0, w_of_u((f - 1) / (frames - 1)))
    return act


w2g = build_transition("WalkToGallop", 30, smoothstep)
g2w = build_transition("GallopToWalk", 44, lambda u: 1.0 - smoothstep(u))

# ---- finalize: Walk active, stash all clips in NLA (muted) for the .blend ---------
CLIP_END = {"Walk": CYC, "Graze": GLEN, "Alert": ALEN, "Gallop": GCYC,
            "Idle": ILEN, "WalkToGallop": 30, "GallopToWalk": 44}


def activate(act):
    arm.animation_data.action = act
    if act.slots:
        arm.animation_data.action_slot = act.slots[0]
    scene.frame_start, scene.frame_end = 1, CLIP_END[act.name]


try:
    ad = arm.animation_data
    for act in (walk, graze, alert, gallop, idle, w2g, g2w):
        tr = ad.nla_tracks.new()
        tr.name = act.name
        strip = tr.strips.new(act.name, 1, act)
        if hasattr(strip, "action_slot") and act.slots:
            strip.action_slot = act.slots[0]
        tr.mute = True
except Exception as e:
    print("NLA stash skipped:", e)

activate(walk)

os.makedirs(OUT_DIR, exist_ok=True)
blend_path = os.path.join(OUT_DIR, "Stag_rigged.blend")
bpy.ops.wm.save_as_mainfile(filepath=blend_path)
print("SAVED", blend_path)

# ---- preview renders -----------------------------------------------------------------
def make_cam(name, loc, target=(0, 0.1, 0.85), ortho=None):
    cam_d = bpy.data.cameras.new(name)
    if ortho:
        cam_d.type = "ORTHO"
        cam_d.ortho_scale = ortho
    cam = bpy.data.objects.new(name, cam_d)
    bpy.context.collection.objects.link(cam)
    cam.location = loc
    d = Vector(target) - Vector(loc)
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    return cam


if DO_RENDER or DO_VIDEO:
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "VERTEX"
    scene.display.shading.show_object_outline = True
    scene.render.resolution_x = 720
    scene.render.resolution_y = 720
    scene.render.film_transparent = False
    gm = bpy.data.meshes.new("PreviewGround")
    gm.from_pydata([(-6, -6, 0), (6, -6, 0), (6, 6, 0), (-6, 6, 0)], [], [(0, 1, 2, 3)])
    gm.update()
    go = bpy.data.objects.new("PreviewGround", gm)
    bpy.context.collection.objects.link(go)

cam_side = make_cam("CamSide", (4.0, 0.1, 1.6), target=(0, 0.1, 0.95), ortho=3.0)
cam_3q = make_cam("Cam3Q", (2.9, -2.9, 1.9), target=(0, 0.1, 1.0))
cam_front = make_cam("CamFront", (0.15, -3.6, 1.35), target=(0, 0.1, 1.1))

SHOTS = {   # clip -> [(cam, tag, frames)]
    "Walk": [(cam_side, "side", (1, 5, 9, 13, 17, 21, 25, 29, 33)),
             (cam_3q, "3q", (1, 9, 17, 25, 33))],
    "Graze": [(cam_side, "side", (1, 8, 14, 23, 40, 50, 62, 80, 92)),
              (cam_3q, "3q", (14, 45, 63))],
    "Alert": [(cam_3q, "3q", (1, 6, 8, 30, 55, 85, 110)),
              (cam_front, "front", (8, 55, 85))],
    "Gallop": [(cam_side, "side", (1, 3, 5, 7, 9, 11)),
               (cam_3q, "3q", (2, 6, 10))],
    "Idle": [(cam_3q, "3q", (1, 40, 70, 100))],
    "WalkToGallop": [(cam_side, "side", (1, 8, 15, 22, 30))],
    "GallopToWalk": [(cam_side, "side", (1, 10, 22, 34, 44))],
}

if DO_RENDER:
    scene.render.image_settings.file_format = "PNG"
    for act in (walk, graze, alert, gallop, idle, w2g, g2w):
        activate(act)
        for cam, tag, frs in SHOTS[act.name]:
            scene.camera = cam
            for f in frs:
                scene.frame_set(f)
                scene.render.filepath = os.path.join(
                    OUT_DIR, f"{act.name.lower()}_{tag}_{f:03d}.png")
                bpy.ops.render.render(write_still=True)
    print("RENDERED stills")

if DO_VIDEO:
    scene.render.image_settings.media_type = "VIDEO"
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "HIGH"
    for act, loops, cam in ((walk, 3, cam_3q), (graze, 1, cam_3q),
                            (alert, 1, cam_3q), (gallop, 8, cam_3q),
                            (idle, 1, cam_3q)):
        activate(act)
        scene.frame_end = CLIP_END[act.name] * loops
        scene.camera = cam
        scene.render.filepath = os.path.join(
            OUT_DIR, f"Stag_{act.name.lower()}_preview.mp4")
        bpy.ops.render.render(animation=True)
        print(f"VIDEO {act.name} done")

    # seamless chain demo: walk -> accelerate -> gallop x3 -> decelerate -> walk
    try:
        ad = arm.animation_data
        ad.action = None
        demo = ad.nla_tracks.new()
        demo.name = "DemoChain"
        start = 1
        for act, rep in ((walk, 2), (w2g, 1), (gallop, 3), (g2w, 1), (walk, 2)):
            s = demo.strips.new(act.name, start, act)
            if hasattr(s, "action_slot") and act.slots:
                s.action_slot = act.slots[0]
            s.repeat = rep
            s.extrapolation = "NOTHING"
            start = int(s.frame_end) + 1
        scene.frame_start, scene.frame_end = 1, start - 1
        scene.camera = cam_side
        scene.render.filepath = os.path.join(OUT_DIR, "Stag_gait_demo.mp4")
        bpy.ops.render.render(animation=True)
        ad.nla_tracks.remove(demo)
        print("VIDEO Demo done")
    except Exception as e:
        print("DEMO video skipped:", e)
    activate(walk)

if DO_EXPORT:
    activate(walk)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    arm.select_set(True)
    bpy.ops.export_scene.gltf(
        filepath=os.path.join(OUT_DIR, "Stag_animated.glb"),
        export_format="GLB", use_selection=True,
        export_animations=True, export_animation_mode="ACTIONS")
    print("EXPORTED glb")

print("DONE")
