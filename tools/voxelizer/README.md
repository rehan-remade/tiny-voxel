# voxelizer

Converts a textured mesh (OBJ or GLB, e.g. a Hunyuan-3D export) into a
MagicaVoxel `.vox` model — the in-repo replacement for the drububu.com web
voxelizer the asset pipeline used manually. Verified against that tool's
outputs on the Bear/Player/Stag source meshes: 0.97–0.998 occupancy IoU,
~3% mean color difference.

Build (part of the normal CMake tree, `VOXEL_TOOLS` option, ON by default):

```
cmake --build build --config Release --target voxelizer
```

Use:

```
voxelizer <mesh.obj|mesh.glb> --out <model.vox> [options]
  --long-axis N       voxels along the longest mesh axis (default 32).
                      Project scale: 1 voxel = 10 cm, so voxels = meters x 10.
  --axis x|y|z        pick the driving axis explicitly (y = up)
  --dims LxHxD        exact grid dims (length x height x depth)
  --cell M            explicit cell size in mesh units
  --texture P.png     albedo for OBJ inputs whose .mtl is missing
  --yaw 0|90|180|270  rotate about the up axis first; creatures must face -Y
  --anchor min|center grid placement against the mesh AABB (default min,
                      which matches the web tool)
  --fill              solid interior (default: hollow shell)
  --previews DIR      write <name>_vox_front/_side/_top.png QA renders
  --ascii             print per-height ASCII slices (bone-rule authoring)
```

Semantics: the mesh AABB anchors a uniform grid (cell = driving extent /
N); a voxel is set when any triangle overlaps its cell (SAT test); voxels
fully enclosed from outside air are dropped unless `--fill`; colors come
from the albedo sampled at the nearest point of an overlapping triangle,
averaged per voxel, quantized to a 255-color palette sorted dark→light.
Y-up mesh space maps to Z-up vox space as `vox(x,y,z) = (x, D-1-gz, gy)` —
a proper rotation (no mirroring), so a mesh facing +Z faces -Y in vox.

Vendored single-header deps (MIT/public-domain) in `third_party/`:
tinyobjloader, cgltf, stb_image, stb_image_write.

The full text→image→3D→vox→rig pipeline is codified in the
`animated-asset` skill (`.claude/skills/animated-asset/`).
