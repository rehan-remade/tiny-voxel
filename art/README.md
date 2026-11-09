# art/ — source assets (Git LFS)

Working files for the creature/asset pipeline (concept image → 3D mesh →
voxelize → rig/animate → engine bake). One folder per asset, PascalCase:

```
art/<Name>/
  <Name>_concept.png     picked concept image
  <Name>_mesh.glb        generated 3D mesh (.obj for legacy assets)
  <Name>.vox             picked voxelization
  <Name>_rigged.blend    rigged/animated Blender file
  <Name>_animated.glb    exported animation clips
  previews/              QA renders and clip previews
  archive/               superseded candidates (never delete)
```

The baked runtime formats (`.cvx2` + `.def`) that the game actually loads
live in `assets/`, not here. See `.claude/skills/animated-asset/SKILL.md`
for the full pipeline.

All binary files here are stored in **Git LFS**. Before cloning or pulling,
run once:

```
git lfs install
```

If you see tiny text files starting with `version https://git-lfs...`
instead of real assets, LFS isn't set up — install it and run `git lfs pull`.
