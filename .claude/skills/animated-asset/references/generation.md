# Generation reference (stages 1–2)

## Text-to-image — Seedream 5 Pro

Input: `prompt` (required); `image_size` `{"width":1024,"height":1024}`;
`num_images` 4 for a candidate round (max 6); `output_format` "png".
Output: `images[]`, one `.url` per candidate — keep the URLs, stage 2
consumes the picked one directly. Wall time ~2 min for one image — poll
every 5 s, budget 6 min for a 4-image round.
Alternates: GPT Image 2, Nano Banana 2 (check their inputs first).

### Prompt template (proven)

> A 3D render of a voxel art game asset: a {adjective} {creature}
> {pose, e.g. "standing"}, viewed from a side three-quarter angle. Built
> from large chunky visible cubes on a strict grid, roughly {L} cubes long
> and {H} cubes tall, MagicaVoxel style, crisp axis-aligned blocky edges,
> no smooth curves. {2–3 sentences of anatomy as BLOCKY shapes: "extremely
> short thick legs one cube tall", "oversized head", "fluff rendered as
> solid stepped cube layers, not strands"; suppress thin parts: "no tail"
> or "tail a thick plume slab lying flat against its side"}. Flat colors
> in clean regions: {color regions}. Centered, entire model in frame,
> plain solid light-gray background, soft even studio lighting, no
> shadows on background.

Rules of thumb: state cube counts matching the intended in-game size
(voxels = meters x 10) — they control style; the voxelizer enforces exact
dims. Thin appendages (tails, whiskers, thin antler tines) must be
thickened, slabbed against the body, or omitted — they shred at low voxel
resolutions. Legs must be separated from each other and the ground plane.

## Image-to-3D — Hunyuan 3D v3.1 Pro

Input: `input_image_url` (required; 128–5000 px, <8 MB — pass the stage-1
URL directly); `generate_type` "Normal" (textured); `enable_pbr` true;
`face_count` 40000–1500000 (default 500k ≈ 45 MB GLB; 100k is plenty);
optional multi-view conditioning: `back_image_url`, `left/right/top/...`.
Output: `model_glb.url`, `model_urls.{glb,obj,mtl,texture}`, `thumbnail.url`.
Wall time ~2.5 min — poll every 10–15 s, budget 10 min.

Known behavior: unseen views are hallucinated (a rear-facing "nose" blob
appeared on a corgi generated from a single 3/4 view). Usually 1–2 stray
voxels after voxelization — ignore, or supply `back_image_url` for a clean
rear.

## Cost sanity

Both models bill per run (concept ~cents, Hunyuan pro ~tens of cents).
Don't loop generations without a passing QA gate in between.
