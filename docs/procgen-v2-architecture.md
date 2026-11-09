# Procgen v2 — Architecture Review & Migration Plan

Status: review/plan only (no code yet). Scope: take the existing single fixed
world (2048×4096×2048 voxels) to a robust, best-practices, chunk-streamed
procgen architecture — same content as today, but structured so any new biome
is one registry entry, and the world can grow without bound in XZ.

---

## 1. The three load-bearing findings

Everything in this plan follows from three facts about the current code:

**F1 — The field layer is already chunk-ready; the feature layer is not.**
`heightAt`, `stoneAt`, and the cave-noise are pure functions of integer
coordinates (hash-based, no tables that couldn't be virtualized). Any voxel of
*terrain* can already be recomputed in isolation. But every *feature* pass
(caves, boulders, bushes, trees, flowers, grass) consumes **one sequential
`mt19937` across the whole world** (`caves.cpp:29` seed `0x0CA7E5`,
`flora.cpp:26/116/206/480/593`). Feature N's placement depends on every RNG
draw made by features 1..N-1. Per-chunk regeneration is *mathematically
impossible* under this scheme. This is the single deepest incompatibility and
must be fixed first.

**F2 — The renderer wants a window, not chunks.**
The shader does a flat brick-DDA over one global brickmap SSBO
(`voxel.frag:1022-1148`), with `gridSize`/`bricksPerAxis` uniforms and fog
tuned to `gridSize.x/2`. Rewriting this as a chunk-pointer-chasing traversal
would be slow and invasive. The right fit is a **toroidal sliding window**:
keep the GPU layout byte-identical to today (256×512×256 bricks), but make it
a wrapping window into an unbounded world. The shader change is ~10 lines
(window-origin uniform + power-of-two masking); the fog already hides the
window edge at 1024 voxels.

**F3 — The cave pass is 96% of generation time *because* of F1.**
87 s of the 90 s cold regen is `genCaves` — 70 worm systems, each 2–3.6k
steps × O(W²) carve writes, fully serial. The systems are independent *except*
for the shared RNG sequence. Fixing F1 (per-feature seeding) automatically
unlocks parallel cave generation with the existing `StripePool` pattern —
87 s → ~10 s on this machine — before any chunking work even starts.

---

## 2. Current architecture (condensed inventory)

### 2.1 Storage (`common.h`, `world.h/cpp`, `worldio.cpp`)
- Two-level sparse grid: brickmap (33.5 M u32, 128 MB, fully GPU-resident) →
  brickpool (8³ bricks, 128 u32 each, grows monotonically, **no free list**,
  fully GPU-resident, ~1–2.3 GB).
- Encoding: `0`=empty, `0x80000000|mat`=solid, else pool index+1.
- `bmIdx()` linearization assumes fixed `BRICKS_XZ/BRICKS_Y` (`world.h:59`).
- `EditDelta` {dirty brickmap slots, dirty pool bricks} → `uploadDelta` does
  per-slot/per-brick `glBufferSubData`; pool growth = full re-upload.
- `world.dat` v2: 48-byte header validating exact grid dims + two zlib blobs;
  mtime invalidation against 6 source files. Monolithic — incompatible with
  per-chunk saves by design.
- `wetBricks` rebuilt is *skipped* on load (perf); slot-index encoded with
  fixed dims (`water.cpp:18-20`).

### 2.2 Generation (`src/worldgen/`)
Phases, costs, and chunk-readiness:

| Phase | Cost | RNG | Footprint | Chunk-ready? |
|---|---|---|---|---|
| Bulk fill | 1.7 s (parallel) | none (pure hash) | per-column | **Yes** |
| Caves | **87 s** | sequential | worms ~3 k steps × ~24 vox; megas r≤170 | No (F1) |
| Boulders | 0.15 s | sequential | r ≤ ~17 + cluster ≤ 22 | No (F1) |
| Bushes | 0.03 s | sequential | r ≤ ~10 | No (F1) |
| Trees | 1.18 s | sequential | canopy r ~40–60 | No (F1) |
| Lakes | 0.21 s | none | per-column at `LAKE_LEVEL` | **Yes** |
| Flowers | 0.02 s | sequential | r ≤ 2 | No (F1) |
| Grass | 0.08 s | sequential | r ≤ 4 per 8-vox cell | No (F1) |

Cross-phase **read-back contracts** (must be preserved by stage ordering):
boulders/bushes/trees/flowers/grass require `MAT_GRASS` underfoot (i.e. read
the bulk-fill + lake output); boulders preserve `MAT_WOOD/LEAVES` (read trees);
cave formations probe *carved* geometry (floor/ceiling scans up to 90 voxels);
ivy/mushrooms/moss anchor on carved surfaces.

Global mutable state: `g_caveLUT`, `g_heightMap` (GRID²), `g_lightVol`,
`g_caveEntrances` (lost on cache load — existing bug), `g_megaCaverns` +
`g_caveLights` (persisted in a sidecar).

`StripePool` (bulk fill) demonstrates the determinism pattern we'll reuse:
parallel pure compute into per-column buffers, serial ordered merge for pool
index assignment → bit-identical output regardless of thread count.

### 2.3 Consumers (everything that breaks if we're careless)
- **Shader**: brick DDA + `MAX_OUTER=1024`; fog = `gridSize.x/2`; empty-space
  skip plane at `surfaceY+220`; height texture (GRID²) and light volume
  sampled with world-normalized coords; per-voxel noise hashes **absolute
  float** positions (precision + seam risk at large coords); water refraction
  walks 96 voxels; leaf translucency fires 220-voxel sun rays.
- **Physics**: `overlapsSolid` full-AABB scan, `isOnGround`, `isInWater`,
  `eyeSkyExposure` (9 columns × 220 up). All treat out-of-bounds as **air** —
  in a streamed world, "not generated yet" must NOT read as air or the player
  falls through the planet.
- **Edits**: brush spheres up to r=16 (33³ region), `cleanupOrphans` ±1 sweep —
  can straddle chunk borders.
- **Water sim**: scoped by `wetBricks`, neighbor reads ±1 voxel, hard XZ stop.
- **Spawn**: scans the world-center column top-down — requires generated
  terrain *before* placing the player.
- **Journal**: specimen flood fill (26-connected, ≤100 k voxels, hard bounds);
  **plate "studio" stamps voxels into the live world at Y=3900** — a hack that
  must die in v2.
- **Console**: `/cave`, `/mega` read the worldgen registries; `/tp` anywhere.

---

## 3. Target architecture

### 3.1 World addressing — unbounded XZ, fixed Y
The game is "deeper than wide" by design; Y stays bounded (512 bricks: bedrock
floor to sky ceiling). Streaming is XZ-only. This kills 80 % of the complexity
of fully-3D chunking (no vertical chunk seams in physics, heightmap stays a
per-column concept, full-height cave probes stay intra-chunk).

- **Chunk = full-height column, 64×4096×64 voxels** (8×512×8 bricks).
  - Brickmap slice: 32 k u32 = 128 KB. Typical allocated bricks ≈ 4.4 k
    (~2.2 MB pool) judging by today's density (4.5 M bricks / 1024 chunks).
  - Big enough that most features (bushes, flowers, grass, boulders) touch at
    most 2×2 chunks; small enough for ~10–90 ms generation each.
- `ChunkCoord {int cx, cz}`; `World` becomes
  `unordered_map<ChunkCoord, Chunk>` + accessors. `getVoxel/setVoxel`
  signatures unchanged — callers (physics, water, edits, journal) largely
  untouched.
- Chunk lifecycle: `Missing → Generating → Resident(clean|dirty) → Evicted`.
  Eviction policy: outside residency ring AND not dirty (dirty chunks save
  first). Ungenerated/missing reads return a sentinel **`MAT_UNLOADED`-style
  blocking semantic** for physics (solid), and "skip" semantics for
  cosmetic scans (sky exposure, translucency).

### 3.2 GPU — the toroidal window (the beautiful part)
Keep today's exact GPU layout as a **sliding window**:

- `brickmapSSBO` stays 256×512×256 entries (128 MB). Slot for world brick
  `(bx,by,bz)` = `(bx & 255, by, bz & 255)` — pure power-of-two wrap, no
  remapping table needed in the hot path.
- New uniforms: `ivec2 windowOriginBricks` (and a float voxel version). The
  DDA clips the ray to the window AABB `[origin, origin+256)²` instead of
  `[0, bricksPerAxis)`. Bricks outside = fog/sky (already visually hidden:
  window half-width 1024 voxels ≥ full-fog distance).
- **Brick pool becomes a slot allocator with a free list.** All bricks are
  uniform 128-u32 slots → free list is trivial, zero fragmentation, and
  finally bounds GPU memory (today the pool only ever grows). Chunk upload =
  allocate slots, write bricks, point window brickmap entries; chunk evict =
  push slots back on the free list, zero the entries.
- Window scroll: when the player crosses a chunk boundary, the strip of
  newly-entered chunks uploads into the slots the departing strip vacated.
  Budgeted (N bricks/frame) to avoid hitches; double-buffer the brickmap
  strip writes if needed.
- Height texture and light volume become the same kind of window (height: one
  GRID² R32F window with toroidal wrap + origin uniform; light volume:
  re-tiled per chunk-column at its existing coarse resolution).
- **Float precision + noise seams**: shader keeps working in window-local
  coordinates (0..2048 — precision solved by construction); per-voxel grain
  hashes switch from `hash31(floor(hitPos))` (float) to integer hashing of
  **world-space ivec3** (exact at any distance, no seams ever). Scrolling UVs
  (clouds, caustics) keyed to world-int + fract to avoid drift.

### 3.3 Generation — deterministic feature lattice
This is the core redesign. Three stage classes:

**Stage A — Fields (pure, per-column).** Height, strata, cave-noise band,
lakes. Already coordinate-pure. `g_caveLUT` is replaced by direct hash noise
(it's already hash-based; the LUT is just a world-bounded cache — make it
per-chunk or drop it).

**Stage B — Local features (lattice-seeded, bounded footprint).**
Boulders, bushes, trees, flowers, grass. Pattern:

```
for each feature kind K with cell size S_K and max radius R_K:
  for each lattice cell C overlapping (chunkBounds expanded by R_K):
    rng = PCG(hash(worldSeed, K, C.x, C.z))     // per-cell stream
    enumerate features in C from rng (count, positions, params)
    for each feature F whose footprint intersects this chunk:
      generate F fully, but WRITE only voxels inside this chunk
```

Properties: any chunk can be generated alone; the same feature replayed by
two neighboring chunks produces identical voxels (idempotent clipping); no
chunk-loading cascades (Minecraft's "populate neighbors" problem avoided
entirely). Each feature kind declares `R_K` as a hard contract
(tree ≈ 80, boulder ≈ 40, bush ≈ 12, grass ≈ 6, flower ≈ 3).
Within-chunk stage order = today's global phase order, which preserves every
read-back contract (trees-before-boulders for leaf preservation, etc. — and
because neighbor-cell features are replayed too, a boulder near a border
correctly sees the leaves of a tree whose trunk lives next door).

**Stage C — Network features (global skeleton, local carve).**
Cave worms and mega caverns span hundreds of meters; they can't be
lattice-local. Split them into *skeleton* and *carve*:

- At world open, compute the **worm skeleton** for any region from seed:
  worm spawn cells on a coarse lattice (e.g. 256² voxel cells), each worm's
  full polyline + per-step width/branch/gallery/room decisions derived from
  its own `PCG(hash(seed, WORM, cell, i))`. Pure math, no voxel writes —
  ~3 k steps × ~70 systems ≈ milliseconds, cached in a spatial index
  (segment → chunk buckets).
- Per chunk: query segments intersecting bounds (+max carve radius ≈ 32),
  carve only voxels inside the chunk. Then run formation/crystal/deco/ivy
  placement for *anchor points inside this chunk* — anchors and their RNG are
  derived per worm-step (deterministic), and their probes (floor/ceiling
  scans ≤ 90 vox vertical, ≤ ~8 horizontal) stay inside the full-height chunk
  + small margin, which the carve stage has already made consistent.
- Bonus: `/cave` and `/mega` read the skeleton index — entrance lists become
  seed-derived and **always available**, fixing the current
  "cached load loses cave entrances" bug for free.

**RNG discipline (the contract):** nothing may share a sequential stream
across features. Every decision derives from
`PCG32(hash64(worldSeed, stageId, featureKey...))`. `std::mt19937` per
feature is acceptable but PCG32 is smaller/faster to seed per-feature.
Cross-platform determinism: integer hashing is exact; float noise is
deterministic per-binary (cross-machine determinism is a non-goal for now —
single-player).

**Acceptance check:** the world *will* change once (different RNG layout) —
do it now, before any player-facing release, and validate with a golden-stats
harness (counts per feature, brick totals, species histogram, screenshot
spots) rather than bit-identity.

### 3.4 Biome abstraction — the extension seam
A biome is **data, not code**:

```cpp
struct BiomeDef {
    const char* name;
    // Stage A params
    HeightParams   height;     // octave amps/periods, surface offset
    StrataParams   strata;     // stone band centers/spreads/materials
    SurfaceParams  surface;    // top material, soil depth, underwater bed
    // Stage B lattice densities/params per feature kind
    FeatureParams  trees, bushes, boulders, flowers, grass;
    // Stage C
    CaveParams     caves;      // worm density, radii, deco mix
    // Presentation + content
    PaletteParams  palette;    // grass/leaf tint ranges fed to the shader
    SpeciesRange   species;    // journal registry rows owned by this biome
};
```

- **Climate fields**: two ultra-low-frequency noise fields (temperature,
  moisture) over XZ → biome selection via a small table (Whittaker-style).
- **Blending**: per column, compute weights of the K≤3 nearest biomes in
  climate space; *scalar* params (height amps, densities, tints) lerp
  smoothly — terrain never steps at a biome border. *Discrete* choices
  (which species spawns at a lattice point) use the dominant biome at that
  point. Feature lattices are global; only their densities/params vary.
- **Step one registers exactly one biome** — `verdant_meadows` — holding
  every constant currently hard-coded in `internal.h`/`flora.cpp`/`caves.cpp`.
  With a single biome, blending is the identity function: provably zero
  behavior change. A `/biome debug2` console biome proves the seam.
- **The extension contract** (the deliverable you asked for): adding a biome =
  1 `BiomeDef` + its journal species rows + (optionally) a new feature
  generator implementing the Stage-B interface
  (`enumerate(cell, rng) → features`, `maxRadius()`, `write(chunk, feature)`).
  No storage, streaming, save, or renderer changes. Ever.

### 3.5 Persistence — edits are the only truth
Procgen is deterministic from seed ⇒ **only player edits need saving.**

- Region files (`region_X_Z.dat`, 32×32 chunks each): header + per-chunk
  offset table + zlib-compressed dirty-chunk snapshots (brickmap slice +
  allocated bricks — same encoding as today, just chunk-scoped).
- Clean chunks: regenerate on load (10–90 ms, async — cheaper than disk for
  most chunks; the old 5 s monolithic cache load disappears as a concept).
- `world.dat` + mtime invalidation retire. Worldgen-source changes now
  invalidate by **bumping a gen version stamped into region headers**; stale
  edited chunks surface a "regenerate edited chunks?" decision rather than a
  silent wipe.
- Sidecar (`world.lights`) retires: crystal lights derive from crystal voxels
  actually present in resident chunks (post-edit, automatically correct after
  mining); mega/entrance lists derive from the worm skeleton.
- Journal save is already separate and untouched.

### 3.6 Runtime systems migration map

| System | Change | Risk |
|---|---|---|
| `getVoxel/setVoxel` | route through chunk map; missing chunk ⇒ policy enum (solid for physics, air for cosmetics) | low |
| CPU raycast | same DDA, chunk-window bounds, early-out on missing chunk | low |
| Physics | unchanged code; relies on voxel policy + **movement gate**: don't let the player into a non-resident ring | low |
| Edits/brush | already bounded (r≤16) — clip per chunk, mark chunks dirty, delta upload via window slots | medium |
| `cleanupOrphans` | bounded sweep — works through accessors | low |
| Water sim | `wetBricks` becomes per-chunk set; cross-border flow allowed when neighbor resident, else deferred-pending flag | medium |
| Spawn | async-generate spawn ring first ⇒ **needs the loading screen** (finally) | low |
| Sky exposure / translucency / god rays | missing-chunk = "open/lit" (cosmetic) | low |
| Journal flood fill | residency ring ≥ specimen reach ⇒ always satisfied near player | low |
| **Plate studio** | stop stamping the live world at Y=3900; render plates from a dedicated scratch volume (small SSBO pair + same shader, studio uniforms) | medium |
| Console `/tp` | teleport = move residency center; show loading gate if far | low |
| Height tex / light vol | windowed tiles (3.2) | medium |
| `[slow]`/fps logging, heatmap | unchanged | — |

### 3.7 Performance engineering
- **Workers**: generalize `StripePool` into a `ChunkWorkerPool` (N = cores−2):
  priority queue keyed by (distance to player, movement direction bias).
  Throughput needed at sprint speed (~77 vox/s): ~1.5 chunks/s; at fly speed
  ~5 chunks/s. Budget says one worker suffices; we'll have 8+.
- **Upload budget**: cap GPU uploads (e.g. 4 MB/frame) — scrolling the window
  across a chunk row is ~35 MB worst case, amortized over ~20 frames without
  visible pop (fog covers the far edge).
- **Per-chunk gen cost estimate** (from current totals / 1024 chunks):
  fields ~2 ms, local features ~3 ms, cave carve ~10–85 ms (worm-density
  dependent), deco ~5–20 ms → worst ~110 ms, typical ~25 ms. Async, so this
  is throughput not latency.
- **Instrumentation to build in from day one**: per-chunk phase timers
  (reuse the `[worldgen]` pattern), gen-queue depth gauge, upload-budget
  saturation counter, resident-chunk/brick-slot gauges in the `[fps]` line.

### 3.8 Immediate optimizations (valuable even before chunks)
1. **Per-feature RNG (Step 1 below) ⇒ parallel cave systems** via StripePool:
   87 s → ~10 s cold regen now.
2. `carveSlice` writes via direct brick access (the bulk-fill fast path)
   instead of `setVoxel`+dummy `EditDelta` — saves a bounds-check + hash-set
   insert per voxel on the hottest loop in the codebase.
3. Hoist the per-voxel `smoothNoise3` wall-wobble in `carveSlice` to
   per-(step, dx) where the math allows — it's sampled per voxel today.
4. Brick-pool free list (3.2) is worth doing early: digging currently grows
   GPU memory forever.

---

## 4. Migration sequence (each step ships, game stays playable)

| Step | What | Validates with | Unlock |
|---|---|---|---|
| 0 | Golden-stats harness: per-phase feature counts, brick totals, timing JSON; 4–5 fixed-camera screenshot spots | manual diff | safety net for everything below |
| 1 | **Per-feature RNG** (lattice seeding for flora; skeleton for caves) — world relayouts once | stats sanity + visual pass | parallel caves (87 s→~10 s); the chunk prerequisite |
| 2 | Worldgen context object: kill globals, thread `worldSeed`, replace caveLUT with pure noise; `generate(World&, Bounds)` API (called once with full bounds for now) | bit-identical to step 1 | testable, bounded gen |
| 3 | Feature-lattice + chunk-clipped replay internally (still generating the full fixed world chunk-by-chunk through the new path) | stats identical to step 2 | proves chunk isolation |
| 4 | CPU chunk storage: `World` = chunk map; per-chunk dirty saves (region files); retire `world.dat` | gameplay regression pass | persistence model |
| 5 | GPU toroidal window + pool free list (window == world size ⇒ zero visual diff by construction) | screenshot diff ≈ 0 | bounded GPU memory |
| 6 | **Unbound XZ**: async workers, residency ring, movement gate, spawn loading screen, window scrolling | walk/fly to former world edge and keep going | the actual big world |
| 7 | Biome registry: extract `verdant_meadows` BiomeDef, climate fields + blending (identity with one biome); `/biome` debug second biome | stats identical; debug biome visibly different | the extension seam |
| 8 | Cleanup: studio scratch buffer, lights-from-voxels, `/cave`-from-skeleton, heightTex/lightVol windows | feature parity checklist | retire all v1 hacks |

Steps 1–2 are pure wins even if the project stops there (10× faster cold
regen, testable worldgen). Step 5 is intentionally a no-visual-diff refactor.
Step 6 is the only step where new player-visible behavior appears, and it
arrives on infrastructure that's already proven by 3–5.

---

## 5. Risks & open questions

1. **World relayout at Step 1** — unavoidable; accept now, before content
   accretes. Re-tune feature densities visually afterward (they were tuned
   against the old RNG stream).
2. **Formation probes near chunk borders** — probes are vertical-dominant
   (full-height chunks make these safe) but horizontal disc placement near a
   border must replay neighbor carves first; covered by carve-before-decorate
   stage order within the chunk + R-margin replay. Needs a dedicated test
   (worm crossing a chunk corner with formations on the seam).
3. **Water across borders** — deferred-pending flags add state; lakes are
   static so scope is player-placed water only. Acceptable to ship Step 6
   with water sim paused at non-resident borders.
4. **Upload hitches while flying fast** — budgeted uploads + fog cover;
   worst case the horizon strip pops in late (already fog-obscured).
5. **Edited-chunk gen-version drift** — worldgen changes after players have
   edits: regenerate-clean vs keep-stale policy. Proposal: keep edited chunks
   as-is (they're the player's), regenerate everything else; surface a
   console command to force-regen a chunk.
6. **`MAX_OUTER=1024`** stays correct (window diagonal ≈ 700 bricks) but
   should become `windowBricks.x * 3` derived, not a literal.
7. **Save-format versioning** from day one: region header carries
   `formatVersion` + `genVersion` + `worldSeed`; refuse nothing, migrate or
   regen explicitly.

---

## 6. Acceptance criteria for "done"

- Cold start to playable: < 3 s (spawn ring only), full ring fills in
  background; no monolithic regen exists.
- Walk/sprint/fly in any direction indefinitely: no hitch > 8 ms attributable
  to streaming (measured via the existing `[slow]` frame log).
- Dig/place/water/journal/console all work identically in any chunk,
  including across chunk borders (test matrix).
- Adding a test biome touches only: one `BiomeDef`, species rows, zero
  storage/render/save code.
- GPU memory bounded and stable over a 30-minute fly-around (pool free list
  gauge flat).
- All worldgen reproducible from `worldSeed` alone; same seed ⇒ same world,
  chunk generation order irrelevant (tested by generating a region in two
  different orders and diffing).

---

## 7. Step 6 execution plan (streaming, decomposed)

Step 6 ships as six sub-steps, each leaving the game playable. The
correctness half (6b–6c) is mandatory for any unbounded world; the quality
half (6d–6f) turns "correct" into "smooth".

| Sub | What | Validates with |
|---|---|---|
| 6a | DONE (776c54f): unbounded world coords, CPU toroidal window, frontier gate, window-correct water/edits | contentHash unchanged |
| 6b | PART 1 DONE: worm-reach CONTRACT (tether 1280 + slack, hard stop; `WORM_REACH`=1898). Measured reach 1988 (border-censored) -> 1456 in-contract; flora bit-identical; chunked hash equality re-proven post-tether. Remaining: the region-query API itself, built when 6e consumes it | chunked-driver contentHash vs monolithic, both seeds |
| 6c | DONE: ivy wander capped (`IVY_WANDER`=64), decoration replays per chunk (`CAVE_DECO_MARGIN`=96), stats/lights anchor-gated. Visual pass approved. (The initial live-read writer showed ~0.015% chunk-order divergence — superseded by 6d's buffered writer.) | golden stats ±small, visual pass |
| 6d | DONE — ORDER-FREE + PARALLEL GENERATION. Part 1: decoration computes into per-chunk buffers against the carve-only world and commits after all chunks (the 6e re-entry rule, shipped early). Part 2: EVERY stage is buffered (`ChunkStageBuffer`, unified `StageWriter` replacing GenWriter) and computed on a persistent `ChunkWorkerPool` (hw_concurrency−2 workers): bulk/carve/flora run as stripes (parallel compute, serial in-order commit — bounds peak buffer memory, keeps pool allocation deterministic); decoration stays batch-all (carve-only-world invariant). Buffers store per-BRICK pages (512B + written bitmask), NOT per-voxel map nodes — the first cut buffered 226M voxel nodes (~9 GB of heap nodes, allocator-lock contention ate the parallelism: 70 s); pages + a one-entry page cache + brick-aligned whole-page commit rejection brought chunked gen to **8.2 s fwd / 8.9 s rev vs 5.9 s monolithic** on 30 workers. PROVEN with the final binary: monolithic == chunked-forward == chunked-REVERSE contentHash `b251dd3747689ad3` (`--chunked-gen` / `=2`; REVERSE confirmed via the driver log line — an earlier check raced a stale process, so always verify the order line in the same log session). `[fps]` line now carries pool gauges (live/high-water/free bricks). Residency-state map deferred to 6e where eviction gives it a consumer | fwd/rev/monolithic hash equality ✓ |
| 6e | Window scrolling, decomposed below (6e-1..6e-4). **Design (decided at 6e start):** (1) FRONTIER first-gen runs as a strip LADDER — per-stage compute fronts separated by their replay margins, each strip running the same buffered stages as the batch driver. Flora is exact under any commit timing because decisions read pre-stage snapshots and live reads are per-voxel at write positions (non-cascading; out-of-bounds reads only affect clipped writes) — this is what fwd/rev hash equality already proves. Two mechanisms make the ladder exact: per-stage SNAPSHOT FRONTS capture each strip's gate band 2 strips before that stage's compute front arrives (so captures never see same-stage commits), and an UNDO BAND for decoration — commits record overwritten values (~6 MB/strip), and decor's cascading probes (floor/ceiling searches that read carve-only state) route through this overlay for the ~4 strips behind the decor front, reconstructing carve-only state exactly. (2) RE-ENTRY loads evicted chunks from a compressed DISK CHUNK CACHE (miniz) — reconstructing intermediate stage states for arbitrary old chunks would need margin recompute of bulk+carve (~50 core-ms/column, infeasible at sprint speed), and the journal game demands byte-identical re-entry. This amends "saves scale with edits" to "saves scale with explored area"; the edits region file remains the durable record, the chunk cache is canonical-but-regenerable (a deleted cache costs a re-walk of the frontier ladder from saved edits). | per sub-step below |
| 6e-1 | DONE — offline LADDER driver (`--chunked-gen=3`): full-world generation strip-by-strip through the moving stage pipeline (fronts: bulk t, carve t-2, decor t-7, flora t-11/15/19/27/31 with lakes t-23; snapshot capture fronts 2 strips ahead of each flora front), exactly the order streaming will use. Built empirically: the naive ladder (snapshot fronts only, no undo band) diverged ONLY in decoration — every flora stat identical, ivyBuilt off by 2 in 31k, 128 bricks of 2.56M — isolating the cascade-read gap precisely as predicted; adding the undo band (decor commits record overwritten values, decor reads route buffer→band→world, strips evict 5 behind the front) made it EXACT: contentHash `b251dd3747689ad3` == monolithic == all other drivers, 10.7 s. Streaming-order generation is proven byte-identical | the 4th leg of the hash gate ✓ |
| 6e-2 | DONE — eviction + session chunk cache + reload, world still fixed 2048. `ChunkCache` (worldio) stores evicted chunks compressed in the edits-file blob format (~55 KB/chunk measured); `evictChunk` frees mixed bricks / blanks brickmap / drops wet slots, `reloadChunk` restores byte-identically (waking water ONLY for edited chunks — static lakes stay out of the sim) and reuses freed slots. A dirty chunk evicted mid-session serves its blob from the cache if a save fires (`saveWorldEdits` gained the cache param). The cache is SESSION-SCOPED by design: ladder determinism means a relaunch regenerates identical content, so persistence stays edits-only. In-game: dev `/stream <R>` residency ring (1 evict + 1 reload per frame, hysteresis at R+1; `/stream off` restores all), `[stream]` gauge line; `uploadDelta` now coalesces dirty indices into contiguous runs (chunk deltas = a few calls, not 32K). GATE: `--evict-check=128` round-trips 128 spread chunks pre-upload — 328K bricks freed and FULLY reused (brickCount AND brickHighWater identical), per-chunk + global hashes identical, PASS. Lights/lightVol deliberately untouched (fixed world keeps them valid; lifecycle lands in 6e-3 with the moving window) | `--evict-check` PASS ✓; hitch measurement in the 6e-3 play pass |
| 6e-3a | DONE — UNBOUNDED WORLDGEN. Every stage generates correctly for bounds anywhere in Z^2: flora lattices derive cell ranges from bounds via floorDiv (anchors at 16+k*CELL for any integer k); all world-border guards/clamps removed (XZ guards became y-only; lakes guard via voxelInWindow); buffer/undo page keys are world-coord int64 (21-bit biased XZ); SurfaceSnapshot is window-origin-relative. The CAVE PLAN is an on-demand lattice: 256-cell systems walk lazily and cache (walkCaveCell), 2048-mega-cells resolve their proposals locally (cap 18 + spacing per mega cell — the old global cap, localized); ensureCaveRegion(bounds) pre-walks serially, carve/build GATHER read-only in lattice order. NEW GATE: `--overlap-check` regenerates the window at origin (8,8) and compares the shared region (shrunk 4 chunks from both windows' edges — EDGE CHUNKS ARE PROVISIONAL by contract, their probes read air beyond the window; 6e-3b must regenerate the rim before finalizing). DENSITY: the old fixed world was border-starved (~62% of unbounded coverage everywhere — the whole window sat within reach of missing outside systems); per-cell density is scaled by 0.62 so the unbounded world reproduces the play-tested interior feel (bricks 2.56M->3.34M, the border band now correctly dense). CANONICAL SEMANTICS = the buffered chunked driver (now the game DEFAULT; monolithic via --chunked-gen=0 diverges ~0.01% in decor cascades — its write-through probes see earlier requests beyond the replay margin; counts identical). Golden hash relocked: fwd == rev == ladder == `17336406923f8c88`; overlap `fa0cbc15bf3d0e42` both origins; evict-check PASS | overlap-check ✓, 3-driver equality ✓, evict-check ✓ |
| 6e-3b | DONE — THE WINDOW MOVES; the world is functionally infinite. `ChunkStreamer` = the chunk-status ladder as a runtime scheduler: per-chunk stage map; a chunk advances to stage s only when every chunk within that stage's dependency radius (decor 4, trees 3, boulders 2, bush/flower/grass 1) is at >= s-1; out-of-window neighbours count as NONE, so the leading rim is a naturally graded pipeline that finalizes only once its margin neighbourhood exists — direction changes and diagonals work by construction. Gates read LIVE world (snapshot "live mode": consistent within a parallel batch; seam-class across batches, accepted — offline drivers keep captured snapshots as the oracle). Scroll: origin follows the player (LEAD 14); leaving FINAL chunks serialize RAW into the cache (compression amortized 2/frame via compactChunkCache — never 32 zlib passes in one frame), partial frontier chunks DROP (never cache); entering strips reload from cache or enter the ladder. GPU sync via budgeted per-chunk uploads (World::uploadChunk; bulk/carve/final milestones; leaving strips need NO GPU work — out-of-window bricks are never sampled; entering-edge staleness tolerated a few frames). heightTex wraps GL_REPEAT = the toroidal mapping; scrolls overwrite the entering strip's texels (updateHeightRect; CPU heightMap synced). Player gated out of sub-BULK chunks (revert-to-last-safe). Perf fix landed: carve ops iterate in place with z-sorted binary search per cell (gather pointer-pushes were 40 MB/chunk of traffic; 16 s -> 12 s batch gen). WALK TEST (--walk-test): 60 s forced march east = x 1024 -> 2757 (709 vox past the old world edge), 26 scrolls, 832 chunks cached, generation kept full walking pace, 1 slow frame (startup). KNOWN v1 [historical — all RESOLVED in later sub-steps; do not re-chase]: cut-side trees at the start/cached boundary → `6e-3e`; new-territory crystals without lightVol glow → `6e-3d`; mega-cell entry stall → incremental cell walks `6e-3c`; journal studio off-centre → `6e-4`. (z-edge corridor bands self-heal on approach by design.) | walk test ✓; play pass pending |
| 6e-3c | MOSTLY DONE — visual seamlessness + sprint-smooth streaming. **REVEAL CURTAIN** (the Minecraft fog-curtain trick, done for this game's painterly haze): terrain fog now reaches FULL sky opacity (was capped at 0.6 — the real reason the frontier was always ~40% visible) at a DYNAMIC distance = the nearest unfinished chunk (`ChunkStreamer::nearestUnrevealedDist`), so the streaming frontier materializes BEHIND the haze; opens back out to the window extent when resident (stationary), pulls in to hide the frontier when moving. Asymmetric smoothing (snap IN fast, ease OUT slow) + low floor (~19m) so outrunning shows fog, never bare terrain. `revealDist` uniform; the studio plate sets it to infinity. **SPRINT THROUGHPUT** (real sprint = WALK_SPEED*SPRINT_MULT = 62 vox/s, 2x the old walk-test): BATCH 16->32, time-budgeted commits (3 ms/frame, not a fixed count — density-agnostic), cave-region walk made INCREMENTAL + parallel-budgeted (12 cells/frame, gated so carve/decor wait rather than hitch on a 300-cell burst). Frontier holds 70m ahead at full sprint — curtain never exposes streaming. **HITCH KILLERS**: cache compression moved to a BACKGROUND THREAD (`compressOneFromCache`; was 30-44 ms/frame on the main thread — the dominant hitch); only EDITED chunks are cached on scroll (unedited regenerate byte-identical, the overlap-gate guarantee — eviction went from 32 extracts/scroll to ~0); GPU brick pool grows via GPU-side copy (`ensurePoolGPU`) not CPU re-upload. Brickmap is CHUNK-CONTIGUOUS (one buffer call per chunk sync, not 4096). Spike profile at FULL sprint through densest caves: 212/50s @ 80-120ms -> ~15/50s @ ~30ms; normal play ~clean. FPS readout in window title (`--walk-test` harness drives it). KNOWN remaining: ~30ms when crossing a single deep cave-system cell (needs resumable worm) — STILL OPEN (parked). [RESOLVED since: ~800ms GPU-pool grow → preallocated `bc5ee2c`; cut-trees seam → `6e-3e`; no lightVol in new land → `6e-3d`; journal studio off-centre → `6e-4`.] Generation determinism intact (overlap `fa0cbc15bf3d0e42`) | overlap ✓; sprint smooth; user playtest ✓ |
| 6e-4 | DONE — journal studio scratch volume. The specimen-plate studio stamped at absolute world-centre (1024,1024, y=3900), which falls outside the resident window once the origin scrolls — so sketching was broken everywhere except spawn. Fix: place the studio at the WINDOW-RELATIVE centre (`originVoxX()+GRID_XZ/2`). The y=3900 empty-sky layer exists across the whole window and the window centre is always FINAL (the player stands there), so that sky IS the scratch volume — no separate SSBO needed. Identical to the old code at spawn (origin 0 → world-centre), so zero regression there. VERIFIED: `--walk-test` sketches at the far end (origin ~37 chunks out) and the plate renders 87% non-background (a blank plate = studio missed the window) | sketch-after-scroll plate non-blank ✓ |
| 6e-3d | DONE — lightVol windowing (cave glow follows the window). The emissive light volume was a fixed texture covering world [0,2048) only, never rebuilt as the window scrolled — so crystals far from spawn had no baked glow. Now it's a TOROIDAL WINDOW like heightTex: `buildLightVolume(ctx, ox, oz)` splats only in-window lights at wrapping X/Z cell indices; lightVolTex wraps GL_REPEAT on S(X)/R(Z), clamp T(Y); the shader's `worldPos/gridSize` sample is unchanged (the wrap does the toroid). Rebuilt + re-uploaded when the origin scrolls or new crystals stream in (throttled ~5/s; out-of-window lights pruned so caveLights stays window-bounded, re-derived on re-entry). recenter rebuilds it origin-aware. Edge caveat: a light within ~80 vox of the window edge casts a faint wrap-tail on the opposite edge, but that edge is beyond the reveal curtain. VERIFIED: `--tp-test` at (60000,60000) shows 4631 in-window lights + populated lightVol (was 0 far from spawn before). NOTE: a follow-on cave-DENSITY experiment (per-cell rate ×3.2 to match main's 70/window, since the lattice loses ~69% of attempts to flat cells with no backfill) was REVERTED — it pocked the surface and blew gen to 54s; caves stay at the committed ×0.62 (334 systems/region, hash 17336406923f8c88). Mega-reachability/connectivity vs main remains a known, accepted gap | `--tp-test` glow populated ✓ |
| 6e-3e | DONE — frozen surface snapshot ends boundary-cut trees/rocks. Streamed flora gated against the LIVE world: the first chunk to commit a tree/boulder overwrote the grass tile its own gate reads, so neighbour chunks re-deriving the same feature failed the gate and skipped their slice → clean vertical cut. Fix: `SurfaceSnapshot` is now TOROIDAL (x&mask, follows the window like heightTex); the streamer captures each chunk's surface band ONCE at carve-completion (frozen — flora never mutates it); flora deps bumped to minStage CARVE so a chunk's margin-neighbours are captured before it gates a feature anchored there. The gate is now a pure function of the frozen snapshot, so every chunk replaying a boundary-spanning feature reaches the identical accept/reject — partial (cut) features impossible by construction, same as the offline drivers. Offline untouched (origin-0 toroidal == old linear; overlap-check fa0cbc15bf3d0e42). Trade-off: single post-carve snapshot (vs offline's per-stage), so a tree may RARELY overlap a boulder — far less jarring than cuts; a 2nd snapshot layer closes it if needed | overlap-check ✓; trees whole at boundaries (visual) |
| 6f | DONE — loading gate + teleport recenter, the formal end of step 6. SPAWN GATE: `drawLoadingVeil` paints a calm sky-wash + "generating world…" title before the ~12 s startup gen, so launch isn't a black/frozen window (spawn itself was already fully generated before the first real frame, so the player was never released into unfinished terrain — this is the UX half). TELEPORT RE-GATE: `/tp` (and `/tp spawn`) to a destination OUTSIDE the resident window now calls `recenterWindow` — drains the async streamer (`ChunkStreamer::reset` waits for the pool, clears flight/stages), flushes edits to disk, snaps the window origin to centre the destination, regenerates it (origin-aware `generateTerrainChunked`), re-uploads brickmap/pool + an origin-aware `buildHeightmap` (toroidal-indexed) + lightVol, re-seeds FINAL, reloads edits — all behind the veil. Lifts the per-frame window clamp's effect (far `/tp` was previously a silent no-op). In-window teleports (`/cave`, `/mega`, nearby `/tp`) are unchanged. GATE: `--tp-test` recenters to (60000,60000) then (-40000,12000) then back — every landing is FINAL with real ground, and the round-trip reproduces the identical contentHash (33f4fc4bbf327f66), proving the recenter is clean + deterministic incl. negative coords ~6 km out. Spawn contentHash unchanged (17336406923f8c88). Sprint acceptance met in 6e-3c. KNOWN: teleport blocks ~6-12 s (loading freeze behind one veil frame; threaded-gen animated loading is the polish) — STILL OPEN (parked). [new-region cave glow: RESOLVED in `6e-3d` (lightVol windowing).] | `--tp-test` PASS ✓; spawn hash ✓; sprint smooth ✓ |

## Open items (current — June 13 2026)

**STEP 6 / streaming epic is COMPLETE and merged into normal play.** The per-row "KNOWN" notes above are HISTORICAL — most were resolved by a later sub-step (annotated inline). The only items still genuinely open:

- **~30–64 ms hitch crossing a deep cave-system cell** — needs resumable / async worm walking (the cell-walk budget hides most of it; a single dense cell can still spike). Parked.
- **Teleport ~6–12 s freeze** behind one veil frame — needs threaded gen with an animated loading screen. Parked.

Everything else once listed as KNOWN is fixed: GPU-pool grow (preallocated, `bc5ee2c`), boundary-cut trees (`6e-3e` frozen snapshot), cave glow in new land (`6e-3d` lightVol windowing), journal studio off-centre (`6e-4`), the reveal-curtain fog pulse + frontier flicker + dark-cave bleed-through (the June 13 stream fixes), and stale-save terrain seams (`GEN_VERSION` discard, `b02d93f`).

**Post-epic work landed since:** regional terrain character — hills/valleys/flat meadow in `heightAt` (`ba00cfd`); journal far-from-spawn crash (toroidal heightMap index + window-relative flood fill, `e01773d`).

**Next direction:** biomes (`biomeAt(x,z)` threaded through height/stone/flora/caves), with the terrain control field already a stepping stone; then fauna. LOD clipmap is the eventual render-distance unlock (deferred).
