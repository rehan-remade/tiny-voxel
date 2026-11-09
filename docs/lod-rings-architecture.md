# Far-Field LOD: Voxel Clipmap Rings — Design (v1, amended)

Status: APPROVED FOR BUILD (adversarial review verdict: BUILD WITH AMENDMENTS).
This document is the source of truth for the render-distance LOD work.
Produced 2026-07-01 by a 6-agent research/design/critique workflow
(Lay of the Land research, Distant Horizons source reading, engine audit,
worldgen audit, architect, adversarial critic). The amendments in Part II
are NORMATIVE — they override Part I where they conflict.

---

# PART I — ARCHITECTURE (architect's design)

# tiny-voxel Far-Field Architecture — "Voxel Clipmap Rings" (v1)

Design verdict up front: **concentric toroidal brickmap rings, directly generated from the pure worldgen core, raymarched by the existing DDA with a per-ring t-handoff, shaded analytically, crossfaded by stochastic switch-radius dither under TAA.** This is Report 2's Representation A. Representation B (multi-run RLE columns) is explicitly rejected as the primary mechanism: it cannot give stepped mesa cliffs/overhangs, it introduces a second trace representation, and the audit shows the brickmap machinery (toroidal scroll, chunk-contiguous upload, coarse mip) transfers to rings nearly verbatim. LotL/DH mesh their LODs; we do not mesh anything — the engine is a raymarcher and the raymarch analog of their design is exactly this ring scheme (VXGI/UE-GDF/Tomorrow-Children precedent).

---

## 1. ARCHITECTURE

### 1.1 Ring set

4 rings + the fine window (treated uniformly as "ring 0"). Every ring keeps the fine texel budget in XZ (`BRICKS_XZ=256` bricks) while its voxel doubles, so extent doubles per ring. Each ring **fully covers its disc** (a clipmap level, not an annulus) — full overlap of all finer rings, per prior-art gotcha #1. Vertical: rings share **Y base 0** (mandatory: `gbuffer.frag`'s `worldTop = surfaceY+500` and all Y math assume absolute Y) and cover the full 4096-world-voxel height via `BRICKS_Y >> N` — full-height coverage is **not** wasteful because brickmap Y shrinks 2×/ring automatically, and above-terrain bricks are `BM_EMPTY` (no pool cost). A terrain-band Y-origin would save <60 MiB total and break the shared Y=0 assumption; rejected.

| Ring | voxel | XZ extent (world vox / m) | brick dims (X×Z×Y) | brickmap | coarse SSBO | pool budget (bricks / MiB) |
|---|---|---|---|---|---|---|
| 0 (fine) | 0.1 m | 2048 / 205 m | 256×256×512 | 128 MiB | 2 MiB | 8.0 M / 3906 |
| 1 | 0.2 m | 4096 / 410 m | 256×256×256 | 64 MiB | 1 MiB | 0.5 M / 244 |
| 2 | 0.4 m | 8192 / 819 m | 256×256×128 | 32 MiB | 0.5 MiB | 0.5 M / 244 |
| 3 | 0.8 m | 16384 / 1.64 km | 256×256×64 | 16 MiB | 0.25 MiB | 0.5 M / 244 |
| 4 | 1.6 m | 32768 / 3.28 km | 256×256×32 | 8 MiB | 0.13 MiB | 0.5 M / 244 |
| **Total** | | horizon radius ≈ 1.6 km marched | 62 M entries | **248 MiB** | ~4 MiB | 10.0 M / **4.88 GiB** |

Ring pool estimate: ~65 k brick columns/ring × 1–3 mixed surface bricks + ~800 tree impostors × ~30 bricks ≈ 160–300 k bricks; 0.5 M/ring reserve is 2× headroom. **Grand VRAM total ≈ 5.1 GiB** vs 24 GiB — trivially fits; CPU mirrors the pool, so budget +1 GiB system RAM. If the user later wants more, ring 5 (3.2 m voxels, 6.5 km) is a table entry, not new architecture.

At ring 4's 1.6 m voxels, a forest tree (trunk 2.2–3.5 m, canopy r 1.2–3 m) is a 2–4-voxel blob — still a discrete silhouetted lump, which is exactly the LotL far look (their far detail also degrades to blobs; footage shows hundreds of meters, not alpine km).

### 1.2 GPU structure (the audit's recommendation, adopted verbatim)

- **One brickpool SSBO (binding 1) shared by all rings.** Brickmap entries store absolute pool indices, so `sampleVoxel` needs zero changes. Single allocator; reserve computed up front as `poolReserveBricks() + 4×RING_POOL_RESERVE` so the no-realloc-mid-gen invariant holds.
- **All ring brickmaps concatenated into the binding-0 SSBO**; likewise coarse mips into binding 2. A `RingTable` UBO (binding 1 — UBO 0 is Materials) holds per-ring `{bmBase, coarseBase, scaleLog2, bricksPerAxis (ivec3), originB (ivec2)}`, fine window included as entry 0. This avoids arrays-of-SSBO-blocks non-uniform indexing UB entirely — ring selection is pure offset math.
- Each ring keeps the **chunk-contiguous slot layout** (`slotOfBrick` mirror) so `uploadChunk`/`gpuBlankChunk`/`refreshCoarseForChunk` work per ring with only a slot-base parameter.
- **Marching trick (minimal diff):** `marchRing(r, ro, rd, tStart, tEnd)` divides `ro` by `scale = 1<<scaleLog2` and runs the existing DDA verbatim in ring units; `t_world = t_ring × scale` (rd unchanged, t scales linearly). `windowLoV`/`bmSlot`/`brickInWindow`/coarse-cell wrap take the ring's `bricksPerAxis`/`originB`/bases as parameters.

### 1.3 Data flow

CPU worldgen workers (same pool as fine gen) → per-ring-chunk fill via the **pure core only** (`internal.h`) → ring chunk buffers → the existing budgeted upload path (`streamer.drainUploads` budget, `uploadChunk` template with ring slot base: pool `SubData` first, then brickmap `SubData`, then coarse refresh) → RingTable UBO update on ring recenter. No downsampling pipeline exists anywhere: **coarse data is generated directly from procgen at stride 2^N** (the LotL neighbor-free pattern, confirmed feasible by the worldgen audit). This is also what makes requirement 4 free: any future biome added to `heightAtEx`/`surfaceInfoAt`/pure flora lattices appears in all rings with zero per-biome LOD work — that contract gets a comment block in `internal.h`.

---

## 2. TRACE INTEGRATION

### 2.1 Primary ray cascade (gbuffer.frag)

Restructure the primary march into a ring cascade with **fixed switch distances** (camera-relative t, in world voxels):

```
D0 = 768   (76.8 m)   fine → ring1     // 12 chunks: inside LEAD=14 scroll margin minus 2-chunk frontier band
D1 = 1536  (153.6 m)  ring1 → ring2
D2 = 3072              ring2 → ring3
D3 = 6144              ring3 → ring4
D4 = 12288 (1.23 km)   ring4 → horizon heightfield → sky
```

`for (r = 0..4): hit = marchRing(r, ro, rd, tPrev, min(D_r, tExit_r)); if hit → shade; tPrev = t`. Doubling D each ring holds the handoff footprint constant at ~3.2 px/voxel (1080p, 90° FOV), degrading to ~1.6 px at each ring's far end — chunky but TAA-stable. Each D_r sits strictly inside its ring's extent (D1=1536 vs R1 half-extent 2048, etc.), so valid data always exists on both sides of every switch (gotcha #1). Handoff is one t-continuation: snap into ring r at `t + ε·scale_r`, never restart from camera.

**Two extra rules:**
- **Fall-through self-healing:** if `marchRing(r)` *misses* before `D_r` (blanked/unfinished/scrolling chunk), the loop simply continues into ring r+1 from the current t. Unfinished fine frontier chunks therefore show ring-1 terrain instead of holes or fog — the artifact becomes a res-pop on upload (the acceptable atomic-reveal behavior the audit predicted), and ring scroll blanking degrades to the next coarser ring instead of black terrain.
- **Seam quantization — floor, not ceil, for terrain:** ring surface height = `floor(heightAt/scale)` ring voxels (ring terrain never pokes above fine terrain). Any-solid/ceil (the VXGI-style conservative rule) would erect a raised rim wall around the player at D0; floor gives at worst a ≤1-coarse-voxel dip that reads as terrain, never a hole (connected surface below). **Flora impostors get the opposite rule** — stamp fat (any-overlap) — because trees must never thin out or vanish. This resolves Report 2's gotcha-#3 conflict explicitly: conservative-occupancy is for downsampled volumes with leaks; directly-generated terrain wants floor.

G-buffer: pack the ring index into `gPosMat.w` as `mat + 256·ring` (audit's suggestion). fp32 world-pos precision at ring-4 extents is ~0.01 voxel — fine.

**Primary coarse-mip skip is mandatory at C1:** rays now travel up to 12 k world voxels; port `anyHit`'s coarse-cell fast-forward (voxel_trace.glsl:82-110 pattern) into the primary march, applied per ring against that ring's coarse mip. Iteration budget: keep 256 outer steps *per ring segment* (a ring segment is ≤ its extent, and each ring's voxels are scaled, so step counts stay bounded).

### 2.2 Secondary rays: none for ring hits

Adopt the audit's verdict wholesale. Ring hits (`ring > 0` flag) get **analytic shading through the existing `shadeHit`** with neutralized window-anchored inputs: `ao = 1`, `shadow = 1` (analytic N·L only), skip lightVol, skip heightTex burial, skip pointSkyVis and water-neighbor `getVoxelAt` (all of these wrap toroidally and return other-side-of-window garbage for out-of-window positions — the #1 source of far-field speckle if missed). `ao.frag` and `reflect.frag` early-out on the ring flag exactly like their existing sky early-outs, keeping their cost flat. Per-voxel face shading (sun tint, day-night, desert warm/cool ambient, dust haze) comes for free through `shadeHit` — this alone kills the "flat 90s lighting" complaint, because distant tree blobs and mesa steps get real per-face N·L that the heightfield could never produce.

Deferred to C3 (optional): one sun `anyHit` per ring hit against the *hit ring's own* coarse SSBO (64²×≤64 cells — cheap) for km-scale cast shadows at low sun. Fine-window sun rays continuing into rings (distant mesa shadowing near terrain): future work, not v1. Never let a secondary ray sample a *coarser* ring than the hit's own (UE-GDF over-occlusion gotcha #4).

### 2.3 Transition band + TAA

**Stochastic switch-radius dither:** per-pixel, offset each D_r by ±1 ring-brick length (8·scale voxels ≈ ±1.6 m at D0) using interleaved-gradient noise. Both representations are opaque; TAA integrates the dither into a smooth DH-style opaque↔opaque crossfade over a ~3 m band. This is chosen **over** DH's literal two-image crossfade (would require a second full G-buffer+lighting pass in our deferred pipeline) and over dual-march blending (2× march cost in the band) — same look, near-zero cost. Ring hits write correct world positions, so TAA motion vectors/reprojection are valid; the res-pop when a fine chunk lands over ring terrain is a one-frame disocclusion the existing neighborhood clamp absorbs.

---

## 3. GENERATION

### 3.1 Factoring (per the worldgen audit, adopted as written)

All in `internal.h` (or `worldgen/surface.h` included by both paths), CPU-side, same TU set / FP flags:

1. `surfaceInfoAt(x,z) -> {h, topMat, grassTop, DesertFields}` — extract fillColumn's material grammar (worldgen.cpp:84-135) verbatim; also kills `packSurfaceColor`'s duplicated rockAmt drift hazard.
2. `deriveTreeSite(seed,kx,kz)` — consumes **exactly draws 1..6** of the SEED_TREE stream (chance, xjit, zjit, sRoll, trunkLen, trunkThick); `genTrees` continues `grow()` from the same rng. Same split for `deriveSaguaroSite`, `derivePalmSites`.
3. `grassTopAt(x,z)` as the pure stand-in for the snapshot gate **ring-side only**; fine gen keeps the snapshot gate (~99% agreement; residual divergence mis-places the odd impostor >100 m out — accepted).
4. **Gate:** the refactor must be `world.contentHash()`-identical + golden-stats-identical before any ring code merges. This is the project's determinism firewall (RNG draw order = API contract; a slip = GEN_VERSION bump and discarded saves).

### 3.2 Ring fill

Per ring texel column (world coords = ring origin + i·scale, sampling `heightAt` at the **same int fine-grid coordinate** convention — column min corner — never re-derived from `baseHAt`): 1 `heightAtEx` + grammar → `BM_SOLID` stone below, one mixed surface brick with `topMat`/dirt band, floor-quantized. Trees: enumerate the 55-vox TREE_CELL lattice over the ring extent, `deriveTreeSite`, stamp impostor (1–2-voxel trunk column + canopy ellipsoid from the shared scale/trunkLen draws, leaf material, fat-stamped). Palms + oasis water (`h < oasisAt().waterY`), mesas, dunes, benches: free inside `heightAt`/grammar. Saguaro in rings 1–2 only. **Skip:** caves (the 12.4 s / 79% dominator — underground, invisible; hillside mouths are 3.6 m dark spots at >100 m, optionally decal-able later from the pure site scan), grass/flowers/creosote/ocotillo/boulders (sub-resolution *and* the O(area) hash costs), lakes (dead code).

### 3.3 Cost budget (audit arithmetic, adopted)

Baseline fine window: 15.65 s CPU (caves 12.36 s). Per ring (2048² texels): terrain ≤0.4 s typical, ~1 s desert-worst (mesaAt 3×3 + oasis rims); tree enumeration tens of ms (ring 4: 355 k cells × hash+noise); impostor writes ~120 k voxels — negligible. **All 4 rings ≈ 1.5–3 s, embarrassingly parallel, zero brickpool contention with fine gen** (own reserve range). Vs 86 s for enlarging the fine window — rings do not multiply the regen problem; they sidestep it by skipping both dominators (caves, per-voxel bulk).

**Scheduling:** at startup/teleport, generate rings **coarsest-first (4→3→2→1)** on the worldgen pool after fine-window tasks drain, behind the existing animated loading screen; reveal does not block on rings (sky beyond the window for the first ~2 s, then terrain pops in coarse-to-fine — Avoyd/LotL pattern). **Recenter:** ring N scrolls in ring-chunks (64·2^N world voxels), i.e. at 1/2^N the window's rate; entering strip = 32 ring-chunk columns ≈ 5–10 ms on one worker, async double-buffered like `buildFarHeightmap`, uploads through the same 4–6/frame budget (never bypass it). Blank-then-fill is safe *because of* cascade fall-through (§2.1) — no reveal-gating needed for rings.

**Saves/edits:** rings are pure procgen — they ignore edits, are never disk-cached (rebuild ≈2 s), and carry no versioning surface (if ever cached, stamp GEN_VERSION). Consequence: player structures are invisible beyond 77 m in v1; future opt-in = re-stamp edited chunks into ring 1 on save. This is consistent with the terrain-save-version lesson: rings can never carry stale heights.

---

## 4. TRANSITION LOOK

- **The translucent band dies by deletion, not tuning.** Remove `terrainFog` (lib/lighting.glsl:154-168) at all four call sites, and the `traceFar/shadeFar/farHeightAt/farColorAt` block + far uniforms from lighting.frag (retiring texture units 6-7). Nothing in the pipeline ever again blends geometry toward sky: the fine window's last voxel at D0 is fully opaque, and 20 cm behind it stands the *same tree* as a ring-1 blob — same lattice site, same height/scale RNG draws, same `heightAt` silhouette. The dissolve was fading terrain against a background that lacked trees; now the far representation contains them (the DH lesson: fade between two opaque representations of the same content, or don't fade at all).
- **Boundary appearance:** a hard representation switch at D0, softened by the ±1-ring-brick stochastic dither + TAA into a ~3 m crossfade band. At 77 m, 2× voxel chunkification at ~3 px/voxel reads as style, not error — this is precisely LotL's visible-but-polished LOD line.
- **terrainFog's second job** (hiding the streaming frontier) is replaced by cascade fall-through: unfinished fine chunks show ring-1 coarse terrain (res-pop on completion) instead of a ghost band.
- **Keep aerial perspective as color-only:** generalize the desert dust haze (lighting.glsl:548-553) into a distance haze that shifts hue/saturation but **never alpha** — depth cueing without translucency, matching LotL's heavy atmospheric haze.
- **The old 2.5D heightfield survives in exactly one role** (per the user's condition): rebuilt at `FAR_STRIDE=64`, 1024 texels → 6.5 km across, rendered only for `t > D4` in the final miss branch as the horizon silhouette layer beyond ring 4, with `shadeFar`. Inside D4 it is dead code.

---

## 5. BUILD PLAN

**C0 — prove the look (single ring, static).** Ring 1 only: RingTable UBO + concatenated bm/coarse segments, shared pool reserve bump, `marchRing` + one handoff at D0, floor-quantized terrain fill + `deriveTreeSite` impostors (factoring PR lands first, gated on contentHash-identical), analytic ring shading + ring flag + ao/reflect early-outs, terrainFog deleted, old heightfield kept beyond ring 1's exit. Ring built async at startup/teleport only (no scrolling — the `buildFarHeightmap` lifecycle). Dev toggle `/rings`.
*Accept:* stand at spawn facing forest: 77–200 m shows individual opaque chunky tree blobs with silhouettes matching the near forest line; sweep the view across D0 — no translucent band, no sky bleed at the seam; teleport to `/mesa` — bench steps visible in ring 1; frame cost delta < 0.5 ms.

**C1 — full cascade + horizon.** Rings 2–4, per-ring switch distances, **primary-ray coarse-mip skip** (this is the milestone where rays get long, so the optimization lands here, not later), heightfield re-scoped to stride-64 horizon beyond D4, retire far uniforms inside D4.
*Accept:* from a highland ridge, terrain reads as 3D voxels to ~1.2 km (mesa at 800 m shows stepped cliffs; forest at 400 m shows discrete blobs); horizon silhouette continues beyond; total primary-march cost delta < 1.5 ms at 1440p; VRAM within the §1 table.

**C2 — motion.** Per-ring toroidal recenter (async strip regen, budgeted uploads, pool-before-brickmap ordering), cascade fall-through verified under sprint, stochastic dither band + TAA polish, shadeHit-input neutralization audit (no far-field speckle at night/underwater/desert transitions).
*Accept:* sprint 500 m continuously and teleport ±2 km repeatedly: no black terrain, no holes, no visible ring seams in motion at normal play speed; ring scroll costs < 1 ms main-thread; the brick-desync repro (`/mega` analog for rings) stays clean.

**C3 — richness (each item optional, independent).** Coarse sun shadows per ring (long dusk shadows across valleys); saguaro/palm impostors verified per biome + big boulder erratics in rings 1–2; cave-mouth decals from the pure site scan; edited-chunk restamp into ring 1; ring telemetry (pool occupancy, upload latency) in the debug HUD.
*Accept:* sunset over the desert casts kilometer-scale mesa shadows; oasis palms visible at 500 m; journal-sketch vista screenshots read like LotL promo shots.

---

## 6. RISKS (ranked)

1. **RNG-refactor determinism break** (world relayout, GEN_VERSION bump, discarded saves). *Mitigation:* factoring PR is standalone and merges only when `world.contentHash()` + golden stats are bit-identical; ring code never touches the fine path after that.
2. **Seam artifacts at D0** (step-wall rim or sky slivers all around the player — the mesa-seam failure mode at 77 m). *Mitigation:* floor quantization; sample `heightAt` at identical int coords; dither band; explicit acceptance test staring at the seam on a mesa bench and a steep forest slope.
3. **Upload-desync re-arm** (ring recenters move 2^N× area; the "jumbled trees" bug). *Mitigation:* rings join the existing budget (never bypass), pool-SubData-before-brickmap-SubData ordering per chunk, and — decisive — cascade fall-through means a stale/blank ring chunk renders as the next coarser ring, not garbage.
4. **shadeHit window-anchored inputs on ring hits** (toroidal-wrap garbage → far-field speckle). *Mitigation:* ring flag in `gPosMat.w` + a written checklist of the five inputs (heightTex burial, lightVol, pointSkyVis, water-neighbor getVoxelAt, softShadow paths), each with a neutral fallback; C2 acceptance includes night/dawn sweeps.
5. **Shader lockstep break** (voxel_trace.glsl signature change ripples through gbuffer/lighting/ao/reflect includes; hot-reload shows a broken frame if any drifts). *Mitigation:* one atomic commit for the include + all four includers; RingTable UBO so uniforms change once, not per-shader.
6. **Perf regression from long rays** (256-iter caps blown, gNormal.w heat). *Mitigation:* coarse-mip skip lands in the same milestone as rings 2–4; per-ring iteration caps; heat visualization already exists.
7. **Memory creep** (shared CPU pool mirror +1 GiB RAM; ring pool overruns in dense desert-forest). *Mitigation:* 2× headroom in per-ring reserves, telemetry in C3, and ring content is re-derivable — a ring pool overflow can degrade to terrain-only (drop impostors) rather than crash.

Key files touched: `src/common.h` (ring constants), `src/worldgen/internal.h` (+`surfaceInfoAt`/`derive*Site`), `src/worldgen/worldgen.cpp` (ring fill driver), `src/worldgen/flora.cpp` (site-derivation split), `src/world.h/.cpp` (ring slot bases, pool reserve, uploadChunk param), `src/shaders/lib/voxel_trace.glsl` (parameterized march), `src/shaders/gbuffer.frag` (cascade + coarse skip), `src/shaders/lighting.frag` + `lib/lighting.glsl` (terrainFog deletion, ring shading), `src/shaders/ao.frag`/`reflect.frag` (ring early-out), `src/render.cpp/.h` (bindings, RingTable UBO), `src/app.cpp` (ring recenter/scheduling).

---

# PART II — ADVERSARIAL REVIEW: REQUIRED AMENDMENTS (normative)

All verification reads complete. Findings below, with the arithmetic.

---

# ADVERSARIAL REVIEW — "Voxel Clipmap Rings" v1

Ground-truth checks performed against the working tree: `src/common.h`, `src/shaders/lib/voxel_trace.glsl`, `src/shaders/gbuffer.frag`, `src/shaders/lib/lighting.glsl`, `src/shaders/lighting.frag`, `src/world.h`, `src/world.cpp:482-597`, `src/app.cpp:2075-2139`, `src/worldgen/internal.h:136-186`, `src/worldgen/worldgen.cpp:40-160`, `src/worldgen/flora.cpp:700-884`. The audits' numbers held up everywhere I re-derived them, with the exceptions called out below.

## CRITICAL

**C1. The shared-pool allocation story is self-contradictory, the allocator is unguarded, and this is the exact mechanism of the project's worst historical bug.**
- Evidence: §1.2 says "Single allocator; reserve computed up front." §3.3 says "embarrassingly parallel, zero brickpool contention with fine gen (own reserve range)." Those are two different designs. Meanwhile `World::allocBrick`/`allocBrickNoFill`/`freeBrick` (world.cpp:84-131) are bare `std::vector` push/pop with **no mutex anywhere in world.cpp** (grep confirms zero `std::mutex` in that TU). Fine gen only survives because workers fill `ColumnResult` staging buffers (worldgen.cpp:46-146) and a single coordinating thread does the alloc/merge.
- Failure scenario: teleport → `recenterWindow` regenerates the fine window (allocBrick storm on the coordinating thread) while the design's "coarsest-first ring fill on the same worker pool" commits ring bricks. One `allocBrick` off-thread = torn `freeBricks` pop = two bricks sharing a pool slot = the canonical "black terrain / jumbled trees" desync (common.h:19-22, project memory). Probability of an implementer getting this wrong given the ambiguous spec: high — the spec itself doesn't say which thread allocates.
- Amendment (required): pin it in the design. Rings get hard slot ranges `[8.0M + r·R_r, 8.0M + (r+1)·R_r)` with per-ring allocators; ring workers produce staged chunk buffers only; slot assignment + CPU-pool memcpy + GPU upload happen exclusively on the main thread inside the existing budgeted drain (app.cpp:2114-2125). Add a range assert on every alloc/free (fine < 8.0M, ring r inside its range).

**C2. Ring 3/4 pool reserves are undersized for forests, and overflow does not "degrade" — it triggers the realloc-corruption path.**
- Numbers: FOREST `treeChance = 0.94` (internal.h:160), TREE_CELL = 55 (flora.cpp:845). Ring 4 extent 32768² → (32768/55)² ≈ 355k tree cells. Pure forest: 334k trees; a realistic 40% forest disc (moisture period 2200 vox → multi-km forest lobes exist): ≈ 133k–150k trees. At ring 4 a canopy blob spans 1–3 of the ring's 8³ bricks (fat-stamped, straddling) → ~2 bricks/tree → 270–670k tree bricks, plus 65k–200k mixed surface bricks (65,536 brick columns × 1–3) → **~350k–870k bricks needed vs a 500k reserve**. Ring 3: (16384/55)²×0.94 ≈ 83k trees × ~3 bricks ≈ 250k + surface ≈ 450k — also brushing the ceiling. The design's "~800 tree impostors" line item is a ring-1 count silently reused for all rings; tree count grows 4× per ring.
- Why CRITICAL and not MAJOR: on overflow the CPU `brickpool` vector grows past its reserve → realloc moves the backing store under concurrent readers (the precise race documented at common.h:28-33), and `ensurePoolGPU` grow-by-copy fires at ~4.8 GiB (transient old+new ≈ 12 GiB VRAM + a multi-hundred-ms hitch, world.cpp:563-597). "Ring pool overflow can degrade to terrain-only" is asserted in Risk 7 but no mechanism exists or is specified.
- Amendment (required): reserve 1.0M bricks each for rings 3/4 (+0.5 GiB VRAM — still under 6 GiB total on a 24 GiB card), and implement the degrade as a hard rule: when a ring allocator hits its range end, skip impostor stamping for the remainder of the fill and log telemetry. Never let ring allocation touch the growth path.

## MAJOR

**M1. The ring-hit shading checklist is provably incomplete — the leaf-translucency ray is missing, and leaves are the dominant ring material.**
- Evidence: Risk 4 enumerates "the five inputs (heightTex burial, lightVol, pointSkyVis, water-neighbor getVoxelAt, softShadow paths)." lighting.glsl:449-457 is a sixth: `if (mat == 5u) ... anyHit(backside, sunDir, 220.0)` inside `shadeHit`, on the main path. Ring tree blobs are MAT_LEAVES.
- Failure scenario, concrete: golden hour, camera facing the sun over a forest. Every ring canopy hit is ≥768 vox from the camera; its 220-vox sun ray starts outside the window → window AABB clip (voxel_trace.glsl:54-61) → returns false → `backLit = 1` → **the entire distant forest gets +transTint × forwardSun² × 0.85 × dayLight — a uniform warm-green bloom** — except pixels whose ray geometrically re-enters the window, which flip to 0. Glow plus speckle, on the flagship vista.
- Amendment (required): the omission proves that enumerating call sites is the wrong protection. Gate structurally: one `bool ringHit` at the top of `shadeHit`, and every `anyHit`/`traceRay`/`softShadow`/texture-wrap input inside lighting.glsl checks it once. Add a debug view that tints any pixel where a secondary ray was cast from `ring > 0`.

**M2. Sun-shadow and AO discontinuity at D0 is a daily-visible artifact, not a C3 nicety.**
- Numbers: fine hits get ray-traced sun visibility (ao.frag, 220-vox = 22 m range) and 48-vox AO; ring hits get `shadow = 1, ao = 1`. Any tree or cliff within 22 m of the D0 boundary casts fine-side shadow that terminates exactly at the dithered seam. At sun elevation 15°, a 25-voxel tree casts a ~93-voxel (9.3 m) shadow — dapple fields will visibly shave off along a circle of radius 76.8 m around the player at every dawn and dusk (the desert's signature time of day, per project direction).
- Amendment: fade `shadowIn`→1 and `aoIn`→1 over the last ~96 vox before D0 on the fine side so both sides of the seam converge on the same analytic value (3-line change in lighting.frag), and keep C3's per-ring coarse sun ray for the km-scale term. Without the fade, the dither band cannot hide a binary lighting step.

**M3. The t > D4 gating of the horizon heightfield breaks warmup: sky band from 102 m to 1.23 km after every teleport/startup.**
- Evidence: §4 "rendered only for t > D4 in the final miss branch; inside D4 it is dead code." Ring build takes 1.5–3 s typical (my re-derivation: bulk_fill = 1.18 s ≈ 8.4M `heightAtEx` → ring = 4.2M `heightAtEx` ≈ 0.6 s/ring on the pool; ×2–3 in mesa/oasis-heavy desert → worst ~6–9 s for all four). During that window the cascade falls through empty rings and the miss branch draws heightfield only beyond 12,288 vox → **terrain between the window edge and 1.23 km renders as sky**, which is strictly worse than today's behavior, on every `/mesa`-style teleport (C2 acceptance literally requires "teleport ±2 km repeatedly").
- Amendment: the heightfield miss branch must march from the cascade's actual last-valid t (traceFar already starts near the camera today — lighting.frag:80-109 — keep that capability), OR the teleport veil/loading reveal blocks on ring-1 + heightfield readiness (+~1 s behind the existing veil). Either is cheap; the design as written specifies neither.

**M4. "256 outer steps per ring segment" is a regression for ring 0 — today's code needs 1024.**
- Numbers: gbuffer.frag:81 uses `MAX_OUTER = 1024` (and has **no coarse skip**); anyHit/traceRay use 256 *with* coarse skip. Ring 0's segment to D0 = 768 vox = 96 bricks/axis; a diagonal grazing ray visits up to ~96×3 = 288 brick cells — over the proposed 256 cap → premature fall-through → coarse ring-1 terrain rendered **inside 77 m** on steep diagonal downhill views, exactly the seam-integrity failure the design's own acceptance test stares at.
- Amendment: per-ring caps must be derived, not asserted: ring 0 keeps 1024 until the primary coarse skip lands (C1), after which 256 is provably sufficient (segment ≤ 96 bricks/axis, empty cells fast-forwarded 4× — mirror voxel_trace.glsl:82-110). State the C0 cap as 1024.

**M5. Tree impostors change texture and opacity at the seam; fat-stamping creates an undefined inside-solid ring-entry case.**
- Evidence: fine canopies are ~65%-filled popcorn bulbs on a shell with the trunk visible through gaps (flora.cpp:716-760, `(rng() & 1023) < 358` skip); the impostor is a solid ellipsoid. At D0 (≈3 px/voxel at 1080p, ≈2.1–4.2 px at 1440p) a tree crossing the dither band visibly densifies/darkens — "switches shape," the user's stated worse-than-fading failure. Position is fine: I verified trunk lateral jitter is only ±4% of length (flora.cpp:782-784), so anchor+(0,trunkLen,0) centering drifts < 2 voxels — position pop is a non-issue; the *fill* mismatch is the issue. Separately: a fat-stamped canopy is fatter than the fine canopy, so a ray crossing D_r inside it starts **inside a solid ring voxel** → hit at segment entry with `hitAxis = -1` → `normal = -rd` fallback → N·L garbage rim sparkle around seam-straddling trees.
- Amendment: stamp impostor canopies with the same ~65% fill using a deterministic per-voxel world-coord hash (no RNG stream needed, preserves airy look and roughly matches transmittance), and define ring-entry-inside-solid: accept the hit but reuse the previous segment's exit axis for the normal.

**M6. The "future content automatically appears" claim is one-third true, and the false two-thirds includes a stated project goal.**
- Verified true for: height, mesas/benches, oasis basin + water, dunes, biome tint/material — all inside `heightAtEx`/grammar (internal.h:381-447, worldgen.cpp:84-135).
- False for flora: every species requires (a) a `derive*Site` split that must consume *exactly* the fine path's RNG draws — I verified trees use exactly 6 draws before `grow()` (flora.cpp:861-879: chance, x-jit, z-jit, sRoll, trunkLen, trunkThick), so the v1 split is sound, but each future species repeats this hash-gated surgery — and (b) bespoke impostor geometry. Three bespoke stamps in v1 already (tree/palm/saguaro).
- False for anything not per-column pure: caves today, and **rivers tomorrow** — a explicit direction in project memory ("river all tried & reverted", still a goal). A carved river visible to 77 m that vanishes into unbroken ring terrain beyond re-creates the exact complaint this design exists to fix.
- Amendment: rewrite the internal.h contract comment to scope the guarantee ("per-(x,z) pure surface content only"); add "ring representation defined" to the definition-of-done for any new species/feature; require future rivers to be `heightAt`-integrated (a pure carve term) or get a bespoke ring pass.

## MINOR

**m1. VRAM table arithmetic slip:** 10.0M bricks × 512 B = 5,120,000,000 B = **4.77 GiB**, not 4.88. Grand total ≈ 5.02 GiB (248 MiB maps + 4.77 GiB pool + 3.9 MiB coarse). Conclusion (fits trivially) unchanged. With C2's reserve bump: ≈ 5.5 GiB. CPU mirror re-check: pool vector reserve 10–11M bricks = 5.1–5.6 GB system RAM (today 4.1 GB) — design's "+1 GiB" is right, but say "system RAM total ~5.5 GB" explicitly.

**m2. Screen-space dither fizzes under motion.** Interleaved-gradient noise re-rolls the ±8·scale switch offset per pixel per frame as the camera translates; at 1.6–3 px/voxel TAA's neighborhood clamp rejects history in the band → a permanent ~3 m shimmer annulus at every D_r during sprint (+25% FOV kick makes it worse). Amendment: hash *world position* along the ray for the switch offset so the boundary is static in world space; TAA then converges.

**m3. Ring water fallback undefined → the whole distant oasis renders as falling droplets.** lighting.glsl:225-232: out-of-window `getVoxelAt` returns 0 → `below == 0` → `isFalling = true` → animated sparkly cyan with time-varying glint, for every ring water voxel, at 200+ m. The design says "skip water-neighbor getVoxelAt" but never defines the substitute. Amendment: ring water hard-codes pool-surface state (above=air, below=water).

**m4. Name the coordinate footguns in the design, don't just avoid them.** The wrap math itself checked out (`bmSlot`/`windowLoV`/`slotOfBrick` generalize by parameter; GLSL `>>` is arithmetic; `ivec3(floor(p/scale))` is negative-safe). But: `bmIdx` (world.h:176-182) and `buildCoarse` are origin-0-only and will be the first thing a ring-fill implementer copies; and any C++ `worldVox / scale` (truncating) instead of `floorDiv`/`>>` mirror-seams the four quadrants around world origin — one texel of misalignment along x=0/z=0, visible as a cross-shaped ring seam after the first recenter. Amendment: design lists `floorDiv` + per-ring `slotOfBrick` as the only sanctioned conversions; ring-side `bmIdx` use is banned in review.

**m5. Fine water reflections now contradict the primary image.** A mesa visible at 300 m in the primary cascade will not appear in a lake reflection at 50 m (reflect.frag rays remain window-clipped, ring flag early-outs). Pre-existing limitation, but the rings make it *newly noticeable* because the far content is now visible. Accept for v1; note as C3+ (march reflection rays through the cascade).

## RECOMPUTED BUDGETS (attack surfaces 1–3, no findings beyond the above)

- **GPU frame cost: PASS with C1's coarse skip, which is load-bearing.** Cascade geometry keeps per-ring work constant: ring segments are 768→1536→3072→6144→12288 world vox = a fixed 384 ring-voxels = 48 bricks/axis each; ring 0 capped at 768 vox is 2.7× *cheaper* than today's 2048-vox window march. Worst horizon-grazing pixel ≈ 96–288 (ring 0) + 4×(48–144) brick visits ≈ 1.1–1.3× today's worst case. Sky-band pixels (below worldTop, above terrain) roughly double without a primary coarse skip: ~1.1M such px at 1440p × ~300–500 extra 4 B SSBO reads ≈ 0.5–1.5 ms (ring brickmap hot slices are L2-resident on a 4090's 72 MB); with per-ring coarse skip ≈ +0.2–0.5 ms. Secondary passes stay flat via ring-flag early-outs. 60+ fps at 1440p holds with margin; C0's "<0.5 ms" acceptance is tight but measurable, and M4's cap fix is required for correctness regardless.
- **Generation: PASS.** Per-ring terrain ≈ 4.2M `heightAtEx` ≈ 0.6 s (derived from measured bulk_fill 1.18 s ≈ 8.4M calls), ×2–3 desert-worst; tree enumeration ring 4 ≈ 355k cells ≈ 0.1–0.3 s; all four rings ≈ 2.4 s typical / ~7 s desert-worst vs 86 s for window enlargement — the sidestep claim is genuine. Recenter strips ≈ 65k texels ≈ 5–10 ms on one worker, and I verified the margin math: ring N's frontier always sits beyond D_N by ≥ 4 ring-chunks under the 1/2^N scroll rate, so upload lag (6/frame budget, ~6-frame backlog per ring scroll) is invisible — the fall-through is genuinely only needed for teleports (which is M3's domain). The race-freedom, however, is not established by the design text — that is C1.
- **Memory: PASS after m1's correction and C2's resize** (~5.5 GiB VRAM, ~5.6 GB RAM — fine on target hardware).
- **Complexity (attack surface 6): the design is already minimal-with-staging.** C0 *is* the "one ring + heightfield" simpler variant; rings 2–4 are table entries on the same code. The one scheduling change I'd make: gate rings 3–4 behind C1 perf/quality validation of rings 1–2 (D2 = 3072 ≈ 307 m covers most everyday sightlines on this terrain), since rings 3–4 carry the pool-sizing risk (C2) for the least-viewed content.

## VERDICT: BUILD WITH AMENDMENTS

1. (C1) Pin the allocation model in the design: per-ring hard slot ranges + main-thread-only slot assignment through the existing budgeted upload path, with range asserts. No ambiguity survives to implementation.
2. (C2) Ring 3/4 reserves → 1.0M bricks each; implement (not just assert) the terrain-only degrade at range exhaustion; ring allocation must never reach the vector-realloc or `ensurePoolGPU` grow paths.
3. (M1) Structural ring-gating of *all* secondary rays in `shadeHit` (one flag, not a checklist); the leaf-translucency ray at lighting.glsl:449-457 is the proof the checklist approach fails.
4. (M2) Fade fine `shadowIn`/`aoIn` to 1 over the last ~96 vox before D0 so the seam converges; keep C3 coarse sun rays for km-scale shadows.
5. (M3) Horizon heightfield marches from the cascade's last valid t (or reveal/veil blocks on ring 1) — kill the sky band during ring warmup.
6. (M4) Ring-0 iteration cap stays 1024 until the primary coarse skip lands; derive per-ring caps from segment length, don't assert 256.
7. (M5) Hash-dithered ~65% fill for impostor canopies + defined inside-solid ring-entry normal.
8. (M6 + m2–m4) Scope the "automatic content" contract honestly (rivers/agent-carved content excluded); world-anchored dither; pool-surface fallback for ring water; `floorDiv`/`slotOfBrick`-only rule with `bmIdx` banned ring-side.

The core architecture — direct pure-procgen rings, one shared pool, concatenated brickmaps with offset math, t-continuation cascade, analytic ring shading, deletion of the translucent band — survived every quantitative attack I ran at it and is the right shape for this engine. The dangers are all in the four places the project has already bled: the allocator, the reserve sizing, the "we listed all the inputs" checklist, and warmup/teleport windows.


---

# PART III — IMPLEMENTATION FINDINGS (C0, 2026-07-02)

Learned building C0; these BIND future milestones:

1. **The 4 GiB SSBO wall (CRITICAL).** The shared brickpool SSBO must stay under
   2^32 BYTES (8,388,608 bricks). Placing ring 1's slot range above the fine
   reserve (8M) pushed the buffer to 4.35 GB and the NVIDIA driver silently
   corrupted sub-range uploads: streamed chunks lost their surface bricks on
   the GPU while all CPU bookkeeping said they were uploaded — reproducing the
   black-terraces desync at GRID_XZ=2048 where the pre-C0 build was verified
   clean. Fix: ring slot base moved to 6.5M (fine peak ~5.6M, warning at 6.3M),
   pool alloc clamped below 2^32. **C1 consequence: rings 2-4 CANNOT share the
   fine pool buffer — give the rings their own pool SSBO on a free binding
   (each buffer stays well under 4 GiB); brickmap entries then hold ring-local
   indices and the ring trace samples the ring pool block.**
2. **The fine-march cap must gate on UPLOAD completion, not streamer stage.**
   fv.revealDist now folds in the distance to the nearest gpuQueue entry.
   A CS_FINAL chunk whose upload is still queued is air on the GPU; rays
   tunnelling through it hit the cut-open interiors of chunks behind (bare
   stone far below heightTex -> burial-black). gbuffer.frag caps the fine
   march at min(D0, revealDist) and falls through to the ring.
3. **The old 19m reveal floor is GONE.** It existed so the fog curtain never
   closed onto the camera; with the opaque ring as the fallback, the correct
   fully-outrun behaviour is "everything coarse until uploads catch up".
4. **Ring recentre shipped in C0** (was C2): drift > GRID_XZ/4 from the ring
   centre triggers a background rebuild (~0.7-0.8s, ring vectors only, bump
   allocator, no World allocator contact) + uploadRing swap. The shader
   addresses the GPU ring via World::ringOriginGPU (set only at upload) so a
   rebuild never desyncs addressing from resident data.
5. **Walk-test has eyes now**: --walk-test dumps walktest_frame.bmp +
   walktest_class.bmp (miss/fine/ring per pixel) + walktest_stats.txt
   (per-class luminance, dark-pixel burial/AO forensics). Use it before
   shipping any far-field change.
   *(Historical: the --walk-test harness was removed together with the
   first-person view — it scripted the FP camera and read the FP G-buffer.
   A future far-field harness would need to be diorama-native: scripted
   orbit poses plus a fog-of-war-off debug flag, with recalibrated
   thresholds. Resurrect the old one from git history if needed.)*
