// Hashing + value/cellular noise helpers, plus the shared per-voxel
// stone-grain tone built on them. Pure functions (no uniform deps), included
// first so the material, sky and icon modules can build on them.

float hash31(vec3 p) {
    p = fract(p * vec3(0.1031, 0.11369, 0.13787));
    p += dot(p, p.yzx + 19.19);
    return fract((p.x + p.y) * p.z);
}
float blockNoise(vec3 p, float freq) {
    return hash31(floor(p * freq) + 0.5);
}
float vnoise2(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash31(vec3(i,                0.0));
    float b = hash31(vec3(i + vec2(1.0, 0), 0.0));
    float c = hash31(vec3(i + vec2(0, 1.0), 0.0));
    float d = hash31(vec3(i + vec2(1.0,1.0),0.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
// ---- Biome climate field (EXACT MIRROR of biomeAt() in worldgen/internal.h)
// Same integer hash, same frequency, same weights, same colors, so the grass
// tint here lands in the same regions as the biome's terrain relief on the
// CPU. If you change biomeAt on the CPU, change this too.
float bHash2(int x, int z) {
    uint h = uint(x) * 374761393u + uint(z) * 668265263u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    h = h ^ (h >> 16u);
    return float(h & 0xFFFFFFu) / float(0xFFFFFF);
}
float bSmoothNoise(float x, float z) {
    int xi = int(floor(x)), zi = int(floor(z));
    float xf = x - float(xi), zf = z - float(zi);
    float a = bHash2(xi,     zi);
    float b = bHash2(xi + 1, zi);
    float c = bHash2(xi,     zi + 1);
    float d = bHash2(xi + 1, zi + 1);
    float u = xf * xf * (3.0 - 2.0 * xf);
    float v = zf * zf * (3.0 - 2.0 * zf);
    return mix(mix(a, b, u), mix(c, d, u), v);
}
// Smooth 3D value noise (trilinear hash) — organic grain for stone.
float vnoise3(vec3 p) {
    vec3 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float n000 = hash31(i + vec3(0,0,0)), n100 = hash31(i + vec3(1,0,0));
    float n010 = hash31(i + vec3(0,1,0)), n110 = hash31(i + vec3(1,1,0));
    float n001 = hash31(i + vec3(0,0,1)), n101 = hash31(i + vec3(1,0,1));
    float n011 = hash31(i + vec3(0,1,1)), n111 = hash31(i + vec3(1,1,1));
    float x00 = mix(n000, n100, f.x), x10 = mix(n010, n110, f.x);
    float x01 = mix(n001, n101, f.x), x11 = mix(n011, n111, f.x);
    return mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z);
}
// 3D cellular / Worley noise. Returns:
//   x = F1            (distance to nearest feature point)
//   y = F2 - F1       (small near a cell boundary -> draws the seam network)
//   z = cell tone     (random shade of the nearest cell -> the "chunk" color)
// Evaluated at voxel centers, this partitions voxels into coherent multi-voxel
// chunks (each its own tone) with darker voxels tracing the chunk boundaries.
vec3 stoneCells3(vec3 p) {
    vec3 ip = floor(p), fp = fract(p);
    float f1 = 9.0, f2 = 9.0; vec3 nearCell = vec3(0.0);
    for (int k = -1; k <= 1; ++k)
    for (int j = -1; j <= 1; ++j)
    for (int i = -1; i <= 1; ++i) {
        vec3 g = vec3(float(i), float(j), float(k));
        vec3 c = ip + g;
        vec3 o = vec3(hash31(c), hash31(c + 19.1), hash31(c + 43.7));
        vec3 r = g + o - fp;
        float d = dot(r, r);
        if (d < f1)      { f2 = f1; f1 = d; nearCell = c; }
        else if (d < f2) { f2 = d; }
    }
    return vec3(sqrt(f1), sqrt(f2) - sqrt(f1), hash31(nearCell + 7.3));
}
float vhash(ivec3 p) {
    uint h = uint(p.x) * 374761393u
           ^ uint(p.y) * 668265263u
           ^ uint(p.z) * 2147483647u;
    h ^= h >> 13;
    h *= 1274126177u;
    h ^= h >> 16;
    return float(h & 0xFFFFu) / 65535.0;
}
// Per-voxel flat stone tone (brightness multiplier). The voxel is the smallest
// paint unit: each voxel face is ONE solid color, evaluated at the voxel center
// so it's constant across the face (clean cubes, no sub-voxel painting). The
// structure comes from Worley cells: voxels group into ~5-voxel CHUNKS that
// share a tone, and voxels on a chunk boundary darken into seam lines — the
// "larger chunks separated by darker stripes" look, instead of pure noise.
// A little per-voxel jitter keeps chunk interiors from being dead flat.
// `cellSize` = Worley chunk size in voxels: how many tiny voxels group into
// one coherent PIECE (bark plate / leaf clump / stone chunk). Each material
// picks its own so a bark piece and a leaf clump read at different scales
// while the world stays tiny-voxel. stoneGrain() keeps the stone default.
// `seamStr` = how dark the cell BORDERS get — the single strongest "make the
// cells visible" control: higher = each piece clearly separated by a groove.
// (Pure function — lives here so the world materials AND the inventory icons
// can share it; see lib/materials.glsl and lib/icon.glsl.)
float stoneGrainSized(ivec3 voxel, float contrast, float cellSize, float seamStr) {
    vec3 vc = vec3(voxel);
    vec3 cell = stoneCells3((vc + 0.5) / cellSize);
    // Punchy structured base: full-weight chunk shade + per-voxel grain,
    // quantized to discrete steps. Snapping pushes voxels toward the palette
    // extremes -> strong region-to-region contrast (the look that read well).
    float chunk  = cell.z - 0.5;                          // coherent region shade
    float jitter = (vhash(voxel) - 0.5) * 0.35;           // per-voxel grain
    float structured = floor((chunk + jitter) * 4.0 + 0.5) / 4.0;
    // Continuous flow ON TOP: two octaves of smooth tonal drift. The low
    // frequency makes big light/dark ZONES across the floor (what gives the
    // reference its sweep); the higher octave adds mid-scale movement. Layered
    // over the punchy regions so it flows without averaging away contrast.
    float flow = (vnoise3(vc * 0.035) - 0.5) * 0.60
               + (vnoise3(vc * 0.11 + 9.0) - 0.5) * 0.28;
    float t = structured + flow;
    // Thin, mid-strength seam on chunk borders — a darker shade of the SAME
    // hue, not a different color or a thick band.
    // Wider border band when the seam is stronger, so pronounced grooves read
    // as grooves and not hairlines.
    float seam = 1.0 - smoothstep(0.04, 0.15 + 0.35 * seamStr, cell.y);
    return (1.0 + contrast * t) * (1.0 - seamStr * seam);
}
float stoneGrain(ivec3 voxel, float contrast) {
    return stoneGrainSized(voxel, contrast, 5.5, 0.30);   // stone default (unchanged)
}
