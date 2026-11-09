// Final present() tonemap.
// Output colour pipeline A/B (/srgb):
//   0 = LEGACY: partial ACES + 30% saturation, NO output transfer encode.
//       Every palette constant in the game was authored visually against
//       this (technically incorrect) curve — it is the shipped look.
//   1 = EXPERIMENT: treat shading as linear light — full ACES roll-off,
//       near-neutral saturation, then a proper sRGB transfer encode. Exists
//       to judge the correct transfer function in-scene BEFORE any palette
//       retuning; expect brighter mid-tones and softer colour until the
//       palettes are re-authored against it. (A fully correct pipeline also
//       wants HDR intermediates and TAA before tonemapping — this toggle is
//       only the output-encode experiment.)
uniform int colorMode;
vec3 present(vec3 col) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    vec3 aces = clamp((col * (a * col + b)) / (col * (c * col + d) + e), 0.0, 1.0);
    if (colorMode == 1) {
        float luma1 = dot(aces, vec3(0.299, 0.587, 0.114));
        vec3 t = mix(vec3(luma1), aces, 1.08);
        return pow(clamp(t, 0.0, 1.0), vec3(1.0 / 2.2));
    }
    col = mix(col, aces, TW_ACES_MIX);         // less ACES so colors stay punchy
    float luma = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(vec3(luma), col, TW_SATURATION); // stronger saturation boost
    return clamp(col, 0.0, 1.0);
}
