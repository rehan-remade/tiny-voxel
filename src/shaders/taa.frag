#version 430 core
// Temporal anti-aliasing resolve. Blends the current (sub-pixel-jittered) frame
// with the motion-reprojected history under a neighbourhood colour clamp (which
// kills ghosting on disocclusion / fast motion). The output is both the
// displayed image and the next frame's history. Accumulating the jittered
// samples over time is what produces the anti-aliasing.
in vec2 vUV;
out vec4 FragColor;

uniform sampler2D curTex;      // current raw lit frame: RGB colour, A = map coverage
uniform sampler2D histTex;     // previous resolved frame
uniform sampler2D motionTex;   // RG screen-space motion (thisUV - prevUV)
uniform vec2  viewport;
uniform int   historyValid;    // 0 on the first frame after enable/resize

void main() {
    vec4 c = texture(curTex, vUV);
    if (historyValid == 0) { FragColor = c; return; }

    // 3x3 neighbourhood colour box of the current frame — the history is clamped
    // to this, so stale/ghosted history snaps back to plausible values.
    vec2 px = 1.0 / viewport;
    vec3 mn = c.rgb, mx = c.rgb;
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        vec3 s = texture(curTex, vUV + vec2(dx, dy) * px).rgb;
        mn = min(mn, s);
        mx = max(mx, s);
    }

    vec2 prevUV = vUV - texture(motionTex, vUV).xy;
    vec3 resolved;
    if (any(lessThan(prevUV, vec2(0.0))) || any(greaterThan(prevUV, vec2(1.0)))) {
        resolved = c.rgb;                                 // reprojects off-screen
    } else {
        vec3 hist = clamp(texture(histTex, prevUV).rgb, mn, mx);
        resolved = mix(c.rgb, hist, 0.9);                 // 90% history accumulation
    }
    FragColor = vec4(resolved, c.a);   // pass the coverage alpha through (mapview.frag composites by it)
}
