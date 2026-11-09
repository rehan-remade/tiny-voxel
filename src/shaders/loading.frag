#version 430 core
// Loading screen — a "naturalist's field journal" page under open sky. Drawn as
// a single fullscreen-triangle pass (fullscreen.vert) with the same 8x8 bitmap
// font-in-a-uniform trick console.frag uses, so it needs no asset pipeline. All
// composited opaque in-shader (no GL blend). Driven by app.cpp while the world
// generates on a background thread.
in vec2 vUV;
out vec4 FragColor;

uniform vec2  viewport;        // framebuffer px
uniform float progress;        // 0..1 overall gen progress
uniform int   font[256];       // 128 glyphs * 2 (rows 0-3 in [g*2], 4-7 in [g*2+1])
// Three centered text lines, each packed 4 ASCII codes / int (12 ints = 48 chars).
uniform int   titleChars[12]; uniform int titleLen;
uniform int   subChars[12];   uniform int subLen;
uniform int   stepChars[12];  uniform int stepLen;

int glyphRowByte(int code, int row) {
    if (code < 0 || code > 127) return 0;
    int word  = font[code * 2 + (row >= 4 ? 1 : 0)];
    int shift  = (row & 3) * 8;
    return (word >> shift) & 0xFF;
}
int charAt(int arr[12], int i) {
    if (i < 0 || i >= 48) return 0;
    return (arr[i >> 2] >> ((i & 3) * 8)) & 0xFF;
}
// Coverage (0/1) of a horizontally-centered text line at this pixel. cy = vertical
// centre in px (origin bottom-left, matching gl_FragCoord); scale = px per font px.
float lineCover(vec2 px, int arr[12], int len, float cy, float scale) {
    float cw = 8.0 * scale;
    float totalW = float(len) * cw;
    float x0 = (viewport.x - totalW) * 0.5;
    float y0 = cy - 4.0 * scale;
    if (px.x < x0 || px.x >= x0 + totalW) return 0.0;
    if (px.y < y0 || px.y >= y0 + 8.0 * scale) return 0.0;
    float lx  = px.x - x0;
    int   col = int(lx / cw);
    int   code = charAt(arr, col);
    if (code == 0 || code == 32) return 0.0;
    int gx   = int((lx - float(col) * cw) / scale);   // 0..7 left->right
    int gy   = int((px.y - y0) / scale);              // 0..7 bottom->top
    return float((glyphRowByte(code, 7 - gy) >> gx) & 1);
}
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i),            hash(i + vec2(1,0)), f.x),
               mix(hash(i + vec2(0,1)), hash(i + vec2(1,1)), f.x), f.y);
}

void main() {
    vec2 px = gl_FragCoord.xy;
    vec2 uv = px / viewport;
    // Parchment (cream, bottom) easing up into a calm loading sky.
    vec3 paper = vec3(0.94, 0.90, 0.79);
    vec3 sky   = vec3(0.62, 0.73, 0.86);
    vec3 col = mix(paper, sky, smoothstep(-0.1, 1.05, uv.y));
    // Hand-washed watercolor mottle (two octaves) + soft edge vignette.
    float m = vnoise(uv * vec2(3.0, 2.0)) * 0.55 + vnoise(uv * 7.3 + 11.0) * 0.45;
    col *= 0.94 + 0.10 * m;
    col *= 1.0 - smoothstep(0.42, 0.98, distance(uv, vec2(0.5, 0.52))) * 0.34;

    vec3 ink     = vec3(0.20, 0.15, 0.10);
    vec3 inkSoft = vec3(0.34, 0.28, 0.20);
    float H = viewport.y;
    float ts = max(2.0, H / 150.0);

    col = mix(col, ink,     lineCover(px, titleChars, titleLen, H * 0.655, ts));
    col = mix(col, inkSoft, lineCover(px, subChars,   subLen,   H * 0.585, ts * 0.40));
    col = mix(col, ink,     lineCover(px, stepChars,  stepLen,  H * 0.420, ts * 0.52));

    // Brush-stroke progress underline: faint sepia track, solid fill to `progress`
    // with a slightly ragged right edge so it reads hand-drawn, not a GUI widget.
    float barW = viewport.x * 0.40;
    float barX0 = (viewport.x - barW) * 0.5;
    float barY = H * 0.360, barH = max(2.0, ts * 0.5);
    if (px.x >= barX0 && px.x <= barX0 + barW && px.y >= barY && px.y <= barY + barH) {
        float t    = (px.x - barX0) / barW;
        float edge = progress + (vnoise(vec2(px.y * 0.7, 3.0)) - 0.5) * 0.012;
        col = (t <= edge) ? ink : mix(col, ink, 0.16);
    }
    FragColor = vec4(col, 1.0);
}
