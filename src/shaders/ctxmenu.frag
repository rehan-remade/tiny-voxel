#version 430 core
// RuneScape-style right-click context menu ("Choose Option"): dark header
// bar, parchment-dark body, yellow hover row, and a little procedural axe /
// spade / pick icon beside the action. Drawn as a fullscreen blended pass
// (discard outside the panel), text via the packed 8x8 console font.
in vec2 vUV;
out vec4 FragColor;

uniform vec2 viewport;
uniform vec2 menuPos;      // panel top-left, WINDOW coords (y down from top)
uniform int  rowN;         // entry rows (excluding the header)
uniform int  hoverRow;     // -1 none
uniform int  chars[28];    // 4 rows x 28 cols packed 4 codes/int (row 0 = header)
uniform int  font[256];    // FONT8X8 packed 2 ints/glyph
uniform int  icons[3];     // per entry row: 0 none, 1 axe, 2 spade, 3 pick

const float W     = 300.0;   // must match FrameView::CTX_W
const float HDR   = 26.0;
const float ROW   = 30.0;
const int   COLS  = 28;

float charPx(int code, vec2 g) {   // g in glyph cells [0,8)
    if (code < 32 || code > 126 || g.x < 0.0 || g.x >= 8.0 || g.y < 0.0 || g.y >= 8.0)
        return 0.0;
    int gy = int(g.y);
    int rowBits = (font[code * 2 + (gy >> 2)] >> ((gy & 3) * 8)) & 0xFF;
    return float((rowBits >> int(g.x)) & 1);
}

// Text row: 8x8 glyphs at 2x — returns coverage for local pixel p, where the
// row's text starts at x0 and is vertically centred in a band of height bh.
float textRow(int row, vec2 p, float x0, float bh) {
    vec2 g = vec2((p.x - x0) / 2.0, (p.y - (bh - 16.0) * 0.5) / 2.0);
    int col = int(floor(g.x / 8.0));
    if (col < 0 || col >= COLS) return 0.0;
    int idx = row * COLS + col;
    int code = (chars[idx >> 2] >> ((idx & 3) * 8)) & 0xFF;
    return charPx(code, vec2(g.x - float(col) * 8.0, g.y));
}

float segDist(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

float inTri(vec2 p, vec2 a, vec2 b, vec2 c) {
    float d1 = (p.x - b.x) * (a.y - b.y) - (a.x - b.x) * (p.y - b.y);
    float d2 = (p.x - c.x) * (b.y - c.y) - (b.x - c.x) * (p.y - c.y);
    float d3 = (p.x - a.x) * (c.y - a.y) - (c.x - a.x) * (p.y - a.y);
    bool neg = (d1 < 0.0) || (d2 < 0.0) || (d3 < 0.0);
    bool pos = (d1 > 0.0) || (d2 > 0.0) || (d3 > 0.0);
    return (neg && pos) ? 0.0 : 1.0;
}

// Icons live in a 22x22 box; q is the local pixel inside it (y down).
vec4 axeIcon(vec2 q) {
    // Handle: diagonal from lower-left to upper-right.
    float hd = segDist(q, vec2(4.5, 18.5), vec2(14.0, 6.5));
    vec4 col = vec4(0.0);
    if (hd < 1.7) col = vec4(0.42, 0.26, 0.12, 1.0);
    // Head: a wedge hanging off the top of the handle, edge facing right.
    float head = max(inTri(q, vec2(10.5, 3.0), vec2(19.0, 7.5), vec2(11.5, 11.5)),
                     inTri(q, vec2(10.5, 3.0), vec2(19.0, 7.5), vec2(17.0, 12.0)));
    if (head > 0.0) {
        col = vec4(0.55, 0.58, 0.64, 1.0);
        if (q.x > 16.0) col.rgb = vec3(0.85, 0.88, 0.93);   // cutting edge
    }
    return col;
}

vec4 spadeIcon(vec2 q) {
    float hd = segDist(q, vec2(11.0, 3.5), vec2(11.0, 10.0));
    vec4 col = vec4(0.0);
    if (hd < 1.6) col = vec4(0.42, 0.26, 0.12, 1.0);
    if (inTri(q, vec2(6.0, 10.0), vec2(16.0, 10.0), vec2(11.0, 19.0)) > 0.0)
        col = vec4(0.55, 0.58, 0.64, 1.0);
    return col;
}

vec4 pickIcon(vec2 q) {
    // Handle: diagonal like the axe's, head crossing it near the top.
    float hd = segDist(q, vec2(6.0, 19.0), vec2(14.5, 6.5));
    vec4 col = vec4(0.0);
    if (hd < 1.7) col = vec4(0.42, 0.26, 0.12, 1.0);
    // Head: a shallow chevron (two segments meeting at the apex) with the
    // points drooping — the classic two-tip pick silhouette.
    float h1 = segDist(q, vec2(4.0, 9.5),  vec2(12.0, 3.5));
    float h2 = segDist(q, vec2(12.0, 3.5), vec2(19.5, 8.0));
    if (min(h1, h2) < 1.8) {
        col = vec4(0.55, 0.58, 0.64, 1.0);
        if (q.x < 6.0 || q.x > 17.5) col.rgb = vec3(0.85, 0.88, 0.93); // tips
    }
    return col;
}

void main() {
    vec2 px = gl_FragCoord.xy;
    // Panel-local coords, y DOWN from the panel's top edge.
    vec2 p = vec2(px.x - menuPos.x, (viewport.y - menuPos.y) - px.y);
    float H = HDR + ROW * float(rowN);
    if (p.x < 0.0 || p.x > W || p.y < 0.0 || p.y > H) discard;

    // 1px black border.
    if (p.x < 1.0 || p.x > W - 1.0 || p.y < 1.0 || p.y > H - 1.0) {
        FragColor = vec4(0.0, 0.0, 0.0, 0.95);
        return;
    }

    if (p.y < HDR) {   // header bar: "Choose Option"
        vec3 bg = vec3(0.28, 0.19, 0.10);
        float t = textRow(0, vec2(p.x, p.y), 10.0, HDR);
        vec3 col = mix(bg, vec3(0.94, 0.87, 0.63), t);
        FragColor = vec4(col, 0.96);
        return;
    }

    int   row = int((p.y - HDR) / ROW);
    row = clamp(row, 0, rowN - 1);
    vec2  rp  = vec2(p.x, p.y - HDR - float(row) * ROW);
    bool  hov = (row == hoverRow);
    vec3  bg  = hov ? vec3(0.32, 0.27, 0.18) : vec3(0.14, 0.12, 0.09);
    vec3  col = bg;

    // Icon box.
    int kind = (row < 3) ? icons[row] : 0;
    if (kind > 0 && rp.x >= 6.0 && rp.x < 28.0) {
        vec2 q = vec2(rp.x - 6.0, rp.y - (ROW - 22.0) * 0.5);
        vec4 ic = (kind == 1) ? axeIcon(q)
                : (kind == 3) ? pickIcon(q) : spadeIcon(q);
        col = mix(col, ic.rgb, ic.a);
    }

    // Entry text (chars row = entry row + 1).
    float t = textRow(row + 1, rp, kind > 0 ? 34.0 : 12.0, ROW);
    vec3 textCol = hov ? vec3(1.0, 0.92, 0.25) : vec3(0.96, 0.94, 0.88);
    col = mix(col, textCol, t);

    FragColor = vec4(col, 0.94);
}
