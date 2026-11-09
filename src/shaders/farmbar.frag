#version 430 core
in vec2 vUV;
out vec4 FragColor;

// Farm Mode bars (drawFarmBar): a bottom-centre row of fixed-width slots, the
// selected one tinted. Drawn once for the tool bar (1 TILL / 2 PLANT /
// 3 HARVEST) and — while the Plant tool is armed — again for the SEED bar
// (one slot per crop species) floating one row above it. Same 8x8 bitmap-font
// scheme as console.frag / tune.frag; the hit-test in app.cpp (farmBarHit)
// must match this geometry.
uniform vec2 viewport;
uniform int  chars[16];    // packed label bytes (cap: 64 columns)
uniform int  font[256];
uniform int  cols;         // label columns in this row
uniform int  slotw;        // columns per slot
uniform int  sel;          // selected slot (tinted)
uniform int  rowIdx;       // 0 = docked above the hotbar tray, 1 = one bar up

const float SCALE  = 2.0;
const float CW     = 8.0 * SCALE;
const float CH     = 8.0 * SCALE;
const float PAD    = 8.0;
const float MARGIN = 164.0;   // docks ABOVE the always-visible hotbar tray (~156px tall)

int glyphRowByte(int code, int row) {
    if (code < 0 || code > 127) return 0;
    int word = font[code * 2 + (row >= 4 ? 1 : 0)];
    int shift = (row & 3) * 8;
    return (word >> shift) & 0xFF;
}

void main() {
    float W = float(cols) * CW + 2.0 * PAD;
    float H = CH + 2.0 * PAD;
    // Bottom-centre, top-origin pixel coords, box-local. Row 1 sits one bar
    // height + 6 px gap above row 0.
    float rowOff = float(rowIdx) * (H + 6.0);
    float x0 = (viewport.x - W) * 0.5;
    float xL = gl_FragCoord.x - x0;
    float yL = (viewport.y - gl_FragCoord.y) - (viewport.y - H - MARGIN - rowOff);
    if (xL < 0.0 || yL < 0.0 || xL >= W || yL >= H) discard;

    vec4 dark = vec4(0.03, 0.05, 0.06, 0.88);
    // Thin border.
    if (xL < 2.0 || yL < 2.0 || xL >= W - 2.0 || yL >= H - 2.0) {
        FragColor = vec4(dark.rgb + 0.16, 0.95);
        return;
    }

    float tx = xL - PAD;
    float ty = yL - PAD;
    int col = int(floor(tx / CW));
    int slot = (col >= 0 && col < cols) ? col / slotw : -1;
    vec4 bg = (slot == sel) ? vec4(0.10, 0.30, 0.12, 0.92) : dark;

    if (tx < 0.0 || ty < 0.0 || ty >= CH || col < 0 || col >= cols) {
        FragColor = bg;
        return;
    }
    int cword = chars[col >> 2];
    int code  = (cword >> ((col & 3) * 8)) & 0xFF;
    if (code == 0 || code == 32) { FragColor = bg; return; }

    int gx   = int((tx - float(col) * CW) / SCALE);
    int frow = int(ty / SCALE);
    int bit  = (glyphRowByte(code, frow) >> gx) & 1;
    vec3 ink = (slot == sel) ? vec3(0.75, 1.0, 0.70) : vec3(0.82, 0.88, 0.95);
    FragColor = (bit != 0) ? vec4(ink, 1.0) : bg;
}
