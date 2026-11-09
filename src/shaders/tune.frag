#version 430 core
in vec2 vUV;
out vec4 FragColor;

// /tune slider-panel overlay (drawTunePanel). Same 8x8 bitmap-font scheme as
// debug.frag / the console, anchored to the LEFT edge at mid height, with a
// highlighted selected row. Grid dims must match drawTunePanel's COLS/ROWS.
uniform vec2 viewport;
uniform int  chars[252];   // >= ceil(maxRows*COLS / 4); lighting page is TWEAK_N+2
                           // rows (25 -> 225 ints), sized to 28 rows of headroom
uniform int  font[256];
uniform int  selRow;       // highlighted grid row (header = 0)
uniform int  rows;         // active row count (page-dependent: lighting vs materials)

const int   COLS   = 36;
const float SCALE  = 2.0;
const float CW     = 8.0 * SCALE;
const float CH     = 8.0 * SCALE;
const float MARGIN = 12.0;

int glyphRowByte(int code, int row) {
    if (code < 0 || code > 127) return 0;
    int word = font[code * 2 + (row >= 4 ? 1 : 0)];
    int shift = (row & 3) * 8;
    return (word >> shift) & 0xFF;
}

void main() {
    vec2 px = gl_FragCoord.xy;
    float panelW = float(COLS) * CW;
    float panelH = float(rows) * CH;
    float xL = px.x - MARGIN;                                  // left-anchored
    float yL = (viewport.y + panelH) * 0.5 - px.y;             // vertically centred
    if (xL < 0.0 || yL < 0.0 || xL >= panelW || yL >= panelH) discard;

    int col = int(xL / CW);
    int row = int(yL / CH);
    vec4 bg = (row == selRow) ? vec4(0.10, 0.16, 0.10, 0.88)
                              : vec4(0.03, 0.04, 0.06, 0.82);
    if (col < 0 || col >= COLS || row < 0 || row >= rows) { FragColor = bg; return; }

    int cellIdx = row * COLS + col;
    int cword   = chars[cellIdx >> 2];
    int code    = (cword >> ((cellIdx & 3) * 8)) & 0xFF;
    if (code == 0 || code == 32) { FragColor = bg; return; }

    float lx = xL - float(col) * CW;
    float ly = yL - float(row) * CH;
    int gx   = int(lx / SCALE);
    int frow = int(ly / SCALE);
    int bit  = (glyphRowByte(code, frow) >> gx) & 1;

    vec3 ink = (row == 0) ? vec3(1.0, 0.95, 0.6)
             : (row == selRow) ? vec3(0.75, 1.0, 0.75) : vec3(0.85, 0.90, 0.95);
    FragColor = (bit != 0) ? vec4(ink, 1.0) : bg;
}
