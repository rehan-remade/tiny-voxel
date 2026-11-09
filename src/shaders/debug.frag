#version 430 core
in vec2 vUV;
out vec4 FragColor;

// Compact top-left debug text overlay (GPU pass timers). Mirrors the console's
// 8x8 bitmap-font scheme but anchored to the TOP-right and on its own small grid.
uniform vec2 viewport;
// Char grid packed 4 ASCII codes per int (row 0 = TOP line). 0/32 = blank.
// ROWS*COLS = 8*16 = 128 codes -> 32 ints.
uniform int  chars[32];
// Packed font: glyph g rows 0-3 in font[g*2], rows 4-7 in font[g*2+1].
// Each row is one byte; bit 0 (LSB) = leftmost column.
uniform int  font[256];

const int   COLS   = 16;
const int   ROWS   = 8;
const float SCALE  = 2.0;            // px per font pixel
const float CW     = 8.0 * SCALE;    // cell width  (16)
const float CH     = 8.0 * SCALE;    // cell height (16)
const float MARGIN = 10.0;           // inset from the top-right corner

int glyphRowByte(int code, int row) {
    if (code < 0 || code > 127) return 0;
    int word = font[code * 2 + (row >= 4 ? 1 : 0)];
    int shift = (row & 3) * 8;
    return (word >> shift) & 0xFF;
}

void main() {
    vec2 px = gl_FragCoord.xy;        // origin bottom-left
    float panelW = float(COLS) * CW;
    float panelH = float(ROWS) * CH;
    // Anchored to the TOP-RIGHT so it clears the top-left DEV badge.
    float xL = px.x - (viewport.x - MARGIN - panelW);   // 0 at panel's left edge
    float yL = (viewport.y - px.y) - MARGIN;            // 0 at TOP, grows downward
    if (xL < 0.0 || yL < 0.0 || xL >= panelW || yL >= panelH) discard;

    vec4 bg = vec4(0.03, 0.04, 0.06, 0.80);
    int col = int(xL / CW);
    int row = int(yL / CH);           // row 0 at the top
    if (col < 0 || col >= COLS || row < 0 || row >= ROWS) { FragColor = bg; return; }

    int cellIdx = row * COLS + col;
    int cword   = chars[cellIdx >> 2];
    int code    = (cword >> ((cellIdx & 3) * 8)) & 0xFF;
    if (code == 0 || code == 32) { FragColor = bg; return; }

    float lx = xL - float(col) * CW;
    float ly = yL - float(row) * CH;  // 0 at cell top, grows downward
    int gx   = int(lx / SCALE);       // 0..7 left->right
    int frow = int(ly / SCALE);       // 0..7 top->bottom (font rows are top->bottom)
    int bit  = (glyphRowByte(code, frow) >> gx) & 1;

    FragColor = (bit != 0) ? vec4(0.80, 1.00, 0.80, 1.0) : bg;
}
