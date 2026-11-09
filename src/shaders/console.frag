#version 430 core
in vec2 vUV;
out vec4 FragColor;

uniform vec2 viewport;
// Char grid packed 4 ASCII codes per int (row 0 = TOP line). 0/32 = blank.
// 14*72 = 1008 codes -> 252 ints.
uniform int  chars[252];
// Packed font: glyph g rows 0-3 in font[g*2], rows 4-7 in font[g*2+1].
// Each row is one byte; bit 0 (LSB) = leftmost column.
uniform int  font[256];            // 128 glyphs * 2

const int COLS  = 72;
const int ROWS  = 14;
const float SCALE = 2.0;           // px per font pixel
const float CW = 8.0 * SCALE;      // cell width  (16)
const float CH = 8.0 * SCALE;      // cell height (16)

int glyphRowByte(int code, int row) {
    if (code < 0 || code > 127) return 0;
    int word = font[code * 2 + (row >= 4 ? 1 : 0)];
    int shift = (row & 3) * 8;
    return (word >> shift) & 0xFF;
}

// Layout padding (px). The panel floats up off the bottom edge, text is
// inset from the left, and a gap band separates the input line (bottom)
// from the scrollback above it — like Minecraft's chat box.
const float PAD_BOTTOM = 28.0;
const float PAD_LEFT   = 12.0;
const float PAD_TOP    = 6.0;     // breathing room above the top line
const float GAP_MID    = 10.0;    // gap between input line and scrollback

void main() {
    vec2 px = gl_FragCoord.xy;     // origin bottom-left

    float panelH = float(ROWS) * CH + PAD_TOP + GAP_MID;
    float yLocal = px.y - PAD_BOTTOM;
    if (yLocal < 0.0 || yLocal >= panelH) discard;

    // Resolve which text row this pixel belongs to. The input line is the
    // bottom cell; a GAP_MID band sits above it; then the scrollback rows.
    int   row    = -1;     // -1 = no text here (background / gap / top pad)
    float ly     = 0.0;    // pixel offset within the cell (0 at cell bottom)
    bool  isInput = false;
    if (yLocal < CH) {
        row = ROWS - 1; ly = yLocal; isInput = true;       // input line
    } else {
        float yOut = yLocal - CH - GAP_MID;
        if (yOut >= 0.0) {
            int rowFromBotOut = int(yOut / CH);
            if (rowFromBotOut < ROWS - 1) {
                row = (ROWS - 2) - rowFromBotOut;
                ly  = yOut - float(rowFromBotOut) * CH;
            }
        }
    }

    // Background — input line a touch lighter; everything else (incl. gap
    // band and top pad) uses the darker scrollback tone.
    vec4 bg = isInput ? vec4(0.10, 0.12, 0.17, 0.86)
                      : vec4(0.04, 0.05, 0.08, 0.74);
    if (row < 0) { FragColor = bg; return; }

    float xLocal = px.x - PAD_LEFT;
    int col = int(xLocal / CW);
    if (xLocal < 0.0 || col < 0 || col >= COLS) { FragColor = bg; return; }

    int cellIdx = row * COLS + col;
    int cword   = chars[cellIdx >> 2];
    int code    = (cword >> ((cellIdx & 3) * 8)) & 0xFF;
    if (code == 0 || code == 32) { FragColor = bg; return; }

    // Within-cell pixel -> glyph pixel. Font rows are top-to-bottom.
    float lx = xLocal - float(col) * CW;
    int gx = int(lx / SCALE);                    // 0..7 left->right
    int gy = int(ly / SCALE);                    // 0..7 bottom->top
    int frow = 7 - gy;                           // flip to top->bottom
    int rowByte = glyphRowByte(code, frow);
    int bit = (rowByte >> gx) & 1;

    if (bit != 0) {
        // Input line text slightly brighter/greenish so it stands out.
        FragColor = (row == ROWS - 1) ? vec4(0.80, 1.00, 0.80, 1.0)
                                      : vec4(0.85, 0.88, 0.92, 1.0);
    } else {
        FragColor = bg;
    }
}
