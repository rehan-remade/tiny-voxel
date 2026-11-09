#version 430 core
in vec2 vUV;
out vec4 FragColor;

// The "satchel" — the player's bag panel (drawBag). A warm card that FLOATS
// over a dimmed-but-still-living world (never a black modal). Row 0 (slots
// 0..7) is the classic hotbar, drawn as the BOTTOM row under a brass rail;
// rows above are storage. Icons are live voxel-DDA renders shared with
// hud.frag via lib/icon.glsl (per-voxel world colouring; the Materials UBO
// supplies the fallback tint). Geometry constants MUST match
// bagCellRect()/bagHitTest() in app.cpp.
uniform vec2 viewport;
uniform int  bagItem[24];    // per-slot material tint id (0 = empty)
uniform int  bagCount[24];
uniform int  bagIcon[24];     // per-slot icon: (iconMode & 3) | (grade << 2)
uniform int  selected;       // hotbar hand cursor (0..7)
uniform int  hover;          // hovered slot 0..23, or -1
uniform int  tidyHot;        // 1 = cursor over the Tidy button
uniform int  font[256];      // 8x8 bitmap font (packed like farmbar.frag)
uniform int  hdr[9];         // header title, packed 4 chars/int
uniform int  ftr[12];        // footer line (hovered item name + count)
uniform int  tdy[2];         // "TIDY" label

layout(std140, binding = 0) uniform Materials { vec4 matColors[256]; };

// ---- layout (keep in lockstep with app.cpp) ----
const float CELL = 64.0, GAP = 8.0, PAD = 22.0;
const int   COLS = 8, ROWS = 3;
const float HEADER = 34.0, FOOTER = 30.0;
const float GRIDW = float(COLS) * CELL + float(COLS - 1) * GAP;   // 568
const float GRIDH = float(ROWS) * CELL + float(ROWS - 1) * GAP;   // 208
const float CARDW = GRIDW + 2.0 * PAD;                            // 612
const float CARDH = PAD + HEADER + GRIDH + FOOTER + PAD;          // 316
const float TIDYW = 88.0, TIDYH = 24.0;

int glyphRowByte(int code, int row) {
    if (code < 0 || code > 127) return 0;
    int word = font[code * 2 + (row >= 4 ? 1 : 0)];
    return (word >> ((row & 3) * 8)) & 0xFF;
}
int charAt(int arr[9], int i)  { return (arr[i >> 2] >> ((i & 3) * 8)) & 0xFF; }
int charAt12(int arr[12], int i){ return (arr[i >> 2] >> ((i & 3) * 8)) & 0xFF; }

// Ink coverage for one glyph `code` at cell-local pixel (lx,ly), cells `scale`px.
float glyphInk(int code, float lx, float ly, float scale) {
    if (code == 0 || code == 32) return 0.0;
    if (lx < 0.0 || ly < 0.0 || lx >= 8.0 * scale || ly >= 8.0 * scale) return 0.0;
    int gx = int(lx / scale), frow = int(ly / scale);
    return float((glyphRowByte(code, frow) >> gx) & 1);
}

// Voxel icons (pumpkin blob + chunky material block) — live true-isometric
// DDA renders, ONE copy shared with hud.frag via the include.
#include "lib/icon.glsl"

void main() {
    float px = gl_FragCoord.x;
    float py = viewport.y - gl_FragCoord.y;         // top-origin

    float cardX = floor((viewport.x - CARDW) * 0.5);
    float cardY = floor((viewport.y - CARDH) * 0.5);

    // Outside the card: dim the living world (translucent, never black).
    if (px < cardX || py < cardY || px >= cardX + CARDW || py >= cardY + CARDH) {
        FragColor = vec4(0.02, 0.02, 0.03, 0.45);
        return;
    }

    float lxc = px - cardX, lyc = py - cardY;        // card-local

    // Card body: warm leather/wood with a soft inner border + drop-shadow lip.
    vec3  wood   = vec3(0.28, 0.20, 0.13);
    vec3  woodLo = vec3(0.20, 0.14, 0.09);
    float edge = min(min(lxc, CARDW - lxc), min(lyc, CARDH - lyc));
    vec3  card = mix(woodLo, wood, smoothstep(0.0, 10.0, edge));
    if (edge < 3.0) card = vec3(0.44, 0.32, 0.18);   // bright rim
    FragColor = vec4(card, 0.96);

    // Header title.
    {
        float ox = cardX + PAD, oy = cardY + PAD + 4.0, sc = 2.0;
        for (int i = 0; i < 12; ++i) {
            float gxo = ox + float(i) * 8.0 * sc;
            if (glyphInk(charAt(hdr, i), px - gxo, py - oy, sc) > 0.5) {
                FragColor = vec4(0.96, 0.90, 0.72, 1.0); return;
            }
        }
    }
    // Tidy button (top-right of header).
    {
        float tx = cardX + CARDW - PAD - TIDYW;
        float ty = cardY + PAD + (HEADER - TIDYH) * 0.5;
        if (px >= tx && px < tx + TIDYW && py >= ty && py < ty + TIDYH) {
            vec3 b = (tidyHot == 1) ? vec3(0.40, 0.30, 0.16) : vec3(0.24, 0.17, 0.11);
            FragColor = vec4(b, 0.98);
            float lx = px - tx - 14.0, ly = py - ty - 4.0, sc = 2.0;
            for (int i = 0; i < 4; ++i) {
                if (glyphInk(charAt(int[9](tdy[0], tdy[1], 0,0,0,0,0,0,0), i),
                             lx - float(i) * 8.0 * sc, ly, sc) > 0.5) {
                    FragColor = vec4(0.96, 0.90, 0.72, 1.0); return;
                }
            }
            return;
        }
    }

    // Grid area.
    float gridX = cardX + PAD;
    float gridY = cardY + PAD + HEADER;
    float relX = px - gridX, relY = py - gridY;
    if (relX >= 0.0 && relY >= 0.0 && relX < GRIDW && relY < GRIDH) {
        float pitch = CELL + GAP;
        int   col = int(relX / pitch);
        int   vrow = int(relY / pitch);             // visual row from TOP
        float inCellX = relX - float(col) * pitch;
        float inCellY = relY - float(vrow) * pitch;
        if (col >= 0 && col < COLS && vrow >= 0 && vrow < ROWS &&
            inCellX < CELL && inCellY < CELL) {
            int row  = (ROWS - 1) - vrow;            // row 0 = hotbar = bottom
            int slot = row * COLS + col;
            bool isHot = (row == 0);
            bool sel   = (isHot && col == selected);
            bool hov   = (slot == hover);

            // Cell recess.
            vec3 cellBg = isHot ? vec3(0.16, 0.12, 0.08) : vec3(0.13, 0.10, 0.07);
            if (hov) cellBg += 0.06;
            float be = min(min(inCellX, CELL - inCellX), min(inCellY, CELL - inCellY));
            vec3 col3 = mix(cellBg * 0.7, cellBg, smoothstep(0.0, 4.0, be));
            FragColor = vec4(col3, 0.98);

            // Selected/hover border.
            float bw = sel ? 4.0 : (hov ? 2.5 : 0.0);
            if (bw > 0.0 && be < bw) {
                FragColor = sel ? vec4(1.00, 0.86, 0.30, 1.0)
                                : vec4(0.85, 0.80, 0.62, 1.0);
                return;
            }

            int mat   = bagItem[slot];
            int count = bagCount[slot];
            if (mat != 0 && count != 0) {
                int iconMode = bagIcon[slot] & 3;
                int grade    = bagIcon[slot] >> 2;
                // Live isometric voxel DDA icons — same functions as hud.frag.
                // Top-origin cell, so flip Y to +Y-up.
                vec2 sp = vec2(inCellX - CELL * 0.5, (CELL * 0.5) - inCellY);
                vec3 icol;
                if (iconMode == 1) {
                    // Whole-produce blob, shaped per species (mat id) + sized by grade.
                    if (produceIcon(sp, mat, grade, CELL * 0.5, matColors[mat].rgb, icol))
                        FragColor = vec4(icol, 1.0);
                } else {
                    // Material: a chunky voxel chip coloured like the world's
                    // own voxels (lib/icon.glsl).
                    if (materialBlockIcon(sp, mat, CELL * 0.5, matColors[mat].rgb, icol))
                        FragColor = vec4(icol, 1.0);
                }
                // Count digits, right-aligned in the cell's bottom-right.
                float sc = 2.0, dw = 8.0 * sc;
                float digBase = CELL - 4.0 - dw;     // rightmost digit box left edge
                float dy = CELL - 4.0 - 8.0 * sc;
                int v = count, place = 0;
                for (int d = 0; d < 3 && (v > 0 || d == 0); ++d) {
                    int digit = v % 10; v /= 10;
                    float bx = digBase - float(place) * (dw - 3.0);
                    float lx = inCellX - bx, ly = inCellY - dy;
                    if (glyphInk(48 + digit, lx, ly, sc) > 0.5)
                        FragColor = vec4(0.98, 0.96, 0.86, 1.0);
                    place++;
                }
            }

            // Brass rail just above the hotbar row.
            if (row == 0 && inCellY < 2.0) FragColor = vec4(0.72, 0.56, 0.24, 1.0);
            return;
        }
    }

    // Footer: hovered item name + count.
    {
        float ox = cardX + PAD, oy = cardY + PAD + HEADER + GRIDH + 6.0, sc = 2.0;
        for (int i = 0; i < 40; ++i) {
            float gxo = ox + float(i) * 8.0 * sc;
            if (glyphInk(charAt12(ftr, i), px - gxo, py - oy, sc) > 0.5) {
                FragColor = vec4(0.86, 0.82, 0.68, 1.0); return;
            }
        }
    }
}
