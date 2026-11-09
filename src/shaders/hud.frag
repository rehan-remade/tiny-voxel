#version 430 core
in vec2 vUV;
out vec4 FragColor;

uniform vec2 viewport;
uniform int  hudMat[8];
uniform int  hudCount[8];
uniform int  hudIcon[8];   // per-slot icon: (iconMode & 3) | (grade << 2)
uniform int  hudSelected;
uniform int  trayOn;       // 0 = hide the tray body (satchel open); the badge
                           // and name label above still draw
uniform int  devMode;
uniform int  hudName[8];  // selected item name, packed 4 chars/int (32 max)
uniform float hudNameFade; // 0..1 opacity for the held-item name label
uniform int  hudFont[256]; // 8x8 bitmap font (packed like farmbar.frag)

layout(std140, binding = 0) uniform Materials { vec4 matColors[256]; };

int hudGlyphRow(int code, int row) {
    if (code < 0 || code > 127) return 0;
    return (hudFont[code * 2 + (row >= 4 ? 1 : 0)] >> ((row & 3) * 8)) & 0xFF;
}

// 3x5 pixel font for "DEV". Returns 1 if the bit is set.
int letterBit(int letter, int row, int col) {
    int p = 0;
    if (letter == 0) {        // D
        if (row == 0 || row == 4) p = 6; // 110
        else                     p = 5; // 101
    } else if (letter == 1) { // E
        if (row == 0 || row == 4) p = 7; // 111
        else if (row == 2)        p = 6; // 110
        else                       p = 4; // 100
    } else if (letter == 2) { // V
        if (row == 4)             p = 2; // 010
        else                       p = 5; // 101
    }
    return (p >> (2 - col)) & 1;
}

// Voxel icons (pumpkin blob + chunky material block) — live true-isometric
// DDA renders, ONE copy shared with bag.frag via the include.
#include "lib/icon.glsl"

void main() {
    vec2 px = gl_FragCoord.xy;

    // -- Dev-mode badge: small yellow "DEV" pill in the top-left ------------
    if (devMode != 0) {
        float scale  = 4.0;
        float letW   = 3.0 * scale;
        float letH   = 5.0 * scale;
        float spacer = scale;
        float bgPadX = 10.0;
        float bgPadY = 7.0;
        float originX = 22.0;
        float originY = viewport.y - 22.0 - letH;          // bottom-left of letter row
        float totalW = 3.0 * letW + 2.0 * spacer;

        if (px.x >= originX - bgPadX && px.x < originX + totalW + bgPadX &&
            px.y >= originY - bgPadY && px.y < originY + letH + bgPadY) {
            FragColor = vec4(0.05, 0.05, 0.07, 0.82);
            float lx = px.x - originX;
            float ly = (originY + letH) - px.y;             // 0 at top of letter row
            if (lx >= 0.0 && lx < totalW && ly >= 0.0 && ly < letH) {
                float charPitch = letW + spacer;
                int   charIdx   = int(floor(lx / charPitch));
                float withinX   = lx - float(charIdx) * charPitch;
                if (withinX < letW) {
                    int col = int(floor(withinX / scale));
                    int row = int(floor(ly / scale));
                    if (letterBit(charIdx, row, col) != 0) {
                        FragColor = vec4(1.00, 0.85, 0.20, 1.0);
                    }
                }
            }
            return;
        }
    }

    // -- Held-item name: fades in just above the hotbar on a selection change,
    // on a soft dark pill so it reads over bright terrain (top-down diorama). --
    if (hudNameFade > 0.001) {
        int n = 0;
        for (int i = 0; i < 32; ++i) {
            int code = (hudName[i >> 2] >> ((i & 3) * 8)) & 0xFF;
            if (code == 0) break; n++;
        }
        if (n > 0) {
            float sc = 2.0;
            float labelW = float(n) * 8.0 * sc, labelH = 8.0 * sc;
            float ox = floor((viewport.x - labelW) * 0.5);
            float oy = 28.0 + 128.0 + 14.0;    // above the wooden tray (trayY + trayH + 14)
            float padX = 10.0, padY = 5.0;
            if (px.x >= ox - padX && px.x < ox + labelW + padX &&
                px.y >= oy - padY && px.y < oy + labelH + padY) {
                FragColor = vec4(0.04, 0.04, 0.06, 0.5 * hudNameFade);   // pill
                if (px.x >= ox && px.x < ox + labelW && px.y >= oy && px.y < oy + labelH) {
                    int idx = int((px.x - ox) / (8.0 * sc));
                    float lx = (px.x - ox) - float(idx) * 8.0 * sc;
                    int frow = int((oy + labelH - px.y) / sc);   // 0 = top scanline
                    int gx = int(lx / sc);
                    int code = (hudName[idx >> 2] >> ((idx & 3) * 8)) & 0xFF;
                    if (((hudGlyphRow(code, frow) >> gx) & 1) != 0)
                        FragColor = vec4(0.98, 0.95, 0.82, hudNameFade);
                }
                return;
            }
        }
    }

    // -- Hotbar: a carved wooden TRAY (same warm-wood family as the satchel /
    // map), 8 felt cells with slot numbers, gold selection, voxel-chip icons
    // and a readable count. Reads as one crafted object, not floating tiles. --
    // Suppressed while the satchel is open (its own bottom row stands in) —
    // the pass still runs then so the badge above never blinks out.
    if (trayOn == 0) discard;
    const float CELL = 96.0, GAP = 8.0, TPAD = 16.0;   // bigger: room for voxel icons
    float trayW = 8.0 * CELL + 7.0 * GAP + 2.0 * TPAD;
    float trayH = CELL + 2.0 * TPAD;
    float trayX = floor((viewport.x - trayW) * 0.5);
    float trayY = 28.0;                              // from the bottom edge
    if (px.x < trayX || px.x >= trayX + trayW ||
        px.y < trayY || px.y >= trayY + trayH) discard;

    // Wood frame: dark body, brighter carved rim (matches bag.frag).
    float lxr = px.x - trayX, lyr = px.y - trayY;
    float fedge = min(min(lxr, trayW - lxr), min(lyr, trayH - lyr));
    vec3  wood   = vec3(0.28, 0.20, 0.13);
    vec3  woodLo = vec3(0.20, 0.14, 0.09);
    vec3  frame  = mix(woodLo, wood, smoothstep(0.0, 10.0, fedge));
    if (fedge < 3.0) frame = vec3(0.44, 0.32, 0.18);   // bright bevel rim
    FragColor = vec4(frame, 0.94);

    float gx0 = px.x - (trayX + TPAD);
    float cy0 = px.y - (trayY + TPAD);
    if (gx0 < 0.0 || cy0 < 0.0 || cy0 >= CELL) return;
    float pitch = CELL + GAP;
    int   slot  = int(gx0 / pitch);
    float inX   = gx0 - float(slot) * pitch;
    if (slot < 0 || slot >= 8 || inX >= CELL) return;   // in a gap -> frame shows

    float inY   = cy0;              // bottom-origin, 0 = cell bottom
    float topY  = CELL - inY;       // 0 = cell top (for glyph rows)
    bool  sel   = (slot == hudSelected);
    int   mat   = hudMat[slot];
    int   count = hudCount[slot];

    // Felt-recessed cell with a soft inner bevel.
    float be = min(min(inX, CELL - inX), min(inY, CELL - inY));
    vec3  cellBg = vec3(0.17, 0.13, 0.09);
    vec3  cell3  = mix(cellBg * 0.62, cellBg, smoothstep(0.0, 4.0, be));
    FragColor = vec4(cell3, 0.98);

    // Selected slot: warm gold border.
    if (sel && be < 4.0) { FragColor = vec4(1.00, 0.86, 0.30, 1.0); return; }

    // Icon, bottom-origin (bright face / top is +y). Both kinds are live
    // isometric voxel DDA — real chunky voxel models in the world's 3-face
    // lighting. Bottom-origin cell: +Y already up.
    if (mat != 0 && count != 0) {
        int iconMode = hudIcon[slot] & 3;
        int grade    = hudIcon[slot] >> 2;
        vec2 sp = vec2(inX, inY) - vec2(CELL * 0.5, CELL * 0.5);
        vec3 icol;
        if (iconMode == 1) {
            // Whole-produce blob, shaped per species (mat id) + sized by grade.
            if (produceIcon(sp, mat, grade, CELL * 0.5, matColors[mat].rgb, icol))
                FragColor = vec4(icol, 1.0);
        } else {
            // Material: a chunky voxel chip coloured like the world's own
            // voxels (grass tops, stone grain, bark plates — lib/icon.glsl).
            if (materialBlockIcon(sp, mat, CELL * 0.5, matColors[mat].rgb, icol))
                FragColor = vec4(icol, 1.0);
        }
        // Count digits, right-aligned in the bottom-right.
        float sc = 2.0, dw = 8.0 * sc;
        float digBase = CELL - 5.0 - dw, dy = CELL - 5.0 - dw;
        int v = count, place = 0;
        for (int d = 0; d < 3 && (v > 0 || d == 0); ++d) {
            int digit = v % 10; v /= 10;
            float bx = digBase - float(place) * (dw - 3.0);
            float dgx = inX - bx, dgy = topY - dy;
            if (dgx >= 0.0 && dgy >= 0.0 && dgx < dw && dgy < dw) {
                int rr = int(dgy / sc), cc2 = int(dgx / sc);
                if (((hudGlyphRow(48 + digit, rr) >> cc2) & 1) != 0)
                    FragColor = vec4(0.98, 0.96, 0.86, 1.0);
            }
            place++;
        }
    }

    // Slot number (1..8) in the top-left corner, tucked and quiet.
    {
        float sc = 1.5, nx = inX - 5.0, ny = topY - 4.0;
        if (nx >= 0.0 && ny >= 0.0 && nx < 8.0 * sc && ny < 8.0 * sc) {
            int rr = int(ny / sc), cc2 = int(nx / sc);
            if (((hudGlyphRow(49 + slot, rr) >> cc2) & 1) != 0)
                FragColor = vec4(0.82, 0.74, 0.54, 1.0);
        }
    }
    return;
}
