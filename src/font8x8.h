#pragma once
// 8x8 monospace bitmap font (printable ASCII 0x20-0x7E). Public-domain
// font8x8_basic by Daniel Hepper (https://github.com/dhepper/font8x8).
// Each glyph is 8 bytes (one per row, top to bottom). Bit 0 (LSB) is the
// LEFTMOST column. Codes outside 0x20-0x7E are all-zero (blank).
extern const unsigned char FONT8X8[128][8];
