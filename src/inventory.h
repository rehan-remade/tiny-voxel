#pragma once
#include <cstdint>
#include <vector>
#include "items.h"

// A stack of one item kind. `item` is an ItemId (0 = empty; 1..255 alias MAT_*;
// >=256 rich items — see items.h). `data` is per-instance payload (growth /
// variant / durability, future) — 0 means fungible, so two data==0 stacks of
// the same item merge; instanced stacks (data!=0) never coalesce.
struct ItemStack {
    ItemId   item  = 0;
    uint16_t count = 0;
    uint16_t data  = 0;

    bool    isEmpty() const { return item == 0 || count == 0; }
    uint8_t mat() const;    // tint / voxel bridge (defined in inventory.cpp)
};

// The one reusable container. The player bag is a Container; chests, the
// crafting grid, and (reserved) equipment will be Containers too — so crafting
// and storage are additive, operating on these same three ops.
struct Container {
    std::vector<ItemStack> slots;
    Container() = default;
    explicit Container(int n) : slots(size_t(n < 0 ? 0 : n)) {}

    // Add n of an item: merges into matching fungible stacks anywhere, then
    // fills EMPTY slots in index order starting at `fillStart` (wrapping). Row
    // 0 of the player bag is the hotbar, so the Hotbar passes fillStart=COLS to
    // land pickups in storage first and keep the hotbar player-curated.
    int  add(ItemId item, int n, uint16_t data = 0, int fillStart = 0);  // returns leftover
    int  count(ItemId item) const;
    void tidy();                                       // group by kind/id, coalesce, empties last
};

// The player's inventory. Row 0 (slots 0..COLS-1) IS the classic iso-cube
// hotbar; the remaining rows are bag storage — one container, "hotbar is the
// bottom row" (Minecraft/Stardew unity). `selected` is the in-hand cursor,
// constrained to the hotbar row. The giveItem/selectedStack surface is
// preserved so the shipped harvest/forestry deposit paths are untouched.
struct Hotbar {
    static constexpr int      COLS      = 8;          // hotbar-row width
    static constexpr int      ROWS      = 3;          // total rows (row 0 = hotbar)
    static constexpr int      SLOTS     = COLS;       // hotbar-row width (HUD strip + number keys)
    static constexpr int      BAG       = COLS * ROWS; // total slots (24)
    static constexpr uint16_t MAX_STACK = 999;

    Container bag{BAG};
    int       selected = 0;                            // 0..COLS-1

    int  giveItem(ItemId item, int n, uint16_t data = 0);   // fills the HOTBAR row first
    int  giveItem(uint8_t mat, int n) { return giveItem(ItemId(mat), n, 0); }  // material compat
    int  giveToStorage(ItemId item, int n, uint16_t data = 0) {  // fills storage rows first
        return bag.add(item, n, data, COLS);
    }
    void tidy() { bag.tidy(); }

    const ItemStack& selectedStack() const { return bag.slots[selected]; }
    ItemStack&       slot(int i)       { return bag.slots[i]; }
    const ItemStack& slot(int i) const { return bag.slots[i]; }
};
