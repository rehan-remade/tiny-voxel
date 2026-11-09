#pragma once
// Item registry — the def-indexed identity layer above raw materials.
//
// An inventory item is an ItemId (uint16). Ids 1..255 ALIAS the MAT_* material
// enum directly (so a material is its own item, no migration, and its icon
// keys straight off the material id); ids >= 256 are "rich" items (produce,
// tools, seeds, and — reserved — wearables) that carry their own def row. ONE table,
// g_items[], describes every kind; adding an item is (usually) one row.
//
// This is the foundation the inventory / crafting / (future) equipment systems
// all key off. See inventory.h (ItemStack/Container/Hotbar).
#include <cstdint>

using ItemId = uint16_t;

enum ItemKind : uint8_t {
    IK_VOID = 0,   // empty / unregistered
    IK_VOXEL,      // a placeable world material
    IK_PRODUCE,    // harvested goods (whole pumpkin, ...)
    IK_TOOL,       // hoe / basket / seed pouch (non-stacking)
    IK_SEED,       // plantable seed
    IK_WEARABLE,   // RESERVED: equipment (no system yet; the id space is held)
};

// How the icon draws (bag.frag / hud.frag branch on this; the raytracers live
// in shaders/lib/icon.glsl). Materials get the chunky voxel-chip block,
// per-voxel coloured like the world; rich kinds add SDF shapes later.
enum IconMode : uint8_t {
    ICON_CUBE = 0,   // chunky voxel block of def.mat, world-coloured per voxel
    ICON_BLOB,       // rounded voxel blob (bulky produce) — reads as "whole/heavy"
    ICON_TOOL,       // small tool glyph
    ICON_SEED,       // seed cluster
};

// Behaviour + crafting-category routing. Kept separate from `category` so the
// crafting matcher's "any wood / any seed" tags never crowd the action flags.
enum ItemFlag : uint32_t {
    IF_TOOL      = 1u << 1,
    IF_TILLS     = 1u << 2,
    IF_PLANTS    = 1u << 3,
    IF_HARVESTS  = 1u << 4,
    IF_DROPWORLD = 1u << 5,   // dropping spawns a world object (needs an entity system; later)
};

struct ItemDef {
    const char* name     = "";
    ItemKind    kind     = IK_VOID;
    IconMode    icon     = ICON_CUBE;
    uint16_t    maxStack = 999;   // 999 materials, 1 tools, small for bulky produce
    uint8_t     mat      = 0;     // MAT_* — drives the icon tint for voxel items
    uint8_t     wearSlot = 0;     // 0 = none. RESERVED for equipment; costs nothing to hold now.
    uint32_t    flags    = 0;
    uint8_t     category = 0;     // RESERVED crafting category ("any wood", ...) — separate from flags
};

constexpr int ITEM_COUNT = 512;   // 0..255 materials, 256.. rich items
extern ItemDef g_items[ITEM_COUNT];

// Rich (non-material) item ids. Authored in initItems(); most are reserved for
// later slices (tools/crafting), but the id space + defs exist now so nothing
// downstream needs a redesign to use them.
enum : ItemId {
    ITEM_WHOLE_PUMPKIN = 256,
    ITEM_HOE,
    ITEM_SEED_POUCH,
    ITEM_BASKET,
    // Whole vegetables (appended — ids above stay stable). Keep this block in
    // CropSpecies order minus pumpkin; itemForCrop() maps species -> item.
    ITEM_WHOLE_CARROT,
    ITEM_WHOLE_TOMATO,
    ITEM_WHOLE_CABBAGE,
    ITEM_WHOLE_CORN,
    ITEM_WHOLE_RADISH,
};

// Produce item for a CropSpecies index (farm.h) — pumpkin is the odd one out
// (its id predates the vegetable range).
inline ItemId itemForCrop(int species) {
    return species == 0 ? ITEM_WHOLE_PUMPKIN
                        : ItemId(ITEM_WHOLE_CARROT + (species - 1));
}

// Populate g_items: auto-register materials 1..255 from the material table, then
// author the rich items. Call ONCE at startup (g_materials is a static global,
// so any time before first inventory use is fine).
void initItems();

inline const ItemDef& itemDef(ItemId id) {
    return g_items[id < ITEM_COUNT ? id : 0];
}

// Display name accounting for per-instance grade (`data`): e.g. a pumpkin reads
// "Small Pumpkin" / "Pumpkin" / "Giant Pumpkin" by its grade. Falls back to the
// def name for everything else. This is the general hook for graded produce.
const char* itemDisplayName(ItemId item, uint16_t data);
