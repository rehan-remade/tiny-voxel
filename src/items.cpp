#include "items.h"
#include "common.h"
#include "materials.h"

#include <cstdio>

ItemDef g_items[ITEM_COUNT];

// Friendly names for the materials the player actually handles. The rest fall
// back to "Block" — a full 256-name table isn't worth maintaining, and unnamed
// materials rarely reach the bag.
static const char* materialName(int m) {
    switch (m) {
        case MAT_DIRT:            return "Dirt";
        case MAT_GRASS:           return "Grass";
        case MAT_STONE:           return "Stone";
        case MAT_WOOD:            return "Wood";
        case MAT_LEAVES:          return "Leaves";
        case MAT_WATER:           return "Water";
        case MAT_GREYSTONE:       return "Greystone";
        case MAT_DARKSTONE:       return "Darkstone";
        case MAT_BEDROCK:         return "Bedrock";
        case MAT_GRASS_BLADE:     return "Grass Tuft";
        case MAT_FLOWER_RED:      return "Red Flower";
        case MAT_FLOWER_YELLOW:   return "Yellow Flower";
        case MAT_FLOWER_BLUE:     return "Blue Flower";
        case MAT_FLOWER_WHITE:    return "White Flower";
        case MAT_CRYSTAL:         return "Glow Crystal";
        case MAT_MUSHROOM_CAP:    return "Mushroom";
        case MAT_MUSHROOM_CAP2:   return "Brown Mushroom";
        case MAT_CAVE_MOSS:       return "Cave Moss";
        case MAT_VINE:            return "Vine";
        case MAT_SAND:            return "Sand";
        case MAT_SAND_STATIC:     return "Sand";
        case MAT_SANDSTONE:       return "Sandstone";
        case MAT_DESERT_PAVEMENT: return "Gravel";
        case MAT_CACTUS_FLESH:    return "Cactus";
        case MAT_SHRUB_SAGE:      return "Sagebrush";
        case MAT_DESERT_WOOD:     return "Dry Wood";
        case MAT_FARMLAND:        return "Tilled Soil";
        case MAT_CROP_VINE:       return "Crop Vine";
        case MAT_CROP_LEAF:       return "Crop Leaf";
        case MAT_PUMPKIN:         return "Pumpkin";
        case MAT_CROP_FLOWER:     return "Blossom";
        case MAT_CARROT:          return "Carrot";
        case MAT_TOMATO:          return "Tomato";
        case MAT_CABBAGE:         return "Cabbage";
        case MAT_CORN:            return "Corn";
        case MAT_RADISH:          return "Radish";
        default:                  return "Block";
    }
}

void initItems() {
    for (int i = 0; i < ITEM_COUNT; ++i) g_items[i] = ItemDef{};

    // Materials 1..255 -> voxel items whose icon is their own colour.
    // No second table to hand-maintain; the icon is always world-synced.
    for (int m = 1; m < 256; ++m) {
        ItemDef& d = g_items[m];
        d.name     = materialName(m);
        d.kind     = IK_VOXEL;
        d.icon     = ICON_CUBE;
        d.maxStack = 999;
        d.mat      = uint8_t(m);
    }

    // Whole produce — collected as WHOLE objects (grade in ItemStack.data),
    // shown with per-species blob icons (ICON_BLOB dispatches its shape on
    // `mat` in lib/icon.glsl), not as flesh voxels.
    //                         name         kind        icon       max mat           wear flags          cat
    g_items[ITEM_WHOLE_PUMPKIN] = { "Pumpkin", IK_PRODUCE, ICON_BLOB, 99, MAT_PUMPKIN, 0, IF_DROPWORLD, 0 };
    g_items[ITEM_WHOLE_CARROT]  = { "Carrot",  IK_PRODUCE, ICON_BLOB, 99, MAT_CARROT,  0, IF_DROPWORLD, 0 };
    g_items[ITEM_WHOLE_TOMATO]  = { "Tomato",  IK_PRODUCE, ICON_BLOB, 99, MAT_TOMATO,  0, IF_DROPWORLD, 0 };
    g_items[ITEM_WHOLE_CABBAGE] = { "Cabbage", IK_PRODUCE, ICON_BLOB, 99, MAT_CABBAGE, 0, IF_DROPWORLD, 0 };
    g_items[ITEM_WHOLE_CORN]    = { "Corn",    IK_PRODUCE, ICON_BLOB, 99, MAT_CORN,    0, IF_DROPWORLD, 0 };
    g_items[ITEM_WHOLE_RADISH]  = { "Radish",  IK_PRODUCE, ICON_BLOB, 99, MAT_RADISH,  0, IF_DROPWORLD, 0 };
    // Tools (reserved — defs exist so the model is proven extensible; not given
    // to the player yet).
    g_items[ITEM_HOE]        = { "Hoe",        IK_TOOL, ICON_TOOL, 1, MAT_WOOD,        0, IF_TOOL | IF_TILLS,    0 };
    g_items[ITEM_SEED_POUCH] = { "Seed Pouch", IK_TOOL, ICON_TOOL, 1, MAT_GRASS,       0, IF_TOOL | IF_PLANTS,   0 };
    g_items[ITEM_BASKET]     = { "Basket",     IK_TOOL, ICON_TOOL, 1, MAT_DESERT_WOOD, 0, IF_TOOL | IF_HARVESTS, 0 };
}

const char* itemDisplayName(ItemId item, uint16_t data) {
    // Graded produce (data 0 small / 1 plain / 2 giant): derive the name from
    // the def instead of a hand-maintained per-species table — new produce
    // items grade automatically. The plain grade returns the def's stable
    // literal; Small/Giant format into a small RING of buffers because
    // callers hold the returned pointer for the rest of the frame (hotbar
    // label + pickup toast + bag footer can be alive at once).
    const ItemDef& d = itemDef(item);
    if (d.kind == IK_PRODUCE && (data == 0 || data == 2)) {
        static char ring[4][32];
        static int  slot = 0;
        char* buf = ring[slot++ & 3];
        std::snprintf(buf, sizeof(ring[0]), "%s %s",
                      data == 0 ? "Small" : "Giant", d.name);
        return buf;
    }
    return d.name;
}
