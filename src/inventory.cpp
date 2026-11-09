#include "inventory.h"
#include <algorithm>

uint8_t ItemStack::mat() const {
    if (item == 0) return 0;
    if (item < 256) return uint8_t(item);        // materials alias their own id
    return itemDef(item).mat;                     // rich items (bounds-clamped)
}

int Container::add(ItemId item, int n, uint16_t data, int fillStart) {
    if (item == 0 || n <= 0 || slots.empty()) return n;
    int maxStack = int(itemDef(item).maxStack);
    if (maxStack < 1) maxStack = 1;
    // 1) merge into existing stacks of the SAME item AND grade (data). Materials
    //    are data==0 so they merge as before; graded produce merges per grade,
    //    so small/medium/giant pumpkins stay distinct stacks.
    for (auto& s : slots) {
        if (n <= 0) break;
        if (s.item == item && s.data == data && int(s.count) < maxStack) {
            int add = std::min(n, maxStack - int(s.count));
            s.count = uint16_t(int(s.count) + add);
            n -= add;
        }
    }
    // 2) fill EMPTY slots in index order from fillStart, wrapping — so the
    //    caller can bias pickups toward storage and leave the hotbar clear.
    int N = int(slots.size());
    if (fillStart < 0) fillStart = 0;
    for (int k = 0; k < N && n > 0; ++k) {
        ItemStack& s = slots[(fillStart + k) % N];
        if (s.isEmpty()) {
            int add = std::min(n, maxStack);
            s.item = item; s.count = uint16_t(add); s.data = data;
            n -= add;
        }
    }
    return n;   // leftover that didn't fit (bag full)
}

int Container::count(ItemId item) const {
    int t = 0;
    for (const auto& s : slots) if (s.item == item) t += int(s.count);
    return t;
}

void Container::tidy() {
    // Collect non-empty, sort by (kind, id, data), coalesce fungible stacks,
    // let empties fall to the end — the one-tap "settle".
    std::vector<ItemStack> items;
    for (const auto& s : slots) if (!s.isEmpty()) items.push_back(s);
    std::stable_sort(items.begin(), items.end(), [](const ItemStack& a, const ItemStack& b) {
        const ItemDef& da = itemDef(a.item);
        const ItemDef& db = itemDef(b.item);
        if (da.kind != db.kind) return da.kind < db.kind;
        if (a.item != b.item)   return a.item < b.item;
        return a.data < b.data;
    });
    std::vector<ItemStack> out;
    for (const auto& s : items) {
        if (!out.empty() && out.back().item == s.item && out.back().data == s.data) {
            int maxStack = int(itemDef(s.item).maxStack);
            int can = maxStack - int(out.back().count);
            int mv  = std::min(int(s.count), can);
            out.back().count = uint16_t(int(out.back().count) + mv);
            int rem = int(s.count) - mv;
            if (rem > 0) { ItemStack t = s; t.count = uint16_t(rem); out.push_back(t); }
        } else {
            out.push_back(s);
        }
    }
    for (size_t i = 0; i < slots.size(); ++i)
        slots[i] = (i < out.size()) ? out[i] : ItemStack{};
}

int Hotbar::giveItem(ItemId item, int n, uint16_t data) {
    // Pickups fill the HOTBAR row first (so an empty hotbar shows what you just
    // got), then spill into storage. Starter stock uses giveToStorage instead.
    return bag.add(item, n, data, 0);
}
