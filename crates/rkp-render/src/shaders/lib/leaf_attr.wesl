// Per-instance leaf-attribute / leaf-color fetch helpers shared by
// the host march, the shadow trace, and any future per-pixel passes
// that need overlay-aware sampling.
//
// Consumer-supplied bindings (each pass declares them at its own
// `@group/@binding` site — names must match for the inlined bodies
// to resolve under ManglerKind::None):
//
//   var<storage, read> leaf_attr_pool:   array<LeafAttr>
//   var<storage, read> color_pool_data:  array<u32>
//   var<storage, read> instance_overlay: array<OverlayEntry>
//   var<storage, read> instance_sculpt:  array<u32>
//
// The overlay slice is per-instance: a leaf is "painted" iff a
// matching `OverlayEntry.leaf_slot` exists in the half-open range
// `[inst.overlay_offset, inst.overlay_offset + inst.overlay_count)`.
// Entries within that slice are kept sorted by `leaf_slot` so the
// lookup is a binary search.
//
// The sculpt slice mirrors the same shape — sorted ascending u32
// `leaf_attr_id`s in `[inst.sculpt_offset, .. + inst.sculpt_count)`
// marking leaves that have been carved away (Phase A — Carve only).
// `is_leaf_removed` returns `true` iff the slot is in that set;
// consumers treat removed leaves as Empty (mesh FS discards, march
// skips the descent, shadow trace treats the cell as non-occluding).

import package::lib::types::RkpInstance;
import package::lib::types::LeafAttr;

// Binary search the per-instance overlay slice for `leaf_slot`.
// Returns the absolute index into `instance_overlay[]`, or
// `0xFFFFFFFFu` (= "no override") if the slot isn't painted.
fn fetch_overlay_index(inst: RkpInstance, leaf_slot: u32) -> u32 {
    if (inst.overlay_count == 0u) {
        return 0xFFFFFFFFu;
    }
    var lo: u32 = 0u;
    var hi: u32 = inst.overlay_count;
    loop {
        if (lo >= hi) { break; }
        let mid = (lo + hi) >> 1u;
        let e = instance_overlay[inst.overlay_offset + mid];
        if (e.leaf_slot < leaf_slot) {
            lo = mid + 1u;
        } else if (e.leaf_slot > leaf_slot) {
            hi = mid;
        } else {
            return inst.overlay_offset + mid;
        }
    }
    return 0xFFFFFFFFu;
}

// Per-leaf attr (normal + materials) for the given instance,
// preferring the overlay if the slot has been painted.
fn fetch_leaf_attr_for(inst: RkpInstance, leaf_slot: u32) -> LeafAttr {
    let idx = fetch_overlay_index(inst, leaf_slot);
    if (idx != 0xFFFFFFFFu) {
        let e = instance_overlay[idx];
        return LeafAttr(e.normal_oct, e.material_packed);
    }
    return leaf_attr_pool[leaf_slot];
}

// Per-leaf packed color for the given instance. Returns 0
// (= "use material base color") when neither the overlay nor the
// asset pool has an override for this slot.
fn fetch_leaf_color_for(inst: RkpInstance, leaf_slot: u32) -> u32 {
    let idx = fetch_overlay_index(inst, leaf_slot);
    if (idx != 0xFFFFFFFFu) {
        return instance_overlay[idx].color_packed;
    }
    return color_pool_data[leaf_slot];
}

// True iff `leaf_slot` has been carved away on this instance.
// Phase A sculpt overlay — binary search the per-instance removal
// set (`instance_sculpt`, sorted ascending). Short-circuits when
// the instance has no carved leaves at all (`sculpt_count == 0`),
// which is the common case.
fn is_leaf_removed(inst: RkpInstance, leaf_slot: u32) -> bool {
    if (inst.sculpt_count == 0u) {
        return false;
    }
    var lo: u32 = 0u;
    var hi: u32 = inst.sculpt_count;
    loop {
        if (lo >= hi) { break; }
        let mid = (lo + hi) >> 1u;
        let v = instance_sculpt[inst.sculpt_offset + mid];
        if (v < leaf_slot) {
            lo = mid + 1u;
        } else if (v > leaf_slot) {
            hi = mid;
        } else {
            return true;
        }
    }
    return false;
}
