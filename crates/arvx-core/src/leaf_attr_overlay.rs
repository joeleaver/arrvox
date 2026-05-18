//! Per-instance sparse paint overlay.
//!
//! [`LeafAttrOverlay`] stores per-leaf paint edits scoped to a single
//! scene instance, decoupled from the asset's shared [`LeafAttrPool`].
//! Multiple instances of the same .arvx asset can each carry their own
//! overlay; the asset's pool stays immutable post-load.
//!
//! This is the data store behind the "paint one of N shared-asset
//! instances paints only that one" fix (memory: project_phase_1_2_shipped
//! Phase 3). Without per-instance overlays, painting any instance of a
//! shared-`octree_root` asset writes into the shared pool and visually
//! affects every sibling.
//!
//! # Layout
//!
//! Entries are kept sorted by `leaf_slot`. Lookups are O(log N) binary
//! search; inserts are O(log N) for position + O(N) for shift on
//! existing entries (or O(1) replace if the slot already exists). For
//! the painting workload — thousands of leaves edited over many strokes
//! — this is the simplest representation that works. Sparse octrees /
//! per-brick override masks are listed in
//! `project_unified_renderer_rethink.md` as future-if-needed.
//!
//! # GPU layout
//!
//! [`OverlayEntry`] is the upload-shape: 16 bytes per entry, padded so
//! WGSL `array<OverlayEntry>` reads align cleanly. The CPU side stores
//! the same data unpacked to keep the API ergonomic.
//!
//! ```text
//! word 0: leaf_slot              (u32)
//! word 1: normal_oct             (u32) — same packing as LeafAttr
//! word 2: material packed        (u16 primary | u16 secondary_blend)
//! word 3: color_packed           (u32) — same packing as LeafAttrPool.colors
//! ```

use crate::leaf_attr::LeafAttr;
use bytemuck::{Pod, Zeroable};

/// GPU-side overlay entry. 16 bytes; one per painted leaf, per instance.
///
/// Stored in `array<OverlayEntry>` on the GPU (binding 13 on the scene
/// bind group). Per-instance slices into this array are described by
/// `ArvxGpuInstance.overlay_offset` + `overlay_count`.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable, PartialEq, Eq)]
pub struct OverlayEntry {
    pub leaf_slot: u32,
    pub normal_oct: u32,
    /// Packed `(material_primary as u32) | ((material_secondary_blend as u32) << 16)`.
    pub material_packed: u32,
    /// Packed R|G|B|A. Matches `LeafAttrPool.colors[slot]`. 0 = no override.
    pub color_packed: u32,
}

impl OverlayEntry {
    /// Decode the [`LeafAttr`] portion of this entry.
    #[inline]
    pub fn attr(self) -> LeafAttr {
        LeafAttr {
            normal_oct: self.normal_oct,
            material_primary: (self.material_packed & 0xFFFF) as u16,
            material_secondary_blend: ((self.material_packed >> 16) & 0xFFFF) as u16,
        }
    }

    #[inline]
    pub fn from_parts(leaf_slot: u32, attr: LeafAttr, color_packed: u32) -> Self {
        let material_packed = (attr.material_primary as u32)
            | ((attr.material_secondary_blend as u32) << 16);
        Self { leaf_slot, normal_oct: attr.normal_oct, material_packed, color_packed }
    }
}

/// Sparse per-instance paint overrides, sorted by `leaf_slot`.
#[derive(Debug, Clone, Default)]
pub struct LeafAttrOverlay {
    entries: Vec<OverlayEntry>,
}

impl LeafAttrOverlay {
    pub fn new() -> Self { Self::default() }

    #[inline]
    pub fn len(&self) -> usize { self.entries.len() }

    #[inline]
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }

    #[inline]
    pub fn entries(&self) -> &[OverlayEntry] { &self.entries }

    /// Find the overlay entry for `leaf_slot`, if any.
    #[inline]
    pub fn get(&self, leaf_slot: u32) -> Option<&OverlayEntry> {
        match self.entries.binary_search_by_key(&leaf_slot, |e| e.leaf_slot) {
            Ok(idx) => Some(&self.entries[idx]),
            Err(_) => None,
        }
    }

    /// Insert or replace the overlay entry at `leaf_slot`. O(log N) for
    /// the binary search + O(N) for the vec shift on insert. Prefer
    /// [`Self::upsert_batch`] when committing many entries at once
    /// (e.g. a single paint stamp touching hundreds of leaves) — the
    /// batch path is O(N + K log K) instead of O(K · N).
    pub fn upsert(&mut self, leaf_slot: u32, attr: LeafAttr, color_packed: u32) {
        let entry = OverlayEntry::from_parts(leaf_slot, attr, color_packed);
        match self.entries.binary_search_by_key(&leaf_slot, |e| e.leaf_slot) {
            Ok(idx) => self.entries[idx] = entry,
            Err(idx) => self.entries.insert(idx, entry),
        }
    }

    /// Commit a batch of `(leaf_slot, attr, color)` upserts in one
    /// merge-pass against the existing sorted vec. Cost is
    /// O(N + K log K) — sort the batch, then walk both vecs once.
    /// Per-entry [`Self::upsert`] is O(K · N) for K entries because
    /// every insert shifts the tail; this is unworkable on paint
    /// stamps where K reaches the thousands and N grows over a drag.
    ///
    /// `batch` may contain duplicate `leaf_slot` values; the last one
    /// wins (matches `upsert`'s replace-on-collision semantics).
    pub fn upsert_batch(&mut self, mut batch: Vec<OverlayEntry>) {
        if batch.is_empty() {
            return;
        }
        // Sort then dedupe-keeping-last so duplicate slots in the batch
        // collapse to a single entry. `dedup_by` keeps the first of a
        // run, so reverse-then-dedupe-then-reverse to keep the last
        // — equivalent to "last write wins" semantics.
        batch.sort_unstable_by(|a, b| {
            a.leaf_slot.cmp(&b.leaf_slot)
        });
        batch.reverse();
        batch.dedup_by_key(|e| e.leaf_slot);
        batch.reverse();

        let mut merged = Vec::with_capacity(self.entries.len() + batch.len());
        let mut i = 0;
        let mut j = 0;
        let existing = std::mem::take(&mut self.entries);
        while i < existing.len() && j < batch.len() {
            let a_slot = existing[i].leaf_slot;
            let b_slot = batch[j].leaf_slot;
            if a_slot < b_slot {
                merged.push(existing[i]);
                i += 1;
            } else if a_slot > b_slot {
                merged.push(batch[j]);
                j += 1;
            } else {
                // Slot collision — batch wins (replace).
                merged.push(batch[j]);
                i += 1;
                j += 1;
            }
        }
        if i < existing.len() {
            merged.extend_from_slice(&existing[i..]);
        }
        if j < batch.len() {
            merged.extend_from_slice(&batch[j..]);
        }
        self.entries = merged;
    }

    /// Update only the [`LeafAttr`] portion at `leaf_slot`. Inserts a new
    /// entry with the asset's fallback color (0 = use material base) if
    /// none exists yet. Used by paint material/erase paths that don't
    /// touch the per-leaf color.
    pub fn upsert_attr(&mut self, leaf_slot: u32, attr: LeafAttr) {
        match self.entries.binary_search_by_key(&leaf_slot, |e| e.leaf_slot) {
            Ok(idx) => {
                self.entries[idx].normal_oct = attr.normal_oct;
                self.entries[idx].material_packed = (attr.material_primary as u32)
                    | ((attr.material_secondary_blend as u32) << 16);
            }
            Err(idx) => {
                self.entries.insert(
                    idx,
                    OverlayEntry::from_parts(leaf_slot, attr, 0),
                );
            }
        }
    }

    /// Update only the color at `leaf_slot`. Inserts a new entry copying
    /// the asset's base [`LeafAttr`] (caller-provided) if none exists.
    pub fn upsert_color(&mut self, leaf_slot: u32, base_attr: LeafAttr, color_packed: u32) {
        match self.entries.binary_search_by_key(&leaf_slot, |e| e.leaf_slot) {
            Ok(idx) => self.entries[idx].color_packed = color_packed,
            Err(idx) => {
                self.entries.insert(
                    idx,
                    OverlayEntry::from_parts(leaf_slot, base_attr, color_packed),
                );
            }
        }
    }

    /// Remove the overlay entry at `leaf_slot`, if any. Returns true on hit.
    pub fn remove(&mut self, leaf_slot: u32) -> bool {
        match self.entries.binary_search_by_key(&leaf_slot, |e| e.leaf_slot) {
            Ok(idx) => { self.entries.remove(idx); true }
            Err(_) => false,
        }
    }

    /// Clear all entries. Capacity preserved.
    pub fn clear(&mut self) { self.entries.clear(); }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attr(material: u16) -> LeafAttr {
        LeafAttr { normal_oct: 0xDEAD_BEEF, material_primary: material, material_secondary_blend: 0 }
    }

    #[test]
    fn entry_size_is_16_bytes() {
        assert_eq!(std::mem::size_of::<OverlayEntry>(), 16);
    }

    #[test]
    fn upsert_then_get() {
        let mut o = LeafAttrOverlay::new();
        o.upsert(42, attr(7), 0x1234_5678);
        let e = o.get(42).expect("present");
        assert_eq!(e.leaf_slot, 42);
        assert_eq!(e.attr().material_primary, 7);
        assert_eq!(e.color_packed, 0x1234_5678);
    }

    #[test]
    fn upsert_replaces_existing() {
        let mut o = LeafAttrOverlay::new();
        o.upsert(42, attr(7), 0x11111111);
        o.upsert(42, attr(9), 0x22222222);
        assert_eq!(o.len(), 1);
        let e = o.get(42).unwrap();
        assert_eq!(e.attr().material_primary, 9);
        assert_eq!(e.color_packed, 0x22222222);
    }

    #[test]
    fn entries_stay_sorted() {
        let mut o = LeafAttrOverlay::new();
        for &s in &[5u32, 1, 100, 42, 17, 3] {
            o.upsert(s, attr(s as u16), 0);
        }
        let slots: Vec<u32> = o.entries().iter().map(|e| e.leaf_slot).collect();
        assert_eq!(slots, vec![1, 3, 5, 17, 42, 100]);
    }

    #[test]
    fn upsert_attr_preserves_color() {
        let mut o = LeafAttrOverlay::new();
        o.upsert(42, attr(7), 0xABCD_1234);
        o.upsert_attr(42, attr(8));
        let e = o.get(42).unwrap();
        assert_eq!(e.attr().material_primary, 8);
        assert_eq!(e.color_packed, 0xABCD_1234);
    }

    #[test]
    fn upsert_color_preserves_attr() {
        let mut o = LeafAttrOverlay::new();
        o.upsert(42, attr(7), 0xABCD_1234);
        o.upsert_color(42, attr(99), 0x5555_5555);
        let e = o.get(42).unwrap();
        assert_eq!(e.attr().material_primary, 7); // unchanged
        assert_eq!(e.color_packed, 0x5555_5555);
    }

    #[test]
    fn upsert_color_with_no_existing_uses_base_attr() {
        let mut o = LeafAttrOverlay::new();
        o.upsert_color(42, attr(13), 0xFEEDFACE);
        let e = o.get(42).unwrap();
        assert_eq!(e.attr().material_primary, 13);
        assert_eq!(e.color_packed, 0xFEEDFACE);
    }

    #[test]
    fn remove_returns_hit() {
        let mut o = LeafAttrOverlay::new();
        o.upsert(42, attr(7), 0);
        assert!(o.remove(42));
        assert!(!o.remove(42));
        assert!(o.is_empty());
    }

    #[test]
    fn upsert_batch_merges_into_sorted_set() {
        let mut o = LeafAttrOverlay::new();
        o.upsert(10, attr(1), 0xAA);
        o.upsert(20, attr(2), 0xBB);
        o.upsert(30, attr(3), 0xCC);
        // Mix new slots with one slot that overlaps an existing entry.
        let batch = vec![
            OverlayEntry::from_parts(15, attr(15), 0x15),
            OverlayEntry::from_parts(20, attr(20), 0x20),  // replaces existing
            OverlayEntry::from_parts(25, attr(25), 0x25),
            OverlayEntry::from_parts(5,  attr(5),  0x05),
        ];
        o.upsert_batch(batch);
        let slots: Vec<u32> = o.entries().iter().map(|e| e.leaf_slot).collect();
        assert_eq!(slots, vec![5, 10, 15, 20, 25, 30]);
        assert_eq!(o.get(20).unwrap().attr().material_primary, 20, "batch overrides existing");
        assert_eq!(o.get(20).unwrap().color_packed, 0x20);
        assert_eq!(o.get(10).unwrap().color_packed, 0xAA, "untouched entry preserved");
    }

    #[test]
    fn upsert_batch_empty_is_noop() {
        let mut o = LeafAttrOverlay::new();
        o.upsert(42, attr(7), 0x1234);
        o.upsert_batch(Vec::new());
        assert_eq!(o.len(), 1);
    }

    #[test]
    fn entry_layout_matches_attr_packing() {
        let a = LeafAttr {
            normal_oct: 0x1234_5678,
            material_primary: 0xABCD,
            material_secondary_blend: 0x4321,
        };
        let mut o = LeafAttrOverlay::new();
        o.upsert(0, a, 0xDEAD_BEEF);
        let e = o.get(0).unwrap();
        assert_eq!(e.normal_oct, 0x1234_5678);
        assert_eq!(e.material_packed, 0xABCD | (0x4321 << 16));
        assert_eq!(e.attr(), a);
    }
}
