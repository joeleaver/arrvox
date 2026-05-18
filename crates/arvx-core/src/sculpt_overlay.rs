//! Per-instance sculpt overlay — sparse "this leaf is removed" set.
//!
//! Parallels [`crate::leaf_attr_overlay::LeafAttrOverlay`] in shape and
//! intent: per-scene-instance, sparse, sorted, binary-searched on the
//! GPU. Paint adds visual overrides; sculpt records structural removals.
//!
//! Phase A scope (memory: `project_sculpt_phase_a_overlay_plan`): Carve
//! only. Each entry is just the `leaf_attr_id` of a leaf that was carved
//! away; renderers treat that slot as Empty during traversal / discard
//! the fragment in raster. Raise (added geometry) is Phase B and will
//! need a separate channel because the leaf_attr_id space for new
//! geometry doesn't exist yet.
//!
//! # Why this isn't fused with [`LeafAttrOverlay`]
//!
//! Paint and sculpt evolve independently; the binary-search cost is
//! negligible (small N per stamp). Keeping them separate keeps the data
//! flows easy to reason about and avoids forcing every paint-overlay
//! consumer to learn about a removed-flag bit. Merge later if perf says.
//!
//! # GPU layout
//!
//! A flat `array<u32>` on the GPU. Per-instance slice via
//! `ArvxGpuInstance.sculpt_offset` + `sculpt_count`. Sorted ascending so
//! the shader can binary search.

/// Sparse per-instance set of removed `leaf_attr_id`s, sorted ascending.
#[derive(Debug, Clone, Default)]
pub struct SculptOverlay {
    /// Sorted ascending. Each entry is a `leaf_attr_id` (== leaf_attr_pool
    /// slot index) that has been carved away on this instance.
    entries: Vec<u32>,
}

impl SculptOverlay {
    pub fn new() -> Self { Self::default() }

    #[inline]
    pub fn len(&self) -> usize { self.entries.len() }

    #[inline]
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }

    /// Sorted view of removed leaf_attr_ids — direct GPU upload shape.
    #[inline]
    pub fn entries(&self) -> &[u32] { &self.entries }

    /// True iff `leaf_attr_id` is recorded as removed.
    #[inline]
    pub fn contains(&self, leaf_attr_id: u32) -> bool {
        self.entries.binary_search(&leaf_attr_id).is_ok()
    }

    /// Insert `leaf_attr_id`; no-op if already present. Returns true on
    /// new insert. Per-entry cost is O(log N) for the search +
    /// O(N) for the shift — prefer [`Self::insert_batch`] when committing
    /// many removals at once (a sculpt stamp can touch thousands).
    pub fn insert(&mut self, leaf_attr_id: u32) -> bool {
        match self.entries.binary_search(&leaf_attr_id) {
            Ok(_) => false,
            Err(idx) => { self.entries.insert(idx, leaf_attr_id); true }
        }
    }

    /// Commit a batch of removed leaf_attr_ids against the existing
    /// sorted set in one merge-pass: O(N + K log K) for K added entries
    /// on top of N existing. Per-entry `insert` is O(K · N) which kills
    /// drag-paint perf at any meaningful stamp size.
    ///
    /// `batch` may contain duplicates; they collapse to a single entry.
    pub fn insert_batch(&mut self, mut batch: Vec<u32>) {
        if batch.is_empty() {
            return;
        }
        batch.sort_unstable();
        batch.dedup();

        let mut merged = Vec::with_capacity(self.entries.len() + batch.len());
        let mut i = 0;
        let mut j = 0;
        let existing = std::mem::take(&mut self.entries);
        while i < existing.len() && j < batch.len() {
            let a = existing[i];
            let b = batch[j];
            if a < b {
                merged.push(a);
                i += 1;
            } else if a > b {
                merged.push(b);
                j += 1;
            } else {
                merged.push(a);
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

    /// Remove `leaf_attr_id` from the carved-away set (i.e. "uncarve").
    /// Returns true on hit. Phase A doesn't expose an undo UX yet, but
    /// the save path uses this implicitly via [`Self::clear`].
    pub fn remove(&mut self, leaf_attr_id: u32) -> bool {
        match self.entries.binary_search(&leaf_attr_id) {
            Ok(idx) => { self.entries.remove(idx); true }
            Err(_) => false,
        }
    }

    /// Clear all entries. Capacity preserved. Used by the save path
    /// after applying the overlay back into the octree.
    pub fn clear(&mut self) { self.entries.clear(); }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_then_contains() {
        let mut o = SculptOverlay::new();
        assert!(o.insert(42));
        assert!(o.contains(42));
        assert!(!o.contains(43));
    }

    #[test]
    fn insert_is_idempotent() {
        let mut o = SculptOverlay::new();
        assert!(o.insert(42));
        assert!(!o.insert(42));
        assert_eq!(o.len(), 1);
    }

    #[test]
    fn entries_stay_sorted() {
        let mut o = SculptOverlay::new();
        for &s in &[5u32, 1, 100, 42, 17, 3] {
            o.insert(s);
        }
        assert_eq!(o.entries(), &[1, 3, 5, 17, 42, 100]);
    }

    #[test]
    fn remove_returns_hit() {
        let mut o = SculptOverlay::new();
        o.insert(42);
        assert!(o.remove(42));
        assert!(!o.remove(42));
        assert!(o.is_empty());
    }

    #[test]
    fn insert_batch_merges_into_sorted_set() {
        let mut o = SculptOverlay::new();
        o.insert(10);
        o.insert(20);
        o.insert(30);
        // Mix new slots with one that overlaps an existing entry, and
        // include duplicates within the batch.
        o.insert_batch(vec![15, 20, 25, 5, 25, 5]);
        assert_eq!(o.entries(), &[5, 10, 15, 20, 25, 30]);
    }

    #[test]
    fn insert_batch_empty_is_noop() {
        let mut o = SculptOverlay::new();
        o.insert(42);
        o.insert_batch(Vec::new());
        assert_eq!(o.len(), 1);
    }

    #[test]
    fn clear_keeps_capacity() {
        let mut o = SculptOverlay::new();
        for s in 0..16 {
            o.insert(s);
        }
        let cap = o.entries.capacity();
        o.clear();
        assert!(o.is_empty());
        assert_eq!(o.entries.capacity(), cap);
    }
}
