//! Track byte ranges within a CPU-side pool that have been mutated since
//! the last GPU upload, so [`upload_geometry`] can `queue.write_buffer`
//! only the deltas instead of rewriting the whole pool every frame.
//!
//! V1 is intentionally dumb: `mark()` appends to a `Vec` without sorting
//! or coalescing. When the accumulated dirty bytes cross a caller-supplied
//! threshold (or when callers know the whole pool is dirty), the caller
//! falls back to a single full-pool write via `mark_full`.

/// A list of `(byte_offset, byte_len)` ranges within a pool that need
/// to be re-uploaded to the GPU.
#[derive(Debug, Default, Clone)]
pub struct DirtyRanges {
    ranges: Vec<(u32, u32)>,
    total_bytes: u64,
    full: bool,
}

impl DirtyRanges {
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark `[start, start + len)` as dirty. No-op when `len == 0` or
    /// when the tracker is already in full-pool mode.
    pub fn mark(&mut self, start: u32, len: u32) {
        if len == 0 || self.full {
            return;
        }
        self.ranges.push((start, len));
        self.total_bytes += len as u64;
    }

    /// Mark the entire pool dirty (clears any previously-tracked ranges
    /// and replaces them with one `(0, total_len)` range). Subsequent
    /// `mark` calls are ignored until `clear`.
    pub fn mark_full(&mut self, total_len: u32) {
        self.ranges.clear();
        self.total_bytes = total_len as u64;
        self.full = true;
        if total_len > 0 {
            self.ranges.push((0, total_len));
        }
    }

    /// Iterate over the currently-tracked ranges in mark order. V1 does
    /// not merge overlaps; callers that care can use `mark_full` to
    /// short-circuit when accumulated ranges get expensive.
    pub fn iter(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.ranges.iter().copied()
    }

    /// Drop all tracked ranges and exit full-pool mode.
    pub fn clear(&mut self) {
        self.ranges.clear();
        self.total_bytes = 0;
        self.full = false;
    }

    /// True when no ranges are currently tracked.
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// True when the tracker was put in full-pool mode via `mark_full`
    /// and the single range covers exactly `total_len` bytes.
    pub fn is_full_pool(&self, total_len: u32) -> bool {
        self.full
            && self.ranges.len() == 1
            && self.ranges[0] == (0, total_len)
    }

    /// Sum of `len` across all tracked ranges. Counts overlapping bytes
    /// multiple times (V1 does not coalesce); callers should treat this
    /// as an upper bound on actual unique dirty bytes.
    pub fn total_dirty_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Heuristic: when accumulated dirty bytes exceed `threshold_bytes`,
    /// a single full-pool write is usually cheaper than N small
    /// `queue.write_buffer` calls.
    pub fn should_coalesce_to_full(&self, threshold_bytes: u64) -> bool {
        self.full || self.total_bytes >= threshold_bytes
    }

    /// Number of distinct ranges currently tracked. Useful for telemetry.
    pub fn range_count(&self) -> usize {
        self.ranges.len()
    }

    /// Sort marked ranges by offset and merge any that overlap or
    /// abut. After this `iter()` yields a minimal list of disjoint
    /// ranges and `total_dirty_bytes()` reports unique bytes (no
    /// double-counting from repeated `mark` calls of the same region).
    ///
    /// Idempotent — calling twice does no extra work past the second
    /// call's O(n) check.
    ///
    /// Cost: O(n log n) sort + O(n) merge, n = current range count.
    /// Sculpt with many overlapping per-cell marks (~32k marks of
    /// ~500 unique bricks on splat5) collapses to ~50-200 disjoint
    /// ranges, cutting both upload bytes and `queue.write_buffer`
    /// syscall count by ~150×.
    pub fn coalesce(&mut self) {
        self.coalesce_with_gap(0);
    }

    /// Coalesce ranges, additionally merging any pair separated by a
    /// gap of `<= max_gap_bytes`. Trades a few non-dirty bytes per
    /// merged group for fewer `queue.write_buffer` calls — the per-call
    /// overhead in modern wgpu drivers is on the order of 0.5-2 ms
    /// (staging buffer acquisition + command record), so cutting the
    /// per-stamp call count from ~2 000 to ~tens is worth the small
    /// over-upload.
    ///
    /// `max_gap_bytes = 0` is equivalent to plain [`coalesce`] — only
    /// overlapping / abutting ranges merge.
    pub fn coalesce_with_gap(&mut self, max_gap_bytes: u32) {
        if self.full || self.ranges.len() <= 1 {
            return;
        }
        self.ranges.sort_unstable_by_key(|(off, _)| *off);
        let mut merged: Vec<(u32, u32)> = Vec::with_capacity(self.ranges.len());
        for &(off, len) in &self.ranges {
            if len == 0 {
                continue;
            }
            let end = off.saturating_add(len);
            if let Some(last) = merged.last_mut() {
                let last_end = last.0.saturating_add(last.1);
                if off <= last_end.saturating_add(max_gap_bytes) {
                    // Overlap, touch, or within the gap budget — extend
                    // the previous range to swallow this one (and any
                    // non-dirty bytes between them).
                    let new_end = end.max(last_end);
                    last.1 = new_end - last.0;
                    continue;
                }
            }
            merged.push((off, len));
        }
        self.ranges = merged;
        self.total_bytes = self.ranges.iter().map(|(_, l)| *l as u64).sum();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_by_default() {
        let d = DirtyRanges::new();
        assert!(d.is_empty());
        assert_eq!(d.total_dirty_bytes(), 0);
        assert_eq!(d.range_count(), 0);
        assert!(!d.is_full_pool(1024));
    }

    #[test]
    fn mark_appends_range() {
        let mut d = DirtyRanges::new();
        d.mark(0, 16);
        d.mark(64, 32);
        let ranges: Vec<_> = d.iter().collect();
        assert_eq!(ranges, vec![(0, 16), (64, 32)]);
        assert_eq!(d.total_dirty_bytes(), 48);
        assert_eq!(d.range_count(), 2);
    }

    #[test]
    fn mark_zero_len_is_noop() {
        let mut d = DirtyRanges::new();
        d.mark(0, 0);
        assert!(d.is_empty());
    }

    #[test]
    fn mark_full_replaces_existing() {
        let mut d = DirtyRanges::new();
        d.mark(0, 16);
        d.mark(64, 32);
        d.mark_full(1024);
        let ranges: Vec<_> = d.iter().collect();
        assert_eq!(ranges, vec![(0, 1024)]);
        assert!(d.is_full_pool(1024));
        assert!(!d.is_full_pool(2048));
        assert_eq!(d.total_dirty_bytes(), 1024);
    }

    #[test]
    fn mark_after_full_is_noop() {
        let mut d = DirtyRanges::new();
        d.mark_full(1024);
        d.mark(0, 16);
        assert!(d.is_full_pool(1024));
        assert_eq!(d.range_count(), 1);
    }

    #[test]
    fn clear_resets_state() {
        let mut d = DirtyRanges::new();
        d.mark(0, 16);
        d.mark_full(1024);
        d.clear();
        assert!(d.is_empty());
        assert!(!d.is_full_pool(1024));
        d.mark(8, 4);
        let ranges: Vec<_> = d.iter().collect();
        assert_eq!(ranges, vec![(8, 4)]);
    }

    #[test]
    fn coalesce_threshold_trips_above_total() {
        let mut d = DirtyRanges::new();
        d.mark(0, 100);
        assert!(!d.should_coalesce_to_full(200));
        d.mark(200, 150);
        assert!(d.should_coalesce_to_full(200));
    }

    #[test]
    fn coalesce_always_true_in_full_mode() {
        let mut d = DirtyRanges::new();
        d.mark_full(8);
        assert!(d.should_coalesce_to_full(u64::MAX));
    }

    #[test]
    fn coalesce_merges_duplicates() {
        let mut d = DirtyRanges::new();
        // Same brick marked 4 times — should collapse to one range.
        d.mark(256, 256);
        d.mark(256, 256);
        d.mark(256, 256);
        d.mark(256, 256);
        d.coalesce();
        let ranges: Vec<_> = d.iter().collect();
        assert_eq!(ranges, vec![(256, 256)]);
        assert_eq!(d.total_dirty_bytes(), 256);
    }

    #[test]
    fn coalesce_merges_adjacent_ranges() {
        let mut d = DirtyRanges::new();
        d.mark(0, 256);    // [0, 256)
        d.mark(256, 256);  // [256, 512) — touches the previous
        d.mark(512, 256);  // [512, 768) — touches again
        d.coalesce();
        let ranges: Vec<_> = d.iter().collect();
        assert_eq!(ranges, vec![(0, 768)]);
    }

    #[test]
    fn coalesce_merges_overlapping_ranges() {
        let mut d = DirtyRanges::new();
        d.mark(0, 100);
        d.mark(50, 100);  // overlaps with [0, 100)
        d.mark(200, 50);  // disjoint
        d.coalesce();
        let ranges: Vec<_> = d.iter().collect();
        assert_eq!(ranges, vec![(0, 150), (200, 50)]);
    }

    #[test]
    fn coalesce_preserves_disjoint_ranges() {
        let mut d = DirtyRanges::new();
        d.mark(0, 16);
        d.mark(100, 16);
        d.mark(200, 16);
        d.coalesce();
        let ranges: Vec<_> = d.iter().collect();
        assert_eq!(ranges, vec![(0, 16), (100, 16), (200, 16)]);
        assert_eq!(d.total_dirty_bytes(), 48);
    }

    #[test]
    fn coalesce_sorts_out_of_order_marks() {
        let mut d = DirtyRanges::new();
        d.mark(200, 16);
        d.mark(0, 16);
        d.mark(100, 16);
        d.coalesce();
        let ranges: Vec<_> = d.iter().collect();
        assert_eq!(ranges, vec![(0, 16), (100, 16), (200, 16)]);
    }

    #[test]
    fn coalesce_idempotent() {
        let mut d = DirtyRanges::new();
        d.mark(0, 50);
        d.mark(60, 50);
        d.coalesce();
        let after_first: Vec<_> = d.iter().collect();
        d.coalesce();
        let after_second: Vec<_> = d.iter().collect();
        assert_eq!(after_first, after_second);
    }

    #[test]
    fn coalesce_skips_full_pool() {
        let mut d = DirtyRanges::new();
        d.mark_full(1024);
        d.coalesce();
        // Full-pool mode stays as the single (0, 1024) range.
        assert!(d.is_full_pool(1024));
    }

    #[test]
    fn coalesce_with_gap_merges_close_ranges() {
        let mut d = DirtyRanges::new();
        d.mark(0, 16);
        d.mark(100, 16);   // gap of 84 from previous
        d.mark(200, 16);   // gap of 84 from previous
        d.mark(10_000, 16); // far away
        d.coalesce_with_gap(128);
        let ranges: Vec<_> = d.iter().collect();
        // First three merged into one (0..216), fourth stays separate.
        assert_eq!(ranges, vec![(0, 216), (10_000, 16)]);
    }

    #[test]
    fn coalesce_with_gap_zero_matches_plain_coalesce() {
        let mut a = DirtyRanges::new();
        let mut b = DirtyRanges::new();
        for off in [0, 16, 100, 200, 216, 10_000] {
            a.mark(off, 16);
            b.mark(off, 16);
        }
        a.coalesce();
        b.coalesce_with_gap(0);
        let ar: Vec<_> = a.iter().collect();
        let br: Vec<_> = b.iter().collect();
        assert_eq!(ar, br);
    }

    #[test]
    fn coalesce_handles_many_duplicates() {
        // Mimics the sculpt brick-pool case: many marks of the same
        // brick from per-cell set_cell calls. 1000 marks of the same
        // 256 B range → 1 range after coalesce.
        let mut d = DirtyRanges::new();
        for _ in 0..1000 {
            d.mark(1024, 256);
        }
        assert_eq!(d.range_count(), 1000);
        assert_eq!(d.total_dirty_bytes(), 256_000); // pre-coalesce: duplicates counted
        d.coalesce();
        assert_eq!(d.range_count(), 1);
        assert_eq!(d.total_dirty_bytes(), 256);
    }

    #[test]
    fn mark_full_with_zero_len_keeps_empty_ranges() {
        let mut d = DirtyRanges::new();
        d.mark_full(0);
        assert!(d.is_empty());
        assert!(!d.is_full_pool(0));
    }
}
