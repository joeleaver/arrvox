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
    fn mark_full_with_zero_len_keeps_empty_ranges() {
        let mut d = DirtyRanges::new();
        d.mark_full(0);
        assert!(d.is_empty());
        assert!(!d.is_full_pool(0));
    }
}
