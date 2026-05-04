//! Variable-size bucketed pool allocator: power-of-2 buckets, free lists per bucket.

// ============================================================
// Variable-size bucketed pool allocator
// ============================================================

/// Free-list allocator over a fixed-capacity pool, bucketed by power
/// of 2. Per-pool bucket range is configurable; `alloc(n)` rounds the
/// request up to the smallest bucket that fits and returns
/// `(offset, allocated_size)`. `free(offset, allocated_size)` pushes
/// the extent back onto its bucket's free list. Bumps a global
/// high-water mark when the matching free list is empty.
///
/// Internal waste is at most 2× per region (worst case: request just
/// over a bucket boundary, get the next bucket). Total reservation
/// matches actual aggregate usage instead of `regions × max_per_region`.
///
/// Not thread-safe — all calls are CPU-side from the render thread.
#[derive(Debug, Clone)]
pub struct BucketPoolAllocator {
    capacity: u32,
    min_bucket: u32,
    max_bucket: u32,
    high_water: u32,
    /// `free_lists[i]` holds free-extent offsets for bucket size
    /// `min_bucket << i`. Length is `log2(max_bucket / min_bucket) + 1`.
    free_lists: Vec<Vec<u32>>,
}

impl BucketPoolAllocator {
    /// Build an allocator over `capacity` slots with bucket sizes
    /// `[min_bucket, 2*min_bucket, …, max_bucket]`. Both bounds must
    /// be powers of 2 and `min_bucket <= max_bucket <= capacity`.
    pub fn new(capacity: u32, min_bucket: u32, max_bucket: u32) -> Self {
        assert!(min_bucket.is_power_of_two(), "min_bucket must be power of 2");
        assert!(max_bucket.is_power_of_two(), "max_bucket must be power of 2");
        assert!(min_bucket <= max_bucket, "min_bucket must be <= max_bucket");
        let n_buckets =
            (max_bucket.trailing_zeros() - min_bucket.trailing_zeros() + 1) as usize;
        Self {
            capacity,
            min_bucket,
            max_bucket,
            high_water: 0,
            free_lists: vec![Vec::new(); n_buckets],
        }
    }

    /// Smallest bucket size at least `requested`, clamped to
    /// `[min_bucket, max_bucket]`.
    fn bucket_for(&self, requested: u32) -> u32 {
        requested
            .max(1)
            .next_power_of_two()
            .max(self.min_bucket)
            .min(self.max_bucket)
    }

    fn bucket_idx(&self, bucket: u32) -> usize {
        (bucket.trailing_zeros() - self.min_bucket.trailing_zeros()) as usize
    }

    /// Allocate at least `requested` slots. Returns `(offset, allocated_size)`
    /// where `allocated_size >= requested` is the bucket size, or
    /// `None` if the request exceeds `max_bucket` or the pool is
    /// exhausted (no matching free extent + no room to bump).
    pub fn alloc(&mut self, requested: u32) -> Option<(u32, u32)> {
        if requested > self.max_bucket {
            return None;
        }
        let bucket = self.bucket_for(requested);
        let idx = self.bucket_idx(bucket);
        if let Some(offset) = self.free_lists[idx].pop() {
            return Some((offset, bucket));
        }
        if self.high_water + bucket > self.capacity {
            return None;
        }
        let offset = self.high_water;
        self.high_water += bucket;
        Some((offset, bucket))
    }

    /// Return an extent to its bucket's free list. `allocated_size`
    /// must match the value returned by the corresponding `alloc`.
    pub fn free(&mut self, offset: u32, allocated_size: u32) {
        debug_assert!(
            allocated_size.is_power_of_two()
                && allocated_size >= self.min_bucket
                && allocated_size <= self.max_bucket,
            "free: allocated_size must come from a previous alloc()",
        );
        let idx = self.bucket_idx(allocated_size);
        self.free_lists[idx].push(offset);
    }

    pub fn high_water(&self) -> u32 { self.high_water }
    pub fn capacity(&self) -> u32 { self.capacity }
    pub fn min_bucket(&self) -> u32 { self.min_bucket }
    pub fn max_bucket(&self) -> u32 { self.max_bucket }

    /// Number of free extents currently held across all buckets —
    /// for diagnostics and tests.
    pub fn free_count(&self) -> usize {
        self.free_lists.iter().map(|l| l.len()).sum()
    }
}
