//! [`SnapshotVec`] — append-only, snapshot-able storage for the voxel pools.
//!
//! Replaces `Arc<Vec<T>>` + `Arc::make_mut` in `LeafAttrPool` / `BrickPool`.
//!
//! ## The problem it solves
//!
//! The voxel pools share their backing with off-thread readers (the paint-walk
//! and collider workers) via an `Arc`. With `Arc<Vec<T>>`, ANY mutation went
//! through `Arc::make_mut`, which **clones the entire buffer** whenever a reader
//! snapshot is outstanding (refcount > 1). Asset / terrain splices only APPEND a
//! tail, but `make_mut` cloned the whole multi-GB pool on every splice while a
//! snapshot was held — `O(scene)` per splice (measured ~700–800 ms on a large
//! scene, even for a 222-voxel asset). That is the load-freeze root cause.
//!
//! ## The fix
//!
//! `SnapshotVec` keeps a **fixed-capacity, stable backing** (`Box<[UnsafeCell<T>]>`)
//! behind an `Arc`. Two write patterns, two costs:
//!
//! - [`tail_mut`](SnapshotVec::tail_mut) — `&mut [T]` over a fresh tail range
//!   `[start..start+len]`. **Never clones**, even with snapshots outstanding. A
//!   snapshot pins a length `W` and only ever reads `[0..W]`; callers only ever
//!   request a tail at `start ≥ W` for every live snapshot (the monotonic
//!   allocation watermark), so the range is **disjoint** from every reader's
//!   pinned prefix. Disjoint regions of one stable allocation accessed by
//!   different threads are not a data race. This is the (hot) splice path.
//! - [`make_mut`](SnapshotVec::make_mut) — exclusive `&mut [T]` over the whole
//!   backing for IN-PLACE edits (sculpt, dealloc-zero) and grow-fill. Copies-on-
//!   write if a snapshot is outstanding — exactly like the old `Arc::make_mut` —
//!   because an in-place edit could touch a slot a reader is reading. This is
//!   the (rare) sculpt path, not the (hot) splice path.
//!
//! ## Soundness contract
//!
//! 1. The backing never moves while a [`SnapshotVecView`] is alive: tail writes
//!    stay within the pre-reserved capacity; [`resize`](SnapshotVec::resize) and
//!    the COW build a *new* `Inner` and leave the old one owned by the
//!    snapshot's `Arc`.
//! 2. Single writer (the pool owner, serialized by the scene-manager lock).
//!    Both `tail_mut` and `make_mut` take `&mut self`, so the borrow checker
//!    forbids overlapping `&`/`&mut` within the owner. Snapshots are taken under
//!    the same lock, so no snapshot is created mid-write.
//! 3. `tail_mut(start, ..)` must not overlap any outstanding snapshot's
//!    `[0..live]`. In the common case `start` is the monotonically-increasing
//!    allocation watermark (`next_free`) and snapshots pinned `live ≤ next_free`,
//!    so the tail is disjoint with no clone. But `next_free` is NOT strictly
//!    monotonic — `deallocate_range` tail-coalescing can shrink it, after which
//!    a re-allocation could hand `tail_mut` a `start` below a live snapshot's
//!    pinned prefix. `tail_mut` defends against this itself: it tracks the
//!    high-watermark of pinned snapshots ([`pinned_high`]) and copies-on-write
//!    (detaches to a private `Inner`) if `start` is below it while a snapshot is
//!    outstanding — so the write can never land in a slot a reader is reading.
//! 4. Off-thread readers only ever call [`SnapshotVecView::as_slice`], reading
//!    `[0..live]` — disjoint from every tail write. Publication is provided by
//!    the scene lock (writes happen-before the under-lock `snapshot`) and the
//!    channel hand-off of the view to the worker.
//!
//! `T: Copy` (the pool elements are `LeafAttr` / `u32`, both `Copy + Pod`), so
//! there is no drop glue to run on overwrite or dealloc.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

struct Inner<T> {
    /// Fixed-capacity contiguous backing. `UnsafeCell` lets the owner write
    /// individual slots while snapshots hold shared refs to *disjoint* slots.
    /// `Box<[UnsafeCell<T>]>` has the same layout as `[T]`, so `as_ptr()`
    /// yields a contiguous `*const T` for zero-copy GPU byte reads.
    cells: Box<[UnsafeCell<T>]>,
}

// SAFETY: the access discipline documented on `SnapshotVec` (single writer;
// disjoint tail-writes vs pinned-prefix reads; COW for in-place edits when
// shared) upholds Rust's aliasing rules across threads.
unsafe impl<T: Send + Sync> Send for Inner<T> {}
unsafe impl<T: Send + Sync> Sync for Inner<T> {}

/// Append-only, snapshot-able vector. See module docs.
pub struct SnapshotVec<T> {
    inner: Arc<Inner<T>>,
    /// Highest `live` pinned by any snapshot taken on the CURRENT `inner`
    /// (reset to 0 whenever `inner` is replaced by COW/resize, or when a
    /// `tail_mut` observes no outstanding snapshots). `tail_mut` COW-detaches if
    /// asked to write below this, guarding the disjointness invariant against a
    /// shrunk-then-reallocated watermark (`deallocate_range`). Owner-side only
    /// (atomic so `snapshot(&self)` can bump it).
    pinned_high: AtomicUsize,
}

impl<T: Copy> SnapshotVec<T> {
    /// New backing of `capacity` slots, all initialized to `fill`.
    pub fn with_capacity(capacity: usize, fill: T) -> Self {
        let cells: Box<[UnsafeCell<T>]> = (0..capacity)
            .map(|_| UnsafeCell::new(fill))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            inner: Arc::new(Inner { cells }),
            pinned_high: AtomicUsize::new(0),
        }
    }

    /// Allocated capacity (slots). Mirrors the old `data.len()`, which was the
    /// resized-to-capacity `Vec` length.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.cells.len()
    }

    /// `*const T` to slot 0 (contiguous). For zero-copy byte reads by the
    /// single-writer owner.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.inner.cells.as_ptr() as *const T
    }

    /// The whole backing as a slice (capacity-length). SOUND for the owner: the
    /// only writes are `make_mut` (COW → exclusive) or `tail_mut` (raw,
    /// disjoint); off-thread snapshots only read. Shared `&` reads coexist.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: contiguous, fully-initialized backing; no `&mut` to it can
        // exist concurrently (see type docs / soundness contract).
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.capacity()) }
    }

    /// Strong refcount of the backing (>1 ⇒ a snapshot is outstanding).
    #[inline]
    pub fn refs(&self) -> usize {
        Arc::strong_count(&self.inner)
    }

    /// O(1) snapshot for off-thread readers: shares the backing `Arc` and pins
    /// `live` slots `[0..live]`. Never clones the data. `live` is the owner's
    /// current allocation watermark (`next_free`).
    pub fn snapshot(&self, live: usize) -> SnapshotVecView<T> {
        debug_assert!(live <= self.capacity());
        // Record the watermark this snapshot can read, so a later `tail_mut`
        // below it (after a `deallocate_range` shrank `next_free`) detaches
        // instead of writing into this reader's prefix.
        self.pinned_high.fetch_max(live, Ordering::Relaxed);
        SnapshotVecView {
            inner: self.inner.clone(),
            live,
        }
    }

    /// `&mut [T]` over the FRESH tail range `[start..start+len]`, WITHOUT
    /// cloning even with snapshots outstanding. The caller writes the new data
    /// (plain copy, transformed copy, fill — whatever).
    ///
    /// SAFETY CONTRACT (see type docs): `start` must be ≥ every outstanding
    /// snapshot's pinned `live` (callers allocate at the monotonic watermark),
    /// so this range is disjoint from every reader's `[0..live]`.
    pub fn tail_mut(&mut self, start: usize, len: usize) -> &mut [T] {
        assert!(start + len <= self.capacity(), "SnapshotVec::tail_mut past capacity");
        // Guard the disjointness invariant. If no snapshot is outstanding the
        // write is trivially exclusive (and we can reset the watermark). If one
        // IS outstanding and `start` is below the highest pinned read range
        // (e.g. a `deallocate_range` shrank `next_free` and this is a re-alloc
        // into the freed tail), the write could overlap a reader's prefix —
        // detach to a private `Inner` (COW) first, exactly as the old
        // `Arc::make_mut` would have.
        if Arc::strong_count(&self.inner) == 1 {
            self.pinned_high.store(0, Ordering::Relaxed);
        } else if start < self.pinned_high.load(Ordering::Relaxed) {
            self.cow_clone();
        }
        // SAFETY: after the guard, either no snapshot exists, or `[start..]` is
        // strictly above every outstanding snapshot's pinned `[0..live]` — so
        // this `&mut` races with no reader. Single writer (`&mut self`); the
        // range is within capacity.
        unsafe {
            let base = self.as_ptr() as *mut T;
            std::slice::from_raw_parts_mut(base.add(start), len)
        }
    }

    /// Exclusive `&mut [T]` over the whole backing for IN-PLACE edits + grow.
    /// Copies-on-write if a snapshot is outstanding (refcount > 1), preserving
    /// every reader's consistent view — same semantics as `Arc::make_mut`.
    pub fn make_mut(&mut self) -> &mut [T] {
        if Arc::strong_count(&self.inner) > 1 {
            self.cow_clone();
        }
        let cap = self.capacity();
        // SAFETY: refcount is now 1 (COW above if it was shared), so no other
        // ref to this `Inner` exists and we hold `&mut self` — this is the only
        // access. `cells` is contiguous + fully initialized.
        unsafe { std::slice::from_raw_parts_mut(self.as_ptr() as *mut T, cap) }
    }

    /// Grow capacity to at least `new_cap`, preserving existing data and filling
    /// new slots with `fill`. Builds a fresh `Inner` (the old one stays alive
    /// for any outstanding snapshot). No-op if already large enough.
    pub fn resize(&mut self, new_cap: usize, fill: T) {
        if new_cap <= self.capacity() {
            return;
        }
        let old = self.as_slice();
        let mut v: Vec<UnsafeCell<T>> = Vec::with_capacity(new_cap);
        v.extend(old.iter().map(|&x| UnsafeCell::new(x)));
        for _ in old.len()..new_cap {
            v.push(UnsafeCell::new(fill));
        }
        self.inner = Arc::new(Inner {
            cells: v.into_boxed_slice(),
        });
        // Fresh `Inner` — no snapshots reference it yet.
        self.pinned_high.store(0, Ordering::Relaxed);
    }

    /// Clone the backing into a fresh, refcount-1 `Inner` of the SAME capacity
    /// (the copy-on-write step for an in-place edit, or a tail write below a
    /// pinned snapshot, while a snapshot is held). The old `Inner` stays alive
    /// via the snapshot's `Arc`.
    fn cow_clone(&mut self) {
        let old = self.as_slice();
        let v: Vec<UnsafeCell<T>> = old.iter().map(|&x| UnsafeCell::new(x)).collect();
        self.inner = Arc::new(Inner {
            cells: v.into_boxed_slice(),
        });
        // Fresh, private `Inner` — no snapshots reference it yet.
        self.pinned_high.store(0, Ordering::Relaxed);
    }
}

impl<T: Copy, I: std::slice::SliceIndex<[T]>> std::ops::Index<I> for SnapshotVec<T> {
    type Output = I::Output;
    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        std::ops::Index::index(self.as_slice(), index)
    }
}

/// An O(1) immutable snapshot of a [`SnapshotVec`]'s `[0..live]` prefix, safe to
/// hand to another thread (e.g. the paint-walk / collider workers). Keeps the
/// backing alive; reads only the pinned prefix.
pub struct SnapshotVecView<T> {
    inner: Arc<Inner<T>>,
    live: usize,
}

// SAFETY: same discipline as `Inner` — the view only ever reads `[0..live]`,
// which no writer mutates (writers append past it or COW to a new `Inner`).
unsafe impl<T: Send + Sync> Send for SnapshotVecView<T> {}
unsafe impl<T: Send + Sync> Sync for SnapshotVecView<T> {}

// Manual `Clone` (avoids the `T: Clone` bound a derive would add — cloning the
// view is just an `Arc` bump + copying the pinned length).
impl<T> Clone for SnapshotVecView<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            live: self.live,
        }
    }
}

impl<T: Copy> SnapshotVecView<T> {
    /// The pinned `[0..live]` prefix.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: `[0..live]` was live when this snapshot was taken and is never
        // mutated thereafter (tail writes touch `[live..]`; in-place edits COW
        // to a different `Inner`). Backing is stable while this `Arc` lives.
        unsafe { std::slice::from_raw_parts(self.inner.cells.as_ptr() as *const T, self.live) }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.live
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.live == 0
    }
}

impl<T: Copy> std::ops::Deref for SnapshotVecView<T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tail_write_does_not_clone_with_snapshot_outstanding() {
        let mut v = SnapshotVec::<u32>::with_capacity(16, 0);
        v.tail_mut(0, 3).copy_from_slice(&[1, 2, 3]);
        let snap = v.snapshot(3);
        assert_eq!(snap.as_slice(), &[1, 2, 3]);
        assert_eq!(v.refs(), 2); // snapshot holds a ref

        // Append while the snapshot is outstanding. The backing must NOT be
        // reallocated/cloned — that's the whole point.
        let before = v.as_ptr();
        v.tail_mut(3, 2).copy_from_slice(&[4, 5]);
        assert_eq!(before, v.as_ptr(), "tail write must not reallocate/clone");

        assert_eq!(snap.as_slice(), &[1, 2, 3], "snapshot keeps its pinned prefix");
        assert_eq!(&v.as_slice()[..5], &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn in_place_edit_cows_when_shared() {
        let mut v = SnapshotVec::<u32>::with_capacity(8, 0);
        v.tail_mut(0, 3).copy_from_slice(&[10, 20, 30]);
        let snap = v.snapshot(3);
        let before = v.as_ptr();
        v.make_mut()[1] = 99; // in-place edit with snapshot outstanding → COW
        assert_ne!(before, v.as_ptr(), "in-place edit must COW when shared");
        assert_eq!(snap.as_slice(), &[10, 20, 30], "snapshot keeps old values");
        assert_eq!(&v.as_slice()[..3], &[10, 99, 30]);
    }

    #[test]
    fn in_place_edit_no_cow_when_unshared() {
        let mut v = SnapshotVec::<u32>::with_capacity(8, 0);
        v.tail_mut(0, 2).copy_from_slice(&[1, 2]);
        let before = v.as_ptr();
        v.make_mut()[0] = 7; // no snapshot → no COW
        assert_eq!(before, v.as_ptr());
        assert_eq!(&v.as_slice()[..2], &[7, 2]);
    }

    #[test]
    fn snapshot_survives_grow() {
        let mut v = SnapshotVec::<u32>::with_capacity(4, 0);
        v.tail_mut(0, 4).copy_from_slice(&[1, 2, 3, 4]);
        let snap = v.snapshot(4);
        v.resize(16, 0); // grow → new Inner; snapshot keeps the old one
        assert_eq!(snap.as_slice(), &[1, 2, 3, 4]);
        assert_eq!(v.capacity(), 16);
        v.tail_mut(4, 1).copy_from_slice(&[5]);
        assert_eq!(&v.as_slice()[..5], &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn index_reads() {
        let mut v = SnapshotVec::<u32>::with_capacity(8, 0);
        v.tail_mut(0, 3).copy_from_slice(&[9, 8, 7]);
        assert_eq!(v[1], 8);
        assert_eq!(&v[0..3], &[9, 8, 7]);
    }

    #[test]
    fn tail_write_below_pinned_snapshot_cows_not_corrupts() {
        // The bug the soundness audit caught: `deallocate_range` shrinks the
        // watermark, then a re-alloc hands `tail_mut` a `start` BELOW a live
        // snapshot's pinned prefix. The write must COW-detach, not corrupt.
        let mut v = SnapshotVec::<u32>::with_capacity(16, 0);
        v.tail_mut(0, 10).copy_from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let snap = v.snapshot(10); // pinned_high = 10; reads [0..10]
        let before = v.as_ptr();

        // Re-allocate into the freed tail at start=8 (< pinned_high) while the
        // snapshot is outstanding.
        v.tail_mut(8, 4).copy_from_slice(&[80, 81, 82, 83]);
        assert_ne!(
            before,
            v.as_ptr(),
            "tail write below a pinned snapshot must COW-detach"
        );
        // Snapshot keeps its consistent prefix; owner sees the new values.
        assert_eq!(snap.as_slice(), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(
            &v.as_slice()[..12],
            &[0, 1, 2, 3, 4, 5, 6, 7, 80, 81, 82, 83]
        );
    }

    #[test]
    fn append_at_watermark_never_cows_even_after_lower_snapshot() {
        // The common path must stay clone-free: a snapshot pins a prefix, then
        // genuine TAIL appends (start >= pinned_high) must NOT detach.
        let mut v = SnapshotVec::<u32>::with_capacity(64, 0);
        v.tail_mut(0, 8).copy_from_slice(&[0, 1, 2, 3, 4, 5, 6, 7]);
        let _snap = v.snapshot(8); // pinned_high = 8
        let before = v.as_ptr();
        v.tail_mut(8, 8).copy_from_slice(&[8, 9, 10, 11, 12, 13, 14, 15]);
        assert_eq!(before, v.as_ptr(), "tail append at/after watermark must not COW");
    }

    /// Models the real usage: a single writer (under a `Mutex`, like the pool
    /// owner under the scene lock) tail-appends while reader threads hold
    /// snapshots and verify their pinned prefix. Slot `i` always holds value
    /// `i` (written once on append, never changed), so any reader observing
    /// `view[i] != i` means a tail write corrupted a live prefix (the aliasing
    /// bug this primitive must not have). No TSan here, but logical corruption
    /// and panics surface; the disjoint-region invariant is what makes it sound.
    #[test]
    fn concurrent_tail_appends_dont_corrupt_snapshot_prefixes() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::{Arc, Mutex};

        const CAP: usize = 1 << 18; // 262_144 slots
        const CHUNK: usize = 97; // odd, to cross arbitrary boundaries
        // (SnapshotVec, next_free) — mirrors the pool's owner + watermark.
        let state = Arc::new(Mutex::new((SnapshotVec::<u64>::with_capacity(CAP, 0), 0usize)));
        let stop = Arc::new(AtomicBool::new(false));

        let readers: Vec<_> = (0..4)
            .map(|_| {
                let state = state.clone();
                let stop = stop.clone();
                std::thread::spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        // Take a snapshot under the lock (as walk_snapshot does),
                        // then read it lock-free (as the workers do).
                        let view = {
                            let g = state.lock().unwrap();
                            g.0.snapshot(g.1)
                        };
                        for (i, &v) in view.as_slice().iter().enumerate() {
                            assert_eq!(v, i as u64, "snapshot prefix corrupted at {i}");
                        }
                    }
                })
            })
            .collect();

        for _ in 0..(CAP / CHUNK).saturating_sub(1) {
            let mut g = state.lock().unwrap();
            let start = g.1;
            let chunk: Vec<u64> = (start..start + CHUNK).map(|x| x as u64).collect();
            g.0.tail_mut(start, CHUNK).copy_from_slice(&chunk);
            g.1 = start + CHUNK;
        }
        stop.store(true, Ordering::Relaxed);
        for r in readers {
            r.join().unwrap();
        }

        let g = state.lock().unwrap();
        for i in 0..g.1 {
            assert_eq!(g.0[i], i as u64);
        }
    }
}
