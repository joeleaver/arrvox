//! Per-Terrain spatial index of stamps.
//!
//! Mirrors the ECS-side `Stamp` entities into a flat, cache-friendly
//! list with pre-computed AABBs. The bake worker reads a snapshot of
//! this at submit time; the index itself is owned by the `Terrain`
//! component and rebuilt by the engine whenever the stamp set changes.
//!
//! ### Why a flat Vec + linear scan
//!
//! Per `docs/TERRAIN.md` § Stamps, V1 expects "dozens to hundreds"
//! of stamps in a typical scene. A 2D grid or a BVH would be the
//! right answer at "thousands of stamps with millions of voxel
//! queries against them per tile bake" — but the bake path already
//! pre-filters the index down to per-tile lists before the per-voxel
//! loop starts (see `bake_tile`), so the index's only hot path is
//! `tile-AABB → relevant Vec<Stamp>` at submit time. A linear scan
//! is the right shape at this size.
//!
//! V2 can swap in a 2D grid keyed on tile coords without touching the
//! call sites — `query_iter` and `relevant_for_tile` are the only
//! public access points.

use arvx_core::Aabb;
use std::sync::Arc;

use crate::stamp::Stamp;
use crate::tile_key::TileKey;

/// Per-Terrain spatial index of stamps.
///
/// Cheap to clone via `Arc`; the bake worker carries an `Arc<StampIndex>`
/// alongside its `Arc<dyn TerrainFn>` to compose layers 1 + 2 in the
/// per-voxel sample callback.
#[derive(Debug, Clone, Default)]
pub struct StampIndex {
    /// Composition order: ascending `(priority, insertion_index)`. The
    /// engine sorts this before construction to keep bake results
    /// deterministic regardless of ECS iteration order.
    stamps: Vec<Stamp>,
    /// Parallel to `stamps`. `aabbs[i] == stamps[i].aabb()`.
    aabbs: Vec<Aabb>,
}

impl StampIndex {
    /// Empty index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of stamps in the index.
    pub fn len(&self) -> usize {
        self.stamps.len()
    }

    /// True if the index has no stamps.
    pub fn is_empty(&self) -> bool {
        self.stamps.is_empty()
    }

    /// Iterate all stamps in composition order.
    pub fn iter(&self) -> impl Iterator<Item = &Stamp> {
        self.stamps.iter()
    }

    /// Build a fresh index from an iterator of stamps. The engine
    /// passes stamps in any order; this constructor sorts them by
    /// `(priority, insertion_index)` and caches their AABBs.
    ///
    /// Stable sort: ties keep input order, so the engine can use any
    /// secondary ordering it wants (e.g. by ECS Entity id) just by
    /// pre-sorting the input.
    pub fn from_stamps<I>(stamps: I) -> Self
    where
        I: IntoIterator<Item = Stamp>,
    {
        let mut buf: Vec<Stamp> = stamps.into_iter().collect();
        buf.sort_by_key(|s| s.priority);
        let aabbs: Vec<Aabb> = buf.iter().map(|s| s.aabb()).collect();
        Self { stamps: buf, aabbs }
    }

    /// Iterate stamps whose AABB intersects the query box.
    pub fn query_iter<'a>(&'a self, query: Aabb) -> impl Iterator<Item = &'a Stamp> + 'a {
        self.stamps
            .iter()
            .zip(self.aabbs.iter())
            .filter_map(move |(s, aabb)| {
                if aabb_intersects(aabb, &query) {
                    Some(s)
                } else {
                    None
                }
            })
    }

    /// Collect the stamps relevant to one tile bake — those whose
    /// AABB overlaps the tile's XZ footprint (Y is ignored because
    /// heightmap stamps influence the full vertical column of any
    /// tile they cross).
    ///
    /// Returns an owned `Vec<Stamp>` rather than borrowing self so
    /// the caller can wrap it in an `Arc` and hand it to a worker
    /// thread without lifetime concerns. With V1's stamp counts the
    /// per-tile allocation is negligible.
    pub fn relevant_for_tile(&self, tile: TileKey) -> Vec<Stamp> {
        let origin = tile.origin_world().to_vec3();
        let extent = tile.extent_m();
        let tile_xz_min = (origin.x, origin.z);
        let tile_xz_max = (origin.x + extent, origin.z + extent);
        self.stamps
            .iter()
            .zip(self.aabbs.iter())
            .filter(|(_, aabb)| {
                aabb.max.x >= tile_xz_min.0
                    && aabb.min.x <= tile_xz_max.0
                    && aabb.max.z >= tile_xz_min.1
                    && aabb.min.z <= tile_xz_max.1
            })
            .map(|(s, _)| *s)
            .collect()
    }
}

/// `StampIndex` plus the cheap `Arc` wrapper the streamer / bake
/// worker share. Snapshot-shaped (the worker reads a frozen index)
/// so writers always allocate a fresh `Arc<StampIndex>` rather than
/// mutating the live one — keeps the worker's view internally
/// consistent without locks.
pub type StampIndexHandle = Arc<StampIndex>;

#[inline]
fn aabb_intersects(a: &Aabb, b: &Aabb) -> bool {
    a.max.x >= b.min.x
        && a.min.x <= b.max.x
        && a.max.y >= b.min.y
        && a.min.y <= b.max.y
        && a.max.z >= b.min.z
        && a.min.z <= b.max.z
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stamp::{FalloffCurve, StampKind};
    use glam::Vec3;

    fn mountain(p: Vec3, h_max: f32, radius: f32, priority: i32) -> Stamp {
        let mut s = Stamp::new(
            StampKind::Mountain {
                h_max,
                radius,
                falloff: FalloffCurve::Smoothstep,
            },
            p,
        );
        s.priority = priority;
        s
    }

    #[test]
    fn empty_index_is_empty() {
        let i = StampIndex::new();
        assert!(i.is_empty());
        assert_eq!(i.len(), 0);
        assert_eq!(i.relevant_for_tile(TileKey::level0(0, 0, 0)).len(), 0);
    }

    #[test]
    fn from_stamps_sorts_by_priority() {
        let s_lo = mountain(Vec3::new(0.0, 0.0, 0.0), 10.0, 5.0, -3);
        let s_mid = mountain(Vec3::new(0.0, 0.0, 0.0), 20.0, 5.0, 0);
        let s_hi = mountain(Vec3::new(0.0, 0.0, 0.0), 30.0, 5.0, 5);
        // Input intentionally unsorted.
        let i = StampIndex::from_stamps([s_mid, s_hi, s_lo]);
        let prios: Vec<i32> = i.iter().map(|s| s.priority).collect();
        assert_eq!(prios, vec![-3, 0, 5]);
    }

    #[test]
    fn stable_sort_preserves_input_order_within_priority() {
        // Two stamps at the same priority but different positions —
        // the second one (B) must stay second in the index.
        let a = mountain(Vec3::new(0.0, 0.0, 0.0), 10.0, 5.0, 0);
        let b = mountain(Vec3::new(100.0, 0.0, 0.0), 10.0, 5.0, 0);
        let i = StampIndex::from_stamps([a, b]);
        let xs: Vec<f32> = i.iter().map(|s| s.position.x).collect();
        assert_eq!(xs, vec![0.0, 100.0]);
    }

    #[test]
    fn query_iter_intersection() {
        // Mountain at (0, 0, 0), 50m radius, 100m h_max. AABB: x[-50,50] y[0,100] z[-50,50].
        let m = mountain(Vec3::new(0.0, 0.0, 0.0), 100.0, 50.0, 0);
        let i = StampIndex::from_stamps([m]);

        // Query overlapping the AABB:
        let q_in = Aabb {
            min: Vec3::new(-10.0, 10.0, -10.0),
            max: Vec3::new(10.0, 90.0, 10.0),
        };
        assert_eq!(i.query_iter(q_in).count(), 1);

        // Query just outside the AABB on X:
        let q_out = Aabb {
            min: Vec3::new(60.0, 10.0, -10.0),
            max: Vec3::new(70.0, 90.0, 10.0),
        };
        assert_eq!(i.query_iter(q_out).count(), 0);
    }

    #[test]
    fn relevant_for_tile_picks_xz_overlapping_stamps_only() {
        // Tile (0, 0, 0): x ∈ [0, 64), z ∈ [0, 64).
        // s_in:  centred at (32, 0, 32), radius 10 → fully inside tile
        let s_in = mountain(Vec3::new(32.0, 0.0, 32.0), 10.0, 10.0, 0);
        // s_edge: centred at (-5, 0, 32), radius 10 → spills into tile
        // from the west (x ∈ [-15, 5] intersects [0, 64)).
        let s_edge = mountain(Vec3::new(-5.0, 0.0, 32.0), 10.0, 10.0, 0);
        // s_far: centred at (200, 0, 32), radius 10 → no overlap.
        let s_far = mountain(Vec3::new(200.0, 0.0, 32.0), 10.0, 10.0, 0);

        let i = StampIndex::from_stamps([s_in, s_edge, s_far]);
        let relevant = i.relevant_for_tile(TileKey::level0(0, 0, 0));
        assert_eq!(relevant.len(), 2);
        // Order preserved (sorted by priority — all 0 here — then input).
        assert!((relevant[0].position.x - 32.0).abs() < 1e-4);
        assert!((relevant[1].position.x - (-5.0)).abs() < 1e-4);
    }

    #[test]
    fn relevant_for_tile_ignores_y_extent() {
        // A high-altitude mountain (position.y = 200) should still
        // come back as relevant for a tile at y = 0 — heightmap
        // stamps affect every Y in the column.
        let s_high = mountain(Vec3::new(32.0, 200.0, 32.0), 50.0, 10.0, 0);
        let i = StampIndex::from_stamps([s_high]);
        let relevant = i.relevant_for_tile(TileKey::level0(0, 0, 0));
        assert_eq!(relevant.len(), 1);
    }

    #[test]
    fn relevant_for_tile_negative_coords() {
        // Tile (-1, 0, 0): x ∈ [-64, 0), z ∈ [0, 64).
        // Mountain at (-32, 0, 32) is centred inside the tile.
        let s = mountain(Vec3::new(-32.0, 0.0, 32.0), 10.0, 10.0, 0);
        let i = StampIndex::from_stamps([s]);
        let relevant = i.relevant_for_tile(TileKey::level0(-1, 0, 0));
        assert_eq!(relevant.len(), 1);
    }

    #[test]
    fn arc_clone_is_cheap_and_consistent() {
        let s = mountain(Vec3::new(0.0, 0.0, 0.0), 10.0, 5.0, 0);
        let i: StampIndexHandle = Arc::new(StampIndex::from_stamps([s]));
        let i2 = Arc::clone(&i);
        assert_eq!(Arc::strong_count(&i), 2);
        assert_eq!(i.len(), i2.len());
    }
}
