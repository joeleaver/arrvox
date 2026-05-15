//! Per-asset spatial index over LOD-0 cluster AABBs. D7 of the drain
//! optimization plan.
//!
//! `clusters_in_brush_grid_aabb` (Phase B R3) used to linear-scan every
//! cluster in the asset, testing each one's grid AABB against the
//! brush AABB. On splat5 elephant (~105 k LOD-0 clusters) this cost
//! 1.1-1.8 ms per stamp — small per-stamp but cumulative across a drag.
//!
//! This index buckets LOD-0 clusters into a coarse grid keyed by
//! `IVec3` bucket coords (finest-grid cell coords divided by
//! `BUCKET_SIZE`). Each cluster is inserted into every bucket its grid
//! AABB overlaps. A query walks the buckets the brush touches and
//! unions the cluster lists, then the caller does the actual overlap
//! filter on the much smaller candidate set.
//!
//! Maintenance:
//! - Built once at asset load (`AssetEntry` populated).
//! - Rebuilt on full mesh re-extract (`sculpt::rebuild_asset_mesh`).
//! - Incrementally updated on patch-cluster append
//!   (`sculpt::rebuild_dirty_clusters` Phase 3).
//!
//! Non-LOD-0 clusters are not indexed (the query filters them anyway).

use glam::{IVec3, Vec3};
use rustc_hash::FxHashMap;

use rkp_core::mesh_cluster::{cluster_grid_aabb, MeshletCluster};

/// Bucket edge length in finest-grid cells. Sized so each bucket
/// holds ~10-100 clusters on typical assets:
///
/// * splat5 elephant: ~105 k LOD-0 clusters over ~80 m extent at
///   `base_vs = 0.02 m` → ~4000 cells extent → ~80 buckets per axis
///   → ~512 k bucket coords. Most are empty (surface clusters span a
///   thin shell), so the HashMap stays compact. Typical drag brush
///   spans 2-3 buckets per axis = 8-27 buckets queried, yielding a
///   few hundred candidate clusters vs 105 k for the linear scan.
///
/// At 50 cells = 1 m for `base_vs = 0.02 m`. The constant trades
/// memory (smaller bucket = more buckets) for query selectivity
/// (smaller bucket = tighter candidate set).
pub const BUCKET_SIZE: i32 = 50;

/// Spatial index over LOD-0 cluster AABBs. Owned per [`AssetEntry`].
#[derive(Debug, Default)]
pub(super) struct ClusterSpatialIndex {
    /// bucket_coord → list of LOD-0 cluster ids whose grid AABB
    /// overlaps that bucket. Clusters spanning multiple buckets are
    /// present in each.
    buckets: FxHashMap<IVec3, Vec<u32>>,
}

impl ClusterSpatialIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Rebuild the index from scratch over `clusters`. O(N) — used by
    /// `rebuild_asset_mesh` and the initial asset-load path. Stores
    /// LOD-0 clusters only; other levels never appear in
    /// `clusters_in_brush_grid_aabb`'s output.
    pub fn rebuild(
        &mut self,
        clusters: &[MeshletCluster],
        grid_origin: Vec3,
        base_voxel_size: f32,
    ) {
        self.buckets.clear();
        for (idx, c) in clusters.iter().enumerate() {
            if c.lod_level != 0 {
                continue;
            }
            self.insert_internal(idx as u32, c, grid_origin, base_voxel_size);
        }
    }

    /// Insert one cluster into the index. Used after appending a patch
    /// cluster on `rebuild_dirty_clusters` Phase 3 so the next stamp
    /// can find it via the index instead of forcing a full rebuild.
    pub fn insert(
        &mut self,
        cluster_id: u32,
        cluster: &MeshletCluster,
        grid_origin: Vec3,
        base_voxel_size: f32,
    ) {
        if cluster.lod_level != 0 {
            return;
        }
        self.insert_internal(cluster_id, cluster, grid_origin, base_voxel_size);
    }

    fn insert_internal(
        &mut self,
        cluster_id: u32,
        cluster: &MeshletCluster,
        grid_origin: Vec3,
        base_voxel_size: f32,
    ) {
        let (cmin, cmax) = cluster_grid_aabb(cluster, grid_origin, base_voxel_size);
        let bucket_lo = IVec3::new(
            cmin.x.div_euclid(BUCKET_SIZE),
            cmin.y.div_euclid(BUCKET_SIZE),
            cmin.z.div_euclid(BUCKET_SIZE),
        );
        let bucket_hi = IVec3::new(
            cmax.x.div_euclid(BUCKET_SIZE),
            cmax.y.div_euclid(BUCKET_SIZE),
            cmax.z.div_euclid(BUCKET_SIZE),
        );
        for z in bucket_lo.z..=bucket_hi.z {
            for y in bucket_lo.y..=bucket_hi.y {
                for x in bucket_lo.x..=bucket_hi.x {
                    self.buckets
                        .entry(IVec3::new(x, y, z))
                        .or_default()
                        .push(cluster_id);
                }
            }
        }
    }

    /// Return the set of LOD-0 cluster ids whose bucket coverage
    /// intersects the brush AABB. **Conservative** — caller must
    /// still run the actual AABB overlap test on each candidate. Both
    /// bounds are interpreted half-open `[brush_lo, brush_hi)` to
    /// match the `brush_cell_range` convention.
    ///
    /// Output is sorted by cluster id with no duplicates (a cluster
    /// spanning multiple buckets within the queried range appears
    /// once).
    pub fn query(&self, brush_lo: IVec3, brush_hi: IVec3) -> Vec<u32> {
        if brush_lo.x >= brush_hi.x
            || brush_lo.y >= brush_hi.y
            || brush_lo.z >= brush_hi.z
        {
            return Vec::new();
        }
        let bucket_lo = IVec3::new(
            brush_lo.x.div_euclid(BUCKET_SIZE),
            brush_lo.y.div_euclid(BUCKET_SIZE),
            brush_lo.z.div_euclid(BUCKET_SIZE),
        );
        // brush_hi is exclusive, so the last cell inside the brush is
        // brush_hi - 1. Its bucket coord is `(brush_hi - 1).div_euclid`.
        let bucket_hi = IVec3::new(
            (brush_hi.x - 1).div_euclid(BUCKET_SIZE),
            (brush_hi.y - 1).div_euclid(BUCKET_SIZE),
            (brush_hi.z - 1).div_euclid(BUCKET_SIZE),
        );
        // Collect into a small sorted set for dedup. A typical drag
        // brush touches ≤ 27 buckets; the candidate list is bounded by
        // the sum of bucket sizes (~100-2700 entries) so a Vec + sort
        // outperforms a HashSet for the dedup.
        let mut candidates: Vec<u32> = Vec::new();
        for z in bucket_lo.z..=bucket_hi.z {
            for y in bucket_lo.y..=bucket_hi.y {
                for x in bucket_lo.x..=bucket_hi.x {
                    if let Some(ids) = self.buckets.get(&IVec3::new(x, y, z)) {
                        candidates.extend_from_slice(ids);
                    }
                }
            }
        }
        candidates.sort_unstable();
        candidates.dedup();
        candidates
    }

    /// Diagnostic — total entries across all buckets (a cluster
    /// spanning N buckets contributes N to this count).
    #[allow(dead_code)]
    pub fn entry_count(&self) -> usize {
        self.buckets.values().map(|v| v.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cluster_at(min: [f32; 3], max: [f32; 3], lod: u32) -> MeshletCluster {
        MeshletCluster {
            aabb_min: min,
            _pad0: 0.0,
            aabb_max: max,
            index_offset: 0,
            index_count: 0,
            lod_level: lod,
            flags: 0,
            cluster_error: 0.0,
            parent_group_error: 1e30,
            group_above_idx: u32::MAX,
            group_below_idx: u32::MAX,
            _pad3: 0,
        }
    }

    #[test]
    fn empty_index_returns_nothing() {
        let idx = ClusterSpatialIndex::new();
        let r = idx.query(IVec3::new(0, 0, 0), IVec3::new(100, 100, 100));
        assert!(r.is_empty());
    }

    #[test]
    fn rebuilt_index_returns_overlapping_lod0_clusters() {
        // Three LOD-0 clusters at distinct positions + one LOD-1 that
        // should never be returned.
        let clusters = vec![
            cluster_at([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0), // id 0 at grid [0, 50]
            cluster_at([5.0, 5.0, 5.0], [6.0, 6.0, 6.0], 0), // id 1 at grid [250, 300]
            cluster_at([20.0, 0.0, 0.0], [21.0, 1.0, 1.0], 0), // id 2 at grid [1000, 1050]
            cluster_at([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1), // id 3 LOD-1
        ];
        let mut idx = ClusterSpatialIndex::new();
        idx.rebuild(&clusters, Vec3::ZERO, 0.02);

        // Brush near cluster 0 → expects [0]; LOD-1 cluster 3 sits in
        // the same buckets but is excluded at insert time.
        let q = idx.query(IVec3::new(0, 0, 0), IVec3::new(60, 60, 60));
        assert_eq!(q, vec![0]);

        // Brush near cluster 1 → expects [1].
        let q = idx.query(IVec3::new(240, 240, 240), IVec3::new(310, 310, 310));
        assert_eq!(q, vec![1]);

        // Wide brush hitting clusters 0 and 1 but not 2.
        let q = idx.query(IVec3::new(0, 0, 0), IVec3::new(400, 400, 400));
        assert_eq!(q, vec![0, 1]);
    }

    #[test]
    fn cluster_spanning_buckets_returns_once() {
        // Cluster spans buckets at grid coords (49,49,49) and (50,50,50).
        let clusters = vec![cluster_at(
            [0.95, 0.95, 0.95],
            [1.05, 1.05, 1.05],
            0,
        )];
        let mut idx = ClusterSpatialIndex::new();
        idx.rebuild(&clusters, Vec3::ZERO, 0.02);
        // Cluster appears in multiple buckets due to the +1 padding in
        // cluster_grid_aabb, plus the natural span across the 50-cell
        // bucket boundary.
        let q = idx.query(IVec3::new(40, 40, 40), IVec3::new(60, 60, 60));
        assert_eq!(q, vec![0]);
    }

    #[test]
    fn incremental_insert_visible_in_next_query() {
        let clusters = vec![cluster_at([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0)];
        let mut idx = ClusterSpatialIndex::new();
        idx.rebuild(&clusters, Vec3::ZERO, 0.02);

        // Insert a second cluster as if it were just appended by a
        // sculpt-V2 patch.
        let patch = cluster_at([10.0, 10.0, 10.0], [11.0, 11.0, 11.0], 0);
        idx.insert(1, &patch, Vec3::ZERO, 0.02);

        let q = idx.query(IVec3::new(0, 0, 0), IVec3::new(600, 600, 600));
        assert_eq!(q, vec![0, 1]);
    }

    #[test]
    fn empty_brush_aabb_returns_nothing() {
        let clusters = vec![cluster_at([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0)];
        let mut idx = ClusterSpatialIndex::new();
        idx.rebuild(&clusters, Vec3::ZERO, 0.02);
        let q = idx.query(IVec3::new(10, 10, 10), IVec3::new(10, 10, 10));
        assert!(q.is_empty());
        let q = idx.query(IVec3::new(10, 10, 10), IVec3::new(5, 5, 5));
        assert!(q.is_empty());
    }
}
