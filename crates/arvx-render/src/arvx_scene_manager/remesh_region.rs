//! `RemeshRegion` — the single dirty-region change-feed between a voxel
//! edit and the mesher.
//!
//! Three re-extract paths used to each hand-roll their own notion of
//! "which grid region did this edit dirty, and which existing triangles
//! must be dropped before re-meshing it":
//!
//!   * sculpt brush footprint   (`sculpt::rebuild_dirty_clusters`)
//!   * sculpt stroke union      (`sculpt::rebuild_stroke_clusters`)
//!   * terrain halo face-band   (`terrain_halo_refresh::rebuild_face_band_clusters`)
//!
//! The three copies of the per-triangle drop predicate had *silently
//! diverged* — sculpt dropped triangles touching a sphere, the terrain
//! band dropped triangles touching a box, and the stroke dropped only
//! triangles fully inside a box. That divergence is exactly the
//! scattered-authority bug class (one concept, no owner, copies drift).
//!
//! This module names the owner. [`RemeshRegion`] is the change-feed
//! entry: the extract span, the dirty-cluster query span, and the
//! [`RemeshFilter`] drop predicate. [`RemeshFilter`] is the single home
//! for all three drop rules. Each edit path now *constructs* a
//! `RemeshRegion` and feeds it to the shared executors
//! ([`ArvxSceneManager::remesh_filter_dirty_clusters`] +
//! [`ArvxSceneManager::append_remesh_patch`]) instead of duplicating the
//! span math, the filter predicate, and the patch-append block.
//!
//! This is also the contract the future `VoxelModel` / `MeshView` split
//! will consume: the mesher becomes `fn(&VoxelModel, &[RemeshRegion],
//! &mut MeshView)`.

use glam::{IVec3, Vec3};

use super::cluster_delta::ClusterFilterRewrite;
use super::mesher::Mesher;
use super::types::{MeshView, VoxelModel};

/// How a re-extract decides which pre-existing triangles to drop before
/// re-meshing a region. The single authority for the three drop rules
/// that were previously copy-diverged across the re-extract paths.
#[derive(Debug, Clone, Copy)]
pub enum RemeshFilter {
    /// Additive edit (Raise / Inflate): the edit only *adds* cells, so
    /// every existing triangle stays a valid part of the surface. Keep
    /// them all — no filter, no index-buffer writes.
    KeepAll,
    /// Drop any triangle with at least one vertex inside the brush
    /// sphere. Used by the sculpt brush footprint (Carve / Deflate /
    /// Smooth / ClayStrip) where the patch re-extracts the whole sphere.
    SphereTouch { center: Vec3, radius_sq: f32 },
    /// Drop any triangle with at least one vertex inside the box. Used
    /// by the terrain halo face-band, where the slab re-extract covers
    /// every cell that could touch the refreshed boundary.
    BoxTouch { min: Vec3, max: Vec3 },
    /// Drop only triangles whose three vertices are *all* inside the
    /// box. Used by the sculpt stroke union: the union patch re-extracts
    /// the box interior, but boundary-straddling triangles are kept so
    /// the seam to untouched geometry stays welded.
    BoxContain { min: Vec3, max: Vec3 },
}

impl RemeshFilter {
    #[inline]
    fn point_in_box(p: Vec3, min: Vec3, max: Vec3) -> bool {
        p.x >= min.x
            && p.x <= max.x
            && p.y >= min.y
            && p.y <= max.y
            && p.z >= min.z
            && p.z <= max.z
    }

    /// Conservative cluster-level reject. Returns `false` only when the
    /// cluster (object-local AABB `[c_min, c_max]`) provably contains no
    /// triangle this filter would drop — the caller then leaves the
    /// whole cluster untouched. Mirrors the per-path early rejects the
    /// three legacy re-extract functions hand-rolled (sculpt's
    /// closest-point sphere test, the terrain band's AABB-overlap test).
    #[inline]
    pub fn cluster_may_have_dropped_tris(&self, c_min: Vec3, c_max: Vec3) -> bool {
        match *self {
            RemeshFilter::KeepAll => false,
            RemeshFilter::SphereTouch { center, radius_sq } => {
                // Closest point on the cluster AABB to the sphere center.
                let closest = center.clamp(c_min, c_max);
                (closest - center).length_squared() <= radius_sq
            }
            RemeshFilter::BoxTouch { min, max } | RemeshFilter::BoxContain { min, max } => {
                // AABB-vs-AABB overlap (a disjoint cluster can hold
                // neither a box-touching nor a box-contained triangle).
                !(c_max.x < min.x
                    || c_min.x > max.x
                    || c_max.y < min.y
                    || c_min.y > max.y
                    || c_max.z < min.z
                    || c_min.z > max.z)
            }
        }
    }

    /// Whether the triangle `(p0, p1, p2)` *survives* (is kept).
    #[inline]
    pub fn keeps_tri(&self, p0: Vec3, p1: Vec3, p2: Vec3) -> bool {
        match *self {
            RemeshFilter::KeepAll => true,
            RemeshFilter::SphereTouch { center, radius_sq } => {
                // Drop if ANY vertex is inside the sphere ⇒ keep iff all
                // three are strictly outside.
                let d0 = (p0 - center).length_squared();
                let d1 = (p1 - center).length_squared();
                let d2 = (p2 - center).length_squared();
                d0 > radius_sq && d1 > radius_sq && d2 > radius_sq
            }
            RemeshFilter::BoxTouch { min, max } => {
                // Drop if ANY vertex is inside the box ⇒ keep iff all out.
                !Self::point_in_box(p0, min, max)
                    && !Self::point_in_box(p1, min, max)
                    && !Self::point_in_box(p2, min, max)
            }
            RemeshFilter::BoxContain { min, max } => {
                // Drop only if ALL vertices are inside the box ⇒ keep iff
                // at least one is outside.
                !Self::point_in_box(p0, min, max)
                    || !Self::point_in_box(p1, min, max)
                    || !Self::point_in_box(p2, min, max)
            }
        }
    }
}

/// A region of an asset's voxel grid that needs re-meshing — the
/// change-feed entry between a voxel edit and the mesher.
#[derive(Debug, Clone, Copy)]
pub struct RemeshRegion {
    /// Grid-cell AABB `[lo, hi)` that surface-nets re-extracts.
    pub extract_lo: IVec3,
    pub extract_hi: IVec3,
    /// Grid-cell AABB `[lo, hi)` used to query which existing clusters
    /// are dirty. Equal to the extract span for the sculpt paths; the
    /// terrain face-band makes it wider/asymmetric than the extract.
    pub query_lo: IVec3,
    pub query_hi: IVec3,
    /// How to drop pre-existing triangles overlapping this region.
    pub filter: RemeshFilter,
}

impl RemeshRegion {
    /// Sculpt brush footprint (single stamp). Extract and query both
    /// span the brush cell AABB; the drop predicate is the brush sphere
    /// (or `KeepAll` for additive Raise / Inflate brushes).
    pub fn brush(brush_lo: IVec3, brush_hi: IVec3, filter: RemeshFilter) -> Self {
        Self {
            extract_lo: brush_lo,
            extract_hi: brush_hi,
            query_lo: brush_lo,
            query_hi: brush_hi,
            filter,
        }
    }

    /// Sculpt stroke union. Extract and query both span the accumulated
    /// stroke AABB; the drop predicate keeps boundary-straddling tris.
    pub fn stroke_union(union_lo: IVec3, union_hi: IVec3, filter: RemeshFilter) -> Self {
        Self {
            extract_lo: union_lo,
            extract_hi: union_hi,
            query_lo: union_lo,
            query_hi: union_hi,
            filter,
        }
    }

    /// Terrain halo face-band. The extract span (narrow slab on the face
    /// axis) and the query/filter span (wider, asymmetric slab) come
    /// from `slab_grid_for_face`; the drop predicate is the filter slab
    /// AABB in object-local space.
    pub fn face_band(
        extract_lo: IVec3,
        extract_hi: IVec3,
        filter_lo: IVec3,
        filter_hi: IVec3,
        grid_origin: Vec3,
        base_vs: f32,
    ) -> Self {
        let min = grid_origin + filter_lo.as_vec3() * base_vs;
        let max = grid_origin + filter_hi.as_vec3() * base_vs;
        Self {
            extract_lo,
            extract_hi,
            query_lo: filter_lo,
            query_hi: filter_hi,
            filter: RemeshFilter::BoxTouch { min, max },
        }
    }
}

/// How much of a terrain tile a halo refresh re-extracts. Replaces the
/// old hand-set `skip_remesh: bool`, whose name lied — it never skipped
/// the re-mesh, it chose a *wider* one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemeshScope {
    /// Re-extract only the boundary slab on the refreshed face. The
    /// common case: the neighbour tile was not independently sculpted
    /// this stamp, so a narrow face-band patch suffices.
    FaceBand,
    /// Re-extract the whole asset. Used when the same tile already
    /// received sculpt edits this stamp — a full haloed re-extract welds
    /// the halo refresh and the sculpt patch in one pass. (The
    /// face-band slab's wider filter region would otherwise drop the
    /// sculpt patch triangles without re-emitting them.)
    FullAsset,
}

/// Stats from a [`ArvxSceneManager::remesh_filter_dirty_clusters`] call,
/// for the per-path telemetry log lines.
#[derive(Debug, Default, Clone, Copy)]
pub struct RemeshFilterStats {
    /// Dirty clusters that short-circuited (KeepAll) or were rejected by
    /// the cluster-level test without running the per-triangle filter.
    pub clusters_rejected: usize,
    pub kept_tris: usize,
    pub dropped_tris: usize,
}

impl Mesher {
    /// **Pure compute half** of the filter executor. Queries the clusters
    /// overlapping `region.query_{lo,hi}`, runs the rayon per-triangle
    /// filter, and returns the dirty-cluster list, the owned
    /// [`ClusterFilterRewrite`]s (in producer order — do not reorder), and
    /// the filter telemetry. **Reads `&view`, mutates nothing** — the
    /// in-place merge is [`MeshView::apply_cluster_delta`]'s job.
    ///
    /// `KeepAll` short-circuits after the dirty query (empty rewrite set),
    /// so additive brushes do zero index-buffer work. `old_offset` /
    /// `old_count` are snapshotted from each cluster here so apply needn't
    /// re-read the (by-then-mutated) table.
    pub(super) fn compute_filter_delta(
        &self,
        model: &VoxelModel,
        view: &MeshView,
        region: &RemeshRegion,
    ) -> (Vec<u32>, Vec<ClusterFilterRewrite>, RemeshFilterStats) {
        let dirty = view.clusters_in_grid_aabb(model, region.query_lo, region.query_hi);
        let mut stats = RemeshFilterStats::default();
        if dirty.is_empty() {
            return (dirty, Vec::new(), stats);
        }
        if matches!(region.filter, RemeshFilter::KeepAll) {
            // Additive edit: existing tris stay valid → no filter, no IBO
            // writes. Report every dirty cluster as short-circuited to
            // mirror the legacy Raise/Inflate telemetry.
            stats.clusters_rejected = dirty.len();
            return (dirty, Vec::new(), stats);
        }

        use rayon::prelude::*;
        let filter = region.filter;

        // Rayon-parallel per-triangle filter. Each cluster's filter is
        // independent (reads its own index slice + indexed vertices,
        // produces a kept-index Vec). `par_iter().collect()` preserves
        // input order so the rewrite Vec — and thus apply's dirty-mark
        // sequence — is deterministic.
        let results: Vec<(u32, Vec<u32>)> = {
            let clusters = &view.meshlet_clusters;
            let indices = &view.mesh_indices;
            let verts = &view.mesh_vertices;
            dirty
                .par_iter()
                .filter_map(|&cid| {
                    let c = &clusters[cid as usize];
                    let count = c.index_count as usize;
                    if count == 0 {
                        return None;
                    }
                    // Cluster-level reject: skip clusters that provably
                    // hold no triangle the filter would drop.
                    if !filter.cluster_may_have_dropped_tris(
                        Vec3::from(c.aabb_min),
                        Vec3::from(c.aabb_max),
                    ) {
                        return None;
                    }
                    let start = c.index_offset as usize;
                    let mut out = Vec::with_capacity(count);
                    for tri_start in (start..start + count).step_by(3) {
                        let i0 = indices[tri_start];
                        let i1 = indices[tri_start + 1];
                        let i2 = indices[tri_start + 2];
                        let p0 = Vec3::from(verts[i0 as usize].local_pos);
                        let p1 = Vec3::from(verts[i1 as usize].local_pos);
                        let p2 = Vec3::from(verts[i2 as usize].local_pos);
                        if filter.keeps_tri(p0, p1, p2) {
                            out.push(i0);
                            out.push(i1);
                            out.push(i2);
                        }
                    }
                    Some((cid, out))
                })
                .collect()
        };
        stats.clusters_rejected = dirty.len() - results.len();

        // Build the rewrites in producer order, snapshotting each
        // cluster's pre-merge `(index_offset, index_count)` and folding in
        // the kept/dropped triangle telemetry. No view mutation here.
        let mut rewrites = Vec::with_capacity(results.len());
        for (cid, kept) in results {
            let old_offset = view.meshlet_clusters[cid as usize].index_offset;
            let old_count = view.meshlet_clusters[cid as usize].index_count;
            let new_count = kept.len() as u32;
            stats.kept_tris += (new_count as usize) / 3;
            stats.dropped_tris += ((old_count - new_count) as usize) / 3;
            rewrites.push(ClusterFilterRewrite {
                cluster_id: cid,
                kept_indices: kept,
                old_offset,
                old_count,
            });
        }

        (dirty, rewrites, stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Equilateral-ish helper triangles around a point.
    fn tri(a: Vec3, b: Vec3, c: Vec3) -> (Vec3, Vec3, Vec3) {
        (a, b, c)
    }

    #[test]
    fn sphere_touch_drops_only_tris_with_a_vertex_inside() {
        let f = RemeshFilter::SphereTouch {
            center: Vec3::ZERO,
            radius_sq: 1.0,
        };
        // All three outside → kept.
        let (a, b, c) = tri(
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, 0.0, 2.0),
        );
        assert!(f.keeps_tri(a, b, c));
        // One vertex inside → dropped.
        let (a, b, c) = tri(
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, 0.0, 2.0),
        );
        assert!(!f.keeps_tri(a, b, c));
    }

    #[test]
    fn box_touch_drops_tris_with_any_vertex_inside() {
        let f = RemeshFilter::BoxTouch {
            min: Vec3::splat(-1.0),
            max: Vec3::splat(1.0),
        };
        let outside = Vec3::new(5.0, 5.0, 5.0);
        let inside = Vec3::ZERO;
        assert!(f.keeps_tri(outside, outside, outside));
        assert!(!f.keeps_tri(outside, inside, outside));
    }

    #[test]
    fn box_contain_drops_only_fully_inside_tris() {
        let f = RemeshFilter::BoxContain {
            min: Vec3::splat(-1.0),
            max: Vec3::splat(1.0),
        };
        let outside = Vec3::new(5.0, 5.0, 5.0);
        let inside = Vec3::ZERO;
        // Straddling (one vertex outside) → kept (welds the seam).
        assert!(f.keeps_tri(outside, inside, inside));
        // Fully inside → dropped.
        assert!(!f.keeps_tri(inside, inside, inside));
    }

    #[test]
    fn keep_all_keeps_everything_and_rejects_no_cluster() {
        let f = RemeshFilter::KeepAll;
        let p = Vec3::ZERO;
        assert!(f.keeps_tri(p, p, p));
        assert!(!f.cluster_may_have_dropped_tris(Vec3::splat(-9.0), Vec3::splat(9.0)));
    }

    #[test]
    fn sphere_cluster_reject_matches_closest_point_test() {
        let f = RemeshFilter::SphereTouch {
            center: Vec3::ZERO,
            radius_sq: 1.0,
        };
        // Cluster straddling the origin → may have dropped tris.
        assert!(f.cluster_may_have_dropped_tris(Vec3::splat(-2.0), Vec3::splat(2.0)));
        // Cluster far away → cannot.
        assert!(!f.cluster_may_have_dropped_tris(Vec3::splat(5.0), Vec3::splat(6.0)));
    }

    #[test]
    fn box_cluster_reject_is_aabb_overlap() {
        let f = RemeshFilter::BoxTouch {
            min: Vec3::splat(-1.0),
            max: Vec3::splat(1.0),
        };
        assert!(f.cluster_may_have_dropped_tris(Vec3::splat(0.0), Vec3::splat(0.5)));
        assert!(!f.cluster_may_have_dropped_tris(Vec3::splat(2.0), Vec3::splat(3.0)));
    }
}
