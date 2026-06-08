//! `ClusterDelta` — a pure-data description of the [`super::types::MeshView`]
//! mutations one *incremental* surface re-extract causes.
//!
//! ## Why this exists (the GPU-mesher seam, layer A)
//!
//! Today the two re-extract executors ([`super::remesh_region`]'s filter +
//! patch-append) compute *and* apply their `MeshView` mutations in one pass.
//! That couples extraction to mutation. The GPU mesher endgame needs the
//! mutation to be a **value**: a GPU surface-nets mesher (layer B) produces
//! the geometry, and a GPU-resident applier (layer C) folds the result into
//! the GPU buffers *without a CPU readback round-trip*.
//!
//! `ClusterDelta` is that value. It captures the two mutation classes an
//! incremental re-extract performs:
//!
//!   * **drop side** — per surviving dirty cluster, the kept index subset
//!     rewritten in place + the freed tail ([`ClusterFilterRewrite`]).
//!   * **add side** — the freshly-extracted region appended as one LOD-0
//!     patch cluster ([`ClusterPatchAppend`]).
//!
//! The CPU path still produces *and* applies it in one process — but through
//! [`super::types::MeshView::apply_cluster_delta`], a single applier that is
//! **bit-identical** to the old inline mutation. Making the delta pure data
//! now buys exhaustive unit-testability (no live GPU) and the exact shape
//! layers B/C plug into.
//!
//! Scope: only the three *incremental* orchestrators (sculpt brush, sculpt
//! stroke, terrain face-band) produce a `ClusterDelta`. The two full-asset
//! rebuilds discard and rebuild the whole slab/cluster/spatial state, so
//! they stay whole-replace — a "delta" there would be the entire mesh.

use crate::mesh_pass::MeshVertex;

/// One dirty cluster's filter result. The kept index list is a strict
/// subset of the cluster's existing range, so it is written back to the
/// **same** `index_offset` (`old_offset`); the dropped tail is freed. The
/// cluster's AABB is unchanged (kept triangles fit inside it).
///
/// `old_offset` / `old_count` are snapshotted from the cluster at *compute*
/// time so the apply is self-contained — it does not re-read the (possibly
/// since-mutated) cluster table to locate the range.
#[derive(Debug, Clone)]
pub(super) struct ClusterFilterRewrite {
    pub(super) cluster_id: u32,
    /// The surviving indices, in the SAME global vertex-index space the
    /// cluster already used (not rebased) — a subset, written verbatim to
    /// `old_offset`.
    pub(super) kept_indices: Vec<u32>,
    pub(super) old_offset: u32,
    pub(super) old_count: u32,
}

/// The freshly-extracted region appended as one standalone LOD-0 patch
/// cluster. `verts` are object-local and not yet in the VBO; `local_indices`
/// are 0-based into `verts` (apply rebases them by the realized VBO
/// insertion offset). No `cluster_id` — apply assigns `meshlet_clusters.len()`.
#[derive(Debug, Clone)]
pub(super) struct ClusterPatchAppend {
    pub(super) verts: Vec<MeshVertex>,
    /// 0-based into `verts`. Apply adds the realized `vertex_offset`.
    pub(super) local_indices: Vec<u32>,
    pub(super) aabb_min: [f32; 3],
    pub(super) aabb_max: [f32; 3],
}

impl ClusterPatchAppend {
    /// Build a patch-append from a freshly-extracted `(verts, indices)`
    /// region. Returns `None` when `verts` is empty — mirroring the old
    /// `append_remesh_patch`, which returned `None` and mutated nothing.
    ///
    /// The AABB scan is a verbatim move of `append_remesh_patch`'s bounds
    /// loop, so the patch cluster's `aabb_min`/`aabb_max` are identical.
    pub(super) fn from_extract(verts: Vec<MeshVertex>, local_indices: Vec<u32>) -> Option<Self> {
        if verts.is_empty() {
            return None;
        }
        let mut aabb_min = [f32::INFINITY; 3];
        let mut aabb_max = [f32::NEG_INFINITY; 3];
        for v in &verts {
            for k in 0..3 {
                if v.local_pos[k] < aabb_min[k] {
                    aabb_min[k] = v.local_pos[k];
                }
                if v.local_pos[k] > aabb_max[k] {
                    aabb_max[k] = v.local_pos[k];
                }
            }
        }
        Some(Self {
            verts,
            local_indices,
            aabb_min,
            aabb_max,
        })
    }
}

/// The complete set of `MeshView` mutations one incremental re-extract
/// causes — pure data. `compute_*` halves build it from `&MeshView`;
/// [`super::types::MeshView::apply_cluster_delta`] folds it into
/// `&mut MeshView`.
///
/// **Ordering.** `filter_rewrites` is kept in the order the filter produced
/// them (dirty-list / rayon-collect order). The final buffer + slab state is
/// actually order-*independent* — distinct clusters own disjoint index
/// ranges, and `free_index_range` keeps the free list sorted + coalesced, so
/// freeing them in any order lands in the same canonical state and the
/// patch's first-fit `alloc_index_range` reuses the same range. What *is*
/// order-observable is the `mesh_indices_dirty` mark **sequence** (`DirtyRanges`
/// appends marks without sorting), which the GPU upload replays in order. To
/// stay bit-identical to the old inline path, apply must iterate this Vec in
/// the same order the inline merge did — so we preserve producer order and
/// never `sort`/`dedup` it.
#[derive(Debug, Clone, Default)]
pub(super) struct ClusterDelta {
    pub(super) filter_rewrites: Vec<ClusterFilterRewrite>,
    /// At most one patch per re-extract today; `Option` rather than `Vec`
    /// keeps it honest (a Vec would be premature). `None` when the extract
    /// produced no verts.
    pub(super) patch: Option<ClusterPatchAppend>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable;

    fn vtx(pos: [f32; 3]) -> MeshVertex {
        let mut v = MeshVertex::zeroed();
        v.local_pos = pos;
        v
    }

    #[test]
    fn from_extract_none_on_empty() {
        assert!(ClusterPatchAppend::from_extract(Vec::new(), Vec::new()).is_none());
    }

    #[test]
    fn from_extract_computes_tight_aabb() {
        let verts = vec![
            vtx([1.0, -2.0, 3.0]),
            vtx([-4.0, 5.0, 0.0]),
            vtx([2.0, 2.0, 9.0]),
        ];
        let p = ClusterPatchAppend::from_extract(verts, vec![0, 1, 2]).expect("non-empty");
        assert_eq!(p.aabb_min, [-4.0, -2.0, 0.0]);
        assert_eq!(p.aabb_max, [2.0, 5.0, 9.0]);
        assert_eq!(p.local_indices, vec![0, 1, 2]);
    }
}
