//! Meshlet clustering — Phase 5 of the mesh pivot.
//!
//! Partitions a single asset's `(vertices, indices)` mesh into
//! meshlets capped at `MAX_VERTS_PER_CLUSTER` unique vertices and
//! `MAX_TRIS_PER_CLUSTER` triangles, with a per-cluster object-local
//! AABB. Both primary visibility (Phase 6 LOD selection) and the
//! shadow path will consume these clusters.
//!
//! Clustering runs through `meshopt::build_meshlets` so we get the
//! same locality / cone-fit quality as every other modern engine
//! using meshoptimizer. The output is repackaged into a flat IBO
//! where each cluster's indices land in a contiguous range
//! `[index_offset .. index_offset + index_count)`. The original VBO
//! is **not** reordered — Phase 6 indirect dispatch only needs
//! `first_index` + `index_count`, so the per-cluster vertex range
//! buys nothing and would force a parallel VBO permutation.

use bytemuck::{Pod, Zeroable};
use glam::{IVec3, Vec3};
use meshopt::{build_meshlets, VertexDataAdapter};

use crate::mesh_extract::MeshVertex;

/// NV-mesh-shader-style cluster cap — 64 unique verts per cluster.
///
/// Matches meshoptimizer's recommended value and is what every
/// modern AAA cluster pipeline (UE5 Nanite, Bevy virtual geometry,
/// id Tech 7 megatextures) targets. Smaller clusters mean more
/// indirect-draw entries; larger means worse cone-cull granularity
/// in Phase 6.
pub const MAX_VERTS_PER_CLUSTER: usize = 64;

/// Triangle cap per cluster. Must be `<= 512` and divisible by 4
/// per `meshopt::build_meshlets` contract.
pub const MAX_TRIS_PER_CLUSTER: usize = 124;

/// Cone weight passed to `meshopt::build_meshlets`. Zero disables
/// cone-cull bias (we don't cone-cull yet); Phase 6 may revisit if
/// it turns out to matter for shadow-pass culling.
const CONE_WEIGHT: f32 = 0.0;

const _: () = assert!(MAX_TRIS_PER_CLUSTER % 4 == 0);
const _: () = assert!(MAX_VERTS_PER_CLUSTER <= 256);
const _: () = assert!(MAX_TRIS_PER_CLUSTER <= 512);

/// Sentinel "no parent group" — used for clusters at the coarsest
/// LOD level (top of the DAG); they're always admitted because no
/// further simplification exists. `f32::INFINITY` projects to
/// `+inf` pixels so the Phase 6 selection rule's
/// `parent_group_error_proj >= threshold` arm is always true.
pub const PARENT_GROUP_ERROR_ROOT: f32 = f32::INFINITY;

/// One meshlet cluster. Stored both on `AssetEntry` (CPU side) and
/// uploaded to the GPU verbatim via `bytemuck::cast_slice` for the
/// Phase 6 LOD-selection compute pass to read.
///
/// 64 B. **Field offsets are hand-tuned to match WGSL std430 of the
/// matching `MeshletCluster` struct in `mesh_lod_select.wesl`:**
/// `vec3<f32>` consumes 12 bytes but starts on a 16-byte boundary,
/// and a following `u32` (align 4) sits *immediately* after with no
/// implicit padding to 16 — so `index_offset` is at byte 28, not
/// byte 32. A 4-byte `_pad0` matches std430's auto-pad bringing
/// `aabb_max` to byte 16, but there is **no `_pad1`** between
/// `aabb_max` and `index_offset` because std430 doesn't insert one.
/// Three trailing `u32`s of pad keep the layout 4-aligned and the
/// total stride at 64 B; using `vec2<u32>` would force the trailing
/// pad to start at byte 56 (8-aligned) and shift everything.
///
/// Phase 5 only fills `aabb_*` + `index_*`; `lod_level=0`,
/// `cluster_error=0`, `parent_group_error=∞`. Phase 6's DAG
/// builder grows the cluster set across LOD levels and fills the
/// error metrics.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct MeshletCluster {
    /// Object-local AABB minimum. World transform is applied per
    /// instance at LOD-select time.                              (0..12)
    pub aabb_min: [f32; 3],
    /// std430-mandated padding before the next `vec3<f32>`.      (12..16)
    pub _pad0: f32,
    /// Object-local AABB maximum.                                (16..28)
    pub aabb_max: [f32; 3],
    /// First index in the cluster's IBO range. Becomes
    /// `DrawIndexedIndirect.first_index` in Phase 6.              (28..32)
    pub index_offset: u32,
    /// Index count for this cluster. `index_count / 3` triangles.
    /// Becomes `DrawIndexedIndirect.index_count`.                 (32..36)
    pub index_count: u32,
    /// LOD level this cluster lives at. 0 = finest (original
    /// surface mesh), higher = progressively simplified.          (36..40)
    pub lod_level: u32,
    /// Reserved for future flags (cone-cull bits, etc).           (40..44)
    pub _pad2: u32,
    /// Maximum simplification error introduced at-or-below this
    /// cluster in the DAG (object-local units). Monotonically
    /// non-decreasing along chains from leaf → root. Phase 6
    /// selection rule admits this cluster iff its projected size
    /// is `< pixel_threshold`.                                    (44..48)
    pub cluster_error: f32,
    /// Simplification error of the parent group (the group whose
    /// simplification produced this cluster's parents at the
    /// next-coarser LOD). `PARENT_GROUP_ERROR_ROOT` for clusters
    /// at the coarsest level. Phase 6 admits the cluster iff
    /// `parent_group_error_proj >= pixel_threshold` AND
    /// `cluster_error_proj < pixel_threshold` — guarantees exactly
    /// one cluster picked per chain (Karis '21 SIGGRAPH).         (48..52)
    pub parent_group_error: f32,
    /// Trailing pad to a 64-byte stride.                          (52..64)
    pub _pad3: [u32; 3],
}

const _: () = assert!(std::mem::size_of::<MeshletCluster>() == 64);
const _: () = assert!(std::mem::align_of::<MeshletCluster>() == 4);
// Hand-checked field offsets — must match WGSL std430.
const _: () = {
    use std::mem::offset_of;
    assert!(offset_of!(MeshletCluster, aabb_min) == 0);
    assert!(offset_of!(MeshletCluster, _pad0) == 12);
    assert!(offset_of!(MeshletCluster, aabb_max) == 16);
    assert!(offset_of!(MeshletCluster, index_offset) == 28);
    assert!(offset_of!(MeshletCluster, index_count) == 32);
    assert!(offset_of!(MeshletCluster, lod_level) == 36);
    assert!(offset_of!(MeshletCluster, _pad2) == 40);
    assert!(offset_of!(MeshletCluster, cluster_error) == 44);
    assert!(offset_of!(MeshletCluster, parent_group_error) == 48);
    assert!(offset_of!(MeshletCluster, _pad3) == 52);
};

/// Partition a mesh into meshlet clusters with per-cluster AABBs.
///
/// * `vertices` — the asset's [`MeshVertex`] buffer. Position is
///   read from `local_pos`; the rest of the vertex is opaque to
///   meshopt (only the 12-byte position prefix is used).
/// * `indices` — the original triangle index buffer. Length must
///   be a multiple of 3.
///
/// Returns `(clusters, reordered_indices)`. The reordered indices
/// are a permutation of triangles from the original `indices`
/// (each cluster contributes a contiguous run). The vertex buffer
/// is unchanged — clusters reference the same vertex IDs.
///
/// Empty mesh → `(vec![], vec![])`. A mesh with `< 3` indices is
/// treated as empty (defensive; SN extraction never produces this).
pub fn cluster_mesh(
    vertices: &[MeshVertex],
    indices: &[u32],
) -> (Vec<MeshletCluster>, Vec<u32>) {
    if vertices.is_empty() || indices.len() < 3 {
        return (Vec::new(), Vec::new());
    }
    debug_assert_eq!(
        indices.len() % 3,
        0,
        "indices length must be a multiple of 3 (got {})",
        indices.len()
    );

    let vertex_bytes = bytemuck::cast_slice::<MeshVertex, u8>(vertices);
    let stride = std::mem::size_of::<MeshVertex>();
    // `local_pos` is the first field — position offset is 0.
    let adapter = VertexDataAdapter::new(vertex_bytes, stride, 0)
        .expect("MeshVertex layout matches VertexDataAdapter expectations");

    let meshlets = build_meshlets(
        indices,
        &adapter,
        MAX_VERTS_PER_CLUSTER,
        MAX_TRIS_PER_CLUSTER,
        CONE_WEIGHT,
    );

    let mut clusters = Vec::with_capacity(meshlets.len());
    let mut flat_indices: Vec<u32> = Vec::with_capacity(indices.len());

    for meshlet in meshlets.iter() {
        // meshlet.vertices: original VBO indices used by this cluster.
        // meshlet.triangles: per-cluster local indices into meshlet.vertices.
        let cluster_index_offset = flat_indices.len() as u32;

        // Materialize the cluster's triangles as absolute VBO indices.
        for &local in meshlet.triangles {
            flat_indices.push(meshlet.vertices[local as usize]);
        }

        let cluster_index_count = (flat_indices.len() as u32) - cluster_index_offset;

        // Compute object-local AABB by iterating the cluster's
        // unique vertex set. meshlet.vertices may include verts not
        // touched by any active triangle (meshopt over-allocates the
        // vertex slice up front and `optimizeMeshlet` reorders), so
        // restrict to verts actually referenced from the index slice.
        let mut aabb_min = [f32::INFINITY; 3];
        let mut aabb_max = [f32::NEG_INFINITY; 3];
        for &abs_vid in flat_indices[cluster_index_offset as usize..].iter() {
            let p = vertices[abs_vid as usize].local_pos;
            for k in 0..3 {
                if p[k] < aabb_min[k] {
                    aabb_min[k] = p[k];
                }
                if p[k] > aabb_max[k] {
                    aabb_max[k] = p[k];
                }
            }
        }

        clusters.push(MeshletCluster {
            aabb_min,
            _pad0: 0.0,
            aabb_max,
            index_offset: cluster_index_offset,
            index_count: cluster_index_count,
            lod_level: 0,
            _pad2: 0,
            cluster_error: 0.0,
            parent_group_error: PARENT_GROUP_ERROR_ROOT,
            _pad3: [0; 3],
        });
    }

    (clusters, flat_indices)
}

/// Convert a cluster's object-local float AABB to a finest-grid-coord
/// integer AABB (inclusive on both bounds).
///
/// **Used by Phase B R3** — the sculpt path needs a fast "which clusters
/// does this brush touch" query. The brush walks finest-voxel cells in
/// integer grid coords; cluster AABBs are floats in object-local space.
/// This lifts the cluster bounds into the brush's coordinate system once
/// per lookup so the intersection test is a 3-axis integer compare.
///
/// **One-cell pad on each side.** SN-cube vertices live on grid corners
/// (between cells); a cube at corner `c` reads cells `[c, c+1]` in each
/// axis. A cluster's vertex-position AABB therefore covers slightly
/// fewer cells than the cells whose state it actually depends on. We
/// pad ±1 cell to make the lookup conservative — false positives at the
/// boundary are cheap (re-extract a couple of extra clusters);
/// false negatives produce cracks at cluster seams.
///
/// Returns an inclusive `(min, max)` IVec3 pair.
pub fn cluster_grid_aabb(
    cluster: &MeshletCluster,
    grid_origin: Vec3,
    base_voxel_size: f32,
) -> (IVec3, IVec3) {
    let inv_vs = 1.0 / base_voxel_size;
    let min_local = Vec3::from(cluster.aabb_min);
    let max_local = Vec3::from(cluster.aabb_max);
    let min_grid_f = (min_local - grid_origin) * inv_vs;
    let max_grid_f = (max_local - grid_origin) * inv_vs;
    let grid_min = IVec3::new(
        min_grid_f.x.floor() as i32 - 1,
        min_grid_f.y.floor() as i32 - 1,
        min_grid_f.z.floor() as i32 - 1,
    );
    let grid_max = IVec3::new(
        max_grid_f.x.ceil() as i32 + 1,
        max_grid_f.y.ceil() as i32 + 1,
        max_grid_f.z.ceil() as i32 + 1,
    );
    (grid_min, grid_max)
}

/// AABB-overlap test in integer grid coords.
///
/// `cluster` bounds are **inclusive** (the cells at `cluster_min` and
/// `cluster_max` are inside the cluster's footprint). `brush` bounds are
/// **half-open** `[brush_lo, brush_hi)` to match the `brush_cell_range`
/// convention in `rkp_core::sculpt` — the brush walks cells in
/// `lo.x..hi.x` etc., so a cell at `hi - 1` is the last one inside the
/// brush.
///
/// Returns `true` when the two ranges overlap on every axis.
pub fn cluster_overlaps_brush_grid_aabb(
    cluster_min: IVec3,
    cluster_max: IVec3,
    brush_lo: IVec3,
    brush_hi: IVec3,
) -> bool {
    cluster_min.x < brush_hi.x
        && brush_lo.x <= cluster_max.x
        && cluster_min.y < brush_hi.y
        && brush_lo.y <= cluster_max.y
        && cluster_min.z < brush_hi.z
        && brush_lo.z <= cluster_max.z
}

/// Inflate per-cluster AABBs to cover the rest-pose extents of the
/// bones their vertices are weighted against.
///
/// **Why:** mesh-VS skinning (Phase 6.6) deforms vertex positions at
/// draw time. The static rest-pose AABBs `cluster_mesh` produces no
/// longer bound the animated geometry — a cluster on an arm can leave
/// its rest-pose AABB entirely as the arm raises. The Phase-6 LOD
/// selector would then wrongly cull that cluster from the camera or
/// shadow pass.
///
/// This is the **conservative** variant from the Phase 6.6 plan: at
/// load time, for each cluster, union the rest-pose AABBs of every
/// bone any of its vertices actually weights against. Bones cover the
/// geometry they animate, so unioning their rest AABBs bounds where
/// the cluster's geometry can land for any per-bone *transform* that
/// keeps each bone within the same volume it occupies at rest. That
/// covers typical character animation (limb rotations around
/// stationary roots); exotic poses (full somersaults, scale anims)
/// can still escape — those need the per-frame GPU recompute variant
/// the memory plan flags as a follow-on.
///
/// Inputs:
/// * `clusters` — mutated in place; `aabb_min` / `aabb_max` widened.
/// * `vertices` — the asset's vertex buffer (with `bone_indices` /
///   `bone_weights` baked in by `extract_surface_mesh`).
/// * `flat_indices` — index buffer; cluster slices read here.
/// * `rest_bone_aabbs` — per-bone `[min_x, min_y, min_z,
///   max_x, max_y, max_z]` extents at rest pose, indexed by the same
///   `bone_idx` the vertex's `bone_indices` carries.
///
/// No-op when `rest_bone_aabbs` is empty (unskinned asset) or any
/// cluster's vertex range is empty. Bone indices that fall outside
/// `rest_bone_aabbs` are silently skipped — defensive against stale
/// asset bakes.
pub fn expand_clusters_for_skinning(
    clusters: &mut [MeshletCluster],
    vertices: &[crate::mesh_extract::MeshVertex],
    flat_indices: &[u32],
    rest_bone_aabbs: &[[f32; 6]],
) {
    if rest_bone_aabbs.is_empty() {
        return;
    }
    for c in clusters.iter_mut() {
        let start = c.index_offset as usize;
        let end = start + c.index_count as usize;
        if end > flat_indices.len() {
            continue;
        }
        let mut seen_bones = [false; 256];
        for &vid in &flat_indices[start..end] {
            let v = match vertices.get(vid as usize) {
                Some(v) => v,
                None => continue,
            };
            // Only count bones that actually carry weight on this
            // vertex — `bone_indices` slot whose matching `bone_weights`
            // byte is zero is just zero-padding, not a real influence.
            for slot in 0..4u32 {
                let w = (v.bone_weights >> (slot * 8)) & 0xFFu32;
                if w == 0 {
                    continue;
                }
                let bone_idx = ((v.bone_indices >> (slot * 8)) & 0xFFu32) as usize;
                if bone_idx < seen_bones.len() {
                    seen_bones[bone_idx] = true;
                }
            }
        }
        let mut aabb_min = c.aabb_min;
        let mut aabb_max = c.aabb_max;
        for (bone_idx, &touched) in seen_bones.iter().enumerate() {
            if !touched {
                continue;
            }
            let aabb = match rest_bone_aabbs.get(bone_idx) {
                Some(a) => a,
                None => continue,
            };
            // Sentinel empty AABB (min > max) — skip; a bone the
            // skeleton declares but no geometry touches.
            if aabb[0] > aabb[3] || aabb[1] > aabb[4] || aabb[2] > aabb[5] {
                continue;
            }
            for k in 0..3 {
                if aabb[k] < aabb_min[k] {
                    aabb_min[k] = aabb[k];
                }
                if aabb[k + 3] > aabb_max[k] {
                    aabb_max[k] = aabb[k + 3];
                }
            }
        }
        c.aabb_min = aabb_min;
        c.aabb_max = aabb_max;
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn vert(p: [f32; 3]) -> MeshVertex {
        MeshVertex {
            local_pos: p,
            normal_oct: 0,
            leaf_attr_id: 0,
            bone_indices: 0,
            bone_weights: 0,
            _pad: 0,
        }
    }

    /// Helper: vertex with explicit bone weight on a single bone.
    fn vert_with_bone(p: [f32; 3], bone_idx: u8) -> MeshVertex {
        MeshVertex {
            local_pos: p,
            normal_oct: 0,
            leaf_attr_id: 0,
            bone_indices: u32::from_le_bytes([bone_idx, 0, 0, 0]),
            bone_weights: u32::from_le_bytes([255, 0, 0, 0]),
            _pad: 0,
        }
    }

    #[test]
    fn expand_clusters_unioning_referenced_bone_aabbs() {
        // Two-triangle cluster: 3 verts on bone 1, 1 vert on bone 2.
        // Rest AABBs: bone 1 lives near origin; bone 2 way out at +X.
        // Expanded cluster AABB must cover both.
        let v = vec![
            vert_with_bone([0.0, 0.0, 0.0], 1),
            vert_with_bone([1.0, 0.0, 0.0], 1),
            vert_with_bone([0.0, 1.0, 0.0], 1),
            vert_with_bone([0.5, 0.5, 0.0], 2),
        ];
        let i = vec![0, 1, 2, 0, 2, 3];
        let (mut clusters, flat) = cluster_mesh(&v, &i);
        assert_eq!(clusters.len(), 1, "small mesh fits in one cluster");
        let pre_min = clusters[0].aabb_min;
        let pre_max = clusters[0].aabb_max;
        // bone 0 unused, bone 1 at origin±0.5, bone 2 way out at +X.
        let bone_aabbs = vec![
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   // 0 (unused)
            [-0.5, -0.5, -0.5, 1.0, 1.0, 0.5],// 1
            [10.0, 0.0, 0.0, 12.0, 1.0, 0.5], // 2
        ];
        expand_clusters_for_skinning(&mut clusters, &v, &flat, &bone_aabbs);
        let c = &clusters[0];
        assert!(
            c.aabb_min[0] <= pre_min[0] && c.aabb_max[0] >= pre_max[0],
            "x-extent must not shrink ({}/{} → {}/{})",
            pre_min[0], pre_max[0], c.aabb_min[0], c.aabb_max[0],
        );
        assert!(
            c.aabb_max[0] >= 12.0,
            "expanded aabb_max.x must reach bone-2 rest-AABB max (12.0); got {}",
            c.aabb_max[0],
        );
    }

    #[test]
    fn expand_skips_unreferenced_bones() {
        // Cluster only weights bone 1; bone 2's rest AABB must NOT
        // pollute the cluster's expanded AABB.
        let v = vec![
            vert_with_bone([0.0, 0.0, 0.0], 1),
            vert_with_bone([1.0, 0.0, 0.0], 1),
            vert_with_bone([0.0, 1.0, 0.0], 1),
        ];
        let i = vec![0, 1, 2];
        let (mut clusters, flat) = cluster_mesh(&v, &i);
        let bone_aabbs = vec![
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.5, -0.5, -0.5, 1.0, 1.0, 0.5],
            [100.0, 100.0, 100.0, 200.0, 200.0, 200.0], // unused — must NOT count
        ];
        expand_clusters_for_skinning(&mut clusters, &v, &flat, &bone_aabbs);
        assert!(clusters[0].aabb_max[0] < 50.0, "unreferenced bone 2 leaked into AABB");
    }

    #[test]
    fn expand_no_op_for_unskinned_assets() {
        // Empty rest_bone_aabbs → no mutation regardless of vertex
        // bone fields.
        let v = vec![
            vert_with_bone([0.0, 0.0, 0.0], 1),
            vert_with_bone([1.0, 0.0, 0.0], 1),
            vert_with_bone([0.0, 1.0, 0.0], 1),
        ];
        let i = vec![0, 1, 2];
        let (mut clusters, flat) = cluster_mesh(&v, &i);
        let snapshot = clusters[0];
        expand_clusters_for_skinning(&mut clusters, &v, &flat, &[]);
        assert_eq!(clusters[0], snapshot, "unskinned asset must not mutate");
    }

    /// Each triangle as a sorted (a,b,c) tuple, for set-equality
    /// comparison across permutations.
    fn triangle_multiset(indices: &[u32]) -> HashMap<[u32; 3], usize> {
        let mut m = HashMap::new();
        for tri in indices.chunks_exact(3) {
            let mut t = [tri[0], tri[1], tri[2]];
            t.sort();
            *m.entry(t).or_insert(0) += 1;
        }
        m
    }

    #[test]
    fn cluster_layout_size_is_64() {
        // Belt-and-suspenders for the const_assert above.
        assert_eq!(std::mem::size_of::<MeshletCluster>(), 64);
    }

    #[test]
    fn phase5_cluster_defaults_lod0_zero_error_root_parent() {
        let v = vec![
            vert([0.0, 0.0, 0.0]),
            vert([1.0, 0.0, 0.0]),
            vert([0.0, 1.0, 0.0]),
        ];
        let (clusters, _) = cluster_mesh(&v, &[0, 1, 2]);
        let c = &clusters[0];
        assert_eq!(c.lod_level, 0);
        assert_eq!(c.cluster_error, 0.0);
        assert!(c.parent_group_error.is_infinite());
    }

    #[test]
    fn cluster_empty_mesh_yields_no_clusters() {
        let (clusters, indices) = cluster_mesh(&[], &[]);
        assert!(clusters.is_empty());
        assert!(indices.is_empty());
    }

    #[test]
    fn cluster_too_few_indices_yields_no_clusters() {
        // Two indices is below a triangle; treated as empty.
        let v = vec![vert([0.0, 0.0, 0.0]), vert([1.0, 0.0, 0.0])];
        let (clusters, indices) = cluster_mesh(&v, &[0, 1]);
        assert!(clusters.is_empty());
        assert!(indices.is_empty());
    }

    #[test]
    fn cluster_single_triangle_aabb_matches_triangle() {
        let v = vec![
            vert([0.0, 0.0, 0.0]),
            vert([1.0, 0.0, 0.0]),
            vert([0.0, 2.0, 3.0]),
        ];
        let i = vec![0, 1, 2];
        let (clusters, flat) = cluster_mesh(&v, &i);
        assert_eq!(clusters.len(), 1);
        assert_eq!(flat.len(), 3);
        let c = &clusters[0];
        assert_eq!(c.index_offset, 0);
        assert_eq!(c.index_count, 3);
        assert_eq!(c.aabb_min, [0.0, 0.0, 0.0]);
        assert_eq!(c.aabb_max, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn cluster_cube_fits_in_one_cluster() {
        // 8 verts, 12 tris — well under 64v/124t.
        let v = vec![
            vert([0.0, 0.0, 0.0]),
            vert([1.0, 0.0, 0.0]),
            vert([1.0, 1.0, 0.0]),
            vert([0.0, 1.0, 0.0]),
            vert([0.0, 0.0, 1.0]),
            vert([1.0, 0.0, 1.0]),
            vert([1.0, 1.0, 1.0]),
            vert([0.0, 1.0, 1.0]),
        ];
        #[rustfmt::skip]
        let i = vec![
            0, 1, 2,  0, 2, 3, // -z
            5, 4, 7,  5, 7, 6, // +z
            4, 0, 3,  4, 3, 7, // -x
            1, 5, 6,  1, 6, 2, // +x
            4, 5, 1,  4, 1, 0, // -y
            3, 2, 6,  3, 6, 7, // +y
        ];
        let (clusters, flat) = cluster_mesh(&v, &i);
        assert_eq!(clusters.len(), 1, "cube fits in one cluster");
        assert_eq!(flat.len(), 36, "12 tris × 3 indices preserved");
        let c = &clusters[0];
        assert_eq!(c.aabb_min, [0.0, 0.0, 0.0]);
        assert_eq!(c.aabb_max, [1.0, 1.0, 1.0]);
        assert_eq!(triangle_multiset(&i), triangle_multiset(&flat));
    }

    /// Build a flat 9×9 grid of vertices on the XZ plane and a
    /// triangle list that covers all 8×8 quads. 81 verts, 128 tris —
    /// exceeds the 64-vert cap so meshopt must split.
    fn grid_mesh(side: usize) -> (Vec<MeshVertex>, Vec<u32>) {
        let n = side as u32;
        let mut verts = Vec::new();
        for y in 0..side {
            for x in 0..side {
                verts.push(vert([x as f32, 0.0, y as f32]));
            }
        }
        let mut idx = Vec::new();
        for y in 0..(side as u32 - 1) {
            for x in 0..(side as u32 - 1) {
                let a = y * n + x;
                let b = y * n + x + 1;
                let c = (y + 1) * n + x + 1;
                let d = (y + 1) * n + x;
                idx.extend_from_slice(&[a, b, c, a, c, d]);
            }
        }
        (verts, idx)
    }

    #[test]
    fn cluster_grid_splits_when_over_vertex_cap() {
        let (v, i) = grid_mesh(9);
        assert_eq!(v.len(), 81);
        assert_eq!(i.len(), 128 * 3);

        let (clusters, flat) = cluster_mesh(&v, &i);
        assert!(
            clusters.len() >= 2,
            "81 verts must split across multiple clusters (got {})",
            clusters.len()
        );

        // Per-cluster caps respected.
        for (ci, c) in clusters.iter().enumerate() {
            assert!(
                c.index_count <= (MAX_TRIS_PER_CLUSTER * 3) as u32,
                "cluster {} index_count {} exceeds tri cap",
                ci,
                c.index_count
            );
            assert_eq!(
                c.index_count % 3,
                0,
                "cluster {} index_count must be tri-aligned",
                ci
            );
            // Unique-vertex count under MAX_VERTS_PER_CLUSTER.
            let unique: std::collections::HashSet<u32> = flat
                [c.index_offset as usize..(c.index_offset + c.index_count) as usize]
                .iter()
                .copied()
                .collect();
            assert!(
                unique.len() <= MAX_VERTS_PER_CLUSTER,
                "cluster {} has {} unique verts, cap {}",
                ci,
                unique.len(),
                MAX_VERTS_PER_CLUSTER
            );
        }

        // Index ranges tile the IBO with no gaps and no overlap.
        let mut cursor = 0u32;
        for c in &clusters {
            assert_eq!(c.index_offset, cursor, "clusters must be contiguous");
            cursor += c.index_count;
        }
        assert_eq!(cursor as usize, flat.len(), "all indices accounted for");

        // Triangle multiset preserved (permutation, no loss/dup).
        assert_eq!(triangle_multiset(&i), triangle_multiset(&flat));
    }

    #[test]
    fn cluster_grid_aabb_pads_one_cell_each_side() {
        // Single triangle with a tiny float AABB at the origin should
        // produce a 1-cell-padded grid AABB that comfortably encloses
        // the contributing cells.
        let c = MeshletCluster {
            aabb_min: [0.25, 0.5, 0.75],
            _pad0: 0.0,
            aabb_max: [1.25, 1.5, 1.75],
            index_offset: 0,
            index_count: 0,
            lod_level: 0,
            _pad2: 0,
            cluster_error: 0.0,
            parent_group_error: 0.0,
            _pad3: [0; 3],
        };
        let (gmin, gmax) = cluster_grid_aabb(&c, Vec3::ZERO, 1.0);
        assert_eq!(gmin, IVec3::new(-1, -1, -1), "floor(0.x) - 1");
        assert_eq!(gmax, IVec3::new(3, 3, 3), "ceil(1.x) + 1");
    }

    #[test]
    fn cluster_grid_aabb_respects_grid_origin_and_voxel_size() {
        // grid_origin shifts the cluster's float coords; base_vs scales them.
        let c = MeshletCluster {
            aabb_min: [4.0, 4.0, 4.0],
            _pad0: 0.0,
            aabb_max: [6.0, 6.0, 6.0],
            index_offset: 0,
            index_count: 0,
            lod_level: 0,
            _pad2: 0,
            cluster_error: 0.0,
            parent_group_error: 0.0,
            _pad3: [0; 3],
        };
        // grid_origin = (2, 2, 2), base_vs = 0.5
        // local 4.0 → grid (4-2)/0.5 = 4.0 → floor-1 = 3
        // local 6.0 → grid (6-2)/0.5 = 8.0 → ceil+1 = 9
        let (gmin, gmax) = cluster_grid_aabb(&c, Vec3::splat(2.0), 0.5);
        assert_eq!(gmin, IVec3::new(3, 3, 3));
        assert_eq!(gmax, IVec3::new(9, 9, 9));
    }

    #[test]
    fn cluster_overlap_brush_aabb_basic_cases() {
        // Two clusters: one at [0..2], one at [10..12] (inclusive).
        let a_min = IVec3::ZERO;
        let a_max = IVec3::new(2, 2, 2);
        let b_min = IVec3::splat(10);
        let b_max = IVec3::splat(12);

        // Brush at [1..3) overlaps A (cell 1 + 2 inside both), misses B.
        assert!(cluster_overlaps_brush_grid_aabb(a_min, a_max, IVec3::ONE, IVec3::splat(3)));
        assert!(!cluster_overlaps_brush_grid_aabb(b_min, b_max, IVec3::ONE, IVec3::splat(3)));

        // Brush at [3..10) hits neither — strictly between them.
        assert!(!cluster_overlaps_brush_grid_aabb(a_min, a_max, IVec3::splat(3), IVec3::splat(10)));
        assert!(!cluster_overlaps_brush_grid_aabb(b_min, b_max, IVec3::splat(3), IVec3::splat(10)));

        // Brush at [10..15) overlaps B only.
        assert!(!cluster_overlaps_brush_grid_aabb(a_min, a_max, IVec3::splat(10), IVec3::splat(15)));
        assert!(cluster_overlaps_brush_grid_aabb(b_min, b_max, IVec3::splat(10), IVec3::splat(15)));
    }

    #[test]
    fn cluster_overlap_brush_aabb_edge_touches_inclusive_max() {
        // Inclusive cluster_max + half-open brush: a brush at [2..3) and
        // a cluster ending at exactly 2 should overlap (cell 2 is the
        // last brush cell, and is inside the cluster).
        assert!(cluster_overlaps_brush_grid_aabb(
            IVec3::ZERO,
            IVec3::splat(2),
            IVec3::splat(2),
            IVec3::splat(3),
        ));
        // Whereas a brush at [3..4) and the same cluster do NOT overlap
        // (the first brush cell is past the cluster's last cell).
        assert!(!cluster_overlaps_brush_grid_aabb(
            IVec3::ZERO,
            IVec3::splat(2),
            IVec3::splat(3),
            IVec3::splat(4),
        ));
    }

    #[test]
    fn cluster_overlap_brush_aabb_per_axis_independence() {
        // Cluster overlaps brush only on X axis — must report no overlap.
        assert!(!cluster_overlaps_brush_grid_aabb(
            IVec3::new(0, 100, 100),
            IVec3::new(10, 110, 110),
            IVec3::new(5, 0, 0),
            IVec3::new(15, 50, 50),
        ));
        // Overlap on all axes — overlap reported.
        assert!(cluster_overlaps_brush_grid_aabb(
            IVec3::new(0, 100, 100),
            IVec3::new(10, 110, 110),
            IVec3::new(5, 105, 105),
            IVec3::new(15, 115, 115),
        ));
    }

    #[test]
    fn cluster_aabbs_contain_their_vertices() {
        let (v, i) = grid_mesh(9);
        let (clusters, flat) = cluster_mesh(&v, &i);
        for (ci, c) in clusters.iter().enumerate() {
            for vid in &flat[c.index_offset as usize..(c.index_offset + c.index_count) as usize] {
                let p = v[*vid as usize].local_pos;
                for k in 0..3 {
                    assert!(
                        c.aabb_min[k] <= p[k] && p[k] <= c.aabb_max[k],
                        "cluster {}: vertex {} pos[{}] = {} outside [{}, {}]",
                        ci,
                        vid,
                        k,
                        p[k],
                        c.aabb_min[k],
                        c.aabb_max[k]
                    );
                }
            }
        }
    }
}
