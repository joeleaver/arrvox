//! Cluster DAG (LOD pyramid) construction — Phase 6.1.
//!
//! Builds a Karis-Nanite-style cluster DAG over a per-asset surface
//! mesh: each LOD level groups its clusters into spatial groups of
//! ~`GROUP_SIZE_TARGET`, locks the group's exterior-boundary
//! vertices (those shared with clusters outside the group),
//! `meshopt::simplify_with_locks` reduces the merged-and-locked
//! triangle set to `LOD_REDUCTION_TARGET` of its triangle count,
//! and the simplified result is re-clustered into the next level.
//!
//! Each cluster carries `cluster_error` (max simplification error
//! at-or-below in the DAG, monotonically non-decreasing along
//! root-ward chains) and `parent_group_error` (the simplification
//! error of the group that consumed this cluster's siblings, or
//! `PARENT_GROUP_ERROR_ROOT` when this cluster has no parent group
//! — i.e., it's at the coarsest level the DAG reached). The Phase
//! 6.2 GPU LOD-select compute pass admits a cluster iff its
//! parent's projected error is at-or-above the pixel threshold AND
//! its own projected error is below it; this guarantees exactly
//! one cluster per chain is rendered (Karis SIGGRAPH '21).
//!
//! Crack-avoidance: locking the *group's* exterior boundary
//! (rather than each cluster's boundary) preserves the geometry
//! shared with neighbouring groups across LOD levels. So when
//! group A picks LOD N and adjacent group B picks LOD N-1, the
//! group-A LOD-N cluster's boundary vertices remain at the same
//! positions as the group-B LOD-(N-1) cluster's boundary vertices —
//! no T-junction, no crack. Karis paper §3.2.

use std::collections::{HashMap, HashSet};

use glam::Vec3;
use meshopt::{simplify_with_locks, SimplifyOptions, VertexDataAdapter};

use super::cluster::{cluster_mesh, MeshletCluster, PARENT_GROUP_ERROR_ROOT};
use super::extract::MeshVertex;

/// Number of LOD levels the DAG attempts to build (LOD 0 is the
/// finest; LOD `LOD_LEVELS - 1` is the coarsest the DAG converges
/// to). Construction may stop early if a level's simplification
/// makes no progress or only one cluster remains.
pub const LOD_LEVELS: usize = 4;

/// Target group size — Nanite uses 4. Per LOD level, the algorithm
/// greedy-clusters the previous level's cluster set into groups of
/// roughly this many.
pub const GROUP_SIZE_TARGET: usize = 4;

/// Per-group simplification target — fraction of input triangles
/// remaining after `meshopt::simplify_with_locks`. 0.5 means each
/// LOD level has ~50% the previous level's triangles.
pub const LOD_REDUCTION_TARGET: f32 = 0.5;

/// The DAG output for one asset.
///
/// `clusters` is a flat list across all LOD levels; the LOD-0
/// clusters come first (sorted by `index_offset`), then LOD 1,
/// etc. `indices` is the concatenated index buffer; each cluster's
/// `index_offset` is global into this single buffer.
///
/// The vertex buffer is unchanged from
/// [`super::extract::extract_surface_mesh`] — `meshopt::simplify`
/// only ever drops vertices from the index buffer; all LOD levels
/// reference the same VBO.
#[derive(Debug, Clone)]
pub struct ClusterDag {
    pub clusters: Vec<MeshletCluster>,
    pub indices: Vec<u32>,
    /// Index range `(start, end)` of LOD-0 clusters in
    /// `clusters`. Useful for a Phase 6.1-only dispatch path that
    /// still wants to render only the original surface (visuals
    /// unchanged) while the DAG is uploaded.
    pub lod0_cluster_range: (u32, u32),
    /// Index range `(start, end)` of LOD-0 indices in `indices`.
    pub lod0_index_range: (u32, u32),
}

impl ClusterDag {
    pub fn empty() -> Self {
        Self {
            clusters: Vec::new(),
            indices: Vec::new(),
            lod0_cluster_range: (0, 0),
            lod0_index_range: (0, 0),
        }
    }
}

/// Build the cluster DAG.
///
/// Returns [`ClusterDag::empty`] for empty input. Otherwise produces
/// up to `LOD_LEVELS` levels of clusters; the build may converge
/// early (1 cluster remaining at some level) or stall (a level's
/// simplification fails to reduce triangles below the target),
/// either of which terminates DAG growth cleanly.
pub fn build_cluster_dag(vertices: &[MeshVertex], indices: &[u32]) -> ClusterDag {
    if vertices.is_empty() || indices.len() < 3 {
        return ClusterDag::empty();
    }

    let mut all_clusters: Vec<MeshletCluster> = Vec::new();
    let mut all_indices: Vec<u32> = Vec::new();

    // LOD 0 — cluster the original mesh.
    let (lod0_clusters, lod0_indices) = cluster_mesh(vertices, indices);
    let lod0_index_base = all_indices.len() as u32;
    all_indices.extend(lod0_indices);
    let lod0_cluster_base = all_clusters.len();
    for mut c in lod0_clusters {
        c.index_offset += lod0_index_base;
        // lod_level/cluster_error/parent_group_error already at
        // sentinels from cluster_mesh.
        all_clusters.push(c);
    }
    let lod0_cluster_end = all_clusters.len();
    let lod0_cluster_range = (lod0_cluster_base as u32, lod0_cluster_end as u32);
    let lod0_index_range = (lod0_index_base, all_indices.len() as u32);

    let mut prev_level_range: std::ops::Range<usize> = lod0_cluster_base..lod0_cluster_end;

    for lod in 1..LOD_LEVELS {
        if prev_level_range.len() <= 1 {
            break; // DAG converged to a single cluster
        }

        let prev_indices: Vec<usize> = prev_level_range.clone().collect();
        let groups = group_clusters_spatially(&all_clusters, &prev_indices, GROUP_SIZE_TARGET);
        if groups.is_empty() {
            break;
        }

        // Per-vertex group ownership for boundary-lock detection.
        // Map: vertex_id → set of group_ids that reference it from
        // any of their member prev-level clusters.
        let vert_to_groups = build_vert_to_groups(&all_clusters, &all_indices, &prev_indices, &groups);

        let new_level_start = all_clusters.len();

        for (g_idx, group_local) in groups.iter().enumerate() {
            // Translate group's prev-level positions back to global cluster indices.
            let group_global: Vec<usize> =
                group_local.iter().map(|&li| prev_indices[li]).collect();

            // Merge member clusters' triangles into a single index buffer.
            let mut merged_tris: Vec<u32> = Vec::new();
            for &gi in &group_global {
                let c = &all_clusters[gi];
                merged_tris.extend_from_slice(
                    &all_indices
                        [c.index_offset as usize..(c.index_offset + c.index_count) as usize],
                );
            }
            if merged_tris.len() < 3 {
                continue;
            }

            // Group-boundary vertex locks. Lock any vertex shared
            // with a cluster in another group at this LOD level.
            let lock_flags = compute_group_boundary_locks(
                &merged_tris,
                vertices.len(),
                &vert_to_groups,
                g_idx,
            );

            // meshopt::simplify_with_locks. Target ~50% triangles;
            // unbounded error budget (the simplifier returns the
            // actual error in `result_error`, which we capture as
            // the group's parametric error metric).
            let target_index_count = ((merged_tris.len() as f32 * LOD_REDUCTION_TARGET) as usize
                / 3)
                * 3;
            let mut group_error = 0.0_f32;
            let simplified =
                simplify_meshopt(&merged_tris, vertices, &lock_flags, target_index_count, &mut group_error);

            if simplified.len() < 3 || simplified.len() >= merged_tris.len() {
                // No reduction (simplifier was blocked by locks or
                // topology). The group's prev-level clusters retain
                // `parent_group_error = ∞` and become DAG leaves at
                // this branch, which is correct: the LOD selection
                // rule will always render them.
                continue;
            }

            // Re-cluster the simplified triangles.
            let (sub_clusters, sub_indices) = cluster_mesh(vertices, &simplified);
            if sub_clusters.is_empty() {
                continue;
            }
            let sub_index_base = all_indices.len() as u32;
            all_indices.extend(sub_indices);

            // cluster_error: max along chain from leaves to here.
            let max_input_error = group_global
                .iter()
                .map(|&gi| all_clusters[gi].cluster_error)
                .fold(0.0_f32, f32::max);
            let new_cluster_error = max_input_error.max(group_error);

            for mut sc in sub_clusters {
                sc.index_offset += sub_index_base;
                sc.lod_level = lod as u32;
                sc.cluster_error = new_cluster_error;
                sc.parent_group_error = PARENT_GROUP_ERROR_ROOT;
                all_clusters.push(sc);
            }

            // Backfill parent_group_error on prev-level clusters
            // that this group consumed — they're no longer DAG
            // leaves, so the LOD-selection rule needs to know what
            // error the next coarser level introduced.
            for &gi in &group_global {
                all_clusters[gi].parent_group_error = group_error;
            }
        }

        let new_level_end = all_clusters.len();
        if new_level_end == new_level_start {
            // No group produced new clusters this level; DAG growth
            // has stalled (every group either failed to simplify or
            // had < 3 input tris). Stop.
            break;
        }
        prev_level_range = new_level_start..new_level_end;
    }

    ClusterDag {
        clusters: all_clusters,
        indices: all_indices,
        lod0_cluster_range,
        lod0_index_range,
    }
}

/// Greedy spatial grouping: pick an unused seed cluster, add its
/// `target_size - 1` AABB-centroid-nearest unused neighbours,
/// repeat until every cluster is in a group. Returns groups as
/// vectors of indices into `prev_indices` (positional within the
/// LOD level — *not* into `clusters` directly).
///
/// Brute-force O(N²) on cluster count per level. With ≤ low-
/// thousands of clusters per LOD level this is ~milliseconds at
/// load time and trivially correct.
fn group_clusters_spatially(
    clusters: &[MeshletCluster],
    prev_indices: &[usize],
    target_size: usize,
) -> Vec<Vec<usize>> {
    if prev_indices.is_empty() {
        return Vec::new();
    }
    let n = prev_indices.len();
    let centroids: Vec<Vec3> = prev_indices
        .iter()
        .map(|&gi| {
            let lo = Vec3::from_array(clusters[gi].aabb_min);
            let hi = Vec3::from_array(clusters[gi].aabb_max);
            (lo + hi) * 0.5
        })
        .collect();

    let mut used = vec![false; n];
    let mut groups: Vec<Vec<usize>> = Vec::new();

    for seed in 0..n {
        if used[seed] {
            continue;
        }
        used[seed] = true;
        let mut group = vec![seed];

        // Sort remaining unused clusters by distance to seed.
        let mut candidates: Vec<(usize, f32)> = (0..n)
            .filter(|&i| !used[i])
            .map(|i| (i, centroids[seed].distance_squared(centroids[i])))
            .collect();
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        for (i, _) in candidates.iter().take(target_size - 1) {
            used[*i] = true;
            group.push(*i);
        }
        groups.push(group);
    }
    groups
}

/// Build the per-vertex → set-of-group-ids map used to detect
/// group-boundary verts during simplify-with-locks.
fn build_vert_to_groups(
    clusters: &[MeshletCluster],
    indices: &[u32],
    prev_indices: &[usize],
    groups: &[Vec<usize>],
) -> HashMap<u32, HashSet<u32>> {
    let mut map: HashMap<u32, HashSet<u32>> = HashMap::new();
    for (g_idx, group_local) in groups.iter().enumerate() {
        for &li in group_local {
            let gi = prev_indices[li];
            let c = &clusters[gi];
            let span = &indices[c.index_offset as usize..(c.index_offset + c.index_count) as usize];
            for &v in span {
                map.entry(v).or_default().insert(g_idx as u32);
            }
        }
    }
    map
}

/// Compute the per-vertex lock flags for a single group's
/// simplification call. A vertex is locked iff it's referenced by
/// any prev-level cluster *outside* this group — i.e., it's on the
/// group's exterior boundary and must keep its position so
/// adjacent groups' boundary geometry continues to match.
fn compute_group_boundary_locks(
    merged_tris: &[u32],
    vertex_count: usize,
    vert_to_groups: &HashMap<u32, HashSet<u32>>,
    this_group: usize,
) -> Vec<bool> {
    let mut locks = vec![false; vertex_count];
    for &v in merged_tris {
        if let Some(groups) = vert_to_groups.get(&v) {
            if groups.iter().any(|&g| g != this_group as u32) {
                locks[v as usize] = true;
            }
        }
    }
    locks
}

/// Thin wrapper around `meshopt::simplify_with_locks`. Pulled out
/// so tests can swap in a deterministic stub.
fn simplify_meshopt(
    indices: &[u32],
    vertices: &[MeshVertex],
    lock_flags: &[bool],
    target_index_count: usize,
    out_error: &mut f32,
) -> Vec<u32> {
    let vertex_bytes = bytemuck::cast_slice::<MeshVertex, u8>(vertices);
    let stride = std::mem::size_of::<MeshVertex>();
    let adapter = VertexDataAdapter::new(vertex_bytes, stride, 0)
        .expect("MeshVertex layout matches VertexDataAdapter");

    // `target_error = f32::MAX` lets meshopt simplify until it can't
    // reach `target_index_count`, returning the actual parametric
    // error in `out_error`. ErrorAbsolute makes the metric
    // object-local so we can compare across assets.
    simplify_with_locks(
        indices,
        &adapter,
        lock_flags,
        target_index_count,
        f32::MAX,
        SimplifyOptions::ErrorAbsolute,
        Some(out_error),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh_pass::cluster::MAX_VERTS_PER_CLUSTER;

    fn vert(p: [f32; 3]) -> MeshVertex {
        MeshVertex {
            local_pos: p,
            normal_oct: 0,
            leaf_attr_id: 0,
            _pad: [0; 3],
        }
    }

    /// 17×17 grid → 289 verts, 512 tris. Big enough to produce
    /// multiple LOD-0 clusters and let the DAG actually grow.
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
    fn dag_empty_input_is_empty() {
        let dag = build_cluster_dag(&[], &[]);
        assert!(dag.clusters.is_empty());
        assert!(dag.indices.is_empty());
        assert_eq!(dag.lod0_cluster_range, (0, 0));
        assert_eq!(dag.lod0_index_range, (0, 0));
    }

    #[test]
    fn dag_single_triangle_has_one_lod0_cluster_and_no_higher_levels() {
        let v = vec![
            vert([0.0, 0.0, 0.0]),
            vert([1.0, 0.0, 0.0]),
            vert([0.0, 1.0, 0.0]),
        ];
        let dag = build_cluster_dag(&v, &[0, 1, 2]);
        assert_eq!(dag.clusters.len(), 1);
        assert_eq!(dag.clusters[0].lod_level, 0);
        assert_eq!(dag.clusters[0].cluster_error, 0.0);
        assert!(dag.clusters[0].parent_group_error.is_infinite());
        // LOD-0 range covers the only cluster.
        assert_eq!(dag.lod0_cluster_range, (0, 1));
    }

    #[test]
    fn dag_lod0_indices_match_phase5_clustering() {
        let (v, i) = grid_mesh(17);
        let dag = build_cluster_dag(&v, &i);
        let (phase5_clusters, phase5_indices) = cluster_mesh(&v, &i);

        // Phase 5 clustering of the same input must reproduce the
        // DAG's LOD-0 cluster set bit-for-bit. Same offsets, same
        // index data — the DAG is purely additive over Phase 5.
        let lod0_indices = &dag.indices[dag.lod0_index_range.0 as usize..dag.lod0_index_range.1 as usize];
        assert_eq!(lod0_indices, phase5_indices.as_slice());

        let dag_lod0: Vec<MeshletCluster> = dag.clusters[dag.lod0_cluster_range.0 as usize..dag.lod0_cluster_range.1 as usize]
            .iter()
            .map(|c| {
                // Mask the LOD-fields the DAG sets (cluster_error
                // and parent_group_error stay at sentinels for
                // LOD-0 leaves; lod_level is 0).
                MeshletCluster {
                    parent_group_error: f32::INFINITY,
                    ..*c
                }
            })
            .collect();
        // The DAG's LOD-0 entries differ from Phase 5's only in
        // `parent_group_error` (Phase 5 sets ∞; DAG may backfill
        // when a group consumes them). Strip that for comparison.
        let phase5_normalized: Vec<MeshletCluster> = phase5_clusters
            .iter()
            .map(|c| MeshletCluster {
                parent_group_error: f32::INFINITY,
                ..*c
            })
            .collect();
        assert_eq!(dag_lod0, phase5_normalized);
    }

    #[test]
    fn dag_grows_at_least_two_levels_on_grid_mesh() {
        let (v, i) = grid_mesh(17);
        let dag = build_cluster_dag(&v, &i);
        let max_lod = dag.clusters.iter().map(|c| c.lod_level).max().unwrap_or(0);
        assert!(
            max_lod >= 1,
            "DAG should grow past LOD 0 on a non-trivial mesh (got max_lod={})",
            max_lod
        );
    }

    #[test]
    fn dag_per_cluster_caps_respected_at_every_level() {
        let (v, i) = grid_mesh(17);
        let dag = build_cluster_dag(&v, &i);
        for c in &dag.clusters {
            assert_eq!(c.index_count % 3, 0);
            // Vertex-cap is the meshopt invariant; verify on the
            // unique-vertex set from the IBO range.
            let span = &dag.indices
                [c.index_offset as usize..(c.index_offset + c.index_count) as usize];
            let unique: HashSet<u32> = span.iter().copied().collect();
            assert!(
                unique.len() <= MAX_VERTS_PER_CLUSTER,
                "cluster lod={} has {} unique verts (cap {})",
                c.lod_level,
                unique.len(),
                MAX_VERTS_PER_CLUSTER
            );
        }
    }

    #[test]
    fn dag_cluster_error_monotonic_along_lod_level() {
        // Within a level, cluster_error is uniform-per-group; across
        // levels it must be monotonically non-decreasing.
        let (v, i) = grid_mesh(17);
        let dag = build_cluster_dag(&v, &i);
        let max_lod = dag.clusters.iter().map(|c| c.lod_level).max().unwrap_or(0);

        let mut max_err_per_lod = vec![0.0_f32; (max_lod + 1) as usize];
        for c in &dag.clusters {
            let m = max_err_per_lod[c.lod_level as usize];
            max_err_per_lod[c.lod_level as usize] = m.max(c.cluster_error);
        }
        for w in max_err_per_lod.windows(2) {
            assert!(
                w[0] <= w[1],
                "max cluster_error must be monotonically non-decreasing across LOD levels: got {:?}",
                max_err_per_lod
            );
        }
    }

    #[test]
    fn dag_consumed_lod0_clusters_have_finite_parent_group_error() {
        // Any LOD-0 cluster whose group was successfully simplified
        // must have `parent_group_error` < ∞. Conversely, any
        // LOD-0 cluster that became a DAG leaf (its group failed
        // to simplify, or there is no level above it) keeps `∞`.
        let (v, i) = grid_mesh(17);
        let dag = build_cluster_dag(&v, &i);
        let max_lod = dag.clusters.iter().map(|c| c.lod_level).max().unwrap_or(0);
        if max_lod == 0 {
            // Mesh too small for the DAG to grow — nothing to assert
            // beyond what `dag_grows_at_least_two_levels_on_grid_mesh`
            // already covers. Skip.
            return;
        }
        let any_consumed = dag.clusters[..dag.lod0_cluster_range.1 as usize]
            .iter()
            .any(|c| c.parent_group_error.is_finite());
        assert!(
            any_consumed,
            "at least one LOD-0 cluster must have been consumed by a higher-LOD group when max_lod={}",
            max_lod
        );
    }

    #[test]
    fn dag_lod0_aabbs_cover_the_input_mesh() {
        let (v, i) = grid_mesh(17);
        let dag = build_cluster_dag(&v, &i);
        let lod0 = &dag.clusters[..dag.lod0_cluster_range.1 as usize];
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for c in lod0 {
            for k in 0..3 {
                if c.aabb_min[k] < min[k] {
                    min[k] = c.aabb_min[k];
                }
                if c.aabb_max[k] > max[k] {
                    max[k] = c.aabb_max[k];
                }
            }
        }
        // Grid spans (0, 0, 0)..(side-1, 0, side-1) = (0,0,0)..(16,0,16)
        assert!(min[0] <= 0.0 && max[0] >= 16.0);
        assert!(min[2] <= 0.0 && max[2] >= 16.0);
    }

    #[test]
    fn group_clusters_spatially_partitions_all() {
        // Synthetic: 7 clusters at distinct positions; group_size=3
        // → groups of {3, 3, 1}.
        let mk = |center: [f32; 3]| MeshletCluster {
            aabb_min: [center[0] - 0.1, center[1] - 0.1, center[2] - 0.1],
            _pad0: 0.0,
            aabb_max: [center[0] + 0.1, center[1] + 0.1, center[2] + 0.1],
            _pad1: 0.0,
            index_offset: 0,
            index_count: 3,
            lod_level: 0,
            _pad2: 0,
            cluster_error: 0.0,
            parent_group_error: f32::INFINITY,
            _pad3: [0; 2],
        };
        let clusters: Vec<MeshletCluster> = (0..7).map(|i| mk([i as f32, 0.0, 0.0])).collect();
        let prev: Vec<usize> = (0..7).collect();
        let groups = group_clusters_spatially(&clusters, &prev, 3);

        let total: usize = groups.iter().map(|g| g.len()).sum();
        assert_eq!(total, 7, "every cluster appears exactly once");
        let mut seen: HashSet<usize> = HashSet::new();
        for g in &groups {
            for &i in g {
                assert!(seen.insert(i), "no cluster appears in more than one group");
            }
        }
    }

    #[test]
    fn compute_group_boundary_locks_marks_cross_group_verts() {
        // Two groups: {tri 0-1-2} and {tri 2-3-4}. Vertex 2 is
        // shared → must be locked when simplifying either group.
        let mut vert_to_groups: HashMap<u32, HashSet<u32>> = HashMap::new();
        vert_to_groups.entry(0).or_default().insert(0);
        vert_to_groups.entry(1).or_default().insert(0);
        vert_to_groups.entry(2).or_default().insert(0);
        vert_to_groups.entry(2).or_default().insert(1);
        vert_to_groups.entry(3).or_default().insert(1);
        vert_to_groups.entry(4).or_default().insert(1);

        // Group 0's merged tris = [0,1,2]; vert 2 should be locked.
        let g0_locks = compute_group_boundary_locks(&[0, 1, 2], 5, &vert_to_groups, 0);
        assert_eq!(g0_locks, vec![false, false, true, false, false]);

        // Group 1's merged tris = [2,3,4]; vert 2 should be locked.
        let g1_locks = compute_group_boundary_locks(&[2, 3, 4], 5, &vert_to_groups, 1);
        assert_eq!(g1_locks, vec![false, false, true, false, false]);
    }
}
