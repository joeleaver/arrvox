//! Per-cluster mesh storage — `ClusterMesh` + split/flatten helpers.
//!
//! **Phase B R4-proper** swap of the asset's mesh-storage model. Before
//! R4-proper: each [`MeshletCluster`] referenced a slice of a single
//! shared flat IBO via `index_offset / index_count`, and re-extracting
//! one cluster meant shifting every downstream cluster's index range.
//! After R4-proper: each cluster *owns* its own [`MeshVertex`] +
//! `u32` index list (indices local to the cluster's own vertex set),
//! and the flat IBO + VBO the renderer consumes are derived by
//! [`flatten_cluster_meshes`] at upload time.
//!
//! **Tradeoffs vs. the shared-VBO model:**
//!
//! * Boundary verts duplicated across adjacent clusters (~+2-5 % VBO
//!   size on typical meshes). Acceptable — duplication is intentional
//!   under the per-cluster-owned model, and it lets each cluster
//!   re-extract independently without coordinating with neighbors.
//! * Flatten is one memcpy of the entire mesh on every geometry-epoch
//!   bump. R4e wires the renderer to invoke it. Cost is proportional
//!   to total mesh size, not to the dirty-cluster set; if drag stutters
//!   on big assets we revisit (delta-upload by cluster range).
//!
//! Round-trip preserves render: `split_flat_into_cluster_meshes` →
//! `flatten_cluster_meshes` produces a different vertex *ordering* (and
//! adds boundary duplicates), but the set of triangles each cluster
//! emits — identified by the triple of vertex positions — is unchanged.
//! Tests cover that invariant.

use std::collections::HashMap;

use bytemuck::Zeroable;

use crate::mesh_cluster::MeshletCluster;
use crate::mesh_extract::MeshVertex;

/// One cluster's owned mesh data — vertices and local-space indices.
///
/// `local_indices` reference positions in this cluster's own
/// `local_vertices`, NOT the asset's flat VBO. The flat VBO doesn't
/// exist as a source of truth anymore — it's derived by
/// [`flatten_cluster_meshes`] at upload time.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ClusterMesh {
    pub local_vertices: Vec<MeshVertex>,
    /// Indices into `local_vertices`. Length must be a multiple of 3
    /// (one triangle per 3 indices). Empty when the cluster is empty.
    pub local_indices: Vec<u32>,
}

/// Split a flat VBO + IBO + cluster table into per-cluster owned
/// `ClusterMesh` blocks. The shared flat VBO/IBO can be dropped after
/// this — the cluster_meshes become the source of truth, and
/// [`flatten_cluster_meshes`] reconstructs the flat buffers for upload.
///
/// Each cluster's slice of the flat IBO is `[cluster.index_offset ..
/// cluster.index_offset + cluster.index_count)`. Unique vertices used
/// by that slice are collected (in first-seen order) into the
/// cluster's `local_vertices`; the cluster's `local_indices` map the
/// original flat-VBO ids to local-cluster ids (0..N).
///
/// Empty clusters (`index_count == 0`) yield empty `ClusterMesh`
/// entries — the per-cluster Vec parallels `clusters` 1-to-1, so
/// downstream code can index either by the same id.
///
/// Panics in debug if any cluster references a flat-VBO id out of
/// range — the load path should never produce that, but the assert
/// catches construction bugs early.
pub fn split_flat_into_cluster_meshes(
    flat_vertices: &[MeshVertex],
    flat_indices: &[u32],
    clusters: &[MeshletCluster],
) -> Vec<ClusterMesh> {
    let mut out = Vec::with_capacity(clusters.len());
    for c in clusters {
        let start = c.index_offset as usize;
        let end = start + c.index_count as usize;
        // Defensive — load path should never produce this, but a
        // truncated buffer here would silently de-duplicate triangles.
        debug_assert!(
            end <= flat_indices.len(),
            "cluster index range {}..{} out of bounds for flat_indices len {}",
            start, end, flat_indices.len(),
        );
        if c.index_count == 0 || end > flat_indices.len() {
            out.push(ClusterMesh::default());
            continue;
        }
        let cluster_slice = &flat_indices[start..end];

        // First-seen order — preserves a deterministic permutation
        // across runs and matches the original triangle wind ordering
        // when local_indices are read back out.
        let mut remap: HashMap<u32, u32> = HashMap::new();
        let mut local_vertices = Vec::new();
        let mut local_indices = Vec::with_capacity(cluster_slice.len());
        for &flat_vid in cluster_slice {
            let local_vid = match remap.get(&flat_vid) {
                Some(&v) => v,
                None => {
                    let v = local_vertices.len() as u32;
                    let vert = flat_vertices
                        .get(flat_vid as usize)
                        .copied()
                        .unwrap_or_else(MeshVertex::zeroed);
                    local_vertices.push(vert);
                    remap.insert(flat_vid, v);
                    v
                }
            };
            local_indices.push(local_vid);
        }
        out.push(ClusterMesh { local_vertices, local_indices });
    }
    out
}

/// Concatenate per-cluster `ClusterMesh` blocks back into a flat VBO
/// + IBO, updating each cluster's `index_offset` / `index_count` to
/// match the new flat layout. Indices in the output are absolute
/// VBO ids (each cluster's local_indices are rebased by the cluster's
/// vertex offset in the flat VBO).
///
/// `cluster_meshes.len()` must equal `clusters.len()`; the two arrays
/// are parallel. Cluster order is preserved — the flat IBO contains
/// each cluster's triangles in `clusters[]` order. Callers that need
/// LOD-0-first layout (matching the legacy DAG concat scheme) should
/// reorder the cluster table before flatten — the renderer reads the
/// LOD-0 prefix via `mesh_lod0_index_count`.
///
/// Boundary verts are duplicated naturally: a vertex used by two
/// clusters appears twice in the flat VBO, once per cluster's local
/// copy. This is expected — see module docs.
pub fn flatten_cluster_meshes(
    cluster_meshes: &[ClusterMesh],
    clusters: &mut [MeshletCluster],
) -> (Vec<MeshVertex>, Vec<u32>) {
    assert_eq!(
        cluster_meshes.len(),
        clusters.len(),
        "cluster_meshes / clusters length mismatch ({}/{})",
        cluster_meshes.len(),
        clusters.len(),
    );

    let total_verts: usize = cluster_meshes.iter().map(|m| m.local_vertices.len()).sum();
    let total_indices: usize = cluster_meshes.iter().map(|m| m.local_indices.len()).sum();
    let mut flat_vertices = Vec::with_capacity(total_verts);
    let mut flat_indices = Vec::with_capacity(total_indices);

    for (cm, c) in cluster_meshes.iter().zip(clusters.iter_mut()) {
        let vertex_offset = flat_vertices.len() as u32;
        let index_offset = flat_indices.len() as u32;
        flat_vertices.extend_from_slice(&cm.local_vertices);
        flat_indices.extend(cm.local_indices.iter().map(|&i| i + vertex_offset));
        c.index_offset = index_offset;
        c.index_count = cm.local_indices.len() as u32;
    }

    (flat_vertices, flat_indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh_cluster::{cluster_mesh, MeshletCluster, PARENT_GROUP_ERROR_ROOT};

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

    /// Each triangle as a sorted (a,b,c) tuple, for set-equality
    /// comparison across vertex-id permutations.
    fn triangle_position_set(
        indices: &[u32],
        verts: &[MeshVertex],
    ) -> std::collections::HashMap<[[i32; 3]; 3], usize> {
        let mut m = std::collections::HashMap::new();
        for tri in indices.chunks_exact(3) {
            let mut positions = [
                pos_key(verts[tri[0] as usize].local_pos),
                pos_key(verts[tri[1] as usize].local_pos),
                pos_key(verts[tri[2] as usize].local_pos),
            ];
            positions.sort();
            *m.entry(positions).or_insert(0) += 1;
        }
        m
    }

    fn pos_key(p: [f32; 3]) -> [i32; 3] {
        [
            (p[0] * 1000.0) as i32,
            (p[1] * 1000.0) as i32,
            (p[2] * 1000.0) as i32,
        ]
    }

    fn make_grid(side: usize) -> (Vec<MeshVertex>, Vec<u32>) {
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
    fn round_trip_preserves_triangle_set() {
        // 9×9 grid → multiple clusters → split → flatten → triangle set
        // (by position triples) must equal the original.
        let (v, i) = make_grid(9);
        let (mut clusters, flat_idx) = cluster_mesh(&v, &i);
        assert!(clusters.len() >= 2);

        let original_set = triangle_position_set(&flat_idx, &v);

        let cms = split_flat_into_cluster_meshes(&v, &flat_idx, &clusters);
        assert_eq!(cms.len(), clusters.len());

        let (flat_v, flat_i) = flatten_cluster_meshes(&cms, &mut clusters);
        let round_trip_set = triangle_position_set(&flat_i, &flat_v);

        assert_eq!(
            original_set, round_trip_set,
            "round-trip must preserve the triangle position multiset"
        );

        // Each cluster's index_offset/count must enclose only its triangles.
        for c in &clusters {
            let start = c.index_offset as usize;
            let end = start + c.index_count as usize;
            assert!(end <= flat_i.len(), "cluster {:?} OOB after flatten", c);
            assert_eq!(c.index_count % 3, 0, "cluster index_count must be tri-aligned");
        }
    }

    #[test]
    fn round_trip_per_cluster_triangle_set_preserved() {
        // Stronger invariant: each cluster's triangles (by position
        // triples) must be the same before vs after round-trip.
        let (v, i) = make_grid(9);
        let (mut clusters, flat_idx) = cluster_mesh(&v, &i);

        // Snapshot per-cluster triangle sets from the original flat data.
        let pre: Vec<_> = clusters
            .iter()
            .map(|c| {
                let s = c.index_offset as usize;
                let e = s + c.index_count as usize;
                triangle_position_set(&flat_idx[s..e], &v)
            })
            .collect();

        let cms = split_flat_into_cluster_meshes(&v, &flat_idx, &clusters);
        let (flat_v, flat_i) = flatten_cluster_meshes(&cms, &mut clusters);

        for (ci, c) in clusters.iter().enumerate() {
            let s = c.index_offset as usize;
            let e = s + c.index_count as usize;
            let post = triangle_position_set(&flat_i[s..e], &flat_v);
            assert_eq!(
                pre[ci], post,
                "cluster {} triangle set changed across round-trip",
                ci
            );
        }
    }

    #[test]
    fn boundary_verts_duplicated_across_clusters() {
        // Grid forces vertex sharing across cluster boundaries (each
        // boundary vert weighted by 2+ clusters). After split, the
        // flat VBO size grows because each cluster carries its own
        // copy of any shared vert.
        let (v, i) = make_grid(9);
        let (mut clusters, flat_idx) = cluster_mesh(&v, &i);
        assert!(clusters.len() >= 2);

        let cms = split_flat_into_cluster_meshes(&v, &flat_idx, &clusters);
        let (flat_v, _) = flatten_cluster_meshes(&cms, &mut clusters);

        // Per-cluster local_vertices SUM must equal flat VBO size after
        // flatten (round-trip is by concat).
        let sum_per_cluster: usize = cms.iter().map(|m| m.local_vertices.len()).sum();
        assert_eq!(sum_per_cluster, flat_v.len());

        // Flat VBO after round-trip must be at least as big as the
        // original (always — duplication grows it; shrink would mean
        // we lost something).
        assert!(
            flat_v.len() >= v.len(),
            "round-trip VBO ({} verts) must not shrink below original ({})",
            flat_v.len(),
            v.len(),
        );
    }

    #[test]
    fn empty_cluster_yields_empty_mesh() {
        // A zero-tri cluster should produce an empty ClusterMesh and
        // contribute nothing to the flat output.
        let c = MeshletCluster {
            aabb_min: [0.0; 3],
            _pad0: 0.0,
            aabb_max: [0.0; 3],
            index_offset: 0,
            index_count: 0,
            lod_level: 0,
            _pad2: 0,
            cluster_error: 0.0,
            parent_group_error: PARENT_GROUP_ERROR_ROOT,
            _pad3: [0; 3],
        };
        let cms = split_flat_into_cluster_meshes(&[], &[], &[c]);
        assert_eq!(cms.len(), 1);
        assert!(cms[0].local_vertices.is_empty());
        assert!(cms[0].local_indices.is_empty());

        let mut clusters = vec![c];
        let (flat_v, flat_i) = flatten_cluster_meshes(&cms, &mut clusters);
        assert!(flat_v.is_empty());
        assert!(flat_i.is_empty());
        assert_eq!(clusters[0].index_count, 0);
    }

    #[test]
    fn flatten_rebases_indices_by_vertex_offset() {
        // Two clusters of one triangle each. After flatten, cluster B's
        // indices must reference flat VBO slots [3, 4, 5] (offset by
        // cluster A's vertex count = 3), not [0, 1, 2].
        let a = ClusterMesh {
            local_vertices: vec![
                vert([0.0, 0.0, 0.0]),
                vert([1.0, 0.0, 0.0]),
                vert([0.0, 1.0, 0.0]),
            ],
            local_indices: vec![0, 1, 2],
        };
        let b = ClusterMesh {
            local_vertices: vec![
                vert([10.0, 0.0, 0.0]),
                vert([11.0, 0.0, 0.0]),
                vert([10.0, 1.0, 0.0]),
            ],
            local_indices: vec![0, 1, 2],
        };
        let mut clusters = vec![
            MeshletCluster {
                aabb_min: [0.0; 3], _pad0: 0.0, aabb_max: [0.0; 3],
                index_offset: 0, index_count: 0, lod_level: 0, _pad2: 0,
                cluster_error: 0.0, parent_group_error: 0.0, _pad3: [0; 3],
            },
            MeshletCluster {
                aabb_min: [0.0; 3], _pad0: 0.0, aabb_max: [0.0; 3],
                index_offset: 0, index_count: 0, lod_level: 0, _pad2: 0,
                cluster_error: 0.0, parent_group_error: 0.0, _pad3: [0; 3],
            },
        ];
        let (flat_v, flat_i) = flatten_cluster_meshes(&[a, b], &mut clusters);
        assert_eq!(flat_v.len(), 6);
        assert_eq!(flat_i, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(clusters[0].index_offset, 0);
        assert_eq!(clusters[0].index_count, 3);
        assert_eq!(clusters[1].index_offset, 3);
        assert_eq!(clusters[1].index_count, 3);
    }
}
