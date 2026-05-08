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

use meshopt::{
    partition_clusters_with_positions, simplify_with_locks, SimplifyOptions, VertexDataAdapter,
};
use rayon::prelude::*;

use crate::mesh_cluster::{cluster_mesh, MeshletCluster, PARENT_GROUP_ERROR_ROOT};
use crate::mesh_extract::MeshVertex;

/// Number of LOD levels the DAG attempts to build (LOD 0 is the
/// finest; LOD `LOD_LEVELS - 1` is the coarsest the DAG converges
/// to). Construction may stop early if a level's simplification
/// makes no progress or only one cluster remains.
///
/// `4` since the per-group simplify perf fix landed (compact
/// local VBO + lock array, rayon across groups). Real-asset load
/// numbers post-fix: elephant 51K LOD-0 clusters ⇒ ~8s DAG build,
/// box_8 46K ⇒ ~7s, bunny 26K ⇒ ~2.6s; everything below 15K
/// clusters builds in well under a second. Multi-asset scenes
/// can still spend tens of seconds in DAG build at editor open
/// — bake-at-import (`project_dag_bake_at_import.md`) is the
/// proper fix once Phase 6.5 freezes the DAG params. Set
/// `RKP_MESH_LOD_LEVELS=1..=LOD_LEVELS_MAX` at bake time to
/// override. (At runtime the v5 load path uses whatever was
/// baked into the asset; the env var only affects
/// `build_cluster_dag` in `--rebuild-mesh` and the v4 legacy
/// load fallback.)
pub const LOD_LEVELS: usize = 4;

/// Hard cap on `lod_levels` for the runtime
/// `RKP_MESH_LOD_LEVELS` override. The DAG converges naturally
/// when a level produces ≤ 1 cluster, so going higher than the
/// scene actually needs is a no-op. 8 covers the splat5
/// elephant scene's worst-case (51K LOD-0 clusters → ~200
/// clusters at LOD-7 with 50% reduction per level).
pub const LOD_LEVELS_MAX: usize = 8;

/// Read the runtime-configurable LOD-level count. Reads
/// `RKP_MESH_LOD_LEVELS` once and clamps to `1..=LOD_LEVELS_MAX`;
/// falls back to the [`LOD_LEVELS`] compile-time default.
fn lod_levels_runtime() -> usize {
    std::env::var("RKP_MESH_LOD_LEVELS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .map(|n| n.clamp(1, LOD_LEVELS_MAX))
        .unwrap_or(LOD_LEVELS)
}

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

/// Build the cluster DAG using the runtime-configured LOD level
/// count. Reads `RKP_MESH_LOD_LEVELS` (clamped 1..=LOD_LEVELS_MAX)
/// once per call and forwards to [`build_cluster_dag_with_levels`].
pub fn build_cluster_dag(vertices: &[MeshVertex], indices: &[u32]) -> ClusterDag {
    build_cluster_dag_with_levels(vertices, indices, lod_levels_runtime())
}

/// Build the cluster DAG to exactly `lod_levels` levels. `lod_levels
/// = 1` skips the simplify-and-regroup loop entirely (returns the
/// LOD-0 clustering, no parent links — every cluster is a DAG leaf
/// with `parent_group_error = ∞`, admitted by the LOD-select rule
/// as "can't go coarser"). `lod_levels >= 2` runs the full DAG
/// build. Empty input → [`ClusterDag::empty`].
pub fn build_cluster_dag_with_levels(
    vertices: &[MeshVertex],
    indices: &[u32],
    lod_levels: usize,
) -> ClusterDag {
    if vertices.is_empty() || indices.len() < 3 {
        return ClusterDag::empty();
    }
    let lod_levels = lod_levels.clamp(1, LOD_LEVELS_MAX);

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
    eprintln!(
        "[lod] LOD 0: {} clusters (target lod_levels={})",
        prev_level_range.len(),
        lod_levels,
    );

    for lod in 1..lod_levels {
        if prev_level_range.len() <= 1 {
            break; // DAG converged to a single cluster
        }

        let lod_t0 = std::time::Instant::now();
        let prev_indices: Vec<usize> = prev_level_range.clone().collect();
        let groups = group_clusters_meshopt(
            &all_clusters,
            &all_indices,
            vertices,
            &prev_indices,
            GROUP_SIZE_TARGET,
        );
        if groups.is_empty() {
            break;
        }

        // Per-vertex group ownership for boundary-lock detection.
        // Map: vertex_id → set of group_ids that reference it from
        // any of their member prev-level clusters.
        let vert_to_groups = build_vert_to_groups(&all_clusters, &all_indices, &prev_indices, &groups);

        let new_level_start = all_clusters.len();

        // Per-group simplify is read-only over `all_clusters` and
        // `all_indices` (it only consults the prev-level clusters'
        // index ranges + cluster_error to compute the new
        // cluster_error). The mutations — appending sub-clusters /
        // sub-indices and backfilling parent_group_error on
        // consumed prev-level clusters — happen sequentially after
        // the parallel collect so cluster `index_offset` values
        // stay deterministic and bit-equivalent to the sequential
        // build.
        let results: Vec<GroupResult> = groups
            .par_iter()
            .enumerate()
            .map(|(g_idx, group_local)| {
                simplify_one_group(
                    g_idx,
                    group_local,
                    &prev_indices,
                    &all_clusters,
                    &all_indices,
                    vertices,
                    &vert_to_groups,
                    lod as u32,
                )
            })
            .collect();

        // Stats for the diagnostic log: how many groups produced
        // simplified output, how many bailed (degenerate input or
        // simplify_with_locks unable to reduce), and the cluster_error
        // distribution at the new level. If most groups bail, the
        // simplifier is being blocked by group-boundary locks (the
        // new sub-clusters' input tris are mostly locked verts) —
        // which means the DAG mostly stalls and the LOD-select admit
        // rule keeps prev-level (or LOD-0) clusters as DAG leaves.
        // High-impact stat for "is LOD doing real work?".
        let groups_count = groups.len();
        let skipped_groups = results.iter().filter(|r| r.skipped).count();
        let mut consumed_prev: usize = 0;
        let mut new_cluster_errors: Vec<f32> = Vec::new();

        for result in results {
            if result.skipped {
                continue;
            }
            consumed_prev += result.consumed_global_ids.len();
            new_cluster_errors.extend(result.sub_clusters.iter().map(|c| c.cluster_error));

            let sub_index_base = all_indices.len() as u32;
            all_indices.extend(result.sub_indices);
            for mut sc in result.sub_clusters {
                sc.index_offset += sub_index_base;
                all_clusters.push(sc);
            }
            // Backfill parent_group_error on prev-level clusters
            // that this group consumed — they're no longer DAG
            // leaves, so the LOD-selection rule needs to know what
            // error the next coarser level introduced.
            for gi in result.consumed_global_ids {
                all_clusters[gi].parent_group_error = result.group_error;
            }
        }

        let new_level_end = all_clusters.len();
        if new_level_end == new_level_start {
            // No group produced new clusters this level; DAG growth
            // has stalled (every group either failed to simplify or
            // had < 3 input tris). Stop.
            break;
        }
        let prev_count = prev_indices.len();
        let consumed_pct = if prev_count == 0 {
            0.0
        } else {
            100.0 * consumed_prev as f32 / prev_count as f32
        };
        let (err_p50, err_p95) = percentiles_p50_p95(&mut new_cluster_errors);
        eprintln!(
            "[lod] LOD {}: {} clusters from {} groups ({} ok, {} skip); \
             consumed {}/{} prev-level ({:.1}%); cluster_error p50={:.4} p95={:.4} ({:.2}s)",
            lod,
            new_level_end - new_level_start,
            groups_count,
            groups_count - skipped_groups,
            skipped_groups,
            consumed_prev,
            prev_count,
            consumed_pct,
            err_p50,
            err_p95,
            lod_t0.elapsed().as_secs_f32(),
        );
        prev_level_range = new_level_start..new_level_end;
    }

    ClusterDag {
        clusters: all_clusters,
        indices: all_indices,
        lod0_cluster_range,
        lod0_index_range,
    }
}

/// Group prev-level clusters via `meshopt::partition_clusters_with_positions`.
/// meshopt prioritises grouping clusters that share vertices (so the
/// group's shared-edge boundary stays small, which is what gives the
/// simplifier real reduction headroom under our group-boundary lock
/// regime); spatial proximity from the position adapter is the
/// tie-breaker. Linear-ish time in the cluster count + total
/// vertex-index references — orders of magnitude faster than the
/// O(N²) seed-and-nearest-neighbour pass it replaces (which would
/// take many minutes on the 100K+-cluster meshes real assets produce).
///
/// Returns groups as vectors of positions within `prev_indices`
/// (not global cluster ids); empty groups are filtered out.
fn group_clusters_meshopt(
    clusters: &[MeshletCluster],
    indices: &[u32],
    vertices: &[MeshVertex],
    prev_indices: &[usize],
    target_size: usize,
) -> Vec<Vec<usize>> {
    if prev_indices.is_empty() {
        return Vec::new();
    }
    let n = prev_indices.len();
    if n == 1 {
        return vec![vec![0]];
    }

    // Build the per-cluster unique-vertex list meshopt expects:
    // sequential concatenation of each cluster's vertex IDs +
    // a parallel `cluster_index_counts` giving each cluster's
    // unique-vertex count.
    let mut cluster_indices_flat: Vec<u32> = Vec::new();
    let mut cluster_index_counts: Vec<u32> = Vec::with_capacity(n);
    let mut seen: HashSet<u32> = HashSet::new();
    for &gi in prev_indices {
        let c = &clusters[gi];
        let span = &indices[c.index_offset as usize..(c.index_offset + c.index_count) as usize];
        seen.clear();
        let start = cluster_indices_flat.len();
        for &v in span {
            if seen.insert(v) {
                cluster_indices_flat.push(v);
            }
        }
        cluster_index_counts.push((cluster_indices_flat.len() - start) as u32);
    }

    let vertex_bytes = bytemuck::cast_slice::<MeshVertex, u8>(vertices);
    let stride = std::mem::size_of::<MeshVertex>();
    let adapter = VertexDataAdapter::new(vertex_bytes, stride, 0)
        .expect("MeshVertex layout matches VertexDataAdapter");

    let mut destination: Vec<u32> = vec![0; n];
    let partition_count = partition_clusters_with_positions(
        &mut destination,
        &cluster_indices_flat,
        &cluster_index_counts,
        &adapter,
        target_size,
    );

    // Invert: partition id → list of cluster positions in prev_indices.
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); partition_count];
    for (cluster_idx, &part_id) in destination.iter().enumerate() {
        groups[part_id as usize].push(cluster_idx);
    }
    groups.retain(|g| !g.is_empty());
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

/// Output of one group's parallel simplify worker. Index offsets
/// in `sub_clusters` are relative to the start of `sub_indices`
/// — the sequential flatten step rebases them into the level's
/// shared `all_indices`. `lod_level`, `cluster_error`, and
/// `parent_group_error` are already final.
struct GroupResult {
    sub_clusters: Vec<MeshletCluster>,
    sub_indices: Vec<u32>,
    /// Parametric error from `simplify_with_locks`. Backfills the
    /// `parent_group_error` of every prev-level cluster in
    /// `consumed_global_ids`. Only meaningful when `!skipped`.
    group_error: f32,
    /// Prev-level global cluster IDs consumed by this group.
    consumed_global_ids: Vec<usize>,
    /// True when the group produced no new clusters — simplifier
    /// failed to reduce or `cluster_mesh` returned empty. The
    /// flatten step skips these without touching `all_clusters` /
    /// `all_indices`; the consumed clusters remain DAG leaves.
    skipped: bool,
}

impl GroupResult {
    fn skipped() -> Self {
        Self {
            sub_clusters: Vec::new(),
            sub_indices: Vec::new(),
            group_error: 0.0,
            consumed_global_ids: Vec::new(),
            skipped: true,
        }
    }
}

/// Per-group simplify worker. Pure over its inputs (only reads
/// `all_clusters` / `all_indices`), so safe to run in parallel
/// across `groups` via rayon. Returns a [`GroupResult`] the
/// caller flattens sequentially into the level's shared cluster
/// + index buffers.
fn simplify_one_group(
    g_idx: usize,
    group_local: &[usize],
    prev_indices: &[usize],
    all_clusters: &[MeshletCluster],
    all_indices: &[u32],
    vertices: &[MeshVertex],
    vert_to_groups: &HashMap<u32, HashSet<u32>>,
    lod: u32,
) -> GroupResult {
    // Translate group's prev-level positions back to global cluster indices.
    let group_global: Vec<usize> = group_local.iter().map(|&li| prev_indices[li]).collect();

    // Merge member clusters' triangles into a single index buffer.
    let mut merged_tris: Vec<u32> = Vec::new();
    for &gi in &group_global {
        let c = &all_clusters[gi];
        merged_tris.extend_from_slice(
            &all_indices[c.index_offset as usize..(c.index_offset + c.index_count) as usize],
        );
    }
    if merged_tris.len() < 3 {
        return GroupResult::skipped();
    }

    // Compact the group's geometry into a local VBO + lock array
    // so meshopt's per-vertex sweep is bounded by the group's
    // unique-vertex count (~256 typical) rather than the asset's
    // vertex count (~1-3M on real meshes). See
    // `build_local_simplify_inputs` for the rationale.
    let (local_verts, local_tris, local_locks, local_to_global) =
        build_local_simplify_inputs(&merged_tris, vertices, vert_to_groups, g_idx);

    // meshopt::simplify_with_locks. Target ~50% triangles;
    // unbounded error budget (the simplifier returns the actual
    // error in `result_error`, which we capture as the group's
    // parametric error metric).
    let target_index_count =
        ((local_tris.len() as f32 * LOD_REDUCTION_TARGET) as usize / 3) * 3;
    let mut group_error = 0.0_f32;
    let simplified_local = simplify_meshopt(
        &local_tris,
        &local_verts,
        &local_locks,
        target_index_count,
        &mut group_error,
    );

    if simplified_local.len() < 3 || simplified_local.len() >= local_tris.len() {
        // No reduction (simplifier was blocked by locks or
        // topology). The group's prev-level clusters retain
        // `parent_group_error = ∞` and become DAG leaves — the LOD
        // selection rule will always render them.
        return GroupResult::skipped();
    }

    // Map the simplifier's local-id output back to global VBO ids
    // so the re-cluster step + DAG accumulator stay in
    // global-VBO space.
    let simplified: Vec<u32> = simplified_local
        .iter()
        .map(|&li| local_to_global[li as usize])
        .collect();

    let (sub_clusters_raw, sub_indices) = cluster_mesh(vertices, &simplified);
    if sub_clusters_raw.is_empty() {
        return GroupResult::skipped();
    }

    // cluster_error: max along chain from leaves to here.
    let max_input_error = group_global
        .iter()
        .map(|&gi| all_clusters[gi].cluster_error)
        .fold(0.0_f32, f32::max);
    let new_cluster_error = max_input_error.max(group_error);

    let sub_clusters: Vec<MeshletCluster> = sub_clusters_raw
        .into_iter()
        .map(|mut sc| {
            // index_offset stays at the meshlet builder's local
            // offset into sub_indices; the flatten step rebases.
            sc.lod_level = lod;
            sc.cluster_error = new_cluster_error;
            sc.parent_group_error = PARENT_GROUP_ERROR_ROOT;
            sc
        })
        .collect();

    GroupResult {
        sub_clusters,
        sub_indices,
        group_error,
        consumed_global_ids: group_global,
        skipped: false,
    }
}

/// Build a compact local VBO + lock array for one group's
/// `simplify_with_locks` call. Doing this avoids meshopt walking
/// the asset's full per-vertex lock array on every group call —
/// real meshes have 1-3M vertices and ~thousands of groups per
/// LOD level, and meshopt's FFI per-vertex sweep dominates wall
/// clock when the lock array is full-sized (memory note
/// `project_mesh_phase5_shipped`). With typical groups of ≤ 256
/// unique verts the inner cost drops by ~4 orders of magnitude.
///
/// Returns `(local_verts, local_tris, local_locks, local_to_global)`:
///
/// * `local_verts` — `MeshVertex` for each unique global vertex
///   referenced by `merged_tris`, in first-encounter order.
/// * `local_tris` — `merged_tris` remapped through the local
///   numbering. Indices into `local_verts`.
/// * `local_locks` — same length as `local_verts`. `true` iff
///   the corresponding global vertex appears in any group other
///   than `this_group` (exterior boundary; must be preserved so
///   adjacent groups' boundary geometry continues to match — the
///   crack-avoidance invariant from Karis '21 §3.2).
/// * `local_to_global` — local id → global VBO id. Caller
///   remaps the simplifier's output back through this before
///   feeding it to `cluster_mesh` / the DAG accumulator.
fn build_local_simplify_inputs(
    merged_tris: &[u32],
    vertices: &[MeshVertex],
    vert_to_groups: &HashMap<u32, HashSet<u32>>,
    this_group: usize,
) -> (Vec<MeshVertex>, Vec<u32>, Vec<bool>, Vec<u32>) {
    let mut global_to_local: HashMap<u32, u32> = HashMap::new();
    let mut local_to_global: Vec<u32> = Vec::new();
    let mut local_verts: Vec<MeshVertex> = Vec::new();
    let mut local_locks: Vec<bool> = Vec::new();
    let mut local_tris: Vec<u32> = Vec::with_capacity(merged_tris.len());

    for &g in merged_tris {
        let local_id = match global_to_local.get(&g) {
            Some(&id) => id,
            None => {
                let id = local_to_global.len() as u32;
                global_to_local.insert(g, id);
                local_to_global.push(g);
                local_verts.push(vertices[g as usize]);
                let locked = vert_to_groups
                    .get(&g)
                    .is_some_and(|groups| groups.iter().any(|&og| og != this_group as u32));
                local_locks.push(locked);
                id
            }
        };
        local_tris.push(local_id);
    }

    (local_verts, local_tris, local_locks, local_to_global)
}

/// p50 + p95 of a `Vec<f32>` (used for the per-LOD `cluster_error`
/// distribution in the diagnostic log). Sorts the vec in place;
/// returns `(0.0, 0.0)` for empty input. Total-order sort via
/// `f32::total_cmp` so NaNs sort consistently rather than
/// pessimising the partial-cmp into a panic.
fn percentiles_p50_p95(v: &mut Vec<f32>) -> (f32, f32) {
    if v.is_empty() {
        return (0.0, 0.0);
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    let p50 = v[n / 2];
    let p95 = v[(n * 95 / 100).min(n - 1)];
    (p50, p95)
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
    use crate::mesh_cluster::MAX_VERTS_PER_CLUSTER;

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
        // Explicit lod_levels=4 — the shipped default `LOD_LEVELS=1`
        // is a temporary perf-bypass while the per-group simplify is
        // optimised; the DAG-build correctness this test guards
        // against still has to work end-to-end.
        let dag = build_cluster_dag_with_levels(&v, &i, 4);
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
        let dag = build_cluster_dag_with_levels(&v, &i, 4);
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
        let dag = build_cluster_dag_with_levels(&v, &i, 4);
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
        let dag = build_cluster_dag_with_levels(&v, &i, 4);
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
    fn dag_groups_partition_all_prev_clusters() {
        // The DAG builder's grouper (now meshopt-backed) must
        // assign every prev-level cluster to exactly one group at
        // the next level. Verified end-to-end on the grid mesh:
        // every LOD-0 cluster either has a finite
        // `parent_group_error` (consumed by a group) OR `∞` (its
        // own group failed to simplify and it became a DAG leaf).
        let (v, i) = grid_mesh(17);
        let dag = build_cluster_dag(&v, &i);
        for c in &dag.clusters[..dag.lod0_cluster_range.1 as usize] {
            assert!(
                c.parent_group_error.is_finite() || c.parent_group_error.is_infinite(),
                "every LOD-0 cluster must have a defined parent_group_error",
            );
        }
        // For the 17×17 grid, the DAG should grow past LOD 0 →
        // at least some LOD-0 clusters get consumed (covered by
        // `dag_consumed_lod0_clusters_have_finite_parent_group_error`).
    }

    #[test]
    fn dag_build_is_deterministic_across_invocations() {
        // The per-group simplify runs in parallel via rayon, then
        // flattens sequentially. This test guards against any
        // future shared-mutable state that would let thread
        // scheduling affect cluster ordering or index_offsets.
        let (v, i) = grid_mesh(17);
        let a = build_cluster_dag_with_levels(&v, &i, 4);
        let b = build_cluster_dag_with_levels(&v, &i, 4);
        assert_eq!(a.clusters, b.clusters);
        assert_eq!(a.indices, b.indices);
        assert_eq!(a.lod0_cluster_range, b.lod0_cluster_range);
        assert_eq!(a.lod0_index_range, b.lod0_index_range);
    }

    #[test]
    fn build_local_simplify_inputs_compacts_and_locks_correctly() {
        // 6 global verts; group 0 owns {0,1,2,3}, group 1 owns
        // {2,3,4,5}. Verts 2 and 3 are exterior-boundary for either
        // group → locked when simplifying that group.
        let mut vert_to_groups: HashMap<u32, HashSet<u32>> = HashMap::new();
        for v in 0..=3 {
            vert_to_groups.entry(v).or_default().insert(0);
        }
        for v in 2..=5 {
            vert_to_groups.entry(v).or_default().insert(1);
        }

        // Distinct positions per global vert so the round-trip
        // assertion below has bite.
        let verts: Vec<MeshVertex> = (0..6)
            .map(|i| vert([i as f32, i as f32 * 0.5, i as f32 * 0.25]))
            .collect();

        // Group 0's merged_tris in global IDs (two triangles
        // sharing the boundary).
        let merged = vec![0u32, 1, 2, 1, 2, 3];

        let (local_verts, local_tris, local_locks, local_to_global) =
            build_local_simplify_inputs(&merged, &verts, &vert_to_groups, 0);

        // 4 unique global verts referenced → 4 local entries.
        assert_eq!(local_verts.len(), 4);
        assert_eq!(local_to_global.len(), 4);
        assert_eq!(local_locks.len(), 4);
        assert_eq!(local_tris.len(), merged.len());

        // local_to_global is a permutation of the unique input verts.
        let global_set: HashSet<u32> = local_to_global.iter().copied().collect();
        let expected: HashSet<u32> = [0u32, 1, 2, 3].into_iter().collect();
        assert_eq!(global_set, expected);

        // Locks: only verts 2 and 3 (shared with group 1).
        for (li, &gi) in local_to_global.iter().enumerate() {
            let want = matches!(gi, 2 | 3);
            assert_eq!(local_locks[li], want, "vert g={} li={}", gi, li);
        }

        // Triangle round-trip: local_to_global[local_tris[i]] == merged[i].
        let round: Vec<u32> = local_tris
            .iter()
            .map(|&li| local_to_global[li as usize])
            .collect();
        assert_eq!(round, merged);

        // Local VBO carries the original positions for each remapped vert.
        for (li, &gi) in local_to_global.iter().enumerate() {
            assert_eq!(local_verts[li].local_pos, verts[gi as usize].local_pos);
        }
    }

    #[test]
    fn build_local_simplify_inputs_no_locks_when_group_owns_all_verts() {
        // Single-group scenario: every vert in vert_to_groups
        // belongs only to this_group → no locks.
        let mut vert_to_groups: HashMap<u32, HashSet<u32>> = HashMap::new();
        for v in 0..=2 {
            vert_to_groups.entry(v).or_default().insert(0);
        }
        let verts: Vec<MeshVertex> = (0..3).map(|i| vert([i as f32, 0.0, 0.0])).collect();

        let (_lv, _lt, locks, _l2g) =
            build_local_simplify_inputs(&[0u32, 1, 2], &verts, &vert_to_groups, 0);
        assert_eq!(locks, vec![false, false, false]);
    }
}
