// Per-cluster LOD selection.
//
// One thread per (asset_instance, cluster). Reads the asset's
// `MeshletCluster` table; applies the Karis-Nanite admit rule
//
//     parent_group_error_proj >= pixel_threshold  AND
//     cluster_error_proj      <  pixel_threshold
//
// using the cluster's object-local error metrics projected to
// screen-pixels through the per-instance world matrix and the
// camera's perspective focal length. Admitted clusters atomicAdd
// into the per-draw `draw_count` and write their args at the
// compacted slot. Non-admitted clusters skip writing entirely —
// the GPU walks only the compacted prefix at draw time via
// `multi_draw_indexed_indirect_count`.
//
// Bind-group layout:
//   @group(0) @binding(0)  CameraUniforms (read)            — per-VR
//   @group(0) @binding(1)  MeshLodSelectParams (read)       — per-VR
//   @group(1) @binding(0)  MeshInstance (read)              — per-draw
//   @group(2) @binding(0)  array<MeshletCluster> (read)     — per-asset
//   @group(2) @binding(1)  array<DrawIndexedIndirectArgs>   — per-draw, RW
//   @group(2) @binding(2)  array<atomic<u32>, 16> stats     — pass-shared, RW
//   @group(2) @binding(3)  atomic<u32> draw_count           — per-draw, RW
//
// `stats` is a 16-slot histogram: indices 0..8 are the per-LOD
// admitted-cluster counts, indices 8..16 are the per-LOD total-cluster
// counts (everything we evaluated, regardless of admit). 8 covers
// `LOD_LEVELS_MAX`. Pass-shared means every draw within a pass
// (primary or shadow) atomic-adds into the same buffer; the CPU
// pre-clears it at the start of each pass. Diagnostic-only — values
// are read back to host and logged behind `ARVX_MESH_LOD_STATS=1`.
//
// `draw_count` is the per-draw indirect-draw count consumed by
// `multi_draw_indexed_indirect_count`. CPU pre-clears to 0 before
// the dispatch; admitted threads atomicAdd 1 to claim their args
// slot. Bound with INDIRECT usage so the same buffer the compute
// pass writes is read by the renderer's count parameter.

import package::lib::types::CameraUniforms;

struct MeshLodSelectParams {
    /// Pixel threshold for the Karis admit rule. Primary path uses
    /// 1.0 (one-pixel error budget); shadow path doubles this so the
    /// chain effectively picks `lod + 1`.
    pixel_threshold: f32,
    /// Number of clusters in the asset's cluster table for this
    /// dispatch. Threads beyond this are no-ops.
    cluster_count: u32,
    /// Debug bypass: when non-zero, every cluster is admitted
    /// regardless of the Karis rule. CPU sets this from
    /// `ARVX_MESH_DEBUG_FORCE_ADMIT`.
    force_admit: u32,
    /// When non-zero, atomic-add into the pass-shared `stats`
    /// histogram. CPU sets this from `ARVX_MESH_LOD_STATS=1`.
    record_stats: u32,
    /// Debug: when not 0xFFFFFFFF, admit ONLY clusters whose
    /// `lod_level` exactly equals this value. Bypasses Karis
    /// admit entirely. Used to A/B-test whether mixed-level
    /// admit is the cause of cross-cluster cracks.
    /// CPU sets from `ARVX_MESH_DEBUG_FORCE_LEVEL=N`.
    force_level: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct MeshInstance {
    world: mat4x4<f32>,
    object_id: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct MeshletCluster {
    aabb_min: vec3<f32>,
    aabb_max: vec3<f32>,
    index_offset: u32,
    index_count: u32,
    lod_level: u32,
    /// Per-cluster flags. Bit 0 = `LOD_DIRTY` (sculpt R4d) — set on
    /// dirty LOD-0 leaves AND on coarser-LOD ancestors whose chain
    /// touches a dirty leaf. The admit logic below force-admits dirty
    /// LOD-0 leaves (fresh geometry at any camera distance) and skips
    /// dirty LOD>0 ancestors entirely.
    flags: u32,
    cluster_error: f32,
    parent_group_error: f32,
    /// DAG topology pointers — CPU-only (sculpt CC walk). Mirrored here
    /// only to keep the WGSL std430 stride at 64 B; the shader doesn't
    /// read them.
    group_above_idx: u32,
    group_below_idx: u32,
    _pad3: u32,
}

const CLUSTER_FLAG_LOD_DIRTY: u32 = 1u;

struct DrawIndexedIndirectArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> params: MeshLodSelectParams;
@group(1) @binding(0) var<uniform> instance: MeshInstance;
@group(2) @binding(0) var<storage, read> clusters: array<MeshletCluster>;
@group(2) @binding(1) var<storage, read_write> args: array<DrawIndexedIndirectArgs>;
@group(2) @binding(2) var<storage, read_write> stats: array<atomic<u32>, 16>;
@group(2) @binding(3) var<storage, read_write> draw_count: atomic<u32>;

/// Approximate the maximum axis-scale of an affine 4x4 transform.
/// Cluster errors are object-local lengths; we multiply by this to
/// get world-space lengths for the perspective projection.
fn world_scale_factor(m: mat4x4<f32>) -> f32 {
    // Length of each basis row of the upper 3x3 — the per-axis scale
    // for the world transform applied to (1,0,0) / (0,1,0) / (0,0,1).
    let sx = length(m[0].xyz);
    let sy = length(m[1].xyz);
    let sz = length(m[2].xyz);
    return max(max(sx, sy), sz);
}

@compute @workgroup_size(64, 1, 1)
fn lod_select(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cidx = gid.x;
    if (cidx >= params.cluster_count) {
        return;
    }

    let c = clusters[cidx];

    // INSTANCE pivot in world (NOT this cluster's AABB center). All
    // clusters in the same instance see the same `view_distance` and
    // `world_to_pixels`, which is what makes the Karis admit rule
    // chain-consistent in floating-point.
    //
    // Why this matters: the rule assumes `parent_err_px` for a child
    // cluster *exactly equals* `cluster_err_px` for that child's
    // parent — they're both `parent_group_error * scale *
    // world_to_pixels`. If we compute `world_to_pixels` from each
    // cluster's own AABB center, the ratio drifts ~10-20% between
    // siblings and the parent (different AABB centers → different
    // distances). At the threshold boundary that's enough to leave a
    // chain with *zero* admitted clusters: the LOD-N child says
    // `parent_err_px < threshold` ("parent will admit") while the
    // LOD-N+1 parent says `cluster_err_px >= threshold` ("descend to
    // me"). Result: cluster-shaped holes that flicker and move with
    // tiny camera motion (`project_mesh_close_up_hole_bug.md`).
    //
    // Using the instance pivot loses per-cluster LOD precision — one
    // LOD level per instance, regardless of where the camera is
    // relative to individual clusters within the instance. For our
    // typical 0.5–5 m props the precision loss is well below the
    // pixel threshold; the proper fix is a Nanite-style monotonic
    // bounding-sphere DAG (each cluster's sphere encloses all its
    // children's), to be revisited if/when LOD oscillation across an
    // instance becomes visible.
    let world_pivot = (instance.world * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
    let scale = world_scale_factor(instance.world);

    // `view_proj[1][1]` ≈ cot(fov_y/2) for perspective and `2/height`
    // for orthographic projections (glam `Mat4::perspective_*` /
    // `Mat4::orthographic_*`). Multiplied by half viewport height it
    // gives pixels per world-unit at unit distance.
    let focal_pixels = abs(camera.view_proj[1][1]) * 0.5 * camera.resolution.y;

    // Ortho cameras (used by directional shadow maps) put `1` in
    // `view_proj[3][3]`; perspective puts `0`. For ortho the screen
    // projection is depth-independent so view_distance drops out
    // (we substitute `1.0`); for perspective we use the standard
    // `world_to_pixels = focal_pixels / view_distance` formula.
    let is_ortho = abs(camera.view_proj[3][3] - 1.0) < 0.01;
    let view_distance = select(
        max(length(world_pivot - camera.position.xyz), 0.001),
        1.0,
        is_ortho,
    );
    let world_to_pixels = focal_pixels / view_distance;

    let cluster_err_px = c.cluster_error * scale * world_to_pixels;
    let parent_err_px = c.parent_group_error * scale * world_to_pixels;

    // Karis '21 admit rule, written NaN-safely:
    //
    //   `parent_group_error = +∞` is the DAG-root sentinel
    //   (no level above this cluster). Computing
    //   `parent_err_px = ∞ * scale * world_to_pixels` produces NaN
    //   on any frame where `scale` or `world_to_pixels` happens to
    //   be 0 (degenerate transform / pre-camera-write first frame),
    //   and `NaN >= threshold` is `false`. The sentinel check below
    //   bypasses the projection math entirely so DAG roots are
    //   always admitted on the parent side.
    //
    //   `cluster_error = 0` is the LOD-0-leaf sentinel; the
    //   cluster's own error is exactly zero so it always fits any
    //   pixel budget, no projection needed.
    let parent_is_root = c.parent_group_error >= 1.0e20;
    let cluster_is_leaf = c.cluster_error == 0.0;
    let admit_parent = parent_is_root || parent_err_px >= params.pixel_threshold;
    let admit_self = cluster_is_leaf || cluster_err_px < params.pixel_threshold;
    let karis_admit = admit_parent && admit_self;
    let level_admit = (params.force_level != 0xFFFFFFFFu) && (c.lod_level == params.force_level);

    // Phase B R4d (V1) — when sculpt touches an asset, the CPU marks
    // EVERY cluster of that asset with `CLUSTER_FLAG_LOD_DIRTY` (not
    // just the brush-overlapping subset). This shader then admits
    // ONLY the LOD-0 leaves of the affected asset and drops every
    // LOD>0 cluster, giving an asset-wide LOD-0 clamp.
    //
    // Why asset-wide and not per-chain: the cluster DAG topology
    // (parent ↔ child) is implicit in the cluster_error /
    // parent_group_error chain, with no explicit cross-references in
    // the cluster struct. Marking only brush-overlapping ancestors
    // dirty broke chains: a non-brush LOD-0 leaf in a chain whose
    // LOD>0 ancestor got dropped couldn't Karis-admit at the user's
    // camera distance, leaving cluster-shaped holes. Asset-wide
    // marking ensures every chain admits at LOD-0 — fresh for the
    // re-extracted dirty leaves, pre-sculpt-correct for the
    // non-dirty leaves. R5's async ancestor rebuild can refine to
    // per-chain marking once DAG topology is exposed.
    let lod_dirty = (c.flags & CLUSTER_FLAG_LOD_DIRTY) != 0u;
    let force_dirty_leaf = lod_dirty && cluster_is_leaf;
    let drop_dirty_ancestor = lod_dirty && !cluster_is_leaf;

    let admit = !drop_dirty_ancestor
        && (level_admit
            || (params.force_admit != 0u)
            || force_dirty_leaf
            || (params.force_level == 0xFFFFFFFFu && karis_admit));

    if (admit) {
        // Claim a compacted slot. All admitted threads atomicAdd 1
        // into the per-draw counter; the returned index is the slot
        // they own in the args buffer. Non-admitted threads skip
        // entirely — the draw count tells the renderer's
        // `multi_draw_indexed_indirect_count` exactly how many
        // entries to walk, so the args[cidx] zero-fill of the old
        // path is no longer needed.
        let slot = atomicAdd(&draw_count, 1u);
        var out: DrawIndexedIndirectArgs;
        out.index_count = c.index_count;
        out.instance_count = 1u;
        out.first_index = c.index_offset;
        out.base_vertex = 0;
        // `first_instance` must be 0 unless `INDIRECT_FIRST_INSTANCE`
        // is enabled (it isn't in our context.rs). The current
        // mesh.wesl reads instance state from a uniform binding,
        // not via `@builtin(instance_index)`, so 0 is the right
        // value either way.
        out.first_instance = 0u;
        args[slot] = out;
    }

    // Diagnostic histogram — gated on `record_stats` because atomic
    // contention on 16 hot cache lines from millions of threads adds
    // measurable GPU time, and we don't want to pay it on the
    // production path. Slots 0..8: admitted per LOD; slots 8..16:
    // total per LOD. CPU pre-clears the buffer to 0 at the start of
    // each pass and reads it back behind `ARVX_MESH_LOD_STATS=1`.
    // `lod_level` is clamped so a corrupt cluster entry can't index
    // out-of-bounds (matches `LOD_LEVELS_MAX = 8`).
    if (params.record_stats != 0u) {
        let lvl = min(c.lod_level, 7u);
        if (admit) {
            atomicAdd(&stats[lvl], 1u);
        }
        atomicAdd(&stats[lvl + 8u], 1u);
    }
}
