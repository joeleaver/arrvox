//! Octree-based surface voxelization from a signed distance field.
//!
//! Level-by-level BFS: at each octree depth we classify every active
//! node's region against the SDF (9 samples — 8 corners + center),
//! then march the Mixed nodes' children into the next level. Terminal
//! levels (brick_depth if bricks are enabled, else max_depth) emit
//! geometry — per-cell center + 6 gradient taps for normals.
//!
//! All SDF queries at a given level batch into one call to the
//! caller's `sdf_fn`, so a GPU-backed callback can dispatch the whole
//! level in a single compute pass. The algorithm was recursive
//! previously; the BFS restructure was motivated by eliminating the
//! dual CPU/GPU procedural math implementation (see the Phase 3
//! rework: the `sdf_fn` is now the GPU evaluator for procedurals, and
//! per-point dispatches would have destroyed bake throughput).
//!
//! * **EMPTY** if every sample's distance is greater than `extent / 2` —
//!   the surface can't be inside this node (SDFs are 1-Lipschitz).
//! * **INTERIOR** if every sample's distance is less than `-extent / 2` —
//!   the node is strictly inside the solid.
//! * **MIXED** otherwise. Descend, or at the terminal level emit bricks
//!   / leaves.
//!
//! Produces a [`SparseOctree`] with variable-depth leaves. Each leaf
//! carries a single [`LeafAttr`] directly: a prefiltered surface
//! normal (SDF gradient) and a material reference. No per-voxel
//! opacity, no voxel_pool indirection.
//!
//! The SDF convention matches rkf-core: negative = inside the surface,
//! positive = outside, zero = on the surface.

use glam::{UVec3, Vec3};
use crate::Aabb;

use crate::brick_pool::{BrickPool, BRICK_CELLS, BRICK_LEVELS};
use crate::leaf_attr::LeafAttr;
use crate::leaf_attr_pool::LeafAttrPool;
use crate::sparse_octree::SparseOctree;

mod bfs;
mod emit;

#[cfg(test)]
mod tests;

/// Result of voxelizing an SDF.
pub struct VoxelizeOctreeResult {
    pub octree: SparseOctree,
    /// Number of leaves in the tree (before collapse/dedup). For brick-based
    /// trees this counts every populated cell across all bricks.
    pub voxel_count: u32,
    /// Number of leaf_attr pool slots allocated. Equals the count of
    /// unique (material, normal) tuples across the whole voxelization.
    pub leaf_attr_unique_count: u32,
    /// First leaf_attr_pool slot used. Together with
    /// `leaf_attr_unique_count` this is the contiguous range to free.
    pub leaf_attr_slot_start: u32,
    /// Every brick id allocated during this voxelization. `BrickPool::allocate`
    /// may return ids reclaimed from the free list (e.g. from a prior asset
    /// release), so the set isn't a contiguous range — track each id
    /// explicitly so `deallocate_geometry` can free them later.
    pub brick_ids: Vec<u32>,
    /// 6 face-adjacent brick ids per brick (or FACE_EMPTY / FACE_INTERIOR
    /// sentinel) indexed by `brick_id`. See `brick_face_links.rs`.
    /// Length is `max_brick_id + 1`. Empty if the tree has no bricks.
    pub brick_face_links: Vec<[u32; 6]>,
    pub grid_origin: Vec3,
}

/// Voxelize a signed distance function into a sparse octree.
///
/// `sdf_fn` takes a batch of world-space positions and returns
/// `(signed_distance, primary_material, secondary_material,
/// blend_weight_u4)` per position. Negative distance = inside the
/// surface. The 1-Lipschitz property of an SDF is what makes the
/// coarse-level Empty/Interior classifier provably correct — the
/// input should be a true signed distance, not an arbitrary scalar
/// field that's merely sign-correct.
///
/// The batched signature lets GPU-backed evaluators dispatch one
/// compute pass per octree level (O(depth) dispatches total). A CPU
/// caller can implement the callback by looping and calling a per-
/// point function — the overhead from the extra Vec allocation is
/// negligible next to the SDF evaluation itself.
///
/// `blend_weight_u4` is in `[0, 15]` (the 4-bit range `LeafAttr` can
/// store). Pass `0` for single-material voxelization — the secondary
/// material field is then ignored by the shader's dual-material lerp
/// (which is guarded behind `blend_weight > 0`). Callers coming from
/// a float blend (`0.0..1.0`) should quantize via
/// `(b * 15.0).round().clamp(0.0, 15.0) as u8`.
///
/// `aabb`: world-space bounding box of the object.
/// `base_voxel_size`: voxel size at the finest level.
pub fn voxelize_octree<F>(
    mut sdf_fn: F,
    aabb: &Aabb,
    base_voxel_size: f32,
    leaf_attr_pool: &mut LeafAttrPool,
    brick_pool: &mut BrickPool,
) -> Option<VoxelizeOctreeResult>
where
    F: FnMut(&[Vec3]) -> Vec<(f32, u16, u16, u8, u32)>,
{
    let aabb_size = aabb.max - aabb.min;
    let max_dim = aabb_size.x.max(aabb_size.y).max(aabb_size.z);

    // Depth is the smallest power of 2 that covers the AABB in voxels.
    let voxels_needed = (max_dim / base_voxel_size).ceil().max(1.0) as u32;
    let depth = if voxels_needed <= 1 {
        1
    } else {
        (32 - (voxels_needed - 1).leading_zeros()) as u8
    };

    let t_start = std::time::Instant::now();
    let mut octree = SparseOctree::new(depth, base_voxel_size);
    let mut voxel_count = 0u32;
    // No cell-attr dedup. Bake-time `(LeafAttr, color)` dedup used to
    // collapse identical cells (a flat 20m face baked to ~100 unique
    // attrs over millions of cells) into shared slots, which broke any
    // editor operation that needs per-cell identity — paint cursor
    // floods the whole face, paint writes recolor every cell sharing
    // the slot, blending across stamps had no per-cell prior to blend
    // with. Allocating one slot per cell makes paint and cursor "just
    // work" by reading/writing the per-cell slot directly. Cost: a
    // 2.3M-cell flat box grows from ~1KB to ~27MB resident; mesh
    // imports barely change because their normals already differ
    // per cell, breaking dedup anyway. LZ4 compresses the repeated
    // attrs effectively on disk.
    let leaf_attr_slot_start = leaf_attr_pool.allocated_count();
    let mut brick_ids: Vec<u32> = Vec::new();

    // Center the octree on the AABB.
    let extent = octree.extent_world();
    let aabb_center = (aabb.min + aabb.max) * 0.5;
    let grid_origin = aabb_center - Vec3::splat(extent * 0.5);

    // Bricks terminate the octree `BRICK_LEVELS` levels above
    // max_depth. For a tree at or below that depth, the entire octree
    // would degenerate to a single brick — disable bricking and fall
    // back to per-leaf.
    let brick_depth: Option<u8> = if depth > BRICK_LEVELS {
        Some(depth - BRICK_LEVELS)
    } else {
        None
    };

    let mut bake_stats = BakeStats::default();
    bfs::subdivide_bfs(
        &mut sdf_fn,
        &mut octree,
        leaf_attr_pool,
        brick_pool,
        &mut voxel_count,
        &mut brick_ids,
        grid_origin,
        depth,
        brick_depth,
        base_voxel_size,
        &mut bake_stats,
    )?;
    let t_after_subdivide = t_start.elapsed();

    // Post-passes, in order:
    //   compact()              — reclaim orphan storage from try_collapse.
    //   deduplicate_subtrees() — share identical 8-child blocks as DAG refs.
    //   morton_reorder()       — rewrite node storage in BFS/Morton order
    //                            so descent-time cache lines pack siblings'
    //                            children adjacently. Pure data-layout pass;
    //                            same tree semantics, just better L2 hit
    //                            rate on warp-coherent ray descents.
    //   prefilter_internals()  — bottom-up walk that emits a prefiltered
    //                            LeafAttr for each branch node, enabling
    //                            the GPU march's screen-footprint early
    //                            exit. Shares attr_dedup with the leaf
    //                            allocations above, so any new attrs bump
    //                            the existing contiguous pool range.
    let nodes_before_compact = octree.node_count();
    octree.compact();
    let nodes_after_compact = octree.node_count();
    octree.deduplicate_subtrees();
    let nodes_after_dedup = octree.node_count();
    octree.morton_reorder();
    let t_after_structure = t_start.elapsed();
    let attrs_before_prefilter = leaf_attr_pool.allocated_count() - leaf_attr_slot_start;
    crate::prefilter::prefilter_octree_internals(
        &mut octree,
        leaf_attr_pool,
        brick_pool,
    );
    let attrs_after_prefilter = leaf_attr_pool.allocated_count() - leaf_attr_slot_start;
    let t_after_prefilter = t_start.elapsed();

    // Compute the face-adjacency links for this voxelization's bricks.
    // The table spans 0..=max_brick_id so the GPU can index it by the
    // brick_id stored in octree BRICK nodes without any remapping.
    let brick_face_links = if let Some(&max_brick) = brick_ids.iter().max() {
        crate::brick_face_links::compute_brick_face_links(&octree, max_brick)
    } else {
        Vec::new()
    };
    let t_total = t_start.elapsed();

    let ms = |d: std::time::Duration| d.as_secs_f32() * 1000.0;
    eprintln!(
        "[voxelize_octree] depth={} voxels={} nodes {}→{}→{}  attrs={}(+{} prefilter)  face_links={}  \
         subdivide={:.2}ms structure={:.2}ms prefilter={:.2}ms face_links={:.2}ms total={:.2}ms",
        depth, voxel_count,
        nodes_before_compact, nodes_after_compact, nodes_after_dedup,
        attrs_before_prefilter, attrs_after_prefilter - attrs_before_prefilter,
        brick_face_links.len(),
        ms(t_after_subdivide),
        ms(t_after_structure - t_after_subdivide),
        ms(t_after_prefilter - t_after_structure),
        ms(t_total - t_after_prefilter),
        ms(t_total),
    );
    let subdivide_cpu = t_after_subdivide
        .saturating_sub(bake_stats.sdf_classify_total + bake_stats.sdf_bricks);
    eprintln!(
        "[voxelize_octree/subdivide] classify {} samples in {} dispatches={:.2}ms (cpu={:.2}ms)  \
         bricks={} brick_sdf={:.2}ms brick_cpu={:.2}ms (cpu total={:.2}ms)",
        bake_stats.classify_samples,
        bake_stats.classify_dispatches,
        ms(bake_stats.sdf_classify_total),
        ms(bake_stats.classify_cpu),
        bake_stats.brick_sample_total,
        ms(bake_stats.sdf_bricks),
        ms(bake_stats.brick_cpu),
        ms(subdivide_cpu),
    );

    Some(VoxelizeOctreeResult {
        octree,
        voxel_count,
        leaf_attr_unique_count: leaf_attr_pool.allocated_count() - leaf_attr_slot_start,
        leaf_attr_slot_start,
        brick_ids,
        brick_face_links,
        grid_origin,
    })
}

/// Accumulator for per-phase timings in one bake. Populated by
/// `subdivide_bfs` + `emit_bricks_batched`, logged once at the end of
/// `voxelize_octree`.
#[derive(Default)]
struct BakeStats {
    /// Total wall time spent inside `sdf_fn` for classify dispatches
    /// (one per octree level).
    sdf_classify_total: std::time::Duration,
    /// CPU time spent building sample lists + running
    /// `classify_from_samples` for every level.
    classify_cpu: std::time::Duration,
    classify_dispatches: u32,
    classify_samples: u64,
    /// Wall time inside `sdf_fn` for the single brick-emission batch.
    sdf_bricks: std::time::Duration,
    /// CPU time walking brick readbacks, allocating leaf_attr / bricks,
    /// and writing cells.
    brick_cpu: std::time::Duration,
    brick_sample_total: usize,
}

/// Classification of one octree node's cubic region. Used in both
/// coarse-descent and leaf-emission paths.
#[derive(Debug, Clone, Copy, PartialEq)]
enum RegionClass {
    Empty,
    Interior,
    Mixed,
}

/// Classify a region from its 9-sample slice (8 corners in xyz-minor
/// order, then center). Same semantics as the old per-node
/// `classify_region` — a 1-Lipschitz SDF means the surface can be at
/// most `|d|` away from a sample, so if every `|d| > extent / 2` the
/// node is definitively empty-or-interior depending on sign.
fn classify_from_samples(samples: &[(f32, u16, u16, u8, u32)], extent: f32) -> RegionClass {
    debug_assert_eq!(samples.len(), 9);
    let threshold = extent * 0.5;
    let mut all_outside = true;
    let mut all_inside = true;
    for &(d, _, _, _, _) in samples {
        if d <= threshold {
            all_outside = false;
        }
        if d >= -threshold {
            all_inside = false;
        }
    }
    if all_outside {
        RegionClass::Empty
    } else if all_inside {
        RegionClass::Interior
    } else {
        RegionClass::Mixed
    }
}

/// Push 9 classify samples (8 corners + center) for the node at
/// `world_min` with side length `extent` into `out`. Corner order is
/// `(cx, cy, cz)` with `cx` varying fastest, matching the loop order in
/// `classify_from_samples`'s implicit convention.
fn push_classify_positions(out: &mut Vec<Vec3>, world_min: Vec3, extent: f32) {
    for cz in 0..2u32 {
        for cy in 0..2u32 {
            for cx in 0..2u32 {
                out.push(
                    world_min
                        + Vec3::new(
                            cx as f32 * extent,
                            cy as f32 * extent,
                            cz as f32 * extent,
                        ),
                );
            }
        }
    }
    out.push(world_min + Vec3::splat(extent * 0.5));
}

/// Build-time descriptor for a Mixed brick-level node awaiting cell
/// processing. We collect these across the BFS pass and resolve them
/// in one batched `sdf_fn` call afterwards.
struct BrickJob {
    coord: UVec3,
    /// World-space origin of the brick — the `world_min` of the
    /// brick-depth node.
    world_min: Vec3,
}

/// Same idea as `BrickJob` but for the non-brick path: shallow trees
/// where every Mixed finest-level node becomes a single leaf rather
/// than a brick.
struct LeafJob {
    coord: UVec3,
    world_min: Vec3,
}

/// Shared layout for a brick cell's 7-sample block: center + 6 axis-
/// aligned gradient taps at `±eps` along each axis. Gradient kernel
/// matches the old recursive path exactly so bake output is bit-stable
/// up to leaf_attr_pool allocation ordering.
///
/// Offsets in the returned slice (after the cell_center at index 0):
///   1 = +x, 2 = -x, 3 = +y, 4 = -y, 5 = +z, 6 = -z
fn push_cell_samples(out: &mut Vec<Vec3>, cell_center: Vec3, eps: f32) {
    out.push(cell_center);
    out.push(cell_center + Vec3::new(eps, 0.0, 0.0));
    out.push(cell_center - Vec3::new(eps, 0.0, 0.0));
    out.push(cell_center + Vec3::new(0.0, eps, 0.0));
    out.push(cell_center - Vec3::new(0.0, eps, 0.0));
    out.push(cell_center + Vec3::new(0.0, 0.0, eps));
    out.push(cell_center - Vec3::new(0.0, 0.0, eps));
}



/// Convenience: voxelize a sphere into a sparse octree. Wraps the
/// per-point SDF in a trivial batching adapter so we keep the one
/// canonical voxelize_octree signature.
pub fn voxelize_sphere_octree(
    center: Vec3,
    radius: f32,
    material_id: u16,
    voxel_size: f32,
    leaf_attr_pool: &mut LeafAttrPool,
    brick_pool: &mut BrickPool,
) -> Option<VoxelizeOctreeResult> {
    let padding = voxel_size * 2.0;
    let aabb = Aabb {
        min: center - Vec3::splat(radius + padding),
        max: center + Vec3::splat(radius + padding),
    };

    let sdf_fn = |positions: &[Vec3]| -> Vec<(f32, u16, u16, u8, u32)> {
        positions
            .iter()
            .map(|pos| {
                let d = (*pos - center).length() - radius;
                // No per-voxel color on the sphere helper — the
                // color_pool's `0` sentinel means the voxel shader
                // falls back to the material's base color, which is
                // what we want for an untextured primitive.
                (d, material_id, 0, 0, 0u32)
            })
            .collect()
    };

    voxelize_octree(sdf_fn, &aabb, voxel_size, leaf_attr_pool, brick_pool)
}

/// Self-contained bake result — everything needed to integrate a
/// voxelization into a scene, independent of which pools the baker was
/// using. Produced by [`voxelize_to_artifact`] (fresh private pools)
/// for the async-bake worker; the main thread remaps all of the IDs
/// into the shared scene pools at integrate time.
///
/// Invariants (guaranteed by `voxelize_to_artifact`):
/// * Leaf-attr IDs referenced inside `octree`, `brick_cells`, and
///   `octree.internal_attr_slice()` are in `0..leaf_attrs.len()` — the
///   baker used a fresh pool, so allocations are a dense bump range.
/// * Brick IDs referenced inside `octree.as_slice()` and
///   `brick_face_links` are in `0..brick_cells.len()`, same reason.
/// * Sentinel values stay: `EMPTY_NODE`, `INTERIOR_NODE`,
///   [`BRICK_EMPTY`](crate::brick_pool::BRICK_EMPTY),
///   [`BRICK_INTERIOR`](crate::brick_pool::BRICK_INTERIOR),
///   [`FACE_EMPTY`](crate::brick_face_links::FACE_EMPTY),
///   [`FACE_INTERIOR`](crate::brick_face_links::FACE_INTERIOR),
///   [`INTERNAL_ATTR_NONE`](crate::sparse_octree::INTERNAL_ATTR_NONE)
///   are all passed through unchanged by the integrator.
pub struct BakeArtifact {
    pub octree: SparseOctree,
    pub voxel_count: u32,
    pub grid_origin: Vec3,
    /// Worker-local leaf attrs, index = worker-local ID.
    pub leaf_attrs: Vec<LeafAttr>,
    /// Parallel per-attr color overrides (0 = no override).
    pub leaf_attr_colors: Vec<u32>,
    /// Per-brick cell payloads (worker-local brick ID = outer index).
    /// Each `[u32; BRICK_CELLS]` entry's cells are worker-local
    /// leaf_attr IDs, or `BRICK_EMPTY`/`BRICK_INTERIOR` sentinels.
    pub brick_cells: Vec<[u32; BRICK_CELLS as usize]>,
    /// 6 face-adjacent brick ids per brick, indexed by worker-local
    /// brick ID. Length matches `brick_cells`. Contains `FACE_EMPTY`/
    /// `FACE_INTERIOR` sentinels for non-brick neighbors.
    pub brick_face_links: Vec<[u32; 6]>,
}

/// `voxelize_octree` against fresh private pools, packaged as a
/// [`BakeArtifact`]. This is the async-bake worker's entry point — it
/// runs entirely off the engine thread and produces a self-contained
/// result the main thread can integrate at its leisure.
pub fn voxelize_to_artifact<F>(
    sdf_fn: F,
    aabb: &Aabb,
    base_voxel_size: f32,
) -> Option<BakeArtifact>
where
    F: FnMut(&[Vec3]) -> Vec<(f32, u16, u16, u8, u32)>,
{
    use crate::brick_pool::BRICK_CELLS as BC;
    // Small initial capacities — pools grow on demand and this keeps
    // the allocator pressure low on small bakes.
    let mut leaf_attr_pool = LeafAttrPool::new(1024);
    let mut brick_pool = BrickPool::new(256);

    let result =
        voxelize_octree(sdf_fn, aabb, base_voxel_size, &mut leaf_attr_pool, &mut brick_pool)?;

    let n_attrs = result.leaf_attr_unique_count as usize;
    let mut leaf_attrs: Vec<LeafAttr> = Vec::with_capacity(n_attrs);
    let mut leaf_attr_colors: Vec<u32> = Vec::with_capacity(n_attrs);
    for i in 0..n_attrs as u32 {
        leaf_attrs.push(*leaf_attr_pool.get(i));
        leaf_attr_colors.push(leaf_attr_pool.color(i));
    }

    let brick_cells: Vec<[u32; BC as usize]> = result
        .brick_ids
        .iter()
        .map(|&id| {
            let cells = brick_pool.brick_cells(id);
            let mut arr = [crate::brick_pool::BRICK_EMPTY; BC as usize];
            arr.copy_from_slice(cells);
            arr
        })
        .collect();

    Some(BakeArtifact {
        octree: result.octree,
        voxel_count: result.voxel_count,
        grid_origin: result.grid_origin,
        leaf_attrs,
        leaf_attr_colors,
        brick_cells,
        brick_face_links: result.brick_face_links,
    })
}

