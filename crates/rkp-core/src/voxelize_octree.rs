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

use crate::brick_pool::{BrickPool, BRICK_CELLS, BRICK_DIM, BRICK_LEVELS};
use crate::leaf_attr::LeafAttr;
use crate::leaf_attr_pool::LeafAttrPool;
use crate::sparse_octree::SparseOctree;

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
    subdivide_bfs(
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

/// BFS octree classification + terminal-level emission.
///
/// Iterates from level 0 down to the terminal level (brick_depth if
/// bricks are enabled, else max_depth). At each level, batches the
/// classify samples for every active node into one `sdf_fn` call.
/// Mixed nodes feed the next level (as 8 children) or are queued for
/// terminal-level geometry emission. After the BFS loop, two more
/// batched calls emit bricks (all in one dispatch) and finest-level
/// leaves (all in one dispatch).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn subdivide_bfs<F>(
    sdf_fn: &mut F,
    octree: &mut SparseOctree,
    leaf_attr_pool: &mut LeafAttrPool,
    brick_pool: &mut BrickPool,
    voxel_count: &mut u32,
    brick_ids: &mut Vec<u32>,
    grid_origin: Vec3,
    max_depth: u8,
    brick_depth: Option<u8>,
    base_voxel_size: f32,
    stats: &mut BakeStats,
) -> Option<()>
where
    F: FnMut(&[Vec3]) -> Vec<(f32, u16, u16, u8, u32)>,
{
    // The octree's root spans `2^max_depth * base_voxel_size` on a side.
    let root_extent = (1u64 << max_depth) as f32 * base_voxel_size;

    // Active set for the current level. Holds coord in finest-level
    // units — multiplying by `base_voxel_size` and adding
    // `grid_origin` gives the node's world_min. Size: octree integer
    // coords fit in u32 because `depth <= 30` in practice.
    let mut active: Vec<UVec3> = vec![UVec3::ZERO];

    // Queues populated at the terminal level during BFS, drained in a
    // single batched pass after classification finishes.
    let mut brick_queue: Vec<BrickJob> = Vec::new();
    let mut leaf_queue: Vec<LeafJob> = Vec::new();

    // Classify level-by-level. We always visit level 0 (root) even if
    // the tree is trivial; beyond that, the loop body early-exits
    // when `active` becomes empty.
    let terminal_level = brick_depth.unwrap_or(max_depth);
    for level in 0..=terminal_level {
        if active.is_empty() {
            break;
        }

        // Extent of a node at this level, in world units.
        let level_extent = (1u64 << (max_depth - level)) as f32 * base_voxel_size;

        // Generate 9 classify samples per node. The layout assumes
        // `classify_from_samples`'s corner + center order.
        let t_level_cpu_start = std::time::Instant::now();
        let mut samples: Vec<Vec3> = Vec::with_capacity(active.len() * 9);
        for &coord in &active {
            let world_min =
                grid_origin + Vec3::new(coord.x as f32, coord.y as f32, coord.z as f32)
                    * base_voxel_size;
            push_classify_positions(&mut samples, world_min, level_extent);
        }
        stats.classify_cpu += t_level_cpu_start.elapsed();

        let t_sdf_start = std::time::Instant::now();
        let results = sdf_fn(&samples);
        stats.sdf_classify_total += t_sdf_start.elapsed();
        stats.classify_dispatches += 1;
        stats.classify_samples += samples.len() as u64;
        debug_assert_eq!(results.len(), samples.len());

        let t_process_start = std::time::Instant::now();
        // Process each node's classification. Mixed nodes either
        // recurse (schedule 8 children for the next level) or are
        // queued for terminal-level geometry emission.
        let mut next_active: Vec<UVec3> = Vec::new();
        let child_voxels = if level < max_depth {
            1u32 << (max_depth - level - 1)
        } else {
            0
        };

        for (i, &coord) in active.iter().enumerate() {
            let slice = &results[i * 9..i * 9 + 9];
            let class = classify_from_samples(slice, level_extent);

            match class {
                RegionClass::Empty => {
                    // Default octree state is EMPTY — no write needed.
                }
                RegionClass::Interior => {
                    octree.set_at_level(coord, level, crate::sparse_octree::INTERIOR_NODE);
                }
                RegionClass::Mixed => {
                    if brick_depth == Some(level) {
                        // Terminal: emit a brick for this node after the
                        // BFS classification loop.
                        let world_min = grid_origin
                            + Vec3::new(coord.x as f32, coord.y as f32, coord.z as f32)
                                * base_voxel_size;
                        brick_queue.push(BrickJob { coord, world_min });
                    } else if level == max_depth {
                        // Terminal: emit a finest-level leaf after the
                        // BFS classification loop. Only fires for
                        // shallow trees (depth ≤ BRICK_LEVELS) where
                        // bricking is disabled.
                        let world_min = grid_origin
                            + Vec3::new(coord.x as f32, coord.y as f32, coord.z as f32)
                                * base_voxel_size;
                        leaf_queue.push(LeafJob { coord, world_min });
                    } else {
                        // Descend: schedule 8 children for the next
                        // level. octant xyz-minor ordering matches the
                        // old recursive path.
                        for octant in 0u32..8 {
                            let dx = octant & 1;
                            let dy = (octant >> 1) & 1;
                            let dz = (octant >> 2) & 1;
                            next_active.push(UVec3::new(
                                coord.x + dx * child_voxels,
                                coord.y + dy * child_voxels,
                                coord.z + dz * child_voxels,
                            ));
                        }
                    }
                }
            }
        }

        active = next_active;
        stats.classify_cpu += t_process_start.elapsed();
    }

    // ── Terminal-level geometry: bricks ──
    if !brick_queue.is_empty() {
        emit_bricks_batched(
            &mut *sdf_fn,
            octree,
            leaf_attr_pool,
            brick_pool,
            voxel_count,
            brick_ids,
            &brick_queue,
            brick_depth.expect("brick_queue non-empty ⇒ brick_depth set"),
            base_voxel_size,
            stats,
        )?;
    }

    // ── Terminal-level geometry: finest-level leaves (shallow tree) ──
    if !leaf_queue.is_empty() {
        emit_leaves_batched(
            &mut *sdf_fn,
            octree,
            leaf_attr_pool,
            voxel_count,
            &leaf_queue,
            max_depth,
            base_voxel_size,
            stats,
        )?;
    }

    let _ = root_extent; // kept for future diagnostics / overflow guard
    Some(())
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

/// Two-phase brick emission.
///
/// **Phase 1**: sample only `d_center` per cell (one sample per cell,
/// BRICK_DIM³ per brick). Classify each cell via 1-Lipschitz bounds:
///
/// * `d_center >  cell_size * sqrt(3)/2` → **EMPTY**. No corner of the
///   cell can be inside; leave as `BRICK_EMPTY`.
/// * `d_center < -cell_size * sqrt(3)/2` → **INTERIOR**. No corner of
///   the cell can be outside; set `BRICK_INTERIOR` (no leaf_attr, no
///   gradient, same render cost as EMPTY).
/// * otherwise → **SURFACE**. Queue a 6-tap gradient fetch for phase
///   2 and store the center sample's material for later.
///
/// **Phase 2**: dispatch 6 axis-aligned taps at `±eps` per surface
/// cell. Build the gradient normal, allocate a `LeafAttr`, write the
/// cell.
///
/// Previously this was a single 7-sample dispatch per cell that
/// fetched the gradient taps even for clearly-EMPTY or clearly-
/// INTERIOR cells. For solid objects that's 6-8× wasted GPU work —
/// the vast majority of cells sit well away from the surface and the
/// gradient would never be read. The 2-phase rework cuts a 20 m ramp
/// bake from ~200 M brick samples to ~25 M.
#[allow(clippy::too_many_arguments)]
fn emit_bricks_batched<F>(
    sdf_fn: &mut F,
    octree: &mut SparseOctree,
    leaf_attr_pool: &mut LeafAttrPool,
    brick_pool: &mut BrickPool,
    voxel_count: &mut u32,
    brick_ids: &mut Vec<u32>,
    brick_queue: &[BrickJob],
    brick_depth: u8,
    base_voxel_size: f32,
    stats: &mut BakeStats,
) -> Option<()>
where
    F: FnMut(&[Vec3]) -> Vec<(f32, u16, u16, u8, u32)>,
{
    let t_start = std::time::Instant::now();
    let cell_size = base_voxel_size;
    let eps = cell_size * 0.5;
    // 1-Lipschitz threshold for "no point in the cell crosses zero"
    // using the cell-center sample alone. The cell's far corner sits
    // at `cell_size * sqrt(3)/2 ≈ 0.866 * cell_size` from the center,
    // so if `|d_center|` exceeds that, the surface is definitively
    // outside the cell.
    let lipschitz_threshold = cell_size * (3.0_f32.sqrt() * 0.5);
    let cells_per_brick = (BRICK_DIM * BRICK_DIM * BRICK_DIM) as usize;

    // ── Allocate brick IDs up-front so phase 2's surface queue can
    //    reference them by (brick_id, cx, cy, cz).
    let mut brick_slots: Vec<u32> = Vec::with_capacity(brick_queue.len());
    for _ in brick_queue {
        brick_slots.push(brick_pool.allocate()?);
    }
    // Mirror into the caller's `brick_ids` so deallocate knows what
    // we held. Done here so an early-return on phase 2 alloc failure
    // still leaves state deallocatable.
    brick_ids.extend_from_slice(&brick_slots);

    // ── Phase 1: d_center per cell. ──────────────────────────────
    let phase1_count = brick_queue.len() * cells_per_brick;
    let mut phase1_samples: Vec<Vec3> = Vec::with_capacity(phase1_count);
    for job in brick_queue {
        for cz in 0..BRICK_DIM {
            for cy in 0..BRICK_DIM {
                for cx in 0..BRICK_DIM {
                    let cell_min = job.world_min
                        + Vec3::new(
                            cx as f32 * cell_size,
                            cy as f32 * cell_size,
                            cz as f32 * cell_size,
                        );
                    let cell_center = cell_min + Vec3::splat(cell_size * 0.5);
                    phase1_samples.push(cell_center);
                }
            }
        }
    }

    let t_phase1_prep = t_start.elapsed();
    let t_phase1_sdf = std::time::Instant::now();
    let phase1_results = sdf_fn(&phase1_samples);
    stats.sdf_bricks += t_phase1_sdf.elapsed();
    stats.brick_sample_total += phase1_count;
    debug_assert_eq!(phase1_results.len(), phase1_count);

    // ── Classify + queue surface cells. ──────────────────────────
    let t_classify = std::time::Instant::now();
    // Each surface entry records everything needed to populate its
    // cell after phase 2 reads back: the brick slot, the 3D cell
    // coord, and the phase-1 sample that carries material/color/blend.
    struct SurfaceCell {
        brick_slot: u32,
        cx: u32,
        cy: u32,
        cz: u32,
        d_center: f32,
        primary: u16,
        secondary: u16,
        blend: u8,
        color: u32,
    }
    let mut surface_cells: Vec<SurfaceCell> = Vec::new();
    for (brick_idx, _job) in brick_queue.iter().enumerate() {
        let brick_slot = brick_slots[brick_idx];
        for cz in 0..BRICK_DIM {
            for cy in 0..BRICK_DIM {
                for cx in 0..BRICK_DIM {
                    let cell_idx = (cz * BRICK_DIM * BRICK_DIM + cy * BRICK_DIM + cx) as usize;
                    let flat = brick_idx * cells_per_brick + cell_idx;
                    let (d_center, primary, secondary, blend, color) = phase1_results[flat];
                    if d_center > lipschitz_threshold {
                        // Fully outside. Default brick cell is already
                        // BRICK_EMPTY, nothing to write.
                        continue;
                    }
                    if d_center < -lipschitz_threshold {
                        // Fully inside. Sentinel → no leaf_attr, no
                        // gradient fetch, same cost as EMPTY at march
                        // time.
                        brick_pool.set_cell(
                            brick_slot, cx, cy, cz,
                            crate::brick_pool::BRICK_INTERIOR,
                        );
                        continue;
                    }
                    // Surface cell — defer to phase 2.
                    surface_cells.push(SurfaceCell {
                        brick_slot, cx, cy, cz,
                        d_center, primary, secondary, blend, color,
                    });
                }
            }
        }
    }
    stats.brick_cpu += t_classify.elapsed();

    // ── Phase 2: 6 gradient taps per surface cell. ───────────────
    if !surface_cells.is_empty() {
        let t_phase2_prep = std::time::Instant::now();
        let phase2_count = surface_cells.len() * 6;
        let mut phase2_samples: Vec<Vec3> = Vec::with_capacity(phase2_count);
        for sc in &surface_cells {
            let cell_min = Vec3::new(
                sc.cx as f32 * cell_size,
                sc.cy as f32 * cell_size,
                sc.cz as f32 * cell_size,
            );
            // Reconstruct brick world_min from the brick's first cell
            // in phase1. Easier: keep world_min on the SurfaceCell.
            // Avoid by iterating brick_queue too. Simpler: we stored
            // the cell-local offset; grab world_min from brick_queue.
            let _ = cell_min;
        }
        // Build phase 2 sample list. Re-derive world positions from
        // brick_queue; cheaper than threading world_min through
        // SurfaceCell for each of the millions of surface cells.
        phase2_samples.clear();
        // Index surface cells by brick so we can reuse each brick's
        // world_min. SurfaceCell already carries (brick_slot, cx..cz)
        // — brick index is `brick_slot`'s position in `brick_slots`.
        // We stored insertion order so `brick_slot` is unique per
        // brick_idx. Build reverse lookup once.
        let mut slot_to_idx = std::collections::HashMap::with_capacity(brick_slots.len());
        for (i, &id) in brick_slots.iter().enumerate() {
            slot_to_idx.insert(id, i);
        }
        for sc in &surface_cells {
            let brick_idx = slot_to_idx[&sc.brick_slot];
            let job = &brick_queue[brick_idx];
            let cell_min = job.world_min
                + Vec3::new(
                    sc.cx as f32 * cell_size,
                    sc.cy as f32 * cell_size,
                    sc.cz as f32 * cell_size,
                );
            let cell_center = cell_min + Vec3::splat(cell_size * 0.5);
            phase2_samples.push(cell_center + Vec3::new(eps, 0.0, 0.0));
            phase2_samples.push(cell_center - Vec3::new(eps, 0.0, 0.0));
            phase2_samples.push(cell_center + Vec3::new(0.0, eps, 0.0));
            phase2_samples.push(cell_center - Vec3::new(0.0, eps, 0.0));
            phase2_samples.push(cell_center + Vec3::new(0.0, 0.0, eps));
            phase2_samples.push(cell_center - Vec3::new(0.0, 0.0, eps));
        }
        let _ = t_phase2_prep;

        let t_phase2_sdf = std::time::Instant::now();
        let phase2_results = sdf_fn(&phase2_samples);
        stats.sdf_bricks += t_phase2_sdf.elapsed();
        stats.brick_sample_total += phase2_count;
        debug_assert_eq!(phase2_results.len(), phase2_count);

        // ── Populate surface cells from phase 2 readback. ────────
        let t_populate = std::time::Instant::now();
        for (i, sc) in surface_cells.iter().enumerate() {
            let base = i * 6;
            let d_xp = phase2_results[base    ].0;
            let d_xm = phase2_results[base + 1].0;
            let d_yp = phase2_results[base + 2].0;
            let d_ym = phase2_results[base + 3].0;
            let d_zp = phase2_results[base + 4].0;
            let d_zm = phase2_results[base + 5].0;
            // Second-chance INTERIOR / EMPTY checks with the tighter
            // sample set. Occasionally the center falls inside the
            // Lipschitz band but all 6 face-taps land on one side —
            // cheaper to reclassify than emit a leaf_attr that never
            // contributes a visible surface.
            let max_tap = d_xp.max(d_xm).max(d_yp).max(d_ym).max(d_zp).max(d_zm);
            let min_tap = d_xp.min(d_xm).min(d_yp).min(d_ym).min(d_zp).min(d_zm);
            if min_tap > 0.0 && sc.d_center > 0.0 {
                // Stays EMPTY.
                continue;
            }
            if max_tap < 0.0 && sc.d_center < 0.0 {
                brick_pool.set_cell(
                    sc.brick_slot, sc.cx, sc.cy, sc.cz,
                    crate::brick_pool::BRICK_INTERIOR,
                );
                continue;
            }

            let grad = Vec3::new(d_xp - d_xm, d_yp - d_ym, d_zp - d_zm);
            let normal = if grad.length_squared() > 1e-12 {
                grad.normalize()
            } else {
                Vec3::Y
            };
            let attr = LeafAttr::new_blended(normal, sc.primary, sc.secondary, sc.blend);
            // No dedup — every cell gets its own slot so paint and
            // cursor have per-cell identity to work with. Bump-only
            // allocate keeps the asset's pool range contiguous, which
            // `release_asset` relies on for a single deallocate_range
            // to free everything.
            let leaf_attr_id = leaf_attr_pool.allocate_contiguous_bump(1)?;
            *leaf_attr_pool.get_mut(leaf_attr_id) = attr;
            if sc.color != 0 {
                leaf_attr_pool.set_color(leaf_attr_id, sc.color);
            }
            brick_pool.set_cell(sc.brick_slot, sc.cx, sc.cy, sc.cz, leaf_attr_id);
            *voxel_count += 1;
        }
        stats.brick_cpu += t_populate.elapsed();
    }

    // ── Wire each allocated brick into the octree. ───────────────
    for (brick_idx, job) in brick_queue.iter().enumerate() {
        octree.set_at_level(
            job.coord,
            brick_depth,
            crate::sparse_octree::make_brick(brick_slots[brick_idx]),
        );
    }

    stats.brick_cpu += t_phase1_prep;
    Some(())
}

/// Emit finest-level single-cell leaves for shallow trees (depth ≤
/// BRICK_LEVELS, where bricks are disabled). Same 7-sample layout
/// per cell as the brick path.
#[allow(clippy::too_many_arguments)]
fn emit_leaves_batched<F>(
    sdf_fn: &mut F,
    octree: &mut SparseOctree,
    leaf_attr_pool: &mut LeafAttrPool,
    voxel_count: &mut u32,
    leaf_queue: &[LeafJob],
    max_depth: u8,
    base_voxel_size: f32,
    stats: &mut BakeStats,
) -> Option<()>
where
    F: FnMut(&[Vec3]) -> Vec<(f32, u16, u16, u8, u32)>,
{
    let t_start = std::time::Instant::now();
    let eps = base_voxel_size * 0.5;
    let samples_per_leaf = 7usize;
    let total_samples = leaf_queue.len() * samples_per_leaf;
    stats.brick_sample_total += total_samples;

    let mut samples: Vec<Vec3> = Vec::with_capacity(total_samples);
    for job in leaf_queue {
        let voxel_center = job.world_min + Vec3::splat(base_voxel_size * 0.5);
        push_cell_samples(&mut samples, voxel_center, eps);
    }
    let t_prep = t_start.elapsed();

    let t_sdf_start = std::time::Instant::now();
    let results = sdf_fn(&samples);
    stats.sdf_bricks += t_sdf_start.elapsed();
    debug_assert_eq!(results.len(), total_samples);

    let t_cpu_start = std::time::Instant::now();
    for (leaf_idx, job) in leaf_queue.iter().enumerate() {
        let cell_base = leaf_idx * samples_per_leaf;
        let (d_center, primary, secondary, blend, color) = results[cell_base];
        if d_center > 0.0 {
            // Center is outside — this corner of a Mixed region is
            // not itself solid. Leave it EMPTY.
            continue;
        }
        let d_xp = results[cell_base + 1].0;
        let d_xm = results[cell_base + 2].0;
        let d_yp = results[cell_base + 3].0;
        let d_ym = results[cell_base + 4].0;
        let d_zp = results[cell_base + 5].0;
        let d_zm = results[cell_base + 6].0;
        let grad = Vec3::new(d_xp - d_xm, d_yp - d_ym, d_zp - d_zm);
        let normal = if grad.length_squared() > 1e-12 {
            grad.normalize()
        } else {
            Vec3::Y
        };
        let attr = LeafAttr::new_blended(normal, primary, secondary, blend);
        // No dedup — see emit_bricks_batched for the reasoning. Plain
        // LEAF emission only fires for shallow trees (depth ≤ BRICK_LEVELS),
        // so this path is rare in practice.
        let leaf_attr_id = leaf_attr_pool.allocate()?;
        *leaf_attr_pool.get_mut(leaf_attr_id) = attr;
        if color != 0 {
            leaf_attr_pool.set_color(leaf_attr_id, color);
        }
        octree.set_at_level(
            job.coord,
            max_depth,
            crate::sparse_octree::make_leaf(leaf_attr_id),
        );
        *voxel_count += 1;
    }

    stats.brick_cpu += t_prep + t_cpu_start.elapsed();
    Some(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse_octree::INTERIOR_NODE;

    /// Wrap a per-point SDF as a batched callback for the tests.
    fn batched<Fp>(f: Fp) -> impl Fn(&[Vec3]) -> Vec<(f32, u16, u16, u8, u32)>
    where
        Fp: Fn(Vec3) -> (f32, u16, u16, u8, u32),
    {
        move |positions: &[Vec3]| positions.iter().map(|p| f(*p)).collect()
    }

    #[test]
    fn sphere_produces_brick_cells() {
        let mut attrs = LeafAttrPool::new(1_000_000);
        let mut bricks = BrickPool::new(10_000);
        let r = voxelize_sphere_octree(Vec3::ZERO, 0.5, 0, 0.1, &mut attrs, &mut bricks).unwrap();

        assert!(r.voxel_count > 0, "should populate cells for the sphere surface");
        assert!(!r.brick_ids.is_empty(), "surface should be encoded as bricks at brick_depth");
    }

    #[test]
    fn sphere_has_interior_nodes() {
        let mut attrs = LeafAttrPool::new(1_000_000);
        let mut bricks = BrickPool::new(10_000);
        let r = voxelize_sphere_octree(Vec3::ZERO, 3.0, 0, 0.1, &mut attrs, &mut bricks).unwrap();

        let ext = r.octree.extent();
        let mid = ext / 2;
        let val = r.octree.lookup(glam::UVec3::new(mid, mid, mid));
        assert_eq!(val, Some(INTERIOR_NODE), "large sphere should have interior at center");
    }

    #[test]
    fn empty_region_produces_no_voxels() {
        let mut attrs = LeafAttrPool::new(256);
        let mut bricks = BrickPool::new(64);
        let aabb = Aabb { min: Vec3::ZERO, max: Vec3::splat(1.0) };
        let r = voxelize_octree(batched(|_| (1000.0, 0, 0, 0, 0)), &aabb, 0.1, &mut attrs, &mut bricks).unwrap();

        assert_eq!(r.voxel_count, 0);
        assert_eq!(r.brick_ids.len(), 0);
        assert_eq!(r.octree.leaf_count(), 0);
    }

    #[test]
    fn fully_interior_region_is_interior() {
        let mut attrs = LeafAttrPool::new(256);
        let mut bricks = BrickPool::new(64);
        let aabb = Aabb { min: Vec3::ZERO, max: Vec3::splat(0.05) };
        let r = voxelize_octree(batched(|_| (-1000.0, 0, 0, 0, 0)), &aabb, 0.1, &mut attrs, &mut bricks).unwrap();

        assert_eq!(r.voxel_count, 0, "fully inside should collapse to INTERIOR");
        assert_eq!(r.brick_ids.len(), 0);
        assert_eq!(r.octree.as_slice()[0], INTERIOR_NODE);
    }

    #[test]
    fn leaf_attrs_carry_correct_material() {
        // Walk every leaf_attr this voxelize allocated and check material_primary.
        let mut attrs = LeafAttrPool::new(1_000_000);
        let mut bricks = BrickPool::new(10_000);
        let r = voxelize_sphere_octree(Vec3::ZERO, 0.3, 42, 0.1, &mut attrs, &mut bricks).unwrap();

        assert!(r.leaf_attr_unique_count > 0);
        for i in r.leaf_attr_slot_start..(r.leaf_attr_slot_start + r.leaf_attr_unique_count) {
            assert_eq!(attrs.get(i).material_primary, 42);
        }
    }

    #[test]
    fn sphere_normals_point_outward() {
        // Walk the brick nodes in the octree, expand each brick's 64 cells,
        // verify cell normals point outward from the sphere center.
        use crate::sparse_octree::{brick_id as get_brick_id, is_brick};

        let center = Vec3::ZERO;
        let radius = 0.5;
        let vs = 0.05;
        let mut attrs = LeafAttrPool::new(1_000_000);
        let mut bricks = BrickPool::new(10_000);
        let r = voxelize_sphere_octree(center, radius, 0, vs, &mut attrs, &mut bricks).unwrap();

        // Find brick nodes by walking the octree; for each, iterate cells.
        let mut checked = 0u32;
        let nodes = r.octree.as_slice().to_vec();
        let max_depth = r.octree.depth();
        let brick_depth = max_depth - BRICK_LEVELS;
        // Visit nodes recursively from root, tracking origin coord at each level.
        fn walk(
            nodes: &[u32],
            node_idx: usize,
            coord: UVec3,
            level: u8,
            brick_depth: u8,
            max_depth: u8,
            vs: f32,
            center: Vec3,
            radius: f32,
            bricks: &BrickPool,
            attrs: &LeafAttrPool,
            grid_origin: Vec3,
            checked: &mut u32,
        ) {
            use crate::sparse_octree::{
                brick_id as get_brick_id, is_brick, is_leaf, INTERIOR_NODE,
            };
            let node = nodes[node_idx];
            if is_brick(node) && level == brick_depth {
                let brick_id = get_brick_id(node);
                let brick_world_min = grid_origin
                    + Vec3::new(coord.x as f32, coord.y as f32, coord.z as f32) * vs;
                for cz in 0..BRICK_DIM {
                    for cy in 0..BRICK_DIM {
                        for cx in 0..BRICK_DIM {
                            let cell = bricks.get_cell(brick_id, cx, cy, cz);
                            if cell == crate::brick_pool::BRICK_EMPTY
                                || cell == crate::brick_pool::BRICK_INTERIOR
                            {
                                continue;
                            }
                            let attr = *attrs.get(cell);
                            let normal = attr.normal();
                            let cell_min = brick_world_min
                                + Vec3::new(cx as f32 * vs, cy as f32 * vs, cz as f32 * vs);
                            let cell_center = cell_min + Vec3::splat(vs * 0.5);
                            let radial = (cell_center - center).normalize();
                            // Normal should point outward (same half-space
                            // as the radial direction from the sphere
                            // center). Stricter than "> 0" because we're
                            // well outside the origin.
                            let dot = normal.dot(radial);
                            assert!(
                                dot > 0.0,
                                "cell normal {normal:?} at {cell_center:?} should point outward from sphere center (dot={dot})",
                            );
                            *checked += 1;
                        }
                    }
                }
                return;
            }
            if node == INTERIOR_NODE || is_leaf(node) {
                return;
            }
            // Otherwise it's an internal node — descend.
            if level >= brick_depth {
                return;
            }
            let first_child = node as usize;
            if first_child == 0 || first_child + 8 > nodes.len() {
                return;
            }
            let child_voxels = 1u32 << (max_depth - level - 1);
            for octant in 0u32..8 {
                let dx = octant & 1;
                let dy = (octant >> 1) & 1;
                let dz = (octant >> 2) & 1;
                let child_coord = UVec3::new(
                    coord.x + dx * child_voxels,
                    coord.y + dy * child_voxels,
                    coord.z + dz * child_voxels,
                );
                walk(
                    nodes,
                    first_child + octant as usize,
                    child_coord,
                    level + 1,
                    brick_depth,
                    max_depth,
                    vs,
                    center,
                    radius,
                    bricks,
                    attrs,
                    grid_origin,
                    checked,
                );
            }
        }
        walk(
            &nodes,
            0,
            UVec3::ZERO,
            0,
            brick_depth,
            max_depth,
            vs,
            center,
            radius,
            &bricks,
            &attrs,
            r.grid_origin,
            &mut checked,
        );
        assert!(checked > 0, "should have checked at least one cell normal");
        let _ = get_brick_id; let _ = is_brick; // silence unused-import warnings
    }
}
