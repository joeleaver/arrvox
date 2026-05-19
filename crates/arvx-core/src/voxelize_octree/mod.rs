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

use glam::{IVec3, UVec3, Vec3};
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
    /// Halo cells sampled outside the nominal AABB on all 6 faces (and
    /// 12 edges + 8 corners) when `voxelize_octree` was called with
    /// `halo > 0`. Each entry is `(cell_coord, leaf_attr_id)` where
    /// `cell_coord` is a signed offset in finest-grid integer units
    /// relative to `grid_origin` — i.e., coords with any axis in
    /// `[-halo, 0) ∪ [N, N + halo)` where `N = 1 << depth`. Interior
    /// halo cells (deep inside the solid neighbour) use the
    /// [`CELL_INTERIOR`](crate::mesh_extract::CELL_INTERIOR) sentinel;
    /// surface halo cells carry a real `leaf_attr_id` allocated
    /// contiguously alongside the interior cells (so `unique_count`
    /// covers both). Empty when `halo = 0`. Phase 3 of the terrain
    /// system uses this to give the surface-mesh extractor enough
    /// boundary data to produce watertight seams between adjacent
    /// tiles — see `docs/TERRAIN.md`.
    pub halo_cells: Vec<(IVec3, u32)>,
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
/// `aabb`: world-space bounding box of the object. **Must be cubic
/// (all three axes equal extent) AND `extent / base_voxel_size` must
/// be a power of 2.** The function returns `None` and logs an error
/// otherwise. Callers with arbitrary mesh bounds should pre-align via
/// [`pad_to_pow2_cubic`]. See `arvx_core::constants::RESOLUTION_TIERS`
/// for the engine-wide pow2-aligned voxel-size table.
///
/// `base_voxel_size`: voxel size at the finest level. Should come from
/// the unified tier table for grid-snap compatibility, though any
/// value that makes the AABB pow2-cubic-aligned is accepted.
///
/// `halo`: number of finest-grid voxels to sample OUTSIDE the AABB on
/// every face, edge, and corner. The octree itself is unchanged
/// (cells `[0, N)³` only), but `result.halo_cells` is populated with
/// the solidity / `leaf_attr_id` of every solid cell in
/// `[-halo, N + halo)³ \ [0, N)³`. Pass `0` (the common case) for
/// stand-alone assets. Pass `1` from the terrain bake path so the
/// surface-mesh extractor has the boundary-cell corner data it needs
/// to produce watertight seams between adjacent tiles.
pub fn voxelize_octree<F>(
    mut sdf_fn: F,
    aabb: &Aabb,
    base_voxel_size: f32,
    leaf_attr_pool: &mut LeafAttrPool,
    brick_pool: &mut BrickPool,
    halo: u32,
) -> Option<VoxelizeOctreeResult>
where
    F: FnMut(&[Vec3]) -> Vec<(f32, u16, u16, u8, u32)>,
{
    // Strict contract: AABB must be a cube whose extent equals
    // `(2^depth) * base_voxel_size` for some `depth`. This was loosened
    // pre-unification (the function silently rounded extent up to a
    // power-of-2 of voxels and re-centred the octree on the AABB), but
    // the resulting hidden padding broke voxel-aligned grid-snap and
    // any system that expected the octree extent to match the asset's
    // declared AABB. Now the caller MUST pre-align. See
    // `arvx_core::constants::RESOLUTION_TIERS` for the engine-wide
    // pow2-aligned tier table.
    let depth = match validate_pow2_cubic(aabb, base_voxel_size) {
        Some(d) => d,
        None => {
            log::error!(
                "voxelize_octree: AABB not pow2-cubic-aligned to voxel_size={base_voxel_size}: \
                 extent={:?}. Callers must pre-pad — see `arvx_core::constants::RESOLUTION_TIERS`.",
                aabb.max - aabb.min,
            );
            return None;
        }
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

    // Post-validation, octree extent equals AABB extent exactly —
    // no centring math, no hidden padding. The octree's lo corner
    // is the AABB's lo corner.
    let grid_origin = aabb.min;

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

    // Halo pass — sample SDF at the ring of cells one wider than the
    // nominal AABB on every face (+ edges + corners). Solid halo cells
    // get allocated `LeafAttr`s contiguous with the interior allocations
    // above so the asset's pool range is still a single bump-allocated
    // run; interior-bulk halo cells get the `CELL_INTERIOR` sentinel
    // and don't consume a pool slot. `halo = 0` skips this entirely.
    let halo_cells: Vec<(IVec3, u32)> = if halo > 0 {
        sample_halo_cells(
            &mut sdf_fn,
            grid_origin,
            depth,
            base_voxel_size,
            halo as i32,
            leaf_attr_pool,
        )?
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
        halo_cells,
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

/// Sample SDF at the halo shell of cells in `[-halo, N + halo)³ \
/// [0, N)³` and return the solid ones as `(cell_coord, leaf_attr_id)`
/// tuples.
///
/// ### Classification must mirror the neighbour's BFS
///
/// Naïve per-cell halo classification (a 2-phase center + 6-grad
/// classifier with cell-level Lipschitz threshold `cell_size · √3 / 2`)
/// produces a CRACK at every tile seam where the surface band's
/// Lipschitz radius exceeds one voxel. Reason: the neighbour tile's
/// BFS prunes branches at coarser scales — bricks classified Empty or
/// Interior by the lax 9-sample test (`extent / 2` threshold) are
/// never per-cell sampled, so individual cells inside such a brick get
/// the brick-level verdict regardless of what cell-level SDF says
/// about them. If our halo sampled per-cell, it would classify cells
/// at the band edge differently from the neighbour's interior path
/// (e.g., a cell with `d = -0.3` inside a `d < -0.5` brick is "Interior"
/// to the neighbour but "Surface" to a cell-level halo classifier),
/// and SN cubes straddling the seam would see asymmetric corner
/// classifications → divergent vertex positions → see-through cracks.
///
/// We instead match BFS exactly:
///
/// 1. Group halo cells by the world-aligned brick they belong to
///    (`brick_extent = BRICK_DIM · cell_size`, here 1 m).
/// 2. For each unique brick, sample 9 BFS-style positions (8 corners
///    + 1 centre) and classify with the lax `extent / 2` threshold
///    that `classify_from_samples` uses. (The 9-sample arrangement
///    makes `extent / 2` Lipschitz-correct — the worst-case interior
///    point is at most `extent / 2` from its nearest sample.)
/// 3. Empty brick → halo cells in it are skipped (the neighbour's
///    octree has nothing here).
/// 4. Interior brick → all halo cells in it are emitted with
///    `CELL_INTERIOR` (the neighbour's octree marks the whole region
///    as `INTERIOR_NODE`).
/// 5. Mixed brick → per-cell 2-phase classify, matching
///    `emit_bricks_batched`'s rules verbatim (`cell_size · √3 / 2`
///    threshold + phase-2 reclassify). Surface cells get a
///    contiguous-bump `LeafAttr`; interior cells get the
///    `CELL_INTERIOR` sentinel.
///
/// The brick-level decision propagates the neighbour's coarse-level
/// classification: by Lipschitz, any coarser branch classified
/// Empty/Interior implies all bricks inside are also Empty/Interior
/// (the coarse threshold is `coarse_extent / 2 ≥ brick_extent / 2`).
/// So this 2-level mirror is sufficient — no need to walk the full
/// BFS up to the root.
fn sample_halo_cells<F>(
    sdf_fn: &mut F,
    grid_origin: Vec3,
    octree_depth: u8,
    base_voxel_size: f32,
    halo: i32,
    leaf_attr_pool: &mut LeafAttrPool,
) -> Option<Vec<(IVec3, u32)>>
where
    F: FnMut(&[Vec3]) -> Vec<(f32, u16, u16, u8, u32)>,
{
    debug_assert!(halo > 0);
    let n = 1i32 << octree_depth;
    let cell_size = base_voxel_size;
    let eps = cell_size * 0.5;
    let cell_lipschitz = cell_size * (3.0_f32.sqrt() * 0.5);
    let brick_dim_cells = crate::brick_pool::BRICK_DIM as i32;
    let brick_extent_m = brick_dim_cells as f32 * cell_size;

    // ── Group halo cells by their containing brick. ──
    use std::collections::HashMap;
    let mut halo_by_brick: HashMap<IVec3, Vec<IVec3>> = HashMap::new();
    let lo = -halo;
    let hi = n + halo;
    for z in lo..hi {
        for y in lo..hi {
            for x in lo..hi {
                let inside = x >= 0 && x < n && y >= 0 && y < n && z >= 0 && z < n;
                if inside {
                    continue;
                }
                let cell = IVec3::new(x, y, z);
                let brick = IVec3::new(
                    cell.x.div_euclid(brick_dim_cells),
                    cell.y.div_euclid(brick_dim_cells),
                    cell.z.div_euclid(brick_dim_cells),
                );
                halo_by_brick.entry(brick).or_default().push(cell);
            }
        }
    }
    if halo_by_brick.is_empty() {
        return Some(Vec::new());
    }

    // ── Brick-level classify (9 samples per brick). ──
    let bricks: Vec<IVec3> = halo_by_brick.keys().copied().collect();
    let mut brick_samples: Vec<Vec3> = Vec::with_capacity(bricks.len() * 9);
    for &brick in &bricks {
        let world_min = grid_origin
            + Vec3::new(brick.x as f32, brick.y as f32, brick.z as f32) * brick_extent_m;
        push_classify_positions(&mut brick_samples, world_min, brick_extent_m);
    }
    let brick_results = sdf_fn(&brick_samples);
    debug_assert_eq!(brick_results.len(), bricks.len() * 9);

    // ── Decide per brick: Empty (skip), Interior (mark all), Mixed (queue). ──
    struct HaloSurface {
        coord: IVec3,
        primary: u16,
        secondary: u16,
        blend: u8,
        color: u32,
    }
    let mut out: Vec<(IVec3, u32)> = Vec::new();
    let mut mixed_cells: Vec<IVec3> = Vec::new();
    for (i, &brick) in bricks.iter().enumerate() {
        let slice = &brick_results[i * 9..i * 9 + 9];
        let class = classify_from_samples(slice, brick_extent_m);
        let cells = halo_by_brick.get(&brick).expect("brick groups complete");
        match class {
            RegionClass::Empty => {
                // No entries in CellMap for cells in this brick.
            }
            RegionClass::Interior => {
                for &c in cells {
                    out.push((c, crate::mesh_extract::CELL_INTERIOR));
                }
            }
            RegionClass::Mixed => {
                mixed_cells.extend(cells.iter().copied());
            }
        }
    }

    if mixed_cells.is_empty() {
        return Some(out);
    }

    // ── Per-cell 2-phase classify for Mixed-brick halo cells. ──
    let mut phase1_samples: Vec<Vec3> = Vec::with_capacity(mixed_cells.len());
    for &cell in &mixed_cells {
        let cell_center = grid_origin
            + Vec3::new(
                cell.x as f32 + 0.5,
                cell.y as f32 + 0.5,
                cell.z as f32 + 0.5,
            ) * cell_size;
        phase1_samples.push(cell_center);
    }
    let phase1_results = sdf_fn(&phase1_samples);
    debug_assert_eq!(phase1_results.len(), mixed_cells.len());

    let mut surface_queue: Vec<HaloSurface> = Vec::new();
    for (i, &coord) in mixed_cells.iter().enumerate() {
        let (d, primary, secondary, blend, color) = phase1_results[i];
        if d > cell_lipschitz {
            // Empty at cell level — skip (matches `emit_bricks_batched`'s
            // BRICK_EMPTY-by-default behaviour for cells with d_center
            // above the per-cell threshold).
            continue;
        }
        if d < -cell_lipschitz {
            // Cell-level interior — matches `BRICK_INTERIOR`.
            out.push((coord, crate::mesh_extract::CELL_INTERIOR));
            continue;
        }
        surface_queue.push(HaloSurface {
            coord,
            primary,
            secondary,
            blend,
            color,
        });
    }

    // ── Phase 2: 6 gradient taps per surface halo cell. ──
    if !surface_queue.is_empty() {
        let mut phase2_samples: Vec<Vec3> =
            Vec::with_capacity(surface_queue.len() * 6);
        for hs in &surface_queue {
            let cell_center = grid_origin
                + Vec3::new(
                    hs.coord.x as f32 + 0.5,
                    hs.coord.y as f32 + 0.5,
                    hs.coord.z as f32 + 0.5,
                ) * cell_size;
            phase2_samples.push(cell_center + Vec3::new(eps, 0.0, 0.0));
            phase2_samples.push(cell_center - Vec3::new(eps, 0.0, 0.0));
            phase2_samples.push(cell_center + Vec3::new(0.0, eps, 0.0));
            phase2_samples.push(cell_center - Vec3::new(0.0, eps, 0.0));
            phase2_samples.push(cell_center + Vec3::new(0.0, 0.0, eps));
            phase2_samples.push(cell_center - Vec3::new(0.0, 0.0, eps));
        }
        let phase2_results = sdf_fn(&phase2_samples);
        debug_assert_eq!(phase2_results.len(), surface_queue.len() * 6);

        for (i, hs) in surface_queue.iter().enumerate() {
            let base = i * 6;
            let d_xp = phase2_results[base    ].0;
            let d_xm = phase2_results[base + 1].0;
            let d_yp = phase2_results[base + 2].0;
            let d_ym = phase2_results[base + 3].0;
            let d_zp = phase2_results[base + 4].0;
            let d_zm = phase2_results[base + 5].0;
            // Second-chance reclassify with the tighter 6-tap set
            // (matches the interior emit path).
            let max_tap = d_xp.max(d_xm).max(d_yp).max(d_ym).max(d_zp).max(d_zm);
            let min_tap = d_xp.min(d_xm).min(d_yp).min(d_ym).min(d_zp).min(d_zm);
            if min_tap > 0.0 {
                continue;
            }
            if max_tap < 0.0 {
                out.push((hs.coord, crate::mesh_extract::CELL_INTERIOR));
                continue;
            }
            let grad = Vec3::new(d_xp - d_xm, d_yp - d_ym, d_zp - d_zm);
            let normal = if grad.length_squared() > 1e-12 {
                grad.normalize()
            } else {
                Vec3::Y
            };
            let attr = LeafAttr::new_blended(normal, hs.primary, hs.secondary, hs.blend);
            let leaf_attr_id = leaf_attr_pool.allocate_contiguous_bump(1)?;
            *leaf_attr_pool.get_mut(leaf_attr_id) = attr;
            if hs.color != 0 {
                leaf_attr_pool.set_color(leaf_attr_id, hs.color);
            }
            out.push((hs.coord, leaf_attr_id));
        }
    }

    Some(out)
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



/// Returns the octree depth for an AABB that satisfies the pow2-cubic
/// contract, or `None` if it doesn't. Pure validation — no side effects.
pub fn validate_pow2_cubic(aabb: &Aabb, voxel_size: f32) -> Option<u8> {
    let extent = aabb.max - aabb.min;
    // Cubic: x == y == z within voxel-tolerance.
    let tol = voxel_size * 0.01;
    if (extent.x - extent.y).abs() > tol || (extent.x - extent.z).abs() > tol {
        return None;
    }
    let cells = extent.x / voxel_size;
    let cells_int = cells.round() as i64;
    if cells_int < 1 || (cells - cells_int as f32).abs() > 0.01 {
        return None;
    }
    let cells_u32 = cells_int as u32;
    if !cells_u32.is_power_of_two() {
        return None;
    }
    Some(cells_u32.trailing_zeros() as u8)
}

/// Round an arbitrary AABB up to a pow2-cubic AABB aligned to
/// `voxel_size`. The returned AABB always contains the input, is
/// cubic, and has `extent / voxel_size` equal to a power of 2 — i.e.,
/// the contract `voxelize_octree` requires. `aabb.min` snaps to the
/// nearest lower multiple of `voxel_size` so adjacent assets / tiles
/// with the same voxel size land on a shared world grid.
///
/// Use this when you have a natural mesh AABB and need to feed it to
/// [`voxelize_octree`]. Terrain doesn't need it — terrain tile AABBs
/// are pow2-cubic-aligned by construction.
pub fn pad_to_pow2_cubic(aabb: &Aabb, voxel_size: f32) -> Aabb {
    // Snap min down to the nearest voxel-grid line for shared-grid
    // alignment across assets.
    let snap = |v: f32| (v / voxel_size).floor() * voxel_size;
    let min = Vec3::new(snap(aabb.min.x), snap(aabb.min.y), snap(aabb.min.z));

    let raw_extent = aabb.max - min;
    let max_dim = raw_extent.x.max(raw_extent.y).max(raw_extent.z);
    let cells_needed = (max_dim / voxel_size).ceil().max(1.0) as u32;
    let pow2_cells = cells_needed.next_power_of_two();
    let cubic_extent = pow2_cells as f32 * voxel_size;
    Aabb {
        min,
        max: min + Vec3::splat(cubic_extent),
    }
}

/// Convenience: voxelize a sphere into a sparse octree. Pre-aligns the
/// AABB to satisfy `voxelize_octree`'s pow2-cubic contract.
pub fn voxelize_sphere_octree(
    center: Vec3,
    radius: f32,
    material_id: u16,
    voxel_size: f32,
    leaf_attr_pool: &mut LeafAttrPool,
    brick_pool: &mut BrickPool,
) -> Option<VoxelizeOctreeResult> {
    let padding = voxel_size * 2.0;
    let natural = Aabb {
        min: center - Vec3::splat(radius + padding),
        max: center + Vec3::splat(radius + padding),
    };
    let aabb = pad_to_pow2_cubic(&natural, voxel_size);

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

    voxelize_octree(sdf_fn, &aabb, voxel_size, leaf_attr_pool, brick_pool, 0)
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
    /// Halo cells sampled outside the nominal AABB on every face when
    /// [`voxelize_to_artifact`] was called with `halo > 0`. Each entry
    /// is `(cell_coord, leaf_attr_id)` in finest-grid integer units
    /// relative to `grid_origin`. `leaf_attr_id` values are
    /// worker-local indices into the same `leaf_attrs` vec above
    /// (interior cells use the `mesh_extract::CELL_INTERIOR` sentinel).
    /// Empty when `halo = 0`. Consumed by `build_mesh_sections_blob`
    /// to produce watertight tile seams; see `docs/TERRAIN.md` Phase 3.
    pub halo_cells: Vec<(IVec3, u32)>,
}

/// `voxelize_octree` against fresh private pools, packaged as a
/// [`BakeArtifact`]. This is the async-bake worker's entry point — it
/// runs entirely off the engine thread and produces a self-contained
/// result the main thread can integrate at its leisure.
pub fn voxelize_to_artifact<F>(
    sdf_fn: F,
    aabb: &Aabb,
    base_voxel_size: f32,
    halo: u32,
) -> Option<BakeArtifact>
where
    F: FnMut(&[Vec3]) -> Vec<(f32, u16, u16, u8, u32)>,
{
    use crate::brick_pool::BRICK_CELLS as BC;
    // Small initial capacities — pools grow on demand and this keeps
    // the allocator pressure low on small bakes.
    let mut leaf_attr_pool = LeafAttrPool::new(1024);
    let mut brick_pool = BrickPool::new(256);

    let result = voxelize_octree(
        sdf_fn,
        aabb,
        base_voxel_size,
        &mut leaf_attr_pool,
        &mut brick_pool,
        halo,
    )?;

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
        halo_cells: result.halo_cells,
    })
}

