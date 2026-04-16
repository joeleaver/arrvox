//! Per-voxel surface-shell emission.
//!
//! Given per-brick inside/outside flags and pre-computed face normals
//! from [`super::classify::process_brick`], this module:
//!
//! 1. Classifies every outside voxel as "shell" iff at least one of
//!    its 26 neighbours is inside. Produces a 1-voxel-thick outer
//!    shell the Laplacian smoother can operate on without ring
//!    artifacts.
//! 2. Emits each shell voxel as an octree leaf, storing the face
//!    normal, material, and colour that `process_brick` already
//!    computed. No gradient re-derivation happens here — face
//!    normals come straight from the nearest triangle.
//! 3. Collapses fully-inside sub-bricks into `INTERIOR_NODE` and
//!    leaves shell-less regions as EMPTY at the octree level.
//! 4. Fills solid-interior cells of shell bricks with `BRICK_INTERIOR`
//!    (cost-free — those slots are pre-allocated in the brick pool).
//!    Enables neighborhood kernels to distinguish "empty space" from
//!    "solid mesh interior" across brick boundaries.

use std::collections::{HashMap, HashSet};

use glam::UVec3;

use rkf_core::companion::ColorVoxel;
use rkf_core::voxel::VoxelSample;
use rkp_core::{SparseOctree, brick_pool, sparse_octree};

use super::classify::{BrickResult, BrickWork};

/// Output of [`emit_shell_leaves`] — everything the `.rkp` writer needs.
pub struct ShellOutput {
    /// Flat brick-cell storage, one `u32` per cell, grouped into
    /// `BRICK_DIM³` contiguous runs. Indices into this array are
    /// stored inside the octree via `sparse_octree::make_brick`.
    pub file_bricks: Vec<u32>,
    /// Per-leaf `VoxelSample`s — material + padding. Normals are
    /// stored separately (one `u32` per leaf) because `VoxelSample`
    /// doesn't have a normal slot in the opacity-octree format.
    pub voxel_data: Vec<VoxelSample>,
    /// Per-leaf albedo colours (same order as `voxel_data`).
    pub color_voxels: Vec<ColorVoxel>,
    /// Per-leaf octahedrally-packed normals (same order).
    pub normals_packed: Vec<u32>,
    /// `true` if any leaf carries non-black albedo data — signals the
    /// writer to emit the color payload.
    pub has_color: bool,
    /// Total number of shell voxels emitted.
    pub voxel_count: u32,
}

/// Core emission routine. Consumes the parallel per-brick sampling
/// results, classifies cells into shell / interior / empty, and
/// returns the flat arrays ready to serialize.
///
/// Mutates `octree` in place — caller supplies an empty octree sized
/// for the finest level and gets back a populated one.
pub fn emit_shell_leaves(
    octree: &mut SparseOctree,
    results: Vec<(BrickWork, BrickResult)>,
    interior_brick_set: HashSet<(u32, u32, u32)>,
    brick_depth: u8,
    octree_bricks: u32,
    depth: u8,
) -> ShellOutput {
    let octree_brick_dim: u32 = brick_pool::BRICK_DIM;
    let octree_brick_levels = brick_pool::BRICK_LEVELS;
    assert!(
        depth > octree_brick_levels,
        "octree depth ({depth}) must exceed BRICK_LEVELS ({octree_brick_levels}) to host bricks",
    );
    let octree_brick_depth = depth - octree_brick_levels;
    let subbricks_per_axis = 8 / octree_brick_dim;
    let brick_cells_u32 =
        (octree_brick_dim * octree_brick_dim * octree_brick_dim) as usize;

    // Build lookup tables: every surface brick we sampled goes in the
    // index so shell-emission can read its per-voxel inside/outside.
    //
    // The old `all_inside`-promotion shortcut (push the brick into
    // interior_brick_set and skip shell emission) has been removed:
    // `process_brick`'s per-voxel sign test is fragile at concavities
    // and thin features, and a single false-positive flip on every
    // voxel of an 8³ brick silently discards 512 shell voxels and
    // produces a brick-sized INTERIOR-node cube in the render.
    // classify_bricks' brick-center winding test is still trusted to
    // populate interior_brick_set — that's robust; per-voxel
    // aggregation-to-interior is not.
    let mut brick_result_index: HashMap<(u32, u32, u32), usize> =
        HashMap::with_capacity(results.len());
    for (i, (w, _result)) in results.iter().enumerate() {
        brick_result_index.insert((w.bx, w.by, w.bz), i);
    }

    // Inside/outside lookup across brick boundaries. Surface bricks
    // have per-voxel `is_inside` data from `process_brick`; interior
    // bricks are inside by definition; out-of-bounds / empty bricks
    // are outside. Used by the 26-neighbour shell-classification
    // scan below.
    let inside_at = |gx: i64, gy: i64, gz: i64| -> bool {
        if gx < 0 || gy < 0 || gz < 0 {
            return false;
        }
        let (gx, gy, gz) = (gx as u32, gy as u32, gz as u32);
        let bx = gx / 8;
        let by = gy / 8;
        let bz = gz / 8;
        if bx >= octree_bricks || by >= octree_bricks || bz >= octree_bricks {
            return false;
        }
        if let Some(&idx) = brick_result_index.get(&(bx, by, bz)) {
            let lx = gx % 8;
            let ly = gy % 8;
            let lz = gz % 8;
            results[idx].1.is_inside[(lx + ly * 8 + lz * 64) as usize]
        } else {
            interior_brick_set.contains(&(bx, by, bz))
        }
    };

    // Insert fully-interior bricks as single octree-level INTERIOR
    // nodes. classify_bricks' brick-center inside-test is now
    // ray-cast-based (topologically robust to non-watertight /
    // self-intersecting meshes), so we can trust the interior list
    // again — the scale-dependent `outside classified inside`
    // bricks that forced us to disable this write have been fixed
    // at their source.
    for &(bx, by, bz) in &interior_brick_set {
        octree.set_at_level(
            UVec3::new(bx * 8, by * 8, bz * 8),
            brick_depth,
            sparse_octree::INTERIOR_NODE,
        );
    }

    let mut file_bricks: Vec<u32> = Vec::new();
    let mut voxel_data: Vec<VoxelSample> = Vec::new();
    let mut color_voxels: Vec<ColorVoxel> = Vec::new();
    let mut normals_packed: Vec<u32> = Vec::new();
    let mut has_color = false;
    let mut voxel_count = 0u32;

    // Emit shell leaves per surface brick, one 4³ sub-brick at a time.
    for (w, result) in &results {
        if result.all_inside {
            continue; // handled above as INTERIOR_NODE
        }

        for sbz in 0..subbricks_per_axis {
            for sby in 0..subbricks_per_axis {
                for sbx in 0..subbricks_per_axis {
                    emit_subbrick(
                        w,
                        result,
                        (sbx, sby, sbz),
                        octree_brick_dim,
                        octree_brick_depth,
                        brick_cells_u32,
                        octree,
                        &inside_at,
                        &mut file_bricks,
                        &mut voxel_data,
                        &mut color_voxels,
                        &mut normals_packed,
                        &mut has_color,
                        &mut voxel_count,
                    );
                }
            }
        }
    }

    ShellOutput {
        file_bricks,
        voxel_data,
        color_voxels,
        normals_packed,
        has_color,
        voxel_count,
    }
}

/// One 4³ sub-brick of an 8³ surface region. Emits the sub-brick to
/// the file_bricks pool + octree if any shell cells are present;
/// collapses to `INTERIOR_NODE` if fully inside; emits nothing
/// otherwise (stays as EMPTY in the octree).
#[allow(clippy::too_many_arguments)]
fn emit_subbrick(
    w: &BrickWork,
    result: &BrickResult,
    (sbx, sby, sbz): (u32, u32, u32),
    octree_brick_dim: u32,
    octree_brick_depth: u8,
    brick_cells_u32: usize,
    octree: &mut SparseOctree,
    inside_at: &impl Fn(i64, i64, i64) -> bool,
    file_bricks: &mut Vec<u32>,
    voxel_data: &mut Vec<VoxelSample>,
    color_voxels: &mut Vec<ColorVoxel>,
    normals_packed: &mut Vec<u32>,
    has_color: &mut bool,
    voxel_count: &mut u32,
) {
    let sub_origin_x = sbx * octree_brick_dim;
    let sub_origin_y = sby * octree_brick_dim;
    let sub_origin_z = sbz * octree_brick_dim;

    struct ShellEntry {
        cell_flat: u32,
        flat8: usize,
    }

    let mut all_interior = true;
    let mut shell_entries: Vec<ShellEntry> = Vec::new();

    for cz in 0..octree_brick_dim {
        for cy in 0..octree_brick_dim {
            for cx in 0..octree_brick_dim {
                let vx = sub_origin_x + cx;
                let vy = sub_origin_y + cy;
                let vz = sub_origin_z + cz;
                let flat8 = (vx + vy * 8 + vz * 64) as usize;
                if !result.is_inside[flat8] {
                    all_interior = false;
                    let gx = w.bx * 8 + vx;
                    let gy = w.by * 8 + vy;
                    let gz = w.bz * 8 + vz;
                    if any_inside_within(inside_at, gx, gy, gz, 2) {
                        let cell_flat = cx
                            + cy * octree_brick_dim
                            + cz * octree_brick_dim * octree_brick_dim;
                        shell_entries.push(ShellEntry { cell_flat, flat8 });
                    }
                }
            }
        }
    }

    let sub_origin_coord = UVec3::new(
        w.bx * 8 + sub_origin_x,
        w.by * 8 + sub_origin_y,
        w.bz * 8 + sub_origin_z,
    );

    if all_interior {
        // Previously: `set_at_level(..., INTERIOR_NODE)` — promoted any
        // 4³ subbrick whose per-voxel is_inside flags were all true to
        // an INTERIOR node. Removed for the same reason as the brick-
        // level all_inside promotion: process_brick's sign test is
        // fragile near concavities and a whole-subbrick flip turns
        // into a rendered cube. Leaving the subbrick EMPTY is safe:
        // genuinely-interior subbricks sit behind the shell voxels of
        // an exterior-facing subbrick, so ray coverage is preserved.
        let _ = (sub_origin_coord, octree_brick_depth);
        return;
    }
    if shell_entries.is_empty() {
        return; // stays as EMPTY
    }

    // Allocate a new brick in the file-local pool and populate its cells.
    let brick_id = (file_bricks.len() / brick_cells_u32) as u32;
    file_bricks.extend(std::iter::repeat_n(brick_pool::BRICK_EMPTY, brick_cells_u32));
    let brick_base = brick_id as usize * brick_cells_u32;

    // Mark inside cells of this sub-brick as `BRICK_INTERIOR`. Zero
    // memory cost (slots are pre-allocated regardless of content).
    // Enables neighbourhood kernels to see solid mass behind the
    // shell when resolving cross-brick queries. Now safe — the
    // per-voxel inside/outside classifier in `process_brick` uses
    // winding-number alone (no concavity-induced false positives).
    for cz in 0..octree_brick_dim {
        for cy in 0..octree_brick_dim {
            for cx in 0..octree_brick_dim {
                let vx = sub_origin_x + cx;
                let vy = sub_origin_y + cy;
                let vz = sub_origin_z + cz;
                let flat8 = (vx + vy * 8 + vz * 64) as usize;
                if result.is_inside[flat8] {
                    let cell_flat =
                        cx + cy * octree_brick_dim + cz * octree_brick_dim * octree_brick_dim;
                    file_bricks[brick_base + cell_flat as usize] = brick_pool::BRICK_INTERIOR;
                }
            }
        }
    }

    // Emit each shell entry. Face normal + material + colour all
    // come straight from `BrickResult` — pre-computed from the
    // nearest-triangle BVH query at sample time. No gradient
    // reconstruction here, so brick-boundary precision issues no
    // longer factor in.
    for e in &shell_entries {
        let normal_oct = result.face_normals[e.flat8];
        let mat_id = result.material_ids[e.flat8];
        let cv = result.color_brick.data[e.flat8];
        if cv.intensity() > 0 {
            *has_color = true;
        }

        let slot = *voxel_count;
        voxel_data.push(VoxelSample::new(0.0, mat_id, 0));
        normals_packed.push(normal_oct);
        color_voxels.push(cv);
        *voxel_count += 1;

        file_bricks[brick_base + e.cell_flat as usize] = slot;
    }

    octree.set_at_level(sub_origin_coord, octree_brick_depth, sparse_octree::make_brick(brick_id));
}

/// `true` iff any voxel within Chebyshev distance `radius` of
/// `(gx, gy, gz)` is inside the mesh. `radius = 1` gives the 26-
/// neighbourhood used for a 1-voxel-thick shell; `radius = 2` gives
/// the 5³-1 = 124-neighbourhood used for a 2-voxel-thick shell.
///
/// A thicker shell closes the glancing-ray leaks where the march can
/// slip between 1-cell-thick shell voxels and reach an INTERIOR
/// neighbour face (rendered by the shader as a flat-gray cube). With
/// 2-voxel thickness every ray crossing the surface at any angle is
/// guaranteed to traverse at least one shell cell before reaching
/// interior bulk.
fn any_inside_within(
    inside_at: &impl Fn(i64, i64, i64) -> bool,
    gx: u32,
    gy: u32,
    gz: u32,
    radius: i64,
) -> bool {
    for dz in -radius..=radius {
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx == 0 && dy == 0 && dz == 0 {
                    continue;
                }
                if inside_at(gx as i64 + dx, gy as i64 + dy, gz as i64 + dz) {
                    return true;
                }
            }
        }
    }
    false
}
