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

use rkp_core::companion::{BoneVoxel, ColorVoxel};
use rkp_core::voxel::VoxelSample;
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
    /// Per-leaf skinning weights (same order). Zero-filled for
    /// unskinned imports; only emitted to `.rkp` when
    /// [`Self::has_bones`] is set.
    pub bone_voxels: Vec<BoneVoxel>,
    /// File-local brick origin per brick, in finest-voxel grid units.
    /// `brick_origins.len() == file_bricks.len() / BRICK_CELLS`. Index
    /// is the brick's file-local id (before any load-time
    /// `scene_brick_offset` shift). Phase-3 skin-deform reads these to
    /// forward-skin each brick's voxels without walking the octree.
    pub brick_origins: Vec<[u32; 3]>,
    /// Per-bone rest-pose AABB in object-local space (the same
    /// `(grid_coord * voxel_size)` frame the march shader uses). Index
    /// is the bone id; length = `1 + max_bone_index_seen`. Empty slots
    /// are zero-extent sentinels at the origin — harmless when unioned
    /// with an identity bone matrix during the per-frame deformed-AABB
    /// computation.
    pub rest_bone_aabbs: Vec<[f32; 6]>,
    /// `true` if any leaf carries non-black albedo data — signals the
    /// writer to emit the color payload.
    pub has_color: bool,
    /// `true` if any leaf carries non-zero bone weights — signals the
    /// writer to emit the bones payload.
    pub has_bones: bool,
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
    mut interior_brick_set: HashSet<(u32, u32, u32)>,
    brick_depth: u8,
    octree_bricks: u32,
    depth: u8,
    voxel_size: f32,
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

    // Build lookup tables: surface bricks → per-voxel inside/outside,
    // plus the set of solid-interior bricks (including any surface
    // bricks that turned out all-inside after per-voxel sampling).
    //
    // The all_inside promotion is safe now that `process_brick` uses
    // ray-cast parity for inside/outside — a whole 8³ brick landing
    // all-inside corresponds to a brick entirely inside the mesh,
    // not to a classifier flip, so promoting it to an octree-level
    // INTERIOR node is both correct and saves emitting 512 wasted
    // shell slots.
    let mut brick_result_index: HashMap<(u32, u32, u32), usize> =
        HashMap::with_capacity(results.len());
    for (i, (w, result)) in results.iter().enumerate() {
        if result.all_inside {
            interior_brick_set.insert((w.bx, w.by, w.bz));
        } else {
            brick_result_index.insert((w.bx, w.by, w.bz), i);
        }
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
    let mut bone_voxels: Vec<BoneVoxel> = Vec::new();
    let mut brick_origins: Vec<[u32; 3]> = Vec::new();
    // Per-bone rest AABB accumulator — grown on demand as bone indices
    // appear in the shell. `[INF, INF, INF, -INF, -INF, -INF]` sentinels
    // flag "no voxel has claimed this bone yet"; writer collapses those
    // back to zero-extent before serialising.
    let mut rest_bone_aabbs: Vec<[f32; 6]> = Vec::new();
    let mut has_color = false;
    let mut has_bones = false;
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
                        voxel_size,
                        octree,
                        &inside_at,
                        &mut file_bricks,
                        &mut voxel_data,
                        &mut color_voxels,
                        &mut normals_packed,
                        &mut bone_voxels,
                        &mut brick_origins,
                        &mut rest_bone_aabbs,
                        &mut has_color,
                        &mut has_bones,
                        &mut voxel_count,
                    );
                }
            }
        }
    }

    // Collapse unclaimed bone slots (sentinel +INF/-INF) to zero AABBs
    // so the on-disk representation is well-formed.
    for aabb in rest_bone_aabbs.iter_mut() {
        if aabb[0] > aabb[3] { *aabb = [0.0; 6]; }
    }

    ShellOutput {
        file_bricks,
        voxel_data,
        color_voxels,
        normals_packed,
        bone_voxels,
        brick_origins,
        rest_bone_aabbs,
        has_color,
        has_bones,
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
    voxel_size: f32,
    octree: &mut SparseOctree,
    inside_at: &impl Fn(i64, i64, i64) -> bool,
    file_bricks: &mut Vec<u32>,
    voxel_data: &mut Vec<VoxelSample>,
    color_voxels: &mut Vec<ColorVoxel>,
    normals_packed: &mut Vec<u32>,
    bone_voxels: &mut Vec<BoneVoxel>,
    brick_origins: &mut Vec<[u32; 3]>,
    rest_bone_aabbs: &mut Vec<[f32; 6]>,
    has_color: &mut bool,
    has_bones: &mut bool,
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
                    if any_inside_26_neighbor(inside_at, gx, gy, gz) {
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
        octree.set_at_level(sub_origin_coord, octree_brick_depth, sparse_octree::INTERIOR_NODE);
        return;
    }
    if shell_entries.is_empty() {
        return; // stays as EMPTY
    }

    // Allocate a new brick in the file-local pool and populate its cells.
    let brick_id = (file_bricks.len() / brick_cells_u32) as u32;
    file_bricks.extend(std::iter::repeat_n(brick_pool::BRICK_EMPTY, brick_cells_u32));
    let brick_base = brick_id as usize * brick_cells_u32;
    // Record the brick's origin in finest-voxel grid units (matches the
    // octree's finest-level indexing). The scatter pass reads this at
    // runtime to derive per-cell rest positions.
    brick_origins.push([sub_origin_coord.x, sub_origin_coord.y, sub_origin_coord.z]);

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
        let bv = result.bone_voxels[e.flat8];
        if cv.intensity() > 0 {
            *has_color = true;
        }
        if result.has_bones && (0..4).any(|i| bv.bone_weight(i) > 0) {
            *has_bones = true;

            // Voxel grid coord inside the mesh brick.
            let vx = (e.flat8 % 8) as u32;
            let vy = ((e.flat8 / 8) % 8) as u32;
            let vz = (e.flat8 / 64) as u32;
            let gx = w.bx * 8 + vx;
            let gy = w.by * 8 + vy;
            let gz = w.bz * 8 + vz;
            // Center of the voxel in object-local space — matches the
            // space the scatter shader forward-skins from.
            let cx = (gx as f32 + 0.5) * voxel_size;
            let cy = (gy as f32 + 0.5) * voxel_size;
            let cz = (gz as f32 + 0.5) * voxel_size;

            // Dominant bone = slot with the max weight. Accumulate the
            // voxel centre into that bone's rest AABB. LBS during the
            // scatter blends across bones, but the AABB only needs to
            // contain voxels that predominantly move with each bone —
            // good enough for sizing the deformed grid with a small
            // inflation margin at runtime.
            let mut dom_slot = 0usize;
            let mut dom_w = bv.bone_weight(0);
            for s in 1..4 {
                if bv.bone_weight(s) > dom_w {
                    dom_w = bv.bone_weight(s);
                    dom_slot = s;
                }
            }
            let dom_bone = bv.bone_index(dom_slot) as usize;
            if rest_bone_aabbs.len() <= dom_bone {
                // Sentinel: +INF min / -INF max flags "not yet
                // accumulated". Collapsed to zero-extent by the outer
                // caller before serialise.
                rest_bone_aabbs.resize(dom_bone + 1, [
                    f32::INFINITY, f32::INFINITY, f32::INFINITY,
                    f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY,
                ]);
            }
            let aabb = &mut rest_bone_aabbs[dom_bone];
            aabb[0] = aabb[0].min(cx);
            aabb[1] = aabb[1].min(cy);
            aabb[2] = aabb[2].min(cz);
            aabb[3] = aabb[3].max(cx);
            aabb[4] = aabb[4].max(cy);
            aabb[5] = aabb[5].max(cz);
        }

        let slot = *voxel_count;
        voxel_data.push(VoxelSample::new(0.0, mat_id, 0));
        normals_packed.push(normal_oct);
        color_voxels.push(cv);
        bone_voxels.push(bv);
        *voxel_count += 1;

        file_bricks[brick_base + e.cell_flat as usize] = slot;
    }

    octree.set_at_level(sub_origin_coord, octree_brick_depth, sparse_octree::make_brick(brick_id));
}

/// `true` iff any of the 26 neighbours of `(gx, gy, gz)` is inside
/// the mesh. Used to identify the 1-voxel-thick outer shell.
fn any_inside_26_neighbor(
    inside_at: &impl Fn(i64, i64, i64) -> bool,
    gx: u32,
    gy: u32,
    gz: u32,
) -> bool {
    for dz in -1i64..=1 {
        for dy in -1i64..=1 {
            for dx in -1i64..=1 {
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
