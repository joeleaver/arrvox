//! CPU leaf extraction for the splat-rasterizer prototype.
//!
//! Walks an asset's brick-terminated octree and emits one [`SplatVertex`]
//! per occupied surface voxel. The vertex carries world-space center,
//! disc radius, and the absolute `leaf_attr_pool` slot the GPU shader
//! reads to recover the prefiltered normal + material — the same slot
//! `fetch_leaf_attr_for` consumes in `octree_march.wesl`.
//!
//! No dependency on the runtime scene manager; takes raw octree + brick
//! data so the prototype's integration test can call it directly with
//! the blobs that come out of `rkp_core::asset_file`.

use glam::{UVec3, Vec3};
use rkp_core::brick_pool::{BRICK_CELLS, BRICK_DIM, BRICK_EMPTY, BRICK_INTERIOR};
use rkp_core::sparse_octree::{
    brick_id, is_branch, is_brick, is_leaf, leaf_slot, EMPTY_NODE, INTERIOR_NODE,
};

/// One surface voxel as a splat-rasterizer input.
///
/// 32 B, `repr(C)` so it can be `bytemuck`-cast directly into a vertex
/// buffer. `_pad` keeps the struct's size at 32 B for clean `array_stride`
/// arithmetic in the wgpu pipeline layout.
///
/// Positions are **object-local** — the per-instance world matrix is
/// applied in the vertex shader from the per-instance uniform. One
/// `Vec<SplatVertex>` per asset; reused across every scene-instance of
/// that asset. The earlier prototype baked `asset_world` into the
/// positions; that's been removed because it forced re-extraction on
/// every transform change in a multi-instance scene.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SplatVertex {
    pub local_pos: [f32; 3],
    /// Disc radius in object-local units. Sized so adjacent voxel discs
    /// overlap at any voxel resolution — a 0.6× factor empirically
    /// covers the √3/2 worst-case diagonal between cell centers without
    /// blowing silhouettes way past the cell.
    pub radius: f32,
    /// Absolute index into the global `leaf_attr_pool` (matches
    /// `fetch_leaf_attr_for`'s `leaf_slot` argument). The fragment
    /// shader does `leaf_attr_pool[leaf_attr_id]` to recover the
    /// prefiltered oct-normal + material packing.
    pub leaf_attr_id: u32,
    pub _pad: [u32; 3],
}

const _: () = assert!(std::mem::size_of::<SplatVertex>() == 32);

/// Disc-radius factor applied to the cell size. Tunable; 0.6 covers the
/// inter-cell-center diagonal (√3 / 2 ≈ 0.866) with some slack and keeps
/// silhouette overshoot modest. If silhouettes look ragged at coarse
/// voxel sizes during visual review, raise toward 0.7–0.75.
pub const DISC_RADIUS_FACTOR: f32 = 0.6;

/// Walk a brick-terminated octree and emit one [`SplatVertex`] per
/// occupied cell.
///
/// `radius_factor` scales the disc radius by `cell_size`. The default
/// for ad-hoc callers is [`DISC_RADIUS_FACTOR`] (0.6); silhouettes can
/// step at glancing angles when this is too tight, so the prototype
/// test exposes it via env var. 0.6–0.9 is the sensible range.
///
/// * `octree_nodes` — raw `nodes` slice from the asset's `SparseOctree`
///   (or directly from `rkp_core::asset_file::read_rkp_octree`). Root at
///   index 0; branch nodes hold a child-array offset, leaf-likes are
///   tagged via [`LEAF_BIT`] / [`BRICK_BIT`].
/// * `octree_depth` — the asset's `depth` field. Determines the size of
///   each level's cell coverage.
/// * `base_voxel_size` — finest cell edge length (object-local units;
///   metres in this engine).
/// * `grid_origin` — object-local position of the octree extent's lo
///   corner. Same field that's stored on `RkpGpuAsset` and that the
///   GPU shader uses for `oc_origin = local_origin - asset.grid_origin`.
/// * `brick_cells` — flat brick storage; `brick_id * BRICK_CELLS + flat`
///   indexes into it. For runtime use this is the SCENE-global
///   `brick_pool.as_slice()` after `load_asset_from_disk` has remapped
///   the asset's brick ids and slot indices to scene-global values.
///
/// Returns one vertex per non-empty, non-interior cell, in
/// **object-local** coordinates. Capacity is pre-reserved to the
/// maximum possible (every brick fully populated) so the inner loop
/// never reallocates.
pub fn extract_splats(
    octree_nodes: &[u32],
    octree_depth: u8,
    base_voxel_size: f32,
    grid_origin: Vec3,
    brick_cells: &[u32],
) -> Vec<SplatVertex> {
    extract_splats_with_radius(
        octree_nodes,
        octree_depth,
        base_voxel_size,
        grid_origin,
        brick_cells,
        DISC_RADIUS_FACTOR,
    )
}

/// Like [`extract_splats`] but takes an explicit `radius_factor` so
/// callers can sweep the silhouette-overlap-vs-overshoot trade-off
/// without recompiling.
pub fn extract_splats_with_radius(
    octree_nodes: &[u32],
    octree_depth: u8,
    base_voxel_size: f32,
    grid_origin: Vec3,
    brick_cells: &[u32],
    radius_factor: f32,
) -> Vec<SplatVertex> {
    // Cap pre-allocation at a few million to avoid wild OOM on huge
    // assets; the Vec will still grow if the cap is too low.
    let max_capacity = (brick_cells.len()).min(8_000_000);
    let mut out = Vec::with_capacity(max_capacity);
    if octree_nodes.is_empty() {
        return out;
    }
    walk(
        0,
        UVec3::ZERO,
        0,
        octree_depth,
        octree_nodes,
        brick_cells,
        base_voxel_size,
        grid_origin,
        radius_factor,
        &mut out,
    );
    out
}

#[allow(clippy::too_many_arguments)]
fn walk(
    node_idx: usize,
    origin_voxels: UVec3,
    level: u8,
    max_depth: u8,
    nodes: &[u32],
    bricks: &[u32],
    vs: f32,
    grid_origin: Vec3,
    radius_factor: f32,
    out: &mut Vec<SplatVertex>,
) {
    let node = nodes[node_idx];
    if node == EMPTY_NODE || node == INTERIOR_NODE {
        return;
    }
    if is_leaf(node) {
        // A LEAF node terminates the descent at this level. The cell
        // covers `1 << (max_depth - level)` finest voxels per axis —
        // emit one disc sized to the full cell.
        let cell_voxels = 1u32 << (max_depth - level);
        let cell_size = vs * cell_voxels as f32;
        let center_local = grid_origin
            + Vec3::new(
                (origin_voxels.x as f32 + cell_voxels as f32 * 0.5) * vs,
                (origin_voxels.y as f32 + cell_voxels as f32 * 0.5) * vs,
                (origin_voxels.z as f32 + cell_voxels as f32 * 0.5) * vs,
            );
        out.push(SplatVertex {
            local_pos: center_local.to_array(),
            radius: cell_size * radius_factor,
            leaf_attr_id: leaf_slot(node),
            _pad: [0; 3],
        });
        return;
    }
    if is_brick(node) {
        // BRICK: walk the 4³ cell array. Each cell either holds an
        // absolute leaf_attr_pool slot or a sentinel (EMPTY / INTERIOR).
        let bid = brick_id(node);
        let base = (bid * BRICK_CELLS) as usize;
        for cz in 0..BRICK_DIM {
            for cy in 0..BRICK_DIM {
                for cx in 0..BRICK_DIM {
                    let flat =
                        (cx + cy * BRICK_DIM + cz * BRICK_DIM * BRICK_DIM) as usize;
                    let c = bricks[base + flat];
                    if c == BRICK_EMPTY || c == BRICK_INTERIOR {
                        continue;
                    }
                    let cell_voxel = origin_voxels + UVec3::new(cx, cy, cz);
                    let center_local = grid_origin
                        + Vec3::new(
                            (cell_voxel.x as f32 + 0.5) * vs,
                            (cell_voxel.y as f32 + 0.5) * vs,
                            (cell_voxel.z as f32 + 0.5) * vs,
                        );
                    out.push(SplatVertex {
                        local_pos: center_local.to_array(),
                        radius: vs * radius_factor,
                        leaf_attr_id: c,
                        _pad: [0; 3],
                    });
                }
            }
        }
        return;
    }
    if is_branch(node) {
        let children_offset = node as usize;
        let half = 1u32 << (max_depth - level - 1);
        for octant in 0u32..8 {
            let dx = octant & 1;
            let dy = (octant >> 1) & 1;
            let dz = (octant >> 2) & 1;
            let child_origin = UVec3::new(
                origin_voxels.x + dx * half,
                origin_voxels.y + dy * half,
                origin_voxels.z + dz * half,
            );
            walk(
                children_offset + octant as usize,
                child_origin,
                level + 1,
                max_depth,
                nodes,
                bricks,
                vs,
                grid_origin,
                radius_factor,
                out,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rkp_core::sparse_octree::{make_brick, make_leaf};

    #[test]
    fn splat_vertex_size_is_32() {
        assert_eq!(std::mem::size_of::<SplatVertex>(), 32);
    }

    #[test]
    fn empty_octree_yields_nothing() {
        let nodes = vec![EMPTY_NODE];
        let splats = extract_splats(&nodes, 4, 0.001, Vec3::ZERO, &[]);
        assert!(splats.is_empty());
    }

    #[test]
    fn single_leaf_at_root_emits_one_full_extent_splat() {
        // depth=0 → root is the only node; one LEAF covering the full
        // 1-voxel extent.
        let nodes = vec![make_leaf(42)];
        let vs = 0.5;
        let splats = extract_splats(&nodes, 0, vs, Vec3::new(1.0, 2.0, 3.0), &[]);
        assert_eq!(splats.len(), 1);
        let s = splats[0];
        assert_eq!(s.leaf_attr_id, 42);
        // grid_origin + half-extent. depth 0 → cell_voxels = 1, so center
        // is at grid_origin + vs * 0.5 on each axis (object-local coords).
        assert_eq!(s.local_pos, [1.0 + vs * 0.5, 2.0 + vs * 0.5, 3.0 + vs * 0.5]);
        // Disc radius = cell_size × factor.
        assert!((s.radius - vs * DISC_RADIUS_FACTOR).abs() < 1e-6);
    }

    #[test]
    fn single_brick_with_two_filled_cells() {
        // Root is a BRICK referencing brick 0. Brick storage holds 64
        // cells; we mark exactly two as filled (slots 100 and 101) and
        // verify both come back at the right object-local positions.
        let nodes = vec![make_brick(0)];
        let mut bricks = vec![BRICK_EMPTY; BRICK_CELLS as usize];
        // Cell (0,0,0) = leaf slot 100.
        bricks[0] = 100;
        // Cell (3,3,3) = leaf slot 101 (opposite corner).
        let last = (BRICK_DIM - 1
            + (BRICK_DIM - 1) * BRICK_DIM
            + (BRICK_DIM - 1) * BRICK_DIM * BRICK_DIM) as usize;
        bricks[last] = 101;

        // depth 2 = brick at root spans 1<<2 = 4 finest voxels per axis.
        let vs = 1.0;
        let splats = extract_splats(&nodes, 2, vs, Vec3::ZERO, &bricks);
        assert_eq!(splats.len(), 2);

        // The (0,0,0) cell.
        let a = splats.iter().find(|s| s.leaf_attr_id == 100).unwrap();
        assert_eq!(a.local_pos, [0.5, 0.5, 0.5]);
        assert!((a.radius - vs * DISC_RADIUS_FACTOR).abs() < 1e-6);

        // The (3,3,3) cell — center at (3.5, 3.5, 3.5) for vs=1.
        let b = splats.iter().find(|s| s.leaf_attr_id == 101).unwrap();
        assert_eq!(b.local_pos, [3.5, 3.5, 3.5]);
    }

    #[test]
    fn empty_and_interior_cells_skipped_in_brick() {
        let nodes = vec![make_brick(0)];
        let mut bricks = vec![BRICK_EMPTY; BRICK_CELLS as usize];
        bricks[0] = 50;
        bricks[1] = BRICK_EMPTY;
        bricks[2] = BRICK_INTERIOR;
        bricks[3] = 51;

        let splats = extract_splats(&nodes, 2, 1.0, Vec3::ZERO, &bricks);
        assert_eq!(splats.len(), 2);
        let ids: std::collections::HashSet<u32> =
            splats.iter().map(|s| s.leaf_attr_id).collect();
        assert!(ids.contains(&50));
        assert!(ids.contains(&51));
    }
}
