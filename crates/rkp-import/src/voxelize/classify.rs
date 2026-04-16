//! Brick-level narrow-band classification.
//!
//! For each 8³ brick region in the grid we check its centre against
//! the BVH. Bricks within the narrow band become *surface* work
//! (scheduled for parallel per-voxel sampling); bricks outside the
//! band are classified as solid-interior or empty by winding-number
//! sign.
//!
//! Per-voxel sampling happens in [`process_brick`]: signed distance
//! (unsigned distance × winding sign), nearest-triangle material,
//! optional per-voxel albedo. Bone-weight sampling is deliberately
//! NOT performed here — the `.rkp` v4 file format has no per-voxel
//! bone slot yet, so the work would be discarded. Skeleton-level
//! data still ships via the `.rkskel` sidecar.

use glam::Vec3;

use rkf_core::Aabb;
use rkf_core::companion::{ColorBrick, ColorVoxel};
use rkf_core::constants::BRICK_DIM;

use crate::bvh::TriangleBvh;
use crate::config::ImportConfig;
use crate::mesh::MeshData;
use crate::sample::texture::sample_texture_at_triangle;

/// Tier-based auto voxel-size picker. Chooses the coarsest tier that
/// still resolves the longest axis with at least 8 bricks. Matches
/// the editor's tier table so auto + manual imports stay in sync.
pub fn auto_voxel_size(aabb: &Aabb) -> f32 {
    const TIERS: [f32; 4] = [0.005, 0.02, 0.08, 0.32];
    let extent = aabb.max - aabb.min;
    let longest = extent.x.max(extent.y).max(extent.z);
    for &vs in &TIERS {
        let brick_world = vs * BRICK_DIM as f32;
        let bricks_on_longest = (longest / brick_world).ceil() as u32;
        if bricks_on_longest >= 8 {
            return vs;
        }
    }
    TIERS[0]
}

/// A brick scheduled for per-voxel sampling. `brick_min` is the
/// brick's minimum corner in world coordinates.
pub struct BrickWork {
    /// Brick coordinate (x) in the grid.
    pub bx: u32,
    /// Brick coordinate (y) in the grid.
    pub by: u32,
    /// Brick coordinate (z) in the grid.
    pub bz: u32,
    /// World-space minimum corner of this brick.
    pub brick_min: Vec3,
}

/// Output of [`process_brick`] — per-voxel inside/outside flag,
/// material, colour, and outward-pointing face normal for every
/// voxel in the brick's 8³ grid.
///
/// No per-voxel signed distance: mesh-to-voxel voxelization only
/// needs the inside/outside bit for shell classification and the
/// nearest-triangle face normal for shading. The SDF magnitude
/// middleware that used to sit here was inherited from the
/// SDF-primitive voxelization pipeline and isn't load-bearing for
/// meshes — removing it eliminated the whole class of brick-boundary
/// gradient bugs we were chasing.
pub struct BrickResult {
    /// RGB albedo per voxel (row-major, 512 entries). Set only for
    /// voxels whose nearest triangle had a texture hit.
    pub color_brick: ColorBrick,
    /// Nearest-triangle material ID per voxel (row-major, 512 entries).
    pub material_ids: [u16; 512],
    /// `true` if this voxel is inside the mesh (per the face-normal
    /// sign test near the surface, or generalized winding number far
    /// from the surface). The shell classifier reads this flag via
    /// `inside_at` during 26-neighbour scans.
    pub is_inside: [bool; 512],
    /// Nearest-triangle face normal, octahedrally packed and already
    /// oriented outward (CCW/CW winding handled by `winding_sign`).
    /// Only meaningful for voxels where `is_inside[flat] == false`
    /// (the shell candidates); stored as `0` for inside voxels where
    /// it's never read.
    pub face_normals: [u32; 512],
    /// `true` if *every* voxel in the brick is inside the mesh —
    /// triggers the surface-brick → interior-brick promotion in
    /// [`super::shell`] so we don't emit wasted empty shell data.
    pub all_inside: bool,
}

/// Flat voxel index within a brick (matches `rkf-core` convention).
pub fn voxel_index(x: u8, y: u8, z: u8) -> u32 {
    x as u32 + y as u32 * 8 + z as u32 * 64
}

/// Sample material, colour, face-normal, and inside/outside at every
/// voxel of one 8³ brick. Caller is expected to have already
/// narrow-band-filtered this brick (every voxel is worth sampling).
///
/// Inside/outside uses axis-aligned ray-cast parity
/// ([`TriangleBvh::is_inside_raycast`]): 3-ray majority vote,
/// topologically robust to self-intersections / duplicated triangles
/// / non-manifold patches in real scan and CAD meshes.
pub fn process_brick(
    mesh: &MeshData,
    bvh: &TriangleBvh,
    brick_min: Vec3,
    voxel_size: f32,
    config: &ImportConfig,
) -> BrickResult {
    let half_voxel = voxel_size * 0.5;
    let mut color_brick = ColorBrick::default();
    let mut material_ids = [0u16; 512];
    let mut is_inside_buf = [false; 512];
    let mut face_normals = [0u32; 512];
    let mut all_inside = true;

    for vz in 0..BRICK_DIM {
        for vy in 0..BRICK_DIM {
            for vx in 0..BRICK_DIM {
                let pos = brick_min
                    + Vec3::new(
                        vx as f32 * voxel_size + half_voxel,
                        vy as f32 * voxel_size + half_voxel,
                        vz as f32 * voxel_size + half_voxel,
                    );

                let nearest = bvh.nearest(pos);

                let is_inside = bvh.is_inside_raycast(pos);

                let flat = voxel_index(vx as u8, vy as u8, vz as u8) as usize;
                is_inside_buf[flat] = is_inside;
                if !is_inside {
                    all_inside = false;
                }

                material_ids[flat] = if let Some(id) = config.material_id_override {
                    id
                } else if nearest.triangle_index < mesh.material_indices.len() {
                    mesh.material_indices[nearest.triangle_index] as u16
                } else {
                    0
                };

                if config.import_colors {
                    if let Some(color) = sample_texture_at_triangle(
                        mesh,
                        nearest.triangle_index,
                        &nearest.barycentric,
                    ) {
                        color_brick.set(
                            vx,
                            vy,
                            vz,
                            ColorVoxel::new(color.r, color.g, color.b, 255),
                        );
                    }
                }

                // Stored per-voxel normal: gradient of the mesh's
                // unsigned distance field, sampled via six BVH
                // `nearest` queries at `±voxel_size` offsets from
                // the voxel centre. This is the standard OpenVDB /
                // level-set voxelization approach (Museth 2006) for
                // smooth surface normals on voxelized meshes.
                //
                // Why this gives smooth normals where face-normal
                // and Gouraud-interpolation did not:
                //
                // * The unsigned distance function `d(x)` is
                //   C¹-continuous everywhere except the medial axis
                //   (a thin set). Its gradient `∇d` therefore
                //   varies smoothly across the mesh surface,
                //   including across triangle edges and vertex
                //   Voronoi boundaries — the exact places where
                //   piecewise face-normal / barycentric-
                //   interpolation go flat and produce the "cube
                //   face" artifacts we chased.
                // * For an outside voxel (d > 0), `∇d` points
                //   outward (away from the surface) — by
                //   construction the correct shading normal.
                //
                // No per-brick SDF cache: taps are fresh BVH
                // queries, so there's no cache-miss / brick-
                // boundary precision problem. Six extra BVH queries
                // per shell-candidate voxel; rayon-parallelised
                // inside the caller so it parallelises cleanly.
                //
                // Only written for outside voxels (shell candidates);
                // inside voxels never become shell leaves and their
                // slot stays zero-packed.
                if !is_inside {
                    let eps = voxel_size;
                    let d_px = bvh.nearest(pos + Vec3::new(eps, 0.0, 0.0)).distance;
                    let d_nx = bvh.nearest(pos + Vec3::new(-eps, 0.0, 0.0)).distance;
                    let d_py = bvh.nearest(pos + Vec3::new(0.0, eps, 0.0)).distance;
                    let d_ny = bvh.nearest(pos + Vec3::new(0.0, -eps, 0.0)).distance;
                    let d_pz = bvh.nearest(pos + Vec3::new(0.0, 0.0, eps)).distance;
                    let d_nz = bvh.nearest(pos + Vec3::new(0.0, 0.0, -eps)).distance;
                    let grad = Vec3::new(d_px - d_nx, d_py - d_ny, d_pz - d_nz);
                    let len2 = grad.length_squared();
                    let normal = if len2 > 1e-12 {
                        grad / len2.sqrt()
                    } else {
                        Vec3::Y
                    };
                    face_normals[flat] = rkp_core::leaf_attr::pack_oct(normal);
                }
            }
        }
    }

    BrickResult {
        color_brick,
        material_ids,
        is_inside: is_inside_buf,
        face_normals,
        all_inside,
    }
}

/// Classify the brick grid into surface work (narrow-band) and
/// solid-interior bricks (inside-the-mesh, outside the band).
///
/// Bricks outside the band and outside the mesh are neither returned
/// — they're implicit empty space. Caller iterates the grid `(bx, by,
/// bz)` in z-major order to produce stable partitioning.
pub fn classify_bricks(
    bvh: &TriangleBvh,
    grid_origin: Vec3,
    brick_world_size: f32,
    octree_bricks: u32,
    narrow_band: f32,
) -> (Vec<BrickWork>, Vec<(u32, u32, u32)>) {
    let mut surface_work = Vec::new();
    let mut interior_bricks = Vec::new();

    for bz in 0..octree_bricks {
        for by in 0..octree_bricks {
            for bx in 0..octree_bricks {
                let brick_min = grid_origin
                    + Vec3::new(
                        bx as f32 * brick_world_size,
                        by as f32 * brick_world_size,
                        bz as f32 * brick_world_size,
                    );
                let brick_center = brick_min + Vec3::splat(brick_world_size * 0.5);
                let nearest = bvh.nearest(brick_center);

                if nearest.distance < narrow_band {
                    surface_work.push(BrickWork { bx, by, bz, brick_min });
                } else if bvh.is_inside_raycast(brick_center) {
                    interior_bricks.push((bx, by, bz));
                }
            }
        }
    }

    (surface_work, interior_bricks)
}
