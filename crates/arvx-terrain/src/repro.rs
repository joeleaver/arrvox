//! Smooth-stairs reproduction harness — bakes the **real** terrain
//! through the **exact** bake path at the **exact** on-screen LOD voxel
//! sizes, so the residual low-frequency "smooth stairs" ripple the user
//! reports can be reproduced, measured, and fix-tested headlessly.
//!
//! This is deliberately in `arvx-terrain` (not `arvx-core`'s
//! `mesh_test_bench`) because only here do we have the **real**
//! [`FbmTerrainFn`](crate::fbm::FbmTerrainFn) and the **real** bake-path
//! SDF closure that `bake_tile_with_skirts` feeds to
//! `voxelize_to_artifact` + `extract_surface_mesh_density_haloed`. The
//! `arvx-core` bench drives the *sculpt/region* extract on a hand-built
//! `CellMap` — a different path. Here we mesh the EXACT terrain surface,
//! through the EXACT bake path, at the EXACT LOD voxel sizes
//! (0.25/0.5/1.0/2.0 m).
//!
//! To keep the 512² software render legible we bake a representative
//! ~16–32 m WINDOW of a tile (the full 64 m tile is thousands of
//! triangles of noise). The window is a pow2-cubic AABB positioned at a
//! real tile origin and driven by the real `FbmTerrainFnResolved::sample`
//! with the real LOD voxel size — so LOD octave-dropping
//! (`octaves_for_voxel`) is faithful too.

use crate::fbm::FbmTerrainFn;
use crate::terrain_fn::{TerrainFn, TerrainSample};
use crate::tile_key::TileKey;
use arvx_core::mesh_extract::{
    extract_surface_mesh_density_haloed, set_blur_override, set_wide_window_project, CellMap,
    MeshVertex,
};
use arvx_core::mesh_test_bench::{BlurParams, Occupancy};
use arvx_core::voxelize_octree::voxelize_to_artifact;
use arvx_core::{Aabb, NullMaterialLookup};
use glam::{IVec3, Vec3};

/// Terrain halo the real bake uses (`bake.rs::TILE_HALO_VOXELS`).
pub const REPRO_TILE_HALO: u32 = 4;

/// A baked real-terrain window: raw extract output + render inputs.
pub struct TerrainWindowMesh {
    /// LOD voxel size this window was baked at (m).
    pub voxel_size: f32,
    /// Object-local vertices from the bake-path extract.
    pub verts: Vec<MeshVertex>,
    /// Triangle indices.
    pub indices: Vec<u32>,
    /// Grid origin (lo corner of cell (0,0,0)) in world coords.
    pub grid_origin: Vec3,
    /// Surface (top-shell) cells for the voxel overlay render.
    pub surface_cells: CellMap,
    /// The window AABB in world coords.
    pub aabb: Aabb,
}

impl TerrainWindowMesh {
    /// Wrap the top-shell cells as an [`Occupancy`] so `arvx-core`'s
    /// software renderer can draw the voxel overlay unchanged.
    pub fn as_occupancy(&self) -> Occupancy {
        let (mut lo, mut hi) = (IVec3::splat(i32::MAX), IVec3::splat(i32::MIN));
        for &c in self.surface_cells.keys() {
            lo = lo.min(c);
            hi = hi.max(c);
        }
        if self.surface_cells.is_empty() {
            lo = IVec3::ZERO;
            hi = IVec3::ZERO;
        }
        Occupancy {
            cells: self.surface_cells.clone(),
            grid_origin: self.grid_origin,
            voxel_size: self.voxel_size,
            region_min: lo - IVec3::splat(2),
            region_max: hi + IVec3::splat(3),
        }
    }
}

/// Default representative terrain function: `FbmTerrainFn::default()`
/// resolved against the null material lookup (slot-0 materials — geometry
/// is identical regardless of material).
pub fn default_terrain_fn() -> impl TerrainFn {
    FbmTerrainFn::default().resolve(&NullMaterialLookup)
}

/// A GENTLE planar-slope terrain source `y = mx·x + mz·z + base`, returning
/// the TRUE point-to-plane Euclidean signed distance (1-Lipschitz, as the
/// voxelizer's coarse classifier requires). This is the canonical
/// wide-tread "smooth stairs" probe: at `mx = 0.10` the surface rises 1
/// voxel over 10 cells, far wider than the R=2 blur kernel can span.
pub struct SlopeTerrainFn {
    /// dh/dx (gentle: 0.10–0.20).
    pub mx: f32,
    /// Small off-axis tilt so the slope is never grid-aligned.
    pub mz: f32,
    /// Base height at the tile-local origin (world Y).
    pub base: f32,
}

impl TerrainFn for SlopeTerrainFn {
    fn sample(&self, tile: TileKey, local: Vec3, _voxel_size_m: f32) -> TerrainSample {
        let world_origin = tile.origin_world().to_vec3();
        let wx = world_origin.x + local.x;
        let wy = world_origin.y + local.y;
        let wz = world_origin.z + local.z;
        let surf = self.base + self.mx * wx + self.mz * wz;
        let grad_mag = (1.0 + self.mx * self.mx + self.mz * self.mz).sqrt();
        TerrainSample {
            sd: (wy - surf) / grad_mag,
            primary_mat: 1,
            secondary_mat: 1,
            blend: 0.0,
        }
    }
}

/// Build a pow2-cubic window AABB of `~window_m` metres, snapped to the
/// voxel grid, with its lo corner at `world_lo`.
fn window_aabb(world_lo: Vec3, window_m: f32, voxel_size: f32) -> Aabb {
    let cells = (window_m / voxel_size).ceil().max(1.0) as u32;
    let pow2 = cells.next_power_of_two();
    let extent = pow2 as f32 * voxel_size;
    let snap = |v: f32| (v / voxel_size).floor() * voxel_size;
    let lo = Vec3::new(snap(world_lo.x), snap(world_lo.y), snap(world_lo.z));
    Aabb::new(lo, lo + Vec3::splat(extent))
}

/// Position a window so the FBM surface band sits ~60% up the cube's Y
/// extent, leaving a tall solid block below (interior) and air above.
fn position_window_lo(
    terrain_fn: &dyn TerrainFn,
    key: TileKey,
    centre_xz: Vec3,
    window_m: f32,
    voxel_size: f32,
) -> Vec3 {
    let pow2 = ((window_m / voxel_size).ceil().max(1.0) as u32).next_power_of_two();
    let extent = pow2 as f32 * voxel_size;
    let world_origin = key.origin_world().to_vec3();
    // Surface height at the window centre. sample() uses local x/z + the
    // tile origin; sd = wy - surface_y, so at wy=0 (local.y=0):
    // surface_y = world_origin.y - sd.
    let local = Vec3::new(centre_xz.x, 0.0, centre_xz.z);
    let s = terrain_fn.sample(key, local, voxel_size);
    let surface_y = world_origin.y - s.sd;
    let lo_y = surface_y - extent * 0.6;
    Vec3::new(
        world_origin.x + centre_xz.x - extent * 0.5,
        lo_y,
        world_origin.z + centre_xz.z - extent * 0.5,
    )
}

/// Mesh a real-terrain window through the **EXACT bake path**:
/// the real `FbmTerrainFnResolved::sample` SDF →
/// `voxelize_to_artifact` (octree + bricks + halo) →
/// `extract_surface_mesh_density_haloed`. Same as
/// `bake_tile_with_skirts` minus stamps/regions/skirts/cluster-DAG
/// (none of which affect the top-surface ripple).
///
/// `centre_xz` is the tile-local horizontal position of the window
/// centre (so different windows sample different terrain).
pub fn bake_terrain_window(
    terrain_fn: &dyn TerrainFn,
    key: TileKey,
    centre_xz: Vec3,
    window_m: f32,
    voxel_size: f32,
) -> TerrainWindowMesh {
    let world_lo = position_window_lo(terrain_fn, key, centre_xz, window_m, voxel_size);
    let aabb = window_aabb(world_lo, window_m, voxel_size);

    // EXACT bake-path SDF closure (mirrors bake_tile_with_skirts: ask the
    // TerrainFn in tile-local coords, sd = wy - surface_y).
    let tile_origin_world = key.origin_world().to_vec3();
    let sdf_fn = |positions: &[Vec3]| -> Vec<(f32, u16, u16, u8, u32, Option<Vec3>)> {
        positions
            .iter()
            .map(|&world_pos| {
                let local = world_pos - tile_origin_world;
                let s = terrain_fn.sample(key, local, voxel_size);
                let blend_u4 = (s.blend.clamp(0.0, 1.0) * 15.0).round() as u8;
                (s.sd, s.primary_mat, s.secondary_mat, blend_u4, 0, None)
            })
            .collect()
    };

    let artifact = voxelize_to_artifact(sdf_fn, &aabb, voxel_size, REPRO_TILE_HALO)
        .expect("terrain window voxelize_to_artifact");
    let brick_pool_flat: Vec<u32> = artifact.brick_cells.iter().flatten().copied().collect();
    let (verts, indices) = extract_surface_mesh_density_haloed(
        artifact.octree.as_slice(),
        artifact.octree.depth(),
        voxel_size,
        artifact.grid_origin,
        &brick_pool_flat,
        &artifact.leaf_attrs,
        &[],
        &artifact.halo_cells,
        REPRO_TILE_HALO,
        None,
        &[],
    );

    let surface_cells =
        surface_shell_cells(terrain_fn, key, &aabb, voxel_size, tile_origin_world);

    TerrainWindowMesh {
        voxel_size,
        verts,
        indices,
        grid_origin: artifact.grid_origin,
        surface_cells,
        aabb,
    }
}

/// Mesh the SAME terrain-window occupancy through the **REGION/SCULPT
/// PATH** (`extract_mesh_region_from_cells_pooled_haloed` via the
/// `arvx-core` bench's `mesh_occupancy`) so the bake-path ripple can be
/// compared against the region-path ripple on the IDENTICAL occupancy.
/// We build the full solid CellMap (every cell at/below the surface) and
/// a region tight around it — exactly what the sculpt path consumes.
pub fn region_mesh_terrain_window(
    terrain_fn: &dyn TerrainFn,
    key: TileKey,
    centre_xz: Vec3,
    window_m: f32,
    voxel_size: f32,
) -> TerrainWindowMesh {
    use arvx_core::mesh_test_bench::{mesh_occupancy, Occupancy as BenchOcc};

    let world_lo = position_window_lo(terrain_fn, key, centre_xz, window_m, voxel_size);
    let aabb = window_aabb(world_lo, window_m, voxel_size);
    let origin = aabb.min;
    let n = ((aabb.max.x - aabb.min.x) / voxel_size).round() as i32;
    let tile_origin_world = key.origin_world().to_vec3();

    let mut cells = CellMap::default();
    let (mut lo, mut hi) = (IVec3::splat(i32::MAX), IVec3::splat(i32::MIN));
    for cz in 0..n {
        for cx in 0..n {
            for cy in 0..n {
                let world = origin
                    + (Vec3::new(cx as f32, cy as f32, cz as f32) + Vec3::splat(0.5)) * voxel_size;
                let local = world - tile_origin_world;
                if terrain_fn.sample(key, local, voxel_size).sd < 0.0 {
                    let k = IVec3::new(cx, cy, cz);
                    cells.insert(k, 0);
                    lo = lo.min(k);
                    hi = hi.max(k);
                }
            }
        }
    }
    let occ = BenchOcc {
        cells,
        grid_origin: origin,
        voxel_size,
        region_min: lo - IVec3::splat(2),
        region_max: hi + IVec3::splat(3),
    };
    let (verts, indices) = mesh_occupancy(&occ);
    let surface_cells =
        surface_shell_cells(terrain_fn, key, &aabb, voxel_size, tile_origin_world);
    TerrainWindowMesh {
        voxel_size,
        verts,
        indices,
        grid_origin: origin,
        surface_cells,
        aabb,
    }
}

/// Bake a terrain window through the PRODUCTION bake path with the
/// smooth-stairs fix **FORCED OFF** (the raw rippled baseline). Use this
/// for the before/after comparison; [`bake_terrain_window`] is the
/// shipping behaviour (fix default-ON).
pub fn bake_terrain_window_baseline(
    terrain_fn: &dyn TerrainFn,
    key: TileKey,
    centre_xz: Vec3,
    window_m: f32,
    voxel_size: f32,
) -> TerrainWindowMesh {
    set_wide_window_project(Some(0.0)); // force the fix off
    let out = bake_terrain_window(terrain_fn, key, centre_xz, window_m, voxel_size);
    set_wide_window_project(None);
    out
}

/// Same as [`bake_terrain_window_baseline`] (fix OFF) but with a per-thread
/// blur override so the R-sweep contrast can isolate a wider blur `(R, σ,
/// iso)` from the plane-fit fix.
pub fn bake_terrain_window_blur(
    terrain_fn: &dyn TerrainFn,
    key: TileKey,
    centre_xz: Vec3,
    window_m: f32,
    voxel_size: f32,
    bp: BlurParams,
) -> TerrainWindowMesh {
    set_blur_override(Some((bp.r, bp.sigma, bp.iso)));
    set_wide_window_project(Some(0.0)); // isolate the blur from the plane-fit
    let out = bake_terrain_window(terrain_fn, key, centre_xz, window_m, voxel_size);
    set_wide_window_project(None);
    set_blur_override(None);
    out
}

/// Bake a terrain window with the plane-fit fix at an EXPLICIT radius
/// (overrides the production default). For the radius sweep.
pub fn bake_terrain_window_fix(
    terrain_fn: &dyn TerrainFn,
    key: TileKey,
    centre_xz: Vec3,
    window_m: f32,
    voxel_size: f32,
    radius_voxels: f32,
) -> TerrainWindowMesh {
    set_wide_window_project(Some(radius_voxels));
    let out = bake_terrain_window(terrain_fn, key, centre_xz, window_m, voxel_size);
    set_wide_window_project(None);
    out
}

/// Collect the top-shell (boundary) cells of the terrain window for the
/// voxel overlay render. A cell is in the shell if it is solid and its
/// `+Y` neighbour is empty.
fn surface_shell_cells(
    terrain_fn: &dyn TerrainFn,
    key: TileKey,
    aabb: &Aabb,
    voxel_size: f32,
    tile_origin_world: Vec3,
) -> CellMap {
    let origin = aabb.min;
    let n = ((aabb.max.x - aabb.min.x) / voxel_size).round() as i32;
    let mut cells = CellMap::default();
    let solid = |cx: i32, cy: i32, cz: i32| -> bool {
        let world = origin
            + (Vec3::new(cx as f32, cy as f32, cz as f32) + Vec3::splat(0.5)) * voxel_size;
        let local = world - tile_origin_world;
        terrain_fn.sample(key, local, voxel_size).sd < 0.0
    };
    for cz in 0..n {
        for cx in 0..n {
            for cy in 0..n {
                if solid(cx, cy, cz) && !solid(cx, cy + 1, cz) {
                    cells.insert(IVec3::new(cx, cy, cz), 0);
                }
            }
        }
    }
    cells
}
