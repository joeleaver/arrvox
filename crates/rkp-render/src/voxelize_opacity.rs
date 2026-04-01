//! Direct mesh-to-opacity voxelization — bypasses SDF for smooth splat fields.
//!
//! Computes per-voxel opacity from unsigned distance to the nearest triangle
//! surface, with inside/outside determined by generalized winding number.
//! Produces a smooth opacity field whose gradient gives clean surface normals.

use std::path::Path;

use glam::Vec3;
use half::f16;
use rayon::prelude::*;

use rkf_core::Aabb;
use rkf_core::brick::Brick;
use rkf_core::brick_map::{BrickMap, BrickMapAllocator, EMPTY_SLOT, INTERIOR_SLOT};
use rkf_core::brick_pool::BrickPool;
use rkf_core::companion::ColorBrick;
use rkf_core::constants::BRICK_DIM;
use rkf_core::sdf_cache::SdfCache;
use rkf_core::voxel::VoxelSample;
use rkf_import::bvh::TriangleBvh;
use rkf_import::material_transfer::sample_texture_at_triangle;
use rkf_import::mesh::MeshData;
use rkf_import::pipeline::{ImportConfig, ImportResult};

/// Smooth Hermite interpolation (matches WGSL smoothstep).
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Result of processing a single brick.
struct BrickResult {
    /// Opacity brick data (f16 opacity in word0, material in word1).
    brick: Brick,
    /// SDF cache storing opacity as f16 (for .rkf compatibility).
    sdf_cache: SdfCache,
    /// Per-voxel color from mesh textures.
    color_brick: ColorBrick,
    /// Whether any voxel has non-zero opacity.
    has_surface: bool,
    /// Whether all voxels are fully opaque.
    is_fully_solid: bool,
}

/// Process a single brick: compute per-voxel opacity and color.
fn process_brick(
    mesh: &MeshData,
    bvh: &TriangleBvh,
    brick_min: Vec3,
    voxel_size: f32,
    fade_inner: f32,
    fade_outer: f32,
    material_id_override: Option<u16>,
    import_colors: bool,
) -> BrickResult {
    let half_voxel = voxel_size * 0.5;
    let mut brick = Brick::default();
    let mut sdf_cache = SdfCache::default();
    let mut color_brick = ColorBrick::default();
    let mut any_nonzero = false;
    let mut all_solid = true;

    for vz in 0..BRICK_DIM {
        for vy in 0..BRICK_DIM {
            for vx in 0..BRICK_DIM {
                let pos = brick_min
                    + Vec3::new(
                        vx as f32 * voxel_size + half_voxel,
                        vy as f32 * voxel_size + half_voxel,
                        vz as f32 * voxel_size + half_voxel,
                    );

                // Unsigned distance to nearest triangle
                let nearest = bvh.nearest(pos);
                let d = nearest.distance;

                // Inside/outside via winding number
                let w = bvh.winding_number(pos);
                let is_inside = w > 0.5;

                // Signed distance (negative inside, positive outside)
                let signed_d = if is_inside { -d } else { d };

                // Smooth opacity: 1.0 deep inside, 0.0 far outside,
                // smooth transition at the surface.
                let opacity = 1.0 - smoothstep(-fade_inner, fade_outer, signed_d);

                if opacity > 0.001 {
                    any_nonzero = true;
                }
                if opacity < 0.999 {
                    all_solid = false;
                }

                // Material ID
                let mat_id = material_id_override.unwrap_or(0);

                // Store opacity as f16 in the VoxelSample distance field
                let opacity_f16 = f16::from_f32(opacity);
                let sample = VoxelSample::new(opacity_f16.to_f32(), mat_id, 0);
                brick.set(vx, vy, vz, sample);

                // SDF cache stores opacity bits for .rkf compatibility
                sdf_cache.set_distance(vx as u8, vy as u8, vz as u8, opacity_f16.to_f32());

                // Per-voxel color from mesh texture (only for surface voxels)
                if import_colors && opacity > 0.01 {
                    if let Some(color) = sample_texture_at_triangle(
                        mesh,
                        nearest.triangle_index,
                        &nearest.barycentric,
                    ) {
                        color_brick.set(
                            vx,
                            vy,
                            vz,
                            rkf_core::companion::ColorVoxel::new(
                                color.r,
                                color.g,
                                color.b,
                                255, // full intensity
                            ),
                        );
                    }
                }
            }
        }
    }

    BrickResult {
        brick,
        sdf_cache,
        color_brick,
        has_surface: any_nonzero,
        is_fully_solid: all_solid,
    }
}

/// Import a mesh file and produce an opacity-voxelized .rkf file.
///
/// This is the splat-native alternative to `rkf_import::pipeline::import_mesh_to_rkf`.
/// Instead of computing SDF distances, it produces a smooth opacity field from
/// unsigned distance + winding number with a smoothstep transition.
pub fn import_mesh_to_opacity_rkf(
    input_path: &Path,
    output_path: &Path,
    config: &ImportConfig,
) -> Result<ImportResult, String> {
    use rkf_core::asset_file_v5::{save_object_v5, SaveLodV5};
    use rkf_core::brick_geometry::BrickGeometry;
    use rkf_import::mesh::load_mesh;

    log::info!("Splat import: loading {}", input_path.display());

    // 1. Load mesh
    let input_str = input_path.to_string_lossy();
    let mut mesh = load_mesh(&input_str).map_err(|e| format!("load mesh: {e}"))?;

    // Apply rotation, normalization, and scale from import config.
    rkf_import::pipeline::prepare_mesh(&mut mesh, config);

    let aabb = Aabb {
        min: mesh.bounds_min,
        max: mesh.bounds_max,
    };

    // 2. Determine voxel size
    let voxel_size = config.voxel_size.unwrap_or_else(|| {
        let extent = aabb.max - aabb.min;
        let longest = extent.x.max(extent.y).max(extent.z);
        // Target ~200 voxels on longest axis
        (longest / 200.0).max(0.002)
    });

    // 3. Build BVH
    log::info!("Splat import: building BVH ({} triangles)", mesh.triangle_count());
    let bvh = TriangleBvh::build(&mesh);

    // 4. Compute grid dimensions
    let brick_world_size = voxel_size * BRICK_DIM as f32;
    let aabb_size = aabb.max - aabb.min;
    let padding = voxel_size * 4.0; // extra padding for transition zone
    let dims = glam::UVec3::new(
        (((aabb_size.x + padding * 2.0) / brick_world_size).ceil() as u32).max(1),
        (((aabb_size.y + padding * 2.0) / brick_world_size).ceil() as u32).max(1),
        (((aabb_size.z + padding * 2.0) / brick_world_size).ceil() as u32).max(1),
    );

    let grid_origin = -Vec3::new(
        dims.x as f32 * brick_world_size * 0.5,
        dims.y as f32 * brick_world_size * 0.5,
        dims.z as f32 * brick_world_size * 0.5,
    );

    // Transition parameters — wider transition gives smoother gradient normals
    // at the cost of slightly softer surface edges.
    let fade_inner = voxel_size * 1.0;
    let fade_outer = voxel_size * 3.0;

    log::info!(
        "Splat import: grid {}x{}x{}, voxel_size={}, fade=[{}, {}]",
        dims.x, dims.y, dims.z, voxel_size, fade_inner, fade_outer,
    );

    // 5. Pass 1: determine which bricks need allocation (narrow band)
    let narrow_band = brick_world_size * 2.0;
    let total_bricks = (dims.x * dims.y * dims.z) as usize;
    let mut brick_needs_alloc = vec![false; total_bricks];
    let mut interior_bricks = vec![false; total_bricks];

    for bz in 0..dims.z {
        for by in 0..dims.y {
            for bx in 0..dims.x {
                let brick_min = grid_origin
                    + Vec3::new(
                        bx as f32 * brick_world_size,
                        by as f32 * brick_world_size,
                        bz as f32 * brick_world_size,
                    );
                let brick_center = brick_min + Vec3::splat(brick_world_size * 0.5);
                let nearest = bvh.nearest(brick_center);

                let bi = (bx + by * dims.x + bz * dims.x * dims.y) as usize;

                if nearest.distance < narrow_band {
                    brick_needs_alloc[bi] = true;
                } else {
                    // Deep interior or deep exterior
                    let w = bvh.winding_number(brick_center);
                    if w > 0.5 {
                        interior_bricks[bi] = true;
                    }
                }
            }
        }
    }

    let needed_count: u32 = brick_needs_alloc.iter().filter(|&&b| b).count() as u32;
    log::info!("Splat import: {} bricks in narrow band", needed_count);

    // 6. Pass 2: process bricks in parallel
    struct BrickWork {
        bx: u32,
        by: u32,
        bz: u32,
        brick_min: Vec3,
    }

    let mut work = Vec::new();
    for bz in 0..dims.z {
        for by in 0..dims.y {
            for bx in 0..dims.x {
                let bi = (bx + by * dims.x + bz * dims.x * dims.y) as usize;
                if brick_needs_alloc[bi] {
                    let brick_min = grid_origin
                        + Vec3::new(
                            bx as f32 * brick_world_size,
                            by as f32 * brick_world_size,
                            bz as f32 * brick_world_size,
                        );
                    work.push(BrickWork { bx, by, bz, brick_min });
                }
            }
        }
    }

    let results: Vec<(BrickWork, BrickResult)> = work
        .into_par_iter()
        .map(|w| {
            let result = process_brick(
                &mesh,
                &bvh,
                w.brick_min,
                voxel_size,
                fade_inner,
                fade_outer,
                config.material_id_override,
                config.import_colors,
            );
            (w, result)
        })
        .collect();

    // 7. Build brick map and collect data for .rkf writing
    let mut brick_map = BrickMap::new(dims);
    let mut geometries = Vec::new();
    let mut sdf_caches = Vec::new();
    let mut color_bricks = Vec::new();
    let mut allocated_count = 0u32;

    // Mark interior bricks
    for bz in 0..dims.z {
        for by in 0..dims.y {
            for bx in 0..dims.x {
                let bi = (bx + by * dims.x + bz * dims.x * dims.y) as usize;
                if interior_bricks[bi] {
                    brick_map.set(bx, by, bz, INTERIOR_SLOT);
                }
            }
        }
    }

    // Process results
    for (w, result) in results {
        if !result.has_surface {
            // No surface voxels — check if deep interior
            if result.is_fully_solid {
                brick_map.set(w.bx, w.by, w.bz, INTERIOR_SLOT);
            }
            // else: leave as EMPTY_SLOT
            continue;
        }

        let slot = allocated_count;
        brick_map.set(w.bx, w.by, w.bz, slot);
        allocated_count += 1;

        // Build BrickGeometry from opacity data (for .rkf compatibility)
        let mut geo = BrickGeometry::new();
        for vz in 0..8u8 {
            for vy in 0..8u8 {
                for vx in 0..8u8 {
                    let sample = result.brick.sample(vx as u32, vy as u32, vz as u32);
                    let opacity = f16::from_bits((sample.word0 & 0xFFFF) as u16).to_f32();
                    if opacity > 0.5 {
                        geo.set_solid(vx, vy, vz, true);
                    }
                }
            }
        }
        geo.rebuild_surface_list();

        geometries.push(geo);
        sdf_caches.push(result.sdf_cache);
        color_bricks.push(result.color_brick);
    }

    log::info!("Splat import: {} allocated bricks", allocated_count);

    // 8. Write .rkf v5 file
    let grid_aabb = Aabb {
        min: grid_origin,
        max: grid_origin + Vec3::new(
            dims.x as f32 * brick_world_size,
            dims.y as f32 * brick_world_size,
            dims.z as f32 * brick_world_size,
        ),
    };

    let material_ids: Vec<u16> = if let Some(id) = config.material_id_override {
        vec![id]
    } else {
        vec![0]
    };

    let has_color = color_bricks.iter().any(|cb| {
        cb.data.iter().any(|cv| cv.intensity() > 0)
    });

    let lod = SaveLodV5 {
        voxel_size,
        brick_map,
        geometry: geometries,
        sdf_cache: Some(sdf_caches),
        color_bricks: if has_color { Some(color_bricks) } else { None },
        bone_bricks: None,
    };

    let file = std::fs::File::create(output_path)
        .map_err(|e| format!("create output: {e}"))?;
    let mut writer = std::io::BufWriter::new(file);
    save_object_v5(&mut writer, &grid_aabb, None, &material_ids, &[lod])
        .map_err(|e| format!("write .rkf: {e}"))?;

    let file_size = std::fs::metadata(output_path)
        .map(|m| m.len())
        .unwrap_or(0);

    log::info!(
        "Splat import: wrote {} ({} bricks, {:.1} KiB)",
        output_path.display(),
        allocated_count,
        file_size as f64 / 1024.0,
    );

    Ok(ImportResult {
        aabb: grid_aabb,
        total_bricks: allocated_count,
        lod_count: 1,
        finest_voxel_size: voxel_size,
        file_size,
        skeleton_path: None,
    })
}
