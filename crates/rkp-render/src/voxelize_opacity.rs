//! Direct mesh-to-opacity voxelization — bypasses SDF for smooth splat fields.
//!
//! Computes per-voxel opacity from unsigned distance to the nearest triangle
//! surface, with inside/outside determined by generalized winding number.
//! Produces a smooth opacity field whose gradient gives clean surface normals.
//!
//! Full feature parity with rkf-import's SDF pipeline:
//! - Per-triangle material IDs
//! - Per-voxel color from mesh textures
//! - Per-voxel bone weights for skeletal animation
//! - Multi-LOD generation
//! - Skeleton extraction (.rkskel sidecar)

use std::path::Path;

use glam::Vec3;
use half::f16;
use rayon::prelude::*;

use rkf_core::Aabb;
use rkf_core::brick::Brick;
use rkf_core::brick_map::BrickMap;
use rkf_core::companion::{BoneBrick, BoneVoxel, ColorBrick, ColorVoxel};
use rkf_core::constants::BRICK_DIM;
use rkf_core::sdf_cache::SdfCache;
use rkf_core::voxel::VoxelSample;
use rkf_import::bvh::TriangleBvh;
use rkf_import::material_transfer::{sample_texture_at_triangle, sample_bone_weights_at_triangle};
use rkf_import::mesh::MeshData;
use rkf_import::pipeline::{ImportConfig, ImportResult};
use rkf_import::skeleton_extract::{self, VertexSkinning};

/// Smooth Hermite interpolation (matches WGSL smoothstep).
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Flat voxel index within a brick (matches rkf-core convention).
fn voxel_index(x: u8, y: u8, z: u8) -> u32 {
    x as u32 + y as u32 * 8 + z as u32 * 64
}

/// Result of processing a single brick.
struct BrickResult {
    sdf_cache: SdfCache,
    color_brick: ColorBrick,
    bone_brick: Option<BoneBrick>,
    /// Per-voxel opacity (used to build BrickGeometry and VoxelSample).
    opacities: [f32; 512],
    /// Per-voxel material ID from nearest triangle.
    material_ids: [u16; 512],
    /// Whether any voxel has non-zero opacity.
    has_surface: bool,
    /// Whether all voxels are fully opaque.
    is_fully_solid: bool,
}

/// Process a single brick: compute per-voxel opacity, material, color, and bone weights.
fn process_brick(
    mesh: &MeshData,
    bvh: &TriangleBvh,
    brick_min: Vec3,
    voxel_size: f32,
    fade_inner: f32,
    fade_outer: f32,
    material_id_override: Option<u16>,
    import_colors: bool,
    skinning: Option<&VertexSkinning>,
) -> BrickResult {
    let half_voxel = voxel_size * 0.5;
    let mut sdf_cache = SdfCache::default();
    let mut color_brick = ColorBrick::default();
    let mut opacities = [0.0f32; 512];
    let mut material_ids = [0u16; 512];
    let mut any_nonzero = false;
    let mut all_solid = true;

    // Bone brick — only allocated if skinning data exists
    let mut bone_brick = if skinning.is_some() {
        Some(BoneBrick { data: [BoneVoxel::default(); 512] })
    } else {
        None
    };
    let mut has_any_bone = false;

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
                let d = nearest.distance;
                let w = bvh.winding_number(pos);
                let is_inside = w > 0.5;
                let signed_d = if is_inside { -d } else { d };
                let opacity = 1.0 - smoothstep(-fade_inner, fade_outer, signed_d);

                let flat = voxel_index(vx as u8, vy as u8, vz as u8) as usize;
                opacities[flat] = opacity;

                if opacity > 0.001 {
                    any_nonzero = true;
                }
                if opacity < 0.999 {
                    all_solid = false;
                }

                // Per-triangle material ID
                let mat_id = if let Some(override_id) = material_id_override {
                    override_id
                } else {
                    let tri_idx = nearest.triangle_index;
                    if tri_idx < mesh.material_indices.len() {
                        mesh.material_indices[tri_idx] as u16
                    } else {
                        0
                    }
                };
                material_ids[flat] = mat_id;

                // SDF cache stores opacity for .rkf compatibility
                sdf_cache.set_distance(vx as u8, vy as u8, vz as u8, f16::from_f32(opacity).to_f32());

                // Per-voxel color from mesh texture
                if import_colors && opacity > 0.01 {
                    if let Some(color) = sample_texture_at_triangle(
                        mesh,
                        nearest.triangle_index,
                        &nearest.barycentric,
                    ) {
                        color_brick.set(vx, vy, vz, ColorVoxel::new(color.r, color.g, color.b, 255));
                    }
                }

                // Bone weights for any voxel with non-zero opacity
                // (the skin deform pass needs weights for all deformed voxels,
                // including those in the opacity transition zone)
                if let (Some(skin), Some(bb)) = (skinning, bone_brick.as_mut()) {
                    if opacity > 0.01 {
                        let bv = sample_bone_weights_at_triangle(
                            mesh, skin, nearest.triangle_index, &nearest.barycentric,
                        );
                        if bv.weights != 0 {
                            has_any_bone = true;
                        }
                        bb.data[flat] = bv;
                    }
                }
            }
        }
    }

    // Only keep bone brick if any voxel actually has bone weights
    if !has_any_bone {
        bone_brick = None;
    }

    BrickResult {
        sdf_cache,
        color_brick,
        bone_brick,
        opacities,
        material_ids,
        has_surface: any_nonzero,
        is_fully_solid: all_solid,
    }
}

/// Auto-detect voxel size using the same tier-based heuristic as rkf-import.
fn auto_voxel_size(aabb: &Aabb) -> f32 {
    let extent = aabb.max - aabb.min;
    let longest = extent.x.max(extent.y).max(extent.z);
    let tiers = [0.005f32, 0.02, 0.08, 0.32];
    for &vs in &tiers {
        let brick_world = vs * BRICK_DIM as f32;
        let bricks_on_longest = (longest / brick_world).ceil() as u32;
        if bricks_on_longest >= 8 {
            return vs;
        }
    }
    tiers[0]
}

/// Generate a single LOD at the given voxel size.
fn generate_lod(
    mesh: &MeshData,
    bvh: &TriangleBvh,
    voxel_size: f32,
    aabb: &Aabb,
    config: &ImportConfig,
    skinning: Option<&VertexSkinning>,
) -> rkf_core::asset_file_v5::SaveLodV5 {
    use rkf_core::asset_file_v5::SaveLodV5;
    use rkf_core::brick_geometry::BrickGeometry;
    use rkf_core::brick_map::{EMPTY_SLOT, INTERIOR_SLOT};

    let brick_world_size = voxel_size * BRICK_DIM as f32;
    let aabb_size = aabb.max - aabb.min;
    let padding = voxel_size * 4.0;
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

    let fade_inner = voxel_size * 1.0;
    let fade_outer = voxel_size * 3.0;

    // Pass 1: narrow-band culling
    let narrow_band = brick_world_size * 1.8;
    let total_brick_count = (dims.x * dims.y * dims.z) as usize;
    let mut brick_needs_alloc = vec![false; total_brick_count];
    let mut interior_bricks = vec![false; total_brick_count];

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
                    let w = bvh.winding_number(brick_center);
                    if w > 0.5 {
                        interior_bricks[bi] = true;
                    }
                }
            }
        }
    }

    // Pass 2: process bricks in parallel
    struct BrickWork {
        bx: u32,
        by: u32,
        bz: u32,
        brick_min: Vec3,
    }

    let mut work_items = Vec::new();
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
                    work_items.push(BrickWork { bx, by, bz, brick_min });
                }
            }
        }
    }

    let results: Vec<(BrickWork, BrickResult)> = work_items
        .into_par_iter()
        .map(|w| {
            let result = process_brick(
                mesh, bvh, w.brick_min, voxel_size, fade_inner, fade_outer,
                config.material_id_override, config.import_colors, skinning,
            );
            (w, result)
        })
        .collect();

    // Build brick map and collect data
    let mut brick_map = BrickMap::new(dims);
    let mut geometries = Vec::new();
    let mut sdf_caches = Vec::new();
    let mut color_bricks = Vec::new();
    let mut bone_bricks: Vec<BoneBrick> = Vec::new();
    let mut allocated_count = 0u32;
    let mut has_any_bones = false;

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

    for (w, result) in results {
        if !result.has_surface {
            if result.is_fully_solid {
                brick_map.set(w.bx, w.by, w.bz, INTERIOR_SLOT);
            }
            continue;
        }

        let slot = allocated_count;
        brick_map.set(w.bx, w.by, w.bz, slot);
        allocated_count += 1;

        // Build BrickGeometry from opacity data
        let mut geo = BrickGeometry::new();
        for vz in 0..8u8 {
            for vy in 0..8u8 {
                for vx in 0..8u8 {
                    let flat = voxel_index(vx, vy, vz) as usize;
                    // Mark any voxel with non-trivial opacity as solid.
                    // This ensures bone weights cover the full transition zone.
                    if result.opacities[flat] > 0.01 {
                        geo.set_solid(vx, vy, vz, true);
                    }
                }
            }
        }
        geo.rebuild_surface_list();

        // Assign per-triangle material IDs to surface voxels
        for sv in &mut geo.surface_voxels {
            sv.material_id = result.material_ids[sv.index() as usize];
        }

        // Build VoxelSample brick with opacity + material
        let mut brick = Brick::default();
        for i in 0..512 {
            let opacity_f16 = f16::from_f32(result.opacities[i]);
            brick.voxels[i] = VoxelSample::new(opacity_f16.to_f32(), result.material_ids[i], 0);
        }

        geometries.push(geo);
        sdf_caches.push(result.sdf_cache);
        color_bricks.push(result.color_brick);

        if let Some(bb) = result.bone_brick {
            has_any_bones = true;
            bone_bricks.push(bb);
        } else {
            bone_bricks.push(BoneBrick { data: [BoneVoxel::default(); 512] });
        }
    }

    let has_color = color_bricks.iter().any(|cb| cb.data.iter().any(|cv| cv.intensity() > 0));

    SaveLodV5 {
        voxel_size,
        brick_map,
        geometry: geometries,
        sdf_cache: Some(sdf_caches),
        color_bricks: if has_color { Some(color_bricks) } else { None },
        bone_bricks: if has_any_bones { Some(bone_bricks) } else { None },
    }
}

/// Import a mesh file and produce an opacity-voxelized .rkf file.
///
/// Full feature parity with `rkf_import::pipeline::import_mesh_to_rkf`:
/// per-triangle materials, per-voxel color, bone weights, multi-LOD, skeleton.
pub fn import_mesh_to_opacity_rkf(
    input_path: &Path,
    output_path: &Path,
    config: &ImportConfig,
) -> Result<ImportResult, String> {
    use rkf_core::asset_file_v5::save_object_v5;
    use rkf_core::SdfPrimitive;
    use rkf_import::mesh::load_mesh;

    log::info!("Splat import: loading {}", input_path.display());

    // 1. Load and prepare mesh
    let input_str = input_path.to_string_lossy();
    let mut mesh = load_mesh(&input_str).map_err(|e| format!("load mesh: {e}"))?;
    let norm = rkf_import::pipeline::prepare_mesh(&mut mesh, config);

    let aabb = Aabb::new(mesh.bounds_min, mesh.bounds_max);
    let has_textures = mesh.materials.iter().any(|m| m.albedo_texture.is_some());

    // 2. Determine finest voxel size
    let finest_voxel_size = config.voxel_size.unwrap_or_else(|| auto_voxel_size(&aabb));

    // 3. Build BVH
    log::info!("Splat import: building BVH ({} triangles)", mesh.triangle_count());
    let bvh = TriangleBvh::build(&mesh);

    // 4. Extract skinning data (if mesh has bones)
    let skinning = {
        let input_str_ref = input_path.to_str().unwrap_or("");
        match skeleton_extract::extract_skeleton(input_str_ref) {
            Ok(Some(extraction)) => {
                log::info!("Skeleton found: {} bones", extraction.skeleton.bones.len());
                Some(extraction)
            }
            _ => None,
        }
    };
    let skinning_data = skinning.as_ref().map(|s| &s.skinning);

    // 5. Generate LODs
    let lod_count = config.lod_levels.max(1);
    log::info!("Splat import: generating {} LOD level(s), finest voxel_size={}", lod_count, finest_voxel_size);

    let mut lods = Vec::with_capacity(lod_count);
    let mut total_bricks = 0u32;

    for level in 0..lod_count {
        let vs = finest_voxel_size * (1u32 << level) as f32;
        log::info!("Splat import: LOD {} — voxel_size={}", level, vs);
        let lod = generate_lod(&mesh, &bvh, vs, &aabb, config, skinning_data);
        total_bricks += lod.brick_map.allocated_count() as u32;
        lods.push(lod);
    }

    // 6. Build metadata
    let center = aabb.center();
    let bounding_radius = (aabb.max - center).length();
    let analytical_bound = SdfPrimitive::Sphere { radius: bounding_radius };

    let material_ids: Vec<u16> = if let Some(id) = config.material_id_override {
        vec![id]
    } else {
        (0..mesh.materials.len().min(65536) as u16).collect()
    };

    // 7. Write .rkf v5 file
    let file = std::fs::File::create(output_path)
        .map_err(|e| format!("create output: {e}"))?;
    let mut writer = std::io::BufWriter::new(file);
    save_object_v5(&mut writer, &aabb, Some(&analytical_bound), &material_ids, &lods)
        .map_err(|e| format!("write .rkf: {e}"))?;

    let file_size = std::fs::metadata(output_path).map(|m| m.len()).unwrap_or(0);

    // 8. Extract and save skeleton sidecar
    let skeleton_path = if let Some(ref extraction) = skinning {
        let skel_path = output_path.with_extension("rkskel");
        let asset = rkf_animation::skeleton_asset::SkeletonAsset::with_normalization(
            extraction.skeleton.clone(),
            extraction.clips.clone(),
            norm.center.to_array(),
            norm.scale,
            norm.rotation_offset,
            norm.rotation_center.to_array(),
        );
        match rkf_animation::skeleton_asset::save_rkskel(&asset, &skel_path) {
            Ok(()) => {
                log::info!("Saved skeleton: {} bones → {}", extraction.skeleton.bones.len(), skel_path.display());
                Some(skel_path)
            }
            Err(e) => {
                log::error!("Failed to save .rkskel: {e}");
                None
            }
        }
    } else {
        None
    };

    log::info!(
        "Splat import: wrote {} ({} bricks across {} LODs, {:.1} KiB)",
        output_path.display(), total_bricks, lod_count, file_size as f64 / 1024.0,
    );

    Ok(ImportResult {
        aabb,
        total_bricks,
        lod_count,
        finest_voxel_size,
        file_size,
        skeleton_path,
    })
}

// ── Octree-native mesh import (.rkp) ─────────────────────────────────────

/// Import a mesh file and produce an octree-native .rkp file.
///
/// Uses the same BVH-based opacity sampling as `import_mesh_to_opacity_rkf`,
/// but builds a sparse octree instead of a flat BrickMap. The octree provides
/// built-in LOD (coarser leaves at shallower depths for uniform regions).
/// No separate LOD levels — the tree IS the hierarchy.
pub fn import_mesh_to_opacity_rkp(
    input_path: &Path,
    output_path: &Path,
    config: &ImportConfig,
) -> Result<ImportResult, String> {
    use rkf_import::mesh::load_mesh;

    eprintln!("Splat import (octree): loading {}", input_path.display());

    // 1. Load and prepare mesh
    let input_str = input_path.to_string_lossy();
    let mut mesh = load_mesh(&input_str).map_err(|e| format!("load mesh: {e}"))?;
    let norm = rkf_import::pipeline::prepare_mesh(&mut mesh, config);

    let aabb = Aabb::new(mesh.bounds_min, mesh.bounds_max);
    let voxel_size = config.voxel_size.unwrap_or_else(|| auto_voxel_size(&aabb));

    // 2. Build BVH
    eprintln!("Splat import (octree): building BVH ({} triangles)", mesh.triangle_count());
    let bvh = TriangleBvh::build(&mesh);

    // 3. Extract skinning data
    let skinning = {
        let input_str_ref = input_path.to_str().unwrap_or("");
        match skeleton_extract::extract_skeleton(input_str_ref) {
            Ok(Some(extraction)) => {
                eprintln!("Skeleton found: {} bones", extraction.skeleton.bones.len());
                Some(extraction)
            }
            _ => None,
        }
    };
    let skinning_data = skinning.as_ref().map(|s| &s.skinning);

    // 4. Per-voxel octree voxelization
    //
    // Classify regions at brick-level granularity (8x8x8 voxels) using BVH
    // narrow-band culling. Then process each surface region in parallel with
    // process_brick(). Finally, insert individual voxels as octree leaves.
    //
    // The octree depth covers individual voxels, not bricks. Depth = brick_depth + 3
    // since each brick is 8x8x8 = 2^3 voxels per axis.

    let brick_world_size = voxel_size * BRICK_DIM as f32;
    let padding = voxel_size * 4.0;
    let padded_aabb = Aabb::new(
        aabb.min - Vec3::splat(padding),
        aabb.max + Vec3::splat(padding),
    );

    // Compute brick-level depth, then add 3 for per-voxel leaves.
    let aabb_size = padded_aabb.max - padded_aabb.min;
    let max_dim = aabb_size.x.max(aabb_size.y).max(aabb_size.z);
    let bricks_needed = (max_dim / brick_world_size).ceil().max(1.0) as u32;
    let brick_depth = if bricks_needed <= 1 { 1u8 } else { (32 - (bricks_needed - 1).leading_zeros()) as u8 };
    let depth = brick_depth + 3; // per-voxel: 3 extra levels for 8x8x8

    let octree_bricks = 1u32 << brick_depth;
    let extent = octree_bricks as f32 * brick_world_size;
    let aabb_center = (padded_aabb.min + padded_aabb.max) * 0.5;
    let grid_origin = aabb_center - Vec3::splat(extent * 0.5);

    eprintln!(
        "Splat import (per-voxel octree): depth={}, voxel_size={}, grid voxels={}^3",
        depth, voxel_size, 1u32 << depth,
    );

    let fade_inner = voxel_size * 1.0;
    let fade_outer = voxel_size * 3.0;
    let narrow_band = brick_world_size * 1.8;

    // Classify brick-sized regions using BVH narrow-band culling.
    struct BrickWork {
        bx: u32,
        by: u32,
        bz: u32,
        brick_min: Vec3,
    }

    let mut surface_work = Vec::new();
    let mut interior_brick_coords = Vec::new();

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
                } else {
                    let w = bvh.winding_number(brick_center);
                    if w > 0.5 {
                        interior_brick_coords.push((bx, by, bz));
                    }
                }
            }
        }
    }

    eprintln!(
        "Splat import (per-voxel octree): {} surface regions, {} interior regions",
        surface_work.len(), interior_brick_coords.len(),
    );

    // Process surface regions in parallel (still 8x8x8 per region for BVH efficiency).
    let results: Vec<(BrickWork, BrickResult)> = surface_work
        .into_par_iter()
        .map(|w| {
            let result = process_brick(
                &mesh, &bvh, w.brick_min, voxel_size, fade_inner, fade_outer,
                config.material_id_override, config.import_colors, skinning_data,
            );
            (w, result)
        })
        .collect();

    // Build per-voxel octree + flat voxel arrays.
    let mut octree = rkp_core::SparseOctree::new(depth, voxel_size);
    let mut voxel_data: Vec<VoxelSample> = Vec::new();
    let mut color_voxels: Vec<ColorVoxel> = Vec::new();
    let mut has_color = false;
    let mut voxel_count = 0u32;

    // Insert interior regions — mark all 8x8x8 voxels as INTERIOR.
    // At brick_depth + 3, a brick-level coordinate (bx, by, bz) maps to
    // voxel coordinates (bx*8..bx*8+7, by*8..by*8+7, bz*8..bz*8+7).
    // We can use set_at_level to mark the whole 8x8x8 region at once
    // (level = brick_depth, which is 3 levels above max depth).
    for &(bx, by, bz) in &interior_brick_coords {
        let voxel_coord = glam::UVec3::new(bx * 8, by * 8, bz * 8);
        octree.set_at_level(voxel_coord, brick_depth, rkp_core::sparse_octree::INTERIOR_NODE);
    }

    // Insert individual surface voxels as octree leaves.
    for (w, result) in results {
        if !result.has_surface {
            if result.is_fully_solid {
                let voxel_coord = glam::UVec3::new(w.bx * 8, w.by * 8, w.bz * 8);
                octree.set_at_level(voxel_coord, brick_depth, rkp_core::sparse_octree::INTERIOR_NODE);
            }
            continue;
        }

        for vz in 0..8u32 {
            for vy in 0..8u32 {
                for vx in 0..8u32 {
                    let flat = (vx + vy * 8 + vz * 64) as usize;
                    let opacity = result.opacities[flat];
                    if opacity <= 0.001 {
                        continue; // empty voxel — octree default
                    }

                    let slot = voxel_count;
                    let voxel_coord = glam::UVec3::new(
                        w.bx * 8 + vx,
                        w.by * 8 + vy,
                        w.bz * 8 + vz,
                    );

                    if opacity >= 0.999 {
                        octree.insert_interior(voxel_coord);
                    } else {
                        octree.insert(voxel_coord, slot);
                        let mat_id = result.material_ids[flat];
                        voxel_data.push(VoxelSample::new(opacity, mat_id, 0));

                        // Per-voxel color
                        let cv = result.color_brick.data[flat];
                        if cv.intensity() > 0 {
                            has_color = true;
                        }
                        color_voxels.push(cv);

                        voxel_count += 1;
                    }
                }
            }
        }
    }

    let colored_count = color_voxels.iter().filter(|cv| cv.intensity() > 0).count();
    // Spot-check: print first few nonzero colors to verify data integrity.
    let mut sample_colors = Vec::new();
    for (i, cv) in color_voxels.iter().enumerate() {
        if cv.intensity() > 0 && sample_colors.len() < 3 {
            sample_colors.push(format!(
                "slot{}=({},{},{},{})",
                i, cv.red(), cv.green(), cv.blue(), cv.intensity(),
            ));
        }
    }
    eprintln!(
        "Splat import (per-voxel octree): {} leaf voxels, {} octree nodes, {} colored (has_color={}), samples: [{}]",
        voxel_count, octree.node_count(), colored_count, has_color, sample_colors.join(", "),
    );

    // 5. Serialize to .rkp v2 (per-voxel format)
    let material_ids: Vec<u16> = if let Some(id) = config.material_id_override {
        vec![id]
    } else {
        (0..mesh.materials.len().min(65536) as u16).collect()
    };

    let voxel_bytes: Vec<u8> = voxel_data
        .iter()
        .flat_map(|v| bytemuck::bytes_of(v))
        .copied()
        .collect();

    let color_data: Option<Vec<u8>> = if has_color {
        Some(
            color_voxels
                .iter()
                .flat_map(|cv| bytemuck::bytes_of(cv))
                .copied()
                .collect(),
        )
    } else {
        None
    };

    // TODO: Per-voxel bone data (bone weights per leaf voxel, not per brick)
    let bone_data: Option<&[u8]> = None;

    let file = std::fs::File::create(output_path)
        .map_err(|e| format!("create output: {e}"))?;
    let mut writer = std::io::BufWriter::new(file);

    // Expand AABB by the fade zone so it encompasses all voxels with nonzero opacity.
    let fade_margin = Vec3::splat(fade_outer + voxel_size);
    let geometry_aabb = Aabb::new(aabb.min - fade_margin, aabb.max + fade_margin);
    rkp_core::asset_file::write_rkp(
        &mut writer,
        octree.as_slice(),
        depth,
        voxel_size,
        voxel_count,
        geometry_aabb.min.to_array(),
        geometry_aabb.max.to_array(),
        &material_ids,
        &voxel_bytes,
        color_data.as_deref(),
        bone_data,
    )
    .map_err(|e| format!("write .rkp: {e}"))?;

    let file_size = std::fs::metadata(output_path).map(|m| m.len()).unwrap_or(0);

    // 6. Save skeleton sidecar
    let skeleton_path = if let Some(ref extraction) = skinning {
        let skel_path = output_path.with_extension("rkskel");
        let asset = rkf_animation::skeleton_asset::SkeletonAsset::with_normalization(
            extraction.skeleton.clone(),
            extraction.clips.clone(),
            norm.center.to_array(),
            norm.scale,
            norm.rotation_offset,
            norm.rotation_center.to_array(),
        );
        match rkf_animation::skeleton_asset::save_rkskel(&asset, &skel_path) {
            Ok(()) => {
                eprintln!("Saved skeleton: {} bones → {}", extraction.skeleton.bones.len(), skel_path.display());
                Some(skel_path)
            }
            Err(e) => {
                eprintln!("Failed to save .rkskel: {e}");
                None
            }
        }
    } else {
        None
    };

    eprintln!(
        "Splat import (per-voxel octree): wrote {} ({} voxels, {:.1} KiB)",
        output_path.display(), voxel_count, file_size as f64 / 1024.0,
    );

    Ok(ImportResult {
        aabb,
        total_bricks: voxel_count,
        lod_count: 1, // Octree is the LOD hierarchy, no separate levels.
        finest_voxel_size: voxel_size,
        file_size,
        skeleton_path,
    })
}
