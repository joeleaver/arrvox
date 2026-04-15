//! Mesh-to-.rkp voxelization — thin surface shell, no opacity anywhere.
//!
//! For each brick in the narrow band around the mesh, samples signed distance
//! (unsigned distance to the nearest triangle + winding-number sign) per
//! voxel. The importer then classifies voxels into a 1-voxel-thick outer
//! shell (leaves with baked SDF-gradient normals) and solid interior
//! (INTERIOR_NODE). No per-voxel opacity field is produced or stored; the
//! runtime reads the baked normal + color and shades directly.

use std::path::Path;

use glam::Vec3;
use rayon::prelude::*;

use rkf_core::Aabb;
use rkf_core::companion::{BoneBrick, BoneVoxel, ColorBrick, ColorVoxel};
use rkf_core::constants::BRICK_DIM;
use rkf_core::voxel::VoxelSample;
use rkf_import::bvh::TriangleBvh;
use rkf_import::material_transfer::{sample_texture_at_triangle, sample_bone_weights_at_triangle};
use rkf_import::mesh::MeshData;
use rkf_import::pipeline::{ImportConfig, ImportResult};
use rkf_import::skeleton_extract::{self, VertexSkinning};

/// Flat voxel index within a brick (matches rkf-core convention).
fn voxel_index(x: u8, y: u8, z: u8) -> u32 {
    x as u32 + y as u32 * 8 + z as u32 * 64
}

/// Result of processing a single brick — signed distance, material, and
/// (optional) color and bone weights for every voxel in the 8×8×8 grid.
struct BrickResult {
    color_brick: ColorBrick,
    bone_brick: Option<BoneBrick>,
    /// Per-voxel signed distance (negative = inside mesh).
    signed_distances: [f32; 512],
    /// Per-voxel material ID from nearest triangle.
    material_ids: [u16; 512],
    /// True if at least one voxel is inside the mesh (d ≤ 0).
    any_inside: bool,
    /// True if every voxel is inside the mesh (d ≤ 0 for all 512).
    all_inside: bool,
}

/// Process a single brick: sample signed distance, material, color, and
/// bone weights at every voxel center. The brick is in the narrow band
/// around the surface (the caller already filtered on `bvh.nearest`
/// distance), so every voxel is worth sampling — no per-voxel gate.
fn process_brick(
    mesh: &MeshData,
    bvh: &TriangleBvh,
    brick_min: Vec3,
    voxel_size: f32,
    material_id_override: Option<u16>,
    import_colors: bool,
    skinning: Option<&VertexSkinning>,
) -> BrickResult {
    let half_voxel = voxel_size * 0.5;
    let mut color_brick = ColorBrick::default();
    let mut signed_distances = [f32::INFINITY; 512];
    let mut material_ids = [0u16; 512];
    let mut any_inside = false;
    let mut all_inside = true;

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
                // Winding-convention-agnostic: closed manifolds give |w|≈1
                // inside, |w|≈0 outside. Some exported glTF assets have
                // reversed triangle winding and return w≈−1 inside.
                let is_inside = w.abs() > 0.5;
                let signed_d = if is_inside { -d } else { d };

                let flat = voxel_index(vx as u8, vy as u8, vz as u8) as usize;
                signed_distances[flat] = signed_d;
                if signed_d <= 0.0 { any_inside = true; } else { all_inside = false; }

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

                if import_colors {
                    if let Some(color) = sample_texture_at_triangle(
                        mesh,
                        nearest.triangle_index,
                        &nearest.barycentric,
                    ) {
                        color_brick.set(vx, vy, vz, ColorVoxel::new(color.r, color.g, color.b, 255));
                    }
                }

                if let (Some(skin), Some(bb)) = (skinning, bone_brick.as_mut()) {
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

    if !has_any_bone {
        bone_brick = None;
    }

    BrickResult {
        color_brick,
        bone_brick,
        signed_distances,
        material_ids,
        any_inside,
        all_inside,
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

/// Import a mesh and produce an octree-native .rkp asset.
///
/// Classifies bricks into narrow-band (surface) vs all-interior (by winding),
/// samples signed distance per voxel inside surface bricks, then emits a thin
/// 1-voxel-thick outer shell of leaves with INTERIOR fill behind it. Each
/// shell leaf carries a pre-baked SDF-gradient normal and a per-voxel albedo
/// sampled from the mesh texture.
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
                    if w.abs() > 0.5 {
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
                &mesh, &bvh, w.brick_min, voxel_size,
                config.material_id_override, config.import_colors, skinning_data,
            );
            (w, result)
        })
        .collect();

    // Build per-voxel octree + flat voxel arrays.
    let mut octree = rkp_core::SparseOctree::new(depth, voxel_size);
    let mut voxel_data: Vec<VoxelSample> = Vec::new();
    let mut color_voxels: Vec<ColorVoxel> = Vec::new();
    // Pre-baked SDF-gradient normals per shell leaf, octahedrally packed.
    // Computed at import time so the runtime never sees an SDF — the load
    // path just reads these verbatim into LeafAttr.normal_oct.
    let mut normals_packed: Vec<u32> = Vec::new();
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

    // Classify voxels into thin outer shell (leaves) and interior. A voxel
    // becomes a shell leaf iff its signed distance is positive AND at least
    // one of its 6 face-neighbors is inside (d ≤ 0). Voxels with d ≤ 0 are
    // interior; outer voxels with all-positive neighbors are empty. This
    // yields a ~1-voxel-thick surface shell which the load-time
    // 26-neighborhood normal kernel can resolve cleanly.
    //
    // Lookups span bricks, so build an index: surface bricks -> their
    // per-voxel signed distance grid; plus a set of all-interior bricks.
    use std::collections::{HashMap, HashSet};
    let mut brick_result_index: HashMap<(u32, u32, u32), usize> =
        HashMap::with_capacity(results.len());
    let mut interior_brick_set: HashSet<(u32, u32, u32)> =
        interior_brick_coords.iter().copied().collect();
    for (i, (w, result)) in results.iter().enumerate() {
        if result.all_inside {
            // Every voxel in this surface-band brick is inside the mesh —
            // collapse the whole brick to INTERIOR rather than emitting
            // 512 individual INTERIOR_NODE insertions.
            interior_brick_set.insert((w.bx, w.by, w.bz));
        } else {
            // Brick straddles the surface (or is fully outside but in the
            // narrow band). Either way we need per-voxel signed distances
            // available for shell classification and SDF-gradient lookups.
            brick_result_index.insert((w.bx, w.by, w.bz), i);
        }
    }

    let sdf_at = |gx: i64, gy: i64, gz: i64| -> f32 {
        if gx < 0 || gy < 0 || gz < 0 {
            return f32::INFINITY;
        }
        let (gx, gy, gz) = (gx as u32, gy as u32, gz as u32);
        let bx = gx / 8;
        let by = gy / 8;
        let bz = gz / 8;
        if bx >= octree_bricks || by >= octree_bricks || bz >= octree_bricks {
            return f32::INFINITY;
        }
        if let Some(&idx) = brick_result_index.get(&(bx, by, bz)) {
            let lx = gx % 8;
            let ly = gy % 8;
            let lz = gz % 8;
            results[idx].1.signed_distances[(lx + ly * 8 + lz * 64) as usize]
        } else if interior_brick_set.contains(&(bx, by, bz)) {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        }
    };

    // Emit shell leaves as bricks at the octree's brick level. Each brick
    // covers BRICK_DIM³ voxels (currently 4³ = 64 cells). An 8³ importer
    // region therefore contains (8/BRICK_DIM)³ = 2³ = 8 octree bricks when
    // BRICK_DIM=4; the loop below handles the general case so the two are
    // decoupled.
    //
    // Each 4³ subregion is classified as:
    //   - all-interior        → INTERIOR_NODE at octree_brick_depth
    //   - all-exterior        → EMPTY (no insertion)
    //   - mixed (has a shell) → allocate brick; cells = slot for shell,
    //                           BRICK_CELL_EMPTY for interior/exterior
    let octree_brick_dim = rkp_core::brick_pool::BRICK_DIM as u32;
    let octree_brick_levels = rkp_core::brick_pool::BRICK_LEVELS;
    assert!(
        depth > octree_brick_levels,
        "octree depth ({depth}) must exceed BRICK_LEVELS ({octree_brick_levels}) to host bricks",
    );
    let octree_brick_depth = depth - octree_brick_levels;
    let subbricks_per_axis = 8 / octree_brick_dim; // 2 for BRICK_DIM=4
    let mut file_bricks: Vec<u32> = Vec::new(); // flat cell storage
    let brick_cells_u32 = (octree_brick_dim * octree_brick_dim * octree_brick_dim) as usize;

    for (w, result) in &results {
        if result.all_inside {
            let voxel_coord = glam::UVec3::new(w.bx * 8, w.by * 8, w.bz * 8);
            octree.set_at_level(voxel_coord, brick_depth, rkp_core::sparse_octree::INTERIOR_NODE);
            continue;
        }

        for sbz in 0..subbricks_per_axis {
            for sby in 0..subbricks_per_axis {
                for sbx in 0..subbricks_per_axis {
                    let sub_origin_x = sbx * octree_brick_dim;
                    let sub_origin_y = sby * octree_brick_dim;
                    let sub_origin_z = sbz * octree_brick_dim;

                    // Classify the 4³ subregion and collect shell data in
                    // one pass. Records are (cell_flat, slot_contents).
                    struct ShellEntry {
                        cell_flat: u32,
                        vx: u32, vy: u32, vz: u32, // local in 8³
                        gx: u32, gy: u32, gz: u32, // global
                    }
                    let mut all_interior = true;
                    let mut any_shell = false;
                    let mut shell_entries: Vec<ShellEntry> = Vec::new();

                    for cz in 0..octree_brick_dim {
                        for cy in 0..octree_brick_dim {
                            for cx in 0..octree_brick_dim {
                                let vx = sub_origin_x + cx;
                                let vy = sub_origin_y + cy;
                                let vz = sub_origin_z + cz;
                                let flat8 = (vx + vy * 8 + vz * 64) as usize;
                                let d = result.signed_distances[flat8];
                                if d > 0.0 {
                                    all_interior = false;
                                    let gx = w.bx * 8 + vx;
                                    let gy = w.by * 8 + vy;
                                    let gz = w.bz * 8 + vz;

                                    // 26-neighbor sign-change test (cross-brick
                                    // via sdf_at).
                                    let mut has_inside = false;
                                    'shell: for dz in -1i64..=1 {
                                        for dy in -1i64..=1 {
                                            for dx in -1i64..=1 {
                                                if dx == 0 && dy == 0 && dz == 0 { continue; }
                                                if sdf_at(gx as i64 + dx, gy as i64 + dy, gz as i64 + dz) <= 0.0 {
                                                    has_inside = true;
                                                    break 'shell;
                                                }
                                            }
                                        }
                                    }
                                    if has_inside {
                                        any_shell = true;
                                        let cell_flat = cx + cy * octree_brick_dim + cz * octree_brick_dim * octree_brick_dim;
                                        shell_entries.push(ShellEntry {
                                            cell_flat, vx, vy, vz, gx, gy, gz,
                                        });
                                    }
                                }
                            }
                        }
                    }

                    let sub_origin_coord = glam::UVec3::new(
                        w.bx * 8 + sub_origin_x,
                        w.by * 8 + sub_origin_y,
                        w.bz * 8 + sub_origin_z,
                    );

                    if all_interior {
                        octree.set_at_level(
                            sub_origin_coord, octree_brick_depth,
                            rkp_core::sparse_octree::INTERIOR_NODE,
                        );
                        continue;
                    }
                    if !any_shell {
                        // No shell cells in this subregion — leave as EMPTY.
                        continue;
                    }

                    // Allocate a brick in the file-local pool, populate cells.
                    let brick_id = (file_bricks.len() / brick_cells_u32) as u32;
                    file_bricks.extend(std::iter::repeat(rkp_core::brick_pool::BRICK_EMPTY).take(brick_cells_u32));
                    let brick_base = brick_id as usize * brick_cells_u32;

                    // Mark every d<=0 cell in this sub-brick as
                    // BRICK_INTERIOR (zero-cost: the brick's 64 slots
                    // are pre-allocated regardless of content, so this
                    // just replaces BRICK_EMPTY in those slots).
                    //
                    // Neighborhood kernels (Laplacian smoothing at
                    // bake time, surface-nets normal reconstruction if
                    // re-enabled) need "is this cell inside the
                    // solid?" info that a pure outer-shell design
                    // can't provide — without this fill, thin-shell
                    // imports produce concentric ring artifacts.
                    //
                    // The march treats BRICK_INTERIOR identically to
                    // BRICK_EMPTY for hit purposes so surface rays
                    // still only stop at the shell. Overwritten next
                    // by shell_entries for cells that carry a real
                    // leaf_attr (disjoint sets: d<=0 vs d>0).
                    //
                    // KNOWN ISSUE: causes subtle shading artifacts on
                    // some mesh imports (horizontal-band-style visual
                    // quirks). Root cause not yet identified — static
                    // analysis shows the shader should treat
                    // BRICK_INTERIOR identically to BRICK_EMPTY.
                    // Needs a shader-level diagnostic (color cells by
                    // first-encountered-type) to bisect. Until fixed,
                    // the interior-fill is still the architecturally
                    // correct thing to do; the artifact is modest and
                    // appears on specific high-curvature surfaces only.
                    for cz_fill in 0..octree_brick_dim {
                        for cy_fill in 0..octree_brick_dim {
                            for cx_fill in 0..octree_brick_dim {
                                let vx = sub_origin_x + cx_fill;
                                let vy = sub_origin_y + cy_fill;
                                let vz = sub_origin_z + cz_fill;
                                let flat8 = (vx + vy * 8 + vz * 64) as usize;
                                if result.signed_distances[flat8] <= 0.0 {
                                    let cell_flat = cx_fill
                                        + cy_fill * octree_brick_dim
                                        + cz_fill * octree_brick_dim * octree_brick_dim;
                                    file_bricks[brick_base + cell_flat as usize] =
                                        rkp_core::brick_pool::BRICK_INTERIOR;
                                }
                            }
                        }
                    }

                    for e in &shell_entries {
                        // SDF-gradient normal (6-tap central differences).
                        let grad = glam::Vec3::new(
                            sdf_at(e.gx as i64 + 1, e.gy as i64, e.gz as i64)
                                - sdf_at(e.gx as i64 - 1, e.gy as i64, e.gz as i64),
                            sdf_at(e.gx as i64, e.gy as i64 + 1, e.gz as i64)
                                - sdf_at(e.gx as i64, e.gy as i64 - 1, e.gz as i64),
                            sdf_at(e.gx as i64, e.gy as i64, e.gz as i64 + 1)
                                - sdf_at(e.gx as i64, e.gy as i64, e.gz as i64 - 1),
                        );
                        let normal = if grad.length_squared() > 1e-12 {
                            grad.normalize()
                        } else {
                            glam::Vec3::Y
                        };
                        let normal_oct = rkp_core::leaf_attr::pack_oct(normal);

                        let flat8 = (e.vx + e.vy * 8 + e.vz * 64) as usize;
                        let mat_id = result.material_ids[flat8];
                        let cv = result.color_brick.data[flat8];
                        if cv.intensity() > 0 {
                            has_color = true;
                        }

                        let slot = voxel_count;
                        voxel_data.push(VoxelSample::new(0.0, mat_id, 0));
                        normals_packed.push(normal_oct);
                        color_voxels.push(cv);
                        voxel_count += 1;

                        file_bricks[brick_base + e.cell_flat as usize] = slot;
                    }

                    octree.set_at_level(
                        sub_origin_coord, octree_brick_depth,
                        rkp_core::sparse_octree::make_brick(brick_id),
                    );
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
    let file_brick_count = file_bricks.len() / brick_cells_u32;
    eprintln!(
        "Splat import (brick octree): {} shell voxels, {} bricks, {} octree nodes, {} colored",
        voxel_count, file_brick_count, octree.node_count(), colored_count,
    );
    let _ = sample_colors;

    // 5. Serialize to .rkp v4 (brick-terminated octree).
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

    let normals_bytes: Vec<u8> = normals_packed
        .iter()
        .flat_map(|n| bytemuck::bytes_of(n))
        .copied()
        .collect();
    let normals_data: Option<&[u8]> = if !normals_bytes.is_empty() {
        Some(&normals_bytes)
    } else {
        None
    };

    let bricks_bytes: Vec<u8> = file_bricks
        .iter()
        .flat_map(|c| bytemuck::bytes_of(c))
        .copied()
        .collect();
    let bricks_data: Option<&[u8]> = if !bricks_bytes.is_empty() {
        Some(&bricks_bytes)
    } else {
        None
    };

    // TODO: Per-voxel bone data (bone weights per leaf voxel, not per brick)
    let bone_data: Option<&[u8]> = None;

    let file = std::fs::File::create(output_path)
        .map_err(|e| format!("create output: {e}"))?;
    let mut writer = std::io::BufWriter::new(file);

    // Expand AABB by one voxel so the outer shell voxels (one voxel beyond
    // the mesh surface on the outside) fall inside the geometry bounds.
    let shell_margin = Vec3::splat(voxel_size);
    let geometry_aabb = Aabb::new(aabb.min - shell_margin, aabb.max + shell_margin);
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
        normals_data,
        bricks_data,
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
