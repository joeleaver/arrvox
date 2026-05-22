// Diagnostic: bake tile (0,0,2) with default FBM, sample SDF at every
// cube corner in the bug region, and find cubes whose corners are
// MIXED solid/empty (should emit a triangle) but where the baked
// mesh emits NONE. That isolates the bug between
// (a) classification — corners are mis-classified, or
// (b) extract — corners are correct but the cube emission loop skips them.

use arvx_terrain::{bake_tile, FbmTerrainFn, Terrain, TerrainRegionSnapshot, TileKey};
use arvx_terrain::terrain_fn::TerrainFn;
use arvx_core::brick_pool::{BRICK_DIM, BRICK_EMPTY, BRICK_INTERIOR};
use arvx_core::mesh_cluster::MeshletCluster;
use arvx_core::mesh_extract::MeshVertex;
use arvx_core::sparse_octree::{is_brick, is_leaf, brick_id, leaf_slot, EMPTY_NODE, INTERIOR_NODE};
use glam::{IVec3, UVec3, Vec3};
use std::collections::HashSet;

/// Scan the WHOLE tile for cells with surface (LEAF or BRICK_SURFACE)
/// whose surrounding 2³ SN cubes emitted NO vertex. Those are the
/// holes — surface data exists but mesh extract skipped emission.
#[test]
#[ignore = "diagnostic — run manually with --ignored"]
fn scan_whole_tile_for_holes() {
    let t = Terrain::default();
    let vs = t.voxel_size_for_level(0);
    let fbm = FbmTerrainFn::default();
    let baked = bake_tile(
        TileKey::level0(0, 0, 2),
        vs,
        &fbm,
        &[],
        &TerrainRegionSnapshot::new(),
    )
    .expect("bake");
    let tile_origin = Vec3::new(0.0, 0.0, 128.0);

    let octree = &baked.artifact.octree;
    let bricks_flat: Vec<u32> = baked
        .artifact
        .brick_cells
        .iter()
        .flatten()
        .copied()
        .collect();
    let cells_per_axis = (64.0 / vs).round() as i32;

    // Collect all surface cells (any cell with non-empty LeafAttr).
    let mut surface_cells: HashSet<IVec3> = HashSet::new();
    for cz in 0..cells_per_axis {
        for cy in 0..cells_per_axis {
            for cx in 0..cells_per_axis {
                let coord = UVec3::new(cx as u32, cy as u32, cz as u32);
                let Some(node) = octree.lookup(coord) else { continue };
                let is_surface = if is_leaf(node) {
                    true
                } else if is_brick(node) {
                    let bid = brick_id(node);
                    let mask = BRICK_DIM - 1;
                    let lx = (cx as u32) & mask;
                    let ly = (cy as u32) & mask;
                    let lz = (cz as u32) & mask;
                    let flat = (lx + ly * BRICK_DIM + lz * BRICK_DIM * BRICK_DIM) as usize;
                    let v = bricks_flat[bid as usize * (BRICK_DIM * BRICK_DIM * BRICK_DIM) as usize + flat];
                    v != BRICK_EMPTY && v != BRICK_INTERIOR
                } else {
                    false
                };
                if is_surface {
                    surface_cells.insert(IVec3::new(cx, cy, cz));
                }
            }
        }
    }
    eprintln!("Tile (0,0,2): {} surface cells in octree", surface_cells.len());

    // Bucket all LOD-0 vertices by their CELL coord (vertex world →
    // tile-local → cell). A cell "covered" by vertices means SN cubes
    // adjacent to it emitted something.
    let verts: &[MeshVertex] = bytemuck::cast_slice(&baked.mesh.vertices);
    let indices: &[u32] = bytemuck::cast_slice(&baked.mesh.indices);
    let clusters: &[MeshletCluster] = bytemuck::cast_slice(&baked.mesh.clusters);
    let mut covered: HashSet<IVec3> = HashSet::new();
    for c in clusters.iter().filter(|c| c.lod_level == 0) {
        let lo = c.index_offset as usize;
        let hi = lo + c.index_count as usize;
        for i in lo..hi {
            if i >= indices.len() { break; }
            let vi = indices[i] as usize;
            if vi >= verts.len() { continue; }
            let p = Vec3::from_array(verts[vi].local_pos);
            let local = p - tile_origin;
            // Vertex lands inside an SN cube; mark all 8 nearby cells as covered.
            let cell = (local / vs).floor().as_ivec3();
            for dz in 0..=1 {
                for dy in 0..=1 {
                    for dx in 0..=1 {
                        covered.insert(cell - IVec3::new(dx, dy, dz));
                    }
                }
            }
        }
    }
    eprintln!("Covered cells (within 1 cell of any LOD-0 vertex): {}", covered.len());

    // Surface cells with no nearby LOD-0 vertex are HOLES.
    let holes: Vec<IVec3> = surface_cells
        .iter()
        .copied()
        .filter(|c| !covered.contains(c))
        .collect();
    eprintln!("\n## HOLE cells (surface in octree, no LOD-0 vertex within 1 cell): {}", holes.len());

    // Bucket holes by 4m region to find cluster-sized holes.
    let mut hole_buckets: std::collections::HashMap<IVec3, u32> =
        std::collections::HashMap::new();
    for h in &holes {
        let world = tile_origin + h.as_vec3() * vs;
        let bx = (world.x / 4.0).floor() as i32;
        let by = (world.y / 4.0).floor() as i32;
        let bz = (world.z / 4.0).floor() as i32;
        *hole_buckets.entry(IVec3::new(bx, by, bz)).or_insert(0) += 1;
    }
    let mut sorted: Vec<_> = hole_buckets.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    eprintln!("\nTop 20 hole clusters (4m³ buckets):");
    for (b, count) in sorted.iter().take(20) {
        eprintln!(
            "  world ({:6.2},{:6.2},{:6.2}) - ({:6.2},{:6.2},{:6.2})  {count} hole cells",
            (b.x * 4) as f32, (b.y * 4) as f32, (b.z * 4) as f32,
            ((b.x + 1) * 4) as f32, ((b.y + 1) * 4) as f32, ((b.z + 1) * 4) as f32,
        );
    }
    let _ = fbm;
}

#[test]
#[ignore = "diagnostic — run manually with --ignored"]
fn dissect_cells_in_bug_region() {
    let t = Terrain::default();
    let vs = t.voxel_size_for_level(0);
    let fbm = FbmTerrainFn::default();
    let baked = bake_tile(
        TileKey::level0(0, 0, 2),
        vs,
        &fbm,
        &[],
        &TerrainRegionSnapshot::new(),
    )
    .expect("bake");

    let tile_origin = Vec3::new(0.0, 0.0, 128.0);

    // Bug area from user K-key dump: world ~(15, 14, 162). Sample a
    // 4m³ region around it.
    // User flew INTO a hole and dumped camera at this exact pos.
    let bug_center = Vec3::new(23.34, 15.84, 159.78);
    let region_half = Vec3::splat(1.0);
    let region_lo = bug_center - region_half;
    let region_hi = bug_center + region_half;

    // Convert region world coords → tile-local cell coords.
    let cell_lo = ((region_lo - tile_origin) / vs).floor().as_ivec3();
    let cell_hi = ((region_hi - tile_origin) / vs).ceil().as_ivec3();
    let cells_per_axis = (64.0 / vs).round() as i32;
    eprintln!("Bug region world: {bug_center:?} ± {region_half:?}");
    eprintln!("Cell range (tile-local): {cell_lo:?} .. {cell_hi:?}  (vs={vs})");

    // 1) Sample the unscaled SDF at each cell center within the region.
    //    Classification using the unscaled SDF should match the
    //    bake's brick/cell classifier (which sees the same SDF
    //    scaled by the Lipschitz overestimate — sign-equal).
    eprintln!("\n## ground-truth SDF sample at each cell center");
    let mut surface_cells_truth: HashSet<IVec3> = HashSet::new();
    for cz in cell_lo.z..cell_hi.z {
        for cy in cell_lo.y..cell_hi.y {
            for cx in cell_lo.x..cell_hi.x {
                if cx < 0 || cy < 0 || cz < 0
                    || cx >= cells_per_axis || cy >= cells_per_axis || cz >= cells_per_axis
                {
                    continue;
                }
                let world = tile_origin + Vec3::new(
                    cx as f32 + 0.5,
                    cy as f32 + 0.5,
                    cz as f32 + 0.5,
                ) * vs;
                let sample = fbm.sample(
                    TileKey::level0(0, 0, 2),
                    world - tile_origin,
                    vs,
                );
                // Cube emits triangles when any cube spanning this cell
                // has corners on both sides of zero. A simple per-cell
                // proxy: cell is "near-surface" if |sd| ≤ vs*sqrt(3)/2.
                let near_surface = sample.sd.abs() <= vs * 3.0_f32.sqrt() * 0.5;
                if near_surface {
                    surface_cells_truth.insert(IVec3::new(cx, cy, cz));
                }
            }
        }
    }
    eprintln!("  {} cells classified as near-surface by SDF ground truth", surface_cells_truth.len());

    // 2) Walk the baked octree and classify each cell as LEAF / BRICK
    //    cell value / EMPTY / INTERIOR.
    eprintln!("\n## baked octree classification");
    let octree = &baked.artifact.octree;
    let depth = octree.depth();
    let nodes = octree.as_slice();
    let bricks_flat: Vec<u32> = baked
        .artifact
        .brick_cells
        .iter()
        .flatten()
        .copied()
        .collect();

    let mut leaf_count = 0;
    let mut brick_surf_count = 0;
    let mut brick_empty_count = 0;
    let mut brick_interior_count = 0;
    let mut interior_node_count = 0;
    let mut empty_node_count = 0;
    let mut surface_cells_baked: HashSet<IVec3> = HashSet::new();
    let mut mismatched_cells: Vec<(IVec3, f32, &'static str)> = Vec::new();

    for cz in cell_lo.z..cell_hi.z {
        for cy in cell_lo.y..cell_hi.y {
            for cx in cell_lo.x..cell_hi.x {
                if cx < 0 || cy < 0 || cz < 0
                    || cx >= cells_per_axis || cy >= cells_per_axis || cz >= cells_per_axis
                {
                    continue;
                }
                let coord = UVec3::new(cx as u32, cy as u32, cz as u32);
                let node = match octree.lookup(coord) {
                    Some(n) => n,
                    None => continue,
                };
                let classification = if node == EMPTY_NODE {
                    empty_node_count += 1;
                    "EMPTY_NODE"
                } else if node == INTERIOR_NODE {
                    interior_node_count += 1;
                    surface_cells_baked.insert(IVec3::new(cx, cy, cz));
                    "INTERIOR_NODE"
                } else if is_leaf(node) {
                    leaf_count += 1;
                    surface_cells_baked.insert(IVec3::new(cx, cy, cz));
                    "LEAF"
                } else if is_brick(node) {
                    let bid = brick_id(node);
                    let mask = BRICK_DIM - 1;
                    let lx = (cx as u32) & mask;
                    let ly = (cy as u32) & mask;
                    let lz = (cz as u32) & mask;
                    let cells_per_brick = (BRICK_DIM * BRICK_DIM * BRICK_DIM) as usize;
                    let flat = (lx + ly * BRICK_DIM + lz * BRICK_DIM * BRICK_DIM) as usize;
                    let v = bricks_flat[bid as usize * cells_per_brick + flat];
                    if v == BRICK_EMPTY {
                        brick_empty_count += 1;
                        "BRICK_EMPTY"
                    } else if v == BRICK_INTERIOR {
                        brick_interior_count += 1;
                        surface_cells_baked.insert(IVec3::new(cx, cy, cz));
                        "BRICK_INTERIOR"
                    } else {
                        brick_surf_count += 1;
                        surface_cells_baked.insert(IVec3::new(cx, cy, cz));
                        "BRICK_SURFACE"
                    }
                } else {
                    "BRANCH"
                };
                let _ = classification;

                // Mismatch: ground truth says near-surface but baked
                // is EMPTY/INTERIOR (no surface emit).
                let truth_near_surface = surface_cells_truth.contains(&IVec3::new(cx, cy, cz));
                let baked_solid_surface = match classification {
                    "LEAF" | "BRICK_SURFACE" => true,
                    _ => false,
                };
                let world = tile_origin + Vec3::new(
                    cx as f32 + 0.5,
                    cy as f32 + 0.5,
                    cz as f32 + 0.5,
                ) * vs;
                let sd = fbm.sample(TileKey::level0(0, 0, 2), world - tile_origin, vs).sd;
                if truth_near_surface && !baked_solid_surface {
                    mismatched_cells.push((IVec3::new(cx, cy, cz), sd, classification));
                }
            }
        }
    }
    eprintln!(
        "  LEAF={} BRICK_SURFACE={} BRICK_EMPTY={} BRICK_INTERIOR={} INTERIOR_NODE={} EMPTY_NODE={}",
        leaf_count, brick_surf_count, brick_empty_count, brick_interior_count,
        interior_node_count, empty_node_count,
    );
    eprintln!("\n## mismatched cells (ground-truth says near-surface but bake says non-surface):");
    eprintln!("  count: {} / {} near-surface cells", mismatched_cells.len(), surface_cells_truth.len());
    for (c, sd, cls) in mismatched_cells.iter().take(20) {
        eprintln!("  cell ({:3},{:3},{:3}) sd={:+7.4} -> baked as {cls}", c.x, c.y, c.z, sd);
    }

    // 3) Look at the LOD-0 cluster vertex coverage in the same region.
    let verts: &[MeshVertex] = bytemuck::cast_slice(&baked.mesh.vertices);
    let indices: &[u32] = bytemuck::cast_slice(&baked.mesh.indices);
    let clusters: &[MeshletCluster] = bytemuck::cast_slice(&baked.mesh.clusters);
    let mut lod0_verts_in_region: HashSet<u32> = HashSet::new();
    for c in clusters.iter().filter(|c| c.lod_level == 0) {
        let lo = c.index_offset as usize;
        let hi = lo + c.index_count as usize;
        for i in lo..hi {
            if i >= indices.len() { break; }
            let vi = indices[i];
            if (vi as usize) >= verts.len() { continue; }
            let p = Vec3::from_array(verts[vi as usize].local_pos);
            if p.x >= region_lo.x && p.x <= region_hi.x
                && p.y >= region_lo.y && p.y <= region_hi.y
                && p.z >= region_lo.z && p.z <= region_hi.z
            {
                lod0_verts_in_region.insert(vi);
            }
        }
    }
    eprintln!("\n## LOD-0 vertex coverage in bug region: {} unique vertices", lod0_verts_in_region.len());

    // For each LOD-0 cluster touching the region, dump declared AABB
    // vs actual vertex extent.
    // What MATERIAL does the bake assign at this xz? Bug area on a
    // steep slope ⇒ rock material; if the project's rock palette
    // entry has opacity < 0.99 the mesh fragment shader discards
    // every rock fragment → cluster-sized "holes".
    let mat_sample = fbm.sample(
        TileKey::level0(0, 0, 2),
        bug_center - tile_origin,
        vs,
    );
    eprintln!(
        "\n## MATERIAL at bug center: primary={} secondary={} blend={}",
        mat_sample.primary_mat, mat_sample.secondary_mat, mat_sample.blend,
    );
    eprintln!("   (FBM defaults: 1=grass 2=sand 3=rock 4=snow)");

    // Sample materials at each LOD-0 vertex in the region to
    // confirm the dominant material in the hole.
    let mut mat_hist: std::collections::HashMap<u16, u32> =
        std::collections::HashMap::new();
    for &vi in &lod0_verts_in_region {
        let v = verts[vi as usize];
        let attr = baked.artifact.leaf_attrs.get(v.leaf_attr_id as usize);
        if let Some(a) = attr {
            *mat_hist.entry(a.material_primary).or_insert(0) += 1;
        }
    }
    eprintln!("\n## material histogram of LOD-0 vertices in bug region:");
    let mut sorted: Vec<_> = mat_hist.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    for (mat, count) in sorted {
        eprintln!("   material {mat}: {count} vertices");
    }

    eprintln!("\n## LOD-0 clusters with any vertex in the bug region:");
    for (ci, c) in clusters.iter().enumerate().filter(|(_, c)| c.lod_level == 0) {
        let lo = c.index_offset as usize;
        let hi = lo + c.index_count as usize;
        let mut hit_count = 0u32;
        let mut actual_min = Vec3::splat(f32::INFINITY);
        let mut actual_max = Vec3::splat(f32::NEG_INFINITY);
        for i in lo..hi {
            if i >= indices.len() { break; }
            let vi = indices[i] as usize;
            if vi >= verts.len() { continue; }
            let p = Vec3::from_array(verts[vi].local_pos);
            actual_min = actual_min.min(p);
            actual_max = actual_max.max(p);
            if p.x >= region_lo.x && p.x <= region_hi.x
                && p.y >= region_lo.y && p.y <= region_hi.y
                && p.z >= region_lo.z && p.z <= region_hi.z
            {
                hit_count += 1;
            }
        }
        if hit_count == 0 {
            continue;
        }
        let declared_min = Vec3::from_array(c.aabb_min);
        let declared_max = Vec3::from_array(c.aabb_max);
        let aabb_match =
            (actual_min - declared_min).length() < 0.01
            && (actual_max - declared_max).length() < 0.01;
        eprintln!(
            "  cluster {ci:5} flags={:#x} idx={:6} cluster_error={:.6}  declared=[{:5.2},{:5.2},{:5.2}]..[{:5.2},{:5.2},{:5.2}] actual=[{:5.2},{:5.2},{:5.2}]..[{:5.2},{:5.2},{:5.2}]  bug_verts={}  aabb_match={}",
            c.flags, c.index_count, c.cluster_error,
            declared_min.x, declared_min.y, declared_min.z,
            declared_max.x, declared_max.y, declared_max.z,
            actual_min.x, actual_min.y, actual_min.z,
            actual_max.x, actual_max.y, actual_max.z,
            hit_count, aabb_match,
        );
    }
}
