//! Octree-based surface voxelization from a signed distance field.
//!
//! Top-down recursive: at each octree node, sample an SDF at 8 corners +
//! center to classify the region.
//!
//! * **EMPTY** if every sample's distance is greater than `extent / 2` —
//!   the surface can't be inside this node (SDFs are 1-Lipschitz).
//! * **INTERIOR** if every sample's distance is less than `-extent / 2` —
//!   the node is strictly inside the solid.
//! * **MIXED** otherwise. Subdivide until the finest level, then create
//!   a leaf if the node center lies inside the surface.
//!
//! Produces a [`SparseOctree`] with variable-depth leaves. Each leaf carries
//! a single [`LeafAttr`] directly: a prefiltered surface normal (SDF
//! gradient) and a material reference. No per-voxel opacity, no voxel_pool
//! indirection.
//!
//! The SDF convention matches rkf-core: negative = inside the surface,
//! positive = outside, zero = on the surface.
//!
//! # Relationship to the old opacity pipeline
//!
//! Before this refactor the builder took an opacity function and stored a
//! per-voxel f16 between 0 and 1. The shader skipped anything below 0.05
//! and broke on the first above-threshold hit for opaque materials, so the
//! f16 value itself was only used for transparent-material compositing —
//! which we've now made a per-material property. The opacity field was
//! vestigial and its fade band generated leaves the shader would then
//! throw away. This module is the replacement.

use glam::{UVec3, Vec3};
use rkf_core::Aabb;

use crate::brick_pool::{BrickPool, BRICK_DIM, BRICK_EMPTY, BRICK_LEVELS};
use crate::leaf_attr::LeafAttr;
use crate::leaf_attr_pool::LeafAttrPool;
use crate::sparse_octree::SparseOctree;

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
    pub grid_origin: Vec3,
}

/// Voxelize a signed distance function into a sparse octree.
///
/// `sdf_fn`: returns `(signed_distance, material_id)` at a world-space
/// position. Negative distance = inside the surface. The 1-Lipschitz
/// property of an SDF is what makes the coarse-level Empty/Interior
/// classifier provably correct — the input should be a true signed
/// distance, not an arbitrary scalar field that's merely sign-correct.
///
/// `aabb`: world-space bounding box of the object.
/// `base_voxel_size`: voxel size at the finest level.
pub fn voxelize_octree<F>(
    sdf_fn: F,
    aabb: &Aabb,
    base_voxel_size: f32,
    leaf_attr_pool: &mut LeafAttrPool,
    brick_pool: &mut BrickPool,
) -> Option<VoxelizeOctreeResult>
where
    F: Fn(Vec3) -> (f32, u16),
{
    let aabb_size = aabb.max - aabb.min;
    let max_dim = aabb_size.x.max(aabb_size.y).max(aabb_size.z);

    // Depth is the smallest power of 2 that covers the AABB in voxels.
    let voxels_needed = (max_dim / base_voxel_size).ceil().max(1.0) as u32;
    let depth = if voxels_needed <= 1 {
        1
    } else {
        (32 - (voxels_needed - 1).leading_zeros()) as u8
    };

    let mut octree = SparseOctree::new(depth, base_voxel_size);
    let mut voxel_count = 0u32;
    // Dedup keyed on the full LeafAttr value. Two spatial leaves with the
    // same packed (material + normal) share an entry, so flat-shaded
    // regions collapse via subtree dedup; curved surfaces pay one entry
    // per unique (material, normal) tuple.
    let mut attr_dedup: std::collections::HashMap<LeafAttr, u32> =
        std::collections::HashMap::new();
    let leaf_attr_slot_start = leaf_attr_pool.allocated_count();
    let mut brick_ids: Vec<u32> = Vec::new();

    // Center the octree on the AABB.
    let extent = octree.extent_world();
    let aabb_center = (aabb.min + aabb.max) * 0.5;
    let grid_origin = aabb_center - Vec3::splat(extent * 0.5);

    // Bricks terminate the octree `BRICK_LEVELS` levels above max_depth.
    // For a tree at or below that depth, the entire octree would degenerate
    // to a single brick — disable bricking and fall back to per-leaf.
    let brick_depth: Option<u8> = if depth > BRICK_LEVELS {
        Some(depth - BRICK_LEVELS)
    } else {
        None
    };

    subdivide_node(
        &sdf_fn,
        &mut octree,
        leaf_attr_pool,
        brick_pool,
        &mut voxel_count,
        &mut attr_dedup,
        &mut brick_ids,
        UVec3::ZERO,
        0,
        depth,
        brick_depth,
        grid_origin,
        extent,
        base_voxel_size,
    )?;

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
    let attrs_before_prefilter = attr_dedup.len();
    crate::prefilter::prefilter_octree_internals(
        &mut octree,
        leaf_attr_pool,
        brick_pool,
        &mut attr_dedup,
    );
    let attrs_after_prefilter = attr_dedup.len();
    if nodes_before_compact >= 10_000 {
        eprintln!(
            "[voxelize_octree] leaves={}  unique_attrs={}(+{} prefilter)  octree {} → compact {} → dedup {} ({:.1}× total)",
            voxel_count,
            attrs_before_prefilter,
            attrs_after_prefilter - attrs_before_prefilter,
            nodes_before_compact,
            nodes_after_compact,
            nodes_after_dedup,
            if nodes_after_dedup > 0 {
                nodes_before_compact as f64 / nodes_after_dedup as f64
            } else { 0.0 },
        );
    }

    Some(VoxelizeOctreeResult {
        octree,
        voxel_count,
        leaf_attr_unique_count: attr_dedup.len() as u32,
        leaf_attr_slot_start,
        brick_ids,
        grid_origin,
    })
}

#[allow(clippy::too_many_arguments)]
fn subdivide_node<F>(
    sdf_fn: &F,
    octree: &mut SparseOctree,
    leaf_attr_pool: &mut LeafAttrPool,
    brick_pool: &mut BrickPool,
    voxel_count: &mut u32,
    attr_dedup: &mut std::collections::HashMap<LeafAttr, u32>,
    brick_ids: &mut Vec<u32>,
    coord: UVec3,
    level: u8,
    max_depth: u8,
    brick_depth: Option<u8>,
    world_min: Vec3,
    node_extent: f32,
    base_voxel_size: f32,
) -> Option<()>
where
    F: Fn(Vec3) -> (f32, u16),
{
    let classification = classify_region(sdf_fn, world_min, node_extent);

    match classification {
        RegionClass::Empty => {
            // Root is EMPTY by default; nothing to write at other levels
            // either because the parent will collapse the 8 EMPTY children.
        }
        RegionClass::Interior => {
            octree.set_at_level(coord, level, crate::sparse_octree::INTERIOR_NODE);
        }
        RegionClass::Mixed => {
            // At brick_depth, terminate the tree with a brick instead of
            // continuing to subdivide. The brick's BRICK_DIM³ cells cover
            // exactly the leaves at max_depth — replacing BRICK_LEVELS
            // levels of tree descent with a single flat array lookup.
            if brick_depth == Some(level) {
                let brick_id = brick_pool.allocate()?;
                brick_ids.push(brick_id);
                let cell_size = base_voxel_size;
                for cz in 0..BRICK_DIM {
                    for cy in 0..BRICK_DIM {
                        for cx in 0..BRICK_DIM {
                            let cell_min = world_min + Vec3::new(
                                cx as f32 * cell_size,
                                cy as f32 * cell_size,
                                cz as f32 * cell_size,
                            );
                            let cell_center = cell_min + Vec3::splat(cell_size * 0.5);
                            let (d_center, material_id) = sdf_fn(cell_center);
                            if d_center > 0.0 {
                                // Cell outside the surface — leave as EMPTY.
                                continue;
                            }
                            // Cell inside: compute SDF-gradient normal.
                            let eps = cell_size * 0.5;
                            let (d_xp, _) = sdf_fn(cell_center + Vec3::new(eps, 0.0, 0.0));
                            let (d_xm, _) = sdf_fn(cell_center - Vec3::new(eps, 0.0, 0.0));
                            let (d_yp, _) = sdf_fn(cell_center + Vec3::new(0.0, eps, 0.0));
                            let (d_ym, _) = sdf_fn(cell_center - Vec3::new(0.0, eps, 0.0));
                            let (d_zp, _) = sdf_fn(cell_center + Vec3::new(0.0, 0.0, eps));
                            let (d_zm, _) = sdf_fn(cell_center - Vec3::new(0.0, 0.0, eps));
                            let grad = Vec3::new(
                                d_xp - d_xm,
                                d_yp - d_ym,
                                d_zp - d_zm,
                            );
                            let normal = if grad.length_squared() > 1e-12 {
                                grad.normalize()
                            } else {
                                Vec3::Y
                            };
                            let attr = LeafAttr::new(normal, material_id);
                            let leaf_attr_id = if let Some(&existing) = attr_dedup.get(&attr) {
                                existing
                            } else {
                                // Bump-only allocate — the asset's full attr
                                // range must stay contiguous so the scene
                                // manager's release (deallocate_range) frees
                                // exactly what we allocated. A plain
                                // `allocate()` would dip into the free list
                                // and break that invariant when other
                                // assets have been released concurrently.
                                let id = leaf_attr_pool.allocate_contiguous_bump(1)?;
                                *leaf_attr_pool.get_mut(id) = attr;
                                attr_dedup.insert(attr, id);
                                id
                            };
                            brick_pool.set_cell(brick_id, cx, cy, cz, leaf_attr_id);
                            *voxel_count += 1;
                        }
                    }
                }
                octree.set_at_level(coord, level, crate::sparse_octree::make_brick(brick_id));
                return Some(());
            }

            if level == max_depth {
                // At the finest level: sample the center. If it's inside
                // the surface, create a leaf with its SDF-gradient normal.
                let voxel_center = world_min + Vec3::splat(base_voxel_size * 0.5);
                let (d_center, material_id) = sdf_fn(voxel_center);
                if d_center > 0.0 {
                    // Center is outside — this corner of a Mixed region is
                    // not itself solid. Leave it EMPTY.
                    return Some(());
                }

                // Surface normal from SDF gradient via 6-tap central
                // differences. SDF gradient points OUTWARD (toward
                // increasing distance), which is exactly the surface
                // normal we want — no sign flip.
                let eps = base_voxel_size * 0.5;
                let (d_xp, _) = sdf_fn(voxel_center + Vec3::new(eps, 0.0, 0.0));
                let (d_xm, _) = sdf_fn(voxel_center - Vec3::new(eps, 0.0, 0.0));
                let (d_yp, _) = sdf_fn(voxel_center + Vec3::new(0.0, eps, 0.0));
                let (d_ym, _) = sdf_fn(voxel_center - Vec3::new(0.0, eps, 0.0));
                let (d_zp, _) = sdf_fn(voxel_center + Vec3::new(0.0, 0.0, eps));
                let (d_zm, _) = sdf_fn(voxel_center - Vec3::new(0.0, 0.0, eps));
                let grad = Vec3::new(d_xp - d_xm, d_yp - d_ym, d_zp - d_zm);
                let normal = if grad.length_squared() > 1e-12 {
                    grad.normalize()
                } else {
                    // Deep interior: gradient vanishes; the normal won't
                    // be visible there anyway, so +Y is a safe fallback.
                    Vec3::Y
                };

                let attr = LeafAttr::new(normal, material_id);
                let leaf_attr_id = if let Some(&existing) = attr_dedup.get(&attr) {
                    existing
                } else {
                    let id = leaf_attr_pool.allocate()?;
                    *leaf_attr_pool.get_mut(id) = attr;
                    attr_dedup.insert(attr, id);
                    id
                };
                octree.set_at_level(coord, level, crate::sparse_octree::make_leaf(leaf_attr_id));
                *voxel_count += 1;
            } else {
                let half = node_extent * 0.5;
                let child_voxels = 1u32 << (max_depth - level - 1);

                for octant in 0u32..8 {
                    let dx = octant & 1;
                    let dy = (octant >> 1) & 1;
                    let dz = (octant >> 2) & 1;

                    let child_min = world_min
                        + Vec3::new(dx as f32 * half, dy as f32 * half, dz as f32 * half);
                    let child_coord = UVec3::new(
                        coord.x + dx * child_voxels,
                        coord.y + dy * child_voxels,
                        coord.z + dz * child_voxels,
                    );

                    subdivide_node(
                        sdf_fn,
                        octree,
                        leaf_attr_pool,
                        brick_pool,
                        voxel_count,
                        attr_dedup,
                        brick_ids,
                        child_coord,
                        level + 1,
                        max_depth,
                        brick_depth,
                        child_min,
                        half,
                        base_voxel_size,
                    )?;
                }
            }
        }
    }

    Some(())
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum RegionClass {
    Empty,
    Interior,
    Mixed,
}

/// Classify a cubic region by sampling the SDF at 9 points (8 corners +
/// center). A 1-Lipschitz SDF means the surface can be at most `d` away
/// from any sample, so:
///
/// * `min(|d|) > extent / 2` and all samples outside (d > 0) → Empty.
/// * `min(|d|) > extent / 2` and all samples inside (d < 0) → Interior.
/// * Otherwise the surface may intersect the node → Mixed.
fn classify_region<F>(sdf_fn: &F, world_min: Vec3, extent: f32) -> RegionClass
where
    F: Fn(Vec3) -> (f32, u16),
{
    let threshold = extent * 0.5;
    let mut all_outside = true;
    let mut all_inside = true;

    for cz in 0..2u32 {
        for cy in 0..2u32 {
            for cx in 0..2u32 {
                let pos = world_min
                    + Vec3::new(
                        cx as f32 * extent,
                        cy as f32 * extent,
                        cz as f32 * extent,
                    );
                let (d, _) = sdf_fn(pos);
                if d <= threshold { all_outside = false; }
                if d >= -threshold { all_inside = false; }
            }
        }
    }

    let center = world_min + Vec3::splat(extent * 0.5);
    let (d_center, _) = sdf_fn(center);
    if d_center <= threshold { all_outside = false; }
    if d_center >= -threshold { all_inside = false; }

    if all_outside {
        RegionClass::Empty
    } else if all_inside {
        RegionClass::Interior
    } else {
        RegionClass::Mixed
    }
}

/// Convenience: voxelize a sphere into a sparse octree.
pub fn voxelize_sphere_octree(
    center: Vec3,
    radius: f32,
    material_id: u16,
    voxel_size: f32,
    leaf_attr_pool: &mut LeafAttrPool,
    brick_pool: &mut BrickPool,
) -> Option<VoxelizeOctreeResult> {
    let padding = voxel_size * 2.0;
    let aabb = Aabb {
        min: center - Vec3::splat(radius + padding),
        max: center + Vec3::splat(radius + padding),
    };

    let sdf_fn = |pos: Vec3| -> (f32, u16) {
        let d = (pos - center).length() - radius;
        (d, material_id)
    };

    voxelize_octree(sdf_fn, &aabb, voxel_size, leaf_attr_pool, brick_pool)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse_octree::INTERIOR_NODE;

    #[test]
    fn sphere_produces_brick_cells() {
        let mut attrs = LeafAttrPool::new(1_000_000);
        let mut bricks = BrickPool::new(10_000);
        let r = voxelize_sphere_octree(Vec3::ZERO, 0.5, 0, 0.1, &mut attrs, &mut bricks).unwrap();

        assert!(r.voxel_count > 0, "should populate cells for the sphere surface");
        assert!(!r.brick_ids.is_empty(), "surface should be encoded as bricks at brick_depth");
    }

    #[test]
    fn sphere_has_interior_nodes() {
        let mut attrs = LeafAttrPool::new(1_000_000);
        let mut bricks = BrickPool::new(10_000);
        let r = voxelize_sphere_octree(Vec3::ZERO, 3.0, 0, 0.1, &mut attrs, &mut bricks).unwrap();

        let ext = r.octree.extent();
        let mid = ext / 2;
        let val = r.octree.lookup(glam::UVec3::new(mid, mid, mid));
        assert_eq!(val, Some(INTERIOR_NODE), "large sphere should have interior at center");
    }

    #[test]
    fn empty_region_produces_no_voxels() {
        let mut attrs = LeafAttrPool::new(256);
        let mut bricks = BrickPool::new(64);
        let aabb = Aabb { min: Vec3::ZERO, max: Vec3::splat(1.0) };
        let r = voxelize_octree(|_| (1000.0, 0), &aabb, 0.1, &mut attrs, &mut bricks).unwrap();

        assert_eq!(r.voxel_count, 0);
        assert_eq!(r.brick_ids.len(), 0);
        assert_eq!(r.octree.leaf_count(), 0);
    }

    #[test]
    fn fully_interior_region_is_interior() {
        let mut attrs = LeafAttrPool::new(256);
        let mut bricks = BrickPool::new(64);
        let aabb = Aabb { min: Vec3::ZERO, max: Vec3::splat(0.05) };
        let r = voxelize_octree(|_| (-1000.0, 0), &aabb, 0.1, &mut attrs, &mut bricks).unwrap();

        assert_eq!(r.voxel_count, 0, "fully inside should collapse to INTERIOR");
        assert_eq!(r.brick_ids.len(), 0);
        assert_eq!(r.octree.as_slice()[0], INTERIOR_NODE);
    }

    #[test]
    fn leaf_attrs_carry_correct_material() {
        // Walk every leaf_attr this voxelize allocated and check material_primary.
        let mut attrs = LeafAttrPool::new(1_000_000);
        let mut bricks = BrickPool::new(10_000);
        let r = voxelize_sphere_octree(Vec3::ZERO, 0.3, 42, 0.1, &mut attrs, &mut bricks).unwrap();

        assert!(r.leaf_attr_unique_count > 0);
        for i in r.leaf_attr_slot_start..(r.leaf_attr_slot_start + r.leaf_attr_unique_count) {
            assert_eq!(attrs.get(i).material_primary, 42);
        }
    }

    #[test]
    fn sphere_normals_point_outward() {
        // Walk the brick nodes in the octree, expand each brick's 64 cells,
        // verify cell normals point outward from the sphere center.
        use crate::sparse_octree::{is_brick, brick_id as get_brick_id};

        let center = Vec3::ZERO;
        let radius = 0.5;
        let vs = 0.05;
        let mut attrs = LeafAttrPool::new(1_000_000);
        let mut bricks = BrickPool::new(10_000);
        let r = voxelize_sphere_octree(center, radius, 0, vs, &mut attrs, &mut bricks).unwrap();

        // Find brick nodes by walking the octree; for each, iterate cells.
        let mut checked = 0u32;
        let nodes = r.octree.as_slice().to_vec();
        let max_depth = r.octree.depth();
        let brick_depth = max_depth - BRICK_LEVELS;
        // Visit nodes recursively from root, tracking origin coord at each level.
        fn walk(
            nodes: &[u32],
            idx: usize,
            origin: glam::UVec3,
            level: u8,
            brick_depth: u8,
            visit: &mut impl FnMut(glam::UVec3, u32),
        ) {
            let node = nodes[idx];
            if is_brick(node) {
                visit(origin, get_brick_id(node));
                return;
            }
            if !crate::sparse_octree::is_branch(node) { return; }
            let children = node as usize;
            let cells_per_child = 1u32 << ((brick_depth + BRICK_LEVELS) - level - 1);
            for octant in 0u32..8 {
                let dx = octant & 1;
                let dy = (octant >> 1) & 1;
                let dz = (octant >> 2) & 1;
                let child_origin = glam::UVec3::new(
                    origin.x + dx * cells_per_child,
                    origin.y + dy * cells_per_child,
                    origin.z + dz * cells_per_child,
                );
                walk(nodes, children + octant as usize, child_origin, level + 1, brick_depth, visit);
            }
        }

        walk(&nodes, 0, glam::UVec3::ZERO, 0, brick_depth, &mut |brick_origin, bid| {
            for cz in 0..BRICK_DIM {
                for cy in 0..BRICK_DIM {
                    for cx in 0..BRICK_DIM {
                        let cell = bricks.get_cell(bid, cx, cy, cz);
                        if cell == BRICK_EMPTY { continue; }
                        let coord = glam::UVec3::new(
                            brick_origin.x + cx,
                            brick_origin.y + cy,
                            brick_origin.z + cz,
                        );
                        let cell_center = r.grid_origin + Vec3::new(
                            coord.x as f32 * vs + vs * 0.5,
                            coord.y as f32 * vs + vs * 0.5,
                            coord.z as f32 * vs + vs * 0.5,
                        );
                        let to_sphere_center = center - cell_center;
                        if to_sphere_center.length() < radius * 0.5 { continue; }
                        let expected = (cell_center - center).normalize();
                        let actual = attrs.get(cell).normal();
                        let dot = expected.dot(actual);
                        assert!(dot > 0.8,
                            "normal at {cell_center:?} should point outward: expected {expected:?}, got {actual:?}, dot={dot}",
                        );
                        checked += 1;
                    }
                }
            }
        });
        assert!(checked > 0, "should have checked at least one surface cell");
    }
}
