use super::*;
use glam::{UVec3, Vec3};
use crate::Aabb;
use crate::brick_pool::{BrickPool, BRICK_DIM, BRICK_LEVELS};
use crate::leaf_attr_pool::LeafAttrPool;
use crate::sparse_octree::INTERIOR_NODE;


/// Wrap a per-point SDF (5-tuple, no analytic gradient) as a batched
/// callback for the tests. The `None` gradient drives the voxelizer's
/// 6-tap finite-difference path — the analytic path is exercised by the
/// `analytic_gradient_*` tests below with an explicit `Some(grad)`.
fn batched<Fp>(f: Fp) -> impl Fn(&[Vec3]) -> Vec<(f32, u16, u16, u8, u32, Option<Vec3>)>
where
    Fp: Fn(Vec3) -> (f32, u16, u16, u8, u32),
{
    move |positions: &[Vec3]| {
        positions
            .iter()
            .map(|p| {
                let (d, a, b, c, e) = f(*p);
                (d, a, b, c, e, None)
            })
            .collect()
    }
}

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
    // AABB extent must be a power-of-2 multiple of voxel_size — 0.8 / 0.1 = 8.
    let aabb = Aabb { min: Vec3::ZERO, max: Vec3::splat(0.8) };
    let r = voxelize_octree(batched(|_| (1000.0, 0, 0, 0, 0)), &aabb, 0.1, &mut attrs, &mut bricks, 0).unwrap();

    assert_eq!(r.voxel_count, 0);
    assert_eq!(r.brick_ids.len(), 0);
    assert_eq!(r.octree.leaf_count(), 0);
}

#[test]
fn fully_interior_region_is_interior() {
    let mut attrs = LeafAttrPool::new(256);
    let mut bricks = BrickPool::new(64);
    // 0.2 / 0.1 = 2 cells (depth 1, 2x2x2 grid).
    let aabb = Aabb { min: Vec3::ZERO, max: Vec3::splat(0.2) };
    let r = voxelize_octree(batched(|_| (-1000.0, 0, 0, 0, 0)), &aabb, 0.1, &mut attrs, &mut bricks, 0).unwrap();

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
    use crate::sparse_octree::{brick_id as get_brick_id, is_brick};

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
        node_idx: usize,
        coord: UVec3,
        level: u8,
        brick_depth: u8,
        max_depth: u8,
        vs: f32,
        center: Vec3,
        radius: f32,
        bricks: &BrickPool,
        attrs: &LeafAttrPool,
        grid_origin: Vec3,
        checked: &mut u32,
    ) {
        use crate::sparse_octree::{
            brick_id as get_brick_id, is_brick, is_leaf, INTERIOR_NODE,
        };
        let node = nodes[node_idx];
        if is_brick(node) && level == brick_depth {
            let brick_id = get_brick_id(node);
            let brick_world_min = grid_origin
                + Vec3::new(coord.x as f32, coord.y as f32, coord.z as f32) * vs;
            for cz in 0..BRICK_DIM {
                for cy in 0..BRICK_DIM {
                    for cx in 0..BRICK_DIM {
                        let cell = bricks.get_cell(brick_id, cx, cy, cz);
                        if cell == crate::brick_pool::BRICK_EMPTY
                            || cell == crate::brick_pool::BRICK_INTERIOR
                        {
                            continue;
                        }
                        let attr = *attrs.get(cell);
                        let normal = attr.normal();
                        let cell_min = brick_world_min
                            + Vec3::new(cx as f32 * vs, cy as f32 * vs, cz as f32 * vs);
                        let cell_center = cell_min + Vec3::splat(vs * 0.5);
                        let radial = (cell_center - center).normalize();
                        // Normal should point outward (same half-space
                        // as the radial direction from the sphere
                        // center). Stricter than "> 0" because we're
                        // well outside the origin.
                        let dot = normal.dot(radial);
                        assert!(
                            dot > 0.0,
                            "cell normal {normal:?} at {cell_center:?} should point outward from sphere center (dot={dot})",
                        );
                        *checked += 1;
                    }
                }
            }
            return;
        }
        if node == INTERIOR_NODE || is_leaf(node) {
            return;
        }
        // Otherwise it's an internal node — descend.
        if level >= brick_depth {
            return;
        }
        let first_child = node as usize;
        if first_child == 0 || first_child + 8 > nodes.len() {
            return;
        }
        let child_voxels = 1u32 << (max_depth - level - 1);
        for octant in 0u32..8 {
            let dx = octant & 1;
            let dy = (octant >> 1) & 1;
            let dz = (octant >> 2) & 1;
            let child_coord = UVec3::new(
                coord.x + dx * child_voxels,
                coord.y + dy * child_voxels,
                coord.z + dz * child_voxels,
            );
            walk(
                nodes,
                first_child + octant as usize,
                child_coord,
                level + 1,
                brick_depth,
                max_depth,
                vs,
                center,
                radius,
                bricks,
                attrs,
                grid_origin,
                checked,
            );
        }
    }
    walk(
        &nodes,
        0,
        UVec3::ZERO,
        0,
        brick_depth,
        max_depth,
        vs,
        center,
        radius,
        &bricks,
        &attrs,
        r.grid_origin,
        &mut checked,
    );
    assert!(checked > 0, "should have checked at least one cell normal");
    let _ = get_brick_id; let _ = is_brick; // silence unused-import warnings
}

// ── G2: analytic-gradient voxelize path ────────────────────────────

/// Voxelize a surface, optionally supplying an analytic ∇sd at every
/// sample, and return the per-leaf (normal, distance) over the asset's
/// pool range. Slot allocation order is deterministic and identical
/// regardless of whether the gradient is supplied (same sd → same
/// classification → same surface cells, in the same order), so index
/// `i` refers to the same node in two runs that differ only in the
/// gradient.
fn voxelize_collect<Fd, Fg>(
    f: Fd,
    grad: Fg,
    supply_grad: bool,
    aabb: &Aabb,
    vs: f32,
) -> (Vec<Vec3>, Vec<f32>)
where
    Fd: Fn(Vec3) -> f32,
    Fg: Fn(Vec3) -> Vec3,
{
    let mut attrs = LeafAttrPool::new(2_000_000);
    let mut bricks = BrickPool::new(50_000);
    let sdf = |positions: &[Vec3]| -> Vec<(f32, u16, u16, u8, u32, Option<Vec3>)> {
        positions
            .iter()
            .map(|&p| {
                let g = if supply_grad { Some(grad(p)) } else { None };
                (f(p), 1u16, 1u16, 0u8, 0u32, g)
            })
            .collect()
    };
    let r = voxelize_octree(sdf, aabb, vs, &mut attrs, &mut bricks, 0).unwrap();
    let start = r.leaf_attr_slot_start;
    let n = r.leaf_attr_unique_count;
    let mut normals = Vec::with_capacity(n as usize);
    let mut dists = Vec::with_capacity(n as usize);
    for s in start..start + n {
        normals.push(crate::leaf_attr::unpack_oct(attrs.get(s).normal_oct).normalize_or_zero());
        dists.push(attrs.dist(s));
    }
    (normals, dists)
}

/// A linear SDF (tilted plane) has EXACT central differences, so the
/// taps synthesized from the analytic gradient (`d_center ± eps·g`)
/// equal the sampled taps bit-for-bit. The analytic path must therefore
/// produce identical per-leaf normals + distances to the FD path —
/// proving the analytic fast path is a no-op on correctness.
#[test]
fn analytic_gradient_plane_is_bit_identical_to_finite_difference() {
    let (m, k) = (0.4f32, -0.25);
    let f = move |p: Vec3| p.y - m * p.x - k * p.z;
    let g = move |_p: Vec3| Vec3::new(-m, 1.0, -k);
    let aabb = Aabb { min: Vec3::splat(-2.0), max: Vec3::splat(2.0) };
    let (n_fd, d_fd) = voxelize_collect(f, g, false, &aabb, 0.25);
    let (n_an, d_an) = voxelize_collect(f, g, true, &aabb, 0.25);
    assert_eq!(n_fd.len(), n_an.len(), "leaf count differs (fd {} vs an {})", n_fd.len(), n_an.len());
    assert!(!n_fd.is_empty(), "plane should produce leaves");
    for i in 0..n_fd.len() {
        assert_eq!(n_fd[i], n_an[i], "normal differs at leaf {i}: fd {:?} an {:?}", n_fd[i], n_an[i]);
        assert_eq!(d_fd[i], d_an[i], "dist differs at leaf {i}: fd {} an {}", d_fd[i], d_an[i]);
    }
}

/// A curved SDF (sphere): FD has O(eps²) truncation error; the analytic
/// gradient is exact. They should still agree closely on every leaf —
/// normals near-parallel, distances within a small fraction of a voxel.
#[test]
fn analytic_gradient_sphere_matches_finite_difference() {
    let radius = 1.0f32;
    let f = move |p: Vec3| p.length() - radius;
    let g = move |p: Vec3| p.normalize_or_zero();
    let aabb = Aabb { min: Vec3::splat(-2.0), max: Vec3::splat(2.0) };
    let (n_fd, d_fd) = voxelize_collect(f, g, false, &aabb, 0.125);
    let (n_an, d_an) = voxelize_collect(f, g, true, &aabb, 0.125);
    assert_eq!(n_fd.len(), n_an.len());
    assert!(n_fd.len() > 100, "sphere should produce many leaves; got {}", n_fd.len());
    for i in 0..n_fd.len() {
        // Skip prefilter/degenerate (zero) normals.
        if n_fd[i].length_squared() < 0.5 || n_an[i].length_squared() < 0.5 {
            continue;
        }
        let dot = n_fd[i].dot(n_an[i]).clamp(-1.0, 1.0);
        assert!(dot > 0.999, "normal {i}: dot {dot} (fd {:?} an {:?})", n_fd[i], n_an[i]);
        assert!(
            (d_fd[i] - d_an[i]).abs() < 0.05,
            "dist {i}: fd {} an {} differ too much",
            d_fd[i],
            d_an[i]
        );
    }
}

/// Some samples in a batch carry `Some(grad)`, others `None` — the
/// per-cell branching (synthesize taps vs sample them) must reproduce
/// the all-FD result. On a plane the analytic taps equal the FD taps
/// exactly, so a correct implementation is bit-identical to all-FD.
#[test]
fn analytic_gradient_mixed_batch_matches_all_finite_difference() {
    let (m, k) = (0.3f32, 0.2);
    let f = move |p: Vec3| p.y - m * p.x - k * p.z;
    let g = move |_p: Vec3| Vec3::new(-m, 1.0, -k);
    let aabb = Aabb { min: Vec3::splat(-2.0), max: Vec3::splat(2.0) };
    let vs = 0.25f32;
    let (n_fd, d_fd) = voxelize_collect(f, g, false, &aabb, vs);

    let mut attrs = LeafAttrPool::new(2_000_000);
    let mut bricks = BrickPool::new(50_000);
    // Mixed: Some(grad) only where p.x > 0; None elsewhere.
    let sdf = |positions: &[Vec3]| -> Vec<(f32, u16, u16, u8, u32, Option<Vec3>)> {
        positions
            .iter()
            .map(|&p| {
                let gr = if p.x > 0.0 { Some(g(p)) } else { None };
                (f(p), 1u16, 1u16, 0u8, 0u32, gr)
            })
            .collect()
    };
    let r = voxelize_octree(sdf, &aabb, vs, &mut attrs, &mut bricks, 0).unwrap();
    let start = r.leaf_attr_slot_start;
    let cnt = r.leaf_attr_unique_count;
    assert_eq!(cnt as usize, n_fd.len(), "mixed leaf count differs");
    for (i, s) in (start..start + cnt).enumerate() {
        let nm = crate::leaf_attr::unpack_oct(attrs.get(s).normal_oct).normalize_or_zero();
        assert_eq!(nm, n_fd[i], "mixed normal differs at {i}");
        assert_eq!(attrs.dist(s), d_fd[i], "mixed dist differs at {i}");
    }
}
