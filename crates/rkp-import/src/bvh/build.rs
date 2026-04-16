//! BVH construction internals. Midpoint split along the longest AABB
//! axis; falls back to an even partition when the midpoint degenerates.
//! Precomputes Barill dipole data per node for far-field winding
//! approximation at query time.

use glam::Vec3;

/// Axis-aligned bounding box for BVH nodes.
#[derive(Debug, Clone, Copy)]
pub(super) struct BvhAabb {
    pub min: Vec3,
    pub max: Vec3,
}

/// Per-node data for BVH-accelerated winding number (Barill et al. 2018).
///
/// Stores the dipole approximation: area-weighted normal sum,
/// area-weighted centroid, and bounding radius. For far-away clusters
/// the solid-angle contribution is approximated as
/// `dot(dipole, r) / (4π |r|³)`.
#[derive(Debug, Clone, Copy)]
pub(super) struct WindingData {
    /// Sum of `(face_normal × triangle_area)` over all triangles.
    pub dipole: Vec3,
    /// Area-weighted centroid of all triangles in this subtree.
    pub centroid: Vec3,
    /// Max distance from centroid to any vertex in this subtree.
    pub radius: f32,
}

/// BVH node — either an interior with two children, or a leaf with a
/// range of triangles in the reordered index array.
pub(super) enum BvhNode {
    Interior {
        bounds: BvhAabb,
        winding: WindingData,
        left: Box<BvhNode>,
        right: Box<BvhNode>,
    },
    Leaf {
        bounds: BvhAabb,
        winding: WindingData,
        start: usize,
        count: usize,
    },
}

impl BvhNode {
    pub(super) fn bounds(&self) -> BvhAabb {
        match self {
            BvhNode::Interior { bounds, .. } | BvhNode::Leaf { bounds, .. } => *bounds,
        }
    }
    pub(super) fn winding(&self) -> WindingData {
        match self {
            BvhNode::Interior { winding, .. } | BvhNode::Leaf { winding, .. } => *winding,
        }
    }
}

/// Recursive BVH construction using midpoint split along the longest axis.
pub(super) fn build_recursive(
    positions: &[[Vec3; 3]],
    face_area_normals: &[Vec3],
    tri_order: &mut [usize],
    start: usize,
    count: usize,
    depth: usize,
) -> BvhNode {
    let bounds = compute_bounds(positions, &tri_order[start..start + count]);
    let winding = compute_winding_data(positions, face_area_normals, tri_order, start, count);

    if count <= 4 || depth >= 32 {
        return BvhNode::Leaf {
            bounds,
            winding,
            start,
            count,
        };
    }

    let extent = bounds.max - bounds.min;
    let axis = if extent.x >= extent.y && extent.x >= extent.z {
        0
    } else if extent.y >= extent.z {
        1
    } else {
        2
    };
    let mid = (bounds.min[axis] + bounds.max[axis]) * 0.5;

    let slice = &mut tri_order[start..start + count];
    let mut left_count = 0;
    for i in 0..slice.len() {
        let tri = &positions[slice[i]];
        let centroid_axis = (tri[0][axis] + tri[1][axis] + tri[2][axis]) / 3.0;
        if centroid_axis < mid {
            slice.swap(i, left_count);
            left_count += 1;
        }
    }
    if left_count == 0 || left_count == count {
        left_count = count / 2;
    }

    let left = build_recursive(
        positions, face_area_normals, tri_order, start, left_count, depth + 1,
    );
    let right = build_recursive(
        positions,
        face_area_normals,
        tri_order,
        start + left_count,
        count - left_count,
        depth + 1,
    );

    BvhNode::Interior {
        bounds,
        winding,
        left: Box::new(left),
        right: Box::new(right),
    }
}

fn compute_winding_data(
    positions: &[[Vec3; 3]],
    face_area_normals: &[Vec3],
    tri_order: &[usize],
    start: usize,
    count: usize,
) -> WindingData {
    let mut dipole = Vec3::ZERO;
    let mut centroid = Vec3::ZERO;
    let mut total_area = 0.0f32;

    for &idx in &tri_order[start..start + count] {
        let area_normal = face_area_normals[idx];
        let area = area_normal.length() * 0.5;
        let [a, b, c] = positions[idx];
        let tri_centroid = (a + b + c) / 3.0;

        dipole += area_normal * 0.5;
        centroid += tri_centroid * area;
        total_area += area;
    }

    if total_area > 1e-10 {
        centroid /= total_area;
    }

    let mut radius = 0.0f32;
    for &idx in &tri_order[start..start + count] {
        for v in &positions[idx] {
            let d = (*v - centroid).length();
            if d > radius {
                radius = d;
            }
        }
    }

    WindingData { dipole, centroid, radius }
}

fn compute_bounds(positions: &[[Vec3; 3]], indices: &[usize]) -> BvhAabb {
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    for &idx in indices {
        for v in &positions[idx] {
            min = min.min(*v);
            max = max.max(*v);
        }
    }
    BvhAabb { min, max }
}

/// Distance from a point to an AABB (0 if inside).
pub(super) fn aabb_distance(point: Vec3, aabb: &BvhAabb) -> f32 {
    let clamped = point.clamp(aabb.min, aabb.max);
    (point - clamped).length()
}
