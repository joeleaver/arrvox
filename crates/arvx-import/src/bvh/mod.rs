//! Bounding volume hierarchy over mesh triangles. Accelerates
//! nearest-triangle queries (for unsigned distance) and generalized
//! winding-number evaluation (for sign determination).
//!
//! Binary BVH built by midpoint split along the longest AABB axis
//! (see [`build`]). Each node stores a Barill dipole approximation
//! that the [`winding`] module uses for O(log n) far-field winding.
//!
//! The public surface is tiny: build a BVH from a [`crate::mesh::MeshData`],
//! then query with [`TriangleBvh::nearest`] or
//! [`TriangleBvh::winding_number`].

use glam::Vec3;

use crate::mesh::MeshData;

mod build;
mod raycast;
mod winding;

use build::{BvhNode, aabb_distance, build_recursive};

/// Result of a nearest-triangle query.
#[derive(Debug, Clone, Copy)]
pub struct NearestResult {
    /// Unsigned distance from the query point to the closest triangle.
    pub distance: f32,
    /// Index of the closest triangle in the source mesh.
    pub triangle_index: usize,
    /// Barycentric coordinates of the closest point on the triangle.
    pub barycentric: [f32; 3],
    /// The closest point on the triangle surface (world space).
    pub closest_point: Vec3,
}

/// BVH over mesh triangles.
pub struct TriangleBvh {
    root: BvhNode,
    /// Triangle indices reordered by BVH construction; leaves reference
    /// ranges into this vector.
    tri_order: Vec<usize>,
    /// Cached triangle vertex positions for fast access during queries.
    positions: Vec<[Vec3; 3]>,
}

impl TriangleBvh {
    /// Build a BVH from the given mesh's triangle soup.
    pub fn build(mesh: &MeshData) -> Self {
        let tri_count = mesh.triangle_count();
        let positions: Vec<[Vec3; 3]> =
            (0..tri_count).map(|i| mesh.triangle_positions(i)).collect();
        let face_area_normals: Vec<Vec3> = positions
            .iter()
            .map(|[a, b, c]| (*b - *a).cross(*c - *a))
            .collect();

        let mut tri_order: Vec<usize> = (0..tri_count).collect();
        let root = build_recursive(
            &positions,
            &face_area_normals,
            &mut tri_order,
            0,
            tri_count,
            0,
        );

        Self { root, tri_order, positions }
    }

    /// Find the nearest triangle to `point`.
    pub fn nearest(&self, point: Vec3) -> NearestResult {
        let mut best = NearestResult {
            distance: f32::MAX,
            triangle_index: 0,
            barycentric: [1.0, 0.0, 0.0],
            closest_point: Vec3::ZERO,
        };
        self.nearest_recursive(&self.root, point, &mut best);
        best
    }

    fn nearest_recursive(&self, node: &BvhNode, point: Vec3, best: &mut NearestResult) {
        match node {
            BvhNode::Leaf { start, count, .. } => {
                for &tri_idx in &self.tri_order[*start..*start + *count] {
                    let tri = &self.positions[tri_idx];
                    let (dist, bary, closest) =
                        point_triangle_distance(point, tri[0], tri[1], tri[2]);
                    if dist < best.distance {
                        best.distance = dist;
                        best.triangle_index = tri_idx;
                        best.barycentric = bary;
                        best.closest_point = closest;
                    }
                }
            }
            BvhNode::Interior { left, right, .. } => {
                let left_b = left.bounds();
                let right_b = right.bounds();
                let left_d = aabb_distance(point, &left_b);
                let right_d = aabb_distance(point, &right_b);

                // Visit closer child first for better pruning.
                if left_d < right_d {
                    if left_d < best.distance {
                        self.nearest_recursive(left, point, best);
                    }
                    if right_d < best.distance {
                        self.nearest_recursive(right, point, best);
                    }
                } else {
                    if right_d < best.distance {
                        self.nearest_recursive(right, point, best);
                    }
                    if left_d < best.distance {
                        self.nearest_recursive(left, point, best);
                    }
                }
            }
        }
    }

    /// Generalized winding number at `point`.
    ///
    /// Values near ±1 mean inside a closed surface; near 0 means
    /// outside. Handles non-watertight meshes gracefully. Complexity
    /// is O(N) worst case but typically O(log N) due to the dipole
    /// approximation for distant triangle clusters.
    pub fn winding_number(&self, point: Vec3) -> f32 {
        winding::winding_recursive(&self.root, &self.positions, &self.tri_order, point)
            / (4.0 * std::f32::consts::PI)
    }

    /// Topological inside/outside via off-axis ray casting.
    ///
    /// Casts three semi-infinite rays from `point` along deliberately
    /// non-grid-aligned directions and counts triangle intersections
    /// (Möller–Trumbore) on each. Odd crossings ⇒ inside on that
    /// ray. Classification is the majority vote (≥ 2 of 3) — any
    /// single ray that grazes an edge loses to the other two.
    ///
    /// Rays are *almost* but not exactly axis-aligned: a perfectly
    /// axis-aligned ray through a grid-aligned cube hits the shared
    /// edge between two +X triangles and double-counts, so we tilt
    /// each ray by ~0.3° off-axis. The three chosen directions are
    /// linearly independent and non-parallel to any plane the
    /// importer is likely to see, so no two triangles share an
    /// in-ray-plane edge under normal mesh data.
    ///
    /// Chosen over generalized winding for real-world scan/CAD meshes
    /// with self-intersections, duplicated triangles, or non-manifold
    /// patches — winding produces `|w| > 0.5` pockets outside the
    /// surface and inverted-sign bricks on layered geometry. Ray
    /// parity contributes 0 or 1 per triangle regardless of layering.
    pub fn is_inside_raycast(&self, point: Vec3) -> bool {
        // Hand-chosen near-axis directions; the small off-axis
        // components (0.005 / 0.007) break alignment with planar
        // mesh geometry while keeping each ray's primary axis
        // dominant for cheap BVH-AABB pruning.
        let dirs = [
            Vec3::new(1.0, 0.005, 0.007),
            Vec3::new(0.007, 1.0, 0.005),
            Vec3::new(0.005, 0.007, 1.0),
        ];
        let mut votes = 0u32;
        for dir in dirs {
            let count = raycast::count_intersections(
                &self.root, &self.positions, &self.tri_order, point, dir,
            );
            if count % 2 == 1 {
                votes += 1;
            }
        }
        votes >= 2
    }
}

/// Closest point on a triangle to a query point.
///
/// Returns `(distance, barycentric_coords, closest_point)`. Uses the
/// Voronoi-region method from Ericson, "Real-Time Collision Detection".
pub fn point_triangle_distance(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> (f32, [f32; 3], Vec3) {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;

    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return ((p - a).length(), [1.0, 0.0, 0.0], a);
    }

    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return ((p - b).length(), [0.0, 1.0, 0.0], b);
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        let closest = a + v * ab;
        return ((p - closest).length(), [1.0 - v, v, 0.0], closest);
    }

    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return ((p - c).length(), [0.0, 0.0, 1.0], c);
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        let closest = a + w * ac;
        return ((p - closest).length(), [1.0 - w, 0.0, w], closest);
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let closest = b + w * (c - b);
        return ((p - closest).length(), [0.0, 1.0 - w, w], closest);
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    let closest = a + ab * v + ac * w;
    ((p - closest).length(), [1.0 - v - w, v, w], closest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{ImportMaterial, MeshData};

    fn single_triangle_mesh() -> MeshData {
        MeshData {
            positions: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            normals: vec![Vec3::Z; 3],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            indices: vec![0, 1, 2],
            material_indices: vec![0],
            materials: vec![ImportMaterial::default()],
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::new(1.0, 1.0, 0.0),
        }
    }

    #[test]
    fn nearest_single_triangle_above_centroid() {
        let bvh = TriangleBvh::build(&single_triangle_mesh());
        let centroid = Vec3::new(1.0 / 3.0, 1.0 / 3.0, 0.0);
        let query = centroid + Vec3::new(0.0, 0.0, 2.0);
        let r = bvh.nearest(query);
        assert!((r.distance - 2.0).abs() < 1e-5);
        assert_eq!(r.triangle_index, 0);
        assert!((r.closest_point - centroid).length() < 1e-5);
    }

    #[test]
    fn nearest_at_vertex_returns_zero() {
        let bvh = TriangleBvh::build(&single_triangle_mesh());
        let r = bvh.nearest(Vec3::new(1.0, 0.0, 0.0));
        assert!(r.distance < 1e-6);
    }

    #[test]
    fn point_triangle_distance_above_center() {
        let a = Vec3::ZERO;
        let b = Vec3::X;
        let c = Vec3::Y;
        let (dist, bary, closest) = point_triangle_distance(Vec3::new(0.25, 0.25, 3.0), a, b, c);
        assert!((dist - 3.0).abs() < 1e-5);
        assert!(closest.z.abs() < 1e-5);
        let s: f32 = bary.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn point_triangle_distance_to_edge() {
        let (_, bary, closest) = point_triangle_distance(
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(1.0, 2.0, 0.0),
        );
        assert!((closest - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-5);
        assert!(bary[2].abs() < 1e-5);
    }

    #[test]
    fn winding_inside_closed_cube_is_one() {
        // Build a small cube (12 triangles) and check winding at the centre.
        let (min, max) = (Vec3::splat(-1.0), Vec3::splat(1.0));
        let v = [
            Vec3::new(min.x, min.y, min.z),
            Vec3::new(max.x, min.y, min.z),
            Vec3::new(max.x, max.y, min.z),
            Vec3::new(min.x, max.y, min.z),
            Vec3::new(min.x, min.y, max.z),
            Vec3::new(max.x, min.y, max.z),
            Vec3::new(max.x, max.y, max.z),
            Vec3::new(min.x, max.y, max.z),
        ];
        // Outward-facing triangles (counter-clockwise seen from outside).
        #[rustfmt::skip]
        let idx: Vec<u32> = vec![
            0, 2, 1, 0, 3, 2, // -Z
            4, 5, 6, 4, 6, 7, // +Z
            0, 1, 5, 0, 5, 4, // -Y
            3, 7, 6, 3, 6, 2, // +Y
            0, 4, 7, 0, 7, 3, // -X
            1, 2, 6, 1, 6, 5, // +X
        ];
        let mesh = MeshData {
            positions: v.to_vec(),
            normals: vec![Vec3::Y; 8],
            uvs: Vec::new(),
            indices: idx,
            material_indices: vec![0; 12],
            materials: vec![ImportMaterial::default()],
            bounds_min: min,
            bounds_max: max,
        };
        let bvh = TriangleBvh::build(&mesh);
        let w_inside = bvh.winding_number(Vec3::ZERO);
        let w_outside = bvh.winding_number(Vec3::new(5.0, 0.0, 0.0));
        assert!(w_inside.abs() > 0.9, "inside winding = {w_inside}");
        assert!(w_outside.abs() < 0.1, "outside winding = {w_outside}");
    }
}
