//! Axis-aligned bounding boxes in local and world space.
//!
//! [`Aabb`] operates in local/chunk space using `Vec3`.
//! [`WorldAabb`] operates in world space using [`WorldPosition`] for
//! precision-safe large-distance calculations.

use glam::Vec3;

use crate::world_position::WorldPosition;

// ─── Aabb ────────────────────────────────────────────────────────────────────

/// Axis-aligned bounding box in local/chunk space.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Aabb {
    /// Minimum corner (component-wise minimum of the two input corners).
    pub min: Vec3,
    /// Maximum corner (component-wise maximum of the two input corners).
    pub max: Vec3,
}

impl Aabb {
    /// Construct an `Aabb` from two corners, ensuring `min <= max` per
    /// component regardless of argument order.
    #[inline]
    pub fn new(a: Vec3, b: Vec3) -> Self {
        Self {
            min: a.min(b),
            max: a.max(b),
        }
    }

    /// Construct an `Aabb` from a center point and half-extents.
    ///
    /// `half_extents` should be non-negative on each axis; negative values are
    /// treated as zero (the box collapses to a point on that axis).
    #[inline]
    pub fn from_center_half_extents(center: Vec3, half_extents: Vec3) -> Self {
        let he = half_extents.max(Vec3::ZERO);
        Self {
            min: center - he,
            max: center + he,
        }
    }

    /// Midpoint of the box.
    #[inline]
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Side lengths along each axis (`max - min`).
    #[inline]
    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }

    /// Half the side lengths (`size / 2`).
    #[inline]
    pub fn half_extents(&self) -> Vec3 {
        self.size() * 0.5
    }

    /// Returns `true` if `point` is inside or on the boundary of the box.
    #[inline]
    pub fn contains_point(&self, point: Vec3) -> bool {
        point.x >= self.min.x
            && point.y >= self.min.y
            && point.z >= self.min.z
            && point.x <= self.max.x
            && point.y <= self.max.y
            && point.z <= self.max.z
    }

    /// Returns `true` if `other` is fully contained within `self` (boundaries
    /// inclusive).
    #[inline]
    pub fn contains_aabb(&self, other: &Aabb) -> bool {
        self.contains_point(other.min) && self.contains_point(other.max)
    }

    /// Returns `true` if `self` and `other` overlap (touching counts as
    /// overlapping).
    #[inline]
    pub fn intersects(&self, other: &Aabb) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    /// Return the overlap region of `self` and `other`, or `None` if they do
    /// not intersect.
    #[inline]
    pub fn intersection(&self, other: &Aabb) -> Option<Aabb> {
        let min = self.min.max(other.min);
        let max = self.max.min(other.max);
        if min.x <= max.x && min.y <= max.y && min.z <= max.z {
            Some(Aabb { min, max })
        } else {
            None
        }
    }

    /// Return a new `Aabb` that also contains `point`.
    #[inline]
    pub fn expand(&self, point: Vec3) -> Aabb {
        Aabb {
            min: self.min.min(point),
            max: self.max.max(point),
        }
    }

    /// Return the union of `self` and `other` (smallest box containing both).
    #[inline]
    pub fn expand_aabb(&self, other: &Aabb) -> Aabb {
        Aabb {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// Volume of the box.
    #[inline]
    pub fn volume(&self) -> f32 {
        let s = self.size();
        s.x * s.y * s.z
    }

    /// Return all 8 corners of the bounding box.
    ///
    /// Order: iterate Z low/high, then Y low/high, then X low/high.
    #[inline]
    pub fn corners(&self) -> [Vec3; 8] {
        [
            Vec3::new(self.min.x, self.min.y, self.min.z),
            Vec3::new(self.min.x, self.min.y, self.max.z),
            Vec3::new(self.min.x, self.max.y, self.min.z),
            Vec3::new(self.min.x, self.max.y, self.max.z),
            Vec3::new(self.max.x, self.min.y, self.min.z),
            Vec3::new(self.max.x, self.min.y, self.max.z),
            Vec3::new(self.max.x, self.max.y, self.min.z),
            Vec3::new(self.max.x, self.max.y, self.max.z),
        ]
    }

    /// Total surface area (sum of six face areas).
    #[inline]
    pub fn surface_area(&self) -> f32 {
        let s = self.size();
        2.0 * (s.x * s.y + s.y * s.z + s.z * s.x)
    }
}

// ─── WorldAabb ───────────────────────────────────────────────────────────────

/// Axis-aligned bounding box in world space using [`WorldPosition`] for
/// precision-safe large-distance arithmetic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorldAabb {
    /// Minimum corner in world space.
    pub min: WorldPosition,
    /// Maximum corner in world space.
    pub max: WorldPosition,
}

impl WorldAabb {
    /// Construct a `WorldAabb` from two `WorldPosition` corners.
    ///
    /// The caller is responsible for ensuring `min <= max` in world-space
    /// order.  If you need automatic ordering, compute displacements with
    /// [`WorldPosition::relative_to_f64`] and swap if necessary.
    #[inline]
    pub fn new(min: WorldPosition, max: WorldPosition) -> Self {
        Self { min, max }
    }

    /// Returns `true` if `point` is inside or on the boundary of the box.
    ///
    /// All comparisons are performed in `f64` relative to `self.min` to avoid
    /// precision loss.
    pub fn contains_point(&self, point: &WorldPosition) -> bool {
        // Express point and max as offsets from min (all f64).
        let p = point.relative_to_f64(&self.min);
        let s = self.max.relative_to_f64(&self.min);
        p.x >= 0.0
            && p.y >= 0.0
            && p.z >= 0.0
            && p.x <= s.x
            && p.y <= s.y
            && p.z <= s.z
    }

    /// Returns `true` if `self` and `other` overlap (touching counts).
    ///
    /// The check is performed by expressing all corners as offsets from a
    /// common reference point (`self.min`) in `f64`.
    pub fn intersects(&self, other: &WorldAabb) -> bool {
        let origin = &self.min;

        let self_max = self.max.relative_to_f64(origin);
        let other_min = other.min.relative_to_f64(origin);
        let other_max = other.max.relative_to_f64(origin);
        // self.min is the origin, so self_min = (0,0,0).

        // Separating axis test on each axis.
        other_min.x <= self_max.x
            && other_max.x >= 0.0
            && other_min.y <= self_max.y
            && other_max.y >= 0.0
            && other_min.z <= self_max.z
            && other_max.z >= 0.0
    }

    /// Convert to a local (camera-relative) [`Aabb`] by expressing both
    /// corners as offsets from `origin`.
    ///
    /// The resulting `Aabb` is suitable for rendering or physics queries where
    /// positions have already been made camera-relative.
    pub fn to_local_aabb(&self, origin: &WorldPosition) -> Aabb {
        let min = self.min.relative_to(origin);
        let max = self.max.relative_to(origin);
        Aabb::new(min, max)
    }
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{IVec3, Vec3};

    // ── Aabb construction ────────────────────────────────────────────────────

    /// `new` normalises so that min <= max per component.
    #[test]
    fn aabb_new_normalizes_min_max() {
        let a = Aabb::new(Vec3::new(3.0, 2.0, 5.0), Vec3::new(1.0, 4.0, 2.0));
        assert_eq!(a.min, Vec3::new(1.0, 2.0, 2.0));
        assert_eq!(a.max, Vec3::new(3.0, 4.0, 5.0));
    }

    /// Passing min and max in the correct order is also fine.
    #[test]
    fn aabb_new_already_ordered() {
        let a = Aabb::new(Vec3::ZERO, Vec3::ONE);
        assert_eq!(a.min, Vec3::ZERO);
        assert_eq!(a.max, Vec3::ONE);
    }

    /// `from_center_half_extents` produces a symmetric box.
    #[test]
    fn aabb_from_center_half_extents() {
        let center = Vec3::new(2.0, 3.0, 4.0);
        let he = Vec3::new(1.0, 0.5, 2.0);
        let a = Aabb::from_center_half_extents(center, he);
        assert_eq!(a.min, Vec3::new(1.0, 2.5, 2.0));
        assert_eq!(a.max, Vec3::new(3.0, 3.5, 6.0));
    }

    /// Negative half-extents are clamped to zero, producing a point AABB.
    #[test]
    fn aabb_from_center_negative_half_extents_clamps_to_zero() {
        let center = Vec3::new(1.0, 1.0, 1.0);
        let a = Aabb::from_center_half_extents(center, Vec3::new(-1.0, -1.0, -1.0));
        assert_eq!(a.min, center);
        assert_eq!(a.max, center);
    }

    // ── center & size ────────────────────────────────────────────────────────

    #[test]
    fn aabb_center_and_size() {
        let a = Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(4.0, 6.0, 8.0));
        let c = a.center();
        assert!((c.x - 2.0).abs() < 1e-6, "center.x = {}", c.x);
        assert!((c.y - 3.0).abs() < 1e-6, "center.y = {}", c.y);
        assert!((c.z - 4.0).abs() < 1e-6, "center.z = {}", c.z);

        let s = a.size();
        assert!((s.x - 4.0).abs() < 1e-6);
        assert!((s.y - 6.0).abs() < 1e-6);
        assert!((s.z - 8.0).abs() < 1e-6);
    }

    #[test]
    fn aabb_half_extents() {
        let a = Aabb::new(Vec3::ZERO, Vec3::new(2.0, 4.0, 6.0));
        let he = a.half_extents();
        assert!((he.x - 1.0).abs() < 1e-6);
        assert!((he.y - 2.0).abs() < 1e-6);
        assert!((he.z - 3.0).abs() < 1e-6);
    }

    // ── contains_point ───────────────────────────────────────────────────────

    #[test]
    fn aabb_contains_point_inside() {
        let a = Aabb::new(Vec3::ZERO, Vec3::ONE);
        assert!(a.contains_point(Vec3::new(0.5, 0.5, 0.5)));
    }

    #[test]
    fn aabb_contains_point_outside() {
        let a = Aabb::new(Vec3::ZERO, Vec3::ONE);
        assert!(!a.contains_point(Vec3::new(1.5, 0.5, 0.5)));
        assert!(!a.contains_point(Vec3::new(-0.1, 0.5, 0.5)));
    }

    #[test]
    fn aabb_contains_point_on_boundary() {
        let a = Aabb::new(Vec3::ZERO, Vec3::ONE);
        // All eight corners should be contained.
        for &x in &[0.0_f32, 1.0] {
            for &y in &[0.0_f32, 1.0] {
                for &z in &[0.0_f32, 1.0] {
                    assert!(
                        a.contains_point(Vec3::new(x, y, z)),
                        "corner ({x},{y},{z}) not contained"
                    );
                }
            }
        }
    }

    // ── contains_aabb ────────────────────────────────────────────────────────

    #[test]
    fn aabb_contains_aabb_fully_inside() {
        let outer = Aabb::new(Vec3::ZERO, Vec3::new(10.0, 10.0, 10.0));
        let inner = Aabb::new(Vec3::new(2.0, 2.0, 2.0), Vec3::new(8.0, 8.0, 8.0));
        assert!(outer.contains_aabb(&inner));
    }

    #[test]
    fn aabb_contains_aabb_partial_overlap() {
        let a = Aabb::new(Vec3::ZERO, Vec3::new(5.0, 5.0, 5.0));
        let b = Aabb::new(Vec3::new(3.0, 3.0, 3.0), Vec3::new(8.0, 8.0, 8.0));
        assert!(!a.contains_aabb(&b));
    }

    #[test]
    fn aabb_contains_aabb_outside() {
        let a = Aabb::new(Vec3::ZERO, Vec3::ONE);
        let b = Aabb::new(Vec3::new(2.0, 2.0, 2.0), Vec3::new(3.0, 3.0, 3.0));
        assert!(!a.contains_aabb(&b));
    }

    #[test]
    fn aabb_contains_aabb_same_box() {
        let a = Aabb::new(Vec3::ZERO, Vec3::ONE);
        assert!(a.contains_aabb(&a));
    }

    // ── intersects ───────────────────────────────────────────────────────────

    #[test]
    fn aabb_intersects_overlapping() {
        let a = Aabb::new(Vec3::ZERO, Vec3::new(3.0, 3.0, 3.0));
        let b = Aabb::new(Vec3::new(2.0, 2.0, 2.0), Vec3::new(5.0, 5.0, 5.0));
        assert!(a.intersects(&b));
        assert!(b.intersects(&a));
    }

    #[test]
    fn aabb_intersects_touching_faces() {
        let a = Aabb::new(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0));
        let b = Aabb::new(Vec3::new(1.0, 0.0, 0.0), Vec3::new(2.0, 1.0, 1.0));
        assert!(a.intersects(&b));
    }

    #[test]
    fn aabb_intersects_apart() {
        let a = Aabb::new(Vec3::ZERO, Vec3::ONE);
        let b = Aabb::new(Vec3::new(2.0, 0.0, 0.0), Vec3::new(3.0, 1.0, 1.0));
        assert!(!a.intersects(&b));
    }

    // ── intersection ─────────────────────────────────────────────────────────

    #[test]
    fn aabb_intersection_overlap() {
        let a = Aabb::new(Vec3::ZERO, Vec3::new(4.0, 4.0, 4.0));
        let b = Aabb::new(Vec3::new(2.0, 2.0, 2.0), Vec3::new(6.0, 6.0, 6.0));
        let result = a.intersection(&b).expect("should intersect");
        assert_eq!(result.min, Vec3::new(2.0, 2.0, 2.0));
        assert_eq!(result.max, Vec3::new(4.0, 4.0, 4.0));
    }

    #[test]
    fn aabb_intersection_no_overlap() {
        let a = Aabb::new(Vec3::ZERO, Vec3::ONE);
        let b = Aabb::new(Vec3::new(2.0, 2.0, 2.0), Vec3::new(3.0, 3.0, 3.0));
        assert!(a.intersection(&b).is_none());
    }

    #[test]
    fn aabb_intersection_touching_is_some() {
        let a = Aabb::new(Vec3::ZERO, Vec3::ONE);
        let b = Aabb::new(Vec3::new(1.0, 0.0, 0.0), Vec3::new(2.0, 1.0, 1.0));
        let result = a.intersection(&b).expect("touching boxes should produce degenerate intersection");
        // The overlap is a zero-thickness plane at x=1.
        assert!((result.min.x - 1.0).abs() < 1e-6);
        assert!((result.max.x - 1.0).abs() < 1e-6);
    }

    // ── expand ───────────────────────────────────────────────────────────────

    #[test]
    fn aabb_expand_point_outside() {
        let a = Aabb::new(Vec3::ZERO, Vec3::ONE);
        let expanded = a.expand(Vec3::new(2.0, 3.0, -1.0));
        assert_eq!(expanded.min, Vec3::new(0.0, 0.0, -1.0));
        assert_eq!(expanded.max, Vec3::new(2.0, 3.0, 1.0));
    }

    #[test]
    fn aabb_expand_point_inside() {
        let a = Aabb::new(Vec3::ZERO, Vec3::new(4.0, 4.0, 4.0));
        let expanded = a.expand(Vec3::new(2.0, 2.0, 2.0));
        assert_eq!(expanded.min, a.min);
        assert_eq!(expanded.max, a.max);
    }

    // ── expand_aabb ──────────────────────────────────────────────────────────

    #[test]
    fn aabb_expand_aabb_union() {
        let a = Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 3.0, 4.0));
        let b = Aabb::new(Vec3::new(-1.0, 2.0, 0.0), Vec3::new(3.0, 4.0, 5.0));
        let union = a.expand_aabb(&b);
        assert_eq!(union.min, Vec3::new(-1.0, 0.0, 0.0));
        assert_eq!(union.max, Vec3::new(3.0, 4.0, 5.0));
    }

    #[test]
    fn aabb_expand_aabb_one_inside_other() {
        let outer = Aabb::new(Vec3::ZERO, Vec3::new(10.0, 10.0, 10.0));
        let inner = Aabb::new(Vec3::ONE, Vec3::new(5.0, 5.0, 5.0));
        let union = outer.expand_aabb(&inner);
        assert_eq!(union.min, outer.min);
        assert_eq!(union.max, outer.max);
    }

    // ── volume & surface_area ────────────────────────────────────────────────

    #[test]
    fn aabb_volume() {
        let a = Aabb::new(Vec3::ZERO, Vec3::new(2.0, 3.0, 4.0));
        assert!((a.volume() - 24.0).abs() < 1e-5, "volume = {}", a.volume());
    }

    #[test]
    fn aabb_volume_unit_cube() {
        let a = Aabb::new(Vec3::ZERO, Vec3::ONE);
        assert!((a.volume() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn aabb_surface_area() {
        // 2×3×4 box: faces are 2×3, 3×4, 2×4 → 6 + 12 + 8 = 26, times 2 = 52
        let a = Aabb::new(Vec3::ZERO, Vec3::new(2.0, 3.0, 4.0));
        assert!(
            (a.surface_area() - 52.0).abs() < 1e-4,
            "surface_area = {}",
            a.surface_area()
        );
    }

    #[test]
    fn aabb_surface_area_unit_cube() {
        let a = Aabb::new(Vec3::ZERO, Vec3::ONE);
        assert!((a.surface_area() - 6.0).abs() < 1e-6);
    }

    // ── corners ──────────────────────────────────────────────────────────────

    #[test]
    fn aabb_corners_returns_all_eight() {
        let a = Aabb::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 6.0, 9.0));
        let c = a.corners();
        assert_eq!(c.len(), 8);
        // Every corner component should be either min or max on that axis
        for corner in &c {
            assert!(corner.x == 1.0 || corner.x == 4.0);
            assert!(corner.y == 2.0 || corner.y == 6.0);
            assert!(corner.z == 3.0 || corner.z == 9.0);
        }
        // All 8 should be unique
        for i in 0..8 {
            for j in (i + 1)..8 {
                assert_ne!(c[i], c[j], "corners {i} and {j} are identical");
            }
        }
    }

    #[test]
    fn aabb_corners_unit_cube() {
        let a = Aabb::new(Vec3::ZERO, Vec3::ONE);
        let c = a.corners();
        // Check specific corners
        assert!(c.contains(&Vec3::ZERO));
        assert!(c.contains(&Vec3::ONE));
        assert!(c.contains(&Vec3::new(1.0, 0.0, 0.0)));
        assert!(c.contains(&Vec3::new(0.0, 1.0, 1.0)));
    }

    // ── WorldAabb ────────────────────────────────────────────────────────────

    fn make_wp(chunk_x: i32, chunk_y: i32, chunk_z: i32, lx: f32, ly: f32, lz: f32) -> WorldPosition {
        WorldPosition::new(
            IVec3::new(chunk_x, chunk_y, chunk_z),
            Vec3::new(lx, ly, lz),
        )
    }

    /// A point known to be inside the box is reported as contained.
    #[test]
    fn world_aabb_contains_point_inside() {
        let min = make_wp(0, 0, 0, 0.0, 0.0, 0.0);
        let max = make_wp(1, 1, 1, 0.0, 0.0, 0.0); // 8×8×8 metres
        let wa = WorldAabb::new(min, max);
        let inside = make_wp(0, 0, 0, 4.0, 4.0, 4.0);
        assert!(wa.contains_point(&inside));
    }

    /// A point outside the box is correctly rejected.
    #[test]
    fn world_aabb_contains_point_outside() {
        let min = make_wp(0, 0, 0, 0.0, 0.0, 0.0);
        let max = make_wp(0, 0, 0, 4.0, 4.0, 4.0);
        let wa = WorldAabb::new(min, max);
        let outside = make_wp(0, 0, 0, 5.0, 2.0, 2.0);
        assert!(!wa.contains_point(&outside));
    }

    /// A point on the min boundary is inside.
    #[test]
    fn world_aabb_contains_point_on_min_boundary() {
        let min = make_wp(0, 0, 0, 1.0, 1.0, 1.0);
        let max = make_wp(0, 0, 0, 5.0, 5.0, 5.0);
        let wa = WorldAabb::new(min, max);
        assert!(wa.contains_point(&min));
    }

    /// A point on the max boundary is inside.
    #[test]
    fn world_aabb_contains_point_on_max_boundary() {
        let min = make_wp(0, 0, 0, 1.0, 1.0, 1.0);
        let max = make_wp(0, 0, 0, 5.0, 5.0, 5.0);
        let wa = WorldAabb::new(min, max);
        assert!(wa.contains_point(&max));
    }

    /// `to_local_aabb` with origin at min gives an Aabb starting at zero.
    #[test]
    fn world_aabb_to_local_aabb_origin_at_min() {
        let min = make_wp(5, 0, 0, 0.0, 0.0, 0.0);
        let max = make_wp(5, 0, 0, 4.0, 3.0, 2.0);
        let wa = WorldAabb::new(min, max);
        let local = wa.to_local_aabb(&min);

        assert!(local.min.length() < 1e-5, "local.min should be zero: {:?}", local.min);
        assert!((local.max.x - 4.0).abs() < 1e-5, "local.max.x = {}", local.max.x);
        assert!((local.max.y - 3.0).abs() < 1e-5, "local.max.y = {}", local.max.y);
        assert!((local.max.z - 2.0).abs() < 1e-5, "local.max.z = {}", local.max.z);
    }

    /// `to_local_aabb` with origin outside the box gives a shifted Aabb.
    #[test]
    fn world_aabb_to_local_aabb_offset_origin() {
        // Box from (0,0,0)+(0,0,0) to (0,0,0)+(8,0,0)... span 1 chunk on x.
        // Actually use simple f32-friendly values to avoid chunk boundary issues.
        let min = make_wp(0, 0, 0, 2.0, 0.0, 0.0);
        let max = make_wp(0, 0, 0, 6.0, 4.0, 4.0);
        let wa = WorldAabb::new(min, max);

        // Origin is at (0,0,0,0,0,0) — 2 metres before min.
        let origin = make_wp(0, 0, 0, 0.0, 0.0, 0.0);
        let local = wa.to_local_aabb(&origin);

        assert!((local.min.x - 2.0).abs() < 1e-5, "local.min.x = {}", local.min.x);
        assert!((local.max.x - 6.0).abs() < 1e-5, "local.max.x = {}", local.max.x);
        assert!((local.min.y).abs() < 1e-5);
        assert!((local.max.y - 4.0).abs() < 1e-5);
    }

    /// Two overlapping WorldAabbs report intersection.
    #[test]
    fn world_aabb_intersects_overlapping() {
        let a = WorldAabb::new(
            make_wp(0, 0, 0, 0.0, 0.0, 0.0),
            make_wp(0, 0, 0, 6.0, 6.0, 6.0),
        );
        let b = WorldAabb::new(
            make_wp(0, 0, 0, 4.0, 4.0, 4.0),
            make_wp(1, 1, 1, 0.0, 0.0, 0.0), // max at (8,8,8)
        );
        assert!(a.intersects(&b));
        assert!(b.intersects(&a));
    }

    /// Two non-overlapping WorldAabbs do not intersect.
    #[test]
    fn world_aabb_intersects_apart() {
        let a = WorldAabb::new(
            make_wp(0, 0, 0, 0.0, 0.0, 0.0),
            make_wp(0, 0, 0, 3.0, 3.0, 3.0),
        );
        let b = WorldAabb::new(
            make_wp(0, 0, 0, 5.0, 0.0, 0.0),
            make_wp(0, 0, 0, 7.0, 3.0, 3.0),
        );
        assert!(!a.intersects(&b));
        assert!(!b.intersects(&a));
    }
}
