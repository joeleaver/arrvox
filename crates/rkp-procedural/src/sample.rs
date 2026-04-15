//! The output of evaluating a procedural node at a position.

use glam::Vec3;

/// Result of sampling a procedural node tree at a world-space position.
///
/// Carries signed distance (the geometry), material IDs for the dual-material
/// system, and per-voxel color. This is the universal currency between nodes —
/// leaves produce samples, combinators merge them with SDF algebra.
///
/// # Distance semantics
///
/// * `distance < 0` — the point is **inside** the surface.
/// * `distance = 0` — the point is on the surface.
/// * `distance > 0` — the point is **outside** the surface.
///
/// Primitives produce proper 1-Lipschitz signed distances. Boolean combinators
/// (union / intersect / subtract) use `min` / `max` / `max(a, -b)` which also
/// preserve the Lipschitz property; the octree voxelizer depends on this.
#[derive(Debug, Clone, Copy)]
pub struct Sample {
    /// Signed distance to the surface. Negative = inside.
    pub distance: f32,
    /// Primary material ID (indexes into the material palette).
    pub material_id: u16,
    /// Secondary material ID for dual-material blending.
    pub secondary_material_id: u16,
    /// Blend weight between primary and secondary: 0.0 = primary, 1.0 = secondary.
    pub blend_weight: f32,
    /// Per-voxel color (linear RGB, 0..1).
    pub color: Vec3,
}

/// A distance value treated as "infinitely far outside" — used when a combine
/// operation has no inputs, or when a Sample is logically empty. Big enough
/// that any voxelizer's coarse-level classifier will treat it as far-empty
/// regardless of the scene's scale.
pub const FAR_OUTSIDE: f32 = 1.0e9;

impl Sample {
    /// "Empty" sample — effectively far outside any surface, no material.
    pub const EMPTY: Self = Self {
        distance: FAR_OUTSIDE,
        material_id: 0,
        secondary_material_id: 0,
        blend_weight: 0.0,
        color: Vec3::ZERO,
    };

    /// Construct a sample with a single material and no color.
    pub fn new(distance: f32, material_id: u16) -> Self {
        Self {
            distance,
            material_id,
            secondary_material_id: 0,
            blend_weight: 0.0,
            color: Vec3::ZERO,
        }
    }

    /// Construct a sample with a single material and per-voxel color.
    pub fn with_color(distance: f32, material_id: u16, color: Vec3) -> Self {
        Self {
            distance,
            material_id,
            secondary_material_id: 0,
            blend_weight: 0.0,
            color,
        }
    }

    /// Whether this sample is effectively outside any surface (no geometry
    /// here). Used by combinators that want to short-circuit when a branch
    /// contributes nothing.
    pub fn is_empty(&self) -> bool {
        self.distance >= FAR_OUTSIDE * 0.5
    }

    /// Whether this sample is at or inside a surface (distance <= 0).
    pub fn is_inside(&self) -> bool {
        self.distance <= 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_sample() {
        let s = Sample::EMPTY;
        assert!(s.is_empty());
        assert!(!s.is_inside());
        assert_eq!(s.material_id, 0);
    }

    #[test]
    fn new_sample() {
        let s = Sample::new(-0.5, 42);
        assert!(!s.is_empty());
        assert!(s.is_inside());
        assert_eq!(s.distance, -0.5);
        assert_eq!(s.material_id, 42);
        assert_eq!(s.secondary_material_id, 0);
        assert_eq!(s.blend_weight, 0.0);
    }

    #[test]
    fn sample_with_color() {
        let s = Sample::with_color(0.0, 5, Vec3::new(0.5, 0.3, 0.1));
        assert_eq!(s.color, Vec3::new(0.5, 0.3, 0.1));
        assert_eq!(s.material_id, 5);
        assert!(s.is_inside());
    }

    #[test]
    fn far_outside_is_empty() {
        let s = Sample::new(FAR_OUTSIDE, 0);
        assert!(s.is_empty());
    }
}
