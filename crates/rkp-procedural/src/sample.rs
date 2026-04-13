//! The output of evaluating a procedural node at a position.

use glam::Vec3;

/// Result of sampling a procedural node tree at a world-space position.
///
/// Carries opacity (the geometry), material IDs for the dual-material system,
/// and per-voxel color. This is the universal currency between nodes — leaves
/// produce samples, combinators merge them.
#[derive(Debug, Clone, Copy)]
pub struct Sample {
    /// Opacity: 0.0 (empty) to 1.0 (fully opaque).
    pub opacity: f32,
    /// Primary material ID (indexes into the material palette).
    pub material_id: u16,
    /// Secondary material ID for dual-material blending.
    pub secondary_material_id: u16,
    /// Blend weight between primary and secondary: 0.0 = primary, 1.0 = secondary.
    pub blend_weight: f32,
    /// Per-voxel color (linear RGB, 0..1).
    pub color: Vec3,
}

impl Sample {
    /// Empty sample — zero opacity, no material, black.
    pub const EMPTY: Self = Self {
        opacity: 0.0,
        material_id: 0,
        secondary_material_id: 0,
        blend_weight: 0.0,
        color: Vec3::ZERO,
    };

    /// Construct a sample with a single material and no color.
    pub fn new(opacity: f32, material_id: u16) -> Self {
        Self {
            opacity,
            material_id,
            secondary_material_id: 0,
            blend_weight: 0.0,
            color: Vec3::ZERO,
        }
    }

    /// Construct a sample with a single material and per-voxel color.
    pub fn with_color(opacity: f32, material_id: u16, color: Vec3) -> Self {
        Self {
            opacity,
            material_id,
            secondary_material_id: 0,
            blend_weight: 0.0,
            color,
        }
    }

    /// Whether this sample is effectively empty.
    pub fn is_empty(&self) -> bool {
        self.opacity <= 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_sample() {
        let s = Sample::EMPTY;
        assert!(s.is_empty());
        assert_eq!(s.opacity, 0.0);
        assert_eq!(s.material_id, 0);
    }

    #[test]
    fn new_sample() {
        let s = Sample::new(0.75, 42);
        assert!(!s.is_empty());
        assert_eq!(s.opacity, 0.75);
        assert_eq!(s.material_id, 42);
        assert_eq!(s.secondary_material_id, 0);
        assert_eq!(s.blend_weight, 0.0);
    }

    #[test]
    fn sample_with_color() {
        let s = Sample::with_color(1.0, 5, Vec3::new(0.5, 0.3, 0.1));
        assert_eq!(s.color, Vec3::new(0.5, 0.3, 0.1));
        assert_eq!(s.material_id, 5);
    }
}
