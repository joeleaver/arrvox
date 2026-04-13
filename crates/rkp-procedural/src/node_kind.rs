//! Node type definitions — what each node in the procedural tree does.

use glam::Vec3;
use serde::{Deserialize, Serialize};

/// What a node does. Leaves produce geometry, combinators merge children.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeKind {
    // ── Leaves (analytical shapes) ──────────────────────────────────
    Sphere(SphereParams),
    Box(BoxParams),
    Capsule(CapsuleParams),
    Cylinder(CylinderParams),
    Torus(TorusParams),
    Plane(PlaneParams),

    // ── Combinators (boolean ops on children) ───────────────────────
    Union {
        material_combine: MaterialCombine,
    },
    Intersect {
        material_combine: MaterialCombine,
    },
    /// Subtract the second child from the first. Always preserves the
    /// base (first child) material — both primary and secondary.
    Subtract,
}

impl NodeKind {
    /// Whether this node kind is a leaf (no children, produces geometry directly).
    pub fn is_leaf(&self) -> bool {
        matches!(
            self,
            NodeKind::Sphere(_)
                | NodeKind::Box(_)
                | NodeKind::Capsule(_)
                | NodeKind::Cylinder(_)
                | NodeKind::Torus(_)
                | NodeKind::Plane(_)
        )
    }

    /// Whether this node kind is a combinator (operates on children).
    pub fn is_combinator(&self) -> bool {
        matches!(
            self,
            NodeKind::Union { .. } | NodeKind::Intersect { .. } | NodeKind::Subtract
        )
    }
}

/// How materials are combined at boolean boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum MaterialCombine {
    /// Higher opacity takes all (material + color).
    Winner,
    /// Winner's primary becomes output primary, loser's primary becomes output
    /// secondary, opacity ratio becomes blend weight. Lossy: existing secondary
    /// materials on both sides are dropped (two-slot limit).
    Layered,
    /// Smooth blend within a radius of equal opacity.
    Blend { radius: f32 },
}

impl Default for MaterialCombine {
    fn default() -> Self {
        Self::Winner
    }
}

// ── Shape parameters ────────────────────────────────────────────────────────

/// Sphere centered at local origin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SphereParams {
    pub radius: f32,
    /// Falloff distance: how far outside the surface opacity transitions from
    /// 1.0 to 0.0. Controls surface softness.
    pub falloff: f32,
    pub material_id: u16,
    pub color: Vec3,
}

impl Default for SphereParams {
    fn default() -> Self {
        Self {
            radius: 0.5,
            falloff: 0.1,
            material_id: 0,
            color: Vec3::ONE,
        }
    }
}

/// Axis-aligned box centered at local origin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoxParams {
    /// Half-extents along each axis.
    pub half_extents: Vec3,
    /// Edge rounding radius (0 = sharp edges).
    pub rounding: f32,
    pub falloff: f32,
    pub material_id: u16,
    pub color: Vec3,
}

impl Default for BoxParams {
    fn default() -> Self {
        Self {
            half_extents: Vec3::splat(0.5),
            rounding: 0.0,
            falloff: 0.1,
            material_id: 0,
            color: Vec3::ONE,
        }
    }
}

/// Capsule: line segment with radius, along Y axis in local space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapsuleParams {
    /// Half-height of the line segment (total height = 2 * half_height + 2 * radius).
    pub half_height: f32,
    pub radius: f32,
    pub falloff: f32,
    pub material_id: u16,
    pub color: Vec3,
}

impl Default for CapsuleParams {
    fn default() -> Self {
        Self {
            half_height: 0.5,
            radius: 0.25,
            falloff: 0.1,
            material_id: 0,
            color: Vec3::ONE,
        }
    }
}

/// Cylinder along Y axis in local space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CylinderParams {
    pub half_height: f32,
    pub radius: f32,
    pub falloff: f32,
    pub material_id: u16,
    pub color: Vec3,
}

impl Default for CylinderParams {
    fn default() -> Self {
        Self {
            half_height: 0.5,
            radius: 0.25,
            falloff: 0.1,
            material_id: 0,
            color: Vec3::ONE,
        }
    }
}

/// Torus in the XZ plane, centered at local origin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TorusParams {
    /// Distance from center to the tube center.
    pub major_radius: f32,
    /// Radius of the tube.
    pub minor_radius: f32,
    pub falloff: f32,
    pub material_id: u16,
    pub color: Vec3,
}

impl Default for TorusParams {
    fn default() -> Self {
        Self {
            major_radius: 0.5,
            minor_radius: 0.15,
            falloff: 0.1,
            material_id: 0,
            color: Vec3::ONE,
        }
    }
}

/// Infinite plane with Y-up normal at local origin. Opacity is 1.0 below,
/// falls off above.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaneParams {
    pub falloff: f32,
    pub material_id: u16,
    pub color: Vec3,
}

impl Default for PlaneParams {
    fn default() -> Self {
        Self {
            falloff: 0.1,
            material_id: 0,
            color: Vec3::ONE,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leaf_detection() {
        assert!(NodeKind::Sphere(SphereParams::default()).is_leaf());
        assert!(NodeKind::Box(BoxParams::default()).is_leaf());
        assert!(NodeKind::Torus(TorusParams::default()).is_leaf());
        assert!(!NodeKind::Union {
            material_combine: MaterialCombine::Winner
        }
        .is_leaf());
    }

    #[test]
    fn combinator_detection() {
        assert!(NodeKind::Union {
            material_combine: MaterialCombine::Winner
        }
        .is_combinator());
        assert!(NodeKind::Intersect {
            material_combine: MaterialCombine::Layered
        }
        .is_combinator());
        assert!(NodeKind::Subtract.is_combinator());
        assert!(!NodeKind::Sphere(SphereParams::default()).is_combinator());
    }
}
