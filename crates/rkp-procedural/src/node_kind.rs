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
    Ramp(RampParams),

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

    // ── Effects (single-child modifiers) ────────────────────────────
    /// Domain-warp the child's SDF by a 3D simplex-noise vector field.
    /// First child is the operand; additional children are ignored.
    ///
    /// Not strictly a "combinator" in the boolean-op sense, but
    /// `is_combinator()` returns true so it shares the add-child-menu
    /// affordance and tree-widget handling.
    NoiseDisplace(NoiseDisplaceParams),
    /// Mirror the child subtree across an axis-aligned plane. Pointwise
    /// position fold `p -> abs(p[axis] - offset) + offset` — so the
    /// child's geometry on the +axis side is reflected to the -axis
    /// side for free. The fold is length-preserving (1-Lipschitz) so
    /// the child's SDF remains a valid distance; no conservative shrink
    /// needed. First child is the operand; additional children are
    /// ignored (same single-child effect cap as `NoiseDisplace`).
    Mirror(MirrorParams),
    /// Rewrite the child sample's primary material according to a
    /// 3-band rule on the sample point's local Y. Geometry is
    /// untouched. Within a `transition_width` zone around each
    /// threshold, the engine's dual-material path carries both
    /// adjacent bands for smooth seams (primary=below,
    /// secondary=above, blend=smoothstep alpha). Single-child effect.
    MaterialByHeight(MaterialByHeightParams),
    /// Rewrite the child sample's per-voxel color according to a
    /// 3-band rule on the sample point's local Y. Geometry and
    /// material are untouched. Adjacent band colors lerp across a
    /// `transition_width` zone. Single-child effect.
    ColorByHeight(ColorByHeightParams),
    /// Rewrite primary material from a 3-band rule on an FBM noise
    /// sample at the effect-local position. Same dual-material
    /// writeback as `MaterialByHeight`; difference is the classifier
    /// input (noise scalar vs. local Y). Single-child effect.
    MaterialByNoise(MaterialByNoiseParams),
    /// Rewrite per-voxel color from a 3-band rule on an FBM noise
    /// sample at the effect-local position. Orthogonal to
    /// `MaterialByNoise` — stacking them rewrites both channels from
    /// their own noise fields (different seeds give independent
    /// patterns). Single-child effect.
    ColorByNoise(ColorByNoiseParams),
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
                | NodeKind::Ramp(_)
        )
    }

    /// Whether this node kind is a combinator (operates on children).
    ///
    /// Effects (like `NoiseDisplace`) are included here: they're not
    /// boolean-op combinators, but they *do* take a child subtree and
    /// benefit from the same UI affordances (add-child "+" button, drop
    /// targets). Callers that need the stricter "boolean op" meaning
    /// should match on the specific variants instead.
    pub fn is_combinator(&self) -> bool {
        matches!(
            self,
            NodeKind::Union { .. }
                | NodeKind::Intersect { .. }
                | NodeKind::Subtract
                | NodeKind::NoiseDisplace(_)
                | NodeKind::Mirror(_)
                | NodeKind::MaterialByHeight(_)
                | NodeKind::ColorByHeight(_)
                | NodeKind::MaterialByNoise(_)
                | NodeKind::ColorByNoise(_)
        )
    }

    /// Maximum number of children this node kind accepts, or `None`
    /// for unbounded. Single-child effects (NoiseDisplace and the
    /// future warp/mirror family) cap at 1 — the evaluator and flatten
    /// both ignore extras anyway, so the cap is the source of truth.
    /// Leaves return `Some(0)`: `add_child` panics on them (preserving
    /// long-standing behavior), and the cap lets the UI hide the "+"
    /// without special-casing leaves.
    pub fn max_children(&self) -> Option<usize> {
        if self.is_leaf() {
            Some(0)
        } else if matches!(
            self,
            NodeKind::NoiseDisplace(_)
                | NodeKind::Mirror(_)
                | NodeKind::MaterialByHeight(_)
                | NodeKind::ColorByHeight(_)
                | NodeKind::MaterialByNoise(_)
                | NodeKind::ColorByNoise(_)
        ) {
            Some(1)
        } else {
            None
        }
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
    pub material_id: u16,
    pub color: Vec3,
}

impl Default for SphereParams {
    fn default() -> Self {
        Self {
            radius: 0.5,
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
    pub material_id: u16,
    pub color: Vec3,
}

impl Default for BoxParams {
    fn default() -> Self {
        Self {
            half_extents: Vec3::splat(0.5),
            rounding: 0.0,
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
    pub material_id: u16,
    pub color: Vec3,
}

impl Default for CapsuleParams {
    fn default() -> Self {
        Self {
            half_height: 0.5,
            radius: 0.25,
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
    pub material_id: u16,
    pub color: Vec3,
}

impl Default for CylinderParams {
    fn default() -> Self {
        Self {
            half_height: 0.5,
            radius: 0.25,
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
    pub material_id: u16,
    pub color: Vec3,
}

impl Default for TorusParams {
    fn default() -> Self {
        Self {
            major_radius: 0.5,
            minor_radius: 0.15,
            material_id: 0,
            color: Vec3::ONE,
        }
    }
}

/// Infinite plane with Y-up normal at local origin. Occupied below y=0,
/// empty above.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaneParams {
    pub material_id: u16,
    pub color: Vec3,
}

impl Default for PlaneParams {
    fn default() -> Self {
        Self {
            material_id: 0,
            color: Vec3::ONE,
        }
    }
}

/// Ramp: triangular prism centered at local origin. The cross-section is a
/// right triangle in the XY plane with the right-angle at the tall corner
/// (+X, +Y). The slope rises from (-X, -Y) to (+X, +Y), extruded along Z.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RampParams {
    /// Half-extent along X (length of the base).
    pub half_length: f32,
    /// Half-extent along Y (height at the tall end).
    pub half_height: f32,
    /// Half-extent along Z (width of the prism).
    pub half_width: f32,
    pub material_id: u16,
    pub color: Vec3,
}

impl Default for RampParams {
    fn default() -> Self {
        Self {
            half_length: 0.5,
            half_height: 0.25,
            half_width: 0.5,
            material_id: 0,
            color: Vec3::ONE,
        }
    }
}

/// Noise-displacement effect. Warps the sample position by a 3D
/// simplex-noise vector field before recursing into the child
/// subtree — pointwise (no caching needed), so the cost is one noise
/// evaluation per sample plus whatever the child costs.
///
/// The noise is evaluated in the effect's local frame (post-ancestor-
/// transform), so dragging the effect around in the scene shifts the
/// noise pattern with the object rather than sliding surfaces through
/// a fixed world-space field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseDisplaceParams {
    /// Maximum displacement magnitude along each axis, in local units.
    /// A sphere of radius 0.5 with amplitude 0.1 becomes a bumpy ball
    /// with protrusions up to ~10 % its radius.
    pub amplitude: f32,
    /// Spatial frequency of the noise, in cycles per local unit.
    /// Higher = tighter bumps. ~1.0 gives features on the scale of
    /// the primitive itself; 4.0 gives tight stipple.
    pub frequency: f32,
    /// Number of FBM octaves to layer. 1 = plain simplex. 2–4 adds
    /// detail at progressively smaller scales. Capped at 8 in
    /// evaluation to keep per-sample cost bounded.
    pub octaves: u32,
    /// Seed for the noise permutation. Change to re-roll the pattern
    /// without changing amplitude / frequency.
    pub seed: u32,
}

impl Default for NoiseDisplaceParams {
    fn default() -> Self {
        Self {
            amplitude: 0.1,
            frequency: 2.0,
            octaves: 3,
            seed: 0,
        }
    }
}

/// Axis-aligned mirror plane, named by the axis it is perpendicular to.
/// Mirror across X flips along X and leaves Y/Z untouched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MirrorAxis {
    X,
    Y,
    Z,
}

impl MirrorAxis {
    /// GPU/flatten encoding: 0=X, 1=Y, 2=Z. Kept in sync with the
    /// shader's `OP_PUSH_MIRROR` axis decode.
    pub fn to_u32(self) -> u32 {
        match self {
            MirrorAxis::X => 0,
            MirrorAxis::Y => 1,
            MirrorAxis::Z => 2,
        }
    }

    /// Inverse of `to_u32`. Any out-of-range value clamps to X so the
    /// decoder never panics on a malformed instruction stream.
    pub fn from_u32(v: u32) -> Self {
        match v {
            1 => MirrorAxis::Y,
            2 => MirrorAxis::Z,
            _ => MirrorAxis::X,
        }
    }
}

/// Mirror effect params. Applies the fold `p[axis] -> abs(p[axis])`
/// in the effect's local frame before evaluating the child — so the
/// +axis-side child geometry is reflected onto the -axis side across
/// the plane through the node's local origin. To position the mirror
/// plane in world space, move/rotate the Mirror node itself via its
/// transform (same pattern as leaves: a sphere's center, a torus's
/// ring center, etc., all come from the node transform rather than
/// a dedicated position field on the params).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MirrorParams {
    /// Which axis to mirror across.
    pub axis: MirrorAxis,
}

impl Default for MirrorParams {
    fn default() -> Self {
        Self {
            axis: MirrorAxis::X,
        }
    }
}

/// Material-by-height effect params. Three bands along the effect's
/// local Y axis, separated by `low_to_mid` and `mid_to_high`
/// thresholds. `transition_width` widens each threshold into a
/// smooth-blend zone (`± width / 2` around each) that the engine
/// renders via its dual-material path. `transition_width = 0` gives
/// hard band edges.
///
/// To rotate the banding direction (e.g. horizontal stratification)
/// or shift the thresholds in world space, use the node transform —
/// the height test runs in the effect's local frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialByHeightParams {
    pub low_material: u16,
    pub low_to_mid: f32,
    pub mid_material: u16,
    pub mid_to_high: f32,
    pub high_material: u16,
    pub transition_width: f32,
}

impl Default for MaterialByHeightParams {
    fn default() -> Self {
        Self {
            low_material: 0,
            low_to_mid: 0.0,
            mid_material: 0,
            mid_to_high: 1.0,
            high_material: 0,
            transition_width: 0.0,
        }
    }
}

/// Color-by-height effect params. Same band structure as
/// `MaterialByHeightParams`, but rewrites `Sample::color` — the
/// per-voxel RGB tint — leaving `material_id` alone. Adjacent band
/// colors lerp in linear RGB across the `transition_width` zone
/// around each threshold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorByHeightParams {
    pub low_color: Vec3,
    pub low_to_mid: f32,
    pub mid_color: Vec3,
    pub mid_to_high: f32,
    pub high_color: Vec3,
    pub transition_width: f32,
}

impl Default for ColorByHeightParams {
    fn default() -> Self {
        Self {
            low_color: Vec3::new(0.4, 0.3, 0.2),
            low_to_mid: 0.0,
            mid_color: Vec3::new(0.3, 0.6, 0.2),
            mid_to_high: 1.0,
            high_color: Vec3::new(0.95, 0.95, 0.95),
            transition_width: 0.0,
        }
    }
}

/// Material-by-noise effect params. Classifies an FBM noise sample
/// at the effect-local position into three bands, same smooth-seam
/// dual-material writeback as `MaterialByHeightParams`. Noise output
/// sits in roughly `[-1, 1]`, so thresholds in that range are the
/// natural setting; `transition_width` lives on the same scale.
///
/// The noise is evaluated in the effect's local frame so moving /
/// rotating the node shifts the pattern with it — consistent with
/// `NoiseDisplace` once you anchor the field to the local frame via
/// the `inverse_world` in flatten's post-op emit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialByNoiseParams {
    pub low_material: u16,
    pub low_to_mid: f32,
    pub mid_material: u16,
    pub mid_to_high: f32,
    pub high_material: u16,
    pub transition_width: f32,
    /// Spatial frequency of the noise in cycles per local unit —
    /// larger = tighter speckle.
    pub frequency: f32,
    /// Permutation seed; change to re-roll the pattern without
    /// changing frequency / octaves.
    pub seed: u32,
    /// Number of FBM octaves. Clamped to `[1, 8]` in evaluation.
    pub octaves: u32,
}

impl Default for MaterialByNoiseParams {
    fn default() -> Self {
        Self {
            low_material: 0,
            low_to_mid: -0.3,
            mid_material: 0,
            mid_to_high: 0.3,
            high_material: 0,
            transition_width: 0.1,
            frequency: 2.0,
            seed: 0,
            octaves: 3,
        }
    }
}

/// Color-by-noise effect params. Same band structure as
/// `ColorByNoiseParams`'s material sibling, but rewrites
/// `Sample::color` with a Vec3 lerp across transition zones.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorByNoiseParams {
    pub low_color: Vec3,
    pub low_to_mid: f32,
    pub mid_color: Vec3,
    pub mid_to_high: f32,
    pub high_color: Vec3,
    pub transition_width: f32,
    pub frequency: f32,
    pub seed: u32,
    pub octaves: u32,
}

impl Default for ColorByNoiseParams {
    fn default() -> Self {
        Self {
            low_color: Vec3::new(0.4, 0.3, 0.2),
            low_to_mid: -0.3,
            mid_color: Vec3::new(0.3, 0.6, 0.2),
            mid_to_high: 0.3,
            high_color: Vec3::new(0.95, 0.95, 0.95),
            transition_width: 0.1,
            frequency: 2.0,
            seed: 0,
            octaves: 3,
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
