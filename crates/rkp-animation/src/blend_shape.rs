//! Blend shape (morph target) support for facial and body deformation.
//!
//! Blend shapes are **additive distance offsets** applied to a SceneNode's
//! base SDF during ray march evaluation:
//! ```text
//! final_distance = base_distance
//!     + weight_smile * smile_delta
//!     + weight_blink * blink_delta
//!     + ...
//! ```
//!
//! Each blend shape stores a delta-SDF as sparse bricks covering only the
//! affected region. They target a specific SceneNode (`target_node`) and
//! modify that node's voxelized SDF data. Only active blend shapes (non-zero
//! weight) are evaluated.

use rkp_core::aabb::Aabb;
use serde::{Deserialize, Serialize};

// ─── BlendShape ──────────────────────────────────────────────────────────────

/// A blend shape (morph target) for facial/body deformation.
///
/// Each blend shape stores a delta-SDF: sparse bricks containing distance
/// offsets from a SceneNode's base SDF. When active (weight > 0), the delta
/// is added to the base distance during ray march evaluation of the target node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlendShape {
    /// Name of the blend shape (e.g., "smile", "blink_L").
    pub name: String,
    /// Name of the SceneNode this blend shape targets.
    /// The delta-SDF modifies this node's voxelized brick data.
    pub target_node: String,
    /// Current weight (0.0 = inactive, 1.0 = fully active).
    pub weight: f32,
    /// Start offset into the blend shape brick pool region.
    pub brick_offset: u32,
    /// Number of bricks in this blend shape's delta-SDF.
    pub brick_count: u32,
    /// Bounding box of the affected region (rest-pose local space).
    pub bounding_box: Aabb,
}

impl BlendShape {
    /// Create a new blend shape with weight initialised to 0.0.
    ///
    /// `target_node` is the name of the SceneNode whose SDF this blend shape modifies.
    pub fn new(
        name: impl Into<String>,
        target_node: impl Into<String>,
        brick_offset: u32,
        brick_count: u32,
        bounding_box: Aabb,
    ) -> Self {
        Self {
            name: name.into(),
            target_node: target_node.into(),
            weight: 0.0,
            brick_offset,
            brick_count,
            bounding_box,
        }
    }

    /// Whether this blend shape is active (non-zero weight).
    #[inline]
    pub fn is_active(&self) -> bool {
        self.weight.abs() > 1e-6
    }

    /// Set the weight, clamped to \[0.0, 1.0\].
    #[inline]
    pub fn set_weight(&mut self, w: f32) {
        self.weight = w.clamp(0.0, 1.0);
    }
}

// ─── BlendShapeSet ───────────────────────────────────────────────────────────

/// A set of blend shapes for a character (typically facial).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlendShapeSet {
    /// All blend shapes in this set.
    pub shapes: Vec<BlendShape>,
}

impl BlendShapeSet {
    /// Create a new blend shape set.
    pub fn new(shapes: Vec<BlendShape>) -> Self {
        Self { shapes }
    }

    /// Get only active blend shapes (weight > 0).
    pub fn active_shapes(&self) -> Vec<&BlendShape> {
        self.shapes.iter().filter(|s| s.is_active()).collect()
    }

    /// Find a blend shape by name.
    pub fn find(&self, name: &str) -> Option<&BlendShape> {
        self.shapes.iter().find(|s| s.name == name)
    }

    /// Find a mutable blend shape by name.
    pub fn find_mut(&mut self, name: &str) -> Option<&mut BlendShape> {
        self.shapes.iter_mut().find(|s| s.name == name)
    }

    /// Set weight for a named blend shape. Returns `true` if found.
    pub fn set_weight(&mut self, name: &str, weight: f32) -> bool {
        if let Some(shape) = self.find_mut(name) {
            shape.set_weight(weight);
            true
        } else {
            false
        }
    }

    /// Get active blend shapes targeting a specific SceneNode.
    pub fn active_shapes_for_node(&self, node_name: &str) -> Vec<&BlendShape> {
        self.shapes
            .iter()
            .filter(|s| s.is_active() && s.target_node == node_name)
            .collect()
    }

    /// Reset all weights to zero.
    pub fn reset_all(&mut self) {
        for shape in &mut self.shapes {
            shape.weight = 0.0;
        }
    }

    /// Total number of bricks used by active blend shapes.
    pub fn active_brick_count(&self) -> u32 {
        self.shapes
            .iter()
            .filter(|s| s.is_active())
            .map(|s| s.brick_count)
            .sum()
    }

    /// Compute the union bounding box of all active blend shapes.
    ///
    /// Returns `None` if no shapes are active.
    pub fn active_bounding_box(&self) -> Option<Aabb> {
        let active: Vec<_> = self.active_shapes();
        if active.is_empty() {
            return None;
        }
        let mut min = active[0].bounding_box.min;
        let mut max = active[0].bounding_box.max;
        for s in &active[1..] {
            min = min.min(s.bounding_box.min);
            max = max.max(s.bounding_box.max);
        }
        Some(Aabb::new(min, max))
    }
}

// ─── BlendShapeGpu ───────────────────────────────────────────────────────────

/// GPU-side blend shape parameters for the rebake shader.
///
/// Each active blend shape is passed as one of these to the shader.
/// Size: 48 bytes (verified by test).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlendShapeGpu {
    /// Weight of this blend shape.
    pub weight: f32,
    /// Brick offset in the blend shape pool.
    pub brick_offset: u32,
    /// Brick count.
    pub brick_count: u32,
    /// Padding for alignment.
    pub _pad: u32,
    /// Bounding box min (rest-pose local space).
    pub bbox_min: [f32; 3],
    /// Padding for vec3 alignment.
    pub _pad1: f32,
    /// Bounding box max.
    pub bbox_max: [f32; 3],
    /// Padding for vec3 alignment.
    pub _pad2: f32,
}

impl BlendShapeGpu {
    /// Create from a [`BlendShape`] reference.
    pub fn from_blend_shape(shape: &BlendShape) -> Self {
        Self {
            weight: shape.weight,
            brick_offset: shape.brick_offset,
            brick_count: shape.brick_count,
            _pad: 0,
            bbox_min: [
                shape.bounding_box.min.x,
                shape.bounding_box.min.y,
                shape.bounding_box.min.z,
            ],
            _pad1: 0.0,
            bbox_max: [
                shape.bounding_box.max.x,
                shape.bounding_box.max.y,
                shape.bounding_box.max.z,
            ],
            _pad2: 0.0,
        }
    }
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    use std::mem;

    fn make_aabb(min: Vec3, max: Vec3) -> Aabb {
        Aabb::new(min, max)
    }

    fn default_aabb() -> Aabb {
        make_aabb(Vec3::ZERO, Vec3::ONE)
    }

    // ── BlendShape ────────────────────────────────────────────────────────────

    #[test]
    fn blend_shape_new_sets_correct_fields() {
        let bb = make_aabb(Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0));
        let bs = BlendShape::new("smile", "Head", 10, 5, bb);
        assert_eq!(bs.name, "smile");
        assert_eq!(bs.target_node, "Head");
        assert_eq!(bs.weight, 0.0);
        assert_eq!(bs.brick_offset, 10);
        assert_eq!(bs.brick_count, 5);
        assert_eq!(bs.bounding_box.min, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(bs.bounding_box.max, Vec3::new(4.0, 5.0, 6.0));
    }

    #[test]
    fn blend_shape_is_active_false_at_zero() {
        let bs = BlendShape::new("blink_L", "Head", 0, 1, default_aabb());
        assert!(!bs.is_active());
    }

    #[test]
    fn blend_shape_is_active_true_at_half() {
        let mut bs = BlendShape::new("blink_L", "Head", 0, 1, default_aabb());
        bs.weight = 0.5;
        assert!(bs.is_active());
    }

    #[test]
    fn blend_shape_is_active_true_at_tiny_nonzero() {
        let mut bs = BlendShape::new("blink_R", "Head", 0, 1, default_aabb());
        bs.weight = 1e-5; // above 1e-6 threshold
        assert!(bs.is_active());
    }

    #[test]
    fn blend_shape_is_active_false_below_epsilon() {
        let mut bs = BlendShape::new("blink_R", "Head", 0, 1, default_aabb());
        bs.weight = 1e-7; // below 1e-6 threshold
        assert!(!bs.is_active());
    }

    #[test]
    fn blend_shape_set_weight_clamps_above_one() {
        let mut bs = BlendShape::new("jaw_open", "Head", 0, 2, default_aabb());
        bs.set_weight(1.5);
        assert!((bs.weight - 1.0).abs() < 1e-6);
    }

    #[test]
    fn blend_shape_set_weight_clamps_below_zero() {
        let mut bs = BlendShape::new("jaw_open", "Head", 0, 2, default_aabb());
        bs.set_weight(-0.5);
        assert!(bs.weight.abs() < 1e-6);
    }

    #[test]
    fn blend_shape_set_weight_midrange_unchanged() {
        let mut bs = BlendShape::new("brow_up", "Head", 0, 3, default_aabb());
        bs.set_weight(0.75);
        assert!((bs.weight - 0.75).abs() < 1e-6);
    }

    // ── BlendShapeSet ─────────────────────────────────────────────────────────

    fn make_set() -> BlendShapeSet {
        let shapes = vec![
            BlendShape::new("smile", "Head", 0, 4, make_aabb(Vec3::ZERO, Vec3::ONE)),
            BlendShape::new("blink_L", "Head", 4, 2, make_aabb(Vec3::new(1.0, 0.0, 0.0), Vec3::new(2.0, 1.0, 1.0))),
            BlendShape::new("blink_R", "Head", 6, 2, make_aabb(Vec3::new(2.0, 0.0, 0.0), Vec3::new(3.0, 1.0, 1.0))),
        ];
        BlendShapeSet::new(shapes)
    }

    #[test]
    fn blend_shape_set_active_shapes_filters_correctly() {
        let mut set = make_set();
        // Initially all inactive.
        assert!(set.active_shapes().is_empty());

        set.shapes[0].weight = 0.8;
        set.shapes[2].weight = 0.3;

        let active = set.active_shapes();
        assert_eq!(active.len(), 2);
        assert!(active.iter().any(|s| s.name == "smile"));
        assert!(active.iter().any(|s| s.name == "blink_R"));
        assert!(!active.iter().any(|s| s.name == "blink_L"));
    }

    #[test]
    fn blend_shape_set_find_by_name() {
        let set = make_set();
        let found = set.find("blink_L").expect("should find blink_L");
        assert_eq!(found.name, "blink_L");
        assert_eq!(found.brick_offset, 4);

        assert!(set.find("nonexistent").is_none());
    }

    #[test]
    fn blend_shape_set_set_weight_by_name() {
        let mut set = make_set();
        let found = set.set_weight("smile", 0.6);
        assert!(found);
        assert!((set.find("smile").unwrap().weight - 0.6).abs() < 1e-6);

        // Unknown name returns false.
        let not_found = set.set_weight("frown", 0.5);
        assert!(!not_found);
    }

    #[test]
    fn blend_shape_set_reset_all_zeros_weights() {
        let mut set = make_set();
        set.shapes[0].weight = 1.0;
        set.shapes[1].weight = 0.5;
        set.shapes[2].weight = 0.25;

        set.reset_all();

        for s in &set.shapes {
            assert!(s.weight.abs() < 1e-9, "shape {} still has weight {}", s.name, s.weight);
        }
        assert!(set.active_shapes().is_empty());
    }

    #[test]
    fn blend_shape_set_active_brick_count_sums_correctly() {
        let mut set = make_set();
        // All inactive → 0.
        assert_eq!(set.active_brick_count(), 0);

        set.shapes[0].weight = 1.0; // brick_count = 4
        set.shapes[2].weight = 0.5; // brick_count = 2
        assert_eq!(set.active_brick_count(), 6);
    }

    #[test]
    fn blend_shape_set_active_bounding_box_returns_none_when_all_inactive() {
        let set = make_set();
        assert!(set.active_bounding_box().is_none());
    }

    #[test]
    fn blend_shape_set_active_bounding_box_computes_union() {
        let mut set = make_set();
        // Activate smile (0..1) and blink_R (2..3 on x).
        set.shapes[0].weight = 1.0;
        set.shapes[2].weight = 0.5;

        let bb = set.active_bounding_box().expect("should return Some");
        // Union of [0,0,0]–[1,1,1] and [2,0,0]–[3,1,1]
        assert!((bb.min.x - 0.0).abs() < 1e-6);
        assert!((bb.min.y - 0.0).abs() < 1e-6);
        assert!((bb.min.z - 0.0).abs() < 1e-6);
        assert!((bb.max.x - 3.0).abs() < 1e-6);
        assert!((bb.max.y - 1.0).abs() < 1e-6);
        assert!((bb.max.z - 1.0).abs() < 1e-6);
    }

    #[test]
    fn blend_shape_set_active_bounding_box_single_active() {
        let mut set = make_set();
        set.shapes[1].weight = 0.9; // blink_L: [1,0,0]–[2,1,1]

        let bb = set.active_bounding_box().expect("should return Some");
        assert!((bb.min.x - 1.0).abs() < 1e-6);
        assert!((bb.max.x - 2.0).abs() < 1e-6);
    }

    // ── active_shapes_for_node ──────────────────────────────────────────────

    #[test]
    fn blend_shape_set_active_shapes_for_node_filters_by_target() {
        let shapes = vec![
            BlendShape::new("smile", "Head", 0, 4, make_aabb(Vec3::ZERO, Vec3::ONE)),
            BlendShape::new("flex_L", "UpperArm_L", 4, 2, make_aabb(Vec3::ZERO, Vec3::ONE)),
            BlendShape::new("blink", "Head", 6, 1, make_aabb(Vec3::ZERO, Vec3::ONE)),
        ];
        let mut set = BlendShapeSet::new(shapes);

        // Activate smile and blink (both target Head), and flex_L (targets UpperArm_L).
        set.shapes[0].weight = 1.0;
        set.shapes[1].weight = 0.5;
        set.shapes[2].weight = 0.3;

        let head_shapes = set.active_shapes_for_node("Head");
        assert_eq!(head_shapes.len(), 2);
        assert!(head_shapes.iter().any(|s| s.name == "smile"));
        assert!(head_shapes.iter().any(|s| s.name == "blink"));

        let arm_shapes = set.active_shapes_for_node("UpperArm_L");
        assert_eq!(arm_shapes.len(), 1);
        assert_eq!(arm_shapes[0].name, "flex_L");

        let none_shapes = set.active_shapes_for_node("Chest");
        assert!(none_shapes.is_empty());
    }

    // ── BlendShapeGpu ─────────────────────────────────────────────────────────

    #[test]
    fn blend_shape_gpu_from_blend_shape_produces_correct_fields() {
        let bb = make_aabb(Vec3::new(1.0, 2.0, 3.0), Vec3::new(7.0, 8.0, 9.0));
        let mut bs = BlendShape::new("test", "Head", 42, 7, bb);
        bs.weight = 0.35;

        let gpu = BlendShapeGpu::from_blend_shape(&bs);

        assert!((gpu.weight - 0.35).abs() < 1e-6);
        assert_eq!(gpu.brick_offset, 42);
        assert_eq!(gpu.brick_count, 7);
        assert_eq!(gpu._pad, 0);
        assert_eq!(gpu.bbox_min, [1.0, 2.0, 3.0]);
        assert_eq!(gpu._pad1, 0.0);
        assert_eq!(gpu.bbox_max, [7.0, 8.0, 9.0]);
        assert_eq!(gpu._pad2, 0.0);
    }

    #[test]
    fn blend_shape_gpu_is_pod_bytemuck_round_trip() {
        let bb = make_aabb(Vec3::new(-1.0, -2.0, -3.0), Vec3::new(4.0, 5.0, 6.0));
        let mut bs = BlendShape::new("pod_test", "Head", 100, 3, bb);
        bs.weight = 0.9;

        let gpu = BlendShapeGpu::from_blend_shape(&bs);

        // Cast to bytes and back — must be identical.
        let bytes: &[u8] = bytemuck::bytes_of(&gpu);
        assert_eq!(bytes.len(), mem::size_of::<BlendShapeGpu>());
        let roundtripped: BlendShapeGpu = *bytemuck::from_bytes(bytes);

        assert!((roundtripped.weight - gpu.weight).abs() < 1e-9);
        assert_eq!(roundtripped.brick_offset, gpu.brick_offset);
        assert_eq!(roundtripped.brick_count, gpu.brick_count);
        assert_eq!(roundtripped.bbox_min, gpu.bbox_min);
        assert_eq!(roundtripped.bbox_max, gpu.bbox_max);
    }

    #[test]
    fn blend_shape_gpu_size_is_48_bytes() {
        assert_eq!(mem::size_of::<BlendShapeGpu>(), 48);
    }
}
