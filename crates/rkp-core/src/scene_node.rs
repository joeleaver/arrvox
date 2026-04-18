//! Scene hierarchy node — the core v2 data model.
//!
//! A [`SceneNode`] is a node in an object's SDF tree. Each node carries a
//! local [`Transform`], an [`SdfSource`] (geometry), a [`BlendMode`]
//! (combination rule), and zero or more children.
//!
//! # Design
//!
//! - Nodes form a tree hierarchy (no cross-references)
//! - Child transforms are relative to parent, not world space
//! - Blending is scoped to the tree — nodes in different root objects never blend
//! - Uniform scale only — non-uniform breaks SDF distances

use glam::{Mat4, Quat, Vec3};

use crate::aabb::Aabb;

/// Local transform: position, rotation, per-axis scale.
///
/// Non-uniform scale uses conservative `dist * min(sx, sy, sz)` for SDF
/// distance correction. Objects can be re-voxelized to eliminate the
/// march-step overhead from extreme scale ratios.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform {
    /// Offset from parent, in parent's local space.
    pub position: Vec3,
    /// Rotation relative to parent.
    pub rotation: Quat,
    /// Per-axis scale factor. All components must be positive.
    pub scale: Vec3,
}

impl Transform {
    /// Create a new transform.
    #[inline]
    pub fn new(position: Vec3, rotation: Quat, scale: Vec3) -> Self {
        Self {
            position,
            rotation,
            scale,
        }
    }

    /// Convert to a 4×4 matrix (scale × rotation × translation).
    #[inline]
    pub fn to_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(
            self.scale,
            self.rotation,
            self.position,
        )
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

/// How an SDF node contributes geometry.
#[derive(Debug, Clone)]
pub enum SdfSource {
    /// Pure transform node — no geometry. Used for grouping, skeleton joints,
    /// or hierarchy anchors. Zero memory cost.
    None,
    /// Analytical SDF primitive — evaluated as math during ray marching.
    /// Zero memory cost, infinite resolution.
    Analytical {
        /// The mathematical shape definition.
        primitive: SdfPrimitive,
        /// Material table index.
        material_id: u16,
    },
    /// Voxelized geometry — brick data from imported mesh or sculpting.
    Voxelized {
        /// Handle to this node's spatial data (brick map or octree).
        spatial_handle: SpatialHandle,
        /// World-space size of one voxel (e.g. 0.005, 0.02, 0.08).
        voxel_size: f32,
        /// Local-space bounding box.
        aabb: Aabb,
    },
}

impl Default for SdfSource {
    fn default() -> Self {
        Self::None
    }
}

/// Placeholder handle for per-object brick map storage.
///
/// In v2, each voxelized node owns a compact brick map. This handle
/// references its allocation in the `BrickMapAllocator` (Phase 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BrickMapHandle {
    /// Offset into the packed brick map buffer.
    pub offset: u32,
    /// Dimensions of the 3D brick map grid.
    pub dims: glam::UVec3,
}

/// Opaque handle to spatial data managed by the march pass.
///
/// The march pass stores and interprets spatial data in its own format.
/// The engine stores this handle per scene node and passes it back when
/// populating GpuObject fields or uploading spatial data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpatialHandle {
    /// Flat brick map (SDF ray march / legacy path).
    BrickMap(BrickMapHandle),
    /// Sparse octree (splat rasterization).
    Octree {
        root_offset: u32,
        len: u32,
        depth: u8,
        base_voxel_size: f32,
    },
}

/// How this node's SDF combines with its siblings.
///
/// Blending is scoped to the tree — nodes in different root objects never blend.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BlendMode {
    /// Smooth-min blend with radius parameter. Creates organic joins.
    /// This is the default mode.
    SmoothUnion(f32),
    /// Hard union (min). Sharp intersection lines.
    Union,
    /// Removes this node's volume from the combined sibling field.
    Subtract,
    /// Only overlapping volume survives.
    Intersect,
}

impl Default for BlendMode {
    fn default() -> Self {
        Self::SmoothUnion(0.1)
    }
}

/// Analytical SDF primitive — evaluated as math, zero memory cost.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum SdfPrimitive {
    /// Sphere centered at local origin.
    Sphere {
        /// Radius in metres.
        radius: f32,
    },
    /// Axis-aligned box centered at local origin.
    Box {
        /// Half-width, half-height, half-depth from center.
        half_extents: Vec3,
    },
    /// Capsule along the local Y axis.
    Capsule {
        /// Radius of the rounded caps and cylinder.
        radius: f32,
        /// Half-height of the cylindrical section.
        half_height: f32,
    },
    /// Torus in the XZ plane centered at local origin.
    Torus {
        /// Distance from center to middle of tube.
        major_radius: f32,
        /// Radius of the tube itself.
        minor_radius: f32,
    },
    /// Cylinder along the local Y axis.
    Cylinder {
        /// Radius of the circular cross-section.
        radius: f32,
        /// Half-height of the cylinder.
        half_height: f32,
    },
    /// Infinite plane.
    Plane {
        /// Normalized surface normal.
        normal: Vec3,
        /// Signed distance from origin along normal.
        distance: f32,
    },
}

/// Metadata for editor state (visibility, lock, selection).
///
/// These fields do not affect runtime rendering — they are editor-only state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeMetadata {
    /// Whether this node and all descendants are visible.
    pub visible: bool,
    /// Lock node from editing.
    pub locked: bool,
    /// Editor selection state.
    pub selected: bool,
    /// UI tree expansion state.
    pub expand_in_tree: bool,
}

impl Default for NodeMetadata {
    fn default() -> Self {
        Self {
            visible: true,
            locked: false,
            selected: false,
            expand_in_tree: true,
        }
    }
}

/// A node in an object's SDF tree.
///
/// Forms the core of the v2 scene hierarchy. Each node carries geometry
/// (via [`SdfSource`]), a local transform, a blend mode for sibling
/// combination, and zero or more children.
#[derive(Debug, Clone)]
pub struct SceneNode {
    /// Human-readable name.
    pub name: String,
    /// Local transform relative to parent.
    pub local_transform: Transform,
    /// How this node contributes geometry.
    pub sdf_source: SdfSource,
    /// How this node combines with siblings.
    pub blend_mode: BlendMode,
    /// Child nodes.
    pub children: Vec<SceneNode>,
    /// Editor metadata (visibility, lock, selection).
    pub metadata: NodeMetadata,
}

impl SceneNode {
    /// Create a new node with default transform, no geometry, and no children.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            local_transform: Transform::default(),
            sdf_source: SdfSource::None,
            blend_mode: BlendMode::default(),
            children: Vec::new(),
            metadata: NodeMetadata::default(),
        }
    }

    /// Create an analytical primitive node.
    pub fn analytical(
        name: impl Into<String>,
        primitive: SdfPrimitive,
        material_id: u16,
    ) -> Self {
        Self {
            name: name.into(),
            local_transform: Transform::default(),
            sdf_source: SdfSource::Analytical {
                primitive,
                material_id,
            },
            blend_mode: BlendMode::default(),
            children: Vec::new(),
            metadata: NodeMetadata::default(),
        }
    }

    /// Add a child node, returning `&mut Self` for chaining.
    pub fn add_child(&mut self, child: SceneNode) -> &mut Self {
        self.children.push(child);
        self
    }

    /// Set the local transform, returning `Self` for builder-style construction.
    pub fn with_transform(mut self, transform: Transform) -> Self {
        self.local_transform = transform;
        self
    }

    /// Set the blend mode, returning `Self` for builder-style construction.
    pub fn with_blend_mode(mut self, mode: BlendMode) -> Self {
        self.blend_mode = mode;
        self
    }

    /// Total number of nodes in this subtree (including self).
    pub fn node_count(&self) -> usize {
        1 + self.children.iter().map(|c| c.node_count()).sum::<usize>()
    }

    /// Find a node by name (depth-first search).
    pub fn find_by_name(&self, name: &str) -> Option<&SceneNode> {
        if self.name == name {
            return Some(self);
        }
        for child in &self.children {
            if let Some(found) = child.find_by_name(name) {
                return Some(found);
            }
        }
        None
    }

    /// Find a node by name (mutable, depth-first search).
    pub fn find_by_name_mut(&mut self, name: &str) -> Option<&mut SceneNode> {
        if self.name == name {
            return Some(self);
        }
        for child in &mut self.children {
            if let Some(found) = child.find_by_name_mut(name) {
                return Some(found);
            }
        }
        None
    }

    // ── Child access ────────────────────────────────────────────────────

    /// Number of direct children.
    pub fn child_count(&self) -> usize {
        self.children.len()
    }

    /// Get a child by index.
    pub fn child(&self, index: usize) -> Option<&SceneNode> {
        self.children.get(index)
    }

    /// Get a mutable child by index.
    pub fn child_mut(&mut self, index: usize) -> Option<&mut SceneNode> {
        self.children.get_mut(index)
    }

    /// Remove and return the child at `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= child_count()`.
    pub fn remove_child(&mut self, index: usize) -> SceneNode {
        self.children.remove(index)
    }

    /// Remove the first child with the given name, returning it.
    ///
    /// Returns `None` if no direct child has that name.
    pub fn remove_child_by_name(&mut self, name: &str) -> Option<SceneNode> {
        let pos = self.children.iter().position(|c| c.name == name)?;
        Some(self.children.remove(pos))
    }

    /// Insert a child at `index`, shifting existing children right.
    ///
    /// # Panics
    ///
    /// Panics if `index > child_count()`.
    pub fn insert_child_at(&mut self, index: usize, child: SceneNode) -> &mut Self {
        self.children.insert(index, child);
        self
    }

    /// Iterate over children (immutable).
    pub fn iter_children(&self) -> std::slice::Iter<'_, SceneNode> {
        self.children.iter()
    }

    /// Iterate over children (mutable).
    pub fn iter_children_mut(&mut self) -> std::slice::IterMut<'_, SceneNode> {
        self.children.iter_mut()
    }

    // ── Path-based access ───────────────────────────────────────────────

    /// Find a descendant by slash-separated path relative to this node.
    ///
    /// Example: `"spine/chest/head"` walks from this node → `spine` → `chest` → `head`.
    pub fn find_by_path(&self, path: &str) -> Option<&SceneNode> {
        if path.is_empty() {
            return None;
        }
        let mut current = self;
        for segment in path.split('/') {
            current = current.children.iter().find(|c| c.name == segment)?;
        }
        Some(current)
    }

    /// Find a descendant by slash-separated path (mutable).
    pub fn find_by_path_mut(&mut self, path: &str) -> Option<&mut SceneNode> {
        if path.is_empty() {
            return None;
        }
        let mut current = self;
        for segment in path.split('/') {
            current = current.children.iter_mut().find(|c| c.name == segment)?;
        }
        Some(current)
    }

    /// Depth-first pre-order traversal yielding `(depth, &SceneNode)`.
    pub fn walk(&self) -> Vec<(usize, &SceneNode)> {
        let mut result = Vec::new();
        self.walk_inner(0, &mut result);
        result
    }

    fn walk_inner<'a>(&'a self, depth: usize, out: &mut Vec<(usize, &'a SceneNode)>) {
        out.push((depth, self));
        for child in &self.children {
            child.walk_inner(depth + 1, out);
        }
    }

    /// Collect all node names in depth-first pre-order.
    pub fn all_names(&self) -> Vec<String> {
        self.walk().into_iter().map(|(_, n)| n.name.clone()).collect()
    }
}

impl std::fmt::Display for SceneNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SceneNode(\"{}\"", self.name)?;
        match &self.sdf_source {
            SdfSource::None => write!(f, ", None")?,
            SdfSource::Analytical { primitive, .. } => write!(f, ", {:?}", primitive)?,
            SdfSource::Voxelized { voxel_size, .. } => write!(f, ", Voxelized({voxel_size})")?,
        }
        if !self.children.is_empty() {
            write!(f, ", {} children", self.children.len())?;
        }
        write!(f, ")")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    #[test]
    fn transform_default_is_identity() {
        let t = Transform::default();
        assert_eq!(t.position, Vec3::ZERO);
        assert_eq!(t.rotation, Quat::IDENTITY);
        assert_eq!(t.scale, Vec3::ONE);
    }

    #[test]
    fn transform_to_matrix_identity() {
        let t = Transform::default();
        let m = t.to_matrix();
        let diff = (m - Mat4::IDENTITY).abs_diff_eq(Mat4::ZERO, 1e-6);
        assert!(diff, "identity transform should produce identity matrix");
    }

    #[test]
    fn transform_to_matrix_translation() {
        let t = Transform::new(Vec3::new(1.0, 2.0, 3.0), Quat::IDENTITY, Vec3::ONE);
        let m = t.to_matrix();
        let p = m.transform_point3(Vec3::ZERO);
        assert!((p - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-5);
    }

    #[test]
    fn transform_to_matrix_scale() {
        let t = Transform::new(Vec3::ZERO, Quat::IDENTITY, Vec3::splat(2.0));
        let m = t.to_matrix();
        let p = m.transform_point3(Vec3::ONE);
        assert!((p - Vec3::splat(2.0)).length() < 1e-5);
    }

    #[test]
    fn transform_to_matrix_rotation() {
        let t = Transform::new(Vec3::ZERO, Quat::from_rotation_z(FRAC_PI_2), Vec3::ONE);
        let m = t.to_matrix();
        let p = m.transform_point3(Vec3::X);
        // 90° Z rotation: X -> Y
        assert!((p - Vec3::Y).length() < 1e-4);
    }

    #[test]
    fn scene_node_new_defaults() {
        let node = SceneNode::new("test");
        assert_eq!(node.name, "test");
        assert!(matches!(node.sdf_source, SdfSource::None));
        assert!(matches!(node.blend_mode, BlendMode::SmoothUnion(_)));
        assert!(node.children.is_empty());
        assert!(node.metadata.visible);
        assert!(!node.metadata.locked);
    }

    #[test]
    fn scene_node_analytical() {
        let node = SceneNode::analytical(
            "sphere",
            SdfPrimitive::Sphere { radius: 0.5 },
            1,
        );
        assert_eq!(node.name, "sphere");
        match &node.sdf_source {
            SdfSource::Analytical {
                primitive,
                material_id,
            } => {
                assert!(matches!(primitive, SdfPrimitive::Sphere { radius } if (*radius - 0.5).abs() < 1e-6));
                assert_eq!(*material_id, 1);
            }
            _ => panic!("expected Analytical"),
        }
    }

    #[test]
    fn scene_node_builder_pattern() {
        let node = SceneNode::analytical(
            "box",
            SdfPrimitive::Box {
                half_extents: Vec3::ONE,
            },
            2,
        )
        .with_transform(Transform::new(Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY, Vec3::splat(0.5)))
        .with_blend_mode(BlendMode::Subtract);

        assert_eq!(node.local_transform.position, Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(node.local_transform.scale, Vec3::splat(0.5));
        assert!(matches!(node.blend_mode, BlendMode::Subtract));
    }

    #[test]
    fn scene_node_tree_building() {
        let mut root = SceneNode::new("root");
        let child_a = SceneNode::analytical(
            "arm_left",
            SdfPrimitive::Capsule {
                radius: 0.1,
                half_height: 0.3,
            },
            1,
        );
        let child_b = SceneNode::analytical(
            "arm_right",
            SdfPrimitive::Capsule {
                radius: 0.1,
                half_height: 0.3,
            },
            1,
        );
        root.add_child(child_a);
        root.add_child(child_b);

        assert_eq!(root.children.len(), 2);
        assert_eq!(root.node_count(), 3);
    }

    #[test]
    fn scene_node_deep_tree() {
        let mut root = SceneNode::new("hips");
        let mut spine = SceneNode::new("spine");
        let mut chest = SceneNode::new("chest");
        let head = SceneNode::analytical(
            "head",
            SdfPrimitive::Sphere { radius: 0.15 },
            1,
        );
        chest.add_child(head);
        spine.add_child(chest);
        root.add_child(spine);

        assert_eq!(root.node_count(), 4);
    }

    #[test]
    fn scene_node_find_by_name() {
        let mut root = SceneNode::new("root");
        let mut child = SceneNode::new("child");
        let grandchild = SceneNode::new("grandchild");
        child.add_child(grandchild);
        root.add_child(child);

        assert!(root.find_by_name("root").is_some());
        assert!(root.find_by_name("child").is_some());
        assert!(root.find_by_name("grandchild").is_some());
        assert!(root.find_by_name("missing").is_none());
    }

    #[test]
    fn scene_node_find_by_name_mut() {
        let mut root = SceneNode::new("root");
        let child = SceneNode::new("child");
        root.add_child(child);

        if let Some(node) = root.find_by_name_mut("child") {
            node.metadata.locked = true;
        }
        assert!(root.children[0].metadata.locked);
    }

    #[test]
    fn scene_node_display() {
        let node = SceneNode::analytical(
            "sphere",
            SdfPrimitive::Sphere { radius: 1.0 },
            0,
        );
        let s = format!("{node}");
        assert!(s.contains("sphere"));
        assert!(s.contains("Sphere"));
    }

    #[test]
    fn blend_mode_default_is_smooth_union() {
        let mode = BlendMode::default();
        assert!(matches!(mode, BlendMode::SmoothUnion(r) if r > 0.0));
    }

    #[test]
    fn node_metadata_default_is_visible_unlocked() {
        let m = NodeMetadata::default();
        assert!(m.visible);
        assert!(!m.locked);
        assert!(!m.selected);
        assert!(m.expand_in_tree);
    }

    #[test]
    fn all_sdf_primitives_constructable() {
        let _s = SdfPrimitive::Sphere { radius: 1.0 };
        let _b = SdfPrimitive::Box {
            half_extents: Vec3::ONE,
        };
        let _c = SdfPrimitive::Capsule {
            radius: 0.1,
            half_height: 0.5,
        };
        let _t = SdfPrimitive::Torus {
            major_radius: 1.0,
            minor_radius: 0.2,
        };
        let _cy = SdfPrimitive::Cylinder {
            radius: 0.5,
            half_height: 1.0,
        };
        let _p = SdfPrimitive::Plane {
            normal: Vec3::Y,
            distance: 0.0,
        };
    }

    // ── Child access (A.1) ───────────────────────────────────────────────

    #[test]
    fn child_count_empty() {
        let node = SceneNode::new("leaf");
        assert_eq!(node.child_count(), 0);
    }

    #[test]
    fn child_count_with_children() {
        let mut node = SceneNode::new("root");
        node.add_child(SceneNode::new("a"));
        node.add_child(SceneNode::new("b"));
        node.add_child(SceneNode::new("c"));
        assert_eq!(node.child_count(), 3);
    }

    #[test]
    fn child_by_index() {
        let mut node = SceneNode::new("root");
        node.add_child(SceneNode::new("first"));
        node.add_child(SceneNode::new("second"));
        assert_eq!(node.child(0).unwrap().name, "first");
        assert_eq!(node.child(1).unwrap().name, "second");
    }

    #[test]
    fn child_out_of_bounds_returns_none() {
        let node = SceneNode::new("root");
        assert!(node.child(0).is_none());
        assert!(node.child(99).is_none());
    }

    #[test]
    fn child_mut_by_index() {
        let mut node = SceneNode::new("root");
        node.add_child(SceneNode::new("child"));
        node.child_mut(0).unwrap().name = "renamed".to_string();
        assert_eq!(node.child(0).unwrap().name, "renamed");
    }

    #[test]
    fn remove_child_by_index() {
        let mut node = SceneNode::new("root");
        node.add_child(SceneNode::new("a"));
        node.add_child(SceneNode::new("b"));
        node.add_child(SceneNode::new("c"));
        let removed = node.remove_child(1);
        assert_eq!(removed.name, "b");
        assert_eq!(node.child_count(), 2);
        assert_eq!(node.child(0).unwrap().name, "a");
        assert_eq!(node.child(1).unwrap().name, "c");
    }

    #[test]
    fn remove_child_by_name_found() {
        let mut node = SceneNode::new("root");
        node.add_child(SceneNode::new("keep"));
        node.add_child(SceneNode::new("remove_me"));
        let removed = node.remove_child_by_name("remove_me");
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().name, "remove_me");
        assert_eq!(node.child_count(), 1);
    }

    #[test]
    fn remove_child_by_name_not_found() {
        let mut node = SceneNode::new("root");
        node.add_child(SceneNode::new("child"));
        assert!(node.remove_child_by_name("nope").is_none());
        assert_eq!(node.child_count(), 1);
    }

    #[test]
    fn insert_child_at_beginning() {
        let mut node = SceneNode::new("root");
        node.add_child(SceneNode::new("b"));
        node.insert_child_at(0, SceneNode::new("a"));
        assert_eq!(node.child(0).unwrap().name, "a");
        assert_eq!(node.child(1).unwrap().name, "b");
    }

    #[test]
    fn insert_child_at_middle() {
        let mut node = SceneNode::new("root");
        node.add_child(SceneNode::new("a"));
        node.add_child(SceneNode::new("c"));
        node.insert_child_at(1, SceneNode::new("b"));
        assert_eq!(node.child(0).unwrap().name, "a");
        assert_eq!(node.child(1).unwrap().name, "b");
        assert_eq!(node.child(2).unwrap().name, "c");
    }

    #[test]
    fn insert_child_at_end() {
        let mut node = SceneNode::new("root");
        node.add_child(SceneNode::new("a"));
        node.insert_child_at(1, SceneNode::new("b"));
        assert_eq!(node.child(1).unwrap().name, "b");
        assert_eq!(node.child_count(), 2);
    }

    #[test]
    fn iter_children_count() {
        let mut node = SceneNode::new("root");
        node.add_child(SceneNode::new("a"));
        node.add_child(SceneNode::new("b"));
        assert_eq!(node.iter_children().count(), node.child_count());
    }

    #[test]
    fn iter_children_mut_modify() {
        let mut node = SceneNode::new("root");
        node.add_child(SceneNode::new("a"));
        node.add_child(SceneNode::new("b"));
        for child in node.iter_children_mut() {
            child.metadata.locked = true;
        }
        assert!(node.child(0).unwrap().metadata.locked);
        assert!(node.child(1).unwrap().metadata.locked);
    }

    // ── Path-based access (A.2) ────────────────────────────────────────

    #[test]
    fn find_by_path_single_segment() {
        let mut root = SceneNode::new("root");
        root.add_child(SceneNode::new("child"));
        assert_eq!(root.find_by_path("child").unwrap().name, "child");
    }

    #[test]
    fn find_by_path_multi_segment() {
        let mut root = SceneNode::new("root");
        let mut spine = SceneNode::new("spine");
        let mut chest = SceneNode::new("chest");
        chest.add_child(SceneNode::new("head"));
        spine.add_child(chest);
        root.add_child(spine);
        assert_eq!(
            root.find_by_path("spine/chest/head").unwrap().name,
            "head"
        );
    }

    #[test]
    fn find_by_path_not_found() {
        let mut root = SceneNode::new("root");
        root.add_child(SceneNode::new("child"));
        assert!(root.find_by_path("nope").is_none());
        assert!(root.find_by_path("child/deep").is_none());
    }

    #[test]
    fn find_by_path_empty_returns_none() {
        let root = SceneNode::new("root");
        assert!(root.find_by_path("").is_none());
    }

    #[test]
    fn find_by_path_mut_modifies() {
        let mut root = SceneNode::new("root");
        let mut spine = SceneNode::new("spine");
        spine.add_child(SceneNode::new("chest"));
        root.add_child(spine);
        root.find_by_path_mut("spine/chest").unwrap().metadata.locked = true;
        assert!(root.find_by_path("spine/chest").unwrap().metadata.locked);
    }

    #[test]
    fn walk_yields_correct_order() {
        let mut root = SceneNode::new("root");
        let mut a = SceneNode::new("a");
        a.add_child(SceneNode::new("a1"));
        root.add_child(a);
        root.add_child(SceneNode::new("b"));
        let names: Vec<&str> = root.walk().iter().map(|(_, n)| n.name.as_str()).collect();
        assert_eq!(names, vec!["root", "a", "a1", "b"]);
    }

    #[test]
    fn walk_yields_correct_depths() {
        let mut root = SceneNode::new("root");
        let mut a = SceneNode::new("a");
        a.add_child(SceneNode::new("a1"));
        root.add_child(a);
        let depths: Vec<usize> = root.walk().iter().map(|(d, _)| *d).collect();
        assert_eq!(depths, vec![0, 1, 2]);
    }

    #[test]
    fn all_names_collects_tree() {
        let mut root = SceneNode::new("root");
        root.add_child(SceneNode::new("x"));
        root.add_child(SceneNode::new("y"));
        assert_eq!(root.all_names(), vec!["root", "x", "y"]);
    }

    #[test]
    fn all_blend_modes_constructable() {
        let _su = BlendMode::SmoothUnion(0.1);
        let _u = BlendMode::Union;
        let _s = BlendMode::Subtract;
        let _i = BlendMode::Intersect;
    }

    #[test]
    fn all_sdf_sources_constructable() {
        let _none = SdfSource::None;
        let _analytical = SdfSource::Analytical {
            primitive: SdfPrimitive::Sphere { radius: 1.0 },
            material_id: 0,
        };
        let _voxelized = SdfSource::Voxelized {
            spatial_handle: SpatialHandle::BrickMap(BrickMapHandle {
                offset: 0,
                dims: glam::UVec3::new(4, 4, 4),
            }),
            voxel_size: 0.02,
            aabb: Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0)),
        };
    }

    #[test]
    fn transform_combined_trs() {
        // Translation + rotation + scale combined
        let t = Transform::new(
            Vec3::new(5.0, 0.0, 0.0),
            Quat::from_rotation_z(FRAC_PI_2),
            Vec3::splat(2.0),
        );
        let m = t.to_matrix();
        // Point at (1, 0, 0) should be: scale(2) → (2,0,0), rotate 90°Z → (0,2,0), translate → (5,2,0)
        let p = m.transform_point3(Vec3::X);
        assert!((p.x - 5.0).abs() < 1e-4, "x: {}", p.x);
        assert!((p.y - 2.0).abs() < 1e-4, "y: {}", p.y);
        assert!(p.z.abs() < 1e-4, "z: {}", p.z);
    }
}
