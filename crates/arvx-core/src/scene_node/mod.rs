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

#[cfg(test)]
mod tests;
