//! Built-in ECS components for the RKIPatch engine.
//!
//! These are the standard components that the engine knows about.
//! Additional components can be registered via the ComponentRegistry.

use serde::{Deserialize, Serialize};

/// Spatial transform — position, rotation, scale.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transform {
    pub position: glam::Vec3,
    /// Euler rotation in degrees (XYZ order).
    pub rotation: glam::Vec3,
    pub scale: glam::Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: glam::Vec3::ZERO,
            rotation: glam::Vec3::ZERO,
            scale: glam::Vec3::ONE,
        }
    }
}

/// Editor-only metadata (name, locked status, etc.).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EditorMetadata {
    pub name: String,
}

/// Renderable geometry — references a voxelized asset or analytical primitive.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Renderable {
    /// Path to the .rkp asset file (relative to project assets/).
    pub asset_path: Option<String>,
    /// Primitive type if this is an analytical object ("box", "sphere", etc.).
    pub primitive: Option<String>,
    /// Material slot index.
    pub material_id: u16,
    /// Number of voxels (populated after voxelization).
    #[serde(default)]
    pub voxel_count: u32,
    /// Octree spatial reference (populated after voxelization/loading).
    #[serde(skip)]
    pub spatial: Option<SpatialData>,
}

/// Octree spatial data for a renderable entity. Not serialized — rebuilt on load.
#[derive(Debug, Clone)]
pub struct SpatialData {
    pub root_offset: u32,
    pub len: u32,
    pub depth: u8,
    pub base_voxel_size: f32,
    pub aabb: rkf_core::Aabb,
    pub voxel_size: f32,
}

/// Point light source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointLight {
    pub color: [f32; 3],
    pub intensity: f32,
    pub range: f32,
}

impl Default for PointLight {
    fn default() -> Self {
        Self {
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
            range: 10.0,
        }
    }
}

/// Camera entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Camera {
    pub fov: f32,
    pub near: f32,
    pub far: f32,
    pub active: bool,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            fov: 60.0,
            near: 0.01,
            far: 1000.0,
            active: false,
        }
    }
}
