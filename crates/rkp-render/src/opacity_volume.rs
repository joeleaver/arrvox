//! Opacity volume manager — creates procedural geometry volumes for the splat pipeline.
//!
//! When a material with an opacity shader (grass, fur, etc.) is painted onto a
//! surface, an [`OpacityVolume`] is created above the painted region. The volume
//! is a BVH object with no brick map — the splat march evaluates the opacity
//! shader directly at each step.

use std::collections::HashMap;

/// A procedural volume for opacity shader geometry (grass, fur, etc.).
///
/// The volume is inserted into the BVH as a GpuObject with `geom_type = PROCEDURAL`.
/// It has NO brick map — the march evaluates the opacity shader at each step.
#[derive(Debug, Clone)]
pub struct OpacityVolume {
    /// Unique ID for this volume (used as GpuObject::object_id).
    pub id: u32,
    /// Parent object that this volume extends.
    pub parent_object_id: u32,
    /// Material ID that has the opacity shader.
    pub material_id: u16,
    /// ShaderComposer ID for the opacity shader.
    pub shader_id: u32,
    /// Maximum height above the surface for procedural geometry.
    pub shell_height: f32,
    /// Voxel size of the parent (for march step size).
    pub voxel_size: f32,
    /// World-space AABB of the volume (painted region + shell_height above).
    pub world_aabb_min: [f32; 3],
    pub world_aabb_max: [f32; 3],
    /// Parent's inverse_world matrix (for local space transformation).
    pub inverse_world: [[f32; 4]; 4],
    /// Surface Y level in the parent's local space (for h_above computation).
    pub surface_y: f32,
    /// Whether the volume needs to be rebuilt.
    pub dirty: bool,
}

/// Manages opacity volumes for the splat pipeline.
pub struct OpacityVolumeManager {
    /// Volumes keyed by (parent_object_id, material_id).
    volumes: HashMap<(u32, u16), OpacityVolume>,
    /// Next unique volume ID.
    next_id: u32,
}

impl OpacityVolumeManager {
    /// Create a new empty manager.
    pub fn new() -> Self {
        Self {
            volumes: HashMap::new(),
            next_id: 0x8000_0000, // High range to avoid collision with scene object IDs
        }
    }

    /// Get or create a volume for the given (object_id, material_id) pair.
    pub fn get_or_create(
        &mut self,
        parent_object_id: u32,
        material_id: u16,
        shader_id: u32,
        shell_height: f32,
    ) -> &mut OpacityVolume {
        let key = (parent_object_id, material_id);
        self.volumes.entry(key).or_insert_with(|| {
            let id = self.next_id;
            self.next_id += 1;
            OpacityVolume {
                id,
                parent_object_id,
                material_id,
                shader_id,
                shell_height,
                voxel_size: 0.05, // default, updated when parent is known
                world_aabb_min: [0.0; 3],
                world_aabb_max: [0.0; 3],
                inverse_world: [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                surface_y: 0.0,
                dirty: true,
            }
        })
    }

    /// Iterate all volumes.
    pub fn all_volumes(&self) -> impl Iterator<Item = &OpacityVolume> {
        self.volumes.values()
    }

    /// Remove all volumes for a parent object.
    pub fn remove_for_object(&mut self, parent_object_id: u32) {
        self.volumes.retain(|k, _| k.0 != parent_object_id);
    }

    /// Check if any volumes exist.
    pub fn is_empty(&self) -> bool {
        self.volumes.is_empty()
    }

    /// Clear all volumes.
    pub fn clear(&mut self) {
        self.volumes.clear();
    }
}

impl Default for OpacityVolumeManager {
    fn default() -> Self {
        Self::new()
    }
}
