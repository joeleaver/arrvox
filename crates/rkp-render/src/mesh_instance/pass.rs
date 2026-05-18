//! Shared per-instance render infrastructure.
//!
//! Owns the `MeshInstanceUniform` (per-scene-instance uniform — world
//! matrix, object id, bone-skinning state) and the bind-group layouts
//! every raster path consumes:
//!  * `g0` — scene-wide: camera + leaf_attr_pool + bone_matrices +
//!    bone_dual_quats. One bind group per pass.
//!  * `g1` — per-instance: a 96 B `MeshInstanceUniform`. One bind group
//!    per scene instance.

use crate::rkp_scene::CameraUniforms;

/// Per-instance uniform — one mat4 world transform, the entity's
/// `object_id` (written into the pick texture), and the per-instance
/// bone-skinning state. 96 B, multiple of 16. CPU mirror of the
/// `MeshInstance` struct in the mesh raster + shadow shaders.
///
/// **Skinning semantics:**
/// * `skinning_mode == SKINNING_MODE_NONE` → instance is not skinned
///   (no live bone matrices); the mesh VS skips skinning entirely
///   regardless of per-vertex `bone_weights` and emits the rest-pose
///   transform.
/// * `skinning_mode == 0` → linear blend skinning; the VS reads
///   `bone_matrices[bone_offset_lbs + bone_idx]` for the four
///   referenced bones, weighted-sums, and applies.
/// * `skinning_mode == 1` → dual-quaternion skinning; the VS reads
///   `bone_dual_quats[bone_offset_dqs + bone_idx]`, blends, and
///   normalises before applying.
///
/// The two offsets are independent — LBS and DQS palettes are sized
/// and packed separately by `BoneMatrixAllocator`. The unused offset
/// is harmless filler.
///
/// **`grid_origin`** is the asset's voxel-grid origin in object-local
/// (mesh-frame) coordinates — the same value `extract_surface_mesh`
/// added to every cell's centroid. The mesh VS subtracts it from
/// `local_pos` before applying bone matrices (which were trained
/// against grid-frame positions, origin at the octree corner) and
/// adds it back after, before the world transform. For unskinned
/// instances `grid_origin` is irrelevant because the VS skips the
/// subtract/add path entirely; the engine still uploads a sane value
/// for debugging.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshInstanceUniform {
    pub world: [[f32; 4]; 4],
    /// Asset's voxel-grid origin in mesh-frame; bridge between
    /// vertex local_pos (mesh-frame) and bone-matrix input frame
    /// (grid-frame, origin at octree corner).
    pub grid_origin: [f32; 3],
    pub object_id: u32,
    /// First index in `bone_matrices` for this instance's LBS palette.
    pub bone_offset_lbs: u32,
    /// First index in `bone_dual_quats` for this instance's DQS palette.
    pub bone_offset_dqs: u32,
    /// `0` = LBS, `1` = DQS, `SKINNING_MODE_NONE` = not skinned.
    pub skinning_mode: u32,
    /// Trailing pad bumps the struct to 96 B (16-aligned) to match
    /// WGSL's struct-alignment rule for uniform buffers.
    pub _pad: u32,
}

/// Sentinel `skinning_mode` value meaning "this instance carries no
/// live bone matrices; render rest pose." Lives in the value space of
/// `u32` outside the LBS / DQS enum so the VS can branch on it without
/// an extra "is_skinned" flag.
pub const SKINNING_MODE_NONE: u32 = u32::MAX;

const _: () = assert!(std::mem::size_of::<MeshInstanceUniform>() == 96);
// Hand-checked field offsets — must match the WGSL declaration in
// `mesh.wesl` / `mesh_shadow.wesl`. WGSL's uniform buffer layout
// treats vec3 as size-12-align-16, so a u32 following vec3 sits
// immediately after at offset 76 (no auto-pad to 80).
const _: () = {
    use std::mem::offset_of;
    assert!(offset_of!(MeshInstanceUniform, world) == 0);
    assert!(offset_of!(MeshInstanceUniform, grid_origin) == 64);
    assert!(offset_of!(MeshInstanceUniform, object_id) == 76);
    assert!(offset_of!(MeshInstanceUniform, bone_offset_lbs) == 80);
    assert!(offset_of!(MeshInstanceUniform, bone_offset_dqs) == 84);
    assert!(offset_of!(MeshInstanceUniform, skinning_mode) == 88);
    assert!(offset_of!(MeshInstanceUniform, _pad) == 92);
};
pub const MESH_INSTANCE_BYTES: u64 = std::mem::size_of::<MeshInstanceUniform>() as u64;

/// Shared per-instance bind-group layouts (g0 = scene-wide,
/// g1 = per-instance). Owned by every raster path the renderer
/// dispatches.
pub struct MeshInstanceLayouts {
    pub g0_layout: wgpu::BindGroupLayout,
    pub g1_layout: wgpu::BindGroupLayout,
}

impl MeshInstanceLayouts {
    pub fn new(device: &wgpu::Device) -> Self {
        // ── g0: scene-wide bindings ────────────────────────────────
        let g0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mesh g0"),
            entries: &[
                // camera (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(
                            std::mem::size_of::<CameraUniforms>() as u64,
                        ),
                    },
                    count: None,
                },
                // leaf_attr_pool (storage<read>) — vertex stage uses
                // `normal_oct` to build the disc basis. Materials and
                // colours are read by the resolve compute pass, not
                // here.
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // bone_matrices (storage<read>) — mesh-VS LBS skinning.
                // Carries the per-frame `mat3x4`-packed LBS palette
                // concatenated across all skinned entities; the
                // per-instance `bone_offset_lbs` indexes into it.
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // bone_dual_quats (storage<read>) — mesh-VS DQS
                // palette, parallel to `bone_matrices`. Indexed by
                // the per-instance `bone_offset_dqs`.
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // ── g1: per-instance uniform ───────────────────────────────
        // Visibility includes COMPUTE so the same per-VR
        // per-instance bind groups drive the mesh render pipelines
        // (vertex+fragment) AND the `mesh_lod_select` compute pass —
        // one bind group, one layout.
        let g1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mesh g1"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(MESH_INSTANCE_BYTES),
                },
                count: None,
            }],
        });

        Self { g0_layout, g1_layout }
    }

    /// Build the scene-wide `g0` bind group. Bound once per pass, before
    /// the per-instance draws.
    pub fn create_g0_bind_group(
        &self,
        device: &wgpu::Device,
        camera_buffer: &wgpu::Buffer,
        leaf_attr_pool_buffer: &wgpu::Buffer,
        bone_matrices_buffer: &wgpu::Buffer,
        bone_dual_quats_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh g0 bg"),
            layout: &self.g0_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: leaf_attr_pool_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bone_matrices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bone_dual_quats_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Build a per-instance `g1` bind group. One per scene-instance.
    pub fn create_g1_bind_group(
        &self,
        device: &wgpu::Device,
        instance_uniform_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh g1 bg"),
            layout: &self.g1_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: instance_uniform_buffer.as_entire_binding(),
            }],
        })
    }
}
