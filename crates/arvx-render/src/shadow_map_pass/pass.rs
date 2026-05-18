//! `ShadowMapPass` — owns the shadow depth storage buffer + the
//! per-frame `LightCameraCsm` uniform that every shadow consumer
//! reads.
//!
//! Historical: this used to also own the work-list scatter compute
//! chain (clear / setup / emit / finalize / scatter) for the march
//! path. The mesh path replaces all that with `MeshShadowMapPass` —
//! depth raster from the light POV + a blit compute that copies the
//! per-cascade depth slices into `shadow_buffer` for shade to sample.
//! After the march retirement only the buffer + the uniform live
//! here; the dispatch chain is gone.

use super::types::{CSM_CASCADE_COUNT, LightCameraCsm};

/// Shadow-buffer owner. `shadow_buffer` is the atomic-u32-backed
/// `array<u32>` storage the mesh shadow blit writes (bit-cast f32
/// depths) and `arvx_shade` samples; `uniform_buffer` holds the
/// per-frame `LightCameraCsm` (all cascade matrices + cascade count
/// + far view-z splits) the engine fills before each shadow render.
pub struct ShadowMapPass {
    pub size: u32,
    pub shadow_buffer: wgpu::Buffer,
    pub uniform_buffer: wgpu::Buffer,
}

impl ShadowMapPass {
    pub fn new(device: &wgpu::Device, size: u32) -> Self {
        let shadow_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_map shadow_buffer"),
            // `CSM_CASCADE_COUNT` slices, contiguous: `[cascade *
            // size² + ty * size + tx]`. Mesh writes all 4 slices per
            // frame (cascade_count = 4).
            size: (size as u64) * (size as u64) * (CSM_CASCADE_COUNT as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_map light_camera_csm"),
            size: std::mem::size_of::<LightCameraCsm>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { size, shadow_buffer, uniform_buffer }
    }

    /// Resize `shadow_buffer` to back a new map size. Returns `true`
    /// when the buffer handle changed (caller must rebuild bind
    /// groups that reference it). `uniform_buffer` is size-invariant.
    pub fn resize_shadow_buffer(&mut self, device: &wgpu::Device, new_size: u32) -> bool {
        if new_size == self.size {
            return false;
        }
        self.size = new_size;
        self.shadow_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_map shadow_buffer"),
            size: (new_size as u64) * (new_size as u64) * (CSM_CASCADE_COUNT as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        true
    }
}
