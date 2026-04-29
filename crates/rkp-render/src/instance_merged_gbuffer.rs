//! Stage 6c-2 — output target for the Option B instance composite pass.
//!
//! Sibling of [`crate::gbuffer::GBuffer`]'s primary four targets
//! (position, normal, material, leaf_slot) at the same formats. The
//! composite pass writes here; Stage 6c-4 will rebind the shade pass
//! to read from this set instead of the host G-buffer, so the final
//! image incorporates instance hits where they win the depth test.
//!
//! ## Why a separate struct (not extension to GBuffer)
//!
//! `GBuffer` carries seven targets (position, normal, material, motion,
//! glass, leaf_slot, depth) plus two bind-group layouts (write-only
//! storage, sampled read). Stage 6c only needs four — adding the others
//! would duplicate ~32 MB of motion-vector storage per viewport at
//! 1080p with no benefit. A small focused struct keeps the wiring
//! transparent: "this is the instance composite's output, nothing
//! else."
//!
//! ## Sizing
//!
//! Per-viewport, full-resolution. At 1920×1080 the four textures total
//! ~17 MB (position 16 B/px, normal 8 B/px, material 8 B/px, leaf_slot
//! 4 B/px = 36 B/px ≈ 75 MB; correction: 1920·1080·36 ≈ 75 MB). Plus
//! the per-pixel `InstanceMarchHit` buffer at 48 B/px ≈ 100 MB. The
//! editor's MAIN + BUILD viewports together pay this cost; keep an eye
//! on it as Stage 6e (perf) lands.

use crate::gbuffer::{
    GBUFFER_LEAF_SLOT_FORMAT, GBUFFER_MATERIAL_FORMAT, GBUFFER_NORMAL_FORMAT,
    GBUFFER_POSITION_FORMAT,
};

/// Four textures matching the host `GBuffer`'s primary targets, used
/// as the output of [`crate::instance_composite_pass::InstanceCompositePass`].
pub struct InstanceMergedGBuffer {
    pub position_texture: wgpu::Texture,
    pub position_view: wgpu::TextureView,
    pub normal_texture: wgpu::Texture,
    pub normal_view: wgpu::TextureView,
    pub material_texture: wgpu::Texture,
    pub material_view: wgpu::TextureView,
    pub leaf_slot_texture: wgpu::Texture,
    pub leaf_slot_view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
}

impl InstanceMergedGBuffer {
    /// Allocate the four textures at the given resolution. Each carries
    /// `STORAGE_BINDING | TEXTURE_BINDING | COPY_SRC` so the composite
    /// can write via storage and downstream passes (Stage 6c-4 shade
    /// rebind) can read via sampled. `COPY_SRC` keeps the door open for
    /// debug-readback work later.
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let usage = wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC;

        let make = |label: &str, format: wgpu::TextureFormat| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };

        let position_texture = make("inst_merged_gbuf position", GBUFFER_POSITION_FORMAT);
        let position_view = position_texture.create_view(&Default::default());
        let normal_texture = make("inst_merged_gbuf normal", GBUFFER_NORMAL_FORMAT);
        let normal_view = normal_texture.create_view(&Default::default());
        let material_texture = make("inst_merged_gbuf material", GBUFFER_MATERIAL_FORMAT);
        let material_view = material_texture.create_view(&Default::default());
        let leaf_slot_texture = make("inst_merged_gbuf leaf_slot", GBUFFER_LEAF_SLOT_FORMAT);
        let leaf_slot_view = leaf_slot_texture.create_view(&Default::default());

        Self {
            position_texture, position_view,
            normal_texture, normal_view,
            material_texture, material_view,
            leaf_slot_texture, leaf_slot_view,
            width, height,
        }
    }

    /// Total bytes the four textures consume at this resolution.
    /// Useful for diagnostics (see Stage 6e perf work).
    pub fn texture_bytes(&self) -> u64 {
        merged_gbuffer_texture_bytes(self.width, self.height)
    }
}

/// Total bytes the four textures consume at the given resolution.
/// Free function so tests + perf telemetry can compute the answer
/// without holding a live `InstanceMergedGBuffer` (which needs a real
/// device).
///
/// position 16 B/px, normal 8 B/px, material 8 B/px, leaf_slot 4 B/px
/// → 36 B/px total.
pub fn merged_gbuffer_texture_bytes(width: u32, height: u32) -> u64 {
    (width as u64) * (height as u64) * 36
}

/// Bytes the per-pixel `InstanceMarchHit` storage buffer consumes —
/// 48 B per pixel. At 1920×1080 this is ~99.5 MB; keep an eye on
/// totals as Stage 6e perf work lands.
pub fn output_hits_buffer_bytes(width: u32, height: u32) -> u64 {
    (width as u64) * (height as u64)
        * std::mem::size_of::<crate::instance_march_pass::InstanceMarchHit>() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texture_bytes_matches_36_per_pixel() {
        assert_eq!(merged_gbuffer_texture_bytes(100, 50), 100 * 50 * 36);
        // 1080p sanity: 1920×1080 = 2_073_600 px × 36 B ≈ 74.7 MB.
        assert_eq!(merged_gbuffer_texture_bytes(1920, 1080), 1920 * 1080 * 36);
    }

    #[test]
    fn output_hits_bytes_matches_48_per_pixel() {
        assert_eq!(output_hits_buffer_bytes(100, 50), 100 * 50 * 48);
        // 1080p sanity: 2_073_600 px × 48 B ≈ 99.5 MB.
        assert_eq!(output_hits_buffer_bytes(1920, 1080), 1920 * 1080 * 48);
    }

    #[test]
    fn output_hits_zero_dim_is_zero() {
        // Avoids panic on a viewport that hasn't been resized yet.
        assert_eq!(output_hits_buffer_bytes(0, 0), 0);
        assert_eq!(merged_gbuffer_texture_bytes(0, 0), 0);
    }
}
