//! Per-viewport render-target and post-process state.
//!
//! [`RkpRenderer`] owns the device-wide pipelines and shared scene buffers;
//! the resolution-coupled outputs live here so multiple viewports can each
//! render into their own G-buffer + bloom chain. Phase 2 has only one
//! viewport (the engine creates a `ViewportRenderer` per
//! [`rkp_engine::viewport::Viewport`] entry), but the type is shaped for
//! the multi-viewport phases to come.
//!
//! The [`render_to`](crate::rkp_renderer::RkpRenderer::render_to) entry
//! point on the renderer wires its bind groups against this struct's
//! G-buffer before dispatching the march, then runs the bloom / tonemap /
//! composite chain that lives here.

use crate::rkp_renderer::RkpRenderer;

/// Per-viewport render targets and post-process passes.
pub struct ViewportRenderer {
    pub gbuffer: rkf_render::GBuffer,
    pub bloom: rkf_render::BloomPass,
    pub bloom_composite: rkf_render::BloomCompositePass,
    pub tone_map: rkf_render::ToneMapPass,
    /// Final LDR target (tonemapped + wireframe overlay). Used as the
    /// readback source.
    pub composite_texture: wgpu::Texture,
    pub composite_view: wgpu::TextureView,
    /// Double-buffered readback so we never block the GPU. We copy into
    /// `readback_buffers[readback_index]` this frame and map
    /// `readback_buffers[1 - readback_index]` (last frame's data).
    pub readback_buffers: [wgpu::Buffer; 2],
    pub readback_index: usize,
    /// `false` until at least one frame has been copied. Avoids reading
    /// stale memory on the first frame.
    pub readback_ready: bool,
    /// Wireframe overlay pass (gizmos drawn over the composite).
    pub wireframe_pass: rkf_render::WireframePass,
    pub width: u32,
    pub height: u32,
}

impl ViewportRenderer {
    /// Build a viewport renderer at the given size. Wires the supplied
    /// `RkpRenderer`'s march/shade/etc. bind groups to this G-buffer, and
    /// chains bloom from the renderer's god-ray output. The renderer must
    /// already be sized to `(width, height)`; in Phase 2 the engine only
    /// has one viewport so this is automatic.
    pub fn new(
        device: &wgpu::Device,
        renderer: &mut RkpRenderer,
        width: u32,
        height: u32,
    ) -> Self {
        let gbuffer = rkf_render::GBuffer::new(device, width, height);
        renderer.set_gbuffer(&gbuffer);

        let bloom = rkf_render::BloomPass::new(
            device,
            &renderer.god_rays.output_view,
            width,
            height,
        );
        let bloom_composite = rkf_render::BloomCompositePass::new(
            device,
            &renderer.god_rays.output_view,
            bloom.mip_views(),
            width,
            height,
        );
        let tone_map = rkf_render::ToneMapPass::new(
            device,
            &bloom_composite.output_view,
            width,
            height,
        );

        let readback_buffers = [
            create_readback_buffer(device, width, height),
            create_readback_buffer(device, width, height),
        ];

        let wireframe_pass = rkf_render::WireframePass::new(device, rkf_render::LDR_FORMAT);

        let composite_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("rkp composite"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: rkf_render::LDR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let composite_view = composite_texture.create_view(&Default::default());

        Self {
            gbuffer,
            bloom,
            bloom_composite,
            tone_map,
            composite_texture,
            composite_view,
            readback_buffers,
            readback_index: 0,
            readback_ready: false,
            wireframe_pass,
            width,
            height,
        }
    }

    /// Re-create per-resolution resources at a new size and re-wire the
    /// shared renderer's bind groups against the new G-buffer. Called when
    /// the host surface resizes. The double-buffered readback resets to
    /// "not ready" so the next frame doesn't read stale dimensions.
    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        renderer: &mut RkpRenderer,
        width: u32,
        height: u32,
    ) {
        if width == self.width && height == self.height {
            return;
        }
        self.width = width;
        self.height = height;

        self.gbuffer = rkf_render::GBuffer::new(device, width, height);
        renderer.set_gbuffer(&self.gbuffer);

        self.bloom = rkf_render::BloomPass::new(
            device,
            &renderer.god_rays.output_view,
            width,
            height,
        );
        self.bloom_composite = rkf_render::BloomCompositePass::new(
            device,
            &renderer.god_rays.output_view,
            self.bloom.mip_views(),
            width,
            height,
        );
        self.tone_map = rkf_render::ToneMapPass::new(
            device,
            &self.bloom_composite.output_view,
            width,
            height,
        );

        self.readback_buffers = [
            create_readback_buffer(device, width, height),
            create_readback_buffer(device, width, height),
        ];
        self.readback_ready = false;

        self.composite_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("rkp composite"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: rkf_render::LDR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.composite_view = self.composite_texture.create_view(&Default::default());
    }

    /// Padded bytes-per-row for the readback buffer (wgpu requires 256-B
    /// alignment for buffer copies, so the row stride may exceed
    /// `width * 4`).
    pub fn readback_padded_row(&self) -> u32 {
        (self.width * 4 + 255) & !255
    }

    /// Encode the GPU→CPU copy of the composite texture into the active
    /// readback buffer. Pair with [`Self::advance_readback`] after submit
    /// so the next frame reads this one's data.
    pub fn copy_composite_to_readback(&self, encoder: &mut wgpu::CommandEncoder) {
        let padded_row = self.readback_padded_row();
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.composite_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback_buffers[self.readback_index],
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// After submit + readback map, advance to the other buffer for the
    /// next frame's GPU copy. The first call also sets `readback_ready` so
    /// future frames know they can safely map the previous buffer.
    pub fn advance_readback(&mut self) {
        self.readback_ready = true;
        self.readback_index = 1 - self.readback_index;
    }
}

fn create_readback_buffer(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Buffer {
    let padded_row = (width * 4 + 255) & !255;
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rkp readback"),
        size: (padded_row * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}
