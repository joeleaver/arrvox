//! Arrvox volumetric rendering — fog and procedural clouds, split into
//! two independent compute passes so sky and non-sky work never mix in a
//! single shader invocation.
//!
//! Passes (per frame, in order):
//! 1. Fog march       (half-res, every pixel) — height-fog integration to
//!                    scene depth. Writes `fog_texture`.
//! 2. Cloud march     (half-res, sky tiles)    — near-field + slab cloud
//!                    march with temporal reprojection. Non-sky tiles
//!                    early-exit with an identity value so the composite
//!                    becomes a no-op there. Writes `cloud_texture`.
//! 3. History update  (half-res)               — copies this frame's cloud
//!                    output into the history buffer, marking non-sky
//!                    texels with a validity sentinel.
//! 4. Sun-attenuation (1 thread)               — integrates camera→sun
//!                    cloud density for readback-driven direct-sun dimming.
//! 5. Composite       (full-res)               — layers fog, then clouds,
//!                    over scene HDR into `output_texture`.
//!
//! Splitting fog from clouds was a deliberate architectural choice: the old
//! combined shader gated everything on `is_sky`, which meant cloud-specific
//! bindings (history, cloud params) were bound even for voxel pixels, the
//! boundary between sky and non-sky was handled by a neutral-marker dance
//! in a shared buffer, and a hardware bilinear sampler could bleed that
//! marker across validity edges — producing ghost outlines around voxel
//! silhouettes in motion. Separate shaders let each pass touch only the
//! bindings it needs and make the boundary explicit.

use std::sync::{Arc, atomic::{AtomicU32, AtomicBool, Ordering}};


/// Uniform parameters for the volumetric march.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VolumetricParams {
    pub cam_pos: [f32; 4],
    pub cam_forward: [f32; 4],
    pub cam_right: [f32; 4],
    pub cam_up: [f32; 4],
    pub sun_dir: [f32; 4],
    pub sun_color: [f32; 4],
    pub width: u32,
    pub height: u32,
    pub full_width: u32,
    pub full_height: u32,
    pub max_steps: u32,
    pub step_size: f32,
    pub near: f32,
    pub far: f32,
    pub fog_color: [f32; 4],
    pub fog_height: [f32; 4],
    pub frame_index: u32,
    pub vol_ambient_r: f32,
    pub vol_ambient_g: f32,
    pub vol_ambient_b: f32,
    pub prev_view_proj: [[f32; 4]; 4],
}

/// Cloud parameters.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CloudParams {
    pub altitude: [f32; 4],
    pub noise: [f32; 4],
    pub wind: [f32; 4],
    pub flags: [f32; 4],
    /// x = slab_steps, y = shadow_steps, z = detail_octaves, w = ms_octaves.
    pub quality: [f32; 4],
    /// x = taa_alpha, y..w reserved.
    pub quality2: [f32; 4],
}

impl Default for CloudParams {
    fn default() -> Self {
        Self {
            altitude: [1000.0, 3000.0, 0.1, 1.0],
            noise: [0.0003, 0.002, 0.3, 10000.0],
            wind: [1.0, 0.0, 5.0, 0.0],
            flags: [0.0, 0.0, 0.0, 0.0], // disabled by default
            quality: [32.0, 4.0, 4.0, 3.0],
            quality2: [0.25, 0.0, 0.0, 0.0],
        }
    }
}

/// Volumetric rendering pass (fog + cloud + composite).
pub struct ArvxVolumetricPass {
    // Fog march — 3 bindings: params, depth, fog_out.
    fog_march_pipeline: wgpu::ComputePipeline,
    fog_march_bind_group_layout: wgpu::BindGroupLayout,
    fog_march_bind_group: Option<wgpu::BindGroup>,

    // Cloud march — 6 bindings: params, depth, cloud_out, cloud_params, history, history sampler.
    cloud_march_pipeline: wgpu::ComputePipeline,
    cloud_march_bind_group_layout: wgpu::BindGroupLayout,
    cloud_march_bind_group: Option<wgpu::BindGroup>,

    composite_pipeline: wgpu::ComputePipeline,
    composite_bind_group_layout: wgpu::BindGroupLayout,
    composite_bind_group: Option<wgpu::BindGroup>,

    params_buffer: wgpu::Buffer,
    cloud_params_buffer: wgpu::Buffer,

    /// Half-res cloud scatter+transmittance output. Identity (0,0,0,1) on
    /// non-sky pixels so the cloud composite step is a no-op there.
    pub cloud_texture: wgpu::Texture,
    pub cloud_view: wgpu::TextureView,

    /// Half-res fog scatter+transmittance output (all pixels, no TAA).
    pub fog_texture: wgpu::Texture,
    pub fog_view: wgpu::TextureView,

    /// Previous-frame cloud buffer — sampled in the cloud march for temporal
    /// accumulation via a manual 4-tap bilateral with per-texel validity
    /// rejection. No hardware sampler is used (would blend across the
    /// validity boundary and bleed the invalid-pixel marker into valid
    /// samples, producing ghost outlines around voxel silhouettes).
    history_texture: wgpu::Texture,
    history_view: wgpu::TextureView,

    /// Full-res composited HDR output (replaces shade output for tone mapping).
    pub output_texture: wgpu::Texture,
    pub output_view: wgpu::TextureView,

    half_width: u32,
    half_height: u32,
    width: u32,
    height: u32,

    depth_view_set: bool,

    // Selective history update — copies current scatter into history only for
    // sky pixels, leaving object pixels untouched so their stale values don't
    // bleed into sky reprojection next frame.
    history_update_pipeline: wgpu::ComputePipeline,
    history_update_bind_group: Option<wgpu::BindGroup>,
    history_update_bind_group_layout: wgpu::BindGroupLayout,

    // Cloud → sun attenuation: tiny compute pass + async readback.
    sun_atten_pipeline: wgpu::ComputePipeline,
    sun_atten_bind_group: wgpu::BindGroup,
    sun_atten_storage: wgpu::Buffer,  // GPU-only, written by compute
    sun_atten_staging: wgpu::Buffer,  // MAP_READ copy of storage
    /// Latest received exp(-τ·k) value as f32-bits. Updated by map_async callback.
    sun_atten_value: Arc<AtomicU32>,
    /// Raw τ from last readback, for debugging.
    sun_atten_tau_bits: Arc<AtomicU32>,
    /// True while a map_async call is in flight. Blocks re-issuing the readback
    /// until the previous map completes (single-buffer design, lags 1–2 frames
    /// under normal GPU pacing).
    sun_atten_map_pending: Arc<AtomicBool>,
}

mod constructor;

impl ArvxVolumetricPass {
    /// Set the depth view (G-buffer position texture). Rebuilds all bind groups
    /// that depend on it — fog march, cloud march, and history update.
    pub fn set_depth_view(&mut self, device: &wgpu::Device, depth_view: &wgpu::TextureView) {
        self.fog_march_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vol fog march bg"),
            layout: &self.fog_march_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.fog_view) },
            ],
        }));
        self.cloud_march_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vol cloud march bg"),
            layout: &self.cloud_march_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.cloud_view) },
                wgpu::BindGroupEntry { binding: 3, resource: self.cloud_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.history_view) },
            ],
        }));
        self.history_update_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vol history update bg"),
            layout: &self.history_update_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.cloud_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.history_view) },
            ],
        }));
        self.depth_view_set = true;
    }

    /// Copy the current scatter output into the history buffer for next-frame
    /// reprojection — but only for sky pixels. Object pixels leave history
    /// untouched so their transient values don't bleed into sky reprojection.
    pub fn update_history(&self, encoder: &mut wgpu::CommandEncoder) {
        let Some(bg) = &self.history_update_bind_group else { return };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("vol history update"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.history_update_pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(
            (self.half_width + 7) / 8,
            (self.half_height + 7) / 8,
            1,
        );
    }

    /// Set the scene HDR view (shade pass output). Rebuilds composite bind group.
    pub fn set_scene_hdr_view(&mut self, device: &wgpu::Device, hdr_view: &wgpu::TextureView) {
        self.composite_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vol composite bg"),
            layout: &self.composite_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(hdr_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.cloud_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.output_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.fog_view) },
            ],
        }));
    }

    /// Update volumetric parameters.
    pub fn update_params(&self, queue: &wgpu::Queue, params: &VolumetricParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Update cloud parameters.
    pub fn update_cloud_params(&self, queue: &wgpu::Queue, cloud: &CloudParams) {
        queue.write_buffer(&self.cloud_params_buffer, 0, bytemuck::bytes_of(cloud));
    }

    /// Dispatch the fog march (half-res, every pixel).
    pub fn dispatch_fog_march(&self, encoder: &mut wgpu::CommandEncoder) {
        let bg = match &self.fog_march_bind_group { Some(bg) => bg, None => return };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("vol fog march"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.fog_march_pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(
            (self.half_width + 7) / 8,
            (self.half_height + 7) / 8,
            1,
        );
    }

    /// Dispatch the cloud march (half-res; sky tiles do work, non-sky
    /// tiles early-return with an identity output). Must run before
    /// `update_history` — the history copy reads the cloud output this pass writes.
    pub fn dispatch_cloud_march(&self, encoder: &mut wgpu::CommandEncoder) {
        let bg = match &self.cloud_march_bind_group { Some(bg) => bg, None => return };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("vol cloud march"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.cloud_march_pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(
            (self.half_width + 7) / 8,
            (self.half_height + 7) / 8,
            1,
        );
    }

    /// Dispatch the 1-thread sun-attenuation compute pass and queue the GPU→CPU
    /// copy into the staging buffer. The value becomes readable after the next
    /// submit completes and the map_async callback fires (see `issue_sun_atten_map`).
    ///
    /// **Skips the copy** if a previous frame's map_async hasn't been consumed
    /// yet — staging is still mapped in that case and writing to it would
    /// trigger a wgpu validation error. The compute still runs (storage gets
    /// the fresh value); the next frame's copy will pick it up.
    pub fn dispatch_sun_atten(&self, encoder: &mut wgpu::CommandEncoder) {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vol sun atten"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sun_atten_pipeline);
            pass.set_bind_group(0, &self.sun_atten_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        if !self.sun_atten_map_pending.load(Ordering::Acquire) {
            encoder.copy_buffer_to_buffer(&self.sun_atten_storage, 0, &self.sun_atten_staging, 0, 16);
        }
    }

    /// True when a `map_async` is still in flight on the staging buffer
    /// (its callback hasn't fired yet). The encoder block must avoid
    /// writing to the staging while this is true.
    pub fn sun_atten_map_pending(&self) -> bool {
        self.sun_atten_map_pending.load(Ordering::Acquire)
    }

    /// After submit, issue a non-blocking map on the staging buffer. The callback
    /// writes the f32 bits into `sun_atten_value` when the GPU catches up. Skipped
    /// if a prior map is still pending (single-buffer design — one read at a time).
    pub fn issue_sun_atten_map(&self) {
        if self.sun_atten_map_pending.load(Ordering::Acquire) {
            return;
        }
        self.sun_atten_map_pending.store(true, Ordering::Release);

        let value = self.sun_atten_value.clone();
        let tau_bits = self.sun_atten_tau_bits.clone();
        let pending = self.sun_atten_map_pending.clone();
        let staging_for_cb = self.sun_atten_staging.clone();
        self.sun_atten_staging.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            if result.is_ok() {
                let data = staging_for_cb.slice(..).get_mapped_range();
                let atten = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                let tau = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
                value.store(atten, Ordering::Release);
                tau_bits.store(tau, Ordering::Release);
                drop(data);
                staging_for_cb.unmap();
            }
            pending.store(false, Ordering::Release);
        });
    }

    /// Latest received exp(-τ) value (updated asynchronously by `issue_sun_atten_map`).
    pub fn sun_atten_value(&self) -> f32 {
        f32::from_bits(self.sun_atten_value.load(Ordering::Acquire))
    }

    /// Raw τ from the last readback, for debugging. Staging buffer holds
    /// (exp(-τ·k), τ, 0, 0).
    pub fn sun_atten_tau_debug(&self) -> f32 {
        f32::from_bits(self.sun_atten_tau_bits.load(Ordering::Acquire))
    }

    /// Dispatch the volumetric composite (full-res).
    pub fn dispatch_composite(&self, encoder: &mut wgpu::CommandEncoder) {
        let bg = match &self.composite_bind_group { Some(bg) => bg, None => return };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("vol composite"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.composite_pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(
            (self.width + 7) / 8,
            (self.height + 7) / 8,
            1,
        );
    }

    /// Resize textures. Call when window resizes.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let hw = (width / 2).max(1);
        let hh = (height / 2).max(1);
        if hw == self.half_width && hh == self.half_height { return; }
        self.half_width = hw;
        self.half_height = hh;
        self.width = width;
        self.height = height;
        let (ct, cv) = Self::create_march_output_texture(device, "vol cloud", hw, hh);
        self.cloud_texture = ct;
        self.cloud_view = cv;
        let (ft, fv) = Self::create_march_output_texture(device, "vol fog", hw, hh);
        self.fog_texture = ft;
        self.fog_view = fv;
        let (ht, hv) = Self::create_history_texture(device, "vol cloud history", hw, hh);
        self.history_texture = ht;
        self.history_view = hv;
        let (ot, ov) = Self::create_texture(device, "vol output", width, height, wgpu::TextureFormat::Rgba16Float);
        self.output_texture = ot;
        self.output_view = ov;
        // Bind groups need rebuild — caller must call set_depth_view + set_scene_hdr_view.
    }

    fn create_texture(
        device: &wgpu::Device,
        label: &str,
        w: u32,
        h: u32,
        format: wgpu::TextureFormat,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                 | wgpu::TextureUsages::TEXTURE_BINDING
                 | wgpu::TextureUsages::COPY_SRC
                 | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (tex, view)
    }

    fn create_march_output_texture(
        device: &wgpu::Device,
        label: &str,
        w: u32,
        h: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (tex, view)
    }

    fn create_history_texture(
        device: &wgpu::Device,
        label: &str,
        w: u32,
        h: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (tex, view)
    }
}
