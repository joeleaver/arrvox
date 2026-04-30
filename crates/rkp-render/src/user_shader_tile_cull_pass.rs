//! Phase 6 Session 2 — user-shader tile-cull AABB compute pipeline.
//!
//! Per filled instance slot in `instance_pool`, dispatches the user
//! shader's `inst_aabb` hook to compute a world-space AABB, then writes
//! one [`InstanceTileCullEntry`] into a scratch buffer that Session 3's
//! tile-cull (count + prefix + scatter) consumes.
//!
//! ## Dispatch shape
//!
//! One dispatch per [`crate::user_shader_emit_pass::InstanceRegionRequest`].
//! Each thread is one slot in the region's `instance_pool` reservation.
//! Threads with `gid >= instance_alloc[region_index]` write a dead
//! placeholder (`live = 0`); threads with `gid >= block_size` early-
//! return. Workgroup size is 64; dispatch count is
//! `instance_block_size.div_ceil(64)`.
//!
//! ## Scratch layout
//!
//! Scratch is a single flat `array<InstanceTileCullEntry>` covering
//! every region's reservation. Each region's slice starts at its
//! `scratch_offset` (cumulative sum of prior regions' `block_size`).
//! Session 3 walks the entire array; the `live` flag gates which
//! entries become tile-cull candidates.
//!
//! ## Wiring
//!
//! Mirrors the construction shape of
//! [`crate::user_shader_emit_pass::EmitPass`]:
//!
//! * group(0) — `instance_pool` (read), `instance_alloc` (read),
//!   `tile_cull_scratch` (read_write).
//! * group(1) — `TileCullRegionUniform` (uniform, dynamic offset, 256 B
//!   stride for wgpu's dynamic-offset alignment requirement).
//!
//! Pipeline rebuild flow matches `EmitPass::reload_user_shaders` /
//! `OctreeMarchPass::reload_user_shaders` exactly so the engine can
//! call all of them with the same `frame.user_shader_source_hash`.

use crate::shader_composer::splice_inst_chunks;
use crate::validate_wgsl;

/// Maximum simultaneous regions per frame. Same shape as
/// [`crate::user_shader_emit_pass::MAX_INSTANCE_REGIONS`] — keeps the
/// uniform buffer bounded.
pub const MAX_TILE_CULL_REGIONS: u32 = 1024;

/// Stride between consecutive [`TileCullRegionUniform`]s in the upload
/// buffer. wgpu requires uniform dynamic-offset alignment of 256 B.
/// Same value as [`crate::user_shader_emit_pass::EMIT_DISPATCH_UNIFORM_STRIDE`].
pub const TILE_CULL_REGION_UNIFORM_STRIDE: u64 = 256;

/// Per-region uniform — must match `TileCullRegionUniform` in
/// `user_shader_tile_cull.wgsl`. 32 bytes.
///
/// `scratch_offset` is the entry index where this region's slice begins
/// in `tile_cull_scratch`; the engine layer computes it as the running
/// sum of prior regions' `instance_block_size`. Each region's slice
/// covers exactly `instance_block_size` entries (capacity, not the
/// post-emit live count — dead slots are kept inline so Session 3
/// indexes by reservation).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TileCullRegionUniform {
    pub region_index: u32,
    pub asset_id: u32,
    pub material_id: u32,
    pub shader_id: u32,
    pub instance_block_offset: u32,
    pub instance_block_size: u32,
    pub instance_stride_u32: u32,
    pub scratch_offset: u32,
}

const _: () = assert!(std::mem::size_of::<TileCullRegionUniform>() == 32);

/// One entry per reserved instance slot. Wire format must match
/// `InstanceTileCullEntry` in `user_shader_tile_cull.wgsl`. 48 bytes.
///
/// `live = 1` means the emit pass populated this slot and `aabb_min` /
/// `aabb_max` are the world-space AABB returned by the user shader's
/// `inst_aabb` hook. `live = 0` means the slot was reserved but
/// unfilled — Session 3 skips it with one branch.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceTileCullEntry {
    pub aabb_min: [f32; 3],
    pub asset_id: u32,
    pub aabb_max: [f32; 3],
    pub instance_state_offset: u32,
    pub material_id: u32,
    pub live: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

const _: () = assert!(std::mem::size_of::<InstanceTileCullEntry>() == 48);

/// Workgroup count for one region's tile-cull dispatch. Workgroup size
/// is 64; one thread per reserved instance slot. Returns at least 1
/// even for empty regions so the validator doesn't reject the dispatch.
pub fn workgroups_for_region(instance_block_size: u32) -> u32 {
    instance_block_size.div_ceil(64).max(1)
}

/// GPU pipeline owner for the tile-cull AABB compute shader. Mirrors
/// the construction shape of [`crate::user_shader_emit_pass::EmitPass`].
pub struct TileCullPass {
    pub group0_layout: wgpu::BindGroupLayout,
    pub group1_layout: wgpu::BindGroupLayout,
    pub pipeline_layout: wgpu::PipelineLayout,
    pub pipeline: wgpu::ComputePipeline,
    /// Per-region uniform array — one slot per dispatched region,
    /// uploaded via `queue.write_buffer` at frame start. Bound at
    /// group(1) with a 256 B dynamic offset.
    pub regions_buffer: wgpu::Buffer,
    /// Hash of the user-shader source mix the pipeline was last built
    /// against. Comparing against the registry's `source_hash` decides
    /// whether the pipeline needs rebuilding. Same protocol as
    /// `OctreeMarchPass` / `EmitPass`.
    pub source_hash: u64,
}

impl TileCullPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let group0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("user_shader_tile_cull group0"),
            entries: &[
                ro_storage(0), // instance_pool
                ro_storage(1), // instance_alloc
                rw_storage(2), // tile_cull_scratch
            ],
        });
        let group1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("user_shader_tile_cull group1"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: std::num::NonZeroU64::new(
                        std::mem::size_of::<TileCullRegionUniform>() as u64,
                    ),
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("user_shader_tile_cull pipeline layout"),
            bind_group_layouts: &[Some(&group0_layout), Some(&group1_layout)],
            immediate_size: 0,
        });
        let pipeline = build_pipeline(device, &pipeline_layout, "", "");

        let regions_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("user_shader_tile_cull regions"),
            size: TILE_CULL_REGION_UNIFORM_STRIDE * MAX_TILE_CULL_REGIONS as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            group0_layout,
            group1_layout,
            pipeline_layout,
            pipeline,
            regions_buffer,
            source_hash: 0,
        }
    }

    /// Re-build the compute pipeline against fresh user-shader chunks.
    /// Returns `true` if rebuilt, `false` if the hash matched and the
    /// existing pipeline was kept. Empty chunks restore the default
    /// identity stubs.
    pub fn reload_user_shaders(
        &mut self,
        device: &wgpu::Device,
        inst_to_local_chunk: &str,
        inst_aabb_chunk: &str,
        source_hash: u64,
    ) -> bool {
        if source_hash == self.source_hash {
            return false;
        }
        self.pipeline = build_pipeline(
            device, &self.pipeline_layout, inst_to_local_chunk, inst_aabb_chunk,
        );
        self.source_hash = source_hash;
        true
    }

    pub fn source_hash(&self) -> u64 {
        self.source_hash
    }
}

fn ro_storage(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn rw_storage(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn build_pipeline(
    device: &wgpu::Device,
    pipeline_layout: &wgpu::PipelineLayout,
    inst_to_local_chunk: &str,
    inst_aabb_chunk: &str,
) -> wgpu::ComputePipeline {
    let template = include_str!("shaders/user_shader_tile_cull.wgsl");
    let source = splice_inst_chunks(template, inst_to_local_chunk, inst_aabb_chunk);
    validate_wgsl(&source, "user_shader_tile_cull");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("user_shader_tile_cull"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("user_shader_tile_cull"),
        layout: Some(pipeline_layout),
        module: &module,
        entry_point: Some("tile_cull_main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_wgsl_valid(source: &str, label: &str) {
        let module = naga::front::wgsl::parse_str(source).unwrap_or_else(|e| {
            panic!("[{label}] parse error:\n{}", e.emit_to_string(source))
        });
        let mut v = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        v.validate(&module)
            .unwrap_or_else(|e| panic!("[{label}] validation error: {e:?}"));
    }

    #[test]
    fn region_uniform_size_is_32() {
        assert_eq!(std::mem::size_of::<TileCullRegionUniform>(), 32);
    }

    #[test]
    fn entry_size_is_48() {
        assert_eq!(std::mem::size_of::<InstanceTileCullEntry>(), 48);
    }

    #[test]
    fn workgroups_for_region_at_least_one() {
        assert_eq!(workgroups_for_region(0), 1);
        assert_eq!(workgroups_for_region(1), 1);
        assert_eq!(workgroups_for_region(64), 1);
        assert_eq!(workgroups_for_region(65), 2);
        assert_eq!(workgroups_for_region(4096), 64);
    }

    #[test]
    fn template_validates_with_empty_chunks() {
        let template = include_str!("shaders/user_shader_tile_cull.wgsl");
        let source = splice_inst_chunks(template, "", "");
        assert_wgsl_valid(&source, "user_shader_tile_cull");
        assert!(source.contains("tile_cull_main"));
        // Identity stubs are kept verbatim when chunks are empty.
        assert!(source.contains("inst_world_to_local"));
    }

    #[test]
    fn template_validates_with_grass_shader_chunks() {
        // Build a realistic chunk pair the way the composer would, then
        // confirm the spliced source validates with naga.
        use crate::shader_composer::{compose, scan_dir};
        let dir = std::env::temp_dir().join(format!(
            "rkpatch_tile_cull_validate_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("grass.wgsl");
        std::fs::write(
            &path,
            r#"
// @instance_proto Blade
struct Blade {
    pos: vec3<f32>,
    yaw: f32,
    sway_phase: f32,
    height_scale: f32,
    tint: u32,
}
fn user_grass_proto(uvw: vec3<f32>) -> VoxelEmit { var v: VoxelEmit; return v; }
fn user_grass_emit(host_pos: vec3<f32>, host: HostSample, ctx: UserCtx) {
    var b: Blade;
    b.pos = host_pos;
    b.yaw = 0.0;
    b.sway_phase = ctx.time;
    b.height_scale = 1.0;
    b.tint = 0u;
    emit_instance(b);
}
fn user_grass_inst_to_local(world_pos: vec3<f32>, inst: Blade) -> vec3<f32> {
    let half = 0.5 * inst.height_scale;
    let inv = 1.0 / max(half * 2.0, 1e-10);
    return (world_pos - inst.pos) * inv + vec3<f32>(0.5);
}
fn user_grass_inst_aabb(inst: Blade) -> Aabb {
    let half = 0.5 * inst.height_scale * 1.7320508;
    var a: Aabb;
    a.min = inst.pos - vec3<f32>(half);
    a.max = inst.pos + vec3<f32>(half);
    return a;
}
"#,
        )
        .unwrap();
        let registry = scan_dir(&dir).unwrap();
        let chunks = compose(&registry);
        let template = include_str!("shaders/user_shader_tile_cull.wgsl");
        let source = splice_inst_chunks(template, &chunks.inst_to_local, &chunks.inst_aabb);
        assert_wgsl_valid(&source, "tile_cull_with_grass");
        // The composer renames `user_grass_inst_aabb` to
        // `rkp_user_<id>_inst_aabb`; confirm the per-shader switch arm
        // landed.
        assert!(source.contains("rkp_user_1_inst_aabb_at"));
        assert!(source.contains("tile_cull_main"));
    }

    #[test]
    fn entry_field_offsets_match_wgsl() {
        // Sanity-check the Rust struct layout matches the WGSL struct
        // (vec3<f32> packs a trailing u32 into the same 16-byte slot).
        // Use bytemuck to cast and verify offsets.
        let e = InstanceTileCullEntry {
            aabb_min: [1.0, 2.0, 3.0],
            asset_id: 0xAABB,
            aabb_max: [4.0, 5.0, 6.0],
            instance_state_offset: 0xCCDD,
            material_id: 0xEEFF,
            live: 1,
            _pad0: 0,
            _pad1: 0,
        };
        let bytes = bytemuck::bytes_of(&e);
        assert_eq!(bytes.len(), 48);
        // aabb_min at offset 0
        assert_eq!(&bytes[0..4], 1.0_f32.to_le_bytes());
        assert_eq!(&bytes[4..8], 2.0_f32.to_le_bytes());
        assert_eq!(&bytes[8..12], 3.0_f32.to_le_bytes());
        // asset_id at offset 12
        assert_eq!(&bytes[12..16], 0xAABB_u32.to_le_bytes());
        // aabb_max at offset 16
        assert_eq!(&bytes[16..20], 4.0_f32.to_le_bytes());
        // instance_state_offset at offset 28
        assert_eq!(&bytes[28..32], 0xCCDD_u32.to_le_bytes());
        // material_id at offset 32
        assert_eq!(&bytes[32..36], 0xEEFF_u32.to_le_bytes());
        // live at offset 36
        assert_eq!(&bytes[36..40], 1_u32.to_le_bytes());
    }
}
