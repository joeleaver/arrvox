//! `PrototypeBakePass` — the GPU runtime. Owns one compute pipeline
//! (`bake_main` from `user_shader_proto.wgsl`), the global brick +
//! leaf-attr cursor buffers + overflow counter, and the per-shader
//! `compose_proto_source` template splice.
//!
//! Also: `PrototypeUniform` (the 32 B per-bake uniform layout matching
//! WGSL) + `OCTREE_EMPTY` / `INTERNAL_ATTR_NONE` octree-node sentinels.

use super::cache::PrototypeCache;
use super::types::PrototypeEntry;


/// GPU prototype uniform — must match `PrototypeUniform` in
/// `user_shader_proto.wgsl`. 32 bytes.
///
/// Brick + leaf-attr ranges are GLOBAL across prototypes — no per-bake
/// offset; the bake atomic-bumps a single cursor pair from
/// [`PrototypeBakePass`]. The two `*_capacity` fields are ABSOLUTE
/// upper bounds the bake uses to gate overflow: bake stops emitting
/// when `brick_id >= brick_capacity` or `leaf_attr_id >= leaf_attr_capacity`.
/// Cursors start at `pool_brick_base` / `pool_leaf_attr_base` (proto
/// reservation start in the host pool), so the capacities below are
/// `pool_*_base + reservation_size` — i.e. the slot just past the
/// proto reservation.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PrototypeUniform {
    pub shader_id: u32,
    pub max_depth: u32,
    pub octree_leaf_offset: u32,
    pub brick_capacity: u32,
    pub leaf_attr_capacity: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

const _: () = assert!(std::mem::size_of::<PrototypeUniform>() == 32);

impl PrototypeUniform {
    pub fn from_entry(entry: &PrototypeEntry, cache: &PrototypeCache) -> Self {
        // Absolute upper bounds in the host pool: cursor starts at
        // pool_*_base, so the cap is base + reservation_size.
        let brick_cap_abs = cache
            .pool_brick_base
            .saturating_add(cache.pool_brick_capacity);
        let leaf_attr_cap_abs = cache
            .pool_leaf_attr_base
            .saturating_add(cache.pool_leaf_attr_capacity);
        Self {
            shader_id: entry.shader_id,
            max_depth: entry.max_depth,
            octree_leaf_offset: entry.octree_leaf_offset(cache.pool_octree_base),
            brick_capacity: brick_cap_abs,
            leaf_attr_capacity: leaf_attr_cap_abs,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }
}

/// GPU pipeline owner for the prototype bake compute shader. Mirrors
/// the construction shape of [`crate::user_shader_pass::UserShaderPass`]
/// but is much smaller — prototype bakes don't need the BFS classify
/// step, the active queue, or per-region atomic counters.
///
/// Brick + leaf-attr cursors are persistent: the bake atomic-bumps
/// them once per emitted slot, and the engine only zeros them on a
/// cache full-reset (rare). Different prototypes' baked slots
/// interleave in the global pools.
pub struct PrototypeBakePass {
    pub group0_layout: wgpu::BindGroupLayout,
    pub group1_layout: wgpu::BindGroupLayout,
    pub pipeline_layout: wgpu::PipelineLayout,
    pub bake_pipeline: wgpu::ComputePipeline,
    /// Single-pair `GlobalCursors { brick: atomic<u32>, leaf_attr: atomic<u32> }`
    /// at group(0) binding(3). 8 bytes total.
    pub cursors_buffer: wgpu::Buffer,
    /// Overflow counters — same layout the per-region pass uses
    /// (only `OVERFLOW_BRICK` and `OVERFLOW_LEAF_ATTR` are written by
    /// this shader). The proto pass owns its own buffer rather than
    /// sharing with the per-region one because resets and readbacks
    /// are scheduled independently.
    pub overflow_buffer: wgpu::Buffer,
    /// Uniform buffer for `PrototypeUniform`. Bound at group(1).
    pub proto_uniform_buffer: wgpu::Buffer,
    /// Hash of the user-shader source mix the bake pipeline was
    /// last built against. Comparing against the registry's
    /// `source_hash` decides whether the pipeline needs rebuilding.
    pub source_hash: u64,
}

impl PrototypeBakePass {
    pub fn new(device: &wgpu::Device) -> Self {
        let group0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("user_shader_proto group0"),
            entries: &[
                rw_storage(0), // octree_nodes
                rw_storage(1), // brick_pool
                rw_storage(2), // leaf_attr_pool
                rw_storage(3), // cursors (GlobalCursors struct, 8 B)
                rw_storage(4), // overflow
            ],
        });
        let group1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("user_shader_proto group1"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(
                        std::mem::size_of::<PrototypeUniform>() as u64,
                    ),
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("user_shader_proto pipeline layout"),
            bind_group_layouts: &[Some(&group0_layout), Some(&group1_layout)],
            immediate_size: 0,
        });
        let bake_pipeline = build_proto_pipeline(device, &pipeline_layout, "");

        // GlobalCursors = brick: atomic<u32> + leaf_attr: atomic<u32> = 8 B.
        let cursors_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("user_shader_proto cursors"),
            size: 8,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Overflow buffer — must be at least as large as the highest
        // index the WGSL writes (OVERFLOW_LEAF_ATTR = 2). Match the
        // per-region pass's OVERFLOW_COUNTER_COUNT (=12 at MAX_DEPTH=8)
        // so a future shared-buffer setup is trivial.
        let overflow_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("user_shader_proto overflow"),
            size: 12 * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let proto_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("user_shader_proto uniform"),
            size: std::mem::size_of::<PrototypeUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            group0_layout,
            group1_layout,
            pipeline_layout,
            bake_pipeline,
            cursors_buffer,
            overflow_buffer,
            proto_uniform_buffer,
            source_hash: 0,
        }
    }

    /// Initialize the GPU brick + leaf-attr atomic cursors to the proto
    /// pool's base offsets in the host scene buffers. The bake compute
    /// shader bumps these and uses the bumped values directly as
    /// brick_id / leaf_attr_id, so a cursor starting at
    /// `proto_brick_base` ⇒ first baked brick lands at host-pool
    /// `proto_brick_base + 0`.
    ///
    /// Call this whenever the proto pool bases change (e.g. on first
    /// frame, or when CPU geometry growth shifts the proto base). Pair
    /// with [`PrototypeCache::flush`] / [`PrototypeCache::dirty_all`] —
    /// otherwise live prototypes' baked slots become unreferenceable
    /// from new bakes.
    pub fn reset_cursors(&self, queue: &wgpu::Queue, brick_base: u32, leaf_attr_base: u32) {
        let mut bytes = [0u8; 8];
        bytes[0..4].copy_from_slice(&brick_base.to_le_bytes());
        bytes[4..8].copy_from_slice(&leaf_attr_base.to_le_bytes());
        queue.write_buffer(&self.cursors_buffer, 0, &bytes);
    }

    /// Re-build the compute pipeline against a fresh user-shader chunk.
    /// Returns `true` if rebuilt, `false` if the hash matched and the
    /// existing pipeline was kept. Mirrors
    /// `UserShaderPass::reload_user_shaders`.
    pub fn reload_user_shaders(
        &mut self,
        device: &wgpu::Device,
        proto_chunk: &str,
        source_hash: u64,
    ) -> bool {
        if source_hash == self.source_hash {
            return false;
        }
        self.bake_pipeline = build_proto_pipeline(device, &self.pipeline_layout, proto_chunk);
        self.source_hash = source_hash;
        true
    }

    pub fn source_hash(&self) -> u64 {
        self.source_hash
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

fn build_proto_pipeline(
    device: &wgpu::Device,
    pipeline_layout: &wgpu::PipelineLayout,
    proto_chunk: &str,
) -> wgpu::ComputePipeline {
    let source = compose_proto_source(proto_chunk);
    crate::validate_wgsl(&source, "user_shader_proto");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("user_shader_proto"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("user_shader_proto bake"),
        layout: Some(pipeline_layout),
        module: &module,
        entry_point: Some("proto_bake_main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// Splice the composer's `proto` chunk into the bake shader source.
/// Empty chunk returns the in-tree default (which has its own
/// identity stub between the markers); non-empty chunk REPLACES the
/// stub and the markers themselves with `proto_chunk`. Mirrors
/// `compose_geom_source` in `user_shader_pass.rs`.
pub fn compose_proto_source(proto_chunk: &str) -> String {
    let template = include_str!("../shaders/user_shader_proto.wgsl");
    if proto_chunk.is_empty() {
        return template.to_string();
    }
    // Marker strings constructed via concat so the `find` below isn't
    // fooled by literal occurrences in this docstring or elsewhere in
    // the Rust source.
    let begin_marker = concat!("USER_PROTO_DISPATCH", "_BEGIN");
    let end_marker = concat!("USER_PROTO_DISPATCH", "_END");
    let begin = template
        .find(begin_marker)
        .expect("user_shader_proto.wgsl missing USER_PROTO_DISPATCH_BEGIN marker");
    let end_after = template[begin..]
        .find(end_marker)
        .map(|off| begin + off + end_marker.len())
        .expect("user_shader_proto.wgsl missing USER_PROTO_DISPATCH_END marker");
    let mut out = String::with_capacity(template.len() + proto_chunk.len());
    out.push_str(&template[..begin]);
    out.push_str(proto_chunk);
    out.push_str(&template[end_after..]);
    out
}
