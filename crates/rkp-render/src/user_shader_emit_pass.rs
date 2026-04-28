//! Option B per-region instance scatter pipeline.
//!
//! Owns the GPU pipeline + cache that runs each instance shader's
//! `emit` hook over a 3D sample grid at brick-parent granularity,
//! atomic-appending placed instances into a per-region slice of a
//! global `instance_pool` buffer. The march (Stage 5+) reads back from
//! this same buffer when traversing instance shaders.
//!
//! ## Cache
//!
//! [`InstanceRegionCache`] is keyed by `(host_object_id, material_id,
//! tile_index)` — same shape as the per-cell
//! [`crate::user_shader_pass::UserShaderObjectCache`]. Two hashes:
//!
//! * `topology_hash` — host geometry + region thickness + tile + max
//!   depth + cell_size + AABB. Stable across frames as long as the
//!   host's painted surface in this tile hasn't moved.
//! * `fill_hash` — folds shader source + per-material params + paint
//!   epoch + `time` (iff the shader is `@animated`).
//!
//! The per-region instance buffer extent only re-bakes when
//! `fill_hash` changes; otherwise the cached emission survives across
//! frames.
//!
//! ## Pool
//!
//! `instance_pool: array<u32>` is the single global buffer holding all
//! regions' instance bytes. Each region's extent is bucket-allocated
//! in u32 units; the WGSL `rkp_user_<id>_emit_instance` body computes
//! `base = region.instance_block_offset + slot * stride_u32` and
//! writes the struct's fields via `bitcast<u32>`.
//!
//! Stride varies per shader — different instance struct sizes — but
//! the pool storage is a flat u32 array, so the allocator works in
//! u32 units uniformly.

use std::collections::HashMap;

use crate::shader_composer::UserShaderInfo;
use crate::user_shader_pass::BucketPoolAllocator;

/// Maximum simultaneous instance regions per frame. Same shape as
/// [`crate::user_shader_pass::MAX_REGIONS`] — bound for storage
/// binding sizing.
pub const MAX_INSTANCE_REGIONS: u32 = 1024;

/// Default per-region instance cap when a shader doesn't override.
/// At stride 8 (32-byte struct) → 32 768 u32s reserved per region.
pub const DEFAULT_MAX_INSTANCES_PER_REGION: u32 = 4096;

/// Global instance pool capacity in u32s. 64 M u32s × 4 = 256 MB. At
/// the default 4096-instance/region cap and stride 8, fits ~2000
/// fully-packed regions; with overflow handling at the high-water
/// mark, more partial regions fit before pressure shows.
pub const MAX_GLOBAL_INSTANCE_U32S: u32 = 64_000_000;

/// Bucket bounds for the instance pool allocator, in u32 units.
/// MIN of 64 = 8 instances at stride 8 (the smallest allocation we
/// hand out — even a region that emits nothing reserves the bucket
/// minimum). MAX caps how much one region can grab.
pub const INSTANCE_BUCKET_MIN: u32 = 64;
pub const INSTANCE_BUCKET_MAX: u32 = 65536;

/// Overflow buffer — must be large enough for the highest
/// `OVERFLOW_*` slot the WGSL emits to. Today only
/// `OVERFLOW_INSTANCE = 0` is written; sized at 4 u32s for headroom.
const OVERFLOW_COUNTER_COUNT: u64 = 4;

/// Sentinel for "this region has no host" — same value the geom
/// pipeline uses, kept here so the engine layer can pick whichever
/// import path is convenient.
pub const HOST_NO_HOST_SENTINEL: u32 = 0xFFFF_FFFFu32;

/// Sentinel `tile_index` value for non-tiled shaders. Same shape as
/// [`crate::user_shader_pass::NO_TILE`].
pub const NO_TILE: [i32; 3] = [i32::MIN, i32::MIN, i32::MIN];

/// One per-region request from sim to render. Mirrors
/// [`crate::user_shader_pass::ShaderRegionRequest`] but trimmed to the
/// fields the emit pipeline actually reads (no fill-task counts, no
/// brick estimates).
#[derive(Debug, Clone)]
pub struct InstanceRegionRequest {
    pub host_object_id: u32,
    pub material_id: u32,
    pub shader_name: String,
    pub params: Vec<f32>,
    /// Cube AABB the dispatch sweeps at brick-parent granularity.
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    pub cell_size: f32,
    /// Folded into `fill_hash`; sim bumps when paint or shader
    /// inputs change.
    pub input_hash: u64,
    pub animated: bool,
    pub region_thickness: f32,
    pub tile_index: [i32; 3],
    /// Stride between instance records in u32s, derived from the
    /// shader's parsed `InstanceLayout` total_size. Cache keying
    /// folds this in so a stride change forces re-allocation.
    pub stride_u32: u32,
    /// Max instances this region may emit. Defaults to
    /// [`DEFAULT_MAX_INSTANCES_PER_REGION`]; user override comes
    /// from a future `@max_instances_per_region` directive (TBD).
    pub max_instances: u32,
    /// Host octree info — same fields the geom pipeline carries.
    pub host_octree_root: u32,
    pub host_octree_depth: u32,
    pub host_octree_extent: f32,
    pub host_grid_origin: [f32; 3],
    pub host_inverse_world: [[f32; 4]; 4],
}

/// Persistent cache entry for one instance region.
#[derive(Debug, Clone)]
struct CacheEntry {
    /// Extent in the global `instance_pool` buffer, in u32 units.
    instance_extent: (u32, u32),
    /// Cached `max_instances` (= extent_size / stride_u32).
    max_instances: u32,
    stride_u32: u32,
    topology_hash: u64,
    fill_hash: u64,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
    cell_size: f32,
    region_thickness: f32,
    object_id: u32,
    touched_this_frame: bool,
}

const INSTANCE_OBJECT_ID_BASE: u32 = 0xF800_0000;

/// Cache + variable-size pool allocator for instance-shader regions.
pub struct InstanceRegionCache {
    entries: HashMap<(u32, u32, [i32; 3]), CacheEntry>,
    instance_alloc: BucketPoolAllocator,
    pool_instance_base: u32,
    pool_instance_capacity: u32,
    next_object_id: u32,
    last_seen_geometry_epoch: u64,
}

impl InstanceRegionCache {
    pub fn new() -> Self {
        Self::with_capacity(MAX_GLOBAL_INSTANCE_U32S)
    }

    pub fn with_capacity(instance_capacity_u32: u32) -> Self {
        Self {
            entries: HashMap::new(),
            instance_alloc: BucketPoolAllocator::new(
                instance_capacity_u32,
                INSTANCE_BUCKET_MIN,
                INSTANCE_BUCKET_MAX,
            ),
            pool_instance_base: 0,
            pool_instance_capacity: instance_capacity_u32,
            next_object_id: INSTANCE_OBJECT_ID_BASE,
            last_seen_geometry_epoch: 0,
        }
    }

    /// Configure the GPU offset where the instance sub-pool begins.
    /// If the base changes, the cache is flushed (pool layout shifted
    /// under us, every cached extent's absolute offset would be wrong).
    pub fn set_pool_base(&mut self, pool_instance_base: u32) {
        if self.pool_instance_base == pool_instance_base {
            return;
        }
        self.flush();
        self.pool_instance_base = pool_instance_base;
    }

    pub fn pool_instance_base(&self) -> u32 { self.pool_instance_base }
    pub fn pool_instance_capacity(&self) -> u32 { self.pool_instance_capacity }

    pub fn flush(&mut self) {
        self.entries.clear();
        self.instance_alloc = BucketPoolAllocator::new(
            self.pool_instance_capacity,
            INSTANCE_BUCKET_MIN,
            INSTANCE_BUCKET_MAX,
        );
    }

    /// Bump the geometry epoch — flushes the cache when host geometry
    /// changes (since topology depends on the host octree). Same as
    /// `UserShaderObjectCache::reconcile_epoch`.
    pub fn reconcile_epoch(&mut self, geometry_epoch: u64) -> bool {
        if geometry_epoch <= self.last_seen_geometry_epoch {
            return false;
        }
        self.last_seen_geometry_epoch = geometry_epoch;
        if !self.entries.is_empty() {
            self.flush();
            return true;
        }
        false
    }

    pub fn begin_frame(&mut self) {
        for entry in self.entries.values_mut() {
            entry.touched_this_frame = false;
        }
    }

    /// Look up or allocate a cache slot. Returns `Some(slot)` on
    /// success, `None` on pool exhaustion.
    pub fn lookup_or_allocate(
        &mut self,
        request: &InstanceRegionRequest,
        topology_hash: u64,
        fill_hash: u64,
    ) -> Option<CachedSlot> {
        let key = (request.host_object_id, request.material_id, request.tile_index);

        let needed = request
            .stride_u32
            .saturating_mul(request.max_instances)
            .max(INSTANCE_BUCKET_MIN);

        if let Some(entry) = self.entries.get_mut(&key) {
            let extent_fits = entry.instance_extent.1 >= needed;
            let stride_match = entry.stride_u32 == request.stride_u32;
            if extent_fits && stride_match {
                let topology_dirty = entry.topology_hash != topology_hash;
                let fill_dirty = topology_dirty || entry.fill_hash != fill_hash;
                entry.aabb_min = request.aabb_min;
                entry.aabb_max = request.aabb_max;
                entry.cell_size = request.cell_size;
                entry.region_thickness = request.region_thickness;
                entry.touched_this_frame = true;
                if topology_dirty {
                    entry.topology_hash = topology_hash;
                }
                if fill_dirty {
                    entry.fill_hash = fill_hash;
                }
                return Some(slot_from_entry(
                    entry,
                    self.pool_instance_base,
                    topology_dirty,
                    fill_dirty,
                ));
            }
            // Stale extent (stride changed or extent too small) —
            // free and fall through to fresh alloc.
            self.instance_alloc
                .free(entry.instance_extent.0, entry.instance_extent.1);
            self.entries.remove(&key);
        }

        let instance_extent = self.instance_alloc.alloc(needed)?;
        // Re-derive max_instances from the bucket-rounded extent so
        // the GPU can use the full granted space.
        let max_instances = instance_extent.1 / request.stride_u32.max(1);

        let object_id = self.next_object_id;
        self.next_object_id = self.next_object_id.wrapping_add(1);

        let entry = CacheEntry {
            instance_extent,
            max_instances,
            stride_u32: request.stride_u32,
            topology_hash,
            fill_hash,
            aabb_min: request.aabb_min,
            aabb_max: request.aabb_max,
            cell_size: request.cell_size,
            region_thickness: request.region_thickness,
            object_id,
            touched_this_frame: true,
        };
        let slot = slot_from_entry(&entry, self.pool_instance_base, true, true);
        self.entries.insert(key, entry);
        Some(slot)
    }

    pub fn evict_untouched(&mut self) {
        let to_remove: Vec<(u32, u32, [i32; 3])> = self
            .entries
            .iter()
            .filter(|(_, e)| !e.touched_this_frame)
            .map(|(k, _)| *k)
            .collect();
        for k in to_remove {
            if let Some(entry) = self.entries.remove(&k) {
                self.instance_alloc
                    .free(entry.instance_extent.0, entry.instance_extent.1);
            }
        }
    }

    pub fn entry_count(&self) -> usize { self.entries.len() }
    pub fn instance_high_water(&self) -> u32 { self.instance_alloc.high_water() }

    /// Iterate cache keys (host_object_id, material_id, tile_index)
    /// for entries that were touched this frame. Stage 4's
    /// [`crate::instance_tile_index::TileIndex`] consumes this to build
    /// the per-frame tile→region lookup the march reads from.
    pub fn touched_keys(&self) -> impl Iterator<Item = (u32, u32, [i32; 3])> + '_ {
        self.entries
            .iter()
            .filter(|(_, e)| e.touched_this_frame)
            .map(|(k, _)| *k)
    }
}

impl Default for InstanceRegionCache {
    fn default() -> Self {
        Self::new()
    }
}

fn slot_from_entry(
    entry: &CacheEntry,
    pool_instance_base: u32,
    topology_dirty: bool,
    fill_dirty: bool,
) -> CachedSlot {
    CachedSlot {
        region_index: 0,
        instance_block_offset: pool_instance_base + entry.instance_extent.0,
        instance_block_size: entry.max_instances,
        instance_extent_u32: entry.instance_extent.1,
        stride_u32: entry.stride_u32,
        object_id: entry.object_id,
        topology_dirty,
        fill_dirty,
    }
}

/// Slot descriptor returned from `lookup_or_allocate`.
#[derive(Debug, Clone, Copy)]
pub struct CachedSlot {
    /// Region index in this frame's dispatch arrays. Set by the
    /// caller after gathering all dirty slots — `lookup_or_allocate`
    /// returns 0 here.
    pub region_index: u32,
    /// Absolute u32 offset where this region's instance records start.
    pub instance_block_offset: u32,
    /// Number of instance slots reserved for this region (= u32
    /// extent / stride). Lookup result of `extent_size /
    /// stride_u32`, NOT what the caller asked for — bucket rounding
    /// gives the GPU the full bucket extent.
    pub instance_block_size: u32,
    /// Total u32 count reserved (instances × stride). For diagnostics.
    pub instance_extent_u32: u32,
    /// Stride between instance records, in u32s.
    pub stride_u32: u32,
    pub object_id: u32,
    pub topology_dirty: bool,
    pub fill_dirty: bool,
}

/// Per-region uniform — must match `EmitRegionUniform` in
/// `user_shader_emit.wgsl`. 192 bytes.
///
/// Field offsets follow WGSL std430-ish rules: `vec3<f32>` has size 12
/// but alignment 16, so any field after one falls at the next
/// 16-aligned offset unless a 4-byte field packs into the trailing
/// 4 bytes (e.g. `shader_id` packs after `aabb_max`).
/// `_pad_host` (3 u32s) explicitly pads from offset 68 to offset 80
/// where `host_grid_origin` (vec3<f32>) starts — WGSL would insert
/// this padding implicitly; Rust will not, so we spell it out.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EmitRegionUniform {
    pub aabb_min: [f32; 3],                // offset   0
    pub cell_size: f32,                     // offset  12
    pub aabb_max: [f32; 3],                 // offset  16
    pub shader_id: u32,                     // offset  28
    pub time: f32,                          // offset  32
    pub material_id: u32,                   // offset  36
    pub region_thickness: f32,              // offset  40
    pub instance_block_offset: u32,         // offset  44
    pub instance_block_size: u32,           // offset  48
    pub instance_stride_u32: u32,           // offset  52
    pub host_octree_root: u32,              // offset  56
    pub host_octree_depth: u32,             // offset  60
    pub host_octree_extent: f32,            // offset  64
    /// 12 bytes (3 × u32) of padding so `host_grid_origin` lands at
    /// offset 80. WGSL inserts this implicitly; we spell it out.
    pub _pad_host: [u32; 3],                // offset  68 → 80
    pub host_grid_origin: [f32; 3],         // offset  80
    pub _pad_grid: f32,                     // offset  92
    pub params: [[f32; 4]; 2],              // offset  96
    pub host_inverse_world: [[f32; 4]; 4],  // offset 128
}

const _: () = assert!(std::mem::size_of::<EmitRegionUniform>() == 192);

/// Per-dispatch state — uploaded with a dynamic-offset uniform binding
/// so a single uniform buffer can hold many regions' values laid out
/// at `EMIT_DISPATCH_UNIFORM_STRIDE` apart.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EmitDispatchUniform {
    pub region_index: u32,
    pub samples_per_axis: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

const _: () = assert!(std::mem::size_of::<EmitDispatchUniform>() == 16);

/// Stride between consecutive `EmitDispatchUniform`s in the upload
/// buffer. WGPU requires uniform dynamic-offset alignment of 256 B.
pub const EMIT_DISPATCH_UNIFORM_STRIDE: u64 = 256;

/// Build the per-region emit uniform from a request + cached slot.
pub fn build_emit_region_uniform(
    request: &InstanceRegionRequest,
    slot: &CachedSlot,
    shader_id: u32,
    time_seconds: f32,
) -> EmitRegionUniform {
    let mut params = [[0.0f32; 4]; 2];
    for (i, &v) in request.params.iter().take(8).enumerate() {
        params[i / 4][i % 4] = v;
    }
    EmitRegionUniform {
        aabb_min: request.aabb_min,
        cell_size: request.cell_size,
        aabb_max: request.aabb_max,
        shader_id,
        time: time_seconds,
        material_id: request.material_id,
        region_thickness: request.region_thickness,
        instance_block_offset: slot.instance_block_offset,
        instance_block_size: slot.instance_block_size,
        instance_stride_u32: slot.stride_u32,
        host_octree_root: request.host_octree_root,
        host_octree_depth: request.host_octree_depth,
        host_octree_extent: request.host_octree_extent,
        _pad_host: [0; 3],
        host_grid_origin: request.host_grid_origin,
        _pad_grid: 0.0,
        params,
        host_inverse_world: request.host_inverse_world,
    }
}

/// Compute samples_per_axis = ceil(extent / (cell_size * 4)) for the
/// emit dispatch's grid. The dispatch then issues
/// `ceil(samples_per_axis / 4)³` workgroups.
pub fn samples_per_axis(request: &InstanceRegionRequest) -> u32 {
    let extent = (request.aabb_max[0] - request.aabb_min[0]).max(1e-6);
    let bp_cell = (request.cell_size * 4.0).max(1e-6);
    ((extent / bp_cell).ceil() as u32).max(1)
}

/// Resolve a `MaterialDef.shader` name to a shader id and parsed
/// `InstanceLayout.total_size`-derived stride. Returns `None` for
/// shader names that aren't in the registry or that aren't instance
/// shaders.
pub fn resolve_instance_shader<'a>(
    infos: &'a [UserShaderInfo],
    name: &str,
) -> Option<&'a UserShaderInfo> {
    infos
        .iter()
        .find(|info| info.is_instance_pipeline && info.name == name)
}

/// Splice the composer's `emit` chunk into the emit shader source.
/// Empty chunk leaves the in-tree default identity stub. Mirrors
/// `compose_proto_source`.
pub fn compose_emit_source(emit_chunk: &str) -> String {
    let template = include_str!("shaders/user_shader_emit.wgsl");
    if emit_chunk.is_empty() {
        return template.to_string();
    }
    let begin_marker = concat!("USER_EMIT_DISPATCH", "_BEGIN");
    let end_marker = concat!("USER_EMIT_DISPATCH", "_END");
    let begin = template
        .find(begin_marker)
        .expect("user_shader_emit.wgsl missing USER_EMIT_DISPATCH_BEGIN marker");
    let end_after = template[begin..]
        .find(end_marker)
        .map(|off| begin + off + end_marker.len())
        .expect("user_shader_emit.wgsl missing USER_EMIT_DISPATCH_END marker");
    let mut out = String::with_capacity(template.len() + emit_chunk.len());
    out.push_str(&template[..begin]);
    out.push_str(emit_chunk);
    out.push_str(&template[end_after..]);
    out
}

/// GPU pipeline owner for the emit compute shader. Mirrors the shape
/// of [`crate::user_shader_proto_pass::PrototypeBakePass`].
pub struct EmitPass {
    pub group0_layout: wgpu::BindGroupLayout,
    pub group1_layout: wgpu::BindGroupLayout,
    pub group2_layout: wgpu::BindGroupLayout,
    pub pipeline_layout: wgpu::PipelineLayout,
    pub emit_pipeline: wgpu::ComputePipeline,
    /// Per-region atomic counter — `array<atomic<u32>, MAX_INSTANCE_REGIONS>`.
    /// Reset to 0 per region at frame start (for fill-dirty regions).
    pub instance_alloc_buffer: wgpu::Buffer,
    pub overflow_buffer: wgpu::Buffer,
    pub regions_buffer: wgpu::Buffer,
    pub dispatch_uniforms_buffer: wgpu::Buffer,
    pub source_hash: u64,
}

impl EmitPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let group0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("user_shader_emit group0"),
            entries: &[
                rw_storage(0), // instance_pool
                rw_storage(1), // instance_alloc
                ro_storage(2), // octree_nodes
                ro_storage(3), // brick_pool
                ro_storage(4), // leaf_attr_pool
                rw_storage(5), // overflow
            ],
        });
        let group1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("user_shader_emit group1"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(
                        std::mem::size_of::<EmitRegionUniform>() as u64,
                    ),
                },
                count: None,
            }],
        });
        let group2_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("user_shader_emit group2"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: std::num::NonZeroU64::new(
                        std::mem::size_of::<EmitDispatchUniform>() as u64,
                    ),
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("user_shader_emit pipeline layout"),
            bind_group_layouts: &[
                Some(&group0_layout),
                Some(&group1_layout),
                Some(&group2_layout),
            ],
            immediate_size: 0,
        });
        let emit_pipeline = build_emit_pipeline(device, &pipeline_layout, "");

        let alloc_buf_size = (MAX_INSTANCE_REGIONS as u64) * 4;
        let instance_alloc_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("user_shader_emit instance_alloc"),
            size: alloc_buf_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let overflow_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("user_shader_emit overflow"),
            size: OVERFLOW_COUNTER_COUNT * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let regions_buffer_size =
            (std::mem::size_of::<EmitRegionUniform>() as u64) * MAX_INSTANCE_REGIONS as u64;
        let regions_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("user_shader_emit regions"),
            size: regions_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let dispatch_uniforms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("user_shader_emit dispatch_uniforms"),
            size: EMIT_DISPATCH_UNIFORM_STRIDE * MAX_INSTANCE_REGIONS as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            group0_layout,
            group1_layout,
            group2_layout,
            pipeline_layout,
            emit_pipeline,
            instance_alloc_buffer,
            overflow_buffer,
            regions_buffer,
            dispatch_uniforms_buffer,
            source_hash: 0,
        }
    }

    pub fn reload_user_shaders(
        &mut self,
        device: &wgpu::Device,
        emit_chunk: &str,
        source_hash: u64,
    ) -> bool {
        if source_hash == self.source_hash {
            return false;
        }
        self.emit_pipeline = build_emit_pipeline(device, &self.pipeline_layout, emit_chunk);
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

fn build_emit_pipeline(
    device: &wgpu::Device,
    pipeline_layout: &wgpu::PipelineLayout,
    emit_chunk: &str,
) -> wgpu::ComputePipeline {
    let source = compose_emit_source(emit_chunk);
    crate::validate_wgsl(&source, "user_shader_emit");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("user_shader_emit"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("user_shader_emit emit"),
        layout: Some(pipeline_layout),
        module: &module,
        entry_point: Some("emit_main"),
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

    fn make_request(stride_u32: u32, max_instances: u32) -> InstanceRegionRequest {
        InstanceRegionRequest {
            host_object_id: 1,
            material_id: 5,
            shader_name: "grass".to_string(),
            params: vec![],
            aabb_min: [0.0; 3],
            aabb_max: [1.0; 3],
            cell_size: 0.04,
            input_hash: 0,
            animated: false,
            region_thickness: 0.05,
            tile_index: NO_TILE,
            stride_u32,
            max_instances,
            host_octree_root: HOST_NO_HOST_SENTINEL,
            host_octree_depth: 0,
            host_octree_extent: 0.0,
            host_grid_origin: [0.0; 3],
            host_inverse_world: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    #[test]
    fn emit_region_uniform_size_is_192() {
        // WGSL EmitRegionUniform total layout:
        //   aabb_min(0..12) + cell_size(12..16) + aabb_max(16..28) +
        //   shader_id(28..32) + time(32..36) + material_id(36..40) +
        //   region_thickness(40..44) + instance_block_offset(44..48) +
        //   instance_block_size(48..52) + instance_stride_u32(52..56) +
        //   host_octree_root(56..60) + host_octree_depth(60..64) +
        //   host_octree_extent(64..68) + _pad_host(68..80) +
        //   host_grid_origin(80..92) + _pad_grid(92..96) +
        //   params(96..128) + host_inverse_world(128..192) = 192
        assert_eq!(std::mem::size_of::<EmitRegionUniform>(), 192);
    }

    #[test]
    fn emit_dispatch_uniform_size_is_16() {
        assert_eq!(std::mem::size_of::<EmitDispatchUniform>(), 16);
    }

    #[test]
    fn emit_shader_validates_with_empty_chunk() {
        let source = compose_emit_source("");
        assert_wgsl_valid(&source, "user_shader_emit");
        assert!(source.contains("emit_main"));
    }

    #[test]
    fn emit_shader_validates_with_one_instance_shader() {
        // Compose a chunk like the real composer would and confirm the
        // spliced source validates with naga.
        use crate::shader_composer::{compose, scan_dir};
        let dir = std::env::temp_dir().join(format!(
            "rkpatch_emit_validate_{}",
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
"#,
        )
        .unwrap();
        let registry = scan_dir(&dir).unwrap();
        let chunks = compose(&registry);
        let source = compose_emit_source(&chunks.emit);
        assert_wgsl_valid(&source, "emit_with_grass");
        assert!(source.contains("rkp_user_1_emit_instance"));
        assert!(source.contains("rkp_user_1_emit"));
    }

    #[test]
    fn cache_first_lookup_is_dirty() {
        let mut cache = InstanceRegionCache::with_capacity(8192);
        cache.set_pool_base(0);
        let req = make_request(8, 64);
        let slot = cache.lookup_or_allocate(&req, 0xAA, 0xBB).unwrap();
        assert!(slot.topology_dirty);
        assert!(slot.fill_dirty);
        assert_eq!(slot.stride_u32, 8);
        // Bucket-rounded extent ≥ requested stride * max_instances.
        assert!(slot.instance_block_size >= 64);
    }

    #[test]
    fn cache_repeat_lookup_clean() {
        let mut cache = InstanceRegionCache::with_capacity(8192);
        cache.set_pool_base(0);
        let req = make_request(8, 64);
        let _ = cache.lookup_or_allocate(&req, 0xAA, 0xBB).unwrap();
        let slot = cache.lookup_or_allocate(&req, 0xAA, 0xBB).unwrap();
        assert!(!slot.topology_dirty);
        assert!(!slot.fill_dirty);
    }

    #[test]
    fn cache_topology_change_dirties_topology_and_fill() {
        let mut cache = InstanceRegionCache::with_capacity(8192);
        cache.set_pool_base(0);
        let req = make_request(8, 64);
        let _ = cache.lookup_or_allocate(&req, 0xAA, 0xBB).unwrap();
        let slot = cache.lookup_or_allocate(&req, 0xCC, 0xBB).unwrap();
        assert!(slot.topology_dirty);
        assert!(slot.fill_dirty); // topology change implies fill dirty
    }

    #[test]
    fn cache_fill_change_only_dirties_fill() {
        let mut cache = InstanceRegionCache::with_capacity(8192);
        cache.set_pool_base(0);
        let req = make_request(8, 64);
        let _ = cache.lookup_or_allocate(&req, 0xAA, 0xBB).unwrap();
        let slot = cache.lookup_or_allocate(&req, 0xAA, 0xCC).unwrap();
        assert!(!slot.topology_dirty);
        assert!(slot.fill_dirty);
    }

    #[test]
    fn cache_stride_change_reallocs_extent() {
        let mut cache = InstanceRegionCache::with_capacity(65536);
        cache.set_pool_base(0);
        let req8 = make_request(8, 64);
        let slot8 = cache.lookup_or_allocate(&req8, 0xAA, 0xBB).unwrap();
        let req4 = make_request(4, 128);
        let slot4 = cache.lookup_or_allocate(&req4, 0xAA, 0xBB).unwrap();
        // Stride changed → extent freed + re-allocated, possibly at a
        // different offset.
        assert_eq!(slot4.stride_u32, 4);
        // Different stride/instance_count combinations may share extent
        // buckets. The key invariant: the new slot's extent fits the
        // new request.
        assert!(slot4.instance_block_size >= 128);
        let _ = slot8;
    }

    #[test]
    fn cache_distinct_keys_get_distinct_extents() {
        let mut cache = InstanceRegionCache::with_capacity(8192);
        cache.set_pool_base(0);
        let a = make_request(8, 64);
        let mut b = make_request(8, 64);
        b.material_id = 6;
        let sa = cache.lookup_or_allocate(&a, 0xAA, 0xBB).unwrap();
        let sb = cache.lookup_or_allocate(&b, 0xAA, 0xBB).unwrap();
        assert_ne!(sa.instance_block_offset, sb.instance_block_offset);
        // also cover tile_index keying
        let mut c = make_request(8, 64);
        c.tile_index = [1, 0, 0];
        let sc = cache.lookup_or_allocate(&c, 0xAA, 0xBB).unwrap();
        assert_ne!(sa.instance_block_offset, sc.instance_block_offset);
    }

    #[test]
    fn cache_evict_untouched() {
        let mut cache = InstanceRegionCache::with_capacity(8192);
        cache.set_pool_base(0);
        let req1 = make_request(8, 64);
        let mut req2 = make_request(8, 64);
        req2.material_id = 6;
        let _ = cache.lookup_or_allocate(&req1, 0xAA, 0xBB).unwrap();
        let _ = cache.lookup_or_allocate(&req2, 0xAA, 0xBB).unwrap();
        assert_eq!(cache.entry_count(), 2);
        cache.begin_frame();
        let _ = cache.lookup_or_allocate(&req1, 0xAA, 0xBB).unwrap(); // touch only req1
        cache.evict_untouched();
        assert_eq!(cache.entry_count(), 1);
    }

    #[test]
    fn cache_pool_base_change_flushes() {
        let mut cache = InstanceRegionCache::with_capacity(8192);
        cache.set_pool_base(0);
        let req = make_request(8, 64);
        let _ = cache.lookup_or_allocate(&req, 0xAA, 0xBB).unwrap();
        cache.set_pool_base(1000);
        assert_eq!(cache.entry_count(), 0);
    }

    #[test]
    fn cache_pool_exhaustion_returns_none() {
        // Tiny pool — only one bucket fits.
        let mut cache = InstanceRegionCache::with_capacity(INSTANCE_BUCKET_MIN);
        cache.set_pool_base(0);
        let req = make_request(8, 4); // needs stride*count = 32 u32s, bucket = 64
        let _ = cache.lookup_or_allocate(&req, 0xAA, 0xBB).unwrap();
        let mut req2 = make_request(8, 4);
        req2.material_id = 6;
        // Second region — pool out of room, returns None.
        assert!(cache.lookup_or_allocate(&req2, 0xAA, 0xBB).is_none());
    }

    #[test]
    fn samples_per_axis_round_up() {
        let mut req = make_request(8, 64);
        req.aabb_min = [0.0, 0.0, 0.0];
        req.aabb_max = [1.0, 1.0, 1.0];
        req.cell_size = 0.04;
        // bp_cell = 0.16. extent 1.0 → samples = ceil(1.0/0.16) = 7.
        assert_eq!(samples_per_axis(&req), 7);
    }

    #[test]
    fn build_uniform_offsets_match_slot() {
        let mut cache = InstanceRegionCache::with_capacity(8192);
        cache.set_pool_base(2000);
        let req = make_request(8, 64);
        let slot = cache.lookup_or_allocate(&req, 0xAA, 0xBB).unwrap();
        let u = build_emit_region_uniform(&req, &slot, 1, 0.0);
        assert_eq!(u.shader_id, 1);
        assert_eq!(u.instance_block_offset, slot.instance_block_offset);
        assert_eq!(u.instance_block_size, slot.instance_block_size);
        assert_eq!(u.instance_stride_u32, 8);
        assert_eq!(u.material_id, 5);
        assert_eq!(u.region_thickness, 0.05);
    }

    #[test]
    fn touched_keys_yields_only_touched_entries() {
        let mut cache = InstanceRegionCache::with_capacity(8192);
        cache.set_pool_base(0);
        let req1 = make_request(8, 64);
        let mut req2 = make_request(8, 64);
        req2.material_id = 6;
        req2.tile_index = [3, 0, 7];
        let _ = cache.lookup_or_allocate(&req1, 0xAA, 0xBB).unwrap();
        let _ = cache.lookup_or_allocate(&req2, 0xAA, 0xBB).unwrap();

        // First frame: both touched.
        let keys: Vec<_> = cache.touched_keys().collect();
        assert_eq!(keys.len(), 2);
        // Order is HashMap-iteration order — sort for stable assertion.
        let mut sorted_keys = keys.clone();
        sorted_keys.sort();
        assert!(sorted_keys.contains(&(1, 5, NO_TILE)));
        assert!(sorted_keys.contains(&(1, 6, [3, 0, 7])));

        // Next frame: only req1 touched.
        cache.begin_frame();
        let _ = cache.lookup_or_allocate(&req1, 0xAA, 0xBB).unwrap();
        let keys: Vec<_> = cache.touched_keys().collect();
        assert_eq!(keys, vec![(1, 5, NO_TILE)]);
    }

    #[test]
    fn build_uniform_truncates_excess_params() {
        let mut req = make_request(8, 64);
        req.params = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mut cache = InstanceRegionCache::with_capacity(8192);
        cache.set_pool_base(0);
        let slot = cache.lookup_or_allocate(&req, 0xAA, 0xBB).unwrap();
        let u = build_emit_region_uniform(&req, &slot, 1, 0.0);
        assert_eq!(u.params[0], [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(u.params[1], [5.0, 6.0, 7.0, 8.0]);
    }
}
