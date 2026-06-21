//! RKP scene GPU buffer management.
//!
//! Two upload paths, both explicit:
//! - [`ArvxScene::upload_geometry`]: voxel pool, octree, color. Called on geometry change only.
//! - [`ArvxScene::upload_frame`]: objects, camera. Called every frame (cheap — ~200 KB).
//!
//! No incremental updates, no caching, no callbacks. The caller builds the full
//! data each time and passes it in.

use crate::arvx_gpu_object::{ArvxGpuAsset, ArvxGpuInstance};

/// Camera uniforms matching the WGSL `CameraUniforms` struct.
///
/// Layout (208 + 16 = 224 bytes):
/// - 4×vec4<f32> camera basis (position, forward, right, up) — 64 B
/// - resolution + jitter — 16 B
/// - layer_mask + focus_object_id + 8 B padding — 16 B
/// - prev_vp + view_proj — 128 B
///
/// `layer_mask`/`focus_object_id` come from the rendering viewport's
/// `SceneFilter` (see `arvx_engine::viewport`). Defaults of `u32::MAX` for
/// both keep all objects visible (mask matches everything; focus matches
/// no real `object_id`, which are sequential from 0).
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniforms {
    pub position: [f32; 4],
    pub forward: [f32; 4],
    pub right: [f32; 4],
    pub up: [f32; 4],
    pub resolution: [f32; 2],
    pub jitter: [f32; 2],
    pub layer_mask: u32,
    pub focus_object_id: u32,
    pub _pad: [u32; 2],
    pub prev_vp: [[f32; 4]; 4],
    pub view_proj: [[f32; 4]; 4],
}

/// Geometry data — uploaded once when geometry changes (load, sculpt, voxelize).
pub struct GeometryUpload<'a> {
    /// Octree node values (packed u32s), one per node slot.
    pub octree_nodes: &'a [u32],
    /// Parallel prefiltered-LOD attr ids (u32s), one per node slot. Same
    /// length as `octree_nodes`. Entry is `INTERNAL_ATTR_NONE` for non-
    /// branches and for branches without a prefilter. The scene buffer
    /// interleaves these with `octree_nodes` into a single
    /// `array<vec2<u32>>` binding so we stay under the 12-storage-buffer
    /// per-stage limit.
    pub octree_internal_attrs: &'a [u32],
    /// Per-leaf attributes: `LeafAttr { normal_oct, material_primary,
    /// material_secondary_blend }`, 8 B each. Indexed by the leaf_attr_id
    /// stored in octree leaf nodes.
    pub leaf_attr_pool: &'a [u8],
    /// Per-leaf color — parallel to `leaf_attr_pool`, 4 B packed RGBA per slot.
    /// 0 means "no override; use material base_color".
    pub color_pool: &'a [u8],
    /// Per-leaf skinning weights — parallel to `leaf_attr_pool`, 8 B
    /// `BoneVoxel` per slot (4 bone indices + 4 weights quantized to
    /// u8). Zero-filled for unskinned assets; the shader still has to
    /// read-gate on per-object `is_skinned` because the buffer is
    /// scene-wide.
    pub bone_weights: &'a [u8],
    /// Brick storage: each brick is a contiguous run of 64 u32 cells (256 B).
    /// Indexed by `brick_id * 64 + flat_cell_index`. A cell's value is either
    /// 0xFFFFFFFF (empty) or a leaf_attr_id.
    pub brick_pool: &'a [u8],
    /// Brick face-adjacency links — 6 u32 per brick in the order
    /// `(−X, +X, −Y, +Y, −Z, +Z)`, byte-cast. Each entry is a
    /// neighboring brick_id or a FACE_EMPTY/FACE_INTERIOR sentinel.
    /// Used by the Surface-Nets reconstruction shader to traverse into
    /// adjacent bricks for cross-boundary neighbor reads.
    pub brick_face_links: &'a [u8],
    /// Dirty byte ranges in the **interleaved GPU layout** of the
    /// octree buffer (each slot is 16 B = `vec4<u32>`). Drained from
    /// `OctreeAllocator::dirty_ranges()`; empty when no octree writes
    /// happened since the last upload. Falls back to a full upload
    /// when the tracker has `mark_full` set or when total dirty bytes
    /// exceed half the pool.
    pub octree_dirty: arvx_core::DirtyRanges,
    /// Dirty byte ranges in `leaf_attr_pool` (8 B per slot).
    pub leaf_attr_dirty: arvx_core::DirtyRanges,
    /// Dirty byte ranges in `color_pool` (4 B per slot).
    pub color_dirty: arvx_core::DirtyRanges,
    /// Dirty byte ranges in `bone_weights` (8 B per slot).
    pub bone_dirty: arvx_core::DirtyRanges,
    /// Dirty byte ranges in `brick_pool` (256 B per brick).
    pub brick_dirty: arvx_core::DirtyRanges,
}

/// Progress report from [`ArvxScene::upload_geometry_budgeted`] — how far
/// the byte-budgeted, multi-frame geometry upload has advanced for the
/// current epoch. The render worker uses it to decide whether to upload
/// per-asset meshes (only after `refs_uploaded`) and whether to advance
/// `last_uploaded_geometry_epoch` + clear the dirty trackers (only when
/// everything, including meshes, has shipped).
#[derive(Debug, Clone, Copy, Default)]
pub struct GeoBudgetProgress {
    /// Every data pool (leaf_attr / color / bone / brick) is fully
    /// resident on the GPU for this epoch — Phase A complete.
    pub data_drained: bool,
    /// The "references" (octree nodes + brick_face_links) have been
    /// uploaded — Phase B complete. Only ever true once `data_drained`.
    /// Until this is true the appended geometry is invisible (no octree
    /// node references the new leaf/brick slots), so a half-filled pool
    /// tail never shows as garbage.
    pub refs_uploaded: bool,
    /// Bytes written to the GPU this call — for telemetry / pacing.
    pub bytes_consumed: u64,
    /// A pool buffer was reallocated (grew) this call. The first frame of
    /// a burst grows every under-sized pool to its final size (one
    /// `buffers_epoch` bump); subsequent frames write in place.
    pub grew: bool,
}

/// Per-frame data. Camera uniforms are per-viewport and uploaded
/// separately via `ViewportRenderer::upload_camera`.
pub struct FrameUpload<'a> {
    /// Per-asset records, deduped upstream (one per unique octree). The
    /// `ArvxGpuInstance.asset_id` field indexes into this slice.
    pub assets: &'a [ArvxGpuAsset],
    /// Per-instance records — one per scene entity.
    pub instances: &'a [ArvxGpuInstance],
    /// Concatenated skinning palette — one `mat4x4<f32>` per bone across
    /// every skinned entity in the scene, in `ArvxGpuInstance`
    /// `bone_buffer_offset` order. Empty `&[]` when no animated entities
    /// are loaded (in which case the bone buffer keeps its dummy
    /// placeholder size so the shader bind still validates).
    pub bone_matrices: &'a [u8],
    /// Byte ranges within `bone_matrices` that differ from the GPU
    /// buffer's contents. When empty, the bone-matrix upload is
    /// skipped entirely (no bones moved this frame, or the C2-narrow
    /// path skipped the bone-matrix rebuild). When `is_full_pool`
    /// the layout shifted (entity add / remove / bone count change)
    /// and we fall back to a full ensure_and_write. Otherwise each
    /// range becomes one `queue.write_buffer` call. PERF_DEBT.md D1.
    pub bone_matrices_dirty: &'a arvx_core::DirtyRanges,
    /// Concatenated forward-pose dual quaternions — one 32-byte
    /// `DualQuat` per bone, parallel to the forward half of
    /// `bone_matrices`. Scatter's DQS branch reads directly from here,
    /// skipping the ~60-ALU per-influence matrix→quat extraction.
    pub bone_dual_quats: &'a [u8],
    /// Same dirty-range protocol as [`Self::bone_matrices_dirty`]
    /// but for [`Self::bone_dual_quats`]. PERF_DEBT.md D1.
    pub bone_dual_quats_dirty: &'a arvx_core::DirtyRanges,
    /// Per-instance paint overlay entries — one `OverlayEntry` (16 B)
    /// per painted leaf per painted instance. Each
    /// `ArvxGpuInstance.overlay_offset` + `overlay_count` slices into
    /// this buffer. Empty `&[]` when no entity carries paint
    /// (placeholder buffer keeps the bind valid).
    pub instance_overlays: &'a [u8],
    /// PERF_DEBT.md D2: dirty-range protocol matching
    /// [`Self::bone_matrices_dirty`]. Empty → skip the overlay
    /// upload entirely (idle tick between paint stamps); else fall
    /// through to `write_with_dirty`'s `is_full_pool` /
    /// per-range branches.
    pub instance_overlays_dirty: &'a arvx_core::DirtyRanges,
    /// Per-instance sculpt overlay — sorted `u32` array of removed
    /// `leaf_attr_id`s, one slice per carved instance. Each
    /// `ArvxGpuInstance.sculpt_offset` + `sculpt_count` slices into this
    /// buffer. Empty `&[]` when no entity has been carved (placeholder
    /// buffer keeps the bind valid). Phase A: Carve only.
    pub instance_sculpts: &'a [u8],
    /// PERF_DEBT.md D3: same dirty-range protocol as
    /// [`Self::instance_overlays_dirty`] but for `instance_sculpts`.
    pub instance_sculpts_dirty: &'a arvx_core::DirtyRanges,
}

/// Per-pool delta-upload telemetry, returned by `upload_pool_delta` /
/// `upload_octree_delta` so `upload_geometry` can log + propagate the
/// buffer-grew signal up to `buffers_epoch`.
#[derive(Debug, Default, Clone, Copy)]
struct UploadStats {
    grew: bool,
    bytes_written: u64,
    range_count: usize,
}

/// Maximum number of per-range `queue.write_buffer` calls per pool
/// per upload. Past this threshold the per-call driver overhead
/// (~0.5-2 ms each for staging-buffer acquire + command record)
/// dominates the actual byte transfer cost, and a single full-pool
/// write is faster end-to-end even though it transfers far more
/// bytes. `coalesce_with_gap` in `geometry_upload` keeps the count
/// low for typical stamps; this cap covers pathological cases (very
/// scattered per-slot mutations).
const MAX_DELTA_RANGES: usize = 64;

/// Per-frame byte slice of a pool's un-resident tail to upload, given
/// `prev_valid` bytes already on the GPU, a CPU pool of `len` bytes, and a
/// per-call `budget`. Returns `[lo, hi)`: `lo = min(prev_valid, len)`,
/// `hi = min(prev_valid + budget, len)`. `lo == hi` means the tail is fully
/// resident (nothing to write). Pure (no GPU) so the byte accounting that
/// drives [`ArvxScene::upload_pool_budgeted`] is unit-testable.
fn budgeted_tail(prev_valid: u64, len: u64, budget: u64) -> (u64, u64) {
    let lo = prev_valid.min(len);
    let hi = lo.saturating_add(budget).min(len);
    (lo, hi)
}

/// Storage stride (u32 lanes) of one octree node on the GPU. Lanes:
///   .x = node value (EMPTY / INTERIOR / BRANCH offset / LEAF / BRICK id)
///   .y = prefiltered-LOD attr id (INTERNAL_ATTR_NONE when absent)
///   .z = quantized tight occupancy AABB lo (8 bits per axis × xyz, last
///        byte reserved); zeroed during Step 1 of the per-node tight-bounds
///        rollout — bake/march writes land in Steps 2/3.
///   .w = quantized tight occupancy AABB hi (same layout); zeroed during
///        Step 1.
pub const OCTREE_NODE_U32S: usize = 4;
/// Byte stride of one octree node on the GPU.
pub const OCTREE_NODE_BYTES: u64 = (OCTREE_NODE_U32S * 4) as u64;

/// GPU scene buffer manager for Arrvox.
///
/// Bind group layout (group 0):
///   0: brick_pool (storage, read) — flat array of u32 cells, `brick_id * 64 + idx` indexes into it.
///       (Was a dummy voxel_pool slot pre-bricks; repurposed because we
///       were one storage-buffer over the per-stage limit.)
///   1: octree_nodes (storage, read) — `array<vec4<u32>>`: see
///       `OCTREE_NODE_U32S` for the lane layout. The buffer was
///       interleaved (`vec2<u32>`) before per-node tight bounds; now
///       widened to `vec4<u32>` with `.zw` reserved for the quantized
///       AABB written at bake time.
///   2: objects (storage, read)
///   3: camera (uniform)
///   4: color_pool (storage, read) — parallel to leaf_attr_pool
///   5: bone_matrices (storage, read)
///   6: bone_weights (storage, read)
///   7: brick_face_links (storage, read) — 6 u32 per brick giving
///       adjacent brick ids / FACE_{EMPTY,INTERIOR} sentinels. (This
///       slot was `deformed_pool` pre-Surface-Nets; deformed_pool
///       wasn't wired into the active pipeline so the slot was free.)
///   8: leaf_attr_pool (storage, read) — `LeafAttr { normal_oct, material_primary, material_secondary_blend }`
///
/// 8 storage buffers + 1 uniform in group 0; group 2 holds 4 more storage
/// buffers + 1 uniform — total 12 storage buffers per stage, exactly at
/// the arvx-render device limit.
///
/// Shared scene GPU buffers. The camera uniform is **not** here — it's
/// per-viewport (`ViewportRenderer::camera_buffer`) so that two viewports
/// can render different cameras inside one encoder without racing.
/// `build_bind_group` stamps out a bind group pairing these shared buffers
/// with the caller's camera buffer; each VR owns its own group.
pub struct ArvxScene {
    pub brick_pool_buffer: wgpu::Buffer,
    pub octree_nodes_buffer: wgpu::Buffer,
    /// Per-instance records — one entry per scene entity. Sourced from
    /// `FrameUpload::instances`.
    pub objects_buffer: wgpu::Buffer,
    /// Per-asset records — one entry per unique octree. Multiple
    /// instances reference the same slot via `ArvxGpuInstance.asset_id`.
    /// Sourced from `FrameUpload::assets`.
    pub assets_buffer: wgpu::Buffer,
    pub color_pool_buffer: wgpu::Buffer,
    pub bone_matrices_buffer: wgpu::Buffer,
    pub bone_weights_buffer: wgpu::Buffer,
    pub brick_face_links_buffer: wgpu::Buffer,
    pub leaf_attr_pool_buffer: wgpu::Buffer,
    /// Per-frame precomputed forward dual quaternions — one 32-byte
    /// `DualQuat` per bone across every skinned entity, in
    /// `SkinnedBinding.bone_dq_offset` order. The mesh VS's DQS branch
    /// reads this directly; the matrix palette is only used by LBS.
    pub bone_dual_quats_buffer: wgpu::Buffer,
    /// Per-instance sparse paint overlay buffer (Phase 3). One
    /// `OverlayEntry` (16 B) per painted leaf, per painted instance,
    /// concatenated. Each `ArvxGpuInstance.overlay_offset` +
    /// `overlay_count` slices into this. Bound at binding(13).
    pub instance_overlay_buffer: wgpu::Buffer,
    /// Per-instance sculpt overlay buffer (Phase A). Sorted `u32` slice
    /// per carved instance, each entry a removed `leaf_attr_id`. Each
    /// `ArvxGpuInstance.sculpt_offset` + `sculpt_count` slices into
    /// this. Bound at binding(14).
    pub instance_sculpt_buffer: wgpu::Buffer,
    /// User-shader emitted instances. Each entry is one `ArvxGpuInstance`
    /// (128 B) representing a single emitted primitive (grass blade,
    /// scatter object, etc.) with a forward affine `world` matrix and
    /// `asset_id` pointing at the shader's prototype asset.
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// Incremented whenever a shared buffer reallocates. Each VR caches
    /// the epoch it built its bind group at; rebuilds when the scene's
    /// epoch moves ahead.
    buffers_epoch: u64,
    /// Bytes of each delta-uploaded pool buffer that currently hold
    /// valid data on the GPU (`= the last uploaded data.len()`). The
    /// buffer is over-allocated 2× on growth, so its *capacity* exceeds
    /// its *valid* extent — copy-on-grow must carry forward only the
    /// valid prefix, not the padding. See [`Self::upload_pool_delta`].
    brick_pool_valid: u64,
    leaf_attr_pool_valid: u64,
    color_pool_valid: u64,
    bone_weights_valid: u64,
}

impl ArvxScene {
    pub fn new(device: &wgpu::Device) -> Self {
        let brick_pool_buffer = Self::create_storage(device, "arvx_brick_pool", 256);
        // 16-byte stride: each slot is `vec4<u32>` (value, prefilter-id, tight-aabb-lo, tight-aabb-hi).
        let octree_nodes_buffer = Self::create_storage(
            device, "arvx_octree_nodes", OCTREE_NODE_BYTES,
        );
        let objects_buffer = Self::create_storage(
            device, "arvx_objects",
            std::mem::size_of::<ArvxGpuInstance>() as u64,
        );
        let assets_buffer = Self::create_storage(
            device, "arvx_assets",
            std::mem::size_of::<ArvxGpuAsset>() as u64,
        );
        let color_pool_buffer = Self::create_storage(device, "arvx_color_pool", 4);
        let bone_matrices_buffer = Self::create_storage(device, "arvx_bone_matrices", 64);
        let bone_weights_buffer = Self::create_storage(device, "arvx_bone_weights", 4);
        let brick_face_links_buffer = Self::create_storage(device, "arvx_brick_face_links", 24);
        let leaf_attr_pool_buffer = Self::create_storage(device, "arvx_leaf_attr_pool", 8);
        // Start with a 32-byte placeholder so the binding validates
        // even before any skinned entity is loaded.
        let bone_dual_quats_buffer = Self::create_storage(device, "arvx_bone_dual_quats", 32);

        // Per-instance overlay buffer — starts at the 16-byte
        // single-entry placeholder so the bind validates even when no
        // entity is painted yet. `upload_frame` grows it as paint
        // accumulates.
        let instance_overlay_buffer = Self::create_storage(
            device, "arvx_instance_overlay", 16,
        );
        // Per-instance sculpt overlay — 4-byte placeholder (one u32
        // slot) so the bind validates before any carve happens.
        // `upload_frame` grows it as sculpt edits accumulate.
        let instance_sculpt_buffer = Self::create_storage(
            device, "arvx_instance_sculpt", 4,
        );

        let bind_group_layout = Self::create_layout(device);

        Self {
            brick_pool_buffer, octree_nodes_buffer, objects_buffer, assets_buffer,
            color_pool_buffer, bone_matrices_buffer,
            bone_weights_buffer, brick_face_links_buffer, leaf_attr_pool_buffer,
            bone_dual_quats_buffer,
            instance_overlay_buffer,
            instance_sculpt_buffer,
            bind_group_layout,
            buffers_epoch: 0,
            brick_pool_valid: 0,
            leaf_attr_pool_valid: 0,
            color_pool_valid: 0,
            bone_weights_valid: 0,
        }
    }

    /// Current buffers epoch. `ViewportRenderer` compares against this on
    /// every frame and rebuilds its bind group when the scene moves ahead.
    pub fn buffers_epoch(&self) -> u64 {
        self.buffers_epoch
    }

    /// Build a scene bind group using the caller-owned `camera_buffer` at
    /// binding 3 and the scene's shared buffers everywhere else. Called by
    /// `ViewportRenderer` at construction and after every buffer-epoch bump.
    pub fn build_bind_group(
        &self,
        device: &wgpu::Device,
        camera_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        Self::create_bind_group(
            device, &self.bind_group_layout,
            &self.brick_pool_buffer, &self.octree_nodes_buffer, &self.objects_buffer,
            camera_buffer, &self.color_pool_buffer, &self.bone_matrices_buffer,
            &self.bone_weights_buffer, &self.brick_face_links_buffer, &self.leaf_attr_pool_buffer,
            &self.bone_dual_quats_buffer,
            &self.assets_buffer, &self.instance_overlay_buffer,
            &self.instance_sculpt_buffer,
        )
    }

    /// CPU-managed asset data lives in `[0 .. cpu_*_bytes]` of each
    /// shared pool. Returns `true` if any underlying buffer reallocated
    /// — caller must rebuild any cached bind groups referencing them.
    pub fn ensure_pool_layout(
        &mut self,
        device: &wgpu::Device,
        cpu_octree_bytes: u64,
        cpu_brick_bytes: u64,
        cpu_leaf_attr_bytes: u64,
        cpu_face_links_bytes: u64,
    ) -> bool {
        let mut bumped = false;
        bumped |= Self::ensure_capacity(
            device, &mut self.octree_nodes_buffer, "arvx_octree_nodes",
            cpu_octree_bytes,
        );
        bumped |= Self::ensure_capacity(
            device, &mut self.brick_pool_buffer, "arvx_brick_pool",
            cpu_brick_bytes,
        );
        bumped |= Self::ensure_capacity(
            device, &mut self.leaf_attr_pool_buffer, "arvx_leaf_attr_pool",
            cpu_leaf_attr_bytes,
        );
        bumped |= Self::ensure_capacity(
            device, &mut self.brick_face_links_buffer, "arvx_brick_face_links",
            cpu_face_links_bytes,
        );
        if bumped {
            self.buffers_epoch += 1;
        }
        bumped
    }

    /// Buffer-size guarantee without writing data. Used by the
    /// user-shader transient-pool reservation. Returns `true` iff a
    /// new buffer was created (caller must refresh dependent bind
    /// groups).
    ///
    /// Caps the requested size at the device's
    /// `max_storage_buffer_binding_size` so a runaway transient
    /// reservation doesn't silently produce an invalid buffer
    /// (which would corrupt every bind group that references it
    /// and surface as a misleading "BindGroup is invalid"
    /// validation error at submit time). When the cap kicks in we
    /// log loudly so callers know to dial down their per-region
    /// estimates.
    fn ensure_capacity(
        device: &wgpu::Device,
        buffer: &mut wgpu::Buffer,
        label: &str,
        min_bytes: u64,
    ) -> bool {
        if min_bytes == 0 {
            return false;
        }
        let limit = device.limits().max_storage_buffer_binding_size as u64;
        if min_bytes > limit {
            eprintln!(
                "[arvx_scene] {label}: requested {min_bytes} B exceeds \
                 max_storage_buffer_binding_size ({limit} B). Clamping — \
                 the offending writer will see truncated capacity. \
                 Reduce per-region brick cap, MAX_REGIONS, or paint area."
            );
        }
        let request = min_bytes.min(limit);
        if request > buffer.size() {
            *buffer = Self::create_storage(device, label, request);
            true
        } else {
            false
        }
    }

    /// Upload geometry data. Call only when geometry changes (load, sculpt, voxelize).
    /// Grows buffers as needed; bumps the epoch on reallocation so `ViewportRenderer`
    /// rebuilds its cached bind group.
    ///
    /// **Delta-upload path (D5):** per-pool [`arvx_core::DirtyRanges`] from
    /// the scene manager drive `queue.write_buffer` at the marked byte
    /// offsets only. Falls back to a full-pool write when:
    ///
    /// * The buffer needed to grow (no usable pre-existing data on the GPU).
    /// * The tracker is in `mark_full` mode (the source has signalled
    ///   "everything's dirty" — first upload, voxelize, load).
    /// * Total marked bytes exceed half the pool (N small `write_buffer`
    ///   calls cost more than one big one past that threshold).
    pub fn upload_geometry(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &GeometryUpload) {
        let upload_start = std::time::Instant::now();
        assert_eq!(
            data.octree_nodes.len(),
            data.octree_internal_attrs.len(),
            "octree_nodes and octree_internal_attrs must have matching length",
        );

        // Diagnostic: how many prefilter attrs are populated in the upload?
        // Zero means prefilter didn't emit anything for this scene — LOD
        // won't fire in the shader no matter what the uniform says.
        let populated = data.octree_internal_attrs.iter()
            .filter(|&&v| v != 0xFFFF_FFFF).count();
        let total = data.octree_internal_attrs.len();
        let pct = if total > 0 { 100.0 * populated as f32 / total as f32 } else { 0.0 };
        eprintln!(
            "[arvx_scene] prefilter attrs: {populated}/{total} ({pct:.1}%) populated",
        );

        let mut needs_rebuild = false;
        let mut total_uploaded: u64 = 0;
        let mut range_count_total: usize = 0;

        let octree_stats = Self::upload_octree_delta(
            device, queue, &mut self.octree_nodes_buffer, "arvx_octree_nodes",
            data.octree_nodes, data.octree_internal_attrs, &data.octree_dirty,
        );
        needs_rebuild |= octree_stats.grew;
        total_uploaded += octree_stats.bytes_written;
        range_count_total += octree_stats.range_count;

        let brick_stats = Self::upload_pool_delta(
            device, queue, &mut self.brick_pool_buffer, &mut self.brick_pool_valid,
            "arvx_brick_pool", data.brick_pool, &data.brick_dirty,
        );
        needs_rebuild |= brick_stats.grew;
        total_uploaded += brick_stats.bytes_written;
        range_count_total += brick_stats.range_count;

        let leaf_stats = Self::upload_pool_delta(
            device, queue, &mut self.leaf_attr_pool_buffer, &mut self.leaf_attr_pool_valid,
            "arvx_leaf_attr_pool", data.leaf_attr_pool, &data.leaf_attr_dirty,
        );
        needs_rebuild |= leaf_stats.grew;
        total_uploaded += leaf_stats.bytes_written;
        range_count_total += leaf_stats.range_count;

        let color_stats = Self::upload_pool_delta(
            device, queue, &mut self.color_pool_buffer, &mut self.color_pool_valid,
            "arvx_color_pool", data.color_pool, &data.color_dirty,
        );
        needs_rebuild |= color_stats.grew;
        total_uploaded += color_stats.bytes_written;
        range_count_total += color_stats.range_count;

        if !data.bone_weights.is_empty() {
            let bone_stats = Self::upload_pool_delta(
                device, queue, &mut self.bone_weights_buffer, &mut self.bone_weights_valid,
                "arvx_bone_weights", data.bone_weights, &data.bone_dirty,
            );
            needs_rebuild |= bone_stats.grew;
            total_uploaded += bone_stats.bytes_written;
            range_count_total += bone_stats.range_count;
        }

        // face_links has no per-mutation tracker yet — keep the legacy
        // full-write path. Sculpt doesn't mutate face_links per stamp;
        // load/voxelize already pay the full cost so the path is fine
        // until a future change benefits from delta upload here too.
        needs_rebuild |= Self::ensure_and_write(
            device, queue, &mut self.brick_face_links_buffer, "arvx_brick_face_links",
            data.brick_face_links,
        );

        let mib = |bytes: u64| bytes as f64 / (1024.0 * 1024.0);
        let elapsed = upload_start.elapsed();
        eprintln!(
            "[arvx_scene] upload_geometry delta: octree={:.3} MiB ({}r)  brick={:.3} MiB ({}r)  \
             leaf_attr={:.3} MiB ({}r)  color={:.3} MiB ({}r)  bone={:.3} MiB ({}r)  \
             total={:.3} MiB ({}r)  in {:.2} ms",
            mib(octree_stats.bytes_written), octree_stats.range_count,
            mib(brick_stats.bytes_written), brick_stats.range_count,
            mib(leaf_stats.bytes_written), leaf_stats.range_count,
            mib(color_stats.bytes_written), color_stats.range_count,
            mib(0), 0,
            mib(total_uploaded), range_count_total,
            elapsed.as_secs_f64() * 1000.0,
        );

        if needs_rebuild {
            self.buffers_epoch += 1;
        }
    }

    /// Byte-budgeted, multi-frame geometry upload (the render-thread
    /// stall fix). Where [`Self::upload_geometry`] uploads the whole
    /// epoch synchronously — up to ~500 ms for a coalesced multi-asset
    /// load, which freezes the present — this spreads it across frames
    /// under a per-call `budget` (bytes), in three phases:
    ///
    /// - **Phase A** (this call, possibly over many frames): the data
    ///   pools `leaf_attr → color → bone → brick`, each tail-filled up to
    ///   the shared budget. Returns `data_drained` once all four are fully
    ///   resident.
    /// - **Phase B** (single frame, only once `data_drained`): the
    ///   *references* — octree nodes + `brick_face_links` — uploaded whole
    ///   (atomic, never split). Gated by `refs_uploaded`, which the caller
    ///   threads across frames and resets when the epoch changes. The
    ///   octree is uploaded last and whole so an octree node can never
    ///   reference a not-yet-resident leaf/brick slot.
    /// - **Phase C** (per-asset meshes) lives in the render worker — it
    ///   runs only after `refs_uploaded`.
    ///
    /// `*_valid` (the per-pool GPU high-water marks) are the cross-frame
    /// cursors; the caller holds no per-pool budget state. A `budget` of
    /// `u64::MAX` restores single-frame behavior (regression / A-B knob).
    pub fn upload_geometry_budgeted(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &GeometryUpload,
        budget: u64,
        refs_uploaded: &mut bool,
    ) -> GeoBudgetProgress {
        let mut grew = false;
        let mut consumed = 0u64;
        let mut remaining = budget;

        // ── Phase A: data pools, leaf_attr → color → bone → brick. ──
        // Each pool grows to its final size on the first touched frame
        // (so a later frame can fill the tail at a stable buffer object)
        // and writes up to the SHARED remaining budget. We never `break`
        // early on `remaining == 0`: a budget-0 call still grows an
        // under-sized pool (cheap) and re-applies any in-prefix sculpt
        // edits, it just writes no tail — so all pools reach final size
        // promptly even while only the front pool's tail is draining.
        let mut all_drained = true;
        for (buffer, valid, label, bytes, dirty) in [
            (
                &mut self.leaf_attr_pool_buffer, &mut self.leaf_attr_pool_valid,
                "arvx_leaf_attr_pool", data.leaf_attr_pool, &data.leaf_attr_dirty,
            ),
            (
                &mut self.color_pool_buffer, &mut self.color_pool_valid,
                "arvx_color_pool", data.color_pool, &data.color_dirty,
            ),
            (
                &mut self.bone_weights_buffer, &mut self.bone_weights_valid,
                "arvx_bone_weights", data.bone_weights, &data.bone_dirty,
            ),
            (
                &mut self.brick_pool_buffer, &mut self.brick_pool_valid,
                "arvx_brick_pool", data.brick_pool, &data.brick_dirty,
            ),
        ] {
            let (stats, drained) = Self::upload_pool_budgeted(
                device, queue, buffer, valid, label, bytes, dirty, remaining,
            );
            grew |= stats.grew;
            consumed += stats.bytes_written;
            remaining = remaining.saturating_sub(stats.bytes_written);
            all_drained &= drained;
        }

        // ── Phase B: references (octree + face_links), atomic-last. ──
        // Only once every data pool is resident, and only once per epoch
        // (the caller resets `*refs_uploaded` on an epoch change). Uploaded
        // whole — never budget-split — so a partially-written octree can
        // never publish a node that points at a not-yet-resident slot.
        if all_drained && !*refs_uploaded {
            let octree_stats = Self::upload_octree_delta(
                device, queue, &mut self.octree_nodes_buffer, "arvx_octree_nodes",
                data.octree_nodes, data.octree_internal_attrs, &data.octree_dirty,
            );
            grew |= octree_stats.grew;
            consumed += octree_stats.bytes_written;
            // `brick_face_links` has NO live GPU consumer: its only reader,
            // `lib::octree::descend_proto_octree`, is never CALLED in any
            // pass (it's bound only so the WESL composer parses lib::octree;
            // naga DCE drops the reads). The old `ensure_and_write` here
            // re-uploaded the full pool (~33 MiB → ~35 ms) every references
            // frame for data nothing reads. Keep the buffer SIZED so the
            // bind group stays valid, but skip the dead data write.
            // (If a real proto-march consumer is ever wired up, this must
            // become a delta-tracked upload — a full-write here is a perf
            // regression, not a correctness fix.)
            grew |= Self::ensure_capacity(
                device, &mut self.brick_face_links_buffer, "arvx_brick_face_links",
                data.brick_face_links.len() as u64,
            );
            *refs_uploaded = true;
        }

        if grew {
            self.buffers_epoch += 1;
        }
        GeoBudgetProgress {
            data_drained: all_drained,
            refs_uploaded: *refs_uploaded,
            bytes_consumed: consumed,
            grew,
        }
    }

    /// The four data pools' GPU high-water marks `(leaf_attr, color, bone,
    /// brick)` in bytes — how far the byte-budgeted upload has filled each.
    /// The render worker hands these to
    /// `ArvxSceneManager::clear_geometry_dirty_below` so consumed
    /// append-dirty is dropped and a drained pool stops re-uploading.
    pub fn pool_valid_marks(&self) -> (u64, u64, u64, u64) {
        (
            self.leaf_attr_pool_valid,
            self.color_pool_valid,
            self.bone_weights_valid,
            self.brick_pool_valid,
        )
    }

    /// Delta-aware upload for a homogeneous byte pool (brick_pool,
    /// leaf_attr_pool, color_pool, bone_weights). Returns telemetry +
    /// whether the buffer was reallocated. `valid_bytes` tracks how many
    /// bytes of `buffer` currently hold valid data (always `=
    /// data.len()` on return); the buffer is over-allocated 2× on
    /// growth, so this is below its capacity.
    ///
    /// **Growth = copy-on-grow, NOT a full CPU re-upload.** A naive
    /// `queue.write_buffer(new, 0, data)` on every realloc re-sends the
    /// ENTIRE growing pool from CPU — on a cold terrain generation that
    /// climbs to ~100 ms in a single frame, long enough for the window
    /// surface to go Outdated (rinch #42 then never recovers). Instead we
    /// copy the previously-valid prefix GPU→GPU (VRAM-local, ~µs), then
    /// write only the genuinely-new tail + any mutations inside the
    /// prefix. Per-grow cost becomes O(new data), not O(total pool).
    ///
    /// The non-growth path still falls back to a full pool write when:
    /// * Tracker is in `mark_full` mode.
    /// * Total marked bytes exceed `data.len() / 2`.
    /// * Range count exceeds [`MAX_DELTA_RANGES`] — the per-call
    ///   overhead of `queue.write_buffer` on modern wgpu drivers
    ///   (~1 ms in staging + command record) means past ~tens of
    ///   calls a single full-pool write is cheaper end-to-end, even
    ///   though it transfers many more bytes. The gap-merge in
    ///   `geometry_upload` keeps the count low for most stamps;
    ///   this cap covers the pathological case.
    fn upload_pool_delta(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &mut wgpu::Buffer,
        valid_bytes: &mut u64,
        label: &str,
        data: &[u8],
        dirty: &arvx_core::DirtyRanges,
    ) -> UploadStats {
        if data.is_empty() {
            return UploadStats::default();
        }
        let needed = data.len() as u64;
        // Loud guard: a storage buffer bound past
        // `max_storage_buffer_binding_size` is INVALID — the shader reads
        // garbage / the bind group fails / the device may fault. As a
        // scene's pools grow (e.g. terrain streaming many tiles) this is a
        // hard ceiling that's hit at a specific size, not gradually. If
        // this fires near a "surface lost", the growing pool is the cause.
        let max_binding = device.limits().max_storage_buffer_binding_size as u64;
        if needed > max_binding {
            eprintln!(
                "[POOL LIMIT] {label}: {needed} bytes EXCEEDS max_storage_buffer_binding_size \
                 {max_binding} — this binding is now invalid (GPU may fault / surface may drop)",
            );
        }
        // After any path below, `[0, data.len())` is resident on the GPU.
        let prev_valid = (*valid_bytes).min(needed);
        *valid_bytes = needed;
        if needed > buffer.size() {
            let old_buffer = std::mem::replace(
                buffer,
                Self::create_storage(device, label, needed.max(buffer.size().saturating_mul(2)).max(64)),
            );
            let mut bytes_written = 0u64;
            let mut range_count = 0usize;
            // 4-byte-aligned copy length (copy_buffer_to_buffer requires
            // it); any unaligned remainder of the old valid prefix falls
            // into the tail write below, so it's still re-sent.
            let copy_len = prev_valid & !3u64;
            // Carry the previously-resident prefix forward GPU→GPU. Must
            // be SUBMITTED before the `write_buffer`s below, whose staged
            // copies apply at the next submit (the frame's render pass) —
            // i.e. AFTER this copy, so the fresh writes overlay it.
            if copy_len > 0 {
                let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pool-grow-copy"),
                });
                enc.copy_buffer_to_buffer(&old_buffer, 0, buffer, 0, copy_len);
                queue.submit(std::iter::once(enc.finish()));
            }
            // Everything not carried forward (the genuinely-new tail, plus
            // any sub-4-byte remainder of the old prefix).
            if needed > copy_len {
                let tail = &data[copy_len as usize..];
                queue.write_buffer(buffer, copy_len, tail);
                bytes_written += tail.len() as u64;
                range_count += 1;
            }
            // Re-apply mutations that landed inside the carried-forward
            // prefix since the last upload (the GPU copy brought their
            // STALE bytes forward). Ranges in the tail are already covered.
            for (off, len) in dirty.iter() {
                if off as u64 >= copy_len {
                    continue;
                }
                let off_u = off as usize;
                let end = ((off as u64 + len as u64).min(copy_len)) as usize;
                if end <= off_u {
                    continue;
                }
                let slice = &data[off_u..end];
                queue.write_buffer(buffer, off as u64, slice);
                bytes_written += slice.len() as u64;
                range_count += 1;
            }
            return UploadStats { grew: true, bytes_written, range_count };
        }
        if dirty.is_empty() {
            return UploadStats::default();
        }
        let bytes_threshold = (data.len() / 2) as u64;
        if dirty.is_full_pool(data.len() as u32)
            || dirty.should_coalesce_to_full(bytes_threshold)
            || dirty.range_count() > MAX_DELTA_RANGES
        {
            queue.write_buffer(buffer, 0, data);
            return UploadStats { grew: false, bytes_written: data.len() as u64, range_count: 1 };
        }
        let mut bytes_written = 0u64;
        let mut range_count = 0usize;
        for (off, len) in dirty.iter() {
            let off_u = off as usize;
            if off_u >= data.len() {
                continue;
            }
            let end = (off_u + len as usize).min(data.len());
            let slice = &data[off_u..end];
            queue.write_buffer(buffer, off as u64, slice);
            bytes_written += slice.len() as u64;
            range_count += 1;
        }
        UploadStats { grew: false, bytes_written, range_count }
    }

    /// Byte-budgeted variant of [`Self::upload_pool_delta`] for a
    /// homogeneous data pool (brick / leaf_attr / color / bone).
    ///
    /// The whole point of the budget is to keep a cold load's giant
    /// `write_buffer` (e.g. a 125 MiB brick tail) from happening
    /// synchronously in ONE render frame — which freezes the present
    /// for hundreds of ms. Instead we grow the buffer to its **final**
    /// size ONCE (carrying the resident prefix forward GPU→GPU, ~µs),
    /// then write at most `budget` bytes of the un-resident tail per
    /// call, advancing `*valid_bytes` (the GPU high-water cursor). A
    /// later frame writes the next slice. `*valid_bytes == data.len()`
    /// means the pool is fully resident → returns `drained = true`.
    ///
    /// Grow-to-final-ONCE is mandatory, not progressive sizing: a
    /// buffer that grew each frame would realloc every frame (maximal
    /// bind-group churn) AND could leave an octree node (uploaded in
    /// the "references last" phase) pointing past the current buffer
    /// size = OOB read. Sizing to final up front keeps the buffer
    /// object stable while the tail fills.
    ///
    /// Any in-prefix `dirty` ranges (sculpt edits to the already-
    /// resident `[0, valid)` region) are re-applied in full each call —
    /// they are KB-scale by construction (a stamp never spans the
    /// budget), so they never split. A pure append marks only the tail
    /// `[valid, len)`, so the in-prefix loop writes nothing.
    ///
    /// SAFETY of progressive tail fill: the appended tail bytes are not
    /// referenced by any uploaded octree node until the "references
    /// last" phase runs (gated on every data pool being drained), so a
    /// half-filled tail is invisible, never garbage — proven against the
    /// shader descent in the design workflow.
    fn upload_pool_budgeted(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &mut wgpu::Buffer,
        valid_bytes: &mut u64,
        label: &str,
        data: &[u8],
        dirty: &arvx_core::DirtyRanges,
        budget: u64,
    ) -> (UploadStats, bool) {
        if data.is_empty() {
            *valid_bytes = 0;
            return (UploadStats::default(), true);
        }
        let needed = data.len() as u64;
        let max_binding = device.limits().max_storage_buffer_binding_size as u64;
        if needed > max_binding {
            eprintln!(
                "[POOL LIMIT] {label}: {needed} bytes EXCEEDS max_storage_buffer_binding_size \
                 {max_binding} — this binding is now invalid (GPU may fault / surface may drop)",
            );
        }
        let prev_valid = (*valid_bytes).min(needed);
        let mut grew = false;
        let mut bytes_written = 0u64;
        let mut range_count = 0usize;

        // 1. Grow to FINAL size once. Carry the resident prefix forward.
        //    `tail_start` is where the un-resident tail begins; on a grow
        //    the sub-4-byte remainder of `prev_valid` isn't GPU→GPU
        //    copyable, so it folds back into the tail write.
        let mut tail_start = prev_valid;
        if needed > buffer.size() {
            let old_buffer = std::mem::replace(
                buffer,
                Self::create_storage(
                    device, label,
                    needed.max(buffer.size().saturating_mul(2)).max(64),
                ),
            );
            grew = true;
            let copy_len = prev_valid & !3u64;
            if copy_len > 0 {
                let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pool-grow-copy"),
                });
                enc.copy_buffer_to_buffer(&old_buffer, 0, buffer, 0, copy_len);
                queue.submit(std::iter::once(enc.finish()));
            }
            tail_start = copy_len;
        }

        // 2. Re-apply in-place EDITS to already-resident data — dirty
        //    ranges below `tail_start` (the resident prefix). The CALLER
        //    drops consumed append-dirty (`clear_geometry_dirty_below`)
        //    each frame, so by the time the cursor has passed the append
        //    region its dirty is gone and this loop writes only genuine
        //    edits (a sculpt stamp to resident data, KB-scale). Without
        //    that drop, a pool's append-dirty (which spans the whole
        //    appended region) would re-upload the entire resident prefix
        //    every frame and starve the tail budget.
        for (off, len) in dirty.iter() {
            let o = off as u64;
            let e = (o + len as u64).min(tail_start);
            if o < tail_start && e > o {
                queue.write_buffer(buffer, o, &data[o as usize..e as usize]);
                bytes_written += e - o;
                range_count += 1;
            }
        }

        // 3. Budget the un-resident tail `[tail_start, needed)`.
        let (lo, hi) = budgeted_tail(tail_start, needed, budget);
        if hi > lo {
            queue.write_buffer(buffer, lo, &data[lo as usize..hi as usize]);
            bytes_written += hi - lo;
            range_count += 1;
        }

        // 4. Advance the GPU high-water cursor. `hi` is the new resident
        //    boundary of the contiguous tail; it never regresses below
        //    `tail_start`.
        *valid_bytes = hi.max(tail_start);
        let drained = *valid_bytes >= needed;
        if drained {
            *valid_bytes = needed;
        }
        (UploadStats { grew, bytes_written, range_count }, drained)
    }

    /// Delta-aware upload for the octree's interleaved-vec4<u32> GPU
    /// layout. Each CPU slot corresponds to 16 GPU bytes
    /// (node, prefilter_id, 0, 0). The dirty tracker carries byte
    /// offsets in the GPU layout, aligned to the 16-byte slot stride.
    fn upload_octree_delta(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &mut wgpu::Buffer,
        label: &str,
        nodes: &[u32],
        attrs: &[u32],
        dirty: &arvx_core::DirtyRanges,
    ) -> UploadStats {
        let slot_count = nodes.len();
        if slot_count == 0 {
            return UploadStats::default();
        }
        let needed_bytes = (slot_count * OCTREE_NODE_U32S * 4) as u64;
        let full_interleaved = |nodes: &[u32], attrs: &[u32]| -> Vec<u32> {
            let mut v = Vec::with_capacity(slot_count * OCTREE_NODE_U32S);
            for (i, &n) in nodes.iter().enumerate() {
                v.push(n);
                v.push(attrs[i]);
                v.push(0u32);
                v.push(0u32);
            }
            v
        };
        if needed_bytes > buffer.size() {
            let new_size = needed_bytes
                .max(buffer.size().saturating_mul(2))
                .max(64);
            *buffer = Self::create_storage(device, label, new_size);
            let interleaved = full_interleaved(nodes, attrs);
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&interleaved));
            return UploadStats { grew: true, bytes_written: needed_bytes, range_count: 1 };
        }
        if dirty.is_empty() {
            return UploadStats::default();
        }
        let bytes_threshold = needed_bytes / 2;
        if dirty.is_full_pool(needed_bytes as u32)
            || dirty.should_coalesce_to_full(bytes_threshold)
            || dirty.range_count() > MAX_DELTA_RANGES
        {
            let interleaved = full_interleaved(nodes, attrs);
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&interleaved));
            return UploadStats { grew: false, bytes_written: needed_bytes, range_count: 1 };
        }
        // Delta: pack one tiny interleaved scratch per range, then write.
        let mut bytes_written = 0u64;
        let mut range_count = 0usize;
        for (off, len) in dirty.iter() {
            let slot_start = (off / OCTREE_NODE_BYTES as u32) as usize;
            let slot_count_range = (len / OCTREE_NODE_BYTES as u32) as usize;
            if slot_start >= nodes.len() || slot_count_range == 0 {
                continue;
            }
            let slot_end = (slot_start + slot_count_range).min(nodes.len());
            let actual_slots = slot_end - slot_start;
            let mut scratch: Vec<u32> = Vec::with_capacity(actual_slots * OCTREE_NODE_U32S);
            for i in slot_start..slot_end {
                scratch.push(nodes[i]);
                scratch.push(attrs[i]);
                scratch.push(0u32);
                scratch.push(0u32);
            }
            queue.write_buffer(buffer, off as u64, bytemuck::cast_slice(&scratch));
            bytes_written += (actual_slots * OCTREE_NODE_U32S * 4) as u64;
            range_count += 1;
        }
        UploadStats { grew: false, bytes_written, range_count }
    }

    /// Upload per-frame asset + instance data. The caller has already
    /// deduplicated assets upstream; this is a straight write of both
    /// buffers. Bumps the epoch when either buffer reallocates so VRs
    /// rebuild their bind groups.
    /// Upload only the per-instance paint overlay buffer. Used by
    /// callers that need the overlay current before the rest of the
    /// per-frame upload (e.g. the user-shader BFS host-material probe
    /// runs before `upload_frame` because `upload_frame` depends on
    /// the BFS's transient asset list).
    pub fn upload_instance_overlay(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bytes: &[u8],
    ) {
        if Self::ensure_and_write(
            device, queue, &mut self.instance_overlay_buffer,
            "arvx_instance_overlay", bytes,
        ) {
            self.buffers_epoch += 1;
        }
    }

    /// Upload only the per-instance sculpt overlay buffer. Same
    /// out-of-band path as `upload_instance_overlay` — used when
    /// sculpt commits a frame's worth of edits between
    /// `apply_sculpt_brush` and the next `upload_frame`. Bumps the
    /// epoch on reallocation so VRs rebuild their bind groups.
    pub fn upload_instance_sculpt(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bytes: &[u8],
    ) {
        if Self::ensure_and_write(
            device, queue, &mut self.instance_sculpt_buffer,
            "arvx_instance_sculpt", bytes,
        ) {
            self.buffers_epoch += 1;
        }
    }

    pub fn upload_frame(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &FrameUpload) {
        let inst_bytes: &[u8] = bytemuck::cast_slice(data.instances);
        let asset_bytes: &[u8] = bytemuck::cast_slice(data.assets);
        let mut needs_rebuild = Self::ensure_and_write(device, queue, &mut self.objects_buffer, "arvx_objects", inst_bytes);
        needs_rebuild |= Self::ensure_and_write(device, queue, &mut self.assets_buffer, "arvx_assets", asset_bytes);

        // Bone matrices — PERF_DEBT.md D1 delta upload. The dirty
        // ranges from sim's BoneMatrixAllocator describe which bytes
        // changed since last upload; empty = skip entirely (no bones
        // moved this frame, or sim took the C2-narrow path and
        // didn't rebuild the bone buffers at all); `is_full_pool` =
        // fall back to ensure_and_write (covers buffer grow + entity
        // set / bone count changes). When the scene has no skinned
        // entities the slice is empty and the buffer keeps its
        // 64-byte placeholder from new() so the bind group stays
        // valid.
        if !data.bone_matrices.is_empty() {
            needs_rebuild |= Self::write_with_dirty(
                device, queue, &mut self.bone_matrices_buffer,
                "arvx_bone_matrices", data.bone_matrices,
                data.bone_matrices_dirty,
            );
        }
        if !data.bone_dual_quats.is_empty() {
            needs_rebuild |= Self::write_with_dirty(
                device, queue, &mut self.bone_dual_quats_buffer,
                "arvx_bone_dual_quats", data.bone_dual_quats,
                data.bone_dual_quats_dirty,
            );
        }
        // D1/D2/D3 telemetry. Quiet by default; env-gated so the
        // validation pass on splat5 can confirm the upload bytes
        // actually drop without spamming the console in normal runs.
        if std::env::var("ARVX_BONE_UPLOAD_PROFILE").is_ok() {
            let mat_bytes = data.bone_matrices_dirty.total_dirty_bytes();
            let dq_bytes = data.bone_dual_quats_dirty.total_dirty_bytes();
            let mat_ranges = data.bone_matrices_dirty.range_count();
            let dq_ranges = data.bone_dual_quats_dirty.range_count();
            let ovl_bytes = data.instance_overlays_dirty.total_dirty_bytes();
            let scu_bytes = data.instance_sculpts_dirty.total_dirty_bytes();
            let ovl_ranges = data.instance_overlays_dirty.range_count();
            let scu_ranges = data.instance_sculpts_dirty.range_count();
            eprintln!(
                "[frame-upload] bone={:.3} KiB (mat={}r dq={}r) overlay={:.3} KiB ({}r) sculpt={:.3} KiB ({}r) total_buf bone={:.3} ovl={:.3} scu={:.3} KiB",
                (mat_bytes + dq_bytes) as f64 / 1024.0,
                mat_ranges,
                dq_ranges,
                ovl_bytes as f64 / 1024.0,
                ovl_ranges,
                scu_bytes as f64 / 1024.0,
                scu_ranges,
                (data.bone_matrices.len() + data.bone_dual_quats.len()) as f64 / 1024.0,
                data.instance_overlays.len() as f64 / 1024.0,
                data.instance_sculpts.len() as f64 / 1024.0,
            );
        }
        // PERF_DEBT.md D2/D3 — delta upload for the flat overlay /
        // sculpt buffers. Empty `*_dirty` → skip the upload; render's
        // bind still references the buffer with last frame's content
        // (which matches sim's `gpu_instance_overlays`/`_sculpts`
        // since no mutation fired this tick). The mutation sites
        // (paint stamp / sculpt stamp / entity remove / clear_scene)
        // flip a bool that the sim's snapshot converts to
        // `mark_full(buf_len)`; here that resolves through
        // `write_with_dirty`'s `is_full_pool` branch into a single
        // `queue.write_buffer`.
        if !data.instance_overlays.is_empty() {
            needs_rebuild |= Self::write_with_dirty(
                device, queue, &mut self.instance_overlay_buffer,
                "arvx_instance_overlay", data.instance_overlays,
                data.instance_overlays_dirty,
            );
        }
        if !data.instance_sculpts.is_empty() {
            needs_rebuild |= Self::write_with_dirty(
                device, queue, &mut self.instance_sculpt_buffer,
                "arvx_instance_sculpt", data.instance_sculpts,
                data.instance_sculpts_dirty,
            );
        }

        if needs_rebuild {
            self.buffers_epoch += 1;
        }
    }

    /// Delta-upload variant of [`Self::ensure_and_write`]. Routes the
    /// upload based on `dirty`:
    /// * `is_empty()` — nothing to do (bytes match GPU buffer already).
    /// * `is_full_pool()` — single full-buffer write (covers buffer
    ///   grow and layout-shift cases that the dirty ranges can't
    ///   incrementally describe).
    /// * otherwise — one `queue.write_buffer` per range. The buffer
    ///   must already be at least `data.len()` bytes; if not we grow
    ///   first (which forces a full rewrite — every range becomes
    ///   stale against the new buffer).
    ///
    /// Returns `true` when the buffer was reallocated (caller's
    /// bind groups need to rebuild). PERF_DEBT.md D1.
    fn write_with_dirty(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &mut wgpu::Buffer,
        label: &str,
        data: &[u8],
        dirty: &arvx_core::DirtyRanges,
    ) -> bool {
        if data.is_empty() {
            return false;
        }
        let needed = data.len() as u64;
        if needed > buffer.size() {
            // Grow path: reallocate + full rewrite. The 2× headroom
            // matches ensure_and_write so streams of growing uploads
            // reallocate O(log N) times rather than every frame.
            let new_size = needed.max(buffer.size().saturating_mul(2)).max(64);
            *buffer = Self::create_storage(device, label, new_size);
            queue.write_buffer(buffer, 0, data);
            return true;
        }
        if dirty.is_empty() {
            // Sim told us nothing changed since last upload —
            // skipping is the whole point of D1.
            return false;
        }
        if dirty.is_full_pool(data.len() as u32) {
            queue.write_buffer(buffer, 0, data);
            return false;
        }
        for (off, len) in dirty.iter() {
            let off = off as usize;
            let len = len as usize;
            // Defensive: cap at slice length. The allocator computes
            // offsets in the same units we emit here, so a range past
            // `data.len()` would mean the allocator got out of sync
            // with the snapshot — treat as a bug surfaced at runtime
            // by clamping rather than panicking the render thread.
            if off >= data.len() {
                continue;
            }
            let end = (off + len).min(data.len());
            queue.write_buffer(buffer, off as u64, &data[off..end]);
        }
        false
    }

    fn ensure_and_write(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &mut wgpu::Buffer,
        label: &str,
        data: &[u8],
    ) -> bool {
        if data.is_empty() {
            return false;
        }
        let needed = data.len() as u64;
        if needed > buffer.size() {
            // Grow with 2× headroom so a stream of incremental appends
            // (e.g. paint stamps growing the per-instance overlay
            // frame-by-frame) reallocates O(log N) times rather than
            // every frame. Reallocation forces every consumer's bind
            // group to rebuild via `buffers_epoch`, so amortizing it
            // matters on the hot path.
            let new_size = needed.max(buffer.size().saturating_mul(2)).max(64);
            *buffer = Self::create_storage(device, label, new_size);
            queue.write_buffer(buffer, 0, data);
            true
        } else {
            queue.write_buffer(buffer, 0, data);
            false
        }
    }

    fn create_storage(device: &wgpu::Device, label: &str, min_size: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: min_size.max(4),
            // COPY_SRC so a buffer can be the source of the GPU→GPU
            // copy-on-grow in `upload_pool_delta` when it's superseded by
            // a larger one.
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    fn create_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        let storage_ro = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("arvx_scene_layout"),
            entries: &[
                storage_ro(0), // brick_pool
                storage_ro(1), // octree_nodes
                storage_ro(2), // objects
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_ro(4), // color_pool
                storage_ro(5), // bone_matrices
                storage_ro(6), // bone_weights
                storage_ro(7), // brick_face_links (was deformed_pool)
                storage_ro(8), // leaf_attr_pool
                storage_ro(9), // bone_dual_quats (DQS precomputed palette)
                storage_ro(10), // assets (per-asset deduped records)
                storage_ro(11), // instance_overlay (Phase 3 per-instance paint)
                storage_ro(12), // instance_sculpt (Phase A per-instance sculpt overlay)
            ],
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn create_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        brick_pool: &wgpu::Buffer,
        octree_nodes: &wgpu::Buffer,
        objects: &wgpu::Buffer,
        camera: &wgpu::Buffer,
        color_pool: &wgpu::Buffer,
        bone_matrices: &wgpu::Buffer,
        bone_weights: &wgpu::Buffer,
        brick_face_links: &wgpu::Buffer,
        leaf_attr_pool: &wgpu::Buffer,
        bone_dual_quats: &wgpu::Buffer,
        assets: &wgpu::Buffer,
        instance_overlay: &wgpu::Buffer,
        instance_sculpt: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("arvx_scene_bind_group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: brick_pool.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: octree_nodes.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: objects.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: camera.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: color_pool.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: bone_matrices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: bone_weights.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: brick_face_links.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: leaf_attr_pool.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: bone_dual_quats.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: assets.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: instance_overlay.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: instance_sculpt.as_entire_binding() },
            ],
        })
    }
}

#[cfg(test)]
mod budget_tests {
    use super::budgeted_tail;

    /// A tail larger than the budget walks across calls, advancing the
    /// high-water mark by `budget` each time until it reaches `len`.
    #[test]
    fn budget_splits_large_tail() {
        let len = 20;
        let budget = 8;
        let (lo0, hi0) = budgeted_tail(0, len, budget);
        assert_eq!((lo0, hi0), (0, 8));
        let (lo1, hi1) = budgeted_tail(hi0, len, budget);
        assert_eq!((lo1, hi1), (8, 16));
        let (lo2, hi2) = budgeted_tail(hi1, len, budget);
        assert_eq!((lo2, hi2), (16, 20)); // final slice is the 4-byte remainder
        assert_eq!(hi2, len, "tail fully drained after 3 calls");
    }

    /// A small pool (or stamp) under the budget completes in one call —
    /// this is the path that keeps a sculpt edit atomic (never split).
    #[test]
    fn budget_completes_small_in_one() {
        let len = 4096;
        let (lo, hi) = budgeted_tail(0, len, 8 * 1024 * 1024);
        assert_eq!((lo, hi), (0, 4096));
        assert_eq!(hi, len);
    }

    /// Budget exhausted earlier in the frame (remaining == 0) makes no
    /// progress this call — `lo == hi`, the cursor doesn't move.
    #[test]
    fn budget_zero_remaining_is_noop() {
        let (lo, hi) = budgeted_tail(8, 20, 0);
        assert_eq!((lo, hi), (8, 8));
    }

    /// Already-drained pool (cursor at `len`) is idempotent — no write.
    #[test]
    fn hw_at_len_is_noop() {
        let (lo, hi) = budgeted_tail(20, 20, 8);
        assert_eq!((lo, hi), (20, 20));
    }

    /// A cursor somehow past `len` (pool shrank via deallocate) clamps to
    /// `len` and writes nothing.
    #[test]
    fn hw_past_len_clamps() {
        let (lo, hi) = budgeted_tail(40, 20, 8);
        assert_eq!((lo, hi), (20, 20));
    }

    /// A budget larger than the whole remaining tail writes it all in one
    /// slice (the `ARVX_GEO_UPLOAD_BUDGET_BYTES=u64::MAX` regression path).
    #[test]
    fn huge_budget_writes_whole_tail_at_once() {
        let (lo, hi) = budgeted_tail(5, 100, u64::MAX);
        assert_eq!((lo, hi), (5, 100));
    }
}
