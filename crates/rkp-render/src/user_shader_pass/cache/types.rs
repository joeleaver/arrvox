//! Public types for the user-shader cache: sim → render request and
//! the per-frame slot descriptor `lookup_or_allocate` hands back.

// ============================================================
// Sim → render request
// ============================================================

/// One materialization request from sim → render. Stable across frames
/// for cache hit; rebuilt by sim each tick from the ECS scan.
#[derive(Debug, Clone)]
pub struct ShaderRegionRequest {
    /// Stable identifier — typically the host entity's scene id or a
    /// synthetic id for free-standing regions. Used as the cache key
    /// alongside `material_id`.
    pub host_object_id: u32,
    /// The host's leaf-level material that triggered this region. Used
    /// for cache keying so the same host with two shader-using
    /// materials gets two cache entries.
    pub material_id: u32,
    /// Shader name (file stem). Resolved against the registry to a
    /// `shader_id` at dispatch time. Empty / unregistered names skip
    /// the request.
    pub shader_name: String,
    /// Per-material shader params, packed in the shader's declared
    /// order. Length matches the shader's `params` schema; longer is
    /// truncated, shorter is zero-padded. The first 8 entries land in
    /// the GPU param array.
    pub params: Vec<f32>,
    /// World-space AABB the user's `generate` hook is sampled across.
    /// Must be a cube — the BFS subdivides isotropically.
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    /// Voxel size at the deepest level. Sim derives this from the
    /// shader's `@cell_size` directive (clamped against the cube's
    /// extent so the implied depth fits within `max_depth`).
    pub cell_size: f32,
    /// Folded with shader source hash + host geometry epoch into the
    /// cache key. Bumped by sim whenever any input the cache should
    /// invalidate on changes.
    pub input_hash: u64,
    /// `@animated` — regenerate every frame, ignoring the hash.
    pub animated: bool,
    /// `@region_thickness` — Lipschitz band around host surface within
    /// which the classifier keeps cells live. 0 disables the gate.
    pub region_thickness: f32,
    /// Octree depth — derived sim-side as
    /// `ceil(log2(extent / (cell_size * BRICK_DIM)))` and clamped to
    /// the shader's `@max_depth` cap (default 8).
    pub max_depth: u32,
    /// Painted-leaf count from the host scan that produced this region.
    /// Drives per-region pool sizing — more painted leaves means a
    /// larger surface area and more sparse-octree expansion. 0 falls
    /// back to a small floor so test/free-standing regions still get a
    /// usable reservation.
    pub painted_leaf_count: u32,
    /// V10 tile coordinate. For shaders with `@tile_size`, this is
    /// the host-local tile index `floor(painted_leaf_pos / tile_size)`.
    /// For shaders without tiling, set to `NO_TILE` (sentinel).
    /// Folded into the cache key so two tiles on the same
    /// (object, material) get distinct cache entries + pool slices.
    pub tile_index: [i32; 3],
    /// Host octree info for `host_sample_at(world_pos)` queries from
    /// inside the user shader. `host_octree_root == 0xFFFFFFFF` means
    /// "no host" (region is free-standing); `host_sample_at` returns
    /// `(+inf, +Y)` in that case.
    pub host_octree_root: u32,
    pub host_octree_depth: u32,
    pub host_octree_extent: f32,
    pub host_grid_origin: [f32; 3],
    pub host_inverse_world: [[f32; 4]; 4],
    /// Phase B-redux 3b — `true` when the BFS should bake band cells
    /// (with `instance_at` derivation hook) instead of voxel bricks.
    /// Routed by sim from `UserShaderInfo.has_instance_at`. Mutually
    /// exclusive with the voxel-emit path within one region; a shader
    /// that has both `generate` and `instance_at` is rejected by the
    /// composer.
    pub is_band_region: bool,
    /// Phase B-redux band-cell anchor projection target. World-space
    /// y of the painted surface; the BFS uses this directly as the
    /// anchor's y when `is_band_region == true`. Computed CPU-side
    /// from the painted leaves' world-space y. Flat-surface only;
    /// sloped/curved hosts need a more expressive scheme (per-cell
    /// projection or multi-source BFS).
    pub host_surface_y: f32,
    /// Per-instance paint overlay slice (mirrors the host
    /// `RkpGpuInstance`'s fields). The BFS's host-material probe
    /// consults this so it sees painted material — without it, the
    /// probe falls back on the asset's baseline material and
    /// rejects every painted anchor.
    pub host_overlay_offset: u32,
    pub host_overlay_count: u32,
}

// ============================================================
// Cache lookup result
// ============================================================

/// Slot descriptor returned from `lookup_or_allocate`. Carries the
/// per-region state the host needs to populate its `RegionUniform`
/// upload, plus dirty bits indicating whether classify, fill, or
/// neither needs to dispatch this frame.
#[derive(Debug, Clone, Copy)]
pub struct CachedSlot {
    /// Region index in this frame's dispatch arrays. Populated by the
    /// caller after gathering all dirty slots — `lookup_or_allocate`
    /// returns 0 here; the caller assigns sequential indices and
    /// updates the underlying entry.
    pub region_index: u32,
    /// Global pool offset where this region's octree root lives.
    pub octree_root: u32,
    /// Per-pool block offsets (absolute, ready for the GPU) and sizes.
    pub octree_block_offset: u32,
    pub octree_block_size: u32,
    pub brick_block_offset: u32,
    pub brick_block_size: u32,
    pub leaf_attr_block_offset: u32,
    pub leaf_attr_block_size: u32,
    /// Fill-task pool offset is in FillTask units; relative to the
    /// fill-task pool buffer (no separate "base" — the pool is
    /// owned entirely by the user-shader pass).
    pub fill_task_block_offset: u32,
    pub fill_task_block_size: u32,
    pub object_id: u32,
    pub max_depth: u32,
    /// `true` when topology inputs differ from the cached values —
    /// classify must re-run.
    pub topology_dirty: bool,
    /// `true` when fill inputs differ. Always `true` when
    /// `topology_dirty` is.
    pub fill_dirty: bool,
}

