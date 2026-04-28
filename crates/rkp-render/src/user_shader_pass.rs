//! Phase C — GPU runtime geometry from user shaders.
//!
//! Owns the geometry-build compute pipeline that materializes voxels by
//! calling each registered shader's `user_<name>_generate` hook over a
//! requested AABB. The Rust side caches the produced octree slices so a
//! region with stable inputs (shader source + params + host geometry)
//! reuses last frame's GPU writes; `@animated` shaders opt out and
//! regenerate every frame.
//!
//! ## Pool layout
//!
//! Phase C carves transient slices out of the *tail* of the existing
//! `octree_nodes`, `leaf_attr_pool`, and `brick_pool` buffers — the
//! march and shade passes don't need a parallel pool because the
//! transient `RkpGpuObject`s point into the same buffers as bake-built
//! objects. The CPU-managed head is unchanged; the geom pipeline writes
//! the tail via a dedicated `read_write` bind group while the march
//! continues to read the same buffers as `read_only`.
//!
//! ## Compose contract
//!
//! `compose_geom_source` splices the composer's `generate` chunk
//! between the `// USER_GENERATE_DISPATCH_BEGIN/_END` markers in
//! `user_shader_geom.wgsl`. Empty chunk → in-tree identity stub keeps
//! validating; the dispatch loop runs but produces no voxels.

use std::collections::HashMap;

use crate::rkp_gpu_object::{geom_type, RkpGpuObject};
use crate::shader_composer::UserShaderInfo;
use crate::validate_wgsl;

/// One materialization request from sim → render. Stable across frames
/// for cache hit; rebuilt by sim each tick from the ECS scan or
/// (in V1) explicit registrations from the engine.
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
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    /// Voxel size at the deepest level. From the shader's `@cell_size`
    /// directive or a host-derived default (sim is responsible for the
    /// fallback choice).
    pub cell_size: f32,
    /// Folded with shader source hash + host geometry epoch into the
    /// cache key. Bumped by sim whenever any input the cache should
    /// invalidate on changes (host transform, host geometry, param
    /// edit). For `@animated` shaders the cache miss is forced via
    /// `animated = true` and this hash is irrelevant.
    pub input_hash: u64,
    /// Mirrors the shader's `@animated` directive — sim resolves this
    /// from the registry once and ships with the request to avoid
    /// chasing the registry on render. `true` regenerates every frame
    /// without cache lookup.
    pub animated: bool,
    /// Mirrors `@region_thickness` — used by the geom pipeline's
    /// brick-level proximity gate. 0 = no gate.
    pub region_thickness: f32,
    /// Octree depth N for this region (from `@octree_depth` directive
    /// or the engine default). 0 = single-brick root, 2 = 16 cells/axis,
    /// up to 6 (capped). Per-region so different shaders can run at
    /// different resolutions in the same frame.
    pub octree_depth: u32,
    /// Host octree info for `host_sample_at(world_pos)` queries from
    /// inside the user shader. `host_octree_root == 0xFFFFFFFF` means
    /// "no host" (region is free-standing); `host_sample_at` returns
    /// the V1 stub in that case (distance=0, normal=+Y).
    pub host_octree_root: u32,
    pub host_octree_depth: u32,
    pub host_octree_extent: f32,
    pub host_grid_origin: [f32; 3],
    pub host_inverse_world: [[f32; 4]; 4],
}

/// Contents of one cache entry. The slice fields point into the
/// transient tail of the scene's pool buffers — the geom pipeline
/// writes there each time the entry is (re)baked.
#[derive(Debug, Clone)]
struct CacheEntry {
    /// Hash that produced this entry's contents. Compare against the
    /// request's effective hash on lookup; mismatch → re-bake.
    content_hash: u64,
    /// Region index assigned at allocation time. Used as the index
    /// into the `leaf_attr_alloc` atomic counter buffer so each
    /// region's allocations stay confined to its capacity.
    region_index: u32,
    /// Octree slice. Length = `(8^(depth+1) - 1) / 7` nodes — a
    /// perfect tree's internal levels plus the level-N brick
    /// pointers. The transient `RkpGpuObject` references
    /// `octree_offset` as the root.
    octree_offset: u32,
    /// Brick slice — 8^depth bricks of `BRICK_CELLS` u32 cells each.
    /// `brick_offset` is in u32 entries, not bytes.
    brick_offset: u32,
    /// LeafAttr slice — capacity sized to a configurable fraction of
    /// the worst-case `8^depth * 64`; the atomic dispenser refuses
    /// excess.
    leaf_attr_offset: u32,
    leaf_attr_capacity: u32,
    /// V5 — sparse brick reservation. Stores how many brick slots
    /// this region's atomic counter is allowed to claim. Overflow
    /// → brick is skipped (OCTREE_EMPTY).
    brick_capacity: u32,
    /// Octree depth — also drives the dispatch shape (`(2^depth)³`
    /// workgroups per region).
    depth: u32,
    /// AABB of the region — used by sim's tile-cull and copied into
    /// the transient `RkpGpuObject` each frame.
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
    cell_size: f32,
    /// Whether the corresponding shader is `@animated` — animated
    /// entries skip the hash check at upload time but otherwise share
    /// the cache plumbing.
    animated: bool,
    /// Stable object_id used by the transient `RkpGpuObject`. Allocated
    /// once at entry creation so tile-list keying stays stable across
    /// frames for the same logical region.
    object_id: u32,
}

/// Cache + transient pool sub-allocator for user-shader-generated geometry.
///
/// Owns nothing GPU-side itself; instead it manages offsets into the
/// reserved tail of `RkpScene`'s pool buffers (sized at startup via
/// `RkpScene::ensure_user_shader_capacity`). Eviction is naive in V1:
/// LRU replaced by "evict on hash mismatch", followed by full-flush
/// when the geometry epoch bumps (host data changed → all transient
/// data was potentially derived from stale state).
pub struct UserShaderObjectCache {
    entries: HashMap<(u32, u32), CacheEntry>,
    /// Free list of (octree, brick, leaf_attr, region_index) tuples
    /// returned by evicted entries. Allocations prefer the free list
    /// over bumping the high-water mark.
    free_slots: Vec<FreeSlot>,
    /// Bump-allocator high-water marks within the reserved tails.
    /// `*_offset` values returned from `allocate_slot` are absolute
    /// indices into the corresponding scene buffer (i.e. they include
    /// the CPU-managed head's length).
    octree_high_water: u32,
    brick_high_water: u32,
    leaf_attr_high_water: u32,
    region_index_high_water: u32,
    /// Capacity bounds — set by `set_pool_bases` before first
    /// allocation. Allocations beyond these refuse and the entry is
    /// dropped (non-fatal — the requesting region just doesn't render
    /// that frame; sim can lower fidelity or skip).
    octree_base: u32,
    octree_capacity: u32,
    brick_base: u32,
    brick_capacity: u32,
    leaf_attr_base: u32,
    leaf_attr_capacity: u32,
    /// Geometry epoch the cache was last reconciled against. When sim
    /// reports a higher epoch the cache flushes — host geometry has
    /// shifted out from under any host-relative shader.
    last_seen_geometry_epoch: u64,
    /// Per-region atomic counter capacity. Constant in V1 — every
    /// region gets `BRICK_CELLS` (64) leaf_attr slots, mirroring the
    /// V1 single-brick geometry shape.
    next_object_id: u32,
}

#[derive(Debug, Clone, Copy)]
struct FreeSlot {
    octree_offset: u32,
    /// Carried alongside `octree_offset` so a future eviction-aware
    /// allocator can pop a free slot, validate its size, and reuse the
    /// exact range. V1 always returns 1-node slots so the field
    /// reads identical for every slot today; kept on the struct to
    /// avoid a CacheEntry-shape diff when eviction lands.
    #[allow(dead_code)]
    octree_capacity: u32,
    brick_offset: u32,
    #[allow(dead_code)]
    brick_capacity: u32,
    leaf_attr_offset: u32,
    leaf_attr_capacity: u32,
    region_index: u32,
}

/// Object_id range reserved for transient user-shader regions. Far
/// above any reasonable persistent-entity count so picks/cull lists
/// can't collide. Shifted up to the top of the u32 range so casts to
/// `i32` still leave plenty of headroom.
const USER_SHADER_OBJECT_ID_BASE: u32 = 0xF000_0000;

/// Cells per brick — must match `rkp_core::brick_pool::BRICK_CELLS`.
/// Constant in this codebase (4³ bricks); referenced in pool sizing
/// and Morton-index math.
pub const BRICK_CELLS: u32 = 64;

/// Number of bricks at the deepest level of a perfect octree of the
/// given depth. `bricks_per_region(0)` = 1 (the root IS a brick);
/// `bricks_per_region(2)` = 64.
pub fn bricks_per_region(depth: u32) -> u32 {
    1u32 << (depth * 3)
}

/// Total octree node count for a perfect tree of depth N — internals
/// at levels 0..N-1 plus brick-leaf nodes at level N. Equals
/// `(8^(N+1) - 1) / 7`.
pub fn octree_node_count(depth: u32) -> u32 {
    (pow8(depth + 1) - 1) / 7
}

/// Offset of level N's first node within the region's octree slice —
/// equals the count of internal nodes at levels 0..N-1, which is
/// `(8^N - 1) / 7`.
pub fn level_n_start(depth: u32) -> u32 {
    (pow8(depth) - 1) / 7
}

fn pow8(n: u32) -> u32 {
    1u32 << (n * 3)
}

/// Pre-compute the perfect-tree internal nodes (levels 0..depth-1)
/// for a region rooted at `octree_offset`. Each output u32 pair
/// (interleaved with `INTERNAL_ATTR_NONE`) is one octree node ready
/// for `queue.write_buffer` into the `octree_nodes_buffer` at byte
/// offset `octree_offset * 8`. Empty for `depth == 0` (root is a
/// brick, written by the GPU pass).
pub fn build_internal_nodes(octree_offset: u32, depth: u32) -> Vec<u32> {
    if depth == 0 {
        return Vec::new();
    }
    let total_internals = level_n_start(depth) as usize;
    let mut out: Vec<u32> = Vec::with_capacity(total_internals * 2);
    for level in 0..depth {
        let node_count_at_level = pow8(level);
        let next_level_start = level_n_start(level + 1);
        for p in 0..node_count_at_level {
            // Branch value: absolute offset of the first child within
            // octree_nodes. Children of node `p` at level L are 8
            // consecutive nodes starting at level L+1's offset + p*8.
            let child_offset = octree_offset + next_level_start + p * 8;
            out.push(child_offset);
            out.push(0xFFFFFFFFu32); // INTERNAL_ATTR_NONE
        }
    }
    out
}

impl UserShaderObjectCache {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            free_slots: Vec::new(),
            octree_high_water: 0,
            brick_high_water: 0,
            leaf_attr_high_water: 0,
            region_index_high_water: 0,
            octree_base: 0,
            octree_capacity: 0,
            brick_base: 0,
            brick_capacity: 0,
            leaf_attr_base: 0,
            leaf_attr_capacity: 0,
            last_seen_geometry_epoch: 0,
            next_object_id: USER_SHADER_OBJECT_ID_BASE,
        }
    }

    /// Reset the cache and reconfigure pool bases. Called by the
    /// render thread after `RkpScene` reallocates the underlying pool
    /// buffers (the previous transient writes are gone; rebake on
    /// next frame).
    pub fn set_pool_bases(
        &mut self,
        octree_base: u32, octree_capacity: u32,
        brick_base: u32, brick_capacity: u32,
        leaf_attr_base: u32, leaf_attr_capacity: u32,
    ) {
        self.entries.clear();
        self.free_slots.clear();
        self.octree_high_water = 0;
        self.brick_high_water = 0;
        self.leaf_attr_high_water = 0;
        self.region_index_high_water = 0;
        self.octree_base = octree_base;
        self.octree_capacity = octree_capacity;
        self.brick_base = brick_base;
        self.brick_capacity = brick_capacity;
        self.leaf_attr_base = leaf_attr_base;
        self.leaf_attr_capacity = leaf_attr_capacity;
    }

    /// Flush all entries — used when sim reports a higher geometry
    /// epoch (host data changed). Pool capacity bounds stay; only the
    /// allocations reset.
    pub fn flush(&mut self) {
        self.entries.clear();
        self.free_slots.clear();
        self.octree_high_water = 0;
        self.brick_high_water = 0;
        self.leaf_attr_high_water = 0;
        self.region_index_high_water = 0;
    }

    /// Resync against sim's current geometry epoch. Returns `true` if
    /// a flush happened. Caller should re-bake every region this
    /// frame; the GPU has stale or missing data for everything.
    pub fn reconcile_epoch(&mut self, geometry_epoch: u64) -> bool {
        if geometry_epoch <= self.last_seen_geometry_epoch {
            return false;
        }
        self.last_seen_geometry_epoch = geometry_epoch;
        if !self.entries.is_empty() || self.octree_high_water > 0 {
            self.flush();
            return true;
        }
        false
    }

    /// Look up or allocate a cache slot for `(host_object_id, material_id)`.
    /// `depth` selects the octree shape and pool capacities:
    ///   octree   = `(8^(depth+1) - 1) / 7` nodes
    ///   bricks   = `8^depth` × `BRICK_CELLS`
    ///   leaf-attrs = `8^depth` × `BRICK_CELLS / 2` (sparsity assumption;
    ///                geom shader refuses cells past this).
    /// Returns `was_dirty=true` for fresh allocations and
    /// hash-mismatched cache hits; `fresh=true` only for fresh
    /// allocations (caller writes the perfect-tree internals).
    /// Animated shaders always set `was_dirty=true` but `fresh=false`
    /// after the first allocation.
    pub fn lookup_or_allocate(
        &mut self,
        request: &ShaderRegionRequest,
        effective_hash: u64,
        depth: u32,
    ) -> Option<CachedSlot> {
        let key = (request.host_object_id, request.material_id);
        if let Some(entry) = self.entries.get_mut(&key) {
            // Depth change forces re-allocation — the slot's pool
            // capacities don't match the new shape.
            if entry.depth == depth {
                let dirty = entry.animated || entry.content_hash != effective_hash;
                entry.aabb_min = request.aabb_min;
                entry.aabb_max = request.aabb_max;
                entry.cell_size = request.cell_size;
                entry.animated = request.animated;
                if dirty {
                    entry.content_hash = effective_hash;
                }
                return Some(CachedSlot {
                    octree_offset: entry.octree_offset,
                    brick_offset: entry.brick_offset,
                    leaf_attr_offset: entry.leaf_attr_offset,
                    leaf_attr_capacity: entry.leaf_attr_capacity,
                    brick_capacity: entry.brick_capacity,
                    region_index: entry.region_index,
                    object_id: entry.object_id,
                    depth: entry.depth,
                    was_dirty: dirty,
                    fresh: false,
                });
            }
            // Depth changed — drop the old slot, fall through to
            // allocation.
            self.entries.remove(&key);
        }

        let oct_per_region = octree_node_count(depth);
        // V8b — reserve FULL bricks_per_region. With per-cell material
        // checks (the down-walk in user shaders), bricks over un-painted
        // host areas claim slots without emitting grass; a fractional
        // reserve causes deterministic-but-spatially-asymmetric grass
        // dropouts on multi-stroke paint patterns. Memory cost: 1 MB
        // per region at depth=4 with MAX_REGIONS=256 → ~256 MB worst
        // case. Lowering MAX_REGIONS or adding a brick-level material
        // gate is the path back to lower memory.
        let bricks_dense = bricks_per_region(depth);
        // FULL reserve, but clamped to the dense max (depth=0 has
        // bricks_dense=1, so the BRICK_CELLS floor would over-allocate).
        let bricks_reserved = bricks_dense;
        let brick_cells_per_region = bricks_reserved * BRICK_CELLS;
        // Leaf-attrs scale with bricks_reserved at ~50 % cell occupancy.
        let leaf_per_region =
            (bricks_reserved as u64 * BRICK_CELLS as u64 / 2).max(BRICK_CELLS as u64) as u32;

        let slot = if let Some(slot) = self.free_slots.pop() {
            slot
        } else {
            let oct = self.octree_high_water;
            let br = self.brick_high_water;
            let la = self.leaf_attr_high_water;
            let ri = self.region_index_high_water;
            if oct + oct_per_region > self.octree_capacity
                || br + brick_cells_per_region > self.brick_capacity
                || la + leaf_per_region > self.leaf_attr_capacity
            {
                eprintln!(
                    "[user_shader_pass] pool exhausted at depth={depth}: oct={}/{} brick_cells={}/{} leaf={}/{} — dropping region {}.{}",
                    oct + oct_per_region, self.octree_capacity,
                    br + brick_cells_per_region, self.brick_capacity,
                    la + leaf_per_region, self.leaf_attr_capacity,
                    request.host_object_id, request.material_id,
                );
                return None;
            }
            self.octree_high_water = oct + oct_per_region;
            self.brick_high_water = br + brick_cells_per_region;
            self.leaf_attr_high_water = la + leaf_per_region;
            self.region_index_high_water = ri + 1;
            FreeSlot {
                octree_offset: self.octree_base + oct,
                octree_capacity: oct_per_region,
                brick_offset: self.brick_base + br,
                brick_capacity: brick_cells_per_region,
                leaf_attr_offset: self.leaf_attr_base + la,
                leaf_attr_capacity: leaf_per_region,
                region_index: ri,
            }
        };

        let object_id = self.next_object_id;
        self.next_object_id = self.next_object_id.wrapping_add(1);

        let entry = CacheEntry {
            content_hash: effective_hash,
            region_index: slot.region_index,
            octree_offset: slot.octree_offset,
            brick_offset: slot.brick_offset,
            leaf_attr_offset: slot.leaf_attr_offset,
            leaf_attr_capacity: slot.leaf_attr_capacity,
            brick_capacity: bricks_reserved,
            depth,
            aabb_min: request.aabb_min,
            aabb_max: request.aabb_max,
            cell_size: request.cell_size,
            animated: request.animated,
            object_id,
        };
        let result = CachedSlot {
            octree_offset: entry.octree_offset,
            brick_offset: entry.brick_offset,
            leaf_attr_offset: entry.leaf_attr_offset,
            leaf_attr_capacity: entry.leaf_attr_capacity,
            brick_capacity: entry.brick_capacity,
            region_index: entry.region_index,
            object_id: entry.object_id,
            depth: entry.depth,
            was_dirty: true,
            fresh: true,
        };
        self.entries.insert(key, entry);
        Some(result)
    }

    /// Iterate live cache entries and emit a transient `RkpGpuObject`
    /// for each. Sim concatenates the result with persistent objects
    /// before the per-frame `RkpScene::upload_frame`.
    pub fn build_transient_objects(&self) -> Vec<RkpGpuObject> {
        self.entries
            .values()
            .map(transient_gpu_object)
            .collect()
    }

    /// Largest region_index any live cache entry holds. Used by the
    /// dispatch path to size the per-region atomic counter buffer.
    pub fn max_region_index(&self) -> u32 {
        self.region_index_high_water.saturating_sub(1)
    }
}

impl Default for UserShaderObjectCache {
    fn default() -> Self { Self::new() }
}

fn transient_gpu_object(entry: &CacheEntry) -> RkpGpuObject {
    // Region geometry sits in world space — the user shader's
    // `generate` hook receives world positions, so the object's
    // local→world is identity and `grid_origin` is the AABB min.
    //
    // The brick grid covers `bricks_per_axis * BRICK_DIM * cell_size`
    // on each axis from `aabb_min`. The march's ray-clip uses the
    // octree's extent (cube), so we set `aabb_max` to match the brick
    // grid's actual coverage rather than carrying the host's
    // potentially non-cubic AABB through. Otherwise ray entry/exit
    // would not align with the bricks the geom shader populated.
    let bricks_per_axis = (1u32 << entry.depth) as f32;
    let extent = entry.cell_size * 4.0 * bricks_per_axis;
    let aabb_max_cube = [
        entry.aabb_min[0] + extent,
        entry.aabb_min[1] + extent,
        entry.aabb_min[2] + extent,
    ];
    let identity: [[f32; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    RkpGpuObject {
        world: identity,
        aabb_min: entry.aabb_min,
        octree_root: entry.octree_offset,
        aabb_max: aabb_max_cube,
        octree_depth: entry.depth,
        octree_extent_bits: extent.to_bits(),
        voxel_size: entry.cell_size,
        material_id: 0,
        object_id: entry.object_id,
        geom_type: geom_type::VOXELIZED,
        is_skinned: 0,
        bone_count: 0,
        bone_buffer_offset: 0,
        rest_octree_root: 0,
        rest_octree_depth: 0,
        rest_octree_extent_bits: 0,
        bone_field_offset: 0,
        layer_mask: u32::MAX,
        bone_field_dim_x: 0,
        bone_field_dim_y: 0,
        bone_field_dim_z: 0,
        bone_field_origin_x: 0.0,
        bone_field_origin_y: 0.0,
        bone_field_origin_z: 0.0,
        bone_field_occ_offset: 0,
        grid_origin: entry.aabb_min,
        _post_grid_pad: 0,
        inverse_world: identity,
    }
}

/// Slot descriptor returned from cache lookup.
#[derive(Debug, Clone, Copy)]
pub struct CachedSlot {
    pub octree_offset: u32,
    pub brick_offset: u32,
    pub leaf_attr_offset: u32,
    pub leaf_attr_capacity: u32,
    pub brick_capacity: u32,
    pub region_index: u32,
    pub object_id: u32,
    pub depth: u32,
    /// `true` iff the GPU contents need to be (re)written this frame.
    /// Animated shaders always return `true`.
    pub was_dirty: bool,
    /// `true` iff this slot is freshly-allocated and the CPU must
    /// write its perfect-tree internal nodes (levels 0..depth-1)
    /// before the next dispatch reads. Cached slots already have
    /// their internals on the GPU and don't need a re-write unless
    /// the underlying pool buffer was reallocated (handled by
    /// `flush()` invalidating all entries).
    pub fresh: bool,
}

/// One region's per-dispatch uniform — mirrors the WGSL `RegionUniform`.
/// 240 bytes (multiple of 16, satisfies WGSL uniform alignment).
/// V3 adds `host_*` fields so the geom shader can descend the host
/// object's octree to answer `host_sample_at(world_pos)`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RegionUniform {
    pub aabb_min: [f32; 3],
    pub cell_size: f32,
    pub aabb_max: [f32; 3],
    pub shader_id: u32,
    pub octree_offset: u32,
    pub brick_offset: u32,
    pub leaf_attr_offset: u32,
    pub leaf_attr_capacity: u32,
    pub brick_capacity: u32,
    pub _pad_brick_cap: u32,
    pub time: f32,
    /// Host material id the region was triggered by. Surfaces through
    /// `UserCtx.material_id` so user code can `v.material_primary =
    /// ctx.material_id` and pick up the host's PBR / color without
    /// hardcoding a slot. u32 to satisfy WGSL alignment; valid range
    /// is 0..=65535 (matches the underlying u16 material slot id).
    pub material_id: u32,
    /// Per-region atomic-counter index for leaf-attr allocation. The
    /// geom shader does `atomicAdd(&leaf_attr_alloc[region_index], 1u)`
    /// to dispense slots; counters across regions don't collide.
    pub region_index: u32,
    /// Octree depth N. 0 = single-brick root (V1 path), 2 = 16
    /// cells/axis (V2 default), etc.
    pub depth: u32,
    /// 2^depth — pre-computed so the shader avoids a loop.
    pub bricks_per_axis: u32,
    /// (8^depth - 1) / 7 — start of level-N brick leaves within this
    /// region's octree slice.
    pub level_n_start: u32,
    /// Host's octree root offset in `octree_nodes`. `0xFFFFFFFF` =
    /// "no host" (e.g., a free-standing region with no host object);
    /// `host_sample_at` returns `(+inf, +Y)` in that case.
    pub host_octree_root: u32,
    /// Host's octree depth.
    pub host_octree_depth: u32,
    /// Host's octree world-space extent — bitcast<f32> of
    /// `RkpGpuObject.octree_extent_bits`.
    pub host_octree_extent: f32,
    /// Region thickness in world units, from the shader's
    /// `@region_thickness` directive. Used to drive the geom
    /// pipeline's brick-level early-out (skip bricks farther than
    /// this from the host surface). 0 disables the gate.
    pub region_thickness: f32,
    /// 4 padding scalars before `host_grid_origin` — WGSL `vec3<f32>`
    /// has 16-byte alignment in uniform-address-space std140 layout,
    /// so the previous f32 (`region_thickness` at offset 92) must
    /// be followed by 16 bytes of slack to land `host_grid_origin`
    /// at offset 112. Rust's `[f32; 3]` only needs 4-byte alignment,
    /// so this padding has to be explicit on the CPU side.
    pub _pad_thickness: f32,
    pub _pad_thickness2: f32,
    pub _pad_thickness3: f32,
    pub _pad_thickness4: f32,
    /// Host grid origin in object-local space — start of the host's
    /// voxel grid. `vec3<f32>` so 16-byte aligned in WGSL uniform
    /// layout; `_pad_grid` fills the slack.
    pub host_grid_origin: [f32; 3],
    pub _pad_grid: f32,
    pub params: [[f32; 4]; 2],
    /// Host's world→local transform — used to map `cell_world_pos`
    /// back into the host's octree coordinate frame. Identity when
    /// no host (matches `host_octree_root == 0xFFFFFFFF`).
    pub host_inverse_world: [[f32; 4]; 4],
}

const _: () = assert!(std::mem::size_of::<RegionUniform>() == 224);

/// Compose the geom-build WGSL with a user `generate` chunk. Empty
/// chunk leaves the in-tree identity stub in place.
pub fn compose_geom_source(user_chunk: &str) -> String {
    let geom_src = include_str!("shaders/user_shader_geom.wgsl");
    if user_chunk.is_empty() {
        return geom_src.to_string();
    }
    const BEGIN: &str = "// USER_GENERATE_DISPATCH_BEGIN";
    const END: &str = "// USER_GENERATE_DISPATCH_END";
    let begin = geom_src.find(BEGIN).expect(
        "user_shader_geom.wgsl missing USER_GENERATE_DISPATCH_BEGIN marker",
    );
    let end_after = geom_src[begin..]
        .find(END)
        .map(|off| begin + off + END.len())
        .expect("user_shader_geom.wgsl missing USER_GENERATE_DISPATCH_END marker");
    let mut out = String::with_capacity(geom_src.len() + user_chunk.len());
    out.push_str(&geom_src[..begin]);
    out.push_str(user_chunk);
    out.push_str(&geom_src[end_after..]);
    out
}

/// The GPU geom-build pipeline + the per-frame transient resources it
/// owns. One instance per render thread; viewports share the dispatch
/// (the writes happen before any per-VR pass and are read uniformly by
/// every march thereafter).
pub struct UserShaderPass {
    device_group0_layout: wgpu::BindGroupLayout,
    device_group1_layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    pipeline: wgpu::ComputePipeline,
    /// Atomic counter buffer — one `u32` per region. The geom shader
    /// allocates leaf_attr slots within each region's reserved range
    /// via `atomicAdd(leaf_attr_alloc[region_index])`. Reset to 0
    /// before every dispatch.
    leaf_attr_alloc_buffer: wgpu::Buffer,
    leaf_attr_alloc_capacity: u64,
    /// V5 — atomic counter buffer for sparse brick allocation.
    /// One `u32` per region. Geom shader's brick gate `atomicAdd`s
    /// here on pass to claim a slot from the region's reservation.
    /// Reset to 0 before every dispatch alongside `leaf_attr_alloc`.
    brick_alloc_buffer: wgpu::Buffer,
    brick_alloc_capacity: u64,
    /// Per-region uniform buffer — written once per region per frame,
    /// then bound for that region's dispatch. Sized to `MAX_REGIONS *
    /// uniform_stride`.
    region_uniforms_buffer: wgpu::Buffer,
    region_uniforms_capacity: u64,
    /// Hash of the user-shader chunk currently compiled into
    /// `pipeline`. `0` is the empty-registry sentinel.
    source_hash: u64,
    /// Cached group-0 bind group, rebuilt when scene buffers reallocate.
    group0_bind_group: Option<wgpu::BindGroup>,
    group0_buffers_epoch: u64,
}

/// Aligned stride between consecutive region uniforms in the resident
/// buffer. WGSL/wgpu requires uniforms at dynamic offsets to be aligned
/// to `min_uniform_buffer_offset_alignment` (256 bytes on most
/// devices). 256 also gives plenty of room beyond the 96-byte struct.
const REGION_UNIFORM_STRIDE: u64 = 256;

impl UserShaderPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let group0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("user_shader_geom group0"),
            entries: &[
                rw_storage(0), // octree_nodes
                rw_storage(1), // brick_pool
                rw_storage(2), // leaf_attr_pool
                rw_storage(3), // leaf_attr_alloc
                rw_storage(4), // brick_alloc (V5)
            ],
        });
        let group1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("user_shader_geom group1"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: std::num::NonZeroU64::new(
                        std::mem::size_of::<RegionUniform>() as u64,
                    ),
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("user_shader_geom pipeline layout"),
            bind_group_layouts: &[
                Some(&group0_layout),
                Some(&group1_layout),
            ],
            immediate_size: 0,
        });
        let pipeline = build_pipeline(device, &pipeline_layout, "");

        let leaf_attr_alloc_capacity: u64 = 256; // 64 u32s
        let leaf_attr_alloc_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("user_shader_geom leaf_attr_alloc"),
            size: leaf_attr_alloc_capacity,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let brick_alloc_capacity: u64 = 256; // 64 u32s; grows alongside leaf_attr_alloc
        let brick_alloc_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("user_shader_geom brick_alloc"),
            size: brick_alloc_capacity,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let region_uniforms_capacity: u64 = REGION_UNIFORM_STRIDE * 16;
        let region_uniforms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("user_shader_geom region_uniforms"),
            size: region_uniforms_capacity,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            device_group0_layout: group0_layout,
            device_group1_layout: group1_layout,
            pipeline_layout,
            pipeline,
            leaf_attr_alloc_buffer,
            leaf_attr_alloc_capacity,
            brick_alloc_buffer,
            brick_alloc_capacity,
            region_uniforms_buffer,
            region_uniforms_capacity,
            source_hash: 0,
            group0_bind_group: None,
            group0_buffers_epoch: 0,
        }
    }

    /// Rebuild the pipeline against a fresh user-shader generate chunk.
    /// Idempotent on matching `source_hash`. Returns `true` on rebuild.
    pub fn reload_user_shaders(
        &mut self,
        device: &wgpu::Device,
        user_chunk: &str,
        source_hash: u64,
    ) -> bool {
        if source_hash == self.source_hash {
            return false;
        }
        self.pipeline = build_pipeline(device, &self.pipeline_layout, user_chunk);
        self.source_hash = source_hash;
        true
    }

    pub fn source_hash(&self) -> u64 { self.source_hash }

    /// Ensure the per-region atomic counter + uniform buffers are sized
    /// for `region_count` regions.
    fn ensure_capacity(&mut self, device: &wgpu::Device, region_count: usize) {
        let needed_alloc = (region_count.max(1) * 4) as u64;
        if needed_alloc > self.leaf_attr_alloc_capacity {
            let mut cap = self.leaf_attr_alloc_capacity.max(64);
            while cap < needed_alloc { cap = cap.saturating_mul(2); }
            self.leaf_attr_alloc_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("user_shader_geom leaf_attr_alloc"),
                size: cap,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.leaf_attr_alloc_capacity = cap;
            // Group 0 references this buffer — invalidate the cached bg.
            self.group0_bind_group = None;
        }
        if needed_alloc > self.brick_alloc_capacity {
            let mut cap = self.brick_alloc_capacity.max(64);
            while cap < needed_alloc { cap = cap.saturating_mul(2); }
            self.brick_alloc_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("user_shader_geom brick_alloc"),
                size: cap,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.brick_alloc_capacity = cap;
            self.group0_bind_group = None;
        }
        let needed_uni = REGION_UNIFORM_STRIDE * region_count.max(1) as u64;
        if needed_uni > self.region_uniforms_capacity {
            let mut cap = self.region_uniforms_capacity.max(REGION_UNIFORM_STRIDE);
            while cap < needed_uni { cap = cap.saturating_mul(2); }
            self.region_uniforms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("user_shader_geom region_uniforms"),
                size: cap,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.region_uniforms_capacity = cap;
        }
    }

    /// Build the group 0 bind group (scene buffers, read_write). Cached
    /// against the scene's `buffers_epoch`; the caller passes the
    /// current epoch so we know when to rebuild.
    pub fn ensure_group0(
        &mut self,
        device: &wgpu::Device,
        octree_nodes_buffer: &wgpu::Buffer,
        brick_pool_buffer: &wgpu::Buffer,
        leaf_attr_pool_buffer: &wgpu::Buffer,
        buffers_epoch: u64,
    ) {
        if self.group0_bind_group.is_some() && buffers_epoch == self.group0_buffers_epoch {
            return;
        }
        self.group0_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("user_shader_geom group0 bg"),
            layout: &self.device_group0_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: octree_nodes_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: brick_pool_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: leaf_attr_pool_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.leaf_attr_alloc_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.brick_alloc_buffer.as_entire_binding() },
            ],
        }));
        self.group0_buffers_epoch = buffers_epoch;
    }

    /// Encode the per-region dispatches. `region_uniforms` must have
    /// the same length as the number of regions to bake this frame
    /// (dirty cache entries). `max_region_index` is the largest
    /// `region_index` value that any live cache entry holds — sizes
    /// the per-region atomic-counter buffer so the WGSL
    /// `atomicAdd(&leaf_attr_alloc[region_index], ...)` never goes
    /// out of bounds.
    pub fn dispatch_regions(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        region_uniforms: &[RegionUniform],
        max_region_index: u32,
    ) {
        if region_uniforms.is_empty() {
            return;
        }
        let counter_slots = max_region_index.saturating_add(1).max(1) as usize;
        self.ensure_capacity(device, counter_slots);
        // Zero the atomic counters — `clear_buffer` is the cheapest
        // path; only the ranges actually used this frame are read.
        encoder.clear_buffer(&self.leaf_attr_alloc_buffer, 0, None);
        encoder.clear_buffer(&self.brick_alloc_buffer, 0, None);

        // Pack uniforms into the resident buffer at REGION_UNIFORM_STRIDE
        // intervals so each dispatch can address its slot via dynamic
        // offset.
        let mut packed = vec![0u8; (REGION_UNIFORM_STRIDE as usize) * region_uniforms.len()];
        for (i, ru) in region_uniforms.iter().enumerate() {
            let off = i * REGION_UNIFORM_STRIDE as usize;
            packed[off..off + std::mem::size_of::<RegionUniform>()]
                .copy_from_slice(bytemuck::bytes_of(ru));
        }
        queue.write_buffer(&self.region_uniforms_buffer, 0, &packed);

        let group0 = match &self.group0_bind_group {
            Some(bg) => bg,
            None => {
                eprintln!("[user_shader_pass] dispatch skipped — group 0 not bound");
                return;
            }
        };
        let group1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("user_shader_geom group1 bg"),
            layout: &self.device_group1_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &self.region_uniforms_buffer,
                    offset: 0,
                    size: std::num::NonZeroU64::new(
                        std::mem::size_of::<RegionUniform>() as u64,
                    ),
                }),
            }],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("user_shader_geom"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, group0, &[]);
        for (i, ru) in region_uniforms.iter().enumerate() {
            let dynamic_offset = (i as u64 * REGION_UNIFORM_STRIDE) as u32;
            pass.set_bind_group(1, &group1, &[dynamic_offset]);
            // (2^depth)³ workgroups per region — one workgroup per
            // brick at the deepest level. Depth=0 → (1, 1, 1) (V1
            // single-brick); depth=2 → (4, 4, 4) → 64 bricks.
            let bpa = ru.bricks_per_axis;
            pass.dispatch_workgroups(bpa, bpa, bpa);
        }
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
    user_chunk: &str,
) -> wgpu::ComputePipeline {
    let source = compose_geom_source(user_chunk);
    validate_wgsl(&source, "user_shader_geom");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("user_shader_geom"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("user_shader_geom"),
        layout: Some(pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// Build the per-region uniform from a request + cache slot. Sim/render
/// builds these alongside the cache lookup; the GPU-side Rust pass
/// fires them through `dispatch_regions`.
pub fn build_region_uniform(
    request: &ShaderRegionRequest,
    slot: &CachedSlot,
    shader_id: u32,
    time_seconds: f32,
) -> RegionUniform {
    let mut params = [[0.0f32; 4]; 2];
    for (i, &v) in request.params.iter().take(8).enumerate() {
        params[i / 4][i % 4] = v;
    }
    RegionUniform {
        aabb_min: request.aabb_min,
        cell_size: request.cell_size,
        aabb_max: request.aabb_max,
        shader_id,
        octree_offset: slot.octree_offset,
        brick_offset: slot.brick_offset,
        leaf_attr_offset: slot.leaf_attr_offset,
        leaf_attr_capacity: slot.leaf_attr_capacity,
        brick_capacity: slot.brick_capacity,
        _pad_brick_cap: 0,
        time: time_seconds,
        material_id: request.material_id,
        region_index: slot.region_index,
        depth: slot.depth,
        bricks_per_axis: 1u32 << slot.depth,
        level_n_start: level_n_start(slot.depth),
        host_octree_root: request.host_octree_root,
        host_octree_depth: request.host_octree_depth,
        host_octree_extent: request.host_octree_extent,
        region_thickness: request.region_thickness,
        _pad_thickness: 0.0,
        _pad_thickness2: 0.0,
        _pad_thickness3: 0.0,
        _pad_thickness4: 0.0,
        host_grid_origin: request.host_grid_origin,
        _pad_grid: 0.0,
        params,
        host_inverse_world: request.host_inverse_world,
    }
}

/// Compose the effective hash for a request given the registry's
/// `source_hash` and the host's `geometry_epoch`. Sim provides
/// `request.input_hash` already folded with anything else specific to
/// the call site (transform bytes, leaf-material set hash, etc.).
pub fn effective_hash(
    request: &ShaderRegionRequest,
    registry_source_hash: u64,
    geometry_epoch: u64,
) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    let prime = 0x100000001b3u64;
    let mix = |h: &mut u64, b: u8| {
        *h ^= b as u64;
        *h = h.wrapping_mul(prime);
    };
    for &b in &registry_source_hash.to_le_bytes() { mix(&mut h, b); }
    for &b in &geometry_epoch.to_le_bytes() { mix(&mut h, b); }
    for &b in &request.input_hash.to_le_bytes() { mix(&mut h, b); }
    for &p in &request.params {
        for &b in &p.to_le_bytes() { mix(&mut h, b); }
    }
    for &b in &request.cell_size.to_le_bytes() { mix(&mut h, b); }
    for &v in request.aabb_min.iter().chain(request.aabb_max.iter()) {
        for &b in &v.to_le_bytes() { mix(&mut h, b); }
    }
    h
}

/// Resolve a `shader_name` to the registry's `shader_id` via a slice of
/// `UserShaderInfo`s. `0` = identity / unregistered.
pub fn resolve_shader_id(infos: &[UserShaderInfo], name: &str) -> u32 {
    if name.is_empty() {
        return 0;
    }
    // Registry assigns ids alphabetically starting at 1 — replicate
    // here so render doesn't need the full registry on its thread.
    let mut sorted: Vec<&UserShaderInfo> = infos.iter().collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));
    for (i, info) in sorted.iter().enumerate() {
        if info.name == name {
            return (i as u32) + 1;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geom_shader_validates_with_empty_chunk() {
        let src = compose_geom_source("");
        let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| {
            panic!("parse error:\n{}", e.emit_to_string(&src))
        });
        let mut v = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        v.validate(&module).unwrap_or_else(|e| panic!("validation error: {e:?}"));
    }

    #[test]
    fn compose_splices_user_chunk() {
        let chunk = "fn dispatch_user_generate(shader_id: u32, cell_world_pos: vec3<f32>, host: HostSample, ctx: UserCtx) -> VoxelEmit { return voxel_emit_skip(); }";
        let src = compose_geom_source(chunk);
        // The in-tree default body should be replaced.
        assert!(src.contains("dispatch_user_generate"));
        // The user chunk's marker presence implies splice happened.
        assert!(!src.contains("Default identity stub"));
    }

    #[test]
    fn cache_returns_dirty_first_time_and_clean_second() {
        let mut cache = UserShaderObjectCache::new();
        cache.set_pool_bases(0, 32, 0, 4096, 0, 4096);
        let req = ShaderRegionRequest {
            host_object_id: 1, material_id: 1,
            shader_name: "x".to_string(),
            params: vec![1.0],
            aabb_min: [0.0; 3], aabb_max: [1.0; 3],
            cell_size: 0.25,
            input_hash: 42,
            animated: false, region_thickness: 0.0, host_octree_root: 0xFFFFFFFF, host_octree_depth: 0, host_octree_extent: 0.0, host_grid_origin: [0.0; 3], host_inverse_world: [[0.0; 4]; 4], octree_depth: 0,
        };
        let h = effective_hash(&req, 7, 0);
        let s1 = cache.lookup_or_allocate(&req, h, 0).unwrap();
        assert!(s1.was_dirty);
        let s2 = cache.lookup_or_allocate(&req, h, 0).unwrap();
        assert!(!s2.was_dirty);
        assert_eq!(s1.octree_offset, s2.octree_offset);
        assert_eq!(s1.region_index, s2.region_index);
        assert_eq!(s1.object_id, s2.object_id);
    }

    #[test]
    fn cache_animated_always_dirty() {
        let mut cache = UserShaderObjectCache::new();
        cache.set_pool_bases(0, 32, 0, 4096, 0, 4096);
        let req = ShaderRegionRequest {
            host_object_id: 1, material_id: 1,
            shader_name: "x".to_string(),
            params: vec![],
            aabb_min: [0.0; 3], aabb_max: [1.0; 3],
            cell_size: 0.25,
            input_hash: 1,
            animated: true, region_thickness: 0.0, host_octree_root: 0xFFFFFFFF, host_octree_depth: 0, host_octree_extent: 0.0, host_grid_origin: [0.0; 3], host_inverse_world: [[0.0; 4]; 4], octree_depth: 0,
        };
        let h = effective_hash(&req, 1, 0);
        assert!(cache.lookup_or_allocate(&req, h, 0).unwrap().was_dirty);
        assert!(cache.lookup_or_allocate(&req, h, 0).unwrap().was_dirty);
    }

    #[test]
    fn cache_different_keys_get_different_slots() {
        let mut cache = UserShaderObjectCache::new();
        cache.set_pool_bases(0, 32, 0, 4096, 0, 4096);
        let mut req = ShaderRegionRequest {
            host_object_id: 1, material_id: 1,
            shader_name: "x".to_string(),
            params: vec![],
            aabb_min: [0.0; 3], aabb_max: [1.0; 3],
            cell_size: 0.25,
            input_hash: 0,
            animated: false, region_thickness: 0.0, host_octree_root: 0xFFFFFFFF, host_octree_depth: 0, host_octree_extent: 0.0, host_grid_origin: [0.0; 3], host_inverse_world: [[0.0; 4]; 4], octree_depth: 0,
        };
        let s1 = cache.lookup_or_allocate(&req, 0, 0).unwrap();
        req.material_id = 2;
        let s2 = cache.lookup_or_allocate(&req, 0, 0).unwrap();
        assert_ne!(s1.octree_offset, s2.octree_offset);
        assert_ne!(s1.brick_offset, s2.brick_offset);
        assert_ne!(s1.region_index, s2.region_index);
    }

    #[test]
    fn cache_flushes_on_geometry_epoch_bump() {
        let mut cache = UserShaderObjectCache::new();
        cache.set_pool_bases(0, 32, 0, 4096, 0, 4096);
        let req = ShaderRegionRequest {
            host_object_id: 1, material_id: 1,
            shader_name: "x".to_string(),
            params: vec![],
            aabb_min: [0.0; 3], aabb_max: [1.0; 3],
            cell_size: 0.25,
            input_hash: 0,
            animated: false, region_thickness: 0.0, host_octree_root: 0xFFFFFFFF, host_octree_depth: 0, host_octree_extent: 0.0, host_grid_origin: [0.0; 3], host_inverse_world: [[0.0; 4]; 4], octree_depth: 0,
        };
        cache.lookup_or_allocate(&req, 0, 0);
        assert!(cache.reconcile_epoch(1));
        assert!(cache.entries.is_empty());
    }

    #[test]
    fn pool_exhaustion_returns_none() {
        let mut cache = UserShaderObjectCache::new();
        // Capacity for one region only.
        cache.set_pool_bases(0, 1, 0, 64, 0, 64);
        let mut req = ShaderRegionRequest {
            host_object_id: 1, material_id: 1,
            shader_name: "x".to_string(),
            params: vec![],
            aabb_min: [0.0; 3], aabb_max: [1.0; 3],
            cell_size: 0.25,
            input_hash: 0,
            animated: false, region_thickness: 0.0, host_octree_root: 0xFFFFFFFF, host_octree_depth: 0, host_octree_extent: 0.0, host_grid_origin: [0.0; 3], host_inverse_world: [[0.0; 4]; 4], octree_depth: 0,
        };
        assert!(cache.lookup_or_allocate(&req, 0, 0).is_some());
        req.host_object_id = 2;
        assert!(cache.lookup_or_allocate(&req, 0, 0).is_none());
    }

    #[test]
    fn region_uniform_size_is_224() {
        assert_eq!(std::mem::size_of::<RegionUniform>(), 224);
    }

    #[test]
    fn octree_node_count_matches_perfect_tree() {
        assert_eq!(octree_node_count(0), 1);
        assert_eq!(octree_node_count(1), 9);
        assert_eq!(octree_node_count(2), 73);
        assert_eq!(octree_node_count(3), 585);
    }

    #[test]
    fn level_n_start_matches_internal_count() {
        assert_eq!(level_n_start(0), 0);
        assert_eq!(level_n_start(1), 1);
        assert_eq!(level_n_start(2), 9);
        assert_eq!(level_n_start(3), 73);
    }

    #[test]
    fn build_internal_nodes_depth0_empty() {
        assert!(build_internal_nodes(100, 0).is_empty());
    }

    #[test]
    fn build_internal_nodes_depth1_root_only() {
        // Depth-1 tree has 1 internal node (the root) at offset 0,
        // pointing to its 8 brick children starting at offset 1.
        let nodes = build_internal_nodes(100, 1);
        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[0], 101); // root branch points to first child
        assert_eq!(nodes[1], 0xFFFFFFFF); // INTERNAL_ATTR_NONE
    }

    #[test]
    fn build_internal_nodes_depth2_root_and_eight_level1() {
        // Depth-2: 1 root + 8 level-1 nodes = 9 internal nodes,
        // 18 u32s with INTERNAL_ATTR_NONE interleaved.
        let nodes = build_internal_nodes(100, 2);
        assert_eq!(nodes.len(), 9 * 2);
        // Root at slot 100 → first child at slot 101.
        assert_eq!(nodes[0], 101);
        assert_eq!(nodes[1], 0xFFFFFFFF);
        // Level-1 node 0 (slot 101) → its first child at level 2,
        // which starts at slot 100 + 9 = 109. Node p (in [0..8))
        // → first child at 109 + p*8.
        for p in 0..8u32 {
            let pair_idx = (1 + p) as usize * 2;
            let expected_value = 100 + 9 + p * 8;
            assert_eq!(nodes[pair_idx], expected_value);
            assert_eq!(nodes[pair_idx + 1], 0xFFFFFFFF);
        }
    }

    #[test]
    fn resolve_shader_id_alphabetical_one_based() {
        let infos = vec![
            UserShaderInfo { name: "zeta".into(), ..Default::default() },
            UserShaderInfo { name: "alpha".into(), ..Default::default() },
            UserShaderInfo { name: "mu".into(), ..Default::default() },
        ];
        assert_eq!(resolve_shader_id(&infos, "alpha"), 1);
        assert_eq!(resolve_shader_id(&infos, "mu"), 2);
        assert_eq!(resolve_shader_id(&infos, "zeta"), 3);
        assert_eq!(resolve_shader_id(&infos, ""), 0);
        assert_eq!(resolve_shader_id(&infos, "missing"), 0);
    }
}
