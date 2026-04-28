//! Phase C V9 — sparse BFS GPU runtime geometry from user shaders.
//!
//! Owns the GPU pipelines that build per-region transient octrees by
//! atomically allocating nodes / bricks / leaf-attrs from per-region
//! pool ranges. Memory and compute scale with painted surface area
//! rather than the (4·2^depth)³ cube the V8 dense brick model demanded.
//!
//! ## Pool layout
//!
//! Per region, the cache reserves a contiguous slice of:
//!   - `octree_nodes` — capacity sized from painted-leaf count
//!   - `brick_pool`   — same
//!   - `leaf_attr_pool` — same, scaled by `BRICK_CELLS`
//!
//! Slices live in the *tail* of the scene's existing flat pools (the
//! `RkpScene::ensure_user_shader_capacity` call grows the buffers; the
//! march reads transient writes through the same bindings as bake-built
//! geometry). The CPU-managed head is unchanged.
//!
//! Three per-region atomic counters (`octree_alloc`, `brick_alloc`,
//! `leaf_attr_alloc`) bump-allocate within those ranges as the BFS
//! expands. Overflow degrades to `OCTREE_EMPTY` at the offending node —
//! never panics, never corrupts neighbouring regions.
//!
//! ## Dispatch chain (per frame, dirty regions only)
//!
//! 1. Counters reset (clear_buffer).
//! 2. Active queue + level counts seeded from `ShaderRegionRequest`s
//!    via `queue.write_buffer` — one root cell per dirty region into
//!    `active_queue[level=0]`, `active_count[0] = dirty_region_count`.
//! 3. For L in 0..=max(max_depth across dirty regions): one
//!    `classify_main` dispatch with `level_u.current_level = L`. Threads
//!    past `active_count[L]` early-out, so we always issue
//!    `(per_level_cap / 64)` workgroups regardless of true active count
//!    (saves a build-indirect-args dispatch).
//! 4. One `brick_fill_main` dispatch over the surviving fill_queue.
//!
//! All dispatches share group-0 (scene + counter buffers) and group-1
//! (region storage array). Group-2 holds the per-dispatch level uniform
//! at a dynamic offset.
//!
//! ## Compose contract
//!
//! Identical to V8 — `compose_geom_source` splices the composer's
//! `generate` chunk between `// USER_GENERATE_DISPATCH_BEGIN/_END`
//! markers in `user_shader_geom.wgsl`. User shaders that called
//! `host_sample_at(world_pos)` keep working unchanged.

use std::collections::HashMap;

use crate::rkp_gpu_object::{geom_type, RkpGpuObject};
use crate::shader_composer::UserShaderInfo;
use crate::validate_wgsl;

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
}

/// Sentinel "no host" value matching `HOST_NO_HOST_SENTINEL` in WGSL.
pub const HOST_NO_HOST_SENTINEL: u32 = 0xFFFF_FFFFu32;

/// Sentinel `tile_index` value used for non-tiled shaders (those
/// without an `@tile_size` directive). One cache entry per
/// (object, material) pair, V9 behaviour.
pub const NO_TILE: [i32; 3] = [i32::MIN, i32::MIN, i32::MIN];

/// Cells per brick — must match `rkp_core::brick_pool::BRICK_CELLS`.
pub const BRICK_CELLS: u32 = 64;

/// Hard ceiling on octree depth. Mirrors `MAX_DEPTH` in
/// `user_shader_geom.wgsl` — the active-queue / counter buffers are
/// sized for this many levels (+1 for the root). Bumping requires
/// growing the queue buffers and the WGSL constant in lockstep.
pub const MAX_DEPTH: u32 = 8;

/// Worst-case active cells held per level in the BFS queue,
/// multiplexed across all regions processed in one frame. Multiplied
/// by `MAX_DEPTH+1` for the global queue buffer.
///
/// V10 raises this from 65K → 256K because tiled paints can emit
/// hundreds of regions, each contributing thousands of cells at the
/// deepest levels. Total queue storage:
/// `(MAX_DEPTH+1) × PER_LEVEL_QUEUE_CAP × sizeof(ActiveCell)` =
/// 9 × 256K × 32 B ≈ 72 MB. Per-level overflow degrades to
/// `OCTREE_EMPTY` at the offending parent's child slot.
const PER_LEVEL_QUEUE_CAP: u32 = 262144;

/// Total brick fill tasks the queue holds across all regions in one
/// frame. Sized for ~512 regions × 1K bricks each = 512 K fill
/// tasks worst case.
const FILL_QUEUE_CAP: u32 = 524288;

/// Workgroup size for `classify_main`. Determines how many workgroups
/// we dispatch per level: `PER_LEVEL_QUEUE_CAP / CLASSIFY_WG_SIZE`
/// regardless of real active count (per-thread early-out).
const CLASSIFY_WG_SIZE: u32 = 64;

/// Per-CPU-side state for one cache entry. Slice fields point into the
/// transient tail of the scene's pool buffers.
#[derive(Debug, Clone)]
struct CacheEntry {
    content_hash: u64,
    /// Region index in the per-region atomic-counter buffers and the
    /// `regions` storage array. Stable across frames for the same
    /// cache key until the cache is flushed.
    region_index: u32,
    /// Pool slices (element offsets into the scene buffers).
    octree_offset: u32,
    octree_capacity: u32,
    brick_offset: u32,
    brick_capacity: u32,
    leaf_attr_offset: u32,
    leaf_attr_capacity: u32,
    max_depth: u32,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
    cell_size: f32,
    animated: bool,
    /// Stable object_id used by the transient `RkpGpuObject` so tile
    /// lists key consistently across frames.
    object_id: u32,
    /// V10 — set to `true` by `lookup_or_allocate` each frame the
    /// entry is referenced. `evict_untouched` after the per-frame
    /// region pass drops entries that didn't get a request this
    /// frame (paint moved off this tile, host removed, etc.) and
    /// returns their pool slices to the free list.
    touched_this_frame: bool,
}

/// Per-region pool sub-allocator + cache for user-shader-generated
/// geometry. See module docstring for layout & lifecycle.
///
/// Cache key is `(host_object_id, material_id, tile_index)`. For
/// non-tiled shaders `tile_index = NO_TILE` and there's at most one
/// entry per (object, material). For tiled shaders many entries can
/// coexist for the same (object, material) — one per tile that
/// currently has paint.
pub struct UserShaderObjectCache {
    entries: HashMap<(u32, u32, [i32; 3]), CacheEntry>,
    free_slots: Vec<FreeSlot>,
    octree_high_water: u32,
    brick_high_water: u32,
    leaf_attr_high_water: u32,
    region_index_high_water: u32,
    octree_base: u32,
    octree_capacity: u32,
    brick_base: u32,
    brick_capacity: u32,
    leaf_attr_base: u32,
    leaf_attr_capacity: u32,
    last_seen_geometry_epoch: u64,
    next_object_id: u32,
}

#[derive(Debug, Clone, Copy)]
struct FreeSlot {
    octree_offset: u32,
    octree_capacity: u32,
    brick_offset: u32,
    brick_capacity: u32,
    leaf_attr_offset: u32,
    leaf_attr_capacity: u32,
    region_index: u32,
}

const USER_SHADER_OBJECT_ID_BASE: u32 = 0xF000_0000;

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

    pub fn set_pool_bases(
        &mut self,
        octree_base: u32, octree_capacity: u32,
        brick_base: u32, brick_capacity: u32,
        leaf_attr_base: u32, leaf_attr_capacity: u32,
    ) {
        // Idempotent — only flush + reset if the pool layout actually
        // changed (CPU head moved, transient reservation grew). With
        // the V10 stable worst-case reservation, this should be a
        // no-op every frame after the first paint, and the cache
        // (including all baked tile entries) survives across frames.
        if self.octree_base == octree_base
            && self.octree_capacity == octree_capacity
            && self.brick_base == brick_base
            && self.brick_capacity == brick_capacity
            && self.leaf_attr_base == leaf_attr_base
            && self.leaf_attr_capacity == leaf_attr_capacity
        {
            return;
        }
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

    pub fn flush(&mut self) {
        self.entries.clear();
        self.free_slots.clear();
        self.octree_high_water = 0;
        self.brick_high_water = 0;
        self.leaf_attr_high_water = 0;
        self.region_index_high_water = 0;
    }

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

    /// Look up or allocate a cache slot. Returns `None` on pool
    /// exhaustion — caller drops the request for this frame.
    /// `was_dirty=true` means the GPU contents need (re)writing this
    /// frame; `fresh=true` means the slot was just allocated (today
    /// these are equivalent — kept for symmetry with V8).
    pub fn lookup_or_allocate(
        &mut self,
        request: &ShaderRegionRequest,
        effective_hash: u64,
    ) -> Option<CachedSlot> {
        let key = (request.host_object_id, request.material_id, request.tile_index);
        let estimate = estimate_region_pool(request.painted_leaf_count, request.max_depth);

        if let Some(entry) = self.entries.get_mut(&key) {
            // If the request's pool needs match the existing entry's
            // capacity AND the depth is unchanged, reuse the slot.
            // Otherwise (depth bump, painted area grew significantly),
            // drop and re-allocate so the new estimate fits.
            let capacity_ok = entry.octree_capacity >= estimate.octree
                && entry.brick_capacity >= estimate.bricks
                && entry.leaf_attr_capacity >= estimate.leaf_attrs;
            if entry.max_depth == request.max_depth && capacity_ok {
                let dirty = entry.animated || entry.content_hash != effective_hash;
                entry.aabb_min = request.aabb_min;
                entry.aabb_max = request.aabb_max;
                entry.cell_size = request.cell_size;
                entry.animated = request.animated;
                entry.touched_this_frame = true;
                if dirty {
                    entry.content_hash = effective_hash;
                }
                return Some(CachedSlot {
                    octree_offset: entry.octree_offset,
                    octree_capacity: entry.octree_capacity,
                    brick_offset: entry.brick_offset,
                    brick_capacity: entry.brick_capacity,
                    leaf_attr_offset: entry.leaf_attr_offset,
                    leaf_attr_capacity: entry.leaf_attr_capacity,
                    region_index: entry.region_index,
                    object_id: entry.object_id,
                    max_depth: entry.max_depth,
                    was_dirty: dirty,
                    fresh: false,
                });
            }
            // Stale slot — drop and re-alloc below. The freed range goes
            // back to free_slots for the next allocation that fits.
            let freed = FreeSlot {
                octree_offset: entry.octree_offset,
                octree_capacity: entry.octree_capacity,
                brick_offset: entry.brick_offset,
                brick_capacity: entry.brick_capacity,
                leaf_attr_offset: entry.leaf_attr_offset,
                leaf_attr_capacity: entry.leaf_attr_capacity,
                region_index: entry.region_index,
            };
            self.free_slots.push(freed);
            self.entries.remove(&key);
        }

        // Try a free slot that fits before bumping the high-water mark.
        let mut slot_opt: Option<FreeSlot> = self
            .free_slots
            .iter()
            .position(|s| s.octree_capacity >= estimate.octree
                && s.brick_capacity >= estimate.bricks
                && s.leaf_attr_capacity >= estimate.leaf_attrs)
            .map(|idx| self.free_slots.swap_remove(idx));

        // Bump high-water if room.
        if slot_opt.is_none() {
            let oct = self.octree_high_water;
            let br = self.brick_high_water;
            let la = self.leaf_attr_high_water;
            let ri = self.region_index_high_water;
            if oct + estimate.octree <= self.octree_capacity
                && br + estimate.bricks * BRICK_CELLS <= self.brick_capacity
                && la + estimate.leaf_attrs <= self.leaf_attr_capacity
            {
                self.octree_high_water = oct + estimate.octree;
                self.brick_high_water = br + estimate.bricks * BRICK_CELLS;
                self.leaf_attr_high_water = la + estimate.leaf_attrs;
                self.region_index_high_water = ri + 1;
                slot_opt = Some(FreeSlot {
                    octree_offset: self.octree_base + oct,
                    octree_capacity: estimate.octree,
                    brick_offset: self.brick_base + br,
                    brick_capacity: estimate.bricks,
                    leaf_attr_offset: self.leaf_attr_base + la,
                    leaf_attr_capacity: estimate.leaf_attrs,
                    region_index: ri,
                });
            }
        }

        // Pool exhausted — try evicting an existing untouched entry
        // and reusing its slot. Untouched entries are guaranteed to
        // be `evict_untouched`-removed at end-of-frame anyway, so
        // claiming one early just runs that eviction on demand.
        // This handles the steady-state case where last frame's
        // cache filled the buffer but this frame's request set has
        // shifted — old tiles vacate, new tiles claim their slots,
        // total cache size stays bounded by MAX_REGIONS.
        if slot_opt.is_none() {
            let victim_key = self
                .entries
                .iter()
                .find(|(_, e)| !e.touched_this_frame)
                .map(|(k, _)| *k);
            if let Some(victim_key) = victim_key {
                let victim = self.entries.remove(&victim_key).unwrap();
                slot_opt = Some(FreeSlot {
                    octree_offset: victim.octree_offset,
                    octree_capacity: victim.octree_capacity,
                    brick_offset: victim.brick_offset,
                    brick_capacity: victim.brick_capacity,
                    leaf_attr_offset: victim.leaf_attr_offset,
                    leaf_attr_capacity: victim.leaf_attr_capacity,
                    region_index: victim.region_index,
                });
            }
        }

        let slot = match slot_opt {
            Some(s) => s,
            None => {
                eprintln!(
                    "[user_shader_pass] pool exhausted at max_depth={}: \
                     all {} slabs in use this frame, no untouched entry \
                     to evict — dropping region {}.{} tile {:?}",
                    request.max_depth,
                    self.region_index_high_water,
                    request.host_object_id, request.material_id,
                    request.tile_index,
                );
                return None;
            }
        };

        let object_id = self.next_object_id;
        self.next_object_id = self.next_object_id.wrapping_add(1);

        let entry = CacheEntry {
            content_hash: effective_hash,
            region_index: slot.region_index,
            octree_offset: slot.octree_offset,
            octree_capacity: slot.octree_capacity,
            brick_offset: slot.brick_offset,
            brick_capacity: slot.brick_capacity,
            leaf_attr_offset: slot.leaf_attr_offset,
            leaf_attr_capacity: slot.leaf_attr_capacity,
            max_depth: request.max_depth,
            aabb_min: request.aabb_min,
            aabb_max: request.aabb_max,
            cell_size: request.cell_size,
            animated: request.animated,
            object_id,
            touched_this_frame: true,
        };
        let result = CachedSlot {
            octree_offset: entry.octree_offset,
            octree_capacity: entry.octree_capacity,
            brick_offset: entry.brick_offset,
            brick_capacity: entry.brick_capacity,
            leaf_attr_offset: entry.leaf_attr_offset,
            leaf_attr_capacity: entry.leaf_attr_capacity,
            region_index: entry.region_index,
            object_id: entry.object_id,
            max_depth: entry.max_depth,
            was_dirty: true,
            fresh: true,
        };
        self.entries.insert(key, entry);
        Some(result)
    }

    pub fn build_transient_objects(&self) -> Vec<RkpGpuObject> {
        self.entries
            .values()
            .map(transient_gpu_object)
            .collect()
    }

    pub fn max_region_index(&self) -> u32 {
        self.region_index_high_water.saturating_sub(1)
    }

    /// Mark every entry untouched at the start of a frame. The caller
    /// then runs `lookup_or_allocate` for each ShaderRegionRequest in
    /// the frame (which marks the matched entries touched), and
    /// finishes with `evict_untouched` to drop entries that didn't
    /// get a request this frame.
    pub fn begin_frame(&mut self) {
        for entry in self.entries.values_mut() {
            entry.touched_this_frame = false;
        }
    }

    /// Drop entries not referenced this frame and return their pool
    /// slices to the free list. Required for V10 multi-region tiling
    /// — when paint moves off a tile (or a host changes its painted
    /// area), the abandoned tiles' cache entries would otherwise
    /// leak into transient objects forever and stale-render. With
    /// V9's one-entry-per-(object, material) layout this hadn't
    /// mattered; tiling makes leaks visible immediately.
    pub fn evict_untouched(&mut self) {
        let entries = std::mem::take(&mut self.entries);
        for (key, entry) in entries.into_iter() {
            if entry.touched_this_frame {
                self.entries.insert(key, entry);
            } else {
                self.free_slots.push(FreeSlot {
                    octree_offset: entry.octree_offset,
                    octree_capacity: entry.octree_capacity,
                    brick_offset: entry.brick_offset,
                    brick_capacity: entry.brick_capacity,
                    leaf_attr_offset: entry.leaf_attr_offset,
                    leaf_attr_capacity: entry.leaf_attr_capacity,
                    region_index: entry.region_index,
                });
            }
        }
    }
}

impl Default for UserShaderObjectCache {
    fn default() -> Self { Self::new() }
}

/// Per-region pool size estimate from painted-leaf count + depth.
/// Overshoot is cheap (unused slots cost only their reserved bytes,
/// no compute), undershoot drops regions or detail — so the heuristic
/// errs generous.
///
/// Sized to handle grass-style shaders with thick proximity bands
/// (`@region_thickness ~0.5m`). At max_depth=4 (V8's default depth)
/// painted_count=64 → ~1024 transient bricks, comparable to V8's
/// dense reservation but allocated only when actually needed. Sparse
/// alloc means cached idle regions retain the same reservation; the
/// frame-time cost is the BFS expansion, not the buffer footprint.
///
/// Capped per-region at `min(8^max_depth, 8192)` so a single region
/// with a huge painted count can't blow the per-frame pool budget;
/// overflow gracefully drops bricks via `OCTREE_EMPTY`.
pub struct PoolEstimate {
    pub octree: u32,
    pub bricks: u32,
    pub leaf_attrs: u32,
}

/// V10 — uniform per-region slab size. Every tile reserves the same
/// fixed chunk regardless of its actual painted-leaf count. Freed
/// slots from evicted tiles are always reusable by future
/// allocations of any size, eliminating fragmentation.
///
/// Tradeoff: tiny tiles (a single painted leaf) reserve the same
/// slab as fully-saturated tiles (~2048 bricks). Worst-case memory
/// usage equals the steady-state reservation that
/// `run_user_shader_geom` makes, so there's no extra cost relative
/// to the buffer pool we already allocate.
///
/// `painted_leaf_count` and `max_depth` are kept on the function
/// signature for forward compatibility (a future revision could
/// shrink slabs for low-demand tiles via a free-list-of-free-lists
/// — but only after we've measured fragmentation pressure).
pub fn estimate_region_pool(painted_leaf_count: u32, max_depth: u32) -> PoolEstimate {
    let _ = painted_leaf_count;
    // Bricks per region: sized for the realistic per-tile band volume
    // at @max_depth 4 + tile_size 1m + region_thickness 0.5m
    // (= 16² surface bricks × 8 band layers ≈ 2 K bricks). Shaders
    // that exceed this within a single tile (e.g. @max_depth 5 on a
    // 1m tile with a thick band) will degrade gracefully with
    // OCTREE_EMPTY on overflow bricks — bumping `@max_depth` should
    // be paired with a smaller `@tile_size` or tighter
    // `@region_thickness` to keep per-tile demand under this slab
    // size.
    const SLAB_BRICKS: u32 = 2048;
    // Octree per region: enough to hold the max possible internal +
    // brick-leaf nodes a depth-N BFS produces, plus the always-
    // existing root-to-deepest spine. SLAB_BRICKS × 8 covers the
    // brick-leaf nodes; the +overhead covers internal branches.
    let depth_overhead = (max_depth.max(1) + 1) * 8;
    PoolEstimate {
        octree: SLAB_BRICKS * 8 + depth_overhead,
        bricks: SLAB_BRICKS,
        leaf_attrs: SLAB_BRICKS * BRICK_CELLS / 2,
    }
}

fn transient_gpu_object(entry: &CacheEntry) -> RkpGpuObject {
    // Brick grid spans `cell_size * BRICK_DIM * 2^max_depth` per axis
    // from `aabb_min` — same formula as V8 since the BFS still yields a
    // (4 * 2^max_depth)³ cube at full depth.
    let bricks_per_axis = (1u32 << entry.max_depth) as f32;
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
        octree_depth: entry.max_depth,
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
    pub octree_capacity: u32,
    pub brick_offset: u32,
    pub brick_capacity: u32,
    pub leaf_attr_offset: u32,
    pub leaf_attr_capacity: u32,
    pub region_index: u32,
    pub object_id: u32,
    pub max_depth: u32,
    pub was_dirty: bool,
    pub fresh: bool,
}

/// Per-region uniform — laid out to match WGSL's std430 storage layout
/// for `array<RegionUniform>`. `vec3<f32>` always has 16-byte alignment
/// in WGSL regardless of address space, so the explicit `_pad*` slots
/// match what naga/the validator expects.
///
/// 224 bytes total — kept identical to V8 for upload-buffer reuse.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RegionUniform {
    pub aabb_min: [f32; 3],
    pub cell_size: f32,
    pub aabb_max: [f32; 3],
    pub shader_id: u32,
    pub octree_offset: u32,
    pub octree_capacity: u32,
    pub brick_offset: u32,
    pub brick_capacity: u32,
    pub leaf_attr_offset: u32,
    pub leaf_attr_capacity: u32,
    pub max_depth: u32,
    pub time: f32,
    pub material_id: u32,
    pub region_thickness: f32,
    pub host_octree_root: u32,
    pub host_octree_depth: u32,
    pub host_octree_extent: f32,
    /// Padding so `host_grid_origin` (vec3, 16-byte aligned in WGSL)
    /// lands at offset 96. Without this `host_grid_origin` would be at
    /// 84, mis-matching WGSL's std430 layout.
    pub _pad_host: [u32; 3],
    pub host_grid_origin: [f32; 3],
    /// Padding so `params` (vec4, 16-byte aligned) lands at offset 112.
    pub _pad_grid: f32,
    pub params: [[f32; 4]; 2],
    pub host_inverse_world: [[f32; 4]; 4],
}

const _: () = assert!(std::mem::size_of::<RegionUniform>() == 208);

/// Per-dispatch state for `classify_main`. Re-uploaded between levels
/// at a stride of `LEVEL_UNIFORM_STRIDE` bytes, addressed by dynamic
/// offset.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LevelUniform {
    pub current_level: u32,
    pub per_level_cap: u32,
    pub max_active_per_level: u32,
    pub _pad: u32,
}

const _: () = assert!(std::mem::size_of::<LevelUniform>() == 16);

const LEVEL_UNIFORM_STRIDE: u64 = 256;

/// One BFS-side initial active cell (one per region per dispatch
/// chain). Mirrors the WGSL `ActiveCell` struct.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ActiveCell {
    octree_offset: u32,
    region_index: u32,
    center: [f32; 3],
    half_extent: f32,
    _pad0: u32,
    _pad1: u32,
}

const _: () = assert!(std::mem::size_of::<ActiveCell>() == 32);

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

/// Holds the BFS pipelines, the transient resources they own, and the
/// per-frame buffers (active queue, fill queue, level uniforms).
pub struct UserShaderPass {
    group0_layout: wgpu::BindGroupLayout,
    group1_layout: wgpu::BindGroupLayout,
    group2_layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    classify_pipeline: wgpu::ComputePipeline,
    fill_pipeline: wgpu::ComputePipeline,

    /// Per-region atomic alloc counters. One u32 each per region.
    octree_alloc_buffer: wgpu::Buffer,
    brick_alloc_buffer: wgpu::Buffer,
    leaf_attr_alloc_buffer: wgpu::Buffer,
    alloc_capacity: u64,

    /// Active-cell queue — `(MAX_DEPTH+1) * PER_LEVEL_QUEUE_CAP`
    /// `ActiveCell`s. Written by CPU at level 0 and by `classify_main`
    /// at levels >0.
    active_queue_buffer: wgpu::Buffer,
    /// `(MAX_DEPTH+1)` u32 atomic counters — one per level.
    active_count_buffer: wgpu::Buffer,

    /// Fill-task queue + counter.
    fill_queue_buffer: wgpu::Buffer,
    fill_count_buffer: wgpu::Buffer,

    /// Region uniforms — `array<RegionUniform>` storage binding.
    regions_buffer: wgpu::Buffer,
    regions_capacity: u64,

    /// Per-level uniform buffer with `MAX_DEPTH+1` slots at
    /// `LEVEL_UNIFORM_STRIDE` apart.
    level_uniforms_buffer: wgpu::Buffer,

    source_hash: u64,
    group0_bind_group: Option<wgpu::BindGroup>,
    group0_buffers_epoch: u64,
}

impl UserShaderPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let group0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("user_shader_geom group0"),
            entries: &[
                rw_storage(0), // octree_nodes
                rw_storage(1), // brick_pool
                rw_storage(2), // leaf_attr_pool
                rw_storage(3), // octree_alloc
                rw_storage(4), // brick_alloc
                rw_storage(5), // leaf_attr_alloc
                rw_storage(6), // active_queue
                rw_storage(7), // active_count
                rw_storage(8), // fill_queue
                rw_storage(9), // fill_count
            ],
        });
        let group1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("user_shader_geom group1"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(
                        std::mem::size_of::<RegionUniform>() as u64,
                    ),
                },
                count: None,
            }],
        });
        let group2_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("user_shader_geom group2"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: std::num::NonZeroU64::new(
                        std::mem::size_of::<LevelUniform>() as u64,
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
                Some(&group2_layout),
            ],
            immediate_size: 0,
        });
        let (classify_pipeline, fill_pipeline) = build_pipelines(device, &pipeline_layout, "");

        // Initial alloc-counter capacity sized for 64 regions; grows on
        // demand via `ensure_capacity`.
        let alloc_capacity: u64 = 64 * 4;
        let octree_alloc_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("user_shader_geom octree_alloc"),
            size: alloc_capacity,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let brick_alloc_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("user_shader_geom brick_alloc"),
            size: alloc_capacity,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let leaf_attr_alloc_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("user_shader_geom leaf_attr_alloc"),
            size: alloc_capacity,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let queue_size_bytes =
            (MAX_DEPTH + 1) as u64 * PER_LEVEL_QUEUE_CAP as u64 * std::mem::size_of::<ActiveCell>() as u64;
        let active_queue_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("user_shader_geom active_queue"),
            size: queue_size_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let active_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("user_shader_geom active_count"),
            size: (MAX_DEPTH + 1) as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let fill_queue_size = FILL_QUEUE_CAP as u64 * 32; // BrickFillTask = 32 B
        let fill_queue_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("user_shader_geom fill_queue"),
            size: fill_queue_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let fill_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("user_shader_geom fill_count"),
            // 4-element array — array<atomic<u32>> with `arrayLength`
            // resolution requires a non-zero element count; only [0]
            // is read.
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let regions_capacity: u64 = std::mem::size_of::<RegionUniform>() as u64 * 64;
        let regions_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("user_shader_geom regions"),
            size: regions_capacity,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let level_uniforms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("user_shader_geom level_uniforms"),
            size: LEVEL_UNIFORM_STRIDE * (MAX_DEPTH + 1) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            group0_layout,
            group1_layout,
            group2_layout,
            pipeline_layout,
            classify_pipeline,
            fill_pipeline,
            octree_alloc_buffer,
            brick_alloc_buffer,
            leaf_attr_alloc_buffer,
            alloc_capacity,
            active_queue_buffer,
            active_count_buffer,
            fill_queue_buffer,
            fill_count_buffer,
            regions_buffer,
            regions_capacity,
            level_uniforms_buffer,
            source_hash: 0,
            group0_bind_group: None,
            group0_buffers_epoch: 0,
        }
    }

    pub fn reload_user_shaders(
        &mut self,
        device: &wgpu::Device,
        user_chunk: &str,
        source_hash: u64,
    ) -> bool {
        if source_hash == self.source_hash {
            return false;
        }
        let (classify, fill) = build_pipelines(device, &self.pipeline_layout, user_chunk);
        self.classify_pipeline = classify;
        self.fill_pipeline = fill;
        self.source_hash = source_hash;
        true
    }

    pub fn source_hash(&self) -> u64 { self.source_hash }

    /// Test accessor — exposes the leaf-attr-alloc counter buffer so a
    /// reference test can read back exactly how many occupied cells
    /// the BFS wrote without having to scan the brick pool. Not part
    /// of the production API; only used by `tests/user_shader_geom_bfs.rs`.
    #[doc(hidden)]
    pub fn test_leaf_attr_alloc_buffer(&self) -> &wgpu::Buffer {
        &self.leaf_attr_alloc_buffer
    }
    #[doc(hidden)]
    pub fn test_brick_alloc_buffer(&self) -> &wgpu::Buffer {
        &self.brick_alloc_buffer
    }
    #[doc(hidden)]
    pub fn test_fill_count_buffer(&self) -> &wgpu::Buffer {
        &self.fill_count_buffer
    }
    #[doc(hidden)]
    pub fn test_active_count_buffer(&self) -> &wgpu::Buffer {
        &self.active_count_buffer
    }

    /// Ensure the per-region alloc-counter buffers are sized for
    /// `region_count` regions, and the regions storage buffer too.
    fn ensure_capacity(&mut self, device: &wgpu::Device, region_count: usize) {
        let needed_alloc = (region_count.max(1) as u64) * 4;
        if needed_alloc > self.alloc_capacity {
            let mut cap = self.alloc_capacity.max(64);
            while cap < needed_alloc { cap = cap.saturating_mul(2); }
            self.octree_alloc_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("user_shader_geom octree_alloc"),
                size: cap,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.brick_alloc_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("user_shader_geom brick_alloc"),
                size: cap,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.leaf_attr_alloc_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("user_shader_geom leaf_attr_alloc"),
                size: cap,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.alloc_capacity = cap;
            self.group0_bind_group = None;
        }
        let needed_regions = std::mem::size_of::<RegionUniform>() as u64 * region_count.max(1) as u64;
        if needed_regions > self.regions_capacity {
            let mut cap = self.regions_capacity.max(std::mem::size_of::<RegionUniform>() as u64);
            while cap < needed_regions { cap = cap.saturating_mul(2); }
            self.regions_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("user_shader_geom regions"),
                size: cap,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.regions_capacity = cap;
        }
    }

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
            layout: &self.group0_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: octree_nodes_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: brick_pool_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: leaf_attr_pool_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.octree_alloc_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.brick_alloc_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.leaf_attr_alloc_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.active_queue_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: self.active_count_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: self.fill_queue_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: self.fill_count_buffer.as_entire_binding() },
            ],
        }));
        self.group0_buffers_epoch = buffers_epoch;
    }

    /// Encode the BFS dispatch chain. `region_uniforms[i]` must align
    /// with the i'th initial cell's `region_index`.
    ///
    /// Scene buffer refs + `buffers_epoch` are passed in (rather than
    /// requiring a prior `ensure_group0` call) because
    /// `ensure_capacity` below may invalidate group 0 when the
    /// per-region counter buffers grow — we need to (re)build group
    /// 0 *after* that grow so the dispatch sees current bindings.
    /// Calling `ensure_group0` externally beforehand isn't enough:
    /// growing the counter buffers nulls it out and the dispatch
    /// silently no-ops with "group 0 not bound".
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_regions(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        region_uniforms: &[RegionUniform],
        max_max_depth: u32,
        octree_nodes_buffer: &wgpu::Buffer,
        brick_pool_buffer: &wgpu::Buffer,
        leaf_attr_pool_buffer: &wgpu::Buffer,
        buffers_epoch: u64,
    ) {
        if region_uniforms.is_empty() {
            return;
        }
        let region_count = region_uniforms.len();
        // Order matters: grow internal counter buffers FIRST (may
        // null group 0), then ensure group 0 (rebuilds against
        // current scene + counter buffers).
        self.ensure_capacity(device, region_count);
        self.ensure_group0(
            device,
            octree_nodes_buffer,
            brick_pool_buffer,
            leaf_attr_pool_buffer,
            buffers_epoch,
        );

        // Counter resets + seeds. All `queue.write_buffer` calls
        // serialize on the queue and execute BEFORE the encoder's
        // compute passes, so reset-zeroes and seed values both land
        // before the first classify dispatch reads them. Using
        // `encoder.clear_buffer` here would land *after* the writes
        // and wipe the seed values.
        let zero_alloc: Vec<u32> = vec![0u32; region_count.max(1)];
        queue.write_buffer(&self.brick_alloc_buffer, 0, bytemuck::cast_slice(&zero_alloc));
        queue.write_buffer(&self.leaf_attr_alloc_buffer, 0, bytemuck::cast_slice(&zero_alloc));
        let zero_fill = [0u32; 4];
        queue.write_buffer(&self.fill_count_buffer, 0, bytemuck::bytes_of(&zero_fill));

        // Upload region uniforms.
        queue.write_buffer(
            &self.regions_buffer,
            0,
            bytemuck::cast_slice(region_uniforms),
        );

        // Build initial active cells — one per region, at level 0,
        // covering the region's (cube) AABB.
        let mut initial: Vec<ActiveCell> = Vec::with_capacity(region_count);
        for (i, ru) in region_uniforms.iter().enumerate() {
            let center = [
                0.5 * (ru.aabb_min[0] + ru.aabb_max[0]),
                0.5 * (ru.aabb_min[1] + ru.aabb_max[1]),
                0.5 * (ru.aabb_min[2] + ru.aabb_max[2]),
            ];
            let half = 0.5 * (ru.aabb_max[0] - ru.aabb_min[0]);
            initial.push(ActiveCell {
                octree_offset: ru.octree_offset,
                region_index: i as u32,
                center,
                half_extent: half,
                _pad0: 0,
                _pad1: 0,
            });
        }
        // Write initial cells into level-0 slice of active_queue.
        queue.write_buffer(
            &self.active_queue_buffer,
            0,
            bytemuck::cast_slice(&initial),
        );
        // Seed octree_alloc[i] = 1 for each region (root sits at offset 0
        // within each region's slice; subsequent allocs claim 1..N).
        let init_octree_alloc: Vec<u32> = vec![1u32; region_count];
        queue.write_buffer(
            &self.octree_alloc_buffer,
            0,
            bytemuck::cast_slice(&init_octree_alloc),
        );
        // Seed active_count[0] = region_count, others zero.
        let mut init_active_count = vec![0u32; (MAX_DEPTH + 1) as usize];
        init_active_count[0] = region_count as u32;
        queue.write_buffer(
            &self.active_count_buffer,
            0,
            bytemuck::cast_slice(&init_active_count),
        );

        // Build per-level uniform packs.
        let mut level_packed = vec![0u8; LEVEL_UNIFORM_STRIDE as usize * (MAX_DEPTH + 1) as usize];
        for level in 0..=MAX_DEPTH {
            let lu = LevelUniform {
                current_level: level,
                per_level_cap: PER_LEVEL_QUEUE_CAP,
                max_active_per_level: PER_LEVEL_QUEUE_CAP,
                _pad: 0,
            };
            let off = level as usize * LEVEL_UNIFORM_STRIDE as usize;
            level_packed[off..off + std::mem::size_of::<LevelUniform>()]
                .copy_from_slice(bytemuck::bytes_of(&lu));
        }
        queue.write_buffer(&self.level_uniforms_buffer, 0, &level_packed);

        let group0 = match &self.group0_bind_group {
            Some(bg) => bg,
            None => {
                eprintln!("[user_shader_pass] dispatch skipped — group 0 not bound");
                return;
            }
        };
        let group1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("user_shader_geom group1 bg"),
            layout: &self.group1_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.regions_buffer.as_entire_binding(),
            }],
        });
        let group2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("user_shader_geom group2 bg"),
            layout: &self.group2_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &self.level_uniforms_buffer,
                    offset: 0,
                    size: std::num::NonZeroU64::new(
                        std::mem::size_of::<LevelUniform>() as u64,
                    ),
                }),
            }],
        });

        // Classify dispatch chain — one per level 0..=max_max_depth.
        // Each level's dispatch always launches the same workgroup
        // count; threads past `active_count[L]` early-out.
        let workgroups_per_level = PER_LEVEL_QUEUE_CAP / CLASSIFY_WG_SIZE;
        let max_level = max_max_depth.min(MAX_DEPTH);
        for level in 0..=max_level {
            let dynamic_offset = (level as u64 * LEVEL_UNIFORM_STRIDE) as u32;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("user_shader_geom classify"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.classify_pipeline);
            pass.set_bind_group(0, group0, &[]);
            pass.set_bind_group(1, &group1, &[]);
            pass.set_bind_group(2, &group2, &[dynamic_offset]);
            pass.dispatch_workgroups(workgroups_per_level, 1, 1);
        }

        // Brick fill — one workgroup per fill task. wgpu caps each
        // workgroup-dim at 65535, so we dispatch over a 2D grid
        // `(FILL_TILE_X, ceil(FILL_QUEUE_CAP / TILE), 1)`. The shader
        // re-packs `(wid.x, wid.y) → task_idx`. Workgroups past
        // `fill_count` early-out.
        const FILL_TILE_X: u32 = 65535;
        let fill_y = (FILL_QUEUE_CAP + FILL_TILE_X - 1) / FILL_TILE_X;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("user_shader_geom brick_fill"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.fill_pipeline);
        pass.set_bind_group(0, group0, &[]);
        pass.set_bind_group(1, &group1, &[]);
        pass.set_bind_group(2, &group2, &[0]);
        pass.dispatch_workgroups(FILL_TILE_X, fill_y, 1);
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

fn build_pipelines(
    device: &wgpu::Device,
    pipeline_layout: &wgpu::PipelineLayout,
    user_chunk: &str,
) -> (wgpu::ComputePipeline, wgpu::ComputePipeline) {
    let source = compose_geom_source(user_chunk);
    validate_wgsl(&source, "user_shader_geom");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("user_shader_geom"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let classify = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("user_shader_geom classify"),
        layout: Some(pipeline_layout),
        module: &module,
        entry_point: Some("classify_main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let fill = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("user_shader_geom fill"),
        layout: Some(pipeline_layout),
        module: &module,
        entry_point: Some("brick_fill_main"),
        compilation_options: Default::default(),
        cache: None,
    });
    (classify, fill)
}

/// Build the per-region uniform from a request + cache slot.
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
        octree_capacity: slot.octree_capacity,
        brick_offset: slot.brick_offset,
        brick_capacity: slot.brick_capacity,
        leaf_attr_offset: slot.leaf_attr_offset,
        leaf_attr_capacity: slot.leaf_attr_capacity,
        max_depth: slot.max_depth,
        time: time_seconds,
        material_id: request.material_id,
        region_thickness: request.region_thickness,
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

/// Compose the effective hash for a request given the registry's
/// `source_hash` and the host's `geometry_epoch`.
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

/// Resolve a `shader_name` to the registry's `shader_id`. `0` =
/// identity / unregistered.
pub fn resolve_shader_id(infos: &[UserShaderInfo], name: &str) -> u32 {
    if name.is_empty() {
        return 0;
    }
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

    fn req(host: u32, mat: u32) -> ShaderRegionRequest {
        ShaderRegionRequest {
            host_object_id: host,
            material_id: mat,
            shader_name: "x".to_string(),
            params: vec![],
            aabb_min: [0.0; 3],
            aabb_max: [1.0; 3],
            cell_size: 0.25,
            input_hash: 0,
            animated: false,
            region_thickness: 0.0,
            max_depth: 4,
            painted_leaf_count: 8,
            host_octree_root: HOST_NO_HOST_SENTINEL,
            host_octree_depth: 0,
            host_octree_extent: 0.0,
            host_grid_origin: [0.0; 3],
            host_inverse_world: [[0.0; 4]; 4],
            tile_index: NO_TILE,
        }
    }

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
        assert!(src.contains("dispatch_user_generate"));
        assert!(!src.contains("Default identity stub"));
    }

    #[test]
    fn cache_returns_dirty_first_time_and_clean_second() {
        let mut cache = UserShaderObjectCache::new();
        // Slab fits 4 regions: 4 × (32K bricks + 8K octree + 128K leaf-attrs).
        let _slab = estimate_region_pool(0, 4); cache.set_pool_bases(0, _slab.octree * 4, 0, _slab.bricks * BRICK_CELLS * 4, 0, _slab.leaf_attrs * 4);
        let r = req(1, 1);
        let h = effective_hash(&r, 7, 0);
        let s1 = cache.lookup_or_allocate(&r, h).unwrap();
        assert!(s1.was_dirty);
        let s2 = cache.lookup_or_allocate(&r, h).unwrap();
        assert!(!s2.was_dirty);
        assert_eq!(s1.octree_offset, s2.octree_offset);
        assert_eq!(s1.region_index, s2.region_index);
        assert_eq!(s1.object_id, s2.object_id);
    }

    #[test]
    fn cache_animated_always_dirty() {
        let mut cache = UserShaderObjectCache::new();
        // Slab fits 4 regions: 4 × (32K bricks + 8K octree + 128K leaf-attrs).
        let _slab = estimate_region_pool(0, 4); cache.set_pool_bases(0, _slab.octree * 4, 0, _slab.bricks * BRICK_CELLS * 4, 0, _slab.leaf_attrs * 4);
        let mut r = req(1, 1);
        r.animated = true;
        let h = effective_hash(&r, 1, 0);
        assert!(cache.lookup_or_allocate(&r, h).unwrap().was_dirty);
        assert!(cache.lookup_or_allocate(&r, h).unwrap().was_dirty);
    }

    #[test]
    fn cache_different_keys_get_different_slots() {
        let mut cache = UserShaderObjectCache::new();
        // Slab fits 4 regions: 4 × (32K bricks + 8K octree + 128K leaf-attrs).
        let _slab = estimate_region_pool(0, 4); cache.set_pool_bases(0, _slab.octree * 4, 0, _slab.bricks * BRICK_CELLS * 4, 0, _slab.leaf_attrs * 4);
        let mut r = req(1, 1);
        let s1 = cache.lookup_or_allocate(&r, 0).unwrap();
        r.material_id = 2;
        let s2 = cache.lookup_or_allocate(&r, 0).unwrap();
        assert_ne!(s1.octree_offset, s2.octree_offset);
        assert_ne!(s1.brick_offset, s2.brick_offset);
        assert_ne!(s1.region_index, s2.region_index);
    }

    #[test]
    fn cache_distinguishes_tiles_on_same_material() {
        let mut cache = UserShaderObjectCache::new();
        // Slab fits 4 regions: 4 × (32K bricks + 8K octree + 128K leaf-attrs).
        let _slab = estimate_region_pool(0, 4); cache.set_pool_bases(0, _slab.octree * 4, 0, _slab.bricks * BRICK_CELLS * 4, 0, _slab.leaf_attrs * 4);
        let mut r = req(1, 1);
        r.tile_index = [0, 0, 0];
        let s1 = cache.lookup_or_allocate(&r, 0).unwrap();
        r.tile_index = [1, 0, 0];
        let s2 = cache.lookup_or_allocate(&r, 0).unwrap();
        // Same (object, material), different tiles → distinct slots.
        assert_ne!(s1.octree_offset, s2.octree_offset);
        assert_ne!(s1.region_index, s2.region_index);
    }

    #[test]
    fn on_demand_eviction_when_pool_full_steady_state() {
        // V10 — last frame filled the pool with N tiles. This frame
        // shifts to a different N tiles (e.g. user paints in a new
        // area, abandoning some old tiles). Without on-demand
        // eviction the new tiles get dropped because all slots
        // appear taken; with it, untouched entries vacate as new
        // ones request.
        let mut cache = UserShaderObjectCache::new();
        // Pool fits exactly 4 slabs.
        let _slab = estimate_region_pool(0, 4); cache.set_pool_bases(0, _slab.octree * 4, 0, _slab.bricks * BRICK_CELLS * 4, 0, _slab.leaf_attrs * 4);
        // Frame 1 — fill the pool with tiles [0..4].
        cache.begin_frame();
        let mut r = req(1, 1);
        for i in 0..4 {
            r.tile_index = [i, 0, 0];
            assert!(cache.lookup_or_allocate(&r, 0).is_some());
        }
        // Pool now full. Don't evict.
        // Frame 2 — request tiles [4..8]. None of the prior 4 are
        // touched, so on-demand eviction should kick in and let the
        // new ones claim slots.
        cache.begin_frame();
        for i in 4..8 {
            r.tile_index = [i, 0, 0];
            assert!(
                cache.lookup_or_allocate(&r, 0).is_some(),
                "on-demand eviction should free a slot for new tile {i}",
            );
        }
        // After this frame the cache contains the new 4 tiles only.
        assert_eq!(cache.entries.len(), 4);
        for i in 4..8 {
            assert!(cache.entries.contains_key(&(1, 1, [i, 0, 0])));
        }
    }

    #[test]
    fn evict_untouched_drops_abandoned_tiles() {
        let mut cache = UserShaderObjectCache::new();
        // Slab fits 4 regions: 4 × (32K bricks + 8K octree + 128K leaf-attrs).
        let _slab = estimate_region_pool(0, 4); cache.set_pool_bases(0, _slab.octree * 4, 0, _slab.bricks * BRICK_CELLS * 4, 0, _slab.leaf_attrs * 4);
        let mut r = req(1, 1);
        // Frame 1 — three tiles allocated.
        cache.begin_frame();
        for i in 0..3 {
            r.tile_index = [i, 0, 0];
            cache.lookup_or_allocate(&r, 0).unwrap();
        }
        cache.evict_untouched();
        assert_eq!(cache.entries.len(), 3);
        // Frame 2 — only one tile referenced.
        cache.begin_frame();
        r.tile_index = [1, 0, 0];
        cache.lookup_or_allocate(&r, 0).unwrap();
        cache.evict_untouched();
        assert_eq!(cache.entries.len(), 1);
        // The freed slots from tiles 0 and 2 should be reusable.
        assert!(cache.free_slots.len() >= 2);
        // Frame 3 — request a new tile; should pop a free slot
        // rather than bumping the high-water mark.
        cache.begin_frame();
        r.tile_index = [42, 0, 0];
        let pre_high_water = cache.octree_high_water;
        cache.lookup_or_allocate(&r, 0).unwrap();
        assert_eq!(cache.octree_high_water, pre_high_water,
            "free slot should be reused before bumping high-water");
    }

    #[test]
    fn cache_flushes_on_geometry_epoch_bump() {
        let mut cache = UserShaderObjectCache::new();
        // Slab fits 4 regions: 4 × (32K bricks + 8K octree + 128K leaf-attrs).
        let _slab = estimate_region_pool(0, 4); cache.set_pool_bases(0, _slab.octree * 4, 0, _slab.bricks * BRICK_CELLS * 4, 0, _slab.leaf_attrs * 4);
        let r = req(1, 1);
        cache.lookup_or_allocate(&r, 0);
        assert!(cache.reconcile_epoch(1));
        assert!(cache.entries.is_empty());
    }

    #[test]
    fn pool_exhaustion_returns_none() {
        let mut cache = UserShaderObjectCache::new();
        // Tiny pool — first allocation succeeds, second must fail.
        let est = estimate_region_pool(8, 4);
        cache.set_pool_bases(0, est.octree, 0, est.bricks * BRICK_CELLS, 0, est.leaf_attrs);
        let mut r = req(1, 1);
        assert!(cache.lookup_or_allocate(&r, 0).is_some());
        r.host_object_id = 2;
        assert!(cache.lookup_or_allocate(&r, 0).is_none());
    }

    #[test]
    fn region_uniform_size_is_208() {
        assert_eq!(std::mem::size_of::<RegionUniform>(), 208);
    }

    #[test]
    fn active_cell_size_is_32() {
        assert_eq!(std::mem::size_of::<ActiveCell>(), 32);
    }

    #[test]
    fn estimate_uniform_slab_size() {
        // V10 slab allocator: every region reserves the same size,
        // independent of painted_leaf_count. The fixed slab makes
        // freed slots reusable by any future allocation, eliminating
        // the fragmentation that previously broke the cache once
        // tile estimates fluctuated.
        let small = estimate_region_pool(8, 4);
        let large = estimate_region_pool(800, 4);
        assert_eq!(large.octree, small.octree);
        assert_eq!(large.bricks, small.bricks);
        assert_eq!(large.leaf_attrs, small.leaf_attrs);
        assert!(small.bricks > 0);
        assert!(small.octree > 0);
        assert!(small.leaf_attrs > 0);
    }

    #[test]
    fn estimate_depth_8_within_slab_bounds() {
        let est = estimate_region_pool(64, 8);
        // Depth 8 doesn't blow past the slab (it's already sized for
        // max_depth 8 worst case).
        assert!(est.bricks <= 2048);
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
