//! Scene-to-GPU synchronization — builds ArvxGpuAsset + ArvxGpuInstance
//! arrays from scene state.

use bytemuck::Zeroable;
use glam::{Mat4, Vec3, Vec4};
use arvx_render::arvx_gpu_object::{self, ArvxGpuAsset, ArvxGpuInstance};

/// Screen-space AABB for tile culling (pixel coordinates).
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ScreenAabb {
    pub min_x: f32,
    pub min_y: f32,
    pub max_x: f32,
    pub max_y: f32,
}

/// Tile size in pixels — historical 8×8 tile granularity. Used by the
/// CPU-side screen-AABB→tile-list builder.
pub const MARCH_TILE_SIZE: u32 = 8;

/// Per-tile object lists. Replaces the 32-bit bitmask scheme so the
/// number of objects a tile can cover is unbounded — a tile's list is
/// `object_ids[offsets[t]..offsets[t+1]]`.
pub struct TileLists {
    /// Prefix-sum of per-tile object counts. Length = `num_tiles + 1`.
    pub offsets: Vec<u32>,
    /// Flat array of object indices into the scene's `gpu_objects`,
    /// grouped by tile.
    pub object_ids: Vec<u32>,
    /// Tile grid width (in tiles, not pixels). Shaders need this to
    /// compute `tile_idx = ty * tile_count_x + tx` from a pixel coord.
    pub tile_count_x: u32,
    pub tile_count_y: u32,
}

/// Build per-tile object lists from the pre-computed screen AABBs.
/// Two passes over the tile grid: count, then fill. O(sum of tiles
/// each object overlaps) — a few ms for thousands of objects at 1080p.
pub fn build_tile_lists(
    screen_aabbs: &[ScreenAabb],
    width: u32,
    height: u32,
) -> TileLists {
    let tile_size = MARCH_TILE_SIZE;
    let tile_count_x = (width + tile_size - 1) / tile_size;
    let tile_count_y = (height + tile_size - 1) / tile_size;
    let num_tiles = (tile_count_x * tile_count_y) as usize;

    // Per-object tile range, clamped to the grid. Zero-sized AABBs
    // (culled / off-screen objects) are skipped via tx_max < tx_min.
    let object_range = |sa: &ScreenAabb| -> Option<(u32, u32, u32, u32)> {
        if sa.max_x <= sa.min_x || sa.max_y <= sa.min_y {
            return None;
        }
        let tx_min = ((sa.min_x / tile_size as f32).floor() as i32).max(0) as u32;
        let ty_min = ((sa.min_y / tile_size as f32).floor() as i32).max(0) as u32;
        // `(px - 1) / 8` clamps the last pixel into its tile rather than
        // the next one, so an AABB ending exactly at a tile boundary
        // doesn't spuriously claim the neighbor.
        let tx_max_clip = (sa.max_x - 1.0).max(0.0) as u32 / tile_size;
        let ty_max_clip = (sa.max_y - 1.0).max(0.0) as u32 / tile_size;
        let tx_max = tx_max_clip.min(tile_count_x.saturating_sub(1));
        let ty_max = ty_max_clip.min(tile_count_y.saturating_sub(1));
        if tx_min > tx_max || ty_min > ty_max {
            return None;
        }
        Some((tx_min, ty_min, tx_max, ty_max))
    };

    // Pass 1 — count objects per tile.
    let mut counts = vec![0u32; num_tiles];
    for sa in screen_aabbs {
        if let Some((tx_min, ty_min, tx_max, ty_max)) = object_range(sa) {
            for ty in ty_min..=ty_max {
                let row = ty * tile_count_x;
                for tx in tx_min..=tx_max {
                    counts[(row + tx) as usize] += 1;
                }
            }
        }
    }

    // Prefix sum — offsets[t] is where tile t's list starts.
    let mut offsets = vec![0u32; num_tiles + 1];
    let mut running = 0u32;
    for i in 0..num_tiles {
        offsets[i] = running;
        running += counts[i];
    }
    offsets[num_tiles] = running;

    // Pass 2 — fill object_ids. `cursors` tracks per-tile write position.
    let mut object_ids = vec![0u32; running as usize];
    let mut cursors = vec![0u32; num_tiles];
    for (obj_idx, sa) in screen_aabbs.iter().enumerate() {
        if let Some((tx_min, ty_min, tx_max, ty_max)) = object_range(sa) {
            for ty in ty_min..=ty_max {
                let row = ty * tile_count_x;
                for tx in tx_min..=tx_max {
                    let t = (row + tx) as usize;
                    let slot = (offsets[t] + cursors[t]) as usize;
                    object_ids[slot] = obj_idx as u32;
                    cursors[t] += 1;
                }
            }
        }
    }

    TileLists { offsets, object_ids, tile_count_x, tile_count_y }
}

/// Compute screen-space AABBs for all instances.
/// Projects each instance's local AABB (looked up via `asset_id` in the
/// asset table, transformed by the instance's world matrix) to pixel
/// coordinates.
pub fn compute_screen_aabbs(
    instances: &[ArvxGpuInstance],
    assets: &[ArvxGpuAsset],
    view_proj: &Mat4,
    width: f32,
    height: f32,
) -> Vec<ScreenAabb> {
    instances.iter().map(|inst| {
        let asset_idx = inst.asset_id as usize;
        if asset_idx >= assets.len() {
            return ScreenAabb::zeroed();
        }
        let asset = &assets[asset_idx];
        if asset.geom_type == 0 {
            return ScreenAabb::zeroed();
        }

        // Build the 8 corners of the asset's local AABB. Iterate the
        // ACTUAL `aabb_min` / `aabb_max` rather than `±octree_extent/2`
        // — they're not always equivalent. Standard host assets bake
        // their octree centered around the local origin
        // (aabb = [-half, +half]) so the two agree there. User-shader
        // instance protos (Phase 4c) are baked into [0, 1]³ canonical
        // space (aabb_min = [0, 0, 0], aabb_max = [1, 1, 1]); using
        // `±half` would project the wrong cube and miss most pixels.
        let aabb_min = Vec3::from(asset.aabb_min);
        let aabb_max = Vec3::from(asset.aabb_max);
        let world = Mat4::from_cols_array_2d(&inst.world);

        let mut smin = Vec3::splat(f32::MAX);
        let mut smax = Vec3::splat(f32::MIN);

        for corner in 0..8u32 {
            let local = Vec3::new(
                if corner & 1 != 0 { aabb_max.x } else { aabb_min.x },
                if corner & 2 != 0 { aabb_max.y } else { aabb_min.y },
                if corner & 4 != 0 { aabb_max.z } else { aabb_min.z },
            );
            let world_pos = world.transform_point3(local);
            let clip = *view_proj * Vec4::new(world_pos.x, world_pos.y, world_pos.z, 1.0);

            // Behind camera: conservatively expand to full screen.
            if clip.w <= 0.0 {
                return ScreenAabb { min_x: 0.0, min_y: 0.0, max_x: width, max_y: height };
            }

            let ndc = clip.truncate() / clip.w;
            let px = (ndc.x * 0.5 + 0.5) * width;
            let py = (0.5 - ndc.y * 0.5) * height;
            smin = smin.min(Vec3::new(px, py, 0.0));
            smax = smax.max(Vec3::new(px, py, 0.0));
        }

        ScreenAabb {
            min_x: smin.x,
            min_y: smin.y,
            max_x: smax.x,
            max_y: smax.y,
        }
    }).collect()
}

/// Skinning attachment for a GPU object. Produced per-frame for every
/// entity carrying a `Skeleton`. Carries the bone-matrix indexing
/// (produced by `BoneMatrixAllocator`) the mesh VS reads to skin
/// per-vertex.
#[derive(Debug, Copy, Clone)]
pub struct SkinnedBinding {
    /// Number of bones in the skeleton.
    pub bone_count: u32,
    /// Offset into the scene-wide bone-matrix buffer, in `mat4x4<f32>`
    /// units (one mat = 16 f32s = 64 bytes).
    pub bone_buffer_offset: u32,
    /// Offset into the scene-wide precomputed dual-quat buffer, in
    /// `DualQuat` (32-byte) units. DQs are forward-pose-only.
    pub bone_dq_offset: u32,
}

/// Build an `ArvxGpuAsset` from an asset's spatial + voxelization data
/// + optional skinning template (the rest octree refs and bone count).
///
/// All fields here are constant across every instance of one asset, so
/// the upstream caller (`engine/scene_gpu.rs`) builds this once per
/// unique asset (keyed by `octree_root`) per frame.
pub fn build_gpu_asset(
    aabb: &arvx_core::Aabb,
    grid_origin: glam::Vec3,
    spatial: &arvx_core::scene_node::SpatialHandle,
    voxel_size: f32,
    bone_count: u32,
) -> ArvxGpuAsset {
    let mut a = ArvxGpuAsset::zeroed();
    a.aabb_min = aabb.min.into();
    a.aabb_max = aabb.max.into();
    a.grid_origin = grid_origin.into();
    a.voxel_size = voxel_size;
    a.geom_type = arvx_gpu_object::geom_type::VOXELIZED;
    a.bone_count = bone_count;
    if let arvx_core::scene_node::SpatialHandle::Octree {
        root_offset, depth, base_voxel_size, ..
    } = spatial
    {
        a.octree_root = *root_offset;
        a.octree_depth = *depth as u32;
        let extent = (1u32 << depth) as f32 * base_voxel_size;
        a.octree_extent_bits = extent.to_bits();
        // Rest octree mirrors the runtime octree — the skinned march
        // uses the rest-pose structure for empty-space descent after
        // inverse-skinning each sample. (Bone_count == 0 means the
        // asset isn't skinned; the rest_octree_* fields are still set
        // but unused by the march path.)
        a.rest_octree_root = a.octree_root;
        a.rest_octree_depth = a.octree_depth;
        a.rest_octree_extent_bits = a.octree_extent_bits;
    }
    a
}

/// Build an `ArvxGpuInstance` from per-entity transform + asset reference
/// + optional per-frame skinning binding (palette offset + scattered
/// bone-field allocation).
///
/// `asset_id` is the slot index in this frame's assets table — the
/// caller assigns it after deciding whether this entity reuses an
/// existing asset slot or creates a new one.
pub fn build_gpu_instance(
    world_matrix: &glam::Mat4,
    asset_id: u32,
    material_id: u16,
    object_id: u32,
    skinning: Option<SkinnedBinding>,
) -> ArvxGpuInstance {
    let mut i = ArvxGpuInstance::zeroed();
    i.world = world_matrix.to_cols_array_2d();
    i.asset_id = asset_id;
    i.material_id = material_id as u32;
    i.object_id = object_id;

    if let Some(skin) = skinning {
        i.is_skinned = 1;
        i.bone_buffer_offset = skin.bone_buffer_offset;
    }
    i
}

/// Per-frame allocator that concatenates every skinned entity's skinning
/// palettes (forward + inverse) into one contiguous byte buffer.
///
/// Per-entity layout: `[forward_0..forward_N, inverse_0..inverse_N]`
/// where `N = bone_count`. The shader keys off `bone_buffer_offset` for
/// the forward range and `bone_buffer_offset + bone_count` for the
/// inverse range — same pattern as rkifield's bone buffer
/// (`ray_march.wgsl:695: bone_matrices[bone_buffer_offset + bone_count + bone_idx]`).
///
/// Called once per frame after [`crate::animation::tick`] has refreshed
/// each skeleton's `current_pose` + `inverse_pose`.
/// Pair of `vec4<f32>` (real + dual) = 32 bytes, matching the
/// `DualQuat` struct the mesh VS reads.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DualQuat {
    real: [f32; 4],
    dual: [f32; 4],
}

/// Extract a forward-pose dual quaternion from a bone matrix.
/// Assumes the matrix is (close to) a pure rigid transform — any scale
/// is dropped. For arrvox this holds when `normalize_mesh` is a
/// uniform-scale + translation, which is the case for Mixamo rigs
/// through the animation::tick conjugation.
fn mat_to_dual_quat(mat: Mat4) -> DualQuat {
    let (_scale, rot, trans) = mat.to_scale_rotation_translation();
    // Dual part = 0.5 * (t_quat * r)  where  t_quat = (t.xyz, 0).
    let t_quat = glam::Quat::from_xyzw(trans.x, trans.y, trans.z, 0.0);
    let d = (t_quat * rot) * 0.5;
    DualQuat {
        real: [rot.x, rot.y, rot.z, rot.w],
        dual: [d.x, d.y, d.z, d.w],
    }
}

/// One entity's slot in the flat bone-matrix layout. Held in
/// [`BoneMatrixAllocator::layout`] across rebuilds; D1 uses it to
/// detect "layout unchanged" frames where we can do per-entity delta
/// uploads instead of rewriting the whole buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BoneSlot {
    entity_bits: u64,
    bone_count: u32,
    /// In mat4 units (forward palette start). Inverse starts at
    /// `mat_offset + bone_count`.
    mat_offset: u32,
    /// In `DualQuat` (32 B) units. Forward only.
    dq_offset: u32,
}

#[derive(Default)]
pub struct BoneMatrixAllocator {
    /// Concatenated palettes: per entity, `mat4x4<f32>` forward-then-
    /// inverse, entities in `Entity::to_bits()` order (stable across
    /// rebuilds — D1 depends on this for delta-upload correctness).
    /// Wrapped in `Arc<Vec<u8>>` so the per-tick snapshot handoff is a
    /// refcount bump rather than an ~58 MB memcpy of the prior
    /// `.bytes().to_vec()` design (PERF_DEBT A3). Mutations during
    /// [`Self::rebuild`] reach into the Vec via `Arc::make_mut`; the
    /// COW happens once per rebuild when render still holds last
    /// frame's `Arc`, but we immediately clear the Vec, so `make_mut`
    /// on a refcount=2 Arc reallocates fresh — no payload copy of the
    /// old contents.
    bytes: std::sync::Arc<Vec<u8>>,
    /// Concatenated forward dual quats — one `DualQuat` per bone per
    /// entity. Parallel to the forward half of `bytes` (same entity
    /// order, same bone order), but independent byte layout because
    /// DualQuat (32 B) doesn't stride the same as mat4 (64 B).
    bytes_dq: std::sync::Arc<Vec<u8>>,
    /// Entity → `SkinnedBinding` for the current frame.
    bindings: std::collections::HashMap<hecs::Entity, SkinnedBinding>,
    /// Stable slot layout from the previous rebuild. Compared against
    /// the new layout each rebuild — when identical, we know the
    /// byte offsets of every entity's pose didn't shift, so we can
    /// emit per-entity dirty ranges instead of a full-buffer mark.
    /// PERF_DEBT.md D1.
    layout: Vec<BoneSlot>,
    /// Per-entity forward pose snapshot from the previous rebuild.
    /// `current_pose == previous_pose[entity]` means the entity's
    /// bones didn't move this tick and its slot bytes are identical
    /// to last frame — no upload required. Cleared and refilled each
    /// rebuild. PERF_DEBT.md D1.
    previous_poses: std::collections::HashMap<hecs::Entity, Vec<Mat4>>,
    /// Byte ranges in [`Self::bytes`] that differ from the GPU
    /// buffer's contents (i.e. that need re-uploading). Filled in
    /// during rebuild based on per-entity pose comparison. Drained
    /// (via [`Self::take_mat_dirty`]) when the sim builds a render
    /// frame; the consumer translates the ranges to
    /// `queue.write_buffer` calls.
    mat_dirty: arvx_core::DirtyRanges,
    /// Same shape as [`Self::mat_dirty`] but for [`Self::bytes_dq`].
    dq_dirty: arvx_core::DirtyRanges,
}

impl BoneMatrixAllocator {
    pub fn new() -> Self { Self::default() }

    /// Reset and re-pack every skinned entity's forward + inverse
    /// palettes into the flat byte buffer, plus a parallel buffer of
    /// forward-pose dual quaternions (the DQS fast path).
    ///
    /// Offsets:
    /// * `bone_buffer_offset` indexes `bone_matrices` in mat4 units —
    ///   forward slot at `[off + i]`, inverse at `[off + bone_count + i]`.
    /// * `bone_dq_offset` indexes `bone_dual_quats` in DualQuat units —
    ///   forward slot at `[off + i]`. No inverse palette in this buffer.
    ///
    /// Entities are processed in `Entity::to_bits()` order so the
    /// slot layout is stable across rebuilds when the entity set and
    /// per-entity bone counts are unchanged. PERF_DEBT.md D1 uses
    /// that stability to emit per-entity dirty ranges instead of a
    /// full-buffer mark whenever animation::tick advanced only some
    /// players (or none — the dirty set ends up empty and the
    /// render-side upload becomes a no-op).
    pub fn rebuild(&mut self, world: &hecs::World) {
        // `Arc::make_mut` reuses the Vec when refcount==1 (typical
        // case after render dropped last frame's snapshot) and
        // reallocates a fresh Vec when refcount>1 (last frame still
        // in flight). The `.clear()` immediately after means we never
        // pay the memcpy of the COW path — it just hands us a fresh
        // empty Vec with previous capacity gone.
        std::sync::Arc::make_mut(&mut self.bytes).clear();
        std::sync::Arc::make_mut(&mut self.bytes_dq).clear();
        self.bindings.clear();
        self.mat_dirty.clear();
        self.dq_dirty.clear();

        // Stable iteration order. hecs query iteration order shifts
        // when archetypes change (entity add/remove with different
        // component sets), which would shuffle per-entity offsets in
        // the byte buffer and invalidate any per-entity dirty range
        // we tracked against the previous rebuild. We materialize
        // into a Vec so the query's internal QueryBorrow drops before
        // we mutate `self.bindings` / `self.bytes` below.
        let mut query = world.query::<&crate::components::Skeleton>();
        let mut entities: Vec<(hecs::Entity, Vec<Mat4>, Vec<Mat4>)> = query
            .iter()
            .filter(|(_, skel)| !skel.current_pose.is_empty())
            .map(|(e, skel)| (e, skel.current_pose.clone(), skel.inverse_pose.clone()))
            .collect();
        drop(query);
        entities.sort_by_key(|(e, _, _)| e.to_bits());

        // Compute the new slot layout up front so the layout-equality
        // check below has both inputs before we touch the byte
        // buffers.
        let mut new_layout: Vec<BoneSlot> = Vec::with_capacity(entities.len());
        {
            let mut mat_off: u32 = 0;
            let mut dq_off: u32 = 0;
            for (entity, current_pose, _inverse_pose) in &entities {
                let bone_count = current_pose.len() as u32;
                new_layout.push(BoneSlot {
                    entity_bits: entity.to_bits().into(),
                    bone_count,
                    mat_offset: mat_off,
                    dq_offset: dq_off,
                });
                mat_off += bone_count * 2;
                dq_off += bone_count;
            }
        }
        let layout_unchanged = self.layout == new_layout;

        // Rebuild byte buffers + bindings while tracking dirty ranges.
        // `running_mat_offset`/`running_dq_offset` mirror `new_layout`
        // so we don't re-derive them here.
        let mut next_previous_poses: std::collections::HashMap<hecs::Entity, Vec<Mat4>> =
            std::collections::HashMap::with_capacity(entities.len());
        let mut running_mat_offset: u32 = 0;
        let mut running_dq_offset: u32 = 0;
        for (entity, current_pose, inverse_pose) in entities {
            let bone_count = current_pose.len() as u32;
            // Forward palette first, then inverse palette. They must be
            // the same length — `animation::tick` keeps them in sync.
            let fwd: &[u8] = bytemuck::cast_slice(&current_pose);
            let inv: &[u8] = bytemuck::cast_slice(&inverse_pose);
            let bytes = std::sync::Arc::make_mut(&mut self.bytes);
            bytes.extend_from_slice(fwd);
            bytes.extend_from_slice(inv);

            // Precomputed forward dual quats for the DQS scatter branch.
            // One DQ per bone — scatter doesn't need inverse dual quats.
            let dqs: Vec<DualQuat> = current_pose.iter()
                .map(|m| mat_to_dual_quat(*m))
                .collect();
            std::sync::Arc::make_mut(&mut self.bytes_dq)
                .extend_from_slice(bytemuck::cast_slice(&dqs));

            self.bindings.insert(entity, SkinnedBinding {
                bone_count,
                bone_buffer_offset: running_mat_offset,
                bone_dq_offset: running_dq_offset,
            });

            // D1 per-entity dirty tracking. When the layout is
            // unchanged across rebuilds and the entity's forward pose
            // is bit-identical to last frame, this entity contributed
            // nothing new to the byte buffer — leave its slot out of
            // the dirty ranges and the render side skips its upload.
            // Inverse pose is derived from forward in animation::tick,
            // so a forward match implies an inverse match; we don't
            // hash them separately. When the layout shifted (entity
            // add / remove / bone count change) we fall through to
            // the `mark_full` branch below.
            if layout_unchanged {
                let pose_changed = match self.previous_poses.get(&entity) {
                    Some(p) => p != &current_pose,
                    // First time we see this entity at the current
                    // layout — must upload its slot once. (In
                    // practice unreachable when layout_unchanged is
                    // true, but defensive.)
                    None => true,
                };
                if pose_changed {
                    let mat_byte_off = running_mat_offset
                        .checked_mul(64)
                        .expect("bone palette offset overflows u32");
                    let mat_byte_len = bone_count
                        .checked_mul(64 * 2)
                        .expect("bone palette slot size overflows u32");
                    self.mat_dirty.mark(mat_byte_off, mat_byte_len);
                    let dq_byte_off = running_dq_offset
                        .checked_mul(32)
                        .expect("bone dq offset overflows u32");
                    let dq_byte_len = bone_count
                        .checked_mul(32)
                        .expect("bone dq slot size overflows u32");
                    self.dq_dirty.mark(dq_byte_off, dq_byte_len);
                }
            }

            next_previous_poses.insert(entity, current_pose);

            // Advance past both palettes.
            running_mat_offset += bone_count * 2;
            running_dq_offset += bone_count;
        }

        if !layout_unchanged {
            // Layout shifted — every byte in the new buffer needs to
            // overwrite the corresponding byte in the GPU buffer (which
            // still holds last frame's layout). Falling back to
            // `mark_full` lets the render side use its existing
            // grow-or-rewrite ensure_and_write path.
            self.mat_dirty.mark_full(self.bytes.len() as u32);
            self.dq_dirty.mark_full(self.bytes_dq.len() as u32);
        }

        // Stale `previous_poses` entries for entities that vanished
        // would just sit there until the next rebuild that names them
        // (never). Replace wholesale.
        self.previous_poses = next_previous_poses;
        self.layout = new_layout;
    }

    /// Flat byte buffer ready to ship in `FrameUpload.bone_matrices`.
    pub fn bytes(&self) -> &[u8] { &self.bytes }

    /// Cheap shareable handle to the bone-matrix bytes for the per-tick
    /// snapshot handoff. `Arc::clone` — no memcpy. Render holds the
    /// returned Arc until it ships (or drops) the frame; the next
    /// `rebuild()`'s `make_mut` will reallocate when render's clone is
    /// still alive.
    pub fn bytes_arc(&self) -> std::sync::Arc<Vec<u8>> { self.bytes.clone() }

    /// Flat byte buffer of forward-pose dual quaternions, ready to
    /// ship in `FrameUpload.bone_dual_quats`. 32 B per bone per entity.
    pub fn bytes_dq(&self) -> &[u8] { &self.bytes_dq }

    /// Cheap shareable handle to the dual-quat bytes. See [`Self::bytes_arc`].
    pub fn bytes_dq_arc(&self) -> std::sync::Arc<Vec<u8>> { self.bytes_dq.clone() }

    /// Lookup a skinning binding for an entity, or `None` if unskinned.
    pub fn binding(&self, entity: hecs::Entity) -> Option<SkinnedBinding> {
        self.bindings.get(&entity).copied()
    }

    /// Consume the byte ranges that changed in [`Self::bytes`] since
    /// the last [`Self::take_mat_dirty`] call. The sim folds the
    /// returned ranges into the [`crate::render_frame::RenderFrame`]
    /// snapshot it ships to the render thread; the render side
    /// translates them to `queue.write_buffer` calls instead of
    /// rewriting the entire bone-matrix buffer. PERF_DEBT.md D1.
    ///
    /// After this call the allocator's `mat_dirty` is empty — a
    /// subsequent snapshot built without an intervening
    /// [`Self::rebuild`] reports no dirty ranges, so the render side
    /// skips the upload entirely. That handles the C2-narrow path
    /// where `update_scene_gpu` (and therefore `rebuild`) doesn't
    /// run.
    pub fn take_mat_dirty(&mut self) -> arvx_core::DirtyRanges {
        std::mem::take(&mut self.mat_dirty)
    }

    /// Same as [`Self::take_mat_dirty`] for [`Self::bytes_dq`].
    pub fn take_dq_dirty(&mut self) -> arvx_core::DirtyRanges {
        std::mem::take(&mut self.dq_dirty)
    }
}

