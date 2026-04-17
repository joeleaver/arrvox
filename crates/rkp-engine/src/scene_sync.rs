//! Scene-to-GPU synchronization — builds RkpGpuObject arrays from scene state.

use bytemuck::Zeroable;
use glam::{Mat4, Vec3, Vec4};
use rkp_render::rkp_gpu_object::{self, RkpGpuObject};
use rkp_render::{SkinBrickEntry, SkinUniforms, SkinningAssetData};

/// Screen-space AABB for tile culling (pixel coordinates).
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ScreenAabb {
    pub min_x: f32,
    pub min_y: f32,
    pub max_x: f32,
    pub max_y: f32,
}

/// Compute screen-space AABBs for all GPU objects.
/// Projects each object's local AABB (transformed by world matrix) to pixel coordinates.
pub fn compute_screen_aabbs(
    objects: &[RkpGpuObject],
    view_proj: &Mat4,
    width: f32,
    height: f32,
) -> Vec<ScreenAabb> {
    objects.iter().map(|obj| {
        if obj.geom_type == 0 {
            return ScreenAabb::zeroed();
        }

        // Build the 8 corners of the local AABB.
        let extent = f32::from_bits(obj.octree_extent_bits);
        let half = extent * 0.5;
        let world = Mat4::from_cols_array_2d(&obj.world);

        let mut smin = Vec3::splat(f32::MAX);
        let mut smax = Vec3::splat(f32::MIN);

        for corner in 0..8u32 {
            let local = Vec3::new(
                if corner & 1 != 0 { half } else { -half },
                if corner & 2 != 0 { half } else { -half },
                if corner & 4 != 0 { half } else { -half },
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
/// entity carrying a `Skeleton`. Combines bone-matrix indexing
/// (produced by `BoneMatrixAllocator`) with the per-frame bone-field
/// geometry (produced by `plan_skin_dispatch`) so the march shader has
/// everything it needs in one struct.
#[derive(Debug, Copy, Clone)]
pub struct SkinnedBinding {
    /// Number of bones in the skeleton.
    pub bone_count: u32,
    /// Offset into the scene-wide bone-matrix buffer, in `mat4x4<f32>`
    /// units (one mat = 16 f32s = 64 bytes).
    pub bone_buffer_offset: u32,
    /// Offset into `bone_field_buffer` in `vec2<u32>` cells. Matches
    /// the scatter dispatch's uniform.
    pub bone_field_offset: u32,
    /// Bone-field grid dimensions in voxel cells.
    pub bone_field_dims: [u32; 3],
    /// Bone-field grid origin in object-local space.
    pub bone_field_origin: [f32; 3],
    /// Offset into the scene-wide occupancy bitmap in u32 words. Each
    /// bit covers one 4³ brick of this object's bone_field.
    pub bone_field_occ_offset: u32,
}

/// Build an RkpGpuObject from a scene object's transform, spatial handle,
/// and optional skinning binding. When `skinning` is `Some`, the object's
/// `is_skinned` / `bone_count` / `bone_buffer_offset` fields are populated
/// so shaders can index into the bone-matrix buffer.
pub fn build_gpu_object(
    world_matrix: &glam::Mat4,
    aabb: &rkp_core::Aabb,
    spatial: &rkp_core::scene_node::SpatialHandle,
    voxel_size: f32,
    material_id: u16,
    object_id: u32,
    skinning: Option<SkinnedBinding>,
) -> RkpGpuObject {
    let mut gpu = RkpGpuObject::zeroed();
    gpu.world = world_matrix.to_cols_array_2d();
    gpu.inverse_world = world_matrix.inverse().to_cols_array_2d();
    gpu.aabb_min = aabb.min.into();
    gpu.aabb_max = aabb.max.into();
    gpu.voxel_size = voxel_size;
    gpu.material_id = material_id as u32;
    gpu.object_id = object_id;
    gpu.geom_type = rkp_gpu_object::geom_type::VOXELIZED;

    if let rkp_core::scene_node::SpatialHandle::Octree {
        root_offset, depth, base_voxel_size, ..
    } = spatial
    {
        gpu.octree_root = *root_offset;
        gpu.octree_depth = *depth as u32;
        let extent = (1u32 << depth) as f32 * base_voxel_size;
        gpu.octree_extent_bits = extent.to_bits();
    }

    if let Some(skin) = skinning {
        gpu.is_skinned = 1;
        gpu.bone_count = skin.bone_count;
        gpu.bone_buffer_offset = skin.bone_buffer_offset;
        // Rest octree mirrors the runtime octree — the skinned march
        // uses the rest-pose structure for empty-space descent after
        // inverse-skinning each sample.
        gpu.rest_octree_root = gpu.octree_root;
        gpu.rest_octree_depth = gpu.octree_depth;
        gpu.rest_octree_extent_bits = gpu.octree_extent_bits;
        gpu.bone_field_offset = skin.bone_field_offset;
        gpu.bone_field_dim_x = skin.bone_field_dims[0];
        gpu.bone_field_dim_y = skin.bone_field_dims[1];
        gpu.bone_field_dim_z = skin.bone_field_dims[2];
        gpu.bone_field_origin_x = skin.bone_field_origin[0];
        gpu.bone_field_origin_y = skin.bone_field_origin[1];
        gpu.bone_field_origin_z = skin.bone_field_origin[2];
        gpu.bone_field_occ_offset = skin.bone_field_occ_offset;
    }

    gpu
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
#[derive(Default)]
pub struct BoneMatrixAllocator {
    /// Concatenated palettes: per entity, `mat4x4<f32>` forward-then-
    /// inverse, entities in iteration order.
    bytes: Vec<u8>,
    /// Entity → `SkinnedBinding` for the current frame.
    bindings: std::collections::HashMap<hecs::Entity, SkinnedBinding>,
}

impl BoneMatrixAllocator {
    pub fn new() -> Self { Self::default() }

    /// Reset and re-pack every skinned entity's forward + inverse
    /// palettes into the flat byte buffer. Offsets are in
    /// `mat4x4<f32>` units — shader reads
    /// `bone_matrices[offset + i]` for the forward matrix and
    /// `bone_matrices[offset + bone_count + i]` for the inverse.
    pub fn rebuild(&mut self, world: &hecs::World) {
        self.bytes.clear();
        self.bindings.clear();

        let mut running_mat_offset: u32 = 0;
        for (entity, skel) in world.query::<&crate::components::Skeleton>().iter() {
            let bone_count = skel.current_pose.len() as u32;
            if bone_count == 0 {
                continue;
            }
            // Forward palette first, then inverse palette. They must be
            // the same length — `animation::tick` keeps them in sync.
            let fwd: &[u8] = bytemuck::cast_slice(&skel.current_pose);
            let inv: &[u8] = bytemuck::cast_slice(&skel.inverse_pose);
            self.bytes.extend_from_slice(fwd);
            self.bytes.extend_from_slice(inv);
            self.bindings.insert(entity, SkinnedBinding {
                bone_count,
                bone_buffer_offset: running_mat_offset,
                // Bone-field geometry is populated later by the caller
                // after `plan_skin_dispatch` runs. Zero dims = march
                // falls back to rigid path for this object.
                bone_field_offset: 0,
                bone_field_dims: [0, 0, 0],
                bone_field_origin: [0.0, 0.0, 0.0],
                bone_field_occ_offset: 0,
            });
            // Advance past both palettes.
            running_mat_offset += bone_count * 2;
        }
    }

    /// Flat byte buffer ready to ship in `FrameUpload.bone_matrices`.
    pub fn bytes(&self) -> &[u8] { &self.bytes }

    /// Lookup a skinning binding for an entity, or `None` if unskinned.
    pub fn binding(&self, entity: hecs::Entity) -> Option<SkinnedBinding> {
        self.bindings.get(&entity).copied()
    }
}

/// One entity's scatter-dispatch data. Assembled by
/// [`plan_skin_dispatch`] and consumed by the engine's render loop to
/// drive `SkinDeformPass::dispatch`.
pub struct PlannedSkinDispatch {
    pub uniforms: SkinUniforms,
    pub bricks: Vec<SkinBrickEntry>,
}

/// Max allowed extent of any skinned entity's bone field along any
/// single axis, measured in voxel cells. Characters are usually much
/// taller than they are deep, so the per-axis cap is permissive; the
/// `MAX_BONE_FIELD_CELLS` total-volume cap below is what actually
/// guards against absurd memory use (e.g. a 4m boss at 2mm voxels).
const MAX_BONE_FIELD_DIM: u32 = 1024;

/// Max total cell count across all three axes per skinned entity.
/// 32M cells × 8 B/cell = 256 MB per entity — generous for a single
/// character, deliberately constraining so a misconfigured voxel tier
/// doesn't eat a multi-GB GPU budget. A 1.82m character at 5mm voxels
/// fits comfortably (≈9M cells); a 2m character at 2mm voxels hits
/// the limit (≈1B cells) and falls back to rigid with a console
/// warning prompting a coarser voxel tier.
const MAX_BONE_FIELD_CELLS: u64 = 32_000_000;

/// 4³-cell bricks for the deformed bone field's occupancy bitmap.
/// Matches the scatter shader's 4×4×4 workgroup size.
pub const OCC_BRICK_DIM: u32 = 4;

/// Build the per-frame scatter plan for one skinned entity.
///
/// Returns `None` when the entity's deformed AABB degenerates to a
/// single point (no non-trivial bone weights) or exceeds
/// `MAX_BONE_FIELD_DIM`. The caller advances `running_bone_field_cells`
/// (for the dense field) and `running_bone_field_occ_u32s` (for the
/// packed brick bitmap) by this dispatch's sizes on `Some(_)`.
pub fn plan_skin_dispatch(
    bone_buffer_offset: u32,
    bone_count: u32,
    current_pose: &[Mat4],
    skinning_asset: &SkinningAssetData,
    voxel_size: f32,
    running_bone_field_cells: &mut u32,
    running_bone_field_occ_u32s: &mut u32,
) -> Option<PlannedSkinDispatch> {
    // Deformed AABB = union(current_pose[i] × rest_bone_aabbs[i])
    // over every bone that has a non-empty rest AABB. Rest AABBs in
    // object-local voxel space; current_pose transforms rest space to
    // deformed space (the same LBS frame the scatter shader operates
    // in).
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for (bone_idx, aabb) in skinning_asset.rest_bone_aabbs.iter().enumerate() {
        // Skip bones with no voxel influence (zero-extent sentinels).
        let ext = [aabb[3] - aabb[0], aabb[4] - aabb[1], aabb[5] - aabb[2]];
        if ext[0] <= 0.0 && ext[1] <= 0.0 && ext[2] <= 0.0 { continue; }
        let mat = current_pose.get(bone_idx).copied().unwrap_or(Mat4::IDENTITY);
        // Transform the 8 AABB corners — LBS is linear per-bone, so
        // union of transformed corners = transformed AABB exactly.
        for corner in 0..8u32 {
            let x = if corner & 1 != 0 { aabb[3] } else { aabb[0] };
            let y = if corner & 2 != 0 { aabb[4] } else { aabb[1] };
            let z = if corner & 4 != 0 { aabb[5] } else { aabb[2] };
            let p = mat.transform_point3(Vec3::new(x, y, z));
            min = min.min(p);
            max = max.max(p);
        }
    }

    if !min.is_finite() || !max.is_finite() || min.x > max.x {
        return None; // no bone has any voxels
    }

    // Inflate by one voxel on each side so the 8-neighbour scatter at
    // joints has room to land without clipping.
    min -= Vec3::splat(voxel_size);
    max += Vec3::splat(voxel_size);

    // Quantise origin to the voxel grid so the scatter's floor() gives
    // integer cell indices with no half-voxel bias.
    let quant = |v: f32| (v / voxel_size).floor() * voxel_size;
    let grid_origin = Vec3::new(quant(min.x), quant(min.y), quant(min.z));
    let extent = max - grid_origin;
    let dims_f = extent / voxel_size;
    let dims = [
        (dims_f.x.ceil() as u32).max(1),
        (dims_f.y.ceil() as u32).max(1),
        (dims_f.z.ceil() as u32).max(1),
    ];

    if dims.iter().any(|&d| d > MAX_BONE_FIELD_DIM) {
        eprintln!(
            "[scene_sync] skin dispatch skipped — deformed dims {:?} exceed per-axis cap {}. \
             Re-import at a coarser voxel tier.",
            dims, MAX_BONE_FIELD_DIM,
        );
        return None;
    }

    let cell_count_u64 = dims[0] as u64 * dims[1] as u64 * dims[2] as u64;
    if cell_count_u64 > MAX_BONE_FIELD_CELLS {
        eprintln!(
            "[scene_sync] skin dispatch skipped — deformed dims {:?} = {} cells, over {}-cell cap. \
             Re-import at a coarser voxel tier.",
            dims, cell_count_u64, MAX_BONE_FIELD_CELLS,
        );
        return None;
    }
    let cell_count = cell_count_u64 as u32;

    // Brick-level occupancy bitmap: one bit per 4³ cell brick.
    let brick_dims = [
        (dims[0] + OCC_BRICK_DIM - 1) / OCC_BRICK_DIM,
        (dims[1] + OCC_BRICK_DIM - 1) / OCC_BRICK_DIM,
        (dims[2] + OCC_BRICK_DIM - 1) / OCC_BRICK_DIM,
    ];
    let brick_count = brick_dims[0] as u64 * brick_dims[1] as u64 * brick_dims[2] as u64;
    let occ_u32_count = ((brick_count + 31) / 32) as u32;

    let uniforms = SkinUniforms {
        bone_buffer_offset,
        bone_count,
        bone_field_offset: *running_bone_field_cells,
        bone_field_dim_x: dims[0],
        bone_field_dim_y: dims[1],
        bone_field_dim_z: dims[2],
        grid_origin_x: grid_origin.x,
        grid_origin_y: grid_origin.y,
        grid_origin_z: grid_origin.z,
        voxel_size,
        bone_field_occ_offset: *running_bone_field_occ_u32s,
        _pad0: 0, _pad1: 0, _pad2: 0, _pad3: 0, _pad4: 0,
    };
    *running_bone_field_cells = running_bone_field_cells.saturating_add(cell_count);
    *running_bone_field_occ_u32s = running_bone_field_occ_u32s.saturating_add(occ_u32_count);

    // `uniform_idx` is filled in by `SkinBatchScratch::push` when the
    // dispatch is folded into the per-frame batch — we can't know it
    // here without knowing our position in the batch.
    let bricks: Vec<SkinBrickEntry> = skinning_asset.bricks.iter()
        .map(|b| SkinBrickEntry {
            brick_id: b.brick_id,
            origin_x: b.origin[0],
            origin_y: b.origin[1],
            origin_z: b.origin[2],
            uniform_idx: 0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        })
        .collect();

    Some(PlannedSkinDispatch { uniforms, bricks })
}
