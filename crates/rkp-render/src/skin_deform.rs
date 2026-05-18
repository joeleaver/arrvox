//! Skin-deform planning types — wire format only.
//!
//! Historical: this module owned the scatter compute pipeline that
//! wrote a deformed-space bone field for the now-retired march path
//! to sample. The mesh path skins in the vertex shader against the
//! per-frame `bone_matrices` / `bone_dual_quats` buffers directly, so
//! the pipeline is gone. The engine still threads the *planning*
//! structs through sim → render frame snapshots for backwards
//! compatibility — they're inert until Phase 2 cleanup audit removes
//! the dead pipeline end-to-end.

/// Per-entity sim → render skin uniform.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkinUniforms {
    pub bone_buffer_offset: u32,
    pub bone_count: u32,
    pub bone_field_offset: u32,
    pub bone_field_dim_x: u32,
    pub bone_field_dim_y: u32,
    pub bone_field_dim_z: u32,
    pub grid_origin_x: f32,
    pub grid_origin_y: f32,
    pub grid_origin_z: f32,
    pub voxel_size: f32,
    pub bone_field_occ_offset: u32,
    pub skinning_mode: u32,
    pub bone_dq_offset: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// One entry in the per-entity brick list.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkinBrickEntry {
    pub brick_id: u32,
    pub origin_x: u32,
    pub origin_y: u32,
    pub origin_z: u32,
    pub uniform_idx: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// One per-entity plan.
pub struct SkinDispatch<'a> {
    pub uniforms: SkinUniforms,
    pub bricks: &'a [SkinBrickEntry],
}

/// Scratch the sim reuses each frame.
#[derive(Default, Clone)]
pub struct SkinBatchScratch {
    pub uniforms: Vec<SkinUniforms>,
    pub bricks: Vec<SkinBrickEntry>,
}

impl SkinBatchScratch {
    pub fn clear(&mut self) {
        self.uniforms.clear();
        self.bricks.clear();
    }

    pub fn push(&mut self, dispatch: SkinDispatch<'_>) {
        let idx = self.uniforms.len() as u32;
        self.uniforms.push(dispatch.uniforms);
        for b in dispatch.bricks {
            let mut entry = *b;
            entry.uniform_idx = idx;
            self.bricks.push(entry);
        }
    }

    pub fn is_empty(&self) -> bool {
        self.uniforms.is_empty()
    }
}
