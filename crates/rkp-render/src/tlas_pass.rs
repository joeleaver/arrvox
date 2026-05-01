//! Phase 7 Session 1 — TLAS (Top-Level Acceleration Structure) for shadow rays.
//!
//! Owns the GPU-resident BVH that the shadow trace pass will traverse
//! once Sessions 2–4 ship. Session 1 is foundation only: type
//! definitions, buffer storage, capacity tracking, capacity-grow
//! helpers. No build pipeline, no traversal — those come in 2 and 4
//! respectively.
//!
//! ## Wire format
//!
//! Two parallel buffers:
//!
//! * `tlas_nodes` — [`TlasNode`] (32 B). Internal nodes carry left + right
//!   child indices. Leaf nodes set the high bit of `left_or_leaf` and
//!   index into `tlas_leaves`.
//! * `tlas_leaves` — [`TlasInstanceLeaf`] (16 B). One per instance.
//!   Carries enough metadata for the shadow trace to dispatch its
//!   per-instance octree descent without `instances[]` lookup for
//!   user-shader instances and with one for host instances.
//!
//! ## Sizing
//!
//! V1 starts with 256 B placeholders so the buffers exist for binding
//! validation. [`TlasPass::ensure_capacity`] grows them on demand each
//! frame as the builder fills. Worst-case estimate: 200 K instances
//! → ~400 K nodes × 32 B = 12.8 MB + 200 K leaves × 16 B = 3.2 MB.
//! Comfortably within budget.

/// One BVH node — internal or leaf, distinguished by the high bit of
/// `left_or_leaf`.
///
/// ```text
/// Internal: left_or_leaf      = left child node index
///           right_or_count    = right child node index
/// Leaf:     left_or_leaf      = 0x80000000 | leaf_index    (into tlas_leaves)
///           right_or_count    = leaf count                  (V1: always 1)
/// ```
///
/// 32 bytes; vec3 alignment in WGSL packs the trailing u32 into the
/// same 16-byte slot as the vec3, so the {vec3, u32} pairs hold
/// without padding. Mirror in `tlas_*.wgsl` once Session 2 introduces
/// the WGSL.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TlasNode {
    pub aabb_min: [f32; 3],
    pub left_or_leaf: u32,
    pub aabb_max: [f32; 3],
    pub right_or_count: u32,
}

const _: () = assert!(std::mem::size_of::<TlasNode>() == 32);

/// High bit marker — set on `left_or_leaf` to flag a leaf node.
pub const TLAS_NODE_LEAF_BIT: u32 = 0x8000_0000;

/// Per-instance leaf payload. 16 bytes — same shape as
/// [`crate::octree_march::UserShaderTileEntry`] except `instance_index`
/// replaces the unused trailing pad. Holds enough info for the shadow
/// trace to set up per-instance ray descent without re-reading the
/// `instances[]` storage buffer for user-shader instances.
///
/// * Host instance: `instance_index = idx into instances[]`,
///   `instance_state_offset = 0` (ignored), the `instances[]` lookup
///   provides world matrix + skinning state.
/// * User-shader instance: `instance_index = TLAS_LEAF_USER_SHADER`,
///   `instance_state_offset` points into `instance_pool`, the user's
///   `inst_to_local` / `inst_aabb` hooks own the world↔local map.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TlasInstanceLeaf {
    pub asset_id: u32,
    pub instance_state_offset: u32,
    pub material_id: u32,
    pub instance_index: u32,
}

const _: () = assert!(std::mem::size_of::<TlasInstanceLeaf>() == 16);

/// `instance_index` sentinel for user-shader leaves (no `instances[]`
/// entry to look up). Distinct from the
/// [`crate::octree_march`]'s `USER_SHADER_PICK_SENTINEL` (0xFFFFFFFE)
/// only by purpose; values can collide harmlessly since both are
/// "this isn't a real instances[] index" markers.
pub const TLAS_LEAF_USER_SHADER: u32 = 0xFFFF_FFFEu32;

/// Initial buffer placeholder size in bytes. Both buffers grow via
/// [`TlasPass::ensure_capacity`] as the builder fills. Sized for one
/// node + one leaf so the buffers exist for bind-group validation
/// before the first build.
pub const TLAS_INITIAL_BYTES: u64 = 256;

/// Owner of the TLAS GPU buffers. Buffers grow on demand via
/// [`TlasPass::ensure_capacity`]; capacity tracking mirrors the
/// pattern used by `OctreeMarchPass` for its tile-list buffers.
///
/// V1 carries only the buffers + capacities. Sessions 2–4 add the
/// CPU builder, GPU upload, and WGSL traversal.
pub struct TlasPass {
    /// `array<TlasNode>` — BVH topology.
    pub nodes_buffer: wgpu::Buffer,
    pub nodes_capacity_bytes: u64,
    /// `array<TlasInstanceLeaf>` — per-instance leaf payloads.
    pub leaves_buffer: wgpu::Buffer,
    pub leaves_capacity_bytes: u64,
    /// Highest node count uploaded to the GPU last frame. Read by
    /// future traversal to know the root layout (root is always at
    /// node 0; this is just for the empty-frame skip path).
    pub last_node_count: u32,
    /// Highest leaf count uploaded last frame.
    pub last_leaf_count: u32,
}

impl TlasPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let nodes_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tlas nodes"),
            size: TLAS_INITIAL_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let leaves_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tlas leaves"),
            size: TLAS_INITIAL_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            nodes_buffer,
            nodes_capacity_bytes: TLAS_INITIAL_BYTES,
            leaves_buffer,
            leaves_capacity_bytes: TLAS_INITIAL_BYTES,
            last_node_count: 0,
            last_leaf_count: 0,
        }
    }


    /// Grow `nodes_buffer` and `leaves_buffer` to fit `node_count` and
    /// `leaf_count`. Reallocates with capacity doubling on overflow,
    /// mirroring `OctreeMarchPass::ensure_us_tile_grid_capacity`.
    /// Returns `true` if either buffer reallocated — caller is
    /// responsible for invalidating any cached bind group that
    /// references these buffers.
    pub fn ensure_capacity(
        &mut self,
        device: &wgpu::Device,
        node_count: u32,
        leaf_count: u32,
    ) -> bool {
        let nodes_needed = (node_count.max(1) as u64) * (std::mem::size_of::<TlasNode>() as u64);
        let leaves_needed =
            (leaf_count.max(1) as u64) * (std::mem::size_of::<TlasInstanceLeaf>() as u64);
        let mut dirty = false;
        if nodes_needed > self.nodes_capacity_bytes {
            let mut new_cap = self.nodes_capacity_bytes.max(TLAS_INITIAL_BYTES);
            while new_cap < nodes_needed {
                new_cap = new_cap.saturating_mul(2);
            }
            self.nodes_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("tlas nodes"),
                size: new_cap,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.nodes_capacity_bytes = new_cap;
            dirty = true;
        }
        if leaves_needed > self.leaves_capacity_bytes {
            let mut new_cap = self.leaves_capacity_bytes.max(TLAS_INITIAL_BYTES);
            while new_cap < leaves_needed {
                new_cap = new_cap.saturating_mul(2);
            }
            self.leaves_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("tlas leaves"),
                size: new_cap,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.leaves_capacity_bytes = new_cap;
            dirty = true;
        }
        dirty
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tlas_node_size_is_32() {
        assert_eq!(std::mem::size_of::<TlasNode>(), 32);
    }

    #[test]
    fn tlas_instance_leaf_size_is_16() {
        assert_eq!(std::mem::size_of::<TlasInstanceLeaf>(), 16);
    }

    #[test]
    fn tlas_node_leaf_bit_is_high_bit() {
        // Distinct from any reasonable child-index value (max useful
        // index is the per-frame node count, well under 2^31).
        assert_eq!(TLAS_NODE_LEAF_BIT, 1u32 << 31);
    }

    #[test]
    fn tlas_node_field_offsets_match_wgsl() {
        // Sanity-check the {vec3, u32} packing matches WGSL's std430
        // alignment: vec3 has size 12 and align 16, so the trailing
        // u32 packs into the same 16-byte slot.
        let n = TlasNode {
            aabb_min: [1.0, 2.0, 3.0],
            left_or_leaf: 0xAABB,
            aabb_max: [4.0, 5.0, 6.0],
            right_or_count: 0xCCDD,
        };
        let bytes = bytemuck::bytes_of(&n);
        assert_eq!(bytes.len(), 32);
        assert_eq!(&bytes[0..4], 1.0_f32.to_le_bytes());
        assert_eq!(&bytes[8..12], 3.0_f32.to_le_bytes());
        assert_eq!(&bytes[12..16], 0xAABB_u32.to_le_bytes());
        assert_eq!(&bytes[16..20], 4.0_f32.to_le_bytes());
        assert_eq!(&bytes[24..28], 6.0_f32.to_le_bytes());
        assert_eq!(&bytes[28..32], 0xCCDD_u32.to_le_bytes());
    }

    #[test]
    fn tlas_instance_leaf_field_layout() {
        let leaf = TlasInstanceLeaf {
            asset_id: 0x1111,
            instance_state_offset: 0x2222,
            material_id: 0x3333,
            instance_index: 0x4444,
        };
        let bytes = bytemuck::bytes_of(&leaf);
        assert_eq!(bytes.len(), 16);
        assert_eq!(&bytes[0..4], 0x1111_u32.to_le_bytes());
        assert_eq!(&bytes[4..8], 0x2222_u32.to_le_bytes());
        assert_eq!(&bytes[8..12], 0x3333_u32.to_le_bytes());
        assert_eq!(&bytes[12..16], 0x4444_u32.to_le_bytes());
    }

}
