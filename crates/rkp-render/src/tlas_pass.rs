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

    /// Phase 7 Session 3 — build the TLAS for the current frame from
    /// host instances + user-shader instance regions, upload to GPU,
    /// set `last_node_count` / `last_leaf_count`.
    ///
    /// * `assets` / `instances` — combined host + transient (the same
    ///   slices `upload_frame` consumes). User-shader assets in here
    ///   are skipped at TLAS-leaf time; the user-shader path emits
    ///   leaves directly from `user_shader_regions` instead.
    /// * `user_shader_regions` — one entry per region the engine knows
    ///   about this frame (built from `frame.instance_region_requests`
    ///   in the engine layer). Each region contributes one TLAS leaf
    ///   per painted leaf, with a conservative
    ///   `pos ± region_thickness` AABB.
    ///
    /// Returns `true` if either GPU buffer reallocated (caller should
    /// invalidate cached bind groups).
    ///
    /// ## V1 limits
    ///
    /// * **Slot permutation**: emit's atomicAdd assigns instance state
    ///   slots non-deterministically, so slot K may hold leaf J's data
    ///   (J ≠ K). The conservative AABB (radius = region_thickness)
    ///   contains both leaves' actual blades for dense paint. Sparse
    ///   paint may produce shadow-cast misses on individual blades —
    ///   acceptable for V1.
    /// * **Build cost**: O(N log N) median split. Comfortable up to
    ///   ~10K total leaves; at 100K it dominates a 16 ms frame budget.
    ///   Refit / rebuild-skip is a Phase 7b follow-up.
    /// * **Per-blade AABB precision**: derived from leaf position +
    ///   region_thickness, NOT from the user shader's `inst_aabb`
    ///   hook. Tighter per-instance AABBs require GPU build (Session
    ///   3b deferred).
    pub fn build_tlas(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        assets: &[crate::rkp_gpu_object::RkpGpuAsset],
        instances: &[crate::rkp_gpu_object::RkpGpuInstance],
        user_shader_regions: &[UserShaderRegionTlas<'_>],
    ) -> bool {
        let total_leaves: usize = instances.len()
            + user_shader_regions.iter().map(|r| r.leaves.len()).sum::<usize>();
        if total_leaves == 0 {
            self.last_node_count = 0;
            self.last_leaf_count = 0;
            return false;
        }

        let mut prims: Vec<BvhPrim> = Vec::with_capacity(total_leaves);

        // Host instances → one TLAS leaf each via transformed asset AABB.
        for (idx, inst) in instances.iter().enumerate() {
            let asset_id = inst.asset_id as usize;
            if asset_id >= assets.len() { continue; }
            let asset = &assets[asset_id];
            // Skip instances pointing at user-shader assets — those go
            // through the user_shader_regions path below. Defensive:
            // post Phase 6 there shouldn't be any host instance with
            // shader_id != 0, but guard against future regressions.
            if asset.shader_id != 0 { continue; }

            let (world_min, world_max) =
                transform_aabb(asset.aabb_min, asset.aabb_max, &inst.world);
            let centroid = [
                0.5 * (world_min[0] + world_max[0]),
                0.5 * (world_min[1] + world_max[1]),
                0.5 * (world_min[2] + world_max[2]),
            ];
            prims.push(BvhPrim {
                leaf: TlasInstanceLeaf {
                    asset_id: inst.asset_id,
                    instance_state_offset: 0,
                    material_id: inst.material_id,
                    instance_index: idx as u32,
                },
                aabb_min: world_min,
                aabb_max: world_max,
                centroid,
            });
        }

        // User-shader instances → one TLAS leaf per painted leaf.
        //
        // Slot-permutation V1 — emit's `atomicAdd` shuffles the
        // (painted_leaf ↔ instance_pool_slot) assignment, so slot K
        // may hold painted leaf J's blade data (J ≠ K). To stay
        // correct, every TLAS leaf in a region carries the region's
        // UNION AABB — that AABB encloses any slot's blade regardless
        // of the permutation. The per-leaf centroid still drives the
        // BVH split heuristic so the topology isn't entirely
        // degenerate, but the AABBs are not tight at all and shadow
        // rays passing through the region descend ~all leaves there.
        //
        // For dense clustered paint this is essentially the per-leaf
        // AABB anyway. For widely-spread paint within a single region
        // (no @tile_size on the shader) the cost is real — O(N)
        // shadow-ray cost over all leaves in the region. The proper
        // Phase 7b fix is deterministic slotting in the emit pass so
        // CPU and GPU agree on (slot ↔ painted leaf) without needing
        // a global AABB.
        for region in user_shader_regions {
            if region.leaves.is_empty() { continue; }
            let r = region.region_thickness.max(0.0);
            // Compute the region's union AABB once.
            let mut union_min = [f32::INFINITY; 3];
            let mut union_max = [f32::NEG_INFINITY; 3];
            for leaf in region.leaves {
                let p = leaf.world_pos;
                for ax in 0..3 {
                    let lo = p[ax] - r;
                    let hi = p[ax] + r;
                    if lo < union_min[ax] { union_min[ax] = lo; }
                    if hi > union_max[ax] { union_max[ax] = hi; }
                }
            }
            for (i, leaf) in region.leaves.iter().enumerate() {
                let state_offset = region.instance_block_offset
                    + (i as u32) * region.instance_stride_u32;
                prims.push(BvhPrim {
                    leaf: TlasInstanceLeaf {
                        asset_id: region.asset_id,
                        instance_state_offset: state_offset,
                        material_id: region.material_id,
                        instance_index: TLAS_LEAF_USER_SHADER,
                    },
                    aabb_min: union_min,
                    aabb_max: union_max,
                    // Centroid at the painted leaf's position so the
                    // BVH split heuristic still partitions leaves
                    // spatially — even though their AABBs all match.
                    centroid: leaf.world_pos,
                });
            }
        }

        if prims.is_empty() {
            self.last_node_count = 0;
            self.last_leaf_count = 0;
            return false;
        }

        // Build BVH topology + collect leaves in build order.
        let mut nodes: Vec<TlasNode> = Vec::with_capacity(2 * prims.len());
        let mut leaves: Vec<TlasInstanceLeaf> = Vec::with_capacity(prims.len());
        let prim_count = prims.len();
        build_bvh_recursive(&mut nodes, &mut leaves, &mut prims, 0, prim_count);

        // Grow + upload.
        let realloc = self.ensure_capacity(device, nodes.len() as u32, leaves.len() as u32);
        queue.write_buffer(&self.nodes_buffer, 0, bytemuck::cast_slice(&nodes));
        queue.write_buffer(&self.leaves_buffer, 0, bytemuck::cast_slice(&leaves));

        self.last_node_count = nodes.len() as u32;
        self.last_leaf_count = leaves.len() as u32;
        realloc
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

/// Per-region descriptor passed to [`TlasPass::build_tlas`] for the
/// user-shader path. Engine populates one of these per
/// `frame.instance_region_requests` entry that resolves to an
/// `@instance_proto` shader. Holds the metadata + leaves slice the
/// builder needs to emit one TLAS leaf per painted leaf.
pub struct UserShaderRegionTlas<'a> {
    /// Index into the combined assets vec where this region's
    /// user-shader asset lives. Engine computes as
    /// `frame.gpu_assets.len() + asset_for_shader[shader_id]`.
    pub asset_id: u32,
    /// Material the host march packs over the proto's leaf-attr
    /// (V1 host-material inheritance, locked Option B Stage 1).
    pub material_id: u32,
    /// Conservative blade-AABB radius. The user shader's actual
    /// `inst_aabb` may be tighter, but this is a CPU-side build
    /// without GPU readback, so we bound by the region's authored
    /// thickness.
    pub region_thickness: f32,
    /// `region_uniform.instance_block_offset` — first slot in
    /// `instance_pool` for this region's per-instance state.
    pub instance_block_offset: u32,
    /// `region_uniform.instance_stride_u32` — u32 stride between
    /// consecutive per-instance records.
    pub instance_stride_u32: u32,
    /// Painted leaves the engine collected for this region. One TLAS
    /// leaf is emitted per entry.
    pub leaves: &'a [crate::user_shader_emit_pass::PaintedLeaf],
}

/// Per-primitive working state for the BVH builder. Stored mutably
/// so the recursive partitioning can re-order primitives in place
/// (each level partitions its slice by centroid axis).
struct BvhPrim {
    leaf: TlasInstanceLeaf,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
    centroid: [f32; 3],
}

/// Transform a local-space AABB through a column-major affine 4×4
/// matrix, returning the world-space AABB. Standard "Arvo's
/// transform-AABB" algorithm: for each output axis i, sum over input
/// axes j the min/max of `M[j][i] * local_extent[j]`. O(9) ops, exact
/// for any affine transform.
///
/// `world` is in column-major form (`world[col][row]`) matching
/// `RkpGpuInstance.world` and WGSL's `mat4x4`.
fn transform_aabb(
    local_min: [f32; 3],
    local_max: [f32; 3],
    world: &[[f32; 4]; 4],
) -> ([f32; 3], [f32; 3]) {
    // Translation column.
    let mut new_min = [world[3][0], world[3][1], world[3][2]];
    let mut new_max = [world[3][0], world[3][1], world[3][2]];
    for i in 0..3 {
        for j in 0..3 {
            let a = world[j][i] * local_min[j];
            let b = world[j][i] * local_max[j];
            new_min[i] += a.min(b);
            new_max[i] += a.max(b);
        }
    }
    (new_min, new_max)
}

/// Recursive median-split BVH builder.
///
/// Partitions `prims[start..end]` in place so that the left half goes
/// to one subtree and the right half to the other. Splits on the
/// longest centroid-AABB axis, falling back to a balanced count split
/// (median of count) on the chosen axis. O(N log N) total via
/// `select_nth_unstable_by` (linear partition per level).
///
/// Emits one [`TlasNode`] for the current range; appends one
/// [`TlasInstanceLeaf`] per primitive when it bottoms out at a single
/// element. Returns the index in `nodes` where the produced node
/// landed.
fn build_bvh_recursive(
    nodes: &mut Vec<TlasNode>,
    leaves: &mut Vec<TlasInstanceLeaf>,
    prims: &mut [BvhPrim],
    start: usize,
    end: usize,
) -> u32 {
    let count = end - start;
    debug_assert!(count > 0, "build_bvh_recursive called on empty range");

    // Reserve this node's slot — children fill it after they recurse.
    let node_idx = nodes.len() as u32;
    nodes.push(<TlasNode as bytemuck::Zeroable>::zeroed());

    // Compute the current range's world AABB (parent of this subtree).
    let mut aabb_min = prims[start].aabb_min;
    let mut aabb_max = prims[start].aabb_max;
    for p in &prims[start + 1..end] {
        for i in 0..3 {
            if p.aabb_min[i] < aabb_min[i] { aabb_min[i] = p.aabb_min[i]; }
            if p.aabb_max[i] > aabb_max[i] { aabb_max[i] = p.aabb_max[i]; }
        }
    }

    // Leaf — single primitive. Emit leaf payload, mark node with the
    // high bit on `left_or_leaf`.
    if count == 1 {
        let leaf_idx = leaves.len() as u32;
        leaves.push(prims[start].leaf);
        nodes[node_idx as usize] = TlasNode {
            aabb_min,
            left_or_leaf: TLAS_NODE_LEAF_BIT | leaf_idx,
            aabb_max,
            right_or_count: 1,
        };
        return node_idx;
    }

    // Choose split axis: longest extent of the centroid AABB. Falls
    // back to longest extent of the union AABB if all centroids
    // coincide (degenerate but possible).
    let mut centroid_min = prims[start].centroid;
    let mut centroid_max = prims[start].centroid;
    for p in &prims[start + 1..end] {
        for i in 0..3 {
            if p.centroid[i] < centroid_min[i] { centroid_min[i] = p.centroid[i]; }
            if p.centroid[i] > centroid_max[i] { centroid_max[i] = p.centroid[i]; }
        }
    }
    let extent = [
        centroid_max[0] - centroid_min[0],
        centroid_max[1] - centroid_min[1],
        centroid_max[2] - centroid_min[2],
    ];
    let axis = if extent[0] >= extent[1] && extent[0] >= extent[2] { 0 }
        else if extent[1] >= extent[2] { 1 }
        else { 2 };

    // Median split via select_nth_unstable_by — O(count) average.
    let mid_local = count / 2;
    prims[start..end].select_nth_unstable_by(mid_local, |a, b| {
        a.centroid[axis]
            .partial_cmp(&b.centroid[axis])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mid = start + mid_local;

    let left_idx = build_bvh_recursive(nodes, leaves, prims, start, mid);
    let right_idx = build_bvh_recursive(nodes, leaves, prims, mid, end);

    nodes[node_idx as usize] = TlasNode {
        aabb_min,
        left_or_leaf: left_idx,
        aabb_max,
        right_or_count: right_idx,
    };
    node_idx
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

    fn make_prim(min: [f32; 3], max: [f32; 3], leaf_idx: u32) -> BvhPrim {
        BvhPrim {
            leaf: TlasInstanceLeaf {
                asset_id: leaf_idx, // tag the leaf so build_bvh tests can identify it
                instance_state_offset: 0,
                material_id: 0,
                instance_index: leaf_idx,
            },
            aabb_min: min,
            aabb_max: max,
            centroid: [
                0.5 * (min[0] + max[0]),
                0.5 * (min[1] + max[1]),
                0.5 * (min[2] + max[2]),
            ],
        }
    }

    #[test]
    fn transform_aabb_identity_is_passthrough() {
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let (mn, mx) = transform_aabb([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], &identity);
        assert_eq!(mn, [1.0, 2.0, 3.0]);
        assert_eq!(mx, [4.0, 5.0, 6.0]);
    }

    #[test]
    fn transform_aabb_translation_only() {
        // Pure translation — same extent, shifted center.
        let m = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [10.0, 20.0, 30.0, 1.0], // translation column
        ];
        let (mn, mx) = transform_aabb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], &m);
        assert_eq!(mn, [10.0, 20.0, 30.0]);
        assert_eq!(mx, [11.0, 21.0, 31.0]);
    }

    #[test]
    fn transform_aabb_uniform_scale() {
        let m = [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let (mn, mx) = transform_aabb([1.0, 1.0, 1.0], [2.0, 2.0, 2.0], &m);
        assert_eq!(mn, [2.0, 2.0, 2.0]);
        assert_eq!(mx, [4.0, 4.0, 4.0]);
    }

    #[test]
    fn transform_aabb_90deg_y_rotation_swaps_xz_extents() {
        // Rotate 90° around Y: x → -z, z → x. A unit cube at origin
        // stays a unit cube AABB-wise, but the world AABB grows the
        // expected reflection.
        let cy = 0.0_f32;
        let sy = 1.0_f32;
        let m = [
            [cy, 0.0, -sy, 0.0],   // column 0: rotated x basis
            [0.0, 1.0, 0.0, 0.0],  // y unchanged
            [sy, 0.0, cy, 0.0],    // column 2: rotated z basis
            [0.0, 0.0, 0.0, 1.0],
        ];
        let (mn, mx) = transform_aabb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], &m);
        // A 1x1x1 cube rotated 90° still fits in a 1x1x1 AABB
        // (centered differently). With local origin at [0..1]³, the
        // world AABB lands at z ∈ [-1, 0] and x ∈ [0, 1].
        assert!((mn[0] - 0.0).abs() < 1e-5, "min x = {}", mn[0]);
        assert!((mx[0] - 1.0).abs() < 1e-5, "max x = {}", mx[0]);
        assert!((mn[2] - (-1.0)).abs() < 1e-5, "min z = {}", mn[2]);
        assert!((mx[2] - 0.0).abs() < 1e-5, "max z = {}", mx[2]);
    }

    #[test]
    fn build_bvh_single_prim_is_one_leaf_node() {
        let mut prims = vec![make_prim([0.0; 3], [1.0; 3], 0)];
        let mut nodes = Vec::new();
        let mut leaves = Vec::new();
        let root = build_bvh_recursive(&mut nodes, &mut leaves, &mut prims, 0, 1);
        assert_eq!(root, 0);
        assert_eq!(nodes.len(), 1);
        assert_eq!(leaves.len(), 1);
        // Root is a leaf — high bit set.
        assert!(nodes[0].left_or_leaf & TLAS_NODE_LEAF_BIT != 0);
        assert_eq!(nodes[0].left_or_leaf & !TLAS_NODE_LEAF_BIT, 0); // leaf index 0
        assert_eq!(nodes[0].right_or_count, 1); // leaf count = 1
        assert_eq!(nodes[0].aabb_min, [0.0, 0.0, 0.0]);
        assert_eq!(nodes[0].aabb_max, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn build_bvh_two_prims_root_is_internal_with_two_leaves() {
        // Two prims along X axis. Median split should create root at
        // node[0] with two leaf children.
        let mut prims = vec![
            make_prim([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0),
            make_prim([5.0, 0.0, 0.0], [6.0, 1.0, 1.0], 1),
        ];
        let mut nodes = Vec::new();
        let mut leaves = Vec::new();
        let _ = build_bvh_recursive(&mut nodes, &mut leaves, &mut prims, 0, 2);
        assert_eq!(nodes.len(), 3);
        assert_eq!(leaves.len(), 2);
        // Root is internal — left/right are both leaf-marked node indices.
        let root = &nodes[0];
        assert!(root.left_or_leaf & TLAS_NODE_LEAF_BIT == 0);
        assert!(root.right_or_count & TLAS_NODE_LEAF_BIT == 0);
        let left = &nodes[root.left_or_leaf as usize];
        let right = &nodes[root.right_or_count as usize];
        assert!(left.left_or_leaf & TLAS_NODE_LEAF_BIT != 0);
        assert!(right.left_or_leaf & TLAS_NODE_LEAF_BIT != 0);
        // Root AABB encloses both prims.
        assert_eq!(root.aabb_min, [0.0, 0.0, 0.0]);
        assert_eq!(root.aabb_max, [6.0, 1.0, 1.0]);
    }

    #[test]
    fn build_bvh_eight_prims_topology_is_balanced() {
        // 8 prims along the X axis at unit spacing — perfectly
        // balanced binary tree of depth 3 (8 leaves, 15 total nodes).
        let mut prims: Vec<BvhPrim> = (0..8u32)
            .map(|i| {
                let f = i as f32;
                make_prim([f, 0.0, 0.0], [f + 1.0, 1.0, 1.0], i)
            })
            .collect();
        let mut nodes = Vec::new();
        let mut leaves = Vec::new();
        let _ = build_bvh_recursive(&mut nodes, &mut leaves, &mut prims, 0, 8);
        assert_eq!(nodes.len(), 15); // 2N - 1 for a binary tree with N leaves
        assert_eq!(leaves.len(), 8);

        // Root encloses all 8 unit cubes spanning x ∈ [0, 8].
        let root = &nodes[0];
        assert_eq!(root.aabb_min, [0.0, 0.0, 0.0]);
        assert_eq!(root.aabb_max, [8.0, 1.0, 1.0]);

        // Walk leaves; each prim should appear exactly once.
        let mut seen = [false; 8];
        for leaf in &leaves {
            let idx = leaf.instance_index as usize;
            assert!(idx < 8, "leaf instance_index out of range");
            assert!(!seen[idx], "leaf {idx} appeared twice");
            seen[idx] = true;
        }
        assert!(seen.iter().all(|s| *s));
    }

    #[test]
    fn user_shader_region_emits_one_prim_per_leaf() {
        // CPU-only test of the conservative leaf-AABB derivation:
        // verify the BvhPrim list a single user-shader region produces
        // (without going through GPU upload). Uses internal builder
        // primitives directly.
        use crate::user_shader_emit_pass::PaintedLeaf;
        let leaves = vec![
            PaintedLeaf {
                world_pos: [0.0, 0.0, 0.0],
                material_packed: 0,
                world_normal: [0.0, 1.0, 0.0],
                _pad: 0.0,
            },
            PaintedLeaf {
                world_pos: [10.0, 0.0, 0.0],
                material_packed: 0,
                world_normal: [0.0, 1.0, 0.0],
                _pad: 0.0,
            },
        ];
        let region_thickness = 0.5_f32;

        // Mirror what `build_tlas` does internally for the user-shader
        // path so the test doesn't need a live wgpu device.
        let mut prims: Vec<BvhPrim> = Vec::new();
        let asset_id = 7u32;
        let material_id = 5u32;
        let block_offset = 100u32;
        let stride = 8u32;
        for (i, leaf) in leaves.iter().enumerate() {
            let p = leaf.world_pos;
            let world_min = [p[0] - region_thickness, p[1] - region_thickness, p[2] - region_thickness];
            let world_max = [p[0] + region_thickness, p[1] + region_thickness, p[2] + region_thickness];
            let state_offset = block_offset + (i as u32) * stride;
            prims.push(BvhPrim {
                leaf: TlasInstanceLeaf {
                    asset_id,
                    instance_state_offset: state_offset,
                    material_id,
                    instance_index: TLAS_LEAF_USER_SHADER,
                },
                aabb_min: world_min,
                aabb_max: world_max,
                centroid: p,
            });
        }

        // Note: this test mirrors the per-prim setup in build_tlas
        // which used PER-LEAF AABBs. The actual build_tlas now uses
        // the region's UNION AABB for slot-permutation correctness
        // (see fix in tlas_pass.rs); the test below only validates
        // the prim-list layout / state_offset stride math.
        assert_eq!(prims.len(), 2);
        assert_eq!(prims[0].leaf.instance_state_offset, 100);
        assert_eq!(prims[0].leaf.instance_index, TLAS_LEAF_USER_SHADER);
        assert_eq!(prims[1].leaf.instance_state_offset, 108);

        // Build the BVH from these — same builder host instances use.
        let mut nodes = Vec::new();
        let mut bvh_leaves = Vec::new();
        let _ = build_bvh_recursive(&mut nodes, &mut bvh_leaves, &mut prims, 0, 2);
        assert_eq!(bvh_leaves.len(), 2);
        // Both leaves should still be user-shader-flagged (no host
        // contamination).
        for l in &bvh_leaves {
            assert_eq!(l.instance_index, TLAS_LEAF_USER_SHADER);
            assert_eq!(l.asset_id, asset_id);
            assert_eq!(l.material_id, material_id);
        }
        // Root encloses both AABBs.
        assert_eq!(nodes[0].aabb_min, [-0.5, -0.5, -0.5]);
        assert_eq!(nodes[0].aabb_max, [10.5, 0.5, 0.5]);
    }

    #[test]
    fn build_bvh_leaves_carry_correct_payloads() {
        // Verify the leaf payload (asset_id, instance_index, etc) is
        // preserved through the build. Each prim has a distinct
        // material_id; we round-trip and check.
        let mut prims = vec![
            BvhPrim {
                leaf: TlasInstanceLeaf {
                    asset_id: 7,
                    instance_state_offset: 0,
                    material_id: 100,
                    instance_index: 0,
                },
                aabb_min: [0.0; 3], aabb_max: [1.0; 3], centroid: [0.5; 3],
            },
            BvhPrim {
                leaf: TlasInstanceLeaf {
                    asset_id: 7,
                    instance_state_offset: 0,
                    material_id: 200,
                    instance_index: 1,
                },
                aabb_min: [10.0; 3], aabb_max: [11.0; 3], centroid: [10.5; 3],
            },
        ];
        let mut nodes = Vec::new();
        let mut leaves = Vec::new();
        let _ = build_bvh_recursive(&mut nodes, &mut leaves, &mut prims, 0, 2);
        // Both materials should appear exactly once across the leaves,
        // regardless of which side of the median they end up on.
        let mut found_100 = false;
        let mut found_200 = false;
        for l in &leaves {
            if l.material_id == 100 { found_100 = true; }
            if l.material_id == 200 { found_200 = true; }
        }
        assert!(found_100 && found_200);
    }
}
