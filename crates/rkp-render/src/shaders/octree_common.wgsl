// Shared octree traversal functions for GPU shaders.
//
// Concatenated into all RKIPatch shader overrides (shadow/AO, radiance inject,
// shading, emit, raster). Provides octree point queries and trilinear sampling
// that replace flat brick_maps[] lookups.
//
// Node encoding (1 u32 per node):
//   0xFFFFFFFF = EMPTY (subtree is all empty air)
//   0xFFFFFFFE = INTERIOR (subtree is fully opaque)
//   Bit 31 clear = BRANCH (value is offset to 8 contiguous children)
//   Bit 31 set, < 0xFFFFFFFE = LEAF (value & 0x7FFFFFFF is brick pool slot)

const OCTREE_EMPTY: u32 = 0xFFFFFFFFu;
const OCTREE_INTERIOR: u32 = 0xFFFFFFFEu;
const OCTREE_LEAF_BIT: u32 = 0x80000000u;

struct OctreeResult {
    /// Brick pool slot, or OCTREE_EMPTY / OCTREE_INTERIOR sentinel.
    slot: u32,
    /// Depth at which the result was found (0 = root, max = finest).
    depth: u32,
}

/// Test if a node value is a branch (offset to 8 children).
fn octree_is_branch(node: u32) -> bool {
    return (node & OCTREE_LEAF_BIT) == 0u && node != OCTREE_EMPTY && node != OCTREE_INTERIOR;
}

/// Test if a node value is a leaf (brick pool slot in lower 31 bits).
fn octree_is_leaf(node: u32) -> bool {
    return (node & OCTREE_LEAF_BIT) != 0u && node != OCTREE_EMPTY && node != OCTREE_INTERIOR;
}

/// Extract brick pool slot from a leaf node.
fn octree_leaf_slot(node: u32) -> u32 {
    return node & ~OCTREE_LEAF_BIT;
}

/// Compute the octant index (0-7) from a position relative to a center.
fn octree_octant(pos: vec3<f32>, center: vec3<f32>) -> u32 {
    let gt = vec3<u32>(pos >= center);
    return gt.x + gt.y * 2u + gt.z * 4u;
}

/// Look up the octree node at a local-space position.
///
/// `root`: offset to root node in octree_nodes buffer.
/// `max_depth`: tree depth.
/// `extent`: world-space extent of the root node (one axis).
/// `local_pos`: position in object-local space (0,0,0 is octree origin).
///
/// Returns the leaf brick_slot (or sentinel) and the depth at which it was found.
fn octree_lookup(root: u32, max_depth: u32, extent: f32, local_pos: vec3<f32>) -> OctreeResult {
    // GpuObject fields (reinterpreted):
    //   brick_map_offset → root
    //   brick_map_dims[0] → max_depth
    //   brick_map_dims[1] → bitcast<f32>(extent) stored as u32, passed as extent

    var offset = root;
    var half = extent * 0.5;
    var center = vec3<f32>(half, half, half);

    for (var level = 0u; level < max_depth; level++) {
        let node = octree_nodes[offset];

        if node == OCTREE_EMPTY {
            return OctreeResult(OCTREE_EMPTY, level);
        }
        if node == OCTREE_INTERIOR {
            return OctreeResult(OCTREE_INTERIOR, level);
        }
        if octree_is_leaf(node) {
            return OctreeResult(octree_leaf_slot(node), level);
        }

        // Branch: pick octant and descend.
        let child = octree_octant(local_pos, center);
        offset = node + child;

        half *= 0.5;
        let signs = vec3<f32>(
            select(-1.0, 1.0, local_pos.x >= center.x),
            select(-1.0, 1.0, local_pos.y >= center.y),
            select(-1.0, 1.0, local_pos.z >= center.z),
        );
        center += signs * half;
    }

    // At max depth — read the leaf.
    let node = octree_nodes[offset];
    if octree_is_leaf(node) {
        return OctreeResult(octree_leaf_slot(node), max_depth);
    }
    if node == OCTREE_INTERIOR {
        return OctreeResult(OCTREE_INTERIOR, max_depth);
    }
    return OctreeResult(OCTREE_EMPTY, max_depth);
}

/// Sample opacity at a local-space position using octree traversal.
///
/// Returns the opacity value (0.0 for EMPTY, 1.0 for INTERIOR, or sampled
/// from brick pool for leaf nodes).
fn sample_opacity_at_octree(
    local_pos: vec3<f32>,
    obj_root: u32,
    obj_depth: u32,
    obj_extent: f32,
    obj_voxel_size: f32,
) -> f32 {
    let result = octree_lookup(obj_root, obj_depth, obj_extent, local_pos);

    if result.slot == OCTREE_EMPTY {
        return 0.0;
    }
    if result.slot == OCTREE_INTERIOR {
        return 1.0;
    }

    // Leaf: compute the effective voxel size at this depth.
    let depth_diff = obj_depth - result.depth;
    let leaf_voxel_size = obj_voxel_size * f32(1u << depth_diff);

    // Compute voxel index within the brick.
    let leaf_extent = leaf_voxel_size * 8.0;
    // Position of this leaf's origin (snapped to leaf grid).
    let leaf_grid = floor(local_pos / leaf_extent) * leaf_extent;
    let local_in_brick = (local_pos - leaf_grid) / leaf_voxel_size;
    let vx = clamp(u32(local_in_brick.x), 0u, 7u);
    let vy = clamp(u32(local_in_brick.y), 0u, 7u);
    let vz = clamp(u32(local_in_brick.z), 0u, 7u);

    let voxel_idx = vx + vy * 8u + vz * 64u;
    let word0 = brick_pool[result.slot * 512u + voxel_idx].word0;
    return clamp(unpack2x16float(word0 & 0xFFFFu).x, 0.0, 1.0);
}

/// 8-tap trilinear sampling of opacity using octree traversal.
///
/// Samples the 8 corners of a voxel-sized cube around `local_pos` and
/// interpolates. Uses the same pattern as the flat `sample_opacity_trilinear`
/// but with octree lookups instead of flat brick map indexing.
fn sample_opacity_trilinear_octree(
    local_pos: vec3<f32>,
    obj_root: u32,
    obj_depth: u32,
    obj_extent: f32,
    obj_voxel_size: f32,
) -> f32 {
    let vs = obj_voxel_size;
    let half = vs * 0.5;

    // 8 corners of a voxel-sized cube centered on local_pos.
    let p000 = local_pos + vec3<f32>(-half, -half, -half);
    let p100 = local_pos + vec3<f32>( half, -half, -half);
    let p010 = local_pos + vec3<f32>(-half,  half, -half);
    let p110 = local_pos + vec3<f32>( half,  half, -half);
    let p001 = local_pos + vec3<f32>(-half, -half,  half);
    let p101 = local_pos + vec3<f32>( half, -half,  half);
    let p011 = local_pos + vec3<f32>(-half,  half,  half);
    let p111 = local_pos + vec3<f32>( half,  half,  half);

    let s000 = sample_opacity_at_octree(p000, obj_root, obj_depth, obj_extent, vs);
    let s100 = sample_opacity_at_octree(p100, obj_root, obj_depth, obj_extent, vs);
    let s010 = sample_opacity_at_octree(p010, obj_root, obj_depth, obj_extent, vs);
    let s110 = sample_opacity_at_octree(p110, obj_root, obj_depth, obj_extent, vs);
    let s001 = sample_opacity_at_octree(p001, obj_root, obj_depth, obj_extent, vs);
    let s101 = sample_opacity_at_octree(p101, obj_root, obj_depth, obj_extent, vs);
    let s011 = sample_opacity_at_octree(p011, obj_root, obj_depth, obj_extent, vs);
    let s111 = sample_opacity_at_octree(p111, obj_root, obj_depth, obj_extent, vs);

    // Fractional position within the voxel.
    let f = fract(local_pos / vs + 0.5);

    // Trilinear interpolation.
    let x00 = mix(s000, s100, f.x);
    let x10 = mix(s010, s110, f.x);
    let x01 = mix(s001, s101, f.x);
    let x11 = mix(s011, s111, f.x);

    let y0 = mix(x00, x10, f.y);
    let y1 = mix(x01, x11, f.y);

    return mix(y0, y1, f.z);
}

/// Compute gradient normal via 6-tap central differences of the trilinear
/// opacity field, using octree-based sampling.
fn compute_normal_octree(
    local_pos: vec3<f32>,
    obj_root: u32,
    obj_depth: u32,
    obj_extent: f32,
    obj_voxel_size: f32,
) -> vec3<f32> {
    let eps = obj_voxel_size * 0.5;

    let dx = sample_opacity_trilinear_octree(local_pos + vec3(eps, 0.0, 0.0), obj_root, obj_depth, obj_extent, obj_voxel_size)
           - sample_opacity_trilinear_octree(local_pos - vec3(eps, 0.0, 0.0), obj_root, obj_depth, obj_extent, obj_voxel_size);
    let dy = sample_opacity_trilinear_octree(local_pos + vec3(0.0, eps, 0.0), obj_root, obj_depth, obj_extent, obj_voxel_size)
           - sample_opacity_trilinear_octree(local_pos - vec3(0.0, eps, 0.0), obj_root, obj_depth, obj_extent, obj_voxel_size);
    let dz = sample_opacity_trilinear_octree(local_pos + vec3(0.0, 0.0, eps), obj_root, obj_depth, obj_extent, obj_voxel_size)
           - sample_opacity_trilinear_octree(local_pos - vec3(0.0, 0.0, eps), obj_root, obj_depth, obj_extent, obj_voxel_size);

    // Gradient points from low to high opacity — surface normal points outward
    // (from high to low), so negate.
    let grad = vec3<f32>(dx, dy, dz);
    let len = length(grad);
    if len < 1e-8 {
        return vec3<f32>(0.0, 1.0, 0.0); // fallback up
    }
    return -grad / len;
}
