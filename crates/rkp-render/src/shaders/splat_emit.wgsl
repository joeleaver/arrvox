// Emit compute shader — path-parallel octree traversal + isosurface face emission.
//
// Each thread represents one potential leaf path through the octree. Thread ID
// encodes the octant sequence: at level L, octant = (tid >> 3*(depth-1-L)) & 7.
// Most threads hit EMPTY/INTERIOR early and exit. Threads that reach leaves
// check 6 neighbors for opacity 0.5 crossings and emit faces at the
// interpolated isosurface position.
//
// Dispatch: (ceil(8^max_depth / WORKGROUP_SIZE), object_count, 1)
//   X dimension = path index, Y dimension = object index.

// --- Constants ---
const OCTREE_EMPTY: u32 = 0xFFFFFFFFu;
const OCTREE_INTERIOR: u32 = 0xFFFFFFFEu;
const OCTREE_LEAF_BIT: u32 = 0x80000000u;
const OPACITY_THRESHOLD: f32 = 0.5;
const WORKGROUP_SIZE: u32 = 256u;

// --- Structs ---

struct FaceInstance {
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    voxel_size: f32,
    voxel_slot: u32,
    packed: u32,
}

struct VoxelSample {
    word0: u32,
    word1: u32,
}

// Must match RkpGpuObject layout (256 bytes).
struct RkpObject {
    world: mat4x4<f32>,
    aabb_min: vec3<f32>,
    octree_root: u32,
    aabb_max: vec3<f32>,
    octree_depth: u32,
    octree_extent_bits: u32,
    voxel_size: f32,
    material_id: u32,
    object_id: u32,
    geom_type: u32,
    is_skinned: u32,
    bone_count: u32,
    bone_buffer_offset: u32,
    rest_octree_root: u32,
    rest_octree_depth: u32,
    rest_octree_extent_bits: u32,
    deformed_pool_offset: u32,
    layer_mask: u32,
    _pad0: u32, _pad1: u32, _pad2: u32, _pad3: u32,
    _pad4: u32, _pad5: u32, _pad6: u32, _pad7: u32,
    _pad8: u32, _pad9: u32, _pad10: u32,
}

struct EmitParams {
    max_faces: u32,
    object_count: u32,
    max_depth: u32,
    _pad1: u32,
}

// --- Bindings ---

@group(0) @binding(0) var<storage, read> voxel_pool: array<VoxelSample>;
@group(0) @binding(1) var<storage, read> octree_nodes: array<u32>;
@group(0) @binding(2) var<storage, read> objects: array<RkpObject>;

@group(1) @binding(0) var<storage, read_write> face_instances: array<FaceInstance>;
@group(1) @binding(1) var<storage, read_write> draw_args: array<atomic<u32>, 4>;

@group(2) @binding(0) var<uniform> emit_params: EmitParams;

// --- Helpers ---

fn extract_opacity(word0: u32) -> f32 {
    return clamp(unpack2x16float(word0 & 0xFFFFu).x, 0.0, 1.0);
}

// Look up opacity at a voxel coordinate by traversing the octree from root.
fn lookup_opacity(coord: vec3<i32>, root: u32, max_depth: u32, extent: f32, vs: f32) -> f32 {
    let max_coord = i32(1u << max_depth);
    if any(coord < vec3<i32>(0)) || any(coord >= vec3<i32>(max_coord)) {
        return 0.0;
    }

    // Position = center of the voxel at this coordinate.
    let pos = vec3<f32>(coord) * vs + vs * 0.5;

    var offset = root;
    var half = extent * 0.5;
    var center = vec3<f32>(half);

    for (var level = 0u; level < max_depth; level++) {
        let node = octree_nodes[offset];
        if node == OCTREE_EMPTY { return 0.0; }
        if node == OCTREE_INTERIOR { return 1.0; }
        if (node & OCTREE_LEAF_BIT) != 0u {
            let slot = node & ~OCTREE_LEAF_BIT;
            return extract_opacity(voxel_pool[slot].word0);
        }
        let gt = vec3<u32>(pos >= center);
        let child = gt.x + gt.y * 2u + gt.z * 4u;
        offset = node + child;
        half *= 0.5;
        center += vec3<f32>(
            select(-half, half, pos.x >= center.x),
            select(-half, half, pos.y >= center.y),
            select(-half, half, pos.z >= center.z),
        );
    }

    let node = octree_nodes[offset];
    if node == OCTREE_EMPTY { return 0.0; }
    if node == OCTREE_INTERIOR { return 1.0; }
    if (node & OCTREE_LEAF_BIT) != 0u {
        let slot = node & ~OCTREE_LEAF_BIT;
        return extract_opacity(voxel_pool[slot].word0);
    }
    return 0.0;
}

// Neighbor offsets: 0=-X, 1=+X, 2=-Y, 3=+Y, 4=-Z, 5=+Z
fn face_offset(face_id: u32) -> vec3<i32> {
    switch face_id {
        case 0u: { return vec3<i32>(-1, 0, 0); }
        case 1u: { return vec3<i32>( 1, 0, 0); }
        case 2u: { return vec3<i32>( 0,-1, 0); }
        case 3u: { return vec3<i32>( 0, 1, 0); }
        case 4u: { return vec3<i32>( 0, 0,-1); }
        case 5u: { return vec3<i32>( 0, 0, 1); }
        default: { return vec3<i32>(0, 0, 0); }
    }
}

// --- Main ---

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let obj_idx = wid.y;
    if obj_idx >= emit_params.object_count { return; }

    let obj = objects[obj_idx];
    let root = obj.octree_root;
    let depth = obj.octree_depth;
    let extent = bitcast<f32>(obj.octree_extent_bits);
    let vs = obj.voxel_size;

    // Thread's path index from X and Z dimensions.
    // X covers up to 65535 workgroups (16.7M threads), Z extends beyond that.
    let path_id = gid.x + wid.z * 65535u * WORKGROUP_SIZE;
    let total_paths = 1u << min(3u * depth, 31u); // 8^depth, clamped to u32 range
    if path_id >= total_paths { return; }

    // Walk the octree following the path encoded in path_id.
    // At each level, the octant is extracted from path_id's bit pattern.
    var offset = root;
    var coord = vec3<u32>(0u);
    var leaf_slot = 0u;
    var leaf_depth = depth;
    var found_leaf = false;

    for (var level = 0u; level < depth; level++) {
        let node = octree_nodes[offset];

        if node == OCTREE_EMPTY || node == OCTREE_INTERIOR {
            return; // No leaf on this path.
        }

        if (node & OCTREE_LEAF_BIT) != 0u {
            // Leaf found at a coarser level than max depth.
            leaf_slot = node & ~OCTREE_LEAF_BIT;
            leaf_depth = level;
            found_leaf = true;
            break;
        }

        // Branch: extract octant from path_id and descend.
        let shift = 3u * (depth - 1u - level);
        let octant = (path_id >> shift) & 7u;
        let half = 1u << (depth - 1u - level);
        coord += vec3<u32>(
            (octant & 1u) * half,
            ((octant >> 1u) & 1u) * half,
            ((octant >> 2u) & 1u) * half,
        );
        offset = node + octant;
    }

    // Check the node at max depth if we haven't found a leaf yet.
    if !found_leaf {
        let node = octree_nodes[offset];
        if (node & OCTREE_LEAF_BIT) != 0u {
            leaf_slot = node & ~OCTREE_LEAF_BIT;
            leaf_depth = depth;
            found_leaf = true;
        }
    }

    if !found_leaf { return; }

    // For coarse leaves (leaf_depth < depth), only process once:
    // the first path within this leaf's subtree (all lower bits zero).
    if leaf_depth < depth {
        let sub_bits = 3u * (depth - leaf_depth);
        let sub_mask = (1u << sub_bits) - 1u;
        if (path_id & sub_mask) != 0u {
            return; // Another thread handles this coarse leaf.
        }
    }

    // Read this voxel's opacity.
    let self_opacity = extract_opacity(voxel_pool[leaf_slot].word0);
    if self_opacity <= 0.01 { return; }

    // Voxel center in octree space.
    let depth_diff = depth - leaf_depth;
    let leaf_vs = vs * f32(1u << depth_diff);
    let self_center = vec3<f32>(coord) * vs + leaf_vs * 0.5;

    let icoord = vec3<i32>(coord);

    // Check 6 neighbors — emit face when bordering empty/low-opacity.
    for (var face = 0u; face < 6u; face++) {
        let off = face_offset(face);
        let nb_coord = icoord + off;
        let nb_opacity = lookup_opacity(nb_coord, root, depth, extent, vs);

        if nb_opacity <= 0.01 {
            let face_idx = atomicAdd(&draw_args[1], 1u);
            if face_idx < emit_params.max_faces {
                face_instances[face_idx] = FaceInstance(
                    self_center.x,
                    self_center.y,
                    self_center.z,
                    leaf_vs,
                    leaf_slot,
                    (face & 0x7u) | ((obj_idx & 0xFFFFFu) << 3u),
                );
            }
        }
    }
}
