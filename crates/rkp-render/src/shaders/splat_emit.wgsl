// Emit compute shader — traverses octrees and emits transition face quad instances.
//
// One workgroup per visible object. Each workgroup traverses the object's octree
// using a stack, and for each leaf brick, reads the occupancy bitmask and emits
// face instances for exposed surface voxel faces.
//
// Output: FaceInstance array + DrawIndirectArgs (atomic instance count).

// --- Octree node constants (must match octree_common.wgsl) ---
const OCTREE_EMPTY: u32 = 0xFFFFFFFFu;
const OCTREE_INTERIOR: u32 = 0xFFFFFFFEu;
const OCTREE_LEAF_BIT: u32 = 0x80000000u;

// DrawIndirectArgs as raw u32 array for atomic access.
// Layout: [0]=vertex_count, [1]=instance_count (atomic), [2]=first_vertex, [3]=first_instance

// --- Face instance: data passed to the raster vertex shader ---
struct FaceInstance {
    // World position of the voxel center.
    world_pos_x: f32,
    world_pos_y: f32,
    world_pos_z: f32,
    // Voxel size in world units (varies with octree depth).
    voxel_size: f32,
    // Brick pool slot for this leaf.
    brick_slot: u32,
    // Packed: voxel_index(9 bits) | face_id(3 bits) | obj_idx(16 bits) | unused(4 bits)
    packed: u32,
}

// --- Octree stack entry for traversal ---
struct StackEntry {
    // Node offset in the octree buffer.
    node_offset: u32,
    // World-space center of this node's region.
    center_x: f32,
    center_y: f32,
    center_z: f32,
    // Half-extent of this node's region (one axis).
    half_extent: f32,
    // Current depth level.
    level: u32,
}

// --- Bindings ---

// Group 0: scene data (must match GpuScene bind group layout)
// binding 0 = brick_pool (not needed by emit, but must be in layout)
@group(0) @binding(1) var<storage, read> octree_nodes: array<u32>;
@group(0) @binding(2) var<storage, read> objects: array<GpuObject>;

// Group 1: surface shell occupancy bitmasks
@group(1) @binding(0) var<storage, read> surface_shell: array<u32>;

// Group 2: output face instances + draw args
@group(2) @binding(0) var<storage, read_write> face_instances: array<FaceInstance>;
@group(2) @binding(1) var<storage, read_write> draw_args: array<atomic<u32>, 4>;

// Group 3: emit params
@group(3) @binding(0) var<uniform> emit_params: EmitParams;

struct EmitParams {
    max_faces: u32,
    object_count: u32,
    _pad0: u32,
    _pad1: u32,
}

// --- Minimal GpuObject struct (only fields we need) ---
struct GpuObject {
    inverse_world: mat4x4<f32>,     // offset 0
    aabb_min: vec4<f32>,            // offset 64
    aabb_max: vec4<f32>,            // offset 80
    octree_root: u32,               // offset 96  (brick_map_offset)
    octree_depth: u32,              // offset 100 (brick_map_dims[0])
    octree_extent_bits: u32,        // offset 104 (brick_map_dims[1])
    _reserved_dims_z: u32,          // offset 108
    voxel_size: f32,                // offset 112
    material_id: u32,               // offset 116
    geom_type: u32,                 // offset 120
    // ... remaining fields omitted (256 bytes total, padding handles the rest)
    _pad: array<u32, 33>,
}

// --- Helpers ---

fn is_occupancy_set(slot: u32, voxel_idx: u32) -> bool {
    // surface_shell is indexed as 16 u32s per slot (512 bits).
    let word_offset = slot * 16u + (voxel_idx / 32u);
    let bit = voxel_idx % 32u;
    return (surface_shell[word_offset] & (1u << bit)) != 0u;
}

fn pack_face_instance(voxel_idx: u32, face_id: u32, obj_idx: u32) -> u32 {
    return (voxel_idx & 0x1FFu) | ((face_id & 0x7u) << 9u) | ((obj_idx & 0xFFFFu) << 12u);
}

fn voxel_xyz(idx: u32) -> vec3<u32> {
    return vec3<u32>(idx % 8u, (idx / 8u) % 8u, idx / 64u);
}

/// Check if the neighbor voxel in the given direction is empty.
/// face_id: 0=-X, 1=+X, 2=-Y, 3=+Y, 4=-Z, 5=+Z
fn is_face_exposed(slot: u32, vx: u32, vy: u32, vz: u32, face_id: u32) -> bool {
    var nx = i32(vx);
    var ny = i32(vy);
    var nz = i32(vz);

    switch face_id {
        case 0u: { nx -= 1; }
        case 1u: { nx += 1; }
        case 2u: { ny -= 1; }
        case 3u: { ny += 1; }
        case 4u: { nz -= 1; }
        case 5u: { nz += 1; }
        default: {}
    }

    // If neighbor is outside the brick, treat as exposed.
    if nx < 0 || nx >= 8 || ny < 0 || ny >= 8 || nz < 0 || nz >= 8 {
        return true;
    }

    let neighbor_idx = u32(nx) + u32(ny) * 8u + u32(nz) * 64u;
    return !is_occupancy_set(slot, neighbor_idx);
}

// --- Main ---

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let obj_idx = gid.x;
    if obj_idx >= emit_params.object_count {
        return;
    }

    let obj = objects[obj_idx];
    let octree_root = obj.octree_root;
    let max_depth = obj.octree_depth;
    let extent = bitcast<f32>(obj.octree_extent_bits);
    let base_vs = obj.voxel_size;

    // Compute world transform (inverse of inverse_world).
    // For emit, we need to transform voxel positions to world space.
    // We'll compute world positions from local positions using the inverse_world
    // matrix's inverse. Since we only need position (not ray direction), we can
    // reconstruct the world matrix.
    //
    // For now, we compute local-space positions and transform them.
    let inv_world = obj.inverse_world;

    // Stack-based octree traversal.
    var stack: array<StackEntry, 32>;
    var sp = 1u;
    let half_extent = extent * 0.5;
    stack[0] = StackEntry(octree_root, half_extent, half_extent, half_extent, half_extent, 0u);

    while sp > 0u {
        sp -= 1u;
        let entry = stack[sp];
        let node = octree_nodes[entry.node_offset];

        if node == OCTREE_EMPTY {
            continue;
        }
        if node == OCTREE_INTERIOR {
            // Entire subtree is solid — no surface faces inside (only at boundaries,
            // which the parent handles). Skip.
            continue;
        }

        if (node & OCTREE_LEAF_BIT) != 0u {
            // Leaf: emit faces for this brick.
            let slot = node & 0x7FFFFFFFu;
            let depth_diff = max_depth - entry.level;
            let leaf_vs = base_vs * f32(1u << depth_diff);
            let brick_extent = leaf_vs * 8.0;

            // Local-space origin of this brick (lower corner).
            let brick_origin = vec3<f32>(
                entry.center_x - entry.half_extent,
                entry.center_y - entry.half_extent,
                entry.center_z - entry.half_extent,
            );

            // Iterate occupied voxels and emit exposed faces.
            for (var vz = 0u; vz < 8u; vz++) {
                for (var vy = 0u; vy < 8u; vy++) {
                    for (var vx = 0u; vx < 8u; vx++) {
                        let voxel_idx = vx + vy * 8u + vz * 64u;
                        if !is_occupancy_set(slot, voxel_idx) {
                            continue;
                        }

                        // Voxel center in local space.
                        let local_center = brick_origin + (vec3<f32>(f32(vx), f32(vy), f32(vz)) + 0.5) * leaf_vs;

                        // Transform to world space.
                        // inv_world transforms world → local, so we need its inverse.
                        // For rigid transforms: world_pos = transpose(R) * (local_pos - t)
                        // but this is complex. Instead, just pass local center + obj_idx
                        // and let the vertex shader do the transform.

                        for (var face = 0u; face < 6u; face++) {
                            if !is_face_exposed(slot, vx, vy, vz, face) {
                                continue;
                            }

                            let idx = atomicAdd(&draw_args[1], 1u); // [1] = instance_count
                            if idx >= emit_params.max_faces {
                                // Overflow — stop emitting.
                                return;
                            }

                            face_instances[idx] = FaceInstance(
                                local_center.x,
                                local_center.y,
                                local_center.z,
                                leaf_vs,
                                slot,
                                pack_face_instance(voxel_idx, face, obj_idx),
                            );
                        }
                    }
                }
            }
            continue;
        }

        // Branch: push 8 children onto the stack.
        let children_offset = node;
        let child_half = entry.half_extent * 0.5;

        // Push in reverse order so octant 0 is processed first.
        for (var i = 7u; ; i--) {
            let dx = f32(i & 1u);
            let dy = f32((i >> 1u) & 1u);
            let dz = f32((i >> 2u) & 1u);

            let child_center = vec3<f32>(
                entry.center_x + (dx * 2.0 - 1.0) * child_half,
                entry.center_y + (dy * 2.0 - 1.0) * child_half,
                entry.center_z + (dz * 2.0 - 1.0) * child_half,
            );

            if sp < 32u {
                stack[sp] = StackEntry(
                    children_offset + i,
                    child_center.x,
                    child_center.y,
                    child_center.z,
                    child_half,
                    entry.level + 1u,
                );
                sp += 1u;
            }

            if i == 0u { break; }
        }
    }
}
