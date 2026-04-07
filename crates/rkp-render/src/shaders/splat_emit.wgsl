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

// Group 1: output face instances + draw args
@group(1) @binding(0) var<storage, read_write> face_instances: array<FaceInstance>;
@group(1) @binding(1) var<storage, read_write> draw_args: array<atomic<u32>, 4>;

// Group 2: emit params
@group(2) @binding(0) var<uniform> emit_params: EmitParams;

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

// --- Main ---
//
// GPU emit is unused — face emission is done on CPU (see splat_raster_pass.rs).
// This shader exists only because the compute pipeline is still created.

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // No-op: CPU emit handles face generation for per-voxel octree.
}
