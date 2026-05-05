// user_shader_proto_rollup.wesl — bottom-up empty-coarsen pass.
//
// The proto bake (`user_shader_proto.wesl`) fills only the leaf level
// (level == max_depth) with brick references or OCTREE_EMPTY. The
// internal levels (0..max_depth-1) are pre-written CPU-side by
// `build_internal_levels` as branch pointers to their 8 children,
// regardless of whether those children are empty.
//
// Without this pass, every miss-march descends from root to leaf to
// discover a cell is empty — the proto's empty space stays at fine
// granularity (one outer-DDA step per fine empty cell). After this
// pass, contiguous empty subtrees collapse into a single OCTREE_EMPTY
// at the highest level where all 8 descendants are empty, so
// `skip_node` jumps across them in one step at march time.
//
// Mirrors the `try_collapse` invariant in
// `rkp-core/src/sparse_octree/insert.rs`. The Rust pass dispatches
// this kernel once per internal level, max_depth-1 down to 0; each
// dispatch covers `8^level` parent nodes.

import package::lib::octree_slot::OCTREE_EMPTY;
import package::lib::octree_slot::OCTREE_LEAF_BIT;
import package::lib::brick::INTERNAL_ATTR_NONE;

struct RollupUniform {
    // Absolute pool offset of the first node at THIS level. For
    // level k of a prototype whose internal levels start at
    // `OCT_BASE`, this is `OCT_BASE + sum(8^i for i in 0..k)` — i.e.
    // `level_starts_inclusive(max_depth)[k]`.
    parent_first_offset: u32,
    // Number of parent nodes at this level = `8^level`.
    parent_count: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> octree_nodes: array<vec4<u32>>;
@group(1) @binding(0) var<uniform> rollup_u: RollupUniform;

// Decode a child node's tight AABB from its packed `.z` / `.w` lanes
// into normalized fractions over the child's own cell. Sentinel
// `(z, w) == (0u, 0u)` (the Step-1 buffer-init default) means "tight
// bounds not set, treat as full cell" — return [0, 1] so the union
// upward is conservative.
fn decode_aabb_lo_norm(z: u32, w: u32) -> vec3<f32> {
    if (z == 0u && w == 0u) {
        return vec3<f32>(0.0);
    }
    return vec3<f32>(
        f32(z & 0xFFu),
        f32((z >> 8u) & 0xFFu),
        f32((z >> 16u) & 0xFFu),
    ) / 255.0;
}
fn decode_aabb_hi_norm(z: u32, w: u32) -> vec3<f32> {
    if (z == 0u && w == 0u) {
        return vec3<f32>(1.0);
    }
    return vec3<f32>(
        f32(w & 0xFFu),
        f32((w >> 8u) & 0xFFu),
        f32((w >> 16u) & 0xFFu),
    ) / 255.0;
}

@compute @workgroup_size(64)
fn proto_rollup_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let parent_idx = gid.x;
    if (parent_idx >= rollup_u.parent_count) { return; }
    let parent_offset = rollup_u.parent_first_offset + parent_idx;

    let parent_slot = octree_nodes[parent_offset].x;
    // Defensive: parent should be a branch pointer at this point. If
    // it's already EMPTY (e.g. the pass ran twice for the same level),
    // nothing to do. LEAF shouldn't appear at internal levels but
    // skip them anyway so a future LOD-prefilter that lifts leaf-attrs
    // up the spine doesn't get clobbered.
    if (parent_slot == OCTREE_EMPTY) { return; }
    if ((parent_slot & OCTREE_LEAF_BIT) != 0u) { return; }
    let children_first = parent_slot;

    var all_empty: bool = true;
    var union_lo = vec3<f32>(1.0e30);
    var union_hi = vec3<f32>(-1.0e30);
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let child_packed = octree_nodes[children_first + i];
        if (child_packed.x == OCTREE_EMPTY) {
            continue;
        }
        all_empty = false;
        // Each child occupies an octant of the parent's [0, 1] cube.
        // Map the child's normalized tight AABB into parent space:
        // octant_i origin is `(i & 1, (i >> 1) & 1, (i >> 2) & 1) *
        // 0.5`; the child's [0, 1] range scales to a 0.5-wide range
        // anchored at that origin.
        let child_lo_norm = decode_aabb_lo_norm(child_packed.z, child_packed.w);
        let child_hi_norm = decode_aabb_hi_norm(child_packed.z, child_packed.w);
        let octant = vec3<f32>(
            f32((i >> 0u) & 1u),
            f32((i >> 1u) & 1u),
            f32((i >> 2u) & 1u),
        ) * 0.5;
        union_lo = min(union_lo, octant + child_lo_norm * 0.5);
        union_hi = max(union_hi, octant + child_hi_norm * 0.5);
    }

    if (all_empty) {
        octree_nodes[parent_offset] = vec4<u32>(OCTREE_EMPTY, INTERNAL_ATTR_NONE, 0u, 0u);
        return;
    }

    // Encode the union into the parent's `.zw`. Floor the lo and ceil
    // the hi to avoid quantization-induced false rejects.
    let lo_x = u32(floor(union_lo.x * 255.0));
    let lo_y = u32(floor(union_lo.y * 255.0));
    let lo_z = u32(floor(union_lo.z * 255.0));
    let hi_x = min(u32(ceil(union_hi.x * 255.0)), 255u);
    let hi_y = min(u32(ceil(union_hi.y * 255.0)), 255u);
    let hi_z = min(u32(ceil(union_hi.z * 255.0)), 255u);
    let aabb_lo = (lo_z << 16u) | (lo_y << 8u) | lo_x;
    let aabb_hi = (hi_z << 16u) | (hi_y << 8u) | hi_x;
    let parent_packed = octree_nodes[parent_offset];
    octree_nodes[parent_offset] = vec4<u32>(
        parent_packed.x,
        parent_packed.y,
        aabb_lo,
        aabb_hi,
    );
}
