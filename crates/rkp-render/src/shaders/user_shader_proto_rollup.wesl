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

@group(0) @binding(0) var<storage, read_write> octree_nodes: array<vec2<u32>>;
@group(1) @binding(0) var<uniform> rollup_u: RollupUniform;

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
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        if (octree_nodes[children_first + i].x != OCTREE_EMPTY) {
            all_empty = false;
            break;
        }
    }
    if (all_empty) {
        octree_nodes[parent_offset] = vec2<u32>(OCTREE_EMPTY, INTERNAL_ATTR_NONE);
    }
}
