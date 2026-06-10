// Phase 7c Session 3 — Karras parallel BVH construction.
//
// Two compute entry points share one bind layout:
//
//   * `build_leaves_main` — one thread per primitive (= leaf).
//     Copies the prim's payload (asset_id / state_offset /
//     material_id / instance_index) into `tlas_leaves[i]`, and
//     writes the leaf-marker node at `tlas_nodes[N-1+i]` with
//     `left_or_leaf = TLAS_NODE_LEAF_BIT | i` and
//     `right_or_count = 1`. AABB is filled in by Session 4's
//     bottom-up propagation.
//
//   * `build_internal_main` — one thread per internal node
//     (0 ≤ i < N-1). Runs Karras' parallel BVH construction (2012)
//     to determine the node's range, split position, and child
//     indices, writing topology into `tlas_nodes[i]`. Children
//     point at either another internal node (`< N-1`) or a leaf-
//     marker node (`≥ N-1`); the shadow trace traversal reads the
//     LEAF_BIT in the LEAF-MARKER's own `left_or_leaf` to
//     distinguish (no special handling needed at this end). AABB
//     filled in by Session 4.
//
// ## Output node layout
//
// `tlas_nodes` is sized for `2N-1` entries:
//   * `[0, N-1)` — internal nodes, written by `build_internal_main`.
//     Root is index 0.
//   * `[N-1, 2N-1)` — leaf-marker nodes, written by `build_leaves_main`.
//
// `tlas_leaves` is sized for `N` entries; each holds one
// `TlasInstanceLeaf` payload sorted in Morton order.
//
// ## Duplicate Morton handling
//
// `delta(i, j)` extends to a 64-bit virtual key
// `(morton[i] << 32) | i` so duplicates produce strictly increasing
// virtual keys, matching Karras' uniqueness assumption.

import package::lib::types::TlasPrim;
import package::lib::types::TlasNode;
import package::lib::types::TlasInstanceLeaf;

const TLAS_NODE_LEAF_BIT: u32 = 0x80000000u;

// Shared state written by `tlas_compute_dispatch_args.wesl`.
// Replaces the old `KarrasUniform` — the per-frame N, internal_wgs,
// and total_node_wgs flow through this buffer, leaving the karras
// shader pipeline with no per-pass uniform of its own.
struct TlasState {
    prim_count: u32,
    radix_workgroups: u32,
    internal_wgs: u32,
    total_node_wgs: u32,
}

@group(0) @binding(0) var<storage, read> sorted_keys: array<u32>;
@group(0) @binding(1) var<storage, read> sorted_vals: array<u32>;
@group(0) @binding(2) var<storage, read> tlas_prims: array<TlasPrim>;
@group(0) @binding(3) var<storage, read_write> tlas_nodes: array<TlasNode>;
@group(0) @binding(4) var<storage, read_write> tlas_leaves: array<TlasInstanceLeaf>;
// Phase 7c Session 4 — parent pointers (legacy: atomic walk-up
// propagation used these). The new iterative propagation
// (Phase 7c.5) doesn't read parents, but `build_internal_main`
// still writes them as a side-effect — kept for now so the
// binding layout is stable. May be retired later.
@group(0) @binding(5) var<storage, read_write> parents: array<u32>;
// Phase 7c.6 — per-component atomic AABB accumulators. Length =
// 3 × (2N-1) for each. `propagate_atomic_main` (one thread per
// leaf) walks up the parent chain and applies atomicMin / atomicMax
// to its ancestors' slots. atomicMin/Max are commutative + idempotent
// across threads, so the final accumulator value is correct
// regardless of thread ordering — no cross-buffer memory visibility
// required (which is what tripped up the previous atomicAdd
// + atomic-walk approach on this user's hardware).
//
// Floats are stored as sortable u32 keys (flip sign for positives,
// flip all bits for negatives) so atomicMin on u32 returns the
// smallest float. `decode_aabb_main` reads back, decodes, writes
// to `tlas_nodes[i].aabb_min/max`.
@group(0) @binding(6) var<storage, read_write> aabb_min_atomic: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> aabb_max_atomic: array<atomic<u32>>;
@group(1) @binding(0) var<uniform> tlas_state: TlasState;

const PARENT_SENTINEL: u32 = 0xFFFFFFFFu;
// Sentinels for the atomic accumulators.
//   FP_MIN_SENTINEL = encoded +infinity (largest finite key for
//                     atomicMin starting point).
//   FP_MAX_SENTINEL = encoded -infinity (smallest finite key for
//                     atomicMax starting point).
const FP_MIN_SENTINEL: u32 = 0xFF800000u;
const FP_MAX_SENTINEL: u32 = 0x007FFFFFu;

// Encode f32 as a u32 sort-key. Positive floats: flip the sign
// bit (0 → 0x80000000 ⇒ 0x80000000 − keeps positive ordering).
// Negative floats: flip all bits (more-negative ⇒ smaller key).
// Resulting u32 ordering matches f32 ordering across the full
// finite range. atomicMin/Max on these keys correctly find
// smallest/largest float values.
fn float_to_sort_key(f: f32) -> u32 {
    let bits = bitcast<u32>(f);
    if ((bits & 0x80000000u) != 0u) {
        return ~bits;
    }
    return bits | 0x80000000u;
}

fn sort_key_to_float(k: u32) -> f32 {
    if ((k & 0x80000000u) != 0u) {
        return bitcast<f32>(k & 0x7FFFFFFFu);
    }
    return bitcast<f32>(~k);
}

// Length of the longest common prefix between virtual keys at
// positions i and j. Out-of-range j returns -1 (sentinel that loses
// to any in-range delta). Virtual key = (morton[i] << 32) | i, so
// duplicate Mortons disambiguate via index — preserves Karras'
// uniqueness guarantee. Returns up to 64.
fn delta(i: i32, j: i32, n: i32) -> i32 {
    if (j < 0 || j >= n) {
        return -1;
    }
    let ki = sorted_keys[u32(i)];
    let kj = sorted_keys[u32(j)];
    if (ki != kj) {
        return i32(countLeadingZeros(ki ^ kj));
    }
    // Mortons collide — fall through to the index tiebreak. The
    // virtual key prefixes share the full 32 Morton bits plus
    // however many leading bits of the index match.
    let xi = u32(i);
    let xj = u32(j);
    return 32 + i32(countLeadingZeros(xi ^ xj));
}

// Sign of `a - b`, in {-1, 0, +1}. Used to pick range direction.
fn sign_i32(a: i32, b: i32) -> i32 {
    if (a > b) { return 1; }
    if (a < b) { return -1; }
    return 0;
}

// ── Leaf packing pass ──────────────────────────────────────────────

@compute @workgroup_size(64, 1, 1)
fn build_leaves_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= tlas_state.prim_count) {
        return;
    }
    let prim_idx = sorted_vals[i];
    let p = tlas_prims[prim_idx];
    var leaf: TlasInstanceLeaf;
    leaf.asset_id = p.asset_id;
    leaf.instance_state_offset = p.instance_state_offset;
    leaf.material_id = p.material_id;
    leaf.instance_index = p.instance_index;
    tlas_leaves[i] = leaf;

    // Leaf-marker node at index N-1+i. Write the leaf AABB
    // directly — only this thread touches this slot, so no race.
    let n = tlas_state.prim_count;
    var leaf_node: TlasNode;
    leaf_node.aabb_min = p.aabb_min;
    leaf_node.aabb_max = p.aabb_max;
    leaf_node.left_or_leaf = TLAS_NODE_LEAF_BIT | i;
    leaf_node.right_or_count = 1u;
    tlas_nodes[n - 1u + i] = leaf_node;
}

// ── Karras internal-node construction ──────────────────────────────

@compute @workgroup_size(64, 1, 1)
fn build_internal_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx_u = gid.x;
    let n_u = tlas_state.prim_count;
    if (n_u < 2u) {
        // Trivial case: 0 or 1 prims. Nothing to build — root is
        // either non-existent or the single leaf-marker (handled
        // by the engine, which sets `tlas_node_count = 0` for
        // empty TLAS and doesn't dispatch this pass for N == 1).
        return;
    }
    if (idx_u >= n_u - 1u) {
        return;
    }
    let idx = i32(idx_u);
    let n = i32(n_u);

    // Direction d ∈ {-1, +1} of the range from idx.
    let d = sign_i32(delta(idx, idx + 1, n), delta(idx, idx - 1, n));

    // Lower bound on common prefix length within the range.
    let delta_min = delta(idx, idx - d, n);

    // Find an upper bound for the range length via doubling.
    var l_max: i32 = 2;
    loop {
        if (delta(idx, idx + l_max * d, n) <= delta_min) { break; }
        l_max = l_max * 2;
        // Safety guard — `delta` returns at most 64 (32-bit Morton +
        // 32-bit index tiebreak), so l_max can't legitimately
        // exceed n. Bail if it does to prevent runaway.
        if (l_max > n) { break; }
    }

    // Binary search for the precise range length.
    var l: i32 = 0;
    var t: i32 = l_max / 2;
    loop {
        if (t < 1) { break; }
        if (delta(idx, idx + (l + t) * d, n) > delta_min) {
            l = l + t;
        }
        t = t / 2;
    }

    // The other end of the range.
    let j = idx + l * d;
    let delta_node = delta(idx, j, n);

    // Find the split position via binary search. `s_step` halves
    // each iteration (rounded up); the loop terminates when
    // `s_step == 0`.
    var s: i32 = 0;
    var divisor: i32 = 2;
    loop {
        let t_split = (l + divisor - 1) / divisor;  // ceil(l / divisor)
        if (delta(idx, idx + (s + t_split) * d, n) > delta_node) {
            s = s + t_split;
        }
        if (t_split <= 1) { break; }
        divisor = divisor * 2;
    }

    // gamma = split position (leaf index just left of the split).
    var gamma: i32;
    if (d > 0) {
        gamma = idx + s;
    } else {
        gamma = idx + s * d - 1;
    }

    // Children:
    //   left  child covers [min(idx,j), gamma]
    //   right child covers [gamma+1, max(idx,j)]
    let range_lo = min(idx, j);
    let range_hi = max(idx, j);

    // If the left child is a single leaf (range_lo == gamma) it
    // points at leaf-marker node N-1+gamma; else it's internal
    // node `gamma`.
    var left_child: u32;
    if (range_lo == gamma) {
        left_child = u32(n - 1 + gamma);
    } else {
        left_child = u32(gamma);
    }
    var right_child: u32;
    if (range_hi == gamma + 1) {
        right_child = u32(n - 1 + gamma + 1);
    } else {
        right_child = u32(gamma + 1);
    }

    var node: TlasNode;
    node.aabb_min = vec3<f32>(0.0);
    node.aabb_max = vec3<f32>(0.0);
    node.left_or_leaf = left_child;
    node.right_or_count = right_child;
    tlas_nodes[u32(idx)] = node;

    // Record parent pointers — Phase 7c Session 4 needs these to
    // walk leaves up to root for AABB propagation. Each child has
    // exactly one parent (this internal node), so no race between
    // sibling internal nodes' writes.
    parents[left_child] = u32(idx);
    parents[right_child] = u32(idx);
}

// ── Atomic-min/max AABB propagation ────────────────────────────────
//
// Phase 7c.6. Replaces the previous atomic-walk + iterative
// approaches. Both relied on cross-buffer memory ordering that
// WGSL doesn't formally guarantee; on this user's hardware the
// race manifested as intermittently-corrupt internal-node AABBs
// (sometimes the root, causing all shadows to disappear).
//
// Insight: atomicMin / atomicMax on the AABB COMPONENTS
// THEMSELVES are commutative + associative + idempotent. The
// final value of the accumulator after all leaves contribute is
// the union AABB, regardless of thread order. No cross-buffer
// ordering required — only the atomic semantics on the addressed
// atomic variable, which WGSL DOES guarantee.
//
// Pipeline:
//   1. `init_atomic_aabb_main` — clear accumulators to ±∞ sentinels.
//   2. `propagate_atomic_main` — one thread per leaf; walks up
//      the parent chain, applying atomicMin/Max to each
//      ancestor's accumulator slots.
//   3. `decode_aabb_main` — one thread per internal node; reads
//      accumulators, decodes, writes to `tlas_nodes[i].aabb_*`.
//
// Three separate compute passes within one encoder; cross-pass
// memory barriers (wgpu's command-buffer guarantee) ensure
// each step's writes are visible to the next.

@compute @workgroup_size(64, 1, 1)
fn init_atomic_aabb_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = tlas_state.prim_count;
    if (n < 2u) {
        return;
    }
    let total_nodes = 2u * n - 1u;
    if (i >= total_nodes) {
        return;
    }
    // Three components per node × min and max accumulator each.
    atomicStore(&aabb_min_atomic[i * 3u + 0u], FP_MIN_SENTINEL);
    atomicStore(&aabb_min_atomic[i * 3u + 1u], FP_MIN_SENTINEL);
    atomicStore(&aabb_min_atomic[i * 3u + 2u], FP_MIN_SENTINEL);
    atomicStore(&aabb_max_atomic[i * 3u + 0u], FP_MAX_SENTINEL);
    atomicStore(&aabb_max_atomic[i * 3u + 1u], FP_MAX_SENTINEL);
    atomicStore(&aabb_max_atomic[i * 3u + 2u], FP_MAX_SENTINEL);
    // Parent slot — initialise every node to PARENT_SENTINEL.
    // `build_internal_main` overwrites parents for non-root nodes
    // immediately after this pass; the root keeps SENTINEL so
    // `propagate_atomic_main`'s walk-up loop terminates. Writing
    // here (rather than CPU `queue.write_buffer`) is a hard
    // ordering requirement: init MUST run before karras_internal,
    // which is guaranteed by the dispatch ordering in
    // `TlasBuildPass::build_gpu_tlas`.
    parents[i] = PARENT_SENTINEL;
}

@compute @workgroup_size(64, 1, 1)
fn propagate_atomic_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = tlas_state.prim_count;
    if (i >= n) {
        return;
    }
    let prim_idx = sorted_vals[i];
    let prim = tlas_prims[prim_idx];

    let key_min_x = float_to_sort_key(prim.aabb_min.x);
    let key_min_y = float_to_sort_key(prim.aabb_min.y);
    let key_min_z = float_to_sort_key(prim.aabb_min.z);
    let key_max_x = float_to_sort_key(prim.aabb_max.x);
    let key_max_y = float_to_sort_key(prim.aabb_max.y);
    let key_max_z = float_to_sort_key(prim.aabb_max.z);

    // Walk up from this leaf to root, contributing to every
    // ancestor's atomic accumulator. atomicMin/Max are commutative
    // — final accumulator value is correct regardless of which
    // leaf gets there first.
    //
    // HOP CAP — defense in depth against a corrupt `parents[]`
    // graph. A legitimate root walk visits at most `total_nodes`
    // distinct nodes, so exceeding that many hops PROVES a cycle;
    // without the cap this loop's only exit is the sentinel, and a
    // cyclic graph parks the thread in an infinite, fully-in-bounds
    // loop — non-preemptible GPU work that froze the whole desktop
    // (NVRM Xid 109 CTX SWITCH TIMEOUT). The cycle source was the
    // non-stable radix scatter feeding Karras unsorted keys (fixed
    // in tlas_radix_sort.wesl; pinned by tests/tlas_sort_stability
    // .rs) — under sorted input "each child has exactly one parent"
    // holds and the cap never fires; it costs one add + compare per
    // hop and guarantees no future parents[] corruption, whatever
    // its source, can hang the device again.
    let total_nodes = 2u * n - 1u;
    let leaf_node_idx = n - 1u + i;
    var cur = leaf_node_idx;
    var hops = 0u;
    loop {
        let parent = parents[cur];
        if (parent == PARENT_SENTINEL) { return; }
        atomicMin(&aabb_min_atomic[parent * 3u + 0u], key_min_x);
        atomicMin(&aabb_min_atomic[parent * 3u + 1u], key_min_y);
        atomicMin(&aabb_min_atomic[parent * 3u + 2u], key_min_z);
        atomicMax(&aabb_max_atomic[parent * 3u + 0u], key_max_x);
        atomicMax(&aabb_max_atomic[parent * 3u + 1u], key_max_y);
        atomicMax(&aabb_max_atomic[parent * 3u + 2u], key_max_z);
        cur = parent;
        hops = hops + 1u;
        if (hops > total_nodes) { return; }
    }
}

@compute @workgroup_size(64, 1, 1)
fn decode_aabb_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = tlas_state.prim_count;
    if (n < 2u) { return; }
    if (i >= n - 1u) {
        // Only internal nodes; leaf-marker AABBs were already
        // written directly by `build_leaves_main`.
        return;
    }
    let mn_x = sort_key_to_float(atomicLoad(&aabb_min_atomic[i * 3u + 0u]));
    let mn_y = sort_key_to_float(atomicLoad(&aabb_min_atomic[i * 3u + 1u]));
    let mn_z = sort_key_to_float(atomicLoad(&aabb_min_atomic[i * 3u + 2u]));
    let mx_x = sort_key_to_float(atomicLoad(&aabb_max_atomic[i * 3u + 0u]));
    let mx_y = sort_key_to_float(atomicLoad(&aabb_max_atomic[i * 3u + 1u]));
    let mx_z = sort_key_to_float(atomicLoad(&aabb_max_atomic[i * 3u + 2u]));
    tlas_nodes[i].aabb_min = vec3<f32>(mn_x, mn_y, mn_z);
    tlas_nodes[i].aabb_max = vec3<f32>(mx_x, mx_y, mx_z);
}
