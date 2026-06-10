// Phase 7c Session 2 — 8-bit GPU radix sort over 32-bit Morton
// codes (with attached `vals` payload = original prim index).
//
// Three compute entry points, one per radix sub-pass; the engine
// fires count → scan → scatter four times (digit shifts 0, 8, 16,
// 24), swapping the (keys_in/keys_out, vals_in/vals_out) bind
// group between passes. After the fourth scatter the (key, val)
// pairs are sorted ascending by full 32-bit Morton.
//
// EVERY pass MUST be stable (preserve the input order of elements
// with equal digits): LSD radix sort *derives* global sortedness
// from per-pass stability — pass k only orders by digit k, and
// relies on lower digits already being in order for keys whose
// digit k ties. A non-stable pass reorders DISTINCT keys that
// share their higher digits, and no later pass ever re-compares
// them, so the final output is NOT globally sorted. The original
// V1 scatter claimed slots with a per-(WG, bucket) `atomicAdd`
// (non-deterministic thread order = non-stable) on the incorrect
// theory that only tie ordering was affected; the unsorted output
// violated the Karras builder's sorted-keys precondition, which
// produced cyclic `parents[]` graphs, which made
// `propagate_atomic_main`'s walk-up loop spin forever — the
// whole-desktop NVRM "Xid 109 CTX SWITCH TIMEOUT" GPU hang at
// ~15-20 resident terrain tiles (grid-clustered tile centroids
// maximize shared-high-bit Morton keys, the exact pattern the
// instability corrupts). Pinned by
// `tests/tlas_sort_stability.rs`, which runs this chain on a real
// adapter and asserts sortedness + parents[] acyclicity.
//
// ## Shared layout
//
// All three entries share group(0) and group(1). The pipeline
// runs the shaders without re-creating bind groups across the
// three sub-passes — only the histogram buffer needs zeroing
// between count passes (engine encodes a `clear_buffer` for that).
//
// * `keys_in` / `vals_in`  — the (Morton, prim_idx) pair for this
//   pass. Read by `count_main` and `scatter_main`.
// * `keys_out` / `vals_out`  — destination of `scatter_main`.
// * `histogram`  — `array<atomic<u32>>` of length
//   `num_workgroups × 256`. `count_main` atomic-adds into
//   `[wg_idx × 256 + digit]`.
// * `scan_offsets`  — `array<atomic<u32>>` of length
//   `num_workgroups × 256`. `scan_main` writes per-WG-per-bucket
//   starting offsets; `scatter_main` reads them as immutable bases
//   and adds each element's stable in-workgroup rank (the array is
//   re-written by the next pass's scan). Kept `atomic` only to
//   avoid a bind-group-layout change from the V1 cursor design.
// * `u: RadixUniform`  — `prim_count`, `digit_shift`,
//   `num_workgroups`. Engine bumps `digit_shift` by 8 between
//   passes.

const WG_SIZE_COUNT_SCATTER: u32 = 64u;
const WG_SIZE_SCAN: u32 = 256u;
const RADIX_BUCKETS: u32 = 256u;

struct RadixUniform {
    digit_shift: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// Shared state written by `tlas_compute_dispatch_args.wesl`.
// `prim_count` and `radix_workgroups` (= num_workgroups for the
// count + scatter passes) flow through this buffer instead of the
// per-pass uniform.
struct TlasState {
    prim_count: u32,
    radix_workgroups: u32,
    internal_wgs: u32,
    total_node_wgs: u32,
}

@group(0) @binding(0) var<storage, read> keys_in: array<u32>;
@group(0) @binding(1) var<storage, read> vals_in: array<u32>;
@group(0) @binding(2) var<storage, read_write> keys_out: array<u32>;
@group(0) @binding(3) var<storage, read_write> vals_out: array<u32>;
@group(0) @binding(4) var<storage, read_write> histogram: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> scan_offsets: array<atomic<u32>>;
@group(1) @binding(0) var<uniform> u: RadixUniform;
@group(1) @binding(1) var<uniform> tlas_state: TlasState;

// ── Pass 1 — count digits per workgroup ────────────────────────────

@compute @workgroup_size(64, 1, 1)
fn count_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let i = gid.x;
    if (i >= tlas_state.prim_count) {
        return;
    }
    let key = keys_in[i];
    let digit = (key >> u.digit_shift) & 0xFFu;
    atomicAdd(&histogram[wg.x * RADIX_BUCKETS + digit], 1u);
}

// ── Pass 2 — global scan over the (num_workgroups × 256) histogram ─
//
// Single workgroup of 256 threads, one per bucket. Computes:
//
//   bucket_total[b]      = Σ_wg histogram[wg, b]
//   bucket_global[b]     = exclusive prefix scan of bucket_total[*]
//   scan_offsets[wg, b]  = bucket_global[b] + Σ_{w<wg} histogram[w, b]
//
// `scan_offsets` then feeds the scatter pass — each (WG, bucket)
// knows where its run of items starts in the output buffer, and
// scatter atomicAdds into that offset to claim its slot.
//
// V1 caps the dispatch at 65 K workgroups (about 4 M primitives at
// `WG_SIZE_COUNT_SCATTER = 64`). Larger inputs would need a
// multi-workgroup scan; not yet exercised in practice.

var<workgroup> bucket_total: array<u32, 256>;
var<workgroup> bucket_global: array<u32, 256>;

@compute @workgroup_size(256, 1, 1)
fn scan_main(@builtin(local_invocation_index) lid: u32) {
    let num_workgroups = tlas_state.radix_workgroups;
    // Step 1 — sum histogram[*][lid] across workgroups.
    var sum: u32 = 0u;
    for (var w: u32 = 0u; w < num_workgroups; w = w + 1u) {
        sum = sum + atomicLoad(&histogram[w * RADIX_BUCKETS + lid]);
    }
    bucket_total[lid] = sum;
    workgroupBarrier();

    // Step 2 — exclusive scan over `bucket_total` into
    // `bucket_global`. Hillis-Steele inclusive scan, then shift
    // right by one to get the exclusive form.
    var v = bucket_total[lid];
    for (var stride: u32 = 1u; stride < 256u; stride = stride * 2u) {
        var add: u32 = 0u;
        if (lid >= stride) {
            add = bucket_total[lid - stride];
        }
        workgroupBarrier();
        v = v + add;
        bucket_total[lid] = v;
        workgroupBarrier();
    }
    // Inclusive → exclusive: shift right.
    if (lid == 0u) {
        bucket_global[0u] = 0u;
    } else {
        bucket_global[lid] = bucket_total[lid - 1u];
    }
    workgroupBarrier();

    // Step 3 — write per-(WG, bucket) starting offsets to
    // scan_offsets. Each thread b walks all workgroups for its
    // bucket, accumulating the running prefix.
    var prefix = bucket_global[lid];
    for (var w: u32 = 0u; w < num_workgroups; w = w + 1u) {
        let h = atomicLoad(&histogram[w * RADIX_BUCKETS + lid]);
        atomicStore(&scan_offsets[w * RADIX_BUCKETS + lid], prefix);
        prefix = prefix + h;
    }
}

// ── Pass 3 — scatter (STABLE) ──────────────────────────────────────
//
// Each thread reads its (key, val), computes the bucket digit, and
// derives its destination DETERMINISTICALLY:
//
//   dst = scan_offsets[wg, digit]            (this WG's global base
//                                             for this bucket)
//       + stable_rank                        (count of earlier
//                                             same-digit elements
//                                             in this workgroup)
//
// `scan_main` already lays out per-(WG, bucket) runs contiguously
// in workgroup order, so input order is preserved both across and
// within workgroups — the pass is stable, which the LSD sort's
// correctness requires (see module doc). The rank loop re-reads up
// to WG_SIZE_COUNT_SCATTER-1 keys from storage; at TLAS prim
// counts (tens to low thousands) this is noise. If prim counts
// ever grow to the 4 M scan cap, replace with a workgroup-shared
// prefix sum over per-digit match masks — keep it stable.

@compute @workgroup_size(64, 1, 1)
fn scatter_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let i = gid.x;
    if (i >= tlas_state.prim_count) {
        return;
    }
    let key = keys_in[i];
    let val = vals_in[i];
    let digit = (key >> u.digit_shift) & 0xFFu;
    let wg_base = wg.x * WG_SIZE_COUNT_SCATTER;
    var rank: u32 = 0u;
    for (var j: u32 = wg_base; j < i; j = j + 1u) {
        if (((keys_in[j] >> u.digit_shift) & 0xFFu) == digit) {
            rank = rank + 1u;
        }
    }
    let off = atomicLoad(&scan_offsets[wg.x * RADIX_BUCKETS + digit]) + rank;
    keys_out[off] = key;
    vals_out[off] = val;
}
