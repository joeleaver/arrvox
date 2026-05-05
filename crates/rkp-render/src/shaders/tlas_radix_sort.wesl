// Phase 7c Session 2 — 8-bit GPU radix sort over 32-bit Morton
// codes (with attached `vals` payload = original prim index).
//
// Three compute entry points, one per radix sub-pass; the engine
// fires count → scan → scatter four times (digit shifts 0, 8, 16,
// 24), swapping the (keys_in/keys_out, vals_in/vals_out) bind
// group between passes. After the fourth scatter the (key, val)
// pairs are sorted ascending by full 32-bit Morton; ties within
// the same digit may permute (V1 uses non-stable scatter via
// per-bucket atomic counters), but Karras tree construction
// tolerates non-strict tie ordering.
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
//   starting offsets; `scatter_main` atomic-adds into them as it
//   places elements (cheap "next-offset" cursor; the array is
//   consumed by the scatter and re-written by the next pass's
//   scan).
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

// ── Pass 3 — scatter ───────────────────────────────────────────────
//
// Each thread reads its (key, val), computes the bucket digit,
// atomic-bumps `scan_offsets[wg, digit]` to claim a slot, and
// writes the pair to that slot in `keys_out` / `vals_out`.
//
// Within a WG-bucket, the order of writes is non-deterministic
// (atomic order across threads in a workgroup is not specified by
// WGSL). Across WG-buckets the layout is exact (per-bucket runs
// are contiguous). For the Karras tree builder this is enough —
// it depends on Mortons being globally sorted, not on stable
// tie-breaking.

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
    let off = atomicAdd(&scan_offsets[wg.x * RADIX_BUCKETS + digit], 1u);
    keys_out[off] = key;
    vals_out[off] = val;
}
