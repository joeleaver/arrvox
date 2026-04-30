// Phase 6 Session 3b — user-shader tile-cull prefix sum pass.
//
// Converts `us_tile_counts[t]` into `us_tile_offsets[]`:
//
//     us_tile_offsets[0]   = 0
//     us_tile_offsets[t+1] = sum(counts[0..=t])
//
// Single-workgroup blocked scan supporting up to
// `PREFIX_BLOCK × PREFIX_THREADS = 256 × 256 = 65536` tiles per
// dispatch. Caps the supported viewport at ~2300×800 px (8 px tiles)
// in V1; multi-block dispatch is a follow-up if 4K becomes a target.
//
// Algorithm — three phases inside one `tile_prefix_main` invocation:
//
// 1. Each thread serially scans its 256-tile block, writing the
//    block-local inclusive prefix into `us_tile_offsets[block_start+1
//    .. block_start+256+1]`. The block total goes into shared
//    `wg_block_sums`.
// 2. Hillis-Steele exclusive scan of `wg_block_sums` across the 256
//    threads (in shared memory).
// 3. Each thread adds its block_offset back to its block-local
//    prefixes in `us_tile_offsets`.
//
// `us_tile_offsets[0]` is set to 0 by thread 0 in a final write.
//
// Tile count is read from a small uniform — the dispatch always uses
// 1 workgroup of 256 threads regardless.

const PREFIX_BLOCK: u32 = 256u;
const PREFIX_THREADS: u32 = 256u;
const PREFIX_MAX_TILES: u32 = 65536u; // PREFIX_BLOCK * PREFIX_THREADS

struct PrefixUniform {
    tile_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> us_tile_counts: array<u32>;
@group(0) @binding(1) var<storage, read_write> us_tile_offsets: array<u32>;

@group(1) @binding(0) var<uniform> u: PrefixUniform;

var<workgroup> wg_block_sums: array<u32, PREFIX_THREADS>;
var<workgroup> wg_scan: array<u32, PREFIX_THREADS>;

@compute @workgroup_size(PREFIX_THREADS, 1, 1)
fn tile_prefix_main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let block_start = tid * PREFIX_BLOCK;

    // Phase 1 — block-local inclusive prefix into offsets[block_start+1..].
    var block_total: u32 = 0u;
    for (var i: u32 = 0u; i < PREFIX_BLOCK; i = i + 1u) {
        let global_idx = block_start + i;
        if (global_idx >= u.tile_count) { break; }
        let count = us_tile_counts[global_idx];
        block_total = block_total + count;
        // Inclusive within block: offsets[global_idx+1] = sum(counts[block_start..=global_idx]).
        us_tile_offsets[global_idx + 1u] = block_total;
    }
    wg_block_sums[tid] = block_total;
    workgroupBarrier();

    // Phase 2 — Hillis-Steele exclusive scan of block sums.
    var scan_val = wg_block_sums[tid];
    workgroupBarrier();
    for (var d: u32 = 1u; d < PREFIX_THREADS; d = d * 2u) {
        wg_scan[tid] = scan_val;
        workgroupBarrier();
        if (tid >= d) {
            scan_val = scan_val + wg_scan[tid - d];
        }
        workgroupBarrier();
    }
    // Convert inclusive → exclusive: subtract own contribution.
    let block_offset = scan_val - wg_block_sums[tid];

    // Phase 3 — add block_offset back into block-local prefixes.
    if (block_offset != 0u) {
        for (var i: u32 = 0u; i < PREFIX_BLOCK; i = i + 1u) {
            let global_idx = block_start + i;
            if (global_idx >= u.tile_count) { break; }
            us_tile_offsets[global_idx + 1u] = us_tile_offsets[global_idx + 1u] + block_offset;
        }
    }

    if (tid == 0u) {
        us_tile_offsets[0] = 0u;
    }
}
