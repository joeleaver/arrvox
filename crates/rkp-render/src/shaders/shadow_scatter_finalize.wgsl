// Phase 8 V2 — shadow scatter finalize pass.
//
// Single thread. Reads `total_work` (the post-setup atomic sum
// of every instance's `tile_count`) and packs it into the
// `dispatch_args` indirect-dispatch buffer for the scatter pass.
//
// Layout:
//   dispatch_args[0] = DISPATCH_X         (constant)
//   dispatch_args[1] = ceil(total_work / DISPATCH_X)
//   dispatch_args[2] = 1
//
// 2D dispatch lifts wgpu's 65 535-per-dimension limit: with
// `DISPATCH_X = 256` we cover up to 256 × 65 535 = ~16.7 M tiles,
// equivalent to ~4 K × 4 K shadow map fully covered.

const DISPATCH_X: u32 = 256u;

@group(0) @binding(0) var<storage, read_write> total_work: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> dispatch_args: array<u32>;

@compute @workgroup_size(1, 1, 1)
fn finalize_main() {
    let total = atomicLoad(&total_work[0]);
    dispatch_args[0] = DISPATCH_X;
    dispatch_args[1] = (total + DISPATCH_X - 1u) / DISPATCH_X;
    dispatch_args[2] = 1u;
    // Stash total into slot 3 so the scatter shader can bounds-
    // check `flat_work_idx >= total`.
    dispatch_args[3] = total;
}
