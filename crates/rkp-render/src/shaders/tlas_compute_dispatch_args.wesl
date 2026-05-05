// Phase 7c indirect-dispatch refactor — single thread, single workgroup.
//
// Reads the post-assembly `tlas_prim_count[0]` atomic, clamps it by
// `prims_capacity` (assembly's overflow guard counter can exceed
// the buffer; the buffer-side gate already dropped the writes), and
// emits the dynamic counts that the rest of the chain needs:
//
//   * `tlas_state` — shared per-frame state read by morton, radix,
//     karras, and propagation shaders. Carries `prim_count` plus the
//     three workgroup counts the chain dispatches need.
//   * `tlas_dispatch_args` — packed `array<u32>` whose 7 slots feed
//     `dispatch_workgroups_indirect`. Each slot is 3 × u32 (x/y/z
//     workgroup counts), 12 bytes wide. Indirect-dispatch byte
//     offset for slot `i` is `i * 12`.
//
// Slot layout:
//   0 — morton            (radix_workgroups)
//   1 — radix             (radix_workgroups; shared by 4 sub-passes)
//   2 — karras_leaves     (radix_workgroups)
//   3 — karras_internal   (internal_wgs;     0 when N < 2)
//   4 — init_atomic_aabb  (total_node_wgs;   0 when N < 2)
//   5 — propagate         (radix_workgroups; 0 when N < 2)
//   6 — decode_aabb       (internal_wgs;     0 when N < 2)
//
// Gating to 0 workgroups for N < 2 is a defence-in-depth — every
// downstream shader also carries an `if (n < 2u) return;` early-out,
// but a 0-workgroup indirect dispatch is GPU-cheap and avoids
// launching any threads at all.

struct TlasState {
    prim_count: u32,
    radix_workgroups: u32,
    internal_wgs: u32,
    total_node_wgs: u32,
}

struct AssembleHostUniform {
    instance_count: u32,
    asset_count: u32,
    prims_capacity: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> tlas_prim_count: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> tlas_state: TlasState;
@group(0) @binding(2) var<storage, read_write> tlas_dispatch_args: array<u32>;
@group(1) @binding(0) var<uniform> u: AssembleHostUniform;

const RADIX_WG: u32 = 64u;

@compute @workgroup_size(1, 1, 1)
fn compute_dispatch_args_main() {
    let raw = atomicLoad(&tlas_prim_count[0]);
    let n = min(raw, u.prims_capacity);

    let radix_wgs = (n + RADIX_WG - 1u) / RADIX_WG;
    let internal_count = select(0u, n - 1u, n >= 2u);
    let internal_wgs = (internal_count + RADIX_WG - 1u) / RADIX_WG;
    let total_node_count = select(0u, 2u * n - 1u, n >= 2u);
    let total_node_wgs = (total_node_count + RADIX_WG - 1u) / RADIX_WG;

    tlas_state.prim_count = n;
    tlas_state.radix_workgroups = radix_wgs;
    tlas_state.internal_wgs = internal_wgs;
    tlas_state.total_node_wgs = total_node_wgs;

    // Slot 0 — morton (per-prim).
    tlas_dispatch_args[0] = radix_wgs;
    tlas_dispatch_args[1] = 1u;
    tlas_dispatch_args[2] = 1u;
    // Slot 1 — radix (shared by count, scatter; scan stays direct).
    tlas_dispatch_args[3] = radix_wgs;
    tlas_dispatch_args[4] = 1u;
    tlas_dispatch_args[5] = 1u;
    // Slot 2 — karras_leaves (per-prim).
    tlas_dispatch_args[6] = radix_wgs;
    tlas_dispatch_args[7] = 1u;
    tlas_dispatch_args[8] = 1u;
    // Slot 3 — karras_internal (per-internal-node, gated on N >= 2).
    let internal_dispatch_x = select(0u, internal_wgs, n >= 2u);
    tlas_dispatch_args[9]  = internal_dispatch_x;
    tlas_dispatch_args[10] = 1u;
    tlas_dispatch_args[11] = 1u;
    // Slot 4 — init_atomic_aabb (per total node, gated on N >= 2).
    let init_dispatch_x = select(0u, total_node_wgs, n >= 2u);
    tlas_dispatch_args[12] = init_dispatch_x;
    tlas_dispatch_args[13] = 1u;
    tlas_dispatch_args[14] = 1u;
    // Slot 5 — propagate (per-leaf, gated on N >= 2).
    let propagate_dispatch_x = select(0u, radix_wgs, n >= 2u);
    tlas_dispatch_args[15] = propagate_dispatch_x;
    tlas_dispatch_args[16] = 1u;
    tlas_dispatch_args[17] = 1u;
    // Slot 6 — decode (per-internal-node, gated on N >= 2).
    let decode_dispatch_x = select(0u, internal_wgs, n >= 2u);
    tlas_dispatch_args[18] = decode_dispatch_x;
    tlas_dispatch_args[19] = 1u;
    tlas_dispatch_args[20] = 1u;
}
