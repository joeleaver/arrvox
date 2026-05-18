// Phase 7c Session 2 — Morton-code computation pass.
//
// One thread per primitive. Reads the prim's centroid, normalizes
// against the engine-supplied scene AABB, quantizes each axis to
// 10 bits, interleaves the bits to form a 30-bit Morton code, and
// writes the (Morton, prim_index) pair into the radix-sort input
// buffers (`keys_a` / `vals_a`).
//
// Initializes `vals_a[i] = i` so a stable radix sort preserves
// natural prim_index ordering as a tie-breaker for primitives that
// land in the same Morton bucket (common for sibling blades inside
// one painted leaf).

import package::lib::types::TlasPrim;

struct MortonUniform {
    // Engine-computed scene AABB. CPU walks the per-frame region
    // uniforms (their authoring AABB encloses every per-blade AABB
    // a region's user shader can produce) and host instance AABBs,
    // then takes the union. Conservative is fine — Morton sort
    // just needs a stable coordinate system, not a tight one.
    scene_min: vec3<f32>,
    _pad0: u32,
    scene_max: vec3<f32>,
    _pad1: u32,
}

// Shared state written by `tlas_compute_dispatch_args.wesl`.
// Per-frame `prim_count` flows through this buffer, not the
// MortonUniform — keeps the chain indirect-dispatch-driven.
struct TlasState {
    prim_count: u32,
    radix_workgroups: u32,
    internal_wgs: u32,
    total_node_wgs: u32,
}

@group(0) @binding(0) var<storage, read> tlas_prims: array<TlasPrim>;
@group(0) @binding(1) var<storage, read_write> keys_a: array<u32>;
@group(0) @binding(2) var<storage, read_write> vals_a: array<u32>;
@group(1) @binding(0) var<uniform> u: MortonUniform;
@group(1) @binding(1) var<uniform> tlas_state: TlasState;

// Spread 10 input bits across 30 output bits (positions 0, 3, 6,
// ..., 27). Standard Morton "expand" via shift+mask. 10 bits caps
// the per-axis quantization at 1024 buckets, which is plenty for
// scene-coherent BVH ordering — finer is nice-to-have but the
// tree builder doesn't depend on key uniqueness within a bucket
// (radix sort is stable; ties retain prim_idx order as a
// secondary key).
fn expand_bits_10(v_in: u32) -> u32 {
    var v = v_in & 0x3FFu;
    v = (v | (v << 16u)) & 0x030000FFu;
    v = (v | (v << 8u))  & 0x0300F00Fu;
    v = (v | (v << 4u))  & 0x030C30C3u;
    v = (v | (v << 2u))  & 0x09249249u;
    return v;
}

fn morton_30(x: u32, y: u32, z: u32) -> u32 {
    return (expand_bits_10(x) << 2u)
         | (expand_bits_10(y) << 1u)
         |  expand_bits_10(z);
}

@compute @workgroup_size(64, 1, 1)
fn compute_morton_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= tlas_state.prim_count) {
        return;
    }
    let p = tlas_prims[i];
    let centroid = 0.5 * (p.aabb_min + p.aabb_max);
    // Normalize to [0, 1]^3 against scene bounds; clamp on either
    // side handles primitives that lie outside the (conservative)
    // scene AABB the CPU computed — shouldn't happen in normal
    // scenes but defends against degenerate inputs.
    let extent = max(u.scene_max - u.scene_min, vec3<f32>(1e-6));
    let normalized = clamp((centroid - u.scene_min) / extent, vec3<f32>(0.0), vec3<f32>(1.0));
    // Quantize to 10 bits per axis (1024 buckets per axis).
    // Multiply by 1023, not 1024, so the max value lands in
    // bucket 1023 instead of overflowing.
    let q = vec3<u32>(min(vec3<u32>(normalized * 1023.0), vec3<u32>(1023u)));
    keys_a[i] = morton_30(q.x, q.y, q.z);
    vals_a[i] = i;
}
