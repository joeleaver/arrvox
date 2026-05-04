// Procedural "sample N positions" compute shader.
//
// Reads a batch of query positions, evaluates the flattened tree at
// each, writes a compact per-position result. Exists so the CPU bake
// pipeline (Phase 3+) can use the same RPN interpreter as the live
// raymarch — one implementation of the procedural math, on the GPU.
//
// Imports the RPN interpreter from `proc_eval` and its type/opcode
// constants from `proc_eval_types`. The shared `eval_tree` reads the
// `instructions` binding declared below.

import package::lib::types::ProcInstruction;
import package::proc_eval::eval_tree;

// Pipeline-override constant — kept unmangled so the Rust side
// (`proc_sample.rs`) can override it at pipeline creation by name.
// proc_eval's body references this as a free name resolved at emit.
override HAS_POS_WARPS: bool = true;

struct SampleParams {
    // Instructions in the `instructions` buffer to execute per query.
    instruction_count: u32,
    // Queries this dispatch should process. Guard for the final
    // workgroup: threads past this return without writing.
    position_count: u32,
    _pad0: u32,
    _pad1: u32,
}

// A packed result per input position. 20 bytes → padded to 32 for
// alignment (Rust mirror keeps the `#[repr(C)]` padding explicit):
//   distance       — signed distance at the query point.
//   primary        — low 16 bits = primary material id.
//   secondary      — low 16 bits = secondary material id.
//   blend_u4       — low 4 bits  = quantized blend weight (0..15).
//   color          — packed `R | (G << 8) | (B << 16) | (0xFF << 24)`.
//                    The high byte is a non-zero sentinel so a
//                    deliberate (0,0,0) color doesn't collide with
//                    the color_pool's `0 = no override` semantic.
// Readback code narrows u32 fields to the target widths.
struct SampleResult {
    distance: f32,
    primary: u32,
    secondary: u32,
    blend_u4: u32,
    color: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: SampleParams;
@group(0) @binding(1) var<storage, read> instructions: array<ProcInstruction>;
// Input positions. `vec4` rather than `vec3` so the array's stride is
// 16 bytes and WGSL's storage-array rules are trivially satisfied. The
// `.w` component is ignored.
@group(0) @binding(2) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> results: array<SampleResult>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.position_count) {
        return;
    }

    let s = eval_tree(positions[i].xyz, params.instruction_count);

    // Quantize blend to 4 bits, matching the CPU bake's
    // `(b * 15.0).round().clamp(0, 15) as u8`. `trunc + 0.5` is the
    // standard round-to-nearest on GPUs.
    let blend_f = clamp(s.blend_weight, 0.0, 1.0) * 15.0;
    let blend_u4 = u32(trunc(blend_f + 0.5));

    // Pack per-voxel color into the color_pool's layout. High byte
    // forced to 0xFF so a black leaf doesn't compare equal to 0
    // (the "no override" sentinel the voxel shader checks).
    let cr = u32(clamp(s.color.r, 0.0, 1.0) * 255.0 + 0.5);
    let cg = u32(clamp(s.color.g, 0.0, 1.0) * 255.0 + 0.5);
    let cb = u32(clamp(s.color.b, 0.0, 1.0) * 255.0 + 0.5);
    let packed_color = cr | (cg << 8u) | (cb << 16u) | (0xFFu << 24u);

    var r: SampleResult;
    r.distance = s.distance;
    r.primary = s.material_id & 0xFFFFu;
    r.secondary = s.secondary_material_id & 0xFFFFu;
    r.blend_u4 = blend_u4 & 0xFu;
    r.color = packed_color;
    r._pad0 = 0u;
    r._pad1 = 0u;
    r._pad2 = 0u;
    results[i] = r;
}
