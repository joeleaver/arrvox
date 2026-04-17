// Shared types + opcode constants for the procedural RPN evaluator.
//
// Split out from `proc_eval.wgsl` so callers can declare a storage
// binding of type `array<ProcInstruction>` before the shared functions
// are concatenated in. Concat order at pipeline creation:
//
//   proc_eval_types.wgsl   — this file (types, constants)
//   <caller shader>        — bindings + entry point
//   proc_eval.wgsl         — function bodies (references caller's
//                             `instructions` module-scope binding)
//
// WGSL resolves module-scope functions in any order, so the caller's
// entry point can freely call `eval_tree` even though the function
// body is declared later in the combined source.

// ── Opcodes ────────────────────────────────────────────────────────────
// Match `flatten::OpKind` — values are u32-compared inline, keep in sync.
const OP_SPHERE:    u32 = 0u;
const OP_BOX:       u32 = 1u;
const OP_CAPSULE:   u32 = 2u;
const OP_CYLINDER:  u32 = 3u;
const OP_TORUS:     u32 = 4u;
const OP_PLANE:     u32 = 5u;
const OP_RAMP:      u32 = 6u;
const OP_UNION:     u32 = 100u;
const OP_INTERSECT: u32 = 101u;
const OP_SUBTRACT:  u32 = 102u;

// Position-warp effects bracket a subtree with PUSH/POP.
const OP_PUSH_NOISE_DISPLACE: u32 = 200u;
const OP_POP_NOISE_DISPLACE:  u32 = 201u;
const OP_PUSH_MIRROR:         u32 = 202u;
const OP_POP_MIRROR:          u32 = 203u;

// Attribute-rewrite post-ops — no PUSH/POP pair, a single opcode fires
// after the child's sample is on the stack and mutates the top sample.
const OP_APPLY_MATERIAL_BY_HEIGHT: u32 = 300u;
const OP_APPLY_COLOR_BY_HEIGHT:    u32 = 301u;
const OP_APPLY_MATERIAL_BY_NOISE:  u32 = 302u;
const OP_APPLY_COLOR_BY_NOISE:     u32 = 303u;

const MAT_COMBINE_WINNER:  u32 = 0u;
const MAT_COMBINE_LAYERED: u32 = 1u;
const MAT_COMBINE_BLEND:   u32 = 2u;

// Max RPN stack depth. 16 accommodates pathological nesting while
// keeping on-register use small.
const STACK_CAP: u32 = 16u;

// Max position-stack depth for nested position-warp effects. Overflow
// clamps — the outermost push stops pushing new values.
const POS_STACK_CAP: u32 = 8u;

// Pipeline specialization: true iff the flattened tree contains any
// position-warp opcode (PUSH/POP NoiseDisplace or Mirror). Set by the
// caller at pipeline creation via a WGSL `override`; the shader
// compiler constant-folds the `if (HAS_POS_WARPS)` branches below,
// so simple trees (the common case in the BUILD preview) fully
// eliminate the `pos_stack` — which is dynamic-indexed and gets
// demoted to local memory otherwise, costing every sphere-trace step
// a round trip there.
override HAS_POS_WARPS: bool = true;

// ── Shared structs ─────────────────────────────────────────────────────

// One flattened instruction. Layout MUST match
// `rkp_procedural::flatten::ProcInstruction` byte-for-byte.
struct ProcInstruction {
    op: u32,
    arity: u32,
    material_combine: u32,
    material_id: u32,
    // `node_id` is the source NodeId the primitive came from, used for
    // per-primitive picking; `u32::MAX` on combinators.
    node_id: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    params_lo: vec4<f32>,
    params_hi: vec4<f32>,
    color: vec4<f32>,
    inverse_world: mat4x4<f32>,
}

// A sample at one point. Combinators don't propagate secondary/blend —
// only post-op effects write them, and the winner of any downstream
// combinator carries the values forward.
struct TreeSample {
    distance: f32,
    material_id: u32,
    secondary_material_id: u32,
    blend_weight: f32,
    color: vec3<f32>,
    // Source primitive's NodeId for picking. Propagated through
    // combinators by picking the winner's id.
    node_id: u32,
}

struct HeightClassify {
    lower: u32,
    upper: u32,
    alpha: f32,
}
