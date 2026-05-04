//! RPN-style flattening of a `ProceduralObject` tree into a linear
//! instruction stream, intended for GPU consumption.
//!
//! The CPU `sample_tree` is recursive: at each node it transforms the
//! position and dispatches on kind. WGSL has no recursion, so we emit
//! a post-order instruction sequence and let the shader execute it
//! against a small fixed-size stack:
//!
//! - A **primitive** instruction pushes one sample onto the stack
//!   (after applying the primitive's transform to the query position).
//! - A **combinator** instruction pops `arity` samples, combines them
//!   according to its kind (`Union` / `Intersect` / `Subtract`), and
//!   pushes the single result.
//!
//! The output of executing the full instruction stream is the tree's
//! sample at the query position — equivalent to `sample_tree(obj, pos,
//! voxel_size)` for all well-formed trees.
//!
//! Transforms are composed at flatten time so the shader only needs a
//! single `inverse_world` per primitive: each primitive stores the
//! inverse of the product of all its ancestors' transforms (root → leaf),
//! which the shader applies directly to world-space query positions.
//! The combinator transforms are absorbed into their descendants — they
//! don't appear in the instruction stream.

use glam::{Affine3A, Mat4};

use crate::arena::{NodeId, ProceduralObject};
use crate::node_kind::MaterialCombine;

mod dispatch;

/// Opcode tags. Values match the WGSL `OP_*` constants in
/// `shaders/proc_raymarch.wgsl` — keep in sync.
///
/// The `< 100` range is primitives (push one sample onto the stack);
/// `100..200` is boolean combinators (pop `arity`, push one combined
/// sample); `200+` is position-warp effects that bracket a subtree
/// with a matched PUSH/POP pair — PUSH pushes a new sample position
/// onto the shader's position stack; POP decrements the position stack
/// and shrinks the distance of the top sample by a conservative
/// envelope so sphere-tracing stays 1-Lipschitz-safe.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpKind {
    Sphere = 0,
    Box = 1,
    Capsule = 2,
    Cylinder = 3,
    Torus = 4,
    Plane = 5,
    Ramp = 6,
    Union = 100,
    Intersect = 101,
    Subtract = 102,
    /// Push a noise-displaced position onto the shader's position
    /// stack. Children evaluated between this and the matching
    /// `PopNoiseDisplace` see the warped position. `params[0..4]` =
    /// `[amplitude, frequency, seed_as_f32, octaves_as_f32]`.
    PushNoiseDisplace = 200,
    /// Pop the position stack and shrink the top sample's distance
    /// by a conservative envelope (max axial warp = amplitude * √3).
    /// `params[0]` = amplitude; the rest ignored.
    PopNoiseDisplace = 201,
    /// Push a mirror-folded position onto the shader's position stack.
    /// Unlike `PushNoiseDisplace` which carries scalar warp params,
    /// Mirror's PUSH carries the derived world-space plane:
    /// `params[0..3]` = plane origin (world), `params[4..7]` = plane
    /// normal (world, unit-length). The fold is a world-space
    /// reflection across that plane — length-preserving, so `PopMirror`
    /// does **not** shrink the top sample's distance.
    PushMirror = 202,
    /// Pop the position stack after a mirror PUSH. No distance
    /// adjustment — a reflection is a pure isometry.
    PopMirror = 203,
    /// Post-op: rewrite the top sample's `material_id` /
    /// `secondary_material_id` / `blend_weight` from a 3-band rule on
    /// the effect's local Y. Doesn't touch the position stack. Layout
    /// is described in `emit_material_by_height_post_op`.
    ApplyMaterialByHeight = 300,
    /// Post-op: rewrite the top sample's `color` from a 3-band rule
    /// on the effect's local Y. Layout is described in
    /// `emit_color_by_height_post_op`.
    ApplyColorByHeight = 301,
    /// Post-op: rewrite the top sample's material fields from a
    /// 3-band rule on an FBM noise sample at the effect-local
    /// position. `params[0..6]` = (low_to_mid, mid_to_high,
    /// transition_width, frequency, seed_as_f32, octaves_as_f32).
    /// `params[6..8]` = (low_material, mid_material). `color[0]` =
    /// high_material. `inverse_world` = effect's (A*M).inverse() so
    /// the shader can transform `pos_stack[pos_top]` into the
    /// effect's local frame before sampling noise.
    ApplyMaterialByNoise = 302,
    /// Post-op: rewrite the top sample's `color` from a 3-band rule
    /// on an FBM noise sample. `params[0..6]` same as
    /// `ApplyMaterialByNoise`. `params[6..8]` + `color[0]` = three
    /// RGB colors packed as u24 (bit-cast into f32). `color[1..4]`
    /// unused. `inverse_world` = effect's (A*M).inverse().
    ApplyColorByNoise = 303,
    /// Push an Array-folded position onto the shader's position stack.
    /// Uses the `opRepLim` trick: `p' = p - clamp(round(p / s), -(N-1)/2, (N-1)/2) * s`
    /// per axis, so every sample collapses onto the canonical center
    /// cell and the child evaluates once regardless of total count.
    /// `params[0..3]` = spacings, `params[4..7]` = counts (as f32).
    /// `inverse_world` = effect's (A*M).inverse() so the shader folds
    /// in local space then transforms back.
    PushArray = 204,
    /// Pop the position stack after an Array PUSH. No distance
    /// adjustment — the fold is a translation, hence length-preserving.
    PopArray = 205,
}

/// A single instruction in the flattened tree stream.
///
/// Primitives carry transform + params; combinators carry arity + the
/// material-combine mode. One struct type keeps the GPU buffer stride
/// stable and the shader tight — the cost is some unused fields per
/// instruction, small relative to the overall bandwidth.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ProcInstruction {
    /// Opcode tag matching `OpKind`. WGSL reads as `u32`.
    pub op: u32,
    /// Combinator arity (number of samples to pop). Ignored for
    /// primitives. Stored as the second u32 so the common prefix is
    /// cheap to branch on in the shader.
    pub arity: u32,
    /// `MaterialCombine` encoded as `u32` (see `material_combine_bits`).
    /// Only meaningful for `Union` / `Intersect`. `Subtract` has no
    /// material-combine (minuend wins), but the field is present in
    /// every instruction for alignment.
    pub material_combine: u32,
    /// Primitive material id (16-bit); padded to u32.
    pub material_id: u32,

    /// Source tree NodeId for this primitive, used by the build
    /// viewport's pick path to translate a hit pixel back to the
    /// clicked primitive. `u32::MAX` for combinators. Capped to 16
    /// bits when packed into the G-buffer (see `proc_raymarch.wgsl`);
    /// that supports 65k distinct primitives per procedural, well
    /// past where the raymarch is cost-effective.
    pub node_id: u32,
    /// For primitives: scalar converting local SDF distance to world
    /// distance. Equals the smallest axis scale of the composed
    /// world transform (safe for 1-Lipschitz under non-uniform
    /// scale). The shader multiplies each primitive's returned
    /// distance by this before the classifier / combinator chain
    /// sees it — without this, a `Root.scale = 20` tree returns
    /// distances in the compressed local frame and the octree-build
    /// classifier subdivides everything to MIXED because |d| looks
    /// tiny everywhere. Combinators / effects set this to `1.0` and
    /// rely on their children already being in world-space.
    pub distance_scale: f32,
    pub _pad1: u32,
    pub _pad2: u32,

    /// Primitive local-space params, unioned across shapes. Layout per
    /// opcode (components noted; unused entries are zero):
    ///
    /// - `Sphere`   : `[radius, 0, 0, 0, 0, 0, 0, 0]`
    /// - `Box`      : `[hx, hy, hz, rounding, 0, 0, 0, 0]`
    /// - `Capsule`  : `[radius, half_height, 0, 0, 0, 0, 0, 0]`
    /// - `Cylinder` : `[radius, half_height, 0, 0, 0, 0, 0, 0]`
    /// - `Torus`    : `[major_radius, minor_radius, 0, 0, 0, 0, 0, 0]`
    /// - `Plane`    : `[0, 0, 0, 0, 0, 0, 0, 0]`
    /// - `Ramp`     : `[half_length, half_height, half_width, 0, 0, 0, 0, 0]`
    /// - combinators: ignored
    pub params: [f32; 8],

    /// Leaf color, linear RGB (w unused). Ignored for combinators.
    pub color: [f32; 4],

    /// World-to-local affine inverse for this primitive, composed from
    /// the tree root down. Stored as a 4x4 even though the last row is
    /// always `(0,0,0,1)` — WGSL `mat4x4<f32>` is the cleanest type at
    /// a 16-byte-aligned struct field, and the extra row costs 16 bytes
    /// per instruction (dwarfed by the 64-byte transform block). Layout
    /// is column-major to match `Mat4::to_cols_array_2d`.
    pub inverse_world: [[f32; 4]; 4],
}

/// Pack an RGB color (each channel in `[0, 1]`) into a u24 value
/// bit-laid as `r | (g << 8) | (b << 16)`. Used by the by-noise
/// color post-op to fit three colors into three f32 slots; the u32
/// is then bit-cast to f32 (safe since values ≤ 2^24 are exactly
/// representable as f32). The WGSL side unpacks with
/// `bitcast<u32>(f32) >> {0, 8, 16} & 0xFF`.
pub(super) fn pack_rgb_u24(c: glam::Vec3) -> u32 {
    let r = (c.x.clamp(0.0, 1.0) * 255.0) as u32;
    let g = (c.y.clamp(0.0, 1.0) * 255.0) as u32;
    let b = (c.z.clamp(0.0, 1.0) * 255.0) as u32;
    r | (g << 8) | (b << 16)
}

/// Encode a `MaterialCombine` as a u32 for the GPU. Values match the
/// WGSL `MAT_COMBINE_*` constants.
pub(super) fn material_combine_bits(m: MaterialCombine) -> u32 {
    match m {
        MaterialCombine::Winner => 0,
        MaterialCombine::Layered => 1,
        MaterialCombine::Blend { .. } => 2,
    }
}

/// For `Blend{radius}`, the radius rides along in combinator instructions'
/// `params[0]` — unused on the Winner/Layered paths and zero there.
pub(super) fn combinator_radius(m: MaterialCombine) -> f32 {
    match m {
        MaterialCombine::Blend { radius } => radius.max(1e-6),
        _ => 0.0,
    }
}

/// Flatten a procedural tree into a linear instruction stream ready for
/// GPU upload. Returns an empty vector for an empty tree.
pub fn flatten_tree(obj: &ProceduralObject) -> Vec<ProcInstruction> {
    let mut out = Vec::new();
    dispatch::emit(obj, obj.root(), Affine3A::IDENTITY, &mut out);
    out
}


pub(super) fn emit_primitive(
    op: OpKind,
    node_id: NodeId,
    params: [f32; 8],
    material_id: u16,
    color: glam::Vec3,
    this_world: Affine3A,
    out: &mut Vec<ProcInstruction>,
) {
    let inverse_world = Mat4::from(this_world).inverse().to_cols_array_2d();
    // Smallest axis scale of the composed transform. For uniform
    // scale `S` all three axes have length `S` so this is exact; for
    // non-uniform scale using min() is a conservative
    // 1-Lipschitz-safe bound (it underestimates |d_world|, which
    // just means the classifier subdivides slightly more than
    // necessary in stretched axes — correct, not over-aggressive).
    // `.abs()` handles mirror (negative scale) from Mirror effects
    // or flipped parent transforms.
    let (scale, _rot, _trans) = this_world.to_scale_rotation_translation();
    let distance_scale = scale.abs().min_element().max(1e-6);
    out.push(ProcInstruction {
        op: op as u32,
        arity: 0,
        material_combine: 0,
        material_id: material_id as u32,
        node_id: node_id.0,
        distance_scale,
        _pad1: 0,
        _pad2: 0,
        params,
        color: [color.x, color.y, color.z, 0.0],
        inverse_world,
    });
}


