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
use crate::node_kind::{MaterialCombine, NodeKind};

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
    pub _pad0: u32,
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
fn pack_rgb_u24(c: glam::Vec3) -> u32 {
    let r = (c.x.clamp(0.0, 1.0) * 255.0) as u32;
    let g = (c.y.clamp(0.0, 1.0) * 255.0) as u32;
    let b = (c.z.clamp(0.0, 1.0) * 255.0) as u32;
    r | (g << 8) | (b << 16)
}

/// Inverse of `pack_rgb_u24` — unpack a u24-bit-laid f32 back into
/// a Vec3 color. Used only by the CPU RPN exec in flatten tests so
/// it mirrors the WGSL unpack bit-for-bit.
fn unpack_rgb_u24(packed: f32) -> glam::Vec3 {
    let bits = packed.to_bits();
    let r = (bits & 0xFF) as f32 / 255.0;
    let g = ((bits >> 8) & 0xFF) as f32 / 255.0;
    let b = ((bits >> 16) & 0xFF) as f32 / 255.0;
    glam::Vec3::new(r, g, b)
}

/// Encode a `MaterialCombine` as a u32 for the GPU. Values match the
/// WGSL `MAT_COMBINE_*` constants.
fn material_combine_bits(m: MaterialCombine) -> u32 {
    match m {
        MaterialCombine::Winner => 0,
        MaterialCombine::Layered => 1,
        MaterialCombine::Blend { .. } => 2,
    }
}

/// For `Blend{radius}`, the radius rides along in combinator instructions'
/// `params[0]` — unused on the Winner/Layered paths and zero there.
fn combinator_radius(m: MaterialCombine) -> f32 {
    match m {
        MaterialCombine::Blend { radius } => radius.max(1e-6),
        _ => 0.0,
    }
}

/// Flatten a procedural tree into a linear instruction stream ready for
/// GPU upload. Returns an empty vector for an empty tree.
pub fn flatten_tree(obj: &ProceduralObject) -> Vec<ProcInstruction> {
    use crate::node_kind::*;

    let mut out = Vec::new();
    emit(obj, obj.root(), Affine3A::IDENTITY, &mut out);
    out
}

/// Post-order emit: walk children first, then emit this node.
/// `ancestor_transform` is the composed root→parent transform. Each
/// primitive's stored `inverse_world` is therefore `(ancestor * self).inverse()`.
fn emit(
    obj: &ProceduralObject,
    id: NodeId,
    ancestor_transform: Affine3A,
    out: &mut Vec<ProcInstruction>,
) {
    use crate::node_kind as nk;
    let Some(node) = obj.get(id) else {
        return;
    };
    let this_world = ancestor_transform * node.transform;

    match &node.kind {
        NodeKind::Sphere(p) => emit_primitive(
            OpKind::Sphere,
            id,
            [p.radius, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            p.material_id,
            p.color,
            this_world,
            out,
        ),
        NodeKind::Box(p) => emit_primitive(
            OpKind::Box,
            id,
            [p.half_extents.x, p.half_extents.y, p.half_extents.z, p.rounding, 0.0, 0.0, 0.0, 0.0],
            p.material_id,
            p.color,
            this_world,
            out,
        ),
        NodeKind::Capsule(p) => emit_primitive(
            OpKind::Capsule,
            id,
            [p.radius, p.half_height, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            p.material_id,
            p.color,
            this_world,
            out,
        ),
        NodeKind::Cylinder(p) => emit_primitive(
            OpKind::Cylinder,
            id,
            [p.radius, p.half_height, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            p.material_id,
            p.color,
            this_world,
            out,
        ),
        NodeKind::Torus(p) => emit_primitive(
            OpKind::Torus,
            id,
            [p.major_radius, p.minor_radius, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            p.material_id,
            p.color,
            this_world,
            out,
        ),
        NodeKind::Plane(p) => emit_primitive(
            OpKind::Plane,
            id,
            [0.0; 8],
            p.material_id,
            p.color,
            this_world,
            out,
        ),
        NodeKind::Ramp(p) => emit_primitive(
            OpKind::Ramp,
            id,
            [p.half_length, p.half_height, p.half_width, 0.0, 0.0, 0.0, 0.0, 0.0],
            p.material_id,
            p.color,
            this_world,
            out,
        ),

        NodeKind::Union { material_combine }
        | NodeKind::Intersect { material_combine } => {
            // Emit children first. Combinator transforms are absorbed
            // into children's ancestor_transform — they don't get their
            // own instruction other than the final op.
            let mut emitted = 0u32;
            for &child_id in &node.children {
                let before = out.len();
                emit(obj, child_id, this_world, out);
                if out.len() > before {
                    emitted += 1;
                }
            }
            if emitted == 0 {
                return;
            }
            let op = match &node.kind {
                NodeKind::Union { .. } => OpKind::Union,
                NodeKind::Intersect { .. } => OpKind::Intersect,
                _ => unreachable!(),
            };
            let mut params = [0.0f32; 8];
            params[0] = combinator_radius(*material_combine);
            out.push(ProcInstruction {
                op: op as u32,
                arity: emitted,
                material_combine: material_combine_bits(*material_combine),
                material_id: 0,
                node_id: u32::MAX,
                _pad0: 0, _pad1: 0, _pad2: 0,
                params,
                color: [0.0; 4],
                inverse_world: Mat4::IDENTITY.to_cols_array_2d(),
            });
        }

        NodeKind::Subtract => {
            let mut emitted = 0u32;
            for &child_id in &node.children {
                let before = out.len();
                emit(obj, child_id, this_world, out);
                if out.len() > before {
                    emitted += 1;
                }
            }
            // Subtract needs at least a minuend AND a cutter — with
            // only the minuend the op would degenerate to a copy. The
            // shader's arity-based pop still works either way, so the
            // safer thing is to emit the op only when there's real
            // work to do; otherwise the single emitted child just sits
            // on the stack as the combinator's output (identical to
            // CPU behavior: a Subtract with one child returns that
            // child unchanged).
            if emitted >= 2 {
                out.push(ProcInstruction {
                    op: OpKind::Subtract as u32,
                    arity: emitted,
                    material_combine: 0,
                    material_id: 0,
                    node_id: u32::MAX,
                    _pad0: 0, _pad1: 0, _pad2: 0,
                    params: [0.0; 8],
                    color: [0.0; 4],
                    inverse_world: Mat4::IDENTITY.to_cols_array_2d(),
                });
            }
            // emitted == 0 or 1: leave the stack as-is. A degenerate
            // Subtract with no minuend contributes nothing; a Subtract
            // with only the minuend equals the minuend.
        }

        NodeKind::Mirror(p) => {
            // PUSH/POP brackets around the single child's stream. The
            // fold runs in world space using the plane derived from
            // this Mirror's world transform (`this_world = A*M`):
            //
            //   origin = (A*M).translation
            //   normal = normalize(inverse_transpose((A*M).linear) · axis)
            //
            // The inverse-transpose is the right transformation for a
            // plane normal — it handles non-uniform scale correctly by
            // producing the direction perpendicular to the transformed
            // plane, not to the scaled axis vector. Primitives below
            // still carry `inverse_world = (A*M*C).inverse()` that
            // undoes the composed transform, so a world-space reflection
            // at the Mirror plane yields the same primitive-local
            // position as the CPU evaluator's local-frame fold.
            let Some(&child_id) = node.children.first() else {
                return;
            };
            let axis_unit = match p.axis {
                crate::node_kind::MirrorAxis::X => glam::Vec3::X,
                crate::node_kind::MirrorAxis::Y => glam::Vec3::Y,
                crate::node_kind::MirrorAxis::Z => glam::Vec3::Z,
            };
            let origin = glam::Vec3::from(this_world.translation);
            let linear_inv_t = glam::Mat3::from(this_world.matrix3).inverse().transpose();
            let normal = (linear_inv_t * axis_unit).normalize_or_zero();
            let params_push = [
                origin.x, origin.y, origin.z, 0.0,
                normal.x, normal.y, normal.z, 0.0,
            ];
            out.push(ProcInstruction {
                op: OpKind::PushMirror as u32,
                arity: 0,
                material_combine: 0,
                material_id: 0,
                node_id: id.0,
                _pad0: 0, _pad1: 0, _pad2: 0,
                params: params_push,
                color: [0.0; 4],
                inverse_world: Mat4::IDENTITY.to_cols_array_2d(),
            });
            emit(obj, child_id, this_world, out);
            out.push(ProcInstruction {
                op: OpKind::PopMirror as u32,
                arity: 0,
                material_combine: 0,
                material_id: 0,
                node_id: u32::MAX,
                _pad0: 0, _pad1: 0, _pad2: 0,
                params: [0.0; 8],
                color: [0.0; 4],
                inverse_world: Mat4::IDENTITY.to_cols_array_2d(),
            });
        }

        NodeKind::MaterialByHeight(p) => {
            let Some(&child_id) = node.children.first() else {
                return;
            };
            emit(obj, child_id, this_world, out);
            // Post-op instruction. `inverse_world` = (A*M).inverse()
            // so the shader can transform pos_stack[pos_top] into the
            // effect's local frame and read `local.y`. Band/threshold
            // data packed into the `params` array.
            let inverse_world = Mat4::from(this_world).inverse().to_cols_array_2d();
            let params_post = [
                p.low_to_mid,
                p.mid_to_high,
                p.transition_width,
                p.low_material as f32,
                p.mid_material as f32,
                p.high_material as f32,
                0.0, 0.0,
            ];
            out.push(ProcInstruction {
                op: OpKind::ApplyMaterialByHeight as u32,
                arity: 0,
                material_combine: 0,
                material_id: 0,
                node_id: id.0,
                _pad0: 0, _pad1: 0, _pad2: 0,
                params: params_post,
                color: [0.0; 4],
                inverse_world,
            });
        }

        NodeKind::ColorByHeight(p) => {
            let Some(&child_id) = node.children.first() else {
                return;
            };
            emit(obj, child_id, this_world, out);
            let inverse_world = Mat4::from(this_world).inverse().to_cols_array_2d();
            // Each color rides in the last 3 slots of its band's
            // vec4 (params_lo for low, params_hi for mid, color for
            // high); thresholds/width occupy the respective `.x` slots
            // of `params_lo` and `params_hi`, and the `color` field's
            // `.w` slot carries `transition_width`.
            //
            // Layout:
            //   params_lo = [low_to_mid,   low.r,  low.g,  low.b]
            //   params_hi = [mid_to_high,  mid.r,  mid.g,  mid.b]
            //   color     = [high.r,       high.g, high.b, transition_width]
            let params_post = [
                p.low_to_mid, p.low_color.x, p.low_color.y, p.low_color.z,
                p.mid_to_high, p.mid_color.x, p.mid_color.y, p.mid_color.z,
            ];
            let color_post = [p.high_color.x, p.high_color.y, p.high_color.z, p.transition_width];
            out.push(ProcInstruction {
                op: OpKind::ApplyColorByHeight as u32,
                arity: 0,
                material_combine: 0,
                material_id: 0,
                node_id: id.0,
                _pad0: 0, _pad1: 0, _pad2: 0,
                params: params_post,
                color: color_post,
                inverse_world,
            });
        }

        NodeKind::MaterialByNoise(p) => {
            let Some(&child_id) = node.children.first() else {
                return;
            };
            emit(obj, child_id, this_world, out);
            let inverse_world = Mat4::from(this_world).inverse().to_cols_array_2d();
            // Layout:
            //   params = [low_to_mid, mid_to_high, transition_width,
            //             frequency, seed, octaves,
            //             low_material, mid_material]
            //   color  = [high_material, 0, 0, 0]
            // Material ids fit exactly in f32 (u16 max 65535, f32
            // represents ints up to 2^24 exactly).
            let params_post = [
                p.low_to_mid, p.mid_to_high, p.transition_width,
                p.frequency, p.seed as f32, p.octaves as f32,
                p.low_material as f32, p.mid_material as f32,
            ];
            let color_post = [p.high_material as f32, 0.0, 0.0, 0.0];
            out.push(ProcInstruction {
                op: OpKind::ApplyMaterialByNoise as u32,
                arity: 0,
                material_combine: 0,
                material_id: 0,
                node_id: id.0,
                _pad0: 0, _pad1: 0, _pad2: 0,
                params: params_post,
                color: color_post,
                inverse_world,
            });
        }

        NodeKind::ColorByNoise(p) => {
            let Some(&child_id) = node.children.first() else {
                return;
            };
            emit(obj, child_id, this_world, out);
            let inverse_world = Mat4::from(this_world).inverse().to_cols_array_2d();
            // Layout:
            //   params = [low_to_mid, mid_to_high, transition_width,
            //             frequency, seed, octaves,
            //             low_rgb_packed, mid_rgb_packed]
            //   color  = [high_rgb_packed, 0, 0, 0]
            // Each color is RGB quantized to u8 and packed into u24;
            // the u32 is bit-cast to f32 for storage (values up to
            // 2^24 are exact in f32). The shader unpacks with a u32
            // bit-cast + byte masking to reconstruct the Vec3. The
            // ~1/255 precision matches the raymarch G-buffer's
            // RGB565 quantization, so no visual loss at the seam.
            let params_post = [
                p.low_to_mid, p.mid_to_high, p.transition_width,
                p.frequency, p.seed as f32, p.octaves as f32,
                f32::from_bits(pack_rgb_u24(p.low_color)),
                f32::from_bits(pack_rgb_u24(p.mid_color)),
            ];
            let color_post = [
                f32::from_bits(pack_rgb_u24(p.high_color)),
                0.0, 0.0, 0.0,
            ];
            out.push(ProcInstruction {
                op: OpKind::ApplyColorByNoise as u32,
                arity: 0,
                material_combine: 0,
                material_id: 0,
                node_id: id.0,
                _pad0: 0, _pad1: 0, _pad2: 0,
                params: params_post,
                color: color_post,
                inverse_world,
            });
        }

        NodeKind::NoiseDisplace(p) => {
            // PUSH/POP brackets around the child stream. Position-warp
            // params ride along in `params` so the shader can execute
            // the warp without dereferencing this Rust struct. Only
            // the first child is rendered (same rule as the evaluator);
            // additional children are silently ignored.
            let Some(&child_id) = node.children.first() else {
                return;
            };
            let params_push = [
                p.amplitude,
                p.frequency,
                p.seed as f32,
                p.octaves as f32,
                0.0, 0.0, 0.0, 0.0,
            ];
            let params_pop = [
                p.amplitude,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ];
            out.push(ProcInstruction {
                op: OpKind::PushNoiseDisplace as u32,
                arity: 0,
                material_combine: 0,
                material_id: 0,
                node_id: id.0,
                _pad0: 0, _pad1: 0, _pad2: 0,
                params: params_push,
                color: [0.0; 4],
                inverse_world: Mat4::IDENTITY.to_cols_array_2d(),
            });
            emit(obj, child_id, this_world, out);
            out.push(ProcInstruction {
                op: OpKind::PopNoiseDisplace as u32,
                arity: 0,
                material_combine: 0,
                material_id: 0,
                node_id: u32::MAX,
                _pad0: 0, _pad1: 0, _pad2: 0,
                params: params_pop,
                color: [0.0; 4],
                inverse_world: Mat4::IDENTITY.to_cols_array_2d(),
            });
        }
    }

    let _ = nk::MaterialCombine::Winner; // silence unused-import on macros
}

fn emit_primitive(
    op: OpKind,
    node_id: NodeId,
    params: [f32; 8],
    material_id: u16,
    color: glam::Vec3,
    this_world: Affine3A,
    out: &mut Vec<ProcInstruction>,
) {
    let inverse_world = Mat4::from(this_world).inverse().to_cols_array_2d();
    out.push(ProcInstruction {
        op: op as u32,
        arity: 0,
        material_combine: 0,
        material_id: material_id as u32,
        node_id: node_id.0,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
        params,
        color: [color.x, color.y, color.z, 0.0],
        inverse_world,
    });
}

