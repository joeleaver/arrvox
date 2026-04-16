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

/// Encode a `MaterialCombine` as a u32 for the GPU. Values match the
/// WGSL `MAT_COMBINE_*` constants. `Blend`'s radius is not currently
/// plumbed through — the shader uses `Winner` behavior in that case
/// (same material selection, no smooth band). If/when we wire in the
/// radius, add it as a second u32 field on `ProcInstruction`.
fn material_combine_bits(m: MaterialCombine) -> u32 {
    match m {
        MaterialCombine::Winner => 0,
        MaterialCombine::Layered => 1,
        MaterialCombine::Blend { .. } => 0,
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
            [p.radius, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            p.material_id,
            p.color,
            this_world,
            out,
        ),
        NodeKind::Box(p) => emit_primitive(
            OpKind::Box,
            [p.half_extents.x, p.half_extents.y, p.half_extents.z, p.rounding, 0.0, 0.0, 0.0, 0.0],
            p.material_id,
            p.color,
            this_world,
            out,
        ),
        NodeKind::Capsule(p) => emit_primitive(
            OpKind::Capsule,
            [p.radius, p.half_height, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            p.material_id,
            p.color,
            this_world,
            out,
        ),
        NodeKind::Cylinder(p) => emit_primitive(
            OpKind::Cylinder,
            [p.radius, p.half_height, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            p.material_id,
            p.color,
            this_world,
            out,
        ),
        NodeKind::Torus(p) => emit_primitive(
            OpKind::Torus,
            [p.major_radius, p.minor_radius, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            p.material_id,
            p.color,
            this_world,
            out,
        ),
        NodeKind::Plane(p) => emit_primitive(
            OpKind::Plane,
            [0.0; 8],
            p.material_id,
            p.color,
            this_world,
            out,
        ),
        NodeKind::Ramp(p) => emit_primitive(
            OpKind::Ramp,
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
            out.push(ProcInstruction {
                op: op as u32,
                arity: emitted,
                material_combine: material_combine_bits(*material_combine),
                material_id: 0,
                params: [0.0; 8],
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
                    params: [0.0; 8],
                    color: [0.0; 4],
                    inverse_world: Mat4::IDENTITY.to_cols_array_2d(),
                });
            }
            // emitted == 0 or 1: leave the stack as-is. A degenerate
            // Subtract with no minuend contributes nothing; a Subtract
            // with only the minuend equals the minuend.
        }
    }

    let _ = nk::MaterialCombine::Winner; // silence unused-import on macros
}

fn emit_primitive(
    op: OpKind,
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
        params,
        color: [color.x, color.y, color.z, 0.0],
        inverse_world,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluate::sample_tree;
    use crate::node_kind::*;
    use crate::sample::Sample;
    use glam::Vec3;

    const VS: f32 = 0.02;

    fn sphere(radius: f32, mat: u16) -> NodeKind {
        NodeKind::Sphere(SphereParams { radius, material_id: mat, ..Default::default() })
    }

    /// Execute an RPN instruction stream on the CPU — mirrors what the
    /// WGSL shader does. Used by the tests to check that flattening is
    /// semantically equivalent to `sample_tree`.
    fn exec(instructions: &[ProcInstruction], world_pos: Vec3) -> Sample {
        use crate::combine::{combine_intersect, combine_subtract, combine_union};
        use crate::leaves::{eval_sphere, eval_box, eval_capsule, eval_cylinder, eval_torus, eval_plane, eval_ramp};

        let mut stack: Vec<Sample> = Vec::with_capacity(16);
        for ins in instructions {
            if ins.op < 100 {
                // Primitive.
                let inv = Mat4::from_cols_array_2d(&ins.inverse_world);
                let local = inv.transform_point3(world_pos);
                let color = Vec3::new(ins.color[0], ins.color[1], ins.color[2]);
                let sample = match ins.op {
                    0 => eval_sphere(local, &SphereParams {
                        radius: ins.params[0], material_id: ins.material_id as u16, color,
                    }, VS),
                    1 => eval_box(local, &BoxParams {
                        half_extents: Vec3::new(ins.params[0], ins.params[1], ins.params[2]),
                        rounding: ins.params[3], material_id: ins.material_id as u16, color,
                    }, VS),
                    2 => eval_capsule(local, &CapsuleParams {
                        radius: ins.params[0], half_height: ins.params[1],
                        material_id: ins.material_id as u16, color,
                    }, VS),
                    3 => eval_cylinder(local, &CylinderParams {
                        radius: ins.params[0], half_height: ins.params[1],
                        material_id: ins.material_id as u16, color,
                    }, VS),
                    4 => eval_torus(local, &TorusParams {
                        major_radius: ins.params[0], minor_radius: ins.params[1],
                        material_id: ins.material_id as u16, color,
                    }, VS),
                    5 => eval_plane(local, &PlaneParams {
                        material_id: ins.material_id as u16, color,
                    }, VS),
                    6 => eval_ramp(local, &RampParams {
                        half_length: ins.params[0], half_height: ins.params[1], half_width: ins.params[2],
                        material_id: ins.material_id as u16, color,
                    }, VS),
                    _ => Sample::EMPTY,
                };
                stack.push(sample);
            } else {
                // Combinator.
                let arity = ins.arity as usize;
                assert!(stack.len() >= arity);
                let base = stack.len() - arity;
                let mode = match ins.material_combine {
                    0 => MaterialCombine::Winner,
                    _ => MaterialCombine::Layered,
                };
                let mut acc = stack[base];
                for k in 1..arity {
                    let rhs = stack[base + k];
                    acc = match ins.op {
                        100 => combine_union(&acc, &rhs, mode),
                        101 => combine_intersect(&acc, &rhs, mode),
                        102 => combine_subtract(&acc, &rhs),
                        _ => acc,
                    };
                }
                stack.truncate(base);
                stack.push(acc);
            }
        }
        stack.pop().unwrap_or(Sample::EMPTY)
    }

    /// Flattening + CPU RPN execution must match `sample_tree` exactly
    /// over a dense grid — same bar as `cached_matches_uncached_dense_grid`.
    #[test]
    fn flatten_matches_sample_tree_dense_grid() {
        use glam::Affine3A;
        let mut obj = ProceduralObject::new(NodeKind::Union {
            material_combine: MaterialCombine::Winner,
        });
        for i in 0..4 {
            let s = obj.add_child(obj.root(), sphere(0.4, i as u16));
            obj.set_transform(s, Affine3A::from_translation(Vec3::new(i as f32 * 1.2 - 1.8, 0.0, 0.0)));
        }
        let sub = obj.add_child(obj.root(), NodeKind::Subtract);
        obj.set_transform(sub, Affine3A::from_translation(Vec3::new(0.0, 1.5, 0.0)));
        obj.add_child(sub, sphere(0.8, 10));
        obj.add_child(sub, sphere(0.3, 11));

        let instructions = flatten_tree(&obj);

        let mut disagree = 0usize;
        for ix in -15..=15 {
            for iy in -5..=8 {
                for iz in -5..=5 {
                    let p = Vec3::new(ix as f32 * 0.3, iy as f32 * 0.3, iz as f32 * 0.3);
                    let ref_s = sample_tree(&obj, p, VS);
                    let flat_s = exec(&instructions, p);
                    if (ref_s.distance - flat_s.distance).abs() > 1e-4 {
                        disagree += 1;
                        if disagree <= 3 {
                            eprintln!(
                                "flatten mismatch at {p:?}: ref={} flat={}",
                                ref_s.distance, flat_s.distance
                            );
                        }
                    }
                }
            }
        }
        assert_eq!(disagree, 0, "flatten+exec diverged from sample_tree at {disagree} points");
    }
}
