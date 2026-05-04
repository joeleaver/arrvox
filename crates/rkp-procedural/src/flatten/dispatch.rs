//! Tree-walk that emits a linear `ProcInstruction` stream from a tree:
//! `emit` dispatches on `NodeKind`; `emit_children_with_implicit_union` is
//! the helper for multi-child Root and effect nodes.

use glam::{Affine3A, Mat4};

use crate::arena::{NodeId, ProceduralObject};
use crate::node_kind::NodeKind;

use super::{combinator_radius, emit_primitive, material_combine_bits, OpKind, ProcInstruction, pack_rgb_u24};

/// Emit every child (in order) then, if two or more emitted any
/// instructions, append an implicit `Union(arity = emitted, Winner)`
/// to collapse them onto a single stack slot. Used by `Root` and by
/// multi-child effects like `NoiseDisplace` / `Mirror` / `Array` so
/// primitives dropped directly under them behave as if wrapped in a
/// Union. Returns the number of children that actually produced
/// instructions — callers that need to know whether to keep their
/// PUSH/POP wrapper use this to decide.
fn emit_children_with_implicit_union(
    obj: &ProceduralObject,
    children: &[NodeId],
    ancestor_transform: Affine3A,
    out: &mut Vec<ProcInstruction>,
) -> u32 {
    let mut emitted = 0u32;
    for &child_id in children {
        let before = out.len();
        emit(obj, child_id, ancestor_transform, out);
        if out.len() > before {
            emitted += 1;
        }
    }
    if emitted > 1 {
        out.push(ProcInstruction {
            op: OpKind::Union as u32,
            arity: emitted,
            material_combine: material_combine_bits(
                crate::node_kind::MaterialCombine::Winner,
            ),
            material_id: 0,
            node_id: u32::MAX,
            distance_scale: 1.0, _pad1: 0, _pad2: 0,
            params: [0.0; 8],
            color: [0.0; 4],
            inverse_world: Mat4::IDENTITY.to_cols_array_2d(),
        });
    }
    emitted
}

/// Post-order emit: walk children first, then emit this node.
/// `ancestor_transform` is the composed root→parent transform. Each
/// primitive's stored `inverse_world` is therefore `(ancestor * self).inverse()`.
pub(super) fn emit(
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
        NodeKind::Root => {
            // Top-level container. Children are implicitly unioned
            // (Winner material-combine) — no explicit Union needed.
            // An explicit `Union` node still exists for Layered /
            // Blend material handling, but Root covers the common
            // case where the user just wants shapes side-by-side.
            emit_children_with_implicit_union(obj, &node.children, this_world, out);
        }
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
                distance_scale: 1.0, _pad1: 0, _pad2: 0,
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
                    distance_scale: 1.0, _pad1: 0, _pad2: 0,
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
            // PUSH/POP brackets around the children's combined stream.
            // Children implicitly union into one sample before the
            // mirror fold applies. World-space plane from (A*M):
            //
            //   origin = (A*M).translation
            //   normal = normalize(inverse_transpose((A*M).linear) · axis)
            //
            // The inverse-transpose handles non-uniform scale correctly.
            // Primitives below still carry `inverse_world = (A*M*C).inverse()`,
            // so a world-space reflection at the mirror plane yields
            // the right primitive-local position.
            if node.children.is_empty() {
                return;
            }
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
            let checkpoint = out.len();
            out.push(ProcInstruction {
                op: OpKind::PushMirror as u32,
                arity: 0,
                material_combine: 0,
                material_id: 0,
                node_id: id.0,
                distance_scale: 1.0, _pad1: 0, _pad2: 0,
                params: params_push,
                color: [0.0; 4],
                inverse_world: Mat4::IDENTITY.to_cols_array_2d(),
            });
            let emitted = emit_children_with_implicit_union(
                obj, &node.children, this_world, out,
            );
            if emitted == 0 {
                // No children produced instructions — skip the whole
                // effect. Truncating back undoes the PUSH we optimistically
                // emitted; without this the POP would shrink whatever
                // was on the sample stack from the outer scope.
                out.truncate(checkpoint);
                return;
            }
            out.push(ProcInstruction {
                op: OpKind::PopMirror as u32,
                arity: 0,
                material_combine: 0,
                material_id: 0,
                node_id: u32::MAX,
                distance_scale: 1.0, _pad1: 0, _pad2: 0,
                params: [0.0; 8],
                color: [0.0; 4],
                inverse_world: Mat4::IDENTITY.to_cols_array_2d(),
            });
        }

        NodeKind::MaterialByHeight(p) => {
            // Post-op effect: emit children first (implicit Union
            // collapses multi-child to one sample), then the rewrite
            // instruction that mutates the top sample's material fields.
            let emitted = emit_children_with_implicit_union(
                obj, &node.children, this_world, out,
            );
            if emitted == 0 {
                return;
            }
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
                distance_scale: 1.0, _pad1: 0, _pad2: 0,
                params: params_post,
                color: [0.0; 4],
                inverse_world,
            });
        }

        NodeKind::ColorByHeight(p) => {
            let emitted = emit_children_with_implicit_union(
                obj, &node.children, this_world, out,
            );
            if emitted == 0 {
                return;
            }
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
                distance_scale: 1.0, _pad1: 0, _pad2: 0,
                params: params_post,
                color: color_post,
                inverse_world,
            });
        }

        NodeKind::MaterialByNoise(p) => {
            let emitted = emit_children_with_implicit_union(
                obj, &node.children, this_world, out,
            );
            if emitted == 0 {
                return;
            }
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
                distance_scale: 1.0, _pad1: 0, _pad2: 0,
                params: params_post,
                color: color_post,
                inverse_world,
            });
        }

        NodeKind::ColorByNoise(p) => {
            let emitted = emit_children_with_implicit_union(
                obj, &node.children, this_world, out,
            );
            if emitted == 0 {
                return;
            }
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
                distance_scale: 1.0, _pad1: 0, _pad2: 0,
                params: params_post,
                color: color_post,
                inverse_world,
            });
        }

        NodeKind::Array(p) => {
            // PUSH/POP brackets around the single child's stream. Like
            // Mirror, Array folds in *world space* using the node's
            // baked world transform — the three unit axes come from
            // its rotation, the origin from its translation, the
            // spacings from `params`. Folding in world means the
            // primitives below can keep their usual
            // `inverse_world = (A*M*C).inverse()` transforms unchanged.
            //
            // Packing (all in world space):
            //   inverse_world.x_axis.xyz = unit X axis (column 0)
            //   inverse_world.y_axis.xyz = unit Y axis (column 1)
            //   inverse_world.z_axis.xyz = unit Z axis (column 2)
            //   inverse_world.w_axis.xyz = origin       (column 3)
            //   params_lo.xyz            = spacings (local units)
            //   params_hi.xyz            = counts   (as f32)
            if node.children.is_empty() {
                return;
            }
            // Orthonormalize the baked world rotation. Non-uniform
            // scale on the Array is supported — its per-axis stretch
            // shows up in the spacings here, not in the unit axes —
            // but shear would break the orthogonal-axis assumption
            // (we never emit shear, so this is safe).
            let linear = glam::Mat3::from(this_world.matrix3);
            let x_axis = linear.x_axis.normalize_or_zero();
            let y_axis = linear.y_axis.normalize_or_zero();
            let z_axis = linear.z_axis.normalize_or_zero();
            let origin = glam::Vec3::from(this_world.translation);
            let scale = glam::Vec3::new(
                linear.x_axis.length(),
                linear.y_axis.length(),
                linear.z_axis.length(),
            );
            let spacing = glam::Vec3::from(p.spacings) * scale;
            let counts_f = [
                p.counts[0].max(1) as f32,
                p.counts[1].max(1) as f32,
                p.counts[2].max(1) as f32,
            ];
            let mat = glam::Mat4::from_cols(
                x_axis.extend(0.0),
                y_axis.extend(0.0),
                z_axis.extend(0.0),
                origin.extend(1.0),
            );
            let checkpoint = out.len();
            out.push(ProcInstruction {
                op: OpKind::PushArray as u32,
                arity: 0,
                material_combine: 0,
                material_id: 0,
                node_id: id.0,
                distance_scale: 1.0, _pad1: 0, _pad2: 0,
                params: [
                    spacing.x, spacing.y, spacing.z, 0.0,
                    counts_f[0], counts_f[1], counts_f[2], 0.0,
                ],
                color: [0.0; 4],
                inverse_world: mat.to_cols_array_2d(),
            });
            let emitted = emit_children_with_implicit_union(
                obj, &node.children, this_world, out,
            );
            if emitted == 0 {
                out.truncate(checkpoint);
                return;
            }
            out.push(ProcInstruction {
                op: OpKind::PopArray as u32,
                arity: 0,
                material_combine: 0,
                material_id: 0,
                node_id: u32::MAX,
                distance_scale: 1.0, _pad1: 0, _pad2: 0,
                params: [0.0; 8],
                color: [0.0; 4],
                inverse_world: Mat4::IDENTITY.to_cols_array_2d(),
            });
        }

        NodeKind::NoiseDisplace(p) => {
            // PUSH/POP brackets around the children's combined stream.
            // Multi-child: they're implicitly unioned into a single
            // sample before the warp's POP runs.
            if node.children.is_empty() {
                return;
            }
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
            let checkpoint = out.len();
            out.push(ProcInstruction {
                op: OpKind::PushNoiseDisplace as u32,
                arity: 0,
                material_combine: 0,
                material_id: 0,
                node_id: id.0,
                distance_scale: 1.0, _pad1: 0, _pad2: 0,
                params: params_push,
                color: [0.0; 4],
                inverse_world: Mat4::IDENTITY.to_cols_array_2d(),
            });
            let emitted = emit_children_with_implicit_union(
                obj, &node.children, this_world, out,
            );
            if emitted == 0 {
                out.truncate(checkpoint);
                return;
            }
            out.push(ProcInstruction {
                op: OpKind::PopNoiseDisplace as u32,
                arity: 0,
                material_combine: 0,
                material_id: 0,
                node_id: u32::MAX,
                distance_scale: 1.0, _pad1: 0, _pad2: 0,
                params: params_pop,
                color: [0.0; 4],
                inverse_world: Mat4::IDENTITY.to_cols_array_2d(),
            });
        }
    }

    let _ = nk::MaterialCombine::Winner; // silence unused-import on macros
}
