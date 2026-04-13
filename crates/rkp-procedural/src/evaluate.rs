//! Recursive tree evaluation — the core entry point for sampling a procedural
//! object at a world-space position.

use glam::Vec3;

use crate::arena::{NodeId, ProceduralObject};
use crate::combine::{combine_intersect, combine_subtract, combine_union};
use crate::leaves::eval_leaf;
use crate::node_kind::NodeKind;
use crate::sample::Sample;

/// Sample the procedural object tree at a world-space position.
///
/// Evaluates the full tree from the root, transforming the position into each
/// node's local space and combining results according to combinator rules.
pub fn sample_tree(obj: &ProceduralObject, pos: Vec3) -> Sample {
    sample_node(obj, obj.root(), pos)
}

/// Recursively evaluate a single node and its subtree.
///
/// `pos` is in the **parent's** local space. This function applies the node's
/// own transform (inverse) to get into the node's local space before evaluating.
fn sample_node(obj: &ProceduralObject, id: NodeId, pos: Vec3) -> Sample {
    let node = match obj.get(id) {
        Some(n) => n,
        None => return Sample::EMPTY,
    };

    // Transform position into this node's local space.
    let local_pos = node.transform.inverse().transform_point3(pos);

    match &node.kind {
        // Leaves: evaluate the shape directly.
        kind if kind.is_leaf() => eval_leaf(local_pos, kind),

        // Union: combine all children.
        NodeKind::Union { material_combine } => {
            let mode = *material_combine;
            let mut result = Sample::EMPTY;
            for &child_id in &node.children {
                let child_sample = sample_node(obj, child_id, local_pos);
                if child_sample.is_empty() {
                    continue;
                }
                if result.is_empty() {
                    result = child_sample;
                } else {
                    result = combine_union(&result, &child_sample, mode);
                }
            }
            result
        }

        // Intersect: combine all children.
        NodeKind::Intersect { material_combine } => {
            let mode = *material_combine;
            let children = &node.children;
            if children.is_empty() {
                return Sample::EMPTY;
            }
            let mut result = sample_node(obj, children[0], local_pos);
            for &child_id in &children[1..] {
                if result.is_empty() {
                    return Sample::EMPTY;
                }
                let child_sample = sample_node(obj, child_id, local_pos);
                result = combine_intersect(&result, &child_sample, mode);
            }
            result
        }

        // Subtract: first child is base, remaining are cutters.
        NodeKind::Subtract => {
            let children = &node.children;
            if children.is_empty() {
                return Sample::EMPTY;
            }
            let mut result = sample_node(obj, children[0], local_pos);
            for &cutter_id in &children[1..] {
                if result.is_empty() {
                    return Sample::EMPTY;
                }
                let cutter_sample = sample_node(obj, cutter_id, local_pos);
                result = combine_subtract(&result, &cutter_sample);
            }
            result
        }

        _ => Sample::EMPTY,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node_kind::*;
    use glam::Affine3A;

    const EPS: f32 = 1e-3;

    fn sphere(radius: f32, material_id: u16) -> NodeKind {
        NodeKind::Sphere(SphereParams {
            radius,
            falloff: 0.05,
            material_id,
            ..Default::default()
        })
    }

    /// A single sphere at the origin.
    #[test]
    fn single_sphere() {
        let mut obj = ProceduralObject::new(NodeKind::Union {
            material_combine: MaterialCombine::Winner,
        });
        obj.add_child(obj.root(), sphere(1.0, 0));

        // Center should be opaque.
        let s = sample_tree(&obj, Vec3::ZERO);
        assert!((s.opacity - 1.0).abs() < EPS);

        // Far away should be empty.
        let s = sample_tree(&obj, Vec3::new(10.0, 0.0, 0.0));
        assert!(s.opacity < EPS);
    }

    /// Union of two overlapping spheres.
    #[test]
    fn union_of_two_spheres() {
        let mut obj = ProceduralObject::new(NodeKind::Union {
            material_combine: MaterialCombine::Winner,
        });
        let a = obj.add_child(obj.root(), sphere(0.5, 1));
        let b = obj.add_child(obj.root(), sphere(0.5, 2));

        // Move sphere A left, B right.
        obj.set_transform(a, Affine3A::from_translation(Vec3::new(-0.3, 0.0, 0.0)));
        obj.set_transform(b, Affine3A::from_translation(Vec3::new(0.3, 0.0, 0.0)));

        // Center (overlap region) should be opaque.
        let s = sample_tree(&obj, Vec3::ZERO);
        assert!(s.opacity > 0.5);

        // Deep inside A should have material 1.
        let s = sample_tree(&obj, Vec3::new(-0.3, 0.0, 0.0));
        assert!((s.opacity - 1.0).abs() < EPS);
        assert_eq!(s.material_id, 1);

        // Deep inside B should have material 2.
        let s = sample_tree(&obj, Vec3::new(0.3, 0.0, 0.0));
        assert!((s.opacity - 1.0).abs() < EPS);
        assert_eq!(s.material_id, 2);
    }

    /// Subtract: sphere with a hole cut out.
    #[test]
    fn subtract_cuts_hole() {
        let mut obj = ProceduralObject::new(NodeKind::Subtract);
        obj.add_child(obj.root(), sphere(1.0, 1)); // base
        obj.add_child(obj.root(), sphere(0.5, 2)); // cutter

        // Center should be empty (cutter removes it).
        let s = sample_tree(&obj, Vec3::ZERO);
        assert!(s.opacity < EPS);

        // Edge of base (outside cutter) should still be opaque.
        let s = sample_tree(&obj, Vec3::new(0.8, 0.0, 0.0));
        assert!(s.opacity > 0.5);
        assert_eq!(s.material_id, 1); // base material preserved
    }

    /// Intersect: only the overlap region survives.
    #[test]
    fn intersect_keeps_overlap() {
        let mut obj = ProceduralObject::new(NodeKind::Intersect {
            material_combine: MaterialCombine::Winner,
        });
        let a = obj.add_child(obj.root(), sphere(0.5, 1));
        let b = obj.add_child(obj.root(), sphere(0.5, 2));

        obj.set_transform(a, Affine3A::from_translation(Vec3::new(-0.2, 0.0, 0.0)));
        obj.set_transform(b, Affine3A::from_translation(Vec3::new(0.2, 0.0, 0.0)));

        // Center (overlap) should be opaque.
        let s = sample_tree(&obj, Vec3::ZERO);
        assert!(s.opacity > 0.5);

        // Far inside A but outside B should be empty.
        let s = sample_tree(&obj, Vec3::new(-0.4, 0.0, 0.0));
        assert!(s.opacity < 0.5);
    }

    /// Nested combinators: union of (sphere) and (subtract of two spheres).
    #[test]
    fn nested_tree() {
        let mut obj = ProceduralObject::new(NodeKind::Union {
            material_combine: MaterialCombine::Winner,
        });

        // Left: plain sphere at (-2, 0, 0).
        let left = obj.add_child(obj.root(), sphere(0.5, 1));
        obj.set_transform(left, Affine3A::from_translation(Vec3::new(-2.0, 0.0, 0.0)));

        // Right: subtract node at (2, 0, 0).
        let sub = obj.add_child(obj.root(), NodeKind::Subtract);
        obj.set_transform(sub, Affine3A::from_translation(Vec3::new(2.0, 0.0, 0.0)));
        obj.add_child(sub, sphere(0.5, 2)); // base
        obj.add_child(sub, sphere(0.3, 3)); // cutter

        // Left sphere center.
        let s = sample_tree(&obj, Vec3::new(-2.0, 0.0, 0.0));
        assert!((s.opacity - 1.0).abs() < EPS);
        assert_eq!(s.material_id, 1);

        // Right subtract: center should be empty (cut away).
        let s = sample_tree(&obj, Vec3::new(2.0, 0.0, 0.0));
        assert!(s.opacity < EPS);

        // Right subtract: shell region should be opaque.
        let s = sample_tree(&obj, Vec3::new(2.4, 0.0, 0.0));
        assert!(s.opacity > 0.5);
        assert_eq!(s.material_id, 2);
    }

    /// Transform on a parent combinator affects all children.
    #[test]
    fn parent_transform_propagates() {
        let mut obj = ProceduralObject::new(NodeKind::Union {
            material_combine: MaterialCombine::Winner,
        });
        obj.add_child(obj.root(), sphere(0.5, 1));

        // Move the root union node. The sphere should move with it.
        obj.set_transform(obj.root(), Affine3A::from_translation(Vec3::new(5.0, 0.0, 0.0)));

        // The sphere center is now at (5, 0, 0).
        let s = sample_tree(&obj, Vec3::new(5.0, 0.0, 0.0));
        assert!((s.opacity - 1.0).abs() < EPS);

        // Origin should be empty.
        let s = sample_tree(&obj, Vec3::ZERO);
        assert!(s.opacity < EPS);
    }
}
