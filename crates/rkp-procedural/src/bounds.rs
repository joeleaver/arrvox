//! Per-node bounding box computation.
//!
//! Every node can compute an axis-aligned bounding box in its local space.
//! Parent transforms widen the bounds. Combinators merge children's bounds.

use glam::Vec3;
use rkp_core::Aabb;

use crate::arena::{NodeId, ProceduralObject};
use crate::node_kind::*;

/// An empty AABB sentinel — min > max means "no bounds".
const EMPTY_AABB: Aabb = Aabb {
    min: Vec3::splat(f32::INFINITY),
    max: Vec3::splat(f32::NEG_INFINITY),
};

fn is_empty_aabb(aabb: &Aabb) -> bool {
    aabb.min.x > aabb.max.x
}

/// Compute the world-space AABB for the entire procedural object tree.
pub fn compute_bounds(obj: &ProceduralObject) -> Aabb {
    let aabb = compute_node_bounds(obj, obj.root());
    if is_empty_aabb(&aabb) {
        // Return a zero-size AABB at origin for empty trees.
        Aabb {
            min: Vec3::ZERO,
            max: Vec3::ZERO,
        }
    } else {
        aabb
    }
}

/// Compute the world-space AABB for a node and its subtree.
fn compute_node_bounds(obj: &ProceduralObject, id: NodeId) -> Aabb {
    let node = match obj.get(id) {
        Some(n) => n,
        None => return EMPTY_AABB,
    };

    let local_aabb = match &node.kind {
        NodeKind::Sphere(p) => leaf_sphere_bounds(p),
        NodeKind::Box(p) => leaf_box_bounds(p),
        NodeKind::Capsule(p) => leaf_capsule_bounds(p),
        NodeKind::Cylinder(p) => leaf_cylinder_bounds(p),
        NodeKind::Torus(p) => leaf_torus_bounds(p),
        NodeKind::Plane(p) => leaf_plane_bounds(p),
        NodeKind::Ramp(p) => leaf_ramp_bounds(p),

        // Combinators: union of children's bounds (works for all combinator types
        // because intersect/subtract can only shrink, so the union is conservative).
        NodeKind::Union { .. } | NodeKind::Intersect { .. } | NodeKind::Subtract => {
            let mut combined = EMPTY_AABB;
            for &child_id in &node.children {
                let child_aabb = compute_node_bounds(obj, child_id);
                combined = aabb_union(&combined, &child_aabb);
            }
            combined
        }
    };

    // Don't transform empty AABBs (infinity values produce NaN).
    if is_empty_aabb(&local_aabb) {
        return EMPTY_AABB;
    }

    // Transform the AABB by this node's transform.
    transform_aabb(&local_aabb, &node.transform)
}

// ── Leaf bounds ─────────────────────────────────────────────────────────────

fn leaf_sphere_bounds(p: &SphereParams) -> Aabb {
    Aabb {
        min: Vec3::splat(-p.radius),
        max: Vec3::splat(p.radius),
    }
}

fn leaf_box_bounds(p: &BoxParams) -> Aabb {
    Aabb {
        min: -p.half_extents,
        max: p.half_extents,
    }
}

fn leaf_capsule_bounds(p: &CapsuleParams) -> Aabb {
    let h = p.half_height + p.radius;
    Aabb {
        min: Vec3::new(-p.radius, -h, -p.radius),
        max: Vec3::new(p.radius, h, p.radius),
    }
}

fn leaf_cylinder_bounds(p: &CylinderParams) -> Aabb {
    Aabb {
        min: Vec3::new(-p.radius, -p.half_height, -p.radius),
        max: Vec3::new(p.radius, p.half_height, p.radius),
    }
}

fn leaf_torus_bounds(p: &TorusParams) -> Aabb {
    let r = p.major_radius + p.minor_radius;
    Aabb {
        min: Vec3::new(-r, -p.minor_radius, -r),
        max: Vec3::new(r, p.minor_radius, r),
    }
}

fn leaf_ramp_bounds(p: &RampParams) -> Aabb {
    Aabb {
        min: Vec3::new(-p.half_length, -p.half_height, -p.half_width),
        max: Vec3::new(p.half_length, p.half_height, p.half_width),
    }
}

fn leaf_plane_bounds(_p: &PlaneParams) -> Aabb {
    // Planes are infinite — use a large but finite bound.
    // Occupied below y=0, empty above.
    let extent = 1000.0;
    Aabb {
        min: Vec3::new(-extent, -extent, -extent),
        max: Vec3::new(extent, 0.0, extent),
    }
}

// ── AABB utilities ──────────────────────────────────────────────────────────

fn aabb_union(a: &Aabb, b: &Aabb) -> Aabb {
    if is_empty_aabb(a) {
        return *b;
    }
    if is_empty_aabb(b) {
        return *a;
    }
    Aabb {
        min: a.min.min(b.min),
        max: a.max.max(b.max),
    }
}

/// Transform an AABB by an affine transform, producing a new (potentially larger)
/// axis-aligned bounding box.
fn transform_aabb(aabb: &Aabb, transform: &glam::Affine3A) -> Aabb {
    // Standard technique: transform all 8 corners and take the min/max.
    let corners = [
        Vec3::new(aabb.min.x, aabb.min.y, aabb.min.z),
        Vec3::new(aabb.max.x, aabb.min.y, aabb.min.z),
        Vec3::new(aabb.min.x, aabb.max.y, aabb.min.z),
        Vec3::new(aabb.max.x, aabb.max.y, aabb.min.z),
        Vec3::new(aabb.min.x, aabb.min.y, aabb.max.z),
        Vec3::new(aabb.max.x, aabb.min.y, aabb.max.z),
        Vec3::new(aabb.min.x, aabb.max.y, aabb.max.z),
        Vec3::new(aabb.max.x, aabb.max.y, aabb.max.z),
    ];

    let first = transform.transform_point3(corners[0]);
    let mut min = first;
    let mut max = first;
    for &corner in &corners[1..] {
        let p = transform.transform_point3(corner);
        min = min.min(p);
        max = max.max(p);
    }

    Aabb { min, max }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::ProceduralObject;
    use glam::Affine3A;

    const EPS: f32 = 1e-3;

    #[test]
    fn sphere_bounds() {
        let mut obj = ProceduralObject::new(NodeKind::Union {
            material_combine: MaterialCombine::Winner,
        });
        obj.add_child(
            obj.root(),
            NodeKind::Sphere(SphereParams {
                radius: 1.0,
                ..Default::default()
            }),
        );

        let aabb = compute_bounds(&obj);
        assert!((aabb.min.x - (-1.0)).abs() < EPS);
        assert!((aabb.max.x - 1.0).abs() < EPS);
    }

    #[test]
    fn box_bounds() {
        let mut obj = ProceduralObject::new(NodeKind::Union {
            material_combine: MaterialCombine::Winner,
        });
        obj.add_child(
            obj.root(),
            NodeKind::Box(BoxParams {
                half_extents: Vec3::new(1.0, 2.0, 3.0),
                ..Default::default()
            }),
        );

        let aabb = compute_bounds(&obj);
        assert!((aabb.min.y - (-2.0)).abs() < EPS);
        assert!((aabb.max.z - 3.0).abs() < EPS);
    }

    #[test]
    fn translated_sphere_bounds() {
        let mut obj = ProceduralObject::new(NodeKind::Union {
            material_combine: MaterialCombine::Winner,
        });
        let s = obj.add_child(
            obj.root(),
            NodeKind::Sphere(SphereParams {
                radius: 0.5,
                ..Default::default()
            }),
        );
        obj.set_transform(s, Affine3A::from_translation(Vec3::new(5.0, 0.0, 0.0)));

        let aabb = compute_bounds(&obj);
        assert!(aabb.min.x > 4.0);
        assert!(aabb.max.x < 6.0);
    }

    #[test]
    fn union_of_two_spheres_bounds() {
        let mut obj = ProceduralObject::new(NodeKind::Union {
            material_combine: MaterialCombine::Winner,
        });
        let a = obj.add_child(
            obj.root(),
            NodeKind::Sphere(SphereParams {
                radius: 0.5,
                ..Default::default()
            }),
        );
        let b = obj.add_child(
            obj.root(),
            NodeKind::Sphere(SphereParams {
                radius: 0.5,
                ..Default::default()
            }),
        );
        obj.set_transform(a, Affine3A::from_translation(Vec3::new(-2.0, 0.0, 0.0)));
        obj.set_transform(b, Affine3A::from_translation(Vec3::new(2.0, 0.0, 0.0)));

        let aabb = compute_bounds(&obj);
        assert!(aabb.min.x < -2.0);
        assert!(aabb.max.x > 2.0);
    }

    #[test]
    fn empty_combinator_bounds() {
        let obj = ProceduralObject::new(NodeKind::Union {
            material_combine: MaterialCombine::Winner,
        });
        // No children — bounds should be empty/zero.
        let aabb = compute_bounds(&obj);
        assert_eq!(aabb.min, Vec3::ZERO);
        assert_eq!(aabb.max, Vec3::ZERO);
    }

    #[test]
    fn torus_bounds_are_flat() {
        let mut obj = ProceduralObject::new(NodeKind::Union {
            material_combine: MaterialCombine::Winner,
        });
        obj.add_child(
            obj.root(),
            NodeKind::Torus(TorusParams {
                major_radius: 1.0,
                minor_radius: 0.2,
                ..Default::default()
            }),
        );

        let aabb = compute_bounds(&obj);
        // Torus is flat — Y extent should be much smaller than X/Z.
        let y_extent = aabb.max.y - aabb.min.y;
        let x_extent = aabb.max.x - aabb.min.x;
        assert!(y_extent < x_extent * 0.5);
    }
}
