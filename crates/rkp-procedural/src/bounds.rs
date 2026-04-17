//! Per-node bounding box computation.
//!
//! Every node can compute an axis-aligned bounding box in its local space.
//! Parent transforms widen the bounds. Combinators merge children's bounds.

use glam::Vec3;
use rkf_core::Aabb;

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

/// World-space AABB per node, indexed by `NodeId.0 as usize`. `None`
/// entries correspond to tombstoned ids, empty combinators, or the
/// unbounded-but-in-practice-large Plane case. Built once per bake by
/// `compute_all_bounds` and consumed by the culled sample path
/// (`sample_tree_cached`) to skip Union children whose AABB distance
/// provably exceeds the running min at a sample point.
pub type AabbCache = Vec<Option<Aabb>>;

/// Compute a world-space AABB for every node in the tree in a single
/// bottom-up pass. Each entry is the same AABB `compute_bounds` would
/// produce for that subtree considered as a whole — including the
/// transforms from the node down to its leaves, but *not* the
/// transforms of any ancestors above it. That matches the position
/// frame `sample_tree_cached` uses for the AABB test (see evaluate.rs).
pub fn compute_all_bounds(obj: &ProceduralObject) -> AabbCache {
    let mut cache: AabbCache = vec![None; obj.arena_len()];
    compute_all_bounds_rec(obj, obj.root(), &mut cache);
    cache
}

fn compute_all_bounds_rec(
    obj: &ProceduralObject,
    id: NodeId,
    cache: &mut AabbCache,
) -> Aabb {
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

        NodeKind::Union { .. } => {
            let mut combined = EMPTY_AABB;
            for &child_id in &node.children {
                let child_aabb = compute_all_bounds_rec(obj, child_id, cache);
                combined = aabb_union(&combined, &child_aabb);
            }
            combined
        }
        NodeKind::Intersect { .. } => {
            // Intersect's result ⊆ the intersection of its operands,
            // so use the axis-aligned intersection. Without this, a
            // Plane inside an Intersect (its AABB spans ±1000m by
            // default) would balloon the tree bounds, push the
            // voxelizer to a ~1 m voxel size to fit the 2048-voxel
            // cap, and subsample the actual geometry to nothing.
            let mut iter = node.children.iter();
            let mut combined = match iter.next() {
                Some(&first) => compute_all_bounds_rec(obj, first, cache),
                None => EMPTY_AABB,
            };
            for &child_id in iter {
                let child_aabb = compute_all_bounds_rec(obj, child_id, cache);
                combined = aabb_intersection(&combined, &child_aabb);
            }
            // Keep computing (for cache side-effects) over remaining
            // children even after combined goes empty — above loop
            // handles that naturally: `aabb_intersection(empty, x) =
            // empty` and we still recurse into `x` above.
            combined
        }
        NodeKind::Subtract => {
            // Subtract's result ⊆ the minuend (first child). Cutters
            // can only carve; their AABBs don't expand the bound.
            // Still recurse into every child so the per-node cache
            // is populated — just don't union the cutters in.
            let mut first_aabb = EMPTY_AABB;
            for (i, &child_id) in node.children.iter().enumerate() {
                let child_aabb = compute_all_bounds_rec(obj, child_id, cache);
                if i == 0 {
                    first_aabb = child_aabb;
                }
            }
            first_aabb
        }
        NodeKind::NoiseDisplace(p) => {
            // Effect widens the child's AABB by its max per-axis
            // displacement. Additional children are ignored (same
            // rule as the evaluator). If the first child is missing
            // the effect contributes nothing — EMPTY_AABB.
            let Some(&child_id) = node.children.first() else {
                return EMPTY_AABB;
            };
            let child_aabb = compute_all_bounds_rec(obj, child_id, cache);
            // Keep iterating the ignored siblings to populate their
            // cache entries (they could still be UI-visible / selected
            // even though they don't contribute to geometry).
            for &sibling_id in node.children.iter().skip(1) {
                let _ = compute_all_bounds_rec(obj, sibling_id, cache);
            }
            if is_empty_aabb(&child_aabb) {
                EMPTY_AABB
            } else {
                Aabb {
                    min: child_aabb.min - Vec3::splat(p.amplitude),
                    max: child_aabb.max + Vec3::splat(p.amplitude),
                }
            }
        }
    };

    let world_aabb = if is_empty_aabb(&local_aabb) {
        EMPTY_AABB
    } else {
        transform_aabb(&local_aabb, &node.transform)
    };

    let slot = id.0 as usize;
    if slot < cache.len() {
        cache[slot] = if is_empty_aabb(&world_aabb) { None } else { Some(world_aabb) };
    }
    world_aabb
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

        NodeKind::Union { .. } => {
            let mut combined = EMPTY_AABB;
            for &child_id in &node.children {
                let child_aabb = compute_node_bounds(obj, child_id);
                combined = aabb_union(&combined, &child_aabb);
            }
            combined
        }
        NodeKind::Intersect { .. } => {
            // Axis-aligned intersection of operand AABBs — tighter
            // than a union, and the key fix for Plane-in-Intersect
            // trees (see compute_all_bounds_rec for the full why).
            let mut iter = node.children.iter();
            let mut combined = match iter.next() {
                Some(&first) => compute_node_bounds(obj, first),
                None => EMPTY_AABB,
            };
            for &child_id in iter {
                let child_aabb = compute_node_bounds(obj, child_id);
                combined = aabb_intersection(&combined, &child_aabb);
            }
            combined
        }
        NodeKind::Subtract => {
            // Minuend bounds only; cutters carve but don't expand.
            node.children
                .first()
                .map(|&first| compute_node_bounds(obj, first))
                .unwrap_or(EMPTY_AABB)
        }
        NodeKind::NoiseDisplace(p) => {
            // Widen the child's AABB by the max per-axis displacement
            // (mirror of compute_all_bounds_rec's NoiseDisplace arm).
            let child_aabb = node
                .children
                .first()
                .map(|&first| compute_node_bounds(obj, first))
                .unwrap_or(EMPTY_AABB);
            if is_empty_aabb(&child_aabb) {
                EMPTY_AABB
            } else {
                Aabb {
                    min: child_aabb.min - Vec3::splat(p.amplitude),
                    max: child_aabb.max + Vec3::splat(p.amplitude),
                }
            }
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

/// Axis-aligned intersection of two AABBs. Returns `EMPTY_AABB` if
/// they don't overlap (min > max on any axis). An empty input flows
/// through as empty — intersecting anything with "nothing" gives
/// nothing.
fn aabb_intersection(a: &Aabb, b: &Aabb) -> Aabb {
    if is_empty_aabb(a) || is_empty_aabb(b) {
        return EMPTY_AABB;
    }
    let min = a.min.max(b.min);
    let max = a.max.min(b.max);
    if min.x > max.x || min.y > max.y || min.z > max.z {
        return EMPTY_AABB;
    }
    Aabb { min, max }
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

    /// Intersect with a Plane child should produce bounds no larger
    /// than the tightest non-Plane operand — Plane's raw AABB spans
    /// ±1000 m but the intersection can't extend past whatever bounds
    /// the other children impose. This is the fix for a user-visible
    /// bug where baking an Intersect-with-Plane tree produced nothing
    /// because the ballooned AABB drove voxel size up past the
    /// primitives' actual scale.
    #[test]
    fn intersect_with_plane_uses_other_operand_bounds() {
        let mut obj = ProceduralObject::new(NodeKind::Intersect {
            material_combine: MaterialCombine::Winner,
        });
        obj.add_child(
            obj.root(),
            NodeKind::Sphere(SphereParams { radius: 1.0, ..Default::default() }),
        );
        obj.add_child(obj.root(), NodeKind::Plane(PlaneParams::default()));

        let aabb = compute_bounds(&obj);

        // Max extent along any axis should be bounded by the sphere —
        // not the plane's 2000-m span. Give 10 m of slack for any
        // margining the shader/voxelizer might layer on top later.
        let extent = aabb.max - aabb.min;
        let max_axis = extent.x.max(extent.y).max(extent.z);
        assert!(
            max_axis < 10.0,
            "Intersect-with-Plane AABB ballooned: extent={extent:?}",
        );
    }

    /// Subtract shouldn't expand beyond the minuend — even if a
    /// cutter's unbounded-ish AABB (Plane) would union past it.
    #[test]
    fn subtract_with_plane_cutter_stays_bounded() {
        let mut obj = ProceduralObject::new(NodeKind::Subtract);
        obj.add_child(
            obj.root(),
            NodeKind::Sphere(SphereParams { radius: 1.0, ..Default::default() }),
        );
        obj.add_child(obj.root(), NodeKind::Plane(PlaneParams::default()));

        let aabb = compute_bounds(&obj);
        let extent = aabb.max - aabb.min;
        let max_axis = extent.x.max(extent.y).max(extent.z);
        assert!(max_axis < 10.0, "Subtract-with-Plane ballooned: extent={extent:?}");
    }

    /// NoiseDisplace widens its child's AABB by `amplitude` on every
    /// axis — the voxelizer needs that slack so the classifier doesn't
    /// clip surface perturbations that escape the tight bounds.
    #[test]
    fn noise_displace_widens_child_aabb() {
        use crate::node_kind::NoiseDisplaceParams;
        let mut obj = ProceduralObject::new(NodeKind::Union {
            material_combine: MaterialCombine::Winner,
        });
        let nd = obj.add_child(
            obj.root(),
            NodeKind::NoiseDisplace(NoiseDisplaceParams {
                amplitude: 0.25,
                frequency: 2.0,
                octaves: 2,
                seed: 0,
            }),
        );
        obj.add_child(
            nd,
            NodeKind::Sphere(SphereParams { radius: 1.0, ..Default::default() }),
        );

        let aabb = compute_bounds(&obj);
        assert!((aabb.min.x - (-1.25)).abs() < EPS, "min.x = {}", aabb.min.x);
        assert!((aabb.max.x - 1.25).abs() < EPS, "max.x = {}", aabb.max.x);
        assert!((aabb.min.y - (-1.25)).abs() < EPS, "min.y = {}", aabb.min.y);
        assert!((aabb.max.y - 1.25).abs() < EPS, "max.y = {}", aabb.max.y);
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
