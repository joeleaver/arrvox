//! Recursive tree evaluation — the core entry point for sampling a procedural
//! object at a world-space position.

use glam::Vec3;

use crate::arena::{NodeId, ProceduralObject};
use crate::bounds::AabbCache;
use crate::combine::{combine_intersect, combine_subtract, combine_union};
use crate::leaves::eval_leaf;
use crate::node_kind::NodeKind;
use crate::sample::Sample;

/// Sample the procedural object tree at a world-space position.
///
/// Evaluates the full tree from the root, transforming the position into each
/// node's local space and combining results according to combinator rules.
///
/// `voxel_size` is used by leaf shapes to size their fade band — matches the
/// voxelizer's grid resolution so the fade is exactly a few voxels wide.
///
/// Slow path — for hot usage inside the voxelizer, prefer
/// `sample_tree_cached` which accepts a precomputed per-node AABB cache
/// and uses it to prune Union children that provably can't contribute at
/// the given sample point.
pub fn sample_tree(obj: &ProceduralObject, pos: Vec3, voxel_size: f32) -> Sample {
    sample_node(obj, obj.root(), pos, voxel_size, None)
}

/// Culled sample path. `cache` must have been built by
/// `compute_all_bounds(obj)` from the same tree state. For Union nodes,
/// children whose world-space AABB distance exceeds the running-minimum
/// distance found so far are skipped — their SDF is provably ≥ that
/// distance (AABB-distance is a lower bound on any bounded primitive's
/// SDF), so they can't lower the min. Leaves and non-Union combinators
/// are evaluated as before; the first Union child is always evaluated
/// (no running min to compare against yet).
///
/// Correctness: skipping never changes the returned distance vs. the
/// uncached path, so the voxelizer's classifier sees identical SDF
/// values. Big speedup for large Unions where most children are
/// spatially distant from a given sample point.
pub fn sample_tree_cached(
    obj: &ProceduralObject,
    pos: Vec3,
    voxel_size: f32,
    cache: &AabbCache,
) -> Sample {
    sample_node(obj, obj.root(), pos, voxel_size, Some(cache))
}

/// Signed distance from a point to an AABB. Negative inside (nearest-face
/// penetration), positive outside (euclidean), zero on the boundary.
/// Matches the standard SDF semantics the procedural leaves use, so it
/// can be compared directly against any bounded primitive's SDF.
fn sdist_point_aabb(p: Vec3, aabb: &rkf_core::Aabb) -> f32 {
    let center = (aabb.min + aabb.max) * 0.5;
    let half = (aabb.max - aabb.min) * 0.5;
    let q = (p - center).abs() - half;
    let outside = q.max(Vec3::ZERO).length();
    let inside = q.x.max(q.y).max(q.z).min(0.0);
    outside + inside
}

/// Recursively evaluate a single node and its subtree.
///
/// `pos` is expressed in this node's *parent* frame — the same frame
/// the AABB cache entry for this node was stored in (cache entries
/// include each node's own transform but no ancestor transforms). For
/// the Union cull test we compare `local_pos` (this node's local frame
/// = its children's parent frame) against each child's cache entry —
/// both are in the same frame, no additional transform needed.
fn sample_node(
    obj: &ProceduralObject,
    id: NodeId,
    pos: Vec3,
    voxel_size: f32,
    cache: Option<&AabbCache>,
) -> Sample {
    let node = match obj.get(id) {
        Some(n) => n,
        None => return Sample::EMPTY,
    };

    let local_pos = node.transform.inverse().transform_point3(pos);

    match &node.kind {
        kind if kind.is_leaf() => eval_leaf(local_pos, kind, voxel_size),

        NodeKind::Union { material_combine } => {
            let mode = *material_combine;
            let mut result = Sample::EMPTY;
            // Running minimum signed distance across already-evaluated
            // children. Starts at +∞; the first child's AABB-cull test
            // can't fire because every `sdist_point_aabb` value is ≤ +∞.
            // After one child produces a finite sample, subsequent
            // children with AABB distance ≥ running_min are skipped —
            // their true SDF is ≥ AABB distance, so they can't lower
            // the min.
            let mut running_min = f32::INFINITY;
            for &child_id in &node.children {
                if let Some(c) = cache {
                    if running_min.is_finite() {
                        if let Some(Some(child_aabb)) = c.get(child_id.0 as usize) {
                            let d_lo = sdist_point_aabb(local_pos, child_aabb);
                            if d_lo >= running_min {
                                continue;
                            }
                        }
                    }
                }
                let child_sample = sample_node(obj, child_id, local_pos, voxel_size, cache);
                if child_sample.is_empty() {
                    continue;
                }
                if child_sample.distance < running_min {
                    running_min = child_sample.distance;
                }
                if result.is_empty() {
                    result = child_sample;
                } else {
                    result = combine_union(&result, &child_sample, mode);
                }
            }
            result
        }

        NodeKind::Intersect { material_combine } => {
            let mode = *material_combine;
            let children = &node.children;
            if children.is_empty() {
                return Sample::EMPTY;
            }
            let mut result = sample_node(obj, children[0], local_pos, voxel_size, cache);
            for &child_id in &children[1..] {
                if result.is_empty() {
                    return Sample::EMPTY;
                }
                let child_sample = sample_node(obj, child_id, local_pos, voxel_size, cache);
                result = combine_intersect(&result, &child_sample, mode);
            }
            result
        }

        NodeKind::Subtract => {
            let children = &node.children;
            if children.is_empty() {
                return Sample::EMPTY;
            }
            let mut result = sample_node(obj, children[0], local_pos, voxel_size, cache);
            for &cutter_id in &children[1..] {
                if result.is_empty() {
                    return Sample::EMPTY;
                }
                let cutter_sample = sample_node(obj, cutter_id, local_pos, voxel_size, cache);
                result = combine_subtract(&result, &cutter_sample);
            }
            result
        }

        NodeKind::Mirror(params) => {
            // Single-child effect. Position fold along the chosen axis
            // reflects the child's +axis-side geometry onto the -axis
            // side. The fold is length-preserving (1-Lipschitz) — a
            // single reflection is a pure isometry — so the child's
            // distance passes through unchanged. No conservative shrink
            // like NoiseDisplace needs.
            let children = &node.children;
            let Some(&child_id) = children.first() else {
                return Sample::EMPTY;
            };
            let folded = crate::node_kind::mirror_fold(local_pos, params.axis);
            sample_node(obj, child_id, folded, voxel_size, cache)
        }

        NodeKind::NoiseDisplace(params) => {
            // Single-child effect. Additional children are ignored —
            // the add-child UI treats this as a combinator so users can
            // attach a subtree, but the semantic is "warp the position
            // and evaluate the first child at that warped location."
            let children = &node.children;
            let Some(&child_id) = children.first() else {
                return Sample::EMPTY;
            };
            let warp = crate::noise::fbm_3d_vec(
                local_pos,
                params.frequency,
                params.seed,
                params.octaves,
            ) * params.amplitude;
            let child_sample = sample_node(
                obj, child_id, local_pos + warp, voxel_size, cache,
            );
            // The child's SDF is evaluated at a point up to `amplitude`
            // away (in each axis) from where the caller asked. That
            // breaks 1-Lipschitz: a surface that the displaced sample
            // places `d` units away could actually be only `d -
            // amplitude` units from the original point. Report the
            // conservative lower bound so the voxelizer's classifier
            // and any sphere-tracer step on top stay safe.
            //
            // Material / color / opacity pass through unchanged — they
            // come from the child's sample at the warped position.
            let max_axis_warp = params.amplitude * (3.0f32).sqrt();
            Sample {
                distance: child_sample.distance - max_axis_warp,
                ..child_sample
            }
        }

        // Never hit — is_leaf() path above captures every leaf kind,
        // and all non-leaf variants have explicit arms. Left as a
        // belt-and-braces fallback so a future new NodeKind added
        // without an evaluate arm fails safe (empty) rather than
        // compiling to UB via unreachable!(). Intentionally inclusive
        // of any Sample::EMPTY-valued "no-op" case.
        _ => Sample::EMPTY,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node_kind::*;
    use glam::Affine3A;

    const EPS: f32 = 1e-3;
    const VS: f32 = 0.02;

    fn sphere(radius: f32, material_id: u16) -> NodeKind {
        NodeKind::Sphere(SphereParams {
            radius,
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
        let s = sample_tree(&obj, Vec3::ZERO, VS);
        assert!(s.is_inside(), "expected inside, dist={}", s.distance);

        // Far away should be empty.
        let s = sample_tree(&obj, Vec3::new(10.0, 0.0, 0.0), VS);
        assert!(!s.is_inside(), "expected outside, dist={}", s.distance);
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
        let s = sample_tree(&obj, Vec3::ZERO, VS);
        assert!(s.is_inside(), "expected inside, dist={}", s.distance);

        // Deep inside A should have material 1.
        let s = sample_tree(&obj, Vec3::new(-0.3, 0.0, 0.0), VS);
        assert!(s.is_inside(), "expected inside, dist={}", s.distance);
        assert_eq!(s.material_id, 1);

        // Deep inside B should have material 2.
        let s = sample_tree(&obj, Vec3::new(0.3, 0.0, 0.0), VS);
        assert!(s.is_inside(), "expected inside, dist={}", s.distance);
        assert_eq!(s.material_id, 2);
    }

    /// Subtract: sphere with a hole cut out.
    #[test]
    fn subtract_cuts_hole() {
        let mut obj = ProceduralObject::new(NodeKind::Subtract);
        obj.add_child(obj.root(), sphere(1.0, 1)); // base
        obj.add_child(obj.root(), sphere(0.5, 2)); // cutter

        // Center should be empty (cutter removes it).
        let s = sample_tree(&obj, Vec3::ZERO, VS);
        assert!(!s.is_inside(), "expected outside, dist={}", s.distance);

        // Edge of base (outside cutter) should still be opaque.
        let s = sample_tree(&obj, Vec3::new(0.8, 0.0, 0.0), VS);
        assert!(s.is_inside(), "expected inside, dist={}", s.distance);
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
        let s = sample_tree(&obj, Vec3::ZERO, VS);
        assert!(s.is_inside(), "expected inside, dist={}", s.distance);

        // Far inside A but outside B should be empty.
        let s = sample_tree(&obj, Vec3::new(-0.4, 0.0, 0.0), VS);
        assert!(!s.is_inside(), "expected outside, dist={}", s.distance);
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
        let s = sample_tree(&obj, Vec3::new(-2.0, 0.0, 0.0), VS);
        assert!(s.is_inside(), "expected inside, dist={}", s.distance);
        assert_eq!(s.material_id, 1);

        // Right subtract: center should be empty (cut away).
        let s = sample_tree(&obj, Vec3::new(2.0, 0.0, 0.0), VS);
        assert!(!s.is_inside(), "expected outside, dist={}", s.distance);

        // Right subtract: shell region should be opaque.
        let s = sample_tree(&obj, Vec3::new(2.4, 0.0, 0.0), VS);
        assert!(s.is_inside(), "expected inside, dist={}", s.distance);
        assert_eq!(s.material_id, 2);
    }

    /// The cached sample path must produce byte-identical SDF values to
    /// the uncached path for the same tree. Any divergence is a cull bug.
    #[test]
    fn cached_matches_uncached_dense_grid() {
        use crate::bounds::compute_all_bounds;
        use crate::node_kind::*;

        // Tree: Union of several spheres and boxes at scattered positions
        // + a Subtract node, so every combinator type runs.
        let mut obj = ProceduralObject::new(NodeKind::Union {
            material_combine: MaterialCombine::Winner,
        });
        for i in 0..5 {
            let s = obj.add_child(obj.root(), sphere(0.4 + 0.05 * i as f32, i as u16));
            obj.set_transform(
                s,
                Affine3A::from_translation(Vec3::new(i as f32 * 1.5 - 3.0, 0.0, 0.0)),
            );
        }
        let sub = obj.add_child(obj.root(), NodeKind::Subtract);
        obj.set_transform(sub, Affine3A::from_translation(Vec3::new(0.0, 1.5, 0.0)));
        obj.add_child(sub, sphere(0.8, 10));
        obj.add_child(sub, sphere(0.3, 11));

        let cache = compute_all_bounds(&obj);

        // Sweep a 3D grid across the whole scene including plenty of empty
        // space — cull decisions at "far from everything" points are
        // where frame-mismatch bugs would show up.
        let mut disagree = 0usize;
        for ix in -20..=20 {
            for iy in -10..=10 {
                for iz in -10..=10 {
                    let p = Vec3::new(
                        ix as f32 * 0.25,
                        iy as f32 * 0.25,
                        iz as f32 * 0.25,
                    );
                    let s_uncached = sample_tree(&obj, p, VS);
                    let s_cached = sample_tree_cached(&obj, p, VS, &cache);
                    let diff = (s_uncached.distance - s_cached.distance).abs();
                    if diff > 1e-5 {
                        disagree += 1;
                        if disagree <= 5 {
                            eprintln!(
                                "cull mismatch at {p:?}: uncached={} cached={}",
                                s_uncached.distance, s_cached.distance,
                            );
                        }
                    }
                }
            }
        }
        assert_eq!(disagree, 0, "cached path diverged at {disagree} grid points");
    }

    /// NoiseDisplace wrapping a sphere: the center must still read
    /// inside (warp can't teleport the surface all the way to the
    /// origin if amplitude is less than the radius), and a point well
    /// beyond radius + amplitude must still read outside.
    #[test]
    fn noise_displace_preserves_center_and_outside() {
        use crate::node_kind::NoiseDisplaceParams;
        let mut obj = ProceduralObject::new(NodeKind::Union {
            material_combine: MaterialCombine::Winner,
        });
        let nd = obj.add_child(
            obj.root(),
            NodeKind::NoiseDisplace(NoiseDisplaceParams {
                amplitude: 0.15,
                frequency: 3.0,
                octaves: 3,
                seed: 7,
                ..Default::default()
            }),
        );
        obj.add_child(nd, sphere(0.5, 1));

        let s_center = sample_tree(&obj, Vec3::ZERO, VS);
        assert!(s_center.is_inside(), "center not inside, dist={}", s_center.distance);

        // Well past the sphere's radius + amplitude: must be empty.
        let s_far = sample_tree(&obj, Vec3::new(2.0, 0.0, 0.0), VS);
        assert!(!s_far.is_inside(), "far point not outside, dist={}", s_far.distance);
    }

    /// Amplitude = 0 collapses to the undisplaced child — distances
    /// should match a bare sphere to within f32 noise.
    #[test]
    fn noise_displace_zero_amplitude_is_identity() {
        use crate::node_kind::NoiseDisplaceParams;
        let mut displaced = ProceduralObject::new(NodeKind::Union {
            material_combine: MaterialCombine::Winner,
        });
        let nd = displaced.add_child(
            displaced.root(),
            NodeKind::NoiseDisplace(NoiseDisplaceParams {
                amplitude: 0.0,
                frequency: 4.0,
                octaves: 4,
                seed: 13,
            }),
        );
        displaced.add_child(nd, sphere(0.5, 0));

        let mut plain = ProceduralObject::new(NodeKind::Union {
            material_combine: MaterialCombine::Winner,
        });
        plain.add_child(plain.root(), sphere(0.5, 0));

        for i in -5..=5 {
            let p = Vec3::new(i as f32 * 0.13, i as f32 * -0.09, i as f32 * 0.21);
            let d = sample_tree(&displaced, p, VS).distance;
            let r = sample_tree(&plain, p, VS).distance;
            assert!((d - r).abs() < 1e-5, "mismatch at {p:?}: {d} vs {r}");
        }
    }

    /// Mirror reflects child geometry across the configured plane:
    /// a sphere sitting on the +X side should register inside at the
    /// mirrored -X point too. Also checks distance: the fold is an
    /// isometry, so the child sample's distance reads the same at
    /// symmetric points.
    #[test]
    fn mirror_reflects_sphere_across_x() {
        use crate::node_kind::{MirrorAxis, MirrorParams};
        let mut obj = ProceduralObject::new(NodeKind::Union {
            material_combine: MaterialCombine::Winner,
        });
        let mir = obj.add_child(
            obj.root(),
            NodeKind::Mirror(MirrorParams { axis: MirrorAxis::X }),
        );
        // Sphere placed at +x — the -x side is purely the reflection.
        let s = obj.add_child(mir, sphere(0.5, 1));
        obj.set_transform(s, Affine3A::from_translation(Vec3::new(2.0, 0.0, 0.0)));

        let s_pos = sample_tree(&obj, Vec3::new(2.0, 0.0, 0.0), VS);
        let s_neg = sample_tree(&obj, Vec3::new(-2.0, 0.0, 0.0), VS);
        assert!(s_pos.is_inside(), "+x center not inside: {}", s_pos.distance);
        assert!(s_neg.is_inside(), "-x mirror center not inside: {}", s_neg.distance);
        // Length-preserving fold: distances must be bit-identical.
        assert!(
            (s_pos.distance - s_neg.distance).abs() < 1e-5,
            "mirrored distances diverge: {} vs {}", s_pos.distance, s_neg.distance,
        );
    }

    /// Empty Mirror (no child) must not inject phantom geometry.
    #[test]
    fn mirror_without_child_is_empty() {
        use crate::node_kind::{MirrorAxis, MirrorParams};
        let mut obj = ProceduralObject::new(NodeKind::Union {
            material_combine: MaterialCombine::Winner,
        });
        obj.add_child(
            obj.root(),
            NodeKind::Mirror(MirrorParams { axis: MirrorAxis::Y }),
        );
        let s = sample_tree(&obj, Vec3::ZERO, VS);
        assert!(!s.is_inside(), "unexpected inside at origin, dist={}", s.distance);
    }

    /// Moving the Mirror node via its transform shifts the mirror
    /// plane in world space. Mirror at x=2 with a sphere at local
    /// x=1 (world x=3) should reflect to world x=1, not x=-3.
    #[test]
    fn mirror_transform_shifts_plane() {
        use crate::node_kind::{MirrorAxis, MirrorParams};
        let mut obj = ProceduralObject::new(NodeKind::Union {
            material_combine: MaterialCombine::Winner,
        });
        let mir = obj.add_child(
            obj.root(),
            NodeKind::Mirror(MirrorParams { axis: MirrorAxis::X }),
        );
        // Mirror plane in world at x=2.
        obj.set_transform(mir, Affine3A::from_translation(Vec3::new(2.0, 0.0, 0.0)));
        // Sphere at mirror-local x=1 → world x=3.
        let s = obj.add_child(mir, sphere(0.25, 1));
        obj.set_transform(s, Affine3A::from_translation(Vec3::new(1.0, 0.0, 0.0)));

        // Original side.
        let s_orig = sample_tree(&obj, Vec3::new(3.0, 0.0, 0.0), VS);
        assert!(s_orig.is_inside(), "original not inside: {}", s_orig.distance);

        // Reflection: plane at x=2, source at x=3, mirror to x=1.
        let s_ref = sample_tree(&obj, Vec3::new(1.0, 0.0, 0.0), VS);
        assert!(s_ref.is_inside(), "reflection not inside: {}", s_ref.distance);

        // The naive mirror-across-origin reflection at x=-3 must NOT
        // be inside — the plane moved with the node transform.
        let s_wrong = sample_tree(&obj, Vec3::new(-3.0, 0.0, 0.0), VS);
        assert!(!s_wrong.is_inside(), "wrong side flagged inside: {}", s_wrong.distance);
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
        let s = sample_tree(&obj, Vec3::new(5.0, 0.0, 0.0), VS);
        assert!(s.is_inside(), "expected inside, dist={}", s.distance);

        // Origin should be empty.
        let s = sample_tree(&obj, Vec3::ZERO, VS);
        assert!(!s.is_inside(), "expected outside, dist={}", s.distance);
    }
}
