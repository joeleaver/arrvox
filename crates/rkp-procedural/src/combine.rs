//! Boolean combination of [`Sample`] values with SDF algebra.
//!
//! * **Union:** `min(a, b)` — take the closer surface (deeper inside wins).
//! * **Intersect:** `max(a, b)` — take the farther surface.
//! * **Subtract:** `max(a, -b)` — flip the cutter's sign and intersect.
//!
//! Material transfer is governed by the [`MaterialCombine`] mode; the
//! "winner" for material purposes is the side with the smaller (more
//! inside) signed distance.

use glam::Vec3;

use crate::node_kind::MaterialCombine;
use crate::sample::Sample;

/// Union: `min(a, b)`. Material chosen by [`MaterialCombine`] mode.
pub fn combine_union(a: &Sample, b: &Sample, mode: MaterialCombine) -> Sample {
    if a.is_empty() && b.is_empty() {
        return Sample::EMPTY;
    }
    let distance = a.distance.min(b.distance);
    apply_material_combine(a, b, distance, mode)
}

/// Intersect: `max(a, b)`. Result is empty if the two bodies don't overlap
/// (both distances positive → result's distance > 0, still a valid SDF).
pub fn combine_intersect(a: &Sample, b: &Sample, mode: MaterialCombine) -> Sample {
    let distance = a.distance.max(b.distance);
    apply_material_combine(a, b, distance, mode)
}

/// Subtract: `base` minus `cutter` = `max(base, -cutter)`. Preserves base
/// material (both primary and secondary).
pub fn combine_subtract(base: &Sample, cutter: &Sample) -> Sample {
    let distance = base.distance.max(-cutter.distance);
    Sample {
        distance,
        material_id: base.material_id,
        secondary_material_id: base.secondary_material_id,
        blend_weight: base.blend_weight,
        color: base.color,
    }
}

/// Apply the material combine mode to determine the output material. The
/// "winner" is the side with the smaller (more inside) distance — this is
/// the side whose surface is active at this query point.
fn apply_material_combine(
    a: &Sample,
    b: &Sample,
    distance: f32,
    mode: MaterialCombine,
) -> Sample {
    match mode {
        MaterialCombine::Winner => {
            let winner = if a.distance <= b.distance { a } else { b };
            Sample {
                distance,
                material_id: winner.material_id,
                secondary_material_id: winner.secondary_material_id,
                blend_weight: winner.blend_weight,
                color: winner.color,
            }
        }
        MaterialCombine::Layered => {
            // Winner's primary becomes output primary, loser's primary
            // becomes output secondary. Blend weight is driven by how close
            // the two surfaces are at this point: closer → stronger blend.
            let (winner, loser) = if a.distance <= b.distance { (a, b) } else { (b, a) };
            // Normalize distance difference into [0, 1] over a sensible
            // blend region. The old opacity-based formula used
            // `loser / (a + b)`; in SDF-space we use a soft falloff based
            // on the separation distance relative to the winner's depth.
            let separation = (loser.distance - winner.distance).max(0.0);
            let scale = (-winner.distance).max(1e-3);
            let blend_weight = (1.0 - (separation / scale)).clamp(0.0, 1.0);
            let color = Vec3::lerp(winner.color, loser.color, blend_weight);
            Sample {
                distance,
                material_id: winner.material_id,
                secondary_material_id: loser.material_id,
                blend_weight,
                color,
            }
        }
        MaterialCombine::Blend { radius } => {
            // Smooth blend within a radius where both surfaces are equally
            // close. Outside the blend zone, behaves like Winner.
            let diff = (a.distance - b.distance).abs();
            let radius = radius.max(1e-6);
            if diff >= radius {
                let winner = if a.distance <= b.distance { a } else { b };
                Sample {
                    distance,
                    material_id: winner.material_id,
                    secondary_material_id: winner.secondary_material_id,
                    blend_weight: winner.blend_weight,
                    color: winner.color,
                }
            } else {
                // Inside blend zone — interpolate. t=0 → fully b, t=1 → fully a.
                let t = 0.5 + 0.5 * (b.distance - a.distance) / radius;
                let color = Vec3::lerp(b.color, a.color, t);
                Sample {
                    distance,
                    material_id: a.material_id,
                    secondary_material_id: b.material_id,
                    blend_weight: 1.0 - t,
                    color,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    fn sample_a() -> Sample {
        // Deeper inside (more negative distance).
        Sample {
            distance: -0.8,
            material_id: 1,
            secondary_material_id: 0,
            blend_weight: 0.0,
            color: Vec3::new(1.0, 0.0, 0.0),
        }
    }

    fn sample_b() -> Sample {
        Sample {
            distance: -0.3,
            material_id: 2,
            secondary_material_id: 0,
            blend_weight: 0.0,
            color: Vec3::new(0.0, 0.0, 1.0),
        }
    }

    // ── Union ───────────────────────────────────────────────────────

    #[test]
    fn union_takes_min_distance() {
        let r = combine_union(&sample_a(), &sample_b(), MaterialCombine::Winner);
        assert!((r.distance - (-0.8)).abs() < EPS);
        assert_eq!(r.material_id, 1); // deeper inside wins
    }

    #[test]
    fn union_of_two_empties_is_empty() {
        let r = combine_union(&Sample::EMPTY, &Sample::EMPTY, MaterialCombine::Winner);
        assert!(r.is_empty());
    }

    // ── Intersect ───────────────────────────────────────────────────

    #[test]
    fn intersect_takes_max_distance() {
        let r = combine_intersect(&sample_a(), &sample_b(), MaterialCombine::Winner);
        assert!((r.distance - (-0.3)).abs() < EPS);
        // Winner's material is whichever is smaller — still a (a.distance < b.distance).
        assert_eq!(r.material_id, 1);
    }

    #[test]
    fn intersect_outside_is_outside() {
        let a = Sample::new(0.5, 1);
        let b = Sample::new(0.3, 2);
        let r = combine_intersect(&a, &b, MaterialCombine::Winner);
        assert!(!r.is_inside());
    }

    // ── Subtract ────────────────────────────────────────────────────

    #[test]
    fn subtract_carves_away_cutter() {
        // base is inside (-0.8), cutter is inside (-0.3). Subtract:
        // max(-0.8, 0.3) = 0.3 → now outside.
        let r = combine_subtract(&sample_a(), &sample_b());
        assert!((r.distance - 0.3).abs() < EPS);
        assert!(!r.is_inside());
    }

    #[test]
    fn subtract_preserves_base_material() {
        let r = combine_subtract(&sample_a(), &sample_b());
        assert_eq!(r.material_id, 1);
        assert_eq!(r.color, Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn subtract_deep_base_unaffected_by_outside_cutter() {
        let base = Sample::new(-1.0, 1);
        let cutter = Sample::new(2.0, 2); // cutter is outside
        let r = combine_subtract(&base, &cutter);
        // max(-1.0, -2.0) = -1.0 — base unchanged.
        assert!((r.distance - (-1.0)).abs() < EPS);
        assert!(r.is_inside());
    }

    // ── Blend mode ──────────────────────────────────────────────────

    #[test]
    fn blend_far_from_boundary_acts_like_winner() {
        // Large distance difference vs. small radius.
        let r = combine_union(&sample_a(), &sample_b(), MaterialCombine::Blend { radius: 0.01 });
        assert_eq!(r.material_id, 1);
    }

    #[test]
    fn blend_at_equal_distance_mixes() {
        let a = Sample::new(-0.5, 1);
        let b = Sample::new(-0.5, 2);
        let r = combine_union(&a, &b, MaterialCombine::Blend { radius: 1.0 });
        // At equal distance, t = 0.5, blend_weight = 0.5.
        assert!((r.blend_weight - 0.5).abs() < EPS);
    }
}
