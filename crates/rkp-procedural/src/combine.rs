//! Boolean combination of [`Sample`] values — union, intersect, subtract.
//!
//! Each operation merges opacity and handles material transfer according to the
//! [`MaterialCombine`] mode.

use glam::Vec3;

use crate::node_kind::MaterialCombine;
use crate::sample::Sample;

/// Union: take the maximum opacity. Material is determined by [`MaterialCombine`] mode.
pub fn combine_union(a: &Sample, b: &Sample, mode: MaterialCombine) -> Sample {
    if a.is_empty() && b.is_empty() {
        return Sample::EMPTY;
    }
    let opacity = a.opacity.max(b.opacity);
    apply_material_combine(a, b, opacity, mode)
}

/// Intersect: take the minimum opacity. Material is determined by [`MaterialCombine`] mode.
pub fn combine_intersect(a: &Sample, b: &Sample, mode: MaterialCombine) -> Sample {
    let opacity = a.opacity.min(b.opacity);
    if opacity <= 0.0 {
        return Sample::EMPTY;
    }
    apply_material_combine(a, b, opacity, mode)
}

/// Subtract: `base` minus `cutter`. Always preserves base material (both primary
/// and secondary).
pub fn combine_subtract(base: &Sample, cutter: &Sample) -> Sample {
    let opacity = (base.opacity - cutter.opacity).max(0.0);
    if opacity <= 0.0 {
        return Sample::EMPTY;
    }
    Sample {
        opacity,
        material_id: base.material_id,
        secondary_material_id: base.secondary_material_id,
        blend_weight: base.blend_weight,
        color: base.color,
    }
}

/// Apply the material combine mode to determine the output material.
fn apply_material_combine(
    a: &Sample,
    b: &Sample,
    opacity: f32,
    mode: MaterialCombine,
) -> Sample {
    match mode {
        MaterialCombine::Winner => {
            // Higher opacity wins — takes all material and color.
            let winner = if a.opacity >= b.opacity { a } else { b };
            Sample {
                opacity,
                material_id: winner.material_id,
                secondary_material_id: winner.secondary_material_id,
                blend_weight: winner.blend_weight,
                color: winner.color,
            }
        }
        MaterialCombine::Layered => {
            // Winner's primary becomes output primary, loser's primary becomes
            // output secondary. Opacity ratio becomes blend weight. Existing
            // secondary materials on both sides are dropped (two-slot limit).
            let (winner, loser) = if a.opacity >= b.opacity {
                (a, b)
            } else {
                (b, a)
            };
            let total = a.opacity + b.opacity;
            let blend_weight = if total > 0.0 {
                loser.opacity / total
            } else {
                0.0
            };
            // Color: blend proportionally.
            let color = Vec3::lerp(winner.color, loser.color, blend_weight);
            Sample {
                opacity,
                material_id: winner.material_id,
                secondary_material_id: loser.material_id,
                blend_weight,
                color,
            }
        }
        MaterialCombine::Blend { radius } => {
            // Smooth blend within a radius of equal opacity. Outside the blend
            // zone, behaves like Winner.
            let diff = (a.opacity - b.opacity).abs();
            let radius = radius.max(1e-6);
            if diff >= radius {
                // Outside blend zone — winner takes all.
                let winner = if a.opacity >= b.opacity { a } else { b };
                Sample {
                    opacity,
                    material_id: winner.material_id,
                    secondary_material_id: winner.secondary_material_id,
                    blend_weight: winner.blend_weight,
                    color: winner.color,
                }
            } else {
                // Inside blend zone — interpolate.
                let t = 0.5 + 0.5 * (a.opacity - b.opacity) / radius;
                let color = Vec3::lerp(b.color, a.color, t);
                Sample {
                    opacity,
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
        Sample {
            opacity: 0.8,
            material_id: 1,
            secondary_material_id: 0,
            blend_weight: 0.0,
            color: Vec3::new(1.0, 0.0, 0.0),
        }
    }

    fn sample_b() -> Sample {
        Sample {
            opacity: 0.5,
            material_id: 2,
            secondary_material_id: 0,
            blend_weight: 0.0,
            color: Vec3::new(0.0, 0.0, 1.0),
        }
    }

    // ── Union ───────────────────────────────────────────────────────

    #[test]
    fn union_winner_takes_max_opacity() {
        let r = combine_union(&sample_a(), &sample_b(), MaterialCombine::Winner);
        assert!((r.opacity - 0.8).abs() < EPS);
        assert_eq!(r.material_id, 1); // a wins
    }

    #[test]
    fn union_layered_assigns_both_materials() {
        let r = combine_union(&sample_a(), &sample_b(), MaterialCombine::Layered);
        assert!((r.opacity - 0.8).abs() < EPS);
        assert_eq!(r.material_id, 1); // winner's primary
        assert_eq!(r.secondary_material_id, 2); // loser's primary
        assert!(r.blend_weight > 0.0 && r.blend_weight < 1.0);
    }

    #[test]
    fn union_of_two_empties_is_empty() {
        let r = combine_union(&Sample::EMPTY, &Sample::EMPTY, MaterialCombine::Winner);
        assert!(r.is_empty());
    }

    // ── Intersect ───────────────────────────────────────────────────

    #[test]
    fn intersect_takes_min_opacity() {
        let r = combine_intersect(&sample_a(), &sample_b(), MaterialCombine::Winner);
        assert!((r.opacity - 0.5).abs() < EPS);
    }

    #[test]
    fn intersect_with_empty_is_empty() {
        let r = combine_intersect(&sample_a(), &Sample::EMPTY, MaterialCombine::Winner);
        assert!(r.is_empty());
    }

    // ── Subtract ────────────────────────────────────────────────────

    #[test]
    fn subtract_reduces_opacity() {
        let r = combine_subtract(&sample_a(), &sample_b());
        assert!((r.opacity - 0.3).abs() < EPS);
    }

    #[test]
    fn subtract_preserves_base_material() {
        let r = combine_subtract(&sample_a(), &sample_b());
        assert_eq!(r.material_id, 1);
        assert_eq!(r.color, Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn subtract_clamps_to_zero() {
        let r = combine_subtract(&sample_b(), &sample_a());
        assert!(r.is_empty());
    }

    // ── Blend mode ──────────────────────────────────────────────────

    #[test]
    fn blend_far_from_boundary_acts_like_winner() {
        // Large opacity difference relative to radius.
        let r = combine_union(&sample_a(), &sample_b(), MaterialCombine::Blend { radius: 0.01 });
        assert_eq!(r.material_id, 1);
    }

    #[test]
    fn blend_at_equal_opacity_mixes() {
        let a = Sample::new(0.5, 1);
        let b = Sample::new(0.5, 2);
        let r = combine_union(&a, &b, MaterialCombine::Blend { radius: 1.0 });
        // At equal opacity, t should be ~0.5.
        assert!((r.blend_weight - 0.5).abs() < EPS);
    }
}
