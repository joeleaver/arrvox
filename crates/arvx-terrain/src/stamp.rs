//! Stamps — Layer 2 of the three-layer terrain source model.
//!
//! A `Stamp` is a heightmap feature placed in world space. It composes
//! over the base `TerrainFn` (Layer 1) and underneath sculpt edits
//! (Layer 3). V1 ships five **heightmap** kinds — Mountain, Hill,
//! Lake, Plateau, Flatten — each defined by an absolute target height
//! `target_h(world_xz)` over a 2D footprint. The combine op decides
//! how the stamp's heightmap merges with the base.
//!
//! Volumetric stamps (caves, overhangs) are reserved for V2; the
//! `StampKind` enum will gain new variants without an API break.
//!
//! ### Composition pipeline
//!
//! For each voxel sample, the `bake_tile` path does:
//!
//! 1. Evaluate Layer 1 `TerrainFn` → `base_sample` with `base_h`.
//! 2. Query the `StampIndex` for stamps overlapping `(wx, wz)`.
//! 3. For each stamp in deterministic order, evaluate
//!    `stamp.sample_height(wx, wz)`; if `Some(target_h)`, fold via
//!    `combine(base_h, target_h, op)`.
//! 4. Final `sd = wy - H` where `H` is the composed height.
//!
//! Materials follow a similar overlay: a stamp can carry an optional
//! `material_override` consumed inside its footprint.

use arvx_core::Aabb;
use glam::{Vec2, Vec3};

/// How the stamp's heightmap falls off at the rim of its footprint.
///
/// `apply(t)` returns 1.0 at `t = 0` (centre of the footprint) and
/// 0.0 at `t = 1` (rim). Outside the rim the stamp doesn't apply at
/// all — the curve only shapes the transition inside the footprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FalloffCurve {
    /// `1 - smoothstep(0, 1, t)` — zero slope at both ends. Default.
    Smoothstep,
    /// `1 - t` — constant slope.
    Linear,
    /// `1` for `t < 1`, `0` at the rim. No transition; useful for
    /// debugging and the rare authoring case that wants a hard rim.
    Hard,
}

impl Default for FalloffCurve {
    fn default() -> Self {
        Self::Smoothstep
    }
}

impl FalloffCurve {
    /// Evaluate the curve at normalised radial distance `t` in `[0, 1]`.
    pub fn apply(self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Smoothstep => {
                let s = 1.0 - t;
                s * s * (3.0 - 2.0 * s)
            }
            Self::Linear => 1.0 - t,
            Self::Hard => {
                if t < 1.0 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

/// Per-stamp combine operator. Decides how the stamp's `target_h`
/// merges with `base_h` inside the stamp's footprint.
///
/// Convention: `target_h` is an absolute world-Y. `position.y` is the
/// stamp's "neutral" Y — Mountain's `target_h - position.y` is the
/// peak's height delta above the stamp's base.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CombineOp {
    /// `H = base + (target - position.y)` — adds the stamp's delta
    /// unconditionally. Useful for layering noise-style stamps.
    Add,
    /// `H = base - (target - position.y)` — subtracts the delta.
    Subtract,
    /// `H = min(base, target)` — stamp wins where it's lower. Default
    /// for Lake. V1 uses straight min; the falloff curve shapes the rim.
    SmoothMin,
    /// `H = max(base, target)` — stamp wins where it's higher. Default
    /// for Mountain / Hill / Plateau.
    SmoothMax,
    /// `H = target` — stamp wins outright. Default for Flatten.
    Replace,
}

/// Family of stamp shape. V1 = five heightmap kinds. Volumetric variants
/// (Cave, Overhang) reserved for V2.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum StampKind {
    /// Circular peak. Target height ranges from `position.y` at the
    /// rim to `position.y + h_max` at the centre, shaped by `falloff`.
    /// Default op: SmoothMax.
    Mountain {
        /// Peak height above `position.y`, in metres.
        h_max: f32,
        /// Horizontal radius of the footprint, in metres.
        radius: f32,
        /// Radial transition curve from centre to rim.
        falloff: FalloffCurve,
    },
    /// Smaller circular peak. Same math as Mountain; separate kind so
    /// the editor can ship distinct defaults and Inspector copy.
    Hill {
        /// Peak height above `position.y`, in metres.
        h_max: f32,
        /// Horizontal radius of the footprint, in metres.
        radius: f32,
        /// Radial transition curve from centre to rim.
        falloff: FalloffCurve,
    },
    /// Circular basin. Target height ranges from `position.y` at the
    /// rim down to `position.y - depth` at the centre, shaped by
    /// `falloff`. Default op: SmoothMin.
    Lake {
        /// Maximum depth below `position.y`, in metres.
        depth: f32,
        /// Horizontal radius of the footprint, in metres.
        radius: f32,
        /// Radial transition curve from rim (0) to centre (1). Same
        /// semantics as Mountain — shore softness, not depth profile.
        falloff: FalloffCurve,
    },
    /// Rectangular plateau. Target height is `position.y` everywhere
    /// inside the rotated rectangle; nothing outside. Default op:
    /// SmoothMax (rises terrain to `position.y` where base is lower).
    Plateau {
        /// XZ half-extents in stamp-local frame, in metres.
        half_extents: Vec2,
    },
    /// Rectangular hard flatten. Identical footprint to Plateau but
    /// default op: Replace (forces `position.y` regardless of base).
    Flatten {
        /// XZ half-extents in stamp-local frame, in metres.
        half_extents: Vec2,
    },
}

impl StampKind {
    /// The default combine op for this kind. Authors can override
    /// per-instance via `Stamp.combine_op`.
    pub fn default_combine_op(self) -> CombineOp {
        match self {
            Self::Mountain { .. } | Self::Hill { .. } | Self::Plateau { .. } => {
                CombineOp::SmoothMax
            }
            Self::Lake { .. } => CombineOp::SmoothMin,
            Self::Flatten { .. } => CombineOp::Replace,
        }
    }
}

/// One stamp instance placed in world space.
///
/// `position.y` is the stamp's neutral elevation — Mountain rises
/// above it, Lake carves below, Plateau / Flatten target it directly.
/// `yaw` rotates rectangular stamps around the world Y axis (radians;
/// ignored by circular stamps).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Stamp {
    /// Shape + amplitude parameters.
    pub kind: StampKind,
    /// World-space anchor. For circular stamps this is the centre; for
    /// rectangular stamps it's the rotation origin.
    pub position: Vec3,
    /// Y-axis rotation in radians. Only affects rectangular stamps.
    pub yaw: f32,
    /// How this stamp's heightmap merges with the base. Initialised
    /// to `kind.default_combine_op()`; authors can edit per-instance.
    pub combine_op: CombineOp,
    /// Composition priority. Stamps apply in ascending
    /// `(priority, tiebreaker)` order so the baked result is
    /// deterministic across ECS iteration orders.
    pub priority: i32,
    /// Optional material override that wins inside the stamp's
    /// footprint. `None` falls through to the base TerrainFn material.
    /// V1 leaves the integration with the leaf material rule for
    /// Phase 7; the field is reserved here so the data model and
    /// scene serde don't churn at that point.
    pub material_override: Option<u16>,
}

impl Stamp {
    /// Construct a stamp with the kind's default combine op.
    pub fn new(kind: StampKind, position: Vec3) -> Self {
        Self {
            kind,
            position,
            yaw: 0.0,
            combine_op: kind.default_combine_op(),
            priority: 0,
            material_override: None,
        }
    }

    /// Evaluate the stamp's target absolute height at a world-space XZ.
    /// Returns `None` if the sample falls outside the stamp's footprint
    /// — the caller skips combining for that stamp at that point.
    pub fn sample_height(&self, wx: f32, wz: f32) -> Option<f32> {
        match self.kind {
            StampKind::Mountain { h_max, radius, falloff }
            | StampKind::Hill { h_max, radius, falloff } => {
                if radius <= 0.0 {
                    return None;
                }
                let dx = wx - self.position.x;
                let dz = wz - self.position.z;
                let d = (dx * dx + dz * dz).sqrt();
                if d > radius {
                    return None;
                }
                let t = d / radius;
                Some(self.position.y + h_max * falloff.apply(t))
            }
            StampKind::Lake { depth, radius, falloff } => {
                if radius <= 0.0 {
                    return None;
                }
                let dx = wx - self.position.x;
                let dz = wz - self.position.z;
                let d = (dx * dx + dz * dz).sqrt();
                if d > radius {
                    return None;
                }
                let t = d / radius;
                Some(self.position.y - depth * falloff.apply(t))
            }
            StampKind::Plateau { half_extents } | StampKind::Flatten { half_extents } => {
                let (lx, lz) = self.world_to_local_xz(wx, wz);
                if lx.abs() <= half_extents.x && lz.abs() <= half_extents.y {
                    Some(self.position.y)
                } else {
                    None
                }
            }
        }
    }

    /// Rotate a world-XZ offset into the stamp's local frame. Only the
    /// stamp's `yaw` is applied (Y-up rotation).
    fn world_to_local_xz(&self, wx: f32, wz: f32) -> (f32, f32) {
        let dx = wx - self.position.x;
        let dz = wz - self.position.z;
        let (s, c) = self.yaw.sin_cos();
        let lx = c * dx + s * dz;
        let lz = -s * dx + c * dz;
        (lx, lz)
    }

    /// World-space AABB covering the stamp's region of influence.
    /// Used by the spatial index for tile invalidation: any tile whose
    /// AABB intersects this is dirtied when the stamp is added, moved,
    /// or deleted.
    pub fn aabb(&self) -> Aabb {
        match self.kind {
            StampKind::Mountain { h_max, radius, .. }
            | StampKind::Hill { h_max, radius, .. } => Aabb {
                min: Vec3::new(
                    self.position.x - radius,
                    self.position.y.min(self.position.y + h_max),
                    self.position.z - radius,
                ),
                max: Vec3::new(
                    self.position.x + radius,
                    self.position.y.max(self.position.y + h_max),
                    self.position.z + radius,
                ),
            },
            StampKind::Lake { depth, radius, .. } => Aabb {
                min: Vec3::new(
                    self.position.x - radius,
                    self.position.y - depth.max(0.0),
                    self.position.z - radius,
                ),
                max: Vec3::new(
                    self.position.x + radius,
                    self.position.y,
                    self.position.z + radius,
                ),
            },
            StampKind::Plateau { half_extents } | StampKind::Flatten { half_extents } => {
                // Yaw-only axis-aligned bound of an XZ rectangle:
                // rx = he.x * |cos| + he.y * |sin|, rz = he.x * |sin| + he.y * |cos|.
                let cosy = self.yaw.cos().abs();
                let siny = self.yaw.sin().abs();
                let rx = half_extents.x * cosy + half_extents.y * siny;
                let rz = half_extents.x * siny + half_extents.y * cosy;
                Aabb {
                    min: Vec3::new(self.position.x - rx, self.position.y, self.position.z - rz),
                    max: Vec3::new(self.position.x + rx, self.position.y, self.position.z + rz),
                }
            }
        }
    }
}

/// Fold a stamp's `target_h` onto the running `base_h` per the op.
///
/// V1 SmoothMin/SmoothMax use straight min/max — the per-stamp falloff
/// curve provides the rim transition. A real smooth-min kernel may
/// land in V2 alongside multi-stamp blending; the enum variant names
/// match the design doc and the API surface is stable.
pub fn combine_heights(base: f32, target: f32, op: CombineOp, stamp_neutral_y: f32) -> f32 {
    match op {
        CombineOp::Add => base + (target - stamp_neutral_y),
        CombineOp::Subtract => base - (target - stamp_neutral_y),
        CombineOp::SmoothMin => base.min(target),
        CombineOp::SmoothMax => base.max(target),
        CombineOp::Replace => target,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    // ── FalloffCurve ────────────────────────────────────────────────────

    #[test]
    fn falloff_endpoints_are_one_and_zero() {
        for curve in [FalloffCurve::Smoothstep, FalloffCurve::Linear, FalloffCurve::Hard] {
            assert!((curve.apply(0.0) - 1.0).abs() < 1e-6, "{:?}@0 != 1", curve);
            assert!(curve.apply(1.0).abs() < 1e-6, "{:?}@1 != 0", curve);
        }
    }

    #[test]
    fn falloff_clamps_out_of_range() {
        // Values outside [0, 1] clamp; never NaN.
        assert!((FalloffCurve::Smoothstep.apply(-5.0) - 1.0).abs() < 1e-6);
        assert!(FalloffCurve::Smoothstep.apply(5.0).abs() < 1e-6);
    }

    #[test]
    fn smoothstep_is_monotonic() {
        let mut last = 1.0_f32;
        for i in 0..=10 {
            let t = i as f32 / 10.0;
            let v = FalloffCurve::Smoothstep.apply(t);
            assert!(v <= last + 1e-6, "smoothstep not monotonic at t={t}: {v} > {last}");
            last = v;
        }
    }

    // ── StampKind defaults ─────────────────────────────────────────────

    #[test]
    fn mountain_defaults_to_smoothmax() {
        let m = StampKind::Mountain {
            h_max: 100.0,
            radius: 50.0,
            falloff: FalloffCurve::Smoothstep,
        };
        assert_eq!(m.default_combine_op(), CombineOp::SmoothMax);
    }

    #[test]
    fn lake_defaults_to_smoothmin() {
        let l = StampKind::Lake {
            depth: 20.0,
            radius: 80.0,
            falloff: FalloffCurve::Smoothstep,
        };
        assert_eq!(l.default_combine_op(), CombineOp::SmoothMin);
    }

    #[test]
    fn flatten_defaults_to_replace() {
        let f = StampKind::Flatten {
            half_extents: Vec2::new(10.0, 10.0),
        };
        assert_eq!(f.default_combine_op(), CombineOp::Replace);
    }

    #[test]
    fn plateau_defaults_to_smoothmax() {
        // Plateau levels things UP to position.y, doesn't carve.
        let p = StampKind::Plateau {
            half_extents: Vec2::new(10.0, 10.0),
        };
        assert_eq!(p.default_combine_op(), CombineOp::SmoothMax);
    }

    // ── sample_height ─────────────────────────────────────────────────

    fn mountain_at(p: Vec3, h_max: f32, radius: f32) -> Stamp {
        Stamp::new(
            StampKind::Mountain {
                h_max,
                radius,
                falloff: FalloffCurve::Smoothstep,
            },
            p,
        )
    }

    #[test]
    fn mountain_peak_at_centre_is_position_plus_h_max() {
        let m = mountain_at(Vec3::new(100.0, 50.0, 200.0), 30.0, 40.0);
        let h = m.sample_height(100.0, 200.0).expect("inside footprint");
        assert!((h - 80.0).abs() < 1e-4, "expected 80, got {h}");
    }

    #[test]
    fn mountain_at_rim_equals_position_y() {
        // Smoothstep is exactly 0 at t=1.
        let m = mountain_at(Vec3::new(0.0, 10.0, 0.0), 20.0, 5.0);
        // 5m east is the rim.
        let h = m.sample_height(5.0, 0.0).expect("on rim");
        assert!((h - 10.0).abs() < 1e-4, "rim should equal position.y; got {h}");
    }

    #[test]
    fn mountain_outside_footprint_returns_none() {
        let m = mountain_at(Vec3::new(0.0, 0.0, 0.0), 50.0, 5.0);
        // 10m away — outside the 5m radius.
        assert_eq!(m.sample_height(10.0, 0.0), None);
    }

    #[test]
    fn mountain_zero_radius_is_no_op() {
        // Degenerate stamp: zero radius. Must return None everywhere
        // (else we'd divide by zero in the normalised distance).
        let m = mountain_at(Vec3::new(0.0, 0.0, 0.0), 50.0, 0.0);
        assert_eq!(m.sample_height(0.0, 0.0), None);
    }

    #[test]
    fn mountain_amplitude_scales_linearly() {
        // Same shape at different h_max — the peak ratio matches the
        // h_max ratio. This is the "huge mountains vs gentle texture"
        // authoring property.
        let big = mountain_at(Vec3::new(0.0, 0.0, 0.0), 200.0, 50.0);
        let small = mountain_at(Vec3::new(0.0, 0.0, 0.0), 5.0, 50.0);
        let hb = big.sample_height(0.0, 0.0).unwrap();
        let hs = small.sample_height(0.0, 0.0).unwrap();
        assert!((hb / hs - 40.0).abs() < 1e-3);
    }

    #[test]
    fn lake_centre_is_position_minus_depth() {
        let l = Stamp::new(
            StampKind::Lake {
                depth: 25.0,
                radius: 40.0,
                falloff: FalloffCurve::Smoothstep,
            },
            Vec3::new(0.0, 10.0, 0.0),
        );
        let h = l.sample_height(0.0, 0.0).expect("inside");
        assert!((h - (-15.0)).abs() < 1e-4, "got {h}");
    }

    #[test]
    fn lake_rim_equals_position_y() {
        let l = Stamp::new(
            StampKind::Lake {
                depth: 25.0,
                radius: 4.0,
                falloff: FalloffCurve::Smoothstep,
            },
            Vec3::new(0.0, 10.0, 0.0),
        );
        let h = l.sample_height(4.0, 0.0).expect("on rim");
        assert!((h - 10.0).abs() < 1e-4);
    }

    #[test]
    fn flatten_inside_rect_returns_target_y() {
        let f = Stamp::new(
            StampKind::Flatten {
                half_extents: Vec2::new(10.0, 5.0),
            },
            Vec3::new(0.0, 100.0, 0.0),
        );
        assert_eq!(f.sample_height(0.0, 0.0), Some(100.0));
        assert_eq!(f.sample_height(9.0, 4.0), Some(100.0));
    }

    #[test]
    fn flatten_outside_rect_returns_none() {
        let f = Stamp::new(
            StampKind::Flatten {
                half_extents: Vec2::new(10.0, 5.0),
            },
            Vec3::new(0.0, 100.0, 0.0),
        );
        // Past the x half-extent.
        assert_eq!(f.sample_height(11.0, 0.0), None);
        // Past the z half-extent.
        assert_eq!(f.sample_height(0.0, 6.0), None);
    }

    #[test]
    fn flatten_respects_yaw() {
        // Long thin rectangle 20 × 2 oriented along world-X with yaw=0:
        // (15, 0, 0) is OUTSIDE (|lx|=15 > 10), (0, 0, 1.5) is OUTSIDE
        // (|lz|=1.5 > 1). After yaw=π/2 the long axis aligns with world
        // Z instead: (15, 0, 0) flips to OUTSIDE-along-local-z and
        // (0, 0, 15) becomes INSIDE.
        let mut f = Stamp::new(
            StampKind::Flatten {
                half_extents: Vec2::new(10.0, 1.0),
            },
            Vec3::new(0.0, 50.0, 0.0),
        );
        // No yaw — long axis = world X.
        assert_eq!(f.sample_height(9.0, 0.0), Some(50.0));
        assert_eq!(f.sample_height(0.0, 0.0, ), Some(50.0));
        assert_eq!(f.sample_height(0.0, 1.5), None);
        assert_eq!(f.sample_height(15.0, 0.0), None);

        // 90° yaw — long axis swings to world Z.
        f.yaw = std::f32::consts::FRAC_PI_2;
        assert_eq!(f.sample_height(0.0, 9.0), Some(50.0));
        assert_eq!(f.sample_height(1.5, 0.0), None);
        assert_eq!(f.sample_height(0.0, 15.0), None);
    }

    // ── AABB ──────────────────────────────────────────────────────────

    #[test]
    fn mountain_aabb_contains_full_extent() {
        let m = mountain_at(Vec3::new(100.0, 50.0, 200.0), 30.0, 40.0);
        let aabb = m.aabb();
        assert!((aabb.min.x - 60.0).abs() < 1e-4);
        assert!((aabb.max.x - 140.0).abs() < 1e-4);
        assert!((aabb.min.y - 50.0).abs() < 1e-4);
        assert!((aabb.max.y - 80.0).abs() < 1e-4);
        assert!((aabb.min.z - 160.0).abs() < 1e-4);
        assert!((aabb.max.z - 240.0).abs() < 1e-4);
    }

    #[test]
    fn lake_aabb_extends_down_to_basin_floor() {
        let l = Stamp::new(
            StampKind::Lake {
                depth: 25.0,
                radius: 40.0,
                falloff: FalloffCurve::Smoothstep,
            },
            Vec3::new(0.0, 10.0, 0.0),
        );
        let aabb = l.aabb();
        assert!((aabb.min.y - (-15.0)).abs() < 1e-4);
        assert!((aabb.max.y - 10.0).abs() < 1e-4);
    }

    #[test]
    fn flatten_aabb_expands_under_yaw() {
        // 45° rotation of a 10×10 rect has axis-aligned bound of
        // ±10*sqrt(2)/2 + ±10*sqrt(2)/2 = ±10*sqrt(2) ≈ ±14.14.
        let mut f = Stamp::new(
            StampKind::Flatten {
                half_extents: Vec2::new(10.0, 10.0),
            },
            Vec3::new(0.0, 0.0, 0.0),
        );
        f.yaw = std::f32::consts::FRAC_PI_4;
        let aabb = f.aabb();
        let expected = 10.0 * std::f32::consts::SQRT_2;
        assert!(
            (aabb.max.x - expected).abs() < 1e-3,
            "expected ~{expected}, got {}",
            aabb.max.x
        );
    }

    // ── combine_heights ───────────────────────────────────────────────

    #[test]
    fn combine_smoothmax_picks_higher() {
        // Mountain stamp: target above base → stamp wins.
        let h = combine_heights(10.0, 50.0, CombineOp::SmoothMax, 0.0);
        assert!((h - 50.0).abs() < 1e-6);
        // Mountain stamp: target below base → base wins (mountain
        // doesn't depress terrain).
        let h = combine_heights(100.0, 50.0, CombineOp::SmoothMax, 0.0);
        assert!((h - 100.0).abs() < 1e-6);
    }

    #[test]
    fn combine_smoothmin_picks_lower() {
        // Lake stamp: target below base → stamp wins.
        let h = combine_heights(50.0, 10.0, CombineOp::SmoothMin, 0.0);
        assert!((h - 10.0).abs() < 1e-6);
        // Lake stamp: target above base → base wins (lake doesn't
        // raise terrain).
        let h = combine_heights(10.0, 50.0, CombineOp::SmoothMin, 0.0);
        assert!((h - 10.0).abs() < 1e-6);
    }

    #[test]
    fn combine_add_uses_delta_from_neutral() {
        // Mountain at neutral=0, target=30 → +30 above base.
        let h = combine_heights(10.0, 30.0, CombineOp::Add, 0.0);
        assert!((h - 40.0).abs() < 1e-6);
        // Same delta from a stamp at neutral=100 → still +30.
        let h = combine_heights(10.0, 130.0, CombineOp::Add, 100.0);
        assert!((h - 40.0).abs() < 1e-6);
    }

    #[test]
    fn combine_subtract_inverts_add() {
        let h = combine_heights(10.0, 30.0, CombineOp::Subtract, 0.0);
        assert!((h - (-20.0)).abs() < 1e-6);
    }

    #[test]
    fn combine_replace_forces_target() {
        let h = combine_heights(10.0, 50.0, CombineOp::Replace, 0.0);
        assert!((h - 50.0).abs() < 1e-6);
        let h = combine_heights(1000.0, 50.0, CombineOp::Replace, 0.0);
        assert!((h - 50.0).abs() < 1e-6);
    }

    // ── round-trip ────────────────────────────────────────────────────

    #[test]
    fn stamp_serde_roundtrip() {
        let s = Stamp::new(
            StampKind::Lake {
                depth: 25.0,
                radius: 80.0,
                falloff: FalloffCurve::Linear,
            },
            Vec3::new(100.0, 5.0, -50.0),
        );
        let json = serde_json::to_string(&s).expect("serialise");
        let back: Stamp = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(s, back);
    }
}
