//! Stamps — Layer 2 of the three-layer terrain source model.
//!
//! A `Stamp` is a heightmap feature placed in world space. It composes
//! over the base `TerrainFn` (Layer 1) and underneath sculpt edits
//! (Layer 3). V2 ships five heightmap kinds — Mountain, Hill, Lake,
//! Plateau, Flatten — each with a richer parameter set than V1:
//!
//! * Cross-cutting [`ShapeNoise`] perturbs each stamp's footprint so
//!   circular stamps stop looking like upside-down bowls and rectangles
//!   stop looking like perfect rectangles.
//! * Mountain / Hill / Lake gain an `aspect` (anisotropic radii) and
//!   the spinal-ridge profile (Mountain / Hill).
//! * Lake gains a `floor_flat_frac` for true flat-bottomed basins.
//! * Plateau / Flatten gain `corner_radius_m`, `edge_falloff_m`, and
//!   `tilt` — rounded corners, soft rims, sloped flats.
//! * Falloff curve set expands beyond Smoothstep / Linear / Hard to
//!   Quadratic / Cubic / Exponential.
//!
//! Volumetric stamps (caves, overhangs) are still reserved for V2.x —
//! they need a parallel SDF-contribution branch in the bake fold.
//!
//! ### Composition pipeline
//!
//! For each voxel sample, the `bake_tile` path does:
//!
//! 1. Evaluate Layer 1 `TerrainFn` → `base_sample` with `base_h`.
//! 2. Query the `StampIndex` for stamps overlapping `(wx, wz)`.
//! 3. For each stamp in deterministic order, evaluate
//!    `stamp.sample_height(wx, wz)`; if `Some(StampSample { target_h,
//!    weight })`, combine via `combine_heights(base_h, target_h, op)`
//!    and blend toward `base_h` by `(1 - weight)`. `weight = 1`
//!    reproduces V1's hard-edged behaviour; `weight < 1` lets a
//!    stamp dial down its effect at the rim (rounded corners,
//!    noisy shores).
//! 4. Final `sd = wy - H` where `H` is the composed height.
//!
//! Materials follow a similar overlay: a stamp can carry an optional
//! `material_override` consumed inside its footprint.

use crate::value_noise::fbm_2d;
use arvx_core::Aabb;
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

/// How the stamp's heightmap falls off at the rim of its footprint.
///
/// `apply(t)` returns 1.0 at `t = 0` (centre of the footprint) and
/// 0.0 at `t = 1` (rim). Outside the rim the stamp doesn't apply at
/// all — the curve only shapes the transition inside the footprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FalloffCurve {
    /// `1 - smoothstep(0, 1, t)` — zero slope at both ends. Default.
    Smoothstep,
    /// `1 - t` — constant slope.
    Linear,
    /// `1` for `t < 1`, `0` at the rim. No transition; useful for
    /// debugging and the rare authoring case that wants a hard rim.
    Hard,
    /// `(1 - t)²` — sharper peak than Smoothstep, slower rim drop.
    /// Mountains with this profile have a pointy summit.
    Quadratic,
    /// `(1 - t)³` — even sharper peak; gentle base.
    Cubic,
    /// `exp(-3 * t) * (1 - t)` — long flat tail toward the rim.
    /// Good for hills that blend almost imperceptibly into their
    /// surroundings.
    Exponential,
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
            Self::Quadratic => {
                let s = 1.0 - t;
                s * s
            }
            Self::Cubic => {
                let s = 1.0 - t;
                s * s * s
            }
            Self::Exponential => {
                // Tail factor (1 - t) forces an exact zero at the rim
                // so the stamp doesn't punch a discontinuity into the
                // composed heightfield. The exp(-3t) shapes the
                // interior.
                (-3.0 * t).exp() * (1.0 - t)
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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

/// Cross-cutting FBM shape perturbation applied to a stamp's
/// footprint. Same noise primitive as [`crate::fbm`]; lives on
/// every stamp so any kind can be made organic with a single field.
///
/// For circular kinds (Mountain / Hill / Lake) the noise perturbs the
/// **radial distance** before the falloff lookup — peaks blob, shores
/// scallop.
///
/// For rectangular kinds (Plateau / Flatten) the noise perturbs the
/// **local XZ coordinates** before the rounded-box SDF — rims meander.
///
/// `amp_m == 0` short-circuits the noise sample (zero per-voxel cost
/// when stamps don't use it).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ShapeNoise {
    /// Magnitude of the perturbation in metres of radial / coordinate
    /// displacement. `0.0` disables noise entirely.
    pub amp_m: f32,
    /// Spatial period of the lowest noise octave, in metres.
    pub scale_m: f32,
    /// Determinism seed.
    pub seed: u32,
    /// Octave count for the FBM. Stamps typically want 2–4 — enough
    /// to look natural without spending cycles.
    pub octaves: u8,
}

impl Default for ShapeNoise {
    fn default() -> Self {
        Self {
            amp_m: 0.0,
            scale_m: 8.0,
            seed: 0,
            octaves: 2,
        }
    }
}

impl ShapeNoise {
    /// `true` if this perturbation will produce non-zero output.
    pub fn is_active(self) -> bool {
        self.amp_m > 0.0 && self.scale_m > 0.0 && self.octaves > 0
    }

    /// Evaluate FBM at world-space `(wx, wz)` and scale by `amp_m`.
    /// Returns 0 for disabled noise.
    pub fn eval(self, wx: f32, wz: f32) -> f32 {
        if !self.is_active() {
            return 0.0;
        }
        let n = fbm_2d(wx / self.scale_m, wz / self.scale_m, self.octaves, self.seed);
        n * self.amp_m
    }
}

/// Family of stamp shape. V2 keeps the V1 variant set; each variant
/// gained extra knobs. New fields all carry `#[serde(default)]` so
/// scenes authored against V1 deserialize cleanly.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum StampKind {
    /// Circular peak. Target height ranges from `position.y` at the
    /// rim to `position.y + h_max` at the centre, shaped by `falloff`.
    /// Default op: SmoothMax.
    Mountain {
        /// Peak height above `position.y`, in metres.
        h_max: f32,
        /// Horizontal radius along the local-X axis, in metres.
        radius: f32,
        /// Radial transition curve from centre to rim.
        falloff: FalloffCurve,
        /// Radius along local-Z as a fraction of `radius`. `1.0`
        /// (default) is circular; `0.5` is an east–west ellipse.
        #[serde(default = "default_aspect")]
        aspect: f32,
        /// `0.0` = round dome (V1 behaviour). `> 0` carves valleys
        /// between `ridge_count` radial arms, producing a spine /
        /// ridge profile. Clamped to `[0, 1]` at evaluation.
        #[serde(default)]
        ridge_strength: f32,
        /// Number of radial ridges when `ridge_strength > 0`. `3`
        /// is the default; gives the classic three-spine mountain
        /// silhouette.
        #[serde(default = "default_ridge_count")]
        ridge_count: u8,
    },
    /// Smaller circular peak. Same math as Mountain; separate kind so
    /// the editor can ship distinct defaults and Inspector copy.
    Hill {
        /// Peak height above `position.y`, in metres.
        h_max: f32,
        /// Horizontal radius along local-X, in metres.
        radius: f32,
        /// Radial transition curve.
        falloff: FalloffCurve,
        /// See [`StampKind::Mountain`] field doc.
        #[serde(default = "default_aspect")]
        aspect: f32,
        /// See [`StampKind::Mountain`] field doc.
        #[serde(default)]
        ridge_strength: f32,
        /// See [`StampKind::Mountain`] field doc.
        #[serde(default = "default_ridge_count")]
        ridge_count: u8,
    },
    /// Circular basin. Target height ranges from `position.y` at the
    /// rim down to `position.y - depth` at the centre, shaped by
    /// `falloff`. Default op: SmoothMin.
    Lake {
        /// Maximum depth below `position.y`, in metres.
        depth: f32,
        /// Horizontal radius along local-X, in metres.
        radius: f32,
        /// Wall transition curve from rim (0) to flat-floor edge (1).
        falloff: FalloffCurve,
        /// Radius along local-Z as a fraction of `radius`.
        #[serde(default = "default_aspect")]
        aspect: f32,
        /// Fraction of the radius that's a flat lake floor at full
        /// depth. `0.0` (default) = pure V1 bowl. `0.6` = wide flat
        /// floor with sloped walls covering the outer 40 %.
        #[serde(default)]
        floor_flat_frac: f32,
        /// Outer-rim weight falloff band, in metres. Inside this
        /// band the lake's effect ramps from `0` at `t = 1` (rim)
        /// to `1` at the band's inner edge. Lakes on terrain that
        /// sits *above* `position.y` need this — without it,
        /// SmoothMin clamps the surface down to `position.y` right
        /// at the rim, producing a vertical cliff where the lake
        /// meets the surrounding hillside. `0.0` (default for
        /// pre-V2.2 scenes) gives the original sharp rim.
        #[serde(default)]
        edge_falloff_m: f32,
    },
    /// Rectangular plateau. Target height is `position.y` (plus
    /// optional tilt) inside the rotated rectangle. Default op:
    /// SmoothMax.
    Plateau {
        /// XZ half-extents in stamp-local frame, in metres.
        half_extents: Vec2,
        /// Inner-corner radius. Subtracted from `half_extents` to
        /// shape the rounded box. `0.0` (default) = sharp corners.
        #[serde(default)]
        corner_radius_m: f32,
        /// Soft-rim band width. Inside the band, the stamp's
        /// `weight` ramps from `0` at the boundary to `1` at the
        /// inner edge — the bake fold blends the stamp's effect
        /// back toward base over this distance. `0.0` (default) =
        /// hard edge.
        #[serde(default)]
        edge_falloff_m: f32,
        /// Slope of the plateau in metres-per-metre along each
        /// local axis. `tilt.x = 0.05` raises the +X edge by 5 cm
        /// per metre. `Vec2::ZERO` (default) = dead-level.
        #[serde(default)]
        tilt: Vec2,
    },
    /// Rectangular hard flatten. Identical footprint to Plateau but
    /// default op: Replace (forces `position.y` regardless of base).
    Flatten {
        /// XZ half-extents in stamp-local frame, in metres.
        half_extents: Vec2,
        /// See [`StampKind::Plateau`] field doc.
        #[serde(default)]
        corner_radius_m: f32,
        /// See [`StampKind::Plateau`] field doc.
        #[serde(default)]
        edge_falloff_m: f32,
        /// See [`StampKind::Plateau`] field doc.
        #[serde(default)]
        tilt: Vec2,
    },
}

fn default_aspect() -> f32 {
    1.0
}
fn default_ridge_count() -> u8 {
    3
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
/// `yaw` rotates the stamp's local frame around world Y (radians).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Stamp {
    /// Shape + amplitude parameters.
    pub kind: StampKind,
    /// World-space anchor. For circular stamps this is the centre; for
    /// rectangular stamps it's the rotation origin.
    pub position: Vec3,
    /// Y-axis rotation in radians. Affects all stamps now that
    /// circular kinds carry anisotropic radii.
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
    pub material_override: Option<u16>,
    /// Cross-cutting FBM perturbation. `amp_m = 0` disables.
    #[serde(default)]
    pub shape_noise: ShapeNoise,
}

/// Result of evaluating a stamp at a world-XZ sample.
///
/// `target_h` is the absolute world-Y the stamp wants to impose
/// (passed to `combine_heights` against the running base).
///
/// `weight` is the fraction of the stamp's effect to mix in — `1.0`
/// reproduces V1's hard-stamp behaviour; lower values pull the
/// composed height back toward the base. Used by rounded-rect and
/// edge-falloff features that want a gradual rim.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StampSample {
    /// Absolute world-Y the stamp wants to impose at this point.
    pub target_h: f32,
    /// `[0, 1]`. `1.0` = full stamp effect; `0.0` = no effect (caller
    /// should treat as "no stamp here").
    pub weight: f32,
}

impl Stamp {
    /// Construct a stamp with the kind's default combine op and no
    /// shape noise / material override.
    pub fn new(kind: StampKind, position: Vec3) -> Self {
        Self {
            kind,
            position,
            yaw: 0.0,
            combine_op: kind.default_combine_op(),
            priority: 0,
            material_override: None,
            shape_noise: ShapeNoise::default(),
        }
    }

    /// Evaluate the stamp's target absolute height + weight at a
    /// world-space XZ. Returns `None` if the sample falls outside the
    /// stamp's footprint (caller skips combining for that stamp).
    pub fn sample_height(&self, wx: f32, wz: f32) -> Option<StampSample> {
        match self.kind {
            StampKind::Mountain {
                h_max,
                radius,
                falloff,
                aspect,
                ridge_strength,
                ridge_count,
            }
            | StampKind::Hill {
                h_max,
                radius,
                falloff,
                aspect,
                ridge_strength,
                ridge_count,
            } => self.sample_mountain_like(
                wx,
                wz,
                h_max,
                radius,
                falloff,
                aspect,
                ridge_strength,
                ridge_count,
            ),
            StampKind::Lake {
                depth,
                radius,
                falloff,
                aspect,
                floor_flat_frac,
                edge_falloff_m,
            } => self.sample_lake(
                wx,
                wz,
                depth,
                radius,
                falloff,
                aspect,
                floor_flat_frac,
                edge_falloff_m,
            ),
            StampKind::Plateau {
                half_extents,
                corner_radius_m,
                edge_falloff_m,
                tilt,
            }
            | StampKind::Flatten {
                half_extents,
                corner_radius_m,
                edge_falloff_m,
                tilt,
            } => self.sample_rect_like(
                wx,
                wz,
                half_extents,
                corner_radius_m,
                edge_falloff_m,
                tilt,
            ),
        }
    }

    fn sample_mountain_like(
        &self,
        wx: f32,
        wz: f32,
        h_max: f32,
        radius: f32,
        falloff: FalloffCurve,
        aspect: f32,
        ridge_strength: f32,
        ridge_count: u8,
    ) -> Option<StampSample> {
        if radius <= 0.0 {
            return None;
        }
        let (lx, lz) = self.world_to_local_xz(wx, wz);
        let radius_x = radius;
        let radius_z = radius * aspect.max(1e-3);

        // Anisotropic radial distance, normalised. `t` in [0, ∞);
        // the stamp applies only when t ≤ 1 (the unit-ellipse rim),
        // optionally perturbed by FBM noise.
        let nx = lx / radius_x;
        let nz = lz / radius_z;
        let mut t = (nx * nx + nz * nz).sqrt();

        // Shape noise pulls / pushes the rim radially. Sampled from
        // world coords so adjacent stamps with the same seed don't
        // overlap deterministically — each stamp's noise is anchored
        // to its world position via the offset baked into wx, wz.
        if self.shape_noise.is_active() {
            // Convert noise from world-metres to normalised radial
            // units by dividing by `radius` (the geometric mean
            // would be more correct for very elliptical stamps; the
            // simple form here is good enough for the small
            // perturbations stamps want).
            t += self.shape_noise.eval(wx, wz) / radius;
        }

        // Spinal ridge: scoop valleys between `ridge_count` radial
        // arms. `cos(angle * N)` cycles N times around the circle —
        // peaks at the arms (+1), troughs between (-1). We remap to
        // [0,1] so arms get a `0` valley bias and inter-arm troughs
        // get `1` (push t outward → falloff returns less height).
        // The bias fades to zero at the rim so the stamp still meets
        // its nominal boundary.
        if ridge_strength > 0.0 && ridge_count > 0 {
            let angle = lz.atan2(lx);
            let cos_n = (angle * ridge_count as f32).cos();
            // arm → cos_n = +1, valley → cos_n = -1
            let valley = (1.0 - cos_n) * 0.5; // valley=0 on arm, valley=1 in trough
            let strength = ridge_strength.clamp(0.0, 1.0);
            t += valley * strength * (1.0 - t).max(0.0);
        }

        if t > 1.0 {
            return None;
        }
        let h_at = self.position.y + h_max * falloff.apply(t);
        Some(StampSample { target_h: h_at, weight: 1.0 })
    }

    fn sample_lake(
        &self,
        wx: f32,
        wz: f32,
        depth: f32,
        radius: f32,
        falloff: FalloffCurve,
        aspect: f32,
        floor_flat_frac: f32,
        edge_falloff_m: f32,
    ) -> Option<StampSample> {
        if radius <= 0.0 {
            return None;
        }
        let (lx, lz) = self.world_to_local_xz(wx, wz);
        let radius_x = radius;
        let radius_z = radius * aspect.max(1e-3);
        let nx = lx / radius_x;
        let nz = lz / radius_z;
        let mut t = (nx * nx + nz * nz).sqrt();

        if self.shape_noise.is_active() {
            t += self.shape_noise.eval(wx, wz) / radius;
        }
        if t > 1.0 {
            return None;
        }

        // Flat-bottom: collapse the inner `floor_flat_frac` of the
        // radius to a single "wall-zero" position so the falloff
        // returns 1.0 across the entire flat region.
        let frac = floor_flat_frac.clamp(0.0, 0.99);
        let t_walls = if frac > 0.0 {
            ((t - frac) / (1.0 - frac)).max(0.0)
        } else {
            t
        };

        let h_at = self.position.y - depth * falloff.apply(t_walls);

        // Outer-rim weight falloff. Without this, SmoothMin clamps
        // the surface to position.y at t = 1 — fine when the base
        // terrain is also at position.y, vertical cliff when it
        // isn't. Ramping weight to 0 at the rim lets the base show
        // through at the very edge so the cliff softens.
        let weight = if edge_falloff_m > 0.0 && radius > 0.0 {
            let t_falloff = (edge_falloff_m / radius).clamp(0.0, 1.0);
            // (1 - t) is the distance inward from the rim, in
            // normalised radial units. Divide by the falloff width
            // (also normalised) and clamp.
            let ramp = ((1.0 - t) / t_falloff).clamp(0.0, 1.0);
            // Smoothstep for nicer visuals than a linear ramp.
            ramp * ramp * (3.0 - 2.0 * ramp)
        } else {
            1.0
        };

        Some(StampSample { target_h: h_at, weight })
    }

    fn sample_rect_like(
        &self,
        wx: f32,
        wz: f32,
        half_extents: Vec2,
        corner_radius_m: f32,
        edge_falloff_m: f32,
        tilt: Vec2,
    ) -> Option<StampSample> {
        let (mut lx, mut lz) = self.world_to_local_xz(wx, wz);

        // Noise perturbs the local coordinates → meandering rim.
        if self.shape_noise.is_active() {
            let nx = self.shape_noise.eval(wx, wz);
            let nz = self.shape_noise.eval(wz, wx);
            lx += nx;
            lz += nz;
        }

        // SDF-rounded-box distance. Negative inside, zero on rim,
        // positive outside.
        let corner = corner_radius_m
            .max(0.0)
            .min(half_extents.x.min(half_extents.y));
        let qx = lx.abs() - half_extents.x + corner;
        let qz = lz.abs() - half_extents.y + corner;
        let outside = Vec2::new(qx.max(0.0), qz.max(0.0)).length();
        let inside = qx.max(qz).min(0.0);
        let d = outside + inside - corner;

        if d > 0.0 {
            return None;
        }

        // Tilt: target_h slopes along local axes. Defaults to flat.
        let target_h = self.position.y + lx * tilt.x + lz * tilt.y;

        // Edge falloff: inside the soft-rim band the weight ramps
        // from 0 (at d = 0) to 1 (at d = -edge_falloff_m). Outside
        // the band (deep interior) weight stays at 1.
        let weight = if edge_falloff_m > 0.0 {
            let ramp = (-d / edge_falloff_m).clamp(0.0, 1.0);
            // Smoothstep gives nicer visuals than the linear ramp.
            ramp * ramp * (3.0 - 2.0 * ramp)
        } else {
            1.0
        };

        Some(StampSample { target_h, weight })
    }

    /// Rotate a world-XZ offset into the stamp's local frame.
    fn world_to_local_xz(&self, wx: f32, wz: f32) -> (f32, f32) {
        let dx = wx - self.position.x;
        let dz = wz - self.position.z;
        let (s, c) = self.yaw.sin_cos();
        let lx = c * dx + s * dz;
        let lz = -s * dx + c * dz;
        (lx, lz)
    }

    /// World-space AABB covering the stamp's region of influence.
    /// Includes the shape-noise margin and rectangular edge-falloff
    /// band so tile invalidation captures every voxel the stamp can
    /// reach.
    pub fn aabb(&self) -> Aabb {
        // Margin every kind picks up from shape noise — it pushes the
        // effective rim outward by up to `amp_m` metres.
        let noise_margin = self.shape_noise.amp_m.max(0.0);
        match self.kind {
            StampKind::Mountain { h_max, radius, aspect, .. }
            | StampKind::Hill { h_max, radius, aspect, .. } => {
                let rx = radius + noise_margin;
                let rz = radius * aspect.max(1e-3) + noise_margin;
                // Worst-case rotation of an ellipse fits inside the
                // larger axis (cheap superset).
                let r = rx.max(rz);
                Aabb {
                    min: Vec3::new(
                        self.position.x - r,
                        self.position.y.min(self.position.y + h_max),
                        self.position.z - r,
                    ),
                    max: Vec3::new(
                        self.position.x + r,
                        self.position.y.max(self.position.y + h_max),
                        self.position.z + r,
                    ),
                }
            }
            StampKind::Lake { depth, radius, aspect, .. } => {
                // edge_falloff_m doesn't extend the footprint
                // (the ramp lives INSIDE the rim at t in [1 -
                // t_falloff, 1]) — only noise_margin grows the
                // outer extent. Keeping the AABB tight here.
                let rx = radius + noise_margin;
                let rz = radius * aspect.max(1e-3) + noise_margin;
                let r = rx.max(rz);
                Aabb {
                    min: Vec3::new(
                        self.position.x - r,
                        self.position.y - depth.max(0.0),
                        self.position.z - r,
                    ),
                    max: Vec3::new(
                        self.position.x + r,
                        self.position.y,
                        self.position.z + r,
                    ),
                }
            }
            StampKind::Plateau { half_extents, edge_falloff_m, tilt, .. }
            | StampKind::Flatten { half_extents, edge_falloff_m, tilt, .. } => {
                let pad = noise_margin + edge_falloff_m.max(0.0);
                let cosy = self.yaw.cos().abs();
                let siny = self.yaw.sin().abs();
                let rx = (half_extents.x + pad) * cosy + (half_extents.y + pad) * siny;
                let rz = (half_extents.x + pad) * siny + (half_extents.y + pad) * cosy;
                // Tilt extends the vertical range by tilt.x*half_x +
                // tilt.y*half_y at the farthest corners.
                let dy = (tilt.x.abs() * half_extents.x + tilt.y.abs() * half_extents.y).max(0.0);
                Aabb {
                    min: Vec3::new(
                        self.position.x - rx,
                        self.position.y - dy,
                        self.position.z - rz,
                    ),
                    max: Vec3::new(
                        self.position.x + rx,
                        self.position.y + dy,
                        self.position.z + rz,
                    ),
                }
            }
        }
    }
}

/// Fold a stamp's `target_h` onto the running `base_h` per the op.
///
/// V1 SmoothMin/SmoothMax use straight min/max — the per-stamp falloff
/// curve provides the rim transition. A real smooth-min kernel may
/// land in V2.x alongside multi-stamp blending; the enum variant names
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
        for curve in [
            FalloffCurve::Smoothstep,
            FalloffCurve::Linear,
            FalloffCurve::Hard,
            FalloffCurve::Quadratic,
            FalloffCurve::Cubic,
            FalloffCurve::Exponential,
        ] {
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
    fn quadratic_is_steeper_than_smoothstep_at_centre() {
        // The two should diverge mid-range — Quadratic decays linearly
        // in (1-t), Smoothstep is cubic in (1-t). At t = 0.5,
        // Quadratic = 0.25 ; Smoothstep = 0.5.
        let q = FalloffCurve::Quadratic.apply(0.5);
        let s = FalloffCurve::Smoothstep.apply(0.5);
        assert!(q < s);
    }

    // ── ShapeNoise ─────────────────────────────────────────────────────

    #[test]
    fn shape_noise_zero_amp_is_inactive() {
        let n = ShapeNoise {
            amp_m: 0.0,
            ..Default::default()
        };
        assert!(!n.is_active());
        assert_eq!(n.eval(123.0, -45.0), 0.0);
    }

    #[test]
    fn shape_noise_is_deterministic() {
        let n = ShapeNoise {
            amp_m: 5.0,
            scale_m: 8.0,
            seed: 42,
            octaves: 3,
        };
        assert_eq!(n.eval(1.0, 2.0), n.eval(1.0, 2.0));
    }

    #[test]
    fn shape_noise_responds_to_seed_change() {
        let a = ShapeNoise {
            amp_m: 5.0,
            scale_m: 8.0,
            seed: 1,
            octaves: 3,
        };
        let mut b = a;
        b.seed = 2;
        let mut differ = 0;
        for i in 0..30 {
            let p = i as f32 * 0.7;
            if (a.eval(p, p) - b.eval(p, p)).abs() > 1e-3 {
                differ += 1;
            }
        }
        assert!(differ > 15, "{differ}/30 differed across seeds");
    }

    // ── StampKind defaults ─────────────────────────────────────────────

    fn default_mountain(h_max: f32, radius: f32) -> StampKind {
        StampKind::Mountain {
            h_max,
            radius,
            falloff: FalloffCurve::Smoothstep,
            aspect: 1.0,
            ridge_strength: 0.0,
            ridge_count: 3,
        }
    }

    fn default_lake(depth: f32, radius: f32) -> StampKind {
        StampKind::Lake {
            depth,
            radius,
            falloff: FalloffCurve::Smoothstep,
            aspect: 1.0,
            floor_flat_frac: 0.0,
            edge_falloff_m: 0.0,
        }
    }

    fn default_flatten(half: Vec2) -> StampKind {
        StampKind::Flatten {
            half_extents: half,
            corner_radius_m: 0.0,
            edge_falloff_m: 0.0,
            tilt: Vec2::ZERO,
        }
    }

    fn default_plateau(half: Vec2) -> StampKind {
        StampKind::Plateau {
            half_extents: half,
            corner_radius_m: 0.0,
            edge_falloff_m: 0.0,
            tilt: Vec2::ZERO,
        }
    }

    #[test]
    fn mountain_defaults_to_smoothmax() {
        assert_eq!(
            default_mountain(100.0, 50.0).default_combine_op(),
            CombineOp::SmoothMax
        );
    }

    #[test]
    fn lake_defaults_to_smoothmin() {
        assert_eq!(
            default_lake(20.0, 80.0).default_combine_op(),
            CombineOp::SmoothMin
        );
    }

    #[test]
    fn flatten_defaults_to_replace() {
        assert_eq!(
            default_flatten(Vec2::new(10.0, 10.0)).default_combine_op(),
            CombineOp::Replace
        );
    }

    #[test]
    fn plateau_defaults_to_smoothmax() {
        assert_eq!(
            default_plateau(Vec2::new(10.0, 10.0)).default_combine_op(),
            CombineOp::SmoothMax
        );
    }

    // ── sample_height: V1 parity (no new knobs engaged) ────────────────

    fn mountain_at(p: Vec3, h_max: f32, radius: f32) -> Stamp {
        Stamp::new(default_mountain(h_max, radius), p)
    }

    #[test]
    fn mountain_peak_at_centre_is_position_plus_h_max() {
        let m = mountain_at(Vec3::new(100.0, 50.0, 200.0), 30.0, 40.0);
        let s = m.sample_height(100.0, 200.0).expect("inside footprint");
        assert!((s.target_h - 80.0).abs() < 1e-4, "expected 80, got {}", s.target_h);
        assert!((s.weight - 1.0).abs() < 1e-6);
    }

    #[test]
    fn mountain_at_rim_equals_position_y() {
        // Smoothstep is exactly 0 at t=1.
        let m = mountain_at(Vec3::new(0.0, 10.0, 0.0), 20.0, 5.0);
        let s = m.sample_height(5.0, 0.0).expect("on rim");
        assert!((s.target_h - 10.0).abs() < 1e-4);
    }

    #[test]
    fn mountain_outside_footprint_returns_none() {
        let m = mountain_at(Vec3::new(0.0, 0.0, 0.0), 50.0, 5.0);
        assert_eq!(m.sample_height(10.0, 0.0), None);
    }

    #[test]
    fn mountain_zero_radius_is_no_op() {
        let m = mountain_at(Vec3::new(0.0, 0.0, 0.0), 50.0, 0.0);
        assert_eq!(m.sample_height(0.0, 0.0), None);
    }

    // ── anisotropic radii ──────────────────────────────────────────────

    #[test]
    fn anisotropic_mountain_squeezes_z() {
        // aspect 0.5 → radius_z = radius * 0.5. A point at (radius, 0)
        // is on the rim of the long axis; at (0, radius * 0.5) is on
        // the rim of the short axis; (0, radius) is OUTSIDE.
        let kind = StampKind::Mountain {
            h_max: 10.0,
            radius: 20.0,
            falloff: FalloffCurve::Linear,
            aspect: 0.5,
            ridge_strength: 0.0,
            ridge_count: 3,
        };
        let m = Stamp::new(kind, Vec3::ZERO);
        assert!(m.sample_height(20.0, 0.0).is_some()); // long-axis rim
        assert!(m.sample_height(0.0, 10.0).is_some()); // short-axis rim
        assert!(m.sample_height(0.0, 15.0).is_none()); // outside short
    }

    // ── ridge ──────────────────────────────────────────────────────────

    #[test]
    fn ridge_produces_non_uniform_height_at_fixed_radius() {
        // Compare two points at the same radius — one on a ridge arm
        // and one in a valley. With ridge_strength > 0 they must
        // differ. Without ridges they're equal.
        let kind_flat = StampKind::Mountain {
            h_max: 100.0,
            radius: 50.0,
            falloff: FalloffCurve::Linear,
            aspect: 1.0,
            ridge_strength: 0.0,
            ridge_count: 4,
        };
        let kind_ridge = StampKind::Mountain {
            h_max: 100.0,
            radius: 50.0,
            falloff: FalloffCurve::Linear,
            aspect: 1.0,
            ridge_strength: 0.6,
            ridge_count: 4,
        };
        let flat = Stamp::new(kind_flat, Vec3::ZERO);
        let ridge = Stamp::new(kind_ridge, Vec3::ZERO);

        // Two points at the same mid-radius. With ridge_count = 4 the
        // arms sit at angles 0, π/2, π, 3π/2; the valleys sit at
        // π/4, 3π/4, 5π/4, 7π/4. Pick one of each.
        let r = 25.0_f32;
        let a = (r, 0.0); // angle 0 — arm
        let b = (
            r * std::f32::consts::FRAC_PI_4.cos(),
            r * std::f32::consts::FRAC_PI_4.sin(),
        ); // 45° — valley

        let fa = flat.sample_height(a.0, a.1).unwrap();
        let fb = flat.sample_height(b.0, b.1).unwrap();
        // Flat: same height at same radius (radial symmetry). Tight
        // tolerance since both points are exactly at r=25.
        assert!((fa.target_h - fb.target_h).abs() < 0.1);

        let ra = ridge.sample_height(a.0, a.1).unwrap();
        let rb = ridge.sample_height(b.0, b.1).unwrap();
        // Ridged: arm position should be higher than valley position.
        assert!(
            ra.target_h > rb.target_h + 2.0,
            "arm {} should top valley {}",
            ra.target_h,
            rb.target_h
        );
    }

    // ── flat-bottom Lake ───────────────────────────────────────────────

    /// Lake's outer-rim `edge_falloff_m` ramps `weight` from 0 at the
    /// rim to 1 deeper inside. Lets the base terrain show through at
    /// the rim so SmoothMin doesn't clamp a hillside down to
    /// position.y as a vertical cliff.
    #[test]
    fn lake_edge_falloff_ramps_weight_at_rim() {
        let kind = StampKind::Lake {
            depth: 10.0,
            radius: 20.0,
            falloff: FalloffCurve::Smoothstep,
            aspect: 1.0,
            floor_flat_frac: 0.0,
            edge_falloff_m: 5.0, // 25% of radius
        };
        let l = Stamp::new(kind, Vec3::new(0.0, 10.0, 0.0));
        // Centre: full weight.
        let c = l.sample_height(0.0, 0.0).unwrap();
        assert!((c.weight - 1.0).abs() < 1e-4, "centre weight {}", c.weight);
        // Just inside the rim (r = 19.5): weight should be small —
        // (1 - 0.975) / 0.25 = 0.1; smoothstep(0.1) ≈ 0.028.
        let near_rim = l.sample_height(19.5, 0.0).unwrap();
        assert!(
            near_rim.weight < 0.1,
            "near-rim weight should taper to ~0; got {}",
            near_rim.weight,
        );
        // Inside the falloff band entrance (r ≈ 15, t = 0.75, just
        // beyond the band's inner edge at t = 1 - 0.25 = 0.75): full
        // weight.
        let inner = l.sample_height(14.0, 0.0).unwrap();
        assert!(
            (inner.weight - 1.0).abs() < 1e-4,
            "weight inside band entrance should be 1; got {}",
            inner.weight,
        );
    }

    /// Setting `edge_falloff_m = 0` reproduces V2.1 behaviour: weight
    /// is 1 throughout the footprint. Sanity check that the new
    /// default-disabled value is the back-compat path.
    #[test]
    fn lake_zero_edge_falloff_keeps_weight_one() {
        let kind = StampKind::Lake {
            depth: 5.0,
            radius: 10.0,
            falloff: FalloffCurve::Smoothstep,
            aspect: 1.0,
            floor_flat_frac: 0.0,
            edge_falloff_m: 0.0,
        };
        let l = Stamp::new(kind, Vec3::ZERO);
        for r in [0.0_f32, 3.0, 6.0, 9.5] {
            let s = l.sample_height(r, 0.0).unwrap();
            assert!((s.weight - 1.0).abs() < 1e-6, "weight at r={r} = {}", s.weight);
        }
    }

    #[test]
    fn flat_bottom_lake_holds_max_depth_across_floor() {
        let kind = StampKind::Lake {
            depth: 20.0,
            radius: 40.0,
            falloff: FalloffCurve::Smoothstep,
            aspect: 1.0,
            floor_flat_frac: 0.5,
            edge_falloff_m: 0.0,
        };
        let l = Stamp::new(kind, Vec3::new(0.0, 10.0, 0.0));
        // Centre — full depth.
        let c = l.sample_height(0.0, 0.0).unwrap();
        assert!((c.target_h - (-10.0)).abs() < 1e-4);
        // Half-radius point — still within flat region (frac=0.5) →
        // still full depth.
        let inner = l.sample_height(15.0, 0.0).unwrap();
        assert!((inner.target_h - (-10.0)).abs() < 1e-3);
        // Outside the flat band — walls climb back to position.y.
        let outer = l.sample_height(35.0, 0.0).unwrap();
        assert!(outer.target_h > -5.0, "outer wall {} should rise", outer.target_h);
    }

    // ── rounded Plateau / Flatten ──────────────────────────────────────

    #[test]
    fn rounded_plateau_excludes_corner() {
        // 10×10 plateau with corner_radius=5 leaves a true sharp
        // corner at (10, 10) outside the rounded box (distance from
        // the rounded-rect SDF is positive).
        let kind = StampKind::Plateau {
            half_extents: Vec2::new(10.0, 10.0),
            corner_radius_m: 5.0,
            edge_falloff_m: 0.0,
            tilt: Vec2::ZERO,
        };
        let p = Stamp::new(kind, Vec3::ZERO);
        // Centre — inside.
        assert!(p.sample_height(0.0, 0.0).is_some());
        // Edge-mid — inside.
        assert!(p.sample_height(9.5, 0.0).is_some());
        // Corner — outside the rounded shape.
        assert!(p.sample_height(10.0, 10.0).is_none());
    }

    #[test]
    fn edge_falloff_ramps_weight() {
        // Plateau with a 4 m falloff band. Centre → weight = 1; just
        // inside the rim → weight near 0.
        let kind = StampKind::Plateau {
            half_extents: Vec2::new(20.0, 20.0),
            corner_radius_m: 0.0,
            edge_falloff_m: 4.0,
            tilt: Vec2::ZERO,
        };
        let p = Stamp::new(kind, Vec3::ZERO);
        let centre = p.sample_height(0.0, 0.0).unwrap();
        assert!((centre.weight - 1.0).abs() < 1e-4);
        let near_rim = p.sample_height(19.5, 0.0).unwrap();
        assert!(
            near_rim.weight < 0.3,
            "weight near rim {} should be small",
            near_rim.weight
        );
    }

    #[test]
    fn tilted_plateau_slopes_along_local_axes() {
        // tilt.x = 0.1 → +X edge sits 0.1 * half_x = 1.0 m above
        // position.y. -X edge sits 1.0 m below.
        let kind = StampKind::Plateau {
            half_extents: Vec2::new(10.0, 10.0),
            corner_radius_m: 0.0,
            edge_falloff_m: 0.0,
            tilt: Vec2::new(0.1, 0.0),
        };
        let p = Stamp::new(kind, Vec3::new(0.0, 50.0, 0.0));
        let plus = p.sample_height(10.0, 0.0).unwrap();
        assert!((plus.target_h - 51.0).abs() < 1e-3);
        let minus = p.sample_height(-10.0, 0.0).unwrap();
        assert!((minus.target_h - 49.0).abs() < 1e-3);
    }

    // ── shape noise perturbs the rim ──────────────────────────────────

    #[test]
    fn shape_noise_can_push_point_outside_rim() {
        // Build two identical mountains, one with shape noise. Test
        // that at least one near-rim sample produces a different
        // result (the rim moved).
        let kind = default_mountain(10.0, 20.0);
        let plain = Stamp::new(kind, Vec3::ZERO);
        let mut noisy = plain;
        noisy.shape_noise = ShapeNoise {
            amp_m: 6.0,
            scale_m: 8.0,
            seed: 99,
            octaves: 3,
        };
        // Walk a circle just inside the nominal rim — half the points
        // should land in / out differently between the two stamps.
        let r = 19.5_f32;
        let mut differ = 0usize;
        for i in 0..32 {
            let a = i as f32 * std::f32::consts::TAU / 32.0;
            let p = (r * a.cos(), r * a.sin());
            let pl = plain.sample_height(p.0, p.1);
            let no = noisy.sample_height(p.0, p.1);
            if pl.is_some() != no.is_some()
                || pl.map(|s| s.target_h) != no.map(|s| s.target_h)
            {
                differ += 1;
            }
        }
        assert!(differ > 4, "shape noise should reshape the rim ({differ}/32)");
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
    fn mountain_aabb_includes_shape_noise_margin() {
        let mut m = mountain_at(Vec3::new(0.0, 0.0, 0.0), 10.0, 20.0);
        m.shape_noise = ShapeNoise {
            amp_m: 5.0,
            scale_m: 4.0,
            seed: 0,
            octaves: 2,
        };
        let aabb = m.aabb();
        // Without noise the X extent would be ±20. With 5 m noise it
        // grows to ±25.
        assert!((aabb.max.x - 25.0).abs() < 1e-4);
        assert!((aabb.min.x + 25.0).abs() < 1e-4);
    }

    #[test]
    fn rounded_plateau_aabb_includes_edge_falloff() {
        let kind = StampKind::Plateau {
            half_extents: Vec2::new(10.0, 10.0),
            corner_radius_m: 2.0,
            edge_falloff_m: 4.0,
            tilt: Vec2::ZERO,
        };
        let p = Stamp::new(kind, Vec3::ZERO);
        let aabb = p.aabb();
        // half_extents 10 + falloff 4 = 14.
        assert!((aabb.max.x - 14.0).abs() < 1e-4);
    }

    // ── combine_heights ───────────────────────────────────────────────

    #[test]
    fn combine_smoothmax_picks_higher() {
        assert!((combine_heights(10.0, 50.0, CombineOp::SmoothMax, 0.0) - 50.0).abs() < 1e-6);
        assert!((combine_heights(100.0, 50.0, CombineOp::SmoothMax, 0.0) - 100.0).abs() < 1e-6);
    }

    #[test]
    fn combine_smoothmin_picks_lower() {
        assert!((combine_heights(50.0, 10.0, CombineOp::SmoothMin, 0.0) - 10.0).abs() < 1e-6);
        assert!((combine_heights(10.0, 50.0, CombineOp::SmoothMin, 0.0) - 10.0).abs() < 1e-6);
    }

    #[test]
    fn combine_add_uses_delta_from_neutral() {
        assert!((combine_heights(10.0, 30.0, CombineOp::Add, 0.0) - 40.0).abs() < 1e-6);
        assert!((combine_heights(10.0, 130.0, CombineOp::Add, 100.0) - 40.0).abs() < 1e-6);
    }

    #[test]
    fn combine_subtract_inverts_add() {
        assert!((combine_heights(10.0, 30.0, CombineOp::Subtract, 0.0) - (-20.0)).abs() < 1e-6);
    }

    #[test]
    fn combine_replace_forces_target() {
        assert!((combine_heights(10.0, 50.0, CombineOp::Replace, 0.0) - 50.0).abs() < 1e-6);
        assert!((combine_heights(1000.0, 50.0, CombineOp::Replace, 0.0) - 50.0).abs() < 1e-6);
    }

    // ── serde round-trip ──────────────────────────────────────────────

    #[test]
    fn stamp_serde_roundtrip_with_all_new_fields() {
        let kind = StampKind::Plateau {
            half_extents: Vec2::new(10.0, 5.0),
            corner_radius_m: 1.5,
            edge_falloff_m: 2.5,
            tilt: Vec2::new(0.05, -0.02),
        };
        let mut s = Stamp::new(kind, Vec3::new(7.0, 8.0, 9.0));
        s.shape_noise = ShapeNoise {
            amp_m: 3.0,
            scale_m: 12.0,
            seed: 77,
            octaves: 3,
        };
        let json = serde_json::to_string(&s).expect("serialise");
        let back: Stamp = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(s, back);
    }

    /// V1 scenes have stamps without the new fields. They must
    /// deserialize cleanly (serde defaults to no noise, sharp
    /// corners, etc.).
    #[test]
    fn v1_scene_deserializes_with_new_fields_defaulted() {
        let v1_json = r#"{
            "kind": {
                "Mountain": { "h_max": 30.0, "radius": 20.0, "falloff": "Smoothstep" }
            },
            "position": [0, 0, 0],
            "yaw": 0.0,
            "combine_op": "SmoothMax",
            "priority": 0,
            "material_override": null
        }"#;
        let s: Stamp = serde_json::from_str(v1_json).expect("v1 deserialise");
        match s.kind {
            StampKind::Mountain {
                aspect,
                ridge_strength,
                ridge_count,
                ..
            } => {
                assert_eq!(aspect, 1.0);
                assert_eq!(ridge_strength, 0.0);
                assert_eq!(ridge_count, 3);
            }
            _ => panic!("wrong kind"),
        }
        assert_eq!(s.shape_noise.amp_m, 0.0);
    }
}
