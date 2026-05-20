//! Region falloff curves.
//!
//! A [`Falloff`] shapes the transition between fully-inside a region
//! (membership = 1) and fully-outside (membership = 0). Every shape
//! produces a signed distance to its surface; the falloff turns that
//! distance into a membership weight.
//!
//! Convention: `sd <= 0` means "inside the shape." A point at the
//! shape's surface (`sd == 0`) sits at the start of the transition.
//! A point at `sd == transition_m` sits at the outer edge of the
//! transition (membership = 0). `Hard` ignores `transition_m`.

/// Region transition curve. See module docs for the signed-distance
/// convention.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Falloff {
    /// Binary membership: `1` inside the shape, `0` everywhere else.
    /// Useful for gameplay triggers — no soft edge.
    Hard,
    /// Linear ramp from `1` at the shape's surface to `0` at
    /// `transition_m` outside. Slope is constant in the transition
    /// band.
    Linear {
        /// Width of the falloff band in metres. Must be `> 0`. Values
        /// `<= 0` collapse to `Hard` semantics (avoids divide-by-zero
        /// at runtime).
        transition_m: f32,
    },
    /// Smoothstep ramp from `1` at the surface to `0` at
    /// `transition_m` outside. Zero derivative at both endpoints — the
    /// natural choice for biome blending and audio cross-fades.
    Smoothstep {
        /// Width of the falloff band in metres. Same semantics as
        /// `Linear::transition_m`.
        transition_m: f32,
    },
}

impl Default for Falloff {
    fn default() -> Self {
        Self::Smoothstep { transition_m: 5.0 }
    }
}

impl Falloff {
    /// Width of the falloff band in metres. `Hard` returns `0`.
    pub fn transition_m(self) -> f32 {
        match self {
            Self::Hard => 0.0,
            Self::Linear { transition_m } | Self::Smoothstep { transition_m } => {
                transition_m.max(0.0)
            }
        }
    }

    /// Apply the curve to a signed distance to the shape's surface.
    ///
    /// `sd <= 0` (inside) returns `1`. `sd >= transition_m` returns
    /// `0`. Intermediate values follow the variant's curve.
    pub fn apply(self, sd: f32) -> f32 {
        if sd <= 0.0 {
            return 1.0;
        }
        let t_band = self.transition_m();
        if t_band <= 0.0 || sd >= t_band {
            return 0.0;
        }
        let t = sd / t_band;
        match self {
            Self::Hard => 0.0,
            Self::Linear { .. } => 1.0 - t,
            Self::Smoothstep { .. } => {
                // smoothstep(0, 1, 1-t) — zero derivative at both
                // endpoints so the boundary doesn't show a kink.
                let s = 1.0 - t;
                s * s * (3.0 - 2.0 * s)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hard_is_binary() {
        let f = Falloff::Hard;
        // Inside.
        assert_eq!(f.apply(-1.0), 1.0);
        assert_eq!(f.apply(0.0), 1.0);
        // Outside.
        assert_eq!(f.apply(0.001), 0.0);
        assert_eq!(f.apply(100.0), 0.0);
    }

    #[test]
    fn linear_ramps_uniformly() {
        let f = Falloff::Linear { transition_m: 10.0 };
        assert!((f.apply(0.0) - 1.0).abs() < 1e-6);
        assert!((f.apply(5.0) - 0.5).abs() < 1e-6);
        assert!((f.apply(10.0)).abs() < 1e-6);
        // Far outside saturates to 0.
        assert!((f.apply(50.0)).abs() < 1e-6);
        // Inside saturates to 1.
        assert!((f.apply(-1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn smoothstep_has_zero_derivative_at_endpoints() {
        // Approximate the derivative as a finite difference and check
        // it's ~0 at both ends of the band.
        let f = Falloff::Smoothstep { transition_m: 10.0 };
        let eps = 1e-3;
        // At sd = 0+ (interior edge of band).
        let v0 = f.apply(0.0);
        let v_eps = f.apply(eps);
        let slope_near_surface = (v0 - v_eps).abs() / eps;
        assert!(slope_near_surface < 0.05, "near-surface slope = {slope_near_surface}");
        // At sd = transition_m (outer edge of band).
        let v_inner = f.apply(10.0 - eps);
        let v_outer = f.apply(10.0);
        let slope_near_outer = (v_inner - v_outer).abs() / eps;
        assert!(slope_near_outer < 0.05, "near-outer slope = {slope_near_outer}");
    }

    #[test]
    fn smoothstep_passes_through_half_at_midpoint() {
        // 1 - smoothstep(0, 1, 0.5) = 1 - (0.5 * 0.5 * (3 - 1)) = 0.5
        let f = Falloff::Smoothstep { transition_m: 10.0 };
        assert!((f.apply(5.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn zero_transition_collapses_to_hard() {
        // Authors who type 0 shouldn't divide by zero — collapse to Hard.
        let f = Falloff::Linear { transition_m: 0.0 };
        assert_eq!(f.apply(-1.0), 1.0);
        assert_eq!(f.apply(0.0), 1.0);
        assert_eq!(f.apply(0.001), 0.0);
    }

    #[test]
    fn negative_transition_does_not_panic() {
        // Defensive against authoring bugs — saturates to 0 outside.
        let f = Falloff::Smoothstep { transition_m: -1.0 };
        assert_eq!(f.apply(0.0), 1.0);
        assert_eq!(f.apply(0.001), 0.0);
    }

    #[test]
    fn serde_roundtrips() {
        let f = Falloff::Smoothstep { transition_m: 20.0 };
        let json = serde_json::to_string(&f).unwrap();
        let back: Falloff = serde_json::from_str(&json).unwrap();
        assert_eq!(f, back);
    }
}
