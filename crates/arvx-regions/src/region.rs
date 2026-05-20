//! The `Region` ECS component — *where* a region is.
//!
//! Consumers attach their own data components alongside (e.g.
//! `BiomeRegion` in `arvx-terrain`, or future `AmbientAudio`,
//! `FogVolume`, `GameplayTrigger`, …). Each consumer queries the
//! [`crate::RegionIndex`] for its data-component type and gets back
//! `(entity, membership weight)` pairs.

use arvx_core::Aabb;
use glam::Vec3;

use crate::falloff::Falloff;
use crate::shape::RegionShape;

/// "What makes an entity a region." Carries the shape + falloff +
/// overlap-priority. Position lives on the entity's
/// `components::Transform` (consumed by [`membership`] as the
/// shape's centre); rotation also lives there but only OBB shapes
/// consume it.
///
/// Authors edit shape and falloff in the Inspector; the transform
/// gizmo manipulates `Transform.position` like any other entity.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct Region {
    /// Volume primitive — Sphere / Box / OBB in V1, Voxelized in V2.
    pub shape: RegionShape,
    /// Transition curve from the shape's surface outward.
    pub falloff: Falloff,
    /// Overlap priority for single-valued properties (e.g. "which
    /// biome's TerrainFn wins where they overlap"). Continuously-
    /// blendable properties (height contributions, audio gains) are
    /// blended by membership weight regardless of priority; this only
    /// kicks in when a consumer has to pick one.
    ///
    /// Higher numbers win. Ties resolved by the consumer (typically
    /// "first wins" in iteration order).
    pub priority: i32,
}

impl Region {
    /// Construct a region with `Hard` falloff at priority 0 — useful
    /// for binary gameplay triggers.
    pub fn hard(shape: RegionShape) -> Self {
        Self {
            shape,
            falloff: Falloff::Hard,
            priority: 0,
        }
    }

    /// World-space AABB tightly bounding the shape AND its falloff
    /// transition band. The [`crate::RegionIndex`] uses this to prune
    /// candidates before evaluating membership.
    pub fn world_aabb(&self, center: Vec3) -> Aabb {
        self.shape
            .world_aabb_with_falloff(center, self.falloff.transition_m())
    }
}

/// Soft membership of `point` in `region`, with the region centred at
/// `center`. Returns a weight in `[0, 1]`:
///
/// - `1.0` — well inside the shape.
/// - `0.0` — outside the shape AND outside the falloff transition.
/// - In between — within the transition band, shaped by the falloff
///   curve.
///
/// Cheap enough to call thousands of times per frame (per-leaf during
/// terrain voxelisation in Phase 7).
pub fn membership(region: &Region, center: Vec3, point: Vec3) -> f32 {
    let sd = region.shape.signed_distance(center, point);
    region.falloff.apply(sd)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    #[test]
    fn sphere_with_smoothstep_centre_is_one() {
        let r = Region {
            shape: RegionShape::Sphere { radius: 10.0 },
            falloff: Falloff::Smoothstep { transition_m: 5.0 },
            priority: 0,
        };
        let m = membership(&r, Vec3::ZERO, Vec3::ZERO);
        assert!((m - 1.0).abs() < 1e-6);
    }

    #[test]
    fn sphere_on_surface_is_one() {
        let r = Region {
            shape: RegionShape::Sphere { radius: 10.0 },
            falloff: Falloff::Smoothstep { transition_m: 5.0 },
            priority: 0,
        };
        // Exactly on the surface, sd == 0 → membership == 1.
        let m = membership(&r, Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0));
        assert!((m - 1.0).abs() < 1e-6);
    }

    #[test]
    fn sphere_at_outer_edge_of_band_is_zero() {
        let r = Region {
            shape: RegionShape::Sphere { radius: 10.0 },
            falloff: Falloff::Smoothstep { transition_m: 5.0 },
            priority: 0,
        };
        // 15 m from centre = 10 + 5 = end of band.
        let m = membership(&r, Vec3::ZERO, Vec3::new(15.0, 0.0, 0.0));
        assert!(m.abs() < 1e-6);
    }

    #[test]
    fn sphere_well_outside_is_zero() {
        let r = Region {
            shape: RegionShape::Sphere { radius: 10.0 },
            falloff: Falloff::Smoothstep { transition_m: 5.0 },
            priority: 0,
        };
        let m = membership(&r, Vec3::ZERO, Vec3::new(100.0, 0.0, 0.0));
        assert!(m.abs() < 1e-6);
    }

    #[test]
    fn sphere_midband_smoothstep_is_half() {
        // Smoothstep at the midpoint of the transition band = 0.5.
        let r = Region {
            shape: RegionShape::Sphere { radius: 10.0 },
            falloff: Falloff::Smoothstep { transition_m: 10.0 },
            priority: 0,
        };
        // 15 m from centre → sd = 5, halfway across the 10 m band.
        let m = membership(&r, Vec3::ZERO, Vec3::new(15.0, 0.0, 0.0));
        assert!((m - 0.5).abs() < 1e-5, "m={m}");
    }

    #[test]
    fn obb_membership_consistent_with_rotation() {
        // 90° rotation around Y swings the long axis from X to Z.
        let r = Region {
            shape: RegionShape::Obb {
                half_extents: Vec3::new(5.0, 0.5, 0.5),
                rotation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
            },
            falloff: Falloff::Hard,
            priority: 0,
        };
        let m_along_z = membership(&r, Vec3::ZERO, Vec3::new(0.0, 0.0, 4.0));
        assert!((m_along_z - 1.0).abs() < 1e-6);
        let m_along_x = membership(&r, Vec3::ZERO, Vec3::new(4.0, 0.0, 0.0));
        assert!(m_along_x.abs() < 1e-6);
    }

    #[test]
    fn world_aabb_includes_falloff_band() {
        let r = Region {
            shape: RegionShape::Sphere { radius: 10.0 },
            falloff: Falloff::Linear { transition_m: 3.0 },
            priority: 0,
        };
        let aabb = r.world_aabb(Vec3::new(100.0, 0.0, 0.0));
        // Centre x=100, total reach = radius+transition = 13.
        assert!((aabb.min.x - 87.0).abs() < 1e-4);
        assert!((aabb.max.x - 113.0).abs() < 1e-4);
    }

    #[test]
    fn serde_roundtrips_full_region() {
        let r = Region {
            shape: RegionShape::Obb {
                half_extents: Vec3::new(1.0, 2.0, 3.0),
                rotation: Quat::from_rotation_z(0.4),
            },
            falloff: Falloff::Smoothstep { transition_m: 7.5 },
            priority: 12,
        };
        let json = serde_json::to_string(&r).unwrap();
        let back: Region = serde_json::from_str(&json).unwrap();
        assert_eq!(r, back);
    }
}
