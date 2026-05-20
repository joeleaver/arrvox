//! Policy traits for terrain physics (Phase 8 of `docs/TERRAIN.md`).
//!
//! The design is deliberately split into mechanism (this crate /
//! `arvx-engine`'s `play_mode/terrain_colliders.rs`) and policy
//! (these traits). The user can't predict who's standing on the
//! terrain — vehicles, NPCs, characters, sleeping rocks — so the
//! terrain system must not bake in assumptions about consumers.
//!
//! V1 ships **mechanism-only**: the engine always builds colliders
//! for every live tile, always rebuilds on re-bake, never
//! pre-materialises. The traits below + their default impls are
//! plumbed so a future consumer (vehicles needing predictive
//! materialisation, KCCs needing different rebuild cadence) can
//! swap in a custom policy without touching the engine.
//!
//! ## Traits
//!
//! * [`ColliderResidencyPolicy`] — which tile keys deserve a
//!   collider right now? Editor V1 default: every live tile.
//! * [`EditRebuildPolicy`] — when an edit dirties a tile, rebuild
//!   immediately, defer, or wait? Editor V1 default:
//!   `OnIntegrate` (rebuild as soon as the next bake completes).
//! * [`PredictiveMaterializationPolicy`] — given trajectory hints
//!   (velocity rays from fast-moving bodies), which not-yet-baked
//!   tiles should the streamer prioritise? Editor V1 default:
//!   no-op (vehicle teleportation = consumer's problem).
//!
//! These traits aren't wired to a settable hook in the engine yet —
//! they exist so a Phase 8b/V2 follow-up can lift the wired-in
//! defaults out without an API break.

use crate::tile_key::TileKey;
use arvx_core::Aabb;
use glam::Vec3;
use std::collections::HashSet;

/// Set of tile keys — used as both input and output of the policy
/// traits.
pub type TileSet = HashSet<TileKey>;

// ── ColliderResidencyPolicy ───────────────────────────────────────

/// What the engine knows about the world when asking residency.
pub struct ResidencyContext<'a> {
    /// Currently-live tile entities (by key). The candidate set the
    /// policy can return a subset of.
    pub live_tiles: &'a TileSet,
    /// Consumer-supplied centres of interest (cameras, bodies, AI
    /// agents) in world coords. The default policy expands these to
    /// a radius and selects intersecting tiles.
    pub interest_points: &'a [Vec3],
    /// Radius around each interest point that should have colliders.
    /// Default: 64 m (one tile-size).
    pub interest_radius_m: f32,
}

/// Decides which tiles should carry an active Rapier collider.
pub trait ColliderResidencyPolicy: Send + Sync {
    /// Return the subset of `ctx.live_tiles` that should be active.
    fn residency(&self, ctx: &ResidencyContext) -> TileSet;
}

/// Editor-default policy: every live tile gets a collider. Simple,
/// correct, and matches the design doc's V1 default ("editor uses a
/// simple radius around editor camera + any spawned bodies"). The
/// engine doesn't yet ask this trait — it just installs colliders
/// for every live tile directly — but the type exists so the
/// callsite can be swapped without breaking the API.
#[derive(Default, Clone, Copy)]
pub struct AlwaysResident;

impl ColliderResidencyPolicy for AlwaysResident {
    fn residency(&self, ctx: &ResidencyContext) -> TileSet {
        ctx.live_tiles.clone()
    }
}

/// Radius-based residency: a tile is resident if its AABB
/// intersects any interest point's radius. Cheaper to ship than
/// `AlwaysResident` once tile counts hit hundreds.
#[derive(Clone, Copy)]
pub struct RadiusResident {
    /// Used when `ResidencyContext.interest_radius_m <= 0`.
    pub default_radius_m: f32,
}

impl Default for RadiusResident {
    fn default() -> Self {
        Self {
            default_radius_m: 64.0,
        }
    }
}

impl ColliderResidencyPolicy for RadiusResident {
    fn residency(&self, ctx: &ResidencyContext) -> TileSet {
        let r = if ctx.interest_radius_m > 0.0 {
            ctx.interest_radius_m
        } else {
            self.default_radius_m
        };
        let r2 = r * r;
        ctx.live_tiles
            .iter()
            .copied()
            .filter(|key| {
                let aabb = tile_aabb(*key);
                ctx.interest_points
                    .iter()
                    .any(|p| aabb_point_distance_sq(&aabb, *p) <= r2)
            })
            .collect()
    }
}

// ── EditRebuildPolicy ─────────────────────────────────────────────

/// What the engine knows about a dirty tile when asking the rebuild
/// policy.
pub struct RebuildContext {
    /// Dirty region inside / containing the tile, in world coords.
    pub dirty_aabb: Aabb,
    /// Time since this tile's last rebuild (or 0 if never rebuilt).
    /// Lets a policy debounce: "no more than once per 50 ms during
    /// a drag-stroke."
    pub time_since_last_ms: f32,
    /// True when the source of the dirty event is "ongoing" (drag
    /// stroke, slider scrub) vs "atomic" (released, click). Editor
    /// V1 default rebuilds on atomic events; the in-game
    /// tunnelling system would rebuild on either.
    pub is_in_flight: bool,
}

/// Outcome of an [`EditRebuildPolicy`] decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RebuildDecision {
    /// Rebuild the collider now.
    Rebuild,
    /// Postpone the rebuild — keep the dirty record, ask again next
    /// tick.
    Wait,
    /// Skip the rebuild until the drag/stroke releases (the engine
    /// transitions `is_in_flight` to `false`).
    DeferredOnRelease,
}

/// Decides when a dirty tile's collider should be rebuilt.
pub trait EditRebuildPolicy: Send + Sync {
    fn decide(&self, ctx: &RebuildContext) -> RebuildDecision;
}

/// Editor-default policy: rebuild as soon as the next bake
/// completes. The streamer already takes ~ms to bake a tile;
/// rebuilding the collider in lockstep matches the user
/// expectation that the world they see is the world they collide
/// against.
#[derive(Default, Clone, Copy)]
pub struct OnIntegrate;

impl EditRebuildPolicy for OnIntegrate {
    fn decide(&self, _ctx: &RebuildContext) -> RebuildDecision {
        RebuildDecision::Rebuild
    }
}

/// Defer collider rebuild until a drag/stroke releases. Useful for
/// in-game tunnelling — saves visible-but-cheap collider rebuilds
/// during the drag, snaps to correctness on release.
#[derive(Default, Clone, Copy)]
pub struct OnStrokeRelease;

impl EditRebuildPolicy for OnStrokeRelease {
    fn decide(&self, ctx: &RebuildContext) -> RebuildDecision {
        if ctx.is_in_flight {
            RebuildDecision::DeferredOnRelease
        } else {
            RebuildDecision::Rebuild
        }
    }
}

// ── PredictiveMaterializationPolicy ───────────────────────────────

/// What the engine knows about not-yet-baked tiles when asking the
/// predictive policy. Velocity rays let a vehicle-scale system
/// pre-warm tiles down its travel path.
pub struct TrajectoryContext<'a> {
    /// (origin, direction, length) rays in world coords. Default
    /// editor doesn't supply any.
    pub velocity_rays: &'a [(Vec3, Vec3, f32)],
    /// Currently-unmaterialised tile keys (candidates to prioritise).
    pub candidates: &'a TileSet,
}

/// Decides which not-yet-baked tiles to prioritise.
pub trait PredictiveMaterializationPolicy: Send + Sync {
    fn prioritise(&self, ctx: &TrajectoryContext) -> TileSet;
}

/// Editor-default predictive policy: prioritise nothing. The
/// streamer's normal residency walk handles materialisation;
/// vehicles needing aggressive lookahead swap in a velocity-ray
/// impl.
#[derive(Default, Clone, Copy)]
pub struct NoPredictive;

impl PredictiveMaterializationPolicy for NoPredictive {
    fn prioritise(&self, _ctx: &TrajectoryContext) -> TileSet {
        TileSet::new()
    }
}

// ── helpers ───────────────────────────────────────────────────────

fn tile_aabb(key: TileKey) -> Aabb {
    let origin = key.origin_world().to_vec3();
    Aabb {
        min: origin,
        max: origin + Vec3::splat(key.extent_m()),
    }
}

fn aabb_point_distance_sq(aabb: &Aabb, p: Vec3) -> f32 {
    let cx = p.x.clamp(aabb.min.x, aabb.max.x);
    let cy = p.y.clamp(aabb.min.y, aabb.max.y);
    let cz = p.z.clamp(aabb.min.z, aabb.max.z);
    let dx = p.x - cx;
    let dy = p.y - cy;
    let dz = p.z - cz;
    dx * dx + dy * dy + dz * dz
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(x: i32, y: i32, z: i32) -> TileKey {
        TileKey::level0(x, y, z)
    }

    fn ctx_with<'a>(live: &'a TileSet, points: &'a [Vec3], r: f32) -> ResidencyContext<'a> {
        ResidencyContext {
            live_tiles: live,
            interest_points: points,
            interest_radius_m: r,
        }
    }

    #[test]
    fn always_resident_returns_input_set() {
        let live: TileSet = [key(0, 0, 0), key(1, 0, 0), key(0, 0, 1)]
            .into_iter()
            .collect();
        let ctx = ctx_with(&live, &[], 0.0);
        assert_eq!(AlwaysResident.residency(&ctx), live);
    }

    #[test]
    fn radius_resident_picks_nearby_only() {
        let live: TileSet = [key(0, 0, 0), key(10, 0, 0)].into_iter().collect();
        // tile (0,0,0) spans [0, 64); (10,0,0) spans [640, 704). A
        // point at (10, 0, 10) with radius 50 only reaches (0,0,0).
        let points = [Vec3::new(10.0, 0.0, 10.0)];
        let ctx = ctx_with(&live, &points, 50.0);
        let out = RadiusResident::default().residency(&ctx);
        assert!(out.contains(&key(0, 0, 0)));
        assert!(!out.contains(&key(10, 0, 0)));
    }

    #[test]
    fn radius_resident_uses_default_when_zero_radius() {
        let live: TileSet = [key(0, 0, 0), key(5, 0, 0)].into_iter().collect();
        // Default radius is 64 m. Point at the origin reaches tile
        // (0,0,0) but not (5,0,0) (centred at 320 m).
        let points = [Vec3::ZERO];
        let ctx = ctx_with(&live, &points, 0.0);
        let out = RadiusResident::default().residency(&ctx);
        assert_eq!(out.len(), 1);
        assert!(out.contains(&key(0, 0, 0)));
    }

    #[test]
    fn on_integrate_always_rebuilds() {
        let ctx = RebuildContext {
            dirty_aabb: Aabb {
                min: Vec3::ZERO,
                max: Vec3::splat(1.0),
            },
            time_since_last_ms: 5.0,
            is_in_flight: true,
        };
        assert_eq!(OnIntegrate.decide(&ctx), RebuildDecision::Rebuild);
    }

    #[test]
    fn on_stroke_release_waits_on_in_flight() {
        let ctx = RebuildContext {
            dirty_aabb: Aabb {
                min: Vec3::ZERO,
                max: Vec3::splat(1.0),
            },
            time_since_last_ms: 5.0,
            is_in_flight: true,
        };
        assert_eq!(
            OnStrokeRelease.decide(&ctx),
            RebuildDecision::DeferredOnRelease
        );
    }

    #[test]
    fn on_stroke_release_rebuilds_on_atomic() {
        let ctx = RebuildContext {
            dirty_aabb: Aabb {
                min: Vec3::ZERO,
                max: Vec3::splat(1.0),
            },
            time_since_last_ms: 5.0,
            is_in_flight: false,
        };
        assert_eq!(OnStrokeRelease.decide(&ctx), RebuildDecision::Rebuild);
    }

    #[test]
    fn no_predictive_returns_empty() {
        let candidates: TileSet = [key(0, 0, 0)].into_iter().collect();
        let ctx = TrajectoryContext {
            velocity_rays: &[],
            candidates: &candidates,
        };
        assert!(NoPredictive.prioritise(&ctx).is_empty());
    }
}
