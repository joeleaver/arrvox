//! Stamp lifecycle on the engine side — spawn, sync the
//! `Terrain.stamps` index from the live ECS, and invalidate dirty
//! tiles on add / move / delete.
//!
//! The pattern mirrors the existing terrain tile pipeline: commands
//! mutate ECS state, then a sync step reconciles the streamer's view
//! of the world. For stamps we explicitly call `sync_terrain_stamps`
//! after mutations rather than wiring an on_add/on_remove hook on
//! the registry entry — the hook fires from the command flush
//! without access to `EngineState`, and we need the full state to
//! call `streamer.invalidate_aabb` + evict.

use std::sync::Arc;

use arvx_core::Aabb;
use arvx_terrain::{
    FalloffCurve, ShapeNoise, Stamp, StampIndex, StampKind, Terrain,
};
use glam::{Vec2, Vec3};

use crate::command::StampKindSpec;
use crate::components::{EditorMetadata, Parent, Transform};

impl super::state::EngineState {
    /// Spawn a new Stamp ECS entity, sync the `Terrain.stamps` index,
    /// and invalidate every tile the new stamp's AABB touches.
    pub(crate) fn handle_spawn_stamp(
        &mut self,
        kind: StampKindSpec,
        position: Vec3,
    ) {
        let stamp = build_default_stamp(kind, position);
        let stamp_aabb = stamp.aabb();
        let kind_name_str = match stamp.kind {
            StampKind::Mountain { .. } => "Mountain",
            StampKind::Hill { .. } => "Hill",
            StampKind::Lake { .. } => "Lake",
            StampKind::Plateau { .. } => "Plateau",
            StampKind::Flatten { .. } => "Flatten",
        };

        // Terrain must exist — stamps are a Terrain feature.
        let Some(terrain_entity) = self.find_terrain_entity() else {
            self.console.warn(format!(
                "SpawnStamp({kind_name_str}) ignored — no Terrain in the scene. Spawn a Terrain first."
            ));
            return;
        };
        let terrain_uuid = self.get_entity_uuid(terrain_entity);

        // Spawn the stamp entity. Transform stores the world position
        // (terrain authors place stamps in world coords; the streamer
        // queries world-space too). Parent-link to the Terrain entity
        // so it appears under "Terrain ▸ Stamps" in the scene tree.
        let name = self.unique_name(&format!("{kind_name_str}"));
        let mut transform = Transform::default();
        transform.position = position;
        let entity = self.world.spawn((
            transform,
            EditorMetadata { name: name.clone() },
            stamp,
            Parent {
                parent_id: terrain_uuid,
            },
        ));
        self.assign_entity_uuid(entity);
        self.scene_dirty.mark_entity(entity);
        self.selected_entity = Some(entity);

        // Refresh the index + invalidate.
        self.sync_terrain_stamps_and_invalidate(Some(stamp_aabb));

        self.console.info(format!("Spawned '{name}' stamp"));
    }

    /// Find the Terrain ECS entity, if one exists. There is at most
    /// one — the `SpawnTerrain` handler enforces singleton. Public
    /// to the crate so other ops modules (region_ops, etc.) reuse
    /// this rather than re-declaring the same impl-fn (which Rust
    /// rejects as a duplicate definition).
    pub(crate) fn find_terrain_entity(&self) -> Option<hecs::Entity> {
        self.world
            .query::<&Terrain>()
            .iter()
            .next()
            .map(|(e, _)| e)
    }

    /// Re-collect every `Stamp` ECS component into a fresh
    /// `StampIndex` and install it on the Terrain. Optionally
    /// invalidates the streamer for an explicit AABB (the union of
    /// before/after for a move, or the new stamp's AABB for a spawn).
    ///
    /// Pass `None` for the AABB to rebuild the index without touching
    /// the streamer — useful when the caller has its own evictions
    /// to issue.
    pub(crate) fn sync_terrain_stamps_and_invalidate(
        &mut self,
        invalidate_aabb: Option<Aabb>,
    ) {
        let Some(terrain_entity) = self.find_terrain_entity() else {
            return;
        };
        // Collect every Stamp component's value. World position
        // already lives inside `Stamp.position`, which mirrors the
        // entity's Transform.position via the spawn / move handlers.
        let stamps: Vec<Stamp> = self
            .world
            .query::<&Stamp>()
            .iter()
            .map(|(_e, s)| *s)
            .collect();

        let new_index = Arc::new(StampIndex::from_stamps(stamps));
        if let Ok(mut t) = self.world.get::<&mut Terrain>(terrain_entity) {
            t.stamps = new_index;
        }

        if let Some(aabb) = invalidate_aabb {
            // Hot-swap: bounce intersecting Live tiles back to Queued,
            // but keep their `integrated_token` so the old entity +
            // asset stay resident in the scene. The deferred eviction
            // in `tick_terrain_streamer`'s integrate path releases the
            // predecessor pair once the fresh bake lands.
            //
            // Sculpt preservation: dirty tiles (in-RAM sculpt edits not
            // yet persisted to .arvxtile) are excluded from invalidation.
            // Re-baking them would drop the sculpt since the worker
            // bakes from `TerrainFn + stamps` (or the saved disk file),
            // neither of which carry the live sculpt diff. The tile
            // stays frozen at the sculpted state until the user
            // explicitly Reverts. Matches the Phase 4.3 edit-overlay
            // design in `docs/TERRAIN.md`.
            let Some(runtime) = self.terrain.as_mut() else { return };
            let dirty_count = runtime.dirty_tiles.len();
            runtime
                .streamer
                .invalidate_aabb_excluding(aabb, &runtime.dirty_tiles);
            if std::env::var("ARVX_TERRAIN_DEBUG").is_ok() {
                eprintln!(
                    "[stamp-sync] aabb=({:.1},{:.1},{:.1})..({:.1},{:.1},{:.1}) \
                     hot-swap queued (no synchronous evict), excluded {dirty_count} dirty tiles",
                    aabb.min.x, aabb.min.y, aabb.min.z,
                    aabb.max.x, aabb.max.y, aabb.max.z,
                );
            }
        }
    }

    /// Called after an entity's Transform changes. If the entity has
    /// a Stamp component, mirror the new world position into the
    /// Stamp, then resync + invalidate.
    pub(crate) fn maybe_sync_stamp_after_transform(
        &mut self,
        entity: hecs::Entity,
    ) {
        // Pull old + new world positions. The Transform has just been
        // updated by `SetObjectPosition`; the Stamp still holds the
        // pre-move position. Compute the union AABB for invalidation.
        let (old_aabb, new_aabb) = {
            let Ok(mut s) = self.world.get::<&mut Stamp>(entity) else {
                return;
            };
            let Ok(t) = self.world.get::<&Transform>(entity) else {
                return;
            };
            let before = s.aabb();
            s.position = t.position;
            let after = s.aabb();
            (before, after)
        };

        let union = Aabb {
            min: old_aabb.min.min(new_aabb.min),
            max: old_aabb.max.max(new_aabb.max),
        };
        self.sync_terrain_stamps_and_invalidate(Some(union));
    }

    /// Called before an entity is despawned. If the entity has a
    /// Stamp, capture its AABB so the post-delete sync can invalidate
    /// the right tiles.
    pub(crate) fn capture_stamp_aabb_before_delete(
        &self,
        entity: hecs::Entity,
    ) -> Option<Aabb> {
        self.world.get::<&Stamp>(entity).ok().map(|s| s.aabb())
    }
}

fn build_default_stamp(spec: StampKindSpec, position: Vec3) -> Stamp {
    // V2 defaults: each kind ships with non-zero values for the new
    // shape knobs so a freshly-spawned stamp looks organic rather
    // than geometric. Authors who want the V1 look set the knobs
    // back to zero in the Inspector.
    let (kind, shape_noise) = match spec {
        StampKindSpec::Mountain => {
            let radius = 30.0;
            (
                StampKind::Mountain {
                    h_max: 50.0,
                    radius,
                    falloff: FalloffCurve::Quadratic, // pointier than Smoothstep
                    aspect: 1.0,
                    ridge_strength: 0.30,
                    ridge_count: 3,
                },
                // Shape noise: ~10% of the radius, mid-frequency.
                ShapeNoise {
                    amp_m: radius * 0.10,
                    scale_m: radius * 0.6,
                    seed: position_seed(position, 0x6e_4e_5d_31),
                    octaves: 3,
                },
            )
        }
        StampKindSpec::Hill => {
            let radius = 15.0;
            (
                StampKind::Hill {
                    h_max: 10.0,
                    radius,
                    falloff: FalloffCurve::Smoothstep,
                    aspect: 1.0,
                    ridge_strength: 0.10, // gentler than Mountain
                    ridge_count: 2,
                },
                ShapeNoise {
                    amp_m: radius * 0.08,
                    scale_m: radius * 0.7,
                    seed: position_seed(position, 0x48_1a_3f_91),
                    octaves: 2,
                },
            )
        }
        StampKindSpec::Lake => {
            let radius = 20.0;
            (
                StampKind::Lake {
                    // Shallower than V2.0 (was 8 m). Default basins
                    // should read as ponds, not as quarries; the
                    // world-envelope clamp also catches anything
                    // that dips below the floor, but a gentle
                    // default avoids the clamp triggering on
                    // default-placed lakes.
                    depth: 6.0,
                    radius,
                    falloff: FalloffCurve::Smoothstep,
                    aspect: 1.0,
                    // Smaller than V2.0 (was 0.45). Walls now span
                    // 75% of the radius (15 m at the 20 m default)
                    // for the 6 m drop — about 22° average slope
                    // instead of V2.0's ~36°.
                    floor_flat_frac: 0.25,
                    // V2.2: 25% of radius (5 m at default) gives
                    // the lake a soft outer rim. Without this,
                    // lakes placed on hilly terrain showed a
                    // vertical cliff at the rim — SmoothMin clamped
                    // the surrounding (higher) terrain down to
                    // position.y in one voxel layer. The weight
                    // ramp lets the base show through near the rim
                    // so the cliff smooths into the hillside.
                    edge_falloff_m: radius * 0.25,
                },
                ShapeNoise {
                    // Smaller than V2.0 (was 0.12). 6% of radius is
                    // enough to read as natural without producing
                    // locally-cliff-y shorelines that compound the
                    // wall slope visually.
                    amp_m: radius * 0.06,
                    scale_m: radius * 0.5,
                    seed: position_seed(position, 0xa1_de_0c_27),
                    octaves: 3,
                },
            )
        }
        StampKindSpec::Plateau => {
            let half = Vec2::new(15.0, 15.0);
            (
                StampKind::Plateau {
                    half_extents: half,
                    corner_radius_m: half.min_element() * 0.20,
                    edge_falloff_m: half.min_element() * 0.25,
                    tilt: Vec2::ZERO,
                },
                ShapeNoise {
                    amp_m: half.min_element() * 0.05,
                    scale_m: half.min_element() * 0.5,
                    seed: position_seed(position, 0x37_28_b2_45),
                    octaves: 2,
                },
            )
        }
        StampKindSpec::Flatten => {
            let half = Vec2::new(10.0, 10.0);
            (
                StampKind::Flatten {
                    half_extents: half,
                    corner_radius_m: half.min_element() * 0.15,
                    edge_falloff_m: half.min_element() * 0.10, // crisper than Plateau
                    tilt: Vec2::ZERO,
                },
                // Flatten is for surveyed-flat-ground use cases —
                // keep the rim straight by default.
                ShapeNoise::default(),
            )
        }
    };
    let mut stamp = Stamp::new(kind, position);
    stamp.shape_noise = shape_noise;
    stamp
}

/// Hash a world position into a noise seed so two stamps placed at
/// different XZ get different shape noise patterns by default.
/// Authors can edit `shape_noise.seed` in the Inspector to taste.
fn position_seed(p: Vec3, salt: u32) -> u32 {
    let x = (p.x * 17.31).round() as i32 as u32;
    let z = (p.z * 23.97).round() as i32 as u32;
    salt.wrapping_add(x.wrapping_mul(0x9e37_79b9))
        .wrapping_add(z.wrapping_mul(0x85eb_ca77))
}

#[cfg(test)]
mod build_default_stamp_tests {
    use super::*;

    /// SpawnStamp(Plateau) must produce a stamp with the V2 default
    /// `edge_falloff_m` value (not zero). Regression test for the
    /// "edge falloff seems stuck at 0" report.
    #[test]
    fn plateau_default_has_nonzero_edge_falloff() {
        let s = build_default_stamp(StampKindSpec::Plateau, Vec3::ZERO);
        match s.kind {
            StampKind::Plateau { edge_falloff_m, corner_radius_m, .. } => {
                assert!(
                    edge_falloff_m > 0.0,
                    "Plateau default edge_falloff_m should be > 0; got {edge_falloff_m}",
                );
                assert!(
                    corner_radius_m > 0.0,
                    "Plateau default corner_radius_m should be > 0; got {corner_radius_m}",
                );
            }
            _ => panic!("expected Plateau"),
        }
    }

    /// Flatten ships with a smaller edge_falloff than Plateau (the
    /// "crisp ground survey" feel) but it must still be non-zero.
    #[test]
    fn flatten_default_has_nonzero_edge_falloff() {
        let s = build_default_stamp(StampKindSpec::Flatten, Vec3::ZERO);
        match s.kind {
            StampKind::Flatten { edge_falloff_m, .. } => {
                assert!(
                    edge_falloff_m > 0.0,
                    "Flatten default edge_falloff_m should be > 0; got {edge_falloff_m}",
                );
            }
            _ => panic!("expected Flatten"),
        }
    }
}
