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
    FalloffCurve, Stamp, StampIndex, StampKind, Terrain,
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
    let kind = match spec {
        StampKindSpec::Mountain => StampKind::Mountain {
            h_max: 50.0,
            radius: 30.0,
            falloff: FalloffCurve::Smoothstep,
        },
        StampKindSpec::Hill => StampKind::Hill {
            h_max: 10.0,
            radius: 15.0,
            falloff: FalloffCurve::Smoothstep,
        },
        StampKindSpec::Lake => StampKind::Lake {
            depth: 8.0,
            radius: 20.0,
            falloff: FalloffCurve::Smoothstep,
        },
        StampKindSpec::Plateau => StampKind::Plateau {
            half_extents: Vec2::new(15.0, 15.0),
        },
        StampKindSpec::Flatten => StampKind::Flatten {
            half_extents: Vec2::new(10.0, 10.0),
        },
    };
    Stamp::new(kind, position)
}
