//! Region lifecycle on the engine side — spawn, sync, invalidate.
//!
//! Regions are cross-cutting: any consumer (terrain biomes,
//! audio, fog, triggers) attaches its own data component beside the
//! [`arvx_regions::Region`] shape. Phase 7 hooks the terrain-side
//! [`arvx_terrain::TerrainRegionSnapshot`] up to the live ECS:
//! whenever a Region or BiomeRegion changes shape, position, or
//! data, the engine rebuilds the snapshot on the Terrain entity and
//! invalidates the tiles whose AABB intersected the change.
//!
//! The pattern mirrors `stamp_ops` exactly — see the docs there for
//! the rationale (no on_add/on_remove registry hook; the engine
//! state needs `streamer.invalidate_aabb` access that those hooks
//! don't have).

use std::sync::Arc;

use arvx_core::Aabb;
use arvx_regions::{Falloff, Region, RegionEntry, RegionIndex, RegionShape};
use arvx_terrain::{BiomeRegion, Terrain, TerrainRegionSnapshot};
use glam::{Quat, Vec3};

use crate::command::RegionShapeSpec;
use crate::components::{EditorMetadata, Transform};

impl super::state::EngineState {
    /// Spawn a Region entity with the given shape at the supplied
    /// world position. Selects the new entity (consistent with every
    /// other Spawn handler) so the gizmo and Inspector light up
    /// immediately. Rebuilds the terrain-side region snapshot and
    /// dirties any tiles the new region's AABB touches.
    pub(crate) fn handle_spawn_region(&mut self, shape_spec: RegionShapeSpec, position: Vec3) {
        let (shape, label) = build_default_shape(shape_spec);
        let region = Region {
            shape,
            falloff: Falloff::Smoothstep { transition_m: 5.0 },
            priority: 0,
        };
        let region_aabb = region.world_aabb(position);

        let name = self.unique_name(&format!("{label} Region"));
        let mut transform = Transform::default();
        transform.position = position;
        let entity = self.world.spawn((
            transform,
            EditorMetadata { name: name.clone() },
            region,
        ));
        self.assign_entity_uuid(entity);
        self.scene_dirty.mark_entity(entity);
        self.selected_entity = Some(entity);

        // Refresh the snapshot + invalidate. No-op when no Terrain.
        self.sync_terrain_regions_and_invalidate(Some(region_aabb));

        self.console.info(format!("Spawned '{name}'"));
    }

    /// Re-collect every `(Region, Transform)` pair (with optional
    /// `BiomeRegion`) into a fresh [`TerrainRegionSnapshot`] and
    /// install it on the Terrain. Optionally invalidates tiles
    /// intersecting `aabb`.
    ///
    /// No-op when no Terrain is in the scene — Regions are
    /// cross-cutting and meaningful without one, so unlike
    /// `SpawnStamp` we silently skip rather than warning.
    pub(crate) fn sync_terrain_regions_and_invalidate(
        &mut self,
        invalidate_aabb: Option<Aabb>,
    ) {
        let Some(terrain_entity) = self.find_terrain_entity() else {
            return;
        };

        // Snapshot every Region + its world position + its optional
        // BiomeRegion. Collect into two parallel Vecs so the
        // RegionIndex's BVH and the side table stay aligned.
        let mut entries: Vec<RegionEntry> = Vec::new();
        let mut biomes: Vec<Option<BiomeRegion>> = Vec::new();
        for (entity, (transform, region)) in self.world.query::<(&Transform, &Region)>().iter() {
            entries.push(RegionEntry::new(entity, *region, transform.position));
            biomes.push(
                self.world
                    .get::<&BiomeRegion>(entity)
                    .ok()
                    .map(|b| (*b).clone()),
            );
        }

        let snapshot = Arc::new(TerrainRegionSnapshot {
            index: Arc::new(RegionIndex::from_entries(entries)),
            biomes: Arc::new(biomes),
        });

        if let Ok(mut t) = self.world.get::<&mut Terrain>(terrain_entity) {
            t.regions = snapshot;
        }

        if let Some(aabb) = invalidate_aabb {
            let Some(runtime) = self.terrain.as_mut() else { return };
            let evictions = runtime.streamer.invalidate_aabb(aabb);
            if !evictions.is_empty() {
                self.evict_terrain_tiles_for_region_change(&evictions);
                self.gpu_objects_dirty.mark_all();
            }
        }
    }

    /// Eviction pass shared by the region-invalidation path. Cloned
    /// from `stamp_ops::evict_terrain_tiles_for_stamp_change`; the
    /// two could be unified into a generic helper but the duplication
    /// is small and keeps lifecycle hooks straightforward to follow.
    fn evict_terrain_tiles_for_region_change(
        &mut self,
        evictions: &[(arvx_terrain::TileKey, u64)],
    ) {
        let Some(runtime) = self.terrain.as_mut() else { return };
        let mut to_drop: Vec<(hecs::Entity, arvx_render::AssetHandle)> = Vec::new();
        for (key, token) in evictions {
            runtime.tile_keys.remove(key);
            if let Some(pair) = runtime.live_tiles.remove(token) {
                to_drop.push(pair);
            }
        }
        for (entity, asset_handle) in to_drop {
            let _ = self.world.despawn(entity);
            self.sculpt_overlays.remove(&entity);
            let mut sm = match self.scene_mgr.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            sm.release_asset(asset_handle);
        }
    }

    /// Called after an entity's Transform changes. If the entity has
    /// a Region component, compute the union of (old AABB ∪ new AABB)
    /// and resync + invalidate. Mirrors
    /// `maybe_sync_stamp_after_transform`.
    pub(crate) fn maybe_sync_region_after_transform(&mut self, entity: hecs::Entity) {
        let Some((old_aabb, new_aabb)) = ({
            let region = self.world.get::<&Region>(entity).ok().map(|r| *r);
            let transform = self.world.get::<&Transform>(entity).ok().map(|t| t.position);
            region.zip(transform).map(|(r, pos)| {
                // The Region itself stores no position — its centre is
                // the entity's Transform.position. So the "old AABB"
                // for an in-flight gizmo drag is unknowable from the
                // Region alone. We fall back to using the snapshot's
                // previous AABB if the entity is in the live snapshot.
                let new_aabb = r.world_aabb(pos);
                let old_aabb = self
                    .lookup_region_snapshot_aabb(entity)
                    .unwrap_or(new_aabb);
                (old_aabb, new_aabb)
            })
        }) else {
            return;
        };

        let union = Aabb {
            min: old_aabb.min.min(new_aabb.min),
            max: old_aabb.max.max(new_aabb.max),
        };
        self.sync_terrain_regions_and_invalidate(Some(union));
    }

    /// Look up the previously-snapshot AABB for an entity. Returns
    /// `None` if the entity wasn't in the last snapshot (newly
    /// spawned, or no Terrain).
    fn lookup_region_snapshot_aabb(&self, entity: hecs::Entity) -> Option<Aabb> {
        let terrain_entity = self.find_terrain_entity()?;
        let terrain = self.world.get::<&Terrain>(terrain_entity).ok()?;
        terrain
            .regions
            .index
            .entries()
            .iter()
            .find(|e| e.entity == entity)
            .map(|e| e.aabb)
    }

    /// Called before an entity is despawned. If the entity has a
    /// Region, capture its AABB so the post-delete sync can
    /// invalidate the right tiles. Mirrors
    /// `capture_stamp_aabb_before_delete`.
    pub(crate) fn capture_region_aabb_before_delete(
        &self,
        entity: hecs::Entity,
    ) -> Option<Aabb> {
        let region = self.world.get::<&Region>(entity).ok()?;
        let transform = self.world.get::<&Transform>(entity).ok()?;
        Some(region.world_aabb(transform.position))
    }

    /// Resync regions, invalidating the AABB of the supplied entity's
    /// Region. No-op if the entity has no Region, or no Terrain is
    /// in the scene. Used by the AddComponent flow (a fresh
    /// BiomeRegion changes what the bake sees for the host region).
    pub(crate) fn sync_terrain_regions_for_entity(&mut self, entity: hecs::Entity) {
        let Some(aabb) = self.capture_region_aabb_before_delete(entity) else {
            return;
        };
        self.sync_terrain_regions_and_invalidate(Some(aabb));
    }
}

fn build_default_shape(spec: RegionShapeSpec) -> (RegionShape, &'static str) {
    match spec {
        RegionShapeSpec::Sphere => (RegionShape::Sphere { radius: 25.0 }, "Sphere"),
        RegionShapeSpec::Box => (
            RegionShape::Box {
                half_extents: Vec3::new(15.0, 15.0, 15.0),
            },
            "Box",
        ),
        RegionShapeSpec::Obb => (
            RegionShape::Obb {
                half_extents: Vec3::new(15.0, 15.0, 15.0),
                rotation: Quat::IDENTITY,
            },
            "OBB",
        ),
    }
}
