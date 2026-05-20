//! Per-tile Rapier `TriMesh` colliders, owned by [`PlayModeState`].
//!
//! Phase 8 of `docs/TERRAIN.md`. The terrain bake produces a per-tile
//! [`arvx_terrain::TileColliderMesh`] at integration time; this
//! module turns those snapshots into Rapier static-body + TriMesh-
//! collider pairs and tracks them by `TileKey` so subsequent re-bakes
//! can replace the old collider.
//!
//! ## Lifecycle
//!
//! * Play mode starts → [`TerrainColliders::build_initial_from_world`]
//!   walks every live `(TerrainTile, TileColliderMesh)` entity and
//!   builds its collider.
//! * A tile re-bakes while play is active (sculpt / stamp move /
//!   region edit drives `integrate_terrain_tile`) → the engine calls
//!   [`Self::on_tile_added`] for the new tile. If a collider for the
//!   same key already existed it's removed first and any sleeping
//!   bodies inside the tile's AABB are woken — they may now be
//!   intersecting fresh geometry.
//! * A tile is evicted → [`Self::on_tile_evicted`] drops the collider.
//! * Play mode ends → `PlayModeState` is dropped, which drops the
//!   `TerrainColliders` and the Rapier world that owned the handles.
//!
//! Out of scope for V1: residency policy (we collide every live
//! tile), the predictive materialisation policy (no-op), origin
//! rebase (V2).

use std::collections::HashMap;

use arvx_core::Aabb;
use arvx_physics::rapier_world::{to_rapier_vec3, PhysicsWorld};
use arvx_terrain::{TerrainTile, TileColliderMesh, TileKey};
use glam::Vec3;
use rapier3d::prelude::*;

/// Tracks one Rapier static body + its TriMesh collider per terrain
/// tile.
pub(crate) struct TerrainColliders {
    by_key: HashMap<TileKey, RigidBodyHandle>,
    /// Friction applied to every terrain collider. Surfaced so we can
    /// surface a Terrain Inspector knob later without touching this
    /// module's call sites.
    friction: f32,
    /// Restitution (bounciness) applied to every terrain collider.
    /// V1 default: 0.0 (terrain doesn't bounce balls back).
    restitution: f32,
}

impl TerrainColliders {
    /// Fresh empty map — no colliders yet.
    pub(crate) fn new() -> Self {
        Self {
            by_key: HashMap::new(),
            friction: 0.8,
            restitution: 0.0,
        }
    }

    /// Walk every live `(TerrainTile, TileColliderMesh)` entity in the
    /// world and build its collider. Idempotent: replaces any
    /// existing collider for a key.
    pub(crate) fn build_initial_from_world(
        &mut self,
        physics: &mut PhysicsWorld,
        world: &hecs::World,
    ) {
        for (_entity, (tile, mesh)) in world.query::<(&TerrainTile, &TileColliderMesh)>().iter() {
            self.install(physics, tile.key, mesh);
        }
    }

    /// Called when the engine integrates a new (or re-baked) tile.
    /// Builds the new collider; if one already existed for this key,
    /// removes it first and wakes sleeping bodies inside the
    /// rebuilt-tile AABB — they may now be intersecting fresh
    /// geometry.
    pub(crate) fn on_tile_added(
        &mut self,
        physics: &mut PhysicsWorld,
        world: &hecs::World,
        entity: hecs::Entity,
        key: TileKey,
        tile_world_aabb: Aabb,
    ) {
        // Fetch the snapshot the engine just attached to the entity.
        // If somehow missing (e.g. the bake produced no surface), drop
        // any pre-existing collider for this key and bail.
        let Ok(mesh_guard) = world.get::<&TileColliderMesh>(entity) else {
            self.on_tile_evicted(physics, key);
            return;
        };
        let mesh = (*mesh_guard).clone();
        drop(mesh_guard);

        // If a collider for this key already existed, this is a
        // re-bake — wake sleeping bodies within the tile's AABB
        // BEFORE removing the old collider so the awoken bodies
        // re-evaluate against the (about-to-be) new geometry.
        let is_rebake = self.by_key.contains_key(&key);
        if is_rebake {
            wake_bodies_in_aabb(physics, tile_world_aabb);
            self.remove_for_key(physics, key);
        }

        self.install(physics, key, &mesh);
    }

    /// Called when a tile is evicted (camera moved past the residency
    /// radius, scene cleared, etc.). Drops the collider without
    /// waking bodies — a vanished tile means there's no new geometry
    /// to react to, only the absence of the old surface (handled by
    /// the physics step naturally).
    pub(crate) fn on_tile_evicted(&mut self, physics: &mut PhysicsWorld, key: TileKey) {
        self.remove_for_key(physics, key);
    }

    /// Number of tiles with a live collider — surfaced for tests and
    /// debug overlays.
    pub(crate) fn len(&self) -> usize {
        self.by_key.len()
    }

    // ── internals ──────────────────────────────────────────────────

    /// Build + insert the Rapier body/collider pair for `mesh` at
    /// `key`'s world origin. Skipped silently when the mesh has no
    /// triangles (degenerate tile).
    fn install(&mut self, physics: &mut PhysicsWorld, key: TileKey, mesh: &TileColliderMesh) {
        if mesh.is_empty() {
            return;
        }

        // Translate tile-local positions into world space. The TriMesh
        // could also live in body-local space + the rigid body could
        // carry the translation, but storing world positions makes
        // ray/contact queries against the collider less surprising
        // when debugging.
        let origin = key.origin_world().to_vec3();
        let points: Vec<rapier3d::math::Vector> = mesh
            .vertices
            .iter()
            .map(|v| to_rapier_vec3(*v + origin))
            .collect();

        let body = RigidBodyBuilder::fixed().build();
        let body_handle = physics.add_rigid_body(body);
        let collider = ColliderBuilder::trimesh(points, mesh.triangles.clone())
            .expect("TriMesh build failed — verified non-empty above")
            .friction(self.friction)
            .restitution(self.restitution)
            .build();
        let _collider_handle = physics.add_collider(collider, body_handle);

        // Replace any stale entry for this key. The caller of
        // `on_tile_added` already issued a `remove_for_key` on the
        // re-bake path; this `insert` is the per-key invariant.
        self.by_key.insert(key, body_handle);
    }

    fn remove_for_key(&mut self, physics: &mut PhysicsWorld, key: TileKey) {
        if let Some(handle) = self.by_key.remove(&key) {
            let _ = physics.remove_rigid_body(handle);
        }
    }
}

/// Wake every sleeping dynamic body whose AABB overlaps `aabb`. Static
/// and kinematic bodies are ignored (they can't sleep). The check is
/// cheap (linear scan over the body set + AABB-AABB overlap); typical
/// scenes have a handful of dynamic bodies, not thousands.
///
/// We wake on a *centre-point* check rather than a true AABB overlap:
/// each body's translation + a generous radius (1 m) is the activity
/// zone we want to perturb. Sleeping bodies sitting *on* the rebuilt
/// tile would have their support face changed; bodies floating
/// alongside likely don't care. Erring on the side of waking more
/// than needed is correct — over-waking is a perf nit; missing a wake
/// is a correctness bug ("body fell through new geometry").
fn wake_bodies_in_aabb(physics: &mut PhysicsWorld, aabb: Aabb) {
    // 1 m wake radius around the AABB — covers contact points at the
    // tile boundary that might otherwise just miss the strict
    // intersection check.
    let pad = 1.0;
    let lo = Vec3::new(aabb.min.x - pad, aabb.min.y - pad, aabb.min.z - pad);
    let hi = Vec3::new(aabb.max.x + pad, aabb.max.y + pad, aabb.max.z + pad);

    // Collect handles first; mutating the body set while iterating
    // would require an unsafe borrow split.
    let mut to_wake: Vec<RigidBodyHandle> = Vec::new();
    for (handle, body) in physics.rigid_body_set.iter() {
        if !body.is_sleeping() {
            continue;
        }
        let t = body.translation();
        let p = Vec3::new(t.x, t.y, t.z);
        if p.x >= lo.x && p.x <= hi.x && p.y >= lo.y && p.y <= hi.y && p.z >= lo.z && p.z <= hi.z {
            to_wake.push(handle);
        }
    }
    for handle in to_wake {
        if let Some(body) = physics.rigid_body_set.get_mut(handle) {
            body.wake_up(true);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arvx_physics::rapier_world::PhysicsConfig;
    use glam::Vec3;

    fn two_tri_mesh() -> TileColliderMesh {
        TileColliderMesh {
            vertices: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(1.0, 0.0, 1.0),
            ],
            triangles: vec![[0, 1, 2], [1, 3, 2]],
        }
    }

    fn empty_mesh() -> TileColliderMesh {
        TileColliderMesh {
            vertices: Vec::new(),
            triangles: Vec::new(),
        }
    }

    #[test]
    fn empty_mesh_inserts_no_collider() {
        let mut tc = TerrainColliders::new();
        let mut physics = PhysicsWorld::new(PhysicsConfig::default());
        tc.install(&mut physics, TileKey::level0(0, 0, 0), &empty_mesh());
        assert_eq!(tc.len(), 0);
        assert_eq!(physics.body_count(), 0);
    }

    #[test]
    fn install_creates_one_body_per_tile() {
        let mut tc = TerrainColliders::new();
        let mut physics = PhysicsWorld::new(PhysicsConfig::default());
        tc.install(&mut physics, TileKey::level0(0, 0, 0), &two_tri_mesh());
        tc.install(&mut physics, TileKey::level0(1, 0, 0), &two_tri_mesh());
        assert_eq!(tc.len(), 2);
        assert_eq!(physics.body_count(), 2);
    }

    #[test]
    fn evict_removes_body() {
        let mut tc = TerrainColliders::new();
        let mut physics = PhysicsWorld::new(PhysicsConfig::default());
        let key = TileKey::level0(0, 0, 0);
        tc.install(&mut physics, key, &two_tri_mesh());
        assert_eq!(tc.len(), 1);
        tc.on_tile_evicted(&mut physics, key);
        assert_eq!(tc.len(), 0);
        assert_eq!(physics.body_count(), 0);
    }

    #[test]
    fn evicting_unknown_key_is_a_noop() {
        let mut tc = TerrainColliders::new();
        let mut physics = PhysicsWorld::new(PhysicsConfig::default());
        // Key that was never installed.
        tc.on_tile_evicted(&mut physics, TileKey::level0(99, 99, 99));
        assert_eq!(tc.len(), 0);
    }

    #[test]
    fn wake_bodies_skips_active_bodies() {
        // Bodies that aren't sleeping shouldn't be touched. Walk the
        // wake function with an awake body and verify nothing
        // changes (no panic, no spurious sleep change).
        let mut physics = PhysicsWorld::new(PhysicsConfig::default());
        let body = RigidBodyBuilder::dynamic()
            .translation(to_rapier_vec3(Vec3::new(0.0, 5.0, 0.0)))
            .build();
        let handle = physics.add_rigid_body(body);

        wake_bodies_in_aabb(
            &mut physics,
            Aabb {
                min: Vec3::new(-10.0, 0.0, -10.0),
                max: Vec3::new(10.0, 10.0, 10.0),
            },
        );

        let body = physics.get_body(handle).unwrap();
        assert!(!body.is_sleeping(), "body remained awake");
    }
}
