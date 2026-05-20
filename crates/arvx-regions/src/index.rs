//! [`RegionIndex`] — spatial index over the world's `Region` entities.
//!
//! The index is a snapshot of (entity, region, world-centre, world-AABB)
//! plus a [`crate::RegionBvh`] over the AABBs. Rebuild whenever regions
//! move / change / are added or removed — the design doc explicitly
//! calls out that regions are rare to mutate at runtime, so we don't
//! refit; we rebuild.
//!
//! The crate doesn't depend on `arvx-engine`, so the index can't reach
//! into `components::Transform` directly. The caller (the engine, or a
//! test) builds an [`RegionEntry`] list from whatever world it has and
//! hands it to [`RegionIndex::from_entries`]. Two filtered queries are
//! offered:
//!
//! * [`RegionIndex::query_all`] — every region containing the point,
//!   regardless of data components attached.
//! * [`RegionIndex::query`] — only regions whose ECS entity also
//!   carries the consumer's data component `D` (`BiomeRegion`,
//!   `AmbientAudio`, `FogVolume`, …).

use arvx_core::Aabb;
use glam::Vec3;

use crate::bvh::RegionBvh;
use crate::region::{membership, Region};

/// One snapshot row consumed by [`RegionIndex`].
///
/// `center` is the world position the region's shape is centred on —
/// typically `Transform.position` for the entity. `aabb` is
/// pre-computed (= `region.world_aabb(center)`) so the index doesn't
/// recompute on every rebuild.
#[derive(Debug, Clone, Copy)]
pub struct RegionEntry {
    /// ECS entity carrying this region.
    pub entity: hecs::Entity,
    /// Region component value (shape + falloff + priority).
    pub region: Region,
    /// World-space centre of the region's shape. Currently taken
    /// from `Transform.position`.
    pub center: Vec3,
    /// World-space AABB of the shape including the falloff band.
    /// Pre-computed so the BVH build path doesn't redo the work.
    pub aabb: Aabb,
}

impl RegionEntry {
    /// Build an entry from an entity / region / centre triple.
    pub fn new(entity: hecs::Entity, region: Region, center: Vec3) -> Self {
        let aabb = region.world_aabb(center);
        Self { entity, region, center, aabb }
    }
}

/// Spatial index over a snapshot of the world's regions.
///
/// Build once when the region set changes; query at any cadence after.
/// Cheap to drop and rebuild — the BVH is small at V1 scale.
#[derive(Debug, Clone, Default)]
pub struct RegionIndex {
    entries: Vec<RegionEntry>,
    aabbs: Vec<Aabb>,
    bvh: RegionBvh,
}

impl RegionIndex {
    /// Empty index — `query` returns nothing.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of regions in the index.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True if the index has no regions.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate the snapshot rows in insertion order.
    pub fn entries(&self) -> &[RegionEntry] {
        &self.entries
    }

    /// Build an index from a flat entry list. The order in `entries`
    /// is preserved for downstream `entries()` iteration; internally
    /// the BVH may permute its own index array.
    pub fn from_entries(entries: Vec<RegionEntry>) -> Self {
        let aabbs: Vec<Aabb> = entries.iter().map(|e| e.aabb).collect();
        let bvh = RegionBvh::build(&aabbs);
        Self { entries, aabbs, bvh }
    }

    /// Return every region whose AABB+falloff covers `point`, paired
    /// with its membership weight. Regions returning weight 0 (point
    /// is inside the AABB but outside the actual shape's falloff
    /// band) are filtered out — every entry in the result has weight
    /// `> 0`.
    pub fn query_all(&self, point: Vec3) -> Vec<(hecs::Entity, f32)> {
        let mut out = Vec::new();
        self.bvh.query_point(&self.aabbs, point, |i| {
            let e = &self.entries[i];
            let w = membership(&e.region, e.center, point);
            if w > 0.0 {
                out.push((e.entity, w));
            }
        });
        out
    }

    /// Visit each entry that contributes membership at `point` —
    /// callback receives `(entry_index, weight)`. The index is a
    /// position into [`Self::entries`], stable for the life of the
    /// `RegionIndex`.
    ///
    /// Lets consumer crates carry a parallel side table of typed
    /// per-region data (e.g. `arvx_terrain`'s `BiomeRegion`
    /// snapshot) and look up the payload by entry index without
    /// touching the world. This is the off-main-thread query path
    /// used by the terrain bake worker; the world-aware
    /// [`Self::query`] only works on the main thread where the
    /// `hecs::World` lives.
    pub fn query_indices<F: FnMut(usize, f32)>(&self, point: Vec3, mut visit: F) {
        self.bvh.query_point(&self.aabbs, point, |i| {
            let e = &self.entries[i];
            let w = membership(&e.region, e.center, point);
            if w > 0.0 {
                visit(i, w);
            }
        });
    }

    /// Like [`Self::query_all`] but only returns regions whose entity
    /// carries the data component `D`. Use this from consumer
    /// systems: terrain biomes pass `BiomeRegion`, an audio system
    /// would pass `AmbientAudio`, etc.
    ///
    /// The world reference is required because the index doesn't
    /// snapshot which data components each region carries — they can
    /// be added or removed after the index was built without
    /// invalidating it. (If a region's *shape* changes, you still
    /// need to rebuild.)
    pub fn query<D: hecs::Component>(
        &self,
        world: &hecs::World,
        point: Vec3,
    ) -> Vec<(hecs::Entity, f32)> {
        let mut out = Vec::new();
        self.bvh.query_point(&self.aabbs, point, |i| {
            let e = &self.entries[i];
            if world.get::<&D>(e.entity).is_err() {
                return;
            }
            let w = membership(&e.region, e.center, point);
            if w > 0.0 {
                out.push((e.entity, w));
            }
        });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::falloff::Falloff;
    use crate::shape::RegionShape;
    use hecs::World;

    fn sphere_region(radius: f32, transition: f32) -> Region {
        Region {
            shape: RegionShape::Sphere { radius },
            falloff: Falloff::Smoothstep { transition_m: transition },
            priority: 0,
        }
    }

    // Marker data component for the query<D>() tests. Region crate
    // can't depend on consumers, so we define our own in-test marker.
    struct MarkerA;
    struct MarkerB;

    #[test]
    fn empty_index_returns_nothing() {
        let i = RegionIndex::new();
        assert!(i.is_empty());
        let w = World::new();
        let r = i.query::<MarkerA>(&w, Vec3::ZERO);
        assert!(r.is_empty());
    }

    #[test]
    fn query_all_returns_weight_one_at_centre() {
        let mut w = World::new();
        let r = sphere_region(10.0, 5.0);
        let e = w.spawn((r,));
        let idx = RegionIndex::from_entries(vec![RegionEntry::new(e, r, Vec3::ZERO)]);
        let hits = idx.query_all(Vec3::ZERO);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, e);
        assert!((hits[0].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn query_filters_by_data_component() {
        // Two regions at the same point. Only one carries MarkerA;
        // only that one comes back when we query for MarkerA.
        let mut w = World::new();
        let r = sphere_region(10.0, 5.0);
        let e_with = w.spawn((r, MarkerA));
        let e_without = w.spawn((r,));

        let idx = RegionIndex::from_entries(vec![
            RegionEntry::new(e_with, r, Vec3::ZERO),
            RegionEntry::new(e_without, r, Vec3::ZERO),
        ]);

        let hits = idx.query::<MarkerA>(&w, Vec3::ZERO);
        let entities: Vec<_> = hits.iter().map(|(e, _)| *e).collect();
        assert_eq!(entities, vec![e_with]);

        // The other marker matches the unmarked region zero times.
        let hits_b = idx.query::<MarkerB>(&w, Vec3::ZERO);
        assert!(hits_b.is_empty());
    }

    #[test]
    fn weight_zero_entries_dropped() {
        // Point sits inside the AABB (within the falloff band) for
        // the BVH to deliver a candidate, but at the very edge of the
        // band — membership lands at 0 exactly.
        let mut w = World::new();
        let r = sphere_region(10.0, 5.0);
        let e = w.spawn((r,));
        let idx = RegionIndex::from_entries(vec![RegionEntry::new(e, r, Vec3::ZERO)]);

        // 15 m from centre = outer edge of falloff band.
        let hits = idx.query_all(Vec3::new(15.0, 0.0, 0.0));
        assert!(hits.is_empty(), "expected drop at weight-0 boundary; got {hits:?}");
    }

    #[test]
    fn multiple_overlapping_regions_all_reported() {
        let mut w = World::new();
        let small = sphere_region(5.0, 2.0);
        let large = sphere_region(20.0, 5.0);
        let e_small = w.spawn((small,));
        let e_large = w.spawn((large,));
        let idx = RegionIndex::from_entries(vec![
            RegionEntry::new(e_small, small, Vec3::ZERO),
            RegionEntry::new(e_large, large, Vec3::ZERO),
        ]);
        let hits = idx.query_all(Vec3::ZERO);
        let mut ents: Vec<_> = hits.iter().map(|(e, _)| *e).collect();
        ents.sort_by_key(|e| e.id());
        let mut want = vec![e_small, e_large];
        want.sort_by_key(|e| e.id());
        assert_eq!(ents, want);
    }

    #[test]
    fn point_outside_returns_empty() {
        let mut w = World::new();
        let r = sphere_region(10.0, 5.0);
        let e = w.spawn((r,));
        let idx = RegionIndex::from_entries(vec![RegionEntry::new(e, r, Vec3::ZERO)]);
        let hits = idx.query_all(Vec3::new(1000.0, 0.0, 0.0));
        assert!(hits.is_empty());
    }

    #[test]
    fn query_indices_yields_entry_positions() {
        // Two regions at separated centres so the BVH actually
        // branches. Point inside only the second one yields entry
        // index 1.
        let mut w = World::new();
        let r = sphere_region(2.0, 1.0);
        let a = w.spawn((r,));
        let b = w.spawn((r,));
        let idx = RegionIndex::from_entries(vec![
            RegionEntry::new(a, r, Vec3::new(0.0, 0.0, 0.0)),
            RegionEntry::new(b, r, Vec3::new(100.0, 0.0, 0.0)),
        ]);
        let mut hits: Vec<(usize, f32)> = Vec::new();
        idx.query_indices(Vec3::new(100.0, 0.0, 0.0), |i, w| hits.push((i, w)));
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 1);
        assert!((hits[0].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn query_indices_no_world_required() {
        // query_indices doesn't take &World — proves the off-main-
        // thread path works.
        let mut w = World::new();
        let r = sphere_region(5.0, 1.0);
        let e = w.spawn((r,));
        let idx = RegionIndex::from_entries(vec![RegionEntry::new(e, r, Vec3::ZERO)]);
        // Move the world out of scope to make the point.
        drop(w);
        let mut hit = false;
        idx.query_indices(Vec3::ZERO, |_, _| hit = true);
        assert!(hit);
    }

    #[test]
    fn entries_preserve_input_order() {
        // Same entity / region pattern, two entries at different
        // positions. The Vec<RegionEntry> returned by entries() must
        // match the input order even though the BVH builds a sorted
        // permutation internally.
        let mut w = World::new();
        let r = sphere_region(1.0, 0.5);
        let a = w.spawn((r,));
        let b = w.spawn((r,));
        let idx = RegionIndex::from_entries(vec![
            RegionEntry::new(a, r, Vec3::new(0.0, 0.0, 0.0)),
            RegionEntry::new(b, r, Vec3::new(50.0, 0.0, 0.0)),
        ]);
        let entries = idx.entries();
        assert_eq!(entries[0].entity, a);
        assert_eq!(entries[1].entity, b);
    }
}
