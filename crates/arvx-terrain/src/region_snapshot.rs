//! Per-terrain snapshot of regions + per-entry BiomeRegion data.
//!
//! The terrain bake worker runs off the main thread and can't touch
//! the `hecs::World`. `arvx_regions::RegionIndex` exposes
//! [`arvx_regions::RegionIndex::query_indices`] so consumers can walk
//! the BVH without `&World`; we pair the index with a parallel
//! `Vec<Option<BiomeRegion>>` (aligned to `index.entries()`) so the
//! bake can look up biome data by index in the same scan.
//!
//! This is the [`crate::stamp_index::StampIndexHandle`] of regions:
//! a snapshot allocated when the region set changes, shared cheaply
//! by `Arc` to in-flight bake jobs. Writers always allocate a fresh
//! snapshot rather than mutating the live one — keeps workers'
//! views internally consistent without locks.
//!
//! ## Material override semantics (V1)
//!
//! Phase 7 consumes only `BiomeRegion.material_override`. Overlap
//! resolution: the highest-priority region whose membership weight
//! is positive AND that carries a material_override wins, full
//! override, no blend. Ties resolve in entry order (first wins) —
//! matches the stamp pattern.

use std::sync::Arc;

use arvx_regions::RegionIndex;
use glam::Vec3;

use crate::biome_region::BiomeRegion;

/// `Arc<RegionIndex>` plus a parallel side table of optional
/// per-entry [`BiomeRegion`] data. Empty (`new()`) when there are no
/// regions in the scene; non-empty otherwise.
#[derive(Debug, Clone, Default)]
pub struct TerrainRegionSnapshot {
    /// BVH-backed spatial index over the world's regions. Built by
    /// `arvx-engine` from `(Region, Transform)` queries.
    pub index: Arc<RegionIndex>,
    /// Parallel to `index.entries()`. Slot `i` is `Some(b)` when the
    /// region at `index.entries()[i]` carries a `BiomeRegion`,
    /// `None` otherwise.
    pub biomes: Arc<Vec<Option<BiomeRegion>>>,
}

impl TerrainRegionSnapshot {
    /// Empty snapshot — no regions, no biomes. The shape the bake
    /// path sees when no `Region` entities exist in the scene.
    pub fn new() -> Self {
        Self {
            index: Arc::new(RegionIndex::new()),
            biomes: Arc::new(Vec::new()),
        }
    }

    /// Number of regions in the snapshot.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// True when no regions are indexed.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Walk all biome regions whose membership at `point` is positive,
    /// in entry order. Callback receives `(biome, weight)`.
    ///
    /// Regions without a `BiomeRegion` data component are skipped —
    /// they exist in the spatial index for other consumers (audio,
    /// fog, triggers — Phase 8+) but the terrain bake doesn't care.
    pub fn query_biome<F: FnMut(&BiomeRegion, f32)>(&self, point: Vec3, mut visit: F) {
        let biomes = self.biomes.as_ref();
        self.index.query_indices(point, |i, w| {
            if let Some(biome) = biomes.get(i).and_then(|b| b.as_ref()) {
                visit(biome, w);
            }
        });
    }

    /// Resolve the per-voxel material override at `point`, if any.
    ///
    /// V1 single-valued resolution: among overlapping biome regions
    /// with `Some(material_override)`, return the one whose
    /// `Region.priority` is highest; ties break by entry order (the
    /// first one encountered wins). Returns `None` when no
    /// overlapping biome region overrides the material at this
    /// point — caller keeps the base TerrainFn / stamp material.
    pub fn material_override_at(&self, point: Vec3) -> Option<u16> {
        let biomes = self.biomes.as_ref();
        let entries = self.index.entries();
        let mut best: Option<(i32, u16)> = None;
        self.index.query_indices(point, |i, _w| {
            let Some(biome) = biomes.get(i).and_then(|b| b.as_ref()) else {
                return;
            };
            let Some(material) = biome.material_override else {
                return;
            };
            let priority = entries[i].region.priority;
            match best {
                None => best = Some((priority, material)),
                Some((p, _)) if priority > p => best = Some((priority, material)),
                _ => {}
            }
        });
        best.map(|(_, m)| m)
    }
}

/// Cheap-to-clone handle. Snapshot-shaped, never mutated in place.
pub type TerrainRegionSnapshotHandle = Arc<TerrainRegionSnapshot>;

#[cfg(test)]
mod tests {
    use super::*;
    use arvx_regions::{Falloff, Region, RegionEntry, RegionShape};
    use hecs::World;

    fn sphere_region(radius: f32, transition: f32, priority: i32) -> Region {
        Region {
            shape: RegionShape::Sphere { radius },
            falloff: Falloff::Smoothstep { transition_m: transition },
            priority,
        }
    }

    #[test]
    fn empty_snapshot_returns_none() {
        let snap = TerrainRegionSnapshot::new();
        assert!(snap.is_empty());
        assert!(snap.material_override_at(Vec3::ZERO).is_none());
    }

    #[test]
    fn no_biome_means_no_override() {
        // A region exists but carries no BiomeRegion — the bake
        // shouldn't see an override.
        let mut w = World::new();
        let r = sphere_region(10.0, 5.0, 0);
        let e = w.spawn((r,));
        let snap = TerrainRegionSnapshot {
            index: Arc::new(RegionIndex::from_entries(vec![RegionEntry::new(e, r, Vec3::ZERO)])),
            biomes: Arc::new(vec![None]),
        };
        assert!(snap.material_override_at(Vec3::ZERO).is_none());
    }

    #[test]
    fn single_biome_with_override_wins() {
        let mut w = World::new();
        let r = sphere_region(10.0, 5.0, 0);
        let e = w.spawn((r,));
        let snap = TerrainRegionSnapshot {
            index: Arc::new(RegionIndex::from_entries(vec![RegionEntry::new(e, r, Vec3::ZERO)])),
            biomes: Arc::new(vec![Some(BiomeRegion {
                material_override: Some(7),
                ..Default::default()
            })]),
        };
        assert_eq!(snap.material_override_at(Vec3::ZERO), Some(7));
        // Outside the region's falloff band → no override.
        assert_eq!(snap.material_override_at(Vec3::new(100.0, 0.0, 0.0)), None);
    }

    #[test]
    fn higher_priority_region_wins_overlap() {
        // Two overlapping biomes at the same point with different
        // material overrides. Higher priority wins regardless of
        // entry order.
        let mut w = World::new();
        let low = sphere_region(20.0, 5.0, 0);
        let high = sphere_region(20.0, 5.0, 5);
        let e_low = w.spawn((low,));
        let e_high = w.spawn((high,));

        let snap = TerrainRegionSnapshot {
            index: Arc::new(RegionIndex::from_entries(vec![
                RegionEntry::new(e_low, low, Vec3::ZERO),
                RegionEntry::new(e_high, high, Vec3::ZERO),
            ])),
            biomes: Arc::new(vec![
                Some(BiomeRegion {
                    material_override: Some(1),
                    ..Default::default()
                }),
                Some(BiomeRegion {
                    material_override: Some(2),
                    ..Default::default()
                }),
            ]),
        };
        assert_eq!(snap.material_override_at(Vec3::ZERO), Some(2));
    }

    #[test]
    fn priority_winner_does_not_depend_on_entry_order() {
        // Reversed entry order should still pick the high-priority
        // override.
        let mut w = World::new();
        let low = sphere_region(20.0, 5.0, 0);
        let high = sphere_region(20.0, 5.0, 5);
        let e_low = w.spawn((low,));
        let e_high = w.spawn((high,));

        let snap = TerrainRegionSnapshot {
            index: Arc::new(RegionIndex::from_entries(vec![
                RegionEntry::new(e_high, high, Vec3::ZERO),
                RegionEntry::new(e_low, low, Vec3::ZERO),
            ])),
            biomes: Arc::new(vec![
                Some(BiomeRegion {
                    material_override: Some(2),
                    ..Default::default()
                }),
                Some(BiomeRegion {
                    material_override: Some(1),
                    ..Default::default()
                }),
            ]),
        };
        assert_eq!(snap.material_override_at(Vec3::ZERO), Some(2));
    }

    #[test]
    fn biome_with_no_material_override_is_ignored() {
        // Even if a biome region overlaps the point, no
        // material_override → no contribution.
        let mut w = World::new();
        let r = sphere_region(10.0, 5.0, 0);
        let e = w.spawn((r,));
        let snap = TerrainRegionSnapshot {
            index: Arc::new(RegionIndex::from_entries(vec![RegionEntry::new(e, r, Vec3::ZERO)])),
            biomes: Arc::new(vec![Some(BiomeRegion::default())]),
        };
        assert_eq!(snap.material_override_at(Vec3::ZERO), None);
    }

    #[test]
    fn query_biome_visits_only_biome_carrying_regions() {
        // Three regions: one without BiomeRegion, two with. Walk
        // should visit only the latter pair.
        let mut w = World::new();
        let r = sphere_region(10.0, 5.0, 0);
        let e0 = w.spawn((r,));
        let e1 = w.spawn((r,));
        let e2 = w.spawn((r,));
        let snap = TerrainRegionSnapshot {
            index: Arc::new(RegionIndex::from_entries(vec![
                RegionEntry::new(e0, r, Vec3::ZERO),
                RegionEntry::new(e1, r, Vec3::ZERO),
                RegionEntry::new(e2, r, Vec3::ZERO),
            ])),
            biomes: Arc::new(vec![
                None,
                Some(BiomeRegion::default()),
                Some(BiomeRegion::default()),
            ]),
        };
        let mut count = 0;
        snap.query_biome(Vec3::ZERO, |_b, _w| count += 1);
        assert_eq!(count, 2);
    }
}
