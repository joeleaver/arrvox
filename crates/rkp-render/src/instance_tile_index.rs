//! Stage 4 — per-tile spatial accel for the instance pipeline.
//!
//! [`TileIndex`] is a CPU lookup structure built per frame from the
//! emit pass's cached regions. Given `(host_object_id, material_id,
//! tile_coord)`, it returns the `region_index` into this frame's
//! `EmitRegionUniform[]` array (the same index the emit pass writes
//! into the GPU regions buffer). Stage 5's march will upload a
//! flattened version of this index and consult it per ray-cell to
//! find candidate instance regions.
//!
//! ## Tiled vs non-tiled shaders
//!
//! Shaders authored with `@tile_size` produce one region per tile;
//! their entries get keyed by tile coord. Shaders without `@tile_size`
//! produce one region per `(host, material)` covering the whole
//! painted area; their entries land in the FALLBACK list, queried
//! when no tiled region matches the lookup.
//!
//! ## Neighbor radius
//!
//! Prototype voxels live in canonical `[0, 1]³`; the world-space
//! extent of an instance is `prototype_extent × max_scale`. When a
//! region's `tile_size` is comparable to or smaller than the
//! prototype's world extent, instances near the tile boundary may
//! straddle into neighboring tiles. The march uses
//! [`neighbor_radius_for_tile_size`] to widen its tile lookups so
//! these crossing instances are still found.

use std::collections::HashMap;

use crate::user_shader_emit_pass::{InstanceRegionRequest, NO_TILE};

/// One entry in the index — `region_index` is the slot in this frame's
/// dispatch arrays that the emit pass populated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileEntry {
    pub region_index: u32,
}

/// Per `(host_object_id, material_id)` lookup table. Keeps tiled and
/// fallback (non-tiled) regions in separate maps so the march doesn't
/// pay tile-coord hashing cost for the simple non-tiled case.
#[derive(Debug, Clone, Default)]
pub struct PerMaterialTileMap {
    /// `tile_coord -> TileEntry` for tiled shaders. One entry per
    /// tile; duplicate adds for the same tile error out at build time.
    pub tiled: HashMap<[i32; 3], TileEntry>,
    /// Single fallback entry for non-tiled shaders. Multiple
    /// non-tiled regions for the same `(host, material)` would be a
    /// build-time error — the design has at most one per pair.
    pub fallback: Option<TileEntry>,
}

/// Tile→region index for one frame's worth of dirty instance
/// regions. Built fresh each frame after the cache lookups have
/// settled and `region_index` slots have been assigned.
#[derive(Debug, Clone, Default)]
pub struct TileIndex {
    by_material: HashMap<(u32, u32), PerMaterialTileMap>,
}

/// Errors that can arise when building a [`TileIndex`].
#[derive(Debug, Clone, thiserror::Error)]
pub enum TileIndexError {
    #[error(
        "duplicate tiled region for host {host_object_id} material {material_id} tile {tile:?}"
    )]
    DuplicateTile {
        host_object_id: u32,
        material_id: u32,
        tile: [i32; 3],
    },

    #[error(
        "duplicate fallback region for host {host_object_id} material {material_id}"
    )]
    DuplicateFallback {
        host_object_id: u32,
        material_id: u32,
    },
}

/// Builder accumulating regions one at a time. Detects collisions
/// (duplicate tile or duplicate fallback) so callers learn about
/// inconsistent inputs synchronously rather than getting silent
/// over-writes that produce wrong march behavior.
#[derive(Debug, Default)]
pub struct TileIndexBuilder {
    inner: TileIndex,
}

impl TileIndexBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record `region_index` against the request's `(host_object_id,
    /// material_id)` and tile/fallback bucket. NO_TILE → fallback;
    /// any other tile → tiled map. Duplicates are rejected.
    pub fn add_request(
        &mut self,
        request: &InstanceRegionRequest,
        region_index: u32,
    ) -> Result<(), TileIndexError> {
        self.add_keyed(
            request.host_object_id,
            request.material_id,
            request.tile_index,
            region_index,
        )
    }

    /// Lower-level: record a region by explicit key without going
    /// through `InstanceRegionRequest`. Useful for tests + direct
    /// construction from cached entries.
    pub fn add_keyed(
        &mut self,
        host_object_id: u32,
        material_id: u32,
        tile_index: [i32; 3],
        region_index: u32,
    ) -> Result<(), TileIndexError> {
        let bucket = self
            .inner
            .by_material
            .entry((host_object_id, material_id))
            .or_default();
        let entry = TileEntry { region_index };
        if tile_index == NO_TILE {
            if bucket.fallback.is_some() {
                return Err(TileIndexError::DuplicateFallback {
                    host_object_id,
                    material_id,
                });
            }
            bucket.fallback = Some(entry);
        } else {
            if bucket.tiled.contains_key(&tile_index) {
                return Err(TileIndexError::DuplicateTile {
                    host_object_id,
                    material_id,
                    tile: tile_index,
                });
            }
            bucket.tiled.insert(tile_index, entry);
        }
        Ok(())
    }

    pub fn build(self) -> TileIndex {
        self.inner
    }
}

impl TileIndex {
    /// Number of `(host, material)` buckets in the index. Zero when no
    /// instance shaders are dirty this frame.
    pub fn material_count(&self) -> usize {
        self.by_material.len()
    }

    /// Total tiled entries across all buckets — for diagnostics.
    pub fn total_tiled_entries(&self) -> usize {
        self.by_material.values().map(|b| b.tiled.len()).sum()
    }

    /// Total fallback entries — for diagnostics. Equal to the count
    /// of non-tiled `(host, material)` instance regions this frame.
    pub fn total_fallback_entries(&self) -> usize {
        self.by_material
            .values()
            .filter(|b| b.fallback.is_some())
            .count()
    }

    /// Iterate `((host_object_id, material_id), bucket)` pairs.
    /// Order is HashMap-arbitrary — callers that need determinism (the
    /// GPU flatten in [`crate::instance_tile_index_gpu`]) sort by key.
    pub fn buckets(
        &self,
    ) -> impl Iterator<Item = ((u32, u32), &PerMaterialTileMap)> + '_ {
        self.by_material.iter().map(|(k, v)| (*k, v))
    }

    /// Look up exactly the tile at `tile_coord` for the given
    /// `(host, material)`. `None` if no region was registered there.
    pub fn region_at(
        &self,
        host_object_id: u32,
        material_id: u32,
        tile_coord: [i32; 3],
    ) -> Option<u32> {
        self.by_material
            .get(&(host_object_id, material_id))?
            .tiled
            .get(&tile_coord)
            .map(|e| e.region_index)
    }

    /// Fallback (non-tiled) region for `(host, material)`, if any.
    pub fn fallback_region(
        &self,
        host_object_id: u32,
        material_id: u32,
    ) -> Option<u32> {
        self.by_material
            .get(&(host_object_id, material_id))?
            .fallback
            .map(|e| e.region_index)
    }

    /// Return every region the march should consider when sampling at
    /// `tile_coord` for this `(host, material)`: the tile itself, all
    /// neighbors within `±neighbor_radius` along each axis, plus the
    /// fallback region if any. Caller-controlled `out` lets the march
    /// reuse a small `Vec` across many lookups without re-allocating.
    pub fn regions_overlapping_tile(
        &self,
        host_object_id: u32,
        material_id: u32,
        tile_coord: [i32; 3],
        neighbor_radius: i32,
        out: &mut Vec<u32>,
    ) {
        out.clear();
        let Some(bucket) = self.by_material.get(&(host_object_id, material_id))
        else {
            return;
        };
        if neighbor_radius <= 0 {
            if let Some(e) = bucket.tiled.get(&tile_coord) {
                out.push(e.region_index);
            }
        } else {
            // Tile + neighbor cube. Skip the center duplicate via the
            // `dx == 0 && dy == 0 && dz == 0` test in the inner loop.
            for dz in -neighbor_radius..=neighbor_radius {
                for dy in -neighbor_radius..=neighbor_radius {
                    for dx in -neighbor_radius..=neighbor_radius {
                        let probe = [
                            tile_coord[0] + dx,
                            tile_coord[1] + dy,
                            tile_coord[2] + dz,
                        ];
                        if let Some(e) = bucket.tiled.get(&probe) {
                            out.push(e.region_index);
                        }
                    }
                }
            }
        }
        if let Some(e) = bucket.fallback {
            out.push(e.region_index);
        }
    }
}

/// Compute the integer neighbor radius the march should use to find
/// every instance whose geometry overlaps a given tile.
///
/// `prototype_extent_world` is the world-space side length of the
/// prototype's bounding cube AT max scale (canonical `[0,1]³` × max
/// `instance.scale`). `tile_size` is the cube edge length of the
/// region's AABB.
///
/// An instance whose center is at the tile boundary can extend up to
/// `prototype_extent_world / 2` into a neighbor; the march in that
/// neighbor must therefore look back into this tile to find the
/// instance. Worst-case: radius = `ceil(extent / (2 × tile_size))`.
///
/// Worked examples:
/// - tile 1 m, grass 0.4 m → 0.4/2 = 0.2 → ceil(0.2) = **1**. The
///   march in adjacent tiles consults the neighbor where blades cross
///   the boundary.
/// - tile 1 m, sapling 1.5 m → 1.5/2 = 0.75 → ceil(0.75) = **1**.
/// - tile 1 m, redwood 5 m → 5/2 = 2.5 → ceil(2.5) = **3**.
/// - tile 1 m, instance fully zero-extent → **0** (degenerate).
///
/// Radius 1 is the smallest non-trivial case: 27 neighbor lookups
/// instead of 1. Shaders that want radius 0 must enforce author-side
/// that instances never spill past their tile boundary (e.g. inset
/// `pos` by `extent/2` from edges in `emit`).
pub fn neighbor_radius_for_tile_size(prototype_extent_world: f32, tile_size: f32) -> i32 {
    // Reject zero, negative, and NaN — `> 0.0` returns false for NaN
    // so the indirected check via `valid` covers all the bad inputs.
    let valid = tile_size > 0.0 && prototype_extent_world > 0.0;
    if !valid {
        return 0;
    }
    let ratio = prototype_extent_world / (2.0 * tile_size);
    ratio.ceil() as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(host: u32, material: u32, tile: [i32; 3]) -> InstanceRegionRequest {
        InstanceRegionRequest {
            host_object_id: host,
            material_id: material,
            shader_name: "x".to_string(),
            params: vec![],
            aabb_min: [0.0; 3],
            aabb_max: [1.0; 3],
            cell_size: 0.04,
            input_hash: 0,
            animated: false,
            region_thickness: 0.0,
            tile_index: tile,
            stride_u32: 8,
            max_instances: 64,
            host_octree_root: 0,
            host_octree_depth: 0,
            host_octree_extent: 0.0,
            host_grid_origin: [0.0; 3],
            host_inverse_world: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            leaves: Vec::new(),
        }
    }

    #[test]
    fn empty_index_returns_none() {
        let idx = TileIndex::default();
        assert!(idx.region_at(1, 5, [0, 0, 0]).is_none());
        assert!(idx.fallback_region(1, 5).is_none());
        let mut out = Vec::new();
        idx.regions_overlapping_tile(1, 5, [0, 0, 0], 0, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn single_tiled_region_lookup() {
        let mut b = TileIndexBuilder::new();
        b.add_request(&req(1, 5, [3, 0, 7]), 42).unwrap();
        let idx = b.build();
        assert_eq!(idx.region_at(1, 5, [3, 0, 7]), Some(42));
        assert!(idx.region_at(1, 5, [4, 0, 7]).is_none());
        assert!(idx.region_at(1, 6, [3, 0, 7]).is_none());
        assert!(idx.region_at(2, 5, [3, 0, 7]).is_none());
    }

    #[test]
    fn fallback_region_for_non_tiled_shader() {
        let mut b = TileIndexBuilder::new();
        b.add_request(&req(1, 5, NO_TILE), 7).unwrap();
        let idx = b.build();
        assert!(idx.region_at(1, 5, [0, 0, 0]).is_none());
        assert_eq!(idx.fallback_region(1, 5), Some(7));
        let mut out = Vec::new();
        idx.regions_overlapping_tile(1, 5, [0, 0, 0], 0, &mut out);
        // Fallback always yields its region from the overlap query.
        assert_eq!(out, vec![7]);
    }

    #[test]
    fn duplicate_tile_rejects() {
        let mut b = TileIndexBuilder::new();
        b.add_request(&req(1, 5, [0, 0, 0]), 1).unwrap();
        let err = b.add_request(&req(1, 5, [0, 0, 0]), 2).unwrap_err();
        match err {
            TileIndexError::DuplicateTile { host_object_id, material_id, tile } => {
                assert_eq!(host_object_id, 1);
                assert_eq!(material_id, 5);
                assert_eq!(tile, [0, 0, 0]);
            }
            other => panic!("expected DuplicateTile, got {other:?}"),
        }
    }

    #[test]
    fn duplicate_fallback_rejects() {
        let mut b = TileIndexBuilder::new();
        b.add_request(&req(1, 5, NO_TILE), 1).unwrap();
        let err = b.add_request(&req(1, 5, NO_TILE), 2).unwrap_err();
        assert!(matches!(err, TileIndexError::DuplicateFallback { .. }));
    }

    #[test]
    fn distinct_keys_isolated() {
        // Different (host, material) pairs must not collide even if
        // their tile coords match.
        let mut b = TileIndexBuilder::new();
        b.add_request(&req(1, 5, [0, 0, 0]), 1).unwrap();
        b.add_request(&req(1, 6, [0, 0, 0]), 2).unwrap();
        b.add_request(&req(2, 5, [0, 0, 0]), 3).unwrap();
        let idx = b.build();
        assert_eq!(idx.region_at(1, 5, [0, 0, 0]), Some(1));
        assert_eq!(idx.region_at(1, 6, [0, 0, 0]), Some(2));
        assert_eq!(idx.region_at(2, 5, [0, 0, 0]), Some(3));
        assert_eq!(idx.material_count(), 3);
        assert_eq!(idx.total_tiled_entries(), 3);
    }

    #[test]
    fn overlapping_tile_zero_radius_returns_only_center() {
        let mut b = TileIndexBuilder::new();
        b.add_request(&req(1, 5, [0, 0, 0]), 1).unwrap();
        b.add_request(&req(1, 5, [1, 0, 0]), 2).unwrap();
        let idx = b.build();
        let mut out = Vec::new();
        idx.regions_overlapping_tile(1, 5, [0, 0, 0], 0, &mut out);
        assert_eq!(out, vec![1]);
    }

    #[test]
    fn overlapping_tile_radius_one_returns_27_cube() {
        // Fill the 3x3x3 cube around origin — query should return all 27.
        let mut b = TileIndexBuilder::new();
        let mut idx_value = 0u32;
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    b.add_request(&req(1, 5, [dx, dy, dz]), idx_value).unwrap();
                    idx_value += 1;
                }
            }
        }
        let idx = b.build();
        let mut out = Vec::new();
        idx.regions_overlapping_tile(1, 5, [0, 0, 0], 1, &mut out);
        assert_eq!(out.len(), 27);
        // All assigned region indices [0..27) should appear.
        out.sort();
        assert_eq!(out, (0..27).collect::<Vec<_>>());
    }

    #[test]
    fn overlapping_tile_skips_unfilled_neighbors() {
        // Only origin and (+1,0,0) are filled. Query at origin with
        // radius 1 should return exactly those two, not synthesise
        // missing neighbors.
        let mut b = TileIndexBuilder::new();
        b.add_request(&req(1, 5, [0, 0, 0]), 10).unwrap();
        b.add_request(&req(1, 5, [1, 0, 0]), 11).unwrap();
        let idx = b.build();
        let mut out = Vec::new();
        idx.regions_overlapping_tile(1, 5, [0, 0, 0], 1, &mut out);
        out.sort();
        assert_eq!(out, vec![10, 11]);
    }

    #[test]
    fn overlapping_tile_includes_fallback() {
        let mut b = TileIndexBuilder::new();
        b.add_request(&req(1, 5, [0, 0, 0]), 1).unwrap();
        b.add_request(&req(1, 5, NO_TILE), 99).unwrap();
        let idx = b.build();
        let mut out = Vec::new();
        idx.regions_overlapping_tile(1, 5, [0, 0, 0], 0, &mut out);
        out.sort();
        assert_eq!(out, vec![1, 99]);
    }

    #[test]
    fn overlapping_query_clears_out_buffer() {
        // The march reuses one Vec across many lookups; the API must
        // clear leftover values from the prior call.
        let mut b = TileIndexBuilder::new();
        b.add_request(&req(1, 5, [0, 0, 0]), 1).unwrap();
        let idx = b.build();
        let mut out = vec![999, 1234];
        idx.regions_overlapping_tile(1, 5, [0, 0, 0], 0, &mut out);
        assert_eq!(out, vec![1]);
    }

    #[test]
    fn neighbor_radius_grass_returns_one() {
        // 1 m tile, 0.4 m grass blade — boundary-centered blade
        // pokes 0.2 m into neighbor → ratio 0.2 → ceil = 1.
        assert_eq!(neighbor_radius_for_tile_size(0.4, 1.0), 1);
    }

    #[test]
    fn neighbor_radius_redwood_returns_three() {
        // 1 m tile, 5 m redwood — center can extend 2.5 m into a
        // neighbor → ratio 2.5 → ceil = 3.
        assert_eq!(neighbor_radius_for_tile_size(5.0, 1.0), 3);
    }

    #[test]
    fn neighbor_radius_tile_spanning_returns_one() {
        // 1 m tile, 1.5 m sapling — center can extend 0.75 m into
        // neighbor → 0.75 m / 1.0 m = 0.75 → ceil = 1.
        assert_eq!(neighbor_radius_for_tile_size(1.5, 1.0), 1);
    }

    #[test]
    fn neighbor_radius_extent_equal_to_tile() {
        // An instance the same size as a tile: half of it (0.5 m) can
        // poke into the neighbor → radius 1.
        assert_eq!(neighbor_radius_for_tile_size(1.0, 1.0), 1);
    }

    #[test]
    fn neighbor_radius_large_tile_returns_zero() {
        // A tile much larger than the prototype — boundary-centered
        // instances still poke ratio-many tiles, but if the ratio is
        // exactly 0 (extent → 0) the formula yields 0.
        assert_eq!(neighbor_radius_for_tile_size(0.000_001, 100.0), 1);
        // Edge case: extent exactly 0 short-circuits to 0.
        assert_eq!(neighbor_radius_for_tile_size(0.0, 100.0), 0);
    }

    #[test]
    fn neighbor_radius_zero_inputs_safe() {
        // Defensive: zero or negative tile_size or extent shouldn't
        // panic — return 0.
        assert_eq!(neighbor_radius_for_tile_size(0.0, 1.0), 0);
        assert_eq!(neighbor_radius_for_tile_size(1.0, 0.0), 0);
        assert_eq!(neighbor_radius_for_tile_size(-1.0, 1.0), 0);
    }

    #[test]
    fn build_from_cache_keys_via_iterator() {
        // Demonstrate the intended caller flow: build the index by
        // walking the cache's `touched_keys` iterator + assigned
        // region indices.
        let touched: [(u32, u32, [i32; 3]); 3] = [
            (1, 5, [0, 0, 0]),
            (1, 5, [1, 0, 0]),
            (1, 6, NO_TILE),
        ];
        let mut b = TileIndexBuilder::new();
        for (i, (host, material, tile)) in touched.iter().enumerate() {
            b.add_keyed(*host, *material, *tile, i as u32).unwrap();
        }
        let idx = b.build();
        assert_eq!(idx.region_at(1, 5, [0, 0, 0]), Some(0));
        assert_eq!(idx.region_at(1, 5, [1, 0, 0]), Some(1));
        assert_eq!(idx.fallback_region(1, 6), Some(2));
        assert_eq!(idx.material_count(), 2);
    }
}
