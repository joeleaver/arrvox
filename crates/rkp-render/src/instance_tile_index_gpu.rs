//! Stage 5a — GPU-side flat layout for the per-frame [`TileIndex`].
//!
//! Stage 4's [`crate::instance_tile_index::TileIndex`] is a CPU-only
//! `HashMap` keyed by `(host_object_id, material_id)` with per-tile and
//! fallback sub-buckets. Stage 5b's march needs a deterministic byte
//! layout it can bind as a storage buffer; this module is the bridge.
//!
//! ## Layout
//!
//! The flat layout is a single sorted `array<GpuTileIndexEntry>`. Each
//! entry corresponds to one region in this frame's dispatch arrays —
//! either a tiled region (real `tile` coord) or a fallback region
//! (`tile == NO_TILE`). The sort order is stable and lexicographic on
//! `(host_object_id, material_id, tile_x, tile_y, tile_z)` so:
//!
//! * binary search by `(host, material, tile)` is possible if a future
//!   stage wants per-pixel lookup;
//! * all entries for a given `(host, material)` form a contiguous run,
//!   so a march that iterates per-bucket can locate its slice with one
//!   lower-bound search and walk forward;
//! * fallback entries (NO_TILE) sort first within their `(host,
//!   material)` group because `NO_TILE = i32::MIN` is the smallest int
//!   triple — the march can detect them by checking `tile_x == i32::MIN`.
//!
//! ## V1 consumer
//!
//! Per the locked Stage 5 design (linear region iteration with AABB
//! cull), Stage 5b's march walks the full `array<GpuTileIndexEntry>`
//! and AABB-culls each region. Tile coords ride along but aren't used
//! by V1 — they're future-proofing for the per-pixel lookup variant.

use std::cmp::Ordering;

use crate::instance_tile_index::TileIndex;
use crate::user_shader_emit_pass::NO_TILE;

/// One per-region record on the GPU. 32 bytes — tile coords kept as
/// signed 32-bit ints (matches the CPU [`crate::instance_tile_index`]
/// keys exactly) so flatten doesn't have to repack.
///
/// `host_object_id` and `material_id` mirror the cache key. `tile_x/y/z`
/// equal `i32::MIN` for fallback (non-tiled) regions; otherwise they're
/// the region's tile coord. `region_index` is the slot in this frame's
/// dispatch arrays — the same value the caller passed to
/// [`crate::instance_tile_index::TileIndexBuilder::add_request`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuTileIndexEntry {
    pub host_object_id: u32,
    pub material_id: u32,
    pub tile_x: i32,
    pub tile_y: i32,
    pub tile_z: i32,
    pub region_index: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

const _: () = assert!(std::mem::size_of::<GpuTileIndexEntry>() == 32);

impl GpuTileIndexEntry {
    /// `true` when this entry came from a fallback (non-tiled) region.
    /// Detection: tile coord triple equals [`NO_TILE`] (`i32::MIN`).
    pub fn is_fallback(&self) -> bool {
        [self.tile_x, self.tile_y, self.tile_z] == NO_TILE
    }

    /// `(tile_x, tile_y, tile_z)` for tiled entries. Returns the
    /// sentinel for fallback entries — callers that need to distinguish
    /// should check [`Self::is_fallback`] first.
    pub fn tile(&self) -> [i32; 3] {
        [self.tile_x, self.tile_y, self.tile_z]
    }
}

/// Walk a CPU [`TileIndex`] and emit a flat sorted `Vec` of GPU entries.
/// One entry per region (tiled + fallback). Sort key:
/// `(host_object_id, material_id, tile_x, tile_y, tile_z)`.
///
/// Within a `(host, material)` group the fallback (NO_TILE) entry — if
/// any — appears first because `i32::MIN` is the lexicographically
/// smallest triple. This lets the march detect it without a separate
/// fallback table: lower-bound by `(host, material)`, check the first
/// entry's tile against [`NO_TILE`].
pub fn flatten_tile_index(index: &TileIndex) -> Vec<GpuTileIndexEntry> {
    let total = index.total_tiled_entries() + index.total_fallback_entries();
    let mut out: Vec<GpuTileIndexEntry> = Vec::with_capacity(total);

    for ((host, material), bucket) in iter_buckets_sorted(index) {
        if let Some(fallback) = bucket.fallback_region(host, material) {
            out.push(GpuTileIndexEntry {
                host_object_id: host,
                material_id: material,
                tile_x: NO_TILE[0],
                tile_y: NO_TILE[1],
                tile_z: NO_TILE[2],
                region_index: fallback,
                _pad0: 0,
                _pad1: 0,
            });
        }
        for (tile, region_index) in bucket.tiled_sorted() {
            out.push(GpuTileIndexEntry {
                host_object_id: host,
                material_id: material,
                tile_x: tile[0],
                tile_y: tile[1],
                tile_z: tile[2],
                region_index,
                _pad0: 0,
                _pad1: 0,
            });
        }
    }
    debug_assert_eq!(out.len(), total);
    debug_assert!(is_sorted_lexicographic(&out));
    out
}

/// Sort-by-key view over a [`TileIndex`]'s buckets — collect the keys,
/// sort them, then yield references with deterministic ordering.
fn iter_buckets_sorted(
    index: &TileIndex,
) -> impl Iterator<Item = ((u32, u32), TileBucketView<'_>)> {
    let mut keys: Vec<(u32, u32)> = index.buckets().map(|(k, _)| k).collect();
    keys.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    keys.into_iter().map(move |(host, material)| {
        let bucket = index
            .buckets()
            .find(|(k, _)| k.0 == host && k.1 == material)
            .map(|(_, b)| b)
            .expect("key collected from buckets() must still be present");
        ((host, material), TileBucketView { bucket })
    })
}

/// Read-only view over a single `(host, material)` bucket. Wraps the
/// CPU [`crate::instance_tile_index::PerMaterialTileMap`] with sorted
/// iteration helpers used by [`flatten_tile_index`].
struct TileBucketView<'a> {
    bucket: &'a crate::instance_tile_index::PerMaterialTileMap,
}

impl<'a> TileBucketView<'a> {
    fn fallback_region(&self, _host: u32, _material: u32) -> Option<u32> {
        self.bucket.fallback.map(|e| e.region_index)
    }

    fn tiled_sorted(&self) -> impl Iterator<Item = ([i32; 3], u32)> + '_ {
        let mut entries: Vec<([i32; 3], u32)> = self
            .bucket
            .tiled
            .iter()
            .map(|(t, e)| (*t, e.region_index))
            .collect();
        entries.sort_unstable_by(|a, b| {
            a.0[0]
                .cmp(&b.0[0])
                .then(a.0[1].cmp(&b.0[1]))
                .then(a.0[2].cmp(&b.0[2]))
        });
        entries.into_iter()
    }
}

fn is_sorted_lexicographic(v: &[GpuTileIndexEntry]) -> bool {
    v.windows(2).all(|w| {
        let a = &w[0];
        let b = &w[1];
        compare_entries(a, b) != Ordering::Greater
    })
}

fn compare_entries(a: &GpuTileIndexEntry, b: &GpuTileIndexEntry) -> Ordering {
    a.host_object_id
        .cmp(&b.host_object_id)
        .then(a.material_id.cmp(&b.material_id))
        .then(a.tile_x.cmp(&b.tile_x))
        .then(a.tile_y.cmp(&b.tile_y))
        .then(a.tile_z.cmp(&b.tile_z))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance_tile_index::TileIndexBuilder;

    #[test]
    fn entry_is_pod_and_32_bytes() {
        assert_eq!(std::mem::size_of::<GpuTileIndexEntry>(), 32);
        assert_eq!(std::mem::align_of::<GpuTileIndexEntry>(), 4);
        // Bytemuck round-trip works.
        let entry = GpuTileIndexEntry {
            host_object_id: 7,
            material_id: 3,
            tile_x: -2,
            tile_y: 4,
            tile_z: 9,
            region_index: 42,
            _pad0: 0,
            _pad1: 0,
        };
        let bytes: &[u8] = bytemuck::bytes_of(&entry);
        let round: GpuTileIndexEntry = *bytemuck::from_bytes(bytes);
        assert_eq!(round, entry);
    }

    #[test]
    fn flatten_empty_index_is_empty() {
        let idx = crate::instance_tile_index::TileIndex::default();
        let flat = flatten_tile_index(&idx);
        assert!(flat.is_empty());
    }

    #[test]
    fn flatten_single_tiled_entry() {
        let mut b = TileIndexBuilder::new();
        b.add_keyed(7, 3, [1, -2, 5], 42).unwrap();
        let idx = b.build();
        let flat = flatten_tile_index(&idx);
        assert_eq!(flat.len(), 1);
        let e = &flat[0];
        assert_eq!(e.host_object_id, 7);
        assert_eq!(e.material_id, 3);
        assert_eq!(e.tile(), [1, -2, 5]);
        assert_eq!(e.region_index, 42);
        assert!(!e.is_fallback());
    }

    #[test]
    fn flatten_fallback_only_entry() {
        let mut b = TileIndexBuilder::new();
        b.add_keyed(7, 3, NO_TILE, 99).unwrap();
        let idx = b.build();
        let flat = flatten_tile_index(&idx);
        assert_eq!(flat.len(), 1);
        let e = &flat[0];
        assert!(e.is_fallback());
        assert_eq!(e.region_index, 99);
    }

    #[test]
    fn flatten_fallback_sorts_before_tiled_within_bucket() {
        // Same (host, material), one fallback + two tiled entries.
        // Fallback must appear first because NO_TILE = i32::MIN.
        let mut b = TileIndexBuilder::new();
        b.add_keyed(1, 5, [3, 0, 0], 20).unwrap();
        b.add_keyed(1, 5, [1, 0, 0], 10).unwrap();
        b.add_keyed(1, 5, NO_TILE, 99).unwrap();
        let flat = flatten_tile_index(&b.build());
        assert_eq!(flat.len(), 3);
        assert!(flat[0].is_fallback());
        assert_eq!(flat[0].region_index, 99);
        assert_eq!(flat[1].tile(), [1, 0, 0]);
        assert_eq!(flat[1].region_index, 10);
        assert_eq!(flat[2].tile(), [3, 0, 0]);
        assert_eq!(flat[2].region_index, 20);
    }

    #[test]
    fn flatten_sorts_by_host_then_material_then_tile() {
        // Mix multiple buckets; verify lexicographic ordering.
        let mut b = TileIndexBuilder::new();
        b.add_keyed(2, 5, [0, 0, 0], 1).unwrap();
        b.add_keyed(1, 6, [0, 0, 0], 2).unwrap();
        b.add_keyed(1, 5, [0, 0, 0], 3).unwrap();
        b.add_keyed(1, 5, [1, 0, 0], 4).unwrap();
        let flat = flatten_tile_index(&b.build());
        assert_eq!(flat.len(), 4);
        // (1,5,[0,0,0])  -> region 3
        // (1,5,[1,0,0])  -> region 4
        // (1,6,[0,0,0])  -> region 2
        // (2,5,[0,0,0])  -> region 1
        let keys: Vec<(u32, u32, [i32; 3], u32)> = flat
            .iter()
            .map(|e| (e.host_object_id, e.material_id, e.tile(), e.region_index))
            .collect();
        assert_eq!(
            keys,
            vec![
                (1, 5, [0, 0, 0], 3),
                (1, 5, [1, 0, 0], 4),
                (1, 6, [0, 0, 0], 2),
                (2, 5, [0, 0, 0], 1),
            ]
        );
    }

    #[test]
    fn flatten_total_count_matches_index() {
        let mut b = TileIndexBuilder::new();
        b.add_keyed(1, 5, [0, 0, 0], 1).unwrap();
        b.add_keyed(1, 5, [1, 0, 0], 2).unwrap();
        b.add_keyed(1, 5, NO_TILE, 99).unwrap();
        b.add_keyed(2, 5, NO_TILE, 100).unwrap();
        let idx = b.build();
        let flat = flatten_tile_index(&idx);
        assert_eq!(
            flat.len(),
            idx.total_tiled_entries() + idx.total_fallback_entries(),
        );
    }
}
