//! Persistent per-tile sculpt diff.
//!
//! V2 LOD pyramid follow-up. The sculpt path mutates the live tile's
//! octree in place via [`arvx_core::sculpt::apply_delta`], discarding
//! the [`SculptDelta`] that produced the writes. That makes three
//! things impossible:
//!
//! 1. **Re-baking a sculpted tile.** Once the tile evicts (or a stamp
//!    move triggers invalidation) the sculpt is lost — the bake input
//!    is `TerrainFn + stamps + biome regions`, never the sculpt diff.
//! 2. **Showing the sculpt at coarse LOD.** Level-N≥1 ancestor tiles
//!    bake from the same procedural source, so the sculpt is absent.
//!    V1 worked around this by pinning the level-0 tile and suppressing
//!    overlapping coarse tiles.
//! 3. **Surviving scene reload without persisting the full tile.**
//!    `.arvxtile` saves the post-sculpt artifact, but only at level 0;
//!    nothing tells a coarse-LOD bake "this region was sculpted".
//!
//! [`SculptDiff`] solves all three: at brush time we append the
//! produced [`SculptDelta`]'s edits into a per-tile diff keyed by
//! [`TileKey`]; bake-replay applies the diff after voxelization so a
//! fresh bake matches the live mutated state; coarse-LOD bakes apply
//! the diff downsampled into their coarser grid; and the diff
//! serialises to a `.arvxsculpt` sidecar so it survives reload.

use arvx_core::sculpt::{LeafEdit, LeafEditOp, SculptDelta};
use glam::{IVec3, Vec3};
#[cfg(test)]
use glam::UVec3;
use std::collections::HashMap;

use crate::tile_key::TileKey;

/// Cumulative per-tile sculpt diff. Edits are stored in stroke order
/// (last-write-wins on the same coord, by virtue of `apply_delta`'s
/// own ordering — same coord across two edits leaves the second's
/// state). Each [`LeafEdit::coord`] is in the **fine-tile's octree
/// grid space** (`[0, cells_per_axis)`).
///
/// ## What's NOT captured
///
/// [`LeafEditOp::SetNormal`] carries a pre-resolved `LeafAttrPool`
/// slot id — that slot is per-octree and won't survive replay onto a
/// freshly-baked octree. We filter SetNormal on capture. Smooth's
/// pure-normal smoothing therefore vanishes on re-bake; Smooth's
/// geometry edits (the morph pass's Remove/Add) survive intact since
/// they flow through the captured op variants.
///
/// ## Downsampling for coarse LOD
///
/// [`SculptDiff::downsampled_to`] re-projects every edit into a
/// coarser tile's grid via world-space arithmetic. Multiple fine
/// edits hitting the same coarse cell collapse by op priority:
/// **Add > SetInterior > Remove/Empty**, with last-write-wins on the
/// Add material. Add wins because a visible material patch at coarse
/// LOD is a better approximation of "this region was sculpted" than
/// a void.
#[derive(Debug, Clone, Default)]
pub struct SculptDiff {
    /// Captured edits in stroke order. Out-of-bounds coords for the
    /// owning tile are not produced by the brush — `compute_brush_edits`
    /// clamps to the asset's octree extent before emitting LeafEdits.
    pub edits: Vec<LeafEdit>,
}

impl SculptDiff {
    /// Empty diff. Equivalent to [`Default`].
    pub fn new() -> Self {
        Self::default()
    }

    /// True when no edits have been captured.
    pub fn is_empty(&self) -> bool {
        self.edits.is_empty()
    }

    /// Number of captured edits (post-SetNormal-filter).
    pub fn len(&self) -> usize {
        self.edits.len()
    }

    /// Append every edit in `delta` except [`LeafEditOp::SetNormal`]
    /// (whose `slot` field can't be replayed onto a fresh octree).
    pub fn append_delta(&mut self, delta: &SculptDelta) {
        for edit in &delta.edits {
            if matches!(edit.op, LeafEditOp::SetNormal { .. }) {
                continue;
            }
            self.edits.push(*edit);
        }
    }

    /// Project this diff into `coarse_tile`'s grid via world-space
    /// arithmetic. `fine_tile` is the owning tile of `self`; both
    /// `*_voxel_size_m` values come from
    /// [`crate::terrain::Terrain::voxel_size_for_level`].
    ///
    /// Coarse coords landing outside `[0, cells_per_coarse_axis)` are
    /// dropped (defensive — `compute_brush_edits`' clamp keeps fine
    /// coords in-tile, and a fine tile's footprint is a strict subset
    /// of any coarse ancestor's footprint).
    ///
    /// `coarse_tile.level` must be strictly greater than
    /// `fine_tile.level`; same-level is meaningless and a debug-mode
    /// assertion.
    pub fn downsampled_to(
        &self,
        fine_tile: TileKey,
        fine_voxel_size_m: f32,
        coarse_tile: TileKey,
        coarse_voxel_size_m: f32,
    ) -> SculptDiff {
        debug_assert!(
            coarse_tile.level > fine_tile.level,
            "downsample target must be strictly coarser than source"
        );

        // Bucket by coarse cell, aggregating multiple fine edits per
        // coarse cell into one final op per priority rules.
        let mut buckets: HashMap<IVec3, AggOp> = HashMap::new();
        let fine_origin = fine_tile.origin_world().to_vec3();
        let coarse_origin = coarse_tile.origin_world().to_vec3();

        for edit in &self.edits {
            let new_op = match AggOp::from_leaf_op(edit.op) {
                Some(op) => op,
                None => continue, // SetNormal — already filtered on capture
            };
            let world = fine_origin + edit.coord.as_vec3() * fine_voxel_size_m;
            let rel = world - coarse_origin;
            let cc = (rel / coarse_voxel_size_m).floor().as_ivec3();
            buckets
                .entry(cc)
                .and_modify(|existing| *existing = existing.combine(new_op))
                .or_insert(new_op);
        }

        // Compute coarse-tile's cells-per-axis. With our voxel-size
        // saturation (Tier 0 = 1 m) this is not always `cells_per_fine
        // / 2^N` — for saturated coarse LODs the cell count GROWS.
        let coarse_cells_per_axis =
            (coarse_tile.extent_m() / coarse_voxel_size_m).round() as i32;

        let mut out: Vec<LeafEdit> = buckets
            .into_iter()
            .filter_map(|(coord, agg)| {
                if coord.x < 0
                    || coord.y < 0
                    || coord.z < 0
                    || coord.x >= coarse_cells_per_axis
                    || coord.y >= coarse_cells_per_axis
                    || coord.z >= coarse_cells_per_axis
                {
                    return None;
                }
                Some(LeafEdit {
                    coord: coord.as_uvec3(),
                    op: agg.into_leaf_op(),
                })
            })
            .collect();

        // Stable ordering so two equal HashMaps don't render as two
        // different `Vec`s — bake-replay is order-insensitive within a
        // diff (one edit per cell post-aggregation) but tests and the
        // sidecar format both benefit from a deterministic layout.
        out.sort_by_key(|e| (e.coord.z, e.coord.y, e.coord.x));
        SculptDiff { edits: out }
    }
}

/// Reduced op variant used inside the downsample aggregator.
/// `SetNormal` doesn't survive capture, so the downsample never sees it.
#[derive(Debug, Clone, Copy)]
enum AggOp {
    Add { material: u16, normal: Vec3 },
    Interior,
    Empty,
}

impl AggOp {
    fn from_leaf_op(op: LeafEditOp) -> Option<Self> {
        match op {
            LeafEditOp::Add { material, normal, .. } => Some(AggOp::Add { material, normal }),
            LeafEditOp::SetInterior => Some(AggOp::Interior),
            LeafEditOp::Remove | LeafEditOp::Empty => Some(AggOp::Empty),
            LeafEditOp::SetNormal { .. } => None,
        }
    }

    /// Priority: Add > Interior > Empty. For two Adds the new one
    /// wins (last-write-wins on material/normal). Coalesces under the
    /// idea that a partial sub-volume of fine Adds is a better coarse
    /// representation than a void.
    fn combine(self, new: AggOp) -> AggOp {
        match (self, new) {
            (AggOp::Add { .. }, AggOp::Add { material, normal }) => {
                AggOp::Add { material, normal }
            }
            (AggOp::Add { material, normal }, _) => AggOp::Add { material, normal },
            (_, AggOp::Add { material, normal }) => AggOp::Add { material, normal },
            (AggOp::Interior, _) | (_, AggOp::Interior) => AggOp::Interior,
            (AggOp::Empty, AggOp::Empty) => AggOp::Empty,
        }
    }

    fn into_leaf_op(self) -> LeafEditOp {
        match self {
            AggOp::Add { material, normal } => LeafEditOp::Add { material, normal, dist: 0.0 },
            AggOp::Interior => LeafEditOp::SetInterior,
            AggOp::Empty => LeafEditOp::Remove,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::Terrain;

    fn add(coord: UVec3, material: u16) -> LeafEdit {
        LeafEdit {
            coord,
            op: LeafEditOp::Add {
                material,
                normal: Vec3::Y,
                dist: 0.0,
            },
        }
    }

    fn remove(coord: UVec3) -> LeafEdit {
        LeafEdit {
            coord,
            op: LeafEditOp::Remove,
        }
    }

    fn delta_with(edits: Vec<LeafEdit>) -> SculptDelta {
        SculptDelta {
            edits,
            ..Default::default()
        }
    }

    #[test]
    fn empty_diff_is_empty() {
        let d = SculptDiff::new();
        assert!(d.is_empty());
        assert_eq!(d.len(), 0);
    }

    #[test]
    fn append_delta_accumulates_in_order() {
        let mut d = SculptDiff::new();
        d.append_delta(&delta_with(vec![
            add(UVec3::new(1, 2, 3), 7),
            remove(UVec3::new(4, 5, 6)),
        ]));
        d.append_delta(&delta_with(vec![add(UVec3::new(7, 8, 9), 11)]));
        assert_eq!(d.len(), 3);
        assert_eq!(d.edits[0].coord, UVec3::new(1, 2, 3));
        assert_eq!(d.edits[1].coord, UVec3::new(4, 5, 6));
        assert_eq!(d.edits[2].coord, UVec3::new(7, 8, 9));
    }

    #[test]
    fn append_delta_filters_set_normal() {
        let mut d = SculptDiff::new();
        d.append_delta(&delta_with(vec![
            add(UVec3::new(1, 0, 0), 1),
            LeafEdit {
                coord: UVec3::new(2, 0, 0),
                op: LeafEditOp::SetNormal {
                    slot: 42,
                    normal: Vec3::Y,
                },
            },
            add(UVec3::new(3, 0, 0), 1),
        ]));
        assert_eq!(d.len(), 2);
        assert_eq!(d.edits[0].coord, UVec3::new(1, 0, 0));
        assert_eq!(d.edits[1].coord, UVec3::new(3, 0, 0));
    }

    /// 4-cell Add cluster on a 2x2x1 sub-volume in level-0 tile (0,0,0)
    /// → 1 Add at the ancestor level-1 tile (0,0,0). Tier 2 base ⇒
    /// fine vs = 0.25 m, coarse vs = 0.5 m, so cells (0,0,0)..(1,1,0)
    /// all land in coarse cell (0,0,0).
    #[test]
    fn downsample_collapses_2x2_block_to_one_cell() {
        let terrain = Terrain::default();
        let fine_vs = terrain.voxel_size_for_level(0);
        let coarse_vs = terrain.voxel_size_for_level(1);
        let fine = TileKey::level0(0, 0, 0);
        let coarse = TileKey {
            level: 1,
            x: 0,
            y: 0,
            z: 0,
        };

        let mut d = SculptDiff::new();
        d.append_delta(&delta_with(vec![
            add(UVec3::new(0, 0, 0), 5),
            add(UVec3::new(1, 0, 0), 5),
            add(UVec3::new(0, 1, 0), 5),
            add(UVec3::new(1, 1, 0), 5),
        ]));
        let coarse_diff = d.downsampled_to(fine, fine_vs, coarse, coarse_vs);
        assert_eq!(coarse_diff.len(), 1);
        let edit = coarse_diff.edits[0];
        assert_eq!(edit.coord, UVec3::new(0, 0, 0));
        match edit.op {
            LeafEditOp::Add { material, .. } => assert_eq!(material, 5),
            other => panic!("expected Add, got {other:?}"),
        }
    }

    /// Add + Remove targeting the same coarse cell → Add wins
    /// (visible material patch is a better coarse representation than
    /// a void).
    #[test]
    fn downsample_add_wins_over_remove_on_collision() {
        let terrain = Terrain::default();
        let fine_vs = terrain.voxel_size_for_level(0);
        let coarse_vs = terrain.voxel_size_for_level(1);
        let fine = TileKey::level0(0, 0, 0);
        let coarse = TileKey {
            level: 1,
            x: 0,
            y: 0,
            z: 0,
        };

        let mut d = SculptDiff::new();
        d.append_delta(&delta_with(vec![
            remove(UVec3::new(0, 0, 0)),
            add(UVec3::new(1, 0, 0), 9),
        ]));
        let cd = d.downsampled_to(fine, fine_vs, coarse, coarse_vs);
        assert_eq!(cd.len(), 1);
        match cd.edits[0].op {
            LeafEditOp::Add { material, .. } => assert_eq!(material, 9),
            other => panic!("expected Add, got {other:?}"),
        }
    }

    /// Last-Add-wins on consecutive Adds at the same coarse cell.
    #[test]
    fn downsample_last_add_wins_material() {
        let terrain = Terrain::default();
        let fine_vs = terrain.voxel_size_for_level(0);
        let coarse_vs = terrain.voxel_size_for_level(1);
        let fine = TileKey::level0(0, 0, 0);
        let coarse = TileKey {
            level: 1,
            x: 0,
            y: 0,
            z: 0,
        };

        let mut d = SculptDiff::new();
        d.append_delta(&delta_with(vec![
            add(UVec3::new(0, 0, 0), 3),
            add(UVec3::new(1, 0, 0), 99),
        ]));
        let cd = d.downsampled_to(fine, fine_vs, coarse, coarse_vs);
        assert_eq!(cd.len(), 1);
        match cd.edits[0].op {
            LeafEditOp::Add { material, .. } => assert_eq!(material, 99),
            other => panic!("expected Add, got {other:?}"),
        }
    }

    /// Fine tile (1, 0, 0) lives inside coarse tile (0, 0, 0) at level
    /// 1 (the coarse covers fine tiles {0,1} along x). An edit at fine
    /// coord (0, 0, 0) in fine tile (1, 0, 0) is at world (64, 0, 0);
    /// its coarse cell is (128, 0, 0)/coarse_vs(0.5) = (128, 0, 0)
    /// inside the coarse tile of extent 128 m → coarse coord
    /// (128, 0, 0) but coarse cells_per_axis = 128/0.5 = 256, so
    /// coord 128 is in-bounds.
    #[test]
    fn downsample_offset_fine_tile_lands_correctly() {
        let terrain = Terrain::default();
        let fine_vs = terrain.voxel_size_for_level(0);
        let coarse_vs = terrain.voxel_size_for_level(1);
        let fine = TileKey::level0(1, 0, 0);
        let coarse = TileKey {
            level: 1,
            x: 0,
            y: 0,
            z: 0,
        };

        let mut d = SculptDiff::new();
        d.append_delta(&delta_with(vec![add(UVec3::new(0, 0, 0), 1)]));
        let cd = d.downsampled_to(fine, fine_vs, coarse, coarse_vs);
        assert_eq!(cd.len(), 1);
        assert_eq!(cd.edits[0].coord, UVec3::new(128, 0, 0));
    }

    /// Level-2 downsample saturates voxel size at Tier 0 = 1 m. The
    /// coarse tile is 256 m on each axis, so cells_per_axis = 256.
    /// Four edits spanning a 4-fine-cell band along x at coord
    /// (0..4, 0, 0) land in 1 coarse cell (since each coarse cell
    /// covers 1m = 4 fine cells of 0.25 m).
    #[test]
    fn downsample_level2_saturated_voxel() {
        let terrain = Terrain::default();
        let fine_vs = terrain.voxel_size_for_level(0);
        let coarse_vs = terrain.voxel_size_for_level(2);
        assert!((coarse_vs - 1.0).abs() < 1e-6);
        let fine = TileKey::level0(0, 0, 0);
        let coarse = TileKey {
            level: 2,
            x: 0,
            y: 0,
            z: 0,
        };

        let mut d = SculptDiff::new();
        d.append_delta(&delta_with(vec![
            add(UVec3::new(0, 0, 0), 1),
            add(UVec3::new(1, 0, 0), 1),
            add(UVec3::new(2, 0, 0), 1),
            add(UVec3::new(3, 0, 0), 1),
        ]));
        let cd = d.downsampled_to(fine, fine_vs, coarse, coarse_vs);
        assert_eq!(cd.len(), 1);
        assert_eq!(cd.edits[0].coord, UVec3::new(0, 0, 0));
    }
}
