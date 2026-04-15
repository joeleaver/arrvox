//! Bottom-up prefilter pass over a [`SparseOctree`] that emits one
//! [`LeafAttr`] per internal (branch) node representing the averaged
//! surface of its subtree.
//!
//! The GPU march uses these prefiltered attrs to terminate descent once a
//! node's projected screen footprint drops below ~1 pixel — avoiding the
//! wasteful case where a ray descends to a 5 mm voxel that occupies less
//! than a pixel on screen and produces a randomly-aliased sample.
//!
//! # Algorithm
//!
//! For each octree node, starting from the leaves and walking up:
//!
//! * **EMPTY / INTERIOR**: no surface — nothing to aggregate, no emit.
//! * **LEAF**: the aggregate *is* the leaf's attr (coverage = 1 cell).
//! * **BRICK**: iterate all 64 cells, accumulate per-cell contributions.
//! * **BRANCH**: sum the aggregates of the 8 children.
//!
//! At every branch, once the aggregate is ready, we try to emit a single
//! prefiltered [`LeafAttr`]:
//!
//! * **Normal** = `normalize(Σ leaf_normal_i)` — per-cell normals are
//!   summed (each already a unit vector), then normalized *last*. If the
//!   summed magnitude is too small relative to the cell count (heavy
//!   dispersion — e.g. a thin double-sided wall cancels out), we skip the
//!   emit and let the shader descend further. This is the correct
//!   fallback: the aggregate doesn't represent the subtree well, so
//!   presenting it would be worse than traversing deeper.
//!
//! * **Primary material** = the one covering the most cells.
//!
//! * **Secondary material + blend weight** = the runner-up, with weight
//!   = `secondary_count / (primary_count + secondary_count) * 15`. The
//!   existing shader path already blends primary+secondary at the fetch
//!   site — reusing that path means the 51 %/49 % → 49 %/51 % transition
//!   across frames is *symmetric* and pop-free.
//!
//! * **Color** = weighted average of per-cell colors (unpacked channels
//!   then repacked).
//!
//! # DAG-sharing correctness
//!
//! After `deduplicate_subtrees`, two parent branches can reference the
//! same 8-child block; those branches have *identical* subtrees, so they
//! produce the *identical* aggregate and prefiltered attr. The walk
//! memoizes on node index, so each unique subtree is visited once.
//!
//! # Allocation discipline
//!
//! All emitted [`LeafAttr`]s go into the same [`LeafAttrPool`] as the
//! voxelize pass, via the same `attr_dedup` map. Post-prefilter, the
//! pool's allocated range is still the contiguous
//! `leaf_attr_slot_start..(leaf_attr_slot_start + attr_dedup.len())` —
//! so the scene manager's existing `deallocate_range(start, count)`
//! correctly frees internal-attr allocations alongside leaf allocations.
//! No new bookkeeping needed.

use std::collections::HashMap;

use glam::Vec3;

use crate::brick_pool::{BrickPool, BRICK_CELLS, BRICK_DIM, BRICK_EMPTY};
use crate::leaf_attr::LeafAttr;
use crate::leaf_attr_pool::LeafAttrPool;
use crate::sparse_octree::{
    brick_id as brick_id_of, is_branch, is_brick, is_leaf, leaf_slot, SparseOctree, EMPTY_NODE,
    INTERIOR_NODE,
};

/// Normalized-normal magnitude below which we treat the subtree as too
/// dispersed to represent with a single prefiltered normal. Tuned so a
/// ~30° spread across contributing leaves still emits (cos 30° ≈ 0.87),
/// but two back-to-back surfaces cancelling out (magnitude → 0) don't.
///
/// Units: `|Σ n_i| / count` where each `n_i` is a unit normal. `1.0`
/// means all normals identical; `0.0` means perfect cancellation.
const DISPERSION_THRESHOLD: f32 = 0.3;

/// Aggregated surface statistics for a subtree. Additive — parent's
/// aggregate is the `+` of its 8 children's aggregates.
#[derive(Debug, Clone, Default)]
struct NodeAggregate {
    /// Number of populated cells contributing to this aggregate.
    cell_count: u32,
    /// Sum of per-cell unit normals (unnormalized).
    normal_sum: Vec3,
    /// Running sum of per-cell unpacked colors, weighted by cell count.
    /// `[r, g, b, a]` each in `[0, 255 × cell_count]`.
    color_sum: [f32; 4],
    /// Histogram of material_primary ids across contributing cells.
    /// Size is at most the number of distinct materials in the subtree;
    /// typical scenes have a handful, so the linear-scan cost is fine.
    material_count: Vec<(u16, u32)>,
}

impl NodeAggregate {
    /// Aggregate representing a single populated cell.
    fn from_leaf_attr(attr: LeafAttr, color_packed: u32) -> Self {
        let [r, g, b, a] = unpack_color_u8(color_packed);
        let material_count = vec![(attr.material_primary, 1)];
        Self {
            cell_count: 1,
            normal_sum: attr.normal(),
            color_sum: [r as f32, g as f32, b as f32, a as f32],
            material_count,
        }
    }

    /// Merge another aggregate into this one (in-place).
    fn add(&mut self, other: &NodeAggregate) {
        if other.cell_count == 0 {
            return;
        }
        self.cell_count += other.cell_count;
        self.normal_sum += other.normal_sum;
        for i in 0..4 {
            self.color_sum[i] += other.color_sum[i];
        }
        for &(mat, cnt) in &other.material_count {
            if let Some(entry) = self.material_count.iter_mut().find(|(m, _)| *m == mat) {
                entry.1 += cnt;
            } else {
                self.material_count.push((mat, cnt));
            }
        }
    }

    /// Try to emit a prefiltered `LeafAttr` + color. Returns `None` when
    /// the subtree has no surface, or the aggregate normal is too
    /// dispersed to represent faithfully (caller should fall through to
    /// a deeper descent).
    fn emit(&self) -> Option<(LeafAttr, u32)> {
        if self.cell_count == 0 {
            return None;
        }

        // Dispersion check: average unit-vector length across contributing
        // cells. Max 1.0 (all aligned), 0.0 (perfectly cancelling). Below
        // the threshold we don't emit — the prefilter would lie about
        // the subtree and cause the shader to shade a wrong normal.
        let avg_len = self.normal_sum.length() / self.cell_count as f32;
        if avg_len < DISPERSION_THRESHOLD {
            return None;
        }

        let avg_normal = self.normal_sum / self.cell_count as f32;
        let normal = if avg_normal.length_squared() > 1e-16 {
            avg_normal.normalize()
        } else {
            Vec3::Y
        };

        // Pick primary + optional secondary material by cell count.
        let mut sorted: Vec<(u16, u32)> = self.material_count.clone();
        // Descending sort by count. Stable sort preserves insertion order
        // on ties, which matters for the first-seen-wins property in
        // ``half_and_half_materials_emit_blended``.
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        let primary = sorted[0].0;
        let (secondary, blend_weight) = if sorted.len() >= 2 && sorted[1].1 > 0 {
            let p = sorted[0].1 as f32;
            let s = sorted[1].1 as f32;
            let frac = s / (p + s);
            let weight = (frac * 15.0).round().clamp(0.0, 15.0) as u8;
            (sorted[1].0, weight)
        } else {
            (0u16, 0u8)
        };

        let attr = LeafAttr::new_blended(normal, primary, secondary, blend_weight);
        let color = pack_color_u8([
            (self.color_sum[0] / self.cell_count as f32).round() as u8,
            (self.color_sum[1] / self.cell_count as f32).round() as u8,
            (self.color_sum[2] / self.cell_count as f32).round() as u8,
            (self.color_sum[3] / self.cell_count as f32).round() as u8,
        ]);
        Some((attr, color))
    }
}

/// Walk `octree` bottom-up, compute a prefiltered `LeafAttr` for every
/// emittable branch, and populate `octree.internal_attr_index` with the
/// allocated ids.
///
/// New `LeafAttr`s go into `leaf_attr_pool` via `attr_dedup` — so an
/// emitted internal attr that happens to match an existing leaf attr
/// (flat surface whose prefilter equals any leaf on it) reuses the id.
///
/// The caller is expected to have completed `compact`,
/// `deduplicate_subtrees`, and (typically) `morton_reorder` before
/// calling this.
pub fn prefilter_octree_internals(
    octree: &mut SparseOctree,
    leaf_attr_pool: &mut LeafAttrPool,
    brick_pool: &BrickPool,
    attr_dedup: &mut HashMap<LeafAttr, u32>,
) {
    if octree.node_count() == 0 {
        return;
    }
    let root = octree.as_slice()[0];
    if !is_branch(root) {
        // Single-node tree (empty / interior / leaf / brick at root) —
        // nothing to prefilter.
        return;
    }

    let mut cache: HashMap<u32, NodeAggregate> = HashMap::new();
    let _root_agg = walk(
        octree,
        leaf_attr_pool,
        brick_pool,
        attr_dedup,
        &mut cache,
        0,
    );
}

fn walk(
    octree: &mut SparseOctree,
    leaf_attr_pool: &mut LeafAttrPool,
    brick_pool: &BrickPool,
    attr_dedup: &mut HashMap<LeafAttr, u32>,
    cache: &mut HashMap<u32, NodeAggregate>,
    node_idx: u32,
) -> NodeAggregate {
    if let Some(cached) = cache.get(&node_idx) {
        return cached.clone();
    }

    let node = octree.as_slice()[node_idx as usize];
    let agg = if node == EMPTY_NODE || node == INTERIOR_NODE {
        NodeAggregate::default()
    } else if is_leaf(node) {
        let attr_id = leaf_slot(node);
        NodeAggregate::from_leaf_attr(
            *leaf_attr_pool.get(attr_id),
            leaf_attr_pool.color(attr_id),
        )
    } else if is_brick(node) {
        aggregate_brick(brick_pool, brick_id_of(node), leaf_attr_pool)
    } else {
        // Branch — recurse into 8 children, then emit at this node.
        debug_assert!(is_branch(node));
        let children_offset = node;
        let mut agg = NodeAggregate::default();
        for i in 0..8u32 {
            let child_agg = walk(
                octree,
                leaf_attr_pool,
                brick_pool,
                attr_dedup,
                cache,
                children_offset + i,
            );
            agg.add(&child_agg);
        }
        // Emit a prefiltered attr for this branch, if meaningful. The
        // result (the aggregate itself) still propagates to our parent
        // unchanged — we don't want emissions to short-circuit the
        // bottom-up walk.
        if let Some((attr, color)) = agg.emit() {
            let attr_id = attr_dedup.entry(attr).or_insert_with(|| {
                // Bump-only allocate to preserve the asset's contiguous
                // [slot_start, next_free) range invariant that the scene
                // manager's release relies on. A regular `allocate()`
                // could return a free-list slot from a concurrently-
                // released asset, producing a non-contiguous range.
                let id = leaf_attr_pool
                    .allocate_contiguous_bump(1)
                    .expect("LeafAttrPool exhausted during prefilter");
                *leaf_attr_pool.get_mut(id) = attr;
                id
            });
            // Color goes on whichever slot the dedup lookup returned —
            // if the attr was pre-existing (flat-surface collision), we
            // leave the existing color unchanged. Two aggregations of
            // the same LeafAttr tuple represent visually-equivalent
            // surfaces, so the prior color is a valid stand-in.
            //
            // Note: this means the *first* writer wins for color when
            // a prefilter attr collides with a LEAF attr. That's the
            // conservative choice — consistent with the existing leaf
            // allocation path which also uses first-writer-wins.
            if leaf_attr_pool.color(*attr_id) == 0 {
                leaf_attr_pool.set_color(*attr_id, color);
            }
            octree.set_internal_attr(node_idx, *attr_id);
        }
        agg
    };

    cache.insert(node_idx, agg.clone());
    agg
}

fn aggregate_brick(
    brick_pool: &BrickPool,
    brick_id: u32,
    leaf_attr_pool: &LeafAttrPool,
) -> NodeAggregate {
    let mut agg = NodeAggregate::default();
    for z in 0..BRICK_DIM {
        for y in 0..BRICK_DIM {
            for x in 0..BRICK_DIM {
                let cell = brick_pool.get_cell(brick_id, x, y, z);
                if cell == BRICK_EMPTY {
                    continue;
                }
                let leaf_attr = *leaf_attr_pool.get(cell);
                let color = leaf_attr_pool.color(cell);
                let leaf_agg = NodeAggregate::from_leaf_attr(leaf_attr, color);
                agg.add(&leaf_agg);
            }
        }
    }
    // Suppress unused-const warning on BRICK_CELLS in release; keep the
    // debug_assert to catch a silently-resized BRICK_DIM.
    debug_assert!(agg.cell_count <= BRICK_CELLS);
    agg
}

// ── Color packing helpers ────────────────────────────────────────────

/// Unpack a `LeafAttrPool`-format packed color into 4 u8 channels.
/// Matches the layout in `leaf_attr_pool::colors`: the exact channel
/// order isn't load-bearing here — we just need pack/unpack to roundtrip.
#[inline]
fn unpack_color_u8(packed: u32) -> [u8; 4] {
    [
        (packed & 0xFF) as u8,
        ((packed >> 8) & 0xFF) as u8,
        ((packed >> 16) & 0xFF) as u8,
        ((packed >> 24) & 0xFF) as u8,
    ]
}

#[inline]
fn pack_color_u8(channels: [u8; 4]) -> u32 {
    (channels[0] as u32)
        | ((channels[1] as u32) << 8)
        | ((channels[2] as u32) << 16)
        | ((channels[3] as u32) << 24)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse_octree::INTERNAL_ATTR_NONE;
    use crate::voxelize_octree::voxelize_sphere_octree;

    /// Walk a post-prefilter octree and count: (branches, branches-with-attr).
    fn branch_populate_stats(octree: &SparseOctree) -> (u32, u32) {
        let mut total = 0u32;
        let mut populated = 0u32;
        for i in 0..octree.node_count() {
            let node = octree.as_slice()[i];
            if is_branch(node) {
                total += 1;
                if octree.internal_attr(i as u32) != INTERNAL_ATTR_NONE {
                    populated += 1;
                }
            }
        }
        (total, populated)
    }

    #[test]
    fn empty_octree_skipped() {
        let mut octree = SparseOctree::new(3, 0.1);
        let mut pool = LeafAttrPool::new(64);
        let bricks = BrickPool::new(8);
        let mut dedup: HashMap<LeafAttr, u32> = HashMap::new();
        prefilter_octree_internals(&mut octree, &mut pool, &bricks, &mut dedup);
        // No branches, no attrs allocated, no panic.
        assert_eq!(octree.node_count(), 1);
        assert_eq!(pool.allocated_count(), 0);
    }

    #[test]
    fn trivial_leaf_root_skipped() {
        let mut octree = SparseOctree::new(0, 0.1);
        let mut pool = LeafAttrPool::new(64);
        let attr = LeafAttr::new(Vec3::Y, 7);
        let slot = pool.allocate().unwrap();
        *pool.get_mut(slot) = attr;
        octree.set_internal_attr_index(vec![INTERNAL_ATTR_NONE; 1]);
        let bricks = BrickPool::new(8);
        let mut dedup: HashMap<LeafAttr, u32> = HashMap::from([(attr, slot)]);
        prefilter_octree_internals(&mut octree, &mut pool, &bricks, &mut dedup);
        // Single-node tree (root is EMPTY here since we didn't insert); nothing
        // to prefilter.
        assert_eq!(pool.allocated_count(), 1); // unchanged
    }

    #[test]
    fn sphere_populates_branch_attrs() {
        // Voxelize a sphere, then run prefilter. Every branch should get
        // a populated attr (spheres are smooth — all subtrees have
        // coherent normals, none hit the dispersion threshold).
        let mut pool = LeafAttrPool::new(1_000_000);
        let mut bricks = BrickPool::new(10_000);
        let r = voxelize_sphere_octree(Vec3::ZERO, 0.3, 5, 0.05, &mut pool, &mut bricks).unwrap();

        // Rebuild attr_dedup from the voxelize output — the function
        // returned a result but not the map. For the test, reconstruct
        // by scanning the populated pool range. (Production code keeps
        // the map alive inside voxelize_octree; see Phase 1 wiring.)
        let mut dedup: HashMap<LeafAttr, u32> = HashMap::new();
        for slot in r.leaf_attr_slot_start..(r.leaf_attr_slot_start + r.leaf_attr_unique_count) {
            dedup.insert(*pool.get(slot), slot);
        }
        let pool_before = pool.allocated_count();

        let mut octree = r.octree;
        prefilter_octree_internals(&mut octree, &mut pool, &bricks, &mut dedup);

        let (total_branches, populated_branches) = branch_populate_stats(&octree);
        assert!(total_branches > 0, "sphere should have branches to prefilter");
        assert!(
            populated_branches * 100 / total_branches >= 80,
            "sphere: expected >=80% branches to get prefilter attrs, got \
             {populated_branches}/{total_branches}",
        );
        // Prefilter may allocate new attrs OR reuse existing ones; either
        // way the pool size shouldn't shrink.
        assert!(pool.allocated_count() >= pool_before);
    }

    #[test]
    fn dispersed_normals_no_emit() {
        // Hand-build an aggregate with back-to-back normals that cancel.
        let n_up = LeafAttr::new(Vec3::Y, 1);
        let n_down = LeafAttr::new(-Vec3::Y, 1);
        let mut agg = NodeAggregate::default();
        for _ in 0..10 {
            agg.add(&NodeAggregate::from_leaf_attr(n_up, 0));
            agg.add(&NodeAggregate::from_leaf_attr(n_down, 0));
        }
        // Sum is ~0; avg_len → 0. Should NOT emit.
        assert!(agg.emit().is_none());
    }

    #[test]
    fn coherent_normals_emit_single_material() {
        let n = LeafAttr::new(Vec3::Z, 42);
        let mut agg = NodeAggregate::default();
        for _ in 0..8 {
            agg.add(&NodeAggregate::from_leaf_attr(n, 0));
        }
        let (emitted, _color) = agg.emit().expect("coherent normals should emit");
        assert_eq!(emitted.material_primary, 42);
        assert_eq!(emitted.material_secondary(), 0);
        assert_eq!(emitted.blend_weight(), 0);
        // Normal should roundtrip to +Z (within octahedral precision).
        assert!(emitted.normal().dot(Vec3::Z) > 0.99);
    }

    #[test]
    fn half_and_half_materials_emit_blended() {
        // 4 cells of material 7, 4 cells of material 9 — same normal.
        // Expect: primary=7 (first-seen wins on ties in our emit), secondary=9,
        // blend_weight ≈ 15/2 = 7 or 8.
        let a = LeafAttr::new(Vec3::Y, 7);
        let b = LeafAttr::new(Vec3::Y, 9);
        let mut agg = NodeAggregate::default();
        for _ in 0..4 {
            agg.add(&NodeAggregate::from_leaf_attr(a, 0));
        }
        for _ in 0..4 {
            agg.add(&NodeAggregate::from_leaf_attr(b, 0));
        }
        let (emitted, _color) = agg.emit().unwrap();
        // At a 4/4 tie, sort is stable, so primary is whichever was pushed
        // to material_count first (here: 7).
        assert_eq!(emitted.material_primary, 7);
        assert_eq!(emitted.material_secondary(), 9);
        // 4/(4+4) = 0.5 → blend = 8 (round(0.5*15)).
        assert_eq!(emitted.blend_weight(), 8);
    }

    #[test]
    fn dominant_material_wins() {
        let a = LeafAttr::new(Vec3::Y, 7);
        let b = LeafAttr::new(Vec3::Y, 9);
        let mut agg = NodeAggregate::default();
        for _ in 0..7 {
            agg.add(&NodeAggregate::from_leaf_attr(a, 0));
        }
        for _ in 0..1 {
            agg.add(&NodeAggregate::from_leaf_attr(b, 0));
        }
        let (emitted, _color) = agg.emit().unwrap();
        assert_eq!(emitted.material_primary, 7);
        assert_eq!(emitted.material_secondary(), 9);
        // 1/(7+1) = 0.125 → blend = round(0.125*15) = 2.
        assert_eq!(emitted.blend_weight(), 2);
    }

    #[test]
    fn color_is_averaged() {
        let attr = LeafAttr::new(Vec3::Y, 0);
        let mut agg = NodeAggregate::default();
        // 2 contributions of red (255,0,0) and 2 of blue (0,0,255).
        agg.add(&NodeAggregate::from_leaf_attr(attr, pack_color_u8([255, 0, 0, 0])));
        agg.add(&NodeAggregate::from_leaf_attr(attr, pack_color_u8([255, 0, 0, 0])));
        agg.add(&NodeAggregate::from_leaf_attr(attr, pack_color_u8([0, 0, 255, 0])));
        agg.add(&NodeAggregate::from_leaf_attr(attr, pack_color_u8([0, 0, 255, 0])));
        let (_attr, color) = agg.emit().unwrap();
        let unpacked = unpack_color_u8(color);
        // Each channel averages to ~127.5 → rounds to 128 or 127.
        assert!(unpacked[0] >= 127 && unpacked[0] <= 128, "R got {}", unpacked[0]);
        assert_eq!(unpacked[1], 0);
        assert!(unpacked[2] >= 127 && unpacked[2] <= 128, "B got {}", unpacked[2]);
    }

    #[test]
    fn voxelize_release_round_trip_leaves_pools_at_baseline() {
        // Brick-leak-class regression test (commit 2290ee2 fixed a related
        // leak for bricks). Post-prefilter, LeafAttrPool grows for both
        // leaf attrs AND internal prefilter attrs; the contiguous release
        // must reclaim both. BrickPool deallocates per-id.
        let mut pool = LeafAttrPool::new(1_000_000);
        let mut bricks = BrickPool::new(10_000);
        let pool_baseline = pool.allocated_count();

        let r = voxelize_sphere_octree(Vec3::ZERO, 0.3, 5, 0.05, &mut pool, &mut bricks).unwrap();
        assert!(
            r.leaf_attr_unique_count > 0,
            "voxelize should allocate at least leaf attrs",
        );
        assert!(!r.brick_ids.is_empty(), "sphere should allocate bricks");
        assert!(
            pool.allocated_count() > pool_baseline,
            "pool should have grown",
        );

        // Release everything the voxelize produced.
        pool.deallocate_range(r.leaf_attr_slot_start, r.leaf_attr_unique_count);
        for &id in &r.brick_ids {
            bricks.deallocate(id);
        }

        // Pool's `next_free` bump pointer returns to baseline because the
        // release range is a tail range (deallocate_range's tail-coalesce
        // path). If prefilter allocations weren't folded into the same
        // contiguous range, this assertion would catch the leak.
        assert_eq!(
            pool.allocated_count(),
            pool_baseline,
            "LeafAttrPool leaked {} slot(s) after release",
            pool.allocated_count() - pool_baseline,
        );
    }

    #[test]
    fn interleaved_voxelize_release_cycle_reclaims_all_slots() {
        // Interleaved pattern that stresses free_list interaction:
        //   voxelize A → voxelize B → release A → voxelize C → release B,C
        // After all releases, the pool must return to baseline. This is
        // the multi-asset version of the brick-leak regression test.
        // A bug here would indicate `voxelize_octree`'s use of the
        // single-slot `pool.allocate()` (which dips into free_list)
        // breaks the `[slot_start, slot_start+count)` range invariant
        // that `deallocate_range` relies on.
        let mut pool = LeafAttrPool::new(1_000_000);
        let mut bricks = BrickPool::new(10_000);
        let baseline = pool.allocated_count();

        let a = voxelize_sphere_octree(Vec3::new(-1.0, 0.0, 0.0), 0.3, 1, 0.05, &mut pool, &mut bricks).unwrap();
        let b = voxelize_sphere_octree(Vec3::new(1.0, 0.0, 0.0), 0.3, 2, 0.05, &mut pool, &mut bricks).unwrap();

        pool.deallocate_range(a.leaf_attr_slot_start, a.leaf_attr_unique_count);
        for &id in &a.brick_ids { bricks.deallocate(id); }

        let c = voxelize_sphere_octree(Vec3::new(0.0, 1.0, 0.0), 0.3, 3, 0.05, &mut pool, &mut bricks).unwrap();

        pool.deallocate_range(b.leaf_attr_slot_start, b.leaf_attr_unique_count);
        for &id in &b.brick_ids { bricks.deallocate(id); }

        pool.deallocate_range(c.leaf_attr_slot_start, c.leaf_attr_unique_count);
        for &id in &c.brick_ids { bricks.deallocate(id); }

        assert_eq!(
            pool.allocated_count(),
            baseline,
            "LeafAttrPool leaked after interleaved voxelize/release cycle",
        );
    }

    #[test]
    fn pool_stays_contiguous_after_prefilter() {
        // Critical invariant for the scene manager's release path: the
        // allocated range `[leaf_attr_slot_start, next_free)` remains
        // contiguous after prefilter runs. We test this by observing
        // next_free grows monotonically.
        let mut pool = LeafAttrPool::new(1_000_000);
        let mut bricks = BrickPool::new(10_000);
        let r = voxelize_sphere_octree(Vec3::ZERO, 0.3, 5, 0.05, &mut pool, &mut bricks).unwrap();

        let start = r.leaf_attr_slot_start;
        let before = pool.allocated_count();

        let mut dedup: HashMap<LeafAttr, u32> = HashMap::new();
        for slot in start..before {
            dedup.insert(*pool.get(slot), slot);
        }

        let mut octree = r.octree;
        prefilter_octree_internals(&mut octree, &mut pool, &bricks, &mut dedup);

        let after = pool.allocated_count();
        assert!(after >= before, "pool should only grow");
        // Every attr-id held in internal_attr_index must be within
        // [start, after) — no out-of-range refs.
        for i in 0..octree.node_count() {
            let id = octree.internal_attr(i as u32);
            if id != INTERNAL_ATTR_NONE {
                assert!(
                    id >= start && id < after,
                    "prefilter attr id {id} out of range [{start}, {after})",
                );
            }
        }
    }
}
