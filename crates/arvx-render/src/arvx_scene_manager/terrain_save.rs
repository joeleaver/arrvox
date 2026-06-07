//! Reverse of `integrate_baked_tile`: walk an `AssetEntry`'s live
//! scene-pool state and produce a self-contained `BakeArtifact` whose
//! leaf-attr / brick / halo IDs are file-local. The artifact is
//! what `arvx_core::asset_file::write_artifact_rkp` wants on disk.
//!
//! Used by Phase 4.3 (`.arvxtile` save). The persisted file is an
//! ordinary `.arvx` v6 — terrain tiles deliberately share the asset
//! format so the engine's existing load path can bring them back
//! without a parallel codec.
//!
//! Note: this path runs on the engine save thread under the scene
//! manager lock. Cost scales with the tile's voxel/brick count, not
//! with the scene; a sculpted terrain tile at Tier 2 is ~500 k tris
//! and the artifact build is sub-second.

use arvx_core::asset_file::MeshSectionsBlob;
use arvx_core::brick_face_links::{FACE_EMPTY, FACE_INTERIOR};
use arvx_core::brick_pool::{BRICK_CELLS, BRICK_EMPTY, BRICK_INTERIOR};
use arvx_core::mesh_extract::CELL_INTERIOR;
use arvx_core::sparse_octree::{
    brick_id as node_brick_id, is_brick, is_leaf, leaf_slot as node_leaf_slot,
    make_brick, make_leaf, SparseOctree, INTERNAL_ATTR_NONE,
};
use arvx_core::{BakeArtifact, LeafAttr};

use super::manager::ArvxSceneManager;
use super::types::AssetHandle;

impl ArvxSceneManager {
    /// Build a self-contained [`BakeArtifact`] from a live asset
    /// entry. The artifact's leaf-attr / brick / halo IDs are
    /// file-local (worker-frame), matching what the existing on-disk
    /// `.arvx` format expects and what `integrate_baked_tile`
    /// re-relocates on load.
    ///
    /// Returns `None` if the handle is unknown.
    ///
    /// Cost: O(asset_node_count + brick_count * BRICK_CELLS). The
    /// allocations live entirely inside the returned artifact;
    /// scene-pool state is read-only.
    pub fn extract_artifact_from_handle(
        &self,
        handle: AssetHandle,
    ) -> Option<BakeArtifact> {
        let entry = self.asset_cache.get(handle)?;
        let leaf_attr_slot_start = entry.model.leaf_attr_slot_start;
        let leaf_attr_slot_count = entry.model.leaf_attr_slot_count;
        let brick_start = entry.model.brick_start;
        let brick_count = entry.model.brick_count;

        // ── leaf_attrs + leaf_attr_colors ─────────────────────────
        let mut leaf_attrs: Vec<LeafAttr> =
            Vec::with_capacity(leaf_attr_slot_count as usize);
        let mut leaf_attr_colors: Vec<u32> =
            Vec::with_capacity(leaf_attr_slot_count as usize);
        for i in 0..leaf_attr_slot_count {
            let slot = leaf_attr_slot_start + i;
            leaf_attrs.push(*self.leaf_attr_pool.get(slot));
            leaf_attr_colors.push(self.leaf_attr_pool.color(slot));
        }

        // ── brick_cells (remap scene → file-local) ────────────────
        let bcells = BRICK_CELLS as usize;
        let mut brick_cells_out: Vec<[u32; BRICK_CELLS as usize]> =
            Vec::with_capacity(brick_count as usize);
        for i in 0..brick_count {
            let scene_id = brick_start + i;
            let src = self.brick_pool.brick_cells(scene_id);
            let mut arr = [BRICK_EMPTY; BRICK_CELLS as usize];
            debug_assert_eq!(src.len(), bcells);
            for (d, &s) in arr.iter_mut().zip(src.iter()) {
                *d = if s == BRICK_EMPTY || s == BRICK_INTERIOR {
                    s
                } else {
                    // Scene-global leaf_attr slot → file-local.
                    s.saturating_sub(leaf_attr_slot_start)
                };
            }
            brick_cells_out.push(arr);
        }

        // ── brick_face_links (remap brick refs) ───────────────────
        let mut brick_face_links_out: Vec<[u32; 6]> =
            Vec::with_capacity(brick_count as usize);
        for i in 0..brick_count {
            let scene_id = (brick_start + i) as usize;
            let src = if scene_id < self.brick_face_links.len() {
                self.brick_face_links[scene_id]
            } else {
                [FACE_EMPTY; 6]
            };
            let mut row = [FACE_EMPTY; 6];
            for (k, &s) in src.iter().enumerate() {
                row[k] = if s == FACE_EMPTY || s == FACE_INTERIOR {
                    s
                } else {
                    s.saturating_sub(brick_start)
                };
            }
            brick_face_links_out.push(row);
        }

        // ── Octree (clone + remap leaf/brick refs in every node) ──
        let mut nodes = entry.model.cpu_octree.as_slice().to_vec();
        for node in nodes.iter_mut() {
            let v = *node;
            if is_leaf(v) {
                let scene_slot = node_leaf_slot(v);
                *node = make_leaf(scene_slot.saturating_sub(leaf_attr_slot_start));
            } else if is_brick(v) {
                let scene_brick = node_brick_id(v);
                *node = make_brick(scene_brick.saturating_sub(brick_start));
            }
        }
        let mut octree = SparseOctree::from_raw(
            &nodes,
            entry.model.cpu_octree.depth(),
            entry.model.cpu_octree.base_voxel_size(),
        );
        // Remap internal_attr_index (prefilter): same shift on every
        // non-sentinel entry.
        let internal: Vec<u32> = entry.model
            .cpu_octree
            .internal_attr_slice()
            .iter()
            .map(|&v| {
                if v == INTERNAL_ATTR_NONE {
                    v
                } else {
                    v.saturating_sub(leaf_attr_slot_start)
                }
            })
            .collect();
        octree.set_internal_attr_index(internal);

        // ── Halo cells (remap LeafAttr slots) ─────────────────────
        let halo_cells = entry.model
            .halo_cells
            .iter()
            .map(|&(coord, slot)| {
                let relocated = if slot == CELL_INTERIOR {
                    slot
                } else {
                    slot.saturating_sub(leaf_attr_slot_start)
                };
                (coord, relocated)
            })
            .collect();

        // ── grid_origin ───────────────────────────────────────────
        // AssetEntry's aabb is the tile's world-space AABB; grid
        // origin = aabb.min (same as the integrate path's reverse).
        let grid_origin = entry.model.aabb.min;

        Some(BakeArtifact {
            octree,
            voxel_count: entry.model.voxel_count,
            grid_origin,
            leaf_attrs,
            leaf_attr_colors,
            brick_cells: brick_cells_out,
            brick_face_links: brick_face_links_out,
            halo_cells,
        })
    }

    /// Build the mesh-sections blob for a given handle. Phase 4.3
    /// piggybacks on this so the saved `.arvxtile` ships with its
    /// baked surface mesh + cluster DAG — matching every other v6
    /// `.arvx` on disk. The load path treats it identically.
    ///
    /// Unlike `extract_artifact_from_handle`, this samples mesh data
    /// straight from the asset entry; sculpt edits already populated
    /// `mesh_vertices` / `mesh_indices` / `meshlet_clusters` in place
    /// during the sculpt path. We just need to convert the vertices'
    /// scene-global leaf_attr_ids back to file-local for portability.
    pub fn extract_mesh_blob_from_handle(
        &self,
        handle: AssetHandle,
    ) -> Option<MeshSectionsBlob> {
        let entry = self.asset_cache.get(handle)?;
        let leaf_attr_slot_start = entry.model.leaf_attr_slot_start;

        let mut mesh_vertices = entry.view.mesh_vertices.clone();
        for v in &mut mesh_vertices {
            v.leaf_attr_id = v.leaf_attr_id.saturating_sub(leaf_attr_slot_start);
        }

        // The full mesh_indices buffer includes both LOD-0 and the
        // higher LODs that the DAG bake produced. We pass it through
        // untouched — load relocates leaf_attr_id alongside.
        let mesh_indices = entry.view.mesh_indices.clone();
        let meshlet_clusters = entry.view.meshlet_clusters.clone();
        let dag_groups = entry.view.dag_groups.clone();
        let dag_consumed = entry.view.dag_consumed.clone();
        let dag_produced = entry.view.dag_produced.clone();
        let lod0_index_count = entry.view.mesh_lod0_index_count;

        Some(MeshSectionsBlob {
            vertices: bytemuck::cast_slice(&mesh_vertices).to_vec(),
            indices: bytemuck::cast_slice(&mesh_indices).to_vec(),
            clusters: bytemuck::cast_slice(&meshlet_clusters).to_vec(),
            dag_groups: bytemuck::cast_slice(&dag_groups).to_vec(),
            dag_consumed: bytemuck::cast_slice(&dag_consumed).to_vec(),
            dag_produced: bytemuck::cast_slice(&dag_produced).to_vec(),
            lod0_index_count,
        })
    }
}
