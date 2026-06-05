//! `integrate_baked_tile` — the in-memory analogue of
//! `load_asset_from_disk` for streamed terrain tiles.
//!
//! A terrain tile is "just an arvx asset under streaming control"
//! (docs/TERRAIN.md). The streamer's worker thread produces a
//! `BakeArtifact + MeshSectionsBlob` pair via `arvx_terrain::bake_tile`;
//! this method takes that pair on the main thread (inside the scene
//! manager lock) and threads it into the same scene-global pools and
//! mesh / cluster bookkeeping that disk-loaded assets use. The result
//! is an [`AssetHandle`] that downstream code (per-frame GPU upload,
//! mesh raster, mesh shadow, paint, sculpt) treats identically to any
//! other loaded asset.
//!
//! Compared to `load_asset_from_disk` this path:
//!
//! - Skips file IO + header parsing (we already have parsed structs).
//! - Skips the legacy v4 mesh-fallback branch (terrain always ships v6).
//! - Skips skinning (terrain is never skinned).
//! - Skips the on-load prefilter pass (the artifact already carries
//!   `internal_attr_index` from `voxelize_to_artifact`).
//!
//! The pool-relocation invariants are identical to disk-load:
//! contiguous-bump leaf-attrs, contiguous-bump bricks (so the existing
//! `release_asset` path frees the tile cleanly), per-LOD cluster-error
//! normalization (load-bearing — Karis admit chain consistency depends
//! on it), and a fresh `ClusterSpatialIndex` build.

use std::path::PathBuf;

use arvx_core::asset_file::MeshSectionsBlob;
use arvx_core::BakeArtifact;

use super::manager::ArvxSceneManager;
use super::types::{AssetEntry, AssetHandle, AssetInfo, MESH_INDEX_STRIDE};

impl ArvxSceneManager {
    /// Integrate a `(BakeArtifact, MeshSectionsBlob)` pair produced by
    /// `arvx_terrain::bake_tile` into the shared scene pools, register
    /// the mesh sections as a renderable asset, and return the handle.
    ///
    /// Caller-supplied `synthetic_path` keys the entry in
    /// `path_to_handle`. The convention is
    /// `PathBuf::from(format!("terrain://{level}_{x}_{y}_{z}"))`; the
    /// `terrain://` scheme prefix guarantees no collision with disk
    /// paths. The path is otherwise opaque — there's no file behind it.
    ///
    /// Returns `None` on pool exhaustion (contiguous-bump fail on
    /// leaf-attrs / bricks). Allocations made before the failing step
    /// are leaked at most until the next streamer tick reclaims them
    /// by retrying — the streamer treats a `None` return as
    /// "materialise failed, mark tile state back to Unmaterialised".
    /// Tightening that is V2.
    pub fn integrate_baked_tile(
        &mut self,
        mut artifact: BakeArtifact,
        mesh: MeshSectionsBlob,
        aabb: arvx_core::Aabb,
        voxel_size: f32,
        synthetic_path: PathBuf,
    ) -> Option<(AssetHandle, AssetInfo)> {
        use arvx_core::brick_face_links::{FACE_EMPTY, FACE_INTERIOR};
        use arvx_core::brick_pool::{BRICK_EMPTY, BRICK_INTERIOR, BRICK_CELLS, BRICK_DIM};
        use arvx_core::sparse_octree::{
            brick_id as node_brick_id, is_brick, is_leaf, leaf_slot as node_leaf_slot,
            make_brick, make_leaf, INTERNAL_ATTR_NONE,
        };

        self.bump_geometry_epoch();
        let t_start = std::time::Instant::now();

        // ── Leaf-attr pool ─────────────────────────────────────────────
        let n_attrs = artifact.leaf_attrs.len() as u32;
        let leaf_attr_slot_start = self
            .leaf_attr_pool
            .allocate_contiguous_bump(n_attrs)?;
        for (i, attr) in artifact.leaf_attrs.iter().enumerate() {
            let scene_id = leaf_attr_slot_start + i as u32;
            *self.leaf_attr_pool.get_mut(scene_id) = *attr;
            let color = artifact.leaf_attr_colors[i];
            if color != 0 {
                self.leaf_attr_pool.set_color(scene_id, color);
            }
        }

        // ── Brick pool ─────────────────────────────────────────────────
        // Contiguous-bump (mirrors disk-load) so the existing
        // `release_asset` brick-range iteration cleans up correctly.
        // Procedural integrate_artifact uses per-brick `allocate()`
        // because it tracks brick ids per-instance in `SpatialData`;
        // terrain doesn't need that — every tile is its own
        // `AssetHandle` and releases as a unit.
        let n_bricks = artifact.brick_cells.len() as u32;
        let brick_start = self
            .brick_pool
            .allocate_contiguous_bump(n_bricks)?;

        for (worker_id, cells) in artifact.brick_cells.iter().enumerate() {
            let scene_id = brick_start + worker_id as u32;
            let dst = self.brick_pool.brick_cells_mut(scene_id);
            debug_assert_eq!(dst.len(), cells.len());
            debug_assert_eq!(BRICK_CELLS as usize, cells.len());
            for (d, &c) in dst.iter_mut().zip(cells.iter()) {
                *d = if c == BRICK_EMPTY || c == BRICK_INTERIOR {
                    c
                } else {
                    leaf_attr_slot_start + c
                };
            }
        }

        // ── Octree node remap ──────────────────────────────────────────
        {
            let nodes = artifact.octree.as_slice_mut();
            for node in nodes.iter_mut() {
                let v = *node;
                if is_leaf(v) {
                    let worker_slot = node_leaf_slot(v);
                    *node = make_leaf(leaf_attr_slot_start + worker_slot);
                } else if is_brick(v) {
                    let worker_id = node_brick_id(v);
                    // Worker bricks are dense 0..n_bricks; contiguous
                    // scene allocation means scene_id = brick_start +
                    // worker_id (no per-brick map needed).
                    *node = make_brick(brick_start + worker_id);
                }
            }
        }

        // Prefiltered internal-attr index — relocate by the same
        // leaf-attr offset, identically to `integrate_artifact`.
        {
            let old = artifact.octree.internal_attr_slice().to_vec();
            let new: Vec<u32> = old
                .into_iter()
                .map(|v| if v == INTERNAL_ATTR_NONE {
                    v
                } else {
                    leaf_attr_slot_start + v
                })
                .collect();
            artifact.octree.set_internal_attr_index(new);
        }

        // ── Brick face links ───────────────────────────────────────────
        if n_bricks > 0 {
            let max_scene_brick = brick_start + n_bricks - 1;
            let mut scene_rows: Vec<[u32; 6]> =
                vec![[FACE_EMPTY; 6]; (max_scene_brick + 1) as usize];
            for (worker_id, row) in artifact.brick_face_links.iter().enumerate() {
                if worker_id as u32 >= n_bricks {
                    break;
                }
                let scene_id = brick_start + worker_id as u32;
                let mut remapped = [FACE_EMPTY; 6];
                for (i, &neighbor) in row.iter().enumerate() {
                    remapped[i] = if neighbor == FACE_EMPTY || neighbor == FACE_INTERIOR {
                        neighbor
                    } else {
                        // Worker neighbor IDs are also dense 0..n_bricks.
                        brick_start + neighbor
                    };
                }
                scene_rows[scene_id as usize] = remapped;
            }
            self.merge_face_links(&scene_rows);
        }

        // ── Octree allocation ──────────────────────────────────────────
        let handle = self.octree.allocate(&artifact.octree);
        let t_octree_alloc = t_start.elapsed();

        // ── Mesh sections ──────────────────────────────────────────────
        // The blob's `Vec<u8>` payloads come from
        // `bytemuck::cast_slice(&typed).to_vec()` in `build_mesh_sections_blob`
        // — which produces a byte-aligned Vec that the allocator
        // may hand back at any odd address. `bytemuck::cast_slice::<u8, T>`
        // requires alignment to `align_of::<T>()` and panics with
        // `TargetAlignmentGreaterAndInputNotAligned` when the allocator
        // hands us a misaligned start. Use `pod_read_unaligned` per
        // stride instead — guarantees correctness regardless of allocator
        // luck. The cost is one extra memcpy per element, negligible
        // against the seconds-long bake.
        use arvx_core::mesh_cluster::PARENT_GROUP_ERROR_ROOT;
        use arvx_core::mesh_lod::DagGroup;
        use crate::mesh_pass::{MeshVertex, MeshletCluster};

        fn read_pod_vec<T: bytemuck::Pod>(bytes: &[u8]) -> Vec<T> {
            // Fast path: if the allocator happened to give us a
            // sufficiently-aligned Vec, do a single bulk cast + clone.
            if let Ok(slice) = bytemuck::try_cast_slice::<u8, T>(bytes) {
                return slice.to_vec();
            }
            let stride = std::mem::size_of::<T>();
            if stride == 0 {
                return Vec::new();
            }
            bytes
                .chunks_exact(stride)
                .map(bytemuck::pod_read_unaligned::<T>)
                .collect()
        }

        let mut mesh_vertices: Vec<MeshVertex> = read_pod_vec(&mesh.vertices);
        let mesh_indices: Vec<u32> = read_pod_vec(&mesh.indices);
        let mut meshlet_clusters: Vec<MeshletCluster> =
            read_pod_vec(&mesh.clusters);
        let dag_groups: Vec<DagGroup> = read_pod_vec(&mesh.dag_groups);
        let dag_consumed: Vec<u32> = read_pod_vec(&mesh.dag_consumed);
        let dag_produced: Vec<u32> = read_pod_vec(&mesh.dag_produced);

        // Per-LOD cluster-error normalisation — mirrors
        // `load_asset_from_disk` (the rationale is the Karis-Nanite
        // chain-consistency requirement; without it, adjacent
        // clusters at different LODs pick mismatched levels and the
        // mesh tears at group boundaries on multi-level admit).
        if !meshlet_clusters.is_empty() {
            let mut max_level = 0u32;
            for c in &meshlet_clusters {
                if c.lod_level > max_level {
                    max_level = c.lod_level;
                }
            }
            let mut level_max_error: Vec<f32> = vec![0.0; max_level as usize + 1];
            for c in &meshlet_clusters {
                let l = c.lod_level as usize;
                if c.cluster_error > level_max_error[l] {
                    level_max_error[l] = c.cluster_error;
                }
            }
            for c in &mut meshlet_clusters {
                let l = c.lod_level as usize;
                if c.cluster_error != 0.0 {
                    c.cluster_error = level_max_error[l];
                }
                if c.parent_group_error < PARENT_GROUP_ERROR_ROOT * 0.5 {
                    let next_l = (c.lod_level + 1) as usize;
                    if next_l <= max_level as usize {
                        c.parent_group_error = level_max_error[next_l];
                    }
                }
            }
        }

        // Vertex leaf_attr_id was file-local against `artifact.leaf_attrs`;
        // shift to scene-global. Same relocation as
        // `load_asset_from_disk` lines 625-635 (terrain has no bones,
        // so the file-bone merge step above it is omitted).
        if leaf_attr_slot_start > 0 && !mesh_vertices.is_empty() {
            for v in &mut mesh_vertices {
                v.leaf_attr_id += leaf_attr_slot_start;
            }
        }

        // ── Halo cells ─────────────────────────────────────────────────
        // Phase 3 baked halo cell coords + their file-local
        // leaf_attr_id slots. Relocate the slot ids by
        // `leaf_attr_slot_start` (same shift the leaf/brick nodes get
        // above) so the stored map references the asset's scene-pool
        // slots. `CELL_INTERIOR` halo cells stay as-is — they're a
        // sentinel, not an index.
        let halo_cells: Vec<(glam::IVec3, u32)> = artifact
            .halo_cells
            .iter()
            .map(|&(coord, slot)| {
                let relocated = if slot == arvx_core::mesh_extract::CELL_INTERIOR {
                    slot
                } else {
                    leaf_attr_slot_start + slot
                };
                (coord, relocated)
            })
            .collect();

        // ── Cluster spatial index ──────────────────────────────────────
        let mut cluster_spatial_index =
            super::cluster_spatial_index::ClusterSpatialIndex::new();
        {
            let extent_f = (1u32 << handle.depth) as f32 * voxel_size;
            let aabb_center = (aabb.min + aabb.max) * 0.5;
            let grid_origin = aabb_center - glam::Vec3::splat(extent_f * 0.5);
            cluster_spatial_index.rebuild(
                &meshlet_clusters,
                grid_origin,
                voxel_size,
            );
        }

        // ── Slab-allocator state for the mesh-indices buffer ───────────
        let mesh_indices_next_free = mesh_indices.len() as u32;
        let mut mesh_indices_dirty = arvx_core::DirtyRanges::new();
        let total_bytes = (mesh_indices.len() as u32)
            .saturating_mul(MESH_INDEX_STRIDE);
        if total_bytes > 0 {
            mesh_indices_dirty.mark_full(total_bytes);
        }
        // Phase 5.6 fix: mark the VBO fully dirty so the renderer's
        // tail-only-append optimisation doesn't reuse stale prefix
        // bytes from a previous asset at the same recycled handle.
        // Terrain stamp move evicts + re-integrates tiles continuously,
        // and the freed handle slot's GPU VBO holds the previous
        // asset's vertices until we mark them all dirty.
        let mut mesh_vertices_dirty = arvx_core::DirtyRanges::new();
        let vbo_total_bytes = (mesh_vertices.len()
            * std::mem::size_of::<crate::mesh_pass::MeshVertex>()) as u32;
        if vbo_total_bytes > 0 {
            mesh_vertices_dirty.mark_full(vbo_total_bytes);
        }

        // ── Construct + insert the cache entry ─────────────────────────
        let actual_cell_count = artifact.voxel_count;
        // Suppress unused-variable warning until Phase 4 wires
        // brick-dim into terrain sculpt; the asserts above guarantee
        // the layout matches.
        let _ = BRICK_DIM;

        let entry = AssetEntry {
            path: synthetic_path,
            refcount: 1,
            spatial_handle: handle,
            voxel_size,
            aabb,
            voxel_count: actual_cell_count,
            leaf_attr_slot_start,
            leaf_attr_slot_count: n_attrs,
            brick_start,
            brick_count: n_bricks,
            skinning: None,
            mesh_vertices,
            mesh_indices,
            mesh_indices_free_list: Vec::new(),
            mesh_indices_next_free,
            mesh_indices_dirty,
            mesh_vertices_dirty,
            mesh_lod0_index_count: mesh.lod0_index_count,
            bake_time_cluster_count: meshlet_clusters.len() as u32,
            meshlet_clusters,
            dag_groups,
            dag_consumed,
            dag_produced,
            cpu_octree: artifact.octree,
            mesh_dirty: true,
            clusters_dirty: true,
            cluster_spatial_index,
            sculpt_extra_slots: std::collections::HashSet::new(),
            sculpt_owned_slots: rustc_hash::FxHashSet::default(),
            halo_extra_slots: std::collections::HashSet::new(),
            halo_cells,
            // Terrain tiles don't precompute a distinct-material set, so
            // the engine falls back to the per-leaf walk for them (they're
            // opaque, so that correctly reports no glass).
            distinct_materials: None,
        };

        let info = entry.info();
        let asset_handle = self.asset_cache.insert(entry);

        if std::env::var("ARVX_TERRAIN_DEBUG").is_ok() {
            let ms = |d: std::time::Duration| d.as_secs_f32() * 1000.0;
            eprintln!(
                "[integrate_baked_tile] handle={} voxels={} bricks={} attrs={} verts={} \
                 tris={} clusters={} total={:.2}ms",
                asset_handle.raw(),
                actual_cell_count,
                n_bricks,
                n_attrs,
                info.voxel_count,
                self.asset_cache_get_index_count(asset_handle) / 3,
                self.asset_cache_get_cluster_count(asset_handle),
                ms(t_octree_alloc),
            );
        }

        Some((asset_handle, info))
    }

    /// Small convenience read-back for the debug log above. Kept
    /// `pub(super)` so it lives next to the other asset-cache
    /// accessors; not part of the public API.
    pub(super) fn asset_cache_get_index_count(&self, handle: AssetHandle) -> u32 {
        self.asset_cache
            .get(handle)
            .map(|e| e.mesh_indices.len() as u32)
            .unwrap_or(0)
    }

    pub(super) fn asset_cache_get_cluster_count(&self, handle: AssetHandle) -> u32 {
        self.asset_cache
            .get(handle)
            .map(|e| e.meshlet_clusters.len() as u32)
            .unwrap_or(0)
    }
}
