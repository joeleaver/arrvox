//! Asset lifecycle — `acquire_asset`, `reload_asset`, `release_asset`,
//! `load_asset_from_disk` (the .rkp file parser), plus `skinning_data`
//! and `resolve_rkp_path`.
//!
//! All methods are part of `impl RkpSceneManager` (sibling impl block
//! on the central type defined in [`super::manager`]). Reads private
//! `pub(super)` fields on the struct directly, calls `merge_face_links`
//! and `bump_geometry_epoch` from sibling files.

use std::path::PathBuf;

use rkp_core::{LeafAttr, SparseOctree};

use super::manager::RkpSceneManager;
use super::types::{
    AssetEntry, AssetHandle, AssetInfo, ReloadResult, SkinBrick, SkinningAssetData,
};
use crate::mesh_pass::extract_surface_mesh;
use crate::splat_pass::extract_splats;

impl RkpSceneManager {
    fn resolve_rkp_path(path: &str) -> Result<PathBuf, String> {
        let rkp_path = if path.ends_with(".rkp") {
            PathBuf::from(path)
        } else {
            let p = std::path::Path::new(path);
            let appended = p.with_file_name(format!(
                "{}.rkp",
                p.file_name().map(|f| f.to_string_lossy()).unwrap_or_default()
            ));
            if appended.exists() {
                appended
            } else {
                let replaced = p.with_extension("rkp");
                if replaced.exists() {
                    replaced
                } else {
                    return Err(format!("no .rkp file found for {path}"));
                }
            }
        };
        if !rkp_path.exists() {
            return Err(format!("{} does not exist", rkp_path.display()));
        }
        rkp_path.canonicalize().map_err(|e| format!("canonicalize {}: {e}", rkp_path.display()))
    }

    /// Acquire a shared asset. First call for a given path allocates the
    /// octree / leaf_attr / brick ranges and caches them. Subsequent calls
    /// return the cached handle and bump its refcount. Every successful
    /// `acquire_asset` must be paired with a `release_asset` when the
    /// instance goes away.
    pub fn acquire_asset(
        &mut self,
        path: &str,
    ) -> Result<(AssetHandle, AssetInfo), String> {
        self.bump_geometry_epoch();
        let canonical = Self::resolve_rkp_path(path)?;

        if let Some(handle) = self.asset_cache.lookup_path(&canonical) {
            let entry = self.asset_cache.get_mut(handle).expect("cache/handle mismatch");
            entry.refcount += 1;
            return Ok((handle, entry.info()));
        }

        let entry = self.load_asset_from_disk(&canonical)?;
        let info = entry.info();
        let handle = self.asset_cache.insert(entry);
        Ok((handle, info))
    }

    /// Force a reload of a cached asset from disk. Used after re-import
    /// rewrites the `.rkp` file so existing scene instances pick up the
    /// new geometry. Frees the previous pool allocations, loads the fresh
    /// file, and preserves the refcount so outstanding instances remain
    /// valid once they've been updated to the returned handle.
    ///
    /// Returns `Ok(None)` when the asset isn't currently cached (nothing
    /// to refresh — the next `acquire_asset` will read the new file).
    pub fn reload_asset(&mut self, path: &str) -> Result<Option<ReloadResult>, String> {
        self.bump_geometry_epoch();
        let canonical = Self::resolve_rkp_path(path)?;
        let Some(old_handle) = self.asset_cache.lookup_path(&canonical) else {
            return Ok(None);
        };

        let old_refcount = self.asset_cache.get(old_handle)
            .map(|e| e.refcount).unwrap_or(0);

        let entry = self.asset_cache.remove(old_handle).expect("just looked up");
        self.octree.deallocate(entry.spatial_handle);
        self.leaf_attr_pool.deallocate_range(entry.leaf_attr_slot_start, entry.leaf_attr_slot_count);
        for id in entry.brick_start..(entry.brick_start + entry.brick_count) {
            self.brick_pool.deallocate(id);
        }

        let mut fresh = self.load_asset_from_disk(&canonical)?;
        fresh.refcount = old_refcount;
        let info = fresh.info();
        let new_handle = self.asset_cache.insert(fresh);
        Ok(Some(ReloadResult { old_handle, new_handle, info }))
    }

    /// Release an instance's claim on a cached asset. When the last
    /// outstanding reference drops, we deallocate the shared ranges from
    /// the scene pools.
    pub fn release_asset(&mut self, handle: AssetHandle) {
        self.bump_geometry_epoch();
        let Some(entry) = self.asset_cache.get_mut(handle) else { return; };
        if entry.refcount == 0 { return; }
        entry.refcount -= 1;
        if entry.refcount > 0 { return; }

        // Last reference — free the pool ranges and drop the cache slot.
        let entry = self.asset_cache.remove(handle).expect("just looked up");
        self.octree.deallocate(entry.spatial_handle);
        self.leaf_attr_pool.deallocate_range(entry.leaf_attr_slot_start, entry.leaf_attr_slot_count);
        for id in entry.brick_start..(entry.brick_start + entry.brick_count) {
            self.brick_pool.deallocate(id);
        }
    }

    /// Disk read + pool allocation for one .rkp file. Called exactly once
    /// per unique path — repeated acquisitions share the returned entry
    /// via the cache.
    fn load_asset_from_disk(&mut self, rkp_path: &std::path::Path) -> Result<AssetEntry, String> {
        use rkp_core::voxel::VoxelSample;

        let mut file = std::fs::File::open(rkp_path)
            .map_err(|e| format!("open {}: {e}", rkp_path.display()))?;
        let mut reader = std::io::BufReader::new(&mut file);

        let header = rkp_core::asset_file::read_rkp_header(&mut reader)
            .map_err(|e| format!("read .rkp header: {e}"))?;

        let octree_nodes = rkp_core::asset_file::read_rkp_octree(&mut reader, &header)
            .map_err(|e| format!("read octree: {e}"))?;

        let voxel_data = rkp_core::asset_file::read_rkp_voxels(&mut reader, &header)
            .map_err(|e| format!("read voxels: {e}"))?;

        let voxel_size = header.base_voxel_size;
        let voxel_count = header.voxel_count;
        let aabb = rkp_core::Aabb::new(
            glam::Vec3::from(header.aabb_min),
            glam::Vec3::from(header.aabb_max),
        );

        // Pre-baked octahedrally-packed normals per slot. One u32 per shell
        // voxel, written at import time from the mesh SDF gradient — the
        // runtime never sees an SDF.
        let has_normals = header.flags & rkp_core::asset_file::FLAG_HAS_NORMALS != 0;
        let normals_bytes = if has_normals {
            rkp_core::asset_file::read_rkp_normals(&mut reader, &header).unwrap_or_default()
        } else {
            Vec::new()
        };
        let normals_u32s: &[u32] = if normals_bytes.len() >= 4 {
            bytemuck::cast_slice(&normals_bytes)
        } else {
            &[]
        };

        // Brick-terminated octree (v4). Each brick is a flat run of
        // BRICK_CELLS u32s; cell value is either BRICK_EMPTY or a slot
        // index into the parallel voxel arrays.
        let has_bricks = header.flags & rkp_core::asset_file::FLAG_HAS_BRICKS != 0;
        let bricks_bytes = if has_bricks {
            rkp_core::asset_file::read_rkp_bricks(&mut reader, &header).unwrap_or_default()
        } else {
            Vec::new()
        };
        let file_brick_cells: &[u32] = if !bricks_bytes.is_empty() {
            bytemuck::cast_slice(&bricks_bytes)
        } else {
            &[]
        };

        let has_color = header.flags & rkp_core::asset_file::FLAG_HAS_COLOR != 0;
        let color_bytes = if has_color {
            rkp_core::asset_file::read_rkp_color(&mut reader, &header).unwrap_or_default()
        } else {
            Vec::new()
        };
        let color_u32s: &[u32] = if color_bytes.len() >= 4 {
            bytemuck::cast_slice(&color_bytes)
        } else {
            &[]
        };

        // Skin-meta section — structured payload carrying per-leaf bone
        // weights, per-brick origins, and per-bone rest AABBs. Only
        // present when rkp-import resolved a skinned skeleton.
        let has_bones = header.flags & rkp_core::asset_file::FLAG_HAS_BONES != 0;
        let skin_meta = if has_bones {
            match rkp_core::asset_file::read_rkp_skin_meta(&mut reader, &header) {
                Ok(m) => {
                    eprintln!(
                        "[RkpSceneManager] {}: skin-meta loaded ({} bone voxels, {} bricks, {} bone AABBs)",
                        rkp_path.display(),
                        m.bone_voxels.len() / 8,
                        m.brick_origins.len(),
                        m.rest_bone_aabbs.len(),
                    );
                    m
                }
                Err(e) => {
                    // Old Phase-2 file format wrote the bones section
                    // as a raw `BoneVoxel` array; the new structured
                    // blob fails to decode that. Warn loudly so a
                    // stale `.rkp` on disk doesn't silently mask the
                    // whole skinning pipeline as "nothing broken, no
                    // deformation".
                    eprintln!(
                        "[RkpSceneManager] {}: FLAG_HAS_BONES set but skin-meta decode failed ({e}). \
                         Re-import the asset to write the new wire format.",
                        rkp_path.display(),
                    );
                    rkp_core::asset_file::SkinMetaOut::default()
                }
            }
        } else {
            rkp_core::asset_file::SkinMetaOut::default()
        };
        let file_bones: &[rkp_core::companion::BoneVoxel] = if skin_meta.bone_voxels.len() >= std::mem::size_of::<rkp_core::companion::BoneVoxel>() {
            bytemuck::cast_slice(&skin_meta.bone_voxels)
        } else {
            &[]
        };

        // v5+: pre-built mesh + cluster DAG sections. Replace the
        // load-time `extract_surface_mesh` + `build_cluster_dag` calls
        // (~12s on a 2.5M-vert elephant) with a deserialize. Vertices'
        // `leaf_attr_id`s are file-local; we relocate them to scene-
        // global below, the same pattern bricks already use.
        let mesh_vertices_bytes = rkp_core::asset_file::read_rkp_mesh_vertices(
            &mut reader, &header,
        )
        .map_err(|e| format!("read mesh vertices: {e}"))?;
        let mesh_indices_bytes = rkp_core::asset_file::read_rkp_mesh_indices(
            &mut reader, &header,
        )
        .map_err(|e| format!("read mesh indices: {e}"))?;
        let meshlet_clusters_bytes = rkp_core::asset_file::read_rkp_meshlet_clusters(
            &mut reader, &header,
        )
        .map_err(|e| format!("read meshlet clusters: {e}"))?;
        let mesh_lod0_index_count_from_file = header.mesh_lod0_index_count;

        let bytes_per_voxel = std::mem::size_of::<VoxelSample>();
        // `Option<u32>` for normal so we distinguish "file has no normals"
        // (stays None → leaf_attr keeps its default) from "file has a
        // normal that happens to oct-pack to 0" (which is the legitimate
        // +Z direction; previously the load path skipped that override
        // because it used `if normal_oct != 0`, corrupting every voxel
        // whose baked normal pointed +Z — manifested as one face of a
        // cube rendering with wrong refraction after save/reload, fixed
        // only by re-baking).
        let mut file_voxel_mat: Vec<(u16, u16, u8, u32, Option<u32>)> = Vec::with_capacity(voxel_count as usize);
        for i in 0..voxel_count as usize {
            let src_offset = i * bytes_per_voxel;
            if src_offset + bytes_per_voxel > voxel_data.len() {
                break;
            }
            let vs: &VoxelSample =
                bytemuck::from_bytes(&voxel_data[src_offset..src_offset + bytes_per_voxel]);
            let color = color_u32s.get(i).copied().unwrap_or(0);
            let normal_oct = if has_normals {
                normals_u32s.get(i).copied()
            } else {
                None
            };
            file_voxel_mat.push((
                vs.material_id(), vs.secondary_material_id(), vs.blend_weight(), color, normal_oct,
            ));
        }

        let octree_depth = header.octree_depth as u8;
        let mut tree = SparseOctree::from_raw(&octree_nodes, octree_depth, voxel_size);

        // 1:1 leaf_attr allocation. We don't dedup file slots → leaf_attrs
        // because texture-sampled colors vary per voxel (measured dedup
        // ratio ≈1.0× on mesh imports — HashMap overhead costs more than
        // the trivial savings). Each file slot maps directly to
        // `leaf_attr_slot_start + file_slot`.
        let leaf_attr_slot_count = voxel_count;
        let leaf_attr_slot_start = self.leaf_attr_pool
            .allocate_contiguous_bump(leaf_attr_slot_count)
            .expect("leaf_attr_pool.allocate_contiguous_bump failed");

        for (i, &(mat_p, mat_s, blend, color, normal_oct)) in file_voxel_mat.iter().enumerate() {
            let mut attr = LeafAttr::new_blended(glam::Vec3::Y, mat_p, mat_s, blend);
            if let Some(n) = normal_oct {
                attr.normal_oct = n;
            }
            let slot = leaf_attr_slot_start + i as u32;
            *self.leaf_attr_pool.get_mut(slot) = attr;
            if color != 0 {
                self.leaf_attr_pool.set_color(slot, color);
            }
            // File-local bone slot i → scene-global leaf_attr slot. The
            // `file_bones` slice is empty for unskinned assets, in which
            // case the pool's zero-default BoneVoxel stands.
            if let Some(&bv) = file_bones.get(i) {
                self.leaf_attr_pool.set_bone(slot, bv);
            }
        }

        // v4: copy file brick pool into the scene brick pool. Each file
        // cell holds a file-local slot index; we shift both brick-ids
        // (in the octree nodes) and slot indices (in the cells) by our
        // contiguous allocation offsets.
        let file_brick_count = (file_brick_cells.len() / rkp_core::brick_pool::BRICK_CELLS as usize) as u32;
        let scene_brick_offset = self.brick_pool
            .allocate_contiguous_bump(file_brick_count)
            .expect("brick_pool.allocate_contiguous_bump failed");

        // Remap BRICK node ids in the flat nodes array.
        let nodes = tree.as_slice_mut();
        for n in nodes.iter_mut() {
            if rkp_core::sparse_octree::is_brick(*n) {
                let file_id = rkp_core::sparse_octree::brick_id(*n);
                *n = rkp_core::sparse_octree::make_brick(scene_brick_offset + file_id);
            }
        }

        // Actual surface-cell count across this asset. `header.voxel_count`
        // only counts unique LeafAttr slots (one per unique normal +
        // material + blend + color tuple after bake-time dedup), which
        // badly understates the painted surface on flat-faced primitives
        // — a 20×1×20 procedural cube has ~2.3M cells but ~100 unique
        // attrs, so the header number reads as "126 voxels" after
        // Convert-to-Voxel even though the geometry is fully intact.
        // Count non-sentinel brick cells here (+ LEAF octree nodes
        // below) and report that instead.
        let mut actual_cell_count: u32 = 0;
        let brick_cells = rkp_core::brick_pool::BRICK_CELLS as usize;
        for file_id in 0..file_brick_count {
            let scene_id = scene_brick_offset + file_id;
            let src = &file_brick_cells[
                file_id as usize * brick_cells..(file_id as usize + 1) * brick_cells
            ];
            for (i, &cell) in src.iter().enumerate() {
                if cell == rkp_core::brick_pool::BRICK_EMPTY {
                    continue;
                }
                // BRICK_INTERIOR is a scene-global sentinel (0xFFFFFFFD),
                // not a file-local slot index — pass it through without
                // the leaf_attr_slot_start offset, which would overflow
                // and corrupt the slot into a bogus leaf_attr_id. Also
                // skip it from the user-facing voxel count: interior
                // sentinels mark "inside the solid" and never render /
                // paint as voxels.
                let remapped = if cell == rkp_core::brick_pool::BRICK_INTERIOR {
                    rkp_core::brick_pool::BRICK_INTERIOR
                } else {
                    // Real leaf: cell is a file-local slot index; shift
                    // by our leaf_attr allocation offset to get the
                    // scene-global leaf_attr_id.
                    actual_cell_count += 1;
                    leaf_attr_slot_start + cell
                };
                let x = (i as u32) % rkp_core::brick_pool::BRICK_DIM;
                let y = ((i as u32) / rkp_core::brick_pool::BRICK_DIM) % rkp_core::brick_pool::BRICK_DIM;
                let z = (i as u32) / (rkp_core::brick_pool::BRICK_DIM * rkp_core::brick_pool::BRICK_DIM);
                self.brick_pool.set_cell(scene_id, x, y, z, remapped);
            }
        }
        // Shallow trees (depth ≤ BRICK_LEVELS) skip the brick path and
        // emit LEAF nodes at `max_depth` instead — count those too.
        for &n in &octree_nodes {
            if rkp_core::sparse_octree::is_leaf(n) {
                actual_cell_count += 1;
            }
        }

        if !has_bricks {
            return Err(format!(
                "{}: v4 format requires a bricks section (FLAG_HAS_BRICKS); older files are not supported",
                rkp_path.display(),
            ));
        }

        let raw_count = tree.node_count();
        tree.collapse_all();
        tree.compact();
        let compact_count = tree.node_count();
        tree.deduplicate_subtrees();
        let dedup_count = tree.node_count();
        tree.morton_reorder();

        // Note: Laplacian shell-normal smoothing used to run here.
        // It's now performed at asset-bake time inside `rkp-import`'s
        // `smooth_normals` stage so each asset pays the cost once
        // instead of on every load. Older `.rkp` files written before
        // that change will have un-smoothed SDF-gradient normals
        // (noisier but still correct); re-import to pick up the
        // pre-smoothed variant.

        // Run the prefilter pass on-load so v4 assets (no baked internal
        // attrs) still benefit from the GPU's LOD early-exit. Phase 4
        // bumps the .rkp format to v5 which bakes these at conversion
        // time — this is the fallback until then.
        //
        // The prefilter appends new attrs at the tail of the asset's
        // contiguous leaf_attr range via allocate_contiguous_bump(1), so
        // the `leaf_attr_slot_count` grows to cover them and the
        // existing deallocate_range releases everything on asset drop.
        rkp_core::prefilter::prefilter_octree_internals(
            &mut tree,
            &mut self.leaf_attr_pool,
            &self.brick_pool,
        );
        let final_leaf_attr_slot_count =
            self.leaf_attr_pool.allocated_count() - leaf_attr_slot_start;

        // Splat extraction — flatten the surface into one
        // `SplatVertex` per occupied cell, in object-local coordinates.
        // The render side uploads this to a per-asset GPU vertex buffer
        // and rasterizes it as oriented disc splats when the editor's
        // primary-visibility path is set to splats. Tree nodes are
        // already in their final compacted/morton form here, and brick
        // ids have been remapped to scene-global; the brick cells'
        // leaf_attr ids likewise. So `tree.as_slice()` paired with the
        // scene-global `brick_pool.as_slice()` produces splats whose
        // `leaf_attr_id` indexes directly into `leaf_attr_pool` — the
        // same indirection the splat shader's vertex stage expects.
        let asset_extent =
            (1u32 << header.octree_depth as u8) as f32 * header.base_voxel_size;
        let aabb_center = (aabb.min + aabb.max) * 0.5;
        let asset_grid_origin = aabb_center - glam::Vec3::splat(asset_extent * 0.5);
        let splats = extract_splats(
            tree.as_slice(),
            header.octree_depth as u8,
            header.base_voxel_size,
            asset_grid_origin,
            self.brick_pool.as_slice(),
        );

        // v5+ pre-built mesh deserialization. The .rkp ships
        // `(MeshVertex[], u32[], MeshletCluster[])` so the editor
        // skips the ~12s `extract_surface_mesh` +
        // `build_cluster_dag` it used to do at every load. v4 files
        // (`mesh_vertices_bytes` empty after the v4-header
        // fallback) take the fall-back path below: rebuild from the
        // scene-merged tree as before.
        use rkp_core::mesh_extract::MeshVertex;
        use rkp_core::mesh_cluster::MeshletCluster;
        let have_baked_mesh = !mesh_vertices_bytes.is_empty();
        let (mut mesh_vertices, mesh_indices, meshlet_clusters, mesh_lod0_index_count) =
            if have_baked_mesh {
                let v: Vec<MeshVertex> =
                    bytemuck::cast_slice::<u8, MeshVertex>(&mesh_vertices_bytes).to_vec();
                let i: Vec<u32> =
                    bytemuck::cast_slice::<u8, u32>(&mesh_indices_bytes).to_vec();
                let c: Vec<MeshletCluster> =
                    bytemuck::cast_slice::<u8, MeshletCluster>(&meshlet_clusters_bytes)
                        .to_vec();
                (v, i, c, mesh_lod0_index_count_from_file)
            } else {
                // Legacy v4 fallback — extract + build at load time
                // exactly like the pre-v5 path. Logged so a slow load
                // is attributable to a stale .rkp instead of looking
                // like a perf regression.
                eprintln!(
                    "[RkpSceneManager] {}: v4 .rkp without baked mesh sections — extracting + building DAG at load (re-import to avoid this)",
                    rkp_path.display(),
                );
                let (v, i_unc) = extract_surface_mesh(
                    tree.as_slice(),
                    header.octree_depth as u8,
                    header.base_voxel_size,
                    asset_grid_origin,
                    self.brick_pool.as_slice(),
                    self.leaf_attr_pool.as_slice(),
                );
                let dag_t0 = std::time::Instant::now();
                let dag = crate::mesh_pass::build_cluster_dag(&v, &i_unc);
                eprintln!(
                    "[RkpSceneManager] {}: legacy DAG built in {:.2}s ({} clusters)",
                    rkp_path.display(),
                    dag_t0.elapsed().as_secs_f32(),
                    dag.clusters.len(),
                );
                let lod0 = dag.lod0_index_range.1 - dag.lod0_index_range.0;
                (v, dag.indices, dag.clusters, lod0)
            };

        // Relocate vertex `leaf_attr_id`s from file-local (what
        // rkp-import baked into v5) to scene-global. The legacy v4
        // path already produced scene-global IDs because it ran
        // `extract_surface_mesh` against the scene-merged pools.
        if have_baked_mesh && leaf_attr_slot_start > 0 && !mesh_vertices.is_empty() {
            for v in &mut mesh_vertices {
                v.leaf_attr_id += leaf_attr_slot_start;
            }
        }
        let _ = (mesh_indices.len(), meshlet_clusters.len());

        // Compute brick face-links for this asset. The tree's brick ids
        // have already been remapped to global ids above, so the rows
        // produced are scene-global and ready to merge. When the file
        // had zero bricks there's nothing to compute.
        if file_brick_count > 0 {
            let max_brick = scene_brick_offset + file_brick_count - 1;
            let face_links = rkp_core::brick_face_links::compute_brick_face_links(&tree, max_brick);
            self.merge_face_links(&face_links);
        }

        // Allocate the octree with its now-populated internal_attr_index
        // intact. `allocate(&tree)` preserves both buffers; the legacy
        // `allocate_raw(nodes, …)` would have dropped the prefilter ids
        // by round-tripping through `SparseOctree::from_raw`.
        let handle = self.octree.allocate(&tree);

        eprintln!(
            "[RkpSceneManager] loaded {}: {} voxels ({} unique attrs), {} bricks, octree {} → compact {} → dedup {} ({:.1}× total), +{} prefilter attrs, {} splats, mesh {} verts / lod0 {} tris / dag {} tris / {} clusters max-lod {}",
            rkp_path.display(),
            actual_cell_count,
            voxel_count,
            file_brick_count,
            raw_count,
            compact_count,
            dedup_count,
            if dedup_count > 0 { raw_count as f64 / dedup_count as f64 } else { 0.0 },
            final_leaf_attr_slot_count - leaf_attr_slot_count,
            splats.len(),
            mesh_vertices.len(),
            mesh_lod0_index_count / 3,
            mesh_indices.len() / 3,
            meshlet_clusters.len(),
            meshlet_clusters.iter().map(|c| c.lod_level).max().unwrap_or(0),
        );

        // Promote the baked skin-meta (file-local brick ids) into
        // scene-global SkinBrick entries. Rest bone AABBs are already
        // in object-local voxel space — no transform needed.
        let skinning = if has_bones {
            let bricks: Vec<SkinBrick> = skin_meta.brick_origins.iter().enumerate()
                .map(|(file_id, &origin)| SkinBrick {
                    brick_id: scene_brick_offset + file_id as u32,
                    origin,
                })
                .collect();
            Some(SkinningAssetData {
                bricks,
                rest_bone_aabbs: skin_meta.rest_bone_aabbs,
            })
        } else {
            None
        };

        Ok(AssetEntry {
            path: rkp_path.to_path_buf(),
            refcount: 1,
            spatial_handle: handle,
            voxel_size,
            aabb,
            voxel_count: actual_cell_count,
            leaf_attr_slot_start,
            leaf_attr_slot_count: final_leaf_attr_slot_count,
            brick_start: scene_brick_offset,
            brick_count: file_brick_count,
            skinning,
            splats,
            mesh_vertices,
            mesh_indices,
            mesh_lod0_index_count,
            meshlet_clusters,
        })
    }

    /// Peek at an asset's skinning metadata. Returns `None` when the
    /// asset was imported without bone weights.
    pub fn skinning_data(&self, handle: AssetHandle) -> Option<&SkinningAssetData> {
        self.asset_cache.get(handle)?.skinning.as_ref()
    }

    /// Splat-rasterizer surface data for `handle`. One `SplatVertex` per
    /// occupied surface cell, in object-local coordinates; the per-
    /// instance world matrix is applied in the splat vertex shader.
    /// Returns `None` for an unknown handle. The slice is shared across
    /// every scene-instance of the asset; the render side uploads it
    /// once per geometry epoch.
    pub fn asset_splats(
        &self,
        handle: AssetHandle,
    ) -> Option<&[crate::splat_pass::SplatVertex]> {
        Some(self.asset_cache.get(handle)?.splats.as_slice())
    }

    /// Iterator over `(AssetHandle, &[SplatVertex])` for every loaded
    /// asset. Used by the render thread to keep the per-asset GPU
    /// vertex-buffer cache in sync with the CPU-side asset cache —
    /// called once per geometry epoch after `upload_geometry`.
    ///
    /// Skips empty splat lists (procedural assets, future-proof) so
    /// the caller's upload loop only touches assets that need it.
    pub fn iter_loaded_asset_splats(
        &self,
    ) -> impl Iterator<Item = (AssetHandle, &[crate::splat_pass::SplatVertex])> {
        self.asset_cache
            .entries
            .iter()
            .enumerate()
            .filter_map(|(idx, slot)| {
                let entry = slot.as_ref()?;
                if entry.splats.is_empty() {
                    return None;
                }
                Some((AssetHandle::from_raw(idx as u32), entry.splats.as_slice()))
            })
    }

    /// Surface-mesh `(vertices, indices, lod0_index_count)` for
    /// `handle`. Phase 6.1: `indices` is the **full DAG IBO** with
    /// LOD-0 indices first; `lod0_index_count` is the LOD-0 prefix
    /// length (what dispatch currently draws). Returns `None` for
    /// an unknown handle.
    pub fn asset_mesh(
        &self,
        handle: AssetHandle,
    ) -> Option<(&[crate::mesh_pass::MeshVertex], &[u32], u32)> {
        let entry = self.asset_cache.get(handle)?;
        Some((
            entry.mesh_vertices.as_slice(),
            entry.mesh_indices.as_slice(),
            entry.mesh_lod0_index_count,
        ))
    }

    /// Iterator over `(AssetHandle, &[MeshVertex], &[u32],
    /// lod0_index_count)` for every loaded asset that produced a
    /// non-empty surface mesh. Phase 6.1: `lod0_index_count` is the
    /// LOD-0 prefix of the DAG IBO; the render thread caches it as
    /// the dispatch draw count.
    pub fn iter_loaded_asset_meshes(
        &self,
    ) -> impl Iterator<Item = (AssetHandle, &[crate::mesh_pass::MeshVertex], &[u32], u32)> {
        self.asset_cache
            .entries
            .iter()
            .enumerate()
            .filter_map(|(idx, slot)| {
                let entry = slot.as_ref()?;
                if entry.mesh_vertices.is_empty() {
                    return None;
                }
                Some((
                    AssetHandle::from_raw(idx as u32),
                    entry.mesh_vertices.as_slice(),
                    entry.mesh_indices.as_slice(),
                    entry.mesh_lod0_index_count,
                ))
            })
    }

    /// Iterator over `(AssetHandle, &[MeshletCluster])` for every
    /// loaded asset whose surface mesh has clusters (Phase 5). The
    /// render thread uploads these to a per-asset GPU storage buffer
    /// once per geometry epoch, parallel to `iter_loaded_asset_meshes`.
    pub fn iter_loaded_asset_clusters(
        &self,
    ) -> impl Iterator<Item = (AssetHandle, &[crate::mesh_pass::MeshletCluster])> {
        self.asset_cache
            .entries
            .iter()
            .enumerate()
            .filter_map(|(idx, slot)| {
                let entry = slot.as_ref()?;
                if entry.meshlet_clusters.is_empty() {
                    return None;
                }
                Some((
                    AssetHandle::from_raw(idx as u32),
                    entry.meshlet_clusters.as_slice(),
                ))
            })
    }

}
