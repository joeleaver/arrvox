//! Asset lifecycle — `acquire_asset`, `reload_asset`, `release_asset`,
//! `load_asset_from_disk` (the .arvx file parser), plus `skinning_data`
//! and `resolve_rkp_path`.
//!
//! All methods are part of `impl ArvxSceneManager` (sibling impl block
//! on the central type defined in [`super::manager`]). Reads private
//! `pub(super)` fields on the struct directly, calls `merge_face_links`
//! and `bump_geometry_epoch` from sibling files.

use std::path::PathBuf;

use arvx_core::{LeafAttr, SparseOctree};

use super::manager::ArvxSceneManager;
use super::types::{
    AssetEntry, AssetHandle, AssetInfo, ReloadResult, SkinningAssetData,
};
use crate::mesh_pass::extract_surface_mesh;

impl ArvxSceneManager {
    fn resolve_rkp_path(path: &str) -> Result<PathBuf, String> {
        let arvx_path = if path.ends_with(".arvx") {
            PathBuf::from(path)
        } else {
            let p = std::path::Path::new(path);
            let appended = p.with_file_name(format!(
                "{}.arvx",
                p.file_name().map(|f| f.to_string_lossy()).unwrap_or_default()
            ));
            if appended.exists() {
                appended
            } else {
                let replaced = p.with_extension("arvx");
                if replaced.exists() {
                    replaced
                } else {
                    return Err(format!("no .arvx file found for {path}"));
                }
            }
        };
        if !arvx_path.exists() {
            return Err(format!("{} does not exist", arvx_path.display()));
        }
        arvx_path.canonicalize().map_err(|e| format!("canonicalize {}: {e}", arvx_path.display()))
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
    /// rewrites the `.arvx` file so existing scene instances pick up the
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
        // Free any sculpt-allocated slots outside the bake range so
        // they don't leak. The HashSet guarantees no double-frees
        // even if a slot was freed-then-realloc'd during the
        // session (sculpt's free path removes the entry).
        for &slot in &entry.sculpt_extra_slots {
            self.leaf_attr_pool.deallocate_range(slot, 1);
        }
        // Same for halo-extra slots (Phase 4.2b cross-tile halo
        // refresh): empty→solid transitions on a neighbour tile may
        // have allocated new halo slots outside this asset's bake
        // range. Free them here.
        for &slot in &entry.halo_extra_slots {
            self.leaf_attr_pool.deallocate_range(slot, 1);
        }
        for id in entry.brick_start..(entry.brick_start + entry.brick_count) {
            self.brick_pool.deallocate(id);
        }
    }

    /// Reserve a fresh `AssetHandle` for a procedural proxy-mesh
    /// entity. The handle is allocated from the same flat handle
    /// space as disk assets (so `mesh_buffers[handle.raw()]` works
    /// the same way), but no `AssetEntry` is attached — proxy meshes
    /// have no octree / leaf_attr / brick allocations to refcount,
    /// and aren't shared by path. Caller is responsible for pairing
    /// with `release_procedural_handle` and for uploading
    /// `mesh_buffers` + `mesh_cluster_buffers` on the renderer side.
    pub fn reserve_procedural_handle(&mut self) -> AssetHandle {
        self.asset_cache.reserve_handle()
    }

    /// Release a handle reserved via `reserve_procedural_handle`.
    /// Caller must drop the renderer's `mesh_buffers` /
    /// `mesh_cluster_buffers` for that handle separately.
    pub fn release_procedural_handle(&mut self, handle: AssetHandle) {
        self.asset_cache.release_reserved(handle);
        self.bump_geometry_epoch();
    }

    /// Disk read + pool allocation for one .arvx file. Called exactly once
    /// per unique path — repeated acquisitions share the returned entry
    /// via the cache.
    fn load_asset_from_disk(&mut self, arvx_path: &std::path::Path) -> Result<AssetEntry, String> {
        use arvx_core::voxel::VoxelSample;

        let mut file = std::fs::File::open(arvx_path)
            .map_err(|e| format!("open {}: {e}", arvx_path.display()))?;
        let mut reader = std::io::BufReader::new(&mut file);

        let header = arvx_core::asset_file::read_rkp_header(&mut reader)
            .map_err(|e| format!("read .arvx header: {e}"))?;

        let octree_nodes = arvx_core::asset_file::read_rkp_octree(&mut reader, &header)
            .map_err(|e| format!("read octree: {e}"))?;

        let voxel_data = arvx_core::asset_file::read_rkp_voxels(&mut reader, &header)
            .map_err(|e| format!("read voxels: {e}"))?;

        let voxel_size = header.base_voxel_size;
        let voxel_count = header.voxel_count;
        let aabb = arvx_core::Aabb::new(
            glam::Vec3::from(header.aabb_min),
            glam::Vec3::from(header.aabb_max),
        );

        // Pre-baked octahedrally-packed normals per slot. One u32 per shell
        // voxel, written at import time from the mesh SDF gradient — the
        // runtime never sees an SDF.
        let has_normals = header.flags & arvx_core::asset_file::FLAG_HAS_NORMALS != 0;
        let normals_bytes = if has_normals {
            arvx_core::asset_file::read_rkp_normals(&mut reader, &header).unwrap_or_default()
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
        let has_bricks = header.flags & arvx_core::asset_file::FLAG_HAS_BRICKS != 0;
        let bricks_bytes = if has_bricks {
            arvx_core::asset_file::read_rkp_bricks(&mut reader, &header).unwrap_or_default()
        } else {
            Vec::new()
        };
        let file_brick_cells: &[u32] = if !bricks_bytes.is_empty() {
            bytemuck::cast_slice(&bricks_bytes)
        } else {
            &[]
        };

        let has_color = header.flags & arvx_core::asset_file::FLAG_HAS_COLOR != 0;
        let color_bytes = if has_color {
            arvx_core::asset_file::read_rkp_color(&mut reader, &header).unwrap_or_default()
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
        // present when arvx-import resolved a skinned skeleton.
        let has_bones = header.flags & arvx_core::asset_file::FLAG_HAS_BONES != 0;
        let skin_meta = if has_bones {
            match arvx_core::asset_file::read_rkp_skin_meta(&mut reader, &header) {
                Ok(m) => {
                    eprintln!(
                        "[ArvxSceneManager] {}: skin-meta loaded ({} bone voxels, {} bricks, {} bone AABBs)",
                        arvx_path.display(),
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
                    // stale `.arvx` on disk doesn't silently mask the
                    // whole skinning pipeline as "nothing broken, no
                    // deformation".
                    eprintln!(
                        "[ArvxSceneManager] {}: FLAG_HAS_BONES set but skin-meta decode failed ({e}). \
                         Re-import the asset to write the new wire format.",
                        arvx_path.display(),
                    );
                    arvx_core::asset_file::SkinMetaOut::default()
                }
            }
        } else {
            arvx_core::asset_file::SkinMetaOut::default()
        };
        let file_bones: &[arvx_core::companion::BoneVoxel] = if skin_meta.bone_voxels.len() >= std::mem::size_of::<arvx_core::companion::BoneVoxel>() {
            bytemuck::cast_slice(&skin_meta.bone_voxels)
        } else {
            &[]
        };

        // v5+: pre-built mesh + cluster DAG sections. Replace the
        // load-time `extract_surface_mesh` + `build_cluster_dag` calls
        // (~12s on a 2.5M-vert elephant) with a deserialize. Vertices'
        // `leaf_attr_id`s are file-local; we relocate them to scene-
        // global below, the same pattern bricks already use.
        let mesh_vertices_bytes = arvx_core::asset_file::read_rkp_mesh_vertices(
            &mut reader, &header,
        )
        .map_err(|e| format!("read mesh vertices: {e}"))?;
        let mesh_indices_bytes = arvx_core::asset_file::read_rkp_mesh_indices(
            &mut reader, &header,
        )
        .map_err(|e| format!("read mesh indices: {e}"))?;
        let meshlet_clusters_bytes = arvx_core::asset_file::read_rkp_meshlet_clusters(
            &mut reader, &header,
        )
        .map_err(|e| format!("read meshlet clusters: {e}"))?;
        // v6+ DAG topology sections. Empty for v4 (legacy fallback
        // rebuilds DAG below) and v5 (no DAG metadata baked; sculpt
        // falls back to asset-wide marking).
        let dag_groups_bytes = arvx_core::asset_file::read_rkp_dag_groups(
            &mut reader, &header,
        )
        .map_err(|e| format!("read dag groups: {e}"))?;
        let dag_consumed_bytes = arvx_core::asset_file::read_rkp_dag_consumed(
            &mut reader, &header,
        )
        .map_err(|e| format!("read dag consumed: {e}"))?;
        let dag_produced_bytes = arvx_core::asset_file::read_rkp_dag_produced(
            &mut reader, &header,
        )
        .map_err(|e| format!("read dag produced: {e}"))?;
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
        let file_brick_count = (file_brick_cells.len() / arvx_core::brick_pool::BRICK_CELLS as usize) as u32;
        let scene_brick_offset = self.brick_pool
            .allocate_contiguous_bump(file_brick_count)
            .expect("brick_pool.allocate_contiguous_bump failed");

        // Remap BRICK node ids in the flat nodes array.
        let nodes = tree.as_slice_mut();
        for n in nodes.iter_mut() {
            if arvx_core::sparse_octree::is_brick(*n) {
                let file_id = arvx_core::sparse_octree::brick_id(*n);
                *n = arvx_core::sparse_octree::make_brick(scene_brick_offset + file_id);
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
        let brick_cells = arvx_core::brick_pool::BRICK_CELLS as usize;
        for file_id in 0..file_brick_count {
            let scene_id = scene_brick_offset + file_id;
            let src = &file_brick_cells[
                file_id as usize * brick_cells..(file_id as usize + 1) * brick_cells
            ];
            for (i, &cell) in src.iter().enumerate() {
                if cell == arvx_core::brick_pool::BRICK_EMPTY {
                    continue;
                }
                // BRICK_INTERIOR is a scene-global sentinel (0xFFFFFFFD),
                // not a file-local slot index — pass it through without
                // the leaf_attr_slot_start offset, which would overflow
                // and corrupt the slot into a bogus leaf_attr_id. Also
                // skip it from the user-facing voxel count: interior
                // sentinels mark "inside the solid" and never render /
                // paint as voxels.
                let remapped = if cell == arvx_core::brick_pool::BRICK_INTERIOR {
                    arvx_core::brick_pool::BRICK_INTERIOR
                } else {
                    // Real leaf: cell is a file-local slot index; shift
                    // by our leaf_attr allocation offset to get the
                    // scene-global leaf_attr_id.
                    actual_cell_count += 1;
                    leaf_attr_slot_start + cell
                };
                let x = (i as u32) % arvx_core::brick_pool::BRICK_DIM;
                let y = ((i as u32) / arvx_core::brick_pool::BRICK_DIM) % arvx_core::brick_pool::BRICK_DIM;
                let z = (i as u32) / (arvx_core::brick_pool::BRICK_DIM * arvx_core::brick_pool::BRICK_DIM);
                self.brick_pool.set_cell(scene_id, x, y, z, remapped);
            }
        }
        // Shallow trees (depth ≤ BRICK_LEVELS) skip the brick path and
        // emit LEAF nodes at `max_depth` instead — count those too.
        for &n in &octree_nodes {
            if arvx_core::sparse_octree::is_leaf(n) {
                actual_cell_count += 1;
            }
        }

        if !has_bricks {
            return Err(format!(
                "{}: v4 format requires a bricks section (FLAG_HAS_BRICKS); older files are not supported",
                arvx_path.display(),
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
        // It's now performed at asset-bake time inside `arvx-import`'s
        // `smooth_normals` stage so each asset pays the cost once
        // instead of on every load. Older `.arvx` files written before
        // that change will have un-smoothed SDF-gradient normals
        // (noisier but still correct); re-import to pick up the
        // pre-smoothed variant.

        // Run the prefilter pass on-load so v4 assets (no baked internal
        // attrs) still benefit from the GPU's LOD early-exit. Phase 4
        // bumps the .arvx format to v5 which bakes these at conversion
        // time — this is the fallback until then.
        //
        // The prefilter appends new attrs at the tail of the asset's
        // contiguous leaf_attr range via allocate_contiguous_bump(1), so
        // the `leaf_attr_slot_count` grows to cover them and the
        // existing deallocate_range releases everything on asset drop.
        arvx_core::prefilter::prefilter_octree_internals(
            &mut tree,
            &mut self.leaf_attr_pool,
            &self.brick_pool,
        );
        let final_leaf_attr_slot_count =
            self.leaf_attr_pool.allocated_count() - leaf_attr_slot_start;

        // v5+ pre-built mesh deserialization. The .arvx ships
        // `(MeshVertex[], u32[], MeshletCluster[])` so the editor
        // skips the ~12s `extract_surface_mesh` +
        // `build_cluster_dag` it used to do at every load. v4 files
        // (`mesh_vertices_bytes` empty after the v4-header
        // fallback) take the fall-back path below: rebuild from the
        // scene-merged tree as before.
        use arvx_core::mesh_extract::MeshVertex;
        use arvx_core::mesh_cluster::{MeshletCluster, PARENT_GROUP_ERROR_ROOT};
        use arvx_core::mesh_lod::DagGroup;
        let have_baked_mesh = !mesh_vertices_bytes.is_empty();
        let (
            mut mesh_vertices,
            mesh_indices,
            mut meshlet_clusters,
            mesh_lod0_index_count,
            dag_groups,
            dag_consumed,
            dag_produced,
        ) = if have_baked_mesh {
            let v: Vec<MeshVertex> =
                bytemuck::cast_slice::<u8, MeshVertex>(&mesh_vertices_bytes).to_vec();
            let i: Vec<u32> =
                bytemuck::cast_slice::<u8, u32>(&mesh_indices_bytes).to_vec();
            let c: Vec<MeshletCluster> =
                bytemuck::cast_slice::<u8, MeshletCluster>(&meshlet_clusters_bytes)
                    .to_vec();
            // v6 DAG sections — empty for v5, present + non-empty for
            // v6 + any DAG that grew past LOD-0. Empty triplet keeps
            // sculpt's CC walk on the asset-wide fallback path.
            let dg: Vec<DagGroup> =
                bytemuck::cast_slice::<u8, DagGroup>(&dag_groups_bytes).to_vec();
            let dc: Vec<u32> =
                bytemuck::cast_slice::<u8, u32>(&dag_consumed_bytes).to_vec();
            let dp: Vec<u32> =
                bytemuck::cast_slice::<u8, u32>(&dag_produced_bytes).to_vec();
            if !dg.is_empty() {
                eprintln!(
                    "[ArvxSceneManager] {}: v6 DAG topology loaded ({} groups, {} consumed, {} produced)",
                    arvx_path.display(),
                    dg.len(),
                    dc.len(),
                    dp.len(),
                );
            }
            (v, i, c, mesh_lod0_index_count_from_file, dg, dc, dp)
        } else {
            // Legacy v4 fallback — extract + build at load time
            // exactly like the pre-v5 path. Logged so a slow load
            // is attributable to a stale .arvx instead of looking
            // like a perf regression.
            eprintln!(
                "[ArvxSceneManager] {}: v4 .arvx without baked mesh sections — extracting + building DAG at load (re-import to avoid this)",
                arvx_path.display(),
            );
            let asset_extent =
                (1u32 << header.octree_depth as u8) as f32 * header.base_voxel_size;
            let aabb_center = (aabb.min + aabb.max) * 0.5;
            let asset_grid_origin = aabb_center - glam::Vec3::splat(asset_extent * 0.5);
            let (v, i_unc) = extract_surface_mesh(
                tree.as_slice(),
                header.octree_depth as u8,
                header.base_voxel_size,
                asset_grid_origin,
                self.brick_pool.as_slice(),
                self.leaf_attr_pool.as_slice(),
                // Fallback path runs against scene-merged pools, so
                // by the time we get here the bone slice has been
                // populated via `set_bone(slot, bv)` above. Skinned
                // v4 assets get their bone weights baked into the
                // newly extracted vertices on the spot.
                self.leaf_attr_pool.bones_as_slice(),
                // Fresh asset load: no sculpt history.
                None,
            );
            let dag_t0 = std::time::Instant::now();
            let dag = crate::mesh_pass::build_cluster_dag(&v, &i_unc);
            eprintln!(
                "[ArvxSceneManager] {}: legacy DAG built in {:.2}s ({} clusters)",
                arvx_path.display(),
                dag_t0.elapsed().as_secs_f32(),
                dag.clusters.len(),
            );
            let lod0 = dag.lod0_index_range.1 - dag.lod0_index_range.0;
            (
                v,
                dag.indices,
                dag.clusters,
                lod0,
                dag.dag_groups,
                dag.dag_consumed,
                dag.dag_produced,
            )
        };

        // Per-LOD-level error normalization. The Karis admit rule's
        // chain consistency requires `cluster_error` of a level-N
        // cluster to equal `parent_group_error` of its level-(N-1)
        // children. The DAG builder already enforces this PER GROUP
        // (all sub-clusters of a group share the same cluster_error,
        // and consumed prev-level clusters get their parent_group_error
        // backfilled from the same group_error). But across DIFFERENT
        // groups at the same level the error values differ, so
        // adjacent clusters may pick DIFFERENT LOD levels at runtime.
        // The simplifier's group-boundary lock keeps each level pair
        // watertight in isolation, but mixing 3+ adjacent levels in
        // one frame creates topological cracks at group boundaries
        // (T-junctions / mismatched edge chains across N+2-step LOD
        // gaps). Visually: the elephant scene at LOD_LEVELS=8 falls
        // apart into chunks once Karis admits at multiple levels.
        //
        // Workaround until monotonic bounding spheres land
        // (`project_mesh_lod_monotonic_spheres_followup`): collapse
        // every level's cluster_error to the LEVEL'S MAX, and rewrite
        // each non-root cluster's parent_group_error to the next
        // level's max. This makes the entire instance admit at one
        // level (since all chains see the same boundary error), at
        // the cost of per-cluster LOD precision. For the splat5
        // elephant the precision loss is invisible because all chains
        // were converging on the same level anyway.
        if !meshlet_clusters.is_empty() {
            let mut max_level = 0u32;
            for c in &meshlet_clusters {
                if c.lod_level > max_level {
                    max_level = c.lod_level;
                }
            }
            // Per-level max cluster_error. Index by lod_level.
            let mut level_max_error: Vec<f32> = vec![0.0; max_level as usize + 1];
            for c in &meshlet_clusters {
                let l = c.lod_level as usize;
                if c.cluster_error > level_max_error[l] {
                    level_max_error[l] = c.cluster_error;
                }
            }
            for c in &mut meshlet_clusters {
                let l = c.lod_level as usize;
                // Don't override the leaf sentinel (cluster_error=0
                // means cluster_is_leaf in the shader). Leaves are at
                // LOD 0 by construction; leaving their cluster_error
                // at 0 keeps the leaf admit short-circuit working.
                if c.cluster_error != 0.0 {
                    c.cluster_error = level_max_error[l];
                }
                // Rewrite parent_group_error to the next level's max,
                // preserving the root sentinel for true DAG roots.
                if c.parent_group_error < PARENT_GROUP_ERROR_ROOT * 0.5 {
                    let next_l = (c.lod_level + 1) as usize;
                    if next_l <= max_level as usize {
                        c.parent_group_error = level_max_error[next_l];
                    }
                }
            }
        }

        // For v5 files baked BEFORE the bone-fields-in-vertex change
        // (Phase 6.6 commit 1), the on-disk vertices carry zero
        // `bone_indices/weights` because the corresponding bytes were
        // unnamed `_pad` and zero-written. Newer bakes carry correct
        // file-local bone data already. The load-path merge below
        // runs in both cases — for old bakes it back-fills bone data
        // from the file's skin-meta payload (avoiding a re-bake of
        // every existing splat5 .arvx); for new bakes it writes the
        // same file-local values back, idempotent and cheap. The
        // legacy v4 fallback above already produced correct,
        // scene-global bone data via the extractor, so it skips this
        // pass entirely.
        if have_baked_mesh && !mesh_vertices.is_empty() && !file_bones.is_empty() {
            for v in &mut mesh_vertices {
                if let Some(bv) = file_bones.get(v.leaf_attr_id as usize) {
                    v.bone_indices = bv.indices;
                    v.bone_weights = bv.weights;
                }
            }
        }

        // Relocate vertex `leaf_attr_id`s from file-local (what
        // arvx-import baked into v5) to scene-global. The legacy v4
        // path already produced scene-global IDs because it ran
        // `extract_surface_mesh` against the scene-merged pools.
        // **Order matters:** the bone-merge above must run BEFORE
        // this — `file_bones` is indexed by file-local slot id.
        if have_baked_mesh && leaf_attr_slot_start > 0 && !mesh_vertices.is_empty() {
            for v in &mut mesh_vertices {
                v.leaf_attr_id += leaf_attr_slot_start;
            }
        }

        // Phase 6.6: cluster AABB expansion for skinned assets is
        // intentionally disabled. The original plan (union the
        // `rest_bone_aabbs` of every bone a cluster's vertices weight
        // against) inflates each cluster by roughly the union of all
        // its referenced bones' territory — for a chest cluster
        // weighted to chest + neck + shoulder bones, that's most of
        // the upper body. The Karis LOD rule projects through the
        // AABB; an oversized AABB makes `cluster_error_proj` huge and
        // forces fine clusters to fail their admission test, leaving
        // the chain root (coarsest LOD) as the only admitted level.
        // Visually that prints CesiumMan in chunky LOD-N triangles
        // (the "ugly triangle quilt" report).
        //
        // Why removing it is safe in practice: the LOD selector uses
        // the AABB only to pick an LOD level, not to cull triangles
        // outright. A cluster admitted at the wrong LOD still renders
        // its triangles (deformed by the VS); the result is a small
        // resolution mismatch invisible compared to the quilting the
        // expansion produced. For typical character animation the
        // rest-pose projected size is within ~10-20 % of the
        // deformed projected size, so the LOD pick is close anyway.
        //
        // The proper fix is the per-frame GPU recompute the memory
        // plan flagged as a follow-on (`project_mesh_skinning_rewrite.md`)
        // — kept out of this commit on purpose. The helper
        // `mesh_cluster::expand_clusters_for_skinning` stays in tree
        // for that future variant; it just isn't called today.
        let _ = (mesh_indices.len(), meshlet_clusters.len());

        // Note: an earlier R4a-proper version ran a per-cluster
        // split + flatten round-trip here so sculpt could replace
        // individual clusters' mesh data in `cluster_meshes` and
        // re-flatten on every stamp. The round-trip *duplicated*
        // boundary verts per-cluster — on a ~100 k-cluster
        // multi-LOD asset that 2-3 ×'d the VBO size (~6.5 M verts
        // vs ~2.5 M original) → "mesh asset vbo" OOM on 4-6 GB
        // GPUs. Sculpt now uses an append-only path against the
        // original flat VBO/IBO (see `rebuild_dirty_clusters`), so
        // load keeps the build_cluster_dag output verbatim.

        // Compute brick face-links for this asset. The tree's brick ids
        // have already been remapped to global ids above, so the rows
        // produced are scene-global and ready to merge. When the file
        // had zero bricks there's nothing to compute.
        if file_brick_count > 0 {
            let max_brick = scene_brick_offset + file_brick_count - 1;
            let face_links = arvx_core::brick_face_links::compute_brick_face_links(&tree, max_brick);
            self.merge_face_links(&face_links);
        }

        // Allocate the octree with its now-populated internal_attr_index
        // intact. `allocate(&tree)` preserves both buffers; the legacy
        // `allocate_raw(nodes, …)` would have dropped the prefilter ids
        // by round-tripping through `SparseOctree::from_raw`.
        let handle = self.octree.allocate(&tree);

        eprintln!(
            "[ArvxSceneManager] loaded {}: {} voxels ({} unique attrs), {} bricks, octree {} → compact {} → dedup {} ({:.1}× total), +{} prefilter attrs, mesh {} verts / lod0 {} tris / dag {} tris / {} clusters max-lod {}",
            arvx_path.display(),
            actual_cell_count,
            voxel_count,
            file_brick_count,
            raw_count,
            compact_count,
            dedup_count,
            if dedup_count > 0 { raw_count as f64 / dedup_count as f64 } else { 0.0 },
            final_leaf_attr_slot_count - leaf_attr_slot_count,
            mesh_vertices.len(),
            mesh_lod0_index_count / 3,
            mesh_indices.len() / 3,
            meshlet_clusters.len(),
            meshlet_clusters.iter().map(|c| c.lod_level).max().unwrap_or(0),
        );

        // Rest bone AABBs are already in object-local voxel space —
        // no transform needed.
        let skinning = if has_bones {
            Some(SkinningAssetData {
                rest_bone_aabbs: skin_meta.rest_bone_aabbs,
            })
        } else {
            None
        };

        // `ARVX_RASTER_DIAG=1` — per-LOD breakdown of the loaded cluster
        // table, plus counts of any flags / topology states that bypass
        // the Karis LOD pyramid. Used to track the mesh_raster regression
        // hypothesis: an asset whose on-disk cluster table already carries
        // `LOD_DIRTY` flags (from sculpt's `mark_lod_dirty_chains`) or
        // unbounded post-bake patch clusters (lod=0 + both DAG_GROUP_NONE
        // + cluster_error=0 + parent_group_error=ROOT) will force-admit
        // at LOD-0 on every frame, inflating raster cost.
        if std::env::var("ARVX_RASTER_DIAG").is_ok() && !meshlet_clusters.is_empty() {
            use arvx_core::mesh_cluster::{CLUSTER_FLAG_LOD_DIRTY, DAG_GROUP_NONE, PARENT_GROUP_ERROR_ROOT};
            let max_lod = meshlet_clusters.iter().map(|c| c.lod_level).max().unwrap_or(0);
            let mut clusters_per_lod = vec![0u32; max_lod as usize + 1];
            let mut indices_per_lod = vec![0u32; max_lod as usize + 1];
            let mut lod_dirty_per_lod = vec![0u32; max_lod as usize + 1];
            let mut patch_count = 0u32;
            let dag_present = !dag_groups.is_empty();
            for c in &meshlet_clusters {
                let l = c.lod_level as usize;
                if l < clusters_per_lod.len() {
                    clusters_per_lod[l] += 1;
                    indices_per_lod[l] += c.index_count;
                    if c.flags & CLUSTER_FLAG_LOD_DIRTY != 0 {
                        lod_dirty_per_lod[l] += 1;
                    }
                }
                // "Post-bake patch" heuristic: LOD-0 cluster with no DAG
                // membership on either side AND a leaf+root error pair.
                // Bake-time LOD-0 leaves have `group_below_idx == NONE`
                // but `group_above_idx` points into `dag_groups` when the
                // DAG goes past LOD-0; only sculpt's appended patches
                // have BOTH set to NONE. For v5 files (no DAG topology),
                // every cluster has both = NONE so this heuristic is
                // meaningless — gate on `dag_present`.
                if dag_present
                    && c.lod_level == 0
                    && c.group_above_idx == DAG_GROUP_NONE
                    && c.group_below_idx == DAG_GROUP_NONE
                    && c.cluster_error == 0.0
                    && c.parent_group_error >= PARENT_GROUP_ERROR_ROOT * 0.5
                {
                    patch_count += 1;
                }
            }
            let per_lod_str: String = clusters_per_lod
                .iter()
                .enumerate()
                .map(|(l, &n)| {
                    format!(
                        "lod{l}={}c/{}tri (dirty={})",
                        n,
                        indices_per_lod[l] / 3,
                        lod_dirty_per_lod[l],
                    )
                })
                .collect::<Vec<_>>()
                .join(" ");
            let lod_dirty_total: u32 = lod_dirty_per_lod.iter().sum();
            eprintln!(
                "[raster_diag load] {}: total={} | {} | LOD_DIRTY={} patch_clusters={} dag_present={}",
                arvx_path.display(),
                meshlet_clusters.len(),
                per_lod_str,
                lod_dirty_total,
                patch_count,
                dag_present,
            );
        }

        // D7 — build the cluster spatial index over the loaded
        // LOD-0 clusters so the first sculpt stamp doesn't pay a
        // full linear scan. Grid origin matches the convention in
        // `clusters_in_brush_grid_aabb`: `aabb_center - extent/2`.
        let mut cluster_spatial_index =
            super::cluster_spatial_index::ClusterSpatialIndex::new();
        {
            let extent_f = (1u32 << handle.depth) as f32 * voxel_size;
            let aabb_center = (aabb.min + aabb.max) * 0.5;
            let grid_origin = aabb_center - glam::Vec3::splat(extent_f * 0.5);
            cluster_spatial_index.rebuild(&meshlet_clusters, grid_origin, voxel_size);
        }

        // The slab allocator considers the whole bake-time index buffer
        // as "in use" (every range is owned by a `MeshletCluster`'s
        // `(index_offset, index_count)`). New patch / filter
        // allocations append past `next_free` or reuse interior slots
        // we free during sculpt. Dirty range is `[0, len * 4)` so the
        // first upload pushes the freshly-loaded IBO to the GPU.
        let mesh_indices_next_free = mesh_indices.len() as u32;
        let mut mesh_indices_dirty = arvx_core::DirtyRanges::new();
        let total_bytes = (mesh_indices.len() as u32)
            .saturating_mul(super::types::MESH_INDEX_STRIDE);
        if total_bytes > 0 {
            mesh_indices_dirty.mark_full(total_bytes);
        }
        let mut mesh_vertices_dirty = arvx_core::DirtyRanges::new();
        let vbo_total_bytes = (mesh_vertices.len()
            * std::mem::size_of::<crate::mesh_pass::MeshVertex>()) as u32;
        if vbo_total_bytes > 0 {
            mesh_vertices_dirty.mark_full(vbo_total_bytes);
        }

        Ok(AssetEntry {
            path: arvx_path.to_path_buf(),
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
            mesh_vertices,
            mesh_indices,
            mesh_indices_free_list: Vec::new(),
            mesh_indices_next_free,
            mesh_indices_dirty,
            mesh_vertices_dirty,
            mesh_lod0_index_count,
            bake_time_cluster_count: meshlet_clusters.len() as u32,
            meshlet_clusters,
            dag_groups,
            dag_consumed,
            dag_produced,
            cpu_octree: tree,
            mesh_dirty: true,
            clusters_dirty: true,
            cluster_spatial_index,
            sculpt_extra_slots: std::collections::HashSet::new(),
            sculpt_owned_slots: rustc_hash::FxHashSet::default(),
            halo_extra_slots: std::collections::HashSet::new(),
            // Disk-loaded non-terrain assets have no halo by
            // construction; the slice stays empty. Terrain tiles
            // populate this through `integrate_baked_tile`. Phase 4.4
            // (.arvxtile load) will revisit this for tiles loaded
            // from disk.
            halo_cells: Vec::new(),
        })
    }

    /// Peek at an asset's skinning metadata. Returns `None` when the
    /// asset was imported without bone weights.
    pub fn skinning_data(&self, handle: AssetHandle) -> Option<&SkinningAssetData> {
        self.asset_cache.get(handle)?.skinning.as_ref()
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
    /// &DirtyRanges, lod0_index_count)` for every loaded asset that
    /// is mesh-dirty. The `DirtyRanges` reference targets
    /// `mesh_indices` byte offsets — the renderer iterates it to
    /// drive partial IBO uploads when the slab allocator has done
    /// interior writes. Phase 6.1: `lod0_index_count` is the LOD-0
    /// prefix of the DAG IBO; the render thread caches it as the
    /// dispatch draw count.
    ///
    /// Empty-but-dirty entries are emitted: terrain hot-swap reuses
    /// released `AssetHandle` slots, and an empty re-bake at a
    /// previously-occupied slot must signal the renderer to set
    /// `mesh_buffers[idx] = None`. Without this the stale OLD mesh
    /// sits at the recycled slot and renders at the NEW entity's
    /// world position — the "mountain-on-an-empty-tile" visual
    /// corruption observed when stamping near a hot-swap boundary.
    pub fn iter_loaded_asset_meshes(
        &self,
    ) -> impl Iterator<
        Item = (
            AssetHandle,
            &[crate::mesh_pass::MeshVertex],
            &[u32],
            &arvx_core::DirtyRanges,
            &arvx_core::DirtyRanges,
            u32,
        ),
    > {
        self.asset_cache
            .entries
            .iter()
            .enumerate()
            .filter_map(|(idx, slot)| {
                let entry = slot.as_ref()?;
                if !entry.mesh_dirty {
                    return None;
                }
                Some((
                    AssetHandle::from_raw(idx as u32),
                    entry.mesh_vertices.as_slice(),
                    entry.mesh_indices.as_slice(),
                    &entry.mesh_vertices_dirty,
                    &entry.mesh_indices_dirty,
                    entry.mesh_lod0_index_count,
                ))
            })
    }

    /// Iterator over `(AssetHandle, &[MeshletCluster])` for every
    /// cluster-dirty loaded asset. The render thread uploads these
    /// to a per-asset GPU storage buffer once per geometry epoch,
    /// parallel to `iter_loaded_asset_meshes`.
    ///
    /// Empty-but-dirty entries are emitted for the same reason as
    /// the mesh iterator — an empty re-bake at a recycled handle
    /// must clear `mesh_cluster_buffers[idx]`.
    pub fn iter_loaded_asset_clusters(
        &self,
    ) -> impl Iterator<Item = (AssetHandle, &[crate::mesh_pass::MeshletCluster])> {
        self.asset_cache
            .entries
            .iter()
            .enumerate()
            .filter_map(|(idx, slot)| {
                let entry = slot.as_ref()?;
                if !entry.clusters_dirty {
                    return None;
                }
                Some((
                    AssetHandle::from_raw(idx as u32),
                    entry.meshlet_clusters.as_slice(),
                ))
            })
    }

    /// Clear every loaded asset's `mesh_dirty / clusters_dirty`
    /// flags. The render thread calls this after each geometry-epoch
    /// upload loop completes — assets that didn't upload
    /// (already-clean entries) are no-op writes, but any entry the
    /// upload touched gets its dirty flag dropped so the next epoch
    /// bump doesn't re-upload it.
    pub fn mark_loaded_asset_uploads_clean(&mut self) {
        for slot in self.asset_cache.entries.iter_mut() {
            if let Some(entry) = slot.as_mut() {
                entry.mesh_dirty = false;
                entry.clusters_dirty = false;
                // Slab-allocator dirty ranges live in lockstep with
                // `mesh_dirty` — they were just consumed by
                // `upload_mesh_for_asset` to drive partial IBO writes.
                entry.mesh_indices_dirty.clear();
                entry.mesh_vertices_dirty.clear();
            }
        }
    }
}
