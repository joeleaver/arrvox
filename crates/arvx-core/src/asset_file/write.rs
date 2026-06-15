//! .arvx file writers: header, octree, voxel data, color, normals, bricks,
//! skin metadata, with optional progress reporting.

use std::io::{Seek, Write};

use super::{
    FLAG_HAS_BONES, FLAG_HAS_BRICKS, FLAG_HAS_COLOR, FLAG_HAS_DISTANCE, FLAG_HAS_NORMALS, ARVX_MAGIC,
    ARVX_VERSION, ArvxFileError, ArvxHeader, SkinMetaIn, encode_skin_meta, write_stage,
};

/// Pre-built mesh + cluster DAG to ship in a v5+ .arvx. The first
/// four fields populated together (or `None` for the whole struct);
/// partial population isn't supported — the renderer expects the
/// triplet to be self-consistent.
///
/// The trailing three DAG-topology fields (`dag_groups`,
/// `dag_consumed`, `dag_produced`) are v6+. v5 writers leave them
/// empty; v5 readers see zero-size DAG sections and fall back to
/// asset-wide LOD-0 marking on sculpt.
#[derive(Debug, Clone, Copy)]
pub struct MeshSectionsIn<'a> {
    /// `MeshVertex` bytes from `extract_surface_mesh`. 32 B per vertex.
    pub vertices: &'a [u8],
    /// Concatenated index buffer across all LOD levels, LOD-0 first.
    /// `bytemuck`-castable from `&[u32]`.
    pub indices: &'a [u8],
    /// `MeshletCluster` bytes from `build_cluster_dag`. 64 B each.
    pub clusters: &'a [u8],
    /// Length of the LOD-0 prefix in `indices` (number of u32 entries).
    pub lod0_index_count: u32,
    /// `DagGroup` bytes from `build_cluster_dag`. 16 B each. Empty
    /// `&[]` when the DAG converged at LOD 0 (no simplification).
    pub dag_groups: &'a [u8],
    /// Flat per-group consumed cluster IDs, `bytemuck`-castable from
    /// `&[u32]`.
    pub dag_consumed: &'a [u8],
    /// Flat per-group produced cluster IDs, `bytemuck`-castable from
    /// `&[u32]`.
    pub dag_produced: &'a [u8],
}

/// Thin wrapper that delegates to [`write_rkp_with_progress`] without
/// emitting any progress. Kept for callers (including the arvx-core
/// tests) that don't want progress reporting.
#[allow(clippy::too_many_arguments)]
pub fn write_rkp<W: Write + Seek>(
    writer: &mut W,
    octree_nodes: &[u32],
    octree_depth: u8,
    base_voxel_size: f32,
    voxel_count: u32,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
    material_ids: &[u16],
    voxel_data: &[u8],
    normals_data: Option<&[u8]>,
    bricks_data: Option<&[u8]>,
    color_data: Option<&[u8]>,
    skin_meta: Option<SkinMetaIn<'_>>,
    mesh_sections: Option<MeshSectionsIn<'_>>,
) -> Result<(), ArvxFileError> {
    write_rkp_with_progress(
        writer,
        octree_nodes,
        octree_depth,
        base_voxel_size,
        voxel_count,
        aabb_min,
        aabb_max,
        material_ids,
        voxel_data,
        normals_data,
        bricks_data,
        color_data,
        skin_meta,
        mesh_sections,
        None, // distance_data
        None, // progress
    )
}

/// Like [`write_rkp`] but fires the optional `progress` callback with
/// [`write_stage`] labels as each LZ4 compression section begins and
/// again when the final file write starts. Lets callers render a
/// live per-section progress indicator during large writes (an
/// elephant-scale voxel bake can spend 30+ seconds here, almost
/// entirely inside single-threaded LZ4 calls).
#[allow(clippy::too_many_arguments)]
pub fn write_rkp_with_progress<W: Write + Seek>(
    writer: &mut W,
    octree_nodes: &[u32],
    octree_depth: u8,
    base_voxel_size: f32,
    voxel_count: u32,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
    material_ids: &[u16],
    voxel_data: &[u8],
    normals_data: Option<&[u8]>,
    bricks_data: Option<&[u8]>,
    color_data: Option<&[u8]>,
    skin_meta: Option<SkinMetaIn<'_>>,
    mesh_sections: Option<MeshSectionsIn<'_>>,
    distance_data: Option<&[u8]>,
    progress: Option<&dyn Fn(&'static str)>,
) -> Result<(), ArvxFileError> {
    let tick = |label: &'static str| {
        if let Some(cb) = progress {
            cb(label);
        }
    };

    tick(write_stage::COMPRESS_OCTREE);
    let octree_bytes: &[u8] = bytemuck::cast_slice(octree_nodes);
    let octree_compressed = lz4_flex::compress_prepend_size(octree_bytes);
    tick(write_stage::COMPRESS_VOXELS);
    let voxel_compressed = lz4_flex::compress_prepend_size(voxel_data);
    let normals_compressed = normals_data.map(|d| {
        tick(write_stage::COMPRESS_NORMALS);
        lz4_flex::compress_prepend_size(d)
    });
    let bricks_compressed = bricks_data.map(|d| {
        tick(write_stage::COMPRESS_BRICKS);
        lz4_flex::compress_prepend_size(d)
    });
    let color_compressed = color_data.map(|d| {
        tick(write_stage::COMPRESS_COLORS);
        lz4_flex::compress_prepend_size(d)
    });
    // Skin meta is encoded structurally (bone weights + brick origins
    // + rest-bone AABBs) then LZ4'd as one blob. The header's
    // `bone_compressed_size` measures that whole blob.
    let skin_meta_blob: Option<Vec<u8>> = skin_meta.as_ref().map(encode_skin_meta);
    let bone_compressed = skin_meta_blob.as_deref().map(|d| {
        tick(write_stage::COMPRESS_BONES);
        lz4_flex::compress_prepend_size(d)
    });
    let mesh_vertices_compressed = mesh_sections.map(|m| {
        tick(write_stage::COMPRESS_MESH_VERTICES);
        lz4_flex::compress_prepend_size(m.vertices)
    });
    let mesh_indices_compressed = mesh_sections.map(|m| {
        tick(write_stage::COMPRESS_MESH_INDICES);
        lz4_flex::compress_prepend_size(m.indices)
    });
    let meshlet_clusters_compressed = mesh_sections.map(|m| {
        tick(write_stage::COMPRESS_MESHLET_CLUSTERS);
        lz4_flex::compress_prepend_size(m.clusters)
    });
    // v6+ DAG topology sections. We always run the compression step
    // when `mesh_sections` is present (even if the inner slice is
    // empty for LOD-0-only assets) so the on-disk layout is uniform
    // across asset sizes — readers branch on `header.dag_*_size == 0`.
    let dag_groups_compressed = mesh_sections.and_then(|m| {
        if m.dag_groups.is_empty() {
            None
        } else {
            tick(write_stage::COMPRESS_DAG_GROUPS);
            Some(lz4_flex::compress_prepend_size(m.dag_groups))
        }
    });
    let dag_consumed_compressed = mesh_sections.and_then(|m| {
        if m.dag_consumed.is_empty() {
            None
        } else {
            tick(write_stage::COMPRESS_DAG_CONSUMED);
            Some(lz4_flex::compress_prepend_size(m.dag_consumed))
        }
    });
    let dag_produced_compressed = mesh_sections.and_then(|m| {
        if m.dag_produced.is_empty() {
            None
        } else {
            tick(write_stage::COMPRESS_DAG_PRODUCED);
            Some(lz4_flex::compress_prepend_size(m.dag_produced))
        }
    });
    // v7+ per-leaf signed distance. Skip when empty (no distance baked) so
    // the flag/size stay 0 and the loader takes the blur fallback.
    let distance_compressed = distance_data.and_then(|d| {
        if d.is_empty() {
            None
        } else {
            Some(lz4_flex::compress_prepend_size(d))
        }
    });
    tick(write_stage::WRITE_FILE);

    let mut flags = 0u32;
    if color_data.is_some()   { flags |= FLAG_HAS_COLOR; }
    if skin_meta.is_some()    { flags |= FLAG_HAS_BONES; }
    if normals_data.is_some() { flags |= FLAG_HAS_NORMALS; }
    if bricks_data.is_some()  { flags |= FLAG_HAS_BRICKS; }
    if distance_compressed.is_some() { flags |= FLAG_HAS_DISTANCE; }

    let mut mat_ids = [0u16; 16];
    for (i, &id) in material_ids.iter().take(16).enumerate() {
        mat_ids[i] = id;
    }

    let header = ArvxHeader {
        magic: ARVX_MAGIC,
        version: ARVX_VERSION,
        octree_node_count: octree_nodes.len() as u32,
        octree_depth: octree_depth as u32,
        base_voxel_size,
        voxel_count,
        aabb_min,
        aabb_max,
        flags,
        material_ids: mat_ids,
        analytical_type: 0,
        analytical_params: [0.0; 4],
        octree_compressed_size: octree_compressed.len() as u32,
        voxel_compressed_size: voxel_compressed.len() as u32,
        normals_compressed_size: normals_compressed.as_ref().map(|d| d.len() as u32).unwrap_or(0),
        color_compressed_size: color_compressed.as_ref().map(|d| d.len() as u32).unwrap_or(0),
        bone_compressed_size: bone_compressed.as_ref().map(|d| d.len() as u32).unwrap_or(0),
        bricks_compressed_size: bricks_compressed.as_ref().map(|d| d.len() as u32).unwrap_or(0),
        mesh_vertices_compressed_size: mesh_vertices_compressed
            .as_ref().map(|d| d.len() as u32).unwrap_or(0),
        mesh_indices_compressed_size: mesh_indices_compressed
            .as_ref().map(|d| d.len() as u32).unwrap_or(0),
        meshlet_clusters_compressed_size: meshlet_clusters_compressed
            .as_ref().map(|d| d.len() as u32).unwrap_or(0),
        mesh_lod0_index_count: mesh_sections.map(|m| m.lod0_index_count).unwrap_or(0),
        dag_groups_compressed_size: dag_groups_compressed
            .as_ref().map(|d| d.len() as u32).unwrap_or(0),
        dag_consumed_compressed_size: dag_consumed_compressed
            .as_ref().map(|d| d.len() as u32).unwrap_or(0),
        dag_produced_compressed_size: dag_produced_compressed
            .as_ref().map(|d| d.len() as u32).unwrap_or(0),
        distance_compressed_size: distance_compressed
            .as_ref().map(|d| d.len() as u32).unwrap_or(0),
    };

    writer.write_all(bytemuck::bytes_of(&header))?;
    writer.write_all(&octree_compressed)?;
    writer.write_all(&voxel_compressed)?;
    if let Some(ref data) = normals_compressed {
        writer.write_all(data)?;
    }
    if let Some(ref data) = bricks_compressed {
        writer.write_all(data)?;
    }
    if let Some(ref data) = color_compressed {
        writer.write_all(data)?;
    }
    if let Some(ref data) = bone_compressed {
        writer.write_all(data)?;
    }
    if let Some(ref data) = mesh_vertices_compressed {
        writer.write_all(data)?;
    }
    if let Some(ref data) = mesh_indices_compressed {
        writer.write_all(data)?;
    }
    if let Some(ref data) = meshlet_clusters_compressed {
        writer.write_all(data)?;
    }
    if let Some(ref data) = dag_groups_compressed {
        writer.write_all(data)?;
    }
    if let Some(ref data) = dag_consumed_compressed {
        writer.write_all(data)?;
    }
    if let Some(ref data) = dag_produced_compressed {
        writer.write_all(data)?;
    }
    // v7 distance section — written LAST so v4-v6 readers (which stop
    // after dag_produced) are unaffected, and read_rkp_distance reads it
    // after the DAG sections.
    if let Some(ref data) = distance_compressed {
        writer.write_all(data)?;
    }

    Ok(())
}

/// Serialize a [`BakeArtifact`](crate::voxelize_octree::BakeArtifact) to
/// a `.arvx` file on disk, atomically. The artifact's file-local
/// leaf_attr and brick IDs are passed through unchanged —
/// `load_asset_from_disk` already handles remapping to scene-global IDs
/// on read. Writes first to `{path}.inprogress`, then renames into
/// place so a mid-write failure leaves any pre-existing file
/// untouched. Creates the parent directory if missing.
///
/// Used by the async bake worker to persist procedural bakes alongside
/// the scene file. No skin-meta is emitted (procedurals aren't
/// skinned); the color section is skipped when the artifact has no
/// per-voxel overrides.
pub fn write_artifact_rkp(
    path: &std::path::Path,
    artifact: &crate::voxelize_octree::BakeArtifact,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
    voxel_size: f32,
) -> Result<(), String> {
    use crate::voxel::VoxelSample;

    let voxel_count = artifact.leaf_attrs.len() as u32;

    // LeafAttr material fields round-trip through load; the saved
    // VoxelSample distance is never read (the shader only reads
    // per-slot material + blend + normal from the .arvx), so we store
    // zero.
    let voxel_samples: Vec<VoxelSample> = artifact
        .leaf_attrs
        .iter()
        .map(|a| {
            VoxelSample::new_blended(
                0.0,
                a.material_primary,
                a.material_secondary(),
                a.blend_weight(),
            )
        })
        .collect();
    let voxel_bytes: &[u8] = bytemuck::cast_slice(&voxel_samples);

    let normals: Vec<u32> = artifact
        .leaf_attrs
        .iter()
        .map(|a| a.normal_oct)
        .collect();
    let normals_bytes: &[u8] = bytemuck::cast_slice(&normals);

    let bricks_flat: Vec<u32> = artifact
        .brick_cells
        .iter()
        .flat_map(|c| c.iter().copied())
        .collect();
    let bricks_bytes: &[u8] = bytemuck::cast_slice(&bricks_flat);

    let has_color = artifact.leaf_attr_colors.iter().any(|&c| c != 0);
    let color_bytes: Option<&[u8]> = if has_color {
        Some(bytemuck::cast_slice(&artifact.leaf_attr_colors))
    } else {
        None
    };

    let material_ids: [u16; 0] = [];

    // Pre-build the surface mesh + Karis-Nanite cluster DAG so the
    // editor doesn't have to rebuild it at load. `leaf_attr_id`s
    // baked into the vertices are file-local; the load path adds
    // the asset's global leaf_attr offset before any GPU upload.
    // Same one-time cost the arvx-import path pays — moves DAG
    // build out of the editor's load critical path.
    //
    // **Terrain vs procedural mesh path.** A non-empty `halo_cells`
    // marks a TERRAIN tile bake (procedural object bakes call
    // `voxelize_to_artifact` with `halo = 0` → no halo cells). Terrain
    // meshes through the density-blur path so the persisted mesh
    // matches the in-RAM `bake_tile` output bit-for-bit (smooth, no
    // staircase). Procedural / imported bakes stay on the BINARY path
    // — the blur rounds sub-2-voxel sharp edges, which hard-surface
    // assets must keep. The halo width is derived from the actual halo
    // cells' reach so the seam neighborhood is correct.
    let halo_width: u32 = artifact
        .halo_cells
        .iter()
        .map(|(c, _)| {
            // Distance past the nominal `[0, extent)` cube on the
            // furthest axis — the halo band width.
            let extent = 1i32 << artifact.octree.depth();
            let over = |v: i32| -> i32 {
                if v < 0 {
                    -v
                } else if v >= extent {
                    v - extent + 1
                } else {
                    0
                }
            };
            over(c.x).max(over(c.y)).max(over(c.z))
        })
        .max()
        .unwrap_or(0) as u32;
    let mesh_blob = if artifact.halo_cells.is_empty() {
        build_mesh_sections_blob_haloed(
            artifact.octree.as_slice(),
            artifact.octree.depth(),
            voxel_size,
            artifact.grid_origin,
            &bricks_flat,
            &artifact.leaf_attrs,
            // Procedural bakes never carry skinning data — generator
            // outputs are static geometry.
            &[],
            &artifact.halo_cells,
            halo_width,
        )
    } else {
        build_mesh_sections_blob_density_haloed(
            artifact.octree.as_slice(),
            artifact.octree.depth(),
            voxel_size,
            artifact.grid_origin,
            &bricks_flat,
            &artifact.leaf_attrs,
            &[],
            &artifact.halo_cells,
            halo_width,
            // Terrain bake → QEF-Hermite when the artifact carries distances.
            &artifact.leaf_attr_dists,
            // No procedural source here (artifact write / import) → `∇D`
            // shading. Live terrain bakes supply the analytic normal in
            // `bake_tile`; this path re-extracts a stored artifact.
            None,
        )
    };
    let mesh_sections = if !mesh_blob.vertices.is_empty() {
        Some(mesh_blob.as_in())
    } else {
        None
    };

    let tmp = {
        let mut s = path.as_os_str().to_owned();
        s.push(".inprogress");
        std::path::PathBuf::from(s)
    };
    let _ = std::fs::remove_file(&tmp);

    if let Some(parent) = tmp.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("create parent {}: {e}", parent.display()))?;
    }

    {
        let file = std::fs::File::create(&tmp)
            .map_err(|e| format!("create {}: {e}", tmp.display()))?;
        let mut writer = std::io::BufWriter::new(file);
        // v7 per-leaf signed-distance section. Forward the artifact's
        // quantized dists (indexed 1:1 with `leaf_attrs`, hence with the
        // header `voxel_count = leaf_attrs.len()`) so a save→reload of a
        // QEF/Manifold-DC bake can re-extract / sculpt from the stored
        // field instead of falling back to blur. Empty when the bake
        // carried no distances (e.g. an old artifact) → no section
        // written, flag stays unset. `write_rkp_with_progress` writes it
        // LAST so v4-v6 readers are unaffected.
        debug_assert!(
            artifact.leaf_attr_dists.is_empty()
                || artifact.leaf_attr_dists.len() == voxel_count as usize,
            "distance section must be 1:1 with leaf_attrs ({} dists vs {voxel_count} voxels)",
            artifact.leaf_attr_dists.len(),
        );
        let distance_bytes: &[u8] = if artifact.leaf_attr_dists.is_empty() {
            &[]
        } else {
            bytemuck::cast_slice(&artifact.leaf_attr_dists)
        };
        write_rkp_with_progress(
            &mut writer,
            artifact.octree.as_slice(),
            artifact.octree.depth(),
            voxel_size,
            voxel_count,
            aabb_min,
            aabb_max,
            &material_ids,
            voxel_bytes,
            Some(normals_bytes),
            Some(bricks_bytes),
            color_bytes,
            None, // skin_meta
            mesh_sections,
            if distance_bytes.is_empty() { None } else { Some(distance_bytes) },
            None, // progress
        )
        .map_err(|e| format!("write .arvx: {e}"))?;
    }

    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("rename {} -> {}: {e}", tmp.display(), path.display())
    })?;

    Ok(())
}

/// Owned byte buffers for the v6 mesh + DAG sections, ready to feed
/// into [`MeshSectionsIn`] for [`write_rkp`]. All seven slices empty
/// when there's no surface to extract.
#[derive(Debug, Default, Clone)]
pub struct MeshSectionsBlob {
    pub vertices: Vec<u8>,
    pub indices: Vec<u8>,
    pub clusters: Vec<u8>,
    pub lod0_index_count: u32,
    pub dag_groups: Vec<u8>,
    pub dag_consumed: Vec<u8>,
    pub dag_produced: Vec<u8>,
}

impl MeshSectionsBlob {
    /// Borrow as a [`MeshSectionsIn`] for the writer.
    pub fn as_in(&self) -> MeshSectionsIn<'_> {
        MeshSectionsIn {
            vertices: &self.vertices,
            indices: &self.indices,
            clusters: &self.clusters,
            lod0_index_count: self.lod0_index_count,
            dag_groups: &self.dag_groups,
            dag_consumed: &self.dag_consumed,
            dag_produced: &self.dag_produced,
        }
    }
}

/// Run surface-mesh extraction + Karis-Nanite cluster-DAG build over
/// the asset's geometry, returning the byte buffers for the v6
/// mesh + DAG sections. Empty fields when there's no surface to
/// extract (degenerate input).
///
/// `leaf_attr_id`s baked into the vertices are FILE-LOCAL — i.e.,
/// indexes into the asset's own `leaf_attrs` Vec. The load path is
/// responsible for adding the scene-global leaf_attr offset before
/// any GPU upload, the same way it relocates `brick_id`s today.
pub fn build_mesh_sections_blob(
    octree_nodes: &[u32],
    octree_depth: u8,
    base_voxel_size: f32,
    grid_origin: glam::Vec3,
    brick_pool: &[u32],
    leaf_attrs: &[crate::leaf_attr::LeafAttr],
    bone_voxels: &[crate::companion::BoneVoxel],
) -> MeshSectionsBlob {
    build_mesh_sections_blob_haloed(
        octree_nodes,
        octree_depth,
        base_voxel_size,
        grid_origin,
        brick_pool,
        leaf_attrs,
        bone_voxels,
        &[],
        0,
    )
}

/// Halo-aware variant of [`build_mesh_sections_blob`]. See
/// [`crate::mesh_extract::extract_surface_mesh_haloed`] for the
/// ownership rule that turns the halo data into watertight tile
/// seams. With `halo = 0` this is bit-identical to the non-halo entry.
#[allow(clippy::too_many_arguments)]
pub fn build_mesh_sections_blob_haloed(
    octree_nodes: &[u32],
    octree_depth: u8,
    base_voxel_size: f32,
    grid_origin: glam::Vec3,
    brick_pool: &[u32],
    leaf_attrs: &[crate::leaf_attr::LeafAttr],
    bone_voxels: &[crate::companion::BoneVoxel],
    halo_cells: &[(glam::IVec3, u32)],
    halo: u32,
) -> MeshSectionsBlob {
    if octree_nodes.is_empty() || leaf_attrs.is_empty() {
        return MeshSectionsBlob::default();
    }
    let (vertices, indices_unclustered) = crate::mesh_extract::extract_surface_mesh_haloed(
        octree_nodes,
        octree_depth,
        base_voxel_size,
        grid_origin,
        brick_pool,
        leaf_attrs,
        bone_voxels,
        halo_cells,
        halo,
        None, // bake-time extract: no sculpt history yet.
    );
    if vertices.is_empty() || indices_unclustered.is_empty() {
        return MeshSectionsBlob::default();
    }
    mesh_sections_from_extract(vertices, indices_unclustered)
}

/// **Terrain-only** density-blur variant of
/// [`build_mesh_sections_blob_haloed`]. Meshes the tile through the
/// SAME blurred-occupancy → `D = 0.5`-topology + `∇D`-normal path the
/// sculpt re-extract uses
/// ([`crate::mesh_extract::extract_mesh_region_from_cells_pooled_haloed`]),
/// so baked terrain is smooth from occupancy alone — no post-hoc
/// heightfield Y-projection needed.
///
/// **Why a separate entry (not a flag on the binary path).** Imports
/// may be hard-surface; the `R = 2` blur rounds sub-2-voxel sharp edges
/// (a documented fundamental limit), so the binary
/// [`crate::mesh_extract::extract_surface_mesh_haloed`] stays the
/// import/asset-load default. ONLY the terrain bake opts into blur via
/// this function.
///
/// **Watertight seams.** The blurred density `D[c]` is a pure local
/// function of occupancy in `c ± DENSITY_KERNEL_R`; with the terrain
/// halo (`TILE_HALO_VOXELS = 4`) `≥` the kernel reach (`R + 1 = 3`), two
/// adjacent tiles share the full boundary neighborhood and compute
/// bit-identical seam vertices. The halo cells are folded into the
/// extract's `cells_grid` for 8-corner data exactly like the sculpt
/// path; they are never added to the iterating owner set, so no quad is
/// double-emitted from the wrong side. See
/// [`crate::mesh_extract::extract_surface_mesh_haloed`] for the seam
/// protocol.
///
/// `halo_cells` carry the tile's neighbor-cell occupancy (sampled by
/// `voxelize_to_artifact` with `halo > 0`). `halo` is the halo width in
/// finest-grid voxels (the terrain bake passes `TILE_HALO_VOXELS`).
/// With `halo = 0` and no halo cells the mesh is still smooth (the blur
/// reads the tile's own occupancy), it just has no cross-tile boundary
/// data — the terrain bake always passes both.
#[allow(clippy::too_many_arguments)]
pub fn build_mesh_sections_blob_density_haloed(
    octree_nodes: &[u32],
    octree_depth: u8,
    base_voxel_size: f32,
    grid_origin: glam::Vec3,
    brick_pool: &[u32],
    leaf_attrs: &[crate::leaf_attr::LeafAttr],
    bone_voxels: &[crate::companion::BoneVoxel],
    halo_cells: &[(glam::IVec3, u32)],
    halo: u32,
    // Per-leaf signed distances (voxel units), indexed like `leaf_attrs`.
    // Non-empty selects the QEF-Hermite mesher (the terrain bake passes the
    // artifact's `leaf_attr_dists`); `&[]` keeps the blur fallback.
    dists: &[i16],
    // Optional analytic shading-normal callback (world → outward `∇sd`).
    // In QEF mode the vertex normal is this evaluated at the vertex (the
    // EXACT surface normal) when supplied; `None` falls back to the
    // interpolated `∇D` field. The terrain bake passes
    // `terrain_fn.sample_grad`-at-world on pure-procedural tiles.
    surface_normal_fn: Option<&dyn Fn(glam::Vec3) -> glam::Vec3>,
) -> MeshSectionsBlob {
    if octree_nodes.is_empty() || leaf_attrs.is_empty() {
        return MeshSectionsBlob::default();
    }
    let (vertices, indices_unclustered) =
        crate::mesh_extract::extract_surface_mesh_density_haloed(
            octree_nodes,
            octree_depth,
            base_voxel_size,
            grid_origin,
            brick_pool,
            leaf_attrs,
            bone_voxels,
            halo_cells,
            halo,
            // Bake-time extract: no sculpt history yet.
            None,
            dists,
            surface_normal_fn,
        );
    if vertices.is_empty() || indices_unclustered.is_empty() {
        return MeshSectionsBlob::default();
    }
    mesh_sections_from_extract(vertices, indices_unclustered)
}

/// Shared cluster-DAG build + blob assembly for the
/// `build_mesh_sections_blob*` family. Takes the raw extract output
/// (unclustered vertices + indices) and runs the Karis-Nanite
/// cluster-DAG build, returning the v6 section byte buffers. Empty
/// input → [`MeshSectionsBlob::default`].
fn mesh_sections_from_extract(
    vertices: Vec<crate::mesh_extract::MeshVertex>,
    indices_unclustered: Vec<u32>,
) -> MeshSectionsBlob {
    if vertices.is_empty() || indices_unclustered.is_empty() {
        return MeshSectionsBlob::default();
    }
    let dag = crate::mesh_lod::build_cluster_dag(&vertices, &indices_unclustered);
    let lod0_index_count = dag.lod0_index_range.1 - dag.lod0_index_range.0;
    MeshSectionsBlob {
        vertices: bytemuck::cast_slice(&vertices).to_vec(),
        indices: bytemuck::cast_slice(&dag.indices).to_vec(),
        clusters: bytemuck::cast_slice(&dag.clusters).to_vec(),
        lod0_index_count,
        dag_groups: bytemuck::cast_slice(&dag.dag_groups).to_vec(),
        dag_consumed: bytemuck::cast_slice(&dag.dag_consumed).to_vec(),
        dag_produced: bytemuck::cast_slice(&dag.dag_produced).to_vec(),
    }
}
