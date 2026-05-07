//! .rkp file writers: header, octree, voxel data, color, normals, bricks,
//! skin metadata, with optional progress reporting.

use std::io::{Seek, Write};

use super::{
    FLAG_HAS_BONES, FLAG_HAS_BRICKS, FLAG_HAS_COLOR, FLAG_HAS_NORMALS, RKP_MAGIC,
    RKP_VERSION, RkpFileError, RkpHeader, SkinMetaIn, encode_skin_meta, write_stage,
};

/// Pre-built mesh + cluster DAG to ship in a v5 .rkp. All four
/// fields populated together (or `None` for the whole struct);
/// partial population isn't supported — the renderer expects the
/// triplet to be self-consistent.
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
}

/// Thin wrapper that delegates to [`write_rkp_with_progress`] without
/// emitting any progress. Kept for callers (including the rkp-core
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
) -> Result<(), RkpFileError> {
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
        None,
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
    progress: Option<&dyn Fn(&'static str)>,
) -> Result<(), RkpFileError> {
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
    tick(write_stage::WRITE_FILE);

    let mut flags = 0u32;
    if color_data.is_some()   { flags |= FLAG_HAS_COLOR; }
    if skin_meta.is_some()    { flags |= FLAG_HAS_BONES; }
    if normals_data.is_some() { flags |= FLAG_HAS_NORMALS; }
    if bricks_data.is_some()  { flags |= FLAG_HAS_BRICKS; }

    let mut mat_ids = [0u16; 16];
    for (i, &id) in material_ids.iter().take(16).enumerate() {
        mat_ids[i] = id;
    }

    let header = RkpHeader {
        magic: RKP_MAGIC,
        version: RKP_VERSION,
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

    Ok(())
}

/// Serialize a [`BakeArtifact`](crate::voxelize_octree::BakeArtifact) to
/// a `.rkp` file on disk, atomically. The artifact's file-local
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
    // per-slot material + blend + normal from the .rkp), so we store
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
    // Same one-time cost the rkp-import path pays — moves DAG
    // build out of the editor's load critical path.
    let (mesh_vertex_bytes, mesh_index_bytes, meshlet_cluster_bytes, lod0_index_count) =
        build_mesh_sections_blob(
            artifact.octree.as_slice(),
            artifact.octree.depth(),
            voxel_size,
            artifact.grid_origin,
            &bricks_flat,
            &artifact.leaf_attrs,
        );
    let mesh_sections = if !mesh_vertex_bytes.is_empty() {
        Some(MeshSectionsIn {
            vertices: &mesh_vertex_bytes,
            indices: &mesh_index_bytes,
            clusters: &meshlet_cluster_bytes,
            lod0_index_count,
        })
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
        write_rkp(
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
            None,
            mesh_sections,
        )
        .map_err(|e| format!("write .rkp: {e}"))?;
    }

    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("rename {} -> {}: {e}", tmp.display(), path.display())
    })?;

    Ok(())
}

/// Run surface-mesh extraction + Karis-Nanite cluster-DAG build over
/// the asset's geometry, returning the byte buffers ready for the
/// `MeshSectionsIn` v5 sections (`vertices`, `indices`, `clusters`)
/// plus the LOD-0 index count. Empty Vecs when there's no surface
/// to extract (degenerate input).
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
) -> (Vec<u8>, Vec<u8>, Vec<u8>, u32) {
    if octree_nodes.is_empty() || leaf_attrs.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new(), 0);
    }
    let (vertices, indices_unclustered) = crate::mesh_extract::extract_surface_mesh(
        octree_nodes,
        octree_depth,
        base_voxel_size,
        grid_origin,
        brick_pool,
        leaf_attrs,
    );
    if vertices.is_empty() || indices_unclustered.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new(), 0);
    }
    let dag = crate::mesh_lod::build_cluster_dag(&vertices, &indices_unclustered);
    let lod0_index_count = dag.lod0_index_range.1 - dag.lod0_index_range.0;
    (
        bytemuck::cast_slice(&vertices).to_vec(),
        bytemuck::cast_slice(&dag.indices).to_vec(),
        bytemuck::cast_slice(&dag.clusters).to_vec(),
        lod0_index_count,
    )
}
