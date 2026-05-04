//! .rkp file writers: header, octree, voxel data, color, normals, bricks,
//! skin metadata, with optional progress reporting.

use std::io::{Seek, Write};

use super::{
    FLAG_HAS_BONES, FLAG_HAS_BRICKS, FLAG_HAS_COLOR, FLAG_HAS_NORMALS, RKP_MAGIC,
    RKP_VERSION, RkpFileError, RkpHeader, SkinMetaIn, encode_skin_meta, write_stage,
};

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
        )
        .map_err(|e| format!("write .rkp: {e}"))?;
    }

    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("rename {} -> {}: {e}", tmp.display(), path.display())
    })?;

    Ok(())
}
