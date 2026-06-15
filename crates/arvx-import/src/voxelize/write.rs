//! Atomic `.arvx` + `.arvxskel` writing with staging paths.
//!
//! Strategy: each artifact is written to a sibling `.inprogress`
//! path, then renamed into place on success. A mid-write failure
//! (disk full, panic, user kill) leaves the old file untouched —
//! next asset-acquire call sees the previous version.
//!
//! The `.arvx` is written first (it's the primary artifact); the
//! `.arvxskel` second so a rename failure there doesn't leave the user
//! with a fresh skeleton pointing at stale geometry.

use std::path::{Path, PathBuf};

use glam::Vec3;

use arvx_core::Aabb;
use arvx_animation::skeleton_asset::{SkeletonAsset, save_rkskel};
use arvx_core::asset_file::write_stage;

use crate::event::{ImportEvent, ProgressReporter};
use crate::normalize::NormalizationParams;
use crate::skeleton::SkeletonExtraction;

use super::shell::ShellOutput;

/// Build the sibling `.inprogress` staging path. We append rather than
/// swap extensions so the `.arvx` / `.arvxskel` suffix the asset scanner
/// keys off is preserved.
fn staging_path(final_path: &Path) -> PathBuf {
    let mut s = final_path.as_os_str().to_owned();
    s.push(".inprogress");
    PathBuf::from(s)
}

/// Writes an `.arvx` to `output_path` atomically. Takes a closure
/// `serialize` so the caller controls file-format details (voxel/
/// brick/color/normal payload assembly) while this module owns the
/// staging + rename logic.
pub fn write_rkp_atomic<F>(output_path: &Path, serialize: F) -> Result<u64, String>
where
    F: FnOnce(&mut std::io::BufWriter<std::fs::File>) -> Result<(), String>,
{
    let tmp = staging_path(output_path);
    let _ = std::fs::remove_file(&tmp);

    {
        let file = std::fs::File::create(&tmp).map_err(|e| format!("create .arvx: {e}"))?;
        let mut writer = std::io::BufWriter::new(file);
        if let Err(e) = serialize(&mut writer) {
            drop(writer);
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
    }

    let size = std::fs::metadata(&tmp).map(|m| m.len()).unwrap_or(0);

    if let Err(e) = std::fs::rename(&tmp, output_path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("rename .arvx: {e}"));
    }
    Ok(size)
}

/// Assemble + atomically write the `.arvx` file from a [`ShellOutput`]
/// plus surrounding context (octree, geometry AABB, voxel size,
/// material table). Returns the on-disk file size in bytes. The
/// `reporter` receives `ImportEvent::StageStart` for each LZ4 sub-
/// stage so long writes (millions of voxels) don't look stuck.
#[allow(clippy::too_many_arguments)]
pub fn write_rkp(
    output_path: &Path,
    shell: &ShellOutput,
    octree_nodes: &[u32],
    octree_depth: u8,
    voxel_size: f32,
    aabb: &Aabb,
    material_ids: &[u16],
    reporter: &dyn ProgressReporter,
) -> Result<u64, String> {
    // Expand AABB by one voxel so outer-shell voxels (one voxel
    // beyond the mesh surface on the outside) fall inside the
    // geometry bounds.
    let shell_margin = Vec3::splat(voxel_size);
    let geometry_aabb = Aabb::new(aabb.min - shell_margin, aabb.max + shell_margin);

    let voxel_bytes: &[u8] = bytemuck::cast_slice(&shell.voxel_data);
    let normals_bytes: &[u8] = bytemuck::cast_slice(&shell.normals_packed);
    let bricks_bytes: &[u8] = bytemuck::cast_slice(&shell.file_bricks);
    let color_bytes: Option<&[u8]> = if shell.has_color {
        Some(bytemuck::cast_slice(&shell.color_voxels))
    } else {
        None
    };
    // Skin metadata — bone weights + brick origins + per-bone rest
    // AABBs — only emitted when a skinned skeleton was resolved during
    // import. The three arrays are shipped together in a structured
    // LZ4 blob (see `asset_file::SkinMetaIn`).
    let skin_meta: Option<arvx_core::asset_file::SkinMetaIn<'_>> = if shell.has_bones {
        Some(arvx_core::asset_file::SkinMetaIn {
            bone_voxels: bytemuck::cast_slice(&shell.bone_voxels),
            brick_origins: &shell.brick_origins,
            rest_bone_aabbs: &shell.rest_bone_aabbs,
        })
    } else {
        None
    };

    // Translate arvx-core's section-boundary ticks into per-stage
    // ImportEvents so the UI shows "Compressing octree" → "Compressing
    // voxels" → ... → "Writing file" instead of a single opaque
    // "Writing" stage that can sit frozen for minutes on huge assets.
    let progress_cb = |label: &'static str| {
        let message = match label {
            write_stage::COMPRESS_OCTREE => "Compressing octree",
            write_stage::COMPRESS_VOXELS => "Compressing voxel data",
            write_stage::COMPRESS_NORMALS => "Compressing normals",
            write_stage::COMPRESS_BRICKS => "Compressing bricks",
            write_stage::COMPRESS_COLORS => "Compressing colors",
            write_stage::COMPRESS_BONES => "Compressing bones",
            write_stage::WRITE_FILE => "Writing file",
            _ => label,
        };
        reporter.report(ImportEvent::StageStart {
            stage: label,
            message: message.to_string(),
        });
    };

    // Build the surface mesh + Karis-Nanite cluster DAG up front so
    // the editor doesn't have to rebuild it every load. The
    // extractor only reads `LeafAttr.normal_oct`, so we synthesize a
    // LeafAttr Vec from the normals-packed array (other fields zero).
    // `leaf_attr_id`s baked into the vertices are FILE-LOCAL — the
    // load path adds the scene-global offset before any GPU upload,
    // mirroring how brick ids are remapped.
    use arvx_core::leaf_attr::LeafAttr;
    let leaf_attrs: Vec<LeafAttr> = shell
        .normals_packed
        .iter()
        .map(|&n| LeafAttr {
            normal_oct: n,
            material_primary: 0,
            material_secondary_blend: 0,
        })
        .collect();
    let asset_extent = (1u32 << octree_depth) as f32 * voxel_size;
    let aabb_center = (geometry_aabb.min + geometry_aabb.max) * 0.5;
    let grid_origin = aabb_center - Vec3::splat(asset_extent * 0.5);
    let mesh_blob = arvx_core::asset_file::build_mesh_sections_blob(
        octree_nodes,
        octree_depth,
        voxel_size,
        grid_origin,
        &shell.file_bricks,
        &leaf_attrs,
        // For skinned imports, bone weights come straight from the
        // voxelizer's per-cell BoneVoxel quads; the mesh extractor
        // pulls them by `leaf_attr_id` and bakes them into the vertex
        // so the mesh VS can do LBS/DQS without a runtime merge.
        // Empty for unskinned assets — extractor zeros the fields.
        &shell.bone_voxels,
    );
    let mesh_sections = if !mesh_blob.vertices.is_empty() {
        Some(mesh_blob.as_in())
    } else {
        None
    };

    // v7 per-leaf distance section. The voxelizer captured the true
    // (Euclidean, unit-gradient) signed distance to the mesh surface per
    // shell leaf; persist it so a loaded import re-extracts / sculpts
    // with Manifold-DC instead of the blur fallback (the read side,
    // asset_load::build_loaded_asset, sets has_distances from it). 1:1
    // with the leaves (== voxel_count); empty only for a degenerate bake.
    let distance_bytes: &[u8] = if shell.dists.is_empty() {
        &[]
    } else {
        bytemuck::cast_slice(&shell.dists)
    };

    write_rkp_atomic(output_path, |writer| {
        arvx_core::asset_file::write_rkp_with_progress(
            writer,
            octree_nodes,
            octree_depth,
            voxel_size,
            shell.voxel_count,
            geometry_aabb.min.to_array(),
            geometry_aabb.max.to_array(),
            material_ids,
            voxel_bytes,
            if normals_bytes.is_empty() { None } else { Some(normals_bytes) },
            if bricks_bytes.is_empty() { None } else { Some(bricks_bytes) },
            color_bytes,
            skin_meta,
            mesh_sections,
            if distance_bytes.is_empty() { None } else { Some(distance_bytes) },
            Some(&progress_cb),
        )
        .map_err(|e| format!("write .arvx: {e}"))
    })
}

/// Write the `.arvxskel` skeleton sidecar to `<output_path>.arvxskel`,
/// atomically. Returns the final path on success. Failure is soft —
/// returns `Ok(None)` with a stderr warning so the `.arvx` still ships
/// even if the skeleton save fails. Used only when the source file
/// contains skinning data.
pub fn write_rkskel(
    output_path: &Path,
    extraction: &SkeletonExtraction,
    norm: &NormalizationParams,
) -> Result<Option<PathBuf>, String> {
    let final_path = output_path.with_extension("arvxskel");
    let tmp = staging_path(&final_path);
    let _ = std::fs::remove_file(&tmp);

    let asset = SkeletonAsset::with_normalization(
        extraction.skeleton.clone(),
        extraction.clips.clone(),
        norm.center.to_array(),
        norm.scale,
        norm.rotation_offset,
        norm.rotation_center.to_array(),
    );

    if let Err(e) = save_rkskel(&asset, &tmp) {
        eprintln!("[arvx-import] warn: failed to save .arvxskel: {e}");
        let _ = std::fs::remove_file(&tmp);
        return Ok(None);
    }

    match std::fs::rename(&tmp, &final_path) {
        Ok(()) => Ok(Some(final_path)),
        Err(e) => {
            eprintln!("[arvx-import] warn: failed to swap .arvxskel into place: {e}");
            let _ = std::fs::remove_file(&tmp);
            Ok(None)
        }
    }
}
