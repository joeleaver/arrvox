//! .rkp v4 file format — brick-terminated sparse octree asset.
//!
//! The octree terminates at `depth - BRICK_LEVELS`; leaf regions are packed
//! as 4³ bricks of flat cells. Each cell stores a slot index (into the
//! parallel per-voxel arrays) or `BRICK_CELL_EMPTY`. Deeper (4-voxel)
//! subregions that are uniformly interior collapse to `INTERIOR_NODE`;
//! uniformly exterior regions stay `EMPTY`. The octree IS the LOD.
//!
//! # File layout (v4)
//!
//! ```text
//! [RkpHeader]                 128 bytes, fixed
//! [octree nodes]              LZ4 compressed, u32 per node (BRICK refs)
//! [voxel data]                LZ4 compressed, per slot: 1 VoxelSample (8 bytes)
//! [normals (optional)]        LZ4 compressed, per slot: 1 u32 (octahedrally-packed normal)
//! [bricks (optional)]         LZ4 compressed, BRICK_CELLS u32s per brick
//! [color data (optional)]     LZ4 compressed, per slot: 1 ColorVoxel (4 bytes)
//! [skin meta (optional)]      LZ4 compressed, self-describing:
//!                               u32 bone_voxel_byte_len,
//!                               BoneVoxel × voxel_count,
//!                               u32 brick_origin_count,
//!                               [u32; 3] × brick_origin_count,
//!                               u32 rest_aabb_count,
//!                               [f32; 6] × rest_aabb_count
//! ```
//!
//! The skin-meta section consolidates everything the Phase-3 scatter
//! pass needs: per-leaf bone influences (weights + indices), per-brick
//! origins (for deriving rest voxel centres without walking the octree
//! at load), and per-bone rest-pose AABBs (for sizing the deformed
//! bone field each frame). All three are deterministic from the
//! voxelization, so they ship pre-computed rather than being rebuilt
//! on every load. Carried in the pre-existing `bone_compressed_size`
//! header slot — no header version bump.
//!
//! v4 replaces v3's per-voxel LEAF encoding with BRICK-terminated octrees,
//! matching the procedural voxelizer's on-GPU representation. Rays can now
//! take a flat DDA through 4³ cells instead of descending the final two
//! octree levels per step.
//!
//! Leaf voxels are stored in slot order. The slot-to-leaf mapping is implicit:
//! iterate the octree's leaves to recover which slot corresponds to which spatial
//! position.

use std::io::{Read, Seek, Write};

use bytemuck::{Pod, Zeroable};

/// File magic: "RKP\x01"
pub const RKP_MAGIC: [u8; 4] = [b'R', b'K', b'P', 0x01];

/// Current format version.
pub const RKP_VERSION: u32 = 4;

/// Flags for optional sections.
pub const FLAG_HAS_COLOR: u32 = 1 << 0;
pub const FLAG_HAS_BONES: u32 = 1 << 1;
pub const FLAG_HAS_NORMALS: u32 = 1 << 2;
pub const FLAG_HAS_BRICKS: u32 = 1 << 3;

/// .rkp file header (128 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct RkpHeader {
    /// File magic: `RKP_MAGIC`.
    pub magic: [u8; 4],
    /// Format version.
    pub version: u32,
    /// Number of octree nodes (u32 entries).
    pub octree_node_count: u32,
    /// Octree depth (max levels).
    pub octree_depth: u32,
    /// Base voxel size at finest level.
    pub base_voxel_size: f32,
    /// Number of leaf voxels (with allocated voxel pool slots).
    pub voxel_count: u32,
    /// Object AABB (world-space).
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    /// Feature flags (FLAG_HAS_COLOR, FLAG_HAS_BONES).
    pub flags: u32,
    /// Material palette (up to 16 material IDs used by this asset).
    pub material_ids: [u16; 16],
    /// Analytical primitive type (for LOD fallback). 0 = none.
    pub analytical_type: u32,
    /// Analytical primitive params (4 floats).
    pub analytical_params: [f32; 4],
    /// Compressed size of octree section.
    pub octree_compressed_size: u32,
    /// Compressed size of voxel data section.
    pub voxel_compressed_size: u32,
    /// Compressed size of normals section (0 if no normals). v3+.
    pub normals_compressed_size: u32,
    /// Compressed size of color section (0 if no color).
    pub color_compressed_size: u32,
    /// Compressed size of bone section (0 if no bones).
    pub bone_compressed_size: u32,
    /// Compressed size of bricks section (0 if no bricks). v4+.
    pub bricks_compressed_size: u32,
}

/// Per-write skin-meta input. Fed into [`write_rkp_with_progress`]'s
/// `skin_meta` parameter; serialised into the single LZ4 blob that the
/// file-format-level `bone` section carries.
///
/// All three slices must be populated together — missing any of them
/// means "this asset has no skinning data; write `None`".
#[derive(Debug, Clone, Copy)]
pub struct SkinMetaIn<'a> {
    /// `BoneVoxel` bytes, one entry per leaf slot.
    /// `.len() == voxel_count * sizeof::<BoneVoxel>()`.
    pub bone_voxels: &'a [u8],
    /// Brick origins in finest-voxel units, `[u32; 3]` per entry.
    /// Indexed by file-local brick id (pre-`scene_brick_offset` shift).
    pub brick_origins: &'a [[u32; 3]],
    /// Per-bone rest-pose AABB: `[min_x, min_y, min_z, max_x, max_y, max_z]`.
    /// Length `= 1 + max_bone_index_seen` (empty-AABB sentinels for
    /// unused slots).
    pub rest_bone_aabbs: &'a [[f32; 6]],
}

/// Decoded skin-meta payload produced by [`read_rkp_skin_meta`].
#[derive(Debug, Clone, Default)]
pub struct SkinMetaOut {
    pub bone_voxels: Vec<u8>,
    pub brick_origins: Vec<[u32; 3]>,
    pub rest_bone_aabbs: Vec<[f32; 6]>,
}

/// Error type for .rkp file operations.
#[derive(Debug, thiserror::Error)]
pub enum RkpFileError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid magic: expected RKP\\x01")]
    BadMagic,
    #[error("Unsupported version: {0}")]
    UnsupportedVersion(u32),
    #[error("Decompression error: {0}")]
    Decompress(String),
    #[error("Malformed skin-meta section: {0}")]
    SkinMeta(&'static str),
}

/// Serialise the three skin-meta arrays into a single byte buffer that
/// will be LZ4-compressed as the file's bones section. Wire format:
///
/// ```text
/// u32                    bone_voxel_byte_len
/// [bone_voxel_byte_len]  bone voxel bytes (BoneVoxel × voxel_count)
/// u32                    brick_origin_count
/// [[u32; 3]]             brick_origins (12 B each)
/// u32                    rest_aabb_count
/// [[f32; 6]]             rest_bone_aabbs (24 B each)
/// ```
fn encode_skin_meta(meta: &SkinMetaIn<'_>) -> Vec<u8> {
    let bv_len = meta.bone_voxels.len();
    let bo_bytes: &[u8] = bytemuck::cast_slice(meta.brick_origins);
    let aabb_bytes: &[u8] = bytemuck::cast_slice(meta.rest_bone_aabbs);
    let mut buf = Vec::with_capacity(4 + bv_len + 4 + bo_bytes.len() + 4 + aabb_bytes.len());
    buf.extend_from_slice(&(bv_len as u32).to_le_bytes());
    buf.extend_from_slice(meta.bone_voxels);
    buf.extend_from_slice(&(meta.brick_origins.len() as u32).to_le_bytes());
    buf.extend_from_slice(bo_bytes);
    buf.extend_from_slice(&(meta.rest_bone_aabbs.len() as u32).to_le_bytes());
    buf.extend_from_slice(aabb_bytes);
    buf
}

fn decode_skin_meta(raw: &[u8]) -> Result<SkinMetaOut, RkpFileError> {
    let read_u32 = |raw: &[u8], pos: &mut usize| -> Result<u32, RkpFileError> {
        if *pos + 4 > raw.len() {
            return Err(RkpFileError::SkinMeta("truncated u32"));
        }
        let v = u32::from_le_bytes(raw[*pos..*pos + 4].try_into().unwrap());
        *pos += 4;
        Ok(v)
    };

    let mut pos = 0usize;
    let bv_len = read_u32(raw, &mut pos)? as usize;
    if pos + bv_len > raw.len() {
        return Err(RkpFileError::SkinMeta("truncated bone voxels"));
    }
    let bone_voxels = raw[pos..pos + bv_len].to_vec();
    pos += bv_len;

    let bo_count = read_u32(raw, &mut pos)? as usize;
    let bo_bytes = bo_count.checked_mul(12).ok_or(RkpFileError::SkinMeta("brick origin overflow"))?;
    if pos + bo_bytes > raw.len() {
        return Err(RkpFileError::SkinMeta("truncated brick origins"));
    }
    let brick_origins: Vec<[u32; 3]> = bytemuck::cast_slice(&raw[pos..pos + bo_bytes]).to_vec();
    pos += bo_bytes;

    let aabb_count = read_u32(raw, &mut pos)? as usize;
    let aabb_bytes = aabb_count.checked_mul(24).ok_or(RkpFileError::SkinMeta("rest aabb overflow"))?;
    if pos + aabb_bytes > raw.len() {
        return Err(RkpFileError::SkinMeta("truncated rest aabbs"));
    }
    let rest_bone_aabbs: Vec<[f32; 6]> = bytemuck::cast_slice(&raw[pos..pos + aabb_bytes]).to_vec();
    pos += aabb_bytes;

    if pos != raw.len() {
        return Err(RkpFileError::SkinMeta("trailing bytes in skin meta section"));
    }

    Ok(SkinMetaOut { bone_voxels, brick_origins, rest_bone_aabbs })
}

/// Write a .rkp v3 file (per-voxel format).
///
/// `octree_nodes`: packed node buffer from `SparseOctree::as_slice()`.
/// `voxel_data`: raw VoxelSample data, 1 entry per leaf voxel, in slot order.
/// `normals_data`: optional per-voxel octahedrally-packed normal (u32 each).
/// `color_data`: optional per-voxel ColorVoxel data (4 bytes per leaf).
/// `skin_meta`: optional skinning metadata (bone weights, brick origins,
/// rest-pose bone AABBs) — see [`SkinMetaIn`].
/// Sub-stage labels emitted by [`write_rkp_with_progress`] through its
/// progress callback. Exposed so `rkp-import` can forward them onto
/// its own `ImportEvent::StageStart` pipeline with matching
/// `&'static str` names.
pub mod write_stage {
    pub const COMPRESS_OCTREE: &str = "compress_octree";
    pub const COMPRESS_VOXELS: &str = "compress_voxels";
    pub const COMPRESS_NORMALS: &str = "compress_normals";
    pub const COMPRESS_BRICKS: &str = "compress_bricks";
    pub const COMPRESS_COLORS: &str = "compress_colors";
    pub const COMPRESS_BONES: &str = "compress_bones";
    pub const WRITE_FILE: &str = "write_file";
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

/// Read a .rkp file header.
pub fn read_rkp_header<R: Read>(reader: &mut R) -> Result<RkpHeader, RkpFileError> {
    let mut buf = [0u8; std::mem::size_of::<RkpHeader>()];
    reader.read_exact(&mut buf)?;
    let header: RkpHeader = *bytemuck::from_bytes(&buf);

    if header.magic != RKP_MAGIC {
        return Err(RkpFileError::BadMagic);
    }
    if header.version != RKP_VERSION {
        return Err(RkpFileError::UnsupportedVersion(header.version));
    }

    Ok(header)
}

/// Read and decompress the octree nodes section.
pub fn read_rkp_octree<R: Read>(
    reader: &mut R,
    header: &RkpHeader,
) -> Result<Vec<u32>, RkpFileError> {
    let mut compressed = vec![0u8; header.octree_compressed_size as usize];
    reader.read_exact(&mut compressed)?;
    let decompressed = lz4_flex::decompress_size_prepended(&compressed)
        .map_err(|e| RkpFileError::Decompress(e.to_string()))?;
    Ok(bytemuck::cast_slice(&decompressed).to_vec())
}

/// Read and decompress the voxel data section (1 VoxelSample per leaf).
pub fn read_rkp_voxels<R: Read>(
    reader: &mut R,
    header: &RkpHeader,
) -> Result<Vec<u8>, RkpFileError> {
    let mut compressed = vec![0u8; header.voxel_compressed_size as usize];
    reader.read_exact(&mut compressed)?;
    lz4_flex::decompress_size_prepended(&compressed)
        .map_err(|e| RkpFileError::Decompress(e.to_string()))
}

/// Read and decompress the normals section (if present). One u32 per leaf
/// voxel, in slot order, octahedrally packed (see `rkp_core::leaf_attr::pack_oct`).
pub fn read_rkp_normals<R: Read>(
    reader: &mut R,
    header: &RkpHeader,
) -> Result<Vec<u8>, RkpFileError> {
    if header.normals_compressed_size == 0 {
        return Ok(Vec::new());
    }
    let mut compressed = vec![0u8; header.normals_compressed_size as usize];
    reader.read_exact(&mut compressed)?;
    lz4_flex::decompress_size_prepended(&compressed)
        .map_err(|e| RkpFileError::Decompress(e.to_string()))
}

/// Read and decompress the bricks section (if present). Flat u32 cells,
/// `BRICK_CELLS` per brick in brick-id order.
pub fn read_rkp_bricks<R: Read>(
    reader: &mut R,
    header: &RkpHeader,
) -> Result<Vec<u8>, RkpFileError> {
    if header.bricks_compressed_size == 0 {
        return Ok(Vec::new());
    }
    let mut compressed = vec![0u8; header.bricks_compressed_size as usize];
    reader.read_exact(&mut compressed)?;
    lz4_flex::decompress_size_prepended(&compressed)
        .map_err(|e| RkpFileError::Decompress(e.to_string()))
}

/// Read and decompress the color data section (if present).
pub fn read_rkp_color<R: Read>(
    reader: &mut R,
    header: &RkpHeader,
) -> Result<Vec<u8>, RkpFileError> {
    if header.color_compressed_size == 0 {
        return Ok(Vec::new());
    }
    let mut compressed = vec![0u8; header.color_compressed_size as usize];
    reader.read_exact(&mut compressed)?;
    lz4_flex::decompress_size_prepended(&compressed)
        .map_err(|e| RkpFileError::Decompress(e.to_string()))
}

/// Read and decompress the bone data section (if present).
/// Read and decode the optional skin-meta section. Returns an empty
/// `SkinMetaOut` (all three vectors empty) when the asset has no skin
/// data.
pub fn read_rkp_skin_meta<R: Read>(
    reader: &mut R,
    header: &RkpHeader,
) -> Result<SkinMetaOut, RkpFileError> {
    if header.bone_compressed_size == 0 {
        return Ok(SkinMetaOut::default());
    }
    let mut compressed = vec![0u8; header.bone_compressed_size as usize];
    reader.read_exact(&mut compressed)?;
    let raw = lz4_flex::decompress_size_prepended(&compressed)
        .map_err(|e| RkpFileError::Decompress(e.to_string()))?;
    decode_skin_meta(&raw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Cursor, SeekFrom};

    #[test]
    fn header_size_is_128_bytes() {
        assert_eq!(std::mem::size_of::<RkpHeader>(), 128);
    }

    #[test]
    fn write_and_read_header_roundtrip() {
        let mut buf = Vec::new();
        let mut cursor = Cursor::new(&mut buf);

        let octree_nodes: Vec<u32> = vec![0xFFFF_FFFF]; // single EMPTY root
        let voxel_data: Vec<u8> = Vec::new();

        write_rkp(
            &mut cursor,
            &octree_nodes,
            1,
            0.1,
            0,
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
            &[0, 1],
            &voxel_data,
            None,
            None,
            None,
            None, // skin_meta
        )
        .unwrap();

        cursor.seek(SeekFrom::Start(0)).unwrap();
        let header = read_rkp_header(&mut cursor).unwrap();

        assert_eq!(header.magic, RKP_MAGIC);
        assert_eq!(header.version, RKP_VERSION);
        assert_eq!(header.octree_node_count, 1);
        assert_eq!(header.octree_depth, 1);
        assert!((header.base_voxel_size - 0.1).abs() < 1e-6);
        assert_eq!(header.voxel_count, 0);
        assert_eq!(header.material_ids[0], 0);
        assert_eq!(header.material_ids[1], 1);
        assert_eq!(header.flags & FLAG_HAS_COLOR, 0);
        assert_eq!(header.flags & FLAG_HAS_BONES, 0);
    }

    #[test]
    fn write_and_read_skin_meta_roundtrip() {
        // Three voxels, two bricks, two bones — exercises every part
        // of the structured skin-meta payload: weights, origins, and
        // rest AABBs all survive the LZ4 + length-prefix round trip.
        use crate::companion::BoneVoxel;

        let bones: Vec<BoneVoxel> = vec![
            BoneVoxel::new([0, 1, 2, 3], [64, 64, 64, 63]),
            BoneVoxel::new([4, 0, 0, 0], [255, 0, 0, 0]),
            BoneVoxel::new([7, 3, 0, 0], [200, 55, 0, 0]),
        ];
        let bone_bytes: &[u8] = bytemuck::cast_slice(&bones);
        let brick_origins: Vec<[u32; 3]> = vec![[0, 0, 0], [8, 0, 0]];
        let rest_aabbs: Vec<[f32; 6]> = vec![
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [-1.0, -2.0, -3.0, 2.0, 3.0, 4.0],
        ];
        let voxel_bytes = vec![0u8; 3 * std::mem::size_of::<crate::voxel::VoxelSample>()];

        let mut buf = Vec::new();
        let mut cursor = Cursor::new(&mut buf);
        write_rkp(
            &mut cursor,
            &[0xFFFF_FFFF],   // single EMPTY root octree
            1,
            0.1,
            3,                // voxel_count
            [-1.0; 3], [1.0; 3],
            &[0],
            &voxel_bytes,
            None, None, None,
            Some(SkinMetaIn {
                bone_voxels: bone_bytes,
                brick_origins: &brick_origins,
                rest_bone_aabbs: &rest_aabbs,
            }),
        )
        .unwrap();

        cursor.seek(SeekFrom::Start(0)).unwrap();
        let header = read_rkp_header(&mut cursor).unwrap();
        assert!(header.flags & FLAG_HAS_BONES != 0, "FLAG_HAS_BONES must be set");
        assert!(header.bone_compressed_size > 0, "skin-meta section must be non-empty");

        let _ = read_rkp_octree(&mut cursor, &header).unwrap();
        let _ = read_rkp_voxels(&mut cursor, &header).unwrap();
        let back = read_rkp_skin_meta(&mut cursor, &header).unwrap();

        assert_eq!(back.bone_voxels, bone_bytes, "bone-voxel bytes must roundtrip");
        assert_eq!(back.brick_origins, brick_origins, "brick origins must roundtrip");
        assert_eq!(back.rest_bone_aabbs, rest_aabbs, "rest bone AABBs must roundtrip");

        // Decode bone voxels + weight-sum invariant.
        let decoded: &[BoneVoxel] = bytemuck::cast_slice(&back.bone_voxels);
        for (i, (bv_in, bv_out)) in bones.iter().zip(decoded).enumerate() {
            for slot in 0..4 {
                assert_eq!(bv_in.bone_index(slot), bv_out.bone_index(slot), "bone_index mismatch at voxel {i} slot {slot}");
                assert_eq!(bv_in.bone_weight(slot), bv_out.bone_weight(slot), "bone_weight mismatch at voxel {i} slot {slot}");
            }
            let sum: u16 = (0..4).map(|s| bv_out.bone_weight(s) as u16).sum();
            assert_eq!(sum, 255, "voxel {i} weights must sum to 255");
        }
    }

    #[test]
    fn write_artifact_rkp_roundtrip() {
        // Bake a tiny sphere into a BakeArtifact via the canonical
        // voxelize path, persist through write_artifact_rkp, then read
        // the sections back and check material/normal/brick/color
        // round-trip. This is the procedural bake-cache pipeline end
        // to end minus the scene integration.
        use crate::voxel::VoxelSample;
        use glam::Vec3;

        let voxel_size = 0.05;
        let half = Vec3::splat(0.3);
        let aabb = crate::Aabb::new(-half, half);
        let radius: f32 = 0.25;
        let sdf = |positions: &[Vec3]| -> Vec<(f32, u16, u16, u8, u32)> {
            positions
                .iter()
                .map(|p| (p.length() - radius, 7u16, 0u16, 0u8, 0u32))
                .collect()
        };

        let mut artifact = crate::voxelize_to_artifact(sdf, &aabb, voxel_size)
            .expect("voxelize sphere");
        assert!(artifact.voxel_count > 0, "sphere must produce voxels");
        // Spike a non-zero color on the first leaf so the color
        // section is emitted — verifies `has_color` detection works.
        artifact.leaf_attr_colors[0] = 0xFFAABBCC;

        let tmp = std::env::temp_dir().join(format!(
            "rkp_artifact_roundtrip_{}.rkp",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&tmp);
        write_artifact_rkp(
            &tmp,
            &artifact,
            aabb.min.to_array(),
            aabb.max.to_array(),
            voxel_size,
        )
        .expect("write_artifact_rkp");

        let mut file = std::fs::File::open(&tmp).expect("open");
        let mut reader = std::io::BufReader::new(&mut file);
        let header = read_rkp_header(&mut reader).expect("read header");
        // Header stores leaf_attr-slot count, not cell count — the
        // per-slot voxel_data length is what the loader cares about.
        // `voxelize_to_artifact` already ran prefilter, so this
        // includes internal-node attrs in addition to the surface
        // leaves. On load, a fresh prefilter appends again; the
        // unreferenced "old" prefilter attrs linger harmlessly.
        assert_eq!(header.voxel_count, artifact.leaf_attrs.len() as u32);
        assert_eq!(header.octree_depth as u8, artifact.octree.depth());
        assert!((header.base_voxel_size - voxel_size).abs() < 1e-6);
        assert!(header.flags & FLAG_HAS_BRICKS != 0);
        assert!(header.flags & FLAG_HAS_NORMALS != 0);
        assert!(header.flags & FLAG_HAS_COLOR != 0);

        let octree_nodes = read_rkp_octree(&mut reader, &header).expect("octree");
        assert_eq!(octree_nodes, artifact.octree.as_slice());

        let voxel_bytes = read_rkp_voxels(&mut reader, &header).expect("voxels");
        let voxels: &[VoxelSample] = bytemuck::cast_slice(&voxel_bytes);
        assert_eq!(voxels.len(), artifact.leaf_attrs.len());
        for (v, a) in voxels.iter().zip(artifact.leaf_attrs.iter()) {
            assert_eq!(v.material_id(), a.material_primary);
            assert_eq!(v.secondary_material_id(), a.material_secondary());
            assert_eq!(v.blend_weight(), a.blend_weight());
        }

        let normals_bytes = read_rkp_normals(&mut reader, &header).expect("normals");
        let normals: &[u32] = bytemuck::cast_slice(&normals_bytes);
        assert_eq!(normals.len(), artifact.leaf_attrs.len());
        for (n, a) in normals.iter().zip(artifact.leaf_attrs.iter()) {
            assert_eq!(*n, a.normal_oct);
        }

        let bricks_bytes = read_rkp_bricks(&mut reader, &header).expect("bricks");
        let bricks: &[u32] = bytemuck::cast_slice(&bricks_bytes);
        let expected_brick_u32s = artifact.brick_cells.len() * crate::BRICK_CELLS as usize;
        assert_eq!(bricks.len(), expected_brick_u32s);

        let color_bytes = read_rkp_color(&mut reader, &header).expect("colors");
        let colors: &[u32] = bytemuck::cast_slice(&color_bytes);
        assert_eq!(colors.len(), artifact.leaf_attr_colors.len());
        assert_eq!(colors[0], 0xFFAABBCC);

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn write_and_read_octree_roundtrip() {
        let mut buf = Vec::new();
        let mut cursor = Cursor::new(&mut buf);

        let octree_nodes: Vec<u32> = vec![1, 0xFFFF_FFFF, 0x8000_002A, 0xFFFF_FFFF,
                                          0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF,
                                          0xFFFF_FFFF];

        write_rkp(
            &mut cursor,
            &octree_nodes,
            1,
            0.1,
            1,
            [-1.0; 3],
            [1.0; 3],
            &[],
            &[0u8; 8], // one voxel = 1 VoxelSample * 8 bytes
            None,
            None,
            None,
            None, // skin_meta
        )
        .unwrap();

        cursor.seek(SeekFrom::Start(0)).unwrap();
        let header = read_rkp_header(&mut cursor).unwrap();
        let nodes = read_rkp_octree(&mut cursor, &header).unwrap();

        assert_eq!(nodes, octree_nodes);
    }
}
