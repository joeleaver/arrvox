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
//! [bone data (optional)]      LZ4 compressed, per slot: 1 BoneVoxel (8 bytes)
//! ```
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
}

/// Write a .rkp v3 file (per-voxel format).
///
/// `octree_nodes`: packed node buffer from `SparseOctree::as_slice()`.
/// `voxel_data`: raw VoxelSample data, 1 entry per leaf voxel, in slot order.
/// `normals_data`: optional per-voxel octahedrally-packed normal (u32 each).
/// `color_data`: optional per-voxel ColorVoxel data (4 bytes per leaf).
/// `bone_data`: optional per-voxel BoneVoxel data.
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
    bone_data: Option<&[u8]>,
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
        bone_data,
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
    bone_data: Option<&[u8]>,
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
    let bone_compressed = bone_data.map(|d| {
        tick(write_stage::COMPRESS_BONES);
        lz4_flex::compress_prepend_size(d)
    });
    tick(write_stage::WRITE_FILE);

    let mut flags = 0u32;
    if color_data.is_some()   { flags |= FLAG_HAS_COLOR; }
    if bone_data.is_some()    { flags |= FLAG_HAS_BONES; }
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
pub fn read_rkp_bones<R: Read>(
    reader: &mut R,
    header: &RkpHeader,
) -> Result<Vec<u8>, RkpFileError> {
    if header.bone_compressed_size == 0 {
        return Ok(Vec::new());
    }
    let mut compressed = vec![0u8; header.bone_compressed_size as usize];
    reader.read_exact(&mut compressed)?;
    lz4_flex::decompress_size_prepended(&compressed)
        .map_err(|e| RkpFileError::Decompress(e.to_string()))
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
            None,
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
            None,
        )
        .unwrap();

        cursor.seek(SeekFrom::Start(0)).unwrap();
        let header = read_rkp_header(&mut cursor).unwrap();
        let nodes = read_rkp_octree(&mut cursor, &header).unwrap();

        assert_eq!(nodes, octree_nodes);
    }
}
