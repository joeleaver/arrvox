//! .rkp v1 file format — octree-native splat asset serialization.
//!
//! The .rkp format stores a sparse octree as the spatial structure instead of a
//! flat brick map. The octree IS the LOD hierarchy — no separate LOD levels.
//! Coarser leaves at shallower depths represent lower detail.
//!
//! # File layout
//!
//! ```text
//! [RkpHeader]                 128 bytes, fixed
//! [octree nodes]              LZ4 compressed, u32 per node
//! [brick data]                LZ4 compressed, per leaf: 512 VoxelSamples (4096 bytes)
//! [geometry data]             LZ4 compressed, per leaf: BrickGeometry (occupancy + surface)
//! [color data (optional)]     LZ4 compressed, per leaf: ColorBrick (2048 bytes)
//! [bone data (optional)]      LZ4 compressed, per leaf: BoneBrick (4096 bytes)
//! ```
//!
//! Leaf bricks are stored in slot order. The slot-to-leaf mapping is implicit:
//! iterate the octree's leaves to recover which slot corresponds to which spatial
//! position.

use std::io::{Read, Seek, SeekFrom, Write};

use bytemuck::{Pod, Zeroable};

/// File magic: "RKP\x01"
pub const RKP_MAGIC: [u8; 4] = [b'R', b'K', b'P', 0x01];

/// Current format version.
pub const RKP_VERSION: u32 = 1;

/// Flags for optional sections.
pub const FLAG_HAS_COLOR: u32 = 1 << 0;
pub const FLAG_HAS_BONES: u32 = 1 << 1;

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
    /// Number of leaf bricks (with allocated brick pool slots).
    pub brick_count: u32,
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
    /// Compressed size of brick data section.
    pub brick_compressed_size: u32,
    /// Compressed size of geometry section.
    pub geometry_compressed_size: u32,
    /// Compressed size of color section (0 if no color).
    pub color_compressed_size: u32,
    /// Compressed size of bone section (0 if no bones).
    pub bone_compressed_size: u32,
    /// Reserved for future use.
    pub _reserved: [u8; 4],
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

/// Write a .rkp file.
///
/// `octree_nodes`: packed node buffer from `SparseOctree::as_slice()`.
/// `brick_data`: raw VoxelSample data, 512 entries per leaf brick, in slot order.
/// `geometry_data`: serialized BrickGeometry per leaf (occupancy + surface voxels).
/// `color_data`: optional serialized ColorBrick data per leaf.
/// `bone_data`: optional serialized BoneBrick data per leaf.
pub fn write_rkp<W: Write + Seek>(
    writer: &mut W,
    octree_nodes: &[u32],
    octree_depth: u8,
    base_voxel_size: f32,
    brick_count: u32,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
    material_ids: &[u16],
    brick_data: &[u8],
    geometry_data: &[u8],
    color_data: Option<&[u8]>,
    bone_data: Option<&[u8]>,
) -> Result<(), RkpFileError> {
    // Compress sections.
    let octree_bytes: &[u8] = bytemuck::cast_slice(octree_nodes);
    let octree_compressed = lz4_flex::compress_prepend_size(octree_bytes);
    let brick_compressed = lz4_flex::compress_prepend_size(brick_data);
    let geometry_compressed = lz4_flex::compress_prepend_size(geometry_data);
    let color_compressed = color_data.map(|d| lz4_flex::compress_prepend_size(d));
    let bone_compressed = bone_data.map(|d| lz4_flex::compress_prepend_size(d));

    let mut flags = 0u32;
    if color_data.is_some() {
        flags |= FLAG_HAS_COLOR;
    }
    if bone_data.is_some() {
        flags |= FLAG_HAS_BONES;
    }

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
        brick_count,
        aabb_min,
        aabb_max,
        flags,
        material_ids: mat_ids,
        analytical_type: 0,
        analytical_params: [0.0; 4],
        octree_compressed_size: octree_compressed.len() as u32,
        brick_compressed_size: brick_compressed.len() as u32,
        geometry_compressed_size: geometry_compressed.len() as u32,
        color_compressed_size: color_compressed.as_ref().map(|d| d.len() as u32).unwrap_or(0),
        bone_compressed_size: bone_compressed.as_ref().map(|d| d.len() as u32).unwrap_or(0),
        _reserved: [0; 4],
    };

    writer.write_all(bytemuck::bytes_of(&header))?;
    writer.write_all(&octree_compressed)?;
    writer.write_all(&brick_compressed)?;
    writer.write_all(&geometry_compressed)?;
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

/// Read and decompress the brick data section.
pub fn read_rkp_bricks<R: Read>(
    reader: &mut R,
    header: &RkpHeader,
) -> Result<Vec<u8>, RkpFileError> {
    let mut compressed = vec![0u8; header.brick_compressed_size as usize];
    reader.read_exact(&mut compressed)?;
    lz4_flex::decompress_size_prepended(&compressed)
        .map_err(|e| RkpFileError::Decompress(e.to_string()))
}

/// Read and decompress the geometry data section.
pub fn read_rkp_geometry<R: Read>(
    reader: &mut R,
    header: &RkpHeader,
) -> Result<Vec<u8>, RkpFileError> {
    let mut compressed = vec![0u8; header.geometry_compressed_size as usize];
    reader.read_exact(&mut compressed)?;
    lz4_flex::decompress_size_prepended(&compressed)
        .map_err(|e| RkpFileError::Decompress(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn header_size_is_128_bytes() {
        assert_eq!(std::mem::size_of::<RkpHeader>(), 128);
    }

    #[test]
    fn write_and_read_header_roundtrip() {
        let mut buf = Vec::new();
        let mut cursor = Cursor::new(&mut buf);

        let octree_nodes: Vec<u32> = vec![0xFFFF_FFFF]; // single EMPTY root
        let brick_data: Vec<u8> = Vec::new();
        let geometry_data: Vec<u8> = Vec::new();

        write_rkp(
            &mut cursor,
            &octree_nodes,
            1,
            0.1,
            0,
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
            &[0, 1],
            &brick_data,
            &geometry_data,
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
        assert_eq!(header.brick_count, 0);
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
            &[0u8; 4096], // one brick = 512 VoxelSamples * 8 bytes
            &[],
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
