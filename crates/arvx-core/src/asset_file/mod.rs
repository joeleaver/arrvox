//! .arvx v4 file format — brick-terminated sparse octree asset.
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
//! [ArvxHeader]                 128 bytes, fixed
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
//! [mesh vertices (optional)]  v5+: LZ4, MeshVertex × N (32 B each)
//! [mesh indices (optional)]   v5+: LZ4, u32 × M (concatenated all-LODs)
//! [meshlet clusters (opt)]    v5+: LZ4, MeshletCluster × K (64 B each)
//! ```
//!
//! v5 adds three pre-built mesh sections so the editor doesn't have
//! to re-extract the surface mesh + rebuild the Karis-Nanite cluster
//! DAG every load. Build is done once at import (or procedural bake)
//! and the result ships in the .arvx. v4 files are no longer
//! supported — re-import to migrate.
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

use std::io::Read;

mod write;
pub mod proxy;

#[cfg(test)]
mod tests;

pub use write::{
    build_mesh_sections_blob, write_artifact_rkp, write_rkp, write_rkp_with_progress,
    MeshSectionsIn,
};
pub use proxy::{
    read_arvxproxy, write_arvxproxy, ArvxProxyError, ArvxProxyHeader, ProxyCache,
    ARVXPROXY_MAGIC, ARVXPROXY_VERSION,
};

use bytemuck::{Pod, Zeroable};

/// File magic: "AVX\x01"
pub const ARVX_MAGIC: [u8; 4] = [b'A', b'V', b'X', 0x01];

/// Current format version.
///
/// v6 (current): adds three baked DAG topology sections
/// (`dag_groups`, `dag_consumed`, `dag_produced`) that mirror
/// [`arvx_core::mesh_lod::ClusterDag`]. Lets sculpt's per-chain
/// LOD-0 clamp walk the bipartite cluster ↔ group graph without
/// rebuilding the DAG at load. v5 and v4 still load — the load
/// path leaves the new sections empty for v5 and falls through to
/// the in-process `build_cluster_dag` rebuild for v4.
pub const ARVX_VERSION: u32 = 6;

/// Flags for optional sections.
pub const FLAG_HAS_COLOR: u32 = 1 << 0;
pub const FLAG_HAS_BONES: u32 = 1 << 1;
pub const FLAG_HAS_NORMALS: u32 = 1 << 2;
pub const FLAG_HAS_BRICKS: u32 = 1 << 3;

/// .arvx file header (128 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ArvxHeader {
    /// File magic: `ARVX_MAGIC`.
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
    /// Compressed size of mesh vertices section. v5+. 0 if absent.
    pub mesh_vertices_compressed_size: u32,
    /// Compressed size of mesh indices section. v5+. 0 if absent.
    pub mesh_indices_compressed_size: u32,
    /// Compressed size of meshlet clusters section. v5+. 0 if absent.
    pub meshlet_clusters_compressed_size: u32,
    /// Length of the LOD-0 prefix in `mesh_indices` (number of u32
    /// indices). Lets the renderer access just the LOD-0 portion
    /// without walking the cluster table. v5+. 0 if absent.
    pub mesh_lod0_index_count: u32,
    /// Compressed size of DAG groups section. v6+. 0 if absent. Sized
    /// `dag_groups.len() * size_of::<DagGroup>()` (16 B each).
    pub dag_groups_compressed_size: u32,
    /// Compressed size of DAG consumed-cluster-IDs section. v6+. 0 if
    /// absent. Sized `dag_consumed.len() * 4` (one u32 per entry).
    pub dag_consumed_compressed_size: u32,
    /// Compressed size of DAG produced-cluster-IDs section. v6+. 0 if
    /// absent. Sized `dag_produced.len() * 4` (one u32 per entry).
    pub dag_produced_compressed_size: u32,
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

/// Error type for .arvx file operations.
#[derive(Debug, thiserror::Error)]
pub enum ArvxFileError {
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

fn decode_skin_meta(raw: &[u8]) -> Result<SkinMetaOut, ArvxFileError> {
    let read_u32 = |raw: &[u8], pos: &mut usize| -> Result<u32, ArvxFileError> {
        if *pos + 4 > raw.len() {
            return Err(ArvxFileError::SkinMeta("truncated u32"));
        }
        let v = u32::from_le_bytes(raw[*pos..*pos + 4].try_into().unwrap());
        *pos += 4;
        Ok(v)
    };

    let mut pos = 0usize;
    let bv_len = read_u32(raw, &mut pos)? as usize;
    if pos + bv_len > raw.len() {
        return Err(ArvxFileError::SkinMeta("truncated bone voxels"));
    }
    let bone_voxels = raw[pos..pos + bv_len].to_vec();
    pos += bv_len;

    let bo_count = read_u32(raw, &mut pos)? as usize;
    let bo_bytes = bo_count.checked_mul(12).ok_or(ArvxFileError::SkinMeta("brick origin overflow"))?;
    if pos + bo_bytes > raw.len() {
        return Err(ArvxFileError::SkinMeta("truncated brick origins"));
    }
    let brick_origins: Vec<[u32; 3]> = bytemuck::cast_slice(&raw[pos..pos + bo_bytes]).to_vec();
    pos += bo_bytes;

    let aabb_count = read_u32(raw, &mut pos)? as usize;
    let aabb_bytes = aabb_count.checked_mul(24).ok_or(ArvxFileError::SkinMeta("rest aabb overflow"))?;
    if pos + aabb_bytes > raw.len() {
        return Err(ArvxFileError::SkinMeta("truncated rest aabbs"));
    }
    let rest_bone_aabbs: Vec<[f32; 6]> = bytemuck::cast_slice(&raw[pos..pos + aabb_bytes]).to_vec();
    pos += aabb_bytes;

    if pos != raw.len() {
        return Err(ArvxFileError::SkinMeta("trailing bytes in skin meta section"));
    }

    Ok(SkinMetaOut { bone_voxels, brick_origins, rest_bone_aabbs })
}

/// Write a .arvx v3 file (per-voxel format).
///
/// `octree_nodes`: packed node buffer from `SparseOctree::as_slice()`.
/// `voxel_data`: raw VoxelSample data, 1 entry per leaf voxel, in slot order.
/// `normals_data`: optional per-voxel octahedrally-packed normal (u32 each).
/// `color_data`: optional per-voxel ColorVoxel data (4 bytes per leaf).
/// `skin_meta`: optional skinning metadata (bone weights, brick origins,
/// rest-pose bone AABBs) — see [`SkinMetaIn`].
/// Sub-stage labels emitted by [`write_rkp_with_progress`] through its
/// progress callback. Exposed so `arvx-import` can forward them onto
/// its own `ImportEvent::StageStart` pipeline with matching
/// `&'static str` names.
pub mod write_stage {
    pub const COMPRESS_OCTREE: &str = "compress_octree";
    pub const COMPRESS_VOXELS: &str = "compress_voxels";
    pub const COMPRESS_NORMALS: &str = "compress_normals";
    pub const COMPRESS_BRICKS: &str = "compress_bricks";
    pub const COMPRESS_COLORS: &str = "compress_colors";
    pub const COMPRESS_BONES: &str = "compress_bones";
    pub const COMPRESS_MESH_VERTICES: &str = "compress_mesh_vertices";
    pub const COMPRESS_MESH_INDICES: &str = "compress_mesh_indices";
    pub const COMPRESS_MESHLET_CLUSTERS: &str = "compress_meshlet_clusters";
    pub const COMPRESS_DAG_GROUPS: &str = "compress_dag_groups";
    pub const COMPRESS_DAG_CONSUMED: &str = "compress_dag_consumed";
    pub const COMPRESS_DAG_PRODUCED: &str = "compress_dag_produced";
    pub const WRITE_FILE: &str = "write_file";
}


/// Header size for the v4 layout (128 B). v5 added 16 trailing
/// bytes (3 mesh-section sizes + lod0_index_count). v6 added another
/// 12 bytes (3 DAG-section sizes). Reader detects version after the
/// first 8 bytes (magic + version) and reads the rest accordingly so
/// existing v4/v5 files keep loading.
const V4_HEADER_SIZE: usize = 128;
const V5_HEADER_SIZE: usize = 144;
const V6_HEADER_SIZE: usize = std::mem::size_of::<ArvxHeader>();

/// Read a .arvx file header. Accepts v4 (128 B), v5 (144 B), and v6
/// (156 B); older paths zero-fill the missing fields so the
/// renderer's fallbacks (build DAG at load for v4; empty DAG groups
/// → asset-wide LOD-0 clamp for v5) kick in.
pub fn read_rkp_header<R: Read>(reader: &mut R) -> Result<ArvxHeader, ArvxFileError> {
    // Peek magic + version first.
    let mut prefix = [0u8; 8];
    reader.read_exact(&mut prefix)?;
    let magic: [u8; 4] = prefix[0..4].try_into().unwrap();
    if magic != ARVX_MAGIC {
        return Err(ArvxFileError::BadMagic);
    }
    let version = u32::from_le_bytes(prefix[4..8].try_into().unwrap());

    let body_len = match version {
        4 => V4_HEADER_SIZE - 8,
        5 => V5_HEADER_SIZE - 8,
        6 => V6_HEADER_SIZE - 8,
        v => return Err(ArvxFileError::UnsupportedVersion(v)),
    };
    let mut body = vec![0u8; body_len];
    reader.read_exact(&mut body)?;

    // Reassemble into a fixed-size v6 buffer; older versions' tails
    // are zeroed, which leaves the corresponding section-size fields
    // at zero. The mesh-section readers short-circuit on size == 0,
    // and the renderer falls back to in-process DAG build (v4) or
    // empty DAG groups → asset-wide marking (v5).
    let mut buf = [0u8; V6_HEADER_SIZE];
    buf[..8].copy_from_slice(&prefix);
    buf[8..8 + body_len].copy_from_slice(&body);
    let header: ArvxHeader = *bytemuck::from_bytes(&buf);
    Ok(header)
}

/// Read and decompress the octree nodes section.
pub fn read_rkp_octree<R: Read>(
    reader: &mut R,
    header: &ArvxHeader,
) -> Result<Vec<u32>, ArvxFileError> {
    let mut compressed = vec![0u8; header.octree_compressed_size as usize];
    reader.read_exact(&mut compressed)?;
    let decompressed = lz4_flex::decompress_size_prepended(&compressed)
        .map_err(|e| ArvxFileError::Decompress(e.to_string()))?;
    Ok(bytemuck::cast_slice(&decompressed).to_vec())
}

/// Read and decompress the voxel data section (1 VoxelSample per leaf).
pub fn read_rkp_voxels<R: Read>(
    reader: &mut R,
    header: &ArvxHeader,
) -> Result<Vec<u8>, ArvxFileError> {
    let mut compressed = vec![0u8; header.voxel_compressed_size as usize];
    reader.read_exact(&mut compressed)?;
    lz4_flex::decompress_size_prepended(&compressed)
        .map_err(|e| ArvxFileError::Decompress(e.to_string()))
}

/// Read and decompress the normals section (if present). One u32 per leaf
/// voxel, in slot order, octahedrally packed (see `arvx_core::leaf_attr::pack_oct`).
pub fn read_rkp_normals<R: Read>(
    reader: &mut R,
    header: &ArvxHeader,
) -> Result<Vec<u8>, ArvxFileError> {
    if header.normals_compressed_size == 0 {
        return Ok(Vec::new());
    }
    let mut compressed = vec![0u8; header.normals_compressed_size as usize];
    reader.read_exact(&mut compressed)?;
    lz4_flex::decompress_size_prepended(&compressed)
        .map_err(|e| ArvxFileError::Decompress(e.to_string()))
}

/// Read and decompress the bricks section (if present). Flat u32 cells,
/// `BRICK_CELLS` per brick in brick-id order.
pub fn read_rkp_bricks<R: Read>(
    reader: &mut R,
    header: &ArvxHeader,
) -> Result<Vec<u8>, ArvxFileError> {
    if header.bricks_compressed_size == 0 {
        return Ok(Vec::new());
    }
    let mut compressed = vec![0u8; header.bricks_compressed_size as usize];
    reader.read_exact(&mut compressed)?;
    lz4_flex::decompress_size_prepended(&compressed)
        .map_err(|e| ArvxFileError::Decompress(e.to_string()))
}

/// Read and decompress the color data section (if present).
pub fn read_rkp_color<R: Read>(
    reader: &mut R,
    header: &ArvxHeader,
) -> Result<Vec<u8>, ArvxFileError> {
    if header.color_compressed_size == 0 {
        return Ok(Vec::new());
    }
    let mut compressed = vec![0u8; header.color_compressed_size as usize];
    reader.read_exact(&mut compressed)?;
    lz4_flex::decompress_size_prepended(&compressed)
        .map_err(|e| ArvxFileError::Decompress(e.to_string()))
}

/// Read and decompress the bone data section (if present).
/// Read and decode the optional skin-meta section. Returns an empty
/// `SkinMetaOut` (all three vectors empty) when the asset has no skin
/// data.
pub fn read_rkp_skin_meta<R: Read>(
    reader: &mut R,
    header: &ArvxHeader,
) -> Result<SkinMetaOut, ArvxFileError> {
    if header.bone_compressed_size == 0 {
        return Ok(SkinMetaOut::default());
    }
    let mut compressed = vec![0u8; header.bone_compressed_size as usize];
    reader.read_exact(&mut compressed)?;
    let raw = lz4_flex::decompress_size_prepended(&compressed)
        .map_err(|e| ArvxFileError::Decompress(e.to_string()))?;
    decode_skin_meta(&raw)
}

/// Read + decompress the mesh-vertices section (v5+). Empty vec if
/// `header.mesh_vertices_compressed_size == 0`. Bytes are
/// `bytemuck`-castable to `MeshVertex` (32 B each); caller does the
/// cast since `MeshVertex` lives in the consuming layer.
pub fn read_rkp_mesh_vertices<R: Read>(
    reader: &mut R,
    header: &ArvxHeader,
) -> Result<Vec<u8>, ArvxFileError> {
    if header.mesh_vertices_compressed_size == 0 {
        return Ok(Vec::new());
    }
    let mut compressed = vec![0u8; header.mesh_vertices_compressed_size as usize];
    reader.read_exact(&mut compressed)?;
    lz4_flex::decompress_size_prepended(&compressed)
        .map_err(|e| ArvxFileError::Decompress(e.to_string()))
}

/// Read + decompress the mesh-indices section (v5+). u32 indices
/// concatenated across all LOD levels (LOD-0 first).
pub fn read_rkp_mesh_indices<R: Read>(
    reader: &mut R,
    header: &ArvxHeader,
) -> Result<Vec<u8>, ArvxFileError> {
    if header.mesh_indices_compressed_size == 0 {
        return Ok(Vec::new());
    }
    let mut compressed = vec![0u8; header.mesh_indices_compressed_size as usize];
    reader.read_exact(&mut compressed)?;
    lz4_flex::decompress_size_prepended(&compressed)
        .map_err(|e| ArvxFileError::Decompress(e.to_string()))
}

/// Read + decompress the meshlet-clusters section (v5+). Bytes are
/// `bytemuck`-castable to `MeshletCluster` (64 B each).
pub fn read_rkp_meshlet_clusters<R: Read>(
    reader: &mut R,
    header: &ArvxHeader,
) -> Result<Vec<u8>, ArvxFileError> {
    if header.meshlet_clusters_compressed_size == 0 {
        return Ok(Vec::new());
    }
    let mut compressed = vec![0u8; header.meshlet_clusters_compressed_size as usize];
    reader.read_exact(&mut compressed)?;
    lz4_flex::decompress_size_prepended(&compressed)
        .map_err(|e| ArvxFileError::Decompress(e.to_string()))
}

/// Read + decompress the DAG-groups section (v6+). Bytes are
/// `bytemuck`-castable to `DagGroup` (16 B each). Empty vec when the
/// file is v5 or earlier.
pub fn read_rkp_dag_groups<R: Read>(
    reader: &mut R,
    header: &ArvxHeader,
) -> Result<Vec<u8>, ArvxFileError> {
    if header.dag_groups_compressed_size == 0 {
        return Ok(Vec::new());
    }
    let mut compressed = vec![0u8; header.dag_groups_compressed_size as usize];
    reader.read_exact(&mut compressed)?;
    lz4_flex::decompress_size_prepended(&compressed)
        .map_err(|e| ArvxFileError::Decompress(e.to_string()))
}

/// Read + decompress the DAG consumed-cluster-IDs section (v6+).
/// `bytemuck`-castable to `&[u32]`. Empty when the file is v5 or
/// earlier.
pub fn read_rkp_dag_consumed<R: Read>(
    reader: &mut R,
    header: &ArvxHeader,
) -> Result<Vec<u8>, ArvxFileError> {
    if header.dag_consumed_compressed_size == 0 {
        return Ok(Vec::new());
    }
    let mut compressed = vec![0u8; header.dag_consumed_compressed_size as usize];
    reader.read_exact(&mut compressed)?;
    lz4_flex::decompress_size_prepended(&compressed)
        .map_err(|e| ArvxFileError::Decompress(e.to_string()))
}

/// Read + decompress the DAG produced-cluster-IDs section (v6+).
/// `bytemuck`-castable to `&[u32]`. Empty when the file is v5 or
/// earlier.
pub fn read_rkp_dag_produced<R: Read>(
    reader: &mut R,
    header: &ArvxHeader,
) -> Result<Vec<u8>, ArvxFileError> {
    if header.dag_produced_compressed_size == 0 {
        return Ok(Vec::new());
    }
    let mut compressed = vec![0u8; header.dag_produced_compressed_size as usize];
    reader.read_exact(&mut compressed)?;
    lz4_flex::decompress_size_prepended(&compressed)
        .map_err(|e| ArvxFileError::Decompress(e.to_string()))
}

