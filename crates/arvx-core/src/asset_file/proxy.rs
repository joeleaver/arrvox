//! .arvxproxy — proxy-mesh sidecar cache.
//!
//! Generator children produce proxy meshes (surface-nets-from-SDF) instead
//! of voxel octrees. Their geometry has no octree, no leaf_attr pool, no
//! brick pool, and no cluster DAG — every `ProxyVertex` already carries
//! its full shading payload. The on-disk cache only needs the vertex and
//! index buffers plus the object-local AABB; the `MeshletCluster` is
//! derivable from the AABB and index count at load time (see
//! `SurfaceMesh::single_cluster`).
//!
//! Separate from `.arvx` because voxel sections (octree, voxels, normals,
//! bricks, color, bones, DAG topology) would all be empty placeholders.
//! Keeps the loader's "is this a voxel asset or a proxy?" decision a
//! single extension check at the file-path level.
//!
//! File layout:
//! ```text
//! [ArvxProxyHeader]   32 B, fixed
//! [vertices]          LZ4-prepended-size, ProxyVertex × N (32 B each)
//! [indices]           LZ4-prepended-size, u32 × M
//! ```

use crate::mesh_extract::ProxyVertex;
use bytemuck::{Pod, Zeroable};
use std::io::{Read, Write};

/// File magic: "AVXP" — distinct from .arvx's "AVX\x01" so a path-key
/// lookup error (e.g., trying to load a .arvxproxy as .arvx) fails loud.
pub const ARVXPROXY_MAGIC: [u8; 4] = [b'A', b'V', b'X', b'P'];

/// Current format version. v1 is the only version that has shipped.
pub const ARVXPROXY_VERSION: u32 = 1;

/// .arvxproxy file header (32 B).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ArvxProxyHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    pub vertex_count: u32,
    pub index_count: u32,
    pub vertices_compressed_size: u32,
    pub indices_compressed_size: u32,
}

const _: () = assert!(std::mem::size_of::<ArvxProxyHeader>() == 48);

/// In-memory form of a `.arvxproxy` payload.
#[derive(Debug, Clone)]
pub struct ProxyCache {
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    pub vertices: Vec<ProxyVertex>,
    pub indices: Vec<u32>,
}

/// Error type for .arvxproxy file operations.
#[derive(Debug, thiserror::Error)]
pub enum ArvxProxyError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid magic: expected AVXP")]
    BadMagic,
    #[error("Unsupported version: {0}")]
    UnsupportedVersion(u32),
    #[error("Decompression error: {0}")]
    Decompress(String),
    #[error("Vertex/index byte count mismatch")]
    LengthMismatch,
}

/// Serialize a `ProxyCache` to `.arvxproxy` on disk, atomically. Writes
/// first to `{path}.inprogress`, then renames into place so a mid-write
/// failure leaves any pre-existing cache untouched. Creates the parent
/// directory if missing.
pub fn write_arvxproxy(
    path: &std::path::Path,
    cache: &ProxyCache,
) -> Result<(), String> {
    let vertex_bytes: &[u8] = bytemuck::cast_slice(&cache.vertices);
    let index_bytes: &[u8] = bytemuck::cast_slice(&cache.indices);
    let vertices_compressed = lz4_flex::compress_prepend_size(vertex_bytes);
    let indices_compressed = lz4_flex::compress_prepend_size(index_bytes);

    let header = ArvxProxyHeader {
        magic: ARVXPROXY_MAGIC,
        version: ARVXPROXY_VERSION,
        aabb_min: cache.aabb_min,
        aabb_max: cache.aabb_max,
        vertex_count: cache.vertices.len() as u32,
        index_count: cache.indices.len() as u32,
        vertices_compressed_size: vertices_compressed.len() as u32,
        indices_compressed_size: indices_compressed.len() as u32,
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
        writer
            .write_all(bytemuck::bytes_of(&header))
            .map_err(|e| format!("write header: {e}"))?;
        writer
            .write_all(&vertices_compressed)
            .map_err(|e| format!("write vertices: {e}"))?;
        writer
            .write_all(&indices_compressed)
            .map_err(|e| format!("write indices: {e}"))?;
    }

    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("rename {} -> {}: {e}", tmp.display(), path.display())
    })?;

    Ok(())
}

/// Read a `.arvxproxy` from disk and return its `ProxyCache`.
pub fn read_arvxproxy(path: &std::path::Path) -> Result<ProxyCache, ArvxProxyError> {
    let file = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::new(file);

    let mut header_bytes = [0u8; std::mem::size_of::<ArvxProxyHeader>()];
    reader.read_exact(&mut header_bytes)?;
    let header: ArvxProxyHeader = *bytemuck::from_bytes(&header_bytes);
    if header.magic != ARVXPROXY_MAGIC {
        return Err(ArvxProxyError::BadMagic);
    }
    if header.version != ARVXPROXY_VERSION {
        return Err(ArvxProxyError::UnsupportedVersion(header.version));
    }

    let mut vertices_compressed = vec![0u8; header.vertices_compressed_size as usize];
    reader.read_exact(&mut vertices_compressed)?;
    let mut indices_compressed = vec![0u8; header.indices_compressed_size as usize];
    reader.read_exact(&mut indices_compressed)?;

    let vertex_bytes = lz4_flex::decompress_size_prepended(&vertices_compressed)
        .map_err(|e| ArvxProxyError::Decompress(e.to_string()))?;
    let index_bytes = lz4_flex::decompress_size_prepended(&indices_compressed)
        .map_err(|e| ArvxProxyError::Decompress(e.to_string()))?;

    let expected_vertex_bytes = (header.vertex_count as usize)
        .checked_mul(std::mem::size_of::<ProxyVertex>())
        .ok_or(ArvxProxyError::LengthMismatch)?;
    let expected_index_bytes = (header.index_count as usize)
        .checked_mul(4)
        .ok_or(ArvxProxyError::LengthMismatch)?;
    if vertex_bytes.len() != expected_vertex_bytes
        || index_bytes.len() != expected_index_bytes
    {
        return Err(ArvxProxyError::LengthMismatch);
    }

    // `bytemuck::cast_slice(&[])` panics on alignment because an empty
    // Vec<u8>'s data pointer is `NonNull::dangling::<u8>()` (align 1),
    // not aligned to ProxyVertex's 4-byte requirement. Side-step both
    // empty casts.
    let vertices: Vec<ProxyVertex> = if vertex_bytes.is_empty() {
        Vec::new()
    } else {
        bytemuck::cast_slice(&vertex_bytes).to_vec()
    };
    let indices: Vec<u32> = if index_bytes.is_empty() {
        Vec::new()
    } else {
        bytemuck::cast_slice(&index_bytes).to_vec()
    };

    Ok(ProxyCache {
        aabb_min: header.aabb_min,
        aabb_max: header.aabb_max,
        vertices,
        indices,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cache() -> ProxyCache {
        ProxyCache {
            aabb_min: [-1.0, -2.0, -3.0],
            aabb_max: [4.0, 5.0, 6.0],
            vertices: vec![
                ProxyVertex {
                    local_pos: [0.0, 0.0, 0.0],
                    normal_oct: 0x12345678,
                    material_packed: 0xCAFEBABE,
                    color_packed: 0xDEADBEEF,
                    _reserved: [0, 0],
                },
                ProxyVertex {
                    local_pos: [1.0, 2.0, 3.0],
                    normal_oct: 0xAAAA5555,
                    material_packed: 0x00112233,
                    color_packed: 0x44556677,
                    _reserved: [0, 0],
                },
                ProxyVertex {
                    local_pos: [-0.5, 0.25, 8.75],
                    normal_oct: 0x80808080,
                    material_packed: 0,
                    color_packed: 0xFFFFFFFF,
                    _reserved: [0, 0],
                },
            ],
            indices: vec![0, 1, 2, 0, 2, 1],
        }
    }

    #[test]
    fn roundtrip_populated() {
        let cache = make_cache();
        let dir = std::env::temp_dir().join(format!(
            "arvxproxy_roundtrip_{}.dir",
            std::process::id(),
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.arvxproxy");

        write_arvxproxy(&path, &cache).expect("write");
        let loaded = read_arvxproxy(&path).expect("read");

        assert_eq!(loaded.aabb_min, cache.aabb_min);
        assert_eq!(loaded.aabb_max, cache.aabb_max);
        assert_eq!(loaded.vertices, cache.vertices);
        assert_eq!(loaded.indices, cache.indices);

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn roundtrip_empty() {
        let cache = ProxyCache {
            aabb_min: [0.0, 0.0, 0.0],
            aabb_max: [0.0, 0.0, 0.0],
            vertices: Vec::new(),
            indices: Vec::new(),
        };
        let dir = std::env::temp_dir().join(format!(
            "arvxproxy_empty_{}.dir",
            std::process::id(),
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("empty.arvxproxy");

        write_arvxproxy(&path, &cache).expect("write");
        let loaded = read_arvxproxy(&path).expect("read");

        assert_eq!(loaded.vertices.len(), 0);
        assert_eq!(loaded.indices.len(), 0);

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn bad_magic_rejected() {
        let dir = std::env::temp_dir().join(format!(
            "arvxproxy_bad_magic_{}.dir",
            std::process::id(),
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bogus.arvxproxy");
        // Need at least a full header's worth of bytes — otherwise the
        // read_exact bails with Io(UnexpectedEof) before the magic
        // check fires. Fill with junk that starts with non-AVXP bytes.
        let mut junk = vec![0u8; std::mem::size_of::<ArvxProxyHeader>() + 16];
        junk[..4].copy_from_slice(b"NOPE");
        std::fs::write(&path, &junk).unwrap();

        let err = read_arvxproxy(&path).expect_err("should fail");
        assert!(matches!(err, ArvxProxyError::BadMagic), "got {err:?}");

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }
}
