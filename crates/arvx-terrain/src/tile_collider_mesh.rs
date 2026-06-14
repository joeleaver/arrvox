//! ECS component carrying the per-tile triangle data the physics
//! integration uses to build a Rapier `TriMesh` collider.
//!
//! Phase 8 of `docs/TERRAIN.md`. Captured at tile-integration time
//! (before `MeshSectionsBlob` moves into the scene manager) and
//! parked on the tile entity. `arvx-engine`'s `play_mode` reads it
//! when a Rapier world exists; tile entities outside play mode
//! still carry it so entering play picks them up.
//!
//! ## Why a separate component, not a borrowed slice
//!
//! `play_mode` lives in `arvx-engine` and Rapier's `TriMesh`
//! construction wants `Vec<Point<f32>>` + `Vec<[u32; 3]>` it can
//! own. Re-reading the mesh from the GPU-bound asset blob would
//! mean duplicating the LOD-0 deserialisation. Snapshotting tile-
//! local positions + triangle indices into a dedicated component
//! costs about (vertex_count * 12 + tri_count * 12) bytes per tile —
//! tens of KB at most for a 64 m tile at 0.25 m voxels — and
//! decouples the physics pipeline from the renderer's buffer
//! ownership.

use arvx_core::asset_file::MeshSectionsBlob;
use glam::Vec3;

/// LOD-0 surface mesh for one terrain tile, in tile-local coords.
///
/// `vertices` are positions only — Rapier `TriMesh` doesn't consume
/// normals / materials / leaf_attr_ids. `triangles` is the LOD-0
/// portion of the mesh's index buffer reshaped to triangle triples.
///
/// Empty when the tile has no surface (all-sky / all-solid baked
/// tile). The engine treats empty meshes as "no collider needed"
/// rather than building a degenerate 0-tri TriMesh.
#[derive(Debug, Clone)]
pub struct TileColliderMesh {
    /// Vertex positions in tile-local metres (origin = the tile's
    /// `TileKey::origin_world()`). The renderer applies the same
    /// tile origin when placing geometry, so a collider positioned
    /// at the tile's world origin lines up with the visible surface.
    pub vertices: Vec<Vec3>,
    /// LOD-0 triangles as `[a, b, c]` indices into `vertices`.
    pub triangles: Vec<[u32; 3]>,
}

impl TileColliderMesh {
    /// True when the tile has no surface — caller should skip
    /// collider construction.
    pub fn is_empty(&self) -> bool {
        self.triangles.is_empty()
    }

    /// Snapshot the LOD-0 surface from a baked [`MeshSectionsBlob`], using
    /// the full `lod0_index_count`. Prefer [`Self::from_mesh_blob_prefix`]
    /// with the surface-only count for terrain — the lateral skirts are
    /// folded into `lod0_index_count` but are back-culled (never drawn), so
    /// including them gives bodies invisible walls to snag on.
    pub fn from_mesh_blob(mesh: &MeshSectionsBlob) -> Self {
        Self::from_mesh_blob_prefix(mesh, mesh.lod0_index_count)
    }

    /// Snapshot the first `index_count` LOD-0 indices into a collider mesh.
    ///
    /// Vertex stride is 32 B (matches arvx's `MeshVertex` layout —
    /// `CLAUDE.md` "Key Data Types"); the first 12 B are the f32 position.
    /// Index stride is 4 B (u32). `index_count` caps the iteration so neither
    /// higher-LOD chains nor the back-culled lateral skirts (a suffix of the
    /// LOD-0 range) bleed into the collider. Pass `BakedTile::surface_index_count`.
    pub fn from_mesh_blob_prefix(mesh: &MeshSectionsBlob, index_count: u32) -> Self {
        const VERTEX_STRIDE: usize = 32;
        const INDEX_STRIDE: usize = 4;

        let lod0_n = (index_count as usize).min(mesh.indices.len() / INDEX_STRIDE);
        if lod0_n < 3 || lod0_n % 3 != 0 {
            return Self {
                vertices: Vec::new(),
                triangles: Vec::new(),
            };
        }

        // Decode LOD-0 indices into u32 values + collect the unique
        // vertex set they touch. We compact the global vertex buffer
        // to just the LOD-0 set so the Rapier TriMesh isn't carrying
        // higher-LOD vertices the index buffer never references.
        let mut remap: std::collections::HashMap<u32, u32> =
            std::collections::HashMap::with_capacity(lod0_n);
        let mut vertices: Vec<Vec3> = Vec::new();
        let mut triangles: Vec<[u32; 3]> = Vec::with_capacity(lod0_n / 3);

        let read_pos = |vid: u32| -> Vec3 {
            let base = vid as usize * VERTEX_STRIDE;
            let bytes = &mesh.vertices[base..base + 12];
            Vec3::new(
                f32::from_le_bytes(bytes[0..4].try_into().unwrap()),
                f32::from_le_bytes(bytes[4..8].try_into().unwrap()),
                f32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            )
        };

        let mut intern = |raw_vid: u32, vertices: &mut Vec<Vec3>| -> u32 {
            *remap.entry(raw_vid).or_insert_with(|| {
                let new = vertices.len() as u32;
                vertices.push(read_pos(raw_vid));
                new
            })
        };

        for tri_start in (0..lod0_n).step_by(3) {
            let read_idx = |i: usize| -> u32 {
                let base = i * INDEX_STRIDE;
                u32::from_le_bytes(
                    mesh.indices[base..base + INDEX_STRIDE].try_into().unwrap(),
                )
            };
            let a_raw = read_idx(tri_start);
            let b_raw = read_idx(tri_start + 1);
            let c_raw = read_idx(tri_start + 2);
            let a = intern(a_raw, &mut vertices);
            let b = intern(b_raw, &mut vertices);
            let c = intern(c_raw, &mut vertices);
            triangles.push([a, b, c]);
        }

        Self { vertices, triangles }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic two-triangle blob and verify extraction.
    /// Vertex layout = 32 B stride (12 B pos + 20 B junk).
    fn make_blob(verts: &[Vec3], tris: &[[u32; 3]]) -> MeshSectionsBlob {
        let mut vertices = Vec::with_capacity(verts.len() * 32);
        for v in verts {
            vertices.extend_from_slice(&v.x.to_le_bytes());
            vertices.extend_from_slice(&v.y.to_le_bytes());
            vertices.extend_from_slice(&v.z.to_le_bytes());
            // 20 B trailing junk.
            vertices.extend_from_slice(&[0u8; 20]);
        }
        let mut indices = Vec::with_capacity(tris.len() * 12);
        for t in tris {
            for &i in t {
                indices.extend_from_slice(&i.to_le_bytes());
            }
        }
        MeshSectionsBlob {
            vertices,
            indices,
            clusters: Vec::new(),
            lod0_index_count: (tris.len() * 3) as u32,
            dag_groups: Vec::new(),
            dag_consumed: Vec::new(),
            dag_produced: Vec::new(),
        }
    }

    #[test]
    fn empty_mesh_is_empty_collider() {
        let blob = make_blob(&[], &[]);
        let c = TileColliderMesh::from_mesh_blob(&blob);
        assert!(c.is_empty());
        assert!(c.vertices.is_empty());
    }

    #[test]
    fn extracts_two_triangles() {
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        ];
        let tris = vec![[0u32, 1, 2], [1, 3, 2]];
        let blob = make_blob(&verts, &tris);
        let c = TileColliderMesh::from_mesh_blob(&blob);
        assert_eq!(c.triangles.len(), 2);
        // Vertices were compacted (all four referenced — all kept).
        assert_eq!(c.vertices.len(), 4);
        // Each triangle's vertices match the input positions.
        for (out_tri, in_tri) in c.triangles.iter().zip(tris.iter()) {
            for (out_idx, in_idx) in out_tri.iter().zip(in_tri.iter()) {
                let out_v = c.vertices[*out_idx as usize];
                let in_v = verts[*in_idx as usize];
                assert!((out_v - in_v).length() < 1e-6, "{out_v:?} != {in_v:?}");
            }
        }
    }

    #[test]
    fn unreferenced_vertices_are_dropped() {
        // 5 vertices in the blob, only 3 referenced by triangles.
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(99.0, 99.0, 99.0), // unreferenced
            Vec3::new(100.0, 100.0, 100.0), // unreferenced
        ];
        let tris = vec![[0u32, 1, 2]];
        let blob = make_blob(&verts, &tris);
        let c = TileColliderMesh::from_mesh_blob(&blob);
        assert_eq!(c.triangles.len(), 1);
        assert_eq!(c.vertices.len(), 3, "should compact to 3 referenced verts");
        // None of the dropped vertices made it through.
        for v in &c.vertices {
            assert!(v.length() < 10.0, "unreferenced vertex leaked: {v:?}");
        }
    }

    #[test]
    fn lod0_count_caps_iteration() {
        // The blob carries 2 triangles worth of indices BUT
        // lod0_index_count = 3 (i.e. only the first triangle is
        // LOD-0). The second triangle's vertices shouldn't appear.
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(2.0, 2.0, 2.0),
        ];
        let tris = vec![[0u32, 1, 2], [1, 3, 2]];
        let mut blob = make_blob(&verts, &tris);
        blob.lod0_index_count = 3; // first tri only
        let c = TileColliderMesh::from_mesh_blob(&blob);
        assert_eq!(c.triangles.len(), 1);
        assert_eq!(c.vertices.len(), 3);
    }

    #[test]
    fn odd_lod0_index_count_is_rejected() {
        let verts = vec![Vec3::ZERO, Vec3::X, Vec3::Y];
        let tris = vec![[0u32, 1, 2]];
        let mut blob = make_blob(&verts, &tris);
        blob.lod0_index_count = 2; // not a multiple of 3
        let c = TileColliderMesh::from_mesh_blob(&blob);
        assert!(c.is_empty(), "non-tri-multiple should produce empty");
    }
}
