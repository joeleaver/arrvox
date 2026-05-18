//! Mesh loading and triangle data for the import pipeline.
//!
//! Loads polygon meshes from glTF (`.gltf`, `.glb`), OBJ (`.obj`), and
//! FBX (`.fbx`) files into a unified [`MeshData`] representation ready
//! for BVH construction and voxelization.
//!
//! Per-format loaders live in the sibling modules ([`gltf`], [`obj`],
//! [`fbx`]); this module owns the shared data types and the dispatch
//! entry point [`load_mesh`].

use glam::Vec3;

pub mod fbx;
pub mod gltf;
pub mod obj;
pub mod texture;

/// A triangle referenced by three vertex indices into
/// [`MeshData::positions`].
#[derive(Debug, Clone, Copy)]
pub struct Triangle {
    /// First vertex index.
    pub i0: u32,
    /// Second vertex index.
    pub i1: u32,
    /// Third vertex index.
    pub i2: u32,
}

/// Material properties extracted from a source mesh. PBR-metallic-roughness
/// in linear RGB with an optional baked albedo texture.
#[derive(Debug, Clone)]
pub struct ImportMaterial {
    /// Material name from the source file.
    pub name: String,
    /// Base colour in linear RGB.
    pub base_color: [f32; 3],
    /// Metallic factor in `[0, 1]`.
    pub metallic: f32,
    /// Roughness factor in `[0, 1]`.
    pub roughness: f32,
    /// Albedo texture data (RGBA8, if present).
    pub albedo_texture: Option<TextureData>,
    /// UV transform `[offset_u, offset_v, scale_u, scale_v]` applied
    /// as `final_uv = uv * scale + offset` (KHR_texture_transform convention).
    pub uv_transform: [f32; 4],
}

impl Default for ImportMaterial {
    fn default() -> Self {
        Self {
            name: String::new(),
            base_color: [0.8, 0.8, 0.8],
            metallic: 0.0,
            roughness: 0.5,
            albedo_texture: None,
            uv_transform: [0.0, 0.0, 1.0, 1.0],
        }
    }
}

/// RGBA8 texture loaded from a mesh file (either embedded or via a
/// sidecar image on disk).
#[derive(Debug, Clone)]
pub struct TextureData {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// RGBA8 pixel data (row-major, 4 bytes per pixel).
    pub data: Vec<u8>,
}

/// Loaded mesh data ready for BVH construction and voxelization.
#[derive(Debug, Clone)]
pub struct MeshData {
    /// Vertex positions.
    pub positions: Vec<Vec3>,
    /// Vertex normals (same length as positions; padded with `Y` if the
    /// source didn't supply them).
    pub normals: Vec<Vec3>,
    /// Vertex UVs (same length as positions, or empty if the source had no UVs).
    pub uvs: Vec<[f32; 2]>,
    /// Flat triangle index buffer — length = `triangle_count * 3`.
    pub indices: Vec<u32>,
    /// Per-triangle material index — length = `triangle_count`.
    pub material_indices: Vec<u32>,
    /// Materials parsed from the source file (always at least one — a
    /// grey default is injected when the file has none).
    pub materials: Vec<ImportMaterial>,
    /// Mesh bounding-box minimum (post any loader-internal transforms).
    pub bounds_min: Vec3,
    /// Mesh bounding-box maximum.
    pub bounds_max: Vec3,
}

impl MeshData {
    /// Number of triangles in the mesh.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Three vertex positions of triangle `tri_idx`.
    pub fn triangle_positions(&self, tri_idx: usize) -> [Vec3; 3] {
        let base = tri_idx * 3;
        [
            self.positions[self.indices[base] as usize],
            self.positions[self.indices[base + 1] as usize],
            self.positions[self.indices[base + 2] as usize],
        ]
    }

    /// Three vertex normals of triangle `tri_idx`. Assumes
    /// `normals.len() == positions.len()`, which [`finalize`]
    /// guarantees (computing from face normals when the source file
    /// didn't supply any).
    pub fn triangle_vertex_normals(&self, tri_idx: usize) -> [glam::Vec3; 3] {
        let base = tri_idx * 3;
        [
            self.normals[self.indices[base] as usize],
            self.normals[self.indices[base + 1] as usize],
            self.normals[self.indices[base + 2] as usize],
        ]
    }

    /// Three vertex UVs of triangle `tri_idx`. Returns zeros for meshes
    /// without UV data.
    pub fn triangle_uvs(&self, tri_idx: usize) -> [[f32; 2]; 3] {
        if self.uvs.is_empty() {
            return [[0.0, 0.0]; 3];
        }
        let base = tri_idx * 3;
        [
            self.uvs[self.indices[base] as usize],
            self.uvs[self.indices[base + 1] as usize],
            self.uvs[self.indices[base + 2] as usize],
        ]
    }

    /// Average edge length across all triangle edges. Useful for
    /// picking voxel sizes that resolve the mesh at triangle granularity.
    pub fn average_edge_length(&self) -> f32 {
        let tc = self.triangle_count();
        if tc == 0 {
            return 0.0;
        }
        let mut total = 0.0f32;
        for i in 0..tc {
            let [v0, v1, v2] = self.triangle_positions(i);
            total += (v1 - v0).length();
            total += (v2 - v1).length();
            total += (v0 - v2).length();
        }
        total / (tc as f32 * 3.0)
    }
}

/// Dispatch to the right per-format loader by file extension.
/// Supports `.gltf`, `.glb`, `.obj`, `.fbx`.
pub fn load_mesh(path: &str) -> Result<MeshData, String> {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "gltf" | "glb" => gltf::load(path),
        "obj" => obj::load(path),
        "fbx" => fbx::load(path),
        other => Err(format!(
            "Unsupported mesh format: .{other}. Supported: .gltf, .glb, .obj, .fbx"
        )),
    }
}

/// Finalize a freshly-loaded mesh: compute per-vertex normals if the
/// source didn't provide them (so downstream Gouraud-style
/// barycentric normal sampling always has real data to interpolate),
/// zero out bounds on empty meshes, and inject a default material
/// if the source had none. Called by every loader as its last step
/// so the returned [`MeshData`] obeys the invariants.
///
/// Historical note: this used to pad missing normals with `Vec3::Y`
/// as a placeholder. That worked when the voxelizer only used
/// triangle-face normals (computed from vertex positions), but once
/// we switched to barycentric-interpolated vertex normals for smooth
/// shading, the `Y` padding produced uniformly up-facing surfaces
/// for any OBJ without `vn` lines (e.g., stanford-bunny). Computing
/// real per-vertex normals when the source omits them fixes this.
pub(crate) fn finalize(mesh: &mut MeshData) {
    if mesh.normals.len() != mesh.positions.len() {
        compute_vertex_normals(mesh);
    }
    if mesh.positions.is_empty() {
        mesh.bounds_min = Vec3::ZERO;
        mesh.bounds_max = Vec3::ZERO;
    }
    if mesh.materials.is_empty() {
        mesh.materials.push(ImportMaterial {
            name: "default".to_string(),
            ..ImportMaterial::default()
        });
    }
}

/// Fill `mesh.normals` with area-weighted averages of incident
/// triangles' face normals — produces Gouraud-style smooth shading
/// across shared edges. Overwrites any existing data; [`finalize`]
/// only calls this when the source didn't supply per-vertex normals.
pub(crate) fn compute_vertex_normals(mesh: &mut MeshData) {
    let n = mesh.positions.len();
    let mut normals = vec![Vec3::ZERO; n];
    let tri_count = mesh.indices.len() / 3;
    for tri_idx in 0..tri_count {
        let base = tri_idx * 3;
        let i0 = mesh.indices[base] as usize;
        let i1 = mesh.indices[base + 1] as usize;
        let i2 = mesh.indices[base + 2] as usize;
        let a = mesh.positions[i0];
        let b = mesh.positions[i1];
        let c = mesh.positions[i2];
        // Unnormalized cross product — magnitude equals 2× triangle
        // area, which gives the per-vertex accumulation natural
        // area-weighting without an extra division.
        let face = (b - a).cross(c - a);
        normals[i0] += face;
        normals[i1] += face;
        normals[i2] += face;
    }
    for normal in normals.iter_mut() {
        let len_sq = normal.length_squared();
        if len_sq > 1e-12 {
            *normal /= len_sq.sqrt();
        } else {
            *normal = Vec3::Y;
        }
    }
    mesh.normals = normals;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_mesh() -> MeshData {
        MeshData {
            positions: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            normals: vec![Vec3::Z; 3],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            indices: vec![0, 1, 2],
            material_indices: vec![0],
            materials: vec![ImportMaterial {
                name: "test".to_string(),
                base_color: [1.0, 0.0, 0.0],
                ..ImportMaterial::default()
            }],
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::new(1.0, 1.0, 0.0),
        }
    }

    fn make_empty_mesh() -> MeshData {
        MeshData {
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            indices: Vec::new(),
            material_indices: Vec::new(),
            materials: Vec::new(),
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::ZERO,
        }
    }

    #[test]
    fn triangle_count_empty() {
        assert_eq!(make_empty_mesh().triangle_count(), 0);
    }

    #[test]
    fn triangle_count_single() {
        assert_eq!(make_test_mesh().triangle_count(), 1);
    }

    #[test]
    fn average_edge_length_empty() {
        assert_eq!(make_empty_mesh().average_edge_length(), 0.0);
    }

    #[test]
    fn average_edge_length_unit_triangle() {
        let mesh = make_test_mesh();
        let expected = (1.0 + std::f32::consts::SQRT_2 + 1.0) / 3.0;
        assert!((mesh.average_edge_length() - expected).abs() < 1e-5);
    }

    #[test]
    fn triangle_positions_correct() {
        let mesh = make_test_mesh();
        let [v0, v1, v2] = mesh.triangle_positions(0);
        assert_eq!(v0, Vec3::ZERO);
        assert_eq!(v1, Vec3::X);
        assert_eq!(v2, Vec3::Y);
    }

    #[test]
    fn triangle_uvs_no_uvs_returns_zeros() {
        let mut mesh = make_test_mesh();
        mesh.uvs.clear();
        assert_eq!(mesh.triangle_uvs(0), [[0.0, 0.0]; 3]);
    }

    #[test]
    fn finalize_pads_normals_and_injects_material() {
        let mut mesh = make_empty_mesh();
        mesh.positions = vec![Vec3::ZERO, Vec3::X];
        finalize(&mut mesh);
        assert_eq!(mesh.normals.len(), 2);
        assert!(mesh.normals.iter().all(|n| *n == Vec3::Y));
        assert_eq!(mesh.materials.len(), 1);
    }

    #[test]
    fn load_mesh_unsupported_format() {
        let err = load_mesh("model.stl").unwrap_err();
        assert!(err.contains("Unsupported"));
    }

    #[test]
    fn load_mesh_no_extension() {
        assert!(load_mesh("noext").is_err());
    }
}
