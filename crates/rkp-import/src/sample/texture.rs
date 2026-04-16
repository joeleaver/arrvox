//! Texture-colour sampling at a point on a triangle.
//!
//! Given a triangle index and barycentric coordinates, interpolates
//! the UV and samples the material's albedo texture (nearest-neighbor
//! with `rem_euclid` wrap for tiling). Falls back to the material's
//! base colour if there's no texture or no UVs.

use crate::mesh::{MeshData, TextureData};

/// Per-voxel albedo colour (RGBA8).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VoxelColor {
    /// Red channel `[0, 255]`.
    pub r: u8,
    /// Green channel `[0, 255]`.
    pub g: u8,
    /// Blue channel `[0, 255]`.
    pub b: u8,
    /// Alpha channel `[0, 255]`.
    pub a: u8,
}

/// Sample texture colour at a point on a triangle using barycentric
/// interpolation. Returns `None` only if the triangle's material
/// index is out of range; otherwise always returns either a texture
/// sample or the material's base colour.
pub fn sample_texture_at_triangle(
    mesh: &MeshData,
    tri_idx: usize,
    barycentric: &[f32; 3],
) -> Option<VoxelColor> {
    let uvs = mesh.triangle_uvs(tri_idx);
    let mut u =
        uvs[0][0] * barycentric[0] + uvs[1][0] * barycentric[1] + uvs[2][0] * barycentric[2];
    let mut v =
        uvs[0][1] * barycentric[0] + uvs[1][1] * barycentric[1] + uvs[2][1] * barycentric[2];

    let mat_idx = mesh
        .material_indices
        .get(tri_idx)
        .copied()
        .unwrap_or(0) as usize;
    let material = mesh.materials.get(mat_idx)?;

    let xf = material.uv_transform;
    u = u * xf[2] + xf[0];
    v = v * xf[3] + xf[1];

    if let Some(ref tex) = material.albedo_texture {
        if !mesh.uvs.is_empty() {
            return Some(sample_texture(tex, u, v));
        }
    }

    Some(VoxelColor {
        r: (material.base_color[0] * 255.0).clamp(0.0, 255.0) as u8,
        g: (material.base_color[1] * 255.0).clamp(0.0, 255.0) as u8,
        b: (material.base_color[2] * 255.0).clamp(0.0, 255.0) as u8,
        a: 255,
    })
}

/// Sample a texture at UV coordinates with nearest-neighbor filtering.
/// UVs are wrapped to `[0, 1)` via `rem_euclid` for seamless tiling.
pub fn sample_texture(tex: &TextureData, u: f32, v: f32) -> VoxelColor {
    if tex.width == 0 || tex.height == 0 || tex.data.is_empty() {
        return VoxelColor { r: 128, g: 128, b: 128, a: 255 };
    }

    let u = u.rem_euclid(1.0);
    let v = v.rem_euclid(1.0);

    let ix = ((u * tex.width as f32).floor() as u32).min(tex.width - 1);
    let iy = ((v * tex.height as f32).floor() as u32).min(tex.height - 1);
    let idx = ((iy * tex.width + ix) * 4) as usize;

    if idx + 3 < tex.data.len() {
        VoxelColor {
            r: tex.data[idx],
            g: tex.data[idx + 1],
            b: tex.data[idx + 2],
            a: tex.data[idx + 3],
        }
    } else {
        VoxelColor { r: 128, g: 128, b: 128, a: 255 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{ImportMaterial, MeshData, TextureData};
    use glam::Vec3;

    fn make_test_texture() -> TextureData {
        TextureData {
            width: 2,
            height: 2,
            data: vec![
                255, 0, 0, 255, // (0,0) red
                0, 255, 0, 255, // (1,0) green
                0, 0, 255, 255, // (0,1) blue
                255, 255, 255, 255, // (1,1) white
            ],
        }
    }

    fn make_textured_mesh() -> MeshData {
        MeshData {
            positions: vec![Vec3::ZERO, Vec3::X, Vec3::Y],
            normals: vec![Vec3::Z; 3],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            indices: vec![0, 1, 2],
            material_indices: vec![0],
            materials: vec![ImportMaterial {
                albedo_texture: Some(make_test_texture()),
                ..ImportMaterial::default()
            }],
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::new(1.0, 1.0, 0.0),
        }
    }

    #[test]
    fn sample_texture_at_origin_returns_red() {
        let c = sample_texture(&make_test_texture(), 0.0, 0.0);
        assert_eq!((c.r, c.g, c.b), (255, 0, 0));
    }

    #[test]
    fn sample_texture_wraps_uv_greater_than_one() {
        let c = sample_texture(&make_test_texture(), 1.25, 0.0);
        assert_eq!((c.r, c.g, c.b), (255, 0, 0));
    }

    #[test]
    fn sample_texture_wraps_negative_uv() {
        let c = sample_texture(&make_test_texture(), -0.25, 0.0);
        assert_eq!((c.r, c.g, c.b), (0, 255, 0));
    }

    #[test]
    fn sample_texture_at_triangle_no_uvs_falls_back_to_base_color() {
        let mut mesh = make_textured_mesh();
        mesh.uvs.clear();
        mesh.materials[0].base_color = [1.0, 0.0, 0.0];
        let c = sample_texture_at_triangle(&mesh, 0, &[1.0, 0.0, 0.0]).unwrap();
        assert_eq!((c.r, c.g, c.b), (255, 0, 0));
    }

    #[test]
    fn sample_texture_at_triangle_vertex_samples_texture() {
        let c = sample_texture_at_triangle(&make_textured_mesh(), 0, &[1.0, 0.0, 0.0]).unwrap();
        assert_eq!((c.r, c.g, c.b), (255, 0, 0));
    }

    #[test]
    fn sample_texture_at_triangle_invalid_material_returns_none() {
        let mesh = MeshData {
            positions: vec![Vec3::ZERO, Vec3::X, Vec3::Y],
            normals: vec![Vec3::Z; 3],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            indices: vec![0, 1, 2],
            material_indices: vec![5],
            materials: Vec::new(),
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::new(1.0, 1.0, 0.0),
        };
        assert!(sample_texture_at_triangle(&mesh, 0, &[1.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn sample_texture_empty_returns_gray() {
        let tex = TextureData { width: 0, height: 0, data: Vec::new() };
        let c = sample_texture(&tex, 0.5, 0.5);
        assert_eq!((c.r, c.g, c.b, c.a), (128, 128, 128, 255));
    }
}
