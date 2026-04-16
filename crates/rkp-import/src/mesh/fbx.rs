//! Autodesk FBX loader (binary + ASCII, all versions via `ufbx`).
//!
//! Coordinate system is normalized to right-handed Y-up with units in
//! metres so downstream code can assume the same convention as glTF.
//! Geometry-space transforms are baked (`SpaceConversion::ModifyGeometry`)
//! so nodes carry identity locals.
//!
//! One vertex per triangle corner — we don't deduplicate shared
//! vertices because downstream BVH/voxelization doesn't need them.
//! This costs memory but keeps the mesh→bvh pipeline simple.

use std::path::Path;

use glam::Vec3;

use super::{ImportMaterial, MeshData, finalize};
use super::texture::load_fbx_texture;

/// Standard ufbx load options shared with the skeleton loader so
/// geometry and skin data line up.
pub(crate) fn fbx_load_opts() -> ufbx::LoadOpts<'static> {
    ufbx::LoadOpts {
        target_axes: ufbx::CoordinateAxes::right_handed_y_up(),
        target_unit_meters: 1.0,
        space_conversion: ufbx::SpaceConversion::ModifyGeometry,
        ..Default::default()
    }
}

/// Load an FBX file into [`MeshData`].
pub fn load(path: &str) -> Result<MeshData, String> {
    let fbx_dir = Path::new(path).parent().unwrap_or(Path::new("."));

    let scene = ufbx::load_file(path, fbx_load_opts())
        .map_err(|e| format!("Failed to load FBX '{path}': {}", e.description))?;

    let materials: Vec<ImportMaterial> = scene
        .materials
        .iter()
        .map(|mat| convert_material(mat, fbx_dir))
        .collect();

    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();
    let mut material_indices = Vec::new();
    let mut bounds_min = Vec3::splat(f32::MAX);
    let mut bounds_max = Vec3::splat(f32::MIN);

    let max_face_tris = scene
        .meshes
        .iter()
        .map(|m| m.max_face_triangles)
        .max()
        .unwrap_or(1);
    let mut tri_indices = vec![0u32; max_face_tris * 3];

    for mesh in &scene.meshes {
        let has_normals = mesh.vertex_normal.exists;
        let has_uvs = mesh.vertex_uv.exists;

        // Map mesh-local material index to scene-global index by
        // reference identity — ufbx `mesh.materials` entries are refs
        // into `scene.materials`, so pointer equality locates them.
        let mesh_mat_to_scene: Vec<u32> = mesh
            .materials
            .iter()
            .map(|m| {
                scene
                    .materials
                    .iter()
                    .position(|sm| std::ptr::eq(sm as *const _, m as *const _))
                    .unwrap_or(0) as u32
            })
            .collect();

        for (face_idx, face) in mesh.faces.iter().enumerate() {
            let num_tris = ufbx::triangulate_face(&mut tri_indices, mesh, *face);

            let mat_idx = if !mesh.face_material.is_empty() {
                let local = mesh.face_material[face_idx] as usize;
                mesh_mat_to_scene.get(local).copied().unwrap_or(0)
            } else {
                0
            };

            for t in 0..num_tris as usize {
                for c in 0..3 {
                    let idx = tri_indices[t * 3 + c] as usize;

                    let p = mesh.vertex_position[idx];
                    let pos = Vec3::new(p.x as f32, p.y as f32, p.z as f32);
                    bounds_min = bounds_min.min(pos);
                    bounds_max = bounds_max.max(pos);
                    positions.push(pos);

                    if has_normals {
                        let n = mesh.vertex_normal[idx];
                        normals.push(Vec3::new(n.x as f32, n.y as f32, n.z as f32));
                    }
                    if has_uvs {
                        let uv = mesh.vertex_uv[idx];
                        uvs.push([uv.x as f32, uv.y as f32]);
                    }

                    indices.push(positions.len() as u32 - 1);
                }
                material_indices.push(mat_idx);
            }
        }
    }

    let mut out = MeshData {
        positions,
        normals,
        uvs,
        indices,
        material_indices,
        materials,
        bounds_min,
        bounds_max,
    };
    finalize(&mut out);
    Ok(out)
}

fn convert_material(mat: &ufbx::Material, fbx_dir: &Path) -> ImportMaterial {
    let pbr = &mat.pbr;

    let base_color = if pbr.base_color.has_value {
        let c = pbr.base_color.value_vec4;
        [c.x as f32, c.y as f32, c.z as f32]
    } else if mat.fbx.diffuse_color.has_value {
        let c = mat.fbx.diffuse_color.value_vec4;
        [c.x as f32, c.y as f32, c.z as f32]
    } else {
        [0.8, 0.8, 0.8]
    };

    let metallic = if pbr.metalness.has_value {
        pbr.metalness.value_vec4.x as f32
    } else {
        0.0
    };
    let roughness = if pbr.roughness.has_value {
        pbr.roughness.value_vec4.x as f32
    } else {
        0.5
    };

    let tex_ref = pbr
        .base_color
        .texture
        .as_ref()
        .or(mat.fbx.diffuse_color.texture.as_ref());

    let (albedo_texture, uv_transform) = if let Some(tex) = tex_ref {
        let loaded = load_fbx_texture(tex, fbx_dir);
        let uv_xform = if tex.has_uv_transform {
            let t = &tex.uv_transform;
            [
                t.translation.x as f32,
                t.translation.y as f32,
                t.scale.x as f32,
                t.scale.y as f32,
            ]
        } else {
            [0.0, 0.0, 1.0, 1.0]
        };
        (loaded, uv_xform)
    } else {
        (None, [0.0, 0.0, 1.0, 1.0])
    };

    ImportMaterial {
        name: mat.element.name.to_string(),
        base_color,
        metallic,
        roughness,
        albedo_texture,
        uv_transform,
    }
}

#[cfg(test)]
mod tests {
    use super::super::load_mesh;

    #[test]
    fn load_fbx_not_found() {
        let err = load_mesh("/tmp/nonexistent_rkp_test.fbx").unwrap_err();
        assert!(err.contains("FBX"), "got: {err}");
    }

    #[test]
    fn load_mesh_dispatches_fbx_extension() {
        let err = load_mesh("/tmp/nonexistent_rkp_test.fbx").unwrap_err();
        assert!(!err.contains("Unsupported"));
    }
}
