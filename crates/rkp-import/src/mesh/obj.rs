//! Wavefront OBJ loader.
//!
//! Uses the `tobj` crate with single-index + triangulated loading.
//! Materials come from the companion `.mtl` file; diffuse textures are
//! sampled into RGBA8 via `image`. Roughness/metallic are approximated
//! from the legacy `Ns` shininess exponent (no real PBR in classic OBJ).

use std::path::Path;

use glam::Vec3;

use super::{ImportMaterial, MeshData, TextureData, finalize};

/// Load a Wavefront OBJ file into [`MeshData`].
pub fn load(path: &str) -> Result<MeshData, String> {
    let obj_dir = Path::new(path).parent().unwrap_or(Path::new("."));

    let (models, materials_result) = tobj::load_obj(
        path,
        &tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
    )
    .map_err(|e| format!("Failed to load OBJ '{path}': {e}"))?;

    let tobj_materials = materials_result
        .map_err(|e| format!("Failed to load materials for OBJ '{path}': {e}"))?;

    let materials: Vec<ImportMaterial> = tobj_materials
        .iter()
        .map(|m| convert_material(m, obj_dir))
        .collect();

    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();
    let mut material_indices = Vec::new();
    let mut bounds_min = Vec3::splat(f32::MAX);
    let mut bounds_max = Vec3::splat(f32::MIN);

    for model in &models {
        let mesh = &model.mesh;
        let vertex_offset = positions.len() as u32;

        for chunk in mesh.positions.chunks(3) {
            let v = Vec3::new(chunk[0], chunk[1], chunk[2]);
            bounds_min = bounds_min.min(v);
            bounds_max = bounds_max.max(v);
            positions.push(v);
        }

        if !mesh.normals.is_empty() {
            for chunk in mesh.normals.chunks(3) {
                normals.push(Vec3::new(chunk[0], chunk[1], chunk[2]));
            }
        }

        if !mesh.texcoords.is_empty() {
            for chunk in mesh.texcoords.chunks(2) {
                uvs.push([chunk[0], chunk[1]]);
            }
        }

        let mat_idx = mesh.material_id.unwrap_or(0) as u32;
        let tri_start = indices.len() / 3;
        for &idx in &mesh.indices {
            indices.push(vertex_offset + idx);
        }
        let tri_end = indices.len() / 3;
        for _ in tri_start..tri_end {
            material_indices.push(mat_idx);
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

fn convert_material(m: &tobj::Material, obj_dir: &Path) -> ImportMaterial {
    let base_color = m.diffuse.unwrap_or([0.8, 0.8, 0.8]);
    let shininess = m.shininess.unwrap_or(0.0).clamp(0.0, 1000.0);
    let glossiness = shininess / 1000.0;

    let albedo_texture = m.diffuse_texture.as_ref().and_then(|tex_name| {
        let tex_path = obj_dir.join(tex_name);
        match image::open(&tex_path) {
            Ok(img) => {
                let rgba = img.to_rgba8();
                Some(TextureData {
                    width: rgba.width(),
                    height: rgba.height(),
                    data: rgba.into_raw(),
                })
            }
            Err(e) => {
                eprintln!(
                    "[rkp-import] warn: failed to load OBJ texture '{}': {e}",
                    tex_path.display()
                );
                None
            }
        }
    });

    ImportMaterial {
        name: m.name.clone(),
        base_color,
        metallic: glossiness,
        roughness: 1.0 - glossiness,
        albedo_texture,
        uv_transform: [0.0, 0.0, 1.0, 1.0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_obj_texture_from_disk() {
        let dir = std::env::temp_dir().join("rkp_import_obj_tex");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let mut img = image::RgbaImage::new(2, 2);
        img.put_pixel(0, 0, image::Rgba([255, 0, 0, 255]));
        img.put_pixel(1, 0, image::Rgba([0, 255, 0, 255]));
        img.put_pixel(0, 1, image::Rgba([0, 0, 255, 255]));
        img.put_pixel(1, 1, image::Rgba([255, 255, 255, 255]));
        img.save(dir.join("diffuse.png")).unwrap();

        std::fs::write(
            dir.join("cube.mtl"),
            "newmtl textured\nKd 0.8 0.8 0.8\nmap_Kd diffuse.png\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("cube.obj"),
            "mtllib cube.mtl\nusemtl textured\n\
             v 0 0 0\nv 1 0 0\nv 0 1 0\n\
             vt 0 0\nvt 1 0\nvt 0 1\n\
             vn 0 0 1\n\
             f 1/1/1 2/2/1 3/3/1\n",
        )
        .unwrap();

        let mesh = load(dir.join("cube.obj").to_str().unwrap()).unwrap();
        let tex = mesh.materials[0].albedo_texture.as_ref().unwrap();
        assert_eq!(tex.width, 2);
        assert_eq!(tex.height, 2);
        assert_eq!(tex.data.len(), 16);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_obj_missing_texture_falls_back() {
        let dir = std::env::temp_dir().join("rkp_import_obj_notex");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        std::fs::write(
            dir.join("cube.mtl"),
            "newmtl missing\nKd 0.5 0.5 0.5\nmap_Kd nonexistent.png\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("cube.obj"),
            "mtllib cube.mtl\nusemtl missing\n\
             v 0 0 0\nv 1 0 0\nv 0 1 0\n\
             vn 0 0 1\n\
             f 1//1 2//1 3//1\n",
        )
        .unwrap();

        let mesh = load(dir.join("cube.obj").to_str().unwrap()).unwrap();
        assert!(mesh.materials[0].albedo_texture.is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
