//! glTF / GLB loader.
//!
//! Uses the `gltf` crate to parse the document + buffer + image blob,
//! flattens all scene meshes into one triangle soup, and extracts PBR
//! metallic-roughness materials (base color, metallic, roughness,
//! albedo texture, KHR_texture_transform).

use glam::Vec3;

use super::{ImportMaterial, MeshData, TextureData, finalize};

/// Load a glTF or GLB file into [`MeshData`].
pub fn load(path: &str) -> Result<MeshData, String> {
    let (document, buffers, images) =
        gltf::import(path).map_err(|e| format!("Failed to load glTF '{path}': {e}"))?;

    let materials: Vec<ImportMaterial> = document
        .materials()
        .map(|mat| extract_material(&mat, &images))
        .collect();

    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();
    let mut material_indices = Vec::new();
    let mut bounds_min = Vec3::splat(f32::MAX);
    let mut bounds_max = Vec3::splat(f32::MIN);

    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buf| Some(&buffers[buf.index()]));
            let mat_idx = primitive.material().index().unwrap_or(0) as u32;
            let vertex_offset = positions.len() as u32;

            let Some(prim_positions) = reader.read_positions() else {
                continue;
            };
            for p in prim_positions {
                let v = Vec3::new(p[0], p[1], p[2]);
                bounds_min = bounds_min.min(v);
                bounds_max = bounds_max.max(v);
                positions.push(v);
            }

            if let Some(prim_normals) = reader.read_normals() {
                for n in prim_normals {
                    normals.push(Vec3::new(n[0], n[1], n[2]));
                }
            }

            if let Some(prim_uvs) = reader.read_tex_coords(0) {
                for uv in prim_uvs.into_f32() {
                    uvs.push(uv);
                }
            }

            if let Some(prim_indices) = reader.read_indices() {
                let tri_start = indices.len() / 3;
                for idx in prim_indices.into_u32() {
                    indices.push(vertex_offset + idx);
                }
                let tri_end = indices.len() / 3;
                for _ in tri_start..tri_end {
                    material_indices.push(mat_idx);
                }
            }
        }
    }

    let mut mesh = MeshData {
        positions,
        normals,
        uvs,
        indices,
        material_indices,
        materials,
        bounds_min,
        bounds_max,
    };
    finalize(&mut mesh);
    Ok(mesh)
}

fn extract_material(
    mat: &gltf::Material<'_>,
    images: &[gltf::image::Data],
) -> ImportMaterial {
    let pbr = mat.pbr_metallic_roughness();
    let bc = pbr.base_color_factor();

    let uv_transform = pbr
        .base_color_texture()
        .and_then(|info| {
            let t = info.texture_transform()?;
            let offset = t.offset();
            let scale = t.scale();
            Some([offset[0], offset[1], scale[0], scale[1]])
        })
        .unwrap_or([0.0, 0.0, 1.0, 1.0]);

    let albedo_texture = pbr.base_color_texture().and_then(|info| {
        let img_index = info.texture().source().index();
        images.get(img_index).and_then(decode_image)
    });

    ImportMaterial {
        name: mat.name().unwrap_or("unnamed").to_string(),
        base_color: [bc[0], bc[1], bc[2]],
        metallic: pbr.metallic_factor(),
        roughness: pbr.roughness_factor(),
        albedo_texture,
        uv_transform,
    }
}

fn decode_image(img: &gltf::image::Data) -> Option<TextureData> {
    let rgba_data = match img.format {
        gltf::image::Format::R8G8B8A8 => img.pixels.clone(),
        gltf::image::Format::R8G8B8 => {
            let mut rgba = Vec::with_capacity(img.pixels.len() / 3 * 4);
            for chunk in img.pixels.chunks(3) {
                rgba.extend_from_slice(chunk);
                rgba.push(255);
            }
            rgba
        }
        _ => return None,
    };
    Some(TextureData {
        width: img.width,
        height: img.height,
        data: rgba_data,
    })
}
