//! Mesh preparation — apply the [`ImportConfig`]'s rotation offset,
//! re-center to origin, scale the longest axis to `target_size`, and
//! optionally apply an extra uniform scale.
//!
//! The returned [`NormalizationParams`] captures everything the
//! runtime needs to match the normalized mesh coordinates when
//! skinning — it's persisted into the `.rkskel` sidecar.

use glam::{Quat, Vec3};

use crate::config::ImportConfig;
use crate::mesh::MeshData;

/// Normalization parameters applied during mesh preparation.
///
/// Stored in the `.rkskel` asset so that runtime skinning can match
/// the normalized mesh coordinates.
pub struct NormalizationParams {
    /// Post-rotation mesh centre used for normalization (the translation
    /// that moved the mesh to the origin).
    pub center: Vec3,
    /// Uniform scale factor applied after translation.
    pub scale: f32,
    /// Rotation offset in degrees (XYZ Euler), applied before normalization.
    pub rotation_offset: [f32; 3],
    /// Pre-rotation mesh centre — the pivot point the rotation was
    /// applied around.
    pub rotation_center: Vec3,
}

/// Prepare the mesh according to `config`: rotate in original space,
/// centre on the origin, scale so the longest axis matches
/// `config.target_size`, then apply any `scale_override`. Mutates the
/// mesh in place and returns the normalization parameters actually
/// applied.
pub fn prepare_mesh(mesh: &mut MeshData, config: &ImportConfig) -> NormalizationParams {
    let [rx, ry, rz] = config.rotation_offset;
    let rotation_center = (mesh.bounds_min + mesh.bounds_max) * 0.5;
    if rx != 0.0 || ry != 0.0 || rz != 0.0 {
        let rot = Quat::from_euler(
            glam::EulerRot::XYZ,
            rx.to_radians(),
            ry.to_radians(),
            rz.to_radians(),
        );
        for pos in &mut mesh.positions {
            *pos = rot * (*pos - rotation_center) + rotation_center;
        }
        for n in &mut mesh.normals {
            *n = rot * *n;
        }
        recompute_bounds(mesh);
    }

    let mut norm_center = Vec3::ZERO;
    let mut norm_scale = 1.0f32;

    if !config.no_normalize && config.target_size > 0.0 {
        let extent = mesh.bounds_max - mesh.bounds_min;
        let longest = extent.x.max(extent.y).max(extent.z);
        if longest > 1e-6 {
            let scale = config.target_size / longest;
            let center = (mesh.bounds_min + mesh.bounds_max) * 0.5;
            norm_center = center;
            norm_scale = scale;
            for pos in &mut mesh.positions {
                *pos = (*pos - center) * scale;
            }
            mesh.bounds_min = (mesh.bounds_min - center) * scale;
            mesh.bounds_max = (mesh.bounds_max - center) * scale;
        }
    }

    if let Some(s) = config.scale_override {
        if (s - 1.0).abs() > 1e-6 {
            norm_scale *= s;
            for pos in &mut mesh.positions {
                *pos *= s;
            }
            mesh.bounds_min *= s;
            mesh.bounds_max *= s;
        }
    }

    NormalizationParams {
        center: norm_center,
        scale: norm_scale,
        rotation_offset: config.rotation_offset,
        rotation_center,
    }
}

fn recompute_bounds(mesh: &mut MeshData) {
    let mut bmin = Vec3::splat(f32::MAX);
    let mut bmax = Vec3::splat(f32::MIN);
    for p in &mesh.positions {
        bmin = bmin.min(*p);
        bmax = bmax.max(*p);
    }
    mesh.bounds_min = bmin;
    mesh.bounds_max = bmax;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::ImportMaterial;

    fn simple_cube() -> MeshData {
        let positions = vec![
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(-1.0, 1.0, -1.0),
        ];
        MeshData {
            positions,
            normals: vec![Vec3::Y; 3],
            uvs: Vec::new(),
            indices: vec![0, 1, 2],
            material_indices: vec![0],
            materials: vec![ImportMaterial::default()],
            bounds_min: Vec3::splat(-1.0),
            bounds_max: Vec3::splat(1.0),
        }
    }

    #[test]
    fn no_normalize_preserves_mesh() {
        let mut mesh = simple_cube();
        let original = mesh.positions.clone();
        let config = ImportConfig {
            no_normalize: true,
            ..ImportConfig::default()
        };
        prepare_mesh(&mut mesh, &config);
        assert_eq!(mesh.positions, original);
    }

    #[test]
    fn normalize_scales_to_target_size() {
        let mut mesh = simple_cube();
        let config = ImportConfig {
            target_size: 2.0,
            ..ImportConfig::default()
        };
        prepare_mesh(&mut mesh, &config);
        let extent = mesh.bounds_max - mesh.bounds_min;
        let longest = extent.x.max(extent.y).max(extent.z);
        assert!((longest - 2.0).abs() < 1e-4, "longest = {longest}");
    }

    #[test]
    fn scale_override_compounds() {
        let mut mesh = simple_cube();
        let config = ImportConfig {
            target_size: 1.0,
            scale_override: Some(2.0),
            ..ImportConfig::default()
        };
        let params = prepare_mesh(&mut mesh, &config);
        // Target size 1 gives scale 0.5 (mesh was 2 units across), × 2.0 = 1.0
        assert!((params.scale - 1.0).abs() < 1e-4);
    }
}
