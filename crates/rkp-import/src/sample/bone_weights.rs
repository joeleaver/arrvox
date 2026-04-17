//! Bone-weight sampling at a point on a triangle.
//!
//! Gathers the 3 vertex's up-to-4 bone influences, interpolates them
//! by barycentric coordinates, merges duplicates by bone index, keeps
//! the top 4 by weight, normalizes, and quantizes to `u8` (weights
//! sum to exactly 255 after a rounding fix-up).

use rkp_core::companion::BoneVoxel;

use crate::mesh::MeshData;
use crate::skeleton::VertexSkinning;

/// Interpolate bone weights at a point on a triangle using barycentric
/// coordinates. Given a triangle index and bary `[w0, w1, w2]`,
/// produces a [`BoneVoxel`] with the top 4 bone influences (weights
/// quantized to `u8` summing to 255). Returns a zero `BoneVoxel` when
/// there's no skinning data or the triangle index is out of range.
pub fn sample_bone_weights_at_triangle(
    mesh: &MeshData,
    skinning: &VertexSkinning,
    tri_idx: usize,
    barycentric: &[f32; 3],
) -> BoneVoxel {
    if skinning.joints.is_empty() || mesh.indices.is_empty() {
        return BoneVoxel::default();
    }

    let base = tri_idx * 3;
    if base + 2 >= mesh.indices.len() {
        return BoneVoxel::default();
    }

    let vi = [
        mesh.indices[base] as usize,
        mesh.indices[base + 1] as usize,
        mesh.indices[base + 2] as usize,
    ];

    // Gather interpolated influences, merging by bone index.
    // Up to 12 raw entries (3 vertices × 4 slots); merge collapses
    // duplicates when multiple vertices share a bone.
    let mut influence_map: Vec<(i32, f32)> = Vec::with_capacity(12);

    for v in 0..3 {
        let vertex = vi[v];
        if vertex >= skinning.joints.len() {
            continue;
        }
        let bary = barycentric[v];
        if bary <= 0.0 {
            continue;
        }

        let joints = &skinning.joints[vertex];
        let weights = &skinning.weights[vertex];

        for slot in 0..4 {
            let bone = joints[slot];
            let w = weights[slot];
            if bone < 0 || w <= 0.0 {
                continue;
            }
            let contribution = w * bary;

            if let Some(entry) = influence_map.iter_mut().find(|(b, _)| *b == bone) {
                entry.1 += contribution;
            } else {
                influence_map.push((bone, contribution));
            }
        }
    }

    if influence_map.is_empty() {
        return BoneVoxel::default();
    }

    // Sort descending by weight — NaN-safe fallback keeps ordering stable.
    influence_map.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    influence_map.truncate(4);

    let total: f32 = influence_map.iter().map(|(_, w)| w).sum();
    if total <= 0.0 {
        return BoneVoxel::default();
    }

    let mut indices = [0u8; 4];
    let mut weights = [0u8; 4];
    let mut weight_sum: u16 = 0;

    for (i, &(bone, w)) in influence_map.iter().enumerate() {
        indices[i] = bone as u8;
        let quantized = ((w / total) * 255.0).round() as u8;
        weights[i] = quantized;
        weight_sum += quantized as u16;
    }

    // Rounding fix-up: nudge the largest weight so the four u8s sum
    // to exactly 255. `max_by_key` over a fixed `[u8; 4]` is always
    // `Some`; `unwrap_or(0)` is a defensive no-op in case the array
    // shape ever changes.
    if weight_sum > 0 && weight_sum != 255 {
        let diff = 255i16 - weight_sum as i16;
        let max_slot = weights
            .iter()
            .enumerate()
            .max_by_key(|&(_, &w)| w)
            .map(|(i, _)| i)
            .unwrap_or(0);
        weights[max_slot] = (weights[max_slot] as i16 + diff).clamp(0, 255) as u8;
    }

    BoneVoxel::new(indices, weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{ImportMaterial, MeshData};
    use glam::Vec3;

    fn triangle_mesh() -> MeshData {
        MeshData {
            positions: vec![Vec3::ZERO, Vec3::X, Vec3::Y],
            normals: vec![Vec3::Z; 3],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            indices: vec![0, 1, 2],
            material_indices: vec![0],
            materials: vec![ImportMaterial::default()],
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::new(1.0, 1.0, 0.0),
        }
    }

    fn single_bone() -> VertexSkinning {
        VertexSkinning {
            joints: vec![[0, -1, -1, -1]; 3],
            weights: vec![[1.0, 0.0, 0.0, 0.0]; 3],
        }
    }

    fn two_bones() -> VertexSkinning {
        VertexSkinning {
            joints: vec![[0, -1, -1, -1], [1, -1, -1, -1], [0, 1, -1, -1]],
            weights: vec![
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0, 0.0],
            ],
        }
    }

    #[test]
    fn single_bone_all_vertices() {
        let bv = sample_bone_weights_at_triangle(&triangle_mesh(), &single_bone(), 0, &[0.33, 0.33, 0.34]);
        assert_eq!(bv.bone_index(0), 0);
        assert_eq!(bv.bone_weight(0), 255);
    }

    #[test]
    fn two_bones_interpolated() {
        let mesh = triangle_mesh();
        let skin = two_bones();

        let bv = sample_bone_weights_at_triangle(&mesh, &skin, 0, &[1.0, 0.0, 0.0]);
        assert_eq!(bv.bone_index(0), 0);
        assert_eq!(bv.bone_weight(0), 255);

        let bv = sample_bone_weights_at_triangle(&mesh, &skin, 0, &[0.0, 1.0, 0.0]);
        assert_eq!(bv.bone_index(0), 1);
        assert_eq!(bv.bone_weight(0), 255);

        let bv = sample_bone_weights_at_triangle(&mesh, &skin, 0, &[0.5, 0.5, 0.0]);
        let (w0, w1) = (bv.bone_weight(0), bv.bone_weight(1));
        assert!((w0 as i16 - 128).abs() <= 1);
        assert!((w1 as i16 - 128).abs() <= 1);
        assert_eq!(w0 as u16 + w1 as u16, 255);
    }

    #[test]
    fn empty_skinning_returns_zero() {
        let bv = sample_bone_weights_at_triangle(
            &triangle_mesh(),
            &VertexSkinning::default(),
            0,
            &[0.5, 0.3, 0.2],
        );
        assert_eq!(bv.bone_weight(0), 0);
    }

    #[test]
    fn weights_always_sum_to_255() {
        let bv = sample_bone_weights_at_triangle(&triangle_mesh(), &two_bones(), 0, &[0.33, 0.33, 0.34]);
        let total: u16 = (0..4).map(|i| bv.bone_weight(i) as u16).sum();
        assert_eq!(total, 255);
    }
}
