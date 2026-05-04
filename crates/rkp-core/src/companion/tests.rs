use super::*;
use crate::constants::VOXELS_PER_BRICK;
use glam::UVec3;
use std::mem::size_of;


// ------ Size checks ------

#[test]
fn bone_voxel_is_8_bytes() {
    assert_eq!(size_of::<BoneVoxel>(), 8);
}

#[test]
fn bone_brick_is_4096_bytes() {
    assert_eq!(size_of::<BoneBrick>(), 4096);
    // Verify against VOXELS_PER_BRICK constant
    assert_eq!(size_of::<BoneBrick>(), VOXELS_PER_BRICK as usize * size_of::<BoneVoxel>());
}

#[test]
fn volumetric_voxel_is_4_bytes() {
    assert_eq!(size_of::<VolumetricVoxel>(), 4);
}

#[test]
fn volumetric_brick_is_2048_bytes() {
    assert_eq!(size_of::<VolumetricBrick>(), 2048);
    assert_eq!(size_of::<VolumetricBrick>(), VOXELS_PER_BRICK as usize * size_of::<VolumetricVoxel>());
}

#[test]
fn color_voxel_is_4_bytes() {
    assert_eq!(size_of::<ColorVoxel>(), 4);
}

#[test]
fn color_brick_is_2048_bytes() {
    assert_eq!(size_of::<ColorBrick>(), 2048);
    assert_eq!(size_of::<ColorBrick>(), VOXELS_PER_BRICK as usize * size_of::<ColorVoxel>());
}

// ------ BoneVoxel pack/unpack roundtrip ------

#[test]
fn bone_voxel_roundtrip() {
    let indices = [10u8, 20, 30, 40];
    let weights = [100u8, 80, 50, 25];
    let v = BoneVoxel::new(indices, weights);

    for i in 0..4 {
        assert_eq!(v.bone_index(i), indices[i], "bone_index({i}) mismatch");
        assert_eq!(v.bone_weight(i), weights[i], "bone_weight({i}) mismatch");
    }
}

#[test]
fn bone_voxel_zero_default() {
    let v = BoneVoxel::default();
    for i in 0..4 {
        assert_eq!(v.bone_index(i), 0);
        assert_eq!(v.bone_weight(i), 0);
    }
}

#[test]
fn bone_voxel_max_values() {
    let v = BoneVoxel::new([255; 4], [255; 4]);
    for i in 0..4 {
        assert_eq!(v.bone_index(i), 255);
        assert_eq!(v.bone_weight(i), 255);
    }
}

// ------ VolumetricVoxel pack/unpack roundtrip ------

#[test]
fn volumetric_voxel_roundtrip() {
    let v = VolumetricVoxel::new(0.5, 1.0);
    // f16 has limited precision — compare with f16 round-trip tolerance
    let d = half::f16::from_f32(0.5).to_f32();
    let e = half::f16::from_f32(1.0).to_f32();
    assert!((v.density_f32() - d).abs() < 1e-3, "density mismatch: {} vs {}", v.density_f32(), d);
    assert!((v.emission_intensity_f32() - e).abs() < 1e-3, "emission mismatch: {} vs {}", v.emission_intensity_f32(), e);
}

#[test]
fn volumetric_voxel_zero() {
    let v = VolumetricVoxel::new(0.0, 0.0);
    assert_eq!(v.density_f32(), 0.0);
    assert_eq!(v.emission_intensity_f32(), 0.0);
}

#[test]
fn volumetric_voxel_density_and_emission_independent() {
    // Ensure density bits don't bleed into emission and vice versa
    let v1 = VolumetricVoxel::new(0.25, 0.0);
    let v2 = VolumetricVoxel::new(0.0, 0.75);
    assert!(v1.density_f32() > 0.0);
    assert_eq!(v1.emission_intensity_f32(), 0.0);
    assert_eq!(v2.density_f32(), 0.0);
    assert!(v2.emission_intensity_f32() > 0.0);
}

// ------ ColorVoxel pack/unpack roundtrip ------

#[test]
fn color_voxel_roundtrip() {
    let v = ColorVoxel::new(128, 64, 32, 200);
    assert_eq!(v.red(), 128);
    assert_eq!(v.green(), 64);
    assert_eq!(v.blue(), 32);
    assert_eq!(v.intensity(), 200);
}

#[test]
fn color_voxel_zero_default() {
    let v = ColorVoxel::default();
    assert_eq!(v.red(), 0);
    assert_eq!(v.green(), 0);
    assert_eq!(v.blue(), 0);
    assert_eq!(v.intensity(), 0);
}

#[test]
fn color_voxel_max_values() {
    let v = ColorVoxel::new(255, 255, 255, 255);
    assert_eq!(v.red(), 255);
    assert_eq!(v.green(), 255);
    assert_eq!(v.blue(), 255);
    assert_eq!(v.intensity(), 255);
}

#[test]
fn color_voxel_channels_independent() {
    // Each channel must not bleed into the others
    let r_only = ColorVoxel::new(255, 0, 0, 0);
    assert_eq!(r_only.red(), 255);
    assert_eq!(r_only.green(), 0);
    assert_eq!(r_only.blue(), 0);
    assert_eq!(r_only.intensity(), 0);

    let g_only = ColorVoxel::new(0, 255, 0, 0);
    assert_eq!(g_only.red(), 0);
    assert_eq!(g_only.green(), 255);
    assert_eq!(g_only.blue(), 0);
    assert_eq!(g_only.intensity(), 0);
}

// ------ Brick index helper ------

#[test]
fn brick_index_corners() {
    assert_eq!(brick_index(0, 0, 0), 0);
    assert_eq!(brick_index(7, 7, 7), 511);
    assert_eq!(brick_index(1, 0, 0), 1);
    assert_eq!(brick_index(0, 1, 0), 8);
    assert_eq!(brick_index(0, 0, 1), 64);
}

#[test]
fn brick_index_all_unique() {
    use std::collections::HashSet;
    let mut seen = HashSet::new();
    for z in 0..8u32 {
        for y in 0..8u32 {
            for x in 0..8u32 {
                let idx = brick_index(x, y, z);
                assert!(idx < 512, "index {idx} out of range for ({x},{y},{z})");
                assert!(seen.insert(idx), "duplicate index {idx} for ({x},{y},{z})");
            }
        }
    }
    assert_eq!(seen.len(), 512);
}

// ------ Brick sample/set roundtrips ------

#[test]
fn bone_brick_sample_set_roundtrip() {
    let mut brick = BoneBrick::default();
    let v = BoneVoxel::new([1, 2, 3, 4], [100, 80, 50, 25]);
    brick.set(3, 5, 7, v);
    assert_eq!(brick.sample(3, 5, 7), v);
    // Other voxels remain zero
    assert_eq!(brick.sample(0, 0, 0), BoneVoxel::default());
}

#[test]
fn volumetric_brick_sample_set_roundtrip() {
    let mut brick = VolumetricBrick::default();
    let v = VolumetricVoxel::new(0.8, 2.5);
    brick.set(1, 2, 3, v);
    assert_eq!(brick.sample(1, 2, 3), v);
    assert_eq!(brick.sample(0, 0, 0), VolumetricVoxel::default());
}

#[test]
fn color_brick_sample_set_roundtrip() {
    let mut brick = ColorBrick::default();
    let v = ColorVoxel::new(200, 100, 50, 255);
    brick.set(7, 0, 4, v);
    assert_eq!(brick.sample(7, 0, 4), v);
    assert_eq!(brick.sample(0, 0, 0), ColorVoxel::default());
}

// ------ Pod/Zeroable verification ------

#[test]
fn pod_zeroable_bone_voxel() {
    let _: BoneVoxel = bytemuck::Zeroable::zeroed();
    let bytes = [0u8; size_of::<BoneVoxel>()];
    let _: &BoneVoxel = bytemuck::from_bytes(&bytes);
}

#[test]
fn pod_zeroable_bone_brick() {
    let b: BoneBrick = bytemuck::Zeroable::zeroed();
    let bytes: &[u8] = bytemuck::bytes_of(&b);
    assert_eq!(bytes.len(), 4096);
    assert!(bytes.iter().all(|&x| x == 0));
}

#[test]
fn pod_zeroable_volumetric_voxel() {
    let _: VolumetricVoxel = bytemuck::Zeroable::zeroed();
    let bytes = [0u8; size_of::<VolumetricVoxel>()];
    let _: &VolumetricVoxel = bytemuck::from_bytes(&bytes);
}

#[test]
fn pod_zeroable_volumetric_brick() {
    let b: VolumetricBrick = bytemuck::Zeroable::zeroed();
    let bytes: &[u8] = bytemuck::bytes_of(&b);
    assert_eq!(bytes.len(), 2048);
    assert!(bytes.iter().all(|&x| x == 0));
}

#[test]
fn pod_zeroable_color_voxel() {
    let _: ColorVoxel = bytemuck::Zeroable::zeroed();
    let bytes = [0u8; size_of::<ColorVoxel>()];
    let _: &ColorVoxel = bytemuck::from_bytes(&bytes);
}

#[test]
fn pod_zeroable_color_brick() {
    let b: ColorBrick = bytemuck::Zeroable::zeroed();
    let bytes: &[u8] = bytemuck::bytes_of(&b);
    assert_eq!(bytes.len(), 2048);
    assert!(bytes.iter().all(|&x| x == 0));
}

// ------ BoneBrickLod ------

#[test]
fn bone_brick_lod_new_empty() {
    let lod = BoneBrickLod::new(glam::UVec3::new(2, 3, 4));
    assert_eq!(lod.dims, glam::UVec3::new(2, 3, 4));
    assert_eq!(lod.companion_map.len(), 24); // 2*3*4
    assert_eq!(lod.brick_count(), 0);
    assert!(lod.companion_map.iter().all(|&v| v == crate::brick_map::EMPTY_SLOT));
}

#[test]
fn bone_brick_lod_allocate_and_get() {
    let mut lod = BoneBrickLod::new(glam::UVec3::new(2, 2, 2));
    let idx = lod.allocate(3);
    assert_eq!(idx, 0);
    assert_eq!(lod.brick_count(), 1);

    // Write data and read it back.
    let v = BoneVoxel::new([10, 20, 30, 40], [100, 80, 50, 25]);
    lod.get_mut(3).unwrap().set(1, 2, 3, v);
    assert_eq!(lod.get(3).unwrap().sample(1, 2, 3), v);
}

#[test]
fn bone_brick_lod_unallocated_returns_none() {
    let lod = BoneBrickLod::new(glam::UVec3::new(4, 4, 4));
    assert!(lod.get(0).is_none());
    assert!(lod.get(63).is_none());
}

#[test]
fn bone_brick_lod_out_of_range_returns_none() {
    let lod = BoneBrickLod::new(glam::UVec3::new(2, 2, 2));
    assert!(lod.get(99).is_none());
}

#[test]
fn bone_brick_lod_multiple_allocations() {
    let mut lod = BoneBrickLod::new(glam::UVec3::new(4, 4, 4));
    let a = lod.allocate(0);
    let b = lod.allocate(10);
    let c = lod.allocate(63);
    assert_eq!(a, 0);
    assert_eq!(b, 1);
    assert_eq!(c, 2);
    assert_eq!(lod.brick_count(), 3);
    assert!(lod.get(0).is_some());
    assert!(lod.get(10).is_some());
    assert!(lod.get(63).is_some());
    assert!(lod.get(1).is_none()); // not allocated
}

#[test]
#[should_panic(expected = "already allocated")]
fn bone_brick_lod_double_allocate_panics() {
    let mut lod = BoneBrickLod::new(glam::UVec3::new(2, 2, 2));
    lod.allocate(0);
    lod.allocate(0); // should panic
}
