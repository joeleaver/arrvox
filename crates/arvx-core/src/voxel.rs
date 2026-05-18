use bytemuck::{Pod, Zeroable};
use half::f16;

/// A single voxel sample — 8 bytes, tightly packed for GPU upload.
///
/// Layout:
/// ```text
/// Word 0 (u32): [ f16 distance (bits 0–15) | blend_weight (bits 16–23) | reserved (bits 24–31) ]
/// Word 1 (u32): [ primary material_id (bits 0–15) | secondary material_id (bits 16–31) ]
/// ```
///
/// Per-voxel color is stored in a separate `ColorBrick` companion pool, not inline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct VoxelSample {
    #[allow(missing_docs)]
    pub word0: u32,
    #[allow(missing_docs)]
    pub word1: u32,
}

// SAFETY: VoxelSample is repr(C), all fields are u32 which are Pod
unsafe impl Zeroable for VoxelSample {}
unsafe impl Pod for VoxelSample {}

impl VoxelSample {
    /// Construct a new voxel sample with distance, material, and color.
    ///
    /// `blend_weight` is 0 (primary only) to 255 (fully secondary).
    /// Secondary material defaults to 0; use [`new_blended`] for dual-material.
    pub fn new(distance: f32, material_id: u16, blend_weight: u8) -> Self {
        let word0 = (f16::from_f32(distance).to_bits() as u32)
            | ((blend_weight as u32) << 16);
        let word1 = material_id as u32;
        Self { word0, word1 }
    }

    /// Construct a new voxel sample with distance, primary + secondary material, and blend weight.
    ///
    /// `blend_weight` is 0 (primary only) to 255 (fully secondary).
    pub fn new_blended(
        distance: f32,
        material_id: u16,
        secondary_material_id: u16,
        blend_weight: u8,
    ) -> Self {
        let word0 = (f16::from_f32(distance).to_bits() as u32)
            | ((blend_weight as u32) << 16);
        let word1 = (material_id as u32) | ((secondary_material_id as u32) << 16);
        Self { word0, word1 }
    }

    /// Extract the f16 signed distance stored in the lower 16 bits of word0.
    #[inline]
    pub fn distance(&self) -> f16 {
        f16::from_bits((self.word0 & 0xFFFF) as u16)
    }

    /// Convenience: return distance as f32.
    #[inline]
    pub fn distance_f32(&self) -> f32 {
        self.distance().to_f32()
    }

    /// Extract the primary material id stored in bits 0–15 of word1 (16 bits, 0–65535).
    #[inline]
    pub fn material_id(&self) -> u16 {
        (self.word1 & 0xFFFF) as u16
    }

    /// Replace the primary material id (bits 0–15 of word1), preserving all other fields.
    #[inline]
    pub fn set_material_id(&mut self, id: u16) {
        self.word1 = (self.word1 & 0xFFFF_0000) | (id as u32);
    }

    /// Extract the secondary material id stored in bits 16–31 of word1 (16 bits, 0–65535).
    #[inline]
    pub fn secondary_material_id(&self) -> u16 {
        ((self.word1 >> 16) & 0xFFFF) as u16
    }

    /// Extract the blend weight from bits 16–23 of word0 (0=primary, 255=secondary).
    #[inline]
    pub fn blend_weight(&self) -> u8 {
        ((self.word0 >> 16) & 0xFF) as u8
    }

    /// Construct a voxel from geometry-first data (single material, no blend).
    ///
    /// `distance_f16_bits` is the raw f16 bits from [`SdfCache`].
    pub fn from_geometry_data(distance_f16_bits: u16, material_id: u16, blend_weight: u8) -> Self {
        let word0 = (distance_f16_bits as u32)
            | ((blend_weight as u32) << 16);
        let word1 = material_id as u32;
        Self { word0, word1 }
    }

    /// Construct from geometry-first data with secondary material and blend weight.
    ///
    /// `blend_weight` is 0–255 (0 = primary only, 255 = fully secondary).
    /// Bits 24–31 of word0 are reserved (zero).
    pub fn from_geometry_data_blended(
        distance_f16_bits: u16,
        material_id: u16,
        secondary_material_id: u16,
        blend_weight: u8,
    ) -> Self {
        let word0 = (distance_f16_bits as u32)
            | ((blend_weight as u32) << 16);
        let word1 = (material_id as u32)
            | ((secondary_material_id as u32) << 16);
        Self { word0, word1 }
    }
}

impl Default for VoxelSample {
    /// Returns a voxel far from any surface: distance = f16::INFINITY, material_id = 0, no blend.
    fn default() -> Self {
        Self::new(f32::INFINITY, 0, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    fn size_is_8_bytes() {
        assert_eq!(mem::size_of::<VoxelSample>(), 8);
    }

    #[test]
    fn pod_zeroable_bytes_of_works() {
        let sample = VoxelSample::default();
        let bytes = bytemuck::bytes_of(&sample);
        assert_eq!(bytes.len(), 8);
    }

    #[test]
    fn zero_sample_all_zeros() {
        let sample: VoxelSample = bytemuck::Zeroable::zeroed();
        assert_eq!(sample.word0, 0);
        assert_eq!(sample.word1, 0);
        assert_eq!(sample.material_id(), 0);
        assert_eq!(sample.distance(), f16::from_f32(0.0));
    }

    #[test]
    fn default_has_infinity_distance() {
        let sample = VoxelSample::default();
        assert!(sample.distance().is_infinite());
        assert!(sample.distance_f32().is_infinite());
        assert!(sample.distance_f32() > 0.0);
    }

    #[test]
    fn roundtrip_typical_values() {
        let dist = 1.5_f32;
        let mat = 42_u16;

        let sample = VoxelSample::new(dist, mat, 0);

        assert_eq!(sample.distance_f32(), 1.5_f32);
        assert_eq!(sample.material_id(), mat);
    }

    #[test]
    fn roundtrip_negative_distance() {
        let sample = VoxelSample::new(-0.5, 1, 0);
        assert_eq!(sample.distance_f32(), -0.5_f32);
        assert_eq!(sample.material_id(), 1);
    }

    #[test]
    fn edge_case_max_material_id() {
        let sample = VoxelSample::new(0.0, 65535, 0);
        assert_eq!(sample.material_id(), 65535);
    }

    #[test]
    fn full_range_material_id() {
        // 16-bit material IDs support full u16 range
        for &id in &[0u16, 1, 63, 64, 255, 256, 1000, 65535] {
            let sample = VoxelSample::new(0.0, id, 0);
            assert_eq!(sample.material_id(), id, "material_id roundtrip failed for {id}");
        }
    }

    #[test]
    fn set_material_id_preserves_distance() {
        let mut sample = VoxelSample::new(-0.25, 5, 0);
        assert_eq!(sample.material_id(), 5);
        sample.set_material_id(42);
        assert_eq!(sample.material_id(), 42);
        assert!((sample.distance_f32() - (-0.25)).abs() < 0.01);
    }

    #[test]
    fn edge_case_f16_max_distance() {
        let dist = f16::MAX.to_f32();
        let sample = VoxelSample::new(dist, 0, 0);
        assert_eq!(sample.distance(), f16::MAX);
    }

    #[test]
    fn from_geometry_data_roundtrip() {
        let dist_bits = f16::from_f32(1.5).to_bits();
        let sample = VoxelSample::from_geometry_data(dist_bits, 42, 0);
        assert_eq!(sample.distance_f32(), 1.5);
        assert_eq!(sample.material_id(), 42);
    }

    #[test]
    fn fields_do_not_bleed_into_each_other() {
        let sample = VoxelSample::new(0.0, 0xFFFF, 0);
        assert_eq!(sample.distance(), f16::from_bits(0));
        assert_eq!(sample.material_id(), 0xFFFF);
        assert_eq!(sample.secondary_material_id(), 0);
    }

    #[test]
    fn new_blended_fields_independent() {
        let sample = VoxelSample::new_blended(0.0, 0xFFFF, 0xFFFF, 0xFF);
        assert_eq!(sample.material_id(), 0xFFFF);
        assert_eq!(sample.secondary_material_id(), 0xFFFF);
        assert_eq!(sample.blend_weight(), 0xFF);
        assert_eq!(sample.distance(), f16::from_bits(0));
    }

    #[test]
    fn blended_material_roundtrip() {
        let dist_bits = f16::from_f32(-0.5).to_bits();
        let sample = VoxelSample::from_geometry_data_blended(dist_bits, 300, 500, 128);
        assert_eq!(sample.material_id(), 300);
        assert_eq!(sample.secondary_material_id(), 500);
        assert_eq!(sample.blend_weight(), 128);
        assert!((sample.distance_f32() - (-0.5)).abs() < 0.01);
    }

    #[test]
    fn set_material_id_preserves_secondary() {
        let dist_bits = f16::from_f32(1.0).to_bits();
        let mut sample = VoxelSample::from_geometry_data_blended(dist_bits, 3, 700, 0);
        assert_eq!(sample.secondary_material_id(), 700);
        sample.set_material_id(10);
        assert_eq!(sample.material_id(), 10);
        assert_eq!(sample.secondary_material_id(), 700); // preserved
    }

    #[test]
    fn all_fields_pack_without_overlap() {
        let dist_bits = 0xFFFF_u16;
        let sample = VoxelSample::from_geometry_data_blended(dist_bits, 65535, 65535, 0xFF);
        // word0: distance(0xFFFF) | blend(0xFF << 16) | reserved(0)
        assert_eq!(sample.word0 & 0xFFFF, 0xFFFF); // distance
        assert_eq!((sample.word0 >> 16) & 0xFF, 0xFF); // blend weight
        assert_eq!(sample.word0 >> 24, 0); // reserved bits are zero
        // word1: material(0xFFFF) | secondary(0xFFFF << 16)
        assert_eq!(sample.word1, 0xFFFF_FFFF);
        assert_eq!(sample.material_id(), 65535);
        assert_eq!(sample.secondary_material_id(), 65535);
        assert_eq!(sample.blend_weight(), 0xFF);
    }

    #[test]
    fn bytemuck_cast_slice_works() {
        let samples = vec![
            VoxelSample::new(1.0, 1, 0),
            VoxelSample::new(2.0, 2, 0),
        ];
        let bytes: &[u8] = bytemuck::cast_slice(&samples);
        assert_eq!(bytes.len(), 16);
    }
}
