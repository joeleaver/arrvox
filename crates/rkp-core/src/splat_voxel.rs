use half::f16;
use rkf_core::voxel::VoxelSample;

/// A splat voxel — zero-cost wrapper over [`VoxelSample`] that reinterprets the
/// SDF distance field as opacity.
///
/// ## Layout (8 bytes, identical to VoxelSample)
///
/// ```text
/// word0: f16 opacity (bits 0–15) | blend_weight u8 (bits 16–23) | reserved (bits 24–31)
/// word1: primary material_id u16 (bits 0–15) | secondary material_id u16 (bits 16–31)
/// ```
///
/// Opacity ranges from 0.0 (empty) to 1.0 (fully opaque). The surface is where
/// opacity crosses a threshold in the trilinearly-interpolated field. Surface
/// normals are derived from the opacity gradient at shade time.
///
/// Materials use rkf-core's palette system unchanged — 16-bit IDs, dual-material
/// blending per voxel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct SplatVoxel(VoxelSample);

impl SplatVoxel {
    /// Empty voxel — zero opacity, no material.
    pub const EMPTY: Self = Self(VoxelSample { word0: 0, word1: 0 });

    /// Construct a splat voxel with opacity and a single material.
    ///
    /// `opacity`: 0.0 (empty) to 1.0 (opaque).
    /// `material_id`: primary material (0–65535).
    pub fn new(opacity: f32, material_id: u16) -> Self {
        let word0 = f16::from_f32(opacity).to_bits() as u32;
        let word1 = material_id as u32;
        Self(VoxelSample { word0, word1 })
    }

    /// Construct a splat voxel with dual-material blending.
    ///
    /// `opacity`: 0.0 (empty) to 1.0 (opaque).
    /// `material_id`: primary material (0–65535).
    /// `secondary_material_id`: secondary material (0–65535).
    /// `blend_weight`: 0 (primary only) to 255 (fully secondary).
    pub fn new_blended(
        opacity: f32,
        material_id: u16,
        secondary_material_id: u16,
        blend_weight: u8,
    ) -> Self {
        let word0 =
            (f16::from_f32(opacity).to_bits() as u32) | ((blend_weight as u32) << 16);
        let word1 = (material_id as u32) | ((secondary_material_id as u32) << 16);
        Self(VoxelSample { word0, word1 })
    }

    /// Extract the f16 opacity from bits 0–15 of word0.
    #[inline]
    pub fn opacity(&self) -> f16 {
        f16::from_bits((self.0.word0 & 0xFFFF) as u16)
    }

    /// Convenience: opacity as f32.
    #[inline]
    pub fn opacity_f32(&self) -> f32 {
        self.opacity().to_f32()
    }

    /// Set opacity (bits 0–15 of word0), preserving all other fields.
    #[inline]
    pub fn set_opacity(&mut self, opacity: f32) {
        self.0.word0 =
            (self.0.word0 & 0xFFFF_0000) | (f16::from_f32(opacity).to_bits() as u32);
    }

    /// Extract the primary material ID from bits 0–15 of word1.
    #[inline]
    pub fn material_id(&self) -> u16 {
        self.0.material_id()
    }

    /// Replace the primary material ID, preserving all other fields.
    #[inline]
    pub fn set_material_id(&mut self, id: u16) {
        self.0.set_material_id(id);
    }

    /// Extract the secondary material ID from bits 16–31 of word1.
    #[inline]
    pub fn secondary_material_id(&self) -> u16 {
        self.0.secondary_material_id()
    }

    /// Extract the blend weight from bits 16–23 of word0.
    /// 0 = primary only, 255 = fully secondary.
    #[inline]
    pub fn blend_weight(&self) -> u8 {
        self.0.blend_weight()
    }

    /// Returns `true` if opacity is effectively zero (empty voxel).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.opacity_f32() < f32::EPSILON
    }

    /// Access the underlying `VoxelSample`.
    #[inline]
    pub fn as_voxel_sample(&self) -> &VoxelSample {
        &self.0
    }

    /// Mutable access to the underlying `VoxelSample`.
    #[inline]
    pub fn as_voxel_sample_mut(&mut self) -> &mut VoxelSample {
        &mut self.0
    }
}

impl From<VoxelSample> for SplatVoxel {
    #[inline]
    fn from(v: VoxelSample) -> Self {
        Self(v)
    }
}

impl From<SplatVoxel> for VoxelSample {
    #[inline]
    fn from(s: SplatVoxel) -> Self {
        s.0
    }
}

impl Default for SplatVoxel {
    /// Default is empty — zero opacity, no material.
    fn default() -> Self {
        Self::EMPTY
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    fn same_size_as_voxel_sample() {
        assert_eq!(mem::size_of::<SplatVoxel>(), mem::size_of::<VoxelSample>());
        assert_eq!(mem::size_of::<SplatVoxel>(), 8);
    }

    #[test]
    fn repr_transparent_layout() {
        // SplatVoxel should be transmutable to/from VoxelSample
        let sv = SplatVoxel::new(0.75, 42);
        let vs: VoxelSample = sv.into();
        let back: SplatVoxel = vs.into();
        assert_eq!(sv, back);
    }

    #[test]
    fn new_roundtrip_opacity_and_material() {
        let sv = SplatVoxel::new(0.5, 100);
        assert!((sv.opacity_f32() - 0.5).abs() < 0.01);
        assert_eq!(sv.material_id(), 100);
        assert_eq!(sv.secondary_material_id(), 0);
        assert_eq!(sv.blend_weight(), 0);
    }

    #[test]
    fn new_blended_roundtrip() {
        let sv = SplatVoxel::new_blended(0.8, 300, 500, 128);
        assert!((sv.opacity_f32() - 0.8).abs() < 0.01);
        assert_eq!(sv.material_id(), 300);
        assert_eq!(sv.secondary_material_id(), 500);
        assert_eq!(sv.blend_weight(), 128);
    }

    #[test]
    fn set_opacity_preserves_other_fields() {
        let mut sv = SplatVoxel::new_blended(1.0, 42, 99, 200);
        sv.set_opacity(0.25);
        assert!((sv.opacity_f32() - 0.25).abs() < 0.01);
        assert_eq!(sv.material_id(), 42);
        assert_eq!(sv.secondary_material_id(), 99);
        assert_eq!(sv.blend_weight(), 200);
    }

    #[test]
    fn set_material_id_preserves_other_fields() {
        let mut sv = SplatVoxel::new_blended(0.5, 10, 700, 50);
        sv.set_material_id(999);
        assert_eq!(sv.material_id(), 999);
        assert_eq!(sv.secondary_material_id(), 700);
        assert!((sv.opacity_f32() - 0.5).abs() < 0.01);
        assert_eq!(sv.blend_weight(), 50);
    }

    #[test]
    fn empty_voxel() {
        let sv = SplatVoxel::EMPTY;
        assert!(sv.is_empty());
        assert_eq!(sv.opacity_f32(), 0.0);
        assert_eq!(sv.material_id(), 0);
    }

    #[test]
    fn default_is_empty() {
        let sv = SplatVoxel::default();
        assert!(sv.is_empty());
        assert_eq!(sv, SplatVoxel::EMPTY);
    }

    #[test]
    fn opaque_voxel_is_not_empty() {
        let sv = SplatVoxel::new(1.0, 0);
        assert!(!sv.is_empty());
    }

    #[test]
    fn full_material_id_range() {
        for &id in &[0u16, 1, 63, 255, 1000, 65535] {
            let sv = SplatVoxel::new(1.0, id);
            assert_eq!(sv.material_id(), id, "material_id roundtrip failed for {id}");
        }
    }

    #[test]
    fn fields_do_not_bleed() {
        let sv = SplatVoxel::new_blended(0.0, 0xFFFF, 0xFFFF, 0xFF);
        assert_eq!(sv.opacity_f32(), 0.0);
        assert_eq!(sv.material_id(), 0xFFFF);
        assert_eq!(sv.secondary_material_id(), 0xFFFF);
        assert_eq!(sv.blend_weight(), 0xFF);
    }

    #[test]
    fn all_fields_max_no_overlap() {
        // Opacity 1.0 as f16 = 0x3C00
        let sv = SplatVoxel::new_blended(1.0, 65535, 65535, 255);
        let vs: VoxelSample = sv.into();
        assert_eq!(vs.word0 & 0xFFFF, 0x3C00); // f16 1.0
        assert_eq!((vs.word0 >> 16) & 0xFF, 0xFF); // blend weight
        assert_eq!(vs.word0 >> 24, 0); // reserved bits zero
        assert_eq!(vs.word1, 0xFFFF_FFFF); // both materials maxed
    }

    #[test]
    fn from_voxel_sample_preserves_bits() {
        let vs = VoxelSample::new_blended(0.5, 42, 99, 128);
        let sv = SplatVoxel::from(vs);
        // The distance field bits are now interpreted as opacity
        assert_eq!(sv.opacity(), vs.distance());
        assert_eq!(sv.material_id(), vs.material_id());
        assert_eq!(sv.secondary_material_id(), vs.secondary_material_id());
        assert_eq!(sv.blend_weight(), vs.blend_weight());
    }

    #[test]
    fn as_voxel_sample_roundtrip() {
        let sv = SplatVoxel::new(0.75, 42);
        let vs = sv.as_voxel_sample();
        let back = SplatVoxel::from(*vs);
        assert_eq!(sv, back);
    }
}
