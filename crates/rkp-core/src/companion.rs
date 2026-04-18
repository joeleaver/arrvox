//! Companion brick types for the RKIField SDF engine.
//!
//! Each companion brick type parallels the main [`Brick`] (8×8×8 = 512 voxels) but stores
//! different per-voxel data:
//!
//! - [`BoneBrick`] — skeletal bone influence weights for animated objects (4KB per brick)
//! - [`VolumetricBrick`] — density + emission for fog/smoke/fire volumes (2KB per brick)
//! - [`ColorBrick`] — per-voxel RGBA color data (2KB per brick)
//!
//! All types implement [`bytemuck::Pod`] and [`bytemuck::Zeroable`] for safe GPU upload.

use bytemuck::{Pod, Zeroable};

// ---------------------------------------------------------------------------
// Shared index helper
// ---------------------------------------------------------------------------

/// Compute a flat voxel index from 3D coordinates within a brick.
///
/// Layout: `x + y * 8 + z * 64` — matches the main brick's memory order.
///
/// # Panics
///
/// Panics in debug builds if any coordinate is >= 8.
#[inline]
pub fn brick_index(x: u32, y: u32, z: u32) -> usize {
    debug_assert!(x < 8, "x={x} out of brick bounds");
    debug_assert!(y < 8, "y={y} out of brick bounds");
    debug_assert!(z < 8, "z={z} out of brick bounds");
    (x + y * 8 + z * 64) as usize
}

// ---------------------------------------------------------------------------
// BoneVoxel / BoneBrick
// ---------------------------------------------------------------------------

/// Bone influence data for a single voxel — 8 bytes.
///
/// Layout:
/// - `indices`: 4 × u8 bone indices packed into a u32 (byte0 = bone_index_0, …)
/// - `weights`: 4 × u8 bone weights packed into a u32 (byte0 = bone_weight_0, …)
///
/// Weights are u8-normalized (0–255). They must sum to 255 in well-formed data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct BoneVoxel {
    /// 4 × u8 bone indices packed as little-endian u32.
    pub indices: u32,
    /// 4 × u8 bone weights packed as little-endian u32.
    pub weights: u32,
}

// SAFETY: repr(C), all fields are u32 (no padding, no invalid bit patterns).
unsafe impl Zeroable for BoneVoxel {}
unsafe impl Pod for BoneVoxel {}

impl BoneVoxel {
    /// Construct from separate index and weight arrays.
    ///
    /// Bytes are packed little-endian: `indices[0]` occupies the lowest byte.
    #[inline]
    pub fn new(indices: [u8; 4], weights: [u8; 4]) -> Self {
        Self {
            indices: u32::from_le_bytes(indices),
            weights: u32::from_le_bytes(weights),
        }
    }

    /// Extract the ith bone index (i in 0..4).
    #[inline]
    pub fn bone_index(&self, i: usize) -> u8 {
        debug_assert!(i < 4, "bone index slot {i} out of range");
        self.indices.to_le_bytes()[i]
    }

    /// Extract the ith bone weight (i in 0..4).
    #[inline]
    pub fn bone_weight(&self, i: usize) -> u8 {
        debug_assert!(i < 4, "bone weight slot {i} out of range");
        self.weights.to_le_bytes()[i]
    }
}

/// A bone-data companion brick — 512 [`BoneVoxel`]s = 4 096 bytes.
#[derive(Clone, Copy)]
#[repr(C, align(4))]
pub struct BoneBrick {
    /// Voxel array, indexed via [`brick_index`].
    pub data: [BoneVoxel; 512],
}

// SAFETY: BoneVoxel is Pod/Zeroable; array of Pod is Pod; no padding added by
// repr(C,align(4)) because BoneVoxel is already 8-byte-aligned.
unsafe impl Zeroable for BoneBrick {}
unsafe impl Pod for BoneBrick {}

impl std::fmt::Debug for BoneBrick {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let non_zero = self.data.iter().filter(|v| v.weights != 0).count();
        f.debug_struct("BoneBrick")
            .field("weighted_voxels", &non_zero)
            .finish()
    }
}

impl Default for BoneBrick {
    fn default() -> Self {
        bytemuck::Zeroable::zeroed()
    }
}

impl BoneBrick {
    /// Flat index from 3D brick coordinates.
    #[inline]
    pub fn index(x: u32, y: u32, z: u32) -> usize {
        brick_index(x, y, z)
    }

    /// Read the voxel at `(x, y, z)`.
    #[inline]
    pub fn sample(&self, x: u32, y: u32, z: u32) -> BoneVoxel {
        self.data[Self::index(x, y, z)]
    }

    /// Write `val` to `(x, y, z)`.
    #[inline]
    pub fn set(&mut self, x: u32, y: u32, z: u32, val: BoneVoxel) {
        self.data[Self::index(x, y, z)] = val;
    }
}

// ---------------------------------------------------------------------------
// VolumetricVoxel / VolumetricBrick
// ---------------------------------------------------------------------------

/// Volumetric data for a single voxel — 4 bytes.
///
/// Layout (packed u32, little-endian halves):
/// - lower 16 bits: `f16` density
/// - upper 16 bits: `f16` emission_intensity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct VolumetricVoxel {
    /// `f16` density in lower 16 bits, `f16` emission_intensity in upper 16 bits.
    pub packed: u32,
}

// SAFETY: repr(C), single u32 field — no padding, no invalid bit patterns.
unsafe impl Zeroable for VolumetricVoxel {}
unsafe impl Pod for VolumetricVoxel {}

impl VolumetricVoxel {
    /// Construct from f32 density and emission intensity, converting to f16.
    #[inline]
    pub fn new(density: f32, emission_intensity: f32) -> Self {
        let d = half::f16::from_f32(density);
        let e = half::f16::from_f32(emission_intensity);
        Self {
            packed: (d.to_bits() as u32) | ((e.to_bits() as u32) << 16),
        }
    }

    /// Density as `f16`.
    #[inline]
    pub fn density(&self) -> half::f16 {
        half::f16::from_bits((self.packed & 0xFFFF) as u16)
    }

    /// Density as `f32`.
    #[inline]
    pub fn density_f32(&self) -> f32 {
        self.density().to_f32()
    }

    /// Emission intensity as `f16`.
    #[inline]
    pub fn emission_intensity(&self) -> half::f16 {
        half::f16::from_bits((self.packed >> 16) as u16)
    }

    /// Emission intensity as `f32`.
    #[inline]
    pub fn emission_intensity_f32(&self) -> f32 {
        self.emission_intensity().to_f32()
    }
}

/// A volumetric companion brick — 512 [`VolumetricVoxel`]s = 2 048 bytes.
#[derive(Clone, Copy)]
#[repr(C, align(4))]
pub struct VolumetricBrick {
    /// Voxel array, indexed via [`brick_index`].
    pub data: [VolumetricVoxel; 512],
}

// SAFETY: VolumetricVoxel is Pod/Zeroable; array of Pod is Pod.
unsafe impl Zeroable for VolumetricBrick {}
unsafe impl Pod for VolumetricBrick {}

impl Default for VolumetricBrick {
    fn default() -> Self {
        bytemuck::Zeroable::zeroed()
    }
}

impl VolumetricBrick {
    /// Flat index from 3D brick coordinates.
    #[inline]
    pub fn index(x: u32, y: u32, z: u32) -> usize {
        brick_index(x, y, z)
    }

    /// Read the voxel at `(x, y, z)`.
    #[inline]
    pub fn sample(&self, x: u32, y: u32, z: u32) -> VolumetricVoxel {
        self.data[Self::index(x, y, z)]
    }

    /// Write `val` to `(x, y, z)`.
    #[inline]
    pub fn set(&mut self, x: u32, y: u32, z: u32, val: VolumetricVoxel) {
        self.data[Self::index(x, y, z)] = val;
    }
}

// ---------------------------------------------------------------------------
// ColorVoxel / ColorBrick
// ---------------------------------------------------------------------------

/// Per-voxel color data — 4 bytes.
///
/// Layout (packed u32, little-endian bytes):
/// - byte 0: red
/// - byte 1: green
/// - byte 2: blue
/// - byte 3: intensity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct ColorVoxel {
    /// `r | (g << 8) | (b << 16) | (intensity << 24)` packed into a u32.
    pub packed: u32,
}

// SAFETY: repr(C), single u32 field.
unsafe impl Zeroable for ColorVoxel {}
unsafe impl Pod for ColorVoxel {}

impl ColorVoxel {
    /// Construct from individual RGBA + intensity components.
    #[inline]
    pub fn new(r: u8, g: u8, b: u8, intensity: u8) -> Self {
        Self {
            packed: (r as u32) | ((g as u32) << 8) | ((b as u32) << 16) | ((intensity as u32) << 24),
        }
    }

    /// Red channel.
    #[inline]
    pub fn red(&self) -> u8 {
        (self.packed & 0xFF) as u8
    }

    /// Green channel.
    #[inline]
    pub fn green(&self) -> u8 {
        ((self.packed >> 8) & 0xFF) as u8
    }

    /// Blue channel.
    #[inline]
    pub fn blue(&self) -> u8 {
        ((self.packed >> 16) & 0xFF) as u8
    }

    /// Intensity channel.
    #[inline]
    pub fn intensity(&self) -> u8 {
        ((self.packed >> 24) & 0xFF) as u8
    }
}

/// A color companion brick — 512 [`ColorVoxel`]s = 2 048 bytes.
#[derive(Clone, Copy)]
#[repr(C, align(4))]
pub struct ColorBrick {
    /// Voxel array, indexed via [`brick_index`].
    pub data: [ColorVoxel; 512],
}

// SAFETY: ColorVoxel is Pod/Zeroable; array of Pod is Pod.
unsafe impl Zeroable for ColorBrick {}
unsafe impl Pod for ColorBrick {}

impl std::fmt::Debug for ColorBrick {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let non_zero = self.data.iter().filter(|v| v.packed != 0).count();
        f.debug_struct("ColorBrick")
            .field("non_zero_voxels", &non_zero)
            .finish()
    }
}

impl Default for ColorBrick {
    fn default() -> Self {
        bytemuck::Zeroable::zeroed()
    }
}

impl ColorBrick {
    /// Flat index from 3D brick coordinates.
    #[inline]
    pub fn index(x: u32, y: u32, z: u32) -> usize {
        brick_index(x, y, z)
    }

    /// Read the voxel at `(x, y, z)`.
    #[inline]
    pub fn sample(&self, x: u32, y: u32, z: u32) -> ColorVoxel {
        self.data[Self::index(x, y, z)]
    }

    /// Write `val` to `(x, y, z)`.
    #[inline]
    pub fn set(&mut self, x: u32, y: u32, z: u32, val: ColorVoxel) {
        self.data[Self::index(x, y, z)] = val;
    }
}

// ---------------------------------------------------------------------------
// BoneBrickLod — per-object bone weight data for one LOD level
// ---------------------------------------------------------------------------

/// Per-object bone weight data for one LOD level.
///
/// Maps brick map slots to [`BoneBrick`] companion data, following the same
/// pattern as the color brick companion pool. Only bricks with surface voxels
/// that have bone influences need an allocated [`BoneBrick`].
#[derive(Debug, Clone)]
pub struct BoneBrickLod {
    /// Brick map dimensions (matches the object's [`BrickMap`] dims for this LOD).
    pub dims: glam::UVec3,

    /// Per-brick-slot companion index. Length = `dims.x * dims.y * dims.z`.
    /// Values index into [`Self::bricks`], or [`EMPTY_SLOT`] for no bone data.
    ///
    /// [`EMPTY_SLOT`]: crate::brick_map::EMPTY_SLOT
    pub companion_map: Vec<u32>,

    /// Bone brick data for allocated slots.
    pub bricks: Vec<BoneBrick>,
}

impl BoneBrickLod {
    /// Create an empty bone brick LOD with the given dimensions.
    ///
    /// All companion map entries are initialized to `EMPTY_SLOT`.
    pub fn new(dims: glam::UVec3) -> Self {
        let total = (dims.x * dims.y * dims.z) as usize;
        Self {
            dims,
            companion_map: vec![crate::brick_map::EMPTY_SLOT; total],
            bricks: Vec::new(),
        }
    }

    /// Allocate a new [`BoneBrick`] for the given brick map slot index.
    ///
    /// Returns the index into [`Self::bricks`]. Panics if `slot_index` is
    /// out of range or already allocated.
    pub fn allocate(&mut self, slot_index: usize) -> usize {
        assert!(
            slot_index < self.companion_map.len(),
            "slot_index {slot_index} out of range (map size {})",
            self.companion_map.len()
        );
        assert_eq!(
            self.companion_map[slot_index],
            crate::brick_map::EMPTY_SLOT,
            "slot {slot_index} already allocated"
        );

        let brick_index = self.bricks.len();
        self.bricks.push(BoneBrick::default());
        self.companion_map[slot_index] = brick_index as u32;
        brick_index
    }

    /// Get the [`BoneBrick`] for a brick map slot, or `None` if unallocated.
    pub fn get(&self, slot_index: usize) -> Option<&BoneBrick> {
        if slot_index >= self.companion_map.len() {
            return None;
        }
        let idx = self.companion_map[slot_index];
        if idx == crate::brick_map::EMPTY_SLOT {
            None
        } else {
            self.bricks.get(idx as usize)
        }
    }

    /// Get a mutable reference to the [`BoneBrick`] for a brick map slot.
    pub fn get_mut(&mut self, slot_index: usize) -> Option<&mut BoneBrick> {
        if slot_index >= self.companion_map.len() {
            return None;
        }
        let idx = self.companion_map[slot_index];
        if idx == crate::brick_map::EMPTY_SLOT {
            None
        } else {
            self.bricks.get_mut(idx as usize)
        }
    }

    /// Number of allocated bone bricks.
    pub fn brick_count(&self) -> usize {
        self.bricks.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::VOXELS_PER_BRICK;
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
}
