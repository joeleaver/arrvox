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

#[cfg(test)]
mod tests;
