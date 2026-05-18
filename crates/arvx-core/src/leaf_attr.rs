//! Per-octree-leaf payload.
//!
//! Every unique (material, normal) tuple in an object gets one [`LeafAttr`]
//! entry. The octree leaf encoding stores a `leaf_attr_id` pointing into a
//! [`LeafAttrPool`](crate::leaf_attr_pool::LeafAttrPool).
//!
//! # Memory layout (8 bytes per entry)
//!
//! ```text
//! word0: normal_oct                 (u32 = 2× snorm16 octahedral)
//! word1: material_primary           (u16)  | material_secondary_blend (u16)
//!          └── low 12 bits = material_secondary_id
//!          └── high 4 bits = blend_weight (0-15)
//! ```
//!
//! Removing opacity from the voxel payload collapsed two indirections. The
//! leaf node directly names its material, its blend partner, and its normal;
//! the shader reads one 8-byte entry and has everything it needs to shade.
//!
//! Transparency is now a per-material property (`mat_opacity`), not a
//! per-voxel one. Spatially-varying transparency is achieved by assigning
//! different material IDs — the palette gives 65k slots, well beyond any
//! real authoring need. If a future volumetric material needs smooth
//! per-position density, it can carry a density field on its own rather
//! than forcing every opaque voxel in the scene to store a redundant f16.

use glam::Vec3;

/// Per-leaf payload — 8 bytes, Pod, GPU-uploadable.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LeafAttr {
    /// Octahedrally-packed unit normal (2× snorm16 in a u32).
    pub normal_oct: u32,
    /// Primary material palette ID.
    pub material_primary: u16,
    /// Packed `(secondary_material_id & 0x0FFF) | (blend_weight << 12)`.
    /// 12 bits = 4096 secondary materials, 4 bits = 16 blend levels.
    pub material_secondary_blend: u16,
}

const SECONDARY_MASK: u16 = 0x0FFF;
const BLEND_SHIFT: u16 = 12;

impl LeafAttr {
    /// Empty — all fields zero. Used for unpopulated pool slots.
    pub const EMPTY: Self = Self {
        normal_oct: 0,
        material_primary: 0,
        material_secondary_blend: 0,
    };

    /// Construct with a single material (no secondary blend).
    pub fn new(normal: Vec3, material_primary: u16) -> Self {
        Self {
            normal_oct: pack_oct(normal),
            material_primary,
            material_secondary_blend: 0,
        }
    }

    /// Construct with primary + secondary material + blend weight.
    ///
    /// `material_secondary` is clamped to 12 bits; `blend_weight` is clamped
    /// to 4 bits. Call sites that need the full u16 secondary or u8 blend
    /// precision should reconsider whether they truly need >4096 palette
    /// entries or >16 blend levels — the original u16/u8 was overkill.
    pub fn new_blended(
        normal: Vec3,
        material_primary: u16,
        material_secondary: u16,
        blend_weight: u8,
    ) -> Self {
        let secondary_bits = material_secondary & SECONDARY_MASK;
        let blend_bits = ((blend_weight as u16) & 0x0F) << BLEND_SHIFT;
        Self {
            normal_oct: pack_oct(normal),
            material_primary,
            material_secondary_blend: secondary_bits | blend_bits,
        }
    }

    /// Decode the normal back into a unit vector.
    #[inline]
    pub fn normal(self) -> Vec3 { unpack_oct(self.normal_oct) }

    /// Secondary material ID (0..4095).
    #[inline]
    pub fn material_secondary(self) -> u16 {
        self.material_secondary_blend & SECONDARY_MASK
    }

    /// Blend weight (0..15). 0 = primary only, 15 = fully secondary.
    #[inline]
    pub fn blend_weight(self) -> u8 {
        ((self.material_secondary_blend >> BLEND_SHIFT) & 0x0F) as u8
    }
}

impl Default for LeafAttr {
    fn default() -> Self { Self::EMPTY }
}

// Octahedral packing -----------------------------------------------------------
//
// 16-bit-per-axis snorm, packed into a single u32. Worst-case angular
// roundtrip error <0.05°, which is imperceptible in shading.

/// Pack a unit normal as octahedral 2× snorm16 in a u32.
pub fn pack_oct(n: Vec3) -> u32 {
    let (u, v) = oct_project(n);
    let ui = quantize_snorm(u, 16);
    let vi = quantize_snorm(v, 16);
    (ui as u32 & 0xFFFF) | ((vi as u32 & 0xFFFF) << 16)
}

/// Unpack a u32 that was produced by [`pack_oct`].
pub fn unpack_oct(packed: u32) -> Vec3 {
    let ui_raw = (packed & 0xFFFF) as i16;
    let vi_raw = ((packed >> 16) & 0xFFFF) as i16;
    let u = (ui_raw as f32 / 32767.0).clamp(-1.0, 1.0);
    let v = (vi_raw as f32 / 32767.0).clamp(-1.0, 1.0);
    oct_reconstruct(u, v)
}

#[inline]
fn oct_project(n: Vec3) -> (f32, f32) {
    let len = n.length();
    let n = if len > 1e-8 { n / len } else { Vec3::Y };

    let abs_sum = n.x.abs() + n.y.abs() + n.z.abs();
    let inv = if abs_sum > 1e-8 { 1.0 / abs_sum } else { 0.0 };
    let mut u = n.x * inv;
    let mut v = n.y * inv;
    if n.z < 0.0 {
        let u0 = u;
        u = (1.0 - v.abs()) * sign_nonzero(u0);
        v = (1.0 - u0.abs()) * sign_nonzero(v);
    }
    (u, v)
}

#[inline]
fn oct_reconstruct(u: f32, v: f32) -> Vec3 {
    let mut n = Vec3::new(u, v, 1.0 - u.abs() - v.abs());
    if n.z < 0.0 {
        let nx0 = n.x;
        n.x = (1.0 - n.y.abs()) * sign_nonzero(nx0);
        n.y = (1.0 - nx0.abs()) * sign_nonzero(n.y);
    }
    let len = n.length();
    if len > 1e-8 { n / len } else { Vec3::Y }
}

#[inline]
fn sign_nonzero(x: f32) -> f32 {
    if x >= 0.0 { 1.0 } else { -1.0 }
}

#[inline]
fn quantize_snorm(x: f32, bits: u32) -> i32 {
    let max = ((1i32 << (bits - 1)) - 1) as f32;
    (x.clamp(-1.0, 1.0) * max).round() as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_roundtrip(n: Vec3, tol_deg: f32) {
        let n = n.normalize();
        let back = unpack_oct(pack_oct(n));
        let dot = n.dot(back).clamp(-1.0, 1.0);
        let err = dot.acos().to_degrees();
        assert!(err < tol_deg, "normal {n:?} → {back:?}, err {err:.4}° (tol {tol_deg}°)");
    }

    #[test]
    fn leaf_attr_is_8_bytes() {
        assert_eq!(std::mem::size_of::<LeafAttr>(), 8);
        assert_eq!(std::mem::align_of::<LeafAttr>(), 4);
    }

    #[test]
    fn new_stores_material_and_normal() {
        let a = LeafAttr::new(Vec3::Y, 42);
        assert_eq!(a.material_primary, 42);
        assert_eq!(a.material_secondary(), 0);
        assert_eq!(a.blend_weight(), 0);
        assert!(a.normal().dot(Vec3::Y) > 0.9999);
    }

    #[test]
    fn new_blended_packs_secondary_and_weight() {
        let a = LeafAttr::new_blended(Vec3::X, 7, 1234, 9);
        assert_eq!(a.material_primary, 7);
        assert_eq!(a.material_secondary(), 1234);
        assert_eq!(a.blend_weight(), 9);
    }

    #[test]
    fn new_blended_clamps_to_12_bits_and_4_bits() {
        // Secondary > 4095 is clamped to 12 bits; blend > 15 is clamped to 4.
        let a = LeafAttr::new_blended(Vec3::X, 0, 0xFFFF, 0xFF);
        assert_eq!(a.material_secondary(), 0xFFF);
        assert_eq!(a.blend_weight(), 0x0F);
    }

    #[test]
    fn axis_aligned_normals_roundtrip() {
        for n in [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z] {
            check_roundtrip(n, 0.01);
        }
    }

    #[test]
    fn spherical_sweep_error_bounded() {
        let mut worst = 0.0_f32;
        for theta_deg in (0..=180).step_by(10) {
            for phi_deg in (0..=360).step_by(10) {
                let theta = (theta_deg as f32).to_radians();
                let phi = (phi_deg as f32).to_radians();
                let n = Vec3::new(
                    theta.sin() * phi.cos(),
                    theta.sin() * phi.sin(),
                    theta.cos(),
                ).normalize();
                let back = unpack_oct(pack_oct(n));
                let dot = n.dot(back).clamp(-1.0, 1.0);
                let err = dot.acos().to_degrees();
                if err > worst { worst = err; }
            }
        }
        assert!(worst < 0.05, "worst roundtrip was {worst:.4}°");
    }

    #[test]
    fn empty_is_zero() {
        assert_eq!(LeafAttr::EMPTY.normal_oct, 0);
        assert_eq!(LeafAttr::EMPTY.material_primary, 0);
        assert_eq!(LeafAttr::EMPTY.material_secondary_blend, 0);
    }

    #[test]
    fn zero_vector_is_safe() {
        let a = LeafAttr::new(Vec3::ZERO, 0);
        assert!(a.normal().length() > 0.5);
    }
}
