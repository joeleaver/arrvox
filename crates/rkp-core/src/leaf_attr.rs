//! Per-octree-leaf attribute payload.
//!
//! Every unique (opacity, material, normal) tuple in the scene gets one
//! [`LeafAttr`] entry. The octree leaf encoding stores a `leaf_attr_id`
//! pointing into a [`LeafAttrPool`](crate::leaf_attr_pool::LeafAttrPool) —
//! replacing the previous scheme where the leaf pointed directly at a
//! [`VoxelPool`](crate::voxel_pool::VoxelPool) slot.
//!
//! Why the indirection: the `voxel_pool` dedups by (opacity, material) value
//! across the whole object. Two spatial leaves with identical voxel values
//! share a pool slot. Surface normals vary *per-spatial-position*, so they
//! can't live alongside the voxel without breaking that dedup. `LeafAttr`
//! carries the per-position normal next to a back-reference to the voxel.
//!
//! Dedup applies at the leaf-attr level too: leaves with identical
//! (voxel_slot, normal) share a leaf_attr_id. Flat surfaces collapse
//! naturally; curved surfaces like spheres get one entry per leaf.

use glam::Vec3;

/// Per-leaf attribute: a back-reference to the voxel data plus a packed
/// surface normal. 8 bytes, Pod.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LeafAttr {
    /// Slot in the object's VoxelPool that holds this leaf's opacity+material.
    pub voxel_slot: u32,
    /// Octahedral-packed unit normal (2× snorm16 in a u32). See [`pack_oct`].
    pub normal_oct: u32,
}

impl LeafAttr {
    /// Empty — used for unpopulated slots.
    pub const EMPTY: Self = Self { voxel_slot: 0, normal_oct: 0 };

    /// Construct from a voxel slot and a world-space unit normal.
    pub fn new(voxel_slot: u32, normal: Vec3) -> Self {
        Self { voxel_slot, normal_oct: pack_oct(normal) }
    }
}

impl Default for LeafAttr {
    fn default() -> Self { Self::EMPTY }
}

/// Pack a unit normal as octahedral 2× snorm16 in a single u32.
///
/// Octahedral mapping preserves great-circle uniformity far better than naive
/// spherical coordinates and survives a 16-bit-per-axis quantization with
/// worst-case angular error well under 0.1°. Reference: Meyer et al. 2010,
/// "On Floating-Point Normal Vectors", and Cigolle et al. 2014, "A Survey of
/// Efficient Representations for Independent Unit Vectors".
pub fn pack_oct(n: Vec3) -> u32 {
    // Safeguard against a zero vector — return up.
    let len = n.length();
    let n = if len > 1e-8 { n / len } else { Vec3::Y };

    // Project onto the octahedron (manhattan normalize).
    let abs_sum = n.x.abs() + n.y.abs() + n.z.abs();
    let inv = if abs_sum > 1e-8 { 1.0 / abs_sum } else { 0.0 };
    let mut u = n.x * inv;
    let mut v = n.y * inv;

    // If the original normal was on the lower hemisphere (z < 0), fold the
    // octahedron's lower half onto its upper half via a sign-dependent flip.
    if n.z < 0.0 {
        let u0 = u;
        u = (1.0 - v.abs()) * sign_nonzero(u0);
        v = (1.0 - u0.abs()) * sign_nonzero(v);
    }

    // snorm16 quantization: map [-1, 1] to [-32767, 32767], store as u16.
    let ui = quantize_snorm16(u);
    let vi = quantize_snorm16(v);
    (ui as u32) | ((vi as u32) << 16)
}

/// Unpack an octahedrally-encoded normal back into a unit vector.
pub fn unpack_oct(packed: u32) -> Vec3 {
    let ui = (packed & 0xFFFF) as i16;
    let vi = ((packed >> 16) & 0xFFFF) as i16;
    let u = (ui as f32 / 32767.0).clamp(-1.0, 1.0);
    let v = (vi as f32 / 32767.0).clamp(-1.0, 1.0);

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
fn quantize_snorm16(x: f32) -> u16 {
    let clamped = x.clamp(-1.0, 1.0);
    let scaled = (clamped * 32767.0).round();
    (scaled as i32 as i16) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_roundtrip(n: Vec3, tol_deg: f32) {
        let n = n.normalize();
        let packed = pack_oct(n);
        let unpacked = unpack_oct(packed);
        let dot = n.dot(unpacked).clamp(-1.0, 1.0);
        let angle_deg = dot.acos().to_degrees();
        assert!(
            angle_deg < tol_deg,
            "normal {n:?} roundtripped to {unpacked:?}, error {angle_deg:.4}° (tol {tol_deg}°)",
        );
    }

    #[test]
    fn layout_is_8_bytes() {
        assert_eq!(std::mem::size_of::<LeafAttr>(), 8);
        assert_eq!(std::mem::align_of::<LeafAttr>(), 4);
    }

    #[test]
    fn axis_aligned_normals_roundtrip() {
        // Axis-aligned sit on quantization grid points; accuracy is near-exact.
        for n in [
            Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z,
        ] {
            check_roundtrip(n, 0.01);
        }
    }

    #[test]
    fn diagonal_normals_roundtrip() {
        // Worst-case accuracy sits at octahedral edges/corners where the
        // manhattan-distance projection crowds samples; snorm16 gives us
        // ≤0.05° residual there.
        for n in [
            Vec3::new( 1.0,  1.0,  1.0),
            Vec3::new(-1.0,  1.0,  1.0),
            Vec3::new( 1.0, -1.0,  1.0),
            Vec3::new( 1.0,  1.0, -1.0),
            Vec3::new(-1.0, -1.0, -1.0),
        ] {
            check_roundtrip(n, 0.05);
        }
    }

    #[test]
    fn spherical_sweep_error_bounded() {
        // Sweep a hemisphere at 10° increments. Worst-case angular error on
        // octahedral-snorm16 should sit well under 0.01°; pick 0.05° for slack.
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
        assert!(worst < 0.05, "worst-case octahedral roundtrip error was {worst:.4}°");
    }

    #[test]
    fn zero_vector_is_safe() {
        let n = Vec3::ZERO;
        let packed = pack_oct(n);
        let unpacked = unpack_oct(packed);
        assert!(unpacked.length() > 0.5, "fallback should produce a unit vector, got {unpacked:?}");
    }

    #[test]
    fn leaf_attr_new_packs_normal() {
        let a = LeafAttr::new(42, Vec3::Y);
        assert_eq!(a.voxel_slot, 42);
        let back = unpack_oct(a.normal_oct);
        assert!(back.dot(Vec3::Y) > 0.9999, "Y roundtrip lost precision: {back:?}");
    }

    #[test]
    fn leaf_attr_is_pod_zero_inited() {
        let z = LeafAttr::default();
        assert_eq!(z.voxel_slot, 0);
        assert_eq!(z.normal_oct, 0);
    }
}
