//! World-space position type for float-precision-safe coordinates.
//!
//! [`WorldPosition`] splits a position into a chunk index (`IVec3`) and a
//! local offset within that chunk (`Vec3`).  The chunk size is 8 metres.
//! This avoids the catastrophic cancellation that would occur if two distant
//! positions were subtracted as raw `f32` values.
//!
//! # Example
//! ```
//! use rkp_core::WorldPosition;
//! use glam::{IVec3, Vec3};
//!
//! let a = WorldPosition::new(IVec3::ZERO, Vec3::new(1.0, 2.0, 3.0));
//! let b = WorldPosition::new(IVec3::new(1, 0, 0), Vec3::new(0.5, 2.0, 3.0));
//! // relative_to uses f64 arithmetic internally
//! let disp = b.relative_to(&a);
//! assert!((disp.x - 7.5_f32).abs() < 1e-5);
//! ```

use std::ops::{Add, AddAssign, Sub};

use glam::{DVec3, IVec3, Vec3};

/// Size of one chunk in metres.
pub const CHUNK_SIZE: f32 = 8.0;
const CHUNK_SIZE_F64: f64 = CHUNK_SIZE as f64;

/// A world-space position split into an integer chunk index and a local
/// `f32` offset within that chunk.
///
/// The invariant maintained by [`WorldPosition::normalize`] is:
/// `0.0 <= local.x < CHUNK_SIZE` (and same for y, z).
///
/// Use [`WorldPosition::relative_to`] instead of raw subtraction to avoid
/// `f32` precision loss at large distances.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WorldPosition {
    /// Integer chunk coordinates.  Each unit represents one 8-metre chunk.
    pub chunk: IVec3,
    /// Sub-chunk offset in metres.  Invariant: each component is in `[0, 8)`.
    pub local: Vec3,
}

impl WorldPosition {
    /// Construct a new `WorldPosition`, normalizing the local offset so it
    /// falls within `[0, CHUNK_SIZE)`.
    #[inline]
    pub fn new(chunk: IVec3, local: Vec3) -> Self {
        let mut pos = Self { chunk, local };
        pos.normalize();
        pos
    }

    /// Re-centre `local` into `[0, CHUNK_SIZE)`, adjusting `chunk` as needed.
    ///
    /// This handles both overflow (local >= CHUNK_SIZE) and underflow
    /// (local < 0.0) on each axis independently.
    pub fn normalize(&mut self) {
        // Use floor-division semantics so that e.g. local = -0.1 maps to
        // chunk -= 1, local = 7.9, not local = 0.0.
        let chunk_offset_x = self.local.x.div_euclid(CHUNK_SIZE).floor() as i32;
        let chunk_offset_y = self.local.y.div_euclid(CHUNK_SIZE).floor() as i32;
        let chunk_offset_z = self.local.z.div_euclid(CHUNK_SIZE).floor() as i32;

        self.chunk.x += chunk_offset_x;
        self.chunk.y += chunk_offset_y;
        self.chunk.z += chunk_offset_z;

        self.local.x = self.local.x.rem_euclid(CHUNK_SIZE);
        self.local.y = self.local.y.rem_euclid(CHUNK_SIZE);
        self.local.z = self.local.z.rem_euclid(CHUNK_SIZE);
    }

    /// Return a normalized copy of this position.
    #[inline]
    pub fn normalized(mut self) -> Self {
        self.normalize();
        self
    }

    /// Return the absolute world position as a plain `Vec3`.
    ///
    /// This is a convenience for editor-scale scenes where all objects are near
    /// the origin and `f32` precision is sufficient.  For large-world usage,
    /// prefer [`relative_to`](Self::relative_to) with the camera position.
    #[inline]
    pub fn to_vec3(&self) -> Vec3 {
        Vec3::new(
            self.chunk.x as f32 * CHUNK_SIZE + self.local.x,
            self.chunk.y as f32 * CHUNK_SIZE + self.local.y,
            self.chunk.z as f32 * CHUNK_SIZE + self.local.z,
        )
    }

    /// Compute the displacement vector `self - origin` using `f64` arithmetic
    /// to avoid precision loss at large distances.
    ///
    /// The result is cast to `f32` only at the end.  Callers that need full
    /// `f64` precision for further calculations should use
    /// [`WorldPosition::relative_to_f64`] instead.
    pub fn relative_to(&self, origin: &WorldPosition) -> Vec3 {
        self.relative_to_f64(origin).as_vec3()
    }

    /// Like [`relative_to`](Self::relative_to) but returns a `DVec3` (`f64`)
    /// for callers that need maximum precision.
    pub fn relative_to_f64(&self, origin: &WorldPosition) -> DVec3 {
        let chunk_delta = self.chunk - origin.chunk;
        let chunk_metres = DVec3::new(
            chunk_delta.x as f64 * CHUNK_SIZE_F64,
            chunk_delta.y as f64 * CHUNK_SIZE_F64,
            chunk_delta.z as f64 * CHUNK_SIZE_F64,
        );
        let local_delta = DVec3::new(
            (self.local.x - origin.local.x) as f64,
            (self.local.y - origin.local.y) as f64,
            (self.local.z - origin.local.z) as f64,
        );
        chunk_metres + local_delta
    }

    /// Euclidean distance between two positions, computed in `f64`.
    pub fn distance_f64(&self, other: &WorldPosition) -> f64 {
        self.relative_to_f64(other).length()
    }

    /// Return the position translated by `offset`, normalized.
    pub fn translate(&self, offset: Vec3) -> WorldPosition {
        WorldPosition::new(self.chunk, self.local + offset)
    }

    /// Convert absolute world coordinates (in metres, `f64`) to a
    /// `WorldPosition`.
    pub fn from_world_f64(x: f64, y: f64, z: f64) -> WorldPosition {
        // Compute chunk using floor division so negative coordinates work.
        let cx = (x / CHUNK_SIZE_F64).floor() as i32;
        let cy = (y / CHUNK_SIZE_F64).floor() as i32;
        let cz = (z / CHUNK_SIZE_F64).floor() as i32;

        let lx = (x - cx as f64 * CHUNK_SIZE_F64) as f32;
        let ly = (y - cy as f64 * CHUNK_SIZE_F64) as f32;
        let lz = (z - cz as f64 * CHUNK_SIZE_F64) as f32;

        // Clamp tiny floating-point residuals that could push local just
        // outside [0, CHUNK_SIZE).
        let local = Vec3::new(
            lx.clamp(0.0, CHUNK_SIZE - f32::EPSILON),
            ly.clamp(0.0, CHUNK_SIZE - f32::EPSILON),
            lz.clamp(0.0, CHUNK_SIZE - f32::EPSILON),
        );

        WorldPosition {
            chunk: IVec3::new(cx, cy, cz),
            local,
        }
    }
}

impl Default for WorldPosition {
    fn default() -> Self {
        Self {
            chunk: IVec3::ZERO,
            local: Vec3::ZERO,
        }
    }
}

// ─── operator impls ─────────────────────────────────────────────────────────

impl From<Vec3> for WorldPosition {
    /// Convert a plain `Vec3` (absolute metres) into a `WorldPosition` by
    /// computing the chunk from floor-division and keeping the remainder as local.
    #[inline]
    fn from(v: Vec3) -> Self {
        WorldPosition::new(IVec3::ZERO, v)
    }
}

impl From<DVec3> for WorldPosition {
    /// Convert a `DVec3` (absolute metres, `f64`) into a `WorldPosition`,
    /// preserving precision by computing chunk via floor-division in f64.
    #[inline]
    fn from(v: DVec3) -> Self {
        WorldPosition::from_world_f64(v.x, v.y, v.z)
    }
}

impl Add<Vec3> for WorldPosition {
    type Output = WorldPosition;

    /// Offset a world position by a `Vec3` displacement (in metres).
    #[inline]
    fn add(self, rhs: Vec3) -> WorldPosition {
        self.translate(rhs)
    }
}

impl AddAssign<Vec3> for WorldPosition {
    /// In-place offset by a `Vec3` displacement.
    #[inline]
    fn add_assign(&mut self, rhs: Vec3) {
        self.local += rhs;
        self.normalize();
    }
}

impl Sub for WorldPosition {
    type Output = Vec3;

    /// Compute the displacement `self - rhs` as a `Vec3`, using `f64`
    /// arithmetic internally to avoid precision loss at large distances.
    ///
    /// The result is cast to `f32` at the end — this is fine because
    /// displacement vectors between interacting objects are typically small.
    #[inline]
    fn sub(self, rhs: WorldPosition) -> Vec3 {
        self.relative_to(&rhs)
    }
}

// ─── tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{IVec3, Vec3};

    // ── normalization ────────────────────────────────────────────────────────

    /// local (9.0, -1.0, 16.5) relative to chunk (0,0,0) should normalise to
    /// chunk (1, -1, 2), local (1.0, 7.0, 0.5).
    #[test]
    fn normalize_overflow_and_underflow() {
        let pos = WorldPosition::new(IVec3::ZERO, Vec3::new(9.0, -1.0, 16.5));
        assert_eq!(pos.chunk, IVec3::new(1, -1, 2));
        // Use approximate equality for f32.
        assert!((pos.local.x - 1.0).abs() < 1e-5, "local.x = {}", pos.local.x);
        assert!((pos.local.y - 7.0).abs() < 1e-5, "local.y = {}", pos.local.y);
        assert!((pos.local.z - 0.5).abs() < 1e-5, "local.z = {}", pos.local.z);
    }

    /// Exactly at a chunk boundary: local.x == 8.0 should become
    /// chunk.x += 1, local.x == 0.0.
    #[test]
    fn normalize_exact_boundary() {
        let pos = WorldPosition::new(IVec3::ZERO, Vec3::new(8.0, 0.0, 0.0));
        assert_eq!(pos.chunk, IVec3::new(1, 0, 0));
        assert!((pos.local.x).abs() < 1e-5);
    }

    /// Very large local values: multiple-chunk overflow.
    #[test]
    fn normalize_large_overflow() {
        // 40.0 / 8.0 = 5 chunks
        let pos = WorldPosition::new(IVec3::ZERO, Vec3::new(40.0, 40.0, 40.0));
        assert_eq!(pos.chunk, IVec3::new(5, 5, 5));
        assert!((pos.local.x).abs() < 1e-5);
    }

    /// `normalized()` returns the same result as `normalize()`.
    #[test]
    fn normalized_consuming_matches_mutating() {
        let raw = WorldPosition {
            chunk: IVec3::ZERO,
            local: Vec3::new(9.0, -1.0, 16.5),
        };
        let mut mutated = raw;
        mutated.normalize();
        let consumed = raw.normalized();
        assert_eq!(mutated, consumed);
    }

    // ── relative_to / precision ──────────────────────────────────────────────

    /// Positions billions of metres apart should give a correct displacement.
    #[test]
    fn relative_to_large_distance() {
        // 1 billion metres along x.
        let billion: i32 = 1_000_000_000 / 8; // chunks
        let a = WorldPosition::new(IVec3::ZERO, Vec3::ZERO);
        let b = WorldPosition::new(IVec3::new(billion, 0, 0), Vec3::new(3.0, 0.0, 0.0));

        let disp = b.relative_to(&a);
        // expected: billion * 8 + 3.0 metres — but f32 can't represent that
        // exactly. The f64 intermediate must be correct; the final cast to f32
        // will saturate. What we test here is that the f64 path is correct.
        let disp_f64 = b.relative_to_f64(&a);
        let expected_f64 = billion as f64 * 8.0 + 3.0;
        assert!(
            (disp_f64.x - expected_f64).abs() < 1.0,
            "f64 displacement error: {} vs {}",
            disp_f64.x,
            expected_f64
        );
        // The f32 result should at least be finite.
        assert!(disp.x.is_finite());
    }

    /// relative_to symmetry: a.relative_to(b) ≈ -b.relative_to(a).
    #[test]
    fn relative_to_symmetry() {
        let a = WorldPosition::new(IVec3::new(3, 7, -2), Vec3::new(1.5, 6.0, 0.25));
        let b = WorldPosition::new(IVec3::new(-1, 4, 5), Vec3::new(4.0, 2.5, 7.75));

        let ab = a.relative_to_f64(&b);
        let ba = b.relative_to_f64(&a);

        assert!(
            (ab.x + ba.x).abs() < 1e-9,
            "symmetry x: {} + {} = {}",
            ab.x,
            ba.x,
            ab.x + ba.x
        );
        assert!((ab.y + ba.y).abs() < 1e-9);
        assert!((ab.z + ba.z).abs() < 1e-9);
    }

    /// Simple known displacement test.
    #[test]
    fn relative_to_simple() {
        let a = WorldPosition::new(IVec3::ZERO, Vec3::new(1.0, 1.0, 1.0));
        let b = WorldPosition::new(IVec3::new(1, 0, 0), Vec3::new(0.5, 1.0, 1.0));
        // b - a => (1 chunk + 0.5 - 1.0) = 8.0 - 0.5 = 7.5 on x axis
        let disp = b.relative_to(&a);
        assert!((disp.x - 7.5).abs() < 1e-5, "disp.x = {}", disp.x);
        assert!((disp.y).abs() < 1e-5, "disp.y = {}", disp.y);
        assert!((disp.z).abs() < 1e-5, "disp.z = {}", disp.z);
    }

    // ── distance_f64 ─────────────────────────────────────────────────────────

    #[test]
    fn distance_f64_same_point() {
        let a = WorldPosition::new(IVec3::new(5, -3, 0), Vec3::new(2.0, 4.0, 6.0));
        assert!(a.distance_f64(&a) < 1e-9);
    }

    #[test]
    fn distance_f64_known_value() {
        let a = WorldPosition::new(IVec3::ZERO, Vec3::ZERO);
        let b = WorldPosition::new(IVec3::ZERO, Vec3::new(3.0, 4.0, 0.0));
        // 3-4-5 triangle → distance = 5.0
        assert!((a.distance_f64(&b) - 5.0).abs() < 1e-6);
    }

    // ── translate ────────────────────────────────────────────────────────────

    #[test]
    fn translate_stays_normalized() {
        let pos = WorldPosition::new(IVec3::ZERO, Vec3::new(6.0, 6.0, 6.0));
        let moved = pos.translate(Vec3::new(4.0, 4.0, 4.0));
        // 6 + 4 = 10 → chunk += 1, local = 2
        assert_eq!(moved.chunk, IVec3::new(1, 1, 1));
        assert!((moved.local.x - 2.0).abs() < 1e-5);
    }

    #[test]
    fn translate_negative_offset() {
        let pos = WorldPosition::new(IVec3::new(1, 0, 0), Vec3::new(1.0, 0.0, 0.0));
        let moved = pos.translate(Vec3::new(-2.0, 0.0, 0.0));
        // chunk 1, local 1 - 2 = -1 → chunk 0, local 7
        assert_eq!(moved.chunk, IVec3::new(0, 0, 0));
        assert!((moved.local.x - 7.0).abs() < 1e-5);
    }

    // ── from_world_f64 ───────────────────────────────────────────────────────

    #[test]
    fn from_world_f64_positive() {
        // 20.5 metres = 2 chunks + 4.5 metres
        let pos = WorldPosition::from_world_f64(20.5, 0.0, 0.0);
        assert_eq!(pos.chunk.x, 2);
        assert!((pos.local.x - 4.5).abs() < 1e-4, "local.x = {}", pos.local.x);
    }

    #[test]
    fn from_world_f64_negative() {
        // -1.0 metres = chunk -1, local 7.0
        let pos = WorldPosition::from_world_f64(-1.0, 0.0, 0.0);
        assert_eq!(pos.chunk.x, -1);
        assert!((pos.local.x - 7.0).abs() < 1e-4, "local.x = {}", pos.local.x);
    }

    #[test]
    fn from_world_f64_origin() {
        let pos = WorldPosition::from_world_f64(0.0, 0.0, 0.0);
        assert_eq!(pos.chunk, IVec3::ZERO);
        assert!(pos.local.length() < 1e-5);
    }

    /// Roundtrip: convert to f64 coords and back, should recover the original.
    #[test]
    fn from_world_f64_roundtrip() {
        let original = WorldPosition::new(IVec3::new(7, -3, 12), Vec3::new(3.75, 1.25, 6.5));
        // Convert to absolute f64
        let ax = original.chunk.x as f64 * CHUNK_SIZE_F64 + original.local.x as f64;
        let ay = original.chunk.y as f64 * CHUNK_SIZE_F64 + original.local.y as f64;
        let az = original.chunk.z as f64 * CHUNK_SIZE_F64 + original.local.z as f64;
        let recovered = WorldPosition::from_world_f64(ax, ay, az);

        assert_eq!(recovered.chunk, original.chunk);
        assert!(
            (recovered.local.x - original.local.x).abs() < 1e-4,
            "local.x: {} vs {}",
            recovered.local.x,
            original.local.x
        );
        assert!((recovered.local.y - original.local.y).abs() < 1e-4);
        assert!((recovered.local.z - original.local.z).abs() < 1e-4);
    }

    // ── serde roundtrip ──────────────────────────────────────────────────────

    #[test]
    fn serde_roundtrip() {
        let pos = WorldPosition::new(IVec3::new(1, -2, 3), Vec3::new(4.5, 0.1, 7.9));
        let json = serde_json::to_string(&pos).expect("serialize");
        let recovered: WorldPosition = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(pos, recovered);
    }

    // ── default ──────────────────────────────────────────────────────────────

    #[test]
    fn default_is_origin() {
        let pos = WorldPosition::default();
        assert_eq!(pos.chunk, IVec3::ZERO);
        assert_eq!(pos.local, Vec3::ZERO);
    }

    // ── From<Vec3> ────────────────────────────────────────────────────────

    #[test]
    fn from_vec3_within_one_chunk() {
        let v = Vec3::new(3.0, 5.0, 7.0);
        let pos: WorldPosition = v.into();
        assert_eq!(pos.chunk, IVec3::ZERO);
        assert!((pos.local.x - 3.0).abs() < 1e-5);
        assert!((pos.local.y - 5.0).abs() < 1e-5);
        assert!((pos.local.z - 7.0).abs() < 1e-5);
    }

    #[test]
    fn from_vec3_spanning_multiple_chunks() {
        // 25.0 / 8.0 = 3 chunks + 1.0 remainder
        let v = Vec3::new(25.0, 50.0, 100.0);
        let pos: WorldPosition = v.into();
        assert_eq!(pos.chunk.x, 3);
        assert!((pos.local.x - 1.0).abs() < 1e-4);
        assert_eq!(pos.chunk.y, 6);  // 50/8 = 6.25 → chunk 6, local 2.0
        assert!((pos.local.y - 2.0).abs() < 1e-4);
        assert_eq!(pos.chunk.z, 12); // 100/8 = 12.5 → chunk 12, local 4.0
        assert!((pos.local.z - 4.0).abs() < 1e-4);
    }

    #[test]
    fn from_vec3_negative_values() {
        // -5.0 → chunk -1, local 3.0 (since -5 = -1*8 + 3)
        let v = Vec3::new(-5.0, -0.5, -16.0);
        let pos: WorldPosition = v.into();
        assert_eq!(pos.chunk.x, -1);
        assert!((pos.local.x - 3.0).abs() < 1e-4);
        assert_eq!(pos.chunk.y, -1);
        assert!((pos.local.y - 7.5).abs() < 1e-4);
        assert_eq!(pos.chunk.z, -2); // -16/8 = -2, local 0
        assert!(pos.local.z.abs() < 1e-4);
    }

    // ── Add<Vec3> ─────────────────────────────────────────────────────────

    #[test]
    fn add_vec3_within_chunk() {
        let pos = WorldPosition::new(IVec3::ZERO, Vec3::new(1.0, 2.0, 3.0));
        let result = pos + Vec3::new(2.0, 1.0, 0.5);
        assert_eq!(result.chunk, IVec3::ZERO);
        assert!((result.local.x - 3.0).abs() < 1e-5);
        assert!((result.local.y - 3.0).abs() < 1e-5);
        assert!((result.local.z - 3.5).abs() < 1e-5);
    }

    #[test]
    fn add_vec3_across_chunk_boundary() {
        let pos = WorldPosition::new(IVec3::ZERO, Vec3::new(6.0, 6.0, 6.0));
        let result = pos + Vec3::new(4.0, 4.0, 4.0);
        // 6+4 = 10 → chunk 1, local 2
        assert_eq!(result.chunk, IVec3::new(1, 1, 1));
        assert!((result.local.x - 2.0).abs() < 1e-5);
        assert!((result.local.y - 2.0).abs() < 1e-5);
        assert!((result.local.z - 2.0).abs() < 1e-5);
    }

    // ── Sub ───────────────────────────────────────────────────────────────

    #[test]
    fn sub_same_chunk() {
        let a = WorldPosition::new(IVec3::new(5, 5, 5), Vec3::new(6.0, 3.0, 1.0));
        let b = WorldPosition::new(IVec3::new(5, 5, 5), Vec3::new(2.0, 1.0, 0.5));
        let diff = a - b;
        assert!((diff.x - 4.0).abs() < 1e-5);
        assert!((diff.y - 2.0).abs() < 1e-5);
        assert!((diff.z - 0.5).abs() < 1e-5);
    }

    #[test]
    fn sub_far_apart_via_f64() {
        // Two positions ~8 million metres apart
        let a = WorldPosition::new(IVec3::new(1_000_000, 0, 0), Vec3::new(3.0, 0.0, 0.0));
        let b = WorldPosition::new(IVec3::ZERO, Vec3::new(1.0, 0.0, 0.0));
        let diff = a - b;
        // Expected: 1_000_000 * 8 + 3 - 1 = 8_000_002
        // f32 can represent this approximately
        let expected = 8_000_002.0_f32;
        assert!(
            (diff.x - expected).abs() < 1.0,
            "diff.x = {}, expected = {}",
            diff.x,
            expected
        );
    }

    // ── AddAssign<Vec3> ───────────────────────────────────────────────────

    #[test]
    fn add_assign_matches_add() {
        let pos = WorldPosition::new(IVec3::new(3, -1, 7), Vec3::new(5.5, 2.0, 6.5));
        let offset = Vec3::new(4.0, -3.0, 10.0);

        let added = pos + offset;
        let mut assigned = pos;
        assigned += offset;

        assert_eq!(added.chunk, assigned.chunk);
        assert!((added.local.x - assigned.local.x).abs() < 1e-5);
        assert!((added.local.y - assigned.local.y).abs() < 1e-5);
        assert!((added.local.z - assigned.local.z).abs() < 1e-5);
    }

    // ── From<DVec3> ─────────────────────────────────────────────────────

    #[test]
    fn from_dvec3_normalizes_large_values() {
        // 1_000_000.5 metres on x → chunk 125000, local 0.5
        let v = DVec3::new(1_000_000.5, -20.25, 0.0);
        let pos: WorldPosition = v.into();
        assert_eq!(pos.chunk.x, 125000);
        assert!((pos.local.x - 0.5).abs() < 1e-3, "local.x = {}", pos.local.x);
        assert_eq!(pos.chunk.y, -3); // -20.25 / 8 = -2.53 → floor = -3, local = 3.75
        assert!((pos.local.y - 3.75).abs() < 1e-3, "local.y = {}", pos.local.y);
        assert_eq!(pos.chunk.z, 0);
        assert!(pos.local.z.abs() < 1e-5);
    }

    #[test]
    fn from_dvec3_precision_preserved() {
        // A position that would lose precision as f32: 10_000_000.123
        let v = DVec3::new(10_000_000.123, 0.0, 0.0);
        let pos: WorldPosition = v.into();
        // chunk = floor(10_000_000.123 / 8) = 1_250_000, local = 0.123
        assert_eq!(pos.chunk.x, 1_250_000);
        assert!((pos.local.x - 0.123).abs() < 1e-3, "local.x = {}", pos.local.x);
    }

    // ── precision at large distances ──────────────────────────────────────

    #[test]
    fn sub_precision_at_large_distance() {
        // Two positions very far apart, differing by a small amount
        let far_chunk = 100_000_000; // ~800 million metres
        let a = WorldPosition::new(IVec3::new(far_chunk, 0, 0), Vec3::new(1.0, 0.0, 0.0));
        let b = WorldPosition::new(IVec3::new(far_chunk, 0, 0), Vec3::new(1.5, 0.0, 0.0));
        let diff = b - a;
        // Should be exactly 0.5 despite being 800M metres from origin
        assert!(
            (diff.x - 0.5).abs() < 1e-5,
            "precision lost: diff.x = {}, expected 0.5",
            diff.x
        );
    }

    // ── roundtrip ─────────────────────────────────────────────────────────

    #[test]
    fn roundtrip_vec3_sub_add() {
        let origin = WorldPosition::new(IVec3::new(10, 20, 30), Vec3::new(1.0, 2.0, 3.0));
        let target = WorldPosition::new(IVec3::new(10, 20, 30), Vec3::new(5.0, 6.0, 7.0));

        // Get displacement
        let disp = target - origin;
        // Apply displacement back to origin
        let recovered = origin + disp;

        assert_eq!(recovered.chunk, target.chunk);
        assert!((recovered.local.x - target.local.x).abs() < 1e-4);
        assert!((recovered.local.y - target.local.y).abs() < 1e-4);
        assert!((recovered.local.z - target.local.z).abs() < 1e-4);
    }
}
