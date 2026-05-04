//! u32::MAX-as-sentinel constants shared between Rust and WGSL.
//!
//! The renderer uses `0xFFFFFFFFu` (= [`u32::MAX`]) as a "no value
//! here" sentinel in several distinct contexts:
//!
//! | Constant | Where |
//! |---|---|
//! | [`OCTREE_EMPTY`] | octree node slot — no node here |
//! | [`BRICK_CELL_EMPTY`] | brick cell — no leaf at this cell |
//! | [`INTERNAL_ATTR_NONE`] | branch node — no prefiltered LOD attr |
//! | [`FACE_EMPTY_LINK`] | brick face link — no neighbour brick |
//! | [`HOST_NO_HOST_SENTINEL`] | shader region — free-standing (no host) |
//!
//! Same numeric value, semantically distinct meanings. The names
//! retain intent at the call site; centralising the base value here
//! means a future change to the encoding has one place to land. The
//! WGSL counterpart is documented at `lib/octree_slot.wesl`.

/// Base sentinel — `u32::MAX` (`0xFFFFFFFFu`). Use one of the
/// semantic aliases below at call sites.
pub const SENTINEL_NONE: u32 = u32::MAX;

/// Octree node slot — "no node here".
pub const OCTREE_EMPTY: u32 = SENTINEL_NONE;

/// Brick cell — "no leaf at this cell".
pub const BRICK_CELL_EMPTY: u32 = SENTINEL_NONE;

/// Branch node — "no prefiltered LOD attr available".
pub const INTERNAL_ATTR_NONE: u32 = SENTINEL_NONE;

/// Brick face link — "no neighbouring brick across this face".
pub const FACE_EMPTY_LINK: u32 = SENTINEL_NONE;

/// Shader region — "this region has no host instance".
pub const HOST_NO_HOST_SENTINEL: u32 = SENTINEL_NONE;
