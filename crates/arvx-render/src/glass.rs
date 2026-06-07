//! Glass classification — the single CPU-side authority.
//!
//! A material is "glass" iff its opacity is below
//! [`GLASS_OPACITY_THRESHOLD`]. Every CPU site that asks "is this
//! material glass?" — the engine's per-slot `material_is_glass` table,
//! the asset `has_glass` scan, the glass-pass FS uniform default —
//! routes through [`is_glass`] rather than hand-spelling
//! `opacity < 0.99`.
//!
//! This is a *forced* GPU/CPU boundary: the GPU rule lives in
//! `lib/types.wesl::is_glass` / `GLASS_OPACITY_THRESHOLD`. The two
//! cannot literally share a symbol across the WGSL boundary, so they
//! share the one numeric value and cross-reference each other in
//! comments. If you change the threshold, change both.
//!
//! NOTE: this rule is arvx-specific. RKIField's `rkf-render` classifies
//! glass with a different threshold (`opacity < 1.0`); do not unify
//! the two.

/// Opacity below which a material is treated as glass everywhere in the
/// render pipeline (opaque raster discards it, the glass front/back
/// pass keeps it, shadow passes mirror the split).
///
/// GPU mirror: `lib/types.wesl::GLASS_OPACITY_THRESHOLD`.
pub const GLASS_OPACITY_THRESHOLD: f32 = 0.99;

/// `true` iff a material with this `opacity` classifies as glass.
#[inline]
pub fn is_glass(opacity: f32) -> bool {
    opacity < GLASS_OPACITY_THRESHOLD
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opaque_materials_are_not_glass() {
        assert!(!is_glass(1.0));
        assert!(!is_glass(GLASS_OPACITY_THRESHOLD));
    }

    #[test]
    fn translucent_materials_are_glass() {
        assert!(is_glass(0.0));
        assert!(is_glass(0.5));
        assert!(is_glass(GLASS_OPACITY_THRESHOLD - 0.001));
    }
}
