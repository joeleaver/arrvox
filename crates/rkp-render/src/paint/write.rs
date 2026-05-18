//! Paint write operations against the scene's `LeafAttrPool` + per-leaf
//! color overlay.
//!
//! [`PaintStamp`] is the input — material set, color set, or color
//! erase. The `paint_*` fns mutate the pool; the `compute_*` fns are
//! pure helpers that produce the new value from old + brush weight,
//! exposed for tests + re-use from per-instance overlay paths.
//!
//! `pack_color` / `unpack_color` describe the per-leaf color word
//! layout (RGB565 + 8-bit intensity).

use rkp_core::LeafAttrPool;

/// What a single brush stamp writes. The scene-manager orchestrator
/// takes this + a brush sphere and applies the matching per-leaf write
/// to every voxel under the brush. Rkp-engine converts its command's
/// `PaintMode` + fields into one of these variants at call time.
#[derive(Debug, Clone, Copy)]
pub enum PaintStamp {
    /// Flip the target leaf's primary material (with soft blend on the
    /// sphere shoulder via `LeafAttr.material_secondary`).
    Material { material_id: u16 },
    /// Write `rgb` into the companion color_pool, lerping from the
    /// existing color by the per-leaf weight.
    Color { rgb: [f32; 3] },
    /// Fade the companion color override toward the material base
    /// color (color_pool=0 sentinel).
    Erase,
}

/// Weight-curve used by paint writes: returns a value in `[0, strength]` that
/// falls from the brush center to zero at the radius edge. Matches the
/// rkifield curve (linear core + smoothstep shoulder) so Phase 3's geodesic
/// upgrade can swap the distance metric without touching the weight shape.
///
/// `falloff` is the fraction of the radius occupied by the smoothstep
/// shoulder: 0.0 = hard edge, 1.0 = smoothstep from the center outward.
#[inline]
pub fn brush_weight(distance: f32, radius: f32, strength: f32, falloff: f32) -> f32 {
    if radius <= 0.0 || distance < 0.0 || distance >= radius {
        return 0.0;
    }
    if falloff <= 0.0 {
        return strength;
    }
    let t = distance / radius;
    let edge_start = 1.0 - falloff;
    if t <= edge_start {
        strength
    } else {
        let s = (t - edge_start) / falloff;
        strength * (1.0 - s * s * (3.0 - 2.0 * s))
    }
}

/// Pack a linear RGB triple plus a write intensity (0-255) into the
/// `LeafAttrPool::colors` layout: `R | G<<8 | B<<16 | intensity<<24`. An
/// intensity of 0 is the "no override, fall back to material base_color"
/// sentinel — paint writes that want to zero a voxel's color should use
/// `erase_leaf_color`, not pack intensity=0 here, so partial erasures still
/// blend correctly.
#[inline]
pub fn pack_color(rgb: [f32; 3], intensity: f32) -> u32 {
    let r = (rgb[0].clamp(0.0, 1.0) * 255.0).round() as u32;
    let g = (rgb[1].clamp(0.0, 1.0) * 255.0).round() as u32;
    let b = (rgb[2].clamp(0.0, 1.0) * 255.0).round() as u32;
    let i = (intensity.clamp(0.0, 1.0) * 255.0).round() as u32;
    r | (g << 8) | (b << 16) | (i << 24)
}

/// Unpack a packed color u32 into linear RGB + intensity. Inverse of
/// [`pack_color`].
#[inline]
pub fn unpack_color(packed: u32) -> ([f32; 3], f32) {
    let r = (packed & 0xFF) as f32 / 255.0;
    let g = ((packed >> 8) & 0xFF) as f32 / 255.0;
    let b = ((packed >> 16) & 0xFF) as f32 / 255.0;
    let i = ((packed >> 24) & 0xFF) as f32 / 255.0;
    ([r, g, b], i)
}

/// Write a new primary material to a leaf, with weighted dual-material
/// blending.
///
/// `weight` is in `[0, 1]` — typically [`brush_weight`] output for this
/// leaf. At weight=1 the leaf becomes pure `material_id` (blend weight 0).
/// At intermediate weights the new material rides into `material_secondary`
/// with a blend weight proportional to `weight`, so dragging the brush
/// softly paints a gradient from old to new material at the sphere edge.
///
/// The `LeafAttr` layout only allows 4 bits of blend weight (0-15), so
/// fractional painting quantizes to 16 levels. That's enough for visually
/// smooth transitions at typical voxel sizes; fine-grain material
/// gradients would need the blend-weight field widened in `LeafAttr`.
pub fn paint_leaf_material(
    pool: &mut LeafAttrPool,
    leaf_slot: u32,
    material_id: u16,
    weight: f32,
) {
    let cur = *pool.get(leaf_slot);
    *pool.get_mut(leaf_slot) = compute_painted_attr(cur, material_id, weight);
}

/// Pure version of [`paint_leaf_material`]: takes the current [`LeafAttr`]
/// and the brush parameters, returns the new [`LeafAttr`]. Used by the
/// per-instance overlay path, which reads "current" from the overlay if
/// present (else from the asset's pool) and writes the result back into
/// the overlay rather than mutating the shared pool.
pub fn compute_painted_attr(
    cur: rkp_core::LeafAttr,
    material_id: u16,
    weight: f32,
) -> rkp_core::LeafAttr {
    let w = weight.clamp(0.0, 1.0);
    let cur_primary = cur.material_primary;
    if cur_primary == material_id || w <= 0.0 {
        // Either already painted with this material, or weight is zero.
        // Full-weight case still falls here when primary already matches.
        if w >= 0.999 {
            // Clear any leftover secondary blend toward a different material.
            return rkp_core::LeafAttr {
                normal_oct: cur.normal_oct,
                material_primary: material_id,
                material_secondary_blend: 0,
            };
        }
        return cur;
    }

    if w >= 0.999 {
        // Hard overwrite — primary flips to the new material, blend cleared.
        return rkp_core::LeafAttr {
            normal_oct: cur.normal_oct,
            material_primary: material_id,
            material_secondary_blend: 0,
        };
    }

    // Partial blend. Quantize weight to the 4-bit blend field.
    let blend_weight = (w * 15.0).round().clamp(0.0, 15.0) as u8;
    rkp_core::LeafAttr::new_blended(
        cur.normal(),
        cur_primary,
        material_id,
        blend_weight,
    )
}

/// Write a new color onto a leaf, lerping from the existing color by
/// `weight`. Unpainted leaves (intensity=0 in the colors array) start from
/// the target RGB at reduced intensity so a single dab gives visible color
/// immediately.
pub fn paint_leaf_color(
    pool: &mut LeafAttrPool,
    leaf_slot: u32,
    rgb: [f32; 3],
    weight: f32,
) {
    let cur = pool.color(leaf_slot);
    pool.set_color(leaf_slot, compute_painted_color(cur, rgb, weight));
}

/// Pure version of [`paint_leaf_color`]: returns the new packed color
/// given the current packed color (0 = no override) and brush parameters.
///
/// The brush's soft edge is realized at cell granularity — each leaf
/// inside the radius lerps its current RGB toward the target by
/// `weight`, but every painted cell ends up at full intensity (1.0).
/// Sub-voxel intensity used to be the soft-edge mechanism, but at low
/// intensity the gbuffer's `mix(base_color, painted_rgb, intensity)`
/// fades through whatever the host material's albedo is — including
/// black on assets whose leaves point at an unconfigured material
/// slot. Forcing intensity=1 makes painted cells predictable across
/// every base material; the visible falloff is the per-cell RGB lerp
/// instead.
pub fn compute_painted_color(cur: u32, rgb: [f32; 3], weight: f32) -> u32 {
    let w = weight.clamp(0.0, 1.0);
    if w <= 0.0 {
        return cur;
    }
    let cur_rgb = if cur == 0 {
        // No existing override — there's no "previous color" to lerp
        // from, so seed with the target. The first stamp writes the
        // target RGB at full intensity (a hard cell of `rgb`); later
        // stamps with a different target lerp toward that new target
        // by their own weight.
        rgb
    } else {
        unpack_color(cur).0
    };
    let new_rgb = [
        cur_rgb[0] + (rgb[0] - cur_rgb[0]) * w,
        cur_rgb[1] + (rgb[1] - cur_rgb[1]) * w,
        cur_rgb[2] + (rgb[2] - cur_rgb[2]) * w,
    ];
    pack_color(new_rgb, 1.0)
}

/// Erase a leaf's color by lerping the intensity channel toward zero.
/// Full strength wipes the override entirely (clears `color_pool[slot]`
/// to the 0 sentinel), returning the leaf to its material's base
/// albedo. Partial strength fades toward the material over multiple
/// strokes — same feel as Photoshop's eraser. The shade pass routes
/// `color_pool[slot] == 0` to `mat_albedo(material)` via a 0 in the
/// gbuffer's RGB565 channel; see `rkp_shade.wesl`.
pub fn erase_leaf_color(
    pool: &mut LeafAttrPool,
    leaf_slot: u32,
    weight: f32,
) {
    let cur = pool.color(leaf_slot);
    pool.set_color(leaf_slot, compute_erased_color(cur, weight));
}

/// Pure version of [`erase_leaf_color`]: returns the new packed color.
pub fn compute_erased_color(cur: u32, weight: f32) -> u32 {
    let w = weight.clamp(0.0, 1.0);
    if w <= 0.0 || cur == 0 {
        return cur;
    }
    let (cur_rgb, cur_i) = unpack_color(cur);
    let new_i = cur_i * (1.0 - w);
    if new_i <= 1.0 / 255.0 {
        // Intensity quantized to zero — clear the whole override so the
        // shader takes the material base color fast-path.
        0
    } else {
        pack_color(cur_rgb, new_i)
    }
}
