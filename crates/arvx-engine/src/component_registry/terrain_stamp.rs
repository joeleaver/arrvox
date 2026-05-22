//! `Stamp` component entry — registers the arvx-terrain `Stamp` as
//! an inspectable + serializable ECS component.
//!
//! The Stamp's internal `kind: StampKind` is a tagged enum with
//! variant-specific fields. The Inspector's `FieldMeta` model doesn't
//! support conditional per-variant fields, so V1 surfaces just three
//! flat fields:
//!
//! * `kind_name` — read-only label of the active variant
//!   (Mountain / Hill / Lake / Plateau / Flatten). Switching kinds
//!   in-place is V2 territory; for V1 the user deletes + re-spawns.
//! * `amplitude` — virtual field mapped to the kind's vertical
//!   parameter (`h_max` for Mountain/Hill, `depth` for Lake, ignored
//!   for Plateau/Flatten which use `position.y` as their target).
//! * `radius` — virtual field mapped to the kind's horizontal
//!   parameter (`radius` for circular kinds, the larger of
//!   `half_extents` for rectangular kinds — read-only on rect).
//! * `priority` — composition order; higher applies later.
//!
//! Scene serde is the full `Stamp` value (round-trips every field,
//! including the falloff curve and the rectangle half-extents). The
//! Inspector just doesn't expose every field as an editable widget.

use crate::inspector::{FieldType, FieldValue};
use arvx_terrain::{Stamp, StampKind};

use super::{ComponentEntry, FieldMeta};

static STAMP_KIND_OPTIONS: &[(&str, &str)] = &[
    ("Mountain", "Mountain"),
    ("Hill", "Hill"),
    ("Lake", "Lake"),
    ("Plateau", "Plateau"),
    ("Flatten", "Flatten"),
];

/// Flat-field Inspector surface. A full dedicated stamp editor /
/// stamp library UI is a separate next-session investment; this
/// keeps things scrubbable in the meantime. Fields that don't apply
/// to the current variant get treated as no-ops by the get/set
/// dispatch (read returns 0, write is silently dropped).
static STAMP_FIELDS: [FieldMeta; 13] = [
    FieldMeta {
        name: "kind_name",
        field_type: FieldType::String,
        range: None,
        transient: true,
        struct_fields: None,
        asset_filter: None,
        enum_options: Some(STAMP_KIND_OPTIONS),
        scrub: false,
    },
    FieldMeta {
        name: "amplitude",
        field_type: FieldType::Float,
        range: Some((0.0, 1000.0)),
        transient: false,
        struct_fields: None,
        asset_filter: None,
        enum_options: None,
        scrub: true,
    },
    FieldMeta {
        name: "radius",
        field_type: FieldType::Float,
        range: Some((0.5, 1000.0)),
        transient: false,
        struct_fields: None,
        asset_filter: None,
        enum_options: None,
        scrub: true,
    },
    FieldMeta {
        name: "priority",
        field_type: FieldType::Int,
        range: Some((-1000.0, 1000.0)),
        transient: false,
        struct_fields: None,
        asset_filter: None,
        enum_options: None,
        scrub: false,
    },
    // ── V2 shape knobs ──────────────────────────────────────
    FieldMeta {
        name: "aspect",
        field_type: FieldType::Float,
        range: Some((0.1, 10.0)),
        transient: false,
        struct_fields: None,
        asset_filter: None,
        enum_options: None,
        scrub: true,
    },
    FieldMeta {
        name: "ridge_strength",
        field_type: FieldType::Float,
        range: Some((0.0, 1.0)),
        transient: false,
        struct_fields: None,
        asset_filter: None,
        enum_options: None,
        scrub: true,
    },
    FieldMeta {
        name: "ridge_count",
        field_type: FieldType::Int,
        range: Some((0.0, 12.0)),
        transient: false,
        struct_fields: None,
        asset_filter: None,
        enum_options: None,
        scrub: false,
    },
    FieldMeta {
        name: "floor_flat_frac",
        field_type: FieldType::Float,
        range: Some((0.0, 0.99)),
        transient: false,
        struct_fields: None,
        asset_filter: None,
        enum_options: None,
        scrub: true,
    },
    FieldMeta {
        name: "corner_radius_m",
        field_type: FieldType::Float,
        range: Some((0.0, 100.0)),
        transient: false,
        struct_fields: None,
        asset_filter: None,
        enum_options: None,
        scrub: true,
    },
    FieldMeta {
        name: "edge_falloff_m",
        field_type: FieldType::Float,
        range: Some((0.0, 100.0)),
        transient: false,
        struct_fields: None,
        asset_filter: None,
        enum_options: None,
        scrub: true,
    },
    // ── ShapeNoise (cross-cutting) ─────────────────────────
    FieldMeta {
        name: "noise_amp_m",
        field_type: FieldType::Float,
        range: Some((0.0, 100.0)),
        transient: false,
        struct_fields: None,
        asset_filter: None,
        enum_options: None,
        scrub: true,
    },
    FieldMeta {
        name: "noise_scale_m",
        field_type: FieldType::Float,
        range: Some((0.5, 200.0)),
        transient: false,
        struct_fields: None,
        asset_filter: None,
        enum_options: None,
        scrub: true,
    },
    FieldMeta {
        name: "noise_seed",
        field_type: FieldType::Int,
        range: Some((0.0, u32::MAX as f64)),
        transient: false,
        struct_fields: None,
        asset_filter: None,
        enum_options: None,
        scrub: false,
    },
];

// ── per-knob helpers ──────────────────────────────────────────────

fn aspect_of(s: &Stamp) -> f32 {
    match s.kind {
        StampKind::Mountain { aspect, .. }
        | StampKind::Hill { aspect, .. }
        | StampKind::Lake { aspect, .. } => aspect,
        _ => 1.0,
    }
}

fn set_aspect(s: &mut Stamp, v: f32) {
    match &mut s.kind {
        StampKind::Mountain { aspect, .. }
        | StampKind::Hill { aspect, .. }
        | StampKind::Lake { aspect, .. } => *aspect = v.max(0.01),
        _ => {} // rect kinds use full-rectangle yaw; aspect is N/A
    }
}

fn ridge_strength_of(s: &Stamp) -> f32 {
    match s.kind {
        StampKind::Mountain { ridge_strength, .. }
        | StampKind::Hill { ridge_strength, .. } => ridge_strength,
        _ => 0.0,
    }
}

fn set_ridge_strength(s: &mut Stamp, v: f32) {
    match &mut s.kind {
        StampKind::Mountain { ridge_strength, .. }
        | StampKind::Hill { ridge_strength, .. } => *ridge_strength = v.clamp(0.0, 1.0),
        _ => {}
    }
}

fn ridge_count_of(s: &Stamp) -> u8 {
    match s.kind {
        StampKind::Mountain { ridge_count, .. } | StampKind::Hill { ridge_count, .. } => {
            ridge_count
        }
        _ => 0,
    }
}

fn set_ridge_count(s: &mut Stamp, v: u8) {
    match &mut s.kind {
        StampKind::Mountain { ridge_count, .. } | StampKind::Hill { ridge_count, .. } => {
            *ridge_count = v
        }
        _ => {}
    }
}

fn floor_flat_frac_of(s: &Stamp) -> f32 {
    match s.kind {
        StampKind::Lake { floor_flat_frac, .. } => floor_flat_frac,
        _ => 0.0,
    }
}

fn set_floor_flat_frac(s: &mut Stamp, v: f32) {
    if let StampKind::Lake { floor_flat_frac, .. } = &mut s.kind {
        *floor_flat_frac = v.clamp(0.0, 0.99);
    }
}

fn corner_radius_of(s: &Stamp) -> f32 {
    match s.kind {
        StampKind::Plateau { corner_radius_m, .. } | StampKind::Flatten { corner_radius_m, .. } => {
            corner_radius_m
        }
        _ => 0.0,
    }
}

fn set_corner_radius(s: &mut Stamp, v: f32) {
    match &mut s.kind {
        StampKind::Plateau { corner_radius_m, .. }
        | StampKind::Flatten { corner_radius_m, .. } => *corner_radius_m = v.max(0.0),
        _ => {}
    }
}

fn edge_falloff_of(s: &Stamp) -> f32 {
    match s.kind {
        StampKind::Plateau { edge_falloff_m, .. } | StampKind::Flatten { edge_falloff_m, .. } => {
            edge_falloff_m
        }
        _ => 0.0,
    }
}

fn set_edge_falloff(s: &mut Stamp, v: f32) {
    match &mut s.kind {
        StampKind::Plateau { edge_falloff_m, .. }
        | StampKind::Flatten { edge_falloff_m, .. } => *edge_falloff_m = v.max(0.0),
        _ => {}
    }
}

fn kind_name(s: &Stamp) -> &'static str {
    match s.kind {
        StampKind::Mountain { .. } => "Mountain",
        StampKind::Hill { .. } => "Hill",
        StampKind::Lake { .. } => "Lake",
        StampKind::Plateau { .. } => "Plateau",
        StampKind::Flatten { .. } => "Flatten",
    }
}

fn kind_amplitude(s: &Stamp) -> f32 {
    match s.kind {
        StampKind::Mountain { h_max, .. } | StampKind::Hill { h_max, .. } => h_max,
        StampKind::Lake { depth, .. } => depth,
        // Plateau / Flatten use position.y as the target; no amplitude.
        StampKind::Plateau { .. } | StampKind::Flatten { .. } => 0.0,
    }
}

fn set_kind_amplitude(s: &mut Stamp, v: f32) {
    match &mut s.kind {
        StampKind::Mountain { h_max, .. } | StampKind::Hill { h_max, .. } => *h_max = v,
        StampKind::Lake { depth, .. } => *depth = v,
        StampKind::Plateau { .. } | StampKind::Flatten { .. } => {
            // No-op — rectangular kinds' Y is position.y; edit Transform.position.y instead.
        }
    }
}

fn kind_radius(s: &Stamp) -> f32 {
    match s.kind {
        StampKind::Mountain { radius, .. }
        | StampKind::Hill { radius, .. }
        | StampKind::Lake { radius, .. } => radius,
        StampKind::Plateau { half_extents, .. } | StampKind::Flatten { half_extents, .. } => {
            half_extents.x.max(half_extents.y)
        }
    }
}

fn set_kind_radius(s: &mut Stamp, v: f32) {
    match &mut s.kind {
        StampKind::Mountain { radius, .. }
        | StampKind::Hill { radius, .. }
        | StampKind::Lake { radius, .. } => *radius = v,
        StampKind::Plateau { half_extents, .. } | StampKind::Flatten { half_extents, .. } => {
            // Maintain aspect ratio by scaling both axes to the new value.
            *half_extents = glam::Vec2::new(v, v);
        }
    }
}

/// Construct the `Stamp` component entry.
pub fn stamp_entry() -> ComponentEntry {
    ComponentEntry {
        name: "Stamp",
        meta: &STAMP_FIELDS,
        mandatory: false,
        has: |world, entity| world.get::<&Stamp>(entity).is_ok(),
        get_field: |world, entity, field| {
            let c = world.get::<&Stamp>(entity).map_err(|_| "no Stamp".to_string())?;
            match field {
                "kind_name" => Ok(FieldValue::String(kind_name(&c).to_string())),
                "amplitude" => Ok(FieldValue::Float(kind_amplitude(&c) as f64)),
                "radius" => Ok(FieldValue::Float(kind_radius(&c) as f64)),
                "priority" => Ok(FieldValue::Int(c.priority as i64)),
                "aspect" => Ok(FieldValue::Float(aspect_of(&c) as f64)),
                "ridge_strength" => Ok(FieldValue::Float(ridge_strength_of(&c) as f64)),
                "ridge_count" => Ok(FieldValue::Int(ridge_count_of(&c) as i64)),
                "floor_flat_frac" => Ok(FieldValue::Float(floor_flat_frac_of(&c) as f64)),
                "corner_radius_m" => Ok(FieldValue::Float(corner_radius_of(&c) as f64)),
                "edge_falloff_m" => Ok(FieldValue::Float(edge_falloff_of(&c) as f64)),
                "noise_amp_m" => Ok(FieldValue::Float(c.shape_noise.amp_m as f64)),
                "noise_scale_m" => Ok(FieldValue::Float(c.shape_noise.scale_m as f64)),
                "noise_seed" => Ok(FieldValue::Int(c.shape_noise.seed as i64)),
                _ => Err(format!("unknown field '{field}'")),
            }
        },
        set_field: |world, entity, field, value| {
            let mut c = world
                .get::<&mut Stamp>(entity)
                .map_err(|_| "no Stamp".to_string())?;
            match field {
                "kind_name" => Err(
                    "kind_name is read-only in V1 — delete + re-spawn to change variant".into(),
                ),
                "amplitude" => {
                    if let FieldValue::Float(v) = value {
                        set_kind_amplitude(&mut c, v as f32);
                        Ok(())
                    } else {
                        Err("type mismatch".into())
                    }
                }
                "radius" => {
                    if let FieldValue::Float(v) = value {
                        set_kind_radius(&mut c, v as f32);
                        Ok(())
                    } else {
                        Err("type mismatch".into())
                    }
                }
                "priority" => {
                    if let FieldValue::Int(v) = value {
                        c.priority = v as i32;
                        Ok(())
                    } else {
                        Err("type mismatch".into())
                    }
                }
                "aspect" => {
                    if let FieldValue::Float(v) = value {
                        set_aspect(&mut c, v as f32);
                        Ok(())
                    } else {
                        Err("type mismatch".into())
                    }
                }
                "ridge_strength" => {
                    if let FieldValue::Float(v) = value {
                        set_ridge_strength(&mut c, v as f32);
                        Ok(())
                    } else {
                        Err("type mismatch".into())
                    }
                }
                "ridge_count" => {
                    if let FieldValue::Int(v) = value {
                        set_ridge_count(&mut c, v.clamp(0, 12) as u8);
                        Ok(())
                    } else {
                        Err("type mismatch".into())
                    }
                }
                "floor_flat_frac" => {
                    if let FieldValue::Float(v) = value {
                        set_floor_flat_frac(&mut c, v as f32);
                        Ok(())
                    } else {
                        Err("type mismatch".into())
                    }
                }
                "corner_radius_m" => {
                    if let FieldValue::Float(v) = value {
                        set_corner_radius(&mut c, v as f32);
                        Ok(())
                    } else {
                        Err("type mismatch".into())
                    }
                }
                "edge_falloff_m" => {
                    if let FieldValue::Float(v) = value {
                        set_edge_falloff(&mut c, v as f32);
                        Ok(())
                    } else {
                        Err("type mismatch".into())
                    }
                }
                "noise_amp_m" => {
                    if let FieldValue::Float(v) = value {
                        c.shape_noise.amp_m = (v as f32).max(0.0);
                        Ok(())
                    } else {
                        Err("type mismatch".into())
                    }
                }
                "noise_scale_m" => {
                    if let FieldValue::Float(v) = value {
                        c.shape_noise.scale_m = (v as f32).max(0.1);
                        Ok(())
                    } else {
                        Err("type mismatch".into())
                    }
                }
                "noise_seed" => {
                    if let FieldValue::Int(v) = value {
                        c.shape_noise.seed = v.max(0) as u32;
                        Ok(())
                    } else {
                        Err("type mismatch".into())
                    }
                }
                _ => Err(format!("unknown field '{field}'")),
            }
        },
        add_default: |_world, _entity| {
            // Stamps need a kind on spawn — no meaningful default. Use
            // EngineCommand::SpawnStamp to create them.
            Err("add Stamp via SpawnStamp command — no default kind".into())
        },
        remove: |world, entity| {
            world
                .remove_one::<Stamp>(entity)
                .map(|_| ())
                .map_err(|e| format!("{e}"))
        },
        serialize: |world, entity| {
            let c = world.get::<&Stamp>(entity).ok()?;
            serde_json::to_string(&*c).ok()
        },
        deserialize_insert: |world, entity, json| {
            let c: Stamp = serde_json::from_str(json).map_err(|e| format!("{e}"))?;
            world.insert_one(entity, c).map_err(|e| format!("{e}"))
        },
        on_add: None,
        on_remove: None,
        field_visible: Some(stamp_field_visible),
    }
}

/// Per-variant field visibility. Mountain / Hill don't carry
/// `corner_radius_m` / `edge_falloff_m` / `tilt`; Lake doesn't carry
/// the ridge knobs; rect kinds don't carry `aspect` / `ridge_*` /
/// `floor_flat_frac`. Hiding the irrelevant rows makes the
/// Inspector match each variant's actual storage — no more "drag
/// this field and nothing happens" because the kind silently
/// no-ops the write.
fn stamp_field_visible(world: &hecs::World, entity: hecs::Entity, field: &str) -> bool {
    let Ok(s) = world.get::<&Stamp>(entity) else {
        return true; // surface every field if we can't introspect — caller error
    };
    let is_mountain_or_hill = matches!(s.kind, StampKind::Mountain { .. } | StampKind::Hill { .. });
    let is_circular = matches!(
        s.kind,
        StampKind::Mountain { .. } | StampKind::Hill { .. } | StampKind::Lake { .. }
    );
    let is_lake = matches!(s.kind, StampKind::Lake { .. });
    let is_rect = matches!(s.kind, StampKind::Plateau { .. } | StampKind::Flatten { .. });
    match field {
        // Always-on knobs.
        "kind_name" | "amplitude" | "radius" | "priority"
        | "noise_amp_m" | "noise_scale_m" | "noise_seed" => true,
        // Circular-only (Mountain / Hill / Lake have anisotropic radii).
        "aspect" => is_circular,
        // Mountain / Hill only (spinal ridges).
        "ridge_strength" | "ridge_count" => is_mountain_or_hill,
        // Lake only (flat-bottom basin).
        "floor_flat_frac" => is_lake,
        // Plateau / Flatten only (rounded corners + soft rim + tilt).
        "corner_radius_m" | "edge_falloff_m" => is_rect,
        // Unknown field — surface it so the bug is obvious.
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arvx_terrain::FalloffCurve;
    use glam::Vec3;
    use hecs::World;

    fn mountain() -> Stamp {
        Stamp::new(
            StampKind::Mountain {
                h_max: 50.0,
                radius: 30.0,
                falloff: FalloffCurve::Smoothstep,
                aspect: 1.0,
                ridge_strength: 0.0,
                ridge_count: 3,
            },
            Vec3::new(10.0, 0.0, 20.0),
        )
    }

    #[test]
    fn entry_has_returns_true_after_insert() {
        let mut w = World::new();
        let e = w.spawn((mountain(),));
        let entry = stamp_entry();
        assert!((entry.has)(&w, e));
    }

    #[test]
    fn entry_get_kind_name_round_trips_per_variant() {
        let mut w = World::new();
        let entry = stamp_entry();

        for (kind, expected) in [
            (
                StampKind::Mountain {
                    h_max: 1.0,
                    radius: 1.0,
                    falloff: FalloffCurve::Smoothstep,
                    aspect: 1.0,
                    ridge_strength: 0.0,
                    ridge_count: 3,
                },
                "Mountain",
            ),
            (
                StampKind::Hill {
                    h_max: 1.0,
                    radius: 1.0,
                    falloff: FalloffCurve::Smoothstep,
                    aspect: 1.0,
                    ridge_strength: 0.0,
                    ridge_count: 3,
                },
                "Hill",
            ),
            (
                StampKind::Lake {
                    depth: 1.0,
                    radius: 1.0,
                    falloff: FalloffCurve::Smoothstep,
                    aspect: 1.0,
                    floor_flat_frac: 0.0,
                },
                "Lake",
            ),
            (
                StampKind::Plateau {
                    half_extents: glam::Vec2::new(5.0, 5.0),
                    corner_radius_m: 0.0,
                    edge_falloff_m: 0.0,
                    tilt: glam::Vec2::ZERO,
                },
                "Plateau",
            ),
            (
                StampKind::Flatten {
                    half_extents: glam::Vec2::new(5.0, 5.0),
                    corner_radius_m: 0.0,
                    edge_falloff_m: 0.0,
                    tilt: glam::Vec2::ZERO,
                },
                "Flatten",
            ),
        ] {
            let e = w.spawn((Stamp::new(kind, Vec3::ZERO),));
            let v = (entry.get_field)(&w, e, "kind_name").expect("get");
            match v {
                FieldValue::String(s) => assert_eq!(s, expected),
                _ => panic!("expected String"),
            }
        }
    }

    #[test]
    fn entry_set_amplitude_writes_through_to_h_max() {
        let mut w = World::new();
        let e = w.spawn((mountain(),));
        let entry = stamp_entry();
        (entry.set_field)(&mut w, e, "amplitude", FieldValue::Float(123.0)).expect("set");
        let s = w.get::<&Stamp>(e).unwrap();
        match s.kind {
            StampKind::Mountain { h_max, .. } => assert!((h_max - 123.0).abs() < 1e-4),
            _ => panic!("kind shifted"),
        }
    }

    #[test]
    fn entry_set_edge_falloff_writes_through_on_plateau() {
        let mut w = World::new();
        let p = Stamp::new(
            StampKind::Plateau {
                half_extents: glam::Vec2::new(10.0, 10.0),
                corner_radius_m: 0.0,
                edge_falloff_m: 0.0,
                tilt: glam::Vec2::ZERO,
            },
            Vec3::ZERO,
        );
        let e = w.spawn((p,));
        let entry = stamp_entry();

        // Read: starts at 0.
        match (entry.get_field)(&w, e, "edge_falloff_m").unwrap() {
            FieldValue::Float(v) => assert!((v - 0.0).abs() < 1e-6, "initial got {v}"),
            other => panic!("expected Float, got {other:?}"),
        }

        // Write a non-zero.
        (entry.set_field)(&mut w, e, "edge_falloff_m", FieldValue::Float(5.5))
            .expect("set");

        // Read back: should be 5.5.
        match (entry.get_field)(&w, e, "edge_falloff_m").unwrap() {
            FieldValue::Float(v) => assert!(
                (v - 5.5).abs() < 1e-4,
                "after set, read got {v} — Inspector dispatch broken",
            ),
            other => panic!("expected Float, got {other:?}"),
        }

        // And the actual component field should be updated.
        let s = w.get::<&Stamp>(e).unwrap();
        match s.kind {
            StampKind::Plateau { edge_falloff_m, .. } => {
                assert!((edge_falloff_m - 5.5).abs() < 1e-4);
            }
            _ => panic!("variant shifted"),
        }
    }

    #[test]
    fn field_visible_hides_edge_falloff_on_mountain() {
        let mut w = World::new();
        let m = Stamp::new(
            StampKind::Mountain {
                h_max: 50.0,
                radius: 30.0,
                falloff: arvx_terrain::FalloffCurve::Smoothstep,
                aspect: 1.0,
                ridge_strength: 0.0,
                ridge_count: 3,
            },
            Vec3::ZERO,
        );
        let e = w.spawn((m,));
        let entry = stamp_entry();
        let fv = entry.field_visible.expect("stamp has field_visible");
        assert!(!fv(&w, e, "edge_falloff_m"));
        assert!(!fv(&w, e, "corner_radius_m"));
        assert!(!fv(&w, e, "floor_flat_frac"));
        // Mountain DOES have these.
        assert!(fv(&w, e, "ridge_strength"));
        assert!(fv(&w, e, "ridge_count"));
        assert!(fv(&w, e, "aspect"));
        // Always-on knobs.
        assert!(fv(&w, e, "noise_amp_m"));
        assert!(fv(&w, e, "amplitude"));
    }

    #[test]
    fn field_visible_shows_edge_falloff_on_plateau() {
        let mut w = World::new();
        let p = Stamp::new(
            StampKind::Plateau {
                half_extents: glam::Vec2::new(10.0, 10.0),
                corner_radius_m: 0.0,
                edge_falloff_m: 0.0,
                tilt: glam::Vec2::ZERO,
            },
            Vec3::ZERO,
        );
        let e = w.spawn((p,));
        let entry = stamp_entry();
        let fv = entry.field_visible.expect("stamp has field_visible");
        assert!(fv(&w, e, "edge_falloff_m"));
        assert!(fv(&w, e, "corner_radius_m"));
        // Plateau doesn't have ridge / aspect / floor_flat_frac.
        assert!(!fv(&w, e, "ridge_strength"));
        assert!(!fv(&w, e, "aspect"));
        assert!(!fv(&w, e, "floor_flat_frac"));
    }

    #[test]
    fn entry_set_edge_falloff_is_noop_on_mountain() {
        // Mountain doesn't carry edge_falloff_m. Writing the field
        // should be a silent no-op (not an error) so the flat-field
        // Inspector stays uniform.
        let mut w = World::new();
        let m = Stamp::new(
            StampKind::Mountain {
                h_max: 50.0,
                radius: 30.0,
                falloff: arvx_terrain::FalloffCurve::Smoothstep,
                aspect: 1.0,
                ridge_strength: 0.0,
                ridge_count: 3,
            },
            Vec3::ZERO,
        );
        let e = w.spawn((m,));
        let entry = stamp_entry();
        // Set should succeed.
        (entry.set_field)(&mut w, e, "edge_falloff_m", FieldValue::Float(5.5))
            .expect("set");
        // Read should still return 0 (irrelevant for Mountain).
        match (entry.get_field)(&w, e, "edge_falloff_m").unwrap() {
            FieldValue::Float(v) => assert!((v - 0.0).abs() < 1e-6, "Mountain edge_falloff should stay 0; got {v}"),
            other => panic!("expected Float, got {other:?}"),
        }
    }

    #[test]
    fn entry_serialise_roundtrip_preserves_kind_variant() {
        let mut w = World::new();
        let original = Stamp::new(
            StampKind::Lake {
                depth: 12.0,
                radius: 25.0,
                falloff: FalloffCurve::Linear,
                aspect: 1.0,
                floor_flat_frac: 0.0,
            },
            Vec3::new(5.0, 10.0, 15.0),
        );
        let e = w.spawn((original,));
        let entry = stamp_entry();
        let json = (entry.serialize)(&w, e).expect("serialize");
        // Drop and re-insert via deserialize.
        let _ = w.remove_one::<Stamp>(e);
        (entry.deserialize_insert)(&mut w, e, &json).expect("deserialize");
        let back = w.get::<&Stamp>(e).unwrap();
        assert_eq!(*back, original);
    }
}
