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

static STAMP_FIELDS: [FieldMeta; 4] = [
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
];

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
        StampKind::Plateau { half_extents } | StampKind::Flatten { half_extents } => {
            half_extents.x.max(half_extents.y)
        }
    }
}

fn set_kind_radius(s: &mut Stamp, v: f32) {
    match &mut s.kind {
        StampKind::Mountain { radius, .. }
        | StampKind::Hill { radius, .. }
        | StampKind::Lake { radius, .. } => *radius = v,
        StampKind::Plateau { half_extents } | StampKind::Flatten { half_extents } => {
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
                },
                "Mountain",
            ),
            (
                StampKind::Hill {
                    h_max: 1.0,
                    radius: 1.0,
                    falloff: FalloffCurve::Smoothstep,
                },
                "Hill",
            ),
            (
                StampKind::Lake {
                    depth: 1.0,
                    radius: 1.0,
                    falloff: FalloffCurve::Smoothstep,
                },
                "Lake",
            ),
            (
                StampKind::Plateau {
                    half_extents: glam::Vec2::new(5.0, 5.0),
                },
                "Plateau",
            ),
            (
                StampKind::Flatten {
                    half_extents: glam::Vec2::new(5.0, 5.0),
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
    fn entry_serialise_roundtrip_preserves_kind_variant() {
        let mut w = World::new();
        let original = Stamp::new(
            StampKind::Lake {
                depth: 12.0,
                radius: 25.0,
                falloff: FalloffCurve::Linear,
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
