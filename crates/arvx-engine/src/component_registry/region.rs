//! `Region` component entry — Phase 6 of the terrain plan.
//!
//! Surfaces `arvx_regions::Region` to the Inspector. Like the
//! [`super::terrain_stamp`] entry, the underlying type has a tagged
//! enum (`RegionShape`) whose per-variant fields the flat
//! `FieldMeta` model doesn't render directly. V1 strategy mirrors
//! stamp's: a read-only `shape_name` label + flat virtual fields for
//! the common parameters.
//!
//! Editable virtual fields:
//! * `shape_name` — read-only label of the active variant. Switching
//!   shapes in-place is V2 territory; users delete + re-spawn.
//! * `size` — radius for Sphere, largest half-extent for Box / OBB.
//! * `falloff_name` — Hard / Linear / Smoothstep.
//! * `transition_m` — falloff transition band width (ignored when
//!   falloff is Hard; writes are stored on the next non-Hard variant).
//! * `priority` — overlap arbitration for single-valued properties.
//!
//! Scene serde is the full `Region` value (every variant field
//! round-trips, including OBB rotation). The Inspector just doesn't
//! expose them all as widgets in V1.

use arvx_regions::{Falloff, Region, RegionShape};

use super::{ComponentEntry, FieldMeta};
use crate::inspector::{FieldType, FieldValue};

static SHAPE_OPTIONS: &[(&str, &str)] = &[
    ("Sphere", "Sphere"),
    ("Box", "Box"),
    ("Obb", "OBB"),
];

static FALLOFF_OPTIONS: &[(&str, &str)] = &[
    ("Hard", "Hard"),
    ("Linear", "Linear"),
    ("Smoothstep", "Smoothstep"),
];

static REGION_FIELDS: [FieldMeta; 5] = [
    FieldMeta {
        name: "shape_name",
        field_type: FieldType::String,
        range: None,
        transient: true,
        struct_fields: None,
        asset_filter: None,
        enum_options: Some(SHAPE_OPTIONS),
        scrub: false,
    },
    FieldMeta {
        name: "size",
        field_type: FieldType::Float,
        range: Some((0.1, 1000.0)),
        transient: false,
        struct_fields: None,
        asset_filter: None,
        enum_options: None,
        scrub: true,
    },
    FieldMeta {
        name: "falloff_name",
        field_type: FieldType::String,
        range: None,
        transient: false,
        struct_fields: None,
        asset_filter: None,
        enum_options: Some(FALLOFF_OPTIONS),
        scrub: false,
    },
    FieldMeta {
        name: "transition_m",
        field_type: FieldType::Float,
        range: Some((0.0, 100.0)),
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

fn shape_name(r: &Region) -> &'static str {
    match r.shape {
        RegionShape::Sphere { .. } => "Sphere",
        RegionShape::Box { .. } => "Box",
        RegionShape::Obb { .. } => "Obb",
    }
}

fn shape_size(r: &Region) -> f32 {
    match r.shape {
        RegionShape::Sphere { radius } => radius,
        RegionShape::Box { half_extents } => half_extents.max_element(),
        RegionShape::Obb { half_extents, .. } => half_extents.max_element(),
    }
}

fn set_shape_size(r: &mut Region, v: f32) {
    match &mut r.shape {
        RegionShape::Sphere { radius } => *radius = v,
        RegionShape::Box { half_extents } => *half_extents = glam::Vec3::splat(v),
        RegionShape::Obb { half_extents, .. } => *half_extents = glam::Vec3::splat(v),
    }
}

fn falloff_name(r: &Region) -> &'static str {
    match r.falloff {
        Falloff::Hard => "Hard",
        Falloff::Linear { .. } => "Linear",
        Falloff::Smoothstep { .. } => "Smoothstep",
    }
}

fn set_falloff_name(r: &mut Region, name: &str) -> Result<(), String> {
    let t = r.falloff.transition_m();
    let preserved_t = if t > 0.0 { t } else { 5.0 };
    r.falloff = match name {
        "Hard" => Falloff::Hard,
        "Linear" => Falloff::Linear { transition_m: preserved_t },
        "Smoothstep" => Falloff::Smoothstep { transition_m: preserved_t },
        other => return Err(format!("unknown falloff '{other}'")),
    };
    Ok(())
}

fn set_transition_m(r: &mut Region, v: f32) {
    r.falloff = match r.falloff {
        // No-op on Hard — Hard ignores transition. The field stays
        // editable so a subsequent shift to Linear/Smoothstep picks
        // up the value the author typed.
        Falloff::Hard => Falloff::Hard,
        Falloff::Linear { .. } => Falloff::Linear { transition_m: v },
        Falloff::Smoothstep { .. } => Falloff::Smoothstep { transition_m: v },
    };
}

pub fn region_entry() -> ComponentEntry {
    ComponentEntry {
        name: "Region",
        meta: &REGION_FIELDS,
        mandatory: false,
        has: |world, entity| world.get::<&Region>(entity).is_ok(),
        get_field: |world, entity, field| {
            let r = world.get::<&Region>(entity).map_err(|_| "no Region".to_string())?;
            match field {
                "shape_name" => Ok(FieldValue::String(shape_name(&r).to_string())),
                "size" => Ok(FieldValue::Float(shape_size(&r) as f64)),
                "falloff_name" => Ok(FieldValue::String(falloff_name(&r).to_string())),
                "transition_m" => Ok(FieldValue::Float(r.falloff.transition_m() as f64)),
                "priority" => Ok(FieldValue::Int(r.priority as i64)),
                _ => Err(format!("unknown field '{field}'")),
            }
        },
        set_field: |world, entity, field, value| {
            let mut r = world
                .get::<&mut Region>(entity)
                .map_err(|_| "no Region".to_string())?;
            match field {
                "shape_name" => Err(
                    "shape_name is read-only in V1 — delete + re-spawn to change variant".into(),
                ),
                "size" => {
                    if let FieldValue::Float(v) = value {
                        set_shape_size(&mut r, v as f32);
                        Ok(())
                    } else {
                        Err("type mismatch".into())
                    }
                }
                "falloff_name" => {
                    if let FieldValue::String(s) = value {
                        set_falloff_name(&mut r, &s)
                    } else {
                        Err("type mismatch".into())
                    }
                }
                "transition_m" => {
                    if let FieldValue::Float(v) = value {
                        set_transition_m(&mut r, v as f32);
                        Ok(())
                    } else {
                        Err("type mismatch".into())
                    }
                }
                "priority" => {
                    if let FieldValue::Int(v) = value {
                        r.priority = v as i32;
                        Ok(())
                    } else {
                        Err("type mismatch".into())
                    }
                }
                _ => Err(format!("unknown field '{field}'")),
            }
        },
        add_default: |_world, _entity| {
            // Regions need a shape on spawn — no meaningful default.
            // Use EngineCommand::SpawnRegion.
            Err("add Region via SpawnRegion command — no default shape".into())
        },
        remove: |world, entity| {
            world
                .remove_one::<Region>(entity)
                .map(|_| ())
                .map_err(|e| format!("{e}"))
        },
        serialize: |world, entity| {
            let r = world.get::<&Region>(entity).ok()?;
            serde_json::to_string(&*r).ok()
        },
        deserialize_insert: |world, entity, json| {
            let r: Region = serde_json::from_str(json).map_err(|e| format!("{e}"))?;
            world.insert_one(entity, r).map_err(|e| format!("{e}"))
        },
        on_add: None,
        on_remove: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arvx_regions::{Falloff, RegionShape};
    use hecs::World;

    fn sphere_region() -> Region {
        Region {
            shape: RegionShape::Sphere { radius: 10.0 },
            falloff: Falloff::Smoothstep { transition_m: 3.0 },
            priority: 2,
        }
    }

    #[test]
    fn has_after_insert() {
        let mut w = World::new();
        let e = w.spawn((sphere_region(),));
        assert!((region_entry().has)(&w, e));
    }

    #[test]
    fn get_shape_name_per_variant() {
        let mut w = World::new();
        let entry = region_entry();
        for (shape, expected) in [
            (RegionShape::Sphere { radius: 1.0 }, "Sphere"),
            (
                RegionShape::Box {
                    half_extents: glam::Vec3::ONE,
                },
                "Box",
            ),
            (
                RegionShape::Obb {
                    half_extents: glam::Vec3::ONE,
                    rotation: glam::Quat::IDENTITY,
                },
                "Obb",
            ),
        ] {
            let e = w.spawn((Region {
                shape,
                ..sphere_region()
            },));
            let v = (entry.get_field)(&w, e, "shape_name").unwrap();
            match v {
                FieldValue::String(s) => assert_eq!(s, expected),
                _ => panic!("expected String"),
            }
        }
    }

    #[test]
    fn set_size_writes_through() {
        let mut w = World::new();
        let e = w.spawn((sphere_region(),));
        (region_entry().set_field)(&mut w, e, "size", FieldValue::Float(42.5))
            .expect("set size");
        let r = w.get::<&Region>(e).unwrap();
        match r.shape {
            RegionShape::Sphere { radius } => assert!((radius - 42.5).abs() < 1e-4),
            _ => panic!("variant changed"),
        }
    }

    #[test]
    fn switching_falloff_preserves_transition() {
        let mut w = World::new();
        let e = w.spawn((sphere_region(),));
        let entry = region_entry();
        // Sphere starts with Smoothstep transition_m = 3.
        (entry.set_field)(
            &mut w,
            e,
            "falloff_name",
            FieldValue::String("Linear".into()),
        )
        .expect("set falloff");
        let r = w.get::<&Region>(e).unwrap();
        match r.falloff {
            Falloff::Linear { transition_m } => assert!((transition_m - 3.0).abs() < 1e-4),
            other => panic!("expected Linear, got {other:?}"),
        }
    }

    #[test]
    fn switching_to_hard_then_back_preserves_default() {
        let mut w = World::new();
        let e = w.spawn((sphere_region(),));
        let entry = region_entry();
        // Smoothstep(3) → Hard zeroes transition; flipping back to
        // Linear without an explicit transition_m write picks the
        // fallback (5 m).
        (entry.set_field)(&mut w, e, "falloff_name", FieldValue::String("Hard".into())).unwrap();
        (entry.set_field)(
            &mut w,
            e,
            "falloff_name",
            FieldValue::String("Linear".into()),
        )
        .unwrap();
        let r = w.get::<&Region>(e).unwrap();
        match r.falloff {
            Falloff::Linear { transition_m } => assert!((transition_m - 5.0).abs() < 1e-4),
            other => panic!("expected Linear, got {other:?}"),
        }
    }

    #[test]
    fn serialise_roundtrip() {
        let mut w = World::new();
        let original = Region {
            shape: RegionShape::Obb {
                half_extents: glam::Vec3::new(2.0, 3.0, 4.0),
                rotation: glam::Quat::from_rotation_y(0.7),
            },
            falloff: Falloff::Linear { transition_m: 9.0 },
            priority: -5,
        };
        let e = w.spawn((original,));
        let entry = region_entry();
        let json = (entry.serialize)(&w, e).expect("serialise");
        let _ = w.remove_one::<Region>(e);
        (entry.deserialize_insert)(&mut w, e, &json).expect("deserialise");
        let back = w.get::<&Region>(e).unwrap();
        assert_eq!(*back, original);
    }
}
