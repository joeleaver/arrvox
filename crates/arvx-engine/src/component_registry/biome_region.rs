//! `BiomeRegion` component entry — Phase 6 of the terrain plan.
//!
//! V1 only exposes `material_override` as an editable field: it's a
//! plain integer (the `MaterialId`) authors can leave as -1 ("no
//! override") or set to a positive id. The `terrain_fn_override`
//! field carries `Option<Arc<dyn TerrainFn>>` — a runtime trait
//! object — and is not Inspector-editable in V1; Phase 7+ will add
//! either a preset picker or a node-graph editor for procedural
//! biomes.
//!
//! Scene serde is custom: the field stores `material_override` only.
//! `terrain_fn_override` is reset to `None` on deserialise (matches
//! the V1 "code-defined TerrainFn only" decision in TERRAIN.md).

use arvx_terrain::BiomeRegion;
use serde::{Deserialize, Serialize};

use super::{ComponentEntry, FieldMeta};
use crate::inspector::{FieldType, FieldValue};

/// Sentinel used by the integer Inspector field for "no override."
/// Stored as i64 in the wire/serde form so `-1` is unambiguous.
const NO_MATERIAL_OVERRIDE: i64 = -1;

static BIOME_REGION_FIELDS: [FieldMeta; 1] = [FieldMeta {
    name: "material_override",
    field_type: FieldType::Int,
    range: Some((NO_MATERIAL_OVERRIDE as f64, 65535.0)),
    transient: false,
    struct_fields: None,
    asset_filter: None,
    enum_options: None,
    scrub: false,
}];

#[derive(Serialize, Deserialize)]
struct BiomeRegionSerde {
    material_override: Option<u16>,
}

impl From<&BiomeRegion> for BiomeRegionSerde {
    fn from(b: &BiomeRegion) -> Self {
        Self {
            material_override: b.material_override,
        }
    }
}

pub fn biome_region_entry() -> ComponentEntry {
    ComponentEntry {
        name: "BiomeRegion",
        meta: &BIOME_REGION_FIELDS,
        mandatory: false,
        has: |world, entity| world.get::<&BiomeRegion>(entity).is_ok(),
        get_field: |world, entity, field| {
            let b = world
                .get::<&BiomeRegion>(entity)
                .map_err(|_| "no BiomeRegion".to_string())?;
            match field {
                "material_override" => Ok(FieldValue::Int(
                    b.material_override.map(|m| m as i64).unwrap_or(NO_MATERIAL_OVERRIDE),
                )),
                _ => Err(format!("unknown field '{field}'")),
            }
        },
        set_field: |world, entity, field, value| {
            let mut b = world
                .get::<&mut BiomeRegion>(entity)
                .map_err(|_| "no BiomeRegion".to_string())?;
            match field {
                "material_override" => {
                    if let FieldValue::Int(v) = value {
                        b.material_override = if v <= NO_MATERIAL_OVERRIDE {
                            None
                        } else {
                            Some(v.clamp(0, u16::MAX as i64) as u16)
                        };
                        Ok(())
                    } else {
                        Err("type mismatch".into())
                    }
                }
                _ => Err(format!("unknown field '{field}'")),
            }
        },
        add_default: |world, entity| {
            world
                .insert_one(entity, BiomeRegion::default())
                .map_err(|e| format!("{e}"))
        },
        remove: |world, entity| {
            world
                .remove_one::<BiomeRegion>(entity)
                .map(|_| ())
                .map_err(|e| format!("{e}"))
        },
        serialize: |world, entity| {
            let b = world.get::<&BiomeRegion>(entity).ok()?;
            serde_json::to_string(&BiomeRegionSerde::from(&*b)).ok()
        },
        deserialize_insert: |world, entity, json| {
            let s: BiomeRegionSerde = serde_json::from_str(json).map_err(|e| format!("{e}"))?;
            let b = BiomeRegion {
                terrain_fn_override: None,
                material_override: s.material_override,
            };
            world.insert_one(entity, b).map_err(|e| format!("{e}"))
        },
        on_add: None,
        on_remove: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hecs::World;

    #[test]
    fn add_default_inserts_empty() {
        let mut w = World::new();
        let e = w.spawn(());
        let entry = biome_region_entry();
        (entry.add_default)(&mut w, e).expect("add_default");
        let b = w.get::<&BiomeRegion>(e).unwrap();
        assert!(b.is_empty());
    }

    #[test]
    fn set_get_material_override_roundtrips() {
        let mut w = World::new();
        let e = w.spawn((BiomeRegion::default(),));
        let entry = biome_region_entry();
        (entry.set_field)(&mut w, e, "material_override", FieldValue::Int(7)).unwrap();
        let v = (entry.get_field)(&w, e, "material_override").unwrap();
        assert_eq!(v, FieldValue::Int(7));
    }

    #[test]
    fn negative_one_clears_override() {
        let mut w = World::new();
        let e = w.spawn((BiomeRegion {
            material_override: Some(5),
            ..Default::default()
        },));
        let entry = biome_region_entry();
        (entry.set_field)(&mut w, e, "material_override", FieldValue::Int(-1)).unwrap();
        let b = w.get::<&BiomeRegion>(e).unwrap();
        assert!(b.material_override.is_none());
    }

    #[test]
    fn serialise_drops_terrain_fn_override() {
        // V1: terrain_fn_override is runtime-only — never persisted.
        let mut w = World::new();
        let e = w.spawn((BiomeRegion {
            terrain_fn_override: None, // doesn't matter — Arc<dyn> not constructible cheaply here
            material_override: Some(12),
        },));
        let entry = biome_region_entry();
        let json = (entry.serialize)(&w, e).expect("serialise");
        let _ = w.remove_one::<BiomeRegion>(e);
        (entry.deserialize_insert)(&mut w, e, &json).expect("deserialise");
        let b = w.get::<&BiomeRegion>(e).unwrap();
        assert_eq!(b.material_override, Some(12));
        assert!(b.terrain_fn_override.is_none());
    }
}
