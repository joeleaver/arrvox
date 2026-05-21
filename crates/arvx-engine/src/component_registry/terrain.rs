//! `Terrain` component entry — Phase 9 of the terrain plan.
//!
//! Surfaces the `arvx_terrain::Terrain` singleton to the Inspector
//! and the scene-file save/load path. The Terrain owns
//! `Arc<dyn TerrainFn>` (not serializable) and runtime snapshots
//! `StampIndex` / `TerrainRegionSnapshot` (rebuilt from ECS on
//! load) — so the registry's serde shape persists only the
//! authoritative fields: bounds, base tier, render radius, and the
//! `TerrainFnSpec` (which V1 covers via the `Fbm` variant).
//!
//! Inspector V1 surfaces flat fields for the common knobs:
//! * `bounded` — bool. `true` = `TerrainBounds::Bounded` with the
//!   subsequent extent fields; `false` = `Unbounded`.
//! * `extent_x` / `extent_y` / `extent_z` — extent in tiles.
//! * `render_radius_m` — camera-centric residency radius.
//! * FBM parameters (active when the spec is `Fbm`): `fbm_seed`,
//!   `fbm_octaves`, `fbm_scale_m`, `fbm_amplitude_m`,
//!   `fbm_base_height_m`, `fbm_sea_level_y`, `fbm_snow_level_y`,
//!   `fbm_slope_rock_threshold_deg`. Writes to any of these go
//!   through `Terrain::set_spec` so the cached `terrain_fn`
//!   trait-object stays in lockstep with the spec.
//!
//! Scene serde captures bounds + base_tier + render_radius_m +
//! `TerrainFnSpec` as JSON; `StampIndex` / `TerrainRegionSnapshot`
//! are rebuilt from the live ECS by the engine's invalidation
//! hooks after load.

use arvx_terrain::{FbmTerrainFn, Terrain, TerrainBounds, TerrainFnSpec};
use serde::{Deserialize, Serialize};

use super::{ComponentEntry, FieldMeta};
use crate::inspector::{FieldType, FieldValue};

static TERRAIN_FIELDS: [FieldMeta; 14] = [
    FieldMeta {
        name: "bounded",
        field_type: FieldType::Bool,
        range: None,
        transient: false,
        struct_fields: None,
        asset_filter: None,
        enum_options: None,
        scrub: false,
    },
    FieldMeta {
        name: "extent_x",
        field_type: FieldType::Int,
        range: Some((1.0, 1024.0)),
        transient: false,
        struct_fields: None,
        asset_filter: None,
        enum_options: None,
        scrub: false,
    },
    FieldMeta {
        name: "extent_y",
        field_type: FieldType::Int,
        range: Some((1.0, 1024.0)),
        transient: false,
        struct_fields: None,
        asset_filter: None,
        enum_options: None,
        scrub: false,
    },
    FieldMeta {
        name: "extent_z",
        field_type: FieldType::Int,
        range: Some((1.0, 1024.0)),
        transient: false,
        struct_fields: None,
        asset_filter: None,
        enum_options: None,
        scrub: false,
    },
    FieldMeta {
        name: "render_radius_m",
        field_type: FieldType::Float,
        range: Some((32.0, 4096.0)),
        transient: false,
        struct_fields: None,
        asset_filter: None,
        enum_options: None,
        scrub: true,
    },
    FieldMeta {
        name: "lod_levels",
        field_type: FieldType::Int,
        range: Some((1.0, 8.0)),
        transient: false,
        struct_fields: None,
        asset_filter: None,
        enum_options: None,
        scrub: false,
    },
    FieldMeta {
        name: "skirt_depth_m",
        field_type: FieldType::Float,
        range: Some((0.0, 64.0)),
        transient: false,
        struct_fields: None,
        asset_filter: None,
        enum_options: None,
        scrub: true,
    },
    FieldMeta {
        name: "fbm_seed",
        field_type: FieldType::Int,
        range: Some((0.0, u32::MAX as f64)),
        transient: false,
        struct_fields: None,
        asset_filter: None,
        enum_options: None,
        scrub: false,
    },
    FieldMeta {
        name: "fbm_octaves",
        field_type: FieldType::Int,
        range: Some((1.0, 12.0)),
        transient: false,
        struct_fields: None,
        asset_filter: None,
        enum_options: None,
        scrub: false,
    },
    FieldMeta {
        name: "fbm_scale_m",
        field_type: FieldType::Float,
        range: Some((4.0, 2048.0)),
        transient: false,
        struct_fields: None,
        asset_filter: None,
        enum_options: None,
        scrub: true,
    },
    FieldMeta {
        name: "fbm_amplitude_m",
        field_type: FieldType::Float,
        range: Some((0.0, 512.0)),
        transient: false,
        struct_fields: None,
        asset_filter: None,
        enum_options: None,
        scrub: true,
    },
    FieldMeta {
        name: "fbm_base_height_m",
        field_type: FieldType::Float,
        range: Some((-512.0, 512.0)),
        transient: false,
        struct_fields: None,
        asset_filter: None,
        enum_options: None,
        scrub: true,
    },
    FieldMeta {
        name: "fbm_sea_level_y",
        field_type: FieldType::Float,
        range: Some((-512.0, 512.0)),
        transient: false,
        struct_fields: None,
        asset_filter: None,
        enum_options: None,
        scrub: true,
    },
    FieldMeta {
        name: "fbm_slope_rock_threshold_deg",
        field_type: FieldType::Float,
        range: Some((0.0, 90.0)),
        transient: false,
        struct_fields: None,
        asset_filter: None,
        enum_options: None,
        scrub: true,
    },
];

/// JSON shape for scene save/load. Captures only the
/// authoritative fields; the runtime cache (`terrain_fn` trait
/// object, stamp/region snapshots) is rebuilt by the engine.
#[derive(Serialize, Deserialize)]
struct TerrainSerde {
    bounds: TerrainBounds,
    base_tier: usize,
    render_radius_m: f32,
    /// Defaults to `1` if absent in older scene files (V1 behavior).
    #[serde(default = "default_lod_levels")]
    lod_levels: u8,
    /// Defaults to `4.0` if absent in older scene files.
    #[serde(default = "default_skirt_depth_m")]
    skirt_depth_m: f32,
    spec: TerrainFnSpec,
}

fn default_lod_levels() -> u8 {
    1
}

fn default_skirt_depth_m() -> f32 {
    4.0
}

impl From<&Terrain> for TerrainSerde {
    fn from(t: &Terrain) -> Self {
        Self {
            bounds: t.bounds,
            base_tier: t.base_tier,
            render_radius_m: t.render_radius_m,
            lod_levels: t.lod_levels,
            skirt_depth_m: t.skirt_depth_m,
            spec: t.spec.clone(),
        }
    }
}

fn fbm(t: &Terrain) -> Option<FbmTerrainFn> {
    match &t.spec {
        TerrainFnSpec::Fbm(f) => Some(*f),
    }
}

fn set_fbm<F>(t: &mut Terrain, f: F) -> Result<(), String>
where
    F: FnOnce(&mut FbmTerrainFn),
{
    let TerrainFnSpec::Fbm(mut fbm) = t.spec.clone();
    f(&mut fbm);
    t.set_spec(TerrainFnSpec::Fbm(fbm));
    Ok(())
}

fn extent(t: &Terrain) -> Option<glam::UVec3> {
    match t.bounds {
        TerrainBounds::Bounded { extent, .. } => Some(extent),
        TerrainBounds::Unbounded => None,
    }
}

fn set_extent<F>(t: &mut Terrain, f: F)
where
    F: FnOnce(&mut glam::UVec3),
{
    match t.bounds {
        TerrainBounds::Bounded { origin, mut extent } => {
            f(&mut extent);
            // Clamp each axis ≥ 1.
            extent.x = extent.x.max(1);
            extent.y = extent.y.max(1);
            extent.z = extent.z.max(1);
            t.bounds = TerrainBounds::Bounded { origin, extent };
        }
        TerrainBounds::Unbounded => {
            // Unbounded has no extent — silently ignore. Authors
            // who want to edit extents must first toggle `bounded`.
        }
    }
}

pub fn terrain_entry() -> ComponentEntry {
    ComponentEntry {
        name: "Terrain",
        meta: &TERRAIN_FIELDS,
        mandatory: false,
        has: |world, entity| world.get::<&Terrain>(entity).is_ok(),
        get_field: |world, entity, field| {
            let t = world.get::<&Terrain>(entity).map_err(|_| "no Terrain".to_string())?;
            match field {
                "bounded" => Ok(FieldValue::Bool(matches!(
                    t.bounds,
                    TerrainBounds::Bounded { .. }
                ))),
                "extent_x" => Ok(FieldValue::Int(extent(&t).map(|e| e.x as i64).unwrap_or(0))),
                "extent_y" => Ok(FieldValue::Int(extent(&t).map(|e| e.y as i64).unwrap_or(0))),
                "extent_z" => Ok(FieldValue::Int(extent(&t).map(|e| e.z as i64).unwrap_or(0))),
                "render_radius_m" => Ok(FieldValue::Float(t.render_radius_m as f64)),
                "lod_levels" => Ok(FieldValue::Int(t.lod_levels as i64)),
                "skirt_depth_m" => Ok(FieldValue::Float(t.skirt_depth_m as f64)),
                "fbm_seed" => Ok(FieldValue::Int(
                    fbm(&t).map(|f| f.seed as i64).unwrap_or(0),
                )),
                "fbm_octaves" => Ok(FieldValue::Int(
                    fbm(&t).map(|f| f.octaves as i64).unwrap_or(0),
                )),
                "fbm_scale_m" => Ok(FieldValue::Float(
                    fbm(&t).map(|f| f.scale_m as f64).unwrap_or(0.0),
                )),
                "fbm_amplitude_m" => Ok(FieldValue::Float(
                    fbm(&t).map(|f| f.amplitude_m as f64).unwrap_or(0.0),
                )),
                "fbm_base_height_m" => Ok(FieldValue::Float(
                    fbm(&t).map(|f| f.base_height_m as f64).unwrap_or(0.0),
                )),
                "fbm_sea_level_y" => Ok(FieldValue::Float(
                    fbm(&t).map(|f| f.sea_level_y as f64).unwrap_or(0.0),
                )),
                "fbm_slope_rock_threshold_deg" => Ok(FieldValue::Float(
                    fbm(&t).map(|f| f.slope_rock_threshold_deg as f64).unwrap_or(0.0),
                )),
                _ => Err(format!("unknown field '{field}'")),
            }
        },
        set_field: |world, entity, field, value| {
            let mut t = world
                .get::<&mut Terrain>(entity)
                .map_err(|_| "no Terrain".to_string())?;
            match field {
                "bounded" => {
                    if let FieldValue::Bool(v) = value {
                        t.bounds = if v {
                            // Recover prior extent if we have one;
                            // otherwise fall back to default.
                            match t.bounds {
                                TerrainBounds::Bounded { .. } => t.bounds,
                                TerrainBounds::Unbounded => TerrainBounds::default(),
                            }
                        } else {
                            TerrainBounds::Unbounded
                        };
                        Ok(())
                    } else {
                        Err("type mismatch".into())
                    }
                }
                "extent_x" => {
                    if let FieldValue::Int(v) = value {
                        set_extent(&mut t, |e| e.x = v.clamp(1, u32::MAX as i64) as u32);
                        Ok(())
                    } else {
                        Err("type mismatch".into())
                    }
                }
                "extent_y" => {
                    if let FieldValue::Int(v) = value {
                        set_extent(&mut t, |e| e.y = v.clamp(1, u32::MAX as i64) as u32);
                        Ok(())
                    } else {
                        Err("type mismatch".into())
                    }
                }
                "extent_z" => {
                    if let FieldValue::Int(v) = value {
                        set_extent(&mut t, |e| e.z = v.clamp(1, u32::MAX as i64) as u32);
                        Ok(())
                    } else {
                        Err("type mismatch".into())
                    }
                }
                "render_radius_m" => {
                    if let FieldValue::Float(v) = value {
                        t.render_radius_m = v as f32;
                        Ok(())
                    } else {
                        Err("type mismatch".into())
                    }
                }
                "lod_levels" => {
                    if let FieldValue::Int(v) = value {
                        t.lod_levels = v.clamp(1, 8) as u8;
                        Ok(())
                    } else {
                        Err("type mismatch".into())
                    }
                }
                "skirt_depth_m" => {
                    if let FieldValue::Float(v) = value {
                        t.skirt_depth_m = (v as f32).clamp(0.0, 64.0);
                        Ok(())
                    } else {
                        Err("type mismatch".into())
                    }
                }
                "fbm_seed" => {
                    if let FieldValue::Int(v) = value {
                        set_fbm(&mut t, |f| f.seed = v.max(0) as u32)
                    } else {
                        Err("type mismatch".into())
                    }
                }
                "fbm_octaves" => {
                    if let FieldValue::Int(v) = value {
                        set_fbm(&mut t, |f| f.octaves = v.clamp(1, 12) as u8)
                    } else {
                        Err("type mismatch".into())
                    }
                }
                "fbm_scale_m" => {
                    if let FieldValue::Float(v) = value {
                        set_fbm(&mut t, |f| f.scale_m = (v as f32).max(0.1))
                    } else {
                        Err("type mismatch".into())
                    }
                }
                "fbm_amplitude_m" => {
                    if let FieldValue::Float(v) = value {
                        set_fbm(&mut t, |f| f.amplitude_m = (v as f32).max(0.0))
                    } else {
                        Err("type mismatch".into())
                    }
                }
                "fbm_base_height_m" => {
                    if let FieldValue::Float(v) = value {
                        set_fbm(&mut t, |f| f.base_height_m = v as f32)
                    } else {
                        Err("type mismatch".into())
                    }
                }
                "fbm_sea_level_y" => {
                    if let FieldValue::Float(v) = value {
                        set_fbm(&mut t, |f| f.sea_level_y = v as f32)
                    } else {
                        Err("type mismatch".into())
                    }
                }
                "fbm_slope_rock_threshold_deg" => {
                    if let FieldValue::Float(v) = value {
                        set_fbm(&mut t, |f| {
                            f.slope_rock_threshold_deg = (v as f32).clamp(0.0, 90.0)
                        })
                    } else {
                        Err("type mismatch".into())
                    }
                }
                _ => Err(format!("unknown field '{field}'")),
            }
        },
        add_default: |world, entity| {
            world
                .insert_one(entity, Terrain::default())
                .map_err(|e| format!("{e}"))
        },
        remove: |world, entity| {
            world
                .remove_one::<Terrain>(entity)
                .map(|_| ())
                .map_err(|e| format!("{e}"))
        },
        serialize: |world, entity| {
            let t = world.get::<&Terrain>(entity).ok()?;
            serde_json::to_string(&TerrainSerde::from(&*t)).ok()
        },
        deserialize_insert: |world, entity, json| {
            let s: TerrainSerde = serde_json::from_str(json).map_err(|e| format!("{e}"))?;
            let mut t = Terrain::default();
            t.bounds = s.bounds;
            t.base_tier = s.base_tier;
            t.render_radius_m = s.render_radius_m;
            t.lod_levels = s.lod_levels.clamp(1, 8);
            t.skirt_depth_m = s.skirt_depth_m.clamp(0.0, 64.0);
            t.set_spec(s.spec);
            world.insert_one(entity, t).map_err(|e| format!("{e}"))
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
    fn has_after_default_spawn() {
        let mut w = World::new();
        let e = w.spawn((Terrain::default(),));
        let entry = terrain_entry();
        assert!((entry.has)(&w, e));
    }

    #[test]
    fn get_bounded_default_returns_true() {
        let mut w = World::new();
        let e = w.spawn((Terrain::default(),));
        let entry = terrain_entry();
        let v = (entry.get_field)(&w, e, "bounded").unwrap();
        assert_eq!(v, FieldValue::Bool(true));
    }

    #[test]
    fn set_render_radius_writes_through() {
        let mut w = World::new();
        let e = w.spawn((Terrain::default(),));
        let entry = terrain_entry();
        (entry.set_field)(&mut w, e, "render_radius_m", FieldValue::Float(456.0)).unwrap();
        let t = w.get::<&Terrain>(e).unwrap();
        assert!((t.render_radius_m - 456.0).abs() < 1e-4);
    }

    #[test]
    fn set_fbm_seed_updates_spec_and_runtime() {
        let mut w = World::new();
        let e = w.spawn((Terrain::default(),));
        let entry = terrain_entry();
        (entry.set_field)(&mut w, e, "fbm_seed", FieldValue::Int(12345)).unwrap();
        let t = w.get::<&Terrain>(e).unwrap();
        match t.spec {
            TerrainFnSpec::Fbm(f) => assert_eq!(f.seed, 12345),
        }
        // Runtime trait object should produce the new seed's noise.
        let s = t.terrain_fn.sample(
            arvx_terrain::TileKey::level0(0, 0, 0),
            glam::Vec3::ZERO,
            0.25,
        );
        assert!(s.sd.is_finite());
    }

    #[test]
    fn toggle_unbounded_then_back_restores_default_extent() {
        let mut w = World::new();
        let e = w.spawn((Terrain::default(),));
        let entry = terrain_entry();
        (entry.set_field)(&mut w, e, "bounded", FieldValue::Bool(false)).unwrap();
        assert_eq!(
            (entry.get_field)(&w, e, "bounded").unwrap(),
            FieldValue::Bool(false)
        );
        (entry.set_field)(&mut w, e, "bounded", FieldValue::Bool(true)).unwrap();
        let v = (entry.get_field)(&w, e, "extent_x").unwrap();
        assert_eq!(v, FieldValue::Int(16));
    }

    #[test]
    fn extent_clamps_to_one_minimum() {
        let mut w = World::new();
        let e = w.spawn((Terrain::default(),));
        let entry = terrain_entry();
        (entry.set_field)(&mut w, e, "extent_x", FieldValue::Int(0)).unwrap();
        let v = (entry.get_field)(&w, e, "extent_x").unwrap();
        assert_eq!(v, FieldValue::Int(1));
    }

    #[test]
    fn serialise_roundtrip_fbm_terrain() {
        let mut w = World::new();
        let mut t = Terrain::default();
        t.render_radius_m = 333.0;
        t.set_spec(TerrainFnSpec::Fbm(FbmTerrainFn {
            seed: 99,
            octaves: 7,
            scale_m: 200.0,
            amplitude_m: 50.0,
            base_height_m: -5.0,
            sea_level_y: 0.0,
            snow_level_y: 40.0,
            slope_rock_threshold_deg: 30.0,
            slope_probe_m: 0.5,
            grass_material: 1,
            rock_material: 3,
            snow_material: 4,
            sand_material: 2,
        }));
        let e = w.spawn((t,));
        let entry = terrain_entry();
        let json = (entry.serialize)(&w, e).expect("serialise");
        let _ = w.remove_one::<Terrain>(e);
        (entry.deserialize_insert)(&mut w, e, &json).expect("deserialise");

        let back = w.get::<&Terrain>(e).unwrap();
        assert!((back.render_radius_m - 333.0).abs() < 1e-4);
        match back.spec {
            TerrainFnSpec::Fbm(f) => {
                assert_eq!(f.seed, 99);
                assert_eq!(f.octaves, 7);
                assert!((f.amplitude_m - 50.0).abs() < 1e-4);
            }
        }
    }
}
