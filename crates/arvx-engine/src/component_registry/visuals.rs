//! Visual component entries: Renderable, PointLight, SpotLight, Camera.

use crate::inspector::{FieldType, FieldValue};

use super::{ComponentEntry, FieldMeta};

// ── Renderable ───────────────────────────────────────────────────────

// `material_id` is intentionally omitted — materials are edited via the
// dedicated MaterialUsageSection slots in the properties panel, which
// give one drop-target per material actually in use on the object.
// The field still exists on Renderable for scene I/O and as the
// fallback material for unbaked geometry; it's just not surfaced as a
// raw numeric editor here.
static RENDERABLE_FIELDS: [FieldMeta; 3] = [
    FieldMeta { name: "asset_path", field_type: FieldType::String, range: None, transient: false, struct_fields: None, asset_filter: Some("arvx"), enum_options: None, scrub: false },
    FieldMeta { name: "primitive", field_type: FieldType::String, range: None, transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: false },
    FieldMeta { name: "voxel_count", field_type: FieldType::Int, range: None, transient: true, struct_fields: None, asset_filter: None, enum_options: None, scrub: false },
];

pub(super) fn renderable_entry() -> ComponentEntry {
    use crate::components::Renderable;
    ComponentEntry {
        name: "Renderable",
        meta: &RENDERABLE_FIELDS,
        mandatory: false,
        has: |world, entity| world.get::<&Renderable>(entity).is_ok(),
        get_field: |world, entity, field| {
            let c = world.get::<&Renderable>(entity).map_err(|_| "no Renderable".to_string())?;
            match field {
                "asset_path" => Ok(FieldValue::String(c.asset_path.clone().unwrap_or_default())),
                "primitive" => Ok(FieldValue::String(c.primitive.clone().unwrap_or_default())),
                "material_id" => Ok(FieldValue::Int(c.material_id as i64)),
                "voxel_count" => Ok(FieldValue::Int(c.voxel_count as i64)),
                _ => Err(format!("unknown field '{field}'")),
            }
        },
        set_field: |world, entity, field, value| {
            let mut c = world.get::<&mut Renderable>(entity).map_err(|_| "no Renderable".to_string())?;
            match field {
                "asset_path" => {
                    if let FieldValue::String(v) = value { c.asset_path = if v.is_empty() { None } else { Some(v) }; Ok(()) }
                    else { Err("type mismatch".into()) }
                }
                "primitive" => {
                    if let FieldValue::String(v) = value { c.primitive = if v.is_empty() { None } else { Some(v) }; Ok(()) }
                    else { Err("type mismatch".into()) }
                }
                "material_id" => {
                    if let FieldValue::Int(v) = value { c.material_id = v as u16; Ok(()) }
                    else { Err("type mismatch".into()) }
                }
                _ => Err(format!("field '{field}' is read-only or unknown")),
            }
        },
        add_default: |world, entity| {
            world.insert_one(entity, Renderable::default()).map_err(|e| format!("{e}"))
        },
        remove: |world, entity| {
            world.remove_one::<Renderable>(entity).map(|_| ()).map_err(|e| format!("{e}"))
        },
        serialize: |world, entity| {
            let c = world.get::<&Renderable>(entity).ok()?;
            serde_json::to_string(&*c).ok()
        },
        deserialize_insert: |world, entity, json| {
            let c: Renderable = serde_json::from_str(json).map_err(|e| format!("{e}"))?;
            world.insert_one(entity, c).map_err(|e| format!("{e}"))
        },
        on_add: None,
        on_remove: None,
        field_visible: None,
    }
}

// ── PointLight ────────────────────────────────────────────────────────

static POINT_LIGHT_FIELDS: [FieldMeta; 4] = [
    FieldMeta { name: "color", field_type: FieldType::Color, range: None, transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: false },
    FieldMeta { name: "intensity", field_type: FieldType::Float, range: Some((0.0, 100_000.0)), transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: true },
    FieldMeta { name: "range", field_type: FieldType::Float, range: Some((0.1, 500.0)), transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: true },
    FieldMeta { name: "cast_shadow", field_type: FieldType::Bool, range: None, transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: false },
];

pub(super) fn point_light_entry() -> ComponentEntry {
    use crate::components::PointLight;
    ComponentEntry {
        name: "PointLight",
        meta: &POINT_LIGHT_FIELDS,
        mandatory: false,
        has: |world, entity| world.get::<&PointLight>(entity).is_ok(),
        get_field: |world, entity, field| {
            let c = world.get::<&PointLight>(entity).map_err(|_| "no PointLight".to_string())?;
            match field {
                "color" => Ok(FieldValue::Color([c.color[0], c.color[1], c.color[2], 1.0])),
                "intensity" => Ok(FieldValue::Float(c.intensity as f64)),
                "range" => Ok(FieldValue::Float(c.range as f64)),
                "cast_shadow" => Ok(FieldValue::Bool(c.cast_shadow)),
                _ => Err(format!("unknown field '{field}'")),
            }
        },
        set_field: |world, entity, field, value| {
            let mut c = world.get::<&mut PointLight>(entity).map_err(|_| "no PointLight".to_string())?;
            match field {
                "color" => {
                    if let FieldValue::Color(v) = value { c.color = [v[0], v[1], v[2]]; Ok(()) }
                    else { Err("type mismatch".into()) }
                }
                "intensity" => {
                    if let FieldValue::Float(v) = value { c.intensity = v as f32; Ok(()) }
                    else { Err("type mismatch".into()) }
                }
                "range" => {
                    if let FieldValue::Float(v) = value { c.range = v as f32; Ok(()) }
                    else { Err("type mismatch".into()) }
                }
                "cast_shadow" => {
                    if let FieldValue::Bool(v) = value { c.cast_shadow = v; Ok(()) }
                    else { Err("type mismatch".into()) }
                }
                _ => Err(format!("unknown field '{field}'")),
            }
        },
        add_default: |world, entity| {
            world.insert_one(entity, PointLight::default()).map_err(|e| format!("{e}"))
        },
        remove: |world, entity| {
            world.remove_one::<PointLight>(entity).map(|_| ()).map_err(|e| format!("{e}"))
        },
        serialize: |world, entity| {
            let c = world.get::<&PointLight>(entity).ok()?;
            serde_json::to_string(&*c).ok()
        },
        deserialize_insert: |world, entity, json| {
            let c: PointLight = serde_json::from_str(json).map_err(|e| format!("{e}"))?;
            world.insert_one(entity, c).map_err(|e| format!("{e}"))
        },
        on_add: None,
        on_remove: None,
        field_visible: None,
    }
}

// ── SpotLight ────────────────────────────────────────────────────

static SPOT_LIGHT_FIELDS: [FieldMeta; 7] = [
    FieldMeta { name: "color", field_type: FieldType::Color, range: None, transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: false },
    FieldMeta { name: "intensity", field_type: FieldType::Float, range: Some((0.0, 200_000.0)), transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: true },
    FieldMeta { name: "range", field_type: FieldType::Float, range: Some((0.1, 500.0)), transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: true },
    FieldMeta { name: "outer_angle", field_type: FieldType::Float, range: Some((1.0, 179.0)), transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: true },
    FieldMeta { name: "inner_angle", field_type: FieldType::Float, range: Some((0.0, 178.0)), transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: true },
    FieldMeta { name: "direction", field_type: FieldType::Vec3, range: None, transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: false },
    FieldMeta { name: "cast_shadow", field_type: FieldType::Bool, range: None, transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: false },
];

pub(super) fn spot_light_entry() -> ComponentEntry {
    use crate::components::SpotLight;
    ComponentEntry {
        name: "SpotLight",
        meta: &SPOT_LIGHT_FIELDS,
        mandatory: false,
        has: |world, entity| world.get::<&SpotLight>(entity).is_ok(),
        get_field: |world, entity, field| {
            let c = world.get::<&SpotLight>(entity).map_err(|_| "no SpotLight".to_string())?;
            match field {
                "color" => Ok(FieldValue::Color([c.color[0], c.color[1], c.color[2], 1.0])),
                "intensity" => Ok(FieldValue::Float(c.intensity as f64)),
                "range" => Ok(FieldValue::Float(c.range as f64)),
                "outer_angle" => Ok(FieldValue::Float(c.outer_angle as f64)),
                "inner_angle" => Ok(FieldValue::Float(c.inner_angle as f64)),
                "direction" => Ok(FieldValue::Vec3(c.direction.to_array())),
                "cast_shadow" => Ok(FieldValue::Bool(c.cast_shadow)),
                _ => Err(format!("unknown field '{field}'")),
            }
        },
        set_field: |world, entity, field, value| {
            let mut c = world.get::<&mut SpotLight>(entity).map_err(|_| "no SpotLight".to_string())?;
            match field {
                "color" => {
                    if let FieldValue::Color(v) = value { c.color = [v[0], v[1], v[2]]; Ok(()) }
                    else { Err("type mismatch".into()) }
                }
                "intensity" => {
                    if let FieldValue::Float(v) = value { c.intensity = v as f32; Ok(()) }
                    else { Err("type mismatch".into()) }
                }
                "range" => {
                    if let FieldValue::Float(v) = value { c.range = v as f32; Ok(()) }
                    else { Err("type mismatch".into()) }
                }
                "outer_angle" => {
                    if let FieldValue::Float(v) = value { c.outer_angle = v as f32; Ok(()) }
                    else { Err("type mismatch".into()) }
                }
                "inner_angle" => {
                    if let FieldValue::Float(v) = value { c.inner_angle = v as f32; Ok(()) }
                    else { Err("type mismatch".into()) }
                }
                "direction" => {
                    if let FieldValue::Vec3(v) = value { c.direction = glam::Vec3::from_array(v); Ok(()) }
                    else { Err("type mismatch".into()) }
                }
                "cast_shadow" => {
                    if let FieldValue::Bool(v) = value { c.cast_shadow = v; Ok(()) }
                    else { Err("type mismatch".into()) }
                }
                _ => Err(format!("unknown field '{field}'")),
            }
        },
        add_default: |world, entity| {
            world.insert_one(entity, SpotLight::default()).map_err(|e| format!("{e}"))
        },
        remove: |world, entity| {
            world.remove_one::<SpotLight>(entity).map(|_| ()).map_err(|e| format!("{e}"))
        },
        serialize: |world, entity| {
            let c = world.get::<&SpotLight>(entity).ok()?;
            serde_json::to_string(&*c).ok()
        },
        deserialize_insert: |world, entity, json| {
            let c: SpotLight = serde_json::from_str(json).map_err(|e| format!("{e}"))?;
            world.insert_one(entity, c).map_err(|e| format!("{e}"))
        },
        on_add: None,
        on_remove: None,
        field_visible: None,
    }
}

// ── Camera ───────────────────────────────────────────────────────────

static CAMERA_FIELDS: [FieldMeta; 4] = [
    FieldMeta { name: "fov", field_type: FieldType::Float, range: Some((10.0, 170.0)), transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: false },
    FieldMeta { name: "near", field_type: FieldType::Float, range: Some((0.001, 10.0)), transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: false },
    FieldMeta { name: "far", field_type: FieldType::Float, range: Some((10.0, 100000.0)), transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: false },
    FieldMeta { name: "active", field_type: FieldType::Bool, range: None, transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: false },
];

pub(super) fn camera_entry() -> ComponentEntry {
    use crate::components::Camera;
    ComponentEntry {
        name: "Camera",
        meta: &CAMERA_FIELDS,
        mandatory: false,
        has: |world, entity| world.get::<&Camera>(entity).is_ok(),
        get_field: |world, entity, field| {
            let c = world.get::<&Camera>(entity).map_err(|_| "no Camera".to_string())?;
            match field {
                "fov" => Ok(FieldValue::Float(c.fov as f64)),
                "near" => Ok(FieldValue::Float(c.near as f64)),
                "far" => Ok(FieldValue::Float(c.far as f64)),
                "active" => Ok(FieldValue::Bool(c.active)),
                _ => Err(format!("unknown field '{field}'")),
            }
        },
        set_field: |world, entity, field, value| {
            let mut c = world.get::<&mut Camera>(entity).map_err(|e| format!("{e}"))?;
            match field {
                "fov" => { if let FieldValue::Float(v) = value { c.fov = v as f32; Ok(()) } else { Err("type mismatch".into()) } }
                "near" => { if let FieldValue::Float(v) = value { c.near = v as f32; Ok(()) } else { Err("type mismatch".into()) } }
                "far" => { if let FieldValue::Float(v) = value { c.far = v as f32; Ok(()) } else { Err("type mismatch".into()) } }
                "active" => { if let FieldValue::Bool(v) = value { c.active = v; Ok(()) } else { Err("type mismatch".into()) } }
                _ => Err(format!("unknown field '{field}'")),
            }
        },
        add_default: |world, entity| {
            world.insert_one(entity, Camera::default()).map_err(|e| format!("{e}"))
        },
        remove: |world, entity| {
            world.remove_one::<Camera>(entity).map(|_| ()).map_err(|e| format!("{e}"))
        },
        serialize: |world, entity| {
            let c = world.get::<&Camera>(entity).ok()?;
            serde_json::to_string(&*c).ok()
        },
        deserialize_insert: |world, entity, json| {
            let c: Camera = serde_json::from_str(json).map_err(|e| format!("{e}"))?;
            world.insert_one(entity, c).map_err(|e| format!("{e}"))
        },
        on_add: None,
        on_remove: None,
        field_visible: None,
    }
}

